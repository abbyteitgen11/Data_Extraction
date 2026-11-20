"""
Anthropic Message Batches: the same calls at half price, answered within 24 hours.

A paper's calls depend on each other: the table profiles and the paper overview build
the dictionary that every prose prompt then contains. So a batch run goes in rounds,
driven by tools/run_batch.py:

    1. run the pipeline with DES_BATCH_MODE=collect. An uncached call is not sent:
       `queue` writes its request to runs/<tag>/batch/pending.jsonl and raises
       Deferred, which stops that paper (its other calls in flight are queued too).
    2. `submit` sends the queue as one Message Batch; `wait` polls until it ends.
    3. `collect` writes each answer into the ordinary response cache, at exactly the
       path the pipeline will look for, and logs its usage at the batch price.
    4. repeat: every paper now gets one stage further. A pass that queues nothing has
       produced the complete outputs.

Nothing in the extraction changes: a batch answer and a direct answer are the same
cached JSON, so a paper half done in batch mode can be finished either way.
"""
import json
import threading
import time

from . import budget, config


class Deferred(BaseException):
    """A call was queued for the batch. BaseException, not Exception: the pipeline has
    `except Exception` fallbacks (the paper-type classifier assumes "review") that must
    not mistake "not answered yet" for "failed"."""


_LOCK = threading.Lock()


def batch_dir():
    path = (config.RUN_DIR or config.DATA) / "batch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pending_path():
    return batch_dir() / "pending.jsonl"


def queue(cache_path, section_id, kind, kwargs):
    """Record one request for the next batch, then raise Deferred."""
    record = {"cache_path": str(cache_path), "section_id": section_id, "kind": kind,
              "params": kwargs}
    with _LOCK:
        with open(pending_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    raise Deferred(section_id)


def pending():
    """The queued requests, one per cache path (a re-queued call is sent once)."""
    path = pending_path()
    if not path.exists():
        return []
    seen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            seen.setdefault(record["cache_path"], record)
    return list(seen.values())


def worst_case_usd(records):
    """Every request's input estimate plus its full max_tokens, at the batch price."""
    total = 0.0
    for r in records:
        params = r["params"]
        chars = sum(len(m["content"]) for m in params["messages"])
        total += budget.worst_case(params["model"], chars, params["max_tokens"])
    return total * config.BATCH_DISCOUNT


def custom_id_for(i):
    return f"r{i:05d}"


def ids_for(records):
    """{custom_id: record} as submit assigned them -- for resuming a submitted batch."""
    return {custom_id_for(i): r for i, r in enumerate(records)}


def submit(records, client):
    """Send the queue as one batch. -> (batch id, {custom_id: record}).

    Refused, before anything is sent, if its worst case would pass the monthly or the
    run cap -- the same caps the direct calls obey.
    """
    worst = worst_case_usd(records)
    left = budget.monthly_cap() - budget.month_spent(refresh=True)
    if worst > left:
        raise budget.BudgetExceeded(f"batch worst case ${worst:.2f} exceeds the "
                                    f"${left:.2f} left this month")
    if config.BUDGET_USD and budget.run_spent(refresh=True) + worst > config.BUDGET_USD:
        raise budget.BudgetExceeded(f"batch worst case ${worst:.2f} would pass "
                                    f"DES_BUDGET_USD=${config.BUDGET_USD:.2f}")
    by_id, requests, beta = {}, [], False
    for i, r in enumerate(records):
        params = dict(r["params"])
        if r["kind"] == "beta":
            beta = True
        params.pop("betas", None)
        custom_id = custom_id_for(i)
        by_id[custom_id] = r
        requests.append({"custom_id": custom_id, "params": params})
    if beta:
        betas = sorted({b for r in records for b in r["params"].get("betas", [])})
        created = client.beta.messages.batches.create(requests=requests, betas=betas)
    else:
        created = client.messages.batches.create(requests=requests)
    return created.id, by_id


def wait(client, batch_id, poll=30):
    """Poll until the batch has ended. -> the final batch object."""
    import anthropic

    failures = 0
    while True:
        try:
            status = client.messages.batches.retrieve(batch_id)
            failures = 0
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            # The batch keeps running server-side; a dropped connection (Wi-Fi, sleep)
            # is no reason to abandon it. Keep asking, more slowly.
            failures += 1
            print(f"    batch {batch_id}: {type(exc).__name__}; retrying", flush=True)
            time.sleep(min(600, poll * 2 ** min(failures, 5)))
            continue
        if status.processing_status == "ended":
            return status
        counts = status.request_counts
        print(f"    batch {batch_id}: {counts.processing} processing, "
              f"{counts.succeeded} succeeded, {counts.errored} errored", flush=True)
        time.sleep(poll)


def collect(client, batch_id, by_id, seconds):
    """Write every answer into the response cache and log its usage.

    -> {"succeeded": n, "errored": n, "cost_usd": x}. A request that errored or
    expired writes nothing, so the next pass queues it again.
    """
    from .extract_text_llm import answer_text, log_usage

    out = {"succeeded": 0, "errored": 0, "cost_usd": 0.0}
    n = max(1, len(by_id))
    for result in client.messages.batches.results(batch_id):
        record = by_id.get(result.custom_id)
        if record is None:
            continue
        if result.result.type != "succeeded":
            out["errored"] += 1
            print(f"    {record['section_id']}: {result.result.type}")
            continue
        message = result.result.message
        usage = message.usage
        cost = log_usage("anthropic", message.model, record["section_id"],
                         usage.input_tokens, usage.output_tokens, seconds / n,
                         getattr(usage, "cache_read_input_tokens", 0) or 0,
                         getattr(usage, "cache_creation_input_tokens", 0) or 0,
                         f"batch:{message.stop_reason or ''}",
                         discount=config.BATCH_DISCOUNT)
        out["cost_usd"] += cost
        path = config.LLM_CACHE_DIR / record["cache_path"].rsplit("/", 1)[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(answer_text(message, record["section_id"]), encoding="utf-8")
        out["succeeded"] += 1
    return out
