#!/usr/bin/env python3
"""
Read the PubChem property text for every component through one Message Batch (half
price), then write component_properties.csv from the filled cache.

    DES_RUN_TAG=corpus999 python tools/batch_component_properties.py --budget 20

The components step reads this text one live call at a time (component_properties.run);
this queues every uncached call instead, sends them as one batch, and then runs that
same step, which now finds every answer in the cache. Run `--steps components` first,
so the PubChem text is cached.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="per answer; sets the batch's worst case, which must fit the cap")
    parser.add_argument("--poll", type=int, default=30)
    args = parser.parse_args()
    if not os.environ.get("DES_RUN_TAG"):
        sys.exit("set DES_RUN_TAG -- the batch files and component_properties.csv go there")

    env = dict(os.environ, DES_LLM_BACKEND="anthropic", DES_ANTHROPIC_MODEL=args.model,
               DES_BUDGET_USD=str(args.budget), DES_ANTHROPIC_MAX_TOKENS=str(args.max_tokens))
    os.environ.update(env)
    os.environ["DES_BATCH_MODE"] = "collect"
    import anthropic

    from des_pipeline import batch, budget, config
    from des_pipeline.component_properties import extract_properties
    from des_pipeline.enrich_components import _load_cache

    if batch.pending_path().exists():
        batch.pending_path().rename(batch.batch_dir() / f"stale_{int(time.time())}.jsonl")
    todo = [(n, r) for n, r in _load_cache().items() if r.get("property_strings")]
    queued = 0
    for name, record in todo:
        try:
            extract_properties(name, record["property_strings"], cid=record.get("cid"))
        except batch.Deferred:
            queued += 1
    print(f"{len(todo)} component(s) with PubChem text; {queued} call(s) not cached")

    if queued:
        records = batch.pending()
        client = anthropic.Anthropic(max_retries=4)
        print(f"  worst case ${batch.worst_case_usd(records):.2f}; {budget.status()}")
        batch_id, by_id = batch.submit(records, client)
        print(f"  submitted batch {batch_id}", flush=True)
        started = time.time()
        batch.wait(client, batch_id, poll=args.poll)
        result = batch.collect(client, batch_id, by_id, time.time() - started)
        batch.pending_path().rename(batch.batch_dir() / f"component_properties_{batch_id}.jsonl")
        print(f"  batch ended after {(time.time() - started) / 60:.1f} min: "
              f"{result['succeeded']} ok, {result['errored']} errored, "
              f"${result['cost_usd']:.2f}", flush=True)

    # The ordinary step, now answered from the cache. Collect mode is off, so anything
    # still missing (an errored request) is a live call, which the budget still guards.
    env.pop("DES_BATCH_MODE", None)
    subprocess.run([sys.executable, "-u", str(ROOT / "run_pipeline.py"), "--steps", "components"],
                   cwd=ROOT, env=env, check=True)
    print(budget.status())


if __name__ == "__main__":
    main()
