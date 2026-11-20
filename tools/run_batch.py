#!/usr/bin/env python3
"""
Run papers through the pipeline with Anthropic Message Batches (half price, async).

    python tools/run_batch.py --tag haiku45_batch100 --model claude-haiku-4-5 \
        --budget 20 --xml 'xml/pmc/PMC1.xml,xml/pmc/PMC2.xml'

Rounds (see des_pipeline/batch.py): a collect pass queues every uncached call and stops
each paper at its first; the queue goes out as one batch; the answers fill the response
cache; the next pass gets one stage further. It ends when a pass queues nothing -- that
pass has written the complete outputs. Every round is logged to
runs/<tag>/batch/rounds.csv (requests, wall time, cost), for comparing with a direct run.

Stopping is safe at any point: answers already collected are cached, and re-running the
same command resumes. A batch still processing when this is interrupted can be picked
up with --resume <batch id>.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--steps", default="refs,table,figures,text,validate")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--resume", default="",
                        help="a batch id already submitted from batch/pending.jsonl: "
                             "collect it instead of running round 1's collect pass")
    args = parser.parse_args()

    env = dict(os.environ, DES_RUN_TAG=args.tag, DES_LLM_BACKEND="anthropic",
               DES_ANTHROPIC_MODEL=args.model, DES_BUDGET_USD=str(args.budget))
    os.environ.update(env)                      # config below reads the same settings
    import anthropic

    from des_pipeline import batch, budget, config

    client = anthropic.Anthropic(max_retries=4)
    run_dir = config.RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds_csv = batch.batch_dir() / "rounds.csv"
    started = time.time()

    for round_no in range(1, args.max_rounds + 1):
        if args.resume and round_no == 1:
            # The queue that was submitted is still in pending.jsonl; its order gave
            # the custom ids, so it maps the results back to their cache paths.
            records = batch.pending()
            batch_id, by_id, pass_seconds = args.resume, batch.ids_for(records), 0
            print(f"\n=== round 1: resuming batch {batch_id} ({len(records)} requests) ===",
                  flush=True)
            submitted = time.time()
        else:
            if batch.pending_path().exists():
                batch.pending_path().rename(
                    batch.batch_dir() / f"stale_{int(time.time())}.jsonl")
            print(f"\n=== round {round_no}: collect pass ===", flush=True)
            t0 = time.time()
            with open(run_dir / f"run_round{round_no}.log", "w") as log:
                subprocess.run(
                    [sys.executable, "-u", str(ROOT / "run_pipeline.py"), "--steps",
                     args.steps, "--xml", args.xml, "--continue-on-error"],
                    cwd=ROOT, env=dict(env, DES_BATCH_MODE="collect"), stdout=log,
                    stderr=subprocess.STDOUT, check=False)
            pass_seconds = time.time() - t0
            records = batch.pending()
            if not records:
                print(f"  nothing queued: complete after {round_no - 1} batch round(s)")
                (run_dir / f"run_round{round_no}.log").replace(run_dir / "run.log")
                break
            print(f"  {len(records)} request(s) queued ({pass_seconds:.0f} s); "
                  f"worst case ${batch.worst_case_usd(records):.2f}; {budget.status()}")
            batch_id, by_id = batch.submit(records, client)
            submitted = time.time()
            print(f"  submitted batch {batch_id}", flush=True)
        batch.wait(client, batch_id, poll=args.poll)
        waited = time.time() - submitted
        result = batch.collect(client, batch_id, by_id, waited)
        batch.pending_path().rename(batch.batch_dir() / f"round{round_no}.jsonl")
        print(f"  batch ended after {waited / 60:.1f} min: {result['succeeded']} ok, "
              f"{result['errored']} errored, ${result['cost_usd']:.2f}", flush=True)
        new = not rounds_csv.exists()
        with open(rounds_csv, "a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["round", "batch_id", "requests", "collect_pass_s",
                                 "batch_wait_s", "succeeded", "errored", "cost_usd"])
            writer.writerow([round_no, batch_id, len(records), round(pass_seconds),
                             round(waited), result["succeeded"], result["errored"],
                             round(result["cost_usd"], 4)])
        if not result["succeeded"]:
            print("  no request succeeded; stopping")
            break
    else:
        print(f"  stopped after {args.max_rounds} rounds with requests still queued")
    print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min; {budget.status()}")


if __name__ == "__main__":
    main()
