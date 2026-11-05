#!/usr/bin/env python3
"""
Estimate what a run will cost BEFORE it starts, and refuse it if the estimate does not
fit the remaining monthly budget (or the run's own cap).

    python tools/preflight.py claude-opus-5-5 [--budget 25]

The estimate uses measured tokens per call type (table profile, paper overview, prose
section): from the most recent Claude run when one exists, otherwise from the qwen3 run
scaled for Claude's tokenizer. It is deliberately pessimistic (output x2 for thinking on
the qwen3 basis). Exit code 0 = go, 2 = refused.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from des_pipeline import budget, config  # noqa: E402

CALLS = {"table": 109, "overview": 13, "section": 65}   # one 13-paper run


def kind(call):
    call = str(call)
    return ("table" if call.startswith("tbl-") else "overview" if call.startswith("paper-")
            else "component" if call.startswith("comp-") else "section")


def per_call_tokens(tag):
    path = ROOT / "runs" / tag / "usage.csv"
    if not path.exists():
        return None
    agg = defaultdict(Counter)
    for r in csv.DictReader(open(path)):
        k = kind(r["call"])
        agg[k]["n"] += 1
        agg[k]["in"] += float(r["input_tokens"] or 0)
        agg[k]["out"] += float(r["output_tokens"] or 0)
    return {k: (v["in"] / v["n"], v["out"] / v["n"]) for k, v in agg.items() if v["n"]}


def estimate(model):
    """-> (usd, basis). Prefers a measured Claude run over the qwen3 basis."""
    p_in, p_out = budget.price(model)
    for tag in ("opus55", "sonnet55", "haiku45"):
        tokens = per_call_tokens(tag)
        if tokens and all(k in tokens for k in CALLS):
            usd = sum(n * (tokens[k][0] * p_in + tokens[k][1] * p_out)
                      for k, n in CALLS.items())
            # Opus thinks more than Sonnet at the same effort; allow for it.
            factor = 1.5 if tag != "opus55" and "opus" in model else 1.0
            return usd * factor, f"measured tokens from run {tag!r}" + \
                (" x1.5 for Opus thinking" if factor > 1 else "")
    tokens = per_call_tokens("qwen3")
    usd = sum(n * (tokens[k][0] * 1.35 * p_in + tokens[k][1] * 2.0 * p_out)
              for k, n in CALLS.items())
    return usd, "qwen3 tokens x1.35 (tokenizer), output x2 (thinking)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--budget", type=float, default=0.0)
    args = parser.parse_args()
    usd, basis = estimate(args.model)
    left = budget.monthly_cap() - budget.month_spent(refresh=True)
    print(f"pre-flight: {args.model}: estimated ${usd:.2f} for the 13 papers ({basis})")
    print(f"pre-flight: {budget.status()}")
    if usd > left:
        print(f"pre-flight: REFUSED -- the estimate exceeds the ${left:.2f} left this month")
        sys.exit(2)
    if args.budget and usd > args.budget:
        print(f"pre-flight: REFUSED -- the estimate exceeds this run's cap "
              f"(${args.budget:.2f}); raise it if you accept the estimate")
        sys.exit(2)
    print("pre-flight: OK")


if __name__ == "__main__":
    main()
