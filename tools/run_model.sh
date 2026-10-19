#!/usr/bin/env bash
# Run the 13 validation papers with one Claude model, under its own run tag.
#
#   tools/run_model.sh haiku45  claude-haiku-4-5   10
#   tools/run_model.sh sonnet55 claude-sonnet-5-5  15
#   tools/run_model.sh opus55   claude-opus-5-5    25
#
# Arguments: run tag, model id, budget in US dollars (the run stops before exceeding it;
# every answer already received is cached, so re-running continues where it stopped).
#
# Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Hand-pinned table profiles
# (Sadeghi's Table 2) are honoured, exactly as in the qwen3 baseline, so every model is
# compared on the same footing; add --ignore-human-profiles to BOTH runs to test that
# table too. Runs only the steps a model affects; components, QM9 and the graph are
# model-independent.
# Then score it:  python tools/evaluate.py --tag qwen3 --tag <tag>
set -euo pipefail
tag="${1:?run tag}"; model="${2:?model id}"; budget="${3:-20}"
cd "$(dirname "$0")/.."
mkdir -p "runs/$tag"
DES_RUN_TAG="$tag" DES_LLM_BACKEND=anthropic DES_ANTHROPIC_MODEL="$model" \
DES_BUDGET_USD="$budget" \
  python -u run_pipeline.py --steps refs,table,figures,text,validate \
    --xml 'xml/*.xml,xml/selected/*.xml' --continue-on-error \
    2>&1 | tee "runs/$tag/run.log"
echo
python - "$tag" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(f"runs/{sys.argv[1]}/usage.csv")))
print(f"{len(rows)} calls, ${sum(float(r['cost_usd']) for r in rows):.2f}, "
      f"{sum(float(r['seconds']) for r in rows) / 60:.1f} model-minutes")
EOF
