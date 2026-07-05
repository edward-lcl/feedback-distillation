#!/bin/bash
# Same-source GSM8K generated-label run (HANDOFF_SAKSHAM.md steps 2-3, local).
# Labels the 400 ProcessBench GSM8K source problems with the local Gemma-4
# teacher under both privilege conditions, then flattens to per-step training
# rows for scripts/run_gold_scorehead_gate.sh TRAIN_DATASET=...
# Resumable: label_pipeline skips already-labeled problems on rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMLX_API_KEY=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])")
export OMLX_URL=${OMLX_URL:-http://127.0.0.1:8000/v1}
export OMLX_MODEL=${OMLX_MODEL:-gemma-4-26b-a4b-it-MLX-4bit}

INPUT=data/gsm8k400_for_labeling.jsonl
mkdir -p data/labeled

echo "=== privileged (+solution) labeling: $(date)"
./.venv/bin/python -m data.label_pipeline --input "$INPUT" \
  --output data/labeled/math_priv_gsm8k400.jsonl --use_omlx --privilege solution

echo "=== no-GT labeling: $(date)"
./.venv/bin/python -m data.label_pipeline --input "$INPUT" \
  --output data/labeled/math_nogt_gsm8k400.jsonl --use_omlx --privilege none

echo "=== flattening: $(date)"
for cond in priv nogt; do
  ./.venv/bin/python -m data.flatten_labels \
    --input data/labeled/math_${cond}_gsm8k400.jsonl \
    --output data/labeled/math_${cond}_gsm8k400_steps.jsonl
done

echo "=== done: $(date)"
wc -l data/labeled/math_{priv,nogt}_gsm8k400*.jsonl
./.venv/bin/python - <<'PY'
import json
for cond in ("priv", "nogt"):
    rows = [json.loads(l) for l in open(f"data/labeled/math_{cond}_gsm8k400.jsonl")]
    steps = [s for r in rows for s in r["steps"]]
    pf = sum(s["parse_failed"] for s in steps)
    err = sum(1 for s in steps if s["is_error"])
    print(f"{cond}: {len(rows)} problems, {len(steps)} steps, "
          f"parse_failed {pf} ({pf/len(steps):.1%}), error rate {err/len(steps):.1%}")
PY
