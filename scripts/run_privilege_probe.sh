#!/usr/bin/env bash
# Turnkey privilege × difficulty probe — the validated SLFD experiment.
#
# Prereq: serve a teacher on an OpenAI-compatible endpoint (vLLM), then export:
#   export OMLX_URL=http://localhost:8000/v1
#   export OMLX_MODEL=<served model id, e.g. google/gemma-2-27b-it>
#   export OMLX_API_KEY=<key>        # omit if your server is open
#
# Run:
#   ./scripts/run_privilege_probe.sh
# Override: PB_CONFIG=math|gsm8k|olympiadbench  N=150  SEED=0
set -euo pipefail
PY="$(command -v python || command -v python3)"
CONFIG="${PB_CONFIG:-math}"
N="${N:-150}"
SEED="${SEED:-0}"
RAW="data/processbench_${CONFIG}.jsonl"
SHUF="data/processbench_${CONFIG}_shuffled.jsonl"
TAG="$(echo "${OMLX_MODEL:-teacher}" | tr '/:.' '___')"
mkdir -p data results

echo "== 1/3  Download ProcessBench ${CONFIG} (GT answer + solution joined) =="
"$PY" -m scripts.download_data --processbench --config "$CONFIG" --output "$RAW"

echo "== 2/3  Shuffle (seed ${SEED}) for a balanced prefix =="
"$PY" -m scripts.shuffle_jsonl --input "$RAW" --output "$SHUF" --seed "$SEED"

echo "== 3/3  Privilege probe — teacher=${OMLX_MODEL:-<UNSET>}  N=${N} =="
"$PY" -m experiments.probe_privilege \
    --dataset "$SHUF" --max_samples "$N" \
    --results_dir "results/teacher_eval_${CONFIG}_${TAG}"

echo
echo "Done -> results/teacher_eval_${CONFIG}_${TAG}/privilege_probe.json"
echo "Read gap_solution_f1: clearly > 0 confirms privilege helps on ${CONFIG}."
echo "Reference (local Gemma-4-26b, MATH N=150): gap_solution +0.07; Qwen-27B +0.082."
