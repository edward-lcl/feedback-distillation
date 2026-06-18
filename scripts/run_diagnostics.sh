#!/usr/bin/env bash
# Diagnose the "privilege does not transfer to the student" null — one command.
# Run AFTER the N=1000 ablation (uses its checkpoints + labeled data).
#
# Prereqs (same env you used for the main run):
#   checkpoints/priv_critique.pt, checkpoints/nogt_critique.pt
#   data/labeled/math_priv.jsonl, data/labeled/math_nogt.jsonl
#   data/processbench_math_shuffled.jsonl
#   GEN_OMLX_URL / GEN_OMLX_MODEL        -> your local generator (gemma-2-9b)
#   OMLX_URL / OMLX_MODEL / OMLX_API_KEY -> Edward's served Gemma-4 teacher (for the probe)
#
#   GEN_OMLX_URL=http://localhost:8000/v1 GEN_OMLX_MODEL=google/gemma-2-9b-it \
#   OMLX_URL=https://teacher.elcl.systems/v1 OMLX_MODEL=gemma-4-26b-a4b-it-MLX-4bit \
#   OMLX_API_KEY=... ./scripts/run_diagnostics.sh
set -euo pipefail
PY="$(command -v python || command -v python3)"
MAXP="${MAX_SAMPLES:-1000}"
GEN_OMLX_URL="${GEN_OMLX_URL:-${OMLX_URL:-http://localhost:8000/v1}}"
GEN_OMLX_MODEL="${GEN_OMLX_MODEL:-${OMLX_MODEL:-google/gemma-2-9b-it}}"

echo "== 1/3  Label agreement — do priv vs no-GT teacher labels actually differ? =="
echo "        (if they agree on >95% of steps, that alone explains the null)"
"$PY" -m experiments.label_agreement \
    --priv data/labeled/math_priv.jsonl --nogt data/labeled/math_nogt.jsonl

echo; echo "== 2/3  Same-pool paired Best-of-N — priv vs nogt verifier on ONE shared pool =="
echo "        (generation via the GENERATOR endpoint: $GEN_OMLX_MODEL)"
OMLX_URL="$GEN_OMLX_URL" OMLX_MODEL="$GEN_OMLX_MODEL" \
"$PY" -m experiments.bon_paired --dataset data/processbench_math_shuffled.jsonl \
    --priv checkpoints/priv_critique.pt --nogt checkpoints/nogt_critique.pt \
    --n 8 --max_samples "$MAXP" --omlx_url "$GEN_OMLX_URL"

echo; echo "== 3/3  Gemma-4 privilege probe — confirm priv≠no-GT for the LABELING teacher =="
echo "        (probe via the TEACHER endpoint: ${OMLX_MODEL:-<unset>})"
: "${OMLX_API_KEY:?set OMLX_API_KEY (teacher key) for the probe step}"
N="${N:-150}" PB_CONFIG="${PB_CONFIG:-math}" SEED="${SEED:-0}" ./scripts/run_privilege_probe.sh

echo; echo "== Diagnostics complete =="
echo "Read: results/diagnostics/label_agreement.json · results/bon_paired/bon_paired_results.json"
echo "      results/teacher_eval_math_<gemma-4>/privilege_probe.json"
