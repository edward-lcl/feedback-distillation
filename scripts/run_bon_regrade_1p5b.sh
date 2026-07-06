#!/usr/bin/env bash
# Regrade rerun of the 1.5B paired BoN (results/bon_paired, 2026-06-30):
# that run's grading silently lacked math_verify (answers_match fell back to
# string/numeric matching on symbolic MATH answers), so its numbers are
# superseded. No retraining -- reuses checkpoints/{priv,nogt}_critique_b0v2.pt.
# This time the shared pool is cached and per-candidate PRM scores are
# persisted, so experiments.bon_curve can produce the BoN-vs-N figure data
# without any re-scoring.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
OUT=results/bon_paired_regrade
LOG="$OUT/run.log"
mkdir -p "$OUT" results/overnight

log(){ printf '\n=== [%s] %s ===\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$LOG"; }

GEN_PORT="${GEN_PORT:-8001}"
GEN_MODEL="${GEN_MODEL:-mlx-community/gemma-3-4b-it-4bit}"
GEN_URL="http://localhost:${GEN_PORT}/v1"
BON_N="${BON_N:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

"$PY" -c "import math_verify" || { echo "FATAL: math_verify missing -- the whole point of this rerun" >&2; exit 1; }

log "Launching generator: $GEN_MODEL on port $GEN_PORT"
nohup /opt/homebrew/bin/mlx_lm.server --model "$GEN_MODEL" --port "$GEN_PORT" \
  >> results/overnight/generator_regrade.log 2>&1 &
GEN_PID=$!
trap 'kill "$GEN_PID" 2>/dev/null || true' EXIT
log "generator pid=$GEN_PID"

for i in $(seq 1 90); do
  curl -s "$GEN_URL/models" > /dev/null 2>&1 && break
  sleep 2
  [ "$i" = 90 ] && { echo "FATAL: generator never became ready" >&2; exit 1; }
done
log "Generator ready"

log "Paired BoN: N=$BON_N, $MAX_SAMPLES problems, b0v2 checkpoints, math_verify grading"
OMLX_MODEL="$GEN_MODEL" OMLX_API_KEY="${OMLX_API_KEY:-local}" \
  "$PY" -m experiments.bon_paired \
  --dataset data/processbench_math_shuffled.jsonl \
  --priv checkpoints/priv_critique_b0v2.pt \
  --nogt checkpoints/nogt_critique_b0v2.pt \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --backend omlx --omlx_url "$GEN_URL" \
  --n "$BON_N" --max_samples "$MAX_SAMPLES" --temperature 0.8 \
  --candidates_file "$OUT/pool_gemma3_4b_n${BON_N}_t0.8.jsonl" \
  --scores_file "$OUT/scored_pool.jsonl" \
  --results_dir "$OUT" 2>&1 | tee -a "$LOG"

log "BoN-vs-N curve"
"$PY" -m experiments.bon_curve \
  --scores_file "$OUT/scored_pool.jsonl" \
  --ns 1 2 4 8 --out "$OUT/bon_curve.json" 2>&1 | tee -a "$LOG"

log "DONE -- results in $OUT (push raw JSONs to a branch, do not hand-edit)"
