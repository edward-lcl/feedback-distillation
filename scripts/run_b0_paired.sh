#!/usr/bin/env bash
# B0 runner: retrain priv_critique + nogt_critique, then run paired BoN at N=1000.
# Uses REUSE_LABELS=1 (no teacher, no generation for training).
# BoN generation uses local oMLX at OMLX_URL / OMLX_API_KEY.
# MAX_STEPS defaults to 4000 — enough signal, fits well within 90 min on MPS.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
cd "$REPO"

if [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"
else PY="$(command -v python || command -v python3)"; fi

LOG=results/overnight/b0_paired.log
mkdir -p results/overnight results/bon_paired checkpoints

log(){ printf '\n=== [%s] %s ===\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$LOG"; }

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_STEPS="${MAX_STEPS:-4000}"
N_EVAL="${N_EVAL:-1000}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OMLX_URL="${OMLX_URL:-http://localhost:8000/v1}"
OMLX_MODEL="${OMLX_MODEL:-Qwen3-4B-Instruct-2507-MLX-8bit}"

log "B0: retrain priv_critique + nogt_critique (REUSE_LABELS, MAX_STEPS=$MAX_STEPS)"
log "student=$STUDENT_MODEL  batch=$BATCH_SIZE  omlx=$OMLX_URL  gen_model=$OMLX_MODEL"

for f in data/labeled/math_priv.jsonl data/labeled/math_nogt.jsonl data/processbench_math_shuffled.jsonl; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
  log "data: $f ($(wc -l < "$f") lines)"
done

# --- Train priv_critique ---
if [ -f checkpoints/priv_critique.pt ]; then
  log "priv_critique.pt already exists — skipping retrain (delete it to force)"
else
  log "Training priv_critique (score_critique, priv labels)..."
  "$PY" -m experiments.train_slfd \
    --dataset data/labeled/math_priv.jsonl \
    --ablation score_critique \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --student_model "$STUDENT_MODEL" \
    --checkpoint checkpoints/priv_critique.pt
  log "priv_critique training done: $(ls -lh checkpoints/priv_critique.pt)"
fi

# --- Train nogt_critique ---
if [ -f checkpoints/nogt_critique.pt ]; then
  log "nogt_critique.pt already exists — skipping retrain (delete it to force)"
else
  log "Training nogt_critique (score_critique, nogt labels)..."
  "$PY" -m experiments.train_slfd \
    --dataset data/labeled/math_nogt.jsonl \
    --ablation score_critique \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --student_model "$STUDENT_MODEL" \
    --checkpoint checkpoints/nogt_critique.pt
  log "nogt_critique training done: $(ls -lh checkpoints/nogt_critique.pt)"
fi

log "Checkpoints ready:"
ls -lh checkpoints/priv_critique.pt checkpoints/nogt_critique.pt | tee -a "$LOG"

# --- Eval both checkpoints on processbench (N_EVAL=1000) ---
log "Evaluating priv_critique on processbench (N=$N_EVAL)..."
"$PY" -m experiments.run_processbench \
  --checkpoint checkpoints/priv_critique.pt \
  --student_model "$STUDENT_MODEL" \
  --dataset data/processbench_math_shuffled.jsonl \
  --max_samples "$N_EVAL" \
  --results_dir results/ablation/priv_critique_b0

log "Evaluating nogt_critique on processbench (N=$N_EVAL)..."
"$PY" -m experiments.run_processbench \
  --checkpoint checkpoints/nogt_critique.pt \
  --student_model "$STUDENT_MODEL" \
  --dataset data/processbench_math_shuffled.jsonl \
  --max_samples "$N_EVAL" \
  --results_dir results/ablation/nogt_critique_b0

log "Processbench eval results:"
for tag in priv_critique_b0 nogt_critique_b0; do
  f="results/ablation/$tag/processbench_results.json"
  [ -f "$f" ] && "$PY" -c "
import json
d=json.load(open('$f'))
print('$tag: roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}  f1={f1:.4f}'.format(**d))
" | tee -a "$LOG"
done

# --- BoN paired test at N=1000 ---
log "Running paired BoN test (N=8, max_samples=1000) via oMLX..."
log "Generator: OMLX_URL=$OMLX_URL  OMLX_MODEL=$OMLX_MODEL"
OMLX_API_KEY="${OMLX_API_KEY:-}" \
OMLX_URL="$OMLX_URL" \
OMLX_MODEL="$OMLX_MODEL" \
"$PY" -m experiments.bon_paired \
  --dataset data/processbench_math_shuffled.jsonl \
  --priv checkpoints/priv_critique.pt \
  --nogt checkpoints/nogt_critique.pt \
  --student_model "$STUDENT_MODEL" \
  --n 8 \
  --max_samples 1000 \
  --omlx_url "$OMLX_URL" \
  --results_dir results/bon_paired 2>&1 | tee -a "$LOG"

log "Final BoN results:"
cat results/bon_paired/bon_paired_results.json | tee -a "$LOG"

log "DONE — B0 paired test complete."
log "Do NOT hand-edit RESULTS.md or mark anything validated."
