#!/usr/bin/env bash
# B0 v2: clean retrain of priv_critique / nogt_critique (full 2 epochs, no
# MAX_STEPS cap -- the 2026-06-24 attempt capped at MAX_STEPS=2000, roughly
# half an epoch, and scored 0.55/0.58 ROC-AUC vs. the 0.63/0.64 the paper
# cites; this run matches the original recipe instead) + a shared-pool
# paired BoN at N=1000 against a locally-launched generator on port 8001
# (port 8000 is the Gemma-4 teacher daemon -- left untouched).
#
# New checkpoint/result names (…_b0v2) so this never collides with the
# existing checkpoints/priv_critique.pt / nogt_critique.pt or their results.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
LOG=results/overnight/b0_paired_v2.log
mkdir -p results/overnight results/bon_paired_v2 checkpoints

log(){ printf '\n=== [%s] %s ===\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$LOG"; }

EPOCHS="${EPOCHS:-2}"
# BATCH_SIZE=1 (not the original run's 2) -- this machine now also has an
# unrelated AJAR job and the local generator server running concurrently, so
# trading some speed for OOM safety on an unattended run. Matches the
# documented "working MPS recipe" from the 2026-06-24 overnight sweep fixes.
BATCH_SIZE="${BATCH_SIZE:-1}"
N_EVAL="${N_EVAL:-1000}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GEN_PORT="${GEN_PORT:-8001}"
GEN_MODEL="${GEN_MODEL:-mlx-community/gemma-3-4b-it-4bit}"
GEN_URL="http://localhost:${GEN_PORT}/v1"

log "B0v2: clean retrain priv_critique_b0v2 + nogt_critique_b0v2 (full EPOCHS=$EPOCHS, no MAX_STEPS cap)"
log "student=$STUDENT_MODEL batch=$BATCH_SIZE n_eval=$N_EVAL generator=$GEN_MODEL on $GEN_URL"

for f in data/labeled/math_priv.jsonl data/labeled/math_nogt.jsonl data/processbench_math_shuffled.jsonl; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
  log "data: $f ($(wc -l < "$f") lines)"
done

# --- Launch a local generator on a port distinct from the Gemma-4 teacher (8000) ---
log "Launching local generator: $GEN_MODEL on port $GEN_PORT"
nohup /opt/homebrew/bin/mlx_lm.server --model "$GEN_MODEL" --port "$GEN_PORT" \
  >> results/overnight/generator_b0v2.log 2>&1 &
GEN_PID=$!
log "generator pid=$GEN_PID (log: results/overnight/generator_b0v2.log)"

log "Waiting for generator to become ready..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' "$GEN_URL/models" 2>/dev/null | grep -q "200"; then
    log "generator ready after ${i}0s"
    break
  fi
  sleep 10
done

# --- Train priv_critique_b0v2 ---
if [ -f checkpoints/priv_critique_b0v2.pt ]; then
  log "priv_critique_b0v2.pt already exists — skipping retrain (delete it to force)"
else
  log "Training priv_critique_b0v2 (score_critique, priv labels, full $EPOCHS epochs)..."
  "$PY" -m experiments.train_slfd \
    --dataset data/labeled/math_priv.jsonl \
    --ablation score_critique \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --student_model "$STUDENT_MODEL" \
    --checkpoint checkpoints/priv_critique_b0v2.pt \
    2>&1 | tee -a "$LOG"
  log "priv_critique_b0v2 training done: $(ls -lh checkpoints/priv_critique_b0v2.pt)"
fi

# --- Train nogt_critique_b0v2 ---
if [ -f checkpoints/nogt_critique_b0v2.pt ]; then
  log "nogt_critique_b0v2.pt already exists — skipping retrain (delete it to force)"
else
  log "Training nogt_critique_b0v2 (score_critique, nogt labels, full $EPOCHS epochs)..."
  "$PY" -m experiments.train_slfd \
    --dataset data/labeled/math_nogt.jsonl \
    --ablation score_critique \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --student_model "$STUDENT_MODEL" \
    --checkpoint checkpoints/nogt_critique_b0v2.pt \
    2>&1 | tee -a "$LOG"
  log "nogt_critique_b0v2 training done: $(ls -lh checkpoints/nogt_critique_b0v2.pt)"
fi

# --- Eval both on ProcessBench (N_EVAL=1000, matching current canonical eval size) ---
log "Evaluating priv_critique_b0v2 on processbench (N=$N_EVAL)..."
"$PY" -m experiments.run_processbench \
  --checkpoint checkpoints/priv_critique_b0v2.pt \
  --student_model "$STUDENT_MODEL" \
  --dataset data/processbench_math_shuffled.jsonl \
  --max_samples "$N_EVAL" \
  --results_dir results/ablation/priv_critique_b0v2 \
  2>&1 | tee -a "$LOG"

log "Evaluating nogt_critique_b0v2 on processbench (N=$N_EVAL)..."
"$PY" -m experiments.run_processbench \
  --checkpoint checkpoints/nogt_critique_b0v2.pt \
  --student_model "$STUDENT_MODEL" \
  --dataset data/processbench_math_shuffled.jsonl \
  --max_samples "$N_EVAL" \
  --results_dir results/ablation/nogt_critique_b0v2 \
  2>&1 | tee -a "$LOG"

log "Processbench eval results:"
for tag in priv_critique_b0v2 nogt_critique_b0v2; do
  f="results/ablation/$tag/processbench_results.json"
  [ -f "$f" ] && "$PY" -c "
import json
d=json.load(open('$f'))
print('$tag: roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}  f1={f1:.4f}'.format(**d))
" | tee -a "$LOG"
done

# --- Transfer CI on the new checkpoints (step-level, sanity check vs. the older pair) ---
log "Transfer CI (priv_critique_b0v2 vs nogt_critique_b0v2)..."
"$PY" -m experiments.transfer_ci \
  --model_a results/ablation/priv_critique_b0v2/per_step_scores.json \
  --model_b results/ablation/nogt_critique_b0v2/per_step_scores.json \
  --n_boot 10000 \
  --out results/ablation/transfer_ci_b0v2.json \
  2>&1 | tee -a "$LOG"

# --- Shared-pool paired BoN at N=1000 against the local generator ---
log "Running shared-pool paired BoN (N=8, max_samples=1000) via local generator..."
"$PY" -m experiments.bon_paired \
  --dataset data/processbench_math_shuffled.jsonl \
  --priv checkpoints/priv_critique_b0v2.pt \
  --nogt checkpoints/nogt_critique_b0v2.pt \
  --student_model "$STUDENT_MODEL" \
  --n 8 \
  --max_samples 1000 \
  --backend omlx \
  --omlx_url "$GEN_URL" \
  --results_dir results/bon_paired_v2 \
  2>&1 | tee -a "$LOG"

log "Final BoN results:"
cat results/bon_paired_v2/bon_paired_results.json | tee -a "$LOG"

log "Shutting down local generator (pid=$GEN_PID)"
kill "$GEN_PID" 2>/dev/null || true

log "DONE — B0v2 clean retrain + shared-pool paired BoN complete."
log "Do NOT hand-edit RESULTS.md or the paper from this run without comparing against the existing priv_critique/nogt_critique numbers first."
