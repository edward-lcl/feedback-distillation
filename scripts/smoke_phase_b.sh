#!/usr/bin/env bash
# Smoke-test the Phase-B tooling end-to-end on REAL teacher-labeled data, with
# small real models for speed. Validates every moving part actually runs and
# emits valid outputs — NOT a research run (small student, few steps → the
# numbers are meaningless; only "does it work" matters).
#
# Checks: all 4 score-loss ablations (score_critique/verdict/soft/logit_kd)
# train + eval + write a per_step_scores.json sidecar; the online logit-KD path
# runs with a live same-family teacher; transfer_ci produces a CI on real
# sidecars. Outputs go to a scratch dir (default /tmp/smoke_phase_b), never the
# repo's results/ or checkpoints/.
#
# Prereqs: data/labeled/math_priv.jsonl + math_nogt.jsonl (from a prior
# run_student_ablation run) and data/processbench_math_shuffled.jsonl.
#
# Run:   ./scripts/smoke_phase_b.sh
# Tune:  STUDENT=... KD_TEACHER=... MAX_STEPS=12 N_EVAL=80 OUT=/tmp/smoke_phase_b
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$(command -v python || command -v python3)"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"      # small real student
KD_TEACHER="${KD_TEACHER:-Qwen/Qwen2.5-1.5B-Instruct}" # SAME FAMILY as student
EVAL="${EVAL:-data/processbench_math_shuffled.jsonl}"
MAX_STEPS="${MAX_STEPS:-12}"; N_EVAL="${N_EVAL:-80}"
OUT="${OUT:-/tmp/smoke_phase_b}"
PRIV="${PRIV:-data/labeled/math_priv.jsonl}"; NOGT="${NOGT:-data/labeled/math_nogt.jsonl}"

for f in "$PRIV" "$NOGT" "$EVAL"; do
  [ -f "$f" ] || { echo "MISSING $f — run ./scripts/run_student_ablation.sh first to produce labeled data."; exit 1; }
done
mkdir -p "$OUT"; PASS=1

cell () {  # $1=labeled $2=ablation $3=tag  [$4=extra train args]
  echo "### TRAIN $3 (ablation=$2)"
  $PY -m experiments.train_slfd --dataset "$1" --ablation "$2" --student_model "$STUDENT" \
      --seed 0 --max_steps "$MAX_STEPS" --epochs 1 --checkpoint "$OUT/$3.pt" ${4:-} \
      > "$OUT/$3.train.log" 2>&1 || { echo "  TRAIN FAILED $3"; tail -5 "$OUT/$3.train.log"; PASS=0; return; }
  $PY -m experiments.run_processbench --checkpoint "$OUT/$3.pt" --student_model "$STUDENT" \
      --dataset "$EVAL" --max_samples "$N_EVAL" --results_dir "$OUT/$3" \
      > "$OUT/$3.eval.log" 2>&1 || { echo "  EVAL FAILED $3"; tail -5 "$OUT/$3.eval.log"; PASS=0; return; }
  $PY -c "import json;d=json.load(open('$OUT/$3/processbench_results.json'));print('  %-14s roc_auc=%.3f f1=%.3f pred_err=%.3f'%('$3',d['roc_auc'],d['f1'],d['pred_error_rate']))"
  [ -f "$OUT/$3/per_step_scores.json" ] && echo "  sidecar OK" || { echo "  sidecar MISSING"; PASS=0; }
}

cell "$PRIV" score_critique priv_critique
cell "$NOGT" score_critique nogt_critique
cell "$PRIV" verdict        priv_verdict
cell "$PRIV" soft           priv_soft
cell "$PRIV" logit_kd       priv_logitkd  "--kd_teacher $KD_TEACHER"
cell "$NOGT" logit_kd       nogt_logitkd  "--kd_teacher $KD_TEACHER"

echo "### TRANSFER_CI (score_critique priv vs nogt)"
$PY -m experiments.transfer_ci --priv "$OUT/priv_critique/per_step_scores.json" \
    --nogt "$OUT/nogt_critique/per_step_scores.json" --n_boot 2000 || PASS=0
echo "### TRANSFER_CI (logit_kd priv vs nogt)"
$PY -m experiments.transfer_ci --priv "$OUT/priv_logitkd/per_step_scores.json" \
    --nogt "$OUT/nogt_logitkd/per_step_scores.json" --n_boot 2000 || PASS=0

echo "=================================================="
if [ "$PASS" = 1 ]; then echo "SMOKE: ALL CELLS PASSED"; else echo "SMOKE: SOME CELLS FAILED"; exit 1; fi
