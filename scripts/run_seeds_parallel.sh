#!/usr/bin/env bash
# run_seeds_parallel.sh
# Efficient parallel execution of Phase B generated-label seeds.
# GPU 0: Privileged BCE, error weight 3 (seeds 0-3)
# GPU 1: No-GT rank-only, balanced batches (seeds 0-3)

set -euo pipefail

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PY="${PY:-$(command -v python || command -v python3)}"
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"
EPOCHS=2
BATCH_SIZE=2
N_EVAL=1000
MAX_STEPS=500
PRIV_LABELS="data/labeled/math_priv.jsonl"
NOGT_LABELS="data/labeled/math_nogt.jsonl"
EVAL_DATASET="data/processbench_math_shuffled.jsonl"

mkdir -p checkpoints results/ablation logs

# Function to run a single seed
run_seed() {
  local gpu="$1"
  local type="$2"
  local seed="$3"
  local ablation="$4"
  local score_loss="$5"
  local extra_args="$6"
  local dataset="$7"

  local tag="Qwen_Qwen2.5-3B-Instruct_seed${seed}_ms500_${type}"
  local ckpt="checkpoints/${tag}.pt"
  local res_dir="results/ablation/${tag}"
  local log_file="logs/${tag}.log"

  if [[ -f "$res_dir/processbench_results.json" ]]; then
    echo "[GPU $gpu] Skipping $tag (already completed)"
    return
  fi

  echo "[GPU $gpu] Starting $tag"
  {
    echo "=== TRAINING ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.train_slfd \
      --dataset "$dataset" \
      --student_model "$STUDENT_MODEL" \
      --ablation "$ablation" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --seed "$seed" \
      --score_loss "$score_loss" \
      --checkpoint "$ckpt" \
      --max_steps "$MAX_STEPS" \
      $extra_args

    echo "=== EVALUATION ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.run_processbench \
      --student_model "$STUDENT_MODEL" \
      --checkpoint "$ckpt" \
      --dataset "$EVAL_DATASET" \
      --max_samples "$N_EVAL" \
      --results_dir "$res_dir"
  } > "$log_file" 2>&1

  echo "[GPU $gpu] Finished $tag. Results in $res_dir"
}

# GPU 0: Privileged BCE (error_weight=3)
run_priv() {
  for seed in {0..3}; do
    run_seed 0 "priv_bce_ew3" "$seed" "score_only" "bce" "--error_weight 3.0" "$PRIV_LABELS"
  done
}

# GPU 1: No-GT rank-only (balanced_batches)
run_nogt() {
  for seed in {0..3}; do
    run_seed 1 "nogt_rank_bal" "$seed" "score_only" "rank" "--balanced_batches" "$NOGT_LABELS"
  done
}

echo "Starting parallel execution of 8 seeds..."
run_priv &
PID_PRIV=$!

run_nogt &
PID_NOGT=$!

wait $PID_PRIV
wait $PID_NOGT

echo "All parallel seeds completed."
