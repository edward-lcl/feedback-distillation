#!/usr/bin/env bash
set -euo pipefail

# Train a score-only verifier on a ProcessBench-style flat gold-label file, then
# evaluate on a held-out ProcessBench JSONL. This is for cross-config transfer
# gates, not teacher-label ablations.

PY="${PY:-./.venv/bin/python}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RUN_TAG="${RUN_TAG:-gold_scorehead_gate}"
TRAIN_DATASET="${TRAIN_DATASET:-}"
EVAL_DATASET="${EVAL_DATASET:-data/processbench_math_shuffled.jsonl}"
N_EVAL="${N_EVAL:-400}"
MAX_STEPS="${MAX_STEPS:-500}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SEED="${SEED:-0}"
MODEL_LR="${MODEL_LR:-1e-4}"
SCORE_LR="${SCORE_LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCORE_LOSS="${SCORE_LOSS:-bce}"
ERROR_WEIGHT="${ERROR_WEIGHT:-1}"
RANK_MARGIN="${RANK_MARGIN:-1.0}"
BALANCED_BATCHES="${BALANCED_BATCHES:-1}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$TRAIN_CUDA_VISIBLE_DEVICES}"
SLFD_CUDA_PLACEMENT="${SLFD_CUDA_PLACEMENT:-single}"
CHECKPOINT="${CHECKPOINT:-checkpoints/${RUN_TAG}.pt}"
RESULTS_DIR="${RESULTS_DIR:-results/diagnostics/${RUN_TAG}}"

if [[ -z "$TRAIN_DATASET" ]]; then
  echo "TRAIN_DATASET is required" >&2
  exit 2
fi

mkdir -p "$(dirname "$CHECKPOINT")" "$RESULTS_DIR"

train_env=(
  env
  PYTHONUNBUFFERED=1
  TORCH_COMPILE_DISABLE=1
  TORCHDYNAMO_DISABLE=1
  SLFD_CUDA_PLACEMENT="$SLFD_CUDA_PLACEMENT"
)
if [[ -n "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
  train_env+=(CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES")
fi

eval_env=(
  env
  PYTHONUNBUFFERED=1
  TORCH_COMPILE_DISABLE=1
  TORCHDYNAMO_DISABLE=1
  SLFD_CUDA_PLACEMENT="$SLFD_CUDA_PLACEMENT"
)
if [[ -n "$EVAL_CUDA_VISIBLE_DEVICES" ]]; then
  eval_env+=(CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES")
fi

balanced_args=()
if [[ "$BALANCED_BATCHES" == "1" || "$BALANCED_BATCHES" == "true" ]]; then
  balanced_args+=(--balanced_batches)
fi

echo "run_tag=$RUN_TAG"
echo "train_dataset=$TRAIN_DATASET"
echo "eval_dataset=$EVAL_DATASET n_eval=$N_EVAL"
echo "checkpoint=$CHECKPOINT"
echo "results_dir=$RESULTS_DIR"
echo "student_model=$STUDENT_MODEL"
echo "seed=$SEED max_steps=$MAX_STEPS epochs=$EPOCHS batch_size=$BATCH_SIZE eval_batch_size=$EVAL_BATCH_SIZE"
echo "score_loss=$SCORE_LOSS error_weight=$ERROR_WEIGHT rank_margin=$RANK_MARGIN balanced_batches=$BALANCED_BATCHES"
echo "train_cuda_visible_devices=${TRAIN_CUDA_VISIBLE_DEVICES:-all-visible} eval_cuda_visible_devices=${EVAL_CUDA_VISIBLE_DEVICES:-all-visible}"
echo "slfd_cuda_placement=$SLFD_CUDA_PLACEMENT"

"${train_env[@]}" "$PY" -m experiments.train_slfd \
  --dataset "$TRAIN_DATASET" \
  --student_model "$STUDENT_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_steps "$MAX_STEPS" \
  --seed "$SEED" \
  --model_lr "$MODEL_LR" \
  --score_lr "$SCORE_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --ablation score_only \
  --score_loss "$SCORE_LOSS" \
  --error_weight "$ERROR_WEIGHT" \
  --rank_margin "$RANK_MARGIN" \
  "${balanced_args[@]}"

"${eval_env[@]}" "$PY" -m experiments.run_processbench \
  --student_model "$STUDENT_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --dataset "$EVAL_DATASET" \
  --max_samples "$N_EVAL" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --results_dir "$RESULTS_DIR"
