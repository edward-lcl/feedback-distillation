#!/usr/bin/env bash
# Run independent ProcessBench-gold score-head gates across two GPUs.
#
# Job file format: tab-separated rows with columns:
#   run_tag  train_dataset  seed  n_eval  max_steps  [eval_dataset]  [student_model]
#
# Example:
#   JOB_FILE=/tmp/gold_jobs.tsv DRY_RUN=1 ./scripts/run_gold_scorehead_dual_gpu_queue.sh
#
# The underlying gate writes:
#   checkpoints/${run_tag}.pt
#   results/diagnostics/${run_tag}/processbench_results.json
#
# Existing outputs are skipped by default. Set SKIP_EXISTING=0 OVERWRITE=1 only
# when intentionally replacing a unique run tag.
set -euo pipefail

JOB_FILE="${JOB_FILE:-${1:-}}"
GPUS="${GPUS:-0,1}"
LOG_DIR="${LOG_DIR:-.}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_EVAL_DATASET="${DEFAULT_EVAL_DATASET:-data/processbench_math_shuffled.jsonl}"
DEFAULT_STUDENT_MODEL="${DEFAULT_STUDENT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DEFAULT_N_EVAL="${DEFAULT_N_EVAL:-1000}"
DEFAULT_MAX_STEPS="${DEFAULT_MAX_STEPS:-500}"

if [[ -z "$JOB_FILE" || ! -f "$JOB_FILE" ]]; then
  cat >&2 <<'USAGE'
Usage:
  JOB_FILE=jobs.tsv ./scripts/run_gold_scorehead_dual_gpu_queue.sh

jobs.tsv columns, tab separated:
  run_tag  train_dataset  seed  n_eval  max_steps  [eval_dataset]  [student_model]

Useful env:
  GPUS=0,1                 GPU worker list
  DRY_RUN=1                print commands without running
  SKIP_EXISTING=1          skip existing checkpoints/results (default)
  LOG_DIR=.                where per-job logs go
USAGE
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "ERROR: GPUS must name at least one GPU, e.g. GPUS=0,1" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

declare -a JOB_LINES=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line//[[:space:]]/}" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  JOB_LINES+=("$line")
done < "$JOB_FILE"

if [[ "${#JOB_LINES[@]}" -eq 0 ]]; then
  echo "ERROR: no jobs found in $JOB_FILE" >&2
  exit 2
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_one() {
  local gpu="$1"
  local line="$2"
  local run_tag train_dataset seed n_eval max_steps eval_dataset student_model

  IFS=$'\t' read -r run_tag train_dataset seed n_eval max_steps eval_dataset student_model <<< "$line"
  eval_dataset="${eval_dataset:-$DEFAULT_EVAL_DATASET}"
  student_model="${student_model:-$DEFAULT_STUDENT_MODEL}"
  n_eval="${n_eval:-$DEFAULT_N_EVAL}"
  max_steps="${max_steps:-$DEFAULT_MAX_STEPS}"

  if [[ -z "$run_tag" || -z "$train_dataset" || -z "$seed" ]]; then
    echo "ERROR: malformed job row: $line" >&2
    return 2
  fi
  if [[ ! -f "$train_dataset" ]]; then
    echo "ERROR: train dataset missing for $run_tag: $train_dataset" >&2
    return 2
  fi
  if [[ ! -f "$eval_dataset" ]]; then
    echo "ERROR: eval dataset missing for $run_tag: $eval_dataset" >&2
    return 2
  fi

  local checkpoint="checkpoints/${run_tag}.pt"
  local results_dir="results/diagnostics/${run_tag}"
  local results_json="${results_dir}/processbench_results.json"
  local log="${LOG_DIR}/phaseb_${run_tag}.log"

  if [[ -e "$checkpoint" || -e "$results_json" ]]; then
    if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
      echo "[$(timestamp)] skip existing run_tag=$run_tag checkpoint=$checkpoint results=$results_json"
      return 0
    fi
    if [[ "$OVERWRITE" != "1" ]]; then
      echo "ERROR: output exists for $run_tag; set SKIP_EXISTING=1 or OVERWRITE=1" >&2
      return 2
    fi
  fi

  local cmd=(
    env
    RUN_TAG="$run_tag"
    TRAIN_DATASET="$train_dataset"
    EVAL_DATASET="$eval_dataset"
    STUDENT_MODEL="$student_model"
    SEED="$seed"
    N_EVAL="$n_eval"
    MAX_STEPS="$max_steps"
    CHECKPOINT="$checkpoint"
    RESULTS_DIR="$results_dir"
    TRAIN_CUDA_VISIBLE_DEVICES="$gpu"
    EVAL_CUDA_VISIBLE_DEVICES="$gpu"
    SLFD_CUDA_PLACEMENT=single
    PYTHONUNBUFFERED=1
    ./scripts/run_gold_scorehead_gate.sh
  )

  echo "[$(timestamp)] gpu=$gpu run_tag=$run_tag log=$log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "$log" 2>&1
  echo "[$(timestamp)] done run_tag=$run_tag"
}

worker() {
  local gpu="$1"
  local start_index="$2"
  local stride="${#GPU_LIST[@]}"
  local status=0
  local i
  for ((i = start_index; i < ${#JOB_LINES[@]}; i += stride)); do
    run_one "$gpu" "${JOB_LINES[$i]}" || status=$?
  done
  return "$status"
}

echo "[$(timestamp)] launching ${#JOB_LINES[@]} jobs across GPUs: $GPUS"

declare -a PIDS=()
for i in "${!GPU_LIST[@]}"; do
  worker "${GPU_LIST[$i]}" "$i" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done

exit "$status"
