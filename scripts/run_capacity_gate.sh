#!/usr/bin/env bash
# Capacity gate for Phase B.
#
# Reuses existing Gemma-4 labeled data, trains priv/no-GT PRMs for one student
# model, evaluates ProcessBench, runs transfer CI, and scores a cached same-pool
# BoN candidate set. It writes versioned checkpoints/results and does not delete
# or overwrite prior artifacts unless OVERWRITE=1 is set.
set -euo pipefail

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PY="${PY:-$(command -v python || command -v python3)}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_EVAL="${N_EVAL:-1000}"
N_BOOT="${N_BOOT:-10000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
TRAIN_DTYPE="${TRAIN_DTYPE:-auto}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-}"
SLFD_CUDA_PLACEMENT="${SLFD_CUDA_PLACEMENT:-auto}"
MAX_STEPS="${MAX_STEPS:-}"
ABLATION="${ABLATION:-score_critique}"
MODEL_LR="${MODEL_LR:-1e-4}"
SCORE_LR="${SCORE_LR:-5e-5}"
ALIGN_LR="${ALIGN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LM_WEIGHT="${LM_WEIGHT:-1.0}"
SCORE_WEIGHT="${SCORE_WEIGHT:-1.0}"
HIDDEN_WEIGHT="${HIDDEN_WEIGHT:-1.0}"
SCORE_LOSS="${SCORE_LOSS:-mse}"
ERROR_WEIGHT="${ERROR_WEIGHT:-1.0}"
RANK_MARGIN="${RANK_MARGIN:-1.0}"
BALANCED_BATCHES="${BALANCED_BATCHES:-0}"
OVERWRITE="${OVERWRITE:-0}"
DEV="${DEV:-0}"
DRY_RUN="${DRY_RUN:-0}"

PRIV_LABELS="${PRIV_LABELS:-data/labeled/math_priv.jsonl}"
NOGT_LABELS="${NOGT_LABELS:-data/labeled/math_nogt.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-data/processbench_math_shuffled.jsonl}"

BON_N="${BON_N:-8}"
QUICK_BON="${QUICK_BON:-200}"
FULL_BON="${FULL_BON:-1000}"
RUN_FULL_BON="${RUN_FULL_BON:-auto}"  # auto, 1, or 0
FULL_BON_TRIGGER_MARGIN="${FULL_BON_TRIGGER_MARGIN:--0.005}"
SKIP_BON="${SKIP_BON:-0}"

GEN_MODEL="${GEN_OMLX_MODEL:-google/gemma-2-9b-it}"
PORT="${PORT:-8080}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0,1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
ALLOW_EXISTING_VLLM="${ALLOW_EXISTING_VLLM:-0}"
FORCE_GENERATE_CANDIDATES="${FORCE_GENERATE_CANDIDATES:-0}"

MODEL_TAG="${STUDENT_MODEL//\//_}"
MODEL_TAG="${MODEL_TAG//[^A-Za-z0-9_.-]/_}"
TAG="${RUN_TAG:-${MODEL_TAG}_seed${SEED}}"

PRIV_TAG="${TAG}_priv_critique"
NOGT_TAG="${TAG}_nogt_critique"
PRIV_CKPT="checkpoints/${PRIV_TAG}.pt"
NOGT_CKPT="checkpoints/${NOGT_TAG}.pt"
PRIV_RESULTS="results/ablation/${PRIV_TAG}"
NOGT_RESULTS="results/ablation/${NOGT_TAG}"
CI_OUT="results/ablation/${TAG}_transfer_ci.json"

GEN_TAG="${GEN_MODEL//\//_}"
GEN_TAG="${GEN_TAG//[^A-Za-z0-9_.-]/_}"
CANDIDATES_FILE="${CANDIDATES_FILE:-results/bon_paired/candidates/${GEN_TAG}_n${BON_N}_m${FULL_BON}_t0.8.jsonl}"
QUICK_BON_DIR="results/bon_paired/${TAG}_quick_m${QUICK_BON}"
FULL_BON_DIR="results/bon_paired/${TAG}_full_m${FULL_BON}"
VLLM_LOG="${VLLM_LOG:-vllm_capacity_${TAG}.log}"
VLLM_PID=""

DEVFLAG=()
if [[ "$DEV" == "1" ]]; then
  DEVFLAG=(--dev_mode)
fi

MAX_STEPS_ARGS=()
if [[ -n "$MAX_STEPS" ]]; then
  MAX_STEPS_ARGS=(--max_steps "$MAX_STEPS")
fi

BALANCED_ARGS=()
if [[ "$BALANCED_BATCHES" == "1" ]]; then
  BALANCED_ARGS=(--balanced_batches)
fi

TRAIN_ENV=(env SLFD_CUDA_PLACEMENT="$SLFD_CUDA_PLACEMENT")
if [[ -n "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
  TRAIN_ENV+=(CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES")
fi

cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file missing: $path" >&2
    exit 1
  fi
}

guard_output() {
  local path="$1"
  if [[ "$OVERWRITE" != "1" && -e "$path" ]]; then
    echo "ERROR: output already exists: $path" >&2
    echo "Set OVERWRITE=1 only if you intentionally want to replace this run's artifacts." >&2
    exit 1
  fi
}

require_file "$PRIV_LABELS"
require_file "$NOGT_LABELS"
require_file "$EVAL_DATASET"

guard_output "$PRIV_CKPT"
guard_output "$NOGT_CKPT"
guard_output "$PRIV_RESULTS/processbench_results.json"
guard_output "$NOGT_RESULTS/processbench_results.json"
guard_output "$CI_OUT"
if [[ "$SKIP_BON" != "1" ]]; then
  guard_output "$QUICK_BON_DIR/bon_paired_results.json"
  if [[ "$RUN_FULL_BON" == "1" ]]; then
    guard_output "$FULL_BON_DIR/bon_paired_results.json"
  fi
fi

mkdir -p checkpoints results/ablation results/bon_paired/candidates

echo "== Phase B capacity gate =="
echo "student=${STUDENT_MODEL}"
echo "tag=${TAG}"
echo "seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE} n_eval=${N_EVAL} eval_batch_size=${EVAL_BATCH_SIZE}"
echo "ablation=${ABLATION} model_lr=${MODEL_LR} score_lr=${SCORE_LR}"
echo "train_cuda_visible_devices=${TRAIN_CUDA_VISIBLE_DEVICES:-all-visible} slfd_cuda_placement=${SLFD_CUDA_PLACEMENT}"
echo "loss_weights=lm:${LM_WEIGHT} score:${SCORE_WEIGHT} hidden:${HIDDEN_WEIGHT}"
echo "score_loss=${SCORE_LOSS} error_weight=${ERROR_WEIGHT} rank_margin=${RANK_MARGIN} balanced_batches=${BALANCED_BATCHES}"
echo "priv_labels=${PRIV_LABELS}"
echo "nogt_labels=${NOGT_LABELS}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 set; exiting before training/eval/generation."
  exit 0
fi

run_cell() {
  local labels="$1"
  local checkpoint="$2"
  local results_dir="$3"

  "${TRAIN_ENV[@]}" "$PY" -m experiments.train_slfd \
    --dataset "$labels" \
    --student_model "$STUDENT_MODEL" \
    --ablation "$ABLATION" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --train_dtype "$TRAIN_DTYPE" \
    --model_lr "$MODEL_LR" \
    --score_lr "$SCORE_LR" \
    --align_lr "$ALIGN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --lm_weight "$LM_WEIGHT" \
    --score_weight "$SCORE_WEIGHT" \
    --hidden_weight "$HIDDEN_WEIGHT" \
    --score_loss "$SCORE_LOSS" \
    --error_weight "$ERROR_WEIGHT" \
    --rank_margin "$RANK_MARGIN" \
    --checkpoint "$checkpoint" \
    "${MAX_STEPS_ARGS[@]}" \
    "${BALANCED_ARGS[@]}" \
    "${DEVFLAG[@]}"

  "${TRAIN_ENV[@]}" "$PY" -m experiments.run_processbench \
    --student_model "$STUDENT_MODEL" \
    --checkpoint "$checkpoint" \
    --dataset "$EVAL_DATASET" \
    --max_samples "$N_EVAL" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --results_dir "$results_dir" \
    "${DEVFLAG[@]}"
}

echo "== 1/4 train/eval privileged PRM =="
run_cell "$PRIV_LABELS" "$PRIV_CKPT" "$PRIV_RESULTS"

echo "== 2/4 train/eval no-GT PRM =="
run_cell "$NOGT_LABELS" "$NOGT_CKPT" "$NOGT_RESULTS"

echo "== 3/4 transfer CI =="
"$PY" -m experiments.transfer_ci \
  --priv "$PRIV_RESULTS/per_step_scores.json" \
  --nogt "$NOGT_RESULTS/per_step_scores.json" \
  --n_boot "$N_BOOT" \
  --out "$CI_OUT"

start_vllm() {
  if curl -f -s "http://localhost:${PORT}/v1/models" >/dev/null; then
    if [[ "$ALLOW_EXISTING_VLLM" == "1" ]]; then
      echo "Using existing vLLM server on port ${PORT}; this script will not stop it."
      return
    fi
    echo "ERROR: port ${PORT} already has a vLLM-compatible server." >&2
    echo "Set ALLOW_EXISTING_VLLM=1 to reuse it, or choose a different PORT." >&2
    exit 1
  fi

  echo "Starting vLLM generator ${GEN_MODEL} on port ${PORT}..."
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$GEN_MODEL" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$PORT" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  until curl -f -s "http://localhost:${PORT}/v1/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM crashed before readiness. Check ${VLLM_LOG}." >&2
      exit 1
    fi
    sleep 5
  done
}

ensure_candidates() {
  if [[ -f "$CANDIDATES_FILE" && "$FORCE_GENERATE_CANDIDATES" != "1" ]]; then
    echo "Reusing candidate pool: $CANDIDATES_FILE"
    return
  fi

  start_vllm
  local force_args=()
  if [[ "$FORCE_GENERATE_CANDIDATES" == "1" ]]; then
    force_args=(--force_generate)
  fi

  "$PY" -m experiments.bon_paired \
    --dataset "$EVAL_DATASET" \
    --n "$BON_N" \
    --max_samples "$FULL_BON" \
    --backend omlx \
    --omlx_url "http://localhost:${PORT}/v1" \
    --candidates_file "$CANDIDATES_FILE" \
    --generate_only \
    "${force_args[@]}" \
    "${DEVFLAG[@]}"

  cleanup
  VLLM_PID=""
}

score_bon() {
  local max_samples="$1"
  local results_dir="$2"
  "$PY" -m experiments.bon_paired \
    --dataset "$EVAL_DATASET" \
    --priv "$PRIV_CKPT" \
    --nogt "$NOGT_CKPT" \
    --student_model "$STUDENT_MODEL" \
    --n "$BON_N" \
    --max_samples "$max_samples" \
    --results_dir "$results_dir" \
    --candidates_file "$CANDIDATES_FILE" \
    "${DEVFLAG[@]}"
}

if [[ "$SKIP_BON" == "1" ]]; then
  echo "== 4/4 BoN skipped (SKIP_BON=1) =="
else
  echo "== 4/4 same-pool BoN from cached candidates =="
  ensure_candidates

  echo "Scoring quick BoN gate: max_samples=${QUICK_BON}"
  score_bon "$QUICK_BON" "$QUICK_BON_DIR"

  SHOULD_FULL="$("$PY" - "$QUICK_BON_DIR/bon_paired_results.json" "$FULL_BON_TRIGGER_MARGIN" <<'PY'
import json
import sys

path, margin = sys.argv[1], float(sys.argv[2])
d = json.load(open(path))
best = max(d["prm_rerank_priv"], d["prm_rerank_nogt"])
print("1" if best - d["majority_vote"] >= margin else "0")
PY
)"

  if [[ "$RUN_FULL_BON" == "1" || ( "$RUN_FULL_BON" == "auto" && "$SHOULD_FULL" == "1" ) ]]; then
    guard_output "$FULL_BON_DIR/bon_paired_results.json"
    echo "Scoring full BoN gate: max_samples=${FULL_BON}"
    score_bon "$FULL_BON" "$FULL_BON_DIR"
  else
    echo "Skipping full BoN: quick gate did not meet trigger margin ${FULL_BON_TRIGGER_MARGIN}."
    echo "Set RUN_FULL_BON=1 to force full scoring."
  fi
fi

echo
echo "== Capacity gate complete =="
echo "priv: $PRIV_RESULTS/processbench_results.json"
echo "nogt: $NOGT_RESULTS/processbench_results.json"
echo "ci:   $CI_OUT"
if [[ "$SKIP_BON" != "1" ]]; then
  echo "quick_bon: $QUICK_BON_DIR/bon_paired_results.json"
  echo "full_bon:  $FULL_BON_DIR/bon_paired_results.json (if triggered)"
fi
