#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

PY="$(command -v python || command -v python3)"
N_EVAL="${N_EVAL:-1000}"
N_BON="${N_BON:-1000}"
BON_N="${BON_N:-8}"
GEN_MODEL="${GEN_OMLX_MODEL:-google/gemma-2-9b-it}"
PORT="${PORT:-8080}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_LOG="${VLLM_LOG:-vllm_phaseb_b0.log}"
B0_LOG="${B0_LOG:-phaseb_b0_bon_paired.log}"
B0B_EVAL_LOG="${B0B_EVAL_LOG:-phaseb_b0b_nogt_eval.log}"
TRANSFER_CI_LOG="${TRANSFER_CI_LOG:-phaseb_b0b_transfer_ci.log}"
VLLM_PID=""

cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "== Phase B gate =="
echo "B0b first: regenerate nogt ProcessBench sidecar, then transfer CI."
echo "B0 next: local vLLM generator + same-pool paired BoN."

echo
echo "== B0b/1: eval nogt_critique on ProcessBench N=${N_EVAL} =="
"${PY}" -m experiments.run_processbench \
  --checkpoint checkpoints/nogt_critique.pt \
  --dataset data/processbench_math_shuffled.jsonl \
  --max_samples "${N_EVAL}" \
  --results_dir results/ablation/nogt_critique \
  > "${B0B_EVAL_LOG}" 2>&1

echo "== B0b/2: transfer CI priv - nogt =="
"${PY}" -m experiments.transfer_ci \
  --priv results/ablation/priv_critique/per_step_scores.json \
  --nogt results/ablation/nogt_critique/per_step_scores.json \
  --n_boot 10000 \
  --out results/ablation/transfer_ci.json \
  > "${TRANSFER_CI_LOG}" 2>&1

echo
echo "== B0/1: start local vLLM generator ${GEN_MODEL} on port ${PORT} =="
CUDA_VISIBLE_DEVICES=0,1 "${PY}" -m vllm.entrypoints.openai.api_server \
  --model "${GEN_MODEL}" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --port "${PORT}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

until curl -f -s "http://localhost:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM crashed before readiness. Check ${VLLM_LOG}."
    exit 1
  fi
  sleep 5
done

echo "== B0/2: same-pool paired BoN N=${N_BON}, candidates=${BON_N} =="
OMLX_URL="http://localhost:${PORT}/v1" OMLX_MODEL="${GEN_MODEL}" \
"${PY}" -m experiments.bon_paired \
  --dataset data/processbench_math_shuffled.jsonl \
  --priv checkpoints/priv_critique.pt \
  --nogt checkpoints/nogt_critique.pt \
  --n "${BON_N}" \
  --max_samples "${N_BON}" \
  --omlx_url "http://localhost:${PORT}/v1" \
  --results_dir results/bon_paired \
  > "${B0_LOG}" 2>&1

echo
echo "== Phase B gate complete =="
echo "B0b: results/ablation/transfer_ci.json"
echo "B0:  results/bon_paired/bon_paired_results.json"
