#!/usr/bin/env bash
source .venv/bin/activate
set -euo pipefail

PY="$(command -v python || command -v python3)"

echo "Starting LOCAL vLLM Generator for Phase 3..."
pkill -f vllm.entrypoints.openai.api_server || true
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.70 \
    --port 8080 > vllm_phase3.log 2>&1 &
VLLM_PID=$!
while ! curl -f -s http://localhost:8080/v1/models > /dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed in Phase 3. Check vllm_phase3.log."
    exit 1
  fi
  sleep 5
done

echo "Starting Phase 3 (Best-of-N Re-ranking downstream) N=1000..."
"$PY" -m experiments.bon_rerank --dataset data/processbench_math_shuffled.jsonl \
    --checkpoint checkpoints/priv_critique.pt --n 8 --max_samples 1000 \
    --omlx_url http://localhost:8080/v1 --backend omlx \
    --results_dir results/bon_priv > phase3_priv.log 2>&1

"$PY" -m experiments.bon_rerank --dataset data/processbench_math_shuffled.jsonl \
    --checkpoint checkpoints/nogt_critique.pt --n 8 --max_samples 1000 \
    --omlx_url http://localhost:8080/v1 --backend omlx \
    --results_dir results/bon_nogt > phase3_nogt.log 2>&1

echo "All complete. Cleaning up..."
pkill -f vllm.entrypoints.openai.api_server || true
echo "Phase 3 finished successfully!"
