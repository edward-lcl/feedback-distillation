#!/usr/bin/env bash
source .venv/bin/activate
pip install peft bitsandbytes
pkill -f vllm.entrypoints.openai.api_server || true
echo "Starting vllm on port 8001..."
export HF_TOKEN="${HF_TOKEN}"
CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-2-9b-it --tensor-parallel-size 2 --port 8001 > vllm.log 2>&1 &
VLLM_PID=$!
echo "Waiting for vllm to be ready (can take a few minutes)..."
while ! curl -f -s http://localhost:8001/v1/models > /dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed. Check vllm.log."
    exit 1
  fi
  sleep 5
done
echo "vLLM is ready. Starting Phase 3 (bon_rerank.py)..."
OMLX_URL=http://localhost:8001/v1 OMLX_MODEL=google/gemma-2-9b-it python -m experiments.bon_rerank --dataset data/processbench_math_shuffled.jsonl --checkpoint checkpoints/nogt_critique.pt --n 8 --max_samples 200 > bon_rerank.log 2>&1
echo "Phase 3 finished. Killing vLLM..."
kill $VLLM_PID
echo "Done."
