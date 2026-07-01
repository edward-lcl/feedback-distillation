#!/usr/bin/env bash
source .venv/bin/activate

echo "Killing old vLLM servers..."
pkill -f vllm.entrypoints.openai.api_server || true
sleep 2

echo "Starting LOCAL vLLM Generator (gemma-2-9b-it) on port 8080..."
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --tensor-parallel-size 2 \
    --port 8080 > vllm_generator.log 2>&1 &
VLLM_PID=$!

echo "Waiting for local Generator to be ready..."
while ! curl -f -s http://localhost:8080/v1/models > /dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed. Check vllm_generator.log."
    exit 1
  fi
  sleep 5
done
echo "Local Generator is ready!"

# ---------------------------------------------------------
# DUAL-ENDPOINT TOPOLOGY SETUP
# ---------------------------------------------------------
# 1. Generation Endpoint (Local vLLM)
export GEN_OMLX_URL="http://localhost:8080/v1"
export GEN_OMLX_MODEL="google/gemma-2-9b-it"

# 2. Teacher Endpoint (Remote MLX)
export OMLX_URL="https://teacher.elcl.systems/v1"
export OMLX_MODEL="gemma-4-26b-a4b-it-MLX-4bit"
export OMLX_API_KEY="<SCRUBBED>"
export OMLX_TIMEOUT=600

# Scale parameters
export N_TRAIN=1000
export N_EVAL=400
export EPOCHS=2

echo "Starting Phase 2 (Ablations) N=1000..."
./scripts/run_student_ablation.sh > phase2.log 2>&1
echo "Phase 2 completed."

# NOTE: run_student_ablation.sh might kill all vLLM servers at the end.
# If it did, restart the local generator for Phase 3!
if ! curl -f -s http://localhost:8080/v1/models > /dev/null; then
    echo "Restarting LOCAL vLLM Generator for Phase 3..."
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
fi

echo "Starting Phase 3 (Best-of-N Re-ranking downstream) N=1000..."
# Phase 3 uses the LOCAL generator to create 8 samples per problem, then scores them using the local student checkpoint.
python -m experiments.bon_rerank --dataset data/processbench_math_shuffled.jsonl \
    --checkpoint checkpoints/priv_critique.pt --n 8 --max_samples 1000 \
    --omlx_url http://localhost:8080/v1 --backend omlx \
    --results_dir results/bon_priv > phase3_priv.log 2>&1

python -m experiments.bon_rerank --dataset data/processbench_math_shuffled.jsonl \
    --checkpoint checkpoints/nogt_critique.pt --n 8 --max_samples 1000 \
    --omlx_url http://localhost:8080/v1 --backend omlx \
    --results_dir results/bon_nogt > phase3_nogt.log 2>&1

echo "All complete. Cleaning up..."
pkill -f vllm.entrypoints.openai.api_server || true
echo "Run finished successfully!"
