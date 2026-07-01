#!/usr/bin/env bash
# Positive Control (B3): Strong vs Weak Teacher Distillation
# Validates whether the student can actually learn to differentiate between good and bad targets.

set -euo pipefail
PY="./.venv/bin/python"

# Define our weak teacher (e.g. 3B model) to prove sensitivity
WEAK_TEACHER="Qwen/Qwen2.5-3B-Instruct"

mkdir -p data/raw data/labeled checkpoints results/positive_control

echo "== 1/3 Generate candidate solutions =="
# (Using existing generated math_priv.jsonl to save compute)
cp data/labeled/math_priv.jsonl data/raw/math_sampled_control.jsonl || true

echo "== 2/3 Label with WEAK teacher (vLLM Accelerated on 2x GPUs) =="
# Start vLLM in the background across both 24GB GPUs
$PY -m vllm.entrypoints.openai.api_server \
    --model "$WEAK_TEACHER" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --port 8000 > vllm_server.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to boot up..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM server is online!"

export OMLX_API_KEY="dummy"
export OMLX_MODEL="$WEAK_TEACHER"
$PY -m data.label_pipeline --input data/raw/math_sampled_control.jsonl \
    --output data/labeled/math_weak_teacher.jsonl \
    --use_omlx --omlx_url "http://localhost:8000/v1" --privilege solution

echo "Shutting down vLLM server to free VRAM for PyTorch student training..."
kill $VLLM_PID
wait $VLLM_PID || true

echo "== 3/3 Distill weak labels into student =="
$PY -m experiments.train_slfd --dataset data/labeled/math_weak_teacher.jsonl \
    --ablation score_critique --epochs 2 --batch_size 4 \
    --checkpoint checkpoints/weak_control.pt

echo "== Evaluate Weak Student =="
$PY -m experiments.run_processbench --checkpoint checkpoints/weak_control.pt \
    --dataset data/processbench_math_shuffled.jsonl \
    --results_dir results/positive_control/weak_student
