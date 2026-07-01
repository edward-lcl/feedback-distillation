#!/usr/bin/env bash
# Preliminary Control (Accelerated small-scale B3 run)
# Compares Privileged Teacher vs Weak Teacher on exactly 100 sequences to ensure apples-to-apples

set -euo pipefail
PY="./.venv/bin/python"
WEAK_TEACHER="Qwen/Qwen2.5-3B-Instruct"

echo "== Preparing 100-sample Preliminary Dataset =="
mkdir -p data/raw data/labeled checkpoints results/prelim_control

# Take exactly 100 sequences from the full priv dataset
head -n 100 data/labeled/math_priv.jsonl > data/labeled/math_priv_prelim.jsonl

echo "== 1/4 Training Privileged Teacher Baseline (Apples-to-Apples) =="
$PY -m experiments.train_slfd --dataset data/labeled/math_priv_prelim.jsonl \
    --ablation score_critique --epochs 2 --batch_size 4 \
    --checkpoint checkpoints/priv_control_prelim.pt

echo "== 2/4 Labeling with WEAK Teacher (vLLM Accelerated on 2x GPUs) =="
# Start vLLM in the background, utilizing both 24GB GPUs
# We set gpu_memory_utilization to 0.8 to ensure we don't completely lock out the system
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
$PY -m data.label_pipeline --input data/labeled/math_priv_prelim.jsonl \
    --output data/labeled/math_weak_prelim.jsonl \
    --use_omlx --omlx_url "http://localhost:8000/v1" --privilege solution

echo "Shutting down vLLM server to free VRAM for PyTorch student training..."
kill $VLLM_PID
wait $VLLM_PID || true

echo "== 3/4 Distilling Weak Labels into Student =="
$PY -m experiments.train_slfd --dataset data/labeled/math_weak_prelim.jsonl \
    --ablation score_critique --epochs 2 --batch_size 4 \
    --checkpoint checkpoints/weak_control_prelim.pt

echo "== 4/4 Evaluating Both Preliminary Models =="
$PY -m experiments.run_processbench --checkpoint checkpoints/priv_control_prelim.pt \
    --dataset data/processbench_math_shuffled.jsonl --max_samples 400 \
    --results_dir results/prelim_control/priv_student

$PY -m experiments.run_processbench --checkpoint checkpoints/weak_control_prelim.pt \
    --dataset data/processbench_math_shuffled.jsonl --max_samples 400 \
    --results_dir results/prelim_control/weak_student

echo "== Preliminary Test Complete! =="
