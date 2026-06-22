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

echo "== 2/3 Label with WEAK teacher =="
# In this experiment, we run the label pipeline locally with the weak teacher
$PY -m data.label_pipeline --input data/raw/math_sampled_control.jsonl \
    --output data/labeled/math_weak_teacher.jsonl \
    --local_model "$WEAK_TEACHER" --privilege solution

echo "== 3/3 Distill weak labels into student =="
$PY -m experiments.train_slfd --dataset data/labeled/math_weak_teacher.jsonl \
    --ablation score_critique --epochs 2 --batch_size 4 \
    --checkpoint checkpoints/weak_control.pt

echo "== Evaluate Weak Student =="
$PY -m experiments.run_processbench --checkpoint checkpoints/weak_control.pt \
    --dataset data/processbench_math_shuffled.jsonl \
    --results_dir results/positive_control/weak_student
