#!/bin/bash
# run_experiment.sh — full SLFD experiment loop, real models via local oMLX teacher.
#
# Teacher labeling + solution generation run against your local oMLX server (no
# 72B download, no API credits). Student training + eval run locally on
# MPS/CUDA/CPU. Set OMLX_API_KEY to your server's key first.
#
#   OMLX_API_KEY=... ./run_experiment.sh
#
# Stage 0 (teacher validation) is the go/no-go gate: if with-GT F1 is weak, fix
# teacher prompting before burning compute on training. Skip with SKIP_GATE=1.
# Override sizes/paths via env: N_TRAIN, N_EVAL, N_GATE, EPOCHS, PB_CONFIG.
set -euo pipefail

PY="$(command -v python || command -v python3)"
# Local dev oracle: official Qwen MoE arch (A3B = 3B active), community nvfp4
# quant — noted in paper appendix. Paper teacher remains Qwen2.5-Math-72B
# (device.py PROD_MODELS) for the final labeling pass on rented GPU.
export OMLX_MODEL="${OMLX_MODEL:-Qwen3.6-35B-A3B-nvfp4}"
N_TRAIN="${N_TRAIN:-300}"
N_EVAL="${N_EVAL:-400}"
N_GATE="${N_GATE:-50}"
EPOCHS="${EPOCHS:-2}"
PB_CONFIG="${PB_CONFIG:-gsm8k}"   # gsm8k config has GT answers auto-joined

mkdir -p data/raw data/labeled checkpoints results

echo "== 1/7  Download data (GSM8K train source + ProcessBench eval) =="
"$PY" -m scripts.download_data --train_source gsm8k --n "$N_TRAIN" \
    --output data/raw/gsm8k_train.jsonl
"$PY" -m scripts.download_data --processbench --config "$PB_CONFIG" \
    --output "data/processbench_${PB_CONFIG}.jsonl"

if [ "${SKIP_GATE:-0}" != "1" ]; then
  echo "== 2/7  GATE: validate teacher on ProcessBench (ceiling + privileged gap) =="
  "$PY" -m experiments.eval_teacher \
      --dataset "data/processbench_${PB_CONFIG}.jsonl" \
      --backend omlx --max_samples "$N_GATE"
  echo ">>> Inspect results/teacher_eval/teacher_eval.json before continuing."
  echo ">>> Weak with_gt F1? Fix teacher prompting first. Re-run with SKIP_GATE=1 to proceed."
  read -r -p "Continue to data generation + training? [y/N] " ans
  [ "$ans" = "y" ] || exit 0
fi

echo "== 3/7  Generate balanced correct/incorrect solutions (oMLX) =="
"$PY" -m scripts.generate_solutions \
    --input data/raw/gsm8k_train.jsonl \
    --output data/raw/gsm8k_sampled.jsonl \
    --backend omlx --k 4

echo "== 4/7  Label steps with the teacher (oMLX) =="
"$PY" -m data.label_pipeline \
    --input data/raw/gsm8k_sampled.jsonl \
    --output data/labeled/gsm8k_labeled.jsonl \
    --use_omlx

echo "== 5/7  Label QA + train/dev split =="
"$PY" -m scripts.label_qa --input data/labeled/gsm8k_labeled.jsonl
"$PY" -m scripts.split_jsonl --input data/labeled/gsm8k_labeled.jsonl \
    --train_out data/labeled/train.jsonl --dev_out data/labeled/dev.jsonl

echo "== 6/7  Train the student (offline distillation) + save checkpoint =="
"$PY" -m experiments.train_slfd \
    --dataset data/labeled/train.jsonl \
    --checkpoint checkpoints/slfd_student.pt \
    --epochs "$EPOCHS"

echo "== 7/7  Evaluate on ProcessBench (GT-free at test time) =="
"$PY" -m experiments.run_processbench \
    --dataset "data/processbench_${PB_CONFIG}.jsonl" \
    --checkpoint checkpoints/slfd_student.pt \
    --max_samples "$N_EVAL" \
    --results_dir results/processbench

echo "Done. Teacher gate: results/teacher_eval/ — student: results/processbench/"
