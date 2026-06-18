#!/usr/bin/env bash
source .venv/bin/activate
set -euo pipefail

PY="$(command -v python || command -v python3)"
N_EVAL="${N_EVAL:-400}"
EPOCHS="${EPOCHS:-2}"
DEVFLAG=""

echo "Evaluating priv_critique..."
"$PY" -m experiments.run_processbench --checkpoint "checkpoints/priv_critique.pt" \
    --dataset data/processbench_math_shuffled.jsonl --max_samples "$N_EVAL" \
    --results_dir "results/ablation/priv_critique" $DEVFLAG

run_cell () {  # $1=labeled  $2=ablation  $3=tag
  "$PY" -m experiments.train_slfd --dataset "$1" --ablation "$2" --epochs "$EPOCHS" \
      --checkpoint "checkpoints/$3.pt" $DEVFLAG
  "$PY" -m experiments.run_processbench --checkpoint "checkpoints/$3.pt" \
      --dataset data/processbench_math_shuffled.jsonl --max_samples "$N_EVAL" \
      --results_dir "results/ablation/$3" $DEVFLAG
}

echo "Running cell: priv_scoreonly"
run_cell data/labeled/math_priv.jsonl score_only priv_scoreonly

echo "Running cell: nogt_critique"
run_cell data/labeled/math_nogt.jsonl score_critique nogt_critique

echo; echo "== RESULTS =="
printf "%-16s %7s %7s %8s %8s %10s %9s %10s\n" cell f1 roc_auc pr_auc err_rec clean_spec pred_err first_acc
for d in priv_critique priv_scoreonly nogt_critique; do
  f="results/ablation/$d/processbench_results.json"
  [ -f "$f" ] && "$PY" -c "import json;d=json.load(open('$f'));g=lambda k:(d.get(k) if d.get(k) is not None else float('nan'));print('%-16s %7.3f %7.3f %8.3f %8.3f %10.3f %9.3f %10.3f'%('$d',g('f1'),g('roc_auc'),g('pr_auc'),g('error_recall'),g('clean_specificity'),g('pred_error_rate'),g('first_error_acc')))"
done

echo "Restarting LOCAL vLLM Generator for Phase 3..."
pkill -f vllm.entrypoints.openai.api_server || true
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.70 \
    --port 8080 > vllm_phase3_resume.log 2>&1 &
VLLM_PID=$!
while ! curl -f -s http://localhost:8080/v1/models > /dev/null; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed in Phase 3. Check vllm_phase3_resume.log."
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
echo "Run finished successfully!"
