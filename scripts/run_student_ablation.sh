#!/usr/bin/env bash
# THE paper run: distill the GT-free student and run the two key ablations.
#   (a) score-only vs score+critique  -> does distilling the NL critique help?
#   (b) privileged vs no-GT labels     -> does the teacher's privilege transfer?
#
# Two endpoints, by design (see HANDOFF_SAKSHAM.md "Teacher topology"):
#   GENERATION (expensive, long outputs) -> a small vLLM model on YOUR GPU box.
#       GEN_OMLX_URL / GEN_OMLX_MODEL  (default: fall back to OMLX_URL/OMLX_MODEL)
#   LABELING (cheap, short outputs, NEEDS the privileged teacher) -> Edward's
#       served MLX teacher over the tunnel.  OMLX_URL / OMLX_MODEL / OMLX_API_KEY
# Single-endpoint mode still works: set only OMLX_URL/OMLX_MODEL and both use it.
# Run:    ./scripts/run_student_ablation.sh
# Tune:   N_TRAIN=300 N_EVAL=400 EPOCHS=2  (DEV=1 for a tiny local smoke)
set -euo pipefail
PY="$(command -v python || command -v python3)"
N_TRAIN="${N_TRAIN:-300}"; N_EVAL="${N_EVAL:-400}"; EPOCHS="${EPOCHS:-2}"
GEN_BACKEND="${GEN_BACKEND:-omlx}"
# Generation (the EXPENSIVE step) must run on a LOCAL small model, never the
# served teacher. Default the generator to gemma-2-9b-it; never inherit the
# teacher's model. And REFUSE to route generation at a remote teacher host:
# if OMLX_URL is remote but GEN_OMLX_URL is unset, that would silently send
# thousands of long generations to the teacher (slow; hammers Edward's Mac).
GEN_OMLX_MODEL="${GEN_OMLX_MODEL:-google/gemma-2-9b-it}"
GEN_OMLX_URL="${GEN_OMLX_URL:-}"
GEN_OMLX_API_KEY="${GEN_OMLX_API_KEY:-${OMLX_API_KEY:-}}"
if [ -z "$GEN_OMLX_URL" ]; then
  case "${OMLX_URL:-}" in
    ""|*localhost*|*127.0.0.1*)
      GEN_OMLX_URL="${OMLX_URL:-http://localhost:8000/v1}"   # single-box / local mode: fine
      ;;
    *)
      echo "ERROR: OMLX_URL is a remote teacher ($OMLX_URL) but GEN_OMLX_URL is unset." >&2
      echo "       Generation must run on YOUR local model, not the teacher host. Set e.g.:" >&2
      echo "         export GEN_OMLX_URL=http://localhost:8000/v1" >&2
      echo "         export GEN_OMLX_MODEL=google/gemma-2-9b-it" >&2
      exit 1
      ;;
  esac
fi
DEVFLAG=""; [ "${DEV:-0}" = "1" ] && DEVFLAG="--dev_mode"
mkdir -p data/raw data/labeled checkpoints results/ablation

echo "== 1/4  Data: MATH train problems + ProcessBench MATH eval (shuffled) =="
"$PY" -m scripts.download_data --train_source math --n "$N_TRAIN" --output data/raw/math_train.jsonl
"$PY" -m scripts.download_data --processbench --config math --output data/processbench_math.jsonl
"$PY" -m scripts.shuffle_jsonl --input data/processbench_math.jsonl \
    --output data/processbench_math_shuffled.jsonl --seed 0

echo "== 2/4  Generate candidate solutions (mix of correct/incorrect) — via GEN endpoint =="
OMLX_URL="$GEN_OMLX_URL" OMLX_MODEL="$GEN_OMLX_MODEL" OMLX_API_KEY="${GEN_OMLX_API_KEY:-${OMLX_API_KEY:-}}" \
"$PY" -m scripts.generate_solutions --input data/raw/math_train.jsonl \
    --output data/raw/math_sampled.jsonl --backend "$GEN_BACKEND" \
    ${GEN_OMLX_URL:+--omlx_url "$GEN_OMLX_URL"} --k 4 $DEVFLAG

export OMLX_LABEL_MAX_TOKENS=50
echo "== 3/4  Label twice — privileged (solution) and no-GT — via TEACHER endpoint ($OMLX_URL) =="
"$PY" -m data.label_pipeline --input data/raw/math_sampled.jsonl \
    --output data/labeled/math_priv.jsonl --use_omlx --omlx_url "$OMLX_URL" --privilege solution
"$PY" -m data.label_pipeline --input data/raw/math_sampled.jsonl \
    --output data/labeled/math_nogt.jsonl --use_omlx --omlx_url "$OMLX_URL" --privilege none

echo "== 4/4  Train + eval the ablation cells =="
echo "Killing vLLM to free 44GB VRAM for student training..."
pkill -f vllm.entrypoints.openai.api_server || true

run_cell () {  # $1=labeled  $2=ablation  $3=tag
  "$PY" -m experiments.train_slfd --dataset "$1" --ablation "$2" --epochs "$EPOCHS" \
      --checkpoint "checkpoints/$3.pt" $DEVFLAG
  "$PY" -m experiments.run_processbench --checkpoint "checkpoints/$3.pt" \
      --dataset data/processbench_math_shuffled.jsonl --max_samples "$N_EVAL" \
      --results_dir "results/ablation/$3" $DEVFLAG
}
run_cell data/labeled/math_priv.jsonl score_critique priv_critique     # privileged + critique
run_cell data/labeled/math_priv.jsonl score_only     priv_scoreonly    # privileged, score only
run_cell data/labeled/math_nogt.jsonl score_critique nogt_critique     # no-GT + critique

echo; echo "== RESULTS =="
echo "NOTE: compare on roc_auc / pr_auc (threshold-free) + the split, NOT f1 alone."
echo "      f1/first_error_acc move with the fixed logit<0 cutoff; pred_err≈0 ⇒ silent/degenerate cell."
printf "%-16s %7s %7s %8s %8s %10s %9s %10s\n" cell f1 roc_auc pr_auc err_rec clean_spec pred_err first_acc
for d in priv_critique priv_scoreonly nogt_critique; do
  f="results/ablation/$d/processbench_results.json"
  [ -f "$f" ] && "$PY" -c "import json;d=json.load(open('$f'));g=lambda k:(d.get(k) if d.get(k) is not None else float('nan'));print('%-16s %7.3f %7.3f %8.3f %8.3f %10.3f %9.3f %10.3f'%('$d',g('f1'),g('roc_auc'),g('pr_auc'),g('error_recall'),g('clean_specificity'),g('pred_error_rate'),g('first_error_acc')))"
done
echo
echo "Ablation (a) critique helps?       priv_critique vs priv_scoreonly  -> compare roc_auc/pr_auc"
echo "Ablation (b) privilege transfers?  priv_critique vs nogt_critique   -> compare roc_auc/pr_auc"
echo "Sanity: if nogt_critique pred_err≈0 & err_rec≈0, its low f1 is a silent-collapse artifact, not capability."
