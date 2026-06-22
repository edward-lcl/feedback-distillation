#!/usr/bin/env bash
# Overnight Mac runner — value-ordered, sleep-safe Phase-B queue.
#
# WHY THIS ORDER (see RESEARCH_ROADMAP.md / RUNBOOK_PHASE_B.md):
#   The highest-value cheap work is re-training the N=1000 null across SEEDS and
#   putting a bootstrap CI on the priv−nogt transfer gap (B0b). That answers "is
#   the null even real?" — the question that gates the whole paper. So that runs
#   FIRST and is the DEFAULT. The B4 distillation-method ablations (verdict/soft/
#   logit_kd) are the mechanism slot, gated behind clearing majority vote; they're
#   the lowest-value spend right now, so they're OPT-IN (RUN_B4=1 / RUN_LOGIT_KD=1)
#   and never crowd out the gates.
#
# SLEEP-SAFE: trains on the EXISTING labels (data/labeled/*.jsonl) — no teacher,
#   no generation, no tunnel. If the Mac sleeps mid-run nothing 502s; cells are
#   skip-if-done so you can just re-launch to resume.
#
# NOT INCLUDED (need supervision / the teacher endpoint, run them as their own jobs):
#   - B0 paired McNemar (bon_paired): needs a generation endpoint for the shared
#     candidate pool.
#   - B3 positive control (weak teacher): label with a 2B on a SEPARATE port
#     (mlx_lm.server :8001, point OMLX_URL there) — do NOT stop the :8000 daemon.
#
# RUN (after the current `soft` cell finishes):
#   nohup bash scripts/run_overnight_mac.sh > overnight.log 2>&1 &
# Resume after a sleep/crash: just re-run the same line (done cells are skipped).
#
# KNOBS (env):
#   SEEDS="0 1 2"   N_EVAL=1000   EPOCHS=2   BATCH_SIZE=4   N_BOOT=10000
#   STUDENT_MODEL=  (default Qwen2.5-1.5B)   WAIT_FOR_FREE=1
#   RUN_B4=0        (1 = also run verdict + soft at SEED=0, namespaced by method)
#   RUN_LOGIT_KD=0  (1 = also run the gemma-family logit_kd cell — heavy, loads a live teacher)
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
[ -d "$REPO/experiments" ] || REPO="$PWD"
cd "$REPO"
[ -d experiments ] || { echo "FATAL: run from the feedback-distillation repo root (or set REPO=)" >&2; exit 1; }
PY="$(command -v python || command -v python3)"

SEEDS="${SEEDS:-0 1 2}"
N_EVAL="${N_EVAL:-1000}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_BOOT="${N_BOOT:-10000}"
STUDENT_MODEL="${STUDENT_MODEL:-}"
RUN_B4="${RUN_B4:-0}"
RUN_LOGIT_KD="${RUN_LOGIT_KD:-0}"
SM_ARG=""; [ -n "$STUDENT_MODEL" ] && SM_ARG="--student_model $STUDENT_MODEL"

PRIV=data/labeled/math_priv.jsonl
NOGT=data/labeled/math_nogt.jsonl
EVAL=data/processbench_math_shuffled.jsonl
mkdir -p checkpoints results/ablation results/overnight
LOG=results/overnight/run.log

for f in "$PRIV" "$NOGT" "$EVAL"; do
  [ -f "$f" ] || { echo "FATAL: missing $f — this script trains on EXISTING labels only. Do one full labeling pass first." >&2; exit 1; }
done

log(){ printf '\n=== [%s] %s ===\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$LOG"; }

# Don't fight the currently-running `soft` cell for unified memory — wait it out.
if [ "${WAIT_FOR_FREE:-1}" = "1" ]; then
  while pgrep -f "experiments.train_slfd" >/dev/null 2>&1; do
    log "another train_slfd is running (likely the soft cell) — waiting 5 min to avoid memory contention"
    sleep 300
  done
fi

# train + eval ONE cell.  $1=labeled  $2=ablation  $3=out tag  $4=seed  [$5=kd_teacher  $6=student_model_override]
cell () {
  local data="$1" abl="$2" tag="$3" seed="$4" kd="${5:-}" smo="${6:-}"
  local kd_arg=""; [ -n "$kd" ] && kd_arg="--kd_teacher $kd"
  local sm_arg="$SM_ARG"; [ -n "$smo" ] && sm_arg="--student_model $smo"
  local bs="$BATCH_SIZE"; [ -n "$kd" ] && bs=2   # live KD teacher needs the headroom
  if [ -f "results/ablation/$tag/per_step_scores.json" ]; then
    log "skip $tag (already done)"; return 0
  fi
  log "train $tag  (ablation=$abl seed=$seed${kd:+ kd=$kd})"
  "$PY" -m experiments.train_slfd --dataset "$data" --ablation "$abl" \
      --epochs "$EPOCHS" --batch_size "$bs" --seed "$seed" \
      $sm_arg $kd_arg --checkpoint "checkpoints/$tag.pt"
  log "eval $tag"
  "$PY" -m experiments.run_processbench --checkpoint "checkpoints/$tag.pt" \
      $sm_arg --dataset "$EVAL" --max_samples "$N_EVAL" \
      --results_dir "results/ablation/$tag"
}

ci () {  # $1=priv tag  $2=nogt tag  $3=out json
  log "transfer_ci  $1  vs  $2"
  "$PY" -m experiments.transfer_ci \
     --priv "results/ablation/$1/per_step_scores.json" \
     --nogt "results/ablation/$2/per_step_scores.json" \
     --n_boot "$N_BOOT" --out "$3" 2>&1 | tee -a "$LOG"
}

log "STEP 1 — re-train the N=1000 null across seeds [$SEEDS] + transfer_ci (B0b — highest value)"
for S in $SEEDS; do
  cell "$PRIV" score_critique "priv_null_seed$S" "$S"
  cell "$NOGT" score_critique "nogt_null_seed$S" "$S"
  ci "priv_null_seed$S" "nogt_null_seed$S" "results/ablation/transfer_ci_null_seed$S.json"
done

if [ "$RUN_B4" = "1" ]; then
  log "STEP 2 — B4 distillation-method ablations at SEED=0 (verdict, soft), namespaced BY METHOD"
  for M in verdict soft; do
    cell "$PRIV" "$M" "priv_${M}_seed0" 0
    cell "$NOGT" "$M" "nogt_${M}_seed0" 0
    ci "priv_${M}_seed0" "nogt_${M}_seed0" "results/ablation/transfer_ci_${M}_seed0.json"
  done
fi

if [ "$RUN_LOGIT_KD" = "1" ]; then
  log "STEP 3 — B4a-online logit_kd (gemma-family; loads a LIVE KD teacher — heavy)"
  KDT="${KD_TEACHER:-google/gemma-2-9b-it}"
  SMK="${STUDENT_MODEL:-google/gemma-2-2b-it}"   # MUST be same family as KDT (vocab align)
  cell "$PRIV" logit_kd "priv_logitkd_seed0" 0 "$KDT" "$SMK"
  cell "$NOGT" logit_kd "nogt_logitkd_seed0" 0 "$KDT" "$SMK"
  ci "priv_logitkd_seed0" "nogt_logitkd_seed0" "results/ablation/transfer_ci_logitkd_seed0.json"
fi

log "SUMMARY — roc_auc per cell + transfer_ci gaps"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, glob, os
def auc(name):
    p=f"results/ablation/{name}/processbench_results.json"
    return json.load(open(p)).get("roc_auc") if os.path.exists(p) else None
print(f"{'cell':<26}{'roc_auc':>9}")
for d in sorted(glob.glob("results/ablation/*")):
    if not os.path.isdir(d): continue
    a=auc(os.path.basename(d))
    if a is not None: print(f"{os.path.basename(d):<26}{a:>9.3f}")
print("\ntransfer_ci  (gap = priv - nogt; CI excluding 0 => a real transfer effect):")
for f in sorted(glob.glob("results/ablation/transfer_ci_*.json")):
    r=json.load(open(f))
    print(f"  {os.path.basename(f):<32} gap={r['gap_priv_minus_nogt']:+.4f}  "
          f"ci95={r['ci95']}  p(priv<=nogt)={r['p_one_sided_priv_le_nogt']:.3f}  "
          f"{'SIG' if r['significant_at_95'] else 'ns (indistinguishable from 0)'}")
PYEOF

log "DONE. Push: results/ablation/*/processbench_results.json + results/ablation/transfer_ci_*.json"
log "Do NOT hand-edit RESULTS.md or mark anything validated — Edward verifies the raw JSONs."
