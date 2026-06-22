#!/usr/bin/env bash
# Run the WHOLE pipeline on one box (e.g. SSH'd into the oMLX host) when you have
# no separate GPU: generation + labeling go through the LOCAL oMLX server
# (localhost:8000), and student train/eval run on this machine (MPS/CPU/GPU via
# torch). oMLX is inference-only — it serves gen+label; it does NOT train.
#
# Prereqs:
#   - oMLX serving on localhost:8000 with the teacher + a small generator model.
#   - export OMLX_API_KEY=...   (your oMLX key; NOT stored in this script)
#
# Run (fresh data):    OMLX_API_KEY=... ./scripts/run_single_box.sh
# Knobs passed through: N_TRAIN N_EVAL EPOCHS ABLATION SEED STUDENT_MODEL
#                       KD_TEACHER BATCH_SIZE  (see run_student_ablation.sh)
#
# 👉 For the cheap-first runs (verdict/soft/logit_kd + transfer_ci) you do NOT
#    need oMLX at all — reuse the existing labels and skip gen/label entirely:
#       REUSE_LABELS=1 ABLATION=soft ./scripts/run_student_ablation.sh
#    That avoids loading the teacher, so there's zero memory contention with
#    torch training. Prefer it whenever data/labeled/*.jsonl already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OMLX_API_KEY:?export OMLX_API_KEY first (your oMLX key)}"
URL="${OMLX_URL:-http://localhost:8000/v1}"
case "$URL" in *localhost*|*127.0.0.1*) ;; *)
  echo "NOTE: OMLX_URL=$URL is not localhost. Generation is heavy — run this ON the" >&2
  echo "      oMLX host so gen+label hit localhost, not the tunnel." >&2 ;; esac

# Single-endpoint: both generation and labeling use the local oMLX server.
export OMLX_URL="$URL"
export OMLX_MODEL="${OMLX_MODEL:-gemma-4-26b-a4b-it-MLX-4bit}"   # privileged teacher (labeling)
export OMLX_TIMEOUT="${OMLX_TIMEOUT:-600}"
export GEN_OMLX_URL="$URL"
export GEN_OMLX_MODEL="${GEN_OMLX_MODEL:-gemma-3-4b-it-4bit}"    # small served generator
export GEN_BACKEND=omlx

cat >&2 <<EOF
== single-box config ==
  generation : $GEN_OMLX_MODEL  @ $GEN_OMLX_URL
  labeling   : $OMLX_MODEL  @ $OMLX_URL
  train/eval : local torch (this machine)
⚠️  Memory: labeling loads the ~15GB teacher into Metal; training then competes
   for the same unified memory. If you hit OOM / Metal clamp, free the teacher
   before the train phase (oMLX GUI → unload, or let it idle-unload), or train a
   smaller student. Big-N generation on one Mac is slow — keep N modest here.
EOF

exec ./scripts/run_student_ablation.sh "$@"
