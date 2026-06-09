#!/bin/bash
# run_local.sh — smoke test SLFD on Apple Silicon / local CPU
# Uses dev-mode models (0.5B student, 1.5B teacher) to fit in 16GB
set -e
echo "SLFD local smoke test — Apple Silicon / dev mode"
export DEV_MODE=1

# Resolve a Python binary (some machines only have python3 on PATH)
PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "No python interpreter found" >&2; exit 1; fi

# Step 1: label 10 samples from a tiny synthetic dataset
"$PY" -m data.label_pipeline \
  --input data/sample_10.jsonl \
  --output /tmp/slfd_labeled_smoke.jsonl \
  --max_samples 10 \
  --dev_mode

# Step 2: run processbench eval with untrained student (baseline)
"$PY" -m experiments.run_processbench \
  --dataset /tmp/slfd_labeled_smoke.jsonl \
  --max_samples 10 \
  --dev_mode \
  --results_dir /tmp/slfd_smoke_results

echo "Smoke test complete. Results in /tmp/slfd_smoke_results/"
