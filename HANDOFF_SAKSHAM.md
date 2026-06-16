# Handoff — Saksham (GPU box, 2×3090 / 48 GB)

**Status: UNBLOCKED (2026-06-16).** The cross-teacher gate passed (Qwen-27B confirms the pattern), so you're clear to run. Start at step 0 below.

_Goal: reproduce the privilege × difficulty result with an **official** Gemma checkpoint at scale. This is the validated, ready-to-run experiment. (Full student training is **not** ready yet — see bottom.)_

## 0. Get the code
```bash
git clone https://github.com/edward-lcl/feedback-distillation.git
cd feedback-distillation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CUDA torch, transformers, datasets, etc.
pip install vllm                          # the teacher server
```

## 1. Serve a teacher (OpenAI-compatible endpoint)
The code talks to any OpenAI-compatible `/v1` endpoint, so vLLM drops in. On 48 GB, a full-precision 27B won't fit — use a **4-bit/AWQ 27B** (tensor-parallel across both GPUs) or fall back to `gemma-2-9b-it`:
```bash
# example — pick a real checkpoint your box can fit:
vllm serve google/gemma-2-27b-it --quantization awq --tensor-parallel-size 2 \
     --port 8000 --api-key sk-local
# simpler/smaller fallback:
# vllm serve google/gemma-2-9b-it --tensor-parallel-size 2 --port 8000 --api-key sk-local
```

## 2. Point the client at it
```bash
export OMLX_URL=http://localhost:8000/v1
export OMLX_MODEL=google/gemma-2-27b-it    # must match the served model id exactly
export OMLX_API_KEY=sk-local               # omit if you served with no key
```

## 3. Run the experiment (one command)
```bash
./scripts/run_privilege_probe.sh           # PB_CONFIG=math  N=150  SEED=0 by default
```
It downloads ProcessBench MATH (GT answer + solution joined), shuffles (seed 0), and runs the 3-condition probe (no-GT / +answer / +full-solution).

## 4. What to report
From `results/teacher_eval_math_<model>/privilege_probe.json`:
- **`gap_solution_f1`** — clearly **> 0** confirms the result holds with the official checkpoint.
- `gap_answer_f1` — expected ≈ 0 (a bare answer is inert).

**Reference (local runs, MATH N=150):** Gemma-4-26b solution gap **+0.07**, Qwen-27B **+0.082**. If your official Gemma lands in that ballpark, the headline is locked with a reproducible, official model.

## Optional / nice-to-have
- Harder set: `PB_CONFIG=olympiadbench ./scripts/run_privilege_probe.sh` (may widen the gap).
- Higher N for tighter CIs: `N=300 ./scripts/run_privilege_probe.sh`.
- A second official family (e.g. a Qwen2.5/3 instruct) for an extra cross-family point.

## Not ready yet (don't run): full student training
The label → train → eval student pipeline (`label_pipeline` → `train_slfd` → `run_processbench`) is **blocked on trainer fixes** (real LoRA wiring + score-head reads the boundary token) — Edward's track. Hold until that lands; this probe is the GPU task for now.
