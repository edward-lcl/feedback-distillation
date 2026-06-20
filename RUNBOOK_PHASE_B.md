# Runbook — Phase B: make the student a *competent* verifier, then re-test transfer

_Owner: Saksham (GPU box) · Design: Edward · Updated 2026-06-20 · Roadmap context: `RESEARCH_ROADMAP.md`_

## Why this phase exists (read once)
The verified result so far: privilege helps the **teacher** (+0.07) but does **not** transfer into the 1.5B student — and the diagnostics show why (the gain is diffuse: ~31% of labels churn symmetrically). **But there's a catch that gates the whole paper:** the student PRM currently **loses to majority vote** (prm_rerank 0.34/0.375 < 0.39). So right now the null reads as *"privilege doesn't distill into a weak verifier."* To be publishable it must read as *"...into a verifier that actually works."*

**The one bar to clear:** `prm_rerank > majority_vote` on the symbolic-checked Best-of-N at N=1000.

Then we re-ask the real question at the competent config: **does privilege transfer into a student that's actually a good verifier?**

## Prereqs (same env as the N=1000 run)
```bash
git pull                                  # main has math_verify + STUDENT_MODEL/BATCH_SIZE knobs
# generation (your local small model)
export GEN_OMLX_URL=http://localhost:8000/v1
export GEN_OMLX_MODEL=google/gemma-2-9b-it
# labeling (Edward's served Gemma-4 teacher)
export OMLX_URL=https://teacher.elcl.systems/v1
export OMLX_MODEL=gemma-4-26b-a4b-it-MLX-4bit
export OMLX_API_KEY=<from Edward>   ;  export OMLX_TIMEOUT=600
```

## B0 — close the underpowered paired test (no retrain, do first)
```bash
python -m experiments.bon_paired --dataset data/processbench_math_shuffled.jsonl \
  --priv checkpoints/priv_critique.pt --nogt checkpoints/nogt_critique.pt --n 8 --max_samples 1000
```
Push `results/bon_paired/`. (A2 was N=200, p=0.14 — underpowered. N=1000 makes the null defensible.)

## B1 — scale training data (hold student = 1.5B)
```bash
for N in 5000 10000; do
  N_TRAIN=$N N_EVAL=1000 EPOCHS=2 ./scripts/run_student_ablation.sh
  python -m experiments.bon_paired --dataset data/processbench_math_shuffled.jsonl \
    --priv checkpoints/priv_critique.pt --nogt checkpoints/nogt_critique.pt --n 8 --max_samples 1000
done
```
Report per cell: `roc_auc` + `prm_rerank` vs `majority_vote`. Does more data lift the student over MV?

## B2 — capacity sweep (hold data at the best N from B1)
```bash
for SM in Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  STUDENT_MODEL=$SM N_TRAIN=<best> N_EVAL=1000 EPOCHS=2 BATCH_SIZE=2 ./scripts/run_student_ablation.sh
  python -m experiments.bon_paired --student_model $SM --dataset data/processbench_math_shuffled.jsonl \
    --priv checkpoints/priv_critique.pt --nogt checkpoints/nogt_critique.pt --n 8 --max_samples 1000
done
```
(1.5/3/7B all fit on 2×3090. Lower `BATCH_SIZE` for 7B if OOM.) This doubles as the **capacity arm** of the boundary sweep: does privilege start to transfer at higher capacity?

## B3 — positive control (sensitivity check — critical for the null)
Confirm the pipeline *can* detect a teacher-quality difference when one exists. Re-label with a deliberately **weak** teacher and compare the resulting student to the Gemma-4-labeled one:
```bash
# weak-teacher labels (point labeling at a small model instead of Gemma-4)
OMLX_URL=<weak teacher, e.g. a 2B> OMLX_MODEL=<weak id> \
  python -m data.label_pipeline --input data/raw/math_sampled.jsonl \
  --output data/labeled/math_weak.jsonl --use_omlx --privilege solution
# train a student on weak labels, eval, BoN — compare to the Gemma-4 student
```
**Expectation:** student-from-Gemma-4 > student-from-weak. If it does → the pipeline is sensitive, so the *privilege* null is real (not insensitivity). If even strong-vs-weak doesn't move the student → the student/eval is the bottleneck (stay in B1/B2).

## Decision gates
- **Any (data × capacity) config clears `prm_rerank > majority_vote`** → competent student. **Then the headline experiment:** at that config, does priv beat nogt (paired McNemar at N=1000)? → "does privilege transfer into a *competent* verifier."
- **No config beats MV even at 7B + 10k** → the small-student approach caps out → we reframe to the **teacher-only** paper (the sweet-spot finding stands alone). Either way it's publishable; this tells us which.

## What to push every run (and what NOT to do)
- ✅ Push raw JSONs: `results/ablation/*/processbench_results.json`, `results/bon_paired/bon_paired_results.json`. Commit to a branch, drop the name in the channel.
- ✅ One-line channel summary: `config (N, student) → roc_auc priv/nogt, prm_rerank vs MV`.
- ❌ Do **not** hand-edit `RESULTS.md` or write conclusions — Edward verifies the raw JSONs and propagates.
- ❌ Do **not** mark anything "validated/FINISHED."

## Guardrails (for you and your agent)
1. **Compare cells on `roc_auc` (threshold-free), never F1** — F1 moves with score-head calibration.
2. A cell is a "win" only if `prm_rerank > majority_vote`. Below MV is **not** a result.
3. If eval prints `⚠️ EVAL HEALTH WARNING`, that cell is degenerate — don't report its F1.
4. **One variable at a time** (data OR capacity), so effects are attributable.
5. Labeling 10k via the served teacher is the slow part (sequential, ~hours) — ping Edward if `teacher.elcl.systems` 502s (means his Mac slept).
