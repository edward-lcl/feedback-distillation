# Runbook — Phase B: make the student a *competent* verifier, then re-test transfer

_Owner: Saksham (GPU box) · Design: Edward · Updated 2026-06-20 · Roadmap context: `RESEARCH_ROADMAP.md`_

## Why this phase exists (read once)
The verified result so far: privilege helps the **teacher** (+0.07) but does **not** transfer into the 1.5B student — and the diagnostics show why (the gain is diffuse: ~31% of labels churn symmetrically). **But there's a catch that gates the whole paper:** the student PRM currently **loses to majority vote** (prm_rerank 0.34/0.375 < 0.39). So right now the null reads as *"privilege doesn't distill into a weak verifier."* To be publishable it must read as *"...into a verifier that actually works."*

**The one bar to clear:** `prm_rerank > majority_vote` on the symbolic-checked Best-of-N at N=1000.

Then we re-ask the real question at the competent config: **does privilege transfer into a student that's actually a good verifier?**

> 🧭 **Order of operations (cheap-first).** Two steps gate the whole paper and need **no new labeling** — run them *first*, in parallel: **B0 paired test**, **B0b multi-seed CI** (is the null even real?), and **B3 positive control** (is the pipeline sensitive to teacher quality at all?). Only then spend compute on **B1** (data) → **B2** (capacity) to clear MV. **B4** (distillation method) is the mechanism slot if B1/B2 don't open the gap. Do **not** message the N=1000 null as "validated" until B0b gives it a CI and the student clears MV — the gap today is 0.641 vs 0.631 `roc_auc` on a single seed.

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

## B0b — is the null statistically real? (do alongside B0, no retrain)
The current "null" is `roc_auc` 0.641 vs 0.631 — **0.01 on one seed**. Before anyone says "validated," put numbers on it:
- **Downstream gap:** the **paired McNemar p** that B0 already prints (same shared pool) *is* the significance test on priv−nogt for re-ranking. Report it. `p > 0.05` ⇒ the gap is not distinguishable from zero — that's the honest status, not "privilege doesn't transfer."
- **Ranking gap (`roc_auc`):** a bootstrap CI on the priv−nogt `roc_auc` difference + a 3-seed retrain are **not yet wired** — that's a Phase D rigor task **owned by Edward** (don't fabricate a `--seed`/bootstrap flag; the eval scripts don't have one yet). Ping Edward to land it; until then, lead with the McNemar p above.

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

## B4 — distillation method (the mechanism slot; run only if B1/B2 don't open the gap)
Hypothesis: we distill via **MSE on a single scalar score** (+ optional NL-critique LM loss); the **logit/distribution loss is wired but disabled** (`loss_flags=[LM,hidden,score,logit]`, `logit=False` in both `--ablation` cells) because the **Qwen student and Gemma teacher have different vocabularies**. If privilege lives in the *shape* of the teacher's distribution, scalar-MSE throws it away and privilege can't transfer at any capacity/data. Two arms:

- **B4a — distribution/logit distillation** ⚠️ *needs trainer code first (owner: Edward).* Enable the KL/logit loss against the teacher's token-level distribution. This **requires a same-family student so vocabs match** — switch the student to **Gemma-2-2B** against the Gemma-4 teacher:
  ```bash
  # available NOW: the student swap (set the family-matched student)
  STUDENT_MODEL=google/gemma-2-2b-it N_TRAIN=<best> N_EVAL=1000 EPOCHS=2 ./scripts/run_student_ablation.sh
  # NOT yet runnable: there is no --ablation distribution / logit-loss CLI path.
  # train_slfd.py must expose the logit loss (loss_flags[3]) + teacher-logit plumbing first. Ping Edward.
  ```
  This is the **only** mechanism for the null we currently have zero coverage of — highest-value *why* experiment.
- **B4c — structured-verdict critique** (cheaper, preprocessing-level). Instead of training the 1.5B to *generate* the 26B teacher's free-text critique (capacity-hopeless), extract the teacher's **binary verdict + reason category** and train on those structured labels. Needs a labeling/preprocessing variant in the label step (not just an env knob) — scope with Edward before running.
- *(Held: contrastive/triplet distillation on priv-vs-nogt pairs — novel but speculative; revisit only if B4a/B4c are inconclusive.)*

## Decision gates
- **Any (data × capacity) config clears `prm_rerank > majority_vote`** → competent student. **Then the headline experiment:** at that config, does priv beat nogt (paired McNemar at N=1000)? → "does privilege transfer into a *competent* verifier."
- **No (data × capacity) config beats MV even at 7B + 10k** → before reframing, try **B4 (distillation method)** — the bottleneck may be the *loss* (scalar-MSE), not capacity/data. Only if B4a/B4c also fail to make the student competent do we reframe to the **teacher-only** paper (the sweet-spot finding stands alone). Either way it's publishable; this tells us which.

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
5. Labeling 10k via the served teacher is the slow part (sequential, ~hours). The teacher now **auto-restarts on crash/reboot** (launchd), but it can't serve while Edward's Mac is **asleep** — a `502`/`530` (Cloudflare `1033`) from `teacher.elcl.systems` almost always means the Mac slept. **Text Edward** (number's in Slack) to wake it; resume the run once you get HTTP 200 from `/v1/models`.
