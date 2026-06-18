# Distilling Privileged Feedback into Ground-Truth-Free Process Reward Models
**Experimental Summary — verified N=1000 run (updated 2026-06-18)**

> Supersedes the earlier draft of this file (which reported a stale N=300 run with
> a fixed-threshold F1 artifact). Numbers below are read directly from the committed
> result JSONs.

## 1. Experimental Setup
- **Teacher (labeling):** `gemma-4-26b-a4b-it` (MLX 4-bit), served from Edward's box and
  reached over a tunnel — **confirmed: ~32k labeling requests** hit the served teacher.
- **Generator (candidate solutions):** `gemma-2-9b-it`, local vLLM on the 2×3090 box.
- **Student PRM:** `Qwen/Qwen2.5-1.5B-Instruct`.
- **Eval:** ProcessBench MATH, **threshold-free** (`roc_auc`/`pr_auc`) + split diagnostics.

## 2. Phase 1 — Teacher-level privilege: VALIDATED (the spine)
Giving the teacher the GT reference solution measurably improves step-error detection,
**but only in a tractability sweet spot**:
- GSM8K (easy) ≈ 0 · **MATH +0.05** (N=400, 95% CI [0.01, 0.09], significant) · OlympiadBench ≈ 0.
- Richness matters: a bare answer is inert; only the full worked solution helps.
- Cross-family confirmed (Qwen-27B teacher: +0.082 on MATH).

This finding is unchanged and remains the paper's spine.

## 3. Phase 2 — Student PRM ablations (does privilege TRANSFER to the student?)
N_TRAIN=1000, N_EVAL=400, EPOCHS=2. **Compare on `roc_auc` (threshold-free), not F1** —
F1 at a fixed `logit<0` cutoff moves with score-head calibration and is not a capability measure.

| Condition | `roc_auc` | `pr_auc` | F1 | error_recall | pred_error_rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `priv_critique` (privileged + critique) | 0.631 | 0.130 | 0.170 | 0.357 | 0.280 |
| `priv_scoreonly` (privileged, score only) | 0.624 | 0.135 | 0.184 | 0.427 | 0.319 |
| `nogt_critique` (no-GT + critique) | **0.641** | 0.121 | 0.198 | 0.467 | 0.327 |

**Takeaway — privilege does NOT transfer to the student.** The no-GT student is *highest* on
`roc_auc` (0.641 vs 0.631) — and this run is non-degenerate (`pred_error_rate`≈0.3 across cells,
no silent collapse), so it's a clean comparison. (The earlier "0.037 → 0.197 privilege transfers"
headline was a fixed-threshold artifact from a degenerate prior run; it does not reproduce.)

## 4. Phase 3 — Downstream Best-of-N re-ranking (N=1000)
Generator `gemma-2-9b-it`, N=8 candidates. Each student PRM used as a test-time verifier.

| Verifier | pass@1 | `prm_rerank` | `majority_vote` | `oracle_pass@N` |
| :--- | :--- | :--- | :--- | :--- |
| **No-GT student** (`bon_nogt`) | 0.337 | **0.373** | 0.391 | 0.517 |
| **Privileged student** (`bon_priv`) | 0.335 | 0.349 | 0.382 | 0.518 |

(An earlier N=200 run is in `results/bon/`: pass@1 0.345, prm_rerank 0.335, majority 0.395.)

**Takeaway:** the no-GT verifier reranks *better* (0.373 vs 0.349, +3.6 vs +1.4 over pass@1),
and **neither verifier beats majority vote** (~0.39). Pools are matched in difficulty
(pass@1 ≈0.336, oracle ≈0.517 both), so the comparison is fair — but they are *separate*
candidate generations, not one shared pool (see open thread #2).

## 5. Honest conclusion
- **Teacher-level privilege: real and validated.** (Sweet spot, N=400 +0.05, cross-family.)
- **Transfer into a 1.5B student PRM: NOT observed at this scale.** No-GT ≥ privileged on both
  step-level `roc_auc` and downstream re-ranking; neither verifier beats majority vote.

This is a clean, honest negative — and a more interesting contribution once we understand *why*.

## 6. Open threads (next experiments)
1. **Gemma-4 privilege probe** — only the `gemma-2-9b` probe was saved; run the probe through the
   served Gemma-4 teacher to confirm the privileged labels actually differ from no-GT at the
   teacher level (the teacher-level gap is independently validated, but pin it for *this* teacher).
2. **Same-pool paired Phase 3** — re-rank one shared candidate set with both verifiers; report
   absolute accuracy + a paired (McNemar) significance test, not baseline-relative deltas.
3. **Why no transfer?** candidate hypotheses to test:
   - train/eval distribution shift (train on 9b-generated solutions, eval on ProcessBench's);
   - 1.5B student capacity ceiling;
   - label agreement: how often do priv vs no-GT teacher labels actually differ, and on which steps?
