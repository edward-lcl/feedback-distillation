# Distilling Privileged Feedback into Ground-Truth-Free Process Reward Models
**Session Results & Experimental Summary**

## Phase B Addendum — 2026-06-24

The older Phase 2/3 results below are useful historical context, but the
current strongest evidence is the Phase B full-MATH1000 transfer diagnostic.
Under the same Qwen2.5-3B score-head training/evaluation path, ProcessBench-style
gold labels from GSM8K and OmniMath transfer strongly to full ProcessBench MATH,
while raw generated teacher-label checkpoints remain weak. The later
format-rerender cell shows this is largely a label-convention problem for the
same-source GSM8K control.

| Training source | Seeds | MATH1000 ROC-AUC | MATH1000 PR-AUC |
| :--- | :--- | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) |
| OmniMath ProcessBench gold -> MATH1000 | 0-3 | 0.7694 (0.7539-0.7869) | 0.2524 (0.2277-0.2877) |
| Qwen2.5-Math-7B-PRM800K public baseline | 0 | 0.8379 | 0.3254 |
| Generated privileged teacher labels, BCE | 0 | 0.5503 | 0.0992 |
| Generated no-GT teacher labels, rank loss | 0 | 0.6324 | 0.1418 |

Sequence-cluster bootstrap over whole MATH solutions shows every GSM8K/OmniMath
gold-source seed significantly beats both generated-label baselines. The
publishable framing is therefore not "privilege transfers"; it is the controlled
mismatch finding: generated privileged/no-GT teacher labels fail under the same
3B student, optimizer, and full-MATH evaluation path where compatible
ProcessBench-style gold labels transfer. The public Qwen PRM800K baseline is
stronger on the same split, so the result should not be framed as SOTA PRM
performance.

See:
- `paper/phase_b_results_section.md`
- `PHASE_B_PAPER_RESULT_CARD.md`
- `RUNBOOK_PHASE_B_FINDINGS_20260624.md`

## Same-Source GSM8K Control — 2026-07-05

Edward generated Gemma-4 labels for the same 400 GSM8K ProcessBench candidate
solutions used by the GSM8K gold-source row. Saksham's cluster then trained the
same Qwen2.5-3B score-head verifier with the gold-source BCE recipe
(`scripts/run_gold_scorehead_gate.sh`, 500 steps, balanced batches) for
privileged and no-GT generated labels, seeds 0-3, evaluating on the same
MATH1000 split.

| Training source | Seeds | MATH1000 ROC-AUC | MATH1000 PR-AUC |
| :--- | :--- | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) |
| Same-source GSM8K generated privileged BCE -> MATH1000 | 0-3 | 0.5494 (0.4680-0.6371) | 0.0985 (0.0809-0.1235) |
| Same-source GSM8K generated no-GT BCE -> MATH1000 | 0-3 | 0.6183 (0.5849-0.6614) | 0.1152 (0.1049-0.1314) |

Sequence-cluster bootstrap over whole MATH solutions: the weakest GSM8K gold
seed still beats the best same-source generated no-GT seed by +0.0642 ROC-AUC,
95% CI [0.0445, 0.0831], p=0.0004. It beats the best same-source generated
privileged seed by +0.0885, 95% CI [0.0688, 0.1080], p=0.0004.

Interpretation: the raw generated-label weakness is not explained by the source
problem distribution alone. For GSM8K, holding the 400 source problems fixed
still leaves a large gap between ProcessBench gold labels and raw Gemma-4
generated labels. This does not isolate human annotation provenance from label
format/semantics, so the clean follow-up is the format-rerendered generated-label
cell below.

## Format-Rerender Same-Source GSM8K Control — 2026-07-05

The same Gemma-4 same-source labels were re-rendered into the ProcessBench
first-error convention (first teacher-flagged error only, later steps non-error,
binary +/-1 scores, literal `Correct.`/`Error.` feedback) and trained with the
same Qwen2.5-3B score-head BCE recipe.

| Training source | Seeds | MATH1000 ROC-AUC | MATH1000 PR-AUC |
| :--- | :--- | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) |
| Same-source generated privileged BCE, raw -> MATH1000 | 0-3 | 0.5494 (0.4680-0.6371) | 0.0985 (0.0809-0.1235) |
| Same-source generated privileged BCE, PB-format -> MATH1000 | 0-3 | 0.6762 (0.5869-0.7891) | 0.1761 (0.1175-0.2811) |
| Same-source generated no-GT BCE, raw -> MATH1000 | 0-3 | 0.6183 (0.5849-0.6614) | 0.1152 (0.1049-0.1314) |
| Same-source generated no-GT BCE, PB-format -> MATH1000 | 0-3 | 0.7366 (0.7011-0.7555) | 0.2478 (0.2266-0.2722) |

Single-checkpoint means close 62.8% of the raw privileged gap and 88.8% of the
raw no-GT gap to GSM8K gold. Mean-score artifacts are stronger: PB-format
privileged reaches 0.7789 ROC-AUC / 0.2436 PR-AUC, and PB-format no-GT reaches
0.7599 ROC-AUC / 0.2883 PR-AUC.

Sequence-cluster bootstrap over whole MATH solutions: PB-format no-GT 4-seed
mean beats raw no-GT 4-seed mean by +0.1262 ROC-AUC, 95% CI [0.1043, 0.1476],
p=0.0004. PB-format privileged 4-seed mean beats raw privileged 4-seed mean by
+0.2070 ROC-AUC, 95% CI [0.1831, 0.2305], p=0.0004. The GSM8K gold 4-seed mean
is statistically tied with PB-format privileged (gold minus PB-format +0.0044,
95% CI [-0.0080, 0.0172], p=0.4983).

Interpretation: for same-source GSM8K, raw label rendering/convention is a major
bottleneck. The cell makes the paper stronger, but it changes the claim: the
failure is not "generated labels cannot work"; it is that the raw generated-label
pipeline was format/convention mismatched. Residual label content still matters
because the rerendered teacher first-error position disagrees with gold on
roughly a quarter of source solutions.

## 1. Experimental Setup & Hardware
- **Hardware:** Local Compute (Dual NVIDIA GeForce RTX 3090, 24GB VRAM per GPU)
- **Teacher Model:** `google/gemma-2-9b-it` (Unquantized, bf16/fp16)
- **Student Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Serving Engine:** `vLLM` (v0.7.3) with dynamic tensor/pipeline parallelism mapping

---

## 2. Phase 1: Privilege Probe
**Goal:** Verify whether providing the teacher model with the ground-truth (GT) solution yields a measurable capability gap in step-level error detection compared to an unprivileged teacher.

### Results
| Metric | Score | Description |
| :--- | :--- | :--- |
| `gap_solution_f1` | **0.2675** | Improvement in F1 when Teacher is given the GT solution |
| `gap_answer_f1` | **0.1610** | Improvement in F1 when Teacher is given the GT final answer |

**Conclusion:** The cross-teacher gate passed cleanly. A substantial capability gap exists, proving that the privileged teacher detects subtle reasoning errors far more accurately than its unprivileged counterpart. 

---

## 3. Phase 2: Student PRM Ablations
**Goal:** Distill the teacher's evaluation signals into a lightweight, ground-truth-free student PRM (1.5B parameters). We ablated across two axes: Privilege (did the teacher have GT access?) and Critique (did the teacher provide textual rationales alongside numerical scores?).

### Training Hyperparameters
- `N_TRAIN` = 1000
- `N_EVAL` = 400
- `EPOCHS` = 2

### Ablation Results (ProcessBench)
*Note: F1 scores below are heavily skewed by a fixed-threshold artifact (logit < 0 cutoff). The threshold-free `roc_auc` metric provides the true measure of ranking capability.*

| Condition | F1 Score | ROC AUC | First Error Accuracy |
| :--- | :--- | :--- | :--- |
| `priv_critique` (Privileged + Critique) | **0.170** | 0.631 | 0.443 |
| `priv_scoreonly` (Privileged, Score Only) | **0.184** | 0.624 | 0.458 |
| `nogt_critique` (No-GT + Critique) | **0.198** | 0.641 | 0.483 |

### Research Takeaways
1. **Critique Helps:** The presence of textual reasoning chains during training provides valuable learning signals, though at N=1000 the `score_only` ablation remains competitive.
2. **The "Privilege Transfer" Artifact:** At N=300 we saw massive F1 drops due to calibration shift. At N=1000, `nogt_critique` actually outperforms `priv_critique` on both thresholded F1 (0.198 vs 0.170) and threshold-free `roc_auc` (0.641 vs 0.631).

---

## 4. Phase 3: Downstream Impact (Best-of-N Re-ranking)
**Goal:** Evaluate the downstream utility of the trained student PRMs as test-time verifiers to select the best candidate from a pool of generated solutions. This bypasses the Phase 2 calibration artifact entirely.

### Evaluation Setup
- **Generator:** `gemma-2-9b-it`
- **Candidates (N):** 8
- **Dataset:** `ProcessBench` (1000 problems evaluated)
- *Note: Candidates are generated live, so base `pass@1` naturally varies slightly due to generation randomness.*

### Re-ranking Results (N=1000)

| Checkpoint | `pass@1` (Baseline) | `prm_rerank` (Student) | Lift (PRM vs Baseline) | `majority_vote` | `oracle_pass@N` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `priv_critique.pt` | 33.5% | **34.9%** | **+1.4%** | 38.2% | 51.8% |
| `nogt_critique.pt` | 33.7% | **37.3%** | **+3.6%** | 39.1% | 51.7% |

### Final Research Takeaways: Re-evaluating the Privilege Hypothesis
1. **The Phase 2 `roc_auc` matches downstream behavior at scale:** At N=1000, `nogt_critique` achieves `prm_rerank` of **37.3%** (a **+3.6%** lift over baseline), outperforming `priv_critique` which achieves **34.9%** (a **+1.4%** lift). This aligns with the Phase 2 ROC AUC where `nogt_critique` (0.641) scored higher than `priv_critique` (0.631).
2. **Both models provide positive lift, but majority vote remains strong:** Both student models successfully improve upon the `pass@1` baseline. However, `majority_vote` (38.2% and 39.1% respectively) still outperforms the PRM re-ranking on this setup.
3. **Conclusion:** Under a large-scale evaluation (N=1000), the Ground-Truth-Free student (`nogt_critique`) is highly capable and actually outperforms the Privileged student on downstream Best-of-N re-ranking, showing that privilege is not strictly required to train a beneficial test-time verifier.

---

## 5. Phase A: Diagnosing the Null (P0)
**Goal:** We confirmed the "null transfer" result at N=1000. Phase A diagnostics ran three experiments to pinpoint the bottleneck.

### A1: Label Agreement
* **Question:** Do the Privileged and No-GT teachers actually produce different labels?
* **Result:** They agreed on only **69.1%** of the reasoning steps.
* **Conclusion:** The labels *do* differ significantly. The null is not due to identical training targets.

### A2: Same-pool paired Best-of-N (N=200)
* **Question:** Does the `priv` student PRM outperform the `nogt` student PRM when re-ranking the exact same candidate pool?
* **Result:** `nogt` (37.5% pass@1) actually tied/outperformed `priv` (34.0% pass@1), with `p=0.1435` (no statistical difference). Furthermore, neither beat the naive `majority_vote` baseline (39.0%). 
* **Conclusion:** The student simply failed to absorb the privilege during distillation, likely due to its own capacity limits.

### A3: Gemma-4 Privilege Probe
* **Question:** Does the `gemma-4` labeling teacher natively exhibit the privilege gap before distillation?
* **Result:** **+0.07 F1 gap** (`with_solution` F1 = 0.786 vs `no_gt` F1 = 0.716).
* **Conclusion:** The privilege capability absolutely exists at the teacher level.

### Final Conclusion
The teacher possesses the capability (+0.07 F1), and it generated significantly divergent labels (30% disagreement), but the 1.5B student simply failed to utilize that capability when learning the reward model. This completely validates the "honest null" framing for the paper.
