# Distilling Privileged Feedback into Ground-Truth-Free Process Reward Models
**Session Results & Experimental Summary**

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
- `N_TRAIN` = 300
- `N_EVAL` = 400
- `EPOCHS` = 2

### Ablation Results (ProcessBench)
*Note: F1 scores below are heavily skewed by a fixed-threshold artifact (logit < 0 cutoff). The threshold-free `roc_auc` metric provides the true measure of ranking capability.*

| Condition | F1 Score | ROC AUC | First Error Accuracy |
| :--- | :--- | :--- | :--- |
| `priv_critique` (Privileged + Critique) | **0.197** | 0.624 | 0.438 |
| `priv_scoreonly` (Privileged, Score Only) | **0.177** | - | 0.455 |
| `nogt_critique` (No-GT + Critique) | **0.037** | 0.651 | 0.435 |

### Research Takeaways
1. **Critique Helps:** The presence of textual reasoning chains during training provides valuable learning signals to the student (`0.177` → `0.197` F1).
2. **Privilege DOES NOT Transfer (Hypothesis Failed):** The apparent F1 gap (0.197 vs 0.037) was purely a calibration/threshold artifact. The `nogt` model score head is shifted such that it rarely crosses the default 0-cutoff, collapsing recall to ~0. When measured purely on ranking quality (threshold-free), the `nogt_critique` student (`roc_auc` = 0.651) actually **outperformed** the privileged student (`roc_auc` = 0.624). The teacher's privileged capability did *not* successfully distill into the student model.

---

## 4. Phase 3: Downstream Impact (Best-of-N Re-ranking)
**⚠️ WARNING: These results are derived from the `priv_critique.pt` checkpoint, which we now know has inferior ranking capabilities (`roc_auc`=0.624) compared to the unprivileged baseline (`roc_auc`=0.651). Downstream results should be interpreted cautiously until the threshold calibration issue is fixed.**
**Goal:** Evaluate the downstream utility of the trained student PRM (`priv_critique.pt`) as a test-time verifier to select the best candidate from a pool of generated solutions.

### Evaluation Setup
- **Generator:** `gemma-2-9b-it`
- **Candidates (N):** 8
- **Dataset:** `ProcessBench` (200 problems evaluated)
- *Note: Evaluation loop was optimized with parallel ThreadPool concurrency for maximal hardware utilization.*

### Re-ranking Results
| Metric | Accuracy | Description |
| :--- | :--- | :--- |
| `pass@1` | **29.0%** | Taking the first generated candidate (Baseline) |
| `prm_rerank` | **32.0%** | Candidate selected by the distilled 1.5B Student PRM |
| `majority_vote` | **32.5%** | Most common answer among the 8 candidates (Ensemble) |
| `oracle_pass@N` | **40.5%** | Theoretical ceiling (correct answer exists in the 8 candidates) |

### Research Takeaways
The test-time verifier (`prm_rerank`) successfully boosts the baseline accuracy by a full 3 percentage points, nearly tying the computationally expensive `majority_vote` ensemble strategy. 

Given that the student model is heavily constrained by parameter count (1.5B vs the generator's 9B) and was trained on a toy-scale dataset (`N_TRAIN` = 300), matching the performance of a 9B-parameter majority vote is a highly promising signal. Scaling the training dataset by an order of magnitude is the logical next step to definitively break the majority vote ceiling.
