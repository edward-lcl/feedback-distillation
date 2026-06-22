# Paper Review: Step-Level Feedback Distillation

Here is a condensed, actionable summary of the major empirical and structural issues to fix before submission:

### 1. Fix the Misleading "Best-of-N" Reporting (Section 2.7)
The paper currently blends an **N=1000 independent-pool** evaluation (`0.373 vs 0.349`) with a McNemar statistical test (`p=0.14`) that was actually run on a completely different **N=200 shared-pool** dataset. 
* **Action:** Explicitly state that Table 5 uses independent candidate generations, and clarify that the McNemar test was a separate, smaller (N=200) run. Note that the N=1000 shared-pool run is queued.

### 2. The Null Result is Confounded by a Weak Student
The paper concludes that teacher privilege does not distill into the student PRM. However, because both trained students (1.5B) fail to beat a simple majority vote baseline (~0.39), the student is fundamentally a weak verifier. 
* **Action:** Acknowledge that the failure to distill may simply be a parametric capacity bottleneck of the 1.5B model, rather than a proof that the privileged signal is inherently unlearnable.

### 3. Add Confidence Intervals to the Step-Level Transfer Claim (Section 2.6)
Relying on a raw `0.01` ROC-AUC delta (`0.641` vs `0.631`) to claim the no-GT student is superior is statistically insufficient for a negative result.
* **Action:** Run the `experiments/transfer_ci.py` script already in your repo to compute a paired bootstrap confidence interval on this gap. Report the CI and p-value to prove the verifiers are statistically indistinguishable.

### 4. Quantify the "Diffuse Signal" Mechanism (Section 4)
The claim that the teacher's privilege "churns labels in both directions" is stated too qualitatively.
* **Action:** Add the hard numbers from your Phase A1 diagnostics: `priv` and `nogt` labels only agree on **69%** of steps, with disagreement being near-symmetric (1,001 vs 956 false positive flags). This firmly grounds the "diffuse gain" argument in empirical data.

### 5. "Sweet Spot" vs. Context Length Limitations
The paper argues that the teacher fails on OlympiadBench because it's too difficult to track.
* **Action:** Clarify whether this failure is due to mathematical intractability, or simply an LLM attention degradation artifact caused by the extreme token length of the OlympiadBench reference solutions.
