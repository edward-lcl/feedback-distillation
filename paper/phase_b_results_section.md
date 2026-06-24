# Phase B Result Section Draft

This is a paste-ready manuscript section for the current strongest Phase B
result. It should be treated as a draft: the claim is intentionally narrower
than a state-of-the-art PRM benchmark claim.

## Result: Gold Process Supervision Transfers, Generated Teacher Labels Do Not

We use the same student architecture, optimizer, and ProcessBench-MATH
evaluation path to compare two kinds of supervision for a small verifier:
ProcessBench-style gold step labels from non-MATH source splits, and generated
teacher labels from our privileged/no-ground-truth labeling pipeline. The key
question is whether the weak generated-label transfer is a student-capacity
failure or a supervision/distribution mismatch.

The evidence points to supervision/distribution mismatch. A Qwen2.5-3B score
head trained on ProcessBench-style gold labels from GSM8K or OmniMath transfers
reliably to full ProcessBench MATH. Under the same full-MATH evaluation,
generated teacher-label checkpoints remain much weaker.

### Experimental Setup

All models are evaluated on the same shuffled ProcessBench MATH split:
1,000 solutions, 6,505 total reasoning steps, and 594 error steps. Evaluation is
exact serial scoring with `batch_size=1`; batched scoring was not used for
headline numbers because tiny score perturbations can affect ranking metrics.

Gold-source verifier training uses `Qwen/Qwen2.5-3B-Instruct` with a score-only
head, LoRA, BCE loss, balanced batches, and 500 training steps. The source
training files contain the first 400 ProcessBench examples from the source
configuration, flattened into per-step labels:

- GSM8K: 2,082 source steps.
- OmniMath: 3,384 source steps.
- OlympiadBench: 3,579 source steps.

Exact problem-string overlap against the MATH1000 eval split is zero for all
three source files.

### Main Result

The manuscript tables below are also generated from the raw result JSONs by
`python -m experiments.summarize_phase_b_tables --out_dir paper/generated`.
The generated markdown and LaTeX versions live in `paper/generated/`.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1* | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) | 0.3144 (0.2864-0.3413) | 0.2503 (0.2137-0.2747) | 0.5111 (0.2949-0.7259) |
| OmniMath ProcessBench gold -> MATH1000 | 0-3 | 0.7694 (0.7539-0.7869) | 0.2524 (0.2277-0.2877) | 0.3276 (0.3145-0.3385) | 0.3055 (0.2830-0.3254) | 0.2920 (0.1797-0.3791) |
| Generated privileged teacher labels, BCE | 0 | 0.5503 | 0.0992 | 0.1803 | 0.0000 | 0.0000 |
| Generated no-GT teacher labels, rank loss | 0 | 0.6324 | 0.1418 | 0.2221 | 0.2151 | 0.3819 |

*Best F1 is swept on the full eval slice and should be treated as diagnostic.
For calibrated threshold numbers, see the held-out calibration table below.

The gold-source rows are not just better than the weakest generated-label
checkpoint. They also beat the strongest generated-label checkpoint we found in
the fast gates: the no-GT rank-loss checkpoint at 0.6324 ROC-AUC. This matters
because it shows the training/evaluation path can produce a competent
ProcessBench-MATH verifier when the supervision is compatible, while our
generated teacher labels do not transfer comparably.

As an external calibration point, we also evaluated the public
`Qwen/Qwen2.5-Math-7B-PRM800K` checkpoint on the same MATH1000 split using the
model-card separator-token scoring convention. This model is not directly
comparable as a training recipe because it is a larger public PRM trained on
PRM800K, but it establishes that our numbers are not state of the art:

| Public baseline | ROC-AUC | PR-AUC | Best F1 | Fixed F1 | Pred error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Math-7B-PRM800K | 0.8379 | 0.3254 | 0.3991 | 0.3953 | 0.1436 |

Qwen PRM800K beats our strongest single gold-source seed, OmniMath seed 3, by
+0.0509 ROC-AUC under sequence-cluster bootstrap (95% CI [0.0330, 0.0682],
p=0.0010). It beats the best generated-label baseline by +0.2055 ROC-AUC
(95% CI [0.1810, 0.2309], p=0.0010).

Optional score averaging improves the gold-source verifier without additional
training. Averaging the four GSM8K seeds gives 0.7832 ROC-AUC, averaging the
four OmniMath seeds gives 0.7865 ROC-AUC, and averaging all eight GSM8K+OmniMath
seeds gives 0.8073 ROC-AUC and 0.2935 PR-AUC. This remains below Qwen PRM800K,
but it suggests the single-seed gold-source result is not saturated. Treat this
as a diagnostic or appendix result, not the main claim.

### Paired Robustness Check

Because ProcessBench steps from the same solution are correlated, the primary
uncertainty check resamples whole solutions rather than individual steps. The
table below reports paired sequence-cluster bootstrap ROC-AUC gaps on the same
1,000 MATH solutions, using 2,000 bootstrap samples.

| Model A | Model B | ROC-AUC gap | 95% CI | p |
| --- | --- | ---: | --- | ---: |
| GSM8K gold seed 0 | generated privileged BCE | +0.1932 | [0.1644, 0.2208] | 0.0010 |
| GSM8K gold seed 1 | generated privileged BCE | +0.2256 | [0.2010, 0.2500] | 0.0010 |
| GSM8K gold seed 2 | generated privileged BCE | +0.2099 | [0.1829, 0.2347] | 0.0010 |
| GSM8K gold seed 3 | generated privileged BCE | +0.1749 | [0.1494, 0.1995] | 0.0010 |
| OmniMath gold seed 0 | generated privileged BCE | +0.2033 | [0.1757, 0.2316] | 0.0010 |
| OmniMath gold seed 1 | generated privileged BCE | +0.2287 | [0.1997, 0.2567] | 0.0010 |
| OmniMath gold seed 2 | generated privileged BCE | +0.2072 | [0.1773, 0.2362] | 0.0010 |
| OmniMath gold seed 3 | generated privileged BCE | +0.2366 | [0.2090, 0.2632] | 0.0010 |
| GSM8K gold seed 0 | best generated-label baseline | +0.1113 | [0.0874, 0.1339] | 0.0010 |
| GSM8K gold seed 1 | best generated-label baseline | +0.1436 | [0.1248, 0.1627] | 0.0010 |
| GSM8K gold seed 2 | best generated-label baseline | +0.1279 | [0.1067, 0.1488] | 0.0010 |
| GSM8K gold seed 3 | best generated-label baseline | +0.0930 | [0.0694, 0.1156] | 0.0010 |
| OmniMath gold seed 0 | best generated-label baseline | +0.1214 | [0.0964, 0.1458] | 0.0010 |
| OmniMath gold seed 1 | best generated-label baseline | +0.1468 | [0.1254, 0.1676] | 0.0010 |
| OmniMath gold seed 2 | best generated-label baseline | +0.1253 | [0.1026, 0.1487] | 0.0010 |
| OmniMath gold seed 3 | best generated-label baseline | +0.1546 | [0.1305, 0.1773] | 0.0010 |

Every GSM8K and OmniMath gold-source seed significantly beats both generated
teacher-label baselines under the sequence-cluster bootstrap.

### Held-Out Threshold Calibration

Thresholded F1 is unstable if the threshold is chosen on the same data used for
evaluation. To report a more honest F1-style metric, we choose the threshold
that maximizes F1 on the first 200 MATH sequences and evaluate it on the
remaining 800.

| Model | Calibrated F1 | Eval ROC-AUC | Eval PR-AUC | Pred error rate |
| --- | ---: | ---: | ---: | ---: |
| GSM8K gold seed 0 | 0.3166 | 0.7452 | 0.2339 | 0.2488 |
| GSM8K gold seed 1 | 0.2977 | 0.7707 | 0.2152 | 0.1466 |
| GSM8K gold seed 2 | 0.2969 | 0.7517 | 0.2261 | 0.2559 |
| GSM8K gold seed 3 | 0.2714 | 0.7233 | 0.1940 | 0.3155 |
| OmniMath gold seed 0 | 0.3062 | 0.7483 | 0.2292 | 0.2742 |
| OmniMath gold seed 1 | 0.3294 | 0.7723 | 0.2606 | 0.2321 |
| OmniMath gold seed 2 | 0.3148 | 0.7529 | 0.2327 | 0.2056 |
| OmniMath gold seed 3 | 0.3316 | 0.7855 | 0.2818 | 0.1987 |
| Qwen2.5-Math-7B-PRM800K | 0.3855 | 0.8372 | 0.3242 | 0.1091 |
| Generated privileged BCE | 0.1673 | 0.5490 | 0.1002 | 0.3212 |
| Best generated-label baseline | 0.2085 | 0.6314 | 0.1431 | 0.1695 |

The calibrated F1 result is consistent with the threshold-free ranking result:
the gold-source verifiers remain well above the generated-label baselines.

### Source Boundary Diagnostic

OlympiadBench is not a third clean headline source in the current experiments.
It is high-variance over two seeds:

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1 | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OlympiadBench ProcessBench gold -> MATH1000 | 0,1 | 0.6509 (0.5854-0.7163) | 0.1619 (0.1209-0.2029) | 0.2384 (0.1918-0.2850) | 0.1938 (0.1636-0.2240) | 0.3661 (0.1399-0.5923) |

Seed 0 is significantly below the best generated-label baseline by
-0.0466 ROC-AUC, 95% CI [-0.0747, -0.0184], p=0.0010. Seed 1 is significantly
above it by +0.0835 ROC-AUC, 95% CI [0.0579, 0.1083], p=0.0010. This supports
the source-distribution framing: ProcessBench-style supervision can transfer,
but the source split matters.

### Interpretation

These results separate student capacity from supervision quality. The same
Qwen2.5-3B score-head path that fails to learn a strong transferable verifier
from our generated teacher labels can learn one from ProcessBench-style gold
labels. The failure of privileged/no-ground-truth generated labels is therefore
not explained by the 3B student or by the ProcessBench evaluation machinery
alone. The current best explanation is label and distribution mismatch between
the generated teacher-labeled training data and the ProcessBench-MATH target.

### What We Should Not Claim

- Do not claim state-of-the-art PRM performance. Public 7B/72B PRMs and recent
  weak-supervision PRM methods report much stronger ProcessBench headline
  numbers; on our same MATH1000 split, Qwen2.5-Math-7B-PRM800K reaches
  0.8379 ROC-AUC.
- Do not claim that cross-source ProcessBench transfer itself is wholly novel;
  recent work already studies PRM data sources and cross-source behavior.
- Do not claim that OlympiadBench cleanly transfers based on the current two
  seeds. It is a boundary diagnostic.

The sharper claim is controlled: generated privileged/no-ground-truth teacher
labels fail under the same student, optimizer, and full-MATH evaluation path
where compatible ProcessBench-style gold labels transfer.

### Artifact Pointers

- Result card: `PHASE_B_PAPER_RESULT_CARD.md`
- Full runbook: `RUNBOOK_PHASE_B_FINDINGS_20260624.md`
- Active status: `ACTIVE_RUN_STATUS.md`
- Public baseline evaluator: `experiments/eval_qwen_prm800k_processbench.py`
- Table generator: `experiments/summarize_phase_b_tables.py`
- Generated tables: `paper/generated/`
- Main MATH1000 result JSONs:
  - `results/diagnostics/qwen_prm800k_math1000/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed2/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed3/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed2/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed3/processbench_results.json`
  - `results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0/processbench_results.json`
  - `results/diagnostics/generated_rank_nogt_to_math1000_qwen3b_seed0/processbench_results.json`
