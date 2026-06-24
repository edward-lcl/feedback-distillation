# Phase B Paper Result Card - Full MATH1000 Transfer

Purpose: compact, citation-ready summary of the strongest Phase B result as of
2026-06-24. This is a result card, not the full runbook.

## Claim

Under the same Qwen2.5-3B score-head training/evaluation path, ProcessBench-style
gold step labels from non-MATH source configs transfer strongly to full
ProcessBench MATH, while generated teacher labels transfer weakly.

## Main Table

All rows evaluate on the same 1,000 ProcessBench MATH samples
(`data/processbench_math_shuffled.jsonl`), 6,505 total steps and 594 error
steps. Evaluation is exact serial scoring (`batch_size=1`).

Exact problem-overlap check against the MATH1000 eval split: GSM8K source400
has 0 overlaps; OmniMath source400 has 0 overlaps; OlympiadBench source400 has
0 overlaps.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1* | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0,1 | 0.7600 (0.7439-0.7760) | 0.2249 (0.2234-0.2264) | 0.3313 (0.3213-0.3413) | 0.2564 (0.2421-0.2707) | 0.5119 (0.4776-0.5462) |
| OmniMath ProcessBench gold -> MATH1000 | 0,1 | 0.7665 (0.7539-0.7792) | 0.2455 (0.2277-0.2633) | 0.3265 (0.3145-0.3385) | 0.3069 (0.3052-0.3086) | 0.2719 (0.1797-0.3640) |
| Generated privileged teacher labels, BCE | 0 | 0.5503 | 0.0992 | 0.1803 | 0.0000 | 0.0000 |
| Generated no-GT teacher labels, rank loss | 0 | 0.6324 | 0.1418 | 0.2221 | 0.2151 | 0.3819 |

*Best F1 is threshold-swept on the eval slice. It is useful diagnostically, but
should not be reported as a calibrated claim without a held-out threshold
calibration split.

## Paired Bootstrap Gaps

The main robustness check should use the sequence-cluster bootstrap, which
resamples whole solutions rather than individual steps. `p` is two-sided from
2,000 bootstrap samples.

| Model A | Model B | ROC-AUC gap | 95% CI | p |
| --- | --- | ---: | --- | ---: |
| GSM8K gold seed 0 | generated privileged BCE | +0.1932 | [0.1644, 0.2208] | 0.0010 |
| GSM8K gold seed 1 | generated privileged BCE | +0.2256 | [0.2010, 0.2500] | 0.0010 |
| OmniMath gold seed 0 | generated privileged BCE | +0.2033 | [0.1757, 0.2316] | 0.0010 |
| OmniMath gold seed 1 | generated privileged BCE | +0.2287 | [0.1997, 0.2567] | 0.0010 |
| GSM8K gold seed 0 | best generated-label baseline | +0.1113 | [0.0874, 0.1339] | 0.0010 |
| GSM8K gold seed 1 | best generated-label baseline | +0.1436 | [0.1248, 0.1627] | 0.0010 |
| OmniMath gold seed 0 | best generated-label baseline | +0.1214 | [0.0964, 0.1458] | 0.0010 |
| OmniMath gold seed 1 | best generated-label baseline | +0.1468 | [0.1254, 0.1676] | 0.0010 |

The older step-level bootstrap gives very similar gaps but is less conservative
because steps within a solution are correlated.

## Held-Out Threshold Calibration

Thresholds are chosen to maximize F1 on the first 200 MATH sequences, then
reported on the remaining 800 sequences.

| Model | Calibrated F1 | Eval ROC-AUC | Eval PR-AUC | Pred error rate |
| --- | ---: | ---: | ---: | ---: |
| GSM8K gold seed 0 | 0.3166 | 0.7452 | 0.2339 | 0.2488 |
| GSM8K gold seed 1 | 0.2977 | 0.7707 | 0.2152 | 0.1466 |
| OmniMath gold seed 0 | 0.3062 | 0.7483 | 0.2292 | 0.2742 |
| OmniMath gold seed 1 | 0.3294 | 0.7723 | 0.2606 | 0.2321 |
| Generated privileged BCE | 0.1673 | 0.5490 | 0.1002 | 0.3212 |
| Best generated-label baseline | 0.2085 | 0.6314 | 0.1431 | 0.1695 |

## Interpretation

The model/training path can learn a transferable verifier when supervision is
ProcessBench-style and source-distribution compatible. The failure of generated
privileged/no-GT labels is therefore not explained by Qwen2.5-3B capacity alone;
the current evidence points to label/distribution mismatch.

This should be positioned as a controlled transfer/mismatch result, not as a
state-of-the-art PRM benchmark result.

## Novelty / Scooping Check

Cross-dataset ProcessBench transfer is not entirely untouched. Recent or
concurrent work includes:

- Qwen PRM work showing ProcessBench trends across data sources and strong
  public 7B/72B PRM baselines:
  <https://arxiv.org/pdf/2501.07301>.
- FreePRM, a weak-supervision PRM method without ground-truth process labels,
  reporting ProcessBench F1 by source split:
  <https://arxiv.org/pdf/2506.03570>.
- ProcessLID, an ICLR 2026 under-review internal-reward method with explicit
  transfer tables across GSM8K, MATH, OlympiadBench, and OmniMath:
  <https://openreview.net/pdf?id=5O5AlNVAbs>.
- Preference-based PRM work reporting ProcessBench comparisons for 7B reward
  models and hard/soft/preference labels:
  <https://openreview.net/pdf?id=09Nj40ScvC>.

Positioning implication: do not claim that "cross-config ProcessBench transfer"
alone is novel. The sharper contribution is the controlled mismatch result:
generated privileged/no-GT teacher labels fail badly under the same 3B student,
optimizer, and full-MATH eval path where non-MATH ProcessBench-style gold labels
transfer cleanly.

## Artifact Paths

- GSM8K seed 0: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
- GSM8K seed 1: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
- OmniMath seed 0: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
- OmniMath seed 1: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
- Generated privileged BCE: `results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0/processbench_results.json`
- Best generated-label baseline: `results/diagnostics/generated_rank_nogt_to_math1000_qwen3b_seed0/processbench_results.json`
- Full runbook: `RUNBOOK_PHASE_B_FINDINGS_20260624.md`
