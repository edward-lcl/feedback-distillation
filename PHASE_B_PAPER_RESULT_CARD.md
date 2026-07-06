# Phase B Paper Result Card - Full MATH1000 Transfer

Purpose: compact, citation-ready summary of the strongest Phase B result as of
2026-06-24. This is a result card, not the full runbook.

## Claim

Under the same Qwen2.5-3B score-head training/evaluation path, ProcessBench-style
gold step labels from GSM8K and OmniMath transfer strongly to full
ProcessBench MATH, while generated teacher labels transfer weakly. A follow-up
OlympiadBench check is high-variance, supporting a source-distribution framing
rather than a blanket claim that every non-MATH source transfers equally. The
same-source GSM8K generated-label control is also negative: Gemma-4 labels on
the same 400 GSM8K ProcessBench candidate solutions still fall well below the
GSM8K gold row.

## Main Table

All rows evaluate on the same 1,000 ProcessBench MATH samples
(`data/processbench_math_shuffled.jsonl`), 6,505 total steps and 594 error
steps. Evaluation is exact serial scoring (`batch_size=1`).

Exact problem-overlap check against the MATH1000 eval split: GSM8K source400
has 0 overlaps; OmniMath source400 has 0 overlaps; OlympiadBench source400 has
0 overlaps.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1* | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) | 0.3144 (0.2864-0.3413) | 0.2503 (0.2137-0.2747) | 0.5111 (0.2949-0.7259) |
| OmniMath ProcessBench gold -> MATH1000 | 0-3 | 0.7694 (0.7539-0.7869) | 0.2524 (0.2277-0.2877) | 0.3276 (0.3145-0.3385) | 0.3055 (0.2830-0.3254) | 0.2920 (0.1797-0.3791) |
| Generated privileged teacher labels, BCE | 0 | 0.5503 | 0.0992 | 0.1803 | 0.0000 | 0.0000 |
| Generated no-GT teacher labels, rank loss | 0 | 0.6324 | 0.1418 | 0.2221 | 0.2151 | 0.3819 |

*Best F1 is threshold-swept on the eval slice. It is useful diagnostically, but
should not be reported as a calibrated claim without a held-out threshold
calibration split.

Same-source GSM8K generated-label control (completed 2026-07-05): labels were
run locally against the Gemma-4 teacher on the same 400 GSM8K ProcessBench
candidate solutions used by the gold row, matched back to GSM8K references for
the privileged condition. Student training used the same gold-source BCE recipe.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1* | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) | 0.3144 (0.2864-0.3413) | 0.2503 (0.2137-0.2747) | 0.5111 (0.2949-0.7259) |
| Same-source GSM8K generated privileged BCE -> MATH1000 | 0-3 | 0.5494 (0.4680-0.6371) | 0.0985 (0.0809-0.1235) | 0.1931 (0.1719-0.2208) | 0.1741 (0.1472-0.2142) | 0.4696 (0.3479-0.6103) |
| Same-source GSM8K generated no-GT BCE -> MATH1000 | 0-3 | 0.6183 (0.5849-0.6614) | 0.1152 (0.1049-0.1314) | 0.2168 (0.2035-0.2352) | 0.2088 (0.1875-0.2287) | 0.4629 (0.3860-0.5397) |

Sequence-cluster bootstrap: even the weakest GSM8K gold seed beats the best
same-source generated no-GT seed by +0.0642 ROC-AUC, 95% CI [0.0445, 0.0831],
p=0.0004. It beats the best same-source generated privileged seed by +0.0885,
95% CI [0.0688, 0.1080], p=0.0004. The best same-source generated no-GT seed
also beats the best same-source generated privileged seed by +0.0242, 95% CI
[0.0135, 0.0344], p=0.0004.

Boundary diagnostic: OlympiadBench ProcessBench gold -> MATH1000 is unstable
over two seeds, with ROC-AUC 0.5854 / 0.7163 and PR-AUC 0.1209 / 0.2029
(mean ROC-AUC 0.6509). Seed 0 is significantly below the best generated-label
baseline by sequence-cluster bootstrap (-0.0466 ROC-AUC, 95% CI
[-0.0747, -0.0184], p=0.0010), while seed 1 is significantly above it
(+0.0835 ROC-AUC, 95% CI [0.0579, 0.1083], p=0.0010). Treat this as a
source-boundary/variance diagnostic, not a third clean headline source.

External calibration baseline: `Qwen/Qwen2.5-Math-7B-PRM800K` reaches
ROC-AUC 0.8379, PR-AUC 0.3254, best F1 0.3991, fixed F1 0.3953, and pred error
rate 0.1436 on the same MATH1000 split. It beats the strongest 500-step
single gold-source seed (OmniMath seed 3) by +0.0509 ROC-AUC, 95% CI
[0.0330, 0.0682], p=0.0010. It also beats a stronger GSM8K seed-0
1500-step diagnostic by +0.0435 ROC-AUC, 95% CI [0.0222, 0.0647],
p=0.0004. This confirms the current result is not a SOTA PRM claim; it is a
controlled mismatch/transfer claim.

Training-budget diagnostic: increasing GSM8K from 500 to 1500 training steps
improves seeds 0 and 1 but hurts seed 2. ROC-AUC changes are +0.0506 for seed 0
(95% CI [0.0288, 0.0722], p=0.0004), +0.0351 for seed 1 (95% CI
[0.0197, 0.0498], p=0.0004), and -0.0622 for seed 2 (95% CI
[-0.0801, -0.0444], p=0.0004). Averaging the three GSM8K 1500-step seeds gives
0.8153 ROC-AUC, 0.3130 PR-AUC, best F1 0.3990, and held-out calibrated F1
0.3715. It beats the best generated-label baseline by +0.1830 ROC-AUC
(95% CI [0.1607, 0.2049], p=0.0004), but Qwen PRM800K still beats it by
+0.0225 ROC-AUC (95% CI [0.0031, 0.0416], p=0.0220). Treat this as an
appendix/saturation result, not the main claim.

Appendix-style ensemble diagnostic: averaging saved per-step scores across the
four GSM8K seeds gives 0.7832 ROC-AUC; averaging the four OmniMath seeds gives
0.7865 ROC-AUC; averaging all eight GSM8K+OmniMath seeds gives 0.8073 ROC-AUC
and 0.2935 PR-AUC. Sequence-cluster bootstrap shows this eight-member score
average significantly beats the best single gold-source seed by +0.0203 ROC-AUC
(95% CI [0.0097, 0.0309], p=0.0004), but Qwen PRM800K still beats it by
+0.0305 ROC-AUC (95% CI [0.0121, 0.0487], p=0.0020).
With threshold chosen on the first 200 MATH sequences and evaluated on the
remaining 800, the same eight-member average reaches calibrated F1 0.3508,
versus 0.3855 for Qwen PRM800K.

The strongest post-hoc score-average diagnostic combines the three GSM8K
1500-step seeds with the four OmniMath 500-step seeds. This seven-member
average reaches 0.8276 ROC-AUC, 0.3433 PR-AUC, and best eval-swept F1 0.4050.
It beats the best generated-label baseline by +0.1953 ROC-AUC (95% CI
[0.1742, 0.2161], p=0.0004). Qwen PRM800K is higher by +0.0101 ROC-AUC, but
the sequence-cluster CI includes zero (95% CI [-0.0082, 0.0276], p=0.2675).
On the held-out threshold split, the seven-member average has higher eval
PR-AUC than Qwen (0.3413 vs 0.3242) but lower calibrated F1 (0.3754 vs
0.3855). Treat this as a strong appendix diagnostic, not as a clean SOTA claim.
The subsequently completed 1500-step follow-up grid was too high-variance to
promote: GSM8K seed 3 was healthy but modest, while OmniMath seeds 2-3
collapsed or over-flagged.

## Paired Bootstrap Gaps

The main robustness check should use the sequence-cluster bootstrap, which
resamples whole solutions rather than individual steps. `p` is two-sided from
2,000 bootstrap samples.

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

The older step-level bootstrap gives very similar gaps but is less conservative
because steps within a solution are correlated.

## Held-Out Threshold Calibration

Thresholds are chosen to maximize F1 on the first 200 MATH sequences, then
reported on the remaining 800 sequences.

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
| GSM8K1500+OmniMath500 7-seed mean | 0.3754 | 0.8240 | 0.3413 | 0.0982 |
| Qwen2.5-Math-7B-PRM800K | 0.3855 | 0.8372 | 0.3242 | 0.1091 |
| Generated privileged BCE | 0.1673 | 0.5490 | 0.1002 | 0.3212 |
| Best generated-label baseline | 0.2085 | 0.6314 | 0.1431 | 0.1695 |

## Interpretation

The model/training path can learn a transferable verifier when supervision is
ProcessBench-style and source-distribution compatible. OlympiadBench variance
suggests the source distribution still matters. The failure of generated
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
- Qwen2.5-Math-7B-PRM800K public model card and scoring convention:
  <https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K>.
- FreePRM, a weak-supervision PRM method without ground-truth process labels,
  reporting ProcessBench F1 by source split:
  <https://arxiv.org/pdf/2506.03570>.
- ProcessLID, an ICLR 2026 under-review internal-reward method with explicit
  transfer tables across GSM8K, MATH, OlympiadBench, and OmniMath:
  <https://openreview.net/pdf?id=5O5AlNVAbs>.
- Preference-based PRM work reporting ProcessBench comparisons for 7B reward
  models and hard/soft/preference labels:
  <https://openreview.net/pdf?id=09Nj40ScvC>.
- A 2026 survey that frames PRM data generation, training, and usage as an
  active crowded field:
  <https://arxiv.org/abs/2510.08049>.
- uPRM, a 2026 unsupervised PRM method using next-token probabilities and
  reporting ProcessBench gains without step labels:
  <https://arxiv.org/abs/2605.10158>.
- ThinkPRM, a 2026 generative/verbalized PRM submission with ProcessBench
  results under minimal process supervision:
  <https://openreview.net/forum?id=V727xqBYIW>.
- RetrievalPRM, which explicitly frames PRM failures as question/step
  out-of-distribution issues across GSM8K, MATH, OlympiadBench, and OmniMath:
  <https://arxiv.org/pdf/2502.14361>.
- Learning Discriminative Process Reward Models without Step Labels, a 2026
  outcome-label-only PRM paper adjacent to weak step-supervision claims:
  <https://openreview.net/forum?id=df3p10k2kq>.

Positioning implication: do not claim that "cross-config ProcessBench transfer"
alone is novel, and do not claim that PRM source-distribution/OOD framing is
new. The sharper contribution is the controlled mismatch result: generated
privileged/no-GT teacher labels fail badly under the same 3B student, optimizer,
and full-MATH eval path where non-MATH ProcessBench-style gold labels transfer
cleanly. The same-source GSM8K control means this is no longer only a
source-distribution confound for GSM8K; generated labels still fail when the
source problems are held fixed, though provenance and label format remain
coupled.

## Artifact Paths

- GSM8K seed 0: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
- GSM8K seed 1: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
- GSM8K seed 2: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed2/processbench_results.json`
- GSM8K seed 3: `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed3/processbench_results.json`
- OmniMath seed 0: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
- OmniMath seed 1: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
- OmniMath seed 2: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed2/processbench_results.json`
- OmniMath seed 3: `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed3/processbench_results.json`
- OlympiadBench seed 0: `results/diagnostics/processbench_olympiadbench_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
- OlympiadBench seed 1: `results/diagnostics/processbench_olympiadbench_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
- Qwen PRM800K baseline: `results/diagnostics/qwen_prm800k_math1000/processbench_results.json`
- Generated privileged BCE: `results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0/processbench_results.json`
- Best generated-label baseline: `results/diagnostics/generated_rank_nogt_to_math1000_qwen3b_seed0/processbench_results.json`
- Same-source generated GSM8K job spec: `results/diagnostics/job_specs/same_source_gsm8k400_generated_jobs.tsv`
- Same-source generated priv seeds: `results/diagnostics/generated_priv_gsm8k400_to_math1000_qwen3b_bce_bal_seed{0,1,2,3}/processbench_results.json`
- Same-source generated no-GT seeds: `results/diagnostics/generated_nogt_gsm8k400_to_math1000_qwen3b_bce_bal_seed{0,1,2,3}/processbench_results.json`
- Same-source bootstrap checks: `results/diagnostics/gsm8k_seed3_vs_same_source_generated_nogt_seed2_sequence_ci.json`, `results/diagnostics/gsm8k_seed3_vs_same_source_generated_priv_seed3_sequence_ci.json`, `results/diagnostics/same_source_generated_nogt_seed2_vs_priv_seed3_sequence_ci.json`
- Full runbook: `RUNBOOK_PHASE_B_FINDINGS_20260624.md`
