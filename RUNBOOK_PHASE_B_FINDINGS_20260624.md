# Phase B Findings Log - 2026-06-24

Purpose: record the fast 3B capacity/pivot gates run after the original Phase B
null, so we can decide what to scale and what to stop. This file is a research
log, not a final claims document.

## Current Status

No experiment job is running now. Both GPUs are idle.

The newest completed gates are the four-seed non-leaky ProcessBench
cross-config score-head runs:

- GSM8K ProcessBench gold -> MATH 400, seed 0: ROC-AUC 0.7241, PR-AUC 0.1848.
- GSM8K ProcessBench gold -> MATH 400, seed 1: ROC-AUC 0.7860, PR-AUC 0.2418.
- OmniMath ProcessBench gold -> MATH 400, seed 0: ROC-AUC 0.7569, PR-AUC 0.2107.
- OlympiadBench ProcessBench gold -> MATH 400, seed 0: ROC-AUC 0.5883, PR-AUC 0.1291.
- GSM8K ProcessBench gold -> full MATH1000, seeds 0-3: mean ROC-AUC 0.7515
  (range 0.7256-0.7760), mean PR-AUC 0.2188.
- OmniMath ProcessBench gold -> full MATH1000, seeds 0-3: mean ROC-AUC 0.7694
  (range 0.7539-0.7869), mean PR-AUC 0.2524.
- OlympiadBench ProcessBench gold -> full MATH1000, seeds 0-1: mean ROC-AUC
  0.6509 (range 0.5854-0.7163), mean PR-AUC 0.1619. This is a boundary
  diagnostic, not a third clean headline source.
- Public Qwen2.5-Math-7B-PRM800K -> full MATH1000: ROC-AUC 0.8379, PR-AUC
  0.3254, best eval-swept F1 0.3991. This is an external calibration baseline,
  not our training recipe.

This moves the live strategy away from more teacher-label loss variants and
toward source-distribution experiments: cross-config gold supervision reliably
trains a competent MATH verifier, while the generated teacher labels transfer
only weakly.

## What We Uncovered

1. The original 3B `score_critique` recipe is unstable.
   - `BATCH_SIZE=2`, 100 steps: 33 non-finite skips, weak AUCs.
   - `BATCH_SIZE=1`, 100 steps: fewer skips, but still 14 total skips.

2. A real LM-label truncation bug caused NaN loss steps.
   - Long prefixes could right-truncate away every supervised feedback token.
   - That produced all-ignored LM labels and NaN LM losses.
   - Fixed in `models/student.py` by tokenizing prefix and feedback separately,
     then left-truncating the prefix so feedback tokens are always retained.
   - Tokenizer-only validation over the first 250 shuffled examples:
     `bad=0`, `maxlen=512`, nonzero supervised feedback labels.

3. Fixing the truncation bug removed NaNs but exposed a negative result.
   - Equal-weight `score_critique` became clean but significantly worse for
     privileged labels than no-GT.
   - This means the earlier positive 100-step bs1 signal was probably an
     artifact of unstable/skipped LM examples, not a robust scale candidate.

4. Lowering LR and lowering LM weight did not rescue the critique objective.
   - Lower LR still had NaNs before the truncation fix and collapsed.
   - `LM_WEIGHT=0.1` after the fix was stable but still negative/weak.

5. BCE on the score head is the best clean pivot so far.
   - `score_only + SCORE_LOSS=bce + ERROR_WEIGHT=3` produced a significant
     positive privileged gap, though both cells remain weak/verifier-subcompetent.

6. Pure pairwise rank loss is not enough.
   - `score_only + SCORE_LOSS=rank + BALANCED_BATCHES=1` made no-GT much better
     than privileged.
   - The threshold behavior improved relative to all-negative collapse, but the
     privileged ranking itself was near random.

7. BCE+rank does not rescue BCE.
   - Privileged AUC reached only 0.5214, below the earlier BCE-only 0.5411.
   - no-GT reached 0.6140, so the privileged-vs-noGT gap was significantly
     negative.

8. More BCE training makes the privileged verifier worse.
   - `bce_ew3` at 500 steps gave privileged ROC-AUC 0.4782.
   - The 500-step gap was null/negative, not a stronger version of the 100-step
     positive signal.

9. Frozen-representation probes point to label/distribution mismatch.
   - A 5k-step linear probe trained on privileged labels reached train ROC-AUC
     0.9947 but ProcessBench ROC-AUC only 0.5497.
   - The matching no-GT probe reached train ROC-AUC 0.9970 and ProcessBench
     ROC-AUC 0.5559.
   - A ProcessBench-gold split probe, using the same frozen 3B representation,
     reached held-out ROC-AUC 0.6308.
   - Interpretation: the frozen representation can expose the target when labels
     are distribution-matched, but current generated teacher-label data transfers
     weakly; privileged labels still do not beat no-GT.

10. A matched ProcessBench-gold score head is strong.
    - Training the same 3B score-head path on the first 200 shuffled ProcessBench
      MATH samples and evaluating on the next 200 samples reached ROC-AUC 0.7129
      and PR-AUC 0.2281.
    - This is a diagnostic split, not a paper claim, but it shows the scorer and
      training code can learn the target under matched labels.
    - The gap between matched-gold 0.713 AUC and teacher-label 0.55 AUC makes
      distribution mismatch the strongest current failure explanation.

11. Non-leaky cross-config ProcessBench-style supervision can transfer to MATH.
    - Training on GSM8K ProcessBench gold labels (400 source examples, 2,082
      steps) and evaluating on ProcessBench MATH 400 reached ROC-AUC 0.7241 and
      PR-AUC 0.1848 at seed 0, then ROC-AUC 0.7860 and PR-AUC 0.2418 at seed 1.
    - Training on OlympiadBench ProcessBench gold labels (400 source examples,
      3,579 steps) reached ROC-AUC 0.5883 and PR-AUC 0.1291.
    - Training on OmniMath ProcessBench gold labels (400 source examples, 3,384
      steps) reached ROC-AUC 0.7569 and PR-AUC 0.2107.
    - This is the first non-leaky competent verifier result in the current run:
      cross-config GSM8K and OmniMath gold labels transfer to MATH better than
      current generated teacher labels and better than the matched-MATH split
      diagnostic.

12. Combining strong sources is not automatically stable.
    - GSM8K+OmniMath combined source, seed 0 reached ROC-AUC 0.7516 and PR-AUC
      0.2028, matching the strong source-only band.
    - GSM8K+OmniMath combined source, seed 1 fell to ROC-AUC 0.5166 and PR-AUC
      0.0932, with very high pred_error_rate 0.8803.
    - GSM8K+OmniMath combined source, seed 2 reached only ROC-AUC 0.6328 and
      PR-AUC 0.1289.
    - This argues against immediately scaling combined-source training.

13. OmniMath source-specific transfer replicated strongly.
    - OmniMath seed 0 reached ROC-AUC 0.7569 and PR-AUC 0.2107.
    - OmniMath seed 1 reached ROC-AUC 0.7800 and PR-AUC 0.2430.
    - Mean OmniMath source-specific transfer over two seeds: ROC-AUC 0.7684,
      PR-AUC 0.2268.
    - Together with GSM8K seed 0/1 (mean ROC-AUC 0.7550), this makes
      source-specific ProcessBench-gold transfer the strongest positive result.

14. The source-specific positive result survives full MATH-1000 eval.
    - GSM8K seed 0 evaluated on all 1,000 ProcessBench MATH samples reached
      ROC-AUC 0.7439, PR-AUC 0.2234, and best eval-swept F1 0.3213.
    - GSM8K seed 1 evaluated on all 1,000 ProcessBench MATH samples reached
      ROC-AUC 0.7760, PR-AUC 0.2264, and best eval-swept F1 0.3413 over 6,505
      steps / 594 error steps.
    - GSM8K seed 2 reached ROC-AUC 0.7603, PR-AUC 0.2317, and best eval-swept
      F1 0.3088.
    - GSM8K seed 3 reached ROC-AUC 0.7256, PR-AUC 0.1935, and best eval-swept
      F1 0.2864.
    - OmniMath seed 0 evaluated on all 1,000 ProcessBench MATH samples reached
      ROC-AUC 0.7539, PR-AUC 0.2277, and best eval-swept F1 0.3145.
    - OmniMath seed 1 evaluated on all 1,000 ProcessBench MATH samples reached
      ROC-AUC 0.7792, PR-AUC 0.2633, and best eval-swept F1 0.3385.
    - OmniMath seed 2 reached ROC-AUC 0.7578, PR-AUC 0.2308, and best
      eval-swept F1 0.3193.
    - OmniMath seed 3 reached ROC-AUC 0.7869, PR-AUC 0.2877, and best
      eval-swept F1 0.3380.
    - Four-seed means: GSM8K ROC-AUC 0.7515, PR-AUC 0.2188; OmniMath ROC-AUC
      0.7694, PR-AUC 0.2524.
    - This is now the strongest publishable-direction result: two different
      non-MATH source configs train a 3B verifier that transfers to full
      ProcessBench MATH under exact serial evaluation.

15. Full-MATH generated-label baselines remain much weaker.
    - Best privileged generated-label checkpoint (`bce_ew3` priv) reached
      ROC-AUC 0.5503 and PR-AUC 0.0992 on full MATH-1000; fixed-threshold F1
      collapsed to 0.0 because it predicted no errors.
    - Best generated-label checkpoint overall from the fast gates (`rank_bal`
      noGT) reached ROC-AUC 0.6324 and PR-AUC 0.1418.

16. OlympiadBench is a high-variance boundary source.
    - Seed 0 reached ROC-AUC 0.5854, PR-AUC 0.1209, and best eval-swept F1
      0.1918 on full MATH-1000.
    - Seed 1 reached ROC-AUC 0.7163, PR-AUC 0.2029, and best eval-swept F1
      0.2850.
    - Mean over two seeds: ROC-AUC 0.6509, PR-AUC 0.1619.
    - Sequence-cluster bootstrap vs the best generated-label baseline:
      seed 0 is worse by -0.0466 ROC-AUC, 95% CI [-0.0747, -0.0184],
      p=0.0010; seed 1 is better by +0.0835 ROC-AUC, 95% CI
      [0.0579, 0.1083], p=0.0010.
    - Interpretation: OlympiadBench supports the source-distribution framing,
      but it should not be included as a clean headline transfer source.

17. A public Qwen PRM baseline is substantially stronger on the same split.
    - `Qwen/Qwen2.5-Math-7B-PRM800K`, evaluated with the model-card
      `<extra_0>` separator-token scoring convention, reached ROC-AUC 0.8379,
      PR-AUC 0.3254, fixed-threshold F1 0.3953, and best eval-swept F1 0.3991.
    - The run used revision `9d6e292f6ccfd474fa44461ce6d5b80d08d8f3c7`.
    - It had 0 truncated sequences and 0 score-count mismatches.
    - Sequence-cluster bootstrap: Qwen PRM800K beats OmniMath seed 3 by
      +0.0509 ROC-AUC, 95% CI [0.0330, 0.0682], p=0.0010, and beats the best
      generated-label baseline by +0.2055 ROC-AUC, 95% CI [0.1810, 0.2309],
      p=0.0010.
    - Interpretation: this confirms our result is not a SOTA PRM claim. It is
      still valuable as a controlled mismatch result because the same small
      student path separates compatible gold process supervision from generated
      teacher labels.

18. Sequence-cluster bootstrap confirms the gap.
    - The earlier paired bootstrap resampled individual steps; this is fast but
      optimistic because steps within one solution are correlated.
    - A stricter paired bootstrap that resamples whole solution sequences still
      gives significant gaps for every GSM8K/OmniMath seed against both
      full-MATH generated-label baselines.
    - On the same 6,505 MATH steps, source-specific ProcessBench-gold transfer
      beats both generated-label baselines by large paired-bootstrap margins.
    - Against the best generated-label baseline (`rank_bal` noGT), sequence
      bootstrap ROC-AUC gaps are +0.1113, +0.1436, +0.1279, and +0.0930 for
      GSM8K seeds 0-3, and +0.1214, +0.1468, +0.1253, and +0.1546 for
      OmniMath seeds 0-3; all two-sided p=0.0010 with 2,000 bootstrap samples.

19. Held-out threshold calibration also favors source-specific gold transfer.
    - Thresholds chosen on the first 200 MATH sequences and evaluated on the
      remaining 800 sequences give calibrated F1 0.2714-0.3166 for GSM8K seeds
      0-3 and 0.3062-0.3316 for OmniMath seeds 0-3.
    - Qwen PRM800K reaches calibrated F1 0.3855 on the same split.
    - Generated privileged BCE reaches calibrated F1 0.1673; the best
      generated-label baseline reaches calibrated F1 0.2085.

20. Exact problem-overlap leakage check is clean.
    - GSM8K source train400 vs MATH1000 eval: 0 exact problem overlaps.
    - OmniMath source train400 vs MATH1000 eval: 0 exact problem overlaps.
    - OlympiadBench source train400 vs MATH1000 eval: 0 exact problem overlaps.

## Fast Gate Results

All rows are Qwen/Qwen2.5-3B-Instruct, `N_EVAL=400`, seed 0, no BoN.
Rows are `MAX_STEPS=100` unless the tag says `_500`.

| Tag | Cell | ROC-AUC | PR-AUC | Pred error rate | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `ms100` | priv | 0.4582 | 0.0752 | 0.0000 | collapsed |
| `ms100` | noGT | 0.4870 | 0.0876 | 0.0004 | collapsed |
| `ms100_bs1` | priv | 0.5329 | 0.0977 | 0.9270 | unstable but best pre-fix priv |
| `ms100_bs1` | noGT | 0.4857 | 0.0807 | 0.9301 | unstable |
| `scoreonly` | priv | 0.4844 | 0.0787 | 0.0000 | stable, weak |
| `scoreonly` | noGT | 0.4776 | 0.0880 | 0.8976 | stable, weak |
| `lr3e-5` | priv | 0.4959 | 0.0817 | 0.0004 | collapsed |
| `lr3e-5` | noGT | 0.4697 | 0.0796 | 0.0004 | collapsed |
| `truncfix` | priv | 0.4514 | 0.0737 | 0.0012 | stable, significantly negative |
| `truncfix` | noGT | 0.5093 | 0.0863 | 0.0000 | stable |
| `lmw0p1` | priv | 0.4546 | 0.0748 | 0.0004 | stable, weak |
| `lmw0p1` | noGT | 0.4933 | 0.0893 | 0.0000 | stable, weak |
| `bce_ew3` | priv | 0.5411 | 0.0928 | 0.0000 | best clean privileged AUC |
| `bce_ew3` | noGT | 0.4700 | 0.0823 | 0.8961 | weak, over-flags |
| `rank_bal` | priv | 0.4939 | 0.0821 | 0.2565 | rank-only fails privileged |
| `rank_bal` | noGT | 0.6306 | 0.1351 | 0.3770 | noGT wins strongly |
| `bcerank_bal` | priv | 0.5214 | 0.0852 | 0.3901 | below BCE-only |
| `bcerank_bal` | noGT | 0.6140 | 0.1266 | 0.5643 | noGT wins strongly |
| `bce_ew3_500` | priv | 0.4782 | 0.0783 | 0.4504 | longer BCE degrades |
| `bce_ew3_500` | noGT | 0.4971 | 0.0860 | 0.0630 | weak/null |

## Frozen Probe Diagnostics

All rows use frozen Qwen/Qwen2.5-3B-Instruct step-boundary embeddings and a
class-balanced logistic probe. These are diagnostics, not final model claims.

| Probe | Train labels | Train steps | Eval split | Eval ROC-AUC | Eval PR-AUC | Read |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `repr_1k` | privileged teacher | 1000 | ProcessBench MATH 400 | 0.5179 | 0.0872 | weak |
| `repr_1k` | no-GT teacher | 1000 | ProcessBench MATH 400 | 0.5504 | 0.0921 | weak, noGT better |
| `repr_5k` | privileged teacher | 5000 | ProcessBench MATH 400 | 0.5497 | 0.0916 | weak |
| `repr_5k` | no-GT teacher | 5000 | ProcessBench MATH 400 | 0.5559 | 0.0952 | weak, noGT slightly better |
| `gold_200/200` | ProcessBench gold | 1300 | held-out ProcessBench MATH 200 samples | 0.6308 | 0.1301 | target is linearly learnable when matched |

## Matched Gold Score-Head Diagnostic

This trains the actual LoRA+score-head path, not just a linear probe, on a
diagnostic ProcessBench split. It should not be reported as a benchmark result
because both train/eval are slices of ProcessBench MATH, but it is the clearest
debug signal so far.

| Train split | Eval split | Recipe | Eval ROC-AUC | Eval PR-AUC | F1 | Pred error rate | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| first 200 shuffled ProcessBench MATH samples | next 200 samples | score-only BCE, balanced batches, 500 steps | 0.7129 | 0.2281 | 0.2794 | 0.1567 | training path works when labels/distribution match |
| GSM8K ProcessBench gold, 400 samples | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.7241 | 0.1848 | 0.2292 | 0.5357 | non-leaky cross-config transfer works |
| GSM8K ProcessBench gold, 400 samples, seed 1 | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.7860 | 0.2418 | 0.2724 | 0.4681 | confirms GSM transfer across seeds |
| OlympiadBench ProcessBench gold, 400 samples | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.5883 | 0.1291 | 0.1708 | 0.1294 | weaker cross-config transfer |
| OmniMath ProcessBench gold, 400 samples | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.7569 | 0.2107 | 0.3001 | 0.1723 | strong cross-config transfer from a second source |
| OmniMath ProcessBench gold, 400 samples, seed 1 | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.7800 | 0.2430 | 0.2994 | 0.3561 | confirms OmniMath transfer across seeds |
| GSM8K+OmniMath ProcessBench gold, 800 samples, seed 0 | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.7516 | 0.2028 | 0.2570 | 0.4805 | strong but not better than source-only |
| GSM8K+OmniMath ProcessBench gold, 800 samples, seed 1 | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.5166 | 0.0932 | 0.1612 | 0.8803 | combined-source instability |
| GSM8K+OmniMath ProcessBench gold, 800 samples, seed 2 | ProcessBench MATH 400 samples | score-only BCE, balanced batches, 500 steps | 0.6328 | 0.1289 | 0.2069 | 0.5659 | only moderate; do not scale mixing |
| GSM8K ProcessBench gold, 400 samples, seed 1 | full ProcessBench MATH 1000 samples | score-only BCE, balanced batches, 500 steps | 0.7760 | 0.2264 | 0.2707 | 0.4776 | full-eval positive holds |
| GSM8K ProcessBench gold, 400 samples, seed 0 | full ProcessBench MATH 1000 samples | score-only BCE, balanced batches, 500 steps | 0.7439 | 0.2234 | 0.2421 | 0.5462 | full-eval positive holds |
| OmniMath ProcessBench gold, 400 samples, seed 1 | full ProcessBench MATH 1000 samples | score-only BCE, balanced batches, 500 steps | 0.7792 | 0.2633 | 0.3086 | 0.3640 | full-eval positive holds |
| OmniMath ProcessBench gold, 400 samples, seed 0 | full ProcessBench MATH 1000 samples | score-only BCE, balanced batches, 500 steps | 0.7539 | 0.2277 | 0.3052 | 0.1797 | full-eval positive holds |
| Best privileged generated-label checkpoint (`bce_ew3` priv) | full ProcessBench MATH 1000 samples | score-only BCE, 100 steps | 0.5503 | 0.0992 | 0.0000 | 0.0000 | weak, fixed-threshold silent collapse |
| Best generated-label checkpoint overall (`rank_bal` noGT) | full ProcessBench MATH 1000 samples | rank-only, balanced batches, 100 steps | 0.6324 | 0.1418 | 0.2151 | 0.3819 | best generated-label baseline, still far below gold-transfer |

Compact full-MATH table:

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1* | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K gold -> MATH1000 | 0,1 | 0.7600 (0.7439-0.7760) | 0.2249 (0.2234-0.2264) | 0.3313 (0.3213-0.3413) | 0.2564 (0.2421-0.2707) | 0.5119 (0.4776-0.5462) |
| OmniMath gold -> MATH1000 | 0,1 | 0.7665 (0.7539-0.7792) | 0.2455 (0.2277-0.2633) | 0.3265 (0.3145-0.3385) | 0.3069 (0.3052-0.3086) | 0.2719 (0.1797-0.3640) |
| Generated priv BCE -> MATH1000 | 0 | 0.5503 | 0.0992 | 0.1803 | 0.0000 | 0.0000 |
| Generated noGT rank -> MATH1000 | 0 | 0.6324 | 0.1418 | 0.2221 | 0.2151 | 0.3819 |

*Best F1 is threshold-swept on the eval slice and should be treated as an
optimistic diagnostic unless a held-out calibration split is used.

Cross-config vs generated-label baselines, paired step bootstrap:

| Model A | Model B | ROC-AUC gap | 95% CI | Two-sided p | Read |
| --- | --- | ---: | --- | ---: | --- |
| GSM8K gold -> MATH, seed 1 | best privileged teacher-label BCE run | +0.2451 | [0.2056, 0.2843] | 0.0004 | cross-config gold is clearly stronger |
| OmniMath gold -> MATH, seed 0 | best privileged teacher-label BCE run | +0.2160 | [0.1674, 0.2636] | 0.0004 | cross-config gold is clearly stronger |
| OmniMath gold -> MATH, seed 1 | best privileged teacher-label BCE run | +0.2388 | [0.1945, 0.2821] | 0.0004 | replicated cross-config gold advantage |
| GSM8K gold -> MATH, seed 1 | best generated-label run overall (`rank_bal` noGT) | +0.1554 | [0.1233, 0.1880] | 0.0004 | beats best generated-label baseline |
| OmniMath gold -> MATH, seed 0 | best generated-label run overall (`rank_bal` noGT) | +0.1264 | [0.0813, 0.1716] | 0.0004 | beats best generated-label baseline |
| OmniMath gold -> MATH, seed 1 | best generated-label run overall (`rank_bal` noGT) | +0.1498 | [0.1138, 0.1850] | 0.0004 | replicated advantage over best generated-label baseline |
| GSM8K gold -> full MATH1000, seed 1 | full-MATH privileged teacher-label BCE baseline | +0.2257 | [0.2008, 0.2501] | 0.0004 | full-eval paired gap |
| OmniMath gold -> full MATH1000, seed 1 | full-MATH privileged teacher-label BCE baseline | +0.2290 | [0.1997, 0.2574] | 0.0004 | full-eval paired gap |
| GSM8K gold -> full MATH1000, seed 1 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1434 | [0.1230, 0.1639] | 0.0004 | full-eval paired gap |
| OmniMath gold -> full MATH1000, seed 1 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1466 | [0.1239, 0.1691] | 0.0004 | full-eval paired gap |
| GSM8K gold -> full MATH1000, seed 0 | full-MATH privileged teacher-label BCE baseline | +0.1940 | [0.1645, 0.2235] | 0.0004 | replicated full-eval paired gap |
| OmniMath gold -> full MATH1000, seed 0 | full-MATH privileged teacher-label BCE baseline | +0.2034 | [0.1730, 0.2331] | 0.0004 | replicated full-eval paired gap |
| GSM8K gold -> full MATH1000, seed 0 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1116 | [0.0862, 0.1377] | 0.0004 | replicated full-eval paired gap |
| OmniMath gold -> full MATH1000, seed 0 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1213 | [0.0950, 0.1484] | 0.0004 | replicated full-eval paired gap |

Sequence-cluster bootstrap summary:

| Model A | Model B | ROC-AUC gap | 95% CI | Two-sided p |
| --- | --- | ---: | --- | ---: |
| GSM8K gold -> full MATH1000, seed 0 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1113 | [0.0874, 0.1339] | 0.0010 |
| GSM8K gold -> full MATH1000, seed 1 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1436 | [0.1248, 0.1627] | 0.0010 |
| OmniMath gold -> full MATH1000, seed 0 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1214 | [0.0964, 0.1458] | 0.0010 |
| OmniMath gold -> full MATH1000, seed 1 | full-MATH best generated-label baseline (`rank_bal` noGT) | +0.1468 | [0.1254, 0.1676] | 0.0010 |

Held-out calibrated threshold summary:

| Model | Calibrated F1 | Eval ROC-AUC | Eval PR-AUC | Pred error rate |
| --- | ---: | ---: | ---: | ---: |
| GSM8K gold seed 0 | 0.3166 | 0.7452 | 0.2339 | 0.2488 |
| GSM8K gold seed 1 | 0.2977 | 0.7707 | 0.2152 | 0.1466 |
| OmniMath gold seed 0 | 0.3062 | 0.7483 | 0.2292 | 0.2742 |
| OmniMath gold seed 1 | 0.3294 | 0.7723 | 0.2606 | 0.2321 |
| Generated privileged BCE | 0.1673 | 0.5490 | 0.1002 | 0.3212 |
| Best generated-label baseline (`rank_bal` noGT) | 0.2085 | 0.6314 | 0.1431 | 0.1695 |

Transfer CI summaries:

| Tag | Priv - noGT ROC-AUC gap | 95% CI | Two-sided p | Read |
| --- | ---: | --- | ---: | --- |
| `ms100` | -0.0291 | [-0.0623, 0.0049] | 0.0964 | weak negative |
| `ms100_bs1` | +0.0468 | [-0.0019, 0.0950] | 0.0586 | unstable borderline positive |
| `scoreonly` | +0.0068 | [-0.0384, 0.0517] | 0.7609 | null |
| `lr3e-5` | +0.0262 | [-0.0112, 0.0637] | 0.1760 | null |
| `truncfix` | -0.0582 | [-0.0951, -0.0209] | 0.0024 | significant negative |
| `lmw0p1` | -0.0384 | [-0.0824, 0.0061] | 0.0876 | weak negative |
| `bce_ew3` | +0.0708 | [0.0275, 0.1134] | 0.0014 | first clean positive gap |
| `rank_bal` | -0.1369 | [-0.1842, -0.0904] | 0.0002 | rank-only rejects privileged |
| `bcerank_bal` | -0.0924 | [-0.1365, -0.0472] | 0.0002 | BCE+rank rejects privileged |
| `bce_ew3_500` | -0.0194 | [-0.0692, 0.0307] | 0.4490 | longer BCE null/weak |

## Interpretation

The main blocker is not simply model capacity. The current linear score-head
training signal is poorly aligned with the evaluation metric:

- MSE-to-teacher-score permits near-constant mostly-correct scorers.
- Equal critique LM loss teaches many `Correct.` continuations and can dominate
  or distort the scorer.
- BCE improves the privileged gap but still leaves threshold calibration and
  absolute ranking quality weak.

The best current claim is not yet "publishable positive result." It is:

> Privileged labels can create a clean positive 3B transfer gap when the score
> objective is class-balanced BCE, but the verifier is still not competent enough
> for BoN or a headline claim.

The strongest negative/mechanistic finding is:

> Correcting feedback-token truncation removes NaNs and reveals that naive
> equal-weight critique distillation significantly hurts privileged transfer.

The newest negative result is:

> Pairwise ranking alone can make the no-GT labels look much better than the
> privileged labels, so ranking loss is not a sufficient fix for the privileged
> transfer failure.

The current operational conclusion is:

> Do not spend BoN or larger training budget on these 3B score-head recipes.
> The positive 100-step BCE gate is not robust under BCE+rank or longer BCE.

The new diagnostic conclusion is:

> Distribution/label mismatch is now the leading failure mode. The frozen 3B
> representation can support a 0.63 ROC-AUC ProcessBench-gold probe, but probes
> trained on the generated teacher-labeled data transfer only around 0.55 ROC-AUC.

After the matched score-head gate:

> The score-head training machinery itself is not the blocker: with matched
> ProcessBench-gold labels it reaches 0.713 ROC-AUC on a held-out ProcessBench
> slice. The publishable path should now target distribution-matched data or a
> reframing around teacher-label transfer failure.

After the cross-config gates:

> The strongest current positive result is non-leaky: 3B verifiers trained on
> GSM8K or OmniMath ProcessBench gold labels transfer to full ProcessBench MATH
> at about 0.776-0.779 ROC-AUC. The contrast with generated teacher-label
> baselines (0.550 for privileged BCE; 0.632 for the best generated-label
> checkpoint overall) is now a concrete, potentially publishable
> distribution-mismatch finding.

## Literature / Positioning Snapshot

External check on 2026-06-24:

- ProcessBench frames the benchmark as earliest-error identification over 3,400
  math reasoning cases and reports that existing PRMs often fail to generalize
  beyond GSM8K and MATH: <https://arxiv.org/abs/2412.06559>.
- Math-Shepherd is the key prior for automatically constructed process-wise
  supervision without human step labels: <https://arxiv.org/abs/2312.08935>.
- FreePRM is a closer 2025 weak-supervision prior; it explicitly trains PRMs
  without ground-truth process labels and reports 53.0 average ProcessBench F1:
  <https://arxiv.org/abs/2506.03570>.
- Qwen2.5-Math PRMs are strong public baselines. The PRM paper reports
  ProcessBench average F1 of 73.5 for Qwen2.5-Math-PRM-7B and 78.3 for 72B:
  <https://arxiv.org/pdf/2501.07301>. The PRM800K baseline card also notes that
  Qwen2.5-Math-7B-PRM800K is trained on PRM800K with MATH-test leakage removed:
  <https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K>.
  We evaluated the public 7B PRM800K checkpoint directly on our MATH1000 split
  as a calibration baseline.
- ProcessLID is an ICLR 2026 under-review internal-reward method with explicit
  transfer tables across ProcessBench GSM8K, MATH, OlympiadBench, and OmniMath:
  <https://openreview.net/pdf?id=5O5AlNVAbs>.
- Preference-based PRM work reports ProcessBench comparisons for hard labels,
  soft labels, and preference labels on 7B reward models:
  <https://openreview.net/pdf?id=09Nj40ScvC>.
- A 2026 PRM survey frames the field around data generation, PRM training, and
  PRM usage, and explicitly categorizes automated supervision as a crowded
  design space: <https://arxiv.org/abs/2510.08049>.
- uPRM is a 2026 unsupervised PRM method that uses next-token probabilities and
  reports ProcessBench improvements without step labels or final-answer
  verification: <https://arxiv.org/abs/2605.10158>.
- ThinkPRM is an ICLR 2026 withdrawn submission on generative/verbalized PRMs
  with minimal process labels and ProcessBench results:
  <https://openreview.net/forum?id=V727xqBYIW>.

Positioning implication:

> Do not claim SOTA PRM performance from the current numbers. The plausible
> publishable angle is controlled evidence that generated privileged/no-GT
> teacher labels fail to transfer under the same student/training path where
> ProcessBench-style gold labels do transfer. If we want a benchmark-performance
> claim, we need full ProcessBench-style evaluation and direct baselines against
> Qwen/PRM800K/FreePRM-style methods.

Also do not claim that cross-config ProcessBench transfer itself is wholly new;
the safer novelty is the controlled mismatch comparison between generated
teacher labels and ProcessBench-style gold labels under the same student path.

## Next Best Move

Do not run BoN yet. Do not scale the equal-weight critique recipe, pure
rank-only recipe, BCE+rank recipe, or longer BCE recipe.

Next experiment should change the data distribution, not just run the same loop
longer:

1. Cache frozen embeddings/progress for diagnostics.
   - Batched bf16 score-head evaluation slightly perturbs rankings, so headline
     `run_processbench` should stay serial (`EVAL_BATCH_SIZE=1`).
   - Representation probes are diagnostic and can use batched hidden extraction,
     but they should cache embeddings and print progress before broader sweeps.

2. Build a distribution-matched training/eval test.
   - Already confirmed in a diagnostic ProcessBench split: matched gold labels
     give 0.713 ROC-AUC.
   - Non-leaky cross-config results now exist: GSM8K ProcessBench gold -> MATH
     gives 0.724 and 0.786 ROC-AUC across two seeds; OmniMath -> MATH gives
     0.757 ROC-AUC; OlympiadBench -> MATH is weaker at 0.588 ROC-AUC.
   - Combined GSM8K+OmniMath is unstable across seeds (0.7516, 0.5166, 0.6328
     AUC). Do not scale combined-source training yet.
   - Source-specific GSM8K and OmniMath are replicated positives over four
     seeds each, and both hold on full MATH-1000.
   - OlympiadBench is high-variance over two seeds and should be treated as a
     boundary/source-distribution diagnostic.
   - Full-MATH generated-label baselines are much weaker. Do not run more
     GSM8K/OmniMath replication unless a reviewer specifically needs it.

3. Use both GPUs safely for independent fast gates.
   - Launch each process with `TRAIN_CUDA_VISIBLE_DEVICES=0` or `1`.
   - Use `SLFD_CUDA_PLACEMENT=single` so each 3B job stays on one 3090 instead
     of `device_map=auto` spreading a single run across both GPUs.
   - Keep vLLM BoN generation separate; the default vLLM path uses both GPUs via
     tensor parallelism.
   - For ProcessBench-gold transfer gates, use `scripts/run_gold_scorehead_gate.sh`.
     It sets `PYTHONUNBUFFERED=1` for future launches so step-loss progress is
     visible while jobs are running.

4. If the probe is strong, try better training only then.
   - Candidate changes: generate/train on ProcessBench-style solutions, train
     only the score head on matched labels first, calibrate threshold on held-out
     labels, or use per-solution pair construction instead of global class
     balancing.
   - Continue to BoN only after ProcessBench ROC-AUC is at least about 0.58 and
     the privileged-vs-noGT gap is positive.

## Time Boxes

- Cached embedding/probe pipeline with progress logging: 45-90 min.
- Matched ProcessBench split score-head/probe gate: 45-90 min once cached.
- Distribution-matched non-leaky data plan: 30-60 min.
- Cross-config confirmation seeds/source configs: 1-2 hours.
- Combined-source two-seed gate on both GPUs: about 1-2 hours to get the first
  actionable result, assuming no queueing or memory contention.
- After full-MATH-1000 positives: source-specific transfer is strong enough for
  a result table. Four seeds per strong source are now complete; next compute is
  only for boundary/source-distribution diagnostics such as OlympiadBench.
- New training recipe gate: only after the probe shows usable signal.
- Quick BoN: only after ProcessBench ROC-AUC is at least about 0.58 and the
  priv-vs-noGT gap remains positive.

## Artifact Pointers

- BCE positive gate:
  - `results/ablation/Qwen_Qwen2.5-3B-Instruct_seed0_ms100_bce_ew3_priv_critique/processbench_results.json`
  - `results/ablation/Qwen_Qwen2.5-3B-Instruct_seed0_ms100_bce_ew3_nogt_critique/processbench_results.json`
  - `results/ablation/Qwen_Qwen2.5-3B-Instruct_seed0_ms100_bce_ew3_transfer_ci.json`
- Truncation-fix negative gate:
  - `results/ablation/Qwen_Qwen2.5-3B-Instruct_seed0_ms100_truncfix_transfer_ci.json`
- Logs:
  - `phaseb_capacity_3b_ms100_bce_ew3_20260624.log`
  - `phaseb_capacity_3b_ms100_truncfix_20260624.log`
  - `phaseb_capacity_3b_ms100_rank_bal_20260624.log`
  - `phaseb_capacity_3b_ms100_bcerank_bal_20260624.log`
  - `phaseb_capacity_3b_ms500_bce_ew3_20260624.log`
- Probe diagnostics:
  - `results/diagnostics/representation_probe_Qwen2.5-3B_mtrain1000_meval400_seed0.json`
  - `results/diagnostics/representation_probe_Qwen2.5-3B_mtrain5000_meval400_seed0.json`
  - `results/diagnostics/processbench_gold_probe_Qwen2.5-3B_train200_eval200_seed0.json`
  - `phaseb_repr_probe_3b_1k_20260624.log`
  - `phaseb_repr_probe_3b_5k_20260624.log`
  - `phaseb_gold_probe_3b_200_20260624.log`
- Matched gold score-head diagnostic:
  - `results/diagnostics/processbench_gold_train200_steps.jsonl`
  - `results/diagnostics/processbench_gold_eval200.jsonl`
  - `results/diagnostics/processbench_gold_scorehead_qwen3b_bce_bal_train200_eval200_seed0/processbench_results.json`
  - `results/diagnostics/processbench_gold_scorehead_qwen3b_bce_bal_train200_eval200_seed0/per_step_scores.json`
  - `phaseb_gold_scorehead_3b_500_20260624.log`
- Cross-config gold score-head diagnostics:
  - `results/diagnostics/processbench_gsm8k_gold_train400_steps.jsonl`
  - `results/diagnostics/processbench_olympiadbench_gold_train400_steps.jsonl`
  - `results/diagnostics/processbench_omnimath_gold_train400_steps.jsonl`
  - `results/diagnostics/processbench_gsm8k_omnimath_gold_train800_steps.jsonl`
  - `results/diagnostics/processbench_gsm8k_to_math400_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math400_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_olympiadbench_to_math400_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math400_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math400_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_omnimath_to_math400_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_omnimath_to_math400_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_omnimath_to_math400_scorehead_qwen3b_bce_bal_seed2/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed0/processbench_results.json`
  - `results/diagnostics/processbench_omnimath_to_math1000_scorehead_qwen3b_bce_bal_seed1/processbench_results.json`
  - `results/diagnostics/teacher_bce_priv_to_math1000_qwen3b_seed0/processbench_results.json`
  - `results/diagnostics/generated_rank_nogt_to_math1000_qwen3b_seed0/processbench_results.json`
  - `results/diagnostics/gsm8k_seed1_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/omnimath_seed0_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/omnimath_seed1_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/gsm8k_seed1_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/omnimath_seed0_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/omnimath_seed1_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/gsm8k_math1000_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/omnimath_math1000_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/gsm8k_math1000_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/omnimath_math1000_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/gsm8k_seed0_math1000_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/omnimath_seed0_math1000_vs_teacher_bce_priv_transfer_ci.json`
  - `results/diagnostics/gsm8k_seed0_math1000_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/omnimath_seed0_math1000_vs_best_generated_rank_nogt_transfer_ci.json`
  - `results/diagnostics/gsm8k_seed0_math1000_vs_teacher_bce_priv_sequence_ci.json`
  - `results/diagnostics/gsm8k_seed1_math1000_vs_teacher_bce_priv_sequence_ci.json`
  - `results/diagnostics/omnimath_seed0_math1000_vs_teacher_bce_priv_sequence_ci.json`
  - `results/diagnostics/omnimath_seed1_math1000_vs_teacher_bce_priv_sequence_ci.json`
  - `results/diagnostics/gsm8k_seed0_math1000_vs_best_generated_rank_nogt_sequence_ci.json`
  - `results/diagnostics/gsm8k_seed1_math1000_vs_best_generated_rank_nogt_sequence_ci.json`
  - `results/diagnostics/omnimath_seed0_math1000_vs_best_generated_rank_nogt_sequence_ci.json`
  - `results/diagnostics/omnimath_seed1_math1000_vs_best_generated_rank_nogt_sequence_ci.json`
  - `results/diagnostics/math1000_calibrated_threshold_metrics_cal200_eval800.json`
  - `phaseb_gsm8k_to_math_scorehead_3b_500_20260624.log`
  - `phaseb_gsm8k_to_math_scorehead_3b_500_seed1_20260624.log`
  - `phaseb_olymp_to_math_scorehead_3b_500_20260624.log`
  - `phaseb_omni_to_math_scorehead_3b_500_20260624.log`
  - `phaseb_gsm8k_omnimath_to_math_scorehead_3b_500_seed0_20260624.log`
  - `phaseb_gsm8k_omnimath_to_math_scorehead_3b_500_seed1_20260624.log`
  - `phaseb_gsm8k_omnimath_to_math_scorehead_3b_500_seed2_20260624.log`
  - `phaseb_omni_to_math_scorehead_3b_500_seed1_20260624.log`
  - `phaseb_gsm8k_to_math1000_scorehead_3b_seed0_20260624.log`
  - `phaseb_gsm8k_to_math1000_scorehead_3b_seed1_20260624.log`
  - `phaseb_omnimath_to_math1000_scorehead_3b_seed0_20260624.log`
  - `phaseb_omnimath_to_math1000_scorehead_3b_seed1_20260624.log`
  - `phaseb_teacher_bce_priv_to_math1000_qwen3b_seed0_20260624.log`
  - `phaseb_generated_rank_nogt_to_math1000_qwen3b_seed0_20260624.log`

## Commit Hygiene

Do not stage `data/labeled/*.jsonl`, checkpoints, `.venv`, or large logs unless
explicitly requested. The publishable code changes to keep are:

- feedback-token truncation fix in `models/student.py`
- loss weight controls in `training/slfd_trainer.py`, `experiments/train_slfd.py`,
  and `scripts/run_capacity_gate.sh`
- BCE scoring objective controls in the same training/runner files
- pairwise/ranking and balanced-batch controls in the same training/runner files
- one-GPU placement controls in `models/device.py` and `scripts/run_capacity_gate.sh`
- threshold-swept diagnostic `best_f1` in `evaluation/processbench.py`
- diagnostic probe scripts in `experiments/representation_probe.py` and
  `experiments/processbench_gold_probe.py`
- diagnostic ProcessBench split helper in `experiments/make_processbench_gold_split.py`
- ProcessBench-gold transfer runner in `scripts/run_gold_scorehead_gate.sh`
- sequence-cluster bootstrap in `experiments/sequence_transfer_ci.py`
- held-out threshold calibration in `experiments/calibrated_processbench_metrics.py`
- this findings log
