# Active Run Status - 2026-06-24

## Current Jobs

No experiment job is currently running. Both GPUs were idle at the latest check.

## Latest Results

Full MATH-1000 source-specific positives:

- GSM8K gold -> MATH1000, seeds 0/1: ROC-AUC mean 0.7600
  (0.7439-0.7760), PR-AUC mean 0.2249.
- OmniMath gold -> MATH1000, seeds 0/1: ROC-AUC mean 0.7665
  (0.7539-0.7792), PR-AUC mean 0.2455.

Full MATH-1000 generated-label baselines:

- Best privileged teacher-label BCE: ROC-AUC 0.5503, PR-AUC 0.0992.
- Best generated-label overall (`rank_bal` noGT): ROC-AUC 0.6324,
  PR-AUC 0.1418.

Paired step bootstrap on the same 6,505 MATH steps:

- GSM8K gold vs privileged teacher BCE: +0.2257 ROC-AUC,
  95% CI [0.2008, 0.2501], p=0.0004.
- OmniMath gold vs privileged teacher BCE: +0.2290 ROC-AUC,
  95% CI [0.1997, 0.2574], p=0.0004.
- GSM8K gold vs best generated-label baseline: +0.1434 ROC-AUC,
  95% CI [0.1230, 0.1639], p=0.0004.
- OmniMath gold vs best generated-label baseline: +0.1466 ROC-AUC,
  95% CI [0.1239, 0.1691], p=0.0004.

Seed-0 paired gaps are also significant:

- GSM8K seed 0 vs best generated-label baseline: +0.1116 ROC-AUC,
  95% CI [0.0862, 0.1377], p=0.0004.
- OmniMath seed 0 vs best generated-label baseline: +0.1213 ROC-AUC,
  95% CI [0.0950, 0.1484], p=0.0004.

## Additional Diagnostics

Combined GSM8K+OmniMath source mixing is unstable:

- seed 0: ROC-AUC 0.7516, PR-AUC 0.2028
- seed 1: ROC-AUC 0.5166, PR-AUC 0.0932
- seed 2: ROC-AUC 0.6328, PR-AUC 0.1289

OmniMath-only replicated on MATH-400:

- seed 0: ROC-AUC 0.7569, PR-AUC 0.2107
- seed 1: ROC-AUC 0.7800, PR-AUC 0.2430

## Watch Commands

```bash
tmux list-sessions
pgrep -af 'run_processbench|train_slfd|run_gold_scorehead_gate'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 80 phaseb_gsm8k_to_math1000_scorehead_3b_seed0_20260624.log
tail -n 80 phaseb_omnimath_to_math1000_scorehead_3b_seed0_20260624.log
```

## Next Decision

- Use the two-seed full-MATH table as the main Phase B positive result.
- Do not scale source mixing until source balance or calibration is inspected.
- Next useful compute: one more source-specific seed or a held-out calibrated
  threshold evaluation if we need F1-comparable numbers.
