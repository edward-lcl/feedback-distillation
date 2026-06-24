# Active Run Status - 2026-06-24

## Current Jobs

No experiment job is currently running. Both GPUs were idle at the latest check.

For future independent score-head gates, use
`scripts/run_gold_scorehead_dual_gpu_queue.sh` with `GPUS=0,1`; it pins one
job per GPU via `TRAIN_CUDA_VISIBLE_DEVICES`/`EVAL_CUDA_VISIBLE_DEVICES` and
skips existing run tags by default.

## Latest Results

Full MATH-1000 source-specific positives:

- GSM8K gold -> MATH1000, seeds 0/1/2/3: ROC-AUC mean 0.7515
  (0.7256-0.7760), PR-AUC mean 0.2188.
- OmniMath gold -> MATH1000, seeds 0/1/2/3: ROC-AUC mean 0.7694
  (0.7539-0.7869), PR-AUC mean 0.2524.

Full MATH-1000 generated-label baselines:

- Best privileged teacher-label BCE: ROC-AUC 0.5503, PR-AUC 0.0992.
- Best generated-label overall (`rank_bal` noGT): ROC-AUC 0.6324,
  PR-AUC 0.1418.

External calibration baseline:

- Qwen2.5-Math-7B-PRM800K: ROC-AUC 0.8379, PR-AUC 0.3254, fixed F1 0.3953,
  best eval-swept F1 0.3991. It beats OmniMath seed 3 by +0.0509 ROC-AUC,
  p=0.0010.

Optional score-ensemble diagnostic:

- GSM8K+OmniMath 8-seed mean score: ROC-AUC 0.8073, PR-AUC 0.2935,
  best eval-swept F1 0.3611. This improves over any single gold-source seed
  by +0.0203 ROC-AUC under sequence-cluster bootstrap, 95% CI
  [0.0097, 0.0309], p=0.0004, but remains below Qwen PRM800K by
  -0.0305 ROC-AUC, 95% CI [-0.0487, -0.0121], p=0.0020. Held-out calibrated
  F1 is 0.3508 versus 0.3855 for Qwen PRM800K.

Training-budget diagnostic:

- GSM8K gold improves from 500 to 1500 steps for seeds 0 and 1, but seed 2
  regresses:
  - seed 0: ROC-AUC 0.7439 -> 0.7942; gap +0.0506, 95% CI
    [0.0288, 0.0722], p=0.0004.
  - seed 1: ROC-AUC 0.7760 -> 0.8111; gap +0.0351, 95% CI
    [0.0197, 0.0498], p=0.0004.
  - seed 2: ROC-AUC 0.7603 -> 0.6978; gap -0.0622, 95% CI
    [-0.0801, -0.0444], p=0.0004.
- GSM8K 1500-step 3-seed mean score: ROC-AUC 0.8153, PR-AUC 0.3130,
  best eval-swept F1 0.3990, held-out calibrated F1 0.3715. It beats the best
  generated-label baseline by +0.1830 ROC-AUC, 95% CI [0.1607, 0.2049],
  p=0.0004, but Qwen PRM800K still beats it by +0.0225 ROC-AUC,
  95% CI [0.0031, 0.0416], p=0.0220.
- OmniMath gold seed 0 is mixed at 1500 steps: ROC-AUC 0.7539 -> 0.7422
  (not significant, CI includes 0), but fixed F1 0.3052 -> 0.3414 and
  calibrated F1 0.3062 -> 0.3377.

Paired step bootstrap on the same 6,505 MATH steps:

- GSM8K gold vs privileged teacher BCE: +0.2257 ROC-AUC,
  95% CI [0.2008, 0.2501], p=0.0004.
- OmniMath gold vs privileged teacher BCE: +0.2290 ROC-AUC,
  95% CI [0.1997, 0.2574], p=0.0004.
- GSM8K gold vs best generated-label baseline: +0.1434 ROC-AUC,
  95% CI [0.1230, 0.1639], p=0.0004.
- OmniMath gold vs best generated-label baseline: +0.1466 ROC-AUC,
  95% CI [0.1239, 0.1691], p=0.0004.

Additional sequence-cluster bootstrap gaps from seeds 2/3:

- GSM8K seed 2 vs privileged teacher BCE: +0.2099 ROC-AUC,
  95% CI [0.1829, 0.2347], p=0.0010.
- GSM8K seed 2 vs best generated-label baseline: +0.1279 ROC-AUC,
  95% CI [0.1067, 0.1488], p=0.0010.
- OmniMath seed 2 vs privileged teacher BCE: +0.2072 ROC-AUC,
  95% CI [0.1773, 0.2362], p=0.0010.
- OmniMath seed 2 vs best generated-label baseline: +0.1253 ROC-AUC,
  95% CI [0.1026, 0.1487], p=0.0010.
- GSM8K seed 3 vs best generated-label baseline: +0.0930 ROC-AUC,
  95% CI [0.0694, 0.1156], p=0.0010.
- OmniMath seed 3 vs best generated-label baseline: +0.1546 ROC-AUC,
  95% CI [0.1305, 0.1773], p=0.0010.

Held-out threshold calibration on first 200 MATH sequences, evaluated on
remaining 800:

- GSM8K gold seeds 0/1/2/3: F1 0.3166 / 0.2977 / 0.2969 / 0.2714.
- OmniMath gold seeds 0/1/2/3: F1 0.3062 / 0.3294 / 0.3148 / 0.3316.
- Generated privileged BCE: F1 0.1673.
- Best generated-label baseline: F1 0.2085.

## Additional Diagnostics

Combined GSM8K+OmniMath source mixing is unstable:

- seed 0: ROC-AUC 0.7516, PR-AUC 0.2028
- seed 1: ROC-AUC 0.5166, PR-AUC 0.0932
- seed 2: ROC-AUC 0.6328, PR-AUC 0.1289

OmniMath-only replicated on MATH-400:

- seed 0: ROC-AUC 0.7569, PR-AUC 0.2107
- seed 1: ROC-AUC 0.7800, PR-AUC 0.2430

OlympiadBench full-MATH1000 boundary check:

- seed 0: ROC-AUC 0.5854, PR-AUC 0.1209; significantly below the best
  generated-label baseline by -0.0466 ROC-AUC, p=0.0010.
- seed 1: ROC-AUC 0.7163, PR-AUC 0.2029; significantly above the best
  generated-label baseline by +0.0835 ROC-AUC, p=0.0010.
- Conclusion: high variance; use as a source-distribution boundary diagnostic,
  not as a third clean headline source.

## Watch Commands

```bash
tmux list-sessions
pgrep -af 'run_processbench|train_slfd|run_gold_scorehead_gate'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 80 phaseb_olympiadbench_to_math1000_scorehead_3b_seed0_20260624.log
tail -n 80 phaseb_olympiadbench_to_math1000_scorehead_3b_seed1_20260624.log
```

## Next Decision

- Keep GSM8K longer-training as an appendix/saturation result: the 3-seed mean
  is strong, but per-seed variance is high and Qwen PRM800K still wins on ROC.
- Use the four-seed full-MATH GSM8K/OmniMath table as the main Phase B
  positive result.
- OlympiadBench is resolved as a boundary/variance diagnostic.
- Next useful work is polishing the paper-style result section and deciding
  whether to run any additional public PRM baselines beyond Qwen PRM800K.
- If running more training diagnostics, batch independent jobs through the
  dual-GPU queue rather than serial launch commands.
