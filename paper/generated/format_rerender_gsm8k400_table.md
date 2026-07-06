# Format-rerender same-source GSM8K diagnostic

All rows train the same Qwen2.5-3B score-head verifier with BCE, balanced
batches, 500 training steps, batch size 2, and serial MATH1000 evaluation.
The raw generated rows use Gemma-4 labels on the same 400 GSM8K ProcessBench
candidate solutions. The PB-format rows re-render those labels into the
ProcessBench first-error convention with binary +/-1 scores and literal
`Correct.`/`Error.` feedback.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1 | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) | 0.3144 (0.2864-0.3413) | 0.2503 (0.2137-0.2747) | 0.5111 (0.2949-0.7259) |
| Same-source generated priv BCE, raw -> MATH1000 | 0-3 | 0.5494 (0.4680-0.6371) | 0.0985 (0.0809-0.1235) | 0.1931 (0.1719-0.2208) | 0.1741 (0.1472-0.2142) | 0.4696 (0.3479-0.6103) |
| Same-source generated priv BCE, PB-format -> MATH1000 | 0-3 | 0.6762 (0.5869-0.7891) | 0.1761 (0.1175-0.2811) | 0.2554 (0.1897-0.3560) | 0.2385 (0.1786-0.3464) | 0.5121 (0.1199-0.8040) |
| Same-source generated no-GT BCE, raw -> MATH1000 | 0-3 | 0.6183 (0.5849-0.6614) | 0.1152 (0.1049-0.1314) | 0.2168 (0.2035-0.2352) | 0.2088 (0.1875-0.2287) | 0.4629 (0.3860-0.5397) |
| Same-source generated no-GT BCE, PB-format -> MATH1000 | 0-3 | 0.7366 (0.7011-0.7555) | 0.2478 (0.2266-0.2722) | 0.3215 (0.2968-0.3642) | 0.3120 (0.2870-0.3539) | 0.1703 (0.1090-0.2148) |

Single-checkpoint means close 62.8% of the raw privileged gap to GSM8K gold and
88.8% of the raw no-GT gap. Mean-score artifacts are stronger: PB-format priv
4-seed mean reaches ROC-AUC 0.7789 / PR-AUC 0.2436, and PB-format no-GT
4-seed mean reaches ROC-AUC 0.7599 / PR-AUC 0.2883.

Key sequence-cluster bootstrap checks:

| Model A | Model B | ROC-AUC gap | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| PB-format no-GT 4-seed mean | Raw no-GT 4-seed mean | +0.1262 | [0.1043, 0.1476] | 0.0004 |
| PB-format priv 4-seed mean | Raw priv 4-seed mean | +0.2070 | [0.1831, 0.2305] | 0.0004 |
| GSM8K gold 4-seed mean | PB-format priv 4-seed mean | +0.0044 | [-0.0080, 0.0172] | 0.4983 |
| GSM8K gold 4-seed mean | PB-format no-GT 4-seed mean | +0.0232 | [0.0113, 0.0351] | 0.0004 |
