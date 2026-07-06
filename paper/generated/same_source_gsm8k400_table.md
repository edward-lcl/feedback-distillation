# Same-source GSM8K generated-label transfer

All rows train the same Qwen2.5-3B score-head verifier with BCE, balanced
batches, 500 training steps, batch size 2, and serial MATH1000 evaluation.
The gold row uses the existing ProcessBench GSM8K source labels. The generated
rows use Gemma-4 labels on the same 400 GSM8K ProcessBench candidate solutions
matched to GSM8K references.

| Training source | Seeds | ROC-AUC | PR-AUC | Best F1 | Fixed F1 | Pred error rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K ProcessBench gold -> MATH1000 | 0-3 | 0.7515 (0.7256-0.7760) | 0.2188 (0.1935-0.2317) | 0.3144 (0.2864-0.3413) | 0.2503 (0.2137-0.2747) | 0.5111 (0.2949-0.7259) |
| Same-source GSM8K generated privileged BCE -> MATH1000 | 0-3 | 0.5494 (0.4680-0.6371) | 0.0985 (0.0809-0.1235) | 0.1931 (0.1719-0.2208) | 0.1741 (0.1472-0.2142) | 0.4696 (0.3479-0.6103) |
| Same-source GSM8K generated no-GT BCE -> MATH1000 | 0-3 | 0.6183 (0.5849-0.6614) | 0.1152 (0.1049-0.1314) | 0.2168 (0.2035-0.2352) | 0.2088 (0.1875-0.2287) | 0.4629 (0.3860-0.5397) |

Key sequence-cluster bootstrap checks:

| Model A | Model B | ROC-AUC gap | 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| Weakest GSM8K gold seed (seed 3) | Best same-source generated no-GT seed (seed 2) | +0.0642 | [0.0445, 0.0831] | 0.0004 |
| Weakest GSM8K gold seed (seed 3) | Best same-source generated priv seed (seed 3) | +0.0885 | [0.0688, 0.1080] | 0.0004 |
| Best same-source generated no-GT seed (seed 2) | Best same-source generated priv seed (seed 3) | +0.0242 | [0.0135, 0.0344] | 0.0004 |
