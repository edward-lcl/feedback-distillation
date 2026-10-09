# Scientific figure build and interpretation

Run `python3 poster/scripts/figures.py` from any working directory. Dependencies: Python 3, NumPy, Matplotlib. The self-contained `data/figure_inputs/metrics.json` snapshot includes full-precision per-seed ROC-AUC, teacher aggregate statistics and SHA-256 hashes of source artifacts. Exact original JSON inputs are also bundled under `data/figure_inputs/results/` with original relative paths; `assets/figure_data.json` retains the initial snapshot for compatibility. Run `--refresh` only when rebuilding the snapshot from the parent repository. SVGs keep editable text; PDFs embed TrueType fonts; PNGs are review previews, not production inputs.

## Main result: `figure_rerender`

Two rows make the intervention legible while preserving the four-seed variance. Filled dots are arithmetic means of four independently evaluated seed ROC-AUCs. Open dots show individual seeds, not confidence intervals. Arrow annotations are differences of these same means, not artifact-level paired-bootstrap estimates. The dashed gold reference is also a mean across four seeds. Axes deliberately show the observed 0.45–0.82 ROC-AUC range; this is a dot plot, not a truncated bar chart.

Exact means: raw No-GT 0.6182525361; PB No-GT 0.7365938255; raw privileged 0.5494073282; PB privileged 0.6762036852; GSM8K gold 0.7514507421. Rounded differences are +0.1183 and +0.1268. Descriptive No-GT gap recovery is `(PB - raw) / (gold - raw) = 0.8884600849`, i.e. 88.8%. No confidence interval or significance is attributed to this ratio.

Data are `results/diagnostics/{generated_{nogt,priv}_gsm8k400[_pbformat]_to_math1000_qwen3b_bce_bal_seed{0,1,2,3},processbench_gsm8k_to_math1000_scorehead_qwen3b_bce_bal_seed{0,1,2,3}}/processbench_results.json`. Individual concrete paths and hashes are in the snapshot. No 4-seed mean-score ROC-AUC enters this figure.

## Supporting teacher result: `figure_teacher`

Discrete bars show solution-privileged minus No-GT first-error F1, not a smooth invented difficulty curve. GSM8K uses `results/teacher_eval/privilege_probe.json`; MATH uses `results/evidence_pack_n400/evidence_pack.json`. Only MATH has a plotted interval, taken directly from the recorded paired 95% bootstrap CI. The figure explicitly does not infer an interval for GSM8K. OlympiadBench appears only as a separate-protocol textual within-run difference from `results/teacher_eval_olympiadbench_oe/privilege_probe.json`. Its per-condition scores are not compared on the shared F1 axis.

## Cost: `figure_cost`

Linear bars start at zero. Approximately 0.3 and 15 GPU-hours are manuscript-reported training estimates (authoritative Downloads paper page 2), not independently remeasured runtimes. No runtime ledger was found; this limitation is retained in the figure caption and data snapshot. A single diagnostic is one 500-step run; the figure does not represent the entire replicated experiment or total research cost. The title recommends a check before recipe search, not guaranteed savings in every setting.

## Convention illustration: `figure_convention`

This is explicitly a conceptual trace, not an observed dataset record. It follows `scripts/rerender_labels_processbench_format.py::rerender_record`: retain the first teacher-flagged error only; map its score to −1 and feedback to `Error.`; set other targets to +1 and `Correct.`. The trace is unchanged. In particular, a later teacher-flagged error becomes a non-error target. The transformation changes target semantics and supervision conventions, not merely visual formatting. Free-text feedback is standardized. The three-step example is intentionally simple and the caption warns that `Correct.` after the first error is a convention label, not a mathematical correctness claim.

## Visual inspection

All four preview PNGs were visually inspected. A gold-reference label collision in the first draft was moved to the empty center-right region, and short-panel title clipping was corrected before handoff. Main plot uses navy/teal/slate and readable direct labels; no 3D or decorative illustration. Production must still inspect the composed PDF at actual placement sizes.
