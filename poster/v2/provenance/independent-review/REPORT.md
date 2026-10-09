# Independent COLM poster release review

**Recommendation: requires major revision.** The headline result reproduces. The necessary revisions concern the explanation of the intervention and the gold-reference comparison, not a replacement experiment or a wholesale redesign. One objectively verifiable plotting defect has been fixed. No scientific wording or experimental quantity was changed.

Reviewed 2026-10-09 at repository HEAD `e598eaa077df71463f6bb9ae9a23fd6c1a09f35c`. Selected output: `poster/final-print.pdf`; editable layout: `poster/poster.typ`; editable figures: `poster/scripts/figures.py`. The poster directory was already untracked when this review began. No commits or pushes were made.

## Independence and manuscript authority

`REVIEW.md`, `SCIENTIFIC_REVIEW.md`, and `VISUAL_REVIEW.md` remained unopened throughout this review. The first assessment was saved as `FIRST_ASSESSMENT.md` before any correction. I had read `SOURCE_INVENTORY.md` and `EVIDENCE_AUDIT.md` before the clarification, so this was not a fully blinded assessment. Their assertions were not treated as validation.

I read all nine pages of `/Users/edward/Downloads/171_Before_You_Scale_The_Cost_-1.pdf`. SHA256: `8c0f8acadb203fd6a153c84002dd46de5a10fecd1914d89eff6f015b6f10368e`. Two other downloaded copies are byte-identical. The repository's `paper/slfd_colm_er_final.pdf` is an older 12-page export. The later-dated `Downloads/main(9).tex` is a different paper, not a newer SLFD manuscript. The reviewed nine-page PDF includes the completed rerender results but still carries anonymous-review headers.

A fresh DeepAPI scrape verified the [public paper page](https://openreview.net/forum?id=gKNwyhDxPk), author order, and headline abstract values, and the [official conference page](https://colm.cc/) specifying 34 × 34 inches for workshops. Receipt: `public-check.json`, request `fbc49523-bd0e-4323-9b19-e13ae19ccad4`. This establishes the newest **locally verified detailed manuscript**, not byte identity with the current public PDF revision. Public PDF revision identity remains unverified.

## Findings, ordered by consequence

### F1 — Major: identify the active intervention, not just its file rendering

**Exact location:** central title and navy interpretation panel, `poster.typ:68–85`; lower-left conceptual figure, `scripts/figures.py:78–91`.

**Evidence/reasoning:** The rerender script changes `is_error` so that only the first teacher-flagged error remains an error target. Direct file comparison finds 254 changed targets for no-GT (425 → 171 errors) and 377 for privileged (571 → 194), each over 2,080 steps. The raw and rerendered problem/prefix/step inputs are identical in order. In the score-only BCE path, `training/slfd_trainer.py:198–242` uses `is_error`; feedback LM loss is disabled (`experiments/train_slfd.py:30–35`, `scripts/run_gold_scorehead_gate.sh:88`). Score magnitude and “Correct.”/“Error.” prose therefore are not active supervision channels for this comparison. Changing `is_error` also changes membership of the two balanced sampling pools (`training/slfd_trainer.py:143–170`). This is a substantive target-convention change with a sampling consequence, not evidence that cosmetic text formatting rescued learning.

The poster correctly says “targets” and explains that later labels need not mean mathematical correctness. But its visually prominent score/feedback example and generic “rendering convention” takeaway leave the active mechanism ambiguous.

**Recommended correction:** retain the canonical supervision-mismatch framing, but explicitly say: “Keep the reasoning text fixed; train on the first teacher-flagged error rather than all flagged errors.” Add that target relabeling also changes balanced sampling, and that feedback wording is unused by this score-only diagnostic. Treat target semantics versus sampling as unseparated mechanisms; do not claim an isolated typography/feedback effect. **Author approval required; not edited.**

### F2 — Major: the gold denominator is not an identical-input counterfactual

**Exact location:** setup card `poster.typ:59–61`, central gold line (`scripts/figures.py:45–46`), and “88.8% gap recovery” (`poster.typ:81–85`).

**Evidence/reasoning:** Raw versus rerendered generated traces match exactly. Generated versus gold training inputs do not. The generated files have 2,080 steps; `results/diagnostics/processbench_gsm8k_gold_train400_steps.jsonl` has 2,082. There are 424 generated and 426 gold unmatched exact `(problem, prefix, step)` rows; most differences are whitespace. After whitespace normalization, 15 generated and 17 gold rows remain unmatched across the Sandy, Sarah, and Raymond problems. These include different step segmentation, not merely labels. The 400 generated solutions cover 375 distinct problem strings.

The five mean AUCs and the recovery ratio are correct. The plot does not explicitly assert byte identity with gold, but the combination of “same traces,” a gold target line, and a percentage recovery invites that stronger reading. The observed raw→PB improvement remains valid; the gold gap cannot be attributed purely to annotation provenance or label quality.

**Recommended correction:** qualify “same traces” as applying to raw→rerendered only. Identify gold as a same-source reference with minor input/segmentation differences. Keep 88.8% only as the descriptive gap-to-this-reference ratio it is; do not call it percentage of failure causally explained. If a pure label-only gold comparison is essential, authors must approve construction and evaluation of a truly aligned control. **Author approval required; no data repair or new training performed.**

### F3 — Major, fixed: the teacher confidence interval was partially hidden

**Exact location:** lower-right teacher figure, MATH +0.051 bar; `scripts/figures.py:61–65`.

**Evidence/reasoning:** The bar used z-order 3 while the errorbar used a lower default order. The entire lower half of the interval was painted behind the bar, making the displayed uncertainty look one-sided. The committed evidence pack gives estimate 0.0506 and interval [0.0095, 0.0925].

**Correction applied:** set errorbar z-order to 5, leaving every estimate and endpoint unchanged. Regenerated only the teacher figure assets, then rebuilt the final PDF and both layout PDFs. The corrected render shows both caps. Extracted poster text is identical before/after. At 72 dpi the only changed poster pixels occupy `(2161,1832)–(2178,1871)`, the newly exposed lower whisker. Originals are preserved under `independent-review/original/`.

### F4 — Major for an unfamiliar attendee: the question is clear; the mechanism is not yet a one-minute explanation

**Exact location:** title “PRM Distillation,” card 02 “no-GT,” card 03 “LoRA/BCE,” and conceptual figure “PB score / feedback.”

**Evidence/reasoning:** The question, failed teacher→student transfer, and recommendation to check supervision before scaling are easy to locate. But PRM, no-GT, and PB are never expanded; the teacher and student model families are omitted. A reader can see an AUC rise without understanding that the evaluated target is the first erroneous step rather than all mathematically wrong steps. The navy panel repeats four values already printed in the plot, using space that could explain this distinction. This is a reviewer walkthrough, not a timed user study.

**Recommended correction:** define a process reward model as “a verifier that scores reasoning steps”; replace no-GT with “teacher without reference solution” at first use; name Qwen2.5-3B in the setup. Use one short sentence for the target change (F1). Keep the central arrow plot and conceptual example. Shorten the repeated numerical paragraph, and move recipe abbreviations to a methods line or QR-linked details. **Proposed copy/layout change; not applied.**

### F5 — Minor: seed variability is present but difficult to distinguish

**Exact location:** central plot's hollow circles and dashed gold line; `scripts/figures.py:45–56`.

**Evidence/reasoning:** No-GT PB seeds 0.75449 and 0.75545 nearly overlap. Raw and rerendered seeds share one horizontal strip and the same marker shape, relying on hue to identify conditions; that distinction weakens substantially in grayscale. Gold is shown only as a line despite its four-seed range of 0.72564–0.77602. Privileged PB is especially variable (0.58692–0.78915), so a mean alone is not a reliability guarantee. Positively, the chart labels dots as seeds, not confidence intervals, and correctly avoids attaching ensemble bootstrap p-values to mean-per-seed AUCs.

**Recommended correction:** separate the two seed strips slightly, use distinct marker shapes, and show the gold seeds/range as descriptive variation. Avoid a “ties gold” claim or an uncertainty interval for 88.8% without the corresponding analysis. No significance calculation is needed merely to improve the display. **Not edited.**

### F6 — Minor: a scoped benchmark result needs a visible downstream limitation

**Exact location:** bottom-center scope block, `poster.typ:110–113`, and central title “verification signal.”

**Evidence/reasoning:** Manuscript §7 explicitly says downstream validation of the 3B result is incomplete: no demonstration that the gold-trained 3B verifier beats majority vote on best-of-N. The poster correctly avoids presenting the fragile historical downstream numbers, but an attendee could still equate improved first-error ranking with better mathematical answers.

**Recommended correction:** add “Improved first-error ranking; downstream reasoning gains not established.” Keep the existing off-policy, separate-verifier, and non-capacity-ablation qualifications. **Author approval required; not edited.**

### F7 — Minor: the teacher “window” is a scoped empirical pattern, not a general law

**Exact location:** lower-right title/caption, `poster.typ:95–99`, `scripts/figures.py:66–68`.

**Evidence/reasoning:** GSM8K uses N=50, MATH N=400, and OlympiadBench a separate protocol. MATH's positive estimate and interval match the committed pack. The GSM8K artifact gives −0.0517 but no CI; “not significant” is manuscript-reported, not independently re-established here. Non-significance is not evidence of zero benefit. The underlying MATH-by-level results support context dependence, but the two-bar display alone does not isolate difficulty from dataset/protocol differences.

**Recommended correction:** label this “Observed teacher pattern in these probes,” or explicitly qualify the difficulty-window caption by datasets/protocols. Do not infer a fitted inverted-U or no benefit outside MATH. **Author approval required; not edited.**

### F8 — Minor: the cost comparison's unit is underspecified

**Exact location:** bottom-left GPU-hour bars, `poster.typ:101–108`.

**Evidence/reasoning:** Approximately 0.3 and 15 GPU-hours are manuscript estimates (§1/§8), not independently logged timing measurements. The 0.3 estimate is one 500-step run, not the full four-seed, multi-condition study. “Reported” and approximation signs help, but this distinction is omitted from the selected layout. The bars themselves correctly use a 1:50 length ratio.

**Recommended correction:** label “One 500-step diagnostic run ≈0.3 GPU-h; reported estimates for this setup.” Avoid guaranteed 50× savings or an amortized/end-to-end cost claim without hardware and accounting records. **Author approval required; not edited.**

### F9 — Minor: essential teacher caveats are close-reading text

**Exact location:** lower-right teacher footnotes; `scripts/figures.py:68`, placed at `poster.typ:97`.

**Evidence/reasoning:** Actual exported sizes are 16.625 pt for protocol/CI caveats, about 19 pt for “not significant,” and 20.6 pt for dataset labels. At 34-inch print size these are materially smaller than 25–29 pt body copy. The smallest type has only a 5.9 mm em, so it should not be expected to read comfortably from 1–2 m. The main title and central result hierarchy are strong; no blanket reduction of all text is warranted.

**Recommended correction:** bring the protocol caveat into a ≥24 pt caption, trim duplicate teacher title/caption text, and prioritize it over decorative whitespace. Confirm with a full-size printed crop. **Not edited; physical viewing not tested.**

### F10 — Minor: raster branding is softer than the otherwise vector output

**Exact location:** top-right Algoverse mark; `poster.typ:28`, `assets/algoverse-extracted.png`.

**Evidence/reasoning:** The 576×98 raster is placed 374 pt wide: 110.9 effective ppi. The charts, QR, and text are vector, so the logo is the one conspicuously lower-resolution component. Its proportions are preserved. The navy/teal palette is coherent; small shade differences across figure/layout sources are not a release-level defect.

**Recommended correction:** replace with an authorized vector original when available, or use a smaller placement. Do not redraw the brand or claim upsampling adds real detail. **Not edited; no verified replacement supplied.**

### F11 — Minor: affiliation and release-version provenance remain author checks

**Exact location:** affiliation “Xi’an Jiaotong-Liverpool University / CSU,” `poster.typ:25`; paper QR, `poster.typ:118`.

**Evidence/reasoning:** The public page confirms names and author order, not this institutional shorthand. “CSU” is ambiguous. The detailed local manuscript is anonymous and not byte-verified against the public revision. The QR itself works and reaches the correct titled paper page.

**Recommended correction:** author confirms exact affiliation wording and that the reviewed local manuscript is the intended release authority. Expand CSU only with that confirmation. **Not edited.**

## What works for an attendee

The central raw→rerendered AUC plot is the strongest figure: one axis, paired comparisons, clear endpoints, and a visible reference. Keep it. The three numbered cards establish a useful narrative order. The conceptual example communicates the crucial “later error becomes non-error target” issue, but should foreground the active binary target rather than feedback prose. The teacher chart is supporting context and can shrink in narrative priority, not in font size. The tiny cost bars add less than the takeaway sentence; their space is better spent clarifying “one run.”

A plausible 60-second route is: question (10 s), teacher/student failure (10 s), central plot (20 s), target-convention example (15 s), practical recommendation and scope (5 s). The current version supports the question and result portions; F1/F4 are needed to make the diagnosis self-contained. No real novice timed test was performed.

## Numerical and statistical validation

Recomputed directly from all 20 committed `per_step_scores.json` files, not just the poster snapshot:

| Condition | Four-seed mean ROC-AUC | Poster rounding |
|---|---:|---:|
| Raw no-GT | 0.6182525361 | 0.6183 |
| PB no-GT | 0.7365938255 | 0.7366 |
| Raw privileged | 0.5494073282 | 0.5494 |
| PB privileged | 0.6762036852 | 0.6762 |
| GSM8K gold | 0.7514507421 | 0.7515 |

Every ROC-AUC and average-precision value matches its committed metric JSON. Both metrics and score-file bytes match HEAD. The no-GT arrow is +0.1183413; privileged +0.1267964. Recovery is 88.8460085%, correctly displayed as 88.8%. These are descriptive means, not the AUC of averaged model scores.

Historical 1.5B values 0.6309053 and 0.6405775 match the poster's bundled files and Git commit `7c54a9d4fa8b6103eae7fc7ce098cada252de77b`; those metrics contain 2,589 evaluated steps. The current `results/ablation/transfer_ci.json` instead describes 6,505 steps, so it must not be silently attached as the exact uncertainty for this historical displayed pair. The poster does not print such a CI.

The manuscript's statement that both rerendered ensembles are “statistically tied” with gold is not supported: the committed gold-minus-PB-no-GT ensemble interval is [0.0112818, 0.0350949]. Even an interval including zero would not establish equivalence. The poster appropriately omits that assertion. The positive-control text receipt also differs from the manuscript's exact numerical snapshot (+0.0481 versus +0.0497); the poster avoids repeating either precise CI. These inherited manuscript discrepancies require reconciliation before adding more inferential claims. Existing bootstrap intervals were inspected, not recomputed; no new hypothesis tests were run. Sequence bootstrap also does not account for dependence between distinct solutions sharing a problem string (955 distinct problems in 1,000 evaluation solutions).

## Validation summary

| Check | Result and limit |
|---|---|
| 20 seed ROC-AUC and AP recomputations | **PASS**; committed bytes and snapshot hashes checked |
| Means, arrow deltas, 88.8% arithmetic | **PASS** |
| Historical displayed 1.5B metrics | **PASS** against historical Git bytes; exact matched CI not established |
| Raw→PB input identity | **PASS** for both conditions; target changes counted |
| Gold identical-input control | **FAIL**; segmentation/input differences documented in F2 |
| Evaluation alignment | **PASS**; all saved label vectors match local evaluation order, 1,000 solutions / 6,505 steps / 594 errors |
| Train/eval exact problem overlap | **PASS**: zero; not a proof against paraphrase or pretraining contamination |
| Editable-source compilation | **PASS** for final and both layout PDFs; source and originals preserved |
| Page size/count | **PASS**: one 2448×2448 pt page (34×34 in), live venue match |
| Embedded fonts/selectable text | **PASS**; Arial variants plus embedded Libertinus superscript fallback |
| Clipping/overlap | **PASS** for page bounds and visual inspection of selected layout; no text outside page; CI occlusion fixed |
| Graphics/export | **PASS with caveat**: charts and QR vector, one 110.9-ppi raster logo; unencrypted PDF 1.7 |
| QR digital decode | **PASS** at native 72/150 dpi; 300-dpi raster decoded after resizing to 400×400, not natively |
| QR destination | **PASS** fresh public page extraction; public PDF bytes not verified |
| Grayscale | Main comparisons still readable; seed-condition identity weak, F5 |
| Safe-fix isolation | **PASS**: identical extracted text, pixel changes confined to lower CI whisker |
| Viewing-distance/print acceptance | **NOT TESTED physically**; based on dimensions, type sizes, and rendered inspection |
| Printer-specific bleed/CMYK/PDF-X | **NOT VERIFIED**; MediaBox/CropBox/TrimBox/BleedBox coincide; no printer requirement was supplied |
| End-to-end compute and downstream benefit | **NOT VERIFIED**; no runtime ledger or new downstream evaluation |

Reproduce the checks with `python3 poster/independent-review/check_evidence.py` and `python3 poster/independent-review/check_pdf.py`. Machine-readable receipts: `evidence-check.json`, `pdf-check.json`, `public-check.json`. The old external-review ZIP was not rebuilt and should not be used as the corrected release artifact.

## Release decision

**Requires major revision.** Resolve F1, F2, and F4 through author-approved explanation/qualifications; F3 is already fixed. The remaining issues are bounded polish and explicit limitations. The evidence supports the observed raw→first-error-target improvement; it does not justify treating all gold-gap recovery as a pure formatting effect or general reasoning improvement. No new model run is required merely to state the existing result accurately.
