# V2 design notes — focused revision of V1

V2 retains the 34 × 34-inch square, navy/teal palette, title, prominent question, numbered narrative, dominant AUC comparison and public-paper QR. It is a candidate for the independent reviewer's focused verification, not a fresh layout competition or publication release.

## Surgical changes

| Area | Before | V2 |
|---|---|---|
| Entry | PRM/no-GT jargon; generic fixed 3B | Header defines a PRM; card 02 defines no-GT; card 03 names Qwen2.5-3B-Instruct and score-only setup. ProcessBench is described as a benchmark for locating first reasoning errors. |
| Hero comparison | Raw labels → rendering convention | Raw error flags → first-error targets. Exact means and arrows remain. Circles versus squares and staggered seed strips work without color. |
| Navy companion | Repeated chart endpoints and recovery percentage | Native editable binary-flag example at identical reasoning inputs. States that targets and balanced-sampling pools change together, while feedback/score magnitude are inactive. |
| Gold reference | Dashed mean line alone, recovery near causal interpretation | Gold gets its own seed row, descriptive range and mean. A separate reference panel states input/segmentation differences and defines 88.8% as measured-gap arithmetic, not causal attribution. |
| Teacher context | Difficulty-window title and tiny footnotes | Observed pattern in these probes; both CI caps visible. Essential protocol/sample caveats are native 25 pt text; chart labels 28.5 pt in the poster. |
| Compute | Small 1:50 bars and ambiguous diagnostic total | Two short numeric comparisons identify one 500-step run and setup-specific estimates. Bars removed to give the accounting caveat room; no additional chart added. |
| Scope | Small bottom note | Downstream reasoning limitation leads the footer in 33 pt bold type, followed by off-policy and capacity qualifications. |
| Branding | 374 pt-wide original raster | Same genuine pixels at 216 pt width (192 effective ppi). No redrawing, fabricated replacement, or invented resolution. |
| Attribution | Edward: XJTLU / CSU | Edward: Xi'an Jiaotong-Liverpool University (XJTLU), marker 4, explicitly author approved. Other authors/markers unchanged. PDF metadata updated. |

## Reader route (design intent, not a timed test)

Question → better teacher / failed transfer → fixed-student comparison → active first-error target change → descriptive gold reference → check targets before more compute. The target table is now beside the AUC plot, making it the main explanatory companion. The teacher plot remains supporting context. Small gaps and typographic rhythm are preserved rather than filling every area with another chart.

The 0/1 example denotes **error flags**, not the numerical correctness targets passed to the BCE loss (their complements). It is explicitly conceptual, not claimed to be an observed data record. A later zero means the step is not selected as the first flagged error, not that its reasoning is mathematically correct. Actual problem, prefix and step inputs are unchanged only for raw→first-error generated pairs; that identity is not extended to gold.

## Rendered iterations and accessibility

Rendered the complete PDF in color and grayscale, and 150-dpi crops of the intervention, chart, teacher, footer, reference and compute panels. After the initial render, made no-GT definition unambiguous, added “Step” labels to the conceptual rows, and linked “first-error targets” to the manuscript's “PB-format” term. The complete CI remains above the teacher bar (z-order 5). Small seed markers are separated by shape and vertical offset; those offsets carry no additional experimental quantity.

The chart's AUC x-axis is visibly labeled from .45 to .80. Point positions encode AUC; there are no zero-origin bars whose lengths could be mistaken for relative performance. Gold's min/max line is descriptive seed spread, explicitly distinguished from confidence intervals. No ensemble-bootstrap p-value or interval is attached to seed-mean arrows or the gap ratio.

This file explains design decisions. `REVIEW.md` records technical outcomes and limits. Physical print proof and novice one-minute comprehension were not tested.
