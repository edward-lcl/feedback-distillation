# V2 figure changes and checks

The V2 script regenerates only local V2 assets from `data/figure_inputs/metrics.json`. The snapshot's five four-seed means remain 0.6183, 0.7366, 0.5494, 0.6762 and 0.7515. The descriptive no-GT reference-gap fraction remains 0.8884600848770006. No new experiment or uncertainty estimate was added. `--refresh` resolves the repository root for the nested V2 directory; regular isolated builds use the bundled immutable snapshot.

## Main comparison (`figure_rerender.*`, 12 × 5.4 inches)

- Header identifies the active change: raw error flags → first-error targets.
- Filled circles/squares show four-seed mean AUCs; hollow circles/squares show individual observations in separate strips. Seed-index vertical staggering makes nearly identical AUC values distinguishable. Vertical offsets carry no statistical meaning. Color is redundant with marker shape and position for grayscale reading.
- Same-source GSM8K gold remains a descriptive reference. Hollow diamonds show its four seeds; the short horizontal line is its minimum–maximum range, not a confidence interval. The vertical dashed line is its four-seed mean. The gold caption explicitly notes input differences.
- No ensemble-bootstrap estimate, significance claim, equivalence claim, or gap-recovery causal attribution is printed in the plot.

## Teacher (`figure_teacher.*`, 7 × 2.4 inches)

- Both MATH interval caps remain on top of the bar, at z-order 5. Estimate 0.0506 and 95% interval [0.0095, 0.0925] are unchanged.
- Removed miniature footnotes and universal difficulty-window title. The poster places an observed-pattern heading and required protocol/CI qualifications in native text.
- All figure labels are at least 18 pt before placement, approximately 24.36 pt at 682 pt displayed width. Dataset sample sizes belong to native caption alongside the distinction between bootstrap evidence and manuscript-reported non-significance.

## Conceptual transformation (`figure_convention.*`, 10 × 3.7 inches)

Illustrative binary `is_error` flags [0,1,1] become [0,1,0] with reasoning unchanged. These are error flags, not literal correctness targets fed into BCE (the trainer derives correctness from them). Free-text feedback and score magnitude columns were removed. The caption states that zero after conversion is not a guarantee of mathematical correctness, and that balanced sampling also changes. The poster uses a native version of this same conceptual diagram in the navy panel.

## Visual checks

Rendered and inspected main, teacher and conceptual PNGs directly. Fixed the main-header top margin and shortened the teacher y-axis label onto two lines to avoid edge clipping. The individual no-GT rerendered seeds near 0.755 are now visibly separate. Teacher upper and lower whisker caps are visible. Full-poster placement, final font embedding, grayscale full composition and physical print acceptance are separate integration checks; standalone figure inspection does not establish those.
