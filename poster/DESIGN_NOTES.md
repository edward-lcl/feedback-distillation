# Design rationale

## Source and identity

The existing submitted Saksham Kapoor poster remains untouched. Its navy, teal, and white identity is preserved, as are its public author order and affiliations. The latest downloaded manuscript is anonymous, so it cannot supply public attribution. `CSU` is reproduced from the submitted poster and should be expanded by the authors if desired. The genuine Algoverse logo is extracted losslessly with its alpha mask from the supplied `SID 1st author Kiran (2).pdf`, PDF image object 19. It is not a fabricated wordmark. The extracted 576 × 98 pixel logo is the sole raster design element; using a future official SVG would improve print sharpness.

The coordinator verified the official COLM workshop size as 34 × 34 inches. Typst page dimensions are explicit at the top of `poster.typ`.

## Two rendered directions

- **A — question, contradiction, controlled diagnosis:** a horizontal three-stage story leads into a large paired raw/rerender chart; an adjacent interpretation panel explains the descriptive recovery; concrete target transformation and teacher evidence support the chart below.
- **B — two-column causal investigation:** a continuous teacher/transfer/diagnostic column sits beside the central result and example; compute dominates the bottom row.

Both alternatives are reproducible via `--input layout=a` or `--input layout=b`. Their actual PDF and PNG renders are in `layouts/`.

## Selection criteria

| Criterion | A | B |
|---|---|---|
| Research question understood at a glance | Explicit headline and three numbered stages | More reading across a long left column |
| Teacher result stays supporting | Yes: small bottom-right figure | Less successful: teacher figure starts left column |
| Main same-source comparison dominates | Largest single result panel | Also strong |
| Estimands visibly separated | Dedicated adjacent navy panel | Smaller caveat at bottom |
| Efficient reading path at square size | Clear horizontal bands | Uneven density in long columns |

**A selected.** It gives the failed teacher-to-student transfer an unmistakable place before the fixed-3B diagnosis, and reserves substantial space for the important statistical qualification.

## Scientific visualization decisions

- ROC-AUC is the student metric. Teacher F1 is clearly separate.
- Four-seed mean ROC-AUC is encoded as a large point. Individual seeds are smaller hollow points, never error bars.
- The connecting segment is a raw-to-rerender comparison within the same generated supervision source; the gold reference is a dashed line. It is not a capacity experiment.
- Rendering includes target semantics, not merely typography. The conceptual example shows later negative scores becoming non-error targets under the first-error convention; its caveat states that the label `Correct.` is not mathematical validation.
- The 88.8% callout is the descriptive gap-recovery ratio computed from unrounded per-seed mean AUCs. The independent audit corrected the original poster’s erroneous association with paired mean-score bootstrap results. No bootstrap significance is attached to this descriptive ratio.
- Compute is shown on a linear shared scale and qualified as reported approximate GPU-hours, not measured logs or a universal savings guarantee.
- No decorative 3D shapes, generated images, invented uncertainty, or SOTA claims.

## Iteration and production

Initial renders exposed overflowing author text and panel captions caused by default paragraph spacing. Paragraph/block spacing was set explicitly and PDFs were re-rendered. The final selected layout has native Typst text/shapes and vector SVG charts; Arial embeds in the PDF. The chart labels remain sharp at zoom. The original logo is intentionally small.

The conference-poster skill informed provenance, vector output, real-logo extraction, and visual QA. Its Figma/landscape/three-column defaults were superseded by the explicit user request for a reproducible editable toolchain, verified square dimensions, and alternative layouts.
