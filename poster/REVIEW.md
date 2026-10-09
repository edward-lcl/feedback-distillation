# Integrated release review

Reviewed 2026-10-09. **PASS for the delivered digital artifact**, with physical-print and source limitations below. All work is contained in `poster/`; accepted manuscript and original submitted poster remain unchanged. Nothing was published, committed or pushed.

## Independent reviews

- Scientific auditor: `SCIENTIFIC_REVIEW.md` — independently reconstructed the contribution and inspected actual final PDF text and render.
- Scientific visualization reviewer: `VISUAL_REVIEW.md` — independently checked plot semantics and actual final PDF rendering after production fixes.
- Coordinator: source authority, official dimensions, public-link/privacy check, rendered QR decode, fonts, vector content, isolated rebuild and original-file integrity.

## Validation outcomes

| Gate | Result and evidence |
|---|---|
| Scientific claims | PASS. Exact per-seed values, historical 1.5B results, teacher F1 and recovery formula trace to bundled inputs and `EVIDENCE_AUDIT.md`. No training rerun claimed. |
| Estimands | PASS. Main plot uses mean per-seed AUC; 88.8% is descriptive gap recovery from those unrounded means. Separate mean-score ensemble bootstrap inference is confined to the audit. No transplanted p-values or invented CI. |
| Scope | PASS. Controlled comparison fixes 3B and supervision source. No capacity-ablation, universal-format-cause, SOTA, downstream-performance, or guaranteed cost-savings claim. |
| Figures | PASS. Vector plots reconstructed from 23 bundled source JSONs. Seed variation is shown as points, not CI. Teacher MATH whisker uses the recorded interval. Conceptual example explicitly changes target semantics. |
| Layout | PASS. Two distinct rendered directions retained; A selected by explicit criteria. Reviewed actual raster output and fixed chart-label collision, footnote errors, and conceptual-panel overflow. |
| Workshop dimensions | PASS. Official COLM page explicitly states workshop 34 × 34 inches. PDF is 2448 × 2448 pt, one page, no rotation. See `SOURCE_INVENTORY.md` and saved full-page receipt. |
| Fonts and vector quality | PASS. Three embedded font subsets; scientific figure PDFs contain zero raster images. Poster's only raster is the authentic Algoverse logo, 576 × 98 px at approximately 111 ppi. No rasterization of plots or typography. |
| Clipping | PASS for automated page-bound checks and manual panel inspection. No extracted text span lies outside the page; conceptual footer fits its enlarged container. This is not an assertion that automated bounds alone establish good layout. |
| QR and public destination | PASS. Paper URL publicly readable and matches title/authors/final-result abstract. QR decoded from actual PDF crop at 72 and 150 dpi; the 300-dpi raster decodes after normalizing to 400 px for OpenCV. PDF hyperlink matches exact decoded URL. No private workspace, token, draft link or authenticated URL appears in the poster. |
| Digital delivery | PASS. Approximately 128 KB PDF with selectable text, vector QR and clickable public paper link. The same PDF serves print and digital; a second lossy version is unnecessary. |
| Reproducibility | PASS. Copied whole `poster/` to an isolated temporary directory, regenerated figures and QR, compiled both layouts, and reran PDF checks without parent-repository data or Downloads. Receipt: `provenance/standalone_build.json`. |
| Preservation | PASS. Downloaded manuscript, submitted poster and reference PDF hashes still match discovery receipts. Git status shows only new `poster/`; no existing repository file changed. |

Technical check script: `python3 poster/scripts/verify_pdf.py`; structured result: `provenance/pdf_validation.json`. Build instructions and pinned Python package versions are in `README.md` and `requirements.txt`. Typst 0.14.2 and installed Arial were used; fonts are embedded in the delivered PDF but not redistributed as source assets.

## Scientific corrections from the submitted poster

1. Removed the visual association between seed-mean endpoints and mean-score bootstrap deltas/p-values. The descriptive mean differences are +0.1183 and +0.1268; the separate bootstrap results remain documented under their own estimand.
2. Defined **88.8%** precisely as no-GT `(PB − raw)/(gold − raw)`, using unrounded four-seed mean ROC-AUCs. It is not a confidence level or a bootstrap recovery estimate.
3. Excluded the manuscript's unsupported statement that both PB-format ensembles tie gold. The no-GT ensemble's stored gold-minus-rerender CI excludes zero; the privileged comparison does not establish equivalence either.
4. Made the target-convention transformation visible, including later teacher errors becoming non-error labels. Avoided describing this as merely cosmetic formatting.
5. Replaced a headline guarantee of “50× compute savings” with the paper-reported ≈0.3 versus ≈15 GPU-hour comparison in this study.
6. Kept the 1.5B-to-3B narrative distinct from a controlled model-capacity ablation, omitted unstable/superseded best-of-N numbers, and used threshold-free student metrics.

## Before and after

The submitted landscape slide has a recognizable navy/teal identity and a useful breadth of evidence, but dense three-column text, many small plots, and mixed statistical quantities compete for attention. The square redesign preserves the identity while making one question, one controlled comparison, and one practical decision dominant. A numbered pipeline explains the contradiction; a large raw-to-rerender plot carries the result; the convention example shows what changed. The teacher and cost figures are supporting evidence. Full statistical details move to the audit and linked paper.

## Remaining limitations and human judgment

- `CSU` is preserved verbatim from the original public poster. Authors should confirm whether to expand or update this affiliation; no new institutional attribution was guessed.
- A supplied official vector or higher-resolution Algoverse logo would improve print sharpness. Current logo is genuine, not recreated, and is the only low-resolution element.
- Costs are manuscript-reported estimates; no independently auditable runtime ledger was found. Precise positive-control statistics have a manuscript/artifact snapshot discrepancy and are intentionally omitted from the poster.
- Public OpenReview title, authors, access and abstract were verified. DeepAPI public-PDF extraction returned HTTP 404, so the public PDF's byte identity was not established. Detailed scientific claims use the verified final local manuscript plus original result artifacts.
- Smallest supporting-method text is 16.625 pt, intended for close reading. Body text is generally 25–29 pt. Physical proof printing, color reproduction, and phone scanning at the intended distance remain real-world acceptance checks, not simulated by the software checks.
