# V2 candidate validation and handoff

**Status: candidate for focused independent verification. Not release approval.**

The V2 implements the response to F1–F11 without changing experimental values, accepted manuscript, V1 or reviewer-corrected files. The affiliation follow-up explicitly authorizes XJTLU only for Edward; no affiliation decision remains pending for him.

## Actual artifact checked

`final-print.pdf`, SHA256 `881acdc66487211298ccbe487cc3940a4d9b94a5aef09d095b74d5951e1672f8`, 130,003 bytes. Reviewed the complete color and grayscale compositions plus 150-dpi crops in `review-crops/`. This small vector PDF serves print and digital review without a separate downsampled version.

## Validation results

| Check | Outcome and evidence |
|---|---|
| Scientific interpretation | Internal content review found no new blocking defect in the candidate. `SCIENTIFIC_REVIEW.md` maps actual rendered text/diagram to findings. This does not replace the original independent reviewer's verification. |
| Numerical preservation | All 20 seed AUCs exactly match the independent review receipt; 23 bundled source JSON hashes verified. Five means, arrow deltas, 88.846%→88.8% descriptive fraction, teacher .0506 and [.0095,.0925] interval unchanged. `provenance/numerical_validation.json`. |
| Intervention and reference | Independently checked raw/first-error input identity; 254 no-GT and 377 privileged flags change. Gold remains 2,082 vs generated 2,080 steps, with 15/17 whitespace-normalized unmatched rows. Portable `data/intervention_evidence/check.py`; receipt `provenance/intervention_check.json`. No new training or inference. |
| Dimensions | One page, 2448 × 2448 pt = 34 × 34 inches. Saved independent-review official-page receipt supports the workshop size. |
| Fonts | All three font subsets embedded; selectable text. Exported size range 21.918–70 pt. Teacher chart labels 28.5 pt and essential caveats 25 pt. No Arial source font redistribution; rebuild requires locally installed Arial. |
| Vector/raster | Scientific plots and QR are vector. All four standalone figure PDFs contain zero raster images. The poster's only raster is the unchanged authentic logo, now 192 effective ppi. |
| CI visibility | Reviewer F3 fix retained (errorbar z-order 5). Both caps visible in actual teacher crop. No endpoint or statistical method changed. |
| Clipping/layout | No extracted text lies outside the page. Manual full composition and detail-crop inspection found no panel collisions or clipped figure labels. See `VISUAL_REVIEW.md`; geometry tests alone are not treated as visual acceptance. |
| Grayscale | Raw circles / first-error squares and separate seed strips preserve condition distinction. Gold diamonds, range and dashed mean have explicit legend. Near-equal no-GT seeds distinguishable by staggering; offsets are visual, not data. |
| Links/QR | PDF hyperlink and QR both encode `https://openreview.net/forum?id=gKNwyhDxPk`. Actual PDF QR crops decode natively at 72, 150 and 300 dpi without rescaling. No private destination. Public-page access/title/authors were verified in the saved 2026-10-09 independent-review receipt; no new live request is claimed for V2. |
| Metadata/affiliation | PDF title/subject identify V2 candidate. Author block and metadata use Xi'an Jiaotong-Liverpool University (XJTLU), with no CSU; other author affiliations/markers preserved. |
| Isolated rebuild | Copied only `v2/` to a temporary directory. Regenerated figures and QR, built PDF/previews/crops, checked source hashes/numbers, intervention evidence and PDF. All six commands returned 0. Rebuilt PDF pixels at 0.5 scale and extracted text exactly match delivered PDF; PDF creation timestamps may differ. `provenance/standalone_build.json`. |
| Preservation | All 100 pre-existing poster-project files match pre-V2 hashes, including original review files and earlier ZIP. No tracked repository file changed. Local manuscript hash unchanged. `provenance/preservation_check.json`. |

Scripts: `scripts/verify_pdf.py`, `scripts/verify_evidence.py`, `data/intervention_evidence/check.py`. Build commands and package versions: `README.md`, `requirements.txt`.

## Review scope and remaining decisions

1. Return this candidate to the independent reviewer to verify whether F1–F11 are sufficiently resolved. Our bounded internal scientific and visual checks are not that external decision.
2. Publication authority: local detailed manuscript is verified and its headline results agree with the public page, but byte identity with the current public PDF is **not established**. Authors may confirm the intended publication version; V2 makes no identity claim.
3. The manuscript's gold-ensemble equivalence assertion and precise positive-control snapshot discrepancy remain unresolved. V2 adds neither equivalence nor new inferential claims, and does not edit the accepted manuscript.
4. A truly aligned gold counterfactual, isolation of target semantics from balanced sampling, downstream reasoning evaluation and measured runtime accounting would be separate research tasks. None was run or implied here.
5. Physical full-size crop/print proof, viewing distance, phone-camera scanning, timed novice comprehension, printer bleed/CMYK/PDF-X compliance remain **untested**. Software decoding and visual inspection do not establish physical acceptance.
6. Logo stays genuine and raster. A verified higher-resolution/vector original remains optional future production input; no substitute was invented.

Stop point: V2 candidate and review evidence delivered. No commit, push, upload, accepted-manuscript edit or release action performed.
