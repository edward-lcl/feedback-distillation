# COLM poster V2 — final release

**Status: approved for scientific/digital distribution.** This approval is distinct from physical-print certification. Finalized 2026-10-09.

## Independent scientific approval

`independent-review/FINAL_VERIFICATION.md` is preserved byte-for-byte from the focused independent review. Its final recommendation is **“Release-ready after minor corrections.”** The reviewer found no remaining scientific blockers or major issues and closed the scientific review conditional only on N1: removing obsolete pending-verification wording while preserving reviewed content.

Reviewed candidate PDF SHA256: `881acdc66487211298ccbe487cc3940a4d9b94a5aef09d095b74d5951e1672f8`.

The reviewed V2 directory and ZIP remain unchanged. Final release is separate under `poster/release/`. Historical candidate-stage audits and reports retain their original wording as provenance; this note and the final `REVIEW.md` state the current disposition.

## Exact release changes

Only two Typst source lines differ from the independently reviewed source:

- **Line 6, metadata:** title changed from `Before You Scale — COLM poster V2 candidate` to `Before You Scale — COLM poster V2`. Description changed from `Scientifically qualified V2 poster candidate. Edward affiliation: Xi'an Jiaotong-Liverpool University (XJTLU). Pending independent re-review.` to `COLM 2026 Workshop on Efficient Reasoning poster. Edward affiliation: Xi'an Jiaotong-Liverpool University (XJTLU).` Author metadata is unchanged.
- **Line 142, footer:** `We thank Algoverse AI Research. V2 candidate · pending focused independent verification.` changed to `We thank Algoverse AI Research. COLM 2026 · V2.`

No scientific text, number, uncertainty interval, figure, author affiliation, branding or layout parameter changed. `provenance/status-only.diff` records the complete source diff. Documentation and validation receipts were added/updated to record release status; the reviewed candidate documentation is preserved.

## Final validation

- Exact source comparison permits only the two intended status-line replacements.
- Extracted PDF text is identical after that exact footer substitution. Scientific content and all numerical text are unchanged.
- At 72 and 150 dpi, every changed pixel is confined to the acknowledgement/status line. Final footer visually inspected; no clipping.
- Figure assets, input data and all pre-existing scripts are byte-identical to V2. A new standalone minimal-diff audit script was added.
- One 34 × 34-inch page; embedded fonts; no text outside page bounds; vector scientific figures and QR. Authentic raster logo unchanged at 192 effective ppi.
- Paper hyperlink and its rectangle unchanged; QR decodes from the actual PDF at 72, 150 and 300 dpi without resizing. Author affiliations, including XJTLU only for Edward, verified unchanged.
- Isolated rebuild regenerates figures and QR, compiles, and runs evidence/PDF/diff checks. All commands passed; rebuilt text and 72-dpi pixels are identical to the delivered release.
- Reviewed candidate PDF and ZIP hashes match the independent report; every pre-existing V2 file retains its original hash. No manuscript or tracked repository file was modified.

Detailed receipts: `provenance/minimal_diff_audit.json`, `pdf_validation.json`, `unchanged_content.json`, and `standalone_build.json`. `SHA256SUMS.txt` records final release-file hashes, excluding the manifest itself.

Final PDF SHA256: `3e7d3d3d5321c54d43471c0087a17be26b5916afce85fe476780edd18da11243`.

## Preserved limitations

- Public manuscript PDF byte identity remains **unverified**. The poster does not claim identity with the local reviewed detailed manuscript.
- Physical print proof and printer acceptance remain **unverified**, including printer-specific color/bleed/PDF-X requirements and physical viewing/scan distance.
- **No downstream reasoning improvement has been established.** The poster's first-error ranking result and explicit downstream limitation remain unchanged.
- No new live public-destination check, experiment, runtime measurement, or timed attendee-comprehension test was performed during this status-only finalization. Existing scientific qualifications and saved public-page evidence remain in force.

These are documented limits, not reopened scientific findings. No upload or public posting was performed by this task.
