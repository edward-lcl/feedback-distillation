# Final release validation

**Approved for scientific/digital distribution; not physical-print certification.**

The independent reviewer approved the scientific explanations and required only obsolete pending-status cleanup. See `independent-review/FINAL_VERIFICATION.md` and `RELEASE_NOTES.md`.

| Gate | Outcome |
|---|---|
| N1 minimal source diff | Exactly source lines 6 and 142 changed, metadata and status footer only. |
| Scientific text and quantities | Entire extracted text matches the reviewed PDF after the exact footer substitution. No scientific text or number changed. |
| Visual regression | Pixel differences at 72/150 dpi confined to acknowledgement/status line. Footer inspected in actual final render. |
| Figures, scripts and data | All original assets/data/scripts byte-identical to V2; scientific figure PDFs have no raster images. |
| Dimensions/fonts/clipping | One 2448 × 2448-pt page; all three font subsets embedded; no text spans outside page; no footer clipping. |
| Affiliations | XJTLU only for Edward; all other affiliations/markers and author metadata unchanged. |
| QR/links | Exact paper URI and hyperlink placement unchanged; actual PDF QR decoded at 72/150/300 dpi without normalization. |
| Isolated reproduction | Six commands succeeded from a copied release directory without repository/Downloads dependencies; text and 72-dpi pixels identical. |
| Preservation | Reviewed V2 directory unchanged, including candidate PDF, ZIP and final independent report. |
| Hashes | Final SHA256 manifest in `SHA256SUMS.txt`; final PDF hash in `RELEASE_NOTES.md`. |

Machine-readable validation is in `provenance/`. `provenance/reviewed-v2/` retains the source/PDF and candidate-stage README/REVIEW for comparison. Other inherited audit notes are historical records, not pending release gates. `scripts/audit_release_diff.py` reruns the exact content/pixel comparison.

Public manuscript PDF byte identity, physical print proof and printer acceptance remain unverified. Downstream reasoning improvement is not established. No new experiments or substantive claims were introduced.
