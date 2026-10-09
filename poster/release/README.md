# Before You Scale — final V2 release

**Approved for scientific/digital distribution. Physical-print certification is not established.**

- `final-print.pdf` — full-resolution vector-first 34 × 34-inch PDF, also suitable for digital distribution.
- `final-preview.png` — full composition; `review-crops/footnotes.png` shows final footer.
- `poster.typ`, `assets/`, `scripts/`, `data/` — editable source, reproducible figures and evidence.
- `RELEASE_NOTES.md` — independent scientific approval, exact status-only changes and limitations.
- `REVIEW.md`, `provenance/` — final checks and minimal-diff receipts.
- `independent-review/FINAL_VERIFICATION.md` — unchanged independent approval report.
- `SHA256SUMS.txt` — final file integrity manifest.

## Rebuild

Use Typst 0.14.2, locally installed Arial, and Python dependencies in `requirements.txt`. Font subsets are embedded in the delivered PDF; font source files are not redistributed. From this directory:

```sh
python3 scripts/figures.py
python3 scripts/qr.py
sh build.sh
python3 scripts/verify_evidence.py
python3 scripts/verify_pdf.py
python3 scripts/audit_release_diff.py
```

The default build uses bundled inputs, without repository or network access. The status-only source comparison baseline is under `provenance/reviewed-v2/`. The older V2 candidate itself remains untouched outside this release directory. Historical audit files retain their original candidate-stage wording; `RELEASE_NOTES.md` establishes final status.

Public manuscript PDF byte identity and physical print proof/printer acceptance remain unverified. No downstream reasoning improvement has been established.
