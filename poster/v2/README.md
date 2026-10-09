# Before You Scale — V2 review candidate

**Status: candidate for focused independent verification; not release-ready certification.** All V1 and reviewer-corrected files remain unchanged outside this directory.

Start with:

- `final-print.pdf` — 34 × 34-inch vector-first PDF, usable for print or digital review.
- `final-preview.png` and `grayscale-preview.png` — complete composition.
- `review-crops/` — intervention, central chart, teacher, footer, reference and compute details.
- `REVIEW_DISPOSITION.md` — F1–F11 responses, including author-resolved affiliation.
- `EVIDENCE_AUDIT.md`, `SCIENTIFIC_REVIEW.md`, `VISUAL_REVIEW.md`, `REVIEW.md` — evidence, bounded checks and outstanding limits.
- `poster.typ`, `scripts/figures.py` — editable source and reproducible plots.

## Build from any copy

Requires Typst 0.14.2 and Arial installed locally. The delivered PDF embeds font subsets; font source files are not redistributed. Python dependencies are in `requirements.txt` (Python 3.14 was used). Run from this directory:

```sh
python3 scripts/figures.py
python3 scripts/qr.py
sh build.sh
python3 scripts/verify_evidence.py
python3 data/intervention_evidence/check.py
python3 scripts/verify_pdf.py
```

Default figure regeneration reads bundled data and needs no repository or network. `--refresh` is optional repository maintenance, not part of the portable build. The conceptual table in the actual poster is native editable Typst; `figure_convention.*` is its standalone companion. The standalone cost chart is retained as a reproducible asset, but V2 uses text to make the accounting unit explicit.

The public-paper QR and hyperlink retain `https://openreview.net/forum?id=gKNwyhDxPk`. Saved public-page receipts verify its title/authors and the workshop dimensions; they do not establish byte identity of the local detailed manuscript and current public PDF.

## Review request

Please check F1–F11 against the actual PDF and source evidence, rather than treating our internal checks as final resolution. Focus on the target/sampling qualification, gold reference and descriptive ratio, CI visibility, grayscale seed distinctions, first-minute explanation and limitations. Flag exact locations and distinguish scientific issues from design preferences. No new experiment, manuscript change or release is authorized by this package.
