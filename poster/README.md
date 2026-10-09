# Before You Scale — editable public poster

> Independent review (2026-10-09): **requires major revision**. See [the independent report](independent-review/REPORT.md). The final PDF includes the safe CI visibility fix; scientific wording awaits author decisions. The earlier external-review ZIP predates this fix.

- `final-print.pdf`: selected direction A, vector-first 34 × 34 inch PDF. Already small enough for digital sharing; its paper QR is clickable.
- `poster.typ`: editable Typst source. Text, colors, geometry and native compute bars are editable here.
- `layouts/direction-a.pdf` and `layouts/direction-b.pdf`: two actual alternative layouts.
- `assets/`: generated vector figures, QR, and extracted genuine Algoverse logo.
- `EVIDENCE_AUDIT.md`: authoritative claim ledger and statistical qualifications.
- `DESIGN_NOTES.md`: layout selection, visual rationale, source attribution.
- `REVIEW.md`: independent scientific, visual and technical review.

## Build

Requires Typst (tested 0.14.2) and Arial. Poppler is optional for preview PNGs.

```sh
./poster/build.sh
```

The command compiles both layout directions and the selected full-resolution PDF. It does not fetch remote assets. Regenerate plots with `python3 poster/scripts/figures.py` (Matplotlib and NumPy); its JSON input/provenance files are included. `python3 poster/scripts/qr.py` regenerates and decodes the QR with OpenCV if needed. The logo is a fixed extracted source asset and must not be redrawn.

The complete PDF is approximately 125 KB, with selectable text, embedded fonts and vector charts/QR. It serves print and digital use without downsampling. No separate lossy digital rendition is necessary.

The submitted original and accepted manuscript remain outside this project and unchanged. Author affiliations follow the submitted poster; `CSU` remains as supplied and may need author clarification.

## Full regeneration and QA

Python dependencies are pinned in `requirements.txt`. With those installed:

```sh
python3 poster/scripts/figures.py
python3 poster/scripts/qr.py
sh poster/build.sh
python3 poster/scripts/verify_pdf.py
```

The build was also verified from an isolated copy of this directory. See `SOURCE_INVENTORY.md` for discovered materials and public-source checks, and `provenance/` for machine-readable validation receipts.
