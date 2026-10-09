#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Python figure regeneration is documented in figures/ or scripts/ by its producer.
typst compile --root . --input layout=a poster.typ final-print.pdf
typst compile --root . --input layout=a poster.typ layouts/direction-a.pdf
typst compile --root . --input layout=b poster.typ layouts/direction-b.pdf
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -scale-to 1800 -png -singlefile final-print.pdf final-preview
  pdftoppm -scale-to 1400 -png -singlefile layouts/direction-a.pdf layouts/direction-a
  pdftoppm -scale-to 1400 -png -singlefile layouts/direction-b.pdf layouts/direction-b
fi
