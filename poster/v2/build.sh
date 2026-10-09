#!/bin/sh
set -eu
cd "$(dirname "$0")"
typst compile --root . poster.typ final-print.pdf
python3 scripts/render.py
