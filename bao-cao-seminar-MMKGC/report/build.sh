#!/usr/bin/env bash
# Build the course report and copy it to the submission filename.
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error mmkgc-report.tex >/dev/null
bibtex mmkgc-report >/dev/null || true
xelatex -interaction=nonstopmode -halt-on-error mmkgc-report.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error mmkgc-report.tex >/dev/null
cp mmkgc-report.pdf 25C11050-25C15003.pdf
python -c "import fitz;d=fitz.open('25C11050-25C15003.pdf');print('pages',len(d))" 2>/dev/null || true
echo "built 25C11050-25C15003.pdf"
