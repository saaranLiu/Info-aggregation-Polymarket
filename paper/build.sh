#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../work/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../work/latex main.tex
cp ../work/latex/main.pdf main.pdf
cp ../work/latex/main.aux main.aux
