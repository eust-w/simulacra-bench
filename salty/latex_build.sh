#!/bin/sh
# Usage: sh latex_build.sh <dir-under-/data/personal/latex> <main-tex-basename>
set -e
D=/data/personal/latex/$1; M=$2
rm -rf /tmp/w && cp -r $D /tmp/w && cd /tmp/w
pdflatex -interaction=nonstopmode $M.tex > /dev/null || true
[ -f references.bib ] && bibtex $M > /dev/null || true
pdflatex -interaction=nonstopmode $M.tex > /dev/null || true
pdflatex -interaction=nonstopmode $M.tex > log.txt || true
tail -3 log.txt
grep -E "Error|Overfull|undefined" $M.log | head -20 || true
grep -E "newlabel" $M.aux | head -40 || true
mkdir -p png; (gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r60 -sOutputFile=png/p%02d.png $M.pdf && echo rendered) || echo "no gs"
cp -r $M.pdf $M.log png $D/ 2>/dev/null || true
echo done
