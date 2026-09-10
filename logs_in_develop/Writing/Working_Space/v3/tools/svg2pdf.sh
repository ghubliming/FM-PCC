#!/usr/bin/env bash
# Convert figures/*.svg to figures/*.pdf for the LaTeX build.
#
# WHY THIS IS A SEPARATE STEP. The figures are generated as SVG because the
# container the thesis is written in has no scientific Python stack and no SVG
# converter -- see plots/fmpcc_svg.py. \includegraphics cannot read SVG, so the
# conversion has to happen wherever the document is actually built. Keeping the
# SVGs as the committed artefact and the PDFs as build output means the figures
# stay diffable, stay inspectable in a browser and in Markdown, and never go
# stale relative to the data.
#
# thesis_v3.tex uses extension-less \includegraphics{fig_...}, so LaTeX picks up
# the PDF as soon as this has run and needs no source change either way.
#
# Usage:  tools/svg2pdf.sh [figures_dir]
set -euo pipefail

DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/figures}"
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }

# In preference order. rsvg-convert is the most faithful for this style of SVG
# (plain shapes and text, no filters); inkscape is the most widely installed;
# cairosvg is the pip fallback for a machine where neither is available.
if command -v rsvg-convert >/dev/null 2>&1;  then CONV=rsvg
elif command -v inkscape   >/dev/null 2>&1;  then CONV=inkscape
elif python3 -c 'import cairosvg' 2>/dev/null; then CONV=cairosvg
else
  cat >&2 <<'EOF'
No SVG converter found. Install one of:
  rsvg-convert   apt-get install librsvg2-bin      (smallest dependency)
  inkscape       apt-get install inkscape
  cairosvg       pip install cairosvg
None of these is present in the AI container by design; run this on the machine
that builds the PDF.
EOF
  exit 1
fi

n=0
for svg in "$DIR"/*.svg; do
  [ -e "$svg" ] || { echo "no SVG files in $DIR"; exit 0; }
  pdf="${svg%.svg}.pdf"
  # Skip when the PDF is already newer than its source.
  [ -e "$pdf" ] && [ "$pdf" -nt "$svg" ] && continue
  case "$CONV" in
    rsvg)     rsvg-convert -f pdf -o "$pdf" "$svg" ;;
    inkscape) inkscape "$svg" --export-type=pdf --export-filename="$pdf" >/dev/null 2>&1 ;;
    cairosvg) python3 -c 'import sys,cairosvg; cairosvg.svg2pdf(url=sys.argv[1], write_to=sys.argv[2])' "$svg" "$pdf" ;;
  esac
  echo "  $(basename "$pdf")"
  n=$((n+1))
done
echo "$n figure(s) converted with $CONV into $DIR"
