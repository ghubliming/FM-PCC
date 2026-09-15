#!/usr/bin/env bash
# Convert every SVG in the thesis figure store (Data_Analysis/DA_in_Paper/figures,
# all groups) to PDF, next to the SVG. Then run plotting/export_to_draft.py so the
# draft receives the PDFs too.
#
# WHY THIS IS A SEPARATE STEP. The figures are generated as SVG because the
# container the thesis is written in has no scientific Python stack and no SVG
# converter -- see plotting/svg/fmpcc_svg.py. \includegraphics cannot read SVG, so the
# conversion has to happen wherever the document is actually built. Keeping the
# SVGs as the committed artefact and the PDFs as build output means the figures
# stay diffable, stay inspectable in a browser and in Markdown, and never go
# stale relative to the data.
#
# The drafts use extension-less \includegraphics{fig_...}, so LaTeX picks up the
# PDF as soon as it has been exported and needs no source change either way.
#
# Usage:  plotting/svg/svg2pdf.sh [figures_dir]   (default: the whole store)
set -euo pipefail

DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/figures}"
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
while IFS= read -r svg; do
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
done < <(find "$DIR" -name '*.svg' | sort)
echo "$n figure(s) converted with $CONV into $DIR"
