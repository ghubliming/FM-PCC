---
name: da-figure-pipeline-svg-preview-png
description: "How thesis figures are actually produced and shipped — stdlib SVG builders, PNGs rendered by plotting/svg/preview_png.py --scale 3 (pixel-identical to the store), export_to_draft.py takes a draft path (so v5 works); partial make_figs runs clobber MANIFEST.md"
metadata:
  node_type: memory
  type: reference
  originSessionId: c237bedd-9a5c-4d2e-aa3b-e1623286bf8e
  modified: 2026-09-25T15:17:39.603Z
---

Thesis figures (`Data_Analysis/DA_in_Paper`): builders in `plotting/builders/*.py` write **SVG** with the stdlib helper
`plotting/svg/fmpcc_svg.py` (`Fig`, `save_grid`; `frame()` draws ONE subtitle line, `text()` is single-line — wrap by drawing
a second `text`). The thesis ships **PNG**: the store's PNGs are exactly `python3.14 plotting/svg/preview_png.py <svg> <png>
--scale 3` (PIL + DejaVu; verified pixel-identical 2026-09-25, so re-rendering here is faithful). No rsvg/inkscape/cairosvg in
the container, so `svg2pdf.sh` cannot run (no vector figures → the standing "PNG only" hole).

**Why:** the reviewer's clipped-text and colour findings (2026-09-25) were fixed at the source this way; `make_figs.py <match>`
rebuilds only matching figures but rewrites `figures/MANIFEST.md` with just those rows — restore it with `git checkout` after a
partial build. `export_to_draft.py` accepts a **path** (`…/Working_Space/v5`), refuses a PNG older than its SVG (render after
building), and reports v2's `fig_hardflow_endpoint_generation` as missing from the store (it lives only in the draft).

**How to apply:** fix a figure in its builder → `python3 plotting/make_figs.py <name>` → `git checkout -- figures/MANIFEST.md`
→ `preview_png.py … --scale 3` into the store → `export_to_draft.py <v5 path>` → v5 `check.py` → record the pass in
`v5/CHANGELOG.md` (the DA has no changelog of its own; the thesis changelog + git are the record). Model palette for one-panel
multi-model scatters is `sources.ENGINE_COLOUR_DISTINCT` (blue / orange / teal / near-black). See [[da-in-paper-official-store]],
[[thesis-orchestra-role]].
