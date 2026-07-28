# DA_Code v3 — U9: per-plot "Plot Legend" box (selected candidates → path)

**Scope:** `Data_Analysis/Visualizer/index.html` only. No Python DA / no cluster re-run.
**Depends on:** U8 (bar value labels), U7 (hardflow variants + HTML render fixes).

## What changed

Added a new box **directly under the plot**, between the chart (`#plot-area`) and the
existing **Path Audit Map** (`#path-section`). It shows only the candidates in the *current*
plot — numbered in the same left-to-right order they appear on the x-axis — so each plotted
bar-group's index can be read straight to its source path without scanning the full audit
table (which still lists every candidate in the batch).

Example: the plot's 1st x-group → `Plot X-Pos 1 · CAND_19 · logs/.../CAND_19_path`.

### Edits

1. **New container** (between `#plot-area` and `#path-section`), blue dashed top border to
   distinguish it from the full audit map:
   ```html
   <div class="path-section" id="selection-map-section" style="display:none; border-top: 2px dashed #007bff; margin-bottom: 0;">
     <label style="color:#007bff;">Plot Legend — Selected Candidates (this plot only)</label>
     <div id="selection-map-container"></div>
   </div>
   ```

2. **New `render_selection_map(x_categories, x_axis, checked_cands)`** (next to
   `render_path_map`). Builds the `Candidate → Full_Path` dict the same way the audit map and
   `download_plot` do (`Full_Path` else `Folder_Name`), then renders a 3-col table:
   `Plot X-Pos | ID | Source Path`, reusing the `p-seg-{i%6}` path-segment colouring.

3. **Call in `trigger_plot`**, right after the zoom re-apply, feeding `pivot_mean.index`
   so the legend order **exactly matches the plotted x-axis** (which is the pandas groupby
   sort order, not the checkbox order).

## Design notes

- **Plot-order fidelity** — the legend is built from `pivot_mean.index.tolist()`, i.e. the
  real x-axis categories after the `groupby([x_axis, 'variant'])` unstack. This is why the
  numbering lines up with the bars even though `Candidate` is a string (lexicographic pivot
  sort: `"1","10",...,"2"`), which differs from the sidebar's numeric-aware candidate list.
- **Env mode** — when Analysis Mode = *By Environment*, the x-axis is `halfspace_variant`
  and candidates are averaged into each env group, so there is no candidate x-order. In that
  case the box falls back to listing the **selected** candidates (`checked_cands`) with a
  note that they are averaged, not positional. In *By Candidate* mode the header reads
  `Plot X-Pos` and the note says left-to-right.
- **Path-not-found** — a candidate with no path row shows `— path not found —` rather than
  crashing (mirrors the audit map's tolerance for missing facets, U7 §9/§10).
- **Visibility** — the box starts hidden and is shown by `render_selection_map`; it refreshes
  on every re-draw alongside the plot.

## Verify

- Reload `index.html`, SYNC a batch, select ~6 candidates → a "Plot Legend" table appears
  between the chart and Path Audit Map, one numbered row per plotted bar-group, index → path.
- Toggle Analysis Mode to *By Environment* → box switches to the averaged-candidates listing.
- Static: the HTML/py block compiles; path colouring reuses existing `p-seg-*` CSS.

No cluster re-run — reload the page and re-SYNC.
