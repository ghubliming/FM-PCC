# DA_Code v3 — U18.2: the Pareto gets its OWN size controls (one widget set + a target selector)

**Scope:** the two HTML viewers only —
`Data_Analysis/Visualizer/index.html` (DAv3, avoiding) and
`Data_Analysis/Visualizer_VA_v2/index.html` (DA_VA_v2, visual aligning), the latter
**regenerated** from the former via `Visualizer_VA_v2/build_from_dav3.py`.
No DA pipeline change, no CSV schema change, no cluster re-run, no new dependency.
**Depends on:** U18 (the Pareto itself), U18.1 (the Pareto as a separate figure/section).
Toolbar tag bumped `SCIENTIFIC_SUITE_v3.15` → `v3.16`.

**Not touched:** `Visualizer_UAV_v1/index.html`. `build_from_va2.py` was re-run only to check
every anchor still matches (41 edits, clean) and its output reverted, same policy as U18/U18.1.

---

## The problem

U18.1 gave the Pareto its own figure and its own page section, but not its own **size**. It was
still drawn at `figsize=(calc_width, PARETO_H)` — `calc_width` being the *main plot's*
`8. Matplotlib FigWidth` — and `applyZoom()` stretched both `<img>`s by the one
`SMALLER`/`LARGER` scale. So the two plots were separated everywhere except in the only place
the user actually touches them.

That is worse than it sounds, because the two figures do not want the same width. The bar chart
wants width per x-tick group. The Pareto carries **two right-hand legends** (candidate shapes /
variant colours, and the "not a perfect goal+constraint run" list) outside its axes, so at the
bar chart's width its plotting area is squeezed to whatever the legends leave over, and every
point lands in the left third of the panel. Widening it meant widening the bar chart too —
which then no longer fits the screen, and the trade-off you were trying to read is off-screen
in the other direction.

## What was added

Not a second copy of every widget. Section 7 keeps **one** `SMALLER`/`LARGER` pair and **one**
FigWidth box, and gains a selector saying which plot they are currently editing:

```
7. Visual Zoom (Magnify) — MAIN PLOT          ← label tracks the selection
   [ CONTROL: MAIN PLOT (bar chart)        ▾ ]   id="plot-target"
   [ SMALLER ] [ LARGER ]      Scale: 1.0x
8. Matplotlib FigWidth — MAIN PLOT
   [ 10.0 ]
```

Each plot keeps its own values in its own hidden store — `width-zoom-main` /
`width-zoom-pareto`, `fig-width-main` / `fig-width-pareto`. The visible widgets are an
**editor**, never the state:

| action | effect |
|---|---|
| change `plot-target` | *loads* the chosen plot's stored zoom + width into the widgets. Redraws nothing, writes nothing. |
| `SMALLER` / `LARGER` | writes the targeted plot's zoom store, then `applyZoom()` |
| FigWidth `onchange` | writes the targeted plot's width store, then `trigger_plot()` |

Switching the target deliberately does **not** redraw: choosing which plot you are *about to*
resize must not resize anything.

## The three wiring changes that make it hold

- **`applyZoom()` applies both stores, always.** It runs after every redraw, and a fresh `<img>`
  has no inline width — reading only the targeted plot's zoom would silently snap the *other*
  plot back to 1.0× on every replot. It now calls a shared `_zoomArea(selector, val)` once per
  area with that area's own store.
- **`draw_pareto_section()` takes no width argument.** It lost its `calc_width` parameter and
  reads `_fig_width("pareto")` itself. Passing the width in from `trigger_plot` is precisely the
  coupling being removed; a caller that *can* hand it the main plot's width will eventually do so.
- **`_fig_width(which)` is the single reader**, for both plots. A missing, empty or junk store
  falls back to `10.0`, not to `0.0` — a degenerate zero-inch canvas is a worse failure than a
  wrong-but-readable default, and the store is empty on any page loaded before this change.

## Not changed

`draw_pareto` is untouched again — band, hollow/filled marks, staircase front, both legends,
colours read from `last_bar_colors`. `PARETO_H` (6.5 in) stays fixed: height is what the two
legends need, and exposing it as a fourth knob buys nothing the width does not already buy.
The export ZIP still writes both PNGs (`_pareto.png` from U18.1) — and now each is saved at the
width its plot was actually configured with.

## Files

| file | change |
|---|---|
| `Data_Analysis/Visualizer/index.html` | `plot-target` selector + 4 hidden stores in the sidebar; `_fig_width()`; `draw_pareto_section` drops its width arg; JS `switch_plot_target` / `commit_fig_width` / `_zoomArea`, `applyZoom` and `adjustZoom` rewritten per-plot; suite tag → v3.16 |
| `Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` | suite-tag anchor `v3.15` → `v3.16` (1 line) |
| `Data_Analysis/Visualizer_VA_v2/index.html` | regenerated (41 edits) |
| `Data_Analysis/Visualizer_VA_v2/test_page_offline.py` | per-plot width regression checks; section renamed |

## Validation

Ran **in this container** (no science stack needed):

- both pages' `<script type="py">` blocks parse (`ast.parse`) — DAv3 1818 lines, VA v2 2456;
- wiring sweep: `plot-target` ×2, each of the 4 stores ×1, `_fig_width("main")` ×1,
  `_fig_width("pareto")` ×1, **zero** remaining `width-zoom"` / `calc_width)` references;
- AST unresolved-name sweep over every function in both blocks — only the known
  closure/comprehension false positives (`out`, `seen`, `rgba`, `q`, `c`, `sel`, `metrics`),
  no new ones (this is the sweep that caught a deleted `fig, ax = ...` line during U18.1);
- CRLF preserved on the DAv3 page (2159/2159); `PARETO_ENV_LABEL` still `'env'` / `'geo'`;
- full derive chain: `build_from_dav3.py` 41 edits, downstream `build_from_va2.py` 41 edits.

**Run on cluster** (needs pandas + matplotlib):

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py
python Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py
python Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py
```

The section now seeds `fig-width-main = 11.0` and `fig-width-pareto = 14.0` — deliberately
different, since a test where both plots want the same width cannot tell the two stores apart —
and asserts: the main figure is 11.0 in wide and the Pareto 14.0 in; setting
`fig-width-pareto = 9.0` and redrawing moves the Pareto to 9.0 and leaves the main plot at 11.0;
and `_fig_width('nope')` returns 10.0 rather than 0. All U18/U18.1 assertions are unchanged.

Not covered offline: how it *looks*, and that the selector swaps the displayed numbers (that
part is JS, which the harness does not execute). Serve the repo root and click it —

```bash
python3 -m http.server 8000        # then Data_Analysis/Visualizer/index.html
```
