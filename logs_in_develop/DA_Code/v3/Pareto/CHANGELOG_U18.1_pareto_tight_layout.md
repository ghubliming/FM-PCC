# DA_Code v3 — U18.1: the Pareto is its OWN plot, in its OWN section

**Scope:** the same two HTML viewers as U18 —
`Data_Analysis/Visualizer/index.html` (DAv3, avoiding) and
`Data_Analysis/Visualizer_VA_v2/index.html` (DA_VA_v2, visual aligning), the latter
**regenerated** from the former. No DA pipeline change, no CSV schema change, no cluster
re-run, no new dependency. Toolbar tag `SCIENTIFIC_SUITE_v3.13` → `v3.15`.
**Fixes / supersedes:** U18's "sub-plot of the same figure" decision.

---

## Two problems, one answer

### 1. The warning

```
<exec>:904: UserWarning: This figure includes Axes that are not compatible with
tight_layout, so results might be incorrect.
```

on **every** draw with the panel on, big enough to be the only thing on screen.

U18 built the stacked axes with
`gridspec_kw={'height_ratios': [6, PARETO_H], 'hspace': 0.45}`. `height_ratios` is harmless.
**`hspace` is not** — it is one of matplotlib's `GridSpec._AllowedKeys`
(`left, bottom, right, top, wspace, hspace`), so it makes `gs.locally_modified_subplot_params()`
truthy, and `TightLayoutEngine` → `get_subplotspec_list()` then does:

```python
elif gs.locally_modified_subplot_params():
    subplotspec = None          # -> "this axes is not compatible"
```

Both axes come back `None`, the engine warns, and `get_tight_layout_figure()` returns `{}` —
**it adjusts nothing.** So `tight_layout` was a silent no-op for the whole figure and
everything hanging *outside* an axes' right edge fell off the canvas: the main plot's
`Variant` legend (`bbox_to_anchor=(1.05, 1)`), the Pareto's *shape = candidate* legend and its
*not a perfect goal+constraint run* legend — the honesty column, the one entry that says a
front point bought its position with 7 % failures. It fired only with the panel on; the plain
`plt.subplots(figsize=...)` path has no GridSpec kwargs, which is why the bar chart alone
always looked right.

### 2. One figure was the wrong container anyway

Even with `tight_layout` working, a shared figure computes **one** left/right margin for the
whole GridSpec — the widest legend on *either* row shrinks *both*. The bar chart would have
lost width to a legend that belongs to the panel. And a single `<img>` cannot be zoomed,
scrolled, screenshotted or exported per plot.

## What changed

The Pareto is now **a second matplotlib figure rendered into a second page section**.

```
<div id="plot-area">                        the bar chart, alone, its own figure
<div id="pareto-section"> #pareto-area      PARETO PLOT — its own figure, own <img>
<div id="selection-map-section">            (unchanged, still below)
```

- `trigger_plot` always builds `fig, ax = plt.subplots(figsize=(calc_width, 6))` — no
  branch, no GridSpec kwargs, `fig.tight_layout()` (not `plt.`, "current figure" is ambiguous
  now), then displays into `plot-area` **and only then** calls `draw_pareto_section(...)`.
  A failure in the panel can no longer cost the user the plot they asked for; the panel still
  reports its own problem in its own axes via `_pareto_message`.
- **New `draw_pareto_section(env, checked_vars, checked_cands, env_is_x_axis, calc_width)`** —
  makes `pfig, pax = plt.subplots(figsize=(calc_width, PARETO_H))`, calls the *unchanged*
  `draw_pareto` into it, `pfig.tight_layout()`, shows the section, displays into
  `pareto-area`. Same width as the main plot on purpose: the two are read against each other.
- **New `_hide_pareto()`** — clears `pareto_fig`, empties `#pareto-area`, hides the section.
  Called from all three no-plot early returns in `trigger_plot` (empty selection, and both
  `subset.empty` paths), so a stale trade-off panel can never sit under a `NO DATA` message
  describing a different selection.
- **New global `pareto_fig`**, deliberately separate from `current_fig`.
- `PARETO_H` `5.5` → `6.5` — it is a whole figure now, not a row. `PARETO_HSPACE` (added
  earlier today as a `tight_layout`-compatible way to keep the row gap) is **deleted**: with
  no shared GridSpec there is no gap to manage.
- Sidebar group renamed **6.5 Pareto Sub-Plot** → **6.5 Pareto Plot**; the checkbox now reads
  `AVG_TIME × N_STEPS as its OWN plot, in its own section`.

### What did *not* change

`draw_pareto` itself — every claim U18 makes (the S+C band, the hollow/filled eligibility
mark, the non-dominated staircase, the two legends, colour = variant read from
`last_bar_colors`, shape = candidate) is byte-identical. The panel still reads the **same
widgets** as the main plot, so it still cannot describe a selection the user is not looking at.

## Knock-on changes the split forced

| what | why |
|---|---|
| `download_plot`: new `{base_name}_pareto.png` in the EXPORT ZIP (`written` gains `PARETO_PNG`) | U18 argued the panel must be in the export — while it was a sub-plot that came free from `current_fig.savefig`. A second figure needs a second `savefig`, and a separate file is the point of separating them. |
| `applyZoom()` (JS): selector `#plot-area img` → `#plot-area img, #pareto-area img`, and it now loops instead of taking `imgs[0]` | otherwise the two plots stop being comparable the moment the user touches the zoom. |
| new `#pareto-area` CSS, mirroring `#plot-area` (white, bordered, `overflow-x: auto`, `min-height: 420px`) | the panel gets its own horizontal scroll instead of squeezing the page. |

## Regression tests

`Visualizer_VA_v2/test_page_offline.py`, `[U18 pareto sub-plot]` — the panel-on draw is now
wrapped in `warnings.catch_warnings(record=True)`:

| check | asserts |
|---|---|
| `drawing the pareto raises no tight_layout warning` | problem 1 is gone |
| `the main plot stays a ONE-axes figure — the pareto is not in it` | the split happened |
| `pareto on -> its OWN figure in its OWN section` | `pareto_fig is not current_fig`, 1 axes, section `display: block` |
| `the pareto figure is displayed into #pareto-area` | it reaches the page, not just memory |
| `no gridspec on either figure carries locally-modified subplot params` | the *cause* of problem 1, on both figures |
| `tight_layout ran on the pareto — its right-hand legends are inside the canvas` | the *effect*: axes `x1 < 0.98` |
| `the main plot no longer pays for the pareto legends` | main axes `x1 >= pareto axes x1` — problem 2 |
| `pareto off -> no pareto figure and the section is hidden` | no stale panel |

Existing U18 assertions were repointed from `current_fig.axes[1]` to `pareto_fig.axes[0]`;
everything they check (point-by-point equality with the Result Matrices, band eligibility,
front non-dominance, every sub-1.000 point named) is unchanged.

## Files

| file | change |
|---|---|
| `Data_Analysis/Visualizer/index.html` | `#pareto-area` CSS; `#pareto-section` markup; sidebar 6.5 relabelled; `pareto_fig` global; `PARETO_H` 5.5→6.5, `PARETO_HSPACE` removed; `_hide_pareto` + `draw_pareto_section`; `trigger_plot` single-figure + 3 `_hide_pareto` guards; `download_plot` second PNG; `applyZoom` both canvases; suite tag → v3.15 |
| `Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` | suite-tag anchor follows the DAv3 bump (`v3.13` → `v3.15`) |
| `Data_Analysis/Visualizer_VA_v2/index.html` | regenerated (41 edits) — inherits everything |
| `Data_Analysis/Visualizer_VA_v2/test_page_offline.py` | `[U18 pareto sub-plot]` section reworked (8 structural checks) |

**Not touched:** `Visualizer_UAV_v1/index.html`, same as U18. `build_from_va2.py` was re-run
against the new VA v2 page to prove the chain still applies (41 edits, clean) and its output
reverted. Re-run it when the UAV page should pick up U18 + U18.1.

## Validation

Ran **in this container**:

- both pages' `<script type="py">` blocks parse (`ast.parse`) — DAv3 1799 py lines, VA v2 2436;
- an AST sweep for unresolved names across every function in both blocks (this caught a real
  slip mid-edit: the `fig, ax = plt.subplots(...)` line had been deleted along with the old
  branch, which `ast.parse` alone accepts happily);
- `trigger_plot` wiring asserted structurally: single `plt.subplots(figsize=(calc_width, 6))`,
  `fig.tight_layout()`, `draw_pareto_section(...)`, `current_fig = fig`;
- **zero** `gridspec_kw` / `PARETO_HSPACE` / `ax_pareto` / `subplots_adjust` left on either page;
- `PARETO_ENV_LABEL` still `'env'` (DAv3) / `'geo'` (VA v2);
- `build_from_dav3.py` (41 edits) and downstream `build_from_va2.py` (41 edits) apply every anchor;
- `test_page_offline.py` parses.

**Run on cluster** (needs pandas + matplotlib):

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py            # newest batch_va2_*
python Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py
python Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py
```

Then serve and eyeball — the original bug was invisible to every offline check U18 shipped
with, because none of them looked at where the axes actually ended up:

```bash
python3 -m http.server 8000        # then Data_Analysis/Visualizer/index.html
```

> If the warning persists on the page you open, check you are not opening a **copy**: the
> reported line (`<exec>:904`) matches `plt.tight_layout()` in neither repo page, so the
> browser may be holding a cached or laptop-local export. Hard-reload / re-copy both files.
