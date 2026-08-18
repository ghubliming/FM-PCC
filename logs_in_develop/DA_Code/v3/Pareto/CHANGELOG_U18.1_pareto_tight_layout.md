# DA_Code v3 — U18.1: the Pareto sub-plot silently disabled `tight_layout`

**Scope:** the same two HTML viewers as U18 —
`Data_Analysis/Visualizer/index.html` (DAv3, avoiding) and
`Data_Analysis/Visualizer_VA_v2/index.html` (DA_VA_v2, visual aligning), the latter
**regenerated** from the former. No DA pipeline change, no CSV schema change, no cluster
re-run, no new dependency. Toolbar tag bumped `SCIENTIFIC_SUITE_v3.13` → `v3.14`.
**Fixes:** U18.

---

## The symptom

```
<exec>:904: UserWarning: This figure includes Axes that are not compatible with
tight_layout, so results might be incorrect.
```

printed by PyScript on **every** draw with the Pareto panel on, big enough to be the only
thing on screen.

## The cause

U18 built the two stacked axes like this:

```python
fig, (ax, ax_pareto) = plt.subplots(
    2, 1, figsize=(calc_width, 6 + PARETO_H),
    gridspec_kw={'height_ratios': [6, PARETO_H], 'hspace': 0.45})
```

`height_ratios` is harmless. **`hspace` is not.** It is one of matplotlib's
`GridSpec._AllowedKeys` (`left, bottom, right, top, wspace, hspace`), so setting it makes
`gs.locally_modified_subplot_params()` truthy. `TightLayoutEngine` then walks the figure via
`get_subplotspec_list()`, which contains:

```python
elif gs.locally_modified_subplot_params():
    subplotspec = None          # -> "this axes is not compatible"
```

Every axes on that GridSpec comes back `None`, the engine emits the warning, and
`get_tight_layout_figure()` returns `{}` — **it adjusts nothing at all**.

## Why that mattered beyond the warning

`plt.tight_layout()` became a **no-op for the whole figure**, so the figure was drawn with
matplotlib's default subplot params. Everything U18 hangs *outside* the right edge of an
axes then falls off the canvas:

- the main plot's `Variant` legend (`bbox_to_anchor=(1.05, 1)`),
- the Pareto panel's *shape = candidate* legend (`bbox_to_anchor=(1.02, 1.0)`),
- the Pareto panel's *not a perfect goal+constraint run* legend (`(1.02, 0.0)`) — the honesty
  column, i.e. the one entry that says a front point bought its position with 7 % failures.

So the warning was not cosmetic noise: it was the page telling us that the layout pass which
makes the panel readable had been skipped. Notably it fired **only** with the panel on — with
`pareto-on` unticked the plain `plt.subplots(figsize=...)` path has no GridSpec kwargs and
`tight_layout` worked, which is why the bar chart alone always looked right.

## The fix

Keep the GridSpec clean and set the gap *after* the layout pass, as a floor:

```python
PARETO_HSPACE = 0.45                     # new constant, next to PARETO_H

fig, (ax, ax_pareto) = plt.subplots(
    2, 1, figsize=(calc_width, 6 + PARETO_H),
    gridspec_kw={'height_ratios': [6, PARETO_H]})
...
plt.tight_layout()
if ax_pareto is not None and fig.subplotpars.hspace < PARETO_HSPACE:
    fig.subplots_adjust(hspace=PARETO_HSPACE)
```

`tight_layout` runs first and computes the margins from the artists it can measure (both
legends included — they are axes children and count toward the tight bbox). The
`subplots_adjust` afterwards only ever **raises** the vertical gap, never lowers it: the main
plot's rotated x tick labels plus its red note line (`… bars have NO DATA`, `(G)/(C) …`) are
routinely taller than what `tight_layout` reserves between the two rows, and the Pareto title
is two lines. Lowering would re-open the collision U18's `hspace=0.45` was there to prevent.

Both `plt.subplots` calls and the `tight_layout` call are commented with *why* `hspace` may
not go back into `gridspec_kw` — this is a trap that reads like a style preference.

## Regression tests

New checks in `Visualizer_VA_v2/test_page_offline.py`, `[U18 pareto sub-plot]`, wrapping the
panel-on draw in `warnings.catch_warnings(record=True)`:

| check | asserts |
|---|---|
| `drawing the pareto raises no tight_layout warning` | the symptom itself is gone |
| `the pareto gridspec sets no subplot params …` | `gs.locally_modified_subplot_params()` is empty — the cause |
| `tight_layout actually ran — the right-hand legends are inside the canvas` | main axes `x1 < 0.98`, i.e. the pass did something — the effect |
| `the two stacked plots keep their gap` | `fig.subplotpars.hspace >= PARETO_HSPACE` — the fix did not cost the separation |

Three of the four would have failed before this change; the fourth (`hspace`) would have
passed for the wrong reason, which is why all three of warning / cause / effect are pinned
rather than just the warning.

## Files

| file | change |
|---|---|
| `Data_Analysis/Visualizer/index.html` | `PARETO_HSPACE` constant; `hspace` out of `gridspec_kw`; post-`tight_layout` gap floor; suite tag → v3.14 |
| `Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` | suite-tag anchor follows the DAv3 bump (`v3.13` → `v3.14`) |
| `Data_Analysis/Visualizer_VA_v2/index.html` | regenerated (41 edits) — inherits the fix |
| `Data_Analysis/Visualizer_VA_v2/test_page_offline.py` | 4 new checks in `[U18 pareto sub-plot]` |

**Not touched:** `Visualizer_UAV_v1/index.html`, same as U18. `build_from_va2.py` was re-run
against the new VA v2 page to prove the chain still applies (41 edits, clean) and its output
reverted. Re-run it when the UAV page should pick up U18 + U18.1.

## Validation

Ran **in this container**:

- both pages' `<script type="py">` blocks parse (`ast.parse`) — DAv3 1773 lines, VA v2 2411;
- exactly one `gridspec_kw` per page and it carries `height_ratios` only;
- `PARETO_HSPACE` present on both, `PARETO_ENV_LABEL` still `'env'` / `'geo'` respectively;
- `build_from_dav3.py` (41 edits) and downstream `build_from_va2.py` (41 edits) both apply
  every anchor;
- `test_page_offline.py` parses.

**Run on cluster** (needs pandas + matplotlib):

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py            # newest batch_va2_*
```

Then serve and eyeball — this bug was invisible to every offline check U18 shipped with,
because none of them looked at where the axes actually ended up:

```bash
python3 -m http.server 8000        # then Data_Analysis/Visualizer/index.html
```

> If the warning persists on the page you open, check you are not opening a **copy** — the
> reported line (`<exec>:904`) does not match `plt.tight_layout()` in either repo page
> (DAv3 py-block line 872 after this fix, VA v2 line 1062), so the browser may be holding a
> cached or laptop-local export. Hard-reload / re-copy the two files above.
