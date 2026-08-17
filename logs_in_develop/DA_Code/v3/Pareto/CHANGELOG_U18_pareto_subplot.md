# DA_Code v3 — U18: Pareto sub-plot (AVG_TIME × N_STEPS, gated on goal+constraints)

**Scope:** the two HTML viewers only —
`Data_Analysis/Visualizer/index.html` (DAv3, avoiding) and
`Data_Analysis/Visualizer_VA_v2/index.html` (DA_VA_v2, visual aligning), the latter
**regenerated** from the former via `Visualizer_VA_v2/build_from_dav3.py`.
No DA pipeline change, no CSV schema change, no cluster re-run, no new dependency
(pure matplotlib, already in Pyodide).
**Depends on:** U15 (per-variant colours), U14 (goal/constraint flags), U10 (Result Matrices).
Toolbar tag bumped `SCIENTIFIC_SUITE_v3.12` → `v3.13`.

**Not touched:** `Visualizer_UAV_v1/index.html`. It derives from the VA v2 page via
`Visualizer_UAV_v1/build_from_va2.py`, and that script was re-run to verify every anchor
still matches (it does — 41 edits, clean), then its output reverted. Re-run it when the UAV
page should pick this up; nothing else is needed.

---

## The problem

The main chart draws **one metric at a time**. The trade-off that actually decides every
comparison in this project — *fewer planning steps* vs *less compute per step* — therefore
never appears on screen at once, and the thing that makes such a comparison legitimate at
all (that the compared methods reached the goal and respected the constraints equally well)
appeared nowhere. A faster planner that fails 30 % of its rollouts is not on any frontier,
but on a bare `avg_time` bar chart it is the winner.

## What was added

A second axes **inside the same figure**, below the main plot, drawn from the **same
selection widgets** as the main plot — candidates, variants, environment focus and seed
mode all come from the existing controls, so the panel can never describe a set the user is
not looking at.

```
x = avg_time   "faster  ←   AVG_TIME (computation per planning step)"
y = n_steps    "fewer   ↓   N_STEPS (episode length)"      → lower-left is better
```

One point per **(candidate, variant)** facet, i.e. exactly the facets the Result Matrices
print — the offline test asserts point-by-point equality against `_cell_stats`, so the panel
and the tables cannot drift.

### The two visual keys

| channel | means | source |
|---|---|---|
| **colour** | the **variant** (projection arm: `dpcc-c-tightened`, `hardflow_new-r`, …) | `last_bar_colors`, i.e. the colours the **main plot above actually drew with** |
| **shape** | the **candidate** (config type: `H8 K20 AlphaFlowODE`, `H8 K10 GaussianDiffusion`, …) | `_candidate_markers`, 14 shapes assigned over the *selected* candidates |

Colour is **read from the bar chart, not recomputed.** `_variant_colors` is deterministic,
but the bar chart has a collision fallback that abandons it for a positional palette; a
Pareto that re-derived its own colours would then key the same variant to two different
colours *within one figure*. `trigger_plot` now records what it drew with and the panel
consumes that.

Shapes are assigned over the **selection**, not the batch — the opposite of the colour rule,
deliberately. A batch carries ~150 candidates and there are 14 tellable-apart shapes, so a
batch-stable shape map would collide inside almost every selection. Distinct-within-this-plot
beats stable-across-plots here; when even that is impossible (>14 candidates ticked) the
legend title says so and every point is labelled `C<id>` instead of two configs silently
sharing a shape.

`_cand_tag()` turns a candidate into its config: `H<horizon>`, `K<steps>` and the model
class off the folder name (`H8_K20_Meuler_T0.5_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE`
→ `H8 K20 AlphaFlowODE`). Verified against every candidate of
`batch_avoiding_combined_20260815_135634`: 154/154 produce a tag, none empty. Anything the
pattern does not recognise falls back to the truncated folder name — a *wrong* tag in a
legend is worse than a long one.

### "Only the most S+C in compare"

The comparison is gated on `n_success_and_constraints` (falling back to `n_success` only for
a batch that never recorded the pair — the **strict** score, the same one U14's `(G, C)`
flag uses):

```
best   = max S+C over the plotted points
band   = sidebar "6.5 Pareto Sub-Plot → S+C band"   (default 0.05)
in     = S+C ≥ best − band          → filled marker, eligible for the front
out    = S+C <  best − band, or no S+C at all → hollow marker, drawn but NEVER on the front
```

so "similar goal+constraints" is an explicit, visible, editable number rather than an
assumption. Set the band to `0` for the strictest reading: only the top score is comparable.
The panel's subtitle states the rule it applied every time —
`compared: n_success_and_constraints ≥ 0.950 (best 1.000, band 0.05) — 6 of 14 points;
8 hollow = not comparable`.

The **front** itself is the non-dominated subset of the eligible points, drawn as a
`steps-post` staircase with a ring around each vertex (a ring, not just a line, because with
a single comparable point the staircase has no length and the front would be invisible).

### "If not 100 % S+C, mark it in the legend"

Two legends, right of the panel:

1. **shape = candidate · colour = variant** — one entry per plotted candidate with its
   config tag, plus the front line itself (`Pareto front — 3 of 6 comparable points`).
2. **not a perfect goal+constraint run** — one entry per point with `S+C < 1.000`, named and
   scored: `CAND_31 · dpcc-c-tightened — S+C 0.933`, and `(out)` when the band excluded it.
   Hollow swatch = hollow point. When nothing is imperfect the legend says exactly that
   (`none — every plotted point is at n_success_and_constraints = 1.000`), so silence is
   never ambiguous.

Both cap at 14 entries and then collapse into `+N more`.

## Controls

New sidebar group **6.5 Pareto Sub-Plot**, placed between *6. Candidates* and
`REFRESH_RE_DRAW`:

- checkbox `pareto-on` (default **on**) — `AVG_TIME × N_STEPS below the main plot`
- number `pareto-band` (default **0.05**) — the S+C similarity band described above

Both call the existing `trigger_plot()`, so no new JS.

## Design decisions worth keeping

- **Sub-plot of the same figure, not a second figure.** `EXPORT ZIP` saves `current_fig`, so
  the panel is in the PNG for free, and it is not possible to screenshot the bar chart
  without the trade-off it hides. Stacked vertically, not side-by-side: the main plot's
  variant legend lives outside its right edge and a neighbour would sit on it.
- **Failure is contained.** `draw_pareto` is called inside a `try/except` after the main axes
  are complete; a raise prints the reason *into the panel's own axes* and leaves the main
  plot untouched. Same for "nothing to draw": `NO PARETO — this selection has no
  avg_time / n_steps pair at env = …`, never a blank box.
- **Log x when it earns it.** `avg_time` spans ~0.004 s to ~0.5 s across arms; when
  max/min > 20 the x axis goes log and the label says `[log scale]`.
- **Environment axis naming.** `PARETO_ENV_LABEL` is `'env'` on the DAv3 page and rewritten
  to `'geo'` by the VA build — the VA page renames that axis and a panel captioned with the
  other page's word for it is simply wrong. In *By Environment* mode the title adds
  "(the main plot spans every env; this panel does not)", because the panel is always at one
  environment.

## Files

| file | change |
|---|---|
| `Data_Analysis/Visualizer/index.html` | new `6.5` control group; U18 block (`_pareto_enabled/_pareto_band/_cand_tag/_candidate_markers/_pareto_points/_pareto_front/_legend_left/_pareto_message/draw_pareto`); `trigger_plot` makes 1-or-2 axes, records `last_bar_colors`, calls `draw_pareto`; suite tag → v3.13 |
| `Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` | new edit 6b: `PARETO_ENV_LABEL = 'env'` → `'geo'`; suite-tag anchor follows the DAv3 bump |
| `Data_Analysis/Visualizer_VA_v2/index.html` | regenerated (41 edits) — inherits the whole panel |
| `Data_Analysis/Visualizer_VA_v2/test_page_offline.py` | new `[U18 pareto sub-plot]` section |

## Validation

Runs **in this container** (no science stack needed):

- both pages' `<script type="py">` blocks parse (`ast.parse`) and export every U18 symbol;
- `_pareto_front` / `_candidate_markers` unit-checked on synthetic points — front
  `[(1,10),(2,8),(4,5)]` out of `[(1,10),(2,8),(3,8),(4,5),(5,9),(4,5)]`, single point and
  empty input both handled, 3 candidates → 3 distinct shapes, 20 → repeat flag raised;
- `_cand_tag` checked against all 154 candidates of a real avoiding batch;
- the full derive chain: `build_from_dav3.py` (41 edits) and downstream `build_from_va2.py`
  (41 edits) both apply every anchor.

**Run on cluster** (needs pandas + matplotlib):

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py            # newest batch_va2_*
python Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py
python Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py
```

The new `[U18 pareto sub-plot]` section asserts: off → 1 axes / on → 2 axes in the same
figure; every point equals the Result Matrices' number for that facet; one point per
plottable facet (no silent drops); every eligible point is within the band of the best and
no scoreless point is ever compared; the front is strictly improving on both axes, contains
no dominated member, and never contains an excluded point; the panel's colours are the main
plot's colours; one shape per candidate (or labels drawn); every sub-1.000 point is named in
the legend; the axes state which direction is better; and a selection with nothing to draw
does not break the page.

Not covered offline: how it *looks*. Serve the repo root and open the page —

```bash
python3 -m http.server 8000        # then Data_Analysis/Visualizer/index.html
```
