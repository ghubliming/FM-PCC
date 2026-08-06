# DA_VA_v2 U2 — new `Visualizer_VA_v2` page, and the old-viewer accommodations backed out

**Date:** 2026-08-06 · **Epoch:** U2
**New:** `Data_Analysis/Visualizer_VA_v2/index.html`
**Changed:** `Data_Analysis/DA_VA_v2/{config,utils,reporter,main_da_batch,README}.py|md`
**Untouched:** `Data_Analysis/Visualizer/index.html`, `Data_Analysis/Visualizer_Visual_Aligning/index.html`

---

## 0. First, the requested revert — there was nothing in the HTML to revert

Checked before changing anything:

```
Visualizer/index.html                  last commit 1b3c0800  2026-08-03  (DAv3 viz U11)
Visualizer_Visual_Aligning/index.html  last commit 9539edfa  2026-07-03
git diff HEAD -- "*.html"              (empty)
```

Neither page was edited today, by me or otherwise, so "revert the HTML" had no
target. What *had* bent toward those pages was on the **DA side**, added in Fix_1
this morning, and that is what was backed out (§1).

Both old pages remain byte-identical and fully usable; the git history holds them
either way.

---

## 1. Backed out: the Fix_1 accommodations for the old viewers

Fix_1 solved "the run does not appear in the HTML quick list" by making the
*pipeline* bend around two old pages that disagree on a folder-name prefix. That
was the wrong direction — it grew a symlink and duplicated CSV columns to keep a
page alive that is superseded and known-buggy.

| removed | was for | replaced by |
|---|---|---|
| `utils.create_viewer_alias()` — wrote a `va_batch_va2_*` **symlink** beside every run | VA v1's `href="(va_batch_…)"` regex | nothing; `batch_va2_*` satisfies the new viewer and the DAv3 page on its own |
| **legacy alias columns** in `per_rollout_detail.csv` (`success`, `mean_dist_m`, `steps`, `phys_err_m`, `context_xy_dist_m`, `avg_time_s`, `final_xy_dist_m`, `constraint_sat_rate`, `n_violated_steps`) | VA v1's rollout table looked those names up | the new viewer reads the canonical names |
| **`va_candidates_dynamic.csv`** output + `Reporter._dynamic_table()` | VA v1's only aggregate input | `va2_aggregated_long.csv`, which is strictly richer (mask, split, geo, seeds) |

Replacing the symlink: `utils.check_viewer_visibility()` — a *warning* when
`--output-path` is named something the viewer's QUICK_LIST will not match, with
the CUSTOM_PATH box as the answer. It changes nothing on disk.

**Kept on purpose:** `candidates_multidimensional_raw.csv` / `_aggregated.csv`.
Those feed `Visualizer/index.html`, which is current, actively developed (U11 on
Aug 3) and the tool of record for the avoiding side — not old code being propped
up. Being able to open a VA batch there was an explicit requirement of the
original task.

Output-folder naming stays `batch_va2_<timestamp>`: it matches the new viewer's
`batch_va2_*` **and** the DAv3 page's `batch_*` with a single name.

---

## 2. New: `Data_Analysis/Visualizer_VA_v2/index.html`

Fresh page, no code taken from either old file beyond the parts of their design
that had earned their place. Reads the **native** DA_VA_v2 CSVs directly:

| file | used for |
|---|---|
| `va2_aggregated_long.csv` | everything aggregate (required) |
| `va2_units_long.csv` | per-seed compare (optional) |
| `per_rollout_detail.csv` | rollout table + compare view (optional) |
| `data_quality.csv` | quality view + mask banner (optional) |

A missing optional file degrades that one view with a message; it never breaks
the page.

### Taken from DAv3 (`Visualizer/index.html`)

* batch picker off the directory listing, `results_manifest.json` as fallback,
  plus a CUSTOM_PATH box;
* checkbox filters with **nothing pre-checked** — a big batch locks the page up
  if it draws everything on load, and an auto-picked first bar gets mistaken for
  a meaningful default (their U10 lesson);
* grouped bar chart with std whiskers and **manually annotated** value labels —
  `ax.bar_label` couples to the ErrorbarContainer that `yerr` adds and crashes on
  an absent (NaN) combo (their U8 fix);
* paper-style **result matrices** (rows = checked candidates, cols = every
  variant, `mean ± sem`) with **LaTeX export**, rendered from the candidate
  selection alone so they populate before any variant is ticked;
* path audit map, scorecard, and a single-ZIP export (PNG + `.tex` + audit log).

### Taken from VA v1 (`Visualizer_Visual_Aligning/index.html`)

* the view-mode toggle as the page's primary structure;
* the **per-rollout table**, sortable, now with frozen rollouts tinted and
  flagged ❄, selectable columns, and its own CSV download;
* the rollout-level **compare** view — scatter / bar / box, free choice of x and
  y from any per-rollout column, series grouped by variant or candidate.

### New in v2

* **The mask is a first-class global switch.** `all` vs `unfrozen`, applied to
  every view, with a permanent banner stating what is currently hidden:
  *"mask = UNFROZEN — 152 of 2358 rollouts excluded (D1 box-obstacle conflict…)"*.
  Every plot title and every exported file records the mask it used.
* **Geometry and split are real dropdowns**, not a `geo/variant` name prefix the
  page has to re-split, because the CSVs now carry them as columns.
* **X-axis is selectable** — Candidate, variant, or geometry — instead of the two
  fixed modes.
* **QUALITY view**: the `data_quality.csv` rows worth distrusting (frozen
  rollouts, projector circuit-breaker trips, `npz_complete=0`), with a plain
  statement of why each invalidates a number.
* Per-seed compare reads `va2_units_long.csv` and degrades to pooled with a
  message when that file is absent.

### Deliberately not carried over

DAv3's U11 **"download this run folder as .zip"** crawler (~120 lines of async
directory walking). It is genuinely useful but it is the one feature that cannot
be exercised outside a browser, and an untested async crawler in a brand-new page
is a bad trade. It stays available in the DAv3 page. Everything else in the
export path (PNG + LaTeX + audit log in one ZIP) is here.

---

## 3. Validation

No browser is available here, so the page was built with its **data layer as pure
pandas functions with no DOM access** (`normalise`, `apply_global`, `agg_pivot`,
`matrix_stats`, `rollout_view`, `compare_frame`, `mask_summary`, `quality_flags`,
`build_latex`), and those are lifted straight out of the HTML by a harness and
run against real CSVs — no hand-maintained copy to drift.

`scratchpad/test_viewer_core.py` — 38 checks, all passing against a real Gen14 U7
+ state-only batch:

* mask splits the long frame exactly (6210 + 6210 = 12420) and forgetting it
  double-counts;
* all three x-axes pivot, std aligns to mean, unknown variant returns empty
  instead of raising;
* the real bar-plot call with `yerr` plus manual labels renders (12 bars);
* 191 matrix cells, sem present, LaTeX emits `tabular`, escapes `_`, marks
  missing cells `--`, one row per candidate;
* **the mask changes the answer**: `constraint_exec_sat_rate` on the tightened
  geometry 0.9157 (all) → 0.9027 (unfrozen);
* rollout sorting ascending/descending per metric, empty selection safe;
* compare view builds, and bad column / empty selection / missing CSV each return
  a message rather than raising;
* all three chart paths draw;
* quality flags 38 of 115 units; banner text correct with and without each CSV.

`scratchpad/lint_viewer.py` — static wiring check, clean: every `getElementById`
target exists in the markup, every `py-click` names a Python function, every
inline `onclick`/`onchange` names a JS function, every `window.document.X` has a
`create_proxy`, every checkbox class read is one that gets created.

### Two real bugs the harness caught

1. **The global mask did nothing in the rollout and compare views.**
   `per_rollout_detail.csv` has no `mask` column — it is one row per rollout with
   a `frozen` flag — so `apply_global` silently passed every row through.
   Switching to UNFROZEN would have changed the aggregate numbers while the
   rollout table underneath kept showing the frozen rollouts, which is exactly
   the kind of quiet inconsistency this page exists to prevent. `apply_global`
   now handles both shapes: filter the `mask` column when present, drop
   `frozen == 1` rows otherwise. Verified: 2358 → 2206 rows, 30 → 26 rollouts in
   a tightened variant's own table, 2280 → 2128 compare points.
2. **Rollout table showed duplicate-looking rows.** With split/geo on ALL the
   same `rollout_idx` appears once per (seed, geo). The table now prepends
   whichever of seed/split/geo actually varies.

Also pre-empted from reading the old pages: `ax.boxplot(labels=)` vs
`tick_labels=` swapped over matplotlib 3.9 and pyodide pins an older build, so
ticks are set afterwards instead; `plt.get_cmap` is wrapped with a
`matplotlib.colormaps` fallback.

**Not verified:** an actual browser load. The data contract and the wiring are
checked statically; rendering is not. → **open the page and click through it.**

---

## 4. How to use

```bash
# cluster: produce a batch
sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans

# locally: serve the repo root and open the viewer
python3 -m http.server 8000
# → http://localhost:8000/Data_Analysis/Visualizer_VA_v2/index.html
```

Pick the batch in QUICK_LIST → SYNC_SOURCE → set the mask → tick candidates and
variants. The Result Matrices need candidates only.
