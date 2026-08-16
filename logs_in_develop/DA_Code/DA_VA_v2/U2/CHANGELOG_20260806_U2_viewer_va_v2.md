# DA_VA_v2 U2 — `Visualizer_VA_v2`, derived from the DAv3 page

**Date:** 2026-08-06 · **Epoch:** U2
**New:** `Data_Analysis/Visualizer_VA_v2/{index.html, build_from_dav3.py, test_page_offline.py}`
**Changed:** `Data_Analysis/DA_VA_v2/{config,utils,reporter,main_da_batch,README}`
**Untouched:** `Data_Analysis/Visualizer/index.html`, `Data_Analysis/Visualizer_Visual_Aligning/index.html`

---

## 0. The requested revert had no target in the HTML

```
Visualizer/index.html                  last commit 1b3c0800  2026-08-03  (DAv3 viz U11)
Visualizer_Visual_Aligning/index.html  last commit 9539edfa  2026-07-03
git diff HEAD -- "*.html"              (empty)
```

Neither page was edited today. What *had* bent toward them was on the DA side,
added in Fix_1 this morning — that is what was backed out (§1).

---

## 1. Backed out: the Fix_1 accommodations for the old viewers

Fix_1 fixed "the run does not appear in the quick list" by making the **pipeline**
bend around two pages that disagree on a folder-name prefix. Wrong direction.

| removed | was for |
|---|---|
| `utils.create_viewer_alias()` — a `va_batch_va2_*` **symlink** beside every run | VA v1's `va_batch_` regex |
| **legacy alias columns** in `per_rollout_detail.csv` (`success`, `mean_dist_m`, `steps`, `phys_err_m`, `context_xy_dist_m`, `avg_time_s`, …) | VA v1's rollout table |
| **`va_candidates_dynamic.csv`** + `Reporter._dynamic_table()` | VA v1's only aggregate input |

Replaced by `utils.check_viewer_visibility()` — a warning, nothing written to disk.
Output folders stay `batch_va2_<timestamp>`, which the new viewer lists and which
also satisfies DAv3's own `batch_*` pattern, so one name serves both with no alias.

**Kept:** `candidates_multidimensional_*.csv`. DAv3's page is current (U11, Aug 3)
and the tool of record for the avoiding side — not old code being propped up.

---

## 2. The first attempt at the viewer was wrong, and why

The first U2 page was written **from scratch**, borrowing DAv3's patterns rather
than its code. Two consequences, both fair to call malfunctioning:

1. **The aggregate controls never appeared.** `#agg-controls` was `display:none`
   and only `setViewMode()` — an onclick handler — ever revealed it. After
   SYNC_SOURCE you got the global filters and nothing else: no metric, no
   variants, no candidates, no export, and a plot area asking you to tick boxes
   that were not on screen. Clicking another view tab and back "fixed" it, which
   is exactly how a page earns the word malfunction.
2. **It was thinner than DAv3 everywhere else.** No U9 plot legend, no U11 folder
   ZIP, no U7 seed-missing warnings, no fail flags in the matrices, a reduced
   LaTeX export. Hence "feels like the old VA viz" — it *was* closer to VA v1's
   scope than to DAv3's.

That page was discarded, not patched.

## 3. What replaced it: a real DAv3 derivative

`Visualizer_VA_v2/index.html` is now **`Visualizer/index.html` plus 20 surgical
edits**, applied by `build_from_dav3.py`. Of 1676 lines, ~1000 are DAv3 verbatim.
Every edit asserts its anchor, so a future DAv3 change that moves the ground fails
loudly instead of silently producing half a page. Re-run the script after DAv3
gains something worth having.

*(DAv3's file is CRLF; the builder normalises to LF — that mismatch is what made
the first anchor attempts fail.)*

### Inherited by construction, not reimplemented

U7 no-data messages and seed-missing warnings · U8 manual bar value labels (the
`bar_label`/`yerr` IndexError fix) · U9 plot legend in x-axis order · U10 paper
result matrices with fail flags · U10.1 full standalone LaTeX export · U11
browser-side folder ZIP download · seed modes · zoom · fig width · scorecard ·
path audit map · single-ZIP export · the numeric-Candidate `astype(str)` fix.

### The v2 grafts

* **Native CSV loader.** Reads `va2_aggregated_long.csv`, `va2_units_long.csv`,
  `per_rollout_detail.csv`, `data_quality.csv`, then `derive_frames()` projects
  them onto the column names the DAv3 core already speaks
  (`halfspace_variant` ← `geo`, `count` ← `n`, `value` ← `mean`). That is why the
  inherited half needs no edits — and why the compat CSVs are no longer what the
  page reads.
* **Mask is a live global switch** (`all` / `unfrozen`), plus a **split** switch.
  Changing either re-derives both frames and bumps `data_version`, which is what
  invalidates DAv3's result-matrix cache. A permanent banner states what is
  hidden: *"mask = ALL — every rollout kept, of which 152 of 2358 rollouts are
  D1-FROZEN…"*.
* **Geometry is the environment axis** — DAv3's "Environment Focus" now lists
  `geo`, relabelled "Geometry Focus".
* **Seeds are read from the batch**, replacing the hardcoded 6/7/8/9/10
  checkboxes; a single-seed VA run no longer shows four dead boxes.
* **Four views** (VA v1's structure): AGGREGATE (all of DAv3), PER-ROLLOUT
  (sortable table, frozen rows tinted, selectable columns, CSV download), COMPARE
  (scatter/bar/box over any two per-rollout columns), QUALITY (the units with
  frozen rollouts, breaker trips, or a partial npz).
* `applyViewMode()` is separate from the tab click **and is called by
  `load_data`**, so the active view's panel is visible the moment data lands —
  the §2.1 bug cannot recur.

---

## 4. Validation

`test_page_offline.py` execs the page's **entire** Python block against stubbed
`document` / `window` / `pyscript` objects, feeds it real CSVs, and calls the real
handlers the way the DOM would. 43 checks, all passing:

* 1318 lines of page Python exec clean;
* filters populate — 54 metrics, 5 geometries, 25 variants, 3 candidates, seeds
  `['6']` from the data;
* `derive_frames` yields the DAv3 schema, applies the mask (6210 of 12420 rows —
  forgetting it would double-count), and carries seeds into `df_raw`;
* aggregate view: plot drawn, scorecard filled, **U9 legend**, **4 result
  matrices** with never-run cells marked, path map, U10 empty-selection message,
  U7 no-data message, per-seed mode;
* export: summary context, LaTeX with `\documentclass` … `\end{document}`, one
  table per metric, candidate source paths;
* **the mask moves real numbers** — `constraint_exec_sat_rate` on the tightened
  geometry 0.9157 (all) → 0.9027 (unfrozen) — and `data_version` bumps;
* per-rollout: table renders, shrinks under the mask, re-sorts, and explains a
  missing selection;
* compare: all three charts draw (120 rollouts), and both failure modes — nothing
  selected, and an all-NaN pairing — produce a message rather than a blank;
* quality: 38 of 115 units flagged;
* `refresh_global` drives every view without raising.

Static wiring lint also clean: all 60 ids referenced exist, every `py-click` names
a Python function, every inline handler names a JS function, every
`window.document.X` has a `create_proxy`, every checkbox class read is created.

**Not verified: an actual browser render.** The data flow and wiring are checked;
pixels are not. → **open it and click through.**

A note on the harness's own findings: two "failures" it reported were the page
behaving correctly on the state-only avoiding candidate (no
`context_init_xy_dist` / `mean_dist_per_rollout` columns exist there), so the test
now exercises both that path and a real visual-aligning one.

---

## 5. Use

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans
python3 -m http.server 8000        # from the repo root
# → http://localhost:8000/Data_Analysis/Visualizer_VA_v2/index.html
```

QUICK_LIST → SYNC_SOURCE → set mask/split → tick candidates and variants.
Result Matrices need candidates only.

Maintenance: `python Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` to
re-derive after a DAv3 update, then `test_page_offline.py` to check it.
