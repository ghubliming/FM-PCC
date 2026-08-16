# U_6 — the Path Audit Map names which seeds a candidate has

**Date:** 2026-08-12
**Epoch:** U_6 (this folder's sequence) · **U17** in the VA page's own U7…U16 numbering
**Scope:** `Data_Analysis/Visualizer_VA_v2/` only

| file | change |
|---|---|
| `Data_Analysis/Visualizer_VA_v2/build_from_dav3.py` | **+6 subs** (34 → 40) — the whole fix lives here |
| `Data_Analysis/Visualizer_VA_v2/index.html` | regenerated (2501 lines) |
| `Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py` | +12 checks (102 → 114), stdlib only |
| `Data_Analysis/Visualizer_VA_v2/test_page_offline.py` | new `[U17 audit map seeds]` section (cluster) |
| `Data_Analysis/DA_VA_v2/README.md` | documents the column |

> **`Data_Analysis/Visualizer/index.html` (DAv3) is NOT touched.** U_3 crossed that line once,
> on explicit instruction, because the bug was in shared plotting code and the derivative
> inherited it. This one is the opposite case: the defect exists *only* on the VA page,
> because it comes from the VA CSV schema. So it is fixed the normal way — as substitutions
> in `build_from_dav3.py`, applied to the generated page. DAv3's audit map is unchanged and
> keeps answering the seed question its own way.

---

## 1. The defect

DA_VA_v2's **Path Audit Map** showed no seed information at all. Not a wrong number — the
column simply was not rendered, so a candidate that ran one seed and a candidate that ran
six looked identical in the audit table.

The inherited DAv3 code gates the column on a column name:

```python
has_missing = 'Missing_Seeds' in df_agg.columns          # render_path_map()
```

`Missing_Seeds` is a **DA_Code_v3** column. This page builds `df_agg` inside
`derive_frames()` out of `va2_aggregated_long.csv`, whose columns are:

```
Candidate, FolderName, FullPath, split, geo, geo_base, tightened, variant,
metric, mean, std, min, max, n, mask, n_seeds, LatestSnapshot
```

`n_seeds` — a *count* — and no `Missing_Seeds`. The DA_VA_v2 reporter does compute missing
seeds (`reporter.py:143,163`), but writes them only into the DAv3-compat tables
(`candidates_multidimensional_{raw,aggregated}.csv`), which this page never fetches. The
gate was therefore permanently false: dead markup, no warning, no clue.

**Why it went unnoticed, and why it stops being harmless.** Every VA batch on disk is
single-seed — four batches checked (`batch_va2_20260806_204620`, `…0808_105342`,
`…0809_103838`, `…0812_114647`), all seed 6 only, no candidate short of anything. The
bridged D3IL baselines (U4.2) are the first ragged case:
`d3il_baseline_ddpm_encdec_vision` carries seed 42 alone while `…__Bf_U3` carries 0–4 + 42,
so a comparison over both roots has a reference set of `{0, 1, 2, 3, 4, 42}` and one
candidate covering a sixth of it.

## 2. The fix

Six substitutions, all against inherited DAv3 code, all in `build_from_dav3.py`.

**`_seed_map()` gains two switches** (three subs — signature, docstring + seed-mode line,
early return):

```python
def _seed_map(use_mode=True, raw_only=False):
```

* **`use_mode=False`** — ignore Custom Seed Compare. The audit map describes the batch **on
  disk**; letting a plot-side filter narrow it would make an audit table misreport what was
  downloaded. The Plot Legend keeps following the filter, as before.
* **`raw_only=True`** — refuse the `Missing_Seeds` fallback. That fallback returns
  `(None, missing)` — it knows only what is *absent* — so in an audit row it renders
  `n/a — no per-seed CSV` on every line, restating the `Warnings` column beside it and
  answering nothing.

**`render_path_map()` gains one column** between **ID** and **Warnings** (three subs):

```python
seed_map = _seed_map(use_mode=False, raw_only=True)
...
if seed_map: html += '<th style="width:160px" ...>Seeds</th>'
...
if seed_map: html += _seed_cell(seed_map, row["Candidate"])
```

The data comes from `va2_units_long.csv` (`df_raw`), which this page always loads and
which is what the Plot Legend already reads — so the audit map now answers *which* seeds,
not merely how many, and does it without a schema change.

`_seed_cell` is the Plot Legend's existing renderer, **reused unchanged**. The ⚠ **NOT
FULL** caution therefore comes along at no cost (it was optional in the request; suppressing
it would have been more code than keeping it). One renderer, one styling rule, no third copy
to drift.

```
ID       Seeds                                Source Path (Audit)
CAND_1   42                                   /logs/…/_DA_VA_BRIDGE_d3il_baseline
         ⚠ NOT FULL — missing 0, 1, 2, 3, 4
CAND_2   0, 1, 2, 3, 4, 42                    /logs/…/_DA_VA_BRIDGE_d3il_baseline_Bf_U3
```

### What each page does now

| | DAv3 (untouched) | DA_VA_v2 |
|---|---|---|
| seed source | `candidates_multidimensional_raw.csv` | `va2_units_long.csv` |
| audit map | `Warnings`: `ALL SEEDS` / `MISSING: […]` | **`Seeds`**: the seed list + ⚠ NOT FULL |
| respects Custom Seed Compare | n/a | no — audits the batch on disk |
| no per-seed CSV | unchanged | column dropped, table exactly as before |

## 3. Verification

```
python3 Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py     114 PASS
```

Stdlib only, runs in the AI container. New this epoch:

* a minimal fake `DataFrame` / `Series` / `GroupBy` (rows of dicts) plus `pd.to_numeric` and
  `pd.isna` — enough to drive `_seed_map` and `render_path_map` with no pandas installed;
* the audit seed map names present **and** missing seeds
  (`{'6': ([6,7],[8]), '65': ([6,7,8],[])}`);
* Custom Seed Compare does **not** narrow the audit map while it still narrows the legend
  map — the two must not collapse into one call;
* `raw_only=True` returns `{}` exactly where the legend's `_seed_map()` still takes the
  fallback;
* rendered — a **VA-shaped `df_agg` with no `Missing_Seeds` gets the Seeds column and grows
  no phantom Warnings column** (the assertion that pins this bug); a DAv3-shaped one keeps
  `ALL SEEDS` / `MISSING:` beside it; with no per-seed frame the table is what it was before;
* **the U17 section runs against DA_VA_v2 only**, and two new drift checks assert the
  intended asymmetry: the VA page has `def _seed_map(use_mode=True, raw_only=False)`, the
  DAv3 page still has `def _seed_map():` and no `raw_only=True` call. If someone hand-edits
  DAv3 into agreement, or the subs stop landing, that check fails.
  `_seed_map` and `render_path_map` were removed from the byte-identity list for the same
  reason; the other 16 shared functions are still asserted identical.

Both pages' python blocks compile. The build is reproducible: 40 edits, 2501 lines.

**Run on cluster** (needs pandas + matplotlib):

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py <batch_va2_dir>
```

New `[U17 audit map seeds]` section — the audit seed map covers every candidate in `df_agg`
(not just the plotted ones), each candidate's seed list is actually present in the emitted
markup, the column does not depend on `Missing_Seeds`, and flipping the seed mode leaves the
audit map identical while changing the legend map. Not yet verified: a browser eyeball.

## 4. Deliberately not done

* **`Missing_Seeds` was not added to `va2_aggregated_long.csv`.** It would light up the
  inherited DAv3 branch with no viewer change at all and revive the `_seed_map()` fallback,
  but it is a pipeline schema change that only takes effect after re-running every batch —
  disproportionate to one viewer column.
* **`n_seeds` is still dropped by `derive_frames()`.** The per-seed frame answers the same
  question better: which, not how many.
* **The dead fallback on this page stays dead.** `_seed_map()`'s `Missing_Seeds` branch can
  never fire here, so a batch folder without `va2_units_long.csv` loses the legend's Seeds
  column outright rather than degrading. `reporter.save_all()` always writes that file; the
  only way to hit it is to hand-assemble a batch folder.
