# CHANGELOG — DA_UAV_v1: relocate the log, sync the viewer, pin the fourth copy

**Date:** 2026-08-19 · **Type:** housekeeping + drift sync + test coverage
**Status:** all checks pass in the AI-coding container. **Still nothing run on real data.**

**Moved:** `logs_in_develop/Gen15/U4/CHANGELOG_U4_uav_da_and_viz.md`
→ `logs_in_develop/DA_Code/DA_UAV_v1/CHANGELOG_20260815_DA_UAV_v1.md`
**Added:** `logs_in_develop/Gen15/U4/MOVED.md` (pointer stub), this file
**Regenerated:** `Data_Analysis/Visualizer_UAV_v1/index.html`
**Modified:** `Data_Analysis/DA_VA_v2/test_snapshot_scan.py`
**Untouched:** every `Data_Analysis/DA_UAV_v1/*.py`, `DA_VA_v2/*.py`, `DA_Code_v3/*`,
`Visualizer_VA_v2/`, `mix_uav*/`. No retraining, no eval re-run.

---

## 1. Why the changelog moved

It was filed at `Gen15/U4/`, named for the Gen15 update number. That follows the
`logs_in_develop/<gen>/<epoch>/` convention, but `DA_UAV_v1` is a **tool** that serves
every UAV generation, not a Gen15 code change — so nothing in its path said "DA" and it
was findable only by someone who already knew it was U4. `logs_in_develop/DA_Code/` is
indexed by tool (`DA_VA_v2/`, `v3/`, `v2/`, `DA_Visual_Aligning/`, `METRIC_SMOOTH/`),
each holding a top-level `CHANGELOG_<date>_<tool>.md` plus epoch subfolders. The UAV tool
now sits there in the same shape, and `Gen15/U4/MOVED.md` keeps old links alive.

## 2. The drift was viewer-only — the Python needed nothing

Measured, not assumed. Between `8a543b08` (DA_UAV_v1 created) and HEAD:

```
Data_Analysis/Visualizer/index.html                | 484 ++++
Data_Analysis/Visualizer_VA_v2/build_from_dav3.py  |   6 +-
Data_Analysis/Visualizer_VA_v2/index.html          | 482 ++++
Data_Analysis/Visualizer_VA_v2/test_page_offline.py| 194 ++
```

`DA_VA_v2/*.py` — **zero changes.** The whole U18 Pareto line (`b5ecb6ad` Pareto
sub-plot, `4bb18eb2` tight_layout fix, `e5539419` Pareto as its own figure, `c76d944f`
per-plot zoom + FigWidth stores) landed entirely in the HTML layer, so `DA_UAV_v1`'s
aggregator / discovery / data_loader / reporter had nothing to port.

### 2.1 DA_UAV_v1 is not behind DAv3 — it is ahead

Worth recording, because the folder listing suggests otherwise. `DA_Code_v3` ships five
files the UAV tool lacks: `batch_aggregator.py`, `batch_data_loader.py`,
`batch_reporter.py`, `batch_visualizer.py`, `multi_candidate_discovery.py`. That is the
**older architecture**, not missing features — `DA_VA_v2` lacks all five too, having
folded the batch path into `main_da_batch.py` and multi-candidate handling into a unified
`discovery.py`. Line counts confirm the direction:

| module | DA_Code_v3 | DA_VA_v2 | DA_UAV_v1 |
|---|---|---|---|
| aggregator.py | 268 | 326 | **442** |
| config.py | 144 | 268 | **463** |
| discovery.py | *(absent)* | 577 | **841** |
| reporter.py | 303 | 418 | **602** |

`candidates` appears 33× in the UAV discovery against 27× in VA_v2's, and `n_candidates`
is carried through the aggregator. Nothing to back-port.

## 3. The viewer resync was one command, and it held

`Visualizer_UAV_v1/index.html` is **generated**, not hand-maintained: `build_from_va2.py`
rebuilds it from the VA_v2 page through 41 anchor-asserting edits, so a moved anchor fails
loudly instead of silently dropping half the page. After four upstream commits:

```
python3 Data_Analysis/Visualizer_UAV_v1/build_from_va2.py
  → 41 edits, 145,936 → 166,309 bytes, exit 0 (no anchor moved)
python3 Data_Analysis/Visualizer_UAV_v1/test_page_structure.py
  → ALL CHECKS PASSED
```

Inherited: `pareto` 0 → 54 occurrences, `zoom` 5 → 17, `FigWidth` 1 → 7; 480 changed
lines. The Pareto panel matters here beyond parity — "good" in this project is defined as
Pareto-dominant at equal success and constraints, so it is the plot the UAV verdicts are
written against.

## 4. The real gap: the fourth copy was unpinned

`DA_VA_v2/test_snapshot_scan.py` exists to prove the "Last Run" timestamp feature renders
**identically everywhere it is duplicated** — pipeline and page. It named three sources.
`DA_UAV_v1/discovery.py` has the feature (`scan_snapshot_timestamps`,
`format_snapshot_ts`, `snapshot_by_seed_str`, same as VA_v2) but the test did not
reference it at all, so the UAV fork could drift from its parents silently and the only
symptom would be the same run rendering two ways in the CSV and the Path Audit Map.

Extended to six sources — added `DA_UAV_v1/discovery.py` and
`Visualizer_UAV_v1/index.html`.

### 4.1 Importing a second `discovery` needed isolation

Every DA generation ships a top-level `config` imported by bare name, so two copies of
`discovery` cannot coexist in `sys.modules`. The new `load_discovery(folder, label)`
stashes the shared module names, prepends the target folder, imports, then restores
DA_VA_v2's exactly as they were — and **asserts `mod.__file__` really is inside the
requested folder**, so a path mishap aborts instead of testing the wrong file twice.

### 4.2 Verified it bites

Not just that it passes. Injecting `return 'DRIFTED'` into `DA_UAV_v1.format_snapshot_ts`:

```
FAIL  DA_UAV_v1 formats identically — got 'DRIFTED', want '2026-05-06 03:48:06'
1 FAILED  ·  exit 1
```

Source restored; `git status` clean on `DA_UAV_v1/`.

## 5. What this does NOT do

- **No real-data run.** The 2026-08-15 header still stands: `DA_UAV_v1` has never opened
  a genuine artifact tree. Path assumptions in its README are still untested.
- **No Pareto regression test for the UAV page.** VA_v2's `test_page_offline.py` gained
  194 lines of Pareto-specific checks; the UAV `test_page_structure.py` has no equivalent,
  so it passes without exercising the new panel. Not blocking; worth a later epoch.
- **The `fm` Gen15 artifacts are still not on disk.** `temp/1908/` holds Slurm logs only.
  Aggregating the 2026-08-17 `fm` runs needs the plan trees exported first.

## 6. How to verify

```bash
python3 Data_Analysis/Visualizer_UAV_v1/build_from_va2.py     # 41 edits, exit 0
python3 Data_Analysis/Visualizer_UAV_v1/test_page_structure.py
python3 Data_Analysis/DA_VA_v2/test_snapshot_scan.py          # now 6 sources
```

All three are stdlib-only and run in the AI-coding container.
