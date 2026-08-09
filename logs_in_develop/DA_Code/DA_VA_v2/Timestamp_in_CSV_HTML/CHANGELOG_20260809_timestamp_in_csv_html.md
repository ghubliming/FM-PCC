# Timestamp in CSV + HTML — "when was this run produced?"

**Date:** 2026-08-09
**Scope:** `DA_VA_v2` **and** `DA_Code_v3` (DAv3), plus both HTML viewers
**Status:** written + locally tested (stdlib parts); **pipeline/viewer runs still to be done on the cluster**

---

## Why

A candidate row in the Path Audit Map is just a path. Two candidates that look
identical can be a week apart, and a candidate that was re-run last night is
indistinguishable from one whose npz files have been sitting there since Gen7 —
you had to `ls` the run folder to find out.

The information already exists on disk. `diffuser/utils/setup.py::snapshot_configs`
(and every generation's copy of it) writes, on **every** `Parser()` call:

```
<savepath>/config_snapshot_<config>/
    ├── <config>.py                 # copy of the config module
    ├── projection_eval.yaml        # copy of the projection config
    └── snapshot_20260506_034806    # marker file, one per launch, never deleted
```

For a `.../plans/<exp>/<seed>/` folder that is **one marker per eval launch**, so:

* newest marker = when this seed's results were last (re)generated
* marker count = how many times the folder has been written into

That is an audit trail, not just an mtime — which is why the markers are read
rather than `os.path.getmtime`.

## What was added

### 1. Discovery scans the markers

New in **both** discovery modules (copy-modify sibling pattern, kept byte-for-byte
equivalent and tested for agreement):

* `Data_Analysis/DA_VA_v2/discovery.py`
* `Data_Analysis/DA_Code_v3/multi_candidate_discovery.py`

```python
scan_snapshot_timestamps(candidate_path, seeds=None)
  -> {'latest', 'first', 'count', 'per_seed', 'n_seeds_stamped'}
format_snapshot_ts('20260506_034806')  -> '2026-05-06 03:48:06'
snapshot_by_seed_str({6: ..., 10: ...}) -> '6:2026… | 10:2026…'   (numeric seed order)
```

Each discovered candidate now carries `info['snapshots']`. `SNAPSHOT_DIR_PREFIX`
lives in `DA_VA_v2/config.py` next to the other discovery constants; DAv3 keeps
its own copy at the top of `multi_candidate_discovery.py`.

Never raises: a tree with no markers reports empty strings and `count = 0`.

### 2. The CSVs carry it

| file | column(s) | grain |
|---|---|---|
| `va2_units_long.csv` (`LatestSnapshot`)<br>`candidates_multidimensional_raw.csv` (`Latest_Snapshot`) | 1 | **that seed's** newest marker; blank when that seed has none |
| `va2_aggregated_long.csv` (`LatestSnapshot`)<br>`candidates_multidimensional_aggregated.csv`, `candidates_ranking.csv` (`Latest_Snapshot`) | 1 | newest marker over all the candidate's seeds |
| `candidates_detailed.csv` | 4 | `Latest_Snapshot`, `First_Snapshot`, `Snapshot_Count`, `Snapshot_By_Seed` |
| `candidates_summary.txt` | — | human-readable per candidate, **plus a per-seed line when the seeds disagree** |

Naming follows each schema's own convention: `LatestSnapshot` in the DA_VA_v2
native cube (`FolderName`/`FullPath` style), `Latest_Snapshot` in the DAv3-compat
files (`Folder_Name`/`Full_Path` style). `build_from_dav3.py` translates one into
the other exactly where it already translates the rest.

Deliberate: the per-seed files do **not** fall back to the candidate's `latest`.
A seed that was never re-run stays visibly blank instead of borrowing a sibling
seed's freshness. `Snapshot_By_Seed` is the column to read when a candidate looks
half-stale — seeds are usually separate jobs, so one fresh seed can hide four old
ones behind a recent `Latest_Snapshot`.

### 3. Both viewers show a **Last Run** column

`Data_Analysis/Visualizer/index.html` (DAv3) gained `STAMP_COL`, `_fmt_stamp()`,
`_stamp_map()`, `_stamp_cell()` and renders the column in:

* **Path Audit Map** (after `Warnings`, before `Source Path`)
* **Plot Legend — Selected Candidates (this plot only)** (after `ID`)
* the exported ZIP's audit `.txt` and the LaTeX "Candidate source paths" section

`Data_Analysis/Visualizer_VA_v2/index.html` inherits all of it — it was
**regenerated** with `build_from_dav3.py` (33 edits, 1957 lines), which is also
where `LatestSnapshot -> Latest_Snapshot` is mapped into `derive_frames()`.

Rendered as `2026-05-06 03:48:06`; a candidate with no marker gets `—`.

## Backward compatibility

Batches produced **before** this change have no `Latest_Snapshot` column. Both
pages detect that and **omit the column entirely** rather than rendering a table
of blanks — nothing to migrate, but an old batch folder in `analysis_results/`
will not gain the column until the DA pipeline is re-run over its source tree.

## Files touched

```
Data_Analysis/DA_Code_v3/multi_candidate_discovery.py    scanner + formatters, info['snapshots']
Data_Analysis/DA_Code_v3/batch_reporter.py               4 CSVs + summary txt
Data_Analysis/DA_Code_v3/README.md                       new section
Data_Analysis/DA_VA_v2/config.py                         SNAPSHOT_DIR_PREFIX
Data_Analysis/DA_VA_v2/discovery.py                      scanner + formatters, manifest, summary
Data_Analysis/DA_VA_v2/reporter.py                       6 tables + summary txt + NOTES
Data_Analysis/DA_VA_v2/README.md                         new section
Data_Analysis/DA_VA_v2/test_snapshot_scan.py             NEW — stdlib regression test
Data_Analysis/Visualizer/index.html                      Last Run column, audit txt, LaTeX
Data_Analysis/Visualizer_VA_v2/build_from_dav3.py        _norm + derive_frames mapping
Data_Analysis/Visualizer_VA_v2/index.html                REGENERATED from the above
Data_Analysis/Visualizer_VA_v2/test_page_offline.py      3 new checks
```

## Testing

**Ran here (container, stdlib only — no pandas needed):**

```bash
python3 Data_Analysis/DA_VA_v2/test_snapshot_scan.py     # 30/30 PASS
```

Covers: multi-seed trees, per-seed newest, seeds with no snapshot dir, junk
filenames in the snapshot dir (`snapshot_not_a_timestamp`, `.bak`), missing paths,
seed filtering, numeric seed ordering, DA_VA_v2 vs DA_Code_v3 agreement, and that
the **HTML pages' `_fmt_stamp` matches the pipeline's `format_snapshot_ts`** (it
is extracted from both `index.html` files and exec'd standalone). A drift there
would render the same run two different ways in the CSV and in the viewer.

Also verified end-to-end against real run folders in `temp/0508/` — discovery
reports e.g. `Last run: 2026-08-04 23:44:16 (3 config snapshot(s) over 1 seed(s))`.

Both `index.html` Python blocks and every edited module byte-compile clean.

**Still to run on the cluster:**

```bash
# 1. regenerate a batch and eyeball the new columns
python Data_Analysis/DA_VA_v2/main_da_batch.py --parent-path <plans tree> \
    --output-path Data_Analysis/analysis_results/batch_va2_<ts>
head -1 <out>/va2_aggregated_long.csv          # LatestSnapshot present
head -1 <out>/candidates_multidimensional_aggregated.csv   # Latest_Snapshot present
cat <out>/candidates_summary.txt               # "last run …" per candidate

# 2. the viewer wiring test (needs pandas + matplotlib)
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py <out>

# 3. DAv3 batch, same check
python Data_Analysis/DA_Code_v3/main_da_batch.py --parent-path <plans tree> ...
```

## Known limitations

* The marker is written when the eval **starts** (at `Parser()`), not when it
  finishes. A crashed run still leaves a marker — cross-check `data_quality.csv`
  / `npz_complete` before reading "last run" as "last successful run".
* If a run folder is reused by a *different* config, only the newest marker shows
  in the viewer; `Snapshot_Count` and the raw `config_snapshot_*` directory are
  the full record.
* Timestamps are the wall clock of the machine that ran the eval (cluster time).
