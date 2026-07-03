# Fix 5 — combined-batch candidate labels: incomplete ABC→123 migration

**Date:** 2026-07-03
**Scope:** `Data_Analysis/DA_Code_v3/main_da_batch.py`
**Debug log:** `temp/DA_debug/slurm` (`batch_avoiding_combined_20260703_093728`)

## Symptom

Visualizer "load error" when running `run_da_batch_avoiding_combined.sh` (multi
`--parent-path` combined analysis) — only on the combined run, never on single-path runs.

## Root cause

Commit `8c20b7d` ("ABC→123... across analysis modules") migrated candidate keys to
integers in `multi_candidate_discovery.py`, `batch_data_loader.py`, and
`batch_visualizer.py` — but **not** `main_da_batch.py`, which wasn't in that commit's file
list. Its multi-path merge branch (only exercised by combined runs) still relettered with
the pre-migration scheme:

```python
candidates = {chr(ord('A') + i): info for i, info in enumerate(_all_infos)}
```

With 48 merged candidates, `chr(ord('A')+i)` overflows past `Z` (26 letters) into raw
ASCII punctuation — labels `[`, `\`, `]`, `^`, `_`, `` ` `` for candidates 27–32 (visible
in the debug log), then lowercase `a`–`z` for 33–48. One label is a literal backslash.
These filesystem-unsafe, format-mismatched string keys then hit `BatchDataLoader`/
`BatchVisualizer`, which now expect the clean integer keys the rest of the pipeline
produces — that mismatch is the load error.

## Fix

`main_da_batch.py:189` now matches `multi_candidate_discovery.py`'s own convention
(`cand_idx = letter_index + 1`):

```python
candidates = {i + 1: info for i, info in enumerate(_all_infos)}
```

No overflow ceiling, no punctuation, consistent int keys end to end.

## Not fixed (flagged, not triggered by this run)

`filter_candidates()` (`multi_candidate_discovery.py:187`, only used if `--candidates` is
passed) still assumes string letter keys (`.upper()` on the selection) — wasn't touched by
either migration. Will break if someone filters a combined run by candidate. Left alone
since it wasn't the reported failure.
