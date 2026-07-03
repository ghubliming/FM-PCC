# Data Analysis — batch tools + HTML visualizers

Three batch-analysis pipelines, each pairing a Python CLI (discover → load → aggregate →
plot → report) with a standalone browser-based visualizer that reads the CSVs the Python
side writes. This doc covers running any of them in **visualizer-only mode** — skip the
slow matplotlib PNGs, keep only the CSV(s) the HTML page actually needs.

| Pipeline | sbatch script | Python entry point | CSV(s) for the HTML visualizer |
|---|---|---|---|
| DA v3 (state-only avoiding, single path) | `Slurm_Codes/sbatch/DA/run_da_batch_v3.sh` | `Data_Analysis/DA_Code_v3/main_da_batch.py` | `candidates_multidimensional_aggregated.csv`, `candidates_multidimensional_raw.csv` |
| DA combined (state-only + visual avoiding, merged) | `Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh` | `Data_Analysis/DA_Code_v3/main_da_batch.py` | same as above |
| DA visual aligning | `Slurm_Codes/sbatch/DA/run_da_batch_visual_aligning.sh` | `Data_Analysis/DA_Visual_Aligning/main_da_batch.py` | `va_candidates_dynamic.csv` |

All three share the same underlying `--no-plots` CLI flag (`argparse` on the Python side)
and, as of `logs_in_develop/DA_Code/v3/fix_5/`, all three sbatch wrappers forward extra
args through to it.

## Visualizer-only mode (skip the plots)

```bash
# Old DA v3 — single path
sbatch Slurm_Codes/sbatch/DA/run_da_batch_v3.sh --no-plots

# DA combined — state-only + visual avoiding merged
sbatch Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh --no-plots

# DA visual aligning — $1/$2 (parent-path, source) still positional; --no-plots goes after
sbatch Slurm_Codes/sbatch/DA/run_da_batch_visual_aligning.sh \
    logs/aligning-d3il-visual/plans/fm_visual_aligning json --no-plots
```

`--no-plots` skips only Phase 5A (matplotlib `plot_all`) — Phase 5B (`save_all_reports`,
which writes every CSV above) always runs regardless. This is the fast path when you only
want to browse results in the HTML visualizer, not generate a full plot folder — typically
seconds instead of ~30-60s per candidate batch.

## Viewing results — serve, don't `file://` open

The visualizers `fetch()` CSVs relative to their own path, which browsers block over
`file://`. Serve the repo root over HTTP instead:

```bash
# from the repo root, on the machine/node holding the results
python3 -m http.server 8000
```

Then open, from a browser that can reach that host/port (SSH-tunnel `-L 8000:localhost:8000`
if this is a remote cluster node):

```
http://<host>:8000/Data_Analysis/Visualizer/index.html                  # DA v3 / DA combined
http://<host>:8000/Data_Analysis/Visualizer_Visual_Aligning/index.html  # DA visual aligning
```

Both pages auto-discover available batch folders via the live directory listing from
`http.server` (falls back to `analysis_results/results_manifest.json` if directory listing
is disabled). Pick a batch from the dropdown and sync — no manual path entry needed unless
you want to point at a custom/non-standard output directory.

## Troubleshooting: "LOAD_ERROR" in the visualizer

Means the expected CSV wasn't found or didn't parse — check, in order:
1. Did the batch job actually reach Phase 5B ("GENERATING REPORTS") in its log
   (`Data_Analysis/analysis_results/<batch>/logs/batch_analysis.log`)? If it crashed
   earlier (Phase 1–4), no CSV was ever written.
2. Are you serving the right directory (`http.server` from the repo root, not from inside
   `Data_Analysis/`)? The visualizer's relative fetch paths assume repo-root serving.
3. For **combined** batches specifically: candidate keys must be plain integers
   (`1, 2, 3, …`) end to end. See `logs_in_develop/DA_Code/v3/fix_5/CHANGELOG.md` for a
   real incident where a stale re-lettering scheme (`chr(ord('A')+i)`, unmigrated from an
   earlier alphabetical-label refactor) overflowed past `Z` into raw ASCII punctuation for
   batches with >26 merged candidates, corrupting the CSV and breaking the load step.

## See also

- `logs_in_develop/DA_Code/v3/U2_visual_Avoiding_add/USAGE.md` — combined-analysis feature
  (multi `--parent-path`) origin.
- `logs_in_develop/DA_Code/v3/fix_3/fix_3.md` — zero-manifest directory-listing discovery,
  original `http.server` serving instructions.
- `logs_in_develop/DA_Code/v3/fix_5/CHANGELOG.md` — the integer-key fix for combined batches.
- `Data_Analysis/DA_Code_v3/README.md` — docs for the older single-run `main_da.py` tool
  (not the batch tool covered here — different entry point, same repo).
