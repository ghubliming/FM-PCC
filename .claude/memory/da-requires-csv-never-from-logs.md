---
name: da-requires-csv-never-from-logs
description: DA is only allowed when the DA batch CSVs are on hand; never build a data analysis out of sbatch/eval logs
metadata:
  type: feedback
---

**A DA may only be written from the DA batch CSV folders.** If the CSVs are not present, there is
no DA — say so, list what is missing, and wait for the user to download them. Never substitute
sbatch/eval job logs (`temp/<ddmm>/<date>/*_uav_mix_eval_*.log` and friends) for real DA input, and
never scrape per-variant summary lines out of a log to build tables.

The CSVs arrive as a `batch_<domain>_<YYYYMMDD>_<HHMMSS>/` folder inside the user's `temp/` drop —
e.g. `temp/0309/` holds `batch_uav_20260903_204120/`, `batch_va2_20260903_154740/`,
`batch_avoiding_combined_20260903_133730/`. A UAV batch carries `uav_aggregated_long.csv`,
`uav_units_long.csv`, `uav_k_sweep.csv`, `per_rollout_detail.csv`, `candidates_*.csv`,
`data_quality.csv`, `run_config.csv`, plus `plots/` and `candidates_summary.txt`. A drop with only
`<date>/` log folders and no `batch_*` folder is **logs-only → no DA possible**.

**Why:** the logs are one seed's console spew, already filtered and rounded, with no data-quality or
provenance columns; the CSVs are the aggregated, seed-pooled, circuit-breaker-checked product of
`DA_UAV_v1`. Analysing logs produces numbers that look authoritative but are not the pipeline's
numbers, and it silently skips the quality gates.

**How to apply:** before starting any DA, check for the `batch_*` folder. Missing → stop, report
which runs have no CSV coverage, and ask for the drop. Reporting *run status* from logs (which job
finished, which died, ETAs) is fine and is not a DA. Related: [[fmpcc-dev-logs-navigation]],
[[pareto-definition-of-good]], [[da-target-is-best-baseline-variant]].
