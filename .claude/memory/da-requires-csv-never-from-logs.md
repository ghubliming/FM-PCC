---
name: da-requires-csv-never-from-logs
description: DA needs real result data — batch CSVs OR raw result folders (results.json/rollouts/npz); never build a DA from sbatch/eval console logs alone
metadata:
  type: feedback
---

**A DA may only be written from real result data: the DA batch CSV folders, or the raw result folders
(see Refinement below).** If neither is present, there is
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

**Refinement (2026-09-13, U14):** raw per-variant **result folders** also count as DA input —
`<geo_suffix>_<constraints>/<variant>/results.json` + `rollout_*.log` traces + `*.npz` + `diagnostics/`.
The user, when I refused such a drop for lacking `batch_*`: "this is more than csv". They are the
structured source the CSVs are built from, not console output. When using them: state that
DA_UAV_v1's quality gates / circuit breakers were not run, check `projection_health` and
`divergence` by hand, and validate any trace-derived metric against `results.json` before relying
on it. Still NOT acceptable: a drop with only `<date>/*_uav_mix_eval_*.log` job logs.

**How to apply:** before starting any DA, check for a `batch_*` folder or raw result folders. Neither → stop, report
which runs have no CSV coverage, and ask for the drop. Reporting *run status* from logs (which job
finished, which died, ETAs) is fine and is not a DA. Related: [[fmpcc-dev-logs-navigation]],
[[pareto-definition-of-good]], [[da-target-is-best-baseline-variant]].
