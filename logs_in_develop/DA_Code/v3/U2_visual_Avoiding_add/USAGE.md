# DA v3 — Visual Avoiding + State-Only Avoiding Combined Analysis

## Command

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh
```

Discovers candidates from both `logs/avoiding-d3il/plans` (state-only) and
`logs/avoiding-d3il-visual/plans` (visual avoiding) and compares them together in one run.

Results → `Data_Analysis/analysis_results/va_avoiding_batch_<timestamp>/`

## Code change

`Data_Analysis/DA_Code_v3/main_da_batch.py` — `--parent-path` now accepts comma-separated
paths. Discovery runs on each path and candidates are merged (re-lettered A, B, C... by path).
