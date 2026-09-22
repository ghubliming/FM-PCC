# CHANGELOG — Gen15 U18 · fix6 (2026-09-23) · the paper runs of PENDING_20260923 §2 as tracked drivers

Spec: `Writing/Working_Space/data_status/PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md` §2 (R39).
Runbook §3 rewritten with the commands: `SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md`.
No temp bash: everything is under `Slurm_Codes/sbatch/uav_avoiding_bridge/` and syncs by git (U17 lesson).

| file | change |
| :-- | :-- |
| `turbo.sh` | `MODE=paper` (T1–T5, tag `p23pv2turbo`, clock) and `MODE=papergif` (T1 + T4 only, for `turbo_gif.sh` with `GIF=2`); the five selections are the exact `Full_Path` folder names of the 19-09 batch CSV; `run_turbo()` helper; `TAG` override |
| `live_p2_meanflow.sh` (new) | P2/L1: MeanFM K1 bbunet (`MF_BACKBONE=unet`), `--config config/meanflow_projection_eval_u18_live.yaml`, seed 6, tag `p23pv2live`, plant switched by env |
| `live_p2_dpcc.sh` (new) | P2/L2: diffusion K20 via `scripts/eval.py`, `FMPCC_PROJ_CFG=config/projection_eval_u18_live_dpcc.yaml`, seed 6, tag `p23pv2live` |
| `config/meanflow_projection_eval_u18_live.yaml` (new) | `meanflow_projection_eval.yaml` with `projection_variants: [diffuser, dpcc-t-tightened]` (seeds [6], n_trials 2 already) |
| `config/projection_eval_u18_live_dpcc.yaml` (new) | `projection_eval.yaml` with seeds [6], `projection_variants: [dpcc-c-tightened]` |
| `scripts/eval.py` | `FMPCC_PROJ_CFG` selects the yaml (default `config/projection_eval.yaml`, unchanged) — the same hook the FM eval got in fix5 |

Order: P1a (`papergif`, GPU) → P1b (`paper`, CPU; T1/T4 skipped as done) → P2 (two GPU jobs). Dry-run of P1b must
list exactly ten cells; if an `--engine`/glob misses on the cluster, the dry-run shows 0 for that call and the glob is the
thing to fix (`ls logs/avoiding-d3il/plans/<engine>/<train>/`).

Not run from the container. Syntax-checked (`bash -n`, `py_compile`, yaml parse).
