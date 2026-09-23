# CHANGELOG — Gen15 U18 · fix7 (2026-09-23) · R39 = the four Chapter 6 cells evaluated LIVE (real eval, no turbo)

Author, 23-09: UAV-pillars is the replica of D3IL-avoiding and tests the vehicle/controller; Ch 6 compares the winning
cells (MeanFM, CI-MeanFM × K1, K2) on the table vs in the air, plus one arm-vs-quadrotor path panel. Turbo replays the
Panda's setpoints open loop (the planner never sees the vehicle), so the thesis uses the live evaluation only.

| file | change |
| :-- | :-- |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh` (new, tracked) | one GPU job per `<mf|af> <K>`: plant switched by env (scale 36, clock 1 Hz, ff off, v_max 1.0, fan 4), seeds 6–10 × 3 geometries × 2 episodes, `diffuser` + `dpcc-t-tightened`; mf one call per `--seed`, af `AF_SEEDS`; tag default `p23uavpv2live`, **refuses a tag without `uav`**; sidecars per `<engine>_K<k>`. Dry-run unless `GO=1`. |
| `config/alphaflow_projection_eval_u18_live.yaml` (new) | `alphaflow_projection_eval.yaml` with `projection_variants: [diffuser, dpcc-t-tightened]` only (verified: the only differing key) |
| `Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh` (new, `git add -f`) | `plan` / `submit`: pre-flight + checkpoint check, submits the 4 jobs |

Found: `live_p2_meanflow.sh` / `live_p2_dpcc.sh` default to `TAG=p23pv2live`, which lacks `uav` → `factory.py` raises at the
plant switch. They are superseded (not edited). Spec/runbook: `data_status/PENDING_20260923_…` §2, `SLURM_RUNBOOK_20260923_…` §3.
Not run from the container; `bash -n`, yaml parse, submitter `plan` dry-run.

Claude (Opus 5.5, Claude Code) · 2026-09-23
