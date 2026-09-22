# Slurm — Gen15 U18 `uav_avoiding_bridge` (pillars_v2)

| script | what | GPU |
| :-- | :-- | :-- |
| `turbo.sh` | **Mode T**: replay stored avoiding executions (`obs_all` of every results npz) through the quadrotor plant, rescore, write `<eval>_msguavpv2s10turbo/…`. `MODE=pilot` (default, gate G1) or `MODE=all` (whole corpus). Dry-run unless `GO=1`. | no |

Mode L (live closed loop) needs no new script: run any avoiding eval sbatch with
`FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=uavpv2s10` in the environment (see `uav_avoiding_bridge/README.md`).

Plan / gates: `logs_in_develop/Gen15/U18/PLAN_20260922_U18_pillars_v2_avoiding_bridge.md`.
