# UAV in Env Test (Gen11 Epoch 3)

Extends the Epoch 2 cascaded-PID flight controller to operate inside a
real MuJoCo scene (floor + lighting + skybox + optional obstacles).
Epoch 2 code (`uav_naive_test/`) is untouched; this directory is a
parallel copy with surgical edits.

**See:**
- [`logs_in_develop/Gen11/Epoch_3_uav_in_env/PLAN.md`](../logs_in_develop/Gen11/Epoch_3_uav_in_env/PLAN.md)
- [`logs_in_develop/Gen11/Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md`](../logs_in_develop/Gen11/Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md) — what carried over

## Scenes (XML wrappers around Epoch 1's `quadrotor_modified.xml`)

| Scene | Geometry | Use |
|---|---|---|
| `empty`    | floor + skybox + lights | visual baseline; controller sanity |
| `corridor` | + 2 parallel walls (4 m × 1 m gap × 1.5 m tall) | straight-line traverse |
| `s_curve`  | + 2 offset corridor segments | thread S-shaped path |
| `pillars`  | + 6 cylinder obstacles (2 columns × 3) | sinusoidal weave |

## Files in this directory

- `flight_controller.py` — verbatim copy from Epoch 2.
- `trajectories.py` — Epoch 2 set (`hover_at`, `step_to`, `circle`) + new
  factories (`traverse_line`, `s_curve_path`, `weave`).
- `smoke_load_env.py` — load any scene, drop drone, confirm contact with floor.
- `run_env.py` — driver. Dispatches on `--scene` and `--task`; logs to
  `logs/uav_env/<scene>_<task_label>/`; tracks per-step obstacle-contact
  count (warn-only, never abort).

## How to run (Slurm)

```bash
cd /path/to/FM-PCC
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_empty       # gate
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh empty C           # Epoch-2 circle in real scene
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh corridor traverse
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh s_curve s_curve
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh pillars weave
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh all               # all 4 scenes one job
```

Results land in `logs/uav_env/<scene>_<task_label>/`.
