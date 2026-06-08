# UAV Naive Fly Test (Gen11 Epoch 2)

Standalone test harness for the Skydio X2 model placed in Epoch 1. Validates:

1. The Epoch 1 XML actually loads in MuJoCo at runtime (the smoke load that local Docker couldn't run).
2. A cascaded PID flight controller can track hand-coded position trajectories — proves planning and execution can be decoupled exactly like FM-PCC's aligning does for the Panda.

**See:** [`logs_in_develop/Gen11/Epoch2_env/PREP_PLAN.md`](../../logs_in_develop/Gen11/Epoch2_env/PREP_PLAN.md)
and [`EXECUTION_PLAN.md`](../../logs_in_develop/Gen11/Epoch2_env/EXECUTION_PLAN.md).

## Files

- `smoke_load.py` — load X2 + step 100 ticks + print model dims (Phase 2-α).
- `flight_controller.py` — cascaded PID (Lee/Mellinger structure) with auto allocation matrix.
- `trajectories.py` — three reference trajectories (hover, step, circle).
- `run_naive.py` — driver; loads X2, runs one task, logs + renders.

## How to run (Slurm)

```
cd /path/to/FM-PCC
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh smoke      # Phase α only
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh A          # hover
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh B          # step
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 6D       # circle (6-D traj)
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 9D       # circle (9-D traj)
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh all        # everything
```

Results land in `logs/uav_naive/<task_label>/`.
