---
name: slurm-sbatch-is-real-entrypoint
description: Slurm_Codes/sbatch scripts are the real cluster entry points — keep them updated with code changes; NEVER violate the GPU/EGL isolation rules (IT warning)
metadata:
  type: project
---

The remote cluster executes everything through `/workspaces/FM-PCC/Slurm_Codes/sbatch/` shell scripts — they are the REAL entry points, not the Python scripts directly. When changing training/eval code, CLI flags, seeds, config keys, or conda-env requirements, check whether the matching sbatch script(s) need updating too — forgetting this is a recurring failure mode.

Jobs are submitted with the wrapper `Slurm_Codes/submit.sh`, NOT with a raw `sbatch` command: `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/<script>.sh [args...]`. The wrapper derives the job name from the script filename, routes stdout+stderr to a date-organized log (`Slurm_Codes/logs/<YYYY-MM-DD>/<HH_MM_SS>_<jobname>_<jobid>.log`), and exports `SUBMIT_TIME`/`SUBMIT_DATE` to unify pipeline logs — so never put `#SBATCH --output/--error/--job-name` assumptions or raw `sbatch` calls into docs/instructions; extra args after the script path are forwarded to the job script.

NEVER-FORGET GPU rule — see `/workspaces/FM-PCC/logs_in_develop/SLURM_GPU_IT_WARNING/`: a real IT violation occurred (June 2026, job 21318) because MuJoCo's EGL renderer grabbed unallocated GPU 0. Every GPU-allocated sbatch script must keep the EGL isolation block right after the `MUJOCO_GL`/`PYOPENGL_PLATFORM`/`MPLBACKEND` exports:

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

Never hardcode `EGL_DEVICE_ID=0` and never overwrite `CUDA_VISIBLE_DEVICES`.

**Why:** Using GPUs not allocated by SLURM interferes with other users' jobs and triggers IT warnings against the user's cluster account.

**How to apply:** When writing or editing any sbatch script, include the isolation block (copy from an existing script like `Slurm_Codes/sbatch/iMF/train_imf.sh`, which also has a runtime GPU-leak abort check). When changing code that an sbatch script invokes, update the script in the same task and mention it in the changelog. See also [[docker-no-python-cluster-only]].
