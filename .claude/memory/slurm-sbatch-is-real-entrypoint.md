---
name: slurm-sbatch-is-real-entrypoint
description: Slurm_Codes/sbatch scripts are the real cluster entry points — keep them updated with code changes; NEVER violate GPU/EGL isolation; size --time with a 2x safety margin (24h hard cap); never use tqdm/live progress bars in batch job console output
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

**`#SBATCH --time` sizing — always lean toward too much, never too little.** The cluster's hard ceiling is **24:00:00**; a job killed mid-run for hitting its time limit wastes the whole run (checkpoints/partial results may be lost) and is far more costly than a slightly longer allocation sitting idle at the end. Rule of thumb: **request ~2x the realistically expected duration**, capped at 24h — e.g. expect ~2h of actual work → request `04:00:00`; expect ~10h → request `20:00:00` (not 24h just because you can). Never request exactly the expected duration or less. If a job is expected to exceed ~12h, flag it to the user explicitly (they may want to split it or checkpoint/resume) rather than silently requesting the full 24h.

**NEVER use tqdm / live-updating progress bars in a script an sbatch job invokes.** `Slurm_Codes/submit.sh` redirects stdout+stderr to a log **file**, not a live terminal — tqdm's carriage-return trick to update in place does not collapse in a file, so a per-iteration `pbar.set_postfix(...)` (or similar) dumps every single update as raw text, producing multi-thousand-character unreadable log lines (real incident: HardFlow eval job 23565, `logs_in_develop/Gen13/fix_2/CHANGELOG_Gen13_fix2_pipeline_and_quiet_logs.md`). **How to apply:** any progress reporting inside code that will run under `submit.sh` must either (a) gate the live bar behind `sys.stdout.isatty()` so it only renders in an interactive terminal, or (b) print one compact plain-text line per meaningful unit of work (e.g. per episode/epoch), never per inner-loop step. If the noisy code is pre-existing and off-limits to edit, fork just the reporting wrapper into a new file rather than leaving the noise in place — don't accept "that's just how it logs" as an answer.
