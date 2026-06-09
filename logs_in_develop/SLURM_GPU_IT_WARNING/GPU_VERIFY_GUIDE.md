# SLURM GPU Verification Guide

**Purpose:** How to verify a running job is using the correct GPU (no leak).

---

## The Problem

When a script uses `MUJOCO_GL=egl` without device pinning, EGL's `eglInitialize()`
defaults to **physical GPU 0** — regardless of which GPU SLURM allocated. This means:

- Your job opens FDs on GPU 0 even if SLURM gave you GPU 6
- You silently steal memory/compute from another user's job on GPU 0
- Your own job may crash or produce wrong results if GPU 0 is OOM

---

## The Fix (must be in every GPU sbatch script)

```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"          # stable physical ordering
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"    # first GPU SLURM gave us
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"  # pin EGL to that card
```

For **physics-only scripts** (no rendering), use `MUJOCO_GL=disabled` instead —
avoids EGL entirely, no leak possible.

---

## How to Verify a Running Job

### Step 1 — Get the job's Python PID

```bash
srun --jobid=<JOBID> --pty bash -c "pgrep -a -f <script_name>.py"
```

Example:
```bash
srun --jobid=21376 --pty bash -c "pgrep -a -f eval_visual_avoiding_dpcc.py"
# Output: 3380167 python diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py ...
```

### Step 2 — Read the process environment

```bash
srun --jobid=<JOBID> --pty bash -c "cat /proc/<PID>/environ | tr '\0' '\n' | grep -E 'CUDA_VISIBLE|MUJOCO_EGL'"
```

Example:
```bash
srun --jobid=21376 --pty bash -c "cat /proc/3380167/environ | tr '\0' '\n' | grep -E 'CUDA_VISIBLE|MUJOCO_EGL'"
# Output:
# CUDA_VISIBLE_DEVICES=6
# MUJOCO_EGL_DEVICE_ID=6
```

### What to check

| Result | Meaning |
|--------|---------|
| Both values **match** | ✅ Clean — EGL is on the allocated GPU |
| Values **differ** (e.g. `CUDA=6`, `EGL=0`) | ❌ Leak — EGL defaulted to GPU 0 |
| `MUJOCO_EGL_DEVICE_ID` **missing** | ❌ Pinning block absent from script |

---

## Alternative: Check from the SLURM log

Every GPU script prints hardware info at startup. Check the log:

```bash
grep -A3 "GPU INFO" Slurm_Codes/logs/<date>/<jobid>.log
```

This shows which physical card was assigned at launch (from `nvidia-smi`).

---

## Alternative: nvidia-smi on the node

```bash
# Find which node your job is on
squeue -j <JOBID> -o "%N"

# SSH to that node, then:
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

Cross-reference the PID with step 1 to confirm it is on the right card.

---

## Quick Reference

```bash
# Full one-shot check (replace JOBID and script name)
JOBID=21376
SCRIPT=eval_visual_avoiding_dpcc.py

PID=$(srun --jobid=$JOBID bash -c "pgrep -n -f $SCRIPT")
srun --jobid=$JOBID --pty bash -c "cat /proc/$PID/environ | tr '\0' '\n' | grep -E 'CUDA_VISIBLE|MUJOCO_EGL'"
```

Expected clean output:
```
CUDA_VISIBLE_DEVICES=<N>
MUJOCO_EGL_DEVICE_ID=<N>   ← must equal CUDA_VISIBLE_DEVICES
```
