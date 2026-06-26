# Gen9 E2 U2 — GPU Leak Audit

**Date:** 2026-06-09  
**Trigger:** Pre-rerun verification requested before re-running Gen9 E2 U2 jobs  
**Verdict: ✅ Clean — no GPU leaks in any of the 6 scripts**

---

## What was checked

GPU leak pattern: `MUJOCO_GL=egl` without device pinning → EGL defaults to GPU 0 regardless
of which GPU SLURM allocated, opening device FDs on the wrong card. Fix is the 3-line pinning
block below.

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

For physics-only scripts (no rendering), `MUJOCO_GL=disabled` avoids EGL entirely — no leak
possible.

---

## Script-by-script findings

### GPU-using scripts (4)

| Script | GPU alloc | MUJOCO_GL | PYOPENGL_PLATFORM | 3-line EGL pin |
|--------|-----------|-----------|-------------------|----------------|
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh` | `--gres=gpu:1` ✅ | `egl` ✅ | `egl` ✅ | ✅ (lines 39–41) |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh` | `--gres=gpu:1` ✅ | `egl` ✅ | `egl` ✅ | ✅ (lines 47–49) |
| `Slurm_Codes/sbatch/fm_visual_avoiding/train_fm_visual_avoiding.sh` | `--gres=gpu:1` ✅ | `egl` ✅ | `egl` ✅ | ✅ (lines 47–49) |
| `Slurm_Codes/sbatch/fm_visual_avoiding/eval_fm_visual_avoiding.sh` | `--gres=gpu:1` ✅ | `egl` ✅ | `egl` ✅ | ✅ (lines 47–49) |

All 4 scripts have the complete pinning block. EGL will always bind to the SLURM-allocated GPU
(`${CUDA_VISIBLE_DEVICES%%,*}`) rather than defaulting to GPU 0.

### Coordinator scripts (2)

| Script | GPU alloc | Notes |
|--------|-----------|-------|
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/visual_avoiding_pipeline_dpcc.sh` | none (`--mem=2G`) ✅ | No GPU — only calls `sbatch` to submit train/eval jobs; no MuJoCo |
| `Slurm_Codes/sbatch/fm_visual_avoiding/fm_visual_avoiding_pipeline.sh` | none (`--mem=2G`) ✅ | Same — pure coordinator, no rendering |

No GPU resources requested → no EGL initialisation possible → clean by construction.

---

## No changes made

All scripts were already compliant. No code was modified. This document records the
verification result so future audits can reference it.

---

## Safe to re-run

Re-running any or all Gen9 E2 U2 jobs (`collect_visual_avoiding.sh`, then
`train_visual_avoiding_dpcc.sh` / `train_fm_visual_avoiding.sh`, then the eval scripts
and pipeline orchestrators) will not open stray GPU FDs. Each GPU job correctly pins
EGL to its allocated device.
