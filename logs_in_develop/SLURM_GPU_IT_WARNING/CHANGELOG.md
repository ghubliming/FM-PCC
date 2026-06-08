# EGL GPU Isolation — Change Log

**Date:** 2026-06-08  
**Reason:** IT violation — job 21318 used unallocated GPU 0 via MuJoCo EGL renderer  
**Ref:** `FIX_PLAN.md`, `FIX_PLAN_AUDIT.md`

---

## What Was Changed

Three lines were added to every GPU-allocated script, immediately after the existing
`MUJOCO_GL` / `PYOPENGL_PLATFORM` / `MPLBACKEND` exports:

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

Five scripts additionally had their hard-coded `export EGL_DEVICE_ID=0` line removed
(it was replaced by the block above).

One CPU-only script had its unused EGL exports removed entirely.

---

## To Revert

**Group A** — remove the 3 inserted lines.  
**Group B** — remove the 3 inserted lines and restore `export EGL_DEVICE_ID=0`.  
**Group C** — restore the 2 removed exports and the original comment.

---

## Group A — 3 lines inserted after `MPLBACKEND` (25 files)

| File | Inserted after (original line №) | Original line content |
|------|-----------------------------------|-----------------------|
| `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/diffuser_visual_aligning/train_visual_aligning_dpcc.sh` | 38 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh` | 38 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Drifting/eval_drifting.sh` | 40 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Drifting/train_drifting.sh` | 40 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/eval_dpcc_job.sh` | 59 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh` | 60 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/fm_visual_aligning/train_fm_visual_aligning.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/fm_visual_avoiding/eval_fm_visual_avoiding.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/fm_visual_avoiding/train_fm_visual_avoiding.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/iMF/eval_imf.sh` | 60 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/iMF/train_imf.sh` | 60 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/imf_visual_aligning/eval_imf_visual_aligning.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/train_dpcc_job.sh` | 59 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/train_fmv3_ode_job.sh` | 59 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/verify_env_job.sh` | 59 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Visual_Aligning/eval_visual_aligning_fm.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Visual_Aligning/eval_visual_aligning.sh` | 46 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Visual_Aligning/train_visual_aligning_fm.sh` | 38 | `export MPLBACKEND="agg"` |
| `Slurm_Codes/sbatch/Visual_Aligning/train_visual_aligning.sh` | 38 | `export MPLBACKEND="agg"` |

**Revert any Group A file:** delete the 3 lines immediately following the `MPLBACKEND` export.

---

## Group B — `EGL_DEVICE_ID=0` replaced with 3-line block (5 files)

| File | Original line № | Removed line | Replaced with |
|------|----------------|--------------|---------------|
| `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh` | 71 | `export EGL_DEVICE_ID=0` | 3-line EGL block |
| `Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh` | 73 | `export EGL_DEVICE_ID=0` | 3-line EGL block |
| `Slurm_Codes/sbatch/uav_env/run_env.sh` | 66 | `export EGL_DEVICE_ID=0` | 3-line EGL block |
| `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | 69 | `export EGL_DEVICE_ID=0` | 3-line EGL block |
| `Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh` | 87 | `export EGL_DEVICE_ID=0` | 3-line EGL block |

**Revert any Group B file:** delete the 3 inserted lines and restore `export EGL_DEVICE_ID=0` at the original position.

---

## Group C — Unused EGL exports removed (1 file)

**File:** `Slurm_Codes/sbatch/uav_expert_data/collect.sh`

| Original line № | Removed content |
|----------------|-----------------|
| 74 | `# No GPU needed for headless MuJoCo rollouts, but set EGL in case render is added.` |
| 75 | `export MUJOCO_GL=egl` |
| 76 | `export PYOPENGL_PLATFORM=egl` |

The comment on line 74 was also updated to:  
`# No GPU needed — CPU-only MuJoCo rollouts, no rendering.`

**Revert:** restore the original comment and the two `export` lines at the original positions.

---

## Template updated (1 file)

`Slurm_Codes/sbatch/templates/2026_04_30_job_template.sh` — **not yet updated** (tracked in FIX_PLAN.md Section 6; separate task).
