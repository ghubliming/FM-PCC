# SLURM GPU Check Block — U2 Injection Changelog

**Date:** 2026-06-09  
**Scope:** All `Slurm_Codes/sbatch/**/*.sh` scripts  
**Reference:** [`../GPU_VERIFY_GUIDE.md`](../GPU_VERIFY_GUIDE.md)

---

## What Was Injected

The following 5-line check block was inserted into every GPU-using script,
immediately after the EGL device pin line (`export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"`):

```bash
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi
```

**Effect:**
- Every SLURM log now records the allocated GPU and EGL device at startup
- If EGL and CUDA are on different cards (pinning block broken/missing), the job
  aborts immediately instead of silently leaking onto GPU 0 for hours

---

## Scripts Modified (31 EGL scripts + 1 template)

### EGL-rendering scripts — check block injected after EGL pin

| Script | Injection point |
|--------|----------------|
| `Drifting/eval_drifting.sh` | after line 43 |
| `Drifting/train_drifting.sh` | after line 43 |
| `Visual_Aligning/eval_visual_aligning.sh` | after line 49 |
| `Visual_Aligning/eval_visual_aligning_fm.sh` | after line 49 |
| `Visual_Aligning/train_visual_aligning.sh` | after line 41 |
| `Visual_Aligning/train_visual_aligning_fm.sh` | after line 41 |
| `d3il_visual_aligning_baseline/eval_d3il_baseline.sh` | after line 49 |
| `d3il_visual_aligning_baseline/train_d3il_baseline.sh` | after line 49 |
| `diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` | after line 49 |
| `diffuser_visual_aligning/train_visual_aligning_dpcc.sh` | after line 41 |
| `diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh` | after line 49 |
| `diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh` | after line 41 |
| `eval_dpcc_job.sh` | after line 62 |
| `eval_fmv3_ode_job.sh` | after line 63 |
| `fm_visual_aligning/eval_fm_visual_aligning.sh` | after line 49 |
| `fm_visual_aligning/train_fm_visual_aligning.sh` | after line 49 |
| `fm_visual_avoiding/eval_fm_visual_avoiding.sh` | after line 49 |
| `fm_visual_avoiding/train_fm_visual_avoiding.sh` | after line 49 |
| `iMF/eval_imf.sh` | after line 63 |
| `iMF/train_imf.sh` | after line 63 |
| `imf_visual_aligning/eval_imf_visual_aligning.sh` | after line 49 |
| `imf_visual_aligning/train_imf_visual_aligning.sh` | after line 49 |
| `train_dpcc_job.sh` | after line 62 |
| `train_fmv3_ode_job.sh` | after line 62 |
| `uav_env/run_env.sh` | after line 68 |
| `uav_expert_data/collect_camera_images.sh` | after line 73 |
| `uav_expert_data/generate_gifs.sh` | after line 75 |
| `uav_expert_data/generate_physics_gifs.sh` | after line 80 |
| `uav_naive/run_naive.sh` | after line 71 |
| `verify_env_job.sh` | after line 62 |
| `visual_avoiding/collect_visual_avoiding.sh` | after line 89 |

### Template — full EGL section added as new §3

`templates/2026_04_30_job_template.sh` had no EGL setup at all. Added a new
`# 3) GPU SETUP` section (full EGL pin + check block) between env setup and
execution sections. Old `# 3) EXECUTION` renumbered to `# 4) EXECUTION`.

---

## Scripts Skipped

### No GPU / no EGL anchor (coordinators and non-rendering jobs)

These scripts have `MUJOCO_GL=disabled` or no GPU at all — no EGL, no leak possible.

| Script | Reason |
|--------|--------|
| `uav_expert_data/collect.sh` | `MUJOCO_GL=disabled` — physics-only, no rendering |
| `*/visual_*_pipeline_*.sh` (all pipeline coordinators) | No GPU — only call `sbatch`|
| `dpcc_pipeline.sh`, `fmv3_ode_pipeline.sh` | No GPU — coordinators |
| `DA/run_da_*.sh` | No EGL rendering |
| `*/load_results_*.sh` | No GPU |
| `extract_dataset_job.sh` | No GPU |
| `templates/2026_04_30_pipeline_template.sh` | No GPU — pipeline coordinator template |

---

## How to Read the Output

Every patched job will now print at startup:
```
[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=6  MUJOCO_EGL_DEVICE_ID=6
```

If there is ever a mismatch the job aborts with:
```
[ GPU-LEAK ] EGL device (0) != CUDA (6) -- aborting
```

To verify a running job live, see [`../GPU_VERIFY_GUIDE.md`](../GPU_VERIFY_GUIDE.md).
