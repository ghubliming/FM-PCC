# SLURM GPU Violation — Root Cause Recheck & Fix Plan

**IT Warning:** Job 21318 (`eval_visual_avoiding_dpcc`) — allocated GPUs {2}, used GPUs {0, 2}  
**Date diagnosed:** 2026-06-08

---

## 1. Is the Warning Correct?

**Yes — and it is not an error in the SLURM allocation.** The violation is real and reproducible. Here is why it happens.

### The Two Separate GPU Access Paths

When SLURM allocates `--gres=gpu:1`, it sets `CUDA_VISIBLE_DEVICES=2` (or whichever physical index was free). This variable constrains **CUDA only** — it has zero effect on graphics/rendering contexts.

| Path | Controlled by | Notes |
|------|--------------|-------|
| PyTorch compute (CUDA tensors) | `CUDA_VISIBLE_DEVICES` | Only sees allocated GPU |
| MuJoCo EGL rendering | DRM device nodes (`/dev/dri/renderD*`) | **Ignores** `CUDA_VISIBLE_DEVICES` |

Without explicit pinning, MuJoCo's EGL backend opens the **lowest DRM node = physical GPU 0**, regardless of what CUDA was allocated. Result: compute on GPU 2, rendering quietly on GPU 0 — exactly the `0,2` pattern IT flagged.

### Why it Fires Now

Our scripts that do visual rendering (eval runs with `--record all`, camera image collection, GIF generation) all set:
```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
```
…but none of them pin the EGL device. Five scripts even hard-code `EGL_DEVICE_ID=0`, which *explicitly* forces GPU 0 regardless of allocation.

The i6-gpu-1 node does not use cgroup device isolation for DRM nodes, so the leak is both possible and silent — the job runs fine but touches an unallocated GPU.

---

## 2. The Fix

Add the following 3-line block immediately **after** the existing `MUJOCO_GL` / `PYOPENGL_PLATFORM` exports in every script that requests a GPU:

```bash
# ─── EGL GPU Isolation ──────────────────────────────────────────────────────
# Forces MuJoCo's EGL renderer onto the same physical GPU as CUDA compute.
# Without this, EGL defaults to /dev/dri/renderD128 (GPU 0) regardless of
# CUDA_VISIBLE_DEVICES.
#
# ASSUMPTION: EGL device enumeration order matches PCI bus order on this node.
# Run the node verification check (Section 5) to confirm before relying on this.
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

> **Note:** There is no universal `EGL_VISIBLE_DEVICES` variable — it is not defined by the EGL specification or NVIDIA drivers. `MUJOCO_EGL_DEVICE_ID` is the correct and only effective variable for MuJoCo. Any non-MuJoCo EGL code in the process (custom OpenGL renderers) will not be further isolated.

For the 5 scripts that already hard-code `export EGL_DEVICE_ID=0`:  
**Remove** that line and replace with the 3-line block above.  
(`EGL_DEVICE_ID` is a legacy variable from `mujoco-py` / older `dm_control`. Verify it is not consumed by the current MuJoCo binding before removing — in this project it is unused.)

---

## 3. Scripts That Need Changing

### Group A — GPU-allocated, EGL set, no pinning (25 scripts — add 3-line block)

| Script | Does rendering? |
|--------|----------------|
| `sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh` | Yes (flagged job) |
| `sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh` | Optional |
| `sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` | Yes |
| `sbatch/diffuser_visual_aligning/train_visual_aligning_dpcc.sh` | Optional |
| `sbatch/fm_visual_avoiding/eval_fm_visual_avoiding.sh` | Yes |
| `sbatch/fm_visual_avoiding/train_fm_visual_avoiding.sh` | Optional |
| `sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh` | Yes |
| `sbatch/fm_visual_aligning/train_fm_visual_aligning.sh` | Optional |
| `sbatch/Visual_Aligning/eval_visual_aligning.sh` | Yes |
| `sbatch/Visual_Aligning/eval_visual_aligning_fm.sh` | Yes |
| `sbatch/Visual_Aligning/train_visual_aligning.sh` | Optional |
| `sbatch/Visual_Aligning/train_visual_aligning_fm.sh` | Optional |
| `sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh` | Yes |
| `sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh` | Optional |
| `sbatch/imf_visual_aligning/eval_imf_visual_aligning.sh` | Yes |
| `sbatch/imf_visual_aligning/train_imf_visual_aligning.sh` | Optional |
| `sbatch/iMF/eval_imf.sh` | Yes |
| `sbatch/iMF/train_imf.sh` | Optional |
| `sbatch/Drifting/eval_drifting.sh` | Yes |
| `sbatch/Drifting/train_drifting.sh` | Optional |
| `sbatch/eval_dpcc_job.sh` | Yes |
| `sbatch/train_dpcc_job.sh` | Optional |
| `sbatch/eval_fmv3_ode_job.sh` | Yes |
| `sbatch/train_fmv3_ode_job.sh` | Optional |
| `sbatch/verify_env_job.sh` | Diagnostic |

**Subtotal: 25 scripts → add 3-line block after EGL exports.**

### Group B — GPU-allocated, EGL set, hard-coded `EGL_DEVICE_ID=0` (5 scripts — replace)

These are the worst offenders: `EGL_DEVICE_ID=0` explicitly pins rendering to GPU 0 on every run.

| Script | Current bad line |
|--------|-----------------|
| `sbatch/uav_expert_data/collect_camera_images.sh` | `export EGL_DEVICE_ID=0` |
| `sbatch/uav_expert_data/generate_gifs.sh` | `export EGL_DEVICE_ID=0` |
| `sbatch/uav_env/run_env.sh` | `export EGL_DEVICE_ID=0` |
| `sbatch/uav_naive/run_naive.sh` | `export EGL_DEVICE_ID=0` |
| `sbatch/visual_avoiding/collect_visual_avoiding.sh` | `export EGL_DEVICE_ID=0` |

**Subtotal: 5 scripts → delete `EGL_DEVICE_ID=0` line, add 3-line block.**

### Group C — No GPU allocation, EGL set (1 script — special treatment)

| Script | GPU? | Action |
|--------|------|--------|
| `sbatch/uav_expert_data/collect.sh` | None (`CUDA_VISIBLE_DEVICES` unset) | Remove `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` entirely — script does CPU-only MuJoCo rollouts, no rendering, EGL exports were added "in case" and serve no purpose. Removing them eliminates the risk. |

---

## 4. Total Count

| Group | Scripts | Action |
|-------|---------|--------|
| A — add pinning block | 25 | Insert 3-line block after EGL exports |
| B — replace bad hard-code | 5 | Delete `EGL_DEVICE_ID=0`, insert 3-line block |
| C — remove unused EGL | 1 | Delete MUJOCO_GL + PYOPENGL_PLATFORM lines |
| **Total** | **31** | |

Pipeline scripts (`*_pipeline*.sh`) and `submit.sh` do not set EGL themselves — they only call `sbatch` — so they do not need changes.

---

## 5. Verification After Fix

### Step 1 — Confirm EGL ↔ CUDA index alignment on i6-gpu-1 (do this once)

The fix assumes `MUJOCO_EGL_DEVICE_ID=N` maps to the same physical GPU as `CUDA_VISIBLE_DEVICES=N`. With `CUDA_DEVICE_ORDER=PCI_BUS_ID` this **usually** holds on standard NVIDIA setups, but the EGL driver (`eglQueryDevicesEXT`) is not contractually bound to enumerate in PCI bus order. Confirm before trusting the fix:

1. Submit a 1-GPU job and note which GPU SLURM assigns (e.g. `CUDA_VISIBLE_DEVICES=2`)
2. Inside the job, run a MuJoCo render with `MUJOCO_EGL_DEVICE_ID=2`
3. From a second shell on the node, check which GPU the render process actually opened:

```bash
lsof /dev/dri/renderD*
```

Your Python PID should appear **only** on the DRM node matching your allocation (e.g. `renderD130` for GPU 2), not on `renderD128` (GPU 0).

If the PID appears on the wrong node, the EGL index ordering doesn't match PCI ordering on this node — a PCI UUID translation table will be needed. Flag to IT or document the local mapping.

### Step 2 — Do not use nvidia-smi for this check

`nvidia-smi` monitors CUDA compute contexts and will **not** reliably report headless EGL graphics contexts. `lsof /dev/dri/renderD*` queries the kernel directly and is the correct tool.

---

## 6. Templates

Update `sbatch/templates/2026_04_30_job_template.sh` so all future GPU scripts start correctly. The template currently has `--gres=gpu:1` but **does not** contain `MUJOCO_GL` or `PYOPENGL_PLATFORM` exports. Add the full EGL section to the Environment Setup block (after conda activation):

```bash
# ─── Headless Rendering (EGL) ───────────────────────────────────────────────
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
# Pin EGL renderer to the same GPU as CUDA compute (see FIX_PLAN.md Section 2)
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

Comment out or delete the EGL block for templates that generate CPU-only scripts. The pipeline template (`2026_04_30_pipeline_template.sh`) does not set EGL directly — no change needed there.
