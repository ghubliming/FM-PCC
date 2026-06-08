# Context
Dear Liming Liu,

Our scripts have detected that your GPU server job has repeatedly used GPUs that were not allocated to you through SLURM.

Current Violation:
JOBID: 21318
User: llim
GPUs allocated: 2
GPUS used: 0,2

Please refrain from using any GPUs that were not allocated to you via CUDA_VISIBLE_DEVICES as this interferes with other jobs. Do not overwrite CUDA_VISIBLE_DEVICES.

Best regards
I6 IT

"/bin/bash: /u/home/llim/miniconda3/lib/libtinfo.so.6: no version information available (required by /bin/bash)
================================================================================
JOB START: Mon Jun  8 10:38:18 UTC 2026
JOB NAME:  eval_visual_avoiding_dpcc
JOB ID:    21318
NODE:      i6-gpu-1
GPU INFO:
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
NVIDIA RTX A5000, 530.41.03, 24564 MiB
GIT REV:   bdc2303
================================================================================
[ eval ] Recording mode set to: all
pybullet build time: Nov 28 2023 23:45:17
[ utils/setup ] Made savepath: logs/avoiding-d3il-visual/plans/visual_avoiding_dpcc/H8_K20_Ddiffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps200_bs64/H8_K20_T0.5_Ddiffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_steps200_mpc1/6

...
"

"#!/bin/bash
#SBATCH --job-name=eval_visual_avoiding_dpcc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student

set -e

# ─── Job Metadata ───────────────────────────────────────────────────────
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU info available)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# ─── Environment Setup ──────────────────────────────────────────────────
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"

# Headless rendering
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"

cd "$REPO"

# ─── Run Evaluation ─────────────────────────────────────────────────────
# Uses config/visual_avoiding_eval.yaml for seed/variant configuration.
# Model is loaded from the FM-PCC pickle config system (same as training).
# Results are saved to: logs/avoiding-d3il-visual/plans/visual_avoiding_dpcc/<exp>/<seed>/results/

# Args: $1=seed (optional), $2=record_mode (optional, default=all)
# $1 blank → seed list read from config yaml (default, runs all configured seeds sequentially).
# $1 set   → overrides yaml; only that single seed runs. Use for parallel per-seed Slurm fan-out:
#   sbatch eval_visual_avoiding_dpcc.sh 5
#   sbatch eval_visual_avoiding_dpcc.sh 6
# When record_mode != "none", the eval script auto-enables visual mode even if config
# has if_vision=False, so GIFs/videos are always captured (UF-13).
SEED_ARG=""
if [ -n "$1" ]; then
    SEED_ARG="--seed $1"
    echo "[ eval ] Overriding seed to: $1"
fi

RECORD_MODE="${2:-all}"
echo "[ eval ] Recording mode set to: $RECORD_MODE"

python diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py $SEED_ARG --record "$RECORD_MODE" --eval-on-train

echo "Job completed successfully."


# Suggeestion

Short version: you're not "wrong" in the way the email implies — you didn't overwrite `CUDA_VISIBLE_DEVICES` or grab extra GPUs on purpose — but their monitoring is very likely correct that *something* in your job touched physical GPU 0, which wasn't yours. The culprit is almost certainly EGL rendering, not your CUDA setup.

Here's the reasoning.

On the CUDA side your script is clean. You request `--gres=gpu:1`, you never touch `CUDA_VISIBLE_DEVICES`, and your PyTorch/diffusion compute lands on whatever GPU SLURM assigned you — that's the `2` in "used 0,2," the one you were actually allocated.

The problem is these two lines:

```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
```

EGL device selection is completely independent of `CUDA_VISIBLE_DEVICES`. That variable constrains *CUDA* only — it has zero effect on which GPU an EGL/OpenGL context gets created on. By default MuJoCo's EGL backend opens EGL device 0, i.e. physical GPU 0, no matter what you were allocated. So your compute runs on the allocated GPU while your *rendering* quietly opens a context on GPU 0. That's exactly the `0,2` pattern IT flagged.

Two more things point straight at this:

- Your own script comment says that when `--record` ≠ `none`, visual mode is force-enabled even if `if_vision=False`. You're running `--record all`, so frames are definitely being rendered — meaning an EGL context is definitely being created — even if you assumed this eval did no rendering. That's the smoking gun.
- Your job's `nvidia-smi` shows all 8 GPUs on the node. That tells me the cluster isolates GPUs only via `CUDA_VISIBLE_DEVICES` (no cgroup device isolation), which is precisely the setup where an EGL leak onto GPU 0 is both possible and invisible to you.

**The fix** — pin EGL to your allocated GPU. Add this before the `python` call:

```bash
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
```

SLURM has already exported `CUDA_VISIBLE_DEVICES` by the time your script body runs, so this points MuJoCo's renderer at the same physical GPU as your compute. (The `%%,*` is just defensive in case it's ever a list; with `gpu:1` it'll be a single index.)

One caveat: EGL device ordering isn't *guaranteed* to match CUDA's ordering. It usually does (both tend to follow PCI bus order), but not always — so verify rather than assume.

**Verify:** run the job and, from a shell that can see the node (or inside the job), run

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

and confirm your Python PID shows up only on the allocated GPU, not on GPU 0. If it still lands on the wrong GPU, the EGL/CUDA index ordering doesn't line up on this node and you'd have to select the EGL device by matching PCI/UUID instead — tell me if it comes to that and I'll give you that snippet.

**Replying to IT:** it's fair to push back gently. You can state that you did not overwrite `CUDA_VISIBLE_DEVICES` and weren't requesting extra GPUs; the activity on GPU 0 came from MuJoCo's EGL rendering backend, which ignores `CUDA_VISIBLE_DEVICES` by design, and you've now pinned it with `MUJOCO_EGL_DEVICE_ID`. That's accurate and good-faith. Want me to draft that reply?

---

# Fixing EGL Headless Rendering GPU Leaks in SLURM

## Root Cause Diagnosis: The Hardware Mapping Disconnect

The GPU usage violation occurs because neural network compute and hardware-accelerated rendering access the physical GPU through two completely distinct Linux interfaces.

When SLURM allocates a GPU via `#SBATCH --gres=gpu:1`, it sets the `CUDA_VISIBLE_DEVICES` environment variable.

* **The Compute Path (PyTorch):** The CUDA runtime respects `CUDA_VISIBLE_DEVICES` and binds tensor operations to the allocated GPU (e.g., GPU 2).
* **The Graphics Path (MuJoCo/EGL):** The EGL backend completely ignores CUDA environment variables. It directly probes the Direct Rendering Manager (DRM) nodes (`/dev/dri/renderD*`). Without explicit instructions, EGL defaults to the lowest enumerated DRM node, which corresponds to physical GPU 0.

Because an 8-GPU node like `i6-gpu-1` does not utilize strict cgroup device isolation for DRM nodes, your process successfully opens a file descriptor on GPU 0 for rendering while performing compute on GPU 2.

## The Fix: Aligning Hardware Topologies

To resolve this, you must force the EGL rendering context to target the exact DRM node that corresponds to your CUDA allocation. Furthermore, because NVIDIA's default CUDA enumeration (`FASTEST_FIRST`) does not natively guarantee a 1:1 mapping with the physical PCI bus enumeration used by the Linux DRM, you must explicitly align their topologies.

Inject the following configuration into your SLURM batch script (`eval_visual_avoiding_dpcc.sh`), immediately before executing your Python evaluation script:

```bash
# ─── GPU Isolation Fix for Headless EGL Rendering ───────────────────────

# 1. Enforce physical PCI ordering so CUDA indices match Linux DRM indices
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# 2. Extract the first allocated GPU index (defensive slicing for multi-GPU setups)
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"

# 3. Pin MuJoCo and general EGL contexts strictly to the allocated GPU
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
export EGL_VISIBLE_DEVICES="$ALLOCATED_GPU"

```

## Rigorous Verification

Do not rely solely on `nvidia-smi` to verify this fix. `nvidia-smi` is designed to monitor CUDA compute contexts and often fails to reliably report pure headless EGL graphics contexts.

To strictly prove that your Python process is no longer touching GPU 0, query the Linux kernel directly to see which processes hold open file descriptors on the DRM nodes.

While your job is running, execute the following command from a shell on the allocated node:

```bash
lsof /dev/dri/renderD*

```

**Expected Output Analysis:**
Look for your Python execution process ID (PID). It should only appear next to the DRM node corresponding to your allocated GPU (e.g., `renderD130` for GPU 2). If your PID appears next to `renderD128` (GPU 0), the topology mapping has failed, and strict PCI UUID matching will be required.