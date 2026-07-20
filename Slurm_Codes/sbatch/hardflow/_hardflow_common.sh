#!/bin/bash
# ==============================================================================
# _hardflow_common.sh  —  SOURCED bridge for running the HardFlow repo replication
#                         from a CLONE of the FMPCC conda env (never the live one).
#
# Design: run HardFlow UNMODIFIED. No edits to HardFlow's source. Everything here
# is a *bridge* — env activation, PYTHONPATH order, a logs symlink, GPU/EGL guard.
# It does NOT create or mutate any conda env (build + reconcile the clone by hand,
# see EVALUATION_hardflow_readiness_and_iMF_swap.md §3.05).
#
# Sourced by each hardflow sbatch AFTER its #SBATCH header. Overridable env vars:
#   CONDA_ENV_NAME   clone env to activate   (default hardflow_clone; MUST NOT be FMPCC)
#   HARDFLOW_REPO    path to the pulled HardFlow checkout
#   FMPCC_ROOT       workspace root holding FM-PCC (default $HOME/FMPCC)
#   HARDFLOW_LOG_COLLECT   where outputs collect  (default $REPO/logs/hardflow)
# ==============================================================================
set -e

# ---- 1. Paths + env name (HARDCODED, mjx-style) -------------------------------
# HARDCODED on purpose: this job may be submitted from an already-activated FMPCC
# shell, so inherited env vars would make `${VAR:-default}` unreliable. Keep these
# literal (mirrors sbatch/uav_fm/eval_fm_uav.sh's FMPCC_mjx clone pattern).
# Only $HOME is dynamic (always safe). Edit these lines if your layout differs.
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
# The reconciled CLONE of FMPCC (gym 0.20 + tyro), never the live FMPCC env.
# (FMPCC has gym 0.26 -> HardFlow's avoiding rollout would break there.)
CONDA_ENV_NAME="hardflow_clone"
# HardFlow is VENDORED into FM-PCC (copied code, tracked by FM-PCC git); it ships
# to the cluster via `git pull` at $REPO/HardFlow.
HARDFLOW_REPO="$REPO/HardFlow"
# Collect ALL HardFlow train/eval outputs here, inside FM-PCC (gitignored logs/*).
HARDFLOW_LOG_COLLECT="$REPO/logs/hardflow"

if [ ! -d "$HARDFLOW_REPO" ]; then
    echo "[ HF-BRIDGE ] ERROR: HARDFLOW_REPO not found at '$HARDFLOW_REPO'. Git-pull it there or export HARDFLOW_REPO." >&2
    exit 1
fi

# ---- 2. Pro-logging (latest.log shortcut + banner + end trap) -----------------
CURRENT_LOG=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'StdOut=\K\S+' || true)
[ -n "$CURRENT_LOG" ] && ln -snf "$CURRENT_LOG" "$REPO/Slurm_Codes/logs/latest.log" 2>/dev/null || true
echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   NODE: $(hostname)"
echo "CONDA ENV: $CONDA_ENV_NAME   (clone of FMPCC — live env untouched)"
echo "HARDFLOW_REPO: $HARDFLOW_REPO"
echo "LOG COLLECT:   $HARDFLOW_LOG_COLLECT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "no nvidia-smi"
( cd "$HARDFLOW_REPO" && echo "HARDFLOW GIT: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)" )
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "JOB END: $(date)"; echo "================================================================================"; }
trap on_exit EXIT

# ---- 3. Activate the CLONE (the sbatch never builds/mutates the env) -----------
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
echo "[ HF-BRIDGE ] python: $(which python)"

# ---- 4. Runtime env vars ------------------------------------------------------
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MPLBACKEND="agg"

# ---- 5. PYTHONPATH: HardFlow repo, then the bridge shims ----------------------
#        HardFlow FIRST so `import hardflow` / `import d3il` resolve to HardFlow's
#        OWN bundled packages. Then `shims/` supplies a tiny empty `d4rl` module
#        (HardFlow imports d4rl but never uses it on the avoiding task; the real
#        d4rl is absent from the FMPCC clone). See shims/d4rl.py.
#        Set FRESH (NOT appended) on purpose: if you submit from an activated
#        FMPCC shell, an inherited PYTHONPATH could carry FMPCC's d3il in and
#        shadow HardFlow's. Conda provides all real deps. (No `pip install -e .`.)
export PYTHONPATH="$HARDFLOW_REPO:$REPO/Slurm_Codes/sbatch/hardflow/shims"

# ---- 6. Sanity-check the clone was reconciled (do NOT auto-install) ------------
#        Job-time installs race across concurrent jobs and mask real env problems.
if ! GYM_VER=$(python -c "import tyro, gym; print(gym.__version__)" 2>/dev/null); then
    echo "[ HF-BRIDGE ] ERROR: clone '$CONDA_ENV_NAME' is missing tyro/gym. Reconcile it first (see §3.05)." >&2
    exit 1
fi
echo "[ HF-BRIDGE ] tyro OK, gym==$GYM_VER"
case "$GYM_VER" in
    0.20.*) : ;;
    *) echo "[ HF-BRIDGE ] WARNING: gym==$GYM_VER (HardFlow's d3il expects 0.20.x — avoiding rollout may break)." ;;
esac

# ---- 6b. W&B login (U9) — same convention as sbatch/iMF/train_imf.sh ----------
#        Key file lives at $HOME/FMPCC/.wandb_api_key (Colab-style). If absent,
#        training still runs and falls back to metrics.csv (never blocks a job).
if [ -f "$FMPCC_ROOT/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat "$FMPCC_ROOT/.wandb_api_key")
    export WANDB_MODE="online"
    echo "[ HF-BRIDGE ] W&B key found -> WANDB_MODE=online"
else
    echo "[ HF-BRIDGE ] no $FMPCC_ROOT/.wandb_api_key -> W&B disabled (CSV only)"
fi

# ---- 7. Headless MuJoCo + GPU/EGL isolation (FMPCC standing rule) --------------
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting" >&2
    exit 1
fi

# ---- 8. Redirect HardFlow's ./logs INTO FM-PCC via a symlink -------------------
#        Robust to eval.py's MIXED paths: cfg.log_folder default "logs" AND the
#        HARDCODED "logs" for dynamics (eval.py:517) both resolve to ./logs
#        relative to cwd=HARDFLOW_REPO, so the symlink catches both. A
#        --log_folder flag would NOT (it misses the hardcoded dynamics path).
mkdir -p "$HARDFLOW_LOG_COLLECT"
if [ -L "$HARDFLOW_REPO/logs" ]; then
    ln -snf "$HARDFLOW_LOG_COLLECT" "$HARDFLOW_REPO/logs"
elif [ -e "$HARDFLOW_REPO/logs" ]; then
    echo "[ HF-BRIDGE ] ERROR: $HARDFLOW_REPO/logs is a real dir. Move/remove it so the bridge can symlink it into FM-PCC." >&2
    exit 1
else
    ln -s "$HARDFLOW_LOG_COLLECT" "$HARDFLOW_REPO/logs"
fi
echo "[ HF-BRIDGE ] $HARDFLOW_REPO/logs -> $HARDFLOW_LOG_COLLECT"

# ---- 9. Work from the HardFlow repo root (relative ./logs, run/, d3il/) --------
cd "$HARDFLOW_REPO"
echo "[ HF-BRIDGE ] cwd: $(pwd)"
