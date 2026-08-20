#!/bin/bash
#SBATCH --job-name=hffm_eval
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                   # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit (~2x expected; arm C runs K NLP solves per plan)
#SBATCH --partition=gpu-1-student   # Updated from sinfo output
# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
# 1) Create a shortcut to the latest log for easy monitoring
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "No GPU detected or nvidia-smi failed"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
echo "================================================================================"

# Trap for JOB END
function on_exit {
    echo "================================================================================"
    echo "JOB END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# 1) Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

# 2) Initialize Conda
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 3) Set Environment Variables
export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"

# Rendering variables for MuJoCo on headless remote nodes
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

# W&B Login (Colab-style from key file)
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"

# 4) Gen12 evaluation: arms A (unguided) / B (DPCC projection) / C (hardflow_new),
#    all three in ONE process per K so seeds, env resets and the checkpoint are shared.
#
# Config split (edit -> submit):
#   - which model : plan_fm_v3_hardflow block in config/avoiding-d3il.py (checkpoint_dir/loadpath)
#   - seeds / arms / geometry / arm-C tuning : config/hardflow_projection_eval.yaml
#
# ⚠️ MATCHED BUDGET (PLAN §5): each K in the grid is applied to ALL arms at once, so A/B/C are
# always compared at equal K. Each K writes its own results dir (K<K>_n<n>), so nothing overwrites
# and the sweep also confirms the K-override works (K2/K5/K10 dirs appear).
#
# K: DEFAULT is a SINGLE run at the plan block's `flow_steps` (config/avoiding-d3il.py).
# To SWEEP, set HFFM_FLOW_STEPS to a space-separated list — each K is applied to ALL arms
# (matched budget, PLAN §5) and writes its own results dir (K<K>_n<n>), so nothing overwrites:
#   HFFM_FLOW_STEPS="2 5 10"  sbatch ...
# Re-run a finished directory: FORCE_OVERWRITE=1 sbatch ...   (PLAN §3.6)
#
# 🔴 B4_PARITY (2026-08-20) — arm C's candidate fan. This script never exported HFFM_BATCH,
# so it inherits config/hardflow_projection_eval.yaml `hardflow.batch_size`, which is NOW 4
# (was 1). 4 == the plan block's batch_size, i.e. the fan arms A/B already use — required,
# because both arms loop serially over candidates around their CPU solve and a 1-vs-4 fan
# silently hands arm C a 4x compute discount. Bare `hardflow_new` stays at 1 regardless
# (resolve_hf_batch_size), so the faithful upstream control is still one variant away.
# Set HFFM_BATCH=1 only to deliberately reproduce an old `B1` run.
export HFFM_BATCH="${HFFM_BATCH:-4}"
echo "[ hardflow ] HFFM_BATCH=$HFFM_BATCH (arm-C fan for -r/-c/-t; bare hardflow_new is always 1)"
if [ -n "${HFFM_FLOW_STEPS:-}" ]; then
    echo "[ eval ] K sweep: $HFFM_FLOW_STEPS   FORCE_OVERWRITE=${FORCE_OVERWRITE:-0}"
    for K in $HFFM_FLOW_STEPS; do
        echo "================================================================================"
        echo "[ eval ] K = $K   ($(date))"
        echo "================================================================================"
        python FM_v3_hardflow_test/eval_FM_v3_hardflow.py --flow-steps "$K"
    done
else
    echo "[ eval ] single run, K from plan block   FORCE_OVERWRITE=${FORCE_OVERWRITE:-0}"
    python FM_v3_hardflow_test/eval_FM_v3_hardflow.py
fi

echo "Evaluation completed successfully."
