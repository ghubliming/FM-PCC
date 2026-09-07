#!/bin/bash
#SBATCH --job-name=hffm_load
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=16G                   # Total memory
#SBATCH --time=00:30:00             # Time limit (~2x expected; aggregation only)
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

# 4) Aggregate one table per K bucket (K<K>_n<n>). Buckets are never mixed (PLAN §5 /
#    Gen13 fix_7): --flow-steps selects exactly one bucket. Mirrors the eval job —
#    DEFAULT reports the single plan-block K; set HFFM_FLOW_STEPS to report a swept grid.
#    🔴 K_ENV_SCALAR (2026-08-30) — same fix as the eval job. This loop reads HFFM_FLOW_STEPS as a
#    LIST, but load_results imports config/avoiding-d3il.py, which reads the SAME name as a scalar
#    `int(...)` at import time. A multi-K aggregation therefore died the same way the eval did
#    (job 25161). Snapshot the list, unset the variable, hand each child its own scalar K.
if [ -n "${HFFM_FLOW_STEPS:-}" ]; then
    HFFM_K_LIST="$HFFM_FLOW_STEPS"
    unset HFFM_FLOW_STEPS
    for K in $HFFM_K_LIST; do
        case "$K" in
            ''|*[!0-9]*)
                echo "[ load_results ] FATAL: HFFM_FLOW_STEPS entry '$K' is not a positive integer (list: '$HFFM_K_LIST')" >&2
                exit 2 ;;
        esac
    done
    for K in $HFFM_K_LIST; do
        echo "================================================================================"
        echo "[ load_results ] K = $K"
        echo "================================================================================"
        env "HFFM_FLOW_STEPS=$K" python FM_v3_hardflow_test/load_results_FM_v3_hardflow.py --flow-steps "$K"
    done
else
    env -u HFFM_FLOW_STEPS python FM_v3_hardflow_test/load_results_FM_v3_hardflow.py
fi

echo "Aggregation completed successfully."
