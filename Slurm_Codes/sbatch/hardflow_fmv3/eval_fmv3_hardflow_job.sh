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
# ⚠️ MATCHED BUDGET OR NOTHING (PLAN §5). In HardFlow, K == ode_t_steps drives BOTH
# the NFE and the number of NLP solves — it is not a free axis. Gen13's central error
# was comparing arms at different K, which invalidated a whole round of conclusions.
# The K grid is therefore built in here rather than left as a config edit somebody
# has to remember; each K writes its own results directory (K<K>_n<n>), so nothing
# overwrites anything.
#
# Override without editing this file:  HFFM_FLOW_STEPS="10" sbatch ...
# Re-run a finished directory:         FORCE_OVERWRITE=1 sbatch ...   (PLAN §3.6)
FLOW_STEPS_GRID="${HFFM_FLOW_STEPS:-2 5 10}"
echo "[ eval ] matched-K budgets to evaluate: $FLOW_STEPS_GRID"
echo "[ eval ] FORCE_OVERWRITE=${FORCE_OVERWRITE:-0}"

for K in $FLOW_STEPS_GRID; do
    echo "================================================================================"
    echo "[ eval ] K = $K   ($(date))"
    echo "================================================================================"
    python FM_v3_hardflow_test/eval_FM_v3_hardflow.py --flow-steps "$K"
done

echo "Evaluation completed successfully."
