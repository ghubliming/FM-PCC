#!/bin/bash
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --partition=gpu-1-student   # Updated from sinfo output
# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
# 1) Create a shortcut to the latest log for easy monitoring
# We use scontrol to find the path defined by the submit.sh wrapper
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

# 1) Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

# 2) Initialize Conda
# Point to the conda.sh in your miniconda installation
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

# W&B optimization
export WANDB_MODE="online"

# 4) Run Training
cd "$REPO"

# Example training command (adjust arguments as needed)
# Using 'python' here will use the one from the activated conda env
python scripts/train.py \
    --seeds 6 \
    --num-seeds 1 \
    --use-wandb \
    --wandb-project FMPCC \
    --auto-resume

echo "Job completed successfully."
