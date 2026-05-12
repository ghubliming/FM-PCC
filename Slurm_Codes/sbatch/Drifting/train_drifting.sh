#!/bin/bash
#SBATCH --job-name=drifting_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student

set -e

# Logging setup
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "JOB START: $(date)"

# Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
DRIFTING="$FMPCC_ROOT/drifting"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# W&B Login
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
fi

cd "$DRIFTING"

# Train Generator or MAE based on config
# For generator: python main.py --config configs/gen/pixel_sota_L.yaml --workdir /path/to/runs
# For MAE: python main.py --config configs/mae/pixel_640.yaml --workdir /path/to/runs
python main.py \
    --config configs/gen/pixel_sota_L.yaml \
    --workdir "$FMPCC_ROOT/drifting_runs"

echo "Job completed successfully."
