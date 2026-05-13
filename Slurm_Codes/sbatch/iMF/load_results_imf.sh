#!/bin/bash
#SBATCH --job-name=imf_load
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-1-student

set -e

# Logging setup
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "JOB START: $(date)"

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
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$PYTHONPATH"

# Headless plotting setup
export MPLBACKEND="agg"

# W&B Login
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
fi

# 4) Run iMF Results Analysis and Metrics Tracking
cd "$REPO"

# Load, aggregate, and visualize evaluation results
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results

echo "✓ Results analysis completed successfully."
