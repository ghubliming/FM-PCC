#!/bin/bash
#SBATCH --job-name=da_batch_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

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
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_Code_v3:$PYTHONPATH"

# Headless plotting setup
export MPLBACKEND="agg"

# 4) Run DA Batch Analysis (v3)
cd "$REPO"

python Data_Analysis/DA_Code_v3/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --output-path Data_Analysis/analysis_results/batch_v3_$(date +%Y%m%d_%H%M%S)

echo "DA Batch Analysis v3 job completed successfully."
