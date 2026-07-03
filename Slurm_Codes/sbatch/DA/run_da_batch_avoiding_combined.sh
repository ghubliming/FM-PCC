#!/bin/bash
#SBATCH --job-name=da_batch_avoiding_combined
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

# 4) Run DA Batch Analysis (v3) — state-only avoiding + visual avoiding combined
# Comma-separated parent-path: discovery runs on both trees and candidates are
# merged (re-lettered A, B, C... sorted by path) into a single comparison run.
cd "$REPO"

# Extra args forwarded (e.g. `sbatch run_da_batch_avoiding_combined.sh --no-plots` to
# skip the slow matplotlib PNGs — the CSVs the HTML visualizer needs are written either way).
python Data_Analysis/DA_Code_v3/main_da_batch.py \
    --parent-path "logs/avoiding-d3il/plans,logs/avoiding-d3il-visual/plans" \
    --output-path Data_Analysis/analysis_results/batch_avoiding_combined_$(date +%Y%m%d_%H%M%S) \
    "$@"

echo "DA Batch Analysis (avoiding combined) job completed successfully."
