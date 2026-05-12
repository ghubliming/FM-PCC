#!/bin/bash
#SBATCH --job-name=da_single_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=student

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
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$PYTHONPATH"

# Headless plotting setup
export MPLBACKEND="agg"

# 4) Run DA Single Analysis
# Analyze a specific experimental directory
cd "$REPO"

# Default to the most recent flow_matching_v3_ode_selectable run if not provided via CLI
INPUT_PATH=${1:-"logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable"}

python Data_Analysis/DA_Code/main_da.py \
    --input-path "$INPUT_PATH" \
    --output-path "analysis_results/single_analysis_$(basename $INPUT_PATH)_$(date +%Y%m%d_%H%M%S)"

echo "DA Single Analysis job completed successfully."
