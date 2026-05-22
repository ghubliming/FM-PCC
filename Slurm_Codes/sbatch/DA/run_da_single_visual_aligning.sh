#!/bin/bash
#SBATCH --job-name=da_single_va
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student

set -e

# Pass experiment folder as first argument:
#   sbatch run_da_single_visual_aligning.sh logs/visual-aligning-dpcc/plans/my_model
INPUT_PATH=${1:-"logs/visual-aligning-dpcc/plans/flow_matching_v3_imeanflow"}

# 1) Workspace paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

# 2) Conda
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 3) Environment
export FMPCC="$REPO"
export PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_Visual_Aligning:$PYTHONPATH"
export MPLBACKEND="agg"

# 4) Run
cd "$REPO"

python Data_Analysis/DA_Visual_Aligning/main_da.py \
    --input-path "$INPUT_PATH" \
    --seed 6 \
    --source npz \
    --output-path "Data_Analysis/analysis_results/va_single_$(basename $INPUT_PATH)_$(date +%Y%m%d_%H%M%S)"

echo "DA Visual Aligning single analysis completed."
