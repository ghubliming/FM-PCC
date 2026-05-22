#!/bin/bash
#SBATCH --job-name=da_batch_va
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

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

# --source npz (default, requires U10.2 NPZ)
# Change to --source json for pre-U10.2 runs
python Data_Analysis/DA_Visual_Aligning/main_da_batch.py \
    --parent-path logs/visual-aligning-dpcc/plans \
    --seed 6 \
    --source npz \
    --output-path Data_Analysis/analysis_results/va_batch_$(date +%Y%m%d_%H%M%S)

echo "DA Visual Aligning batch analysis completed."
