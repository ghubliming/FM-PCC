#!/bin/bash
#SBATCH --job-name=da_batch_va
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

# Pass args:  $1 = parent path (folder containing candidate subfolders)
#             $2 = source: 'json' (default, pre-U10.2) or 'npz' (U10.2+)
#             $3 = geo_variant: geo-constraint subfolder name, e.g. 'combined_5'
#                  (omit or leave empty for old flat-schema runs)
# Examples:
#   sbatch run_da_batch_visual_aligning.sh logs/aligning-d3il-visual/plans/fm_visual_aligning json
#   sbatch run_da_batch_visual_aligning.sh logs/aligning-d3il-visual/plans/fm_visual_aligning json combined_5
PARENT_PATH=${1:-"logs/aligning-d3il-visual/plans/fm_visual_aligning"}
SOURCE=${2:-"json"}
GEO_VARIANT=${3:-""}

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

python Data_Analysis/DA_Visual_Aligning/main_da_batch.py \
    --parent-path "$PARENT_PATH" \
    --seed 6 \
    --source "$SOURCE" \
    ${GEO_VARIANT:+--geo-variant "$GEO_VARIANT"} \
    --output-path Data_Analysis/analysis_results/va_batch_$(date +%Y%m%d_%H%M%S)

echo "DA Visual Aligning batch analysis completed."
