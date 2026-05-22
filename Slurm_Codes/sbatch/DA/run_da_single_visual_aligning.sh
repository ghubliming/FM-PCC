#!/bin/bash
#SBATCH --job-name=da_single_va
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student

set -e

# Pass args:  $1 = model experiment folder — the folder that DIRECTLY contains
#                  the seed subfolder (e.g. the folder where {seed}/ lives).
#                  For nested layouts like .../H8_Dfm_.../H8_K100_...mpc4/6/...
#                  pass .../H8_K100_...mpc4  (not a parent level).
#                  Prefer the batch script for auto-discovery of nested layouts.
#             $2 = source: 'json' (default, pre-U10.2) or 'npz' (U10.2+)
# Example:
#   sbatch run_da_single_visual_aligning.sh \
#     logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_.../H8_K100_...mpc4 json
INPUT_PATH=${1:?"ERROR: must pass model experiment folder as \$1"}
SOURCE=${2:-"json"}

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
    --source "$SOURCE" \
    --output-path "Data_Analysis/analysis_results/va_single_$(basename $INPUT_PATH)_$(date +%Y%m%d_%H%M%S)"

echo "DA Visual Aligning single analysis completed."
