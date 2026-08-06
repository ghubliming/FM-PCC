#!/bin/bash
#SBATCH --job-name=da_batch_va_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

# DA_VA_v2 — visual-aligning batch analysis (Gen7 + Gen14 in one run).
#
# Args:  $1  = parent path(s), comma-separated; each is scanned for candidate
#              folders and the results are merged into a single comparison run
#        $2+ = forwarded verbatim to main_da_batch.py
#
# Examples:
#   # every Gen14 engine arm under the visual-aligning plans tree
#   sbatch run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans
#
#   # Gen14 + Gen7 + the state-only avoiding tree side by side
#   sbatch run_da_batch_va_v2.sh \
#       "logs/aligning-d3il-visual/plans,logs/avoiding-d3il/plans"
#
#   # test split only, both geometries, with PNGs
#   sbatch run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans --splits test --plots
#
# Plots are OFF unless --plots is passed: the CSVs are the product and matplotlib
# is by far the slowest stage.
PARENT_PATH=${1:-"logs/aligning-d3il-visual/plans"}
if [ "$#" -ge 1 ]; then shift 1; fi

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
export PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_VA_v2:$PYTHONPATH"
export MPLBACKEND="agg"

# 4) Run
cd "$REPO"

python Data_Analysis/DA_VA_v2/main_da_batch.py \
    --parent-path "$PARENT_PATH" \
    --output-path "Data_Analysis/analysis_results/va2_batch_$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo "DA_VA_v2 batch analysis completed."
