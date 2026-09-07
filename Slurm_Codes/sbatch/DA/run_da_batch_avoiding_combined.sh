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
# Comma-separated parent-path: discovery runs on all three trees and candidates are
# merged (re-lettered A, B, C... sorted by path) into a single comparison run.
#
#   logs/avoiding-d3il/plans             state-only avoiding (Gen0/Gen3v6/Gen3v7/Gen7)
#   logs/avoiding-d3il-visual/plans      Gen9  visual avoiding (diffusion + fm only)
#   logs/avoiding-d3il-visual-mix/plans  Gen16 visual avoiding (diffusion/fm/mf/af)
#
# Gen16 was added 2026-08-23. Its on-disk shape is DA_Code_v3 shape B — the same
# `<candidate>/<seed>/results/halfspace_<geo>/<variant>.npz` the other two trees use
# (mix_visual_avoiding_test/eval_mix_visual_avoiding.py:586) — so discovery needs the
# root and nothing else. This line is safe to carry before the first Gen16 eval has
# ever run: discover_candidates_recursive() returns {} for a path that is not a
# directory (multi_candidate_discovery.py:244) and the other roots analyse normally.
cd "$REPO"

# ...but it does so SILENTLY, which is how a typo'd root disappears without a trace.
# Echo what is actually on disk before handing the list to the DA.
PARENTS="logs/avoiding-d3il/plans,logs/avoiding-d3il-visual/plans,logs/avoiding-d3il-visual-mix/plans"
echo "--- DA roots ---"
IFS=',' read -ra _ROOTS <<< "$PARENTS"
for _r in "${_ROOTS[@]}"; do
    if [ -d "$_r" ]; then
        echo "  [present] $_r  ($(find "$_r" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) groups)"
    else
        echo "  [ABSENT ] $_r  -- skipped (no eval has written here yet)"
    fi
done
echo "----------------"

# Extra args forwarded (e.g. `sbatch run_da_batch_avoiding_combined.sh --no-plots` to
# skip the slow matplotlib PNGs — the CSVs the HTML visualizer needs are written either way).
python Data_Analysis/DA_Code_v3/main_da_batch.py \
    --parent-path "$PARENTS" \
    --output-path Data_Analysis/analysis_results/batch_avoiding_combined_$(date +%Y%m%d_%H%M%S) \
    --no-plots \
    "$@"

echo "DA Batch Analysis (avoiding combined) job completed successfully."
