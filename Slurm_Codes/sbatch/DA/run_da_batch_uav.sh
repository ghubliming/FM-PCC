#!/bin/bash
#SBATCH --job-name=da_batch_uav
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

# DA_UAV_v1 — Gen15 UAV Mix-ML batch analysis.
#
# ZERO-ARGUMENT BY DEFAULT:
#
#     sbatch run_da_batch_uav.sh
#
# With no argument the script AUTO-SCANS the roots in AUTO_ROOTS below, keeps
# the ones that exist, and analyses them all in one comparison. Nothing to type,
# nothing to remember — to change what is scanned, edit AUTO_ROOTS, not the
# command line.
#
# Args (all optional):
#   $1  = parent path(s), comma-separated — OVERRIDES the auto-scan entirely
#   $2+ = forwarded verbatim to main_da_batch.py
#
# Examples:
#   sbatch run_da_batch_uav.sh                                  # everything
#   sbatch run_da_batch_uav.sh "" --scenes corridor --engines fm,mf,af
#   sbatch run_da_batch_uav.sh "" --k 20 --variants diffuser,dpcc-c,dpcc-c-tightened
#   sbatch run_da_batch_uav.sh logs/UAV_MIX/uav-corridor            # one scene tree
#   sbatch run_da_batch_uav.sh "" --plots                        # + the PNG set
#
# Plots are OFF unless --plots is passed: the CSVs are the product and
# matplotlib is by far the slowest stage.
#
# ⚠️ DO NOT add --no-diagnostics-scan to save time. The UAV npz carries NO timing
# group (eval_artifacts.save_npz writes success/physical/constraint/goal only),
# so avg_time / fm_ms / proj_ms come from diagnostics/rollout_*_stats.json alone.
# Skipping the scan produces a batch with no time axis at all — which is the one
# axis the Gen15 K sweep exists to measure.

# Fixed roots to scan when no path is given. A root that does not exist is
# skipped without failing the job.
AUTO_ROOTS=(
    "logs/UAV_MIX"      # Gen15 — fm / mf / af / diffusion arms, all scenes, all K
    "logs/UAV_FM"       # Gen11 — the naive-FM + DPCC target rows Gen15 must beat
)
# Both are in the default set on purpose. Gen15's claim is stated against Gen11
# (PLAN §7.1: "the best Gen11 fm+DPCC row on the same scene, geo variant,
# projection variant, K and seed set"), and a comparison whose target is not in
# the same batch gets made by eye across two runs, which is how mismatched K and
# mismatched seed sets get compared. The two trees are separated by `generation`
# in every CSV, so nothing is pooled that should not be.

USER_PARENT="${1:-}"
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
export PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_UAV_v1:$PYTHONPATH"
export MPLBACKEND="agg"

# 4) Run
cd "$REPO"

# ─── Resolve what to scan (paths are relative to $REPO, hence after the cd) ───
if [ -n "$USER_PARENT" ]; then
    PARENT_PATH="$USER_PARENT"
    echo "[ DA_UAV_v1 ] parent path from CLI: ${PARENT_PATH}"
else
    FOUND=()
    for root in "${AUTO_ROOTS[@]}"; do
        if [ -d "$root" ]; then FOUND+=("$root"); else echo "[ skip ] not present: $root"; fi
    done

    if [ "${#FOUND[@]}" -eq 0 ]; then
        echo "[ FATAL ] auto-scan found no tree to analyse under $REPO/logs"
        echo "[ FATAL ] edit AUTO_ROOTS in $(basename "$0"), or pass a path as \$1"
        exit 1
    fi
    PARENT_PATH=$(IFS=, ; echo "${FOUND[*]}")
    echo "[ DA_UAV_v1 ] auto-scan found ${#FOUND[@]} root(s):"
    for root in "${FOUND[@]}"; do echo "[ DA_UAV_v1 ]   $root"; done
fi

# The folder name matters: the HTML visualizers pick runs out of the
# analysis_results/ directory listing by leading prefix ("batch_uav_" for
# Visualizer_UAV_v1/index.html, "batch_" for Visualizer/index.html — so
# `batch_uav_*` lands in both). Do not rename this without reading
# Data_Analysis/DA_UAV_v1/config.py:OUTPUT_FOLDER_PREFIX.
python Data_Analysis/DA_UAV_v1/main_da_batch.py \
    --parent-path "$PARENT_PATH" \
    --output-path "Data_Analysis/analysis_results/batch_uav_$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo "DA_UAV_v1 batch analysis completed."
