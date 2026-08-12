#!/bin/bash
#SBATCH --job-name=da_batch_va_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student

set -e

# DA_VA_v2 — visual-aligning batch analysis (Gen7 + Gen14 + D3IL baseline in one run).
#
# ZERO-ARGUMENT BY DEFAULT:
#
#     sbatch run_da_batch_va_v2.sh
#
# With no argument the script AUTO-SCANS: it takes the fixed roots listed in
# AUTO_ROOTS below, plus every D3IL-baseline export root it can find
# (`_DA_VA_BRIDGE_*` bridged / `DA_VA_*` native), keeps the ones that exist, and
# analyses them all in one comparison. Nothing to type, nothing to remember —
# to change what is scanned, edit AUTO_ROOTS, not the command line.
#
# Args (all optional):
#   $1  = parent path(s), comma-separated — OVERRIDES the auto-scan entirely
#   $2+ = forwarded verbatim to main_da_batch.py
#
# Examples:
#   sbatch run_da_batch_va_v2.sh                          # auto-scan everything
#   sbatch run_da_batch_va_v2.sh "" --splits test --plots # auto-scan + extra flags
#   sbatch run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans      # one tree only
#
# Plots are OFF unless --plots is passed: the CSVs are the product and matplotlib
# is by far the slowest stage.

# Fixed roots to scan when no path is given. Add a line to include a tree
# permanently; a root that does not exist is skipped without failing the job.
AUTO_ROOTS=(
    "logs/aligning-d3il-visual/plans"       # Gen14 engine arms + Gen7 visual aligning
)
# NOT in the default set: `logs/avoiding-d3il/plans` (state-only AVOIDING). This
# tool can read it, but it is a different task with its own metric set, and it
# carries candidates x 6 seeds x 3 geometries x ~7 variants — merging it into a
# visual-aligning comparison multiplies every table for rows nobody reads. Pass
# it explicitly as $1 when you actually want that comparison.
# Every D3IL-baseline export root is picked up automatically, so a newly bridged
# or newly evaluated baseline joins the comparison without touching this file.
AUTO_EXPORT_GLOB_DEPTH=3

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
export PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_VA_v2:$PYTHONPATH"
export MPLBACKEND="agg"

# 4) Run
cd "$REPO"

# ─── Resolve what to scan (paths are relative to $REPO, hence after the cd) ───
if [ -n "$USER_PARENT" ]; then
    PARENT_PATH="$USER_PARENT"
    echo "[ DA_VA_v2 ] parent path from CLI: ${PARENT_PATH}"
else
    FOUND=()
    for root in "${AUTO_ROOTS[@]}"; do
        if [ -d "$root" ]; then FOUND+=("$root"); else echo "[ skip ] not present: $root"; fi
    done
    # Bridged (`_DA_VA_BRIDGE_*`) and native (`DA_VA_*`) D3IL-baseline export roots.
    # -maxdepth keeps this off the deep run trees; the sort makes the order stable.
    while IFS= read -r found_root; do
        [ -n "$found_root" ] && FOUND+=("$found_root")
    done < <(find logs -maxdepth "$AUTO_EXPORT_GLOB_DEPTH" -type d \
                  \( -name '_DA_VA_BRIDGE_*' -o -name 'DA_VA_*' \) 2>/dev/null | sort)

    if [ "${#FOUND[@]}" -eq 0 ]; then
        echo "[ FATAL ] auto-scan found no tree to analyse under $REPO/logs"
        echo "[ FATAL ] edit AUTO_ROOTS in $(basename "$0"), or pass a path as \$1"
        exit 1
    fi
    PARENT_PATH=$(IFS=, ; echo "${FOUND[*]}")
    echo "[ DA_VA_v2 ] auto-scan found ${#FOUND[@]} root(s):"
    for root in "${FOUND[@]}"; do echo "[ DA_VA_v2 ]   $root"; done
fi

# The folder name matters: both HTML visualizers pick runs out of the
# analysis_results/ directory listing by leading prefix ("batch_" for
# Visualizer/index.html, "va_batch_" for Visualizer_Visual_Aligning/). The run
# writes `batch_va2_<ts>` and symlinks `va_batch_va2_<ts>` beside it, so it shows
# up in both pickers. Do not rename this to something else without reading
# Data_Analysis/DA_VA_v2/config.py:OUTPUT_FOLDER_PREFIX.
python Data_Analysis/DA_VA_v2/main_da_batch.py \
    --parent-path "$PARENT_PATH" \
    --output-path "Data_Analysis/analysis_results/batch_va2_$(date +%Y%m%d_%H%M%S)" \
    "$@"

echo "DA_VA_v2 batch analysis completed."
