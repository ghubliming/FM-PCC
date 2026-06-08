#!/bin/bash
#SBATCH --job-name=revert_checkpoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-1-student

set -e

# ─── TARGET — edit this before submitting ────────────────────────────────────
# Set to the experiment folder path (relative to $HOME or absolute).
# Pass the NEW (mistakenly patched) folder name. If the folder name contains no patched tokens the
# script exits immediately without touching anything (safe to re-run).
TARGET_PATH="FMPCC/FM-PCC/logs/aligning-d3il-visual/visual_aligning_dpcc/H8_K20_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_aw10_VTrue_steps900_bs64"
# Optional flags: --dry-run  --backup  (space-separated if both needed)
EXTRA_FLAGS=""
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# ─── Environment ─────────────────────────────────────────────────────────────
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"

cd "$REPO"

# ─── Resolve path ────────────────────────────────────────────────────────────
if [[ "$TARGET_PATH" != /* ]]; then
    TARGET_PATH="$HOME/$TARGET_PATH"
fi
echo "[ revert ] Target: $TARGET_PATH"
echo "[ revert ] Flags:  ${EXTRA_FLAGS:-none}"
echo ""

# ─── Run ─────────────────────────────────────────────────────────────────────
SCRIPT="$REPO/logs_in_develop/API_UPDATE/Tools/revert_legacy_checkpoints.py"
python "$SCRIPT" --path "$TARGET_PATH" $EXTRA_FLAGS

echo ""
echo "Job completed successfully."
