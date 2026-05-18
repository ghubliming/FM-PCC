#!/bin/bash
#SBATCH --job-name=visual_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student

set -e

# ─── Job Metadata ───────────────────────────────────────────────────────
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU info available)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# ─── Environment Setup ──────────────────────────────────────────────────
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

# Headless rendering
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"

cd "$REPO"

# ─── Run Evaluation ─────────────────────────────────────────────────────
# Uses config/visual_aligning_eval.yaml for seed/variant configuration.
# Model is loaded from the FM-PCC pickle config system (same as training).
# Results are saved to: logs/aligning-d3il-visual/plans/ddpm_encdec_vision/H8/<seed>/results/

# Optional: override seed via command line argument
# Run evaluation
# Args: $1=seed (optional), $2=record_mode (optional, default=all)
SEED_ARG=""
if [ -n "$1" ]; then
    SEED_ARG="--seed $1"
    echo "[ eval ] Overriding seed to: $1"
fi

RECORD_MODE="${2:-all}"
echo "[ eval ] Recording mode set to: $RECORD_MODE"

python ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py $SEED_ARG --record "$RECORD_MODE"

echo "Job completed successfully."
