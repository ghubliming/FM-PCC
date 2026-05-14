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

# Logging setup
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "JOB START: $(date)"

# Setup Workspace Paths
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

# ─────────────────────────────────────────────────────────────────────
# Configuration: Set the path to your trained D3IL checkpoint directory
# This should contain eval_best_ddpm.pth (or last_ddpm.pth)
# ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR="${1:-$REPO/d3il/logs/aligning/runs/ddpm_encdec_vision}"
SEED="${2:-42}"
N_CONTEXTS="${3:-30}"
N_TRAJECTORIES="${4:-1}"

echo "[ eval ] Weights dir: $WEIGHTS_DIR"
echo "[ eval ] Seed: $SEED"
echo "[ eval ] Contexts: $N_CONTEXTS | Trajectories/ctx: $N_TRAJECTORIES"

# Run evaluation using D3IL's native agent infrastructure
python ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py \
    --weights-dir "$WEIGHTS_DIR" \
    --seed "$SEED" \
    --n-contexts "$N_CONTEXTS" \
    --n-trajectories "$N_TRAJECTORIES" \
    --n-cores 1 \
    --device cuda

echo "Job completed successfully."
