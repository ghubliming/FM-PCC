#!/bin/bash
#SBATCH --job-name=eval_d3il_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student

set -e

# ─── Job Metadata ───────────────────────────────────────────────────────────
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

# ─── Environment Setup ──────────────────────────────────────────────────────
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
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

cd "$REPO"

# ─── Args ───────────────────────────────────────────────────────────────────
# $1 = agent_name   (default: ddpm_encdec_vision)
# $2 = seed         (optional; if blank → all seeds in d3il_eval_config.yaml)
# $3 = record_mode  (default: all)
# $4 = scale        (optional: "paper" → 60 ctx × 18 traj, faithful entropy; else config smoke)
#
# Examples:
#   sbatch eval_d3il_baseline.sh ddpm_encdec_vision 42
#   sbatch eval_d3il_baseline.sh ddpm_encdec_vision 42 gif
#   sbatch eval_d3il_baseline.sh ddpm_encdec_vision 42 none paper   # ← paper-faithful (entropy)
#   sbatch eval_d3il_baseline.sh ddpm_encdec_vision    # all seeds from config

AGENT_NAME="${1:-ddpm_encdec_vision}"
RECORD_MODE="${3:-all}"

AGENT_ARG="--agent-name ${AGENT_NAME}"

SEED_ARG=""
if [ -n "$2" ]; then
    SEED_ARG="--seed $2"
    echo "[ eval ] seed override: $2"
fi

# U2: paper-faithful eval scale (60 ctx × 18 traj) → meaningful behavior entropy.
PAPER_ARG=""
if [ "$4" = "paper" ]; then
    PAPER_ARG="--paper"
    echo "[ eval ] --paper preset: 60 contexts x 18 trajectories (faithful entropy)"
fi

echo "[ eval ] agent=${AGENT_NAME}  record_mode=${RECORD_MODE}"

python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py \
    $AGENT_ARG \
    $SEED_ARG \
    $PAPER_ARG \
    --record "${RECORD_MODE}"

echo "Job completed successfully."
