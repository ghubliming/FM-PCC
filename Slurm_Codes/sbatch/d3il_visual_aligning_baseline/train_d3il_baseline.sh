#!/bin/bash
#SBATCH --job-name=train_d3il_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

# Optional W&B login
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/FMPCC/.wandb_api_key")
fi

cd "$REPO"

# ─── Args ───────────────────────────────────────────────────────────────────
# $1 = agent_name  (default: ddpm_encdec_vision)
# $2 = seed        (default: 42)
# $3 = epoch       (default: paper setting — 200 for vision, 500 for state)
#
# Vision agents (*_vision): use aligning_vision_config  → Aligning_Img_Dataset, obs_dim=3
# State agents (no _vision): use aligning_config        → Aligning_Dataset,     obs_dim=20
#
# D3IL paper (ICLR 2024): 200 epochs for image agents, 500 for state agents;
# eval every 1/10th of total training; best val-loss checkpoint saved.
#
# Outputs land in: logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights/
# (already gitignored via root .gitignore  logs/*)

AGENT_NAME="${1:-ddpm_encdec_vision}"
SEED="${2:-42}"

# Auto-select base config: vision agents need the image dataset config
if [[ "$AGENT_NAME" == *"_vision"* ]]; then
    CONFIG_NAME="aligning_vision_config"
    DEFAULT_EPOCH=200
else
    CONFIG_NAME="aligning_config"
    DEFAULT_EPOCH=500
fi

EPOCH="${3:-$DEFAULT_EPOCH}"
EVAL_EVERY=$(( EPOCH / 10 ))   # eval every 1/10th of training (paper setting)
# guard: eval_every must be at least 1
if [[ "$EVAL_EVERY" -lt 1 ]]; then EVAL_EVERY=1; fi

SAVE_DIR="logs/d3il_visual_aligning_baseline/${AGENT_NAME}/seed_${SEED}/weights"

echo "[ train ] agent=${AGENT_NAME}  seed=${SEED}  config=${CONFIG_NAME}"
echo "[ train ] epoch=${EPOCH}  eval_every=${EVAL_EVERY}"
echo "[ train ] weights will be saved to: ${REPO}/${SAVE_DIR}"

python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    --config-name "${CONFIG_NAME}" \
    "agents=${AGENT_NAME}_agent" \
    "agent_name=${AGENT_NAME}" \
    "seed=${SEED}" \
    "epoch=${EPOCH}" \
    "eval_every_n_epochs=${EVAL_EVERY}" \
    "hydra.run.dir=${SAVE_DIR}"

echo "Job completed successfully."
