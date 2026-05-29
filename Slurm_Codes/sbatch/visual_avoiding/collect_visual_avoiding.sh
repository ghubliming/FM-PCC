#!/bin/bash
#SBATCH --job-name=collect_visual_avoiding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1

# GPU required for EGL offscreen MuJoCo rendering (no interactive display needed).
# MUJOCO_GL=egl is set below to force the EGL backend.

set -e

echo "========================================"
echo "COLLECT VISUAL AVOIDING CAMERA DATA"
echo "DATE:   $(date)"
echo "NODE:   $(hostname)"
echo "JOB_ID: $SLURM_JOB_ID"
echo "========================================"

function on_exit {
    echo "========================================"
    echo "JOB END: $(date)"
    echo "========================================"
}
trap on_exit EXIT

# ─── Args ────────────────────────────────────────────────────────────────────
# $1 = resolution    (default: 96)
# $2 = max_episodes  (default: all — leave blank for full run)
# $3 = train_ratio   (default: 0.9)
#
# Examples:
#   # Full collection at 96×96
#   sbatch collect_visual_avoiding.sh
#
#   # Smoke: 10 episodes
#   sbatch collect_visual_avoiding.sh 96 10
#
#   # Custom split
#   sbatch collect_visual_avoiding.sh 96 "" 0.8

RESOLUTION="${1:-96}"
MAX_EP="${2:-}"
TRAIN_RATIO="${3:-0.9}"

# ─── Logging ─────────────────────────────────────────────────────────────────
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# ─── Environment ─────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
echo "[ sbatch ] Repo root: $REPO"

# EGL offscreen rendering — must be set before MuJoCo is imported
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"
echo "[ sbatch ] Resolution: ${RESOLUTION}  MaxEp: ${MAX_EP:-all}  TrainRatio: ${TRAIN_RATIO}"

# ─── Build args ──────────────────────────────────────────────────────────────
SCRIPT_ARGS="--resolution $RESOLUTION --train-ratio $TRAIN_RATIO"
if [ -n "$MAX_EP" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --max-episodes $MAX_EP"
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
python collect_visual_avoiding_data/collect_visual_avoiding_data.py $SCRIPT_ARGS

echo "[ sbatch ] Collection complete."
echo "Output: d3il/environments/dataset/data/avoiding/all_data/"
