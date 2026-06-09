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
# NOTE: SLURM copies the submitted script to a spool dir (e.g. /var/lib/slurmd/...),
# so ${BASH_SOURCE[0]} does NOT point at the repo. Use $SLURM_SUBMIT_DIR (the dir
# from which sbatch was invoked) and locate the repo root from there.
MARKER="collect_visual_avoiding_data"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    # Walk upward until we find the marker dir (or hit /)
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do
        REPO="$(dirname "$REPO")"
    done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

if [ ! -d "$REPO/$MARKER" ]; then
    echo "[ sbatch ] ERROR: could not locate FM-PCC repo root."
    echo "  SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    echo "  Resolved REPO=$REPO  (missing marker dir '$MARKER')"
    echo "  Submit from the repo root, e.g.:"
    echo "    cd /path/to/FM-PCC && sbatch Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh"
    exit 1
fi

cd "$REPO"
echo "[ sbatch ] Repo root: $REPO"

# EGL offscreen rendering — must be set before MuJoCo is imported.
# MuJoCo's egl backend also gates on PYOPENGL_PLATFORM: it must be unset or 'egl'.
# Some cluster shells preset it to 'glx'/'osmesa', which trips ImportError, so pin it.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

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
