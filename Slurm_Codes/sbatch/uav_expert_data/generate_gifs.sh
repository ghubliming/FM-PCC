#!/bin/bash
#SBATCH --job-name=uav_gen_gifs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1
#
# Epoch 5 WS-B — GIF/video generation from Epoch 4 state pickles.
# GPU required for EGL offscreen MuJoCo rendering.
#
# Usage (submit from repo root):
#   # Full generation — all scenes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh
#
#   # Smoke test — 5 episodes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh 5
#
#   # Single scene, with MP4
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" corridor mp4
#
# Args:
#   $1 = max_episodes    (default: all — leave blank for full run)
#   $2 = scene           (default: all — leave blank for all scenes)
#   $3 = "mp4"           (pass "mp4" to also generate MP4 files; default: GIF only)
#   $4 = frame_stride    (default: 1 — every frame; use 3 for smaller GIFs)
#   $5 = per_homotopy    (default: all — pass 1 for one GIF per homotopy bucket, 9 total)

set -e

echo "========================================"
echo "UAV TRAJECTORY GIF GENERATION (Epoch 5 WS-B)"
echo "DATE:     $(date)"
echo "NODE:     $(hostname)"
echo "JOB_ID:   $SLURM_JOB_ID"
echo "========================================"

function on_exit {
    echo "========================================"
    echo "JOB END: $(date)"
    echo "========================================"
}
trap on_exit EXIT

MAX_EP="${1:-}"
SCENE="${2:-}"
MP4_FLAG="${3:-}"
STRIDE="${4:-1}"
PER_HOMO="${5:-}"

# Resolve repo root.
MARKER="d3il/environments/d3il/models/mj/robot/quadrotor/scenes"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do
        REPO="$(dirname "$REPO")"
    done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

if [ ! -d "$REPO/$MARKER" ]; then
    echo "[ sbatch ] ERROR: could not locate FM-PCC repo root."
    exit 1
fi

cd "$REPO"
echo "[ sbatch ] Repo: $REPO"

# EGL offscreen rendering.
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
echo "[ sbatch ] MaxEp: ${MAX_EP:-all}  Scene: ${SCENE:-all}  MP4: ${MP4_FLAG:-no}  Stride: ${STRIDE}  PerHomo: ${PER_HOMO:-all}"

# Build args
SCRIPT_ARGS="--frame-stride $STRIDE"
if [ -n "$MAX_EP" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --max-episodes $MAX_EP"
fi
if [ -n "$SCENE" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --scene $SCENE"
fi
if [ "$MP4_FLAG" = "mp4" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --mp4"
fi
if [ -n "$PER_HOMO" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --per-homotopy $PER_HOMO"
fi

python uav_expert_data_collect/generate_trajectory_gifs.py $SCRIPT_ARGS

echo "[ sbatch ] GIF generation complete."
echo "Output: logs/uav_expert_data/gifs/"
