#!/bin/bash
#SBATCH --job-name=uav_cam_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1
#
# Epoch 5 WS-A — Camera image collection from Epoch 4 state pickles.
# GPU required for EGL offscreen MuJoCo rendering.
#
# Usage (submit from repo root):
#   # Full collection — all scenes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh
#
#   # Smoke test — 5 episodes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh 5
#
#   # Single scene
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh "" corridor
#
# Args:
#   $1 = max_episodes  (default: all — leave blank for full run)
#   $2 = scene         (default: all — leave blank for all scenes)
#   $3 = resolution    (default: 96)

set -e

echo "========================================"
echo "UAV CAMERA IMAGE COLLECTION (Epoch 5 WS-A)"
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
RESOLUTION="${3:-96}"

# Resolve repo root.
MARKER="d3il/environments/d3il/models/mj/robot/quadrotor/scenes"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do
        REPO="$(dirname "$REPO")"
    done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}") /../../.." && pwd)"
fi

if [ ! -d "$REPO/$MARKER" ]; then
    echo "[ sbatch ] ERROR: could not locate FM-PCC repo root."
    exit 1
fi

cd "$REPO"
echo "[ sbatch ] Repo: $REPO"

# EGL offscreen rendering — must be set before MuJoCo is imported.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"
echo "[ sbatch ] Resolution: ${RESOLUTION}  MaxEp: ${MAX_EP:-all}  Scene: ${SCENE:-all}"

# Build args
SCRIPT_ARGS="--resolution $RESOLUTION"
if [ -n "$MAX_EP" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --max-episodes $MAX_EP"
fi
if [ -n "$SCENE" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --scene $SCENE"
fi

python uav_expert_data_collect/collect_camera_images.py $SCRIPT_ARGS

echo "[ sbatch ] Camera collection complete."
echo "Output: logs/uav_expert_data/images/"
