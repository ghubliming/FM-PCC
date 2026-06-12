#!/bin/bash
#SBATCH --job-name=uav_phys_gifs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1
#
# Epoch 5 WS-D (U3) — Physics replay GIF generation from Epoch 4 state pickles.
# GPU required: mujoco.Renderer uses EGL offscreen rendering (unlike collect.sh
# which is physics-only and uses MUJOCO_GL=disabled).
#
# Usage (submit from repo root):
#   # Smoke test — 3 pillar episodes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh 3 pillars
#
#   # Full run — all scenes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh
#
#   # Single scene with MP4
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh "" corridor mp4
#
# Args:
#   $1 = max_episodes    (default: all — leave blank for full run)
#   $2 = scene           (default: all — leave blank for all scenes)
#   $3 = "mp4"           (pass "mp4" to also save MP4 files; default: GIF only)
#   $4 = frame_stride    (default: 3 — every 3rd 33-Hz frame; use 1 for full rate)
#   $5 = per_homotopy    (default: all — pass 1 for one GIF per homotopy bucket, 9 total)
#   $6 = data_dir        (default: logs/uav_expert_data — pass stress root for E5 U4)

set -e

echo "========================================"
echo "UAV PHYSICS REPLAY GIF GENERATION (Epoch 5 WS-D)"
echo "DATE:     $(date)"
echo "NODE:     $(hostname)"
echo "JOB_ID:   $SLURM_JOB_ID"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
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
STRIDE="${4:-3}"
PER_HOMO="${5:-}"
DATA_DIR="${6:-}"

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

# EGL offscreen rendering — mujoco.Renderer IS used here (unlike collect.sh).
# MUJOCO_GL=disabled would crash at Renderer() creation.
# 3-line GPU pinning prevents EGL defaulting to /dev/dri/renderD128 (GPU 0).
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

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"
echo "[ sbatch ] MaxEp: ${MAX_EP:-all}  Scene: ${SCENE:-all}  MP4: ${MP4_FLAG:-no}  Stride: ${STRIDE}  PerHomo: ${PER_HOMO:-all}  DataDir: ${DATA_DIR:-default}"
echo "[ sbatch ] EGL device: $MUJOCO_EGL_DEVICE_ID"

# [DEBUG] EGL device-pinning check — uncomment once, submit, verify output, re-comment.
# Creates a minimal Renderer to trigger EGL init, then reads /proc/{pid}/fd (no lsof needed)
# to find which renderD device was opened.
# Expected output:
#   [ EGL-CHECK ] EGL_DEVICE=<N>  CUDA=<N>
#   [ EGL-CHECK ] renderD fds open (after Renderer): ['/dev/dri/renderD<128+N>']
# If CUDA_VISIBLE_DEVICES=2, MUJOCO_EGL_DEVICE_ID=2, expect renderD130 (not renderD128).
# python3 -c "import os,subprocess,mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>'); r=mujoco.Renderer(m,32,32); pid=os.getpid(); res=subprocess.run(['ls','-la',f'/proc/{pid}/fd'],capture_output=True,text=True); dri=[l.strip() for l in res.stdout.splitlines() if 'renderD' in l]; print('[ EGL-CHECK ] EGL_DEVICE='+os.environ.get('MUJOCO_EGL_DEVICE_ID','NOT SET')+'  CUDA='+os.environ.get('CUDA_VISIBLE_DEVICES','NOT SET')); print('[ EGL-CHECK ] renderD fds open (after Renderer):',dri if dri else 'NONE — EGL failed?'); (r.close() if hasattr(r,'close') else None)"

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
if [ -n "$DATA_DIR" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --data-dir $DATA_DIR"
fi

python uav_expert_data_collect/generate_physics_gifs.py $SCRIPT_ARGS

echo "[ sbatch ] Physics GIF generation complete."
echo "[ sbatch ] Output: logs/uav_expert_data/gifs_physics/"
