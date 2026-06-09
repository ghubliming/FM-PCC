#!/bin/bash
#SBATCH --job-name=uav_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student
#
# Epoch 4 expert data collection — headless MuJoCo PID rollouts (no GPU needed).
# Runs collect.py then stats_validator.py so the full USAGE.md quick-start
# executes entirely on the cluster without needing a local Python runtime.
#
# Usage patterns (submit from repo root via submit.sh):
#   # Smoke test (10 trials, empty scene, validate output)
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 10
#
#   # Full collection — one job per scene
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
#
#   # Single scene, explicit gain
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500 pid_high_gain 1000
#
# Args:
#   $1 = scene       (empty|corridor|s_curve|pillars|all_scenes)  [default: empty]
#   $2 = n_trials    [default: 200]
#   $3 = gain        (pid_default|pid_high_gain|pid_low_gain)      [default: pid_default]
#   $4 = seed_offset (added to SLURM_ARRAY_TASK_ID * 10000)        [default: 0]

set -e

echo "========================================"
echo "UAV EXPERT DATA COLLECTION (Epoch 4)"
echo "DATE:     $(date)"
echo "NODE:     $(hostname)"
echo "JOB_ID:   $SLURM_JOB_ID"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "========================================"

function on_exit {
    echo "========================================"
    echo "JOB END: $(date)"
    echo "========================================"
}
trap on_exit EXIT

SCENE_ARG="${1:-empty}"
N_TRIALS="${2:-200}"
GAIN="${3:-pid_default}"
SEED_OFFSET="${4:-0}"

# Resolve repo root the same way other sbatch scripts do.
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

# No GPU allocated — disable MuJoCo's GL backend entirely.
# MUJOCO_GL=disabled skips the entire gl_context import block in mujoco 2.3.7
# (gl_context.py:25), so neither EGL nor osmesa is loaded.
# Without this, mujoco falls back to osmesa which isn't installed → AttributeError.
# With MUJOCO_GL=egl, mujoco/egl/__init__.py:65 calls eglInitialize() at import
# time → opens a GPU device even with no rendering → GPU leak. "disabled" avoids both.
export MUJOCO_GL="disabled"

source activate FMPCC 2>/dev/null || conda activate FMPCC

# [DEBUG] GPU-leak check — uncomment once to verify MUJOCO_GL=disabled opens no DRM device.
# Expected output: "DRI fds: NONE — clean". Remove after verification.
# python3 -c "import os,mujoco; import subprocess; r=subprocess.run(['lsof',f'/proc/{os.getpid()}/fd'],capture_output=True,text=True); dri=[l for l in r.stdout.splitlines() if 'dri' in l.lower() or 'renderD' in l]; print('[ GPU-LEAK CHECK ] DRI fds:', dri or 'NONE — clean')"

SEED=$(( SEED_OFFSET ))
SCENE="$SCENE_ARG"

echo "[ sbatch ] scene=$SCENE  n_trials=$N_TRIALS  gain=$GAIN  seed=$SEED"

python uav_expert_data_collect/collect.py \
    --scene        "$SCENE" \
    --n-trials     "$N_TRIALS" \
    --gain-variant "$GAIN" \
    --seed         "$SEED" \
    --homotopy     all \
    --noise-sigma  0.02

echo "[ sbatch ] Collection done. Running stats validator …"

python uav_expert_data_collect/stats_validator.py \
    --data-dir "logs/uav_expert_data/$SCENE"

echo "[ sbatch ] Output: logs/uav_expert_data/$SCENE/"
