#!/bin/bash
#SBATCH --job-name=uav_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1
#
# Epoch 4 expert data collection — headless MuJoCo PID rollouts (no GPU needed).
# Runs collect.py then stats_validator.py so the full USAGE.md quick-start
# executes entirely on the cluster without needing a local Python runtime.
#
# Usage patterns (submit from repo root):
#   # Step 1+2 — smoke test (10 trials, empty scene, validate output)
#   sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 10
#
#   # Step 3+4 — full collection, all scenes in parallel
#   sbatch --array=0-3 Slurm_Codes/sbatch/uav_expert_data/collect.sh all_scenes 500
#
#   # Single scene, explicit gain
#   sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500 pid_default 0
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

# No GPU needed for headless MuJoCo rollouts, but set EGL in case render is added.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

source activate FMPCC 2>/dev/null || conda activate FMPCC

# Seed = offset + (array_id * 10000) so parallel array tasks don't overlap.
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( SEED_OFFSET + ARRAY_ID * 10000 ))

# Map array task id to scene when running all_scenes mode.
ALL_SCENES=("empty" "corridor" "s_curve" "pillars")

if [ "$SCENE_ARG" = "all_scenes" ]; then
    SCENE="${ALL_SCENES[$ARRAY_ID % 4]}"
else
    SCENE="$SCENE_ARG"
fi

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
