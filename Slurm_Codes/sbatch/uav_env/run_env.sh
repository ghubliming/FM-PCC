#!/bin/bash
#SBATCH --job-name=uav_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1

# Epoch-3 UAV in env: runs smoke-load OR a (scene, task) pair under the
# Epoch-2 cascaded PID. EGL offscreen rendering for GIFs.
#
# Args:
#   $1 = scene  (empty | corridor | s_curve | pillars | smoke_<scene> | all)
#   $2 = task   (A | B | C | traverse | s_curve | weave)   -- ignored if $1 starts with smoke_
#
# Examples:
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_empty
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh empty C
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh corridor traverse
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh s_curve s_curve
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh pillars weave
#   sbatch Slurm_Codes/sbatch/uav_env/run_env.sh all

set -e

echo "========================================"
echo "UAV IN ENV TEST"
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

SCENE="${1:-smoke_empty}"
TASK="${2:-C}"

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
    echo "  SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR  REPO=$REPO"
    exit 1
fi

cd "$REPO"
echo "[ sbatch ] Repo: $REPO   Scene: $SCENE   Task: $TASK"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"

source activate FMPCC 2>/dev/null || conda activate FMPCC

case "$SCENE" in
    smoke_empty|smoke_corridor|smoke_s_curve|smoke_pillars)
        S="${SCENE#smoke_}"
        python uav_env_test/smoke_load_env.py --scene "$S"
        ;;
    empty|corridor|s_curve|pillars)
        python uav_env_test/run_env.py --scene "$SCENE" --task "$TASK" --render
        ;;
    all)
        python uav_env_test/run_env.py --scene empty    --task C        --render
        python uav_env_test/run_env.py --scene corridor --task traverse --render
        python uav_env_test/run_env.py --scene s_curve  --task s_curve  --render
        python uav_env_test/run_env.py --scene pillars  --task weave    --render
        ;;
    *)
        echo "[ sbatch ] ERROR: unknown scene '$SCENE'"
        exit 1
        ;;
esac

echo "[ sbatch ] Output (if applicable): logs/uav_env/"
