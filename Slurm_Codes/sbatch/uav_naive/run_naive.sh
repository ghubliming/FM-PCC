#!/bin/bash
#SBATCH --job-name=uav_naive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-1-student
#SBATCH --gres=gpu:1

# Epoch-2 naive UAV fly test: runs smoke-load, hover, step, or circle
# tracking under a cascaded PID. EGL offscreen rendering required for GIFs.
#
# Args:
#   $1 = task   (smoke | A | B | C | all)         default: smoke
#   $2 = format (6D | 9D, only used when task=C)  default: 9D
#
# Examples:
#   sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh smoke
#   sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh A
#   sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 6D
#   sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh all

set -e

echo "========================================"
echo "UAV NAIVE FLY TEST"
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

TASK="${1:-smoke}"
FMT="${2:-9D}"

# ─── Resolve repo root (SLURM stages script under /var/lib/slurmd/...) ────
MARKER="d3il/environments/d3il/models/mj/robot/quadrotor"
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
    echo "  SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
    echo "  Resolved REPO=$REPO  (missing marker '$MARKER')"
    echo "  Submit from the repo root, e.g.:"
    echo "    cd /path/to/FM-PCC && sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh smoke"
    exit 1
fi

cd "$REPO"
echo "[ sbatch ] Repo: $REPO   Task: $TASK   Format: $FMT"

# ─── Headless rendering ──────────────────────────────────────────────────
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"

# ─── Dispatch ────────────────────────────────────────────────────────────
case "$TASK" in
    smoke)
        python uav_naive_test/smoke_load.py
        ;;
    A|B)
        python uav_naive_test/run_naive.py --task "$TASK" --render
        ;;
    C)
        python uav_naive_test/run_naive.py --task C --trajectory-format "$FMT" --render
        ;;
    all)
        python uav_naive_test/run_naive.py --task A --render
        python uav_naive_test/run_naive.py --task B --render
        python uav_naive_test/run_naive.py --task C --trajectory-format 6D --render
        python uav_naive_test/run_naive.py --task C --trajectory-format 9D --render
        ;;
    *)
        echo "[ sbatch ] ERROR: unknown task '$TASK' (expected smoke | A | B | C | all)"
        exit 1
        ;;
esac

echo "[ sbatch ] Output (if applicable): logs_in_develop/Gen11/Epoch2_env/results/"
