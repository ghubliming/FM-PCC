#!/bin/bash
#SBATCH --job-name=uav_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student
#
# E4 U10 — stress-test episode collection (CPU-only, no GPU).
# Generates deliberately bad episodes into logs/uav_expert_data_stress/.
# Never touches the production data folder.
#
# Usage (submit from repo root):
#   # All cases, all applicable scenes (~50-60 episodes, ~5 min)
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh all
#
#   # Specific cases
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh wall_crossing,floor_dive
#
#   # One scene
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh tight_fillet pillars
#
# Args:
#   $1 = cases      ("all" or comma-separated list)   [REQUIRED — no default]
#   $2 = scene      (empty|corridor|s_curve|pillars|"")  [default: all applicable]
#   $3 = n_per_case (default: 3)
#   $4 = seed       (default: 0)

set -e

echo "========================================"
echo "UAV STRESS-TEST COLLECTION (E4 U10)"
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

CASES="${1:-}"
SCENE="${2:-}"
N_PER_CASE="${3:-3}"
SEED="${4:-0}"

if [ -z "$CASES" ]; then
    echo "[ sbatch ] ERROR: \$1 (cases) is required. Pass 'all' or a comma-separated list."
    exit 1
fi

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

export MUJOCO_GL="disabled"

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"
echo "[ sbatch ] Cases: ${CASES}  Scene: ${SCENE:-all}  N_per_case: ${N_PER_CASE}  Seed: ${SEED}"

SCRIPT_ARGS="--cases $CASES --n-per-case $N_PER_CASE --seed $SEED"
if [ -n "$SCENE" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --scene $SCENE"
fi

python uav_expert_data_collect/collect_stress.py $SCRIPT_ARGS

echo "[ sbatch ] Stress collection complete."
echo "[ sbatch ] Output: logs/uav_expert_data_stress/"
