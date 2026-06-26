#!/bin/bash
#SBATCH --job-name=uav_plots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student
#
# E5 U5 — 2-D overhead trajectory overview plots.
# CPU-only: pure matplotlib, no MuJoCo, no GPU.
#
# Usage (submit from repo root):
#   # Scene summaries — all production episodes
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh
#
#   # Summaries, one specific scene
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh "" pillars
#
#   # Per-episode, 1 per homotopy (9 plots total), production data
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh per-episode "" 1
#
#   # Both modes, stress root, 1 per homotopy
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh \
#       both "" 1 logs/uav_expert_data_stress
#
# Args:
#   $1 = mode          (summary|per-episode|both)   [default: summary]
#   $2 = scene         (empty|corridor|s_curve|pillars|"")  [default: all]
#   $3 = per_homotopy  (int or "")                  [default: all]
#   $4 = data_dir      (path or "")                 [default: logs/uav_expert_data]
#   $5 = dpi           (int)                        [default: 150]

set -e

echo "========================================"
echo "UAV OVERVIEW PLOT GENERATION (E5 U5)"
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

MODE="${1:-summary}"
SCENE="${2:-}"
PER_HOMO="${3:-}"
DATA_DIR="${4:-}"
DPI="${5:-150}"

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

# Pure matplotlib — no GL backend needed.
export MUJOCO_GL="disabled"
export MPLBACKEND="agg"

source activate FMPCC 2>/dev/null || conda activate FMPCC

echo "[ sbatch ] Python: $(which python)"
echo "[ sbatch ] Mode: ${MODE}  Scene: ${SCENE:-all}  PerHomo: ${PER_HOMO:-all}  DataDir: ${DATA_DIR:-default}  DPI: ${DPI}"

SCRIPT_ARGS="--mode $MODE --dpi $DPI"
if [ -n "$SCENE" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --scene $SCENE"
fi
if [ -n "$PER_HOMO" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --per-homotopy $PER_HOMO"
fi
if [ -n "$DATA_DIR" ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --data-dir $DATA_DIR"
fi

python uav_expert_data_collect/generate_overview_plots.py $SCRIPT_ARGS

echo "[ sbatch ] Plot generation complete."
if [ -n "$DATA_DIR" ]; then
    echo "[ sbatch ] Output: ${DATA_DIR}/plots/"
else
    echo "[ sbatch ] Output: logs/uav_expert_data/plots/"
fi
