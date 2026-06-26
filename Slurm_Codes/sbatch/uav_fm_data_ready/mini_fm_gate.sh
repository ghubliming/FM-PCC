#!/bin/bash
#SBATCH --job-name=uav_fm_mini_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 Aux-1 — OPTIONAL mini-FM sanity gate (GPU). Trains a tiny FM on the
# curated data and checks held-out RMS, catching action-convention/shape bugs
# before a full train run. Run AFTER prepare_uav_fm_data.sh.
#
# Usage:
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/mini_fm_gate.sh           # empty scene
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/mini_fm_gate.sh pillars 100 15000 0.3
# Args: $1=scene [empty]  $2=n_episodes [100]  $3=n_steps [15000]  $4=lr [0.3]
# n_steps/lr defaults mirror mini_fm_sanity.py's own argparse defaults — keep
# them in sync (analytic-backprop trainer needs ~15000 steps @ lr=0.3 to clear
# the RMS gate; the old 1000/1e-3 was tuned for the removed slow finite-diff
# trainer and silently overrides the script's own defaults if left here).
set -e

SCENE="${1:-empty}"
N_EP="${2:-100}"
N_STEPS="${3:-15000}"
LR="${4:-0.3}"

echo "================================================================================"
echo "UAV-FM MINI GATE (Aux-1)  $(date)  | JOB $SLURM_JOB_ID | NODE $(hostname)"
echo "scene=$SCENE  n_episodes=$N_EP  n_steps=$N_STEPS  lr=$LR"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

MARKER="d3il/environments/d3il/models/mj/robot/quadrotor/scenes"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do REPO="$(dirname "$REPO")"; done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
[ -d "$REPO/$MARKER" ] || { echo "[ mini-gate ] ERROR: repo root not found."; exit 1; }
cd "$REPO"
source activate FMPCC 2>/dev/null || conda activate FMPCC
export MUJOCO_GL="egl"; export MPLBACKEND="agg"

echo "[ mini-gate ] python uav_expert_data_collect/mini_fm_sanity.py --data-dir data/uav_fm/v1/$SCENE --n-steps $N_STEPS --lr $LR"
python uav_expert_data_collect/mini_fm_sanity.py \
    --data-dir "data/uav_fm/v1/$SCENE" --n-episodes "$N_EP" --n-steps "$N_STEPS" --lr "$LR"
echo "[ mini-gate ] done — review held-out RMS in the log (< 0.1 m = pass)."
