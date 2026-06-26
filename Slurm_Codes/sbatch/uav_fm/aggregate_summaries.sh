#!/bin/bash
#SBATCH --job-name=uav_fm_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — roll up per-(scene,seed) eval results → per-scene SCENE_SUMMARY.json
# + cross-scene fm_uav_ALL_SCENES_SUMMARY.json. Pure stdlib (no GPU/torch).
#
# Usage:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh "empty corridor s_curve pillars" fm_only
# Args: $1=scenes [all 4]   $2=projection [fm_only]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
PROJ="${2:-fm_only}"

echo "[ aggregate ] $(date)  scenes=[$SCENES]  proj=$PROJ"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

MARKER="d3il/environments/d3il/models/mj/robot/quadrotor/scenes"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO="$SLURM_SUBMIT_DIR"
    while [ "$REPO" != "/" ] && [ ! -d "$REPO/$MARKER" ]; do REPO="$(dirname "$REPO")"; done
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
[ -d "$REPO/$MARKER" ] || { echo "[ aggregate ] ERROR: repo root not found."; exit 1; }
cd "$REPO"
source activate FMPCC 2>/dev/null || conda activate FMPCC

python FM_v3_uav_test/aggregate_scene_summaries.py --scenes $SCENES --projection "$PROJ"
echo "[ aggregate ] done."
