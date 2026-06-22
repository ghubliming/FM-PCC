#!/bin/bash
#SBATCH --job-name=uav_fm_eval_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — outer loop: eval ONE FM per (scene, seed), then a final aggregate job
# (after all evals) rolls up per-scene + cross-scene summaries. Thin submitter; exits immediately.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh "pillars" "5 6 7" 20 fm_only
# Args: $1=scenes [all 4]  $2=seeds ["5 6 7"]  $3=n_trials [20]  $4=projection [fm_only]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
SEEDS="${2:-5 6 7}"
NTRIALS="${3:-20}"
PROJ="${4:-fm_only}"
EVAL="Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh"
AGG="Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh"

echo "================================================================================"
echo "UAV-FM EVAL ALL (per-scene)  $(date)   scenes=[$SCENES]  seeds=[$SEEDS]  n_trials=$NTRIALS  proj=$PROJ"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

DEP="afterok"
for scene in $SCENES; do
    for seed in $SEEDS; do
        ID=$(sbatch --parsable $LOG_OPTS "$EVAL" "$scene" "$seed" "$NTRIALS" "$PROJ")
        echo "  eval   scene=$scene seed=$seed  → Job $ID"
        DEP="${DEP}:${ID}"
    done
done

# Final roll-up after all evals succeed.
AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$DEP" "$AGG" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
echo "Read: logs/uav-<scene>/SCENE_SUMMARY.json  +  logs/fm_uav_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
