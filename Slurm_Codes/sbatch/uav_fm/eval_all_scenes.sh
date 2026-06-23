#!/bin/bash
#SBATCH --job-name=uav_fm_eval_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — outer loop: ONE eval job PER SCENE (seeds loop internally inside
# eval_fm_uav.sh, not as separate sbatch calls), then a final aggregate job (after
# all evals) rolls up per-scene + cross-scene summaries. Thin submitter; exits immediately.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh "pillars" "6 7 8 9 10" 20 fm_only
# Args: $1=scenes [all 4]  $2=seeds ["6"]  $3=n_trials [20]  $4=projection [fm_only]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10" or uncomment below.
SEEDS="${2:-6}"
# SEEDS="${2:-6 7 8 9 10}"   # full run (5 seeds)
NTRIALS="${3:-20}"
PROJ="${4:-fm_only}"
EVAL="Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh"
AGG="Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh"

N_SEEDS=$(echo $SEEDS | wc -w)
EVAL_HOURS=$((N_SEEDS * 8))

echo "================================================================================"
echo "UAV-FM EVAL ALL (per-scene, seed-loop internal)  $(date)   scenes=[$SCENES]  seeds=[$SEEDS]  n_trials=$NTRIALS  proj=$PROJ"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

DEP="afterok"
for scene in $SCENES; do
    ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS "$EVAL" "$scene" "$SEEDS" "$NTRIALS" "$PROJ")
    echo "  eval   scene=$scene seeds=[$SEEDS]  → Job $ID"
    DEP="${DEP}:${ID}"
done

# Final roll-up after all evals succeed.
AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$DEP" "$AGG" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
echo "Read: logs/UAV_FM/uav-<scene>/SCENE_SUMMARY.json  +  logs/UAV_FM/fm_uav_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
