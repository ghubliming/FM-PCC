#!/bin/bash
#SBATCH --job-name=uav_fm_train_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — outer loop: train ONE FM per (scene, seed). Thin submitter; exits immediately.
# Per-scene models (state-only FM cannot tell scenes apart → one model per scene). See U2 PLAN.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh "pillars" "5 6 7"
# Args: $1=scenes (quoted, space-sep) [all 4]   $2=seeds (quoted) ["5 6 7"]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
SEEDS="${2:-5 6 7}"
JOB="Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh"

echo "================================================================================"
echo "UAV-FM TRAIN ALL (per-scene)  $(date)   scenes=[$SCENES]  seeds=[$SEEDS]"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

n=0
for scene in $SCENES; do
    for seed in $SEEDS; do
        ID=$(sbatch --parsable $LOG_OPTS "$JOB" "$scene" "$seed")
        echo "  train  scene=$scene seed=$seed  → Job $ID"
        n=$((n+1))
    done
done
echo "--------------------------------------------------------------------------------"
echo "Submitted $n train jobs. Outputs: logs/uav-<scene>/flow_matching_v3_uav.../<seed>/weights/"
echo "Then eval:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh \"$SCENES\" \"$SEEDS\""
echo "================================================================================"
