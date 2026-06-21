#!/bin/bash
#SBATCH --job-name=uav_fm_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
set -e

# One-shot: train → eval (eval runs only if train succeeds).
# Args: $1=scene (def all)  $2=seed (def 5)  $3=n_trials (def 20)
SCENE="${1:-all}"
SEED="${2:-5}"
NTRIALS="${3:-20}"
SBATCH_DIR="Slurm_Codes/sbatch/uav_fm"

echo "================================================================================"
echo "UAV-FM PIPELINE START: $(date)  scene=$SCENE seed=$SEED n_trials=$NTRIALS"
echo "================================================================================"

# Unify dated logs across the chained jobs (submit.sh exports SUBMIT_DATE/TIME).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_fm_uav.sh" "$SCENE" "$SEED")
echo "Step 1: train submitted — Job $TRAIN_ID"

EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${TRAIN_ID} \
    "${SBATCH_DIR}/eval_fm_uav.sh" "$SCENE" "$SEED" "$NTRIALS")
echo "Step 2: eval scheduled (afterok:${TRAIN_ID}) — Job $EVAL_ID"
echo "--------------------------------------------------------------------------------"
echo "Outputs: logs/uav-${SCENE}/flow_matching_v3_uav/.../${SEED}/{weights,eval}"
echo "================================================================================"
