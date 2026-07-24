#!/bin/bash
#SBATCH --job-name=af_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "PIPELINE START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "================================================================================"

# Trap for PIPELINE END
function on_exit {
    echo "================================================================================"
    echo "PIPELINE END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Smart Unified Session)
# ------------------------------------------------------------------------------
# Inherit session metadata from submit.sh or fallback to current local time
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# Sub-jobs share the SAME timestamp as the pipeline manager for perfect grouping
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# ==============================================================================
# AlphaFlow (Gen3v7) Pipeline Master Script
# ==============================================================================
# Chains two Slurm jobs:
# 1. Training  (train_alphaflow.sh)
# 2. Evaluation (eval_alphaflow.sh) — only if training succeeds
# ==============================================================================

SBATCH_DIR="Slurm_Codes/sbatch/AlphaFlow"

echo "Launching AlphaFlow Pipeline..."

# 1. Submit Training Job
TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_alphaflow.sh")
echo "Step 1: Training submitted. Job ID: $TRAIN_ID"

# 2. Submit Evaluation Job (success dependency on training)
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_alphaflow.sh")
echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

echo "--------------------------------------------------------------------------------"
echo "AlphaFlow Pipeline submitted successfully."
echo "Use 'squeue -u $USER' to monitor progress."
echo "If training fails, evaluation will be cancelled automatically by Slurm."
echo "================================================================================"
