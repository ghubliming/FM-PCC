#!/bin/bash
#SBATCH --job-name=visual_aligning_pipeline_fm
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
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# ==============================================================================
# Visual Aligning Flow Matching Pipeline Master Script
# ==============================================================================
SBATCH_DIR="Slurm_Codes/sbatch/Visual_Aligning"

echo "Launching Visual Aligning Flow Matching Pipeline..."

# 1. Submit Training Job
TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_visual_aligning_fm.sh")
echo "Step 1: Flow Matching Training submitted. Job ID: $TRAIN_ID"

# 2. Submit Evaluation Job (Success Dependency on Training)
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_visual_aligning_fm.sh")
echo "Step 2: Flow Matching Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

echo "--------------------------------------------------------------------------------"
echo "Visual Aligning Flow Matching Pipeline submitted successfully."
echo "Use 'squeue -u $USER' to monitor progress."
echo "================================================================================"
