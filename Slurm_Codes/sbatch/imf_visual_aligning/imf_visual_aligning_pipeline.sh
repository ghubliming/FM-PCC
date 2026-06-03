#!/bin/bash

# ==============================================================================
# Visual-iMF (Gen8) Pipeline Master Script
# ==============================================================================
# Chains two Slurm jobs:
# 1. Training  (train_imf_visual_aligning.sh)
# 2. Evaluation (eval_imf_visual_aligning.sh) — only if training succeeds (afterok)
# ==============================================================================

set -e

function on_exit {
    echo "================================================================================"
    echo "JOB END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# ─── Logging Configuration (Smart Unified Session) ──────────────────────────
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# Sub-jobs share the SAME timestamp as the pipeline manager for perfect grouping
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

SBATCH_DIR="Slurm_Codes/sbatch/imf_visual_aligning"

echo "================================================================================"
echo "Launching Visual-iMF (Gen8) Pipeline..."
echo "Date:      $DATE"
echo "Time tag:  $TIME"
echo "Log dir:   $LOG_DIR"
echo "================================================================================"

# 1. Submit Training Job
TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_imf_visual_aligning.sh")
echo "Step 1: Training submitted. Job ID: $TRAIN_ID"

# 2. Submit Evaluation Job (success dependency on training)
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_imf_visual_aligning.sh")
echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

echo "--------------------------------------------------------------------------------"
echo "Visual-iMF (Gen8) Pipeline submitted successfully."
echo "Use 'squeue -u \$USER' to monitor progress."
echo "If training fails, evaluation will be cancelled automatically by Slurm."
echo "================================================================================"
