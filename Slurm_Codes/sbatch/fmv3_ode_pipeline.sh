#!/bin/bash
#SBATCH --job-name=fmv3_pipeline
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

# Sub-jobs will share the SAME timestamp as the pipeline manager for perfect grouping
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# ==============================================================================
# FMv3 ODE Pipeline Master Script
# ==============================================================================
# This script chains three Slurm jobs:
# 1. Training (train_fmv3_ode_job.sh)
# 2. Evaluation (eval_fmv3_ode_job.sh) - Only if Training succeeds
# 3. Load Results (load_results_fmv3_job.sh) - Only if Evaluation succeeds
# ==============================================================================

# Ensure we are in the right directory
SBATCH_DIR="Slurm_Codes/sbatch"

echo "Launching FMv3 ODE Pipeline..."

# 1. Submit Training Job
# --parsable makes sbatch only return the Job ID
TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_fmv3_ode_job.sh")
echo "Step 1: Training submitted. Job ID: $TRAIN_ID"

# 2. Submit Evaluation Job (Success Dependency on Training)
# afterok:jobid means it only runs if the previous job exits with status 0
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_fmv3_ode_job.sh")
echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

# 3. Submit Load Results Job (Success Dependency on Evaluation)
LOAD_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$EVAL_ID "${SBATCH_DIR}/load_results_fmv3_job.sh")
echo "Step 3: Load Results scheduled (afterok:$EVAL_ID). Job ID: $LOAD_ID"

echo "--------------------------------------------------------------------------------"
echo "Pipeline submitted successfully."
echo "Use 'squeue -u $USER' to monitor progress."
echo "If any step fails, the subsequent steps will be cancelled automatically by Slurm."
echo "================================================================================"
