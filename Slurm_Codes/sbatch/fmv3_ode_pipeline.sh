#!/bin/bash
set -e

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
TRAIN_ID=$(sbatch --parsable "${SBATCH_DIR}/train_fmv3_ode_job.sh")
echo "Step 1: Training submitted. Job ID: $TRAIN_ID"

# 2. Submit Evaluation Job (Success Dependency on Training)
# afterok:jobid means it only runs if the previous job exits with status 0
EVAL_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_fmv3_ode_job.sh")
echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

# 3. Submit Load Results Job (Success Dependency on Evaluation)
LOAD_ID=$(sbatch --parsable --dependency=afterok:$EVAL_ID "${SBATCH_DIR}/load_results_fmv3_job.sh")
echo "Step 3: Load Results scheduled (afterok:$EVAL_ID). Job ID: $LOAD_ID"

echo "--------------------------------------------------------------------------------"
echo "Pipeline submitted successfully."
echo "Use 'squeue -u $USER' to monitor progress."
echo "If any step fails, the subsequent steps will be cancelled automatically by Slurm."
echo "================================================================================"
