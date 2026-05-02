#!/bin/bash
#SBATCH --job-name=pipeline_template
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

# Exit on error
set -e

# ==============================================================================
# PIPELINE TEMPLATE (CHAINED SUCCESS ONLY)
# ==============================================================================
# Use this script to run multiple Slurm jobs in a sequence.
# Subsequent jobs will ONLY start if the previous job finishes successfully (exit 0).
# If a job fails, all subsequent jobs in the chain are automatically cancelled.
# ==============================================================================

SBATCH_DIR="Slurm_Codes/sbatch"
DATE=$(date +%Y-%m-%d)

echo "[${DATE}] Starting Pipeline Submission..."

# 1. Step One
JOB1_ID=$(sbatch --parsable "${SBATCH_DIR}/your_job_1.sh")
echo "Step 1 Submitted: Job ID $JOB1_ID"

# 2. Step Two (Depends on Step One Success)
JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID "${SBATCH_DIR}/your_job_2.sh")
echo "Step 2 Scheduled: Job ID $JOB2_ID (Waiting for $JOB1_ID)"

# 3. Step Three (Depends on Step Two Success)
JOB3_ID=$(sbatch --parsable --dependency=afterok:$JOB2_ID "${SBATCH_DIR}/your_job_3.sh")
echo "Step 3 Scheduled: Job ID $JOB3_ID (Waiting for $JOB2_ID)"

echo "--------------------------------------------------------------------------------"
echo "All jobs submitted to the queue."
echo "Check progress with: squeue -u $USER"
echo "================================================================================"
