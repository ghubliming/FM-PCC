#!/bin/bash
#SBATCH --job-name=hffm_pipeline
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
# Gen12 (HardFlow -> FMv3) Pipeline Master Script
# ==============================================================================
# There is NO training job here, and that is the point: HardFlow's contribution
# is entirely at sampling time, so Gen12 reuses the existing Gen3 FMv3 checkpoint
# (PLAN §1). The chain is:
#
#   1. gates        G0-G3 seam assertions            (PLAN §4 steps 2)
#   2. eval         arms A/B/C at matched K          (PLAN §4 steps 5-7)
#   3. load_results one table per K bucket
#
# The dynamics refit (fit_dynamics_hardflow_fmv3.sh) is NOT chained in: the
# default `hardflow.dynamics_mode: deriv` reuses FMPCC's own kinematics so arms B
# and C enforce an identical feasible set. Submit it separately, before eval, if
# you switch the YAML to `linear_fit`:
#
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/fit_dynamics_hardflow_fmv3.sh
# ==============================================================================

SBATCH_DIR="Slurm_Codes/sbatch/hardflow_fmv3"

echo "Launching Gen12 HardFlow-into-FMv3 Pipeline..."

# 1. Gates — PLAN §4: "Do not proceed past a failing step." A gate failure exits
#    non-zero, so the afterok dependencies below cancel the rest automatically.
GATES_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/gates_hardflow_fmv3.sh")
echo "Step 1: Gates submitted. Job ID: $GATES_ID"

# 2. Evaluation (success dependency on the gates)
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$GATES_ID "${SBATCH_DIR}/eval_fmv3_hardflow_job.sh")
echo "Step 2: Evaluation scheduled (afterok:$GATES_ID). Job ID: $EVAL_ID"

# 3. Aggregation (success dependency on evaluation)
LOAD_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$EVAL_ID "${SBATCH_DIR}/load_results_hardflow_fmv3.sh")
echo "Step 3: Aggregation scheduled (afterok:$EVAL_ID). Job ID: $LOAD_ID"

echo "--------------------------------------------------------------------------------"
echo "Gen12 Pipeline submitted successfully."
echo "Use 'squeue -u $USER' to monitor progress."
echo "If the gates fail, evaluation and aggregation are cancelled automatically by Slurm."
echo "================================================================================"
