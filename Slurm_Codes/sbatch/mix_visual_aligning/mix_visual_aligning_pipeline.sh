#!/bin/bash
#SBATCH --job-name=mix_visual_aligning_pipeline
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
# Visual-Mix-ML (Gen14) Pipeline Master Script
# ==============================================================================
# Chains three Slurm jobs for ONE engine arm:
#   1. Gates      (gates_mix_visual.sh)   — fidelity + objective sanity, ~minutes
#   2. Training   (train_mix_visual_aligning.sh)  — only if the gates pass
#   3. Evaluation (eval_mix_visual_aligning.sh)   — only if training succeeds
#
# Usage:  sbatch mix_visual_aligning_pipeline.sh <engine> [seed]
#           <engine> = ddpm | fm | mf | af      (default: fm, the Gen7 reference arm)
#           [seed]   = training seed            (default: 6)
#
#   sbatch mix_visual_aligning_pipeline.sh mf 6
#
# The gates run FIRST on purpose: a broken copy or a dead alpha schedule is cheap to
# catch in one minute and expensive to discover after a 24-hour training run.
# ==============================================================================

SBATCH_DIR="Slurm_Codes/sbatch/mix_visual_aligning"
ENGINE="${1:-fm}"
SEED="${2:-6}"

case "$ENGINE" in
    ddpm|fm|mf|af) ;;
    *) echo "ERROR: unknown engine '$ENGINE' (want: ddpm | fm | mf | af)"; exit 1 ;;
esac

echo "Launching Visual-Mix-ML (Gen14) Pipeline — engine=$ENGINE seed=$SEED ..."

# 1. Gates
GATE_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/gates_mix_visual.sh")
echo "Step 1: Gates submitted. Job ID: $GATE_ID"

# 2. Training (only if the gates pass)
TRAIN_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$GATE_ID \
    "${SBATCH_DIR}/train_mix_visual_aligning.sh" "$ENGINE" "$SEED")
echo "Step 2: Training scheduled (afterok:$GATE_ID). Job ID: $TRAIN_ID"

# 3. Evaluation (only if training succeeds)
EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID \
    "${SBATCH_DIR}/eval_mix_visual_aligning.sh" "$ENGINE" "$SEED")
echo "Step 3: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"

echo "--------------------------------------------------------------------------------"
echo "Visual-Mix-ML (Gen14) Pipeline submitted — engine=$ENGINE seed=$SEED."
echo "Use 'squeue -u $USER' to monitor progress."
echo "A failed stage cancels the downstream stages automatically."
echo "================================================================================"
