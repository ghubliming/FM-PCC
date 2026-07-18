#!/bin/bash
#SBATCH --job-name=hf_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Pipeline ORCHESTRATOR (fix_3). Chains train_hardflow.sh -> eval_hardflow.sh
# as SEPARATE sbatch jobs via --dependency=afterok, matching the repo
# convention (sbatch/iMF/imf_pipeline.sh, sbatch/dpcc_pipeline.sh, etc.) — NOT
# one inline job. The orchestrator itself just submits and exits.
#
# WHY THIS REPLACED an inline version: the previous hardflow_pipeline.sh ran
# train+eval inline in ONE job and requested --time=36:00:00 (24h train + 12h
# eval) to cover both phases. The cluster's actual partition hard cap is 24h,
# so that request could NEVER be scheduled (PartitionTimeLimit — confirmed via
# the identical bug in imf_pipeline_hardflow.sh, job 23577; see
# logs_in_develop/Gen13/fix_3/CHANGELOG_Gen13_fix3_pipeline_time_limit.md).
# Each CHAINED job below already has its own correctly-sized --time (<=24h).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh
# Skip training (use a downloaded .pth): SKIP_TRAIN=1 ./Slurm_Codes/submit.sh ...
# Eval knob (forwarded to the eval job): METHODS ("hardflow_new original")
# ==============================================================================
set -e

echo "================================================================================"
echo "PIPELINE START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "PIPELINE END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

SBATCH_DIR="Slurm_Codes/sbatch/hardflow"
CKPT="logs/hardflow/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth"

EVAL_EXPORT="ALL"
[ -n "$METHODS" ] && EVAL_EXPORT="$EVAL_EXPORT,METHODS=$METHODS"

if [ "${SKIP_TRAIN:-0}" = "1" ] || [ -f "$CKPT" ]; then
    echo "[ HF-PIPE ] SKIP_TRAIN or checkpoint present ($CKPT) — submitting eval only."
    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS "${SBATCH_DIR}/eval_hardflow.sh")
    echo "Eval submitted standalone. Job ID: $EVAL_ID"
else
    TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_hardflow.sh")
    echo "Step 1: Training submitted. Job ID: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS \
        --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_hardflow.sh")
    echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
fi

echo "--------------------------------------------------------------------------------"
echo "HardFlow pipeline submitted. Use 'squeue -u $USER' to monitor."
echo "If training fails, evaluation is auto-cancelled by Slurm (afterok)."
