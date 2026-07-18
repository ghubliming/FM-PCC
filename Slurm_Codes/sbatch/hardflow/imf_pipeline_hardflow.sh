#!/bin/bash
#SBATCH --job-name=hf_imf_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — iMF pipeline ORCHESTRATOR (fix_3). Chains train_imf_hardflow.sh ->
# eval_imf_hardflow.sh as SEPARATE sbatch jobs via --dependency=afterok, exactly
# like the repo convention (sbatch/iMF/imf_pipeline.sh, sbatch/dpcc_pipeline.sh,
# etc.) — NOT one inline job. The orchestrator itself just submits and exits;
# it needs no GPU/long walltime of its own.
#
# WHY THIS REPLACED an inline version: the previous imf_pipeline_hardflow.sh
# ran train+eval inline in ONE job and requested --time=36:00:00 (24h train +
# 12h eval) to cover both phases. The cluster's actual partition hard cap is
# 24h, so that request could NEVER be scheduled — job 23577 sat PENDING
# forever with reason (PartitionTimeLimit). See
# logs_in_develop/Gen13/fix_3/CHANGELOG_Gen13_fix3_pipeline_time_limit.md.
# Each CHAINED job below already has its own correctly-sized --time (<=24h),
# so this always fits.
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
# Skip training (reuse an existing checkpoint): SKIP_TRAIN=1 ./Slurm_Codes/submit.sh ...
# Eval knobs (forwarded to the eval job via env export):
#   IMF_METHODS ("original hardflow_new"), IMF_KS ("1 2"), IMF_CP (4)
# ==============================================================================
set -e

echo "================================================================================"
echo "PIPELINE START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "PIPELINE END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

# Sub-jobs share the pipeline's own submission timestamp for log grouping
# (same convention as sbatch/iMF/imf_pipeline.sh).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

SBATCH_DIR="Slurm_Codes/sbatch/hardflow"
CKPT="logs/hardflow/avoiding-v0/flow/H16_imf_100k/model_ema_4.pth"

# Forward eval knobs to the eval job's environment (sbatch --export)
EVAL_EXPORT="ALL"
[ -n "$IMF_METHODS" ] && EVAL_EXPORT="$EVAL_EXPORT,IMF_METHODS=$IMF_METHODS"
[ -n "$IMF_KS" ] && EVAL_EXPORT="$EVAL_EXPORT,IMF_KS=$IMF_KS"
[ -n "$IMF_CP" ] && EVAL_EXPORT="$EVAL_EXPORT,IMF_CP=$IMF_CP"

if [ "${SKIP_TRAIN:-0}" = "1" ] || [ -f "$CKPT" ]; then
    echo "[ HF-IMF-PIPE ] SKIP_TRAIN or checkpoint present ($CKPT) — submitting eval only."
    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS "${SBATCH_DIR}/eval_imf_hardflow.sh")
    echo "Eval submitted standalone. Job ID: $EVAL_ID"
else
    TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_imf_hardflow.sh")
    echo "Step 1: Training submitted (gates run inside it first). Job ID: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS \
        --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_imf_hardflow.sh")
    echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
fi

echo "--------------------------------------------------------------------------------"
echo "iMF pipeline submitted. Use 'squeue -u $USER' to monitor."
echo "If training fails, evaluation is auto-cancelled by Slurm (afterok)."
