#!/bin/bash
#SBATCH --job-name=hf_imf_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — iMF pipeline ORCHESTRATOR. Chains train_imf_hardflow.sh ->
# eval_imf_hardflow.sh as SEPARATE sbatch jobs via --dependency=afterok, exactly
# like the repo convention (sbatch/iMF/imf_pipeline.sh, sbatch/dpcc_pipeline.sh).
# The orchestrator itself only submits and exits — no GPU, no long walltime.
#
# WHY NOT INLINE (fix_3): an earlier version ran train+eval in ONE job and asked
# for --time=36:00:00. The partition hard cap is 24h, so job 23577 sat PENDING
# forever (PartitionTimeLimit). Each chained job below has its own <=24h --time.
# See logs_in_develop/Gen13/fix_3/CHANGELOG_Gen13_fix3_pipeline_time_limit.md.
#
# U9.2 — THE POINT OF THIS REVISION: exp_name and checkpoint index are now
# DERIVED here, using the SAME rule as run_scripts/train_imf.sh, and exported to
# the eval job. Before, both were pinned to "H16_imf_100k", so any run with a
# different budget or IMF_EXP_NAME trained a new model and then silently
# evaluated the OLD checkpoint. Same bug class as U9's eval-tagging fix.
#
# ------------------------------------------------------------------ USAGE
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
#
#   U9.2 fix run (train @ corrected LR, then eval it — one submit):
#     IMF_LR=2e-5 IMF_GRAD_CLIP=1.0 N_TRAIN_STEPS=100000 \
#     IMF_EXP_NAME=H16_imf_lrfix_100k IMF_KS=2 RANDOM_REPEAT=200 \
#       ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
#
# Train knobs : N_TRAIN_STEPS, IMF_EXP_NAME, IMF_LR, IMF_GRAD_CLIP,
#               IMF_DATA_PROPORTION, IMF_P_STD, USE_WANDB, FORCE_OVERWRITE
# Eval  knobs : IMF_METHODS ("original hardflow_new"), IMF_KS ("1 2"),
#               RANDOM_REPEAT, IMF_CP (auto-derived; override only to pick an
#               intermediate checkpoint)
# SKIP_TRAIN=1 : eval an already-trained IMF_EXP_NAME, no training job.
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

# ---- resolve WHICH run this pipeline is about --------------------------------
# MUST stay in sync with run_scripts/train_imf.sh (steps_tag / exp_name / final_cp)
# and run_scripts/eval_*_imf.sh (flow_exp_name / flow_cp).
N_TRAIN_STEPS="${N_TRAIN_STEPS:-100000}"
STEPS_TAG="$(( N_TRAIN_STEPS / 1000 ))k"
EXP_NAME="${IMF_EXP_NAME:-H16_imf_${STEPS_TAG}}"
FINAL_CP="${IMF_CP:-$(( N_TRAIN_STEPS / 25000 ))}"
CKPT="logs/hardflow/avoiding-v0/flow/${EXP_NAME}/model_ema_${FINAL_CP}.pth"

echo "[ HF-IMF-PIPE ] exp_name   : $EXP_NAME   (steps=$N_TRAIN_STEPS)"
echo "[ HF-IMF-PIPE ] eval cp    : $FINAL_CP"
echo "[ HF-IMF-PIPE ] checkpoint : $CKPT"
echo "[ HF-IMF-PIPE ] lr=${IMF_LR:-2e-4 (default)}  grad_clip=${IMF_GRAD_CLIP:-1.0 (default)}"
echo "[ HF-IMF-PIPE ] eval       : methods='${IMF_METHODS:-original hardflow_new}' K='${IMF_KS:-1 2}' n=${RANDOM_REPEAT:-50}"

# ---- build the --export list -------------------------------------------------
# EXP_NAME/CP are forwarded EXPLICITLY (not just via ALL) so the eval job loads
# the checkpoint this pipeline actually produced, whatever the caller passed.
EXPORTS="ALL,IMF_EXP_NAME=$EXP_NAME"
TRAIN_EXPORT="$EXPORTS,N_TRAIN_STEPS=$N_TRAIN_STEPS"
EVAL_EXPORT="$EXPORTS,IMF_CP=$FINAL_CP"
for v in IMF_LR IMF_GRAD_CLIP IMF_DATA_PROPORTION IMF_P_STD USE_WANDB FORCE_OVERWRITE; do
    [ -n "${!v}" ] && TRAIN_EXPORT="$TRAIN_EXPORT,$v=${!v}"
done
for v in IMF_METHODS IMF_KS RANDOM_REPEAT IMF_PLOT_FAN; do
    [ -n "${!v}" ] && EVAL_EXPORT="$EVAL_EXPORT,$v=${!v}"
done

# ---- submit ------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    if [ ! -f "$CKPT" ]; then
        echo "[ HF-IMF-PIPE ] ABORT: SKIP_TRAIN=1 but $CKPT does not exist." >&2
        exit 1
    fi
    echo "[ HF-IMF-PIPE ] SKIP_TRAIN=1 -> eval only, on the existing checkpoint."
    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS "${SBATCH_DIR}/eval_imf_hardflow.sh")
    echo "Eval submitted standalone. Job ID: $EVAL_ID"
else
    # U9 safety, mirrored from train_imf.sh: never silently clobber a finished run.
    # (train_imf.sh aborts too, but catching it HERE avoids queueing a doomed job
    # plus a dependent eval that would then never run.)
    if [ -f "$CKPT" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
        echo "[ HF-IMF-PIPE ] ABORT: '$EXP_NAME' already holds a finished run ($CKPT)." >&2
        echo "                Use IMF_EXP_NAME=<new>, a different N_TRAIN_STEPS," >&2
        echo "                SKIP_TRAIN=1 to just re-evaluate it, or FORCE_OVERWRITE=1." >&2
        exit 1
    fi
    TRAIN_ID=$(sbatch --parsable --export="$TRAIN_EXPORT" $LOG_OPTS "${SBATCH_DIR}/train_imf_hardflow.sh")
    echo "Step 1: Training submitted (gates run inside it first). Job ID: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS \
        --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_imf_hardflow.sh")
    echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
fi

echo "--------------------------------------------------------------------------------"
echo "iMF pipeline submitted. Use 'squeue -u $USER' to monitor."
echo "If training fails, evaluation is auto-cancelled by Slurm (afterok)."
echo "Results land in logs/hardflow/avoiding-v0/eval/*_from_${EXP_NAME}*/"
