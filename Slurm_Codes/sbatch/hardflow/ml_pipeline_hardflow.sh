#!/bin/bash
#SBATCH --job-name=hf_ml_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U11/U12 — Mix-ML pipeline ORCHESTRATOR. Chains train_ml_hardflow.sh ->
# eval_ml_hardflow.sh as SEPARATE sbatch jobs via --dependency=afterok (repo
# convention; see imf_pipeline_hardflow.sh). The orchestrator only submits + exits.
#
# EVAL is OBJECTIVE-AGNOSTIC — MF/AF train the SAME TemporalImfUnet, so
# run/eval_imf.py loads their checkpoints and runs ImfFlowPolicy unchanged.
# U12: eval now goes through eval_ml_hardflow.sh (NOT the frozen iMF one), which
# names dirs after the objective. U12.2: EXP_NAME itself now carries the family
# prefix (EXP_NAME=<ml_type>/H16_ml_<ml_type>_<steps>k), so BOTH train and eval
# nest one folder per FAMILY FIRST, then the run, then the arm — for free:
#     logs/<env>/flow/<EXP_NAME>/model_ema_*.pth                  == flow/mf/H16_ml_mf_100k/...
#     logs/<env>/eval/<EXP_NAME>/{raw,hfproj}_K<k>[_n<n>]/         == eval/mf/H16_ml_mf_100k/hfproj_K2_n200/
# We forward ML_EXP_NAME=$EXP_NAME so eval loads exactly this checkpoint and
# lands in the matching family folder.
#
# ------------------------------------------------------------------ USAGE
#   MeanFlow, one seed, train+eval (K=1,2, n=200):
#     ML_TYPE=mf N_TRAIN_STEPS=100000 IMF_KS="1 2" RANDOM_REPEAT=200 \
#       ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
#   α-Flow:
#     ML_TYPE=af N_TRAIN_STEPS=100000 IMF_KS="1 2" RANDOM_REPEAT=200 \
#       ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
#
# Train knobs : ML_TYPE (imf|mf|af), N_TRAIN_STEPS, ML_EXP_NAME, IMF_LR,
#               IMF_GRAD_CLIP, MF_DATA_PROPORTION, MF_P_STD, AF_ALPHA_SCHEDULER,
#               AF_ALPHA_INIT, AF_ALPHA_END, AF_ALPHA_GAMMA, AF_RATIO_FM,
#               USE_WANDB, FORCE_OVERWRITE
# Eval  knobs : ML_METHODS ("raw hfproj"), ML_KS ("1 2"), RANDOM_REPEAT,
#               ML_CP (auto-derived), HF_EVAL_SAVE_PNG (0), IMF_PLOT_FAN
#               (legacy IMF_METHODS/IMF_KS still accepted as fallbacks)
# SKIP_TRAIN=1 : eval an already-trained ML_EXP_NAME, no training job.
# ==============================================================================
set -e

echo "================================================================================"
echo "PIPELINE START: $(date)   JOB ID: $SLURM_JOB_ID"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "PIPELINE END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

SBATCH_DIR="Slurm_Codes/sbatch/hardflow"

# ---- resolve WHICH run this pipeline is about (sync with train_ml.sh) --------
ML_TYPE="${ML_TYPE:-imf}"
N_TRAIN_STEPS="${N_TRAIN_STEPS:-100000}"
STEPS_TAG="$(( N_TRAIN_STEPS / 1000 ))k"
RUN_NAME="H16_ml_${ML_TYPE}_${STEPS_TAG}"
# U12.2: default nests under <ml_type>/ (family-first folder). An explicit
# ML_EXP_NAME override is used VERBATIM (no auto family-prefix).
EXP_NAME="${ML_EXP_NAME:-${ML_TYPE}/${RUN_NAME}}"
FINAL_CP="${IMF_CP:-$(( N_TRAIN_STEPS / 25000 ))}"
CKPT="logs/hardflow/avoiding-v0/flow/${EXP_NAME}/model_ema_${FINAL_CP}.pth"

echo "[ HF-ML-PIPE ] ml_type    : $ML_TYPE"
echo "[ HF-ML-PIPE ] exp_name   : $EXP_NAME   (steps=$N_TRAIN_STEPS)"
echo "[ HF-ML-PIPE ] eval cp    : $FINAL_CP"
echo "[ HF-ML-PIPE ] checkpoint : $CKPT"
echo "[ HF-ML-PIPE ] eval       : methods='${ML_METHODS:-raw hfproj}' K='${ML_KS:-${IMF_KS:-1 2}}' n=${RANDOM_REPEAT:-50}"

# ---- build the --export lists ------------------------------------------------
# EXP_NAME is forwarded to train as ML_EXP_NAME and to eval as ML_EXP_NAME (the
# U12 eval scripts read ML_EXP_NAME for flow_exp_name), so eval loads exactly the
# checkpoint this pipeline produced.
BASE="ALL"
TRAIN_EXPORT="$BASE,ML_TYPE=$ML_TYPE,ML_EXP_NAME=$EXP_NAME,N_TRAIN_STEPS=$N_TRAIN_STEPS"
EVAL_EXPORT="$BASE,ML_EXP_NAME=$EXP_NAME,ML_CP=$FINAL_CP"
for v in IMF_LR IMF_GRAD_CLIP MF_DATA_PROPORTION MF_P_STD \
         AF_ALPHA_SCHEDULER AF_ALPHA_INIT AF_ALPHA_END AF_ALPHA_GAMMA AF_RATIO_FM \
         USE_WANDB FORCE_OVERWRITE WANDB_PROJECT; do
    [ -n "${!v}" ] && TRAIN_EXPORT="$TRAIN_EXPORT,$v=${!v}"
done
# ML_METHODS/ML_KS are the U12 names; IMF_METHODS/IMF_KS still accepted (fallback).
for v in ML_METHODS ML_KS IMF_METHODS IMF_KS RANDOM_REPEAT HF_EVAL_SAVE_PNG IMF_PLOT_FAN; do
    [ -n "${!v}" ] && EVAL_EXPORT="$EVAL_EXPORT,$v=${!v}"
done

# ---- submit ------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    if [ ! -f "$CKPT" ]; then
        echo "[ HF-ML-PIPE ] ABORT: SKIP_TRAIN=1 but $CKPT does not exist." >&2
        exit 1
    fi
    echo "[ HF-ML-PIPE ] SKIP_TRAIN=1 -> eval only, on the existing checkpoint."
    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS "${SBATCH_DIR}/eval_ml_hardflow.sh")
    echo "Eval submitted standalone. Job ID: $EVAL_ID"
else
    if [ -f "$CKPT" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
        echo "[ HF-ML-PIPE ] ABORT: '$EXP_NAME' already holds a finished run ($CKPT)." >&2
        echo "                Use ML_EXP_NAME=<new>, a different N_TRAIN_STEPS," >&2
        echo "                SKIP_TRAIN=1 to just re-evaluate it, or FORCE_OVERWRITE=1." >&2
        exit 1
    fi
    TRAIN_ID=$(sbatch --parsable --export="$TRAIN_EXPORT" $LOG_OPTS "${SBATCH_DIR}/train_ml_hardflow.sh")
    echo "Step 1: Training submitted (gates run inside it first). Job ID: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable --export="$EVAL_EXPORT" $LOG_OPTS \
        --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_ml_hardflow.sh")
    echo "Step 2: Evaluation scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
fi

echo "--------------------------------------------------------------------------------"
echo "Mix-ML pipeline submitted (ml_type=$ML_TYPE). Monitor with 'squeue -u $USER'."
echo "If training fails, evaluation is auto-cancelled by Slurm (afterok)."
echo "Results land in logs/hardflow/avoiding-v0/eval/${EXP_NAME}/{raw,hfproj}_K*[_n*]/  (family folder: ${ML_TYPE}/)"
