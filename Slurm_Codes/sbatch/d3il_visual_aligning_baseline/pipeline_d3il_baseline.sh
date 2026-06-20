#!/bin/bash
#SBATCH --job-name=pipeline_d3il_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

set -e

echo "================================================================================"
echo "PIPELINE START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "================================================================================"

function on_exit {
    echo "================================================================================"
    echo "PIPELINE END: $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# ─── Logging ────────────────────────────────────────────────────────────────
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# ─── Args ───────────────────────────────────────────────────────────────────
# $1 = agent_name   (default: ddpm_encdec_vision)
# $2 = seed         (default: 42)
# $3 = epoch        (default: paper setting — 200 vision / 500 state; passed to train script)
# $4 = record_mode  (default: all)
# $5 = eval_scale   (optional: "paper" → 60 ctx x 18 traj, faithful entropy; else config smoke)
#
# Example:
#   sbatch pipeline_d3il_baseline.sh ddpm_encdec_vision 42 200 all
#   sbatch pipeline_d3il_baseline.sh ddpm_vision 6                  # uses paper defaults
#   sbatch pipeline_d3il_baseline.sh ddpm_encdec_vision 42 200 all paper   # + paper-faithful eval

AGENT_NAME="${1:-ddpm_encdec_vision}"
SEED="${2:-42}"
EPOCH="${3:-}"         # empty → train script uses its own paper default
RECORD_MODE="${4:-all}"
EVAL_SCALE="${5:-}"    # "paper" forwarded to eval_d3il_baseline.sh as its $4

SBATCH_DIR="Slurm_Codes/sbatch/d3il_visual_aligning_baseline"

echo "D3IL Baseline Pipeline: agent=${AGENT_NAME}  seed=${SEED}  epoch=${EPOCH:-paper-default}  record=${RECORD_MODE}  eval_scale=${EVAL_SCALE:-smoke}"

# 1. Training
TRAIN_ID=$(sbatch --parsable $LOG_OPTS \
    "${SBATCH_DIR}/train_d3il_baseline.sh" "${AGENT_NAME}" "${SEED}" "${EPOCH}")
echo "Step 1: Training submitted. Job ID: ${TRAIN_ID}"

# 2. Evaluation (runs only if training succeeds)
EVAL_ID=$(sbatch --parsable $LOG_OPTS \
    --dependency=afterok:${TRAIN_ID} \
    "${SBATCH_DIR}/eval_d3il_baseline.sh" "${AGENT_NAME}" "${SEED}" "${RECORD_MODE}" "${EVAL_SCALE}")

echo "Step 2: Evaluation scheduled (afterok:${TRAIN_ID}). Job ID: ${EVAL_ID}"

echo "--------------------------------------------------------------------------------"
echo "D3IL Baseline Pipeline submitted."
echo "  Train job: ${TRAIN_ID}"
echo "  Eval  job: ${EVAL_ID} (blocked until train succeeds)"
echo "Use 'squeue -u \$USER' to monitor."
echo "Outputs: logs/d3il_visual_aligning_baseline/${AGENT_NAME}/seed_${SEED}/"
echo "  Epoch:  ${EPOCH:-paper-default (200 vision / 500 state)}"
echo "================================================================================"
