#!/bin/bash
#SBATCH --job-name=d3il_all_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

set -e

# One pipeline job that submits train→eval pairs for all paper seeds.
# Mirrors pipeline_d3il_baseline.sh pattern: lightweight orchestrator only.
#
# Usage:
#   sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/run_all_seeds_d3il_baseline.sh
#   sbatch ... ddpm_encdec_vision 200 paper        # overrides
#
# Args: $1=agent_name (def: ddpm_encdec_vision)  $2=epoch (def: 200)  $3=eval_scale (def: paper)

AGENT_NAME="${1:-ddpm_encdec_vision}"
EPOCH="${2:-200}"
EVAL_SCALE="${3:-paper}"
SEEDS=(0 1 2 3 4 42)

SBATCH_DIR="Slurm_Codes/sbatch/d3il_visual_aligning_baseline"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

echo "========================================================"
echo "PIPELINE START: $(date)  |  Job $SLURM_JOB_ID"
echo "Agent: $AGENT_NAME  Epochs: $EPOCH  Eval: $EVAL_SCALE"
echo "Seeds: ${SEEDS[*]}"
echo "========================================================"

for SEED in "${SEEDS[@]}"; do
    TRAIN_ID=$(sbatch --parsable $LOG_OPTS \
        "${SBATCH_DIR}/train_d3il_baseline.sh" "$AGENT_NAME" "$SEED" "$EPOCH")
    EVAL_ID=$(sbatch --parsable $LOG_OPTS \
        --dependency=afterok:${TRAIN_ID} \
        "${SBATCH_DIR}/eval_d3il_baseline.sh" "$AGENT_NAME" "$SEED" "all" "$EVAL_SCALE")
    echo "  seed=$SEED → train=$TRAIN_ID  eval=$EVAL_ID (afterok:$TRAIN_ID)"
done

echo "========================================================"
echo "All seeds submitted. Monitor: squeue -u \$USER"
echo "========================================================"
