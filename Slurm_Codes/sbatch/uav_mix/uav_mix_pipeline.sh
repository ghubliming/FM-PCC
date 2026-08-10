#!/bin/bash
#SBATCH --job-name=uav_mix_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
set -e

# One-shot: train → eval (eval runs only if train succeeds).
# Args: $1=engine (fm|mf|af, def fm)  $2=scene (def all)  $3=seed (def 6)
#        $4=n_trials (omit → yaml default)  $5=projection (def fm_only)
#        $6=record (none|gif|all, def none)  $7=K / flow_steps (omit → plan-block value)
# n_trials: omit $4 (or pass "") → reads from config/uav_projection.yaml; pass int → CLI override.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af) ;; *) echo "[ ERROR ] engine must be fm|mf|af (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
SEED="${3:-6}"
NTRIALS="${4:-}"          # empty = let config/uav_projection.yaml n_trials apply
PROJECTION="${5:-fm_only}"
RECORD="${6:-none}"
FLOW_STEPS="${7:-}"       # empty = use the plan block's flow_steps_v3
SBATCH_DIR="Slurm_Codes/sbatch/uav_mix"

echo "================================================================================"
echo "UAV-MIX PIPELINE START: $(date)  engine=$ENGINE scene=$SCENE seed=$SEED n_trials=${NTRIALS:-'yaml default'} K=${FLOW_STEPS:-'plan block'}"
echo "================================================================================"

# Unify dated logs across the chained jobs (submit.sh exports SUBMIT_DATE/TIME).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED")
echo "Step 1: train submitted — Job $TRAIN_ID"

EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${TRAIN_ID} \
    "${SBATCH_DIR}/eval_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED" "${NTRIALS}" "$PROJECTION" "$RECORD" "$FLOW_STEPS")
echo "Step 2: eval scheduled (afterok:${TRAIN_ID}) — Job $EVAL_ID"
echo "--------------------------------------------------------------------------------"
echo "Outputs: logs/UAV_MIX/uav-${SCENE}/mix_uav_${ENGINE}/.../${SEED}/  (+ plans/ for eval)"
echo "================================================================================"
