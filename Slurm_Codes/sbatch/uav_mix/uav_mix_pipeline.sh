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
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
SEED="${3:-6}"
NTRIALS="${4:-}"          # empty = let config/uav_projection.yaml n_trials apply
PROJECTION="${5:-fm_only}"
RECORD="${6:-none}"
FLOW_STEPS="${7:-}"       # empty = use the plan block's flow_steps_v3
SBATCH_DIR="Slurm_Codes/sbatch/uav_mix"

# ── Gen15 U6 ── carry the af knobs onto BOTH child stages EXPLICITLY.
#
# `sbatch` defaults to --export=ALL, so these would ride along anyway — but two of them are
# CHECKPOINT-PATH keys ('_bb<bone>', '_ae<alpha>') and one is a RESULTS-PATH key ('_EP<sel>'),
# and a stage that does not see them resolves a DIFFERENT directory than the submitter is
# watching. Naming them here makes the chain reproducible from the log alone, and is the same
# doctrine Gen14's pipeline applies to MIX_TRAIN_STEPS.
EXPORT_OPTS="--export=ALL"
for _v in UAV_MIX_BONE_AF UAV_MIX_AF_ALPHA_END UAV_MIX_EPOCH; do
    eval "_val=\${$_v:-}"
    if [ -n "$_val" ]; then
        if [ "$_v" != "UAV_MIX_EPOCH" ] && [ "$ENGINE" != "af" ]; then
            echo "[ ERROR ] $_v is set but engine='$ENGINE'. It applies to the af arm only."
            exit 1
        fi
        EXPORT_OPTS="$EXPORT_OPTS,$_v=$_val"
        echo "[ pipeline ] $_v=$_val"
    fi
done
if [ "$ENGINE" = "af" ]; then
    echo "[ pipeline ] af bone = ${UAV_MIX_BONE_AF:-unet}  (U6 default; 'sit' via UAV_MIX_BONE_AF=sit)"
    if [ -z "${UAV_MIX_AF_ALPHA_END:-}" ]; then
        echo "[ pipeline ]   ⚠  af_alpha_end = 0.0 (shipped): alpha snaps to EXACTLY 0 from"
        echo "[ pipeline ]      ~71.2% of the budget on, so this run ends on the MEANFLOW"
        echo "[ pipeline ]      target. Set UAV_MIX_AF_ALPHA_END>0 to train alpha-Flow proper."
    elif [ -z "${UAV_MIX_EPOCH:-}" ]; then
        echo "[ pipeline ]   ⚠  alpha is floored but UAV_MIX_EPOCH is unset -> eval will load"
        echo "[ pipeline ]      'best', which prefers a MID-CURRICULUM checkpoint and discards"
        echo "[ pipeline ]      what the floor produced. Set UAV_MIX_EPOCH=latest."
    fi
fi

echo "================================================================================"
echo "UAV-MIX PIPELINE START: $(date)  engine=$ENGINE scene=$SCENE seed=$SEED n_trials=${NTRIALS:-'yaml default'} K=${FLOW_STEPS:-'plan block'}"
echo "================================================================================"

# Unify dated logs across the chained jobs (submit.sh exports SUBMIT_DATE/TIME).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

TRAIN_ID=$(sbatch --parsable $EXPORT_OPTS $LOG_OPTS "${SBATCH_DIR}/train_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED")
echo "Step 1: train submitted — Job $TRAIN_ID"

EVAL_ID=$(sbatch --parsable $EXPORT_OPTS $LOG_OPTS --dependency=afterok:${TRAIN_ID} \
    "${SBATCH_DIR}/eval_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED" "${NTRIALS}" "$PROJECTION" "$RECORD" "$FLOW_STEPS")
echo "Step 2: eval scheduled (afterok:${TRAIN_ID}) — Job $EVAL_ID"
echo "--------------------------------------------------------------------------------"
echo "Outputs: logs/UAV_MIX/uav-${SCENE}/mix_uav_${ENGINE}/.../${SEED}/  (+ plans/ for eval)"
echo "================================================================================"
