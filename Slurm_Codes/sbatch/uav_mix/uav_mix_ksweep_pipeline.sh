#!/bin/bash
#SBATCH --job-name=uav_mix_ksweep_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
set -e

# One-shot: train ONCE → fan out one eval per K (each eval runs only if the train succeeded).
#
# Args: $1=engine (fm|mf|af|diffusion, def fm)   $2=scene (def all)      $3=seed (def 6)
#       $4=n_trials (omit → yaml default)        $5=projection (def fm_only)
#       $6=record (none|gif|all, def none)       $7=K list, QUOTED & space-separated
#                                                   (omit/"" → ONE eval at the plan-block K)
#
# Why this exists, and why it is not `uav_mix_pipeline.sh` in a loop: for fm/mf/af, K is an
# EVAL-time knob (engine_registry.apply_nfe) and `train_mix_uav.sh` takes no K argument — so
# one checkpoint serves every K. Looping the single-K pipeline would retrain the same model
# N times. This trains once and hangs N evals off that one job id.
#
# 🔴 `diffusion` is the exception: its K is a TRAINING property (n_diffusion_steps, the beta
# schedule is built from it), so pass NO K list for that arm — the plan block's value applies
# and `apply_nfe` deliberately no-ops. Passing a K list to `diffusion` is a usage error and is
# rejected below rather than silently mislabelling the output folder.
#
# 🔴 MATCHED BUDGET: K appears in the output path as K{n}, so distinct-K evals never overwrite
# each other — but when COMPARING arms, give every arm the same K.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
SEED="${3:-6}"
NTRIALS="${4:-}"          # empty = let config/uav_projection.yaml n_trials apply
PROJECTION="${5:-fm_only}"
RECORD="${6:-none}"
KLIST="${7:-}"            # empty = ONE eval at the plan block's flow_steps_v3
SBATCH_DIR="Slurm_Codes/sbatch/uav_mix"

if [ "$ENGINE" = "diffusion" ] && [ -n "$KLIST" ]; then
    echo "[ ERROR ] engine 'diffusion' bakes K at TRAIN time (config/uav_mix.py n_diffusion_steps)."
    echo "          A K list here would only relabel the folder, not change the NFE. Pass no K list."
    exit 1
fi

echo "================================================================================"
echo "UAV-MIX K-SWEEP PIPELINE START: $(date)"
echo "  engine=$ENGINE  scene=$SCENE  seed=$SEED  n_trials=${NTRIALS:-'yaml default'}"
echo "  projection=$PROJECTION  record=$RECORD  K list=${KLIST:-'<plan block>'}"
echo "================================================================================"

# Unify dated logs across the chained jobs (submit.sh exports SUBMIT_DATE/TIME).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED")
echo "Step 1: train submitted — Job $TRAIN_ID"

# afterok on the SAME train job for every eval: they all read one checkpoint, and none of them
# starts if training fails. They are independent of each other, so Slurm is free to run them
# in any order (or concurrently, if the partition ever allows it).
N=0
if [ -z "$KLIST" ]; then
    EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${TRAIN_ID} \
        "${SBATCH_DIR}/eval_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED" "${NTRIALS}" "$PROJECTION" "$RECORD" "")
    echo "Step 2: eval scheduled (afterok:${TRAIN_ID}) K=<plan block> — Job $EVAL_ID"
    N=1
else
    for K in $KLIST; do
        case "$K" in ''|*[!0-9]*) echo "[ ERROR ] K list must be integers, got '$K'"; exit 1 ;; esac
        EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${TRAIN_ID} \
            "${SBATCH_DIR}/eval_mix_uav.sh" "$ENGINE" "$SCENE" "$SEED" "${NTRIALS}" "$PROJECTION" "$RECORD" "$K")
        echo "Step 2.$((N+1)): eval scheduled (afterok:${TRAIN_ID}) K=$K — Job $EVAL_ID"
        N=$((N+1))
    done
fi

echo "--------------------------------------------------------------------------------"
echo "Submitted: 1 train + $N eval job(s), all gated on train $TRAIN_ID"
echo "Outputs: logs/UAV_MIX/uav-${SCENE}/mix_uav_${ENGINE}/.../${SEED}/  (+ plans/ per K)"
echo "================================================================================"
