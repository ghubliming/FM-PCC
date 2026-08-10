#!/bin/bash
#SBATCH --job-name=uav_mix_all_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — one job PER SCENE (not per scene*seed): each train/eval job loops over
# all seeds internally (see train_fm_uav.sh / eval_fm_uav.sh). Total sbatch calls =
# 2*num_scenes + 1, independent of how many seeds you pass — adding seeds never adds
# jobs, only --time. See logs_in_develop/Gen11/Epoch6_fm_pcc_training/U2/F1_submit_logic/.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_all_pipeline.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_all_pipeline.sh mf "empty corridor s_curve pillars" "6 7 8 9 10" 20
# Args: $1=engine (fm|mf|af) [fm]  $2=scenes [all 4]  $3=seeds ["6"]  $4=n_trials [omit → yaml default]
#       $5=projection [fm_only]  $6=K / flow_steps [omit → plan-block value]
# n_trials: omit $4 → reads from config/uav_projection.yaml; pass int → CLI override.
# This submits ONE ARM. Run it once per engine; the three arms write to disjoint trees.
set -e

ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af) ;; *) echo "[ ERROR ] engine must be fm|mf|af (got '$ENGINE')"; exit 1 ;; esac
SCENES="${2:-empty corridor s_curve pillars}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10".
SEEDS="${3:-6}"
NTRIALS="${4:-}"   # empty = let config/uav_projection.yaml n_trials apply
PROJ="${5:-fm_only}"
FLOW_STEPS="${6:-}"   # empty = use the plan block's flow_steps_v3
DIR="Slurm_Codes/sbatch/uav_mix"

# Per-combo time budgets (the single-seed ceilings baked into the job scripts), scaled
# by seed count since seeds run sequentially inside one job allocation.
N_SEEDS=$(echo $SEEDS | wc -w)
TRAIN_HOURS=$((N_SEEDS * 24))
EVAL_HOURS=$((N_SEEDS * 8))

echo "================================================================================"
echo "UAV-MIX ALL PIPELINE (per-scene, seed-loop internal)  $(date)   engine=$ENGINE"
echo "scenes=[$SCENES]  seeds=[$SEEDS] (x$N_SEEDS)  n_trials=${NTRIALS:-'yaml default'}  proj=$PROJ  K=${FLOW_STEPS:-'plan block'}"
echo "train --time=${TRAIN_HOURS}:00:00   eval --time=${EVAL_HOURS}:00:00   (scaled by seed count)"
echo "[ note ] requires the curated dataset (data/uav_fm/v1/) — run Aux-1 prepare first."
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

EVAL_DEP="afterok"
for scene in $SCENES; do
    T_ID=$(sbatch --parsable --time="${TRAIN_HOURS}:00:00" $LOG_OPTS "$DIR/train_mix_uav.sh" "$ENGINE" "$scene" "$SEEDS")
    E_ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS --dependency=afterok:${T_ID} \
           "$DIR/eval_mix_uav.sh" "$ENGINE" "$scene" "$SEEDS" "$NTRIALS" "$PROJ" "none" "$FLOW_STEPS")
    echo "  engine=$ENGINE scene=$scene  seeds=[$SEEDS]   train=$T_ID → eval=$E_ID"
    EVAL_DEP="${EVAL_DEP}:${E_ID}"
done

AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$EVAL_DEP" "$DIR/aggregate_summaries.sh" "$ENGINE" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
N_SCENES=$(echo $SCENES | wc -w)
echo "Total jobs submitted: $((2*N_SCENES+1))  (was up to $((2*N_SCENES*N_SEEDS+1)) under the old per-seed-job scheme)"
echo "When done: logs/UAV_MIX/uav-<scene>/SCENE_SUMMARY.json + logs/UAV_MIX/uav_mix_${ENGINE}_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
