#!/bin/bash
#SBATCH --job-name=uav_fm_all_pipeline
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
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh "empty corridor s_curve pillars" "5 6 7" 20
# Args: $1=scenes [all 4]  $2=seeds ["5 6 7"]  $3=n_trials [20]  $4=projection [fm_only]
set -e

SCENES="${1:-empty corridor s_curve pillars}"
SEEDS="${2:-5 6 7}"
NTRIALS="${3:-20}"
PROJ="${4:-fm_only}"
DIR="Slurm_Codes/sbatch/uav_fm"

# Per-combo time budgets (the single-seed ceilings baked into the job scripts), scaled
# by seed count since seeds run sequentially inside one job allocation.
N_SEEDS=$(echo $SEEDS | wc -w)
TRAIN_HOURS=$((N_SEEDS * 24))
EVAL_HOURS=$((N_SEEDS * 8))

echo "================================================================================"
echo "UAV-FM ALL PIPELINE (per-scene, seed-loop internal)  $(date)"
echo "scenes=[$SCENES]  seeds=[$SEEDS] (x$N_SEEDS)  n_trials=$NTRIALS  proj=$PROJ"
echo "train --time=${TRAIN_HOURS}:00:00   eval --time=${EVAL_HOURS}:00:00   (scaled by seed count)"
echo "[ note ] requires the curated dataset (data/uav_fm/v1/) — run Aux-1 prepare first."
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

EVAL_DEP="afterok"
for scene in $SCENES; do
    T_ID=$(sbatch --parsable --time="${TRAIN_HOURS}:00:00" $LOG_OPTS "$DIR/train_fm_uav.sh" "$scene" "$SEEDS")
    E_ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS --dependency=afterok:${T_ID} \
           "$DIR/eval_fm_uav.sh" "$scene" "$SEEDS" "$NTRIALS" "$PROJ")
    echo "  scene=$scene  seeds=[$SEEDS]   train=$T_ID → eval=$E_ID"
    EVAL_DEP="${EVAL_DEP}:${E_ID}"
done

AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$EVAL_DEP" "$DIR/aggregate_summaries.sh" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
N_SCENES=$(echo $SCENES | wc -w)
echo "Total jobs submitted: $((2*N_SCENES+1))  (was up to $((2*N_SCENES*N_SEEDS+1)) under the old per-seed-job scheme)"
echo "When done: logs/UAV_FM/uav-<scene>/SCENE_SUMMARY.json + logs/UAV_FM/fm_uav_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
