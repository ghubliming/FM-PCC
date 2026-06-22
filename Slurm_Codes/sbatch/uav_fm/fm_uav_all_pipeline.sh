#!/bin/bash
#SBATCH --job-name=uav_fm_all_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — ONE-SHOT per-scene: for each (scene,seed) train→eval chained, then a single
# aggregate job after ALL evals. Fully batched (no login-session steps). Per-scene models only —
# NOT the pooled --scene all (which is underdetermined for a state-only FM; see U2 PLAN §0).
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

echo "================================================================================"
echo "UAV-FM ALL PIPELINE (per-scene)  $(date)"
echo "scenes=[$SCENES]  seeds=[$SEEDS]  n_trials=$NTRIALS  proj=$PROJ"
echo "[ note ] requires the curated dataset (data/uav_fm/v1/) — run Aux-1 prepare first."
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

EVAL_DEP="afterok"
for scene in $SCENES; do
    for seed in $SEEDS; do
        T_ID=$(sbatch --parsable $LOG_OPTS "$DIR/train_fm_uav.sh" "$scene" "$seed")
        E_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${T_ID} \
               "$DIR/eval_fm_uav.sh" "$scene" "$seed" "$NTRIALS" "$PROJ")
        echo "  scene=$scene seed=$seed   train=$T_ID → eval=$E_ID"
        EVAL_DEP="${EVAL_DEP}:${E_ID}"
    done
done

AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$EVAL_DEP" "$DIR/aggregate_summaries.sh" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
echo "When done: logs/uav-<scene>/SCENE_SUMMARY.json + logs/fm_uav_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
