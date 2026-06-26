#!/bin/bash
#SBATCH --job-name=uav_fm_collect_to_ready
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 Aux-1 — ONE-SHOT: collect all 4 scenes (E4) → prepare (curate+verify) → READY.
# Fully batched: this thin orchestrator submits the child jobs with SLURM dependencies and
# exits. Nothing runs on your login session, so a disconnect can't kill it.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/collect_to_ready_pipeline.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/collect_to_ready_pipeline.sh 500 gate
# Args: $1=n_trials [500]   $2="gate" to also chain the mini-FM GPU gate after prepare [off]
set -e

N_TRIALS="${1:-500}"
WITH_GATE="${2:-}"
SCENES="empty corridor s_curve pillars"
COLLECT="Slurm_Codes/sbatch/uav_expert_data/collect.sh"
PREP="Slurm_Codes/sbatch/uav_fm_data_ready/prepare_uav_fm_data.sh"
GATE="Slurm_Codes/sbatch/uav_fm_data_ready/mini_fm_gate.sh"

echo "================================================================================"
echo "UAV-FM COLLECT→READY PIPELINE  $(date)   n_trials=$N_TRIALS  gate=${WITH_GATE:-no}"
echo "================================================================================"
echo "[ note ] this re-collects ALL 4 scenes — only use it for a FRESH start."
echo "         If you already have good empty/corridor/s_curve, do NOT run this (it would"
echo "         add to / mix those dirs). Instead collect only pillars + run prepare."
echo "[ note ] to keep the old pillars, ARCHIVE (never delete) before a fresh collect:"
echo "         mv logs/uav_expert_data/pillars logs/uav_expert_data/_archive_pillars_274"

# Unify dated logs across the chained jobs (submit.sh exports SUBMIT_DATE/TIME).
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# 1) one collect job per scene; collect their job IDs into an afterok dependency list.
DEP="afterok"
for s in $SCENES; do
    ID=$(sbatch --parsable $LOG_OPTS "$COLLECT" "$s" "$N_TRIALS")
    DEP="${DEP}:${ID}"
    echo "  collect $s        → Job $ID"
done

# 2) prepare (curate+verify) — runs only if ALL collects succeed.
PREP_ID=$(sbatch --parsable $LOG_OPTS --dependency="$DEP" "$PREP")
echo "  prepare (curate)  → Job $PREP_ID   (after all collects)"

# 3) optional mini-FM gate after prepare.
if [ "$WITH_GATE" = "gate" ]; then
    GATE_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:${PREP_ID} "$GATE" empty)
    echo "  mini-FM gate      → Job $GATE_ID   (after prepare)"
fi

echo "--------------------------------------------------------------------------------"
echo "Submitted. When Job $PREP_ID prints 'DATASET READY ✓', run (per-scene, see U2):"
echo "  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh \"empty corridor s_curve pillars\" \"5 6 7\""
echo "Monitor: squeue -u \$USER"
echo "================================================================================"
