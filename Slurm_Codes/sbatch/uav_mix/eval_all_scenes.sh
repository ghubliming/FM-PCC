#!/bin/bash
#SBATCH --job-name=uav_mix_eval_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
#
# Gen11 E6 U2 — outer loop: ONE eval job PER SCENE (seeds loop internally inside
# eval_fm_uav.sh, not as separate sbatch calls), then a final aggregate job (after
# all evals) rolls up per-scene + cross-scene summaries. Thin submitter; exits immediately.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_all_scenes.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_all_scenes.sh "pillars" "6 7 8 9 10" 20 fm_only
# Args: $1=engine (fm|mf|af|diffusion) [fm]  $2=scenes [all 4]  $3=seeds ["6"]  $4=n_trials [omit → yaml default]
#       $5=projection [fm_only]  $6=K / flow_steps [omit → plan-block value]
# n_trials: omit $4 → reads from config/uav_projection.yaml; pass int → CLI override.
# 🔴 K: when comparing arms, pass the SAME K to every arm (matched budget or nothing).
set -e

ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENES="${2:-empty corridor s_curve pillars}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10".
SEEDS="${3:-6}"
NTRIALS="${4:-}"   # empty = let config/uav_projection.yaml n_trials apply
PROJ="${5:-fm_only}"
FLOW_STEPS="${6:-}"   # empty = use the plan block's flow_steps_v3
EVAL="Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh"
AGG="Slurm_Codes/sbatch/uav_mix/aggregate_summaries.sh"

N_SEEDS=$(echo $SEEDS | wc -w)
EVAL_HOURS=$((N_SEEDS * 8))

echo "================================================================================"
echo "UAV-MIX EVAL ALL (per-scene, seed-loop internal)  $(date)   engine=$ENGINE  scenes=[$SCENES]  seeds=[$SEEDS]  n_trials=${NTRIALS:-'yaml default'}  proj=$PROJ  K=${FLOW_STEPS:-'plan block'}"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

DEP="afterok"
for scene in $SCENES; do
    ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS "$EVAL" "$ENGINE" "$scene" "$SEEDS" "$NTRIALS" "$PROJ" "none" "$FLOW_STEPS")
    echo "  eval   engine=$ENGINE scene=$scene seeds=[$SEEDS]  → Job $ID"
    DEP="${DEP}:${ID}"
done

# Final roll-up after all evals succeed.
AGG_ID=$(sbatch --parsable $LOG_OPTS --dependency="$DEP" "$AGG" "$ENGINE" "$SCENES" "$PROJ")
echo "  aggregate (after all evals)  → Job $AGG_ID"
echo "--------------------------------------------------------------------------------"
echo "Read: logs/UAV_MIX/uav-<scene>/SCENE_SUMMARY.json  +  logs/UAV_MIX/uav_mix_ALL_SCENES_SUMMARY.json"
echo "================================================================================"
