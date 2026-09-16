#!/bin/bash
#
# TEMP (2026-09-16) — UAV 5-seed replication: submit training ONLY for the missing seeds.
# PENDING_20260916 R1. Paper arms × 3 scenes × seeds 6–10:
#   fm        → FlowMatchingODE_9D
#   mf        → MeanFlowODE_9D_dp0.5_bbunet
#   af        → AlphaFlowODE_9D_as1_ae0.2_bbunet   (UAV_MIX_BONE_AF=unet, UAV_MIX_AF_ALPHA_END=0.2)
#   diffusion → GaussianDiffusion_9D_K20            (K20 only)
# Scenes: corridor pillars s_curve (corridor_v2 is an eval-side geometry; training uses 'corridor').
#
# A (engine, scene, seed) is skipped when logs/UAV_MIX/uav-<scene>/mix_uav_<engine>/<exp>/<seed>/state_best.pt
# already exists (checked on the cluster at submit time). As of the 16-09 tree only seed 6 exists.
#
# Thin login-node submitter (like train_all_scenes.sh) — run with bash, NOT via submit.sh:
#   bash Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh           # dry run: prints the plan
#   bash Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh --submit  # actually sbatch
#
# ONE seed per job, --time=24:00:00. At most MAX_PARALLEL (=2) jobs run at once: all jobs are split
# into 2 lanes, each lane a single --dependency=afterany chain. Jobs are ordered SEED-FIRST (all seed-7
# jobs, then seed 8, …) so a stopped campaign still leaves complete seed sets; each job goes to the
# lane with the smaller expected load. afterany: a failed/timed-out job does not block its lane.
# Expected h/seed from the seed-6 checkpoint timestamps: fm ~2.7, diffusion ~2.5, af(unet) ~3, mf ~7.5.
set -e

SUBMIT=0; [ "$1" = "--submit" ] && SUBMIT=1

ENGINES="fm mf af diffusion"
SCENES="corridor pillars s_curve"
SEEDS="6 7 8 9 10"
JOB="Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh"
LOGROOT="logs/UAV_MIX"

exp_dir() {   # engine -> checkpoint folder name
    case "$1" in
        fm)        echo "H8_Dmodels.diffusion.FlowMatchingODE_9D" ;;
        mf)        echo "H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet" ;;
        af)        echo "H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet" ;;
        diffusion) echo "H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20" ;;
    esac
}
TIME_LIMIT="24:00:00"
MAX_PARALLEL=2
tenths_per_seed() {  # expected hours ×10, for lane balancing only
    case "$1" in fm) echo 27 ;; mf) echo 75 ;; af) echo 30 ;; diffusion) echo 25 ;; esac
}

[ -f "$JOB" ] || { echo "[ ERROR ] run from repo root ($JOB not found)"; exit 1; }

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

echo "================================================================================"
echo "UAV 5-SEED MISSING-TRAIN  $(date)  mode=$([ $SUBMIT = 1 ] && echo SUBMIT || echo DRY-RUN)"
echo "================================================================================"

declare -a LANE_PREV LANE_LOAD LANE_N
for ((l=0; l<MAX_PARALLEL; l++)); do LANE_PREV[$l]=""; LANE_LOAD[$l]=0; LANE_N[$l]=0; done

n_jobs=0
for s in $SEEDS; do
  for engine in $ENGINES; do
    EXP=$(exp_dir "$engine")
    EXTRA_EXPORT=""
    [ "$engine" = "af" ] && EXTRA_EXPORT=",UAV_MIX_BONE_AF=unet,UAV_MIX_AF_ALPHA_END=0.2"
    for scene in $SCENES; do
        ckpt="$LOGROOT/uav-$scene/mix_uav_$engine/$EXP/$s/state_best.pt"
        if [ -f "$ckpt" ]; then
            echo "  skip   $engine/$scene seed $s  (exists)"
            continue
        fi
        lane=0
        for ((l=1; l<MAX_PARALLEL; l++)); do
            [ ${LANE_LOAD[$l]} -lt ${LANE_LOAD[$lane]} ] && lane=$l
        done
        prev="${LANE_PREV[$lane]}"
        DEP=""; [ -n "$prev" ] && DEP="--dependency=afterany:$prev"
        if [ $SUBMIT = 1 ]; then
            id=$(sbatch --parsable --job-name="uav_mix_train_${engine}_${scene}_s${s}" --time="$TIME_LIMIT" $DEP \
                 --export="ALL,SUBMIT_TIME=$TIME,SUBMIT_DATE=$DATE${EXTRA_EXPORT}" $LOG_OPTS \
                 "$JOB" "$engine" "$scene" "$s")
            echo "  lane$lane  $engine/$scene seed=$s ${DEP:-(lane head)} → Job $id"
        else
            id="DRY${n_jobs}"
            echo "  [dry] lane$lane  $id  $engine/$scene seed=$s ${DEP:-(lane head)}"
        fi
        LANE_PREV[$lane]=$id
        LANE_LOAD[$lane]=$(( ${LANE_LOAD[$lane]} + $(tenths_per_seed "$engine") ))
        LANE_N[$lane]=$(( ${LANE_N[$lane]} + 1 ))
        n_jobs=$((n_jobs+1))
    done
  done
done
for ((l=0; l<MAX_PARALLEL; l++)); do
    echo "  lane$l: ${LANE_N[$l]} jobs, ~$(( ${LANE_LOAD[$l]} / 10 )) h expected"
done
echo "--------------------------------------------------------------------------------"
echo "$n_jobs jobs (1 seed each, max $MAX_PARALLEL running at once) $([ $SUBMIT = 1 ] && echo submitted || echo planned — rerun with --submit)"
echo "================================================================================"
