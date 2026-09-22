#!/bin/bash
#SBATCH --job-name=p23_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
# ─────────────────────────────────────────────────────────────────────────────
# P23 corridor-v3 MASTER — the paper waves one after another, never all in squeue at once.
#
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh start [C1]   # submit the master for the first wave
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh plan         # dry: what each link would submit
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh status       # the chain + this wave's jobs in squeue
#   scancel -n p23_master_C3   (the PENDING link shown by `status`)                  # stop the chain; running eval jobs continue
#
# HOW. One tiny CPU job per wave (10 min wall, no GPU). It submits its wave through the paper driver
# (eval_20260923_p23_corridor_v3.sh, P23_DIRECT=1: every eval job submitted directly, ids recorded), then submits
# ITSELF for the next wave with `--dependency=afterany:<all ids of this wave>` and exits. So at any time squeue
# holds: the running wave's eval jobs + ONE pending master (reason: Dependency). C1 → C2 → C3 → C4 → C5.
# `afterany`, not `afterok`: one failed cell must not block the rest of the grid (re-run it alone afterwards).
# A wave with nothing to submit (e.g. a missing checkpoint) is skipped in the same link.
#
# Same env / variants / cells as `… corridor_v3.sh submit all` (runbook §1–§2): 68 cells per scene, both scenes.
# Overrides are read at `start` and forwarded along the chain: GEOS, SEEDS, NTRIALS, RECORD.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

find_repo_root() {
    local d
    for d in "$PWD" "$(cd "$(dirname "$0")" && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
        [ -n "$d" ] || continue
        while [ "$d" != "/" ] && [ -n "$d" ]; do
            if [ -f "$d/Slurm_Codes/submit.sh" ]; then echo "$d"; return 0; fi
            d="$(dirname "$d")"
        done
    done
    return 1
}
REPO="$(find_repo_root)" || { echo "[FAIL] no Slurm_Codes/submit.sh above \$PWD / this script / SLURM_SUBMIT_DIR."; exit 1; }
cd "$REPO"

SELF="Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh"
DRIVER="Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh"
ORDER=(C1 C2 C3 C4 C5)
MODE="${1:-status}"
WAVE="${2:-C1}"
export GEOS="${GEOS:-corridor_v3_tilt corridor_v3_ablation_hump}" SEEDS="${SEEDS:-6}" NTRIALS="${NTRIALS:-12}" RECORD="${RECORD:-none}"

next_wave() {  # $1=wave -> the following wave, or '' after C5
    local i
    for i in "${!ORDER[@]}"; do
        if [ "${ORDER[$i]}" = "$1" ]; then echo "${ORDER[$((i + 1))]:-}"; return 0; fi
    done
    echo "[FAIL] unknown wave '$1'" >&2; return 1
}
submit_master() {  # $1=wave $2=dependency ids ':'-joined or ''  -> prints the master job id
    local w="$1" dep="$2" DATE TIME LOG_DIR ID
    DATE=$(date +%Y-%m-%d); TIME=$(date +%H_%M_%S); LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
    local depopt=()
    [ -n "$dep" ] && depopt=( --dependency="afterany:${dep}" )
    ID=$(sbatch --parsable --job-name="p23_master_${w}" "${depopt[@]}" \
         --output="$LOG_DIR/${TIME}_%x_%j.log" --error="$LOG_DIR/${TIME}_%x_%j.log" \
         --export=ALL,GEOS="$GEOS",SEEDS="$SEEDS",NTRIALS="$NTRIALS",RECORD="$RECORD",SUBMIT_TIME="$TIME",SUBMIT_DATE="$DATE" \
         "$SELF" run "$w")
    echo "${ID%%;*}"
}

case "$MODE" in
    plan)
        echo "CHAIN plan: ${ORDER[*]} — each link = one 10-min master job that submits its wave and queues the next link behind it."
        for w in "${ORDER[@]}"; do
            echo; echo "──── link ${w} would submit:"
            P23_DIRECT=1 bash "$DRIVER" plan "$w" | sed -n '/════ wave/,$p' | grep -v "^PLAN only\|^Run:"
        done
        ;;
    start)
        bash "$DRIVER" plan "$WAVE" >/dev/null || { echo "[FAIL] the driver's pre-flight failed — fix that first"; exit 1; }
        ID="$(submit_master "$WAVE" "")"
        echo "[ ok ] chain started at ${WAVE}: master job ${ID}  (scenes: ${GEOS}; seeds ${SEEDS}; ${NTRIALS} flights)"
        echo "       watch: bash $SELF status      stop the chain: scancel -n p23_master_<pending link>   (see status)"
        ;;
    status)
        echo "masters:";   squeue -u "$USER" -h -o '  %.10i %-16j %.3t %.10M %R' --name="$(IFS=,; echo "${ORDER[*]/#/p23_master_}")" 2>/dev/null || true
        echo "eval jobs:"; squeue -u "$USER" -h -o '  %.10i %-16j %.3t %.10M %R' --name=uav_mix_eval 2>/dev/null || true
        echo "(the eval jobs above include any non-P23 uav_mix_eval jobs of yours)"
        ;;
    run)
        # ── inside the Slurm master job ──
        echo "════════ P23 master · wave ${WAVE} · $(date) · job ${SLURM_JOB_ID:-?} ════════"
        w="$WAVE"
        while [ -n "$w" ]; do
            IDS_FILE="$(mktemp "${TMPDIR:-/tmp}/p23_${w}_ids.XXXXXX")"
            echo; echo "submitting wave ${w} …"
            P23_DIRECT=1 P23_JOBIDS_FILE="$IDS_FILE" bash "$DRIVER" submit "$w"
            mapfile -t IDS < <(grep -E '^[0-9]+$' "$IDS_FILE" || true)
            rm -f "$IDS_FILE"
            nxt="$(next_wave "$w")"
            if [ "${#IDS[@]}" -eq 0 ]; then
                echo "wave ${w}: nothing submitted (no checkpoint?) → going straight to ${nxt:-<end>}"
                w="$nxt"; continue
            fi
            echo "wave ${w}: ${#IDS[@]} eval job(s): ${IDS[*]}"
            if [ -z "$nxt" ]; then
                echo "wave ${w} was the last link. Chain complete once these jobs finish. Then run the DA per tag (driver's NOTE)."
                break
            fi
            dep="$(IFS=:; echo "${IDS[*]}")"
            NID="$(submit_master "$nxt" "$dep")"
            echo "queued link ${nxt} as master job ${NID}, dependency afterany:${#IDS[@]} job(s)"
            break
        done
        echo "════════ master link ${WAVE} done · $(date) ════════"
        ;;
    *) echo "[FAIL] mode must be start [wave] | plan | status | run <wave>"; exit 1 ;;
esac
