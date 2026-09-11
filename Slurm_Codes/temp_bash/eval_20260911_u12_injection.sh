#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gen15 U12 — `corridor_ball_v2` FAST INJECTION TEST   (1 job, ~30 min)
#
#   mf · corridor · K=2 · geo=corridor_ball_v2 · seed 6 · n=10 · 2 variants
#
# TWO GATES, both decisive — read them before running the full wave:
#   1  `diffuser`  must report  n_violations > 0        -> the ball still binds
#   2  `dpcc-t`    must report  collision_free_completed > 0
#                  -> something actually routed around it. U11 scored 0.000 on
#                     32 of 34 cells because its escape slot was 0.08 m; U12
#                     opens 0.93 m. If this is still 0.000 the slot is STILL
#                     unreachable: stop, do not run the full wave.
#
# This script sets every knob itself and CLEARS the ones it does not use, so a
# polluted login shell cannot leak into the job. (On 2026-09-11 a mangled paste
# left UAV_MIX_GEO_VARIANTS=corridor_ball_v2export and FMPCC_UAV_EVAL_TAG=u7hgexport
# exported in the shell — this script is immune to that.)
#
#     bash Slurm_Codes/temp_bash/eval_20260911_u12_injection.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

find_repo_root() {
    local d
    for d in "$PWD" "$(cd "$(dirname "$0")" && pwd)"; do
        while [ "$d" != "/" ] && [ -n "$d" ]; do
            if [ -f "$d/Slurm_Codes/submit.sh" ]; then echo "$d"; return 0; fi
            d="$(dirname "$d")"
        done
    done
    return 1
}
REPO="$(find_repo_root)" || { echo "[FAIL] no Slurm_Codes/submit.sh above \$PWD or this script."; exit 1; }
cd "$REPO"; echo "[ ok ] repo root: $REPO"

GEO=corridor_ball_v2
VARIANTS=diffuser,dpcc-t

grep -q "name: ${GEO}\b" config/uav_projection.yaml \
    || { echo "[FAIL] '${GEO}' not in config/uav_projection.yaml -- pull U12 first."; exit 1; }
echo "[ ok ] geo      = ${GEO}"
echo "[ ok ] variants = ${VARIANTS}"
echo

# env -u clears anything this run must NOT inherit; the assignments override the rest.
env -u UAV_MIX_BONE_AF -u UAV_MIX_AF_ALPHA_END -u UAV_MIX_EPOCH \
    -u UAV_MIX_CONTROLLER -u UAV_MIX_HF_OFF -u FMPCC_HF_ALLOW_DEGENERATE \
    UAV_EVAL_HOURS=24 \
    FMPCC_SAFE_EPS_MODE=scaled \
    FMPCC_UAV_EVAL_TAG=u7hg \
    UAV_MIX_GEO_VARIANTS="${GEO}" \
    UAV_MIX_VARIANTS="${VARIANTS}" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf corridor "6" "2"

cat <<'NOTE'

Submitted. When the CHILD log appears:

  grep -hE "U11 . geo|E9 geo|variant=(diffuser|dpcc-t) \(B=" \
      Slurm_Codes/logs/$(date +%F)/*uav_mix_eval*.log

  expect:  [ U11 ] geo variants for 'corridor': ['corridor_ball_v2']
           E9 geo ... variant 'corridor_ball_v2' ... (bounds=True, hs=2, obs=5)
           results path containing  corridor_hgb2_

  GATE 1   diffuser  n_violations > 0
  GATE 2   dpcc-t    collision_free_completed > 0   <- the one that decides U12
NOTE
