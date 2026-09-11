#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gen15 U12 — `corridor_ball_v2` ACTION-BOUND PROBE   (1 job, ~12 min)
#
#   mf · corridor · K=2 · geo=corridor_ball_v2 · seed 6 · n=10 · 2 variants
#
# THE QUESTION: why did the projector not move the plan around the ball?
#   `dpcc-t`             action-magnitude bound ON  (as shipped)
#   `dpcc-t-bounds_free` action-magnitude bound OFF, geometry + dynamics still ON
#                        -> `bounds_free` skips ONLY that family (eval_mix_uav.py:1259)
#
# HYPOTHESIS UNDER TEST. The corridor expert flies a straight line at constant y AND z, so
# dy/dz have zero variance in training; `action_bounds='auto'` reads that range and caps the
# projector at ~2.2e-05 m/step in both axes = 8.7 mm over a 396-step episode, against the
# ~0.43 m detour the ball demands.
#
#   dpcc-t-bounds_free  collision_free > 0  -> hypothesis CONFIRMED; the action cap is the
#                                              blocker, fix is an explicit `action_bounds`
#   dpcc-t-bounds_free  collision_free = 0  -> hypothesis WRONG; look elsewhere
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
VARIANTS=diffuser,dpcc-t,dpcc-t-bounds_free

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
