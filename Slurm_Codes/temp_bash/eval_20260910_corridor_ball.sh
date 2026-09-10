#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gen15 U11 — first run of the `corridor_ball` geometry, 3 engines, TWO TIERS
#
#   Tier A   K=2   PCC only            4 variants   (HardFlow is degenerate here)
#   Tier B   K=5   PCC + HF-SLSQP      7 variants   (HardFlow genuine, attributable)
#
#   3 engines x 2 tiers = 6 jobs, seed 6, n=10.
#
# WHY TWO K. HardFlow's guidance lives only in ACTIVE NON-TERMINAL ODE steps:
#     n_genuine = max(K - int(A*K), 1) - 1        with the shipped A = 0.5
#   K=2 -> max(2-1,1)-1 = 0   DEGENERATE. The arm runs no HardFlow arithmetic and the
#                             eval BLOCKS it. Listing a hardflow_* name at K=2 exits 2.
#   K=5 -> max(5-2,1)-1 = 2   genuine, and >=1 is the bar for an attributable effect.
#   So the PCC-only question is asked at the cheap budget and the HardFlow question at
#   the cheapest budget that can legitimately carry it. K is inference-only for the flow
#   family, so both tiers load the SAME corridor checkpoints -- no retraining.
#
# WHY THESE VARIANTS
#   diffuser                  unprojected control      -> MUST violate if the ball binds
#   dpcc-t-geo_free           projects, geometry OFF   -> negative control, should violate
#   dpcc-t / dpcc-t-tightened the PCC projector working  (`-t` is best at every K: T1, T2)
#   hardflow_new              plain HF
#   hardflow_new-t            the MATCHED pair against dpcc-t -- the headline comparison
#   hardflow_new-t-geo_free   HF with geometry OFF     -> symmetric negative control
#
# BASELINE: none needed. corridor_hg K=2 for all three engines is already in
# batch_uav_20260910_092309 (C30/C44/C38). `geo_tag_suffix: _hgb` puts these runs in a
# sibling geo folder under the SAME candidate, so a DA pairs them on the `geo` axis.
# (corridor_hg has no K=5 arm, so Tier B is new geometry AND new budget -- read it as a
#  within-corridor_ball comparison, not against corridor_hg.)
#
# Run from anywhere in the repo:
#     bash Slurm_Codes/eval_20260910_corridor_ball.sh
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

GEO=corridor_ball

# Tier A — K=2, PCC only. No hardflow_* names: the eval would exit 2 at n_genuine=0.
VA=( diffuser dpcc-t-geo_free dpcc-t dpcc-t-tightened )
# Tier B — K=5, PCC + HF-SLSQP, matched selectors.
VB=( diffuser dpcc-t dpcc-t-tightened dpcc-t-geo_free
     hardflow_new hardflow_new-t hardflow_new-t-geo_free )

join() { local IFS=,; echo "$*"; }

check() {                          # $1=label  $2=K  $3..=names
    local label="$1" k="$2"; shift 2
    local list=("$@") bad=0
    for v in "${list[@]}"; do
        case "$v" in
            *[[:space:]]*) echo "[FAIL] $label: whitespace inside '$v'"; bad=1 ;;
            ""|*--*|*-)    echo "[FAIL] $label: malformed '$v'";        bad=1 ;;
        esac
        if [ "$k" -lt 3 ] && case "$v" in hardflow*) true;; *) false;; esac; then
            echo "[FAIL] $label: '$v' is degenerate at K=$k (n_genuine=0) -- the eval will exit 2"; bad=1
        fi
    done
    [ "$bad" -eq 0 ] || { echo "ABORT — not submitting."; exit 1; }
    echo "[ ok ] $label (K=$k): ${#list[@]} names — $(join "${list[@]}")"
}

grep -q "name: ${GEO}\b" config/uav_projection.yaml \
    || { echo "[FAIL] '${GEO}' not in config/uav_projection.yaml -- pull U11 first."; exit 1; }
echo "[ ok ] geo = ${GEO}"
check "Tier A · PCC only"      2 "${VA[@]}"
check "Tier B · PCC + HF-SLSQP" 5 "${VB[@]}"
echo

submit () {                        # $1=engine  $2=K  $3=variant csv  $4..=extra env
    local eng="$1" k="$2" vars="$3"; shift 3
    echo "=== ${eng}  corridor  K=${k}  geo=${GEO}"
    env UAV_EVAL_HOURS=24 \
        FMPCC_SAFE_EPS_MODE=scaled \
        FMPCC_UAV_EVAL_TAG=u7hg \
        UAV_MIX_GEO_VARIANTS="${GEO}" \
        UAV_MIX_VARIANTS="${vars}" \
        "$@" \
        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh "$eng" corridor "6" "$k"
    echo
}

AF_KNOBS=( UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest )

echo "############ TIER A — K=2, PCC only ############"
submit af 2 "$(join "${VA[@]}")" "${AF_KNOBS[@]}"
submit mf 2 "$(join "${VA[@]}")"
submit fm 2 "$(join "${VA[@]}")"

echo "############ TIER B — K=5, PCC + HF-SLSQP ############"
submit af 5 "$(join "${VB[@]}")" "${AF_KNOBS[@]}"
submit mf 5 "$(join "${VB[@]}")"
submit fm 5 "$(join "${VB[@]}")"

cat <<'NOTE'
Submitted 6 wrappers (3 engines x 2 tiers). First-run checks, in the CHILD eval logs:

  1  [ U11 ] geo variants for 'corridor': ['corridor_ball']
  2  [ eval ] E9 geo 'corridor' <- variant 'corridor_ball': ... (bounds=True, hs=2, obs=5)
                                                                              ^^^^^ 4 caps + ball
  3  a results path containing  corridor_hgb_
  4  Tier B only: NO "[hardflow][BLOCKED] ... DEGENERATE" line, and hf_n_genuine = 2
  5  *** THE ONE THAT DECIDES THE TEST ***
     `diffuser` must now report n_violations > 0. If it is still 0.00 the ball is not
     binding and the geometry needs revisiting before anything else is read.

  grep -hE "U11 . geo|E9 geo|BLOCKED|variant=diffuser \(B=" \
      Slurm_Codes/logs/$(date +%F)/*uav_mix_eval*.log
NOTE
