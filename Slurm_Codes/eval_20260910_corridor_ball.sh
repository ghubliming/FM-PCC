#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gen15 U11 — first run of the `corridor_ball` geometry, 3 engines, K=2
#
#   af / mf / fm  x  4 variants  x  10 trials  x  seed 6   (~0.5-2 h per job)
#
# WHY K=2: the only budget where all three engines have a `u7hg` corridor
# checkpoint (C30 af, C44 mf, C38 fm), so it is the matched point. HardFlow is
# degenerate at K=2 (n_genuine=0) and auto-blocks -- this test is about the DPCC
# projector, and dropping HardFlow is what makes it fast.
#
# WHY THESE 4 VARIANTS:
#   diffuser          unprojected control      -> MUST violate if the ball binds
#   dpcc-t-geo_free   projects, no geometry    -> negative control, should also violate
#   dpcc-t            projector doing real work
#   dpcc-t-tightened  strongest projection
# `dpcc-t*` is the best selector at every K on corridor (T2) and pillars (T1).
#
# BASELINE: none needed. corridor_hg at K=2 for all three engines is already in
# batch_uav_20260910_092309 (C30/C44/C38) with all four of these variants at n=10.
# `geo_tag_suffix: _hgb` puts the new runs in a sibling geo folder under the SAME
# candidate, so a DA pairs them on the `geo` axis directly.
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
V=( diffuser dpcc-t-geo_free dpcc-t dpcc-t-tightened )
join() { local IFS=,; echo "$*"; }
VARIANTS="$(join "${V[@]}")"

# ── pre-flight: the geometry must exist and must carry 5 obstacles (4 caps + ball) ──
grep -q "name: ${GEO}\b" config/uav_projection.yaml \
    || { echo "[FAIL] '${GEO}' not found in config/uav_projection.yaml -- pull U11 first."; exit 1; }
for v in "${V[@]}"; do
    case "$v" in *[[:space:]]*|""|*--*|*-) echo "[FAIL] malformed variant '$v'"; exit 1 ;; esac
done
echo "[ ok ] geo = ${GEO}"
echo "[ ok ] variants (${#V[@]}) = ${VARIANTS}"
echo

submit () {                       # $1=engine  $2..=extra env assignments
    local eng="$1"; shift
    echo "=== ${eng}  corridor K=2  geo=${GEO}"
    env UAV_EVAL_HOURS=24 \
        FMPCC_SAFE_EPS_MODE=scaled \
        FMPCC_UAV_EVAL_TAG=u7hg \
        UAV_MIX_GEO_VARIANTS="${GEO}" \
        UAV_MIX_VARIANTS="${VARIANTS}" \
        "$@" \
        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh "$eng" corridor "6" "2"
    echo
}

# af needs its three U6 knobs to resolve the same checkpoint as C30 (_EPlatest_u7hg).
submit af UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest
# mf and fm ride the default `best` checkpoint -> tag `u7hg`, matching C44 / C38.
submit mf
submit fm

cat <<'NOTE'
Submitted 3 wrappers. First-run checks, in the CHILD eval logs:

  1  [ U11 ] geo variants for 'corridor': ['corridor_ball']
  2  [ eval ] E9 geo 'corridor' <- variant 'corridor_ball': ... (bounds=True, hs=2, obs=5)
                                                                              ^^^^^ 4 caps + ball
  3  a results path containing  corridor_hgb_
  4  *** THE ONE THAT DECIDES THE TEST ***
     the `diffuser` row must now report n_violations > 0. If it is still 0.00 the
     ball is not binding and the geometry needs revisiting before anything is read.

  grep -hE "U11 . geo|E9 geo|variant=diffuser \(B=" \
      Slurm_Codes/logs/$(date +%F)/*uav_mix_eval*.log
NOTE
