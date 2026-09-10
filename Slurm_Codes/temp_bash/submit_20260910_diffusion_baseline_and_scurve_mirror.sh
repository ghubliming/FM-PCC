#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Gen15 — close the UAV benchmark matrix: pillars diffusion baseline + the two
#         engines missing from s_curve under `u7hg` (af, diffusion)
#
#   1  pillars  diffusion   TRAIN + EVAL   (no pillars diffusion checkpoint exists)
#   2  s_curve  diffusion   EVAL only      (GaussianDiffusion_9D_K20 already trained)
#   3  s_curve  af  K=5     EVAL only, part A — DPCC     (AlphaFlowODE_..._bbunet exists)
#   4  s_curve  af  K=5     EVAL only, part B — HardFlow
#
# WHY ONLY ONE TRAIN. The `u7hg` change is a CONSTRAINT-SET change (config/uav_projection.yaml),
# not a model change, so every existing checkpoint stays valid and the missing s_curve rows are
# re-evals, not retrains. Only `pillars` has no diffusion checkpoint at all.
#
# WHY DIFFUSION CARRIES NO K LIST. K is an EVAL-time knob for fm/mf/af but a TRAINING parameter
# for DPCC diffusion -- the checkpoint is literally `..._K20`. Passing an empty flow_steps arg
# runs it at the plan-block K (=20), which is the `avoiding-d3il` baseline setup. This arm is a
# REFERENCE ROW for the reader's scale, not a matched-budget claim: label it as such.
#
# WHY af ON s_curve IS SPLIT IN TWO. mf K=10 on s_curve with 8 variants took 17.7 h; af runs
# ~2x fm per variant and the wall is 24 h. Part B carries `dpcc-t-geo_free` because the eval
# refuses a HardFlow-only subset (needs a dpcc row at the same K) -- it is the cheapest legal
# key at 1514 s on this scene, and it is new data (never in the U9 subset).
#
# EXPECTATION MANAGEMENT: max S&C across all 40 existing `u7hg` s_curve cells is 0.100. These
# two rows are for benchmark-matrix completeness -- "all four engines, identical geometry" --
# not because they are expected to rank. See CLOSURE_20260910 §3.
#
# Run from anywhere in the repo:
#     bash Slurm_Codes/submit_20260910_diffusion_baseline_and_scurve_mirror.sh
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

join() { local IFS=,; echo "$*"; }
check() {                          # $1=label  $2..=names
    local label="$1"; shift
    local list=("$@") bad=0
    for v in "${list[@]}"; do
        case "$v" in
            *[[:space:]]*) echo "[FAIL] $label: whitespace inside '$v'"; bad=1 ;;
            ""|*--*|*-)    echo "[FAIL] $label: malformed '$v'";        bad=1 ;;
        esac
    done
    # HardFlow needs a dpcc-* companion at the same K, else the eval exits 2
    local hf=0 pcc=0
    for v in "${list[@]}"; do
        case "$v" in hardflow*) hf=1 ;; diffuser) ;; *) pcc=1 ;; esac
    done
    [ "$hf" -eq 1 ] && [ "$pcc" -eq 0 ] && { echo "[FAIL] $label: HardFlow-only subset"; bad=1; }
    [ "$bad" -eq 0 ] || { echo "ABORT — not submitting."; exit 1; }
    echo "[ ok ] $label (${#list[@]}): $(join "${list[@]}")"
}

# DPCC family only for the baseline reference rows: `-r` is the only unsafe family and is never
# the best cell (T3), and HardFlow is irrelevant to a DPCC-diffusion reference.
DIFF=( diffuser dpcc-c dpcc-t dpcc-c-tightened dpcc-t-tightened )
# af s_curve mirrors the U9 subset the other s_curve arms used, so the rows are comparable.
AF_A=( diffuser dpcc-r dpcc-c dpcc-t )
AF_B=( dpcc-t-geo_free hardflow_new hardflow_new-r hardflow_new-c hardflow_new-t )

check "diffusion reference" "${DIFF[@]}"
check "af s_curve part A"   "${AF_A[@]}"
check "af s_curve part B"   "${AF_B[@]}"
echo

BASE=( UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=u7hg )
AF_KNOBS=( UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest )

echo "### 1/4  pillars · diffusion · TRAIN + EVAL (plan-block K=20)"
env "${BASE[@]}" UAV_MIX_VARIANTS="$(join "${DIFF[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_ksweep_pipeline.sh \
        diffusion pillars 6 "" fm_only none ""
echo

echo "### 2/4  s_curve · diffusion · EVAL only (plan-block K=20)"
env "${BASE[@]}" UAV_MIX_VARIANTS="$(join "${DIFF[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh \
        diffusion s_curve 6 "" fm_only none ""
echo

echo "### 3/4  s_curve · af K=5 · part A — DPCC"
env "${BASE[@]}" "${AF_KNOBS[@]}" UAV_MIX_VARIANTS="$(join "${AF_A[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af s_curve "6" "5"
echo

echo "### 4/4  s_curve · af K=5 · part B — HardFlow (+1 dpcc guard key)"
env "${BASE[@]}" "${AF_KNOBS[@]}" UAV_MIX_VARIANTS="$(join "${AF_B[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af s_curve "6" "5"
echo

cat <<'NOTE'
Submitted 4. Checks when the logs land:

  pillars diffusion : train job succeeded before the eval fired (dependency chain)
                      results path contains  Ediffusion_K20_  and  pillars_hg_
  s_curve  diffusion: results path contains  Ediffusion_K20_  and  s_curve_hg_
  s_curve  af       : [ U6 ] af bone = unet / af_alpha_end = 0.2 / checkpoint = latest
                      results path contains  Eaf_K5_..._EPlatest_u7hg  and  s_curve_hg_
                      part B: NO "[hardflow][BLOCKED] ... DEGENERATE"  (K=5 -> n_genuine=2)

  If part 4 hits the 24 h wall, resume the unfinished variants only -- do NOT restart it.
NOTE
