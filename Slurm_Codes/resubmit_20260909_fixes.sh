#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Resubmit the two unfinished Gen15 UAV arms — 2026-09-09
#
#   Job 1  af pillars K=5   12 variants   ~11-14 h   closes mission 1 + mission 3's af leg
#   Job 2  fm  s_curve K=20  2 variants   ~4.7 h     closes mission 4
#
# Written because pasting the long UAV_MIX_VARIANTS string kept breaking: the
# terminal inserts a line break around column ~160, and `.strip()` in
# mix_uav_test/eval_mix_uav.py only trims the ENDS of a name, never the middle.
# A broken name (`dpcc-r-\n  geo_free` or `dpcc-r-  geo_free`) fails the variant
# check and the whole job exits 2. Killed jobs 25555, 25582 that way.
#
# Run from the repo root on the cluster:
#     bash Slurm_Codes/resubmit_20260909_fixes.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Locate the repo root, wherever this script happens to sit ────────────────
# Walk up from the current directory first, then from the script's own directory,
# looking for Slurm_Codes/submit.sh. Works whether you run it from the repo root,
# from a subdirectory, or with the script copied somewhere else in the tree.
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
REPO="$(find_repo_root)" || {
    echo "[FAIL] could not find Slurm_Codes/submit.sh above \$PWD or above this script."
    echo "       cd to the repo root and re-run."
    exit 1
}
cd "$REPO"
echo "[ ok ] repo root: $REPO"

# ── Job 1 variant list — as an array, so no long line exists anywhere ────────
V1=(
  dpcc-t                      # n=5 partial from 25501 -> must be re-run to n=10
  dpcc-t-tightened
  dpcc-r-geo_free
  dpcc-c-geo_free
  dpcc-t-geo_free
  hardflow_new
  hardflow_new-r
  hardflow_new-c
  hardflow_new-t
  hardflow_new-r-geo_free
  hardflow_new-c-geo_free
  hardflow_new-t-geo_free
)
V2=( dpcc-t-geo_free hardflow_new-t )   # dpcc-* row required by the HardFlow guard

join() { local IFS=,; echo "$*"; }

# ── Self-check: catch a mangled name before Slurm ever sees it ───────────────
check() {
    local label="$1" want="$2"; shift 2
    local list=("$@") bad=0
    [ "${#list[@]}" -eq "$want" ] || { echo "[FAIL] $label: ${#list[@]} names, expected $want"; bad=1; }
    for v in "${list[@]}"; do
        case "$v" in
            *[[:space:]]*) echo "[FAIL] $label: whitespace inside '$v'"; bad=1 ;;
            ""|*--*|*-)    echo "[FAIL] $label: malformed '$v'";        bad=1 ;;
        esac
    done
    [ "$bad" -eq 0 ] || { echo "ABORT — not submitting."; exit 1; }
    echo "[ ok ] $label: $want clean names"
}

check "job 1 (af pillars K5)" 12 "${V1[@]}"
check "job 2 (fm s_curve K20)" 2 "${V2[@]}"
echo

# ── Job 1 — af pillars K=5 ───────────────────────────────────────────────────
echo "=== Job 1: af pillars K=5 — $(join "${V1[@]}")"
UAV_EVAL_HOURS=24 \
FMPCC_SAFE_EPS_MODE=scaled \
FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_BONE_AF=unet \
UAV_MIX_AF_ALPHA_END=0.2 \
UAV_MIX_EPOCH=latest \
UAV_MIX_VARIANTS="$(join "${V1[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "5"
echo

# ── Job 2 — fm s_curve K=20 ──────────────────────────────────────────────────
# NOTE: no UAV_MIX_EPOCH here. This arm was evaluated on the `best` checkpoint;
# setting it would append `_EPlatest` to the eval tag and split the arm into a
# folder that does not match candidate C92.
echo "=== Job 2: fm s_curve K=20 — $(join "${V2[@]}")"
UAV_EVAL_HOURS=24 \
FMPCC_SAFE_EPS_MODE=scaled \
FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_VARIANTS="$(join "${V2[@]}")" \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm s_curve "6" "20"
echo

echo "Both submitted. Verify the knobs actually landed:"
echo "  grep -hE 'af bone|af_alpha_end|checkpoint|EVAL_TAG|variant subset|time per eval' \\"
echo "    Slurm_Codes/logs/\$(date +%F)/*eval_k_sweep*.log"
