#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R39 — UAV-pillars for Chapter 6 (tab:uav-pillars-raw, fig:uav-pillars-paths): REAL evaluation, no turbo.
#   The four table cells, evaluated live with the quadrotor as the plant, at the table's protocol:
#     MeanFM K1, MeanFM K2, CI-MeanFM (α_end 0.2) K1, CI-MeanFM K2   ×   {diffuser, dpcc-t-tightened}
#     seeds 6–10 × 3 geometries × 2 episodes (30 flights per variant per cell)          tag p23uavpv2live
#
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh           # PLAN (default): list the 4 jobs, submit nothing
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh submit    # submit the 4 GPU jobs (one per engine × K)
#   overrides:  CELLS="mf:1 af:1"  TAG=p23uavpv2live  bash ... submit
# Job script: Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh (tracked). Spec: data_status/PENDING_20260923_… §2.
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
cd "$REPO"
MODE="${1:-plan}"
case "$MODE" in plan|submit) ;; *) echo "[FAIL] mode must be plan | submit (got '$MODE')"; exit 1 ;; esac
CELLS="${CELLS:-mf:1 mf:2 af:1 af:2}"
export TAG="${TAG:-p23uavpv2live}"
JOB=Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh

fail=0
case "$(echo "$TAG" | tr 'A-Z' 'a-z')" in *uav*) ;; *) echo "[FAIL] TAG='$TAG' must contain 'uav' (uav_avoiding_bridge/factory.py)"; fail=1 ;; esac
[ -f "$JOB" ]                                                          || { echo "[FAIL] $JOB missing"; fail=1; }
[ -f config/meanflow_projection_eval_u18_live.yaml ]                   || { echo "[FAIL] config/meanflow_projection_eval_u18_live.yaml missing"; fail=1; }
[ -f config/alphaflow_projection_eval_u18_live.yaml ]                  || { echo "[FAIL] config/alphaflow_projection_eval_u18_live.yaml missing"; fail=1; }
grep -q "make_avoiding_env" FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py  || { echo "[FAIL] MeanFM eval lacks the plant hook"; fail=1; }
grep -q "make_avoiding_env" FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py || { echo "[FAIL] CI-MeanFM eval lacks the plant hook"; fail=1; }
[ "$fail" -eq 0 ] || { echo "ABORT — pull first."; exit 1; }
echo "[ ok ] pre-flight passed (tag=$TAG)"

echo; echo "checkpoints:"
compgen -G "logs/avoiding-d3il/flow_matching_v3_meanflow/*bbunet*/6" >/dev/null && echo "  [ ok ] MeanFM bbunet" || echo "  [MISS] MeanFM bbunet (logs/avoiding-d3il/flow_matching_v3_meanflow/*bbunet*/<seed>)"
compgen -G "logs/avoiding-d3il/flow_matching_v3_alphaflow/*bbunet*ae0.2*/6" >/dev/null && echo "  [ ok ] CI-MeanFM bbunet ae0.2" || echo "  [MISS] CI-MeanFM bbunet ae0.2"

echo; echo "jobs (${MODE}):"
n=0
for c in $CELLS; do
    e="${c%%:*}"; k="${c##*:}"; n=$((n + 1))
    printf '  %-2s %-3s K=%s  seeds 6-10 x 3 geometries x 2 episodes, diffuser + dpcc-t-tightened\n' "$n" "$e" "$k"
    [ "$MODE" = "submit" ] && GO=1 ./Slurm_Codes/submit.sh "$JOB" "$e" "$k"
done
echo
if [ "$MODE" = "plan" ]; then echo "PLAN only — ${n} job(s) listed. Run:  bash $0 submit"; exit 0; fi
cat <<NOTE
Submitted ${n} job(s).
Results: logs/avoiding-d3il/plans/flow_matching_v3_{meanflow,alphaflow}/<train bbunet>/H8_K<k>_…_msg${TAG}/<seed>/results/halfspace_<geo>/{diffuser,dpcc-t-tightened}.npz
Plant sidecars: logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/${TAG}/<engine>_K<k>/
Read per job: the eval's Success rate / Constraints satisfied blocks and the [ uav-plant ] sidecar lines.
Record the job ids in SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md §5.
NOTE
