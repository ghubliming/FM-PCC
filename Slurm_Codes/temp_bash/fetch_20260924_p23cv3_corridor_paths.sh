#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# p23cv3 · FETCH — the flown paths the corridor-v3 DA still lacks (Gen15 U19 / R33), as one tar.gz
#
#   bash Slurm_Codes/temp_bash/fetch_20260924_p23cv3_corridor_paths.sh          # PLAN (default): list + verify, write nothing
#   bash Slurm_Codes/temp_bash/fetch_20260924_p23cv3_corridor_paths.sh pack     # write export_tmp/p23cv3_paths_<stamp>.tar.gz
#
# WHY: the tables of the DA come from batch_uav_20260924_081422 (complete). What that CSV cannot give — the flown path of
# every flight (npz obs_all / plans) for the altitude numbers, the residue/stall diagnostics and the paths / altitude
# figures — is missing locally for 52 cells: every CI-MeanFM cell (the 23-09 copy arrived as 0-byte files), MeanFM K3
# (arrived empty) and the four diffusion cells that finished after the copy.
#
# WHAT (nothing else): per cell `<variant>.npz` + `results.json`; per eval folder `run_provenance*.json` and
# `config_snapshot_uav_mix/uav_projection.yaml` (the geometry check). No diagnostics, png, svg, gif or rollout logs.
#   CI-MeanFM (af, bbunet, ae0.2)  K1, K2, K3  × tilt + hump   all variants   32 cells
#   MeanFM    (mf, bbunet, dp0.5)  K3          × tilt + hump   all variants   16 cells
#   Diffusion K20                  tilt dpcc-c; hump dpcc-t, dpcc-r, dpcc-c                      4 cells
# ≈ 220 MiB before compression.
#
# LOCAL (after download): tar -xzf p23cv3_paths_<stamp>.tar.gz -C temp/23-09-Corridor-TEMP     # overwrites the 0-byte files
#   then: python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_grid.py
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
case "$MODE" in plan|pack) ;; *) echo "[FAIL] mode must be plan | pack (got '$MODE')"; exit 1 ;; esac
MARK=p23cv3
ROOT=logs/UAV_MIX/uav-corridor
[ -d "$ROOT/plans" ] || { echo "[FAIL] $ROOT/plans not found under $REPO"; exit 1; }

AF=plans/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet
MF=plans/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet
DF=plans/mix_uav_diffusion/H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20
STACK=-bounds_free-pdes-tightened

# eval folder | variant filter ('*' = every variant folder under 6/corridor_cv3*/)
SPECS=(
  "$AF/Eaf_K1_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3t|*"
  "$AF/Eaf_K2_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3t|*"
  "$AF/Eaf_K3_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3t|*"
  "$AF/Eaf_K1_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3ah|*"
  "$AF/Eaf_K2_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3ah|*"
  "$AF/Eaf_K3_mpc4_pid_stopgo_T0.5_EPlatest_p23cv3ah|*"
  "$MF/Emf_K3_mpc4_pid_stopgo_T0.5_p23cv3t|*"
  "$MF/Emf_K3_mpc4_pid_stopgo_T0.5_p23cv3ah|*"
  "$DF/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23cv3t|dpcc-c$STACK"
  "$DF/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23cv3ah|dpcc-t$STACK"
  "$DF/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23cv3ah|dpcc-r$STACK"
  "$DF/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23cv3ah|dpcc-c$STACK"
)

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p export_tmp
LIST="export_tmp/${MARK}_paths_${STAMP}.filelist"
MANI="export_tmp/${MARK}_paths_${STAMP}_MANIFEST.txt"
: > "$LIST"
fail=0; ncell=0; bytes=0
declare -A SEEN_EVAL
for spec in "${SPECS[@]}"; do
    ev="${spec%%|*}"; filt="${spec#*|}"
    [ -d "$ROOT/$ev/6" ] || { echo "[FAIL] missing eval folder: $ev/6"; fail=1; continue; }
    if [ -z "${SEEN_EVAL[$ev]:-}" ]; then
        SEEN_EVAL[$ev]=1
        for f in "$ROOT/$ev"/6/run_provenance*.json "$ROOT/$ev"/6/config_snapshot_uav_mix/uav_projection.yaml; do
            [ -s "$f" ] && echo "${f#$ROOT/}" >> "$LIST"
        done
    fi
    shopt -s nullglob
    dirs=( "$ROOT/$ev"/6/corridor_cv3*/$filt/ )
    shopt -u nullglob
    [ "${#dirs[@]}" -gt 0 ] || { echo "[FAIL] no variant folder matches $ev/6/corridor_cv3*/$filt"; fail=1; continue; }
    for d in "${dirs[@]}"; do
        d="${d%/}"; v="$(basename "$d")"
        for f in "$d/$v.npz" "$d/results.json"; do
            if [ ! -s "$f" ]; then echo "[FAIL] missing or empty: ${f#$ROOT/}"; fail=1; continue; fi
            echo "${f#$ROOT/}" >> "$LIST"; bytes=$((bytes + $(stat -c %s "$f")))
        done
        ncell=$((ncell + 1))
        printf '  %-3s %s\n' "$ncell" "${d#$ROOT/plans/}" | sed 's|H8_Dmodels\.[^/]*/||'
    done
done
echo
echo "cells: ${ncell} (expected 52) · files: $(wc -l < "$LIST") · $(awk -v b=$bytes 'BEGIN{printf "%.0f MiB", b/1048576}') of npz + json"
[ "$fail" -eq 0 ] || { echo "ABORT — a file is missing or empty (see [FAIL] above); nothing packed."; rm -f "$LIST"; exit 1; }
[ "$ncell" -eq 52 ] || echo "[WARN] expected 52 cells, found ${ncell} — check the list above before packing"

if [ "$MODE" = "plan" ]; then
    rm -f "$LIST"
    echo "PLAN only — nothing written. Run:  bash $0 pack"
    exit 0
fi

( cd "$ROOT" && md5sum $(cat "$REPO/$LIST") ) > "$MANI"
OUT="export_tmp/${MARK}_paths_${STAMP}.tar.gz"
tar -czf "$OUT" -C "$ROOT" -T "$LIST" -C "$REPO/export_tmp" "$(basename "$MANI")"
n_in=$(tar -tzf "$OUT" | wc -l)
echo "[ ok ] $OUT  ($(du -h "$OUT" | cut -f1), ${n_in} entries incl. manifest)"
echo "       md5 $(md5sum "$OUT" | cut -d' ' -f1)"
echo "       local: tar -xzf $(basename "$OUT") -C temp/23-09-Corridor-TEMP   # → plans/… in place; the manifest lands beside plans/"
rm -f "$LIST"
