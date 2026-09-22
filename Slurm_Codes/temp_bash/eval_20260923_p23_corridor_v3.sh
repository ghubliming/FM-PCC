#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# P23 — UAV-corridor v3 PAPER waves (R33), runbook SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md §1–§2
#
#   TWO scenes (author, 2026-09-23: "two scenes as piloted"), each with the full 68-cell grid:
#     corridor_v3_tilt            the v2 slide leaned −60° (one x-y-z plane: sideways AND down)   tag p23cv3t   folder corridor_cv3t_…
#     corridor_v3_ablation_hump   no slide, an x-z roof to climb over (ceiling 2.80)              tag p23cv3ah  folder corridor_cv3ah_…
#   (The runbook's single combined scene `corridor_v3` / tag p23cv3 is NOT run — see cross_draft/to_v3/FROM_U19_20260923_….md)
#
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh                 # PLAN (default): list every submission, submit NOTHING
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh smoke           # C0 re-pilot (already done 22-09 as u19smoke*, gates in Gen15/U19/PILOT_…md)
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh submit C1       # one wave (C1 | C2 | C3 | C4 | C5), both scenes
#   bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh submit all      # C1 → C2 → C3 → C4 → C5, both scenes, in that order
#
#   overrides:  GEOS="corridor_v3_tilt"  SEEDS="6"  NTRIALS=12  RECORD=none  bash ... submit all
#
# PAPER-ONLY (PENDING_20260923 §0): tightened only, `-bounds_free-pdes-tightened` stack, seed 6, 12 flights (4 per route),
# no untightened twin, no ablation variants. `-tightened` is always the LAST token (DA reads it with endswith).
#
# VARIANTS (runbook §1) — input spelling is `hardflow_new-…` (the name registered in config/uav_mix.py); the eval WRITES
# the folders as `hardflow_sls-…`, which is what the runbook and the DA read.
#   PCC_R/C/T  dpcc-{r,c,t}-bounds_free-pdes-tightened
#   HF_S/R/C/T hardflow_new{,-r,-c,-t}-bounds_free-pdes-tightened          (B=1, B=4 ×3)
#
# WAVES (runbook §2, cells per scene = 68; ONE DEVIATION, forced by the eval's own guard: a job that keeps HardFlow
# variants must also keep at least one dpcc-* row, so the HF jobs carry PCC_T and C2 therefore runs PCC_T only at the
# K where no HF job exists. Every cell is run exactly ONCE; nothing is duplicated, nothing is added.)
#   C1  diffuser:            mf K"1 2 3"   af K"1 2 3"   fm K"1 2 3 5 20"   diffusion K20 (plan block)         4 submissions / 12 children
#   C2  PCC_R,PCC_C,PCC_T:   mf/af/fm K"1 2";   PCC_R,PCC_C: mf/af K"3", fm K"3 5"                              6 submissions / 10 children
#   C3  PCC_T + HF_S,R,C,T:  mf K"3"  af K"3"  fm K"3 5"                                                       3 submissions /  4 children
#   C4  fm K20:              PCC_R,PCC_C  |  PCC_T + HF_S,R,C,T                                                2 submissions /  2 children
#   C5  diffusion K20:       PCC_T, then PCC_R, then PCC_C — one variant per job (up to 24 h each)              3 submissions /  3 children
#   Order = fast first, K20 last, diffusion K20 projected very last. `submit all` interleaves the two scenes per wave.
#
# FIXED ENV on every child: UAV_MIX_GEO_VARIANTS=<scene> FMPCC_UAV_EVAL_TAG=<tag> FMPCC_SAFE_EPS_FRAC=1.0
#   FMPCC_SAFE_EPS_MODE=scaled UAV_EVAL_HOURS=24;  af adds UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest.
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

MODE="${1:-plan}"
WAVE="${2:-all}"
case "$MODE" in plan|smoke|submit) ;; *) echo "[FAIL] mode must be plan | smoke | submit (got '$MODE')"; exit 1 ;; esac
case "$WAVE" in all|C1|C2|C3|C4|C5) ;; *) echo "[FAIL] wave must be all | C1..C5 (got '$WAVE')"; exit 1 ;; esac
[ "$WAVE" = "all" ] && WAVES="C1 C2 C3 C4 C5" || WAVES="$WAVE"

XML=d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_corridor_v2.xml
GEOS="${GEOS:-corridor_v3_tilt corridor_v3_ablation_hump}"
SEEDS="${SEEDS:-6}"
NTRIALS="${NTRIALS:-12}"
RECORD="${RECORD:-none}"

geo_tag() {   # $1=geo -> paper tag
    case "$1" in
        corridor_v3_tilt)          echo p23cv3t ;;
        corridor_v3_ablation_hump) echo p23cv3ah ;;
        *) echo "[FAIL] GEOS may only contain corridor_v3_tilt | corridor_v3_ablation_hump (got '$1')" >&2; return 1 ;;
    esac
}
geo_folder() { case "$1" in corridor_v3_tilt) echo corridor_cv3t_ ;; *) echo corridor_cv3ah_ ;; esac; }
for g in $GEOS; do geo_tag "$g" >/dev/null || exit 1; done

PCC_R=dpcc-r-bounds_free-pdes-tightened
PCC_C=dpcc-c-bounds_free-pdes-tightened
PCC_T=dpcc-t-bounds_free-pdes-tightened
HF_S=hardflow_new-bounds_free-pdes-tightened
HF_R=hardflow_new-r-bounds_free-pdes-tightened
HF_C=hardflow_new-c-bounds_free-pdes-tightened
HF_T=hardflow_new-t-bounds_free-pdes-tightened
HF_ALL="${HF_S},${HF_R},${HF_C},${HF_T}"

# ── pre-flight ───────────────────────────────────────────────────────────────
fail=0
for g in $GEOS; do
    grep -q "name: ${g}\b" config/uav_projection.yaml               || { echo "[FAIL] geo '${g}' missing from config/uav_projection.yaml"; fail=1; }
done
grep -q "z_lean:" config/uav_projection.yaml                        || { echo "[FAIL] yaml has no 'z_lean' halfspace (U19 tilt)"; fail=1; }
grep -q "plane: xz" config/uav_projection.yaml                      || { echo "[FAIL] yaml has no 'plane: xz' halfspace (U19 hump)"; fail=1; }
[ -f "$XML" ]                                                       || { echo "[FAIL] $XML missing"; fail=1; }
grep -q "def _hs_lean" mix_uav_test/eval_mix_uav.py                 || { echo "[FAIL] eval_mix_uav.py lacks _hs_lean (U19)"; fail=1; }
grep -q "def _hs_plane" mix_uav_test/eval_mix_uav.py                || { echo "[FAIL] eval_mix_uav.py lacks _hs_plane (U19)"; fail=1; }
grep -q "def _fs_hs_lean" mix_uav_test/eval_artifacts.py            || { echo "[FAIL] eval_artifacts.py lacks _fs_hs_lean (U19)"; fail=1; }
grep -q "_geo_on_pdes" mix_uav_test/eval_mix_uav.py                 || { echo "[FAIL] eval_mix_uav.py lacks the -pdes toggle (U16 fix 1)"; fail=1; }
grep -q "def update_constraint_list" mix_uav/sampling/hardflow_projection.py || { echo "[FAIL] hardflow_projection.py lacks update_constraint_list (U16 fix 2)"; fail=1; }
grep -q "_TOGGLES = ('-pdes'" mix_uav_test/eval_mix_uav.py          || { echo "[FAIL] eval_mix_uav.py lacks the composed-toggle allow-list (U16 fix 2)"; fail=1; }
for v in "'hardflow_new'" "'hardflow_new-r'" "'hardflow_new-c'" "'hardflow_new-t'"; do
    grep -q "$v" config/uav_mix.py                                  || { echo "[FAIL] config/uav_mix.py does not register $v (the HF input names)"; fail=1; }
done
grep -qE "^diffusion_timestep_threshold: 0.5\b" config/uav_projection.yaml || { echo "[FAIL] diffusion_timestep_threshold is not 0.5"; fail=1; }
[ "$fail" -eq 0 ] || { echo "ABORT — pull the U19 commit first."; exit 1; }
echo "[ ok ] pre-flight passed  (scenes: ${GEOS};  waves: ${WAVES};  mode: ${MODE})"

ckpt_dirs() { compgen -G "logs/UAV_MIX/uav-corridor/mix_uav_${1}/*/${2}" || true; }

BASE_ENV=( UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_SAFE_EPS_FRAC=1.0 UAV_MIX_GIF_RES=320 )
UNSET=( -u UAV_MIX_BONE_AF -u UAV_MIX_AF_ALPHA_END -u UAV_MIX_EPOCH -u UAV_MIX_CONTROLLER
        -u UAV_MIX_HF_OFF -u FMPCC_HF_ALLOW_DEGENERATE -u UAV_MIX_TRAJ_GIF )
engine_env() { case "$1" in af) echo "UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest" ;; *) echo "" ;; esac; }

N_SUB=0; N_CHILD=0
run() {  # $1=label $2=n_children $3..=command
    local label="$1" nc="$2"; shift 2
    N_SUB=$((N_SUB + 1)); N_CHILD=$((N_CHILD + nc))
    printf '  %-3s %s\n' "$N_SUB" "$label"
    if [ "$MODE" = "plan" ]; then return 0; fi
    "$@"
}
flow_job() {  # $1=geo $2=engine $3=seed $4="K list" $5=variants(csv) $6=trials $7=record $8=tag
    local geo="$1" e="$2" s="$3" ks="$4" v="$5" n="$6" rec="$7" tag="$8"
    local nk; nk=$(echo $ks | wc -w)
    # shellcheck disable=SC2046
    run "[${geo}] ${e} seed ${s} K=[${ks}] n=${n}: ${v}" "$nk" \
        env "${UNSET[@]}" "${BASE_ENV[@]}" FMPCC_UAV_EVAL_TAG="${tag}" UAV_MIX_GEO_VARIANTS="${geo}" UAV_MIX_VARIANTS="${v}" $(engine_env "$e") \
        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh "$e" corridor "$s" "$ks" "$n" fm_only "$rec"
}
diff_job() {  # $1=geo $2=seed $3=variants(csv) $4=trials $5=record $6=tag   (diffusion K20 = plan block, no K arg)
    local geo="$1" s="$2" v="$3" n="$4" rec="$5" tag="$6"
    run "[${geo}] diffusion seed ${s} K=20 n=${n}: ${v}" 1 \
        env "${UNSET[@]}" "${BASE_ENV[@]}" FMPCC_UAV_EVAL_TAG="${tag}" UAV_MIX_GEO_VARIANTS="${geo}" UAV_MIX_VARIANTS="${v}" \
        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh diffusion corridor "$s" "$n" fm_only "$rec" ""
}

# ── SMOKE (C0) — the 22-09 pilots already answered G1–G3 (Gen15/U19/PILOT_20260922_U19_gates_G1-G3.md); rerun only on demand ──
if [ "$MODE" = "smoke" ]; then
    echo; echo "C0 — mf, seed 6, K=3, 3 trials (L, C, R), GIF on, per scene:"
    for g in $GEOS; do t="$(geo_tag "$g")"; flow_job "$g" mf 6 "3" "diffuser,${PCC_T},${HF_T}" 3 gif "${t/p23/p23smoke}"; done
    echo; echo "gates: python logs_in_develop/Gen15/U19/tools/check_gates_u19.py --geo tilt|hump \"logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/*/Emf_K3_*_p23smoke<tag>/6/corridor_cv3*\""
    exit 0
fi

# ── checkpoints ──────────────────────────────────────────────────────────────
echo; echo "checkpoints (logs/UAV_MIX/uav-corridor/mix_uav_<engine>/*/<seed>):"
declare -A HAVE
for e in mf af fm diffusion; do for s in $SEEDS; do
    d="$(ckpt_dirs "$e" "$s")"
    if [ -n "$d" ]; then HAVE["$e/$s"]=1; echo "  [ ok ] $e seed $s"; else HAVE["$e/$s"]=0; echo "  [MISS] $e seed $s: no checkpoint dir (its jobs are SKIPPED)"; fi
done; done
need() { [ "${HAVE[$1/$2]:-0}" = "1" ] || { echo "  --  SKIP $1 seed $2 (no checkpoint)"; return 1; }; }

# ── WAVES ────────────────────────────────────────────────────────────────────
for w in $WAVES; do
    echo; echo "════ wave ${w} ════"
    for g in $GEOS; do
        tag="$(geo_tag "$g")"
        for s in $SEEDS; do
            case "$w" in
                C1)
                    need mf "$s" && flow_job "$g" mf "$s" "1 2 3"      "diffuser" "$NTRIALS" "$RECORD" "$tag"
                    need af "$s" && flow_job "$g" af "$s" "1 2 3"      "diffuser" "$NTRIALS" "$RECORD" "$tag"
                    need fm "$s" && flow_job "$g" fm "$s" "1 2 3 5 20" "diffuser" "$NTRIALS" "$RECORD" "$tag"
                    need diffusion "$s" && diff_job "$g" "$s" "diffuser" "$NTRIALS" "$RECORD" "$tag"
                    ;;
                C2)
                    for e in mf af fm; do
                        need "$e" "$s" || continue
                        flow_job "$g" "$e" "$s" "1 2" "${PCC_R},${PCC_C},${PCC_T}" "$NTRIALS" "$RECORD" "$tag"
                        [ "$e" = fm ] && k3="3 5" || k3="3"
                        flow_job "$g" "$e" "$s" "$k3" "${PCC_R},${PCC_C}"          "$NTRIALS" "$RECORD" "$tag"
                    done
                    ;;
                C3)
                    for e in mf af fm; do
                        need "$e" "$s" || continue
                        [ "$e" = fm ] && k3="3 5" || k3="3"
                        flow_job "$g" "$e" "$s" "$k3" "${PCC_T},${HF_ALL}" "$NTRIALS" "$RECORD" "$tag"
                    done
                    ;;
                C4)
                    need fm "$s" || continue
                    flow_job "$g" fm "$s" "20" "${PCC_R},${PCC_C}"  "$NTRIALS" "$RECORD" "$tag"
                    flow_job "$g" fm "$s" "20" "${PCC_T},${HF_ALL}" "$NTRIALS" "$RECORD" "$tag"
                    ;;
                C5)
                    need diffusion "$s" || continue
                    for v in "$PCC_T" "$PCC_R" "$PCC_C"; do diff_job "$g" "$s" "$v" "$NTRIALS" "$RECORD" "$tag"; done
                    ;;
            esac
        done
    done
done

echo
if [ "$MODE" = "plan" ]; then
    echo "PLAN only — ${N_SUB} submission(s) = ${N_CHILD} eval job(s) listed, nothing submitted (per scene: 18 submissions / 31 eval jobs / 68 cells)."
    echo "Run:  bash $0 submit all      or one wave:  bash $0 submit C1"
    exit 0
fi
cat <<NOTE
Submitted ${N_SUB} submission(s) = ${N_CHILD} eval job(s) for waves [${WAVES}], scenes [${GEOS}].

Results:  logs/UAV_MIX/uav-corridor/plans/mix_uav_<engine>/<train-id>/E<engine>_K<k>_mpc4_pid_stopgo_T0.5_<tag>/<seed>/corridor_cv3{t,ah}_bounds+dynamics+geo_bounds+halfspace+obstacles/<variant>/
Completion (runbook §4): 68 result folders per scene; projection_health.n_tripped_trials = 0; divergence.n_aborted_trials per cell.
DA per scene, tag-only, never pooled:
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_uav.sh "\$(ls -d logs/UAV_MIX/uav-corridor/plans/mix_uav_*/*/E*_p23cv3t  | paste -sd, -)"
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_uav.sh "\$(ls -d logs/UAV_MIX/uav-corridor/plans/mix_uav_*/*/E*_p23cv3ah | paste -sd, -)"
Record the job ids in SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md §5.
NOTE
