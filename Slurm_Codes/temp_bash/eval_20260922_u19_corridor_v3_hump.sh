#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# U19 — corridor_v3_hump: the WIDE corridor with a constraint on ALTITUDE (plane xz halfspaces)
#
#   bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh          # PLAN  (default): print every job + checkpoint checks, submit NOTHING
#   bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh smoke    # PILOT (gates G1-G3): ONE short job — mf, K=3, 3 trials = L/C/R, GIF on
#   bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh submit   # SUBMIT the full wave (gate G4) — only after the pilot passed
#
#   overrides:  ENGINES="mf fm af diffusion"  SEEDS="6"  NTRIALS=12  RECORD=none  GEO=corridor_v3_hump_lo  bash ... submit
#
# Copy of eval_20260913_u16_corridor_v2_paper.sh with the geometry swapped. Plan + gates:
#   logs_in_develop/Gen15/U19/PLAN_20260922_U19_corridor_v3_z_slide.md
#
# FIXED FOR EVERY ARM (the U16 bench, unchanged except the geometry):
#   scene/geo   corridor_v3_hump  (scene_corridor_v2.xml, walls ±1.0, NO slide, a HUMP in the x-z plane:
#               roof 0 -> 1.10 m at x=0 -> 0 over x in [-1.5, 1.5]; ceiling 2.80; virtual, scored only)
#   normaliser  FMPCC_SAFE_EPS_FRAC=1.0, FMPCC_SAFE_EPS_MODE=scaled
#   projector   geometry on the setpoint (-pdes) + DPCC margin (-tightened), action cap off (-bounds_free)
#               (-bounds_free is what lets z move at all — U12 closure; never drop it here)
#   metrics     exactly as pillars / s_curve / corridor_v2 (strict success, S&C, violations on the real drone)
#   tag         FMPCC_UAV_EVAL_TAG=u19cv3  -> results in E<engine>_K<k>_..._u19cv3/, never mixed with cv2s
#
# ARMS (names keep `-tightened` LAST: DA_UAV_v1 reads it with endswith)
#   diffuser                                      unprojected reference (gate G1: it must VIOLATE the hump)
#   dpcc-{r,c,t}-bounds_free-pdes-tightened       DPCC, random / min-cost / temporal selection
#   hardflow_new-bounds_free-pdes-tightened       HardFlow B=1          (flow engines, K >= 3 only)
#   hardflow_new-t-bounds_free-pdes-tightened     HardFlow B=4, matched to dpcc-t
#
# JOBS PER SEED (same split as U16; one job per K is launched by eval_k_sweep.sh)
#   mf / fm / af   K=1        diffuser + dpcc-r/c/t                (HardFlow is degenerate at K<=2)
#                  K=3 and 5  part A: diffuser + dpcc-r + dpcc-c
#                             part B: dpcc-t + hardflow B=1 + hardflow-t B=4
#   diffusion      K=20 (a TRAINING property — plan block, no K arg)
#                             part A: diffuser + dpcc-r   part B: dpcc-c + dpcc-t   (no HardFlow)
#
# SECOND RUNG: GEO=corridor_v3_hump_lo (H = 0.90) — run ONLY if the pilot fails G2 while G3 passes
# (the climb starts but stops short). Its tag becomes u19cv3lo automatically.
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
case "$MODE" in plan|smoke|submit) ;; *) echo "[FAIL] mode must be plan | smoke | submit (got '$MODE')"; exit 1 ;; esac

GEO="${GEO:-corridor_v3_hump}"
XML=d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_corridor_v2.xml
case "$GEO" in
    corridor_v3_hump)    TAG=u19cv3 ;;
    corridor_v3_hump_lo) TAG=u19cv3lo ;;
    *) echo "[FAIL] GEO must be corridor_v3_hump | corridor_v3_hump_lo (got '$GEO')"; exit 1 ;;
esac
ENGINES="${ENGINES:-mf fm af diffusion}"
SEEDS="${SEEDS:-6}"
NTRIALS="${NTRIALS:-12}"          # 12 = 4 per route (homotopies cycle L, C, R)
RECORD="${RECORD:-none}"          # gif/all renders MuJoCo GIFs for every rollout (slow) — smoke turns it on

PCC_R=dpcc-r-bounds_free-pdes-tightened
PCC_C=dpcc-c-bounds_free-pdes-tightened
PCC_T=dpcc-t-bounds_free-pdes-tightened
HF_1=hardflow_new-bounds_free-pdes-tightened
HF_T=hardflow_new-t-bounds_free-pdes-tightened

# ── pre-flight: the code and config this wave depends on ─────────────────────
fail=0
grep -q "name: ${GEO}\b" config/uav_projection.yaml                || { echo "[FAIL] geo '${GEO}' missing from config/uav_projection.yaml"; fail=1; }
grep -q "plane: xz" config/uav_projection.yaml                      || { echo "[FAIL] config/uav_projection.yaml has no 'plane: xz' halfspace (U19)"; fail=1; }
[ -f "$XML" ]                                                       || { echo "[FAIL] $XML missing"; fail=1; }
grep -q "def _hs_plane" mix_uav_test/eval_mix_uav.py                || { echo "[FAIL] eval_mix_uav.py lacks _hs_plane (U19: plane xz support)"; fail=1; }
grep -q "def _fs_hs_plane" mix_uav_test/eval_artifacts.py           || { echo "[FAIL] eval_artifacts.py lacks _fs_hs_plane (U19: plane xz support)"; fail=1; }
grep -q "_geo_on_pdes" mix_uav_test/eval_mix_uav.py                 || { echo "[FAIL] eval_mix_uav.py lacks the -pdes toggle (U16 fix 1)"; fail=1; }
grep -q "def update_constraint_list" mix_uav/sampling/hardflow_projection.py || { echo "[FAIL] hardflow_projection.py lacks update_constraint_list (U16 fix 2)"; fail=1; }
grep -q "_TOGGLES = ('-pdes'" mix_uav_test/eval_mix_uav.py          || { echo "[FAIL] eval_mix_uav.py lacks composed-toggle allow-list (U16 fix 2)"; fail=1; }
grep -qE "^diffusion_timestep_threshold: 0.5\b" config/uav_projection.yaml || { echo "[FAIL] diffusion_timestep_threshold is not 0.5"; fail=1; }
[ "$fail" -eq 0 ] || { echo "ABORT — pull the U19 commit first."; exit 1; }
echo "[ ok ] pre-flight passed  (geo=${GEO}, tag=${TAG})"

ckpt_dirs() {   # $1=engine $2=seed -> matching checkpoint dirs (training output, not plans/)
    compgen -G "logs/UAV_MIX/uav-corridor/mix_uav_${1}/*/${2}" || true
}

BASE_ENV=( UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_SAFE_EPS_FRAC=1.0
           FMPCC_UAV_EVAL_TAG="${TAG}" UAV_MIX_GEO_VARIANTS="${GEO}" UAV_MIX_GIF_RES=320 )
UNSET=( -u UAV_MIX_BONE_AF -u UAV_MIX_AF_ALPHA_END -u UAV_MIX_EPOCH -u UAV_MIX_CONTROLLER
        -u UAV_MIX_HF_OFF -u FMPCC_HF_ALLOW_DEGENERATE -u UAV_MIX_TRAJ_GIF )
engine_env() {  # extra knobs per engine (af: the corridor af_unet checkpoint of campaign T2)
    case "$1" in af) echo "UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest" ;; *) echo "" ;; esac
}

N_JOBS=0
run() {  # $1=label  $2..=command (after env settings)
    local label="$1"; shift
    N_JOBS=$((N_JOBS + 1))
    printf '  %-3s %s\n' "$N_JOBS" "$label"
    if [ "$MODE" = "plan" ]; then return 0; fi
    "$@"
}

flow_job() {  # $1=engine $2=seed $3="K list" $4=variants(csv) $5=trials $6=record $7=tag
    local e="$1" s="$2" ks="$3" v="$4" n="$5" rec="$6" tag="$7"
    # shellcheck disable=SC2046
    run "${e} seed ${s} K=[${ks}] n=${n} rec=${rec}: ${v}" \
        env "${UNSET[@]}" "${BASE_ENV[@]}" FMPCC_UAV_EVAL_TAG="${tag}" UAV_MIX_VARIANTS="${v}" $(engine_env "$e") \
        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh "$e" corridor "$s" "$ks" "$n" fm_only "$rec"
}

# ── SMOKE = the PILOT (gates G1, G2, G3): one job, three routes, both projector arms ──
if [ "$MODE" = "smoke" ]; then
    STAG="${TAG/u19/u19smoke}"     # u19smokecv3 / u19smokecv3lo
    echo; echo "PILOT — mf, seed 6, K=3, 3 trials (L, C, R), MuJoCo GIF on, tag ${STAG}, geo ${GEO}"
    flow_job mf 6 "3" "diffuser,${PCC_T},${HF_T}" 3 gif "${STAG}"
    cat <<NOTE

  check in the CHILD log:
    UAV_MIX_VARIANTS -> running 3/... variants: ['diffuser', '${PCC_T}', '${HF_T}']
    [ eval ] E9 geo 'corridor' <- variant '${GEO}': ... hs=4, obs=4
    [ U16 ] geo entry '${GEO}': MuJoCo scene .../scene_corridor_v2.xml ... walls ... y=-1.00 ... y=+1.00
  results: logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/<train-id>/Emf_K3_..._${STAG}/6/corridor_cv3h*_bounds+dynamics+geo_bounds+halfspace+obstacles/{diffuser,${PCC_T},hardflow_sls-t-bounds_free-pdes-tightened}/
  gates (plan §3) — run on the cluster env, it only needs numpy:
    python logs_in_develop/Gen15/U19/tools/check_gates_u19.py "logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/*/Emf_K3_*_${STAG}/6/corridor_cv3h*"
      G1  diffuser violates the hump on 3/3          (else the hump does not bind: raise H)
      G2  a projected arm is collision-free >= 2/3 AND succeeds >= 2/3
      G3  paired z (projected - unprojected) at x in [-0.5, 0.5] > 0.15 m on EVERY projected flight
  also read: the per-variant overview PNG (side panel now draws the roof + centre limit) and the foresight SVGs.
  pass -> bash $0 submit        fail G2 with G3 pass -> GEO=corridor_v3_hump_lo bash $0 smoke
NOTE
    exit 0
fi

# ── PLAN / SUBMIT: the full wave (gate G4) ──────────────────────────────────
echo; echo "engines=[${ENGINES}]  seeds=[${SEEDS}]  n_trials=${NTRIALS}  record=${RECORD}  mode=${MODE}  geo=${GEO}"
echo; echo "checkpoints (logs/UAV_MIX/uav-corridor/mix_uav_<engine>/*/<seed>):"
declare -A HAVE
for e in $ENGINES; do
    for s in $SEEDS; do
        d="$(ckpt_dirs "$e" "$s")"
        if [ -n "$d" ]; then HAVE["$e/$s"]=1; echo "  [ ok ] $e seed $s:"; echo "$d" | sed 's/^/           /'
        else HAVE["$e/$s"]=0; echo "  [MISS] $e seed $s: no checkpoint dir"; fi
    done
done

echo; echo "jobs:"
for e in $ENGINES; do
    for s in $SEEDS; do
        if [ "$e" = "diffusion" ]; then
            if [ "${HAVE[$e/$s]}" = "1" ]; then
                for part in "diffuser,${PCC_R}" "${PCC_C},${PCC_T}"; do
                    run "diffusion seed ${s} K=20 (plan block) n=${NTRIALS}: ${part}" \
                        env "${UNSET[@]}" "${BASE_ENV[@]}" UAV_MIX_VARIANTS="${part}" \
                        ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh diffusion corridor "$s" "$NTRIALS" fm_only "$RECORD" ""
                done
            else
                echo "  --  SKIP diffusion seed ${s}: no checkpoint (the corridor diffusion model was trained for U16 — pull logs/ or train: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh diffusion corridor \"${s}\")"
            fi
            continue
        fi
        if [ "${HAVE[$e/$s]}" != "1" ]; then
            echo "  --  SKIP ${e} seed ${s}: no checkpoint (train it first: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh ${e} corridor \"${s}\")"
            continue
        fi
        flow_job "$e" "$s" "1"   "diffuser,${PCC_R},${PCC_C},${PCC_T}" "$NTRIALS" "$RECORD" "$TAG"
        flow_job "$e" "$s" "3 5" "diffuser,${PCC_R},${PCC_C}"          "$NTRIALS" "$RECORD" "$TAG"
        flow_job "$e" "$s" "3 5" "${PCC_T},${HF_1},${HF_T}"             "$NTRIALS" "$RECORD" "$TAG"
    done
done

echo
if [ "$MODE" = "plan" ]; then
    echo "PLAN only — ${N_JOBS} submission(s) listed, nothing submitted."
    echo "(each K list of 2 values launches 2 eval children). Run:  bash $0 smoke   (the pilot)   then   bash $0 submit"
    exit 0
fi
cat <<NOTE
Submitted ${N_JOBS} submission(s).

Results:  logs/UAV_MIX/uav-corridor/plans/mix_uav_<engine>/<train-id>/E<engine>_K<k>_mpc4_pid_stopgo_T0.5_${TAG}/<seed>/corridor_cv3h*_bounds+dynamics+geo_bounds+halfspace+obstacles/<variant>/
DA (after everything finished):
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_uav.sh "\$(ls -d logs/UAV_MIX/uav-corridor/plans/mix_uav_*/*/E*_${TAG} | paste -sd, -)"
NOTE
