#!/bin/bash
#SBATCH --job-name=u18_turbo        # Gen15 U18 — Mode T: replay stored avoiding executions through the quadrotor plant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00             # pilot: minutes. MODE=all (whole corpus): expect < 1 h
#SBATCH --partition=gpu-1-student   # no --gres: CPU-only job (MuJoCo physics + numpy; no network, no NLP, no render)

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Usage (from the repo root on the cluster):
#      ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh            # PLAN ONLY (dry-run)
#      GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh       # run the PILOT (gate G1)
#      GO=1 MODE=all ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh   # the whole corpus
#      GO=1 MODE=paper ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh # P1 paper set (10 cells, tag p23pv2turbo)
#  Knobs (env): MODE=pilot|all|paper|papergif  REPLAYS="clock settle"  HZ=1  VMAX=1.0  GRACE_S=2  SCALE=36  LIMIT=0 (episodes/cell)
#               GIF=N (overhead MuJoCo GIF for the first N episodes per cell; GPU needed -> submit turbo_gif.sh)
#               EXTRA="…" appended to every turbo.py call (e.g. --force, --no-png, --max-cells 5)
#  Pilot = gate G1 of the U18 plan: FM K20 extended cell (msg20trials), seed 6, all three geometries,
#          `diffuser` + `dpcc-r-tightened`, 20 episodes, replayed with BOTH policies (clock, settle).
#  Outputs (fix2): logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/<engine>/<train>/<eval>_msguavpv2s10turbo/… (clock)
#  and …turboset/… (settle) — the UAV-pillars scene folder, avoiding-style layout below it; per run a JSON in
#  …/avoiding_bridge/_uav_turbo_runs/. OUT_ROOT=… overrides. Idempotent: existing cells are skipped.
# ──────────────────────────────────────────────────────────────────────────────────────────────
set -eo pipefail

CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)   NAME: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   NODE: $(hostname)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "JOB END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$PYTHONPATH"
export MPLBACKEND="agg"
FORESIGHT="${FORESIGHT:-3}"   # foresight SVGs per cell (matplotlib, CPU)
GIF="${GIF:-0}"
if [ "$GIF" -gt 0 ]; then
    # GIF recording needs a GL context: EGL pinned to the allocated GPU, exactly as the UAV eval jobs do.
    # Submit through turbo_gif.sh (it carries --gres=gpu:1); this branch only sets the environment.
    export MUJOCO_GL="egl"
    export PYOPENGL_PLATFORM="egl"
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
    echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
    if [ -z "$MUJOCO_EGL_DEVICE_ID" ]; then echo "[ u18 ] GIF>0 but no GPU allocated: use turbo_gif.sh"; exit 1; fi
else
    # CPU job, nothing is rendered: tell MuJoCo to skip GL entirely. Without it `import mujoco` walks the
    # glfw -> egl -> osmesa fallback chain and dies in PyOpenGL (first pilot, job 26067).
    export MUJOCO_GL="disable"
    unset PYOPENGL_PLATFORM
fi
cd "$REPO"

MODE="${MODE:-pilot}"
GO="${GO:-0}"
REPLAYS="${REPLAYS:-clock settle}"
HZ="${HZ:-1}"          # fix3: clock mode = 1 setpoint/s at scale 36 (reference rate-limited to VMAX)
VMAX="${VMAX:-1.0}"       # fix3: reference rate limit; feed-forward OFF (EXTRA="--ff" to opt in, unstable > 0.5 m/s)
GRACE_S="${GRACE_S:-2}"
LIMIT="${LIMIT:-0}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-$REPO/logs/UAV_MIX/uav-pillars/plans/avoiding_bridge}"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-36}"   # fix3: 36 = rod radius (0.01) -> drone reach (0.36)
STAG="s${FMPCC_AVOID_UAV_SCALE}"

# regenerate the scene for this scale (idempotent; asserts the frame<->scene match at plant load)
python uav_avoiding_bridge/scene.py

case "$MODE" in
  pilot)
    SELECT=(--engine flow_matching_v3_ode_selectable --eval-glob '*K20*msg20trials' --seeds 6
            --geos both-hard top-left-hard top-right-hard --variants diffuser dpcc-r-tightened) ;;
  all)
    SELECT=() ;;
  paper|papergif)
    # P1 of PENDING_20260923 §2.3: the representative set at DPCC's protocol. Five calls; folder names are the exact
    # Full_Path entries of the 19-09 batch CSV. `papergif` = only T1 and T4 (the two cells the thesis pictures):
    # run it through turbo_gif.sh (GPU) with GIF=2 BEFORE `paper`, which then skips those two cells as done.
    SELECT=() ;;
  *) echo "MODE must be pilot|all|paper|papergif"; exit 2 ;;
esac
P23_COMMON=(--seeds 6 7 8 9 10 --geos top-left-hard top-right-hard both-hard)
P23_CALLS=(
  "T1|--engine flow_matching_v3_meanflow --train-glob *bbunet* --eval-glob H8_K1_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE --variants diffuser dpcc-t-tightened"
  "T2|--engine flow_matching_v3_alphaflow --train-glob *bbunet*ae0.2* --eval-glob *K1_*msgdpccproto --variants diffuser dpcc-t-tightened"
  "T3|--engine flow_matching_v3_ode_selectable --eval-glob H8_K1_*msgdpccproto --variants diffuser dpcc-t-tightened"
  "T4|--engine diffusion --eval-glob H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5 --variants diffuser dpcc-c-tightened"
)
P23_T5="--engine flow_matching_v3_meanflow --train-glob *bbunet* --eval-glob H8_K3_*A1_B4*msghfmink* --seeds 7 8 9 10 --geos top-left-hard top-right-hard both-hard --variants dpcc-t-tightened hardflow_sls-t-tightened"

DRY=(--dry-run); [ "$GO" = "1" ] && DRY=()
echo "[ u18 ] out_root=$OUT_ROOT"
echo "[ u18 ] MODE=$MODE GO=$GO replays='$REPLAYS' hz=$HZ vmax=$VMAX grace=${GRACE_S}s scale=$FMPCC_AVOID_UAV_SCALE limit=$LIMIT extra='$EXTRA'"
run_turbo() {   # $1 = tag, $2 = replay, rest = selection
  local TAG="$1" R="$2"; shift 2
  python uav_avoiding_bridge/turbo.py "$@" --tag "$TAG" --replay "$R" --hz "$HZ" \
      --grace-s "$GRACE_S" --vmax "$VMAX" --limit-episodes "$LIMIT" --out-root "$OUT_ROOT" --gif "$GIF" --foresight "$FORESIGHT" "${DRY[@]}" $EXTRA
}
if [ "$MODE" = "paper" ] || [ "$MODE" = "papergif" ]; then
  set -f                                              # the selections carry fnmatch globs (*bbunet*): no shell expansion
  TAG="${TAG:-p23pv2turbo}"; R="${REPLAYS%% *}"      # one replay policy for the paper set (clock unless overridden)
  echo "[ u18 ] MODE=$MODE tag=$TAG replay=$R"
  for entry in "${P23_CALLS[@]}"; do
    name="${entry%%|*}"; sel="${entry#*|}"
    if [ "$MODE" = "papergif" ] && [ "$name" != "T1" ] && [ "$name" != "T4" ]; then continue; fi
    echo "--------------------------------------------------------------------------------"; echo "[ u18 ] $name: $sel"
    # shellcheck disable=SC2086
    run_turbo "$TAG" "$R" "${P23_COMMON[@]}" $sel
  done
  if [ "$MODE" = "paper" ]; then
    echo "--------------------------------------------------------------------------------"; echo "[ u18 ] T5: $P23_T5"
    # shellcheck disable=SC2086
    run_turbo "$TAG" "$R" $P23_T5
  fi
  set +f
else
  for R in $REPLAYS; do
    TAG="uavpv2${STAG}turbo"; [ "$R" = "settle" ] && TAG="uavpv2${STAG}turboset"
    echo "--------------------------------------------------------------------------------"
    echo "[ u18 ] replay=$R  tag=$TAG"
    echo "--------------------------------------------------------------------------------"
    run_turbo "$TAG" "$R" "${SELECT[@]}"
  done
fi
[ "$GO" = "1" ] || echo "[ u18 ] dry-run only. Re-submit with GO=1 to fly."
