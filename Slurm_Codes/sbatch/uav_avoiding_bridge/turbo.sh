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
#  Knobs (env): MODE=pilot|all  REPLAYS="clock settle"  HZ=5  GRACE_S=2  SCALE=10  LIMIT=0 (episodes/cell)
#               EXTRA="…" appended to every turbo.py call (e.g. --force, --no-png, --max-cells 5)
#  Pilot = gate G1 of the U18 plan: FM K20 extended cell (msg20trials), seed 6, all three geometries,
#          `diffuser` + `dpcc-r-tightened`, 20 episodes, replayed with BOTH policies (clock, settle).
#  Outputs land beside the sources: <eval>_msguavpv2s10turbo/… (clock) and …turboset/… (settle);
#  per run a JSON in logs/avoiding-d3il/plans/_uav_turbo_runs/. Idempotent: existing cells are skipped.
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
# CPU job, nothing is rendered: tell MuJoCo to skip GL entirely. Without it `import mujoco` walks the
# glfw -> egl -> osmesa fallback chain and dies in PyOpenGL (first pilot, job 26067). The GPU evals keep
# their MUJOCO_GL=egl; this file is the only CPU MuJoCo entrypoint in the repo.
export MUJOCO_GL="disable"
unset PYOPENGL_PLATFORM
cd "$REPO"

MODE="${MODE:-pilot}"
GO="${GO:-0}"
REPLAYS="${REPLAYS:-clock settle}"
HZ="${HZ:-5}"
GRACE_S="${GRACE_S:-2}"
LIMIT="${LIMIT:-0}"
EXTRA="${EXTRA:-}"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-10}"
STAG="s${FMPCC_AVOID_UAV_SCALE}"

# regenerate the scene for this scale (idempotent; asserts the frame<->scene match at plant load)
python uav_avoiding_bridge/scene.py

case "$MODE" in
  pilot)
    SELECT=(--engine flow_matching_v3_ode_selectable --eval-glob '*K20*msg20trials' --seeds 6
            --geos both-hard top-left-hard top-right-hard --variants diffuser dpcc-r-tightened) ;;
  all)
    SELECT=() ;;
  *) echo "MODE must be pilot|all"; exit 2 ;;
esac

DRY=(--dry-run); [ "$GO" = "1" ] && DRY=()
echo "[ u18 ] MODE=$MODE GO=$GO replays='$REPLAYS' hz=$HZ grace=${GRACE_S}s scale=$FMPCC_AVOID_UAV_SCALE limit=$LIMIT extra='$EXTRA'"
for R in $REPLAYS; do
  TAG="uavpv2${STAG}turbo"; [ "$R" = "settle" ] && TAG="uavpv2${STAG}turboset"
  echo "--------------------------------------------------------------------------------"
  echo "[ u18 ] replay=$R  tag=$TAG"
  echo "--------------------------------------------------------------------------------"
  python uav_avoiding_bridge/turbo.py "${SELECT[@]}" --tag "$TAG" --replay "$R" --hz "$HZ" \
      --grace-s "$GRACE_S" --limit-episodes "$LIMIT" "${DRY[@]}" $EXTRA
done
[ "$GO" = "1" ] || echo "[ u18 ] dry-run only. Re-submit with GO=1 to fly."
