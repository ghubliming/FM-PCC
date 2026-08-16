#!/bin/bash
#SBATCH --job-name=uav_mix_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
set -e

# ---- Args: $1=engine (fm|mf|af, def fm)  $2=scene (def all)  $3=seeds (quoted, def "6")
#            $4=n_trials (omit → yaml default)  $5=projection (def fm_only)
#            $6=record (none|gif|all, def none)  $7=K / flow_steps (omit → plan-block value) ----
# Seeds are looped INSIDE this one job allocation — never submit one sbatch job per seed.
# If you add seeds, bump --time (above, or via `sbatch --time=...` override) proportionally.
# n_trials: omit $4 (or pass "") → reads n_trials from config/uav_projection.yaml.
#           pass an int          → CLI override wins over yaml.
#
# 🔴 K (=$7) is the NFE budget and the primary experimental axis of Gen15. MATCHED BUDGET OR
# NOTHING: when comparing arms, pass the SAME K to every one of them. K appears in the output
# path as K{n}, so distinct-K runs never overwrite each other.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10".
SEEDS="${3:-6}"
NTRIALS="${4:-}"             # empty = let config/uav_projection.yaml n_trials apply
PROJECTION="${5:-fm_only}"
RECORD="${6:-none}"          # 'gif'/'all' → overhead GIFs per rollout (slower); 'none' = fast
FLOW_STEPS="${7:-}"          # empty = use the plan block's flow_steps_v3

CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "ENGINE: $ENGINE   SCENE: $SCENE   SEEDS: $SEEDS   N_TRIALS: $NTRIALS   K: ${FLOW_STEPS:-<plan block>}"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"

# Auto-select conda env from config/uav.py's eval-block 'controller' value — no manual
# flag needed. controller='mjpc' needs mujoco>=3.x / mujoco.mjx, which conflicts with
# the mujoco==2.3.7 pin the rest of the repo (e.g. d3il/avoiding) needs, so it lives in
# an isolated clone env (`conda create -n FMPCC_mjx --clone FMPCC && pip install
# "jax[cuda12]" mujoco-mjx`). Every other controller uses the default FMPCC env.
# See CHANGELOG_U6_mjx_tracker.md.
DETECTED_CONTROLLER=$(awk "/'plan_flow_matching_v3_uav': \{/,0" "$REPO/config/uav.py" \
    | grep -m1 "'controller':" | sed -E "s/.*'controller':[[:space:]]*'([^']*)'.*/\1/")
if [ "$DETECTED_CONTROLLER" = "mjpc" ]; then
    CONDA_ENV_NAME="FMPCC_mjx"
else
    CONDA_ENV_NAME="FMPCC"
fi
echo "[ env-select ] config/uav.py controller='$DETECTED_CONTROLLER' -> conda env '$CONDA_ENV_NAME'"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
# third_party/mujoco_mpc: bundled Python package + generated proto stubs (Fix_5).
# The compiled agent_server binary must be at third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server.
export PYTHONPATH="$REPO:$REPO/third_party/mujoco_mpc:$PYTHONPATH"
# Eval rolls out in MuJoCo on a headless node → EGL required.
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
# Fix_11 follow-up: force UNBUFFERED stdout/stderr. Under SLURM, python's stdout is a file
# (not a TTY) → block-buffered (~4-8KB), so print() breadcrumbs sit in the buffer and never
# reach the .log until it fills or the process exits. That silences Fix_11's whole progress
# trail on a running (or SIGKILL-timed-out) job — the log looks frozen with "nothing shown"
# even while the eval is progressing fine. This makes every print appear live. Equivalent to
# `python -u`; set as an env var so it also covers any child python the script spawns.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# GPU-leak guard (same as all FMPCC jobs): pin EGL to the Slurm-allocated GPU and abort if they diverge.
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

# W&B Login (key file on the cluster) — eval_fm_uav.py doesn't log to W&B today, kept for parity.
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"
# Fix_11: seed x/N breadcrumb — if this 24h job gets killed by the time limit, the last
# "seed X/N" line printed (plus eval_fm_uav.py's own scene/variant/trial progress lines)
# tells you where, instead of the log just going silent with no indication of progress.
read -ra SEED_ARR <<< "$SEEDS"
N_SEEDS=${#SEED_ARR[@]}
SEED_IDX=0
for seed in $SEEDS; do
    SEED_IDX=$((SEED_IDX+1))
    echo "--------------------------------------------------------------------------------"
    echo "[ uav_mix_eval ] seed $SEED_IDX/$N_SEEDS: engine=$ENGINE scene=$SCENE seed=$seed  $(date)"
    echo "[ uav_mix_eval ] python mix_uav_test/eval_mix_uav.py --engine $ENGINE --scene $SCENE --seed $seed ${NTRIALS:+--n-trials $NTRIALS} --projection $PROJECTION --record $RECORD ${FLOW_STEPS:+--flow-steps $FLOW_STEPS}"
    python mix_uav_test/eval_mix_uav.py --engine "$ENGINE" --scene "$SCENE" --seed "$seed" \
        ${NTRIALS:+--n-trials "$NTRIALS"} --projection "$PROJECTION" --record "$RECORD" \
        ${FLOW_STEPS:+--flow-steps "$FLOW_STEPS"}
done
echo "Job completed successfully. Evaluated engine=$ENGINE scene=$SCENE for seeds=[$SEEDS]"
