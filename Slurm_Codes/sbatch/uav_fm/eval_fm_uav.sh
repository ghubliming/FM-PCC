#!/bin/bash
#SBATCH --job-name=uav_fm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-1-student
set -e

# ---- Args: $1=scene (def all)  $2=seeds (quoted, def "6")  $3=n_trials (omit → yaml default)  $4=projection (def fm_only)  $5=record (none|gif|all, def none) ----
# Seeds are looped INSIDE this one job allocation — never submit one sbatch job per seed.
# If you add seeds, bump --time (above, or via `sbatch --time=...` override) proportionally.
# n_trials: omit $3 (or pass "") → reads n_trials from config/uav_projection.yaml.
#           pass an int          → CLI override wins over yaml.
SCENE="${1:-all}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10" or uncomment below.
SEEDS="${2:-6}"
# SEEDS="${2:-6 7 8 9 10}"   # full run (5 seeds)
NTRIALS="${3:-}"             # empty = let config/uav_projection.yaml n_trials apply
PROJECTION="${4:-fm_only}"
RECORD="${5:-none}"          # 'gif'/'all' → overhead GIFs per rollout (slower); 'none' = fast

CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "SCENE: $SCENE   SEEDS: $SEEDS   N_TRIALS: $NTRIALS"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"
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
fi

cd "$REPO"
for seed in $SEEDS; do
    echo "--------------------------------------------------------------------------------"
    echo "[ uav_fm_eval ] scene=$SCENE seed=$seed  $(date)"
    echo "[ uav_fm_eval ] python FM_v3_uav_test/eval_fm_uav.py --scene $SCENE --seed $seed ${NTRIALS:+--n-trials $NTRIALS} --projection $PROJECTION --record $RECORD"
    python FM_v3_uav_test/eval_fm_uav.py --scene "$SCENE" --seed "$seed" \
        ${NTRIALS:+--n-trials "$NTRIALS"} --projection "$PROJECTION" --record "$RECORD"
done
echo "Job completed successfully. Evaluated scene=$SCENE for seeds=[$SEEDS]"
