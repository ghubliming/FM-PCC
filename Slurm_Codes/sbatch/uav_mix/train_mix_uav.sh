#!/bin/bash
#SBATCH --job-name=uav_mix_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
set -e

# ---- Args: $1=engine (fm|mf|af, def fm)  $2=scene (all|empty|corridor|s_curve|pillars, def all)
#            $3=seeds (quoted, space-sep, def "6") ----
# Gen15: `engine` is the ML objective — fm (Gen11 flow matching) | mf (Gen3v6 MeanFlow) |
# af (Gen3v7 alpha-Flow). It selects the config block AND the checkpoint tree, so the three
# arms never collide. Seeds are looped INSIDE this one job allocation — never submit one
# sbatch job per seed. If you add seeds, bump --time proportionally.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af) ;; *) echo "[ ERROR ] engine must be fm|mf|af (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
# Default single seed=6 for testing. For the full multi-seed run pass "6 7 8 9 10".
SEEDS="${3:-6}"

# ---- Pro-logging ----
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "ENGINE: $ENGINE   SCENE: $SCENE   SEEDS: $SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"
function on_exit { echo "JOB END: $(date)"; }
trap on_exit EXIT

# ---- Environment ----
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# UAV FM needs only the repo on the path — NO D3IL (the UAV data loader has no d3il dep).
export FMPCC="$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"
# Headless MuJoCo (not used by train, harmless; kept for parity with eval).
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

# W&B Login (key file on the cluster) — enables --use-wandb below.
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"
for seed in $SEEDS; do
    echo "--------------------------------------------------------------------------------"
    echo "[ uav_mix_train ] engine=$ENGINE scene=$SCENE seed=$seed  $(date)"
    echo "[ uav_mix_train ] python mix_uav_test/train_mix_uav.py --engine $ENGINE --scene $SCENE --seed $seed --use-wandb --wandb-project FM-PCC-uav-mix --wandb-group uav-$ENGINE-$SCENE"
    python mix_uav_test/train_mix_uav.py --engine "$ENGINE" --scene "$SCENE" --seed "$seed" \
        --use-wandb --wandb-project FM-PCC-uav-mix --wandb-group "uav-$ENGINE-$SCENE"
done
echo "Job completed successfully. Trained engine=$ENGINE scene=$SCENE for seeds=[$SEEDS]"
