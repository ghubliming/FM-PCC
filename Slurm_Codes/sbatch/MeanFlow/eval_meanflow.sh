#!/bin/bash
#SBATCH --job-name=mf_eval
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit
#SBATCH --partition=gpu-1-student   # Updated from sinfo output

# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
# 1) Create a shortcut to the latest log for easy monitoring
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB NAME:  $SLURM_JOB_NAME"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "No GPU detected or nvidia-smi failed"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
echo "================================================================================"

# Trap for JOB END
function on_exit {
    echo "================================================================================"
    echo "JOB END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# 1) Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

# 2) Initialize Conda
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 3) Set Environment Variables
export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"

# Rendering variables for MuJoCo on headless remote nodes
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

# W&B Login (Colab-style from key file)
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

# 4) Run MeanFlow (Gen3v6) Evaluation
cd "$REPO"

# Evaluation reads seeds and variants from config/meanflow_projection_eval.yaml.
#
# 🔵 U9 — ⚠️ MATCHED BUDGET OR NOTHING (PLAN §7 / fix_7.3 §9). The entire Gen13 claim died because
# one hard-coded k_steps=10 made the decisive control unrunnable, and the confound survived four
# rounds of analysis. Gen3v7 (AlphaFlow) has had the K grid built into its eval sbatch since day
# one; Gen3v6 did not, so every MeanFlow eval silently ran the single HFFM_FLOW_STEPS default (2)
# and a K sweep meant remembering to resubmit with a different env var. The grid is now IN HERE.
# Each K writes its own results directory (flow_steps_v3 is watched as 'K' in the plan exp_name),
# so nothing overwrites anything.
#
# Override the grid without editing this file:  MF_FLOW_STEPS="2" sbatch ...
FLOW_STEPS_GRID="${MF_FLOW_STEPS:-1 2 5 10 20}"
echo "[ eval ] NFE budgets to evaluate: $FLOW_STEPS_GRID"

for K in $FLOW_STEPS_GRID; do
    echo "================================================================================"
    echo "[ eval ] K = $K   ($(date))"
    echo "================================================================================"
    python FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py --flow-steps "$K"
done

echo "Evaluation completed successfully."
