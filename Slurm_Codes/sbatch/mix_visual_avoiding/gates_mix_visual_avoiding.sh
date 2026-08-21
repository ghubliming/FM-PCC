#!/bin/bash
#SBATCH --job-name=gates_mix_visual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student

set -e

# Logging setup
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU info available)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# Setup Workspace Paths
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"

# Headless rendering
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

# W&B Login
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"

# ─── Gen16 gates — run BEFORE believing any visual-avoiding number ──────
# A0 copy fidelity      | A1 spec coherence   | A2 no stray camera/dim literals
# A3 registry wiring    | A4 path round-trip  | A5 backbones agree
# A6 dataset <-> spec   | A7 four arms train  | A8 hardflow hosts
# A9 yaml <-> config
#
#   sbatch gates_mix_visual_avoiding.sh            # all gates (needs a GPU)
#   sbatch gates_mix_visual_avoiding.sh static     # everything that needs no GPU
#   sbatch gates_mix_visual_avoiding.sh offline    # stdlib only (also runs off-cluster)
#   sbatch gates_mix_visual_avoiding.sh a7         # one gate
#
# 🔴 A0 IS THE LOAD-BEARING GATE. Gen16's claim is "Gen14's frame, one task swapped": every
#    file that is not on the declared-edit ledger must be byte-identical to Gen14's after the
#    package rename. If A0 fails, that claim is false and every cross-generation comparison
#    is suspect. Re-open the plan; do not patch over it.
#
# 🔴 A7 IS THE ONE THAT ANSWERS THE RESEARCH QUESTION — it takes one real loss step on each
#    of the four engines with a SINGLE-camera visual batch. The mf/af arms differentiate the
#    network with a forward-mode JVP and keep the vision encoder out of it by pre-encoding
#    the latent; if that repack is wrong for a one-camera payload, A7 is where it surfaces,
#    rather than nine hours into a training job.
#
# Exit code is non-zero if ANY gate fails, so this is safe to chain ahead of training.
GATE="${1:-all}"
echo "[ gates ] running: $GATE"

python mix_visual_avoiding_test/gates_mix_visual_avoiding.py --gate "$GATE"

echo "Job completed successfully."
