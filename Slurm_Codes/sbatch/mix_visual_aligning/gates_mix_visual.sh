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

# ─── Gen14 gates — run BEFORE believing any Visual-Mix-ML number ────────
# G0 copy fidelity | G1 reference-arm wiring | G2 JVP survives vision
# G3 MeanFlow identity at h=0 | G4 alpha spans the budget | G5 alpha->0 == MeanFlow
# G6 projector fires at K=1
#
#   sbatch gates_mix_visual.sh          # all gates
#   sbatch gates_mix_visual.sh static   # the no-GPU subset (g0, g1, g4, g6, gb1, gb6, gb7)
#   sbatch gates_mix_visual.sh bone     # Gen14 U8 only: G-B1/B2/B3/B4-5/B6/B7
#
# ── Gen14 U8 ── the ML-bone gates:
#   G-B1  cond_dim=0 leaves all four transformer bones byte-identical (state-only gens safe)
#   G-B2  every visual bone builds AND is parameter-matched to the ~4.0M U-Net
#   G-B3  gradient actually reaches vis_projector -- the model is not image-blind
#   G-B4/5 mf JVP and af bootstrap both stay finite with the visual token
#   G-B6  prefix_tokens / RoPE-table / pos_embed bookkeeping agree (the silent failure mode)
#   G-B7  bone is a checkpoint-path key, and no '_film..' fragment lands on a DiT path
#
# ── Gen14 U12 ── the checkpoint-selector gate (no GPU; included in `static` and `all`):
#   G-B12 MIX_EPOCH tags the RESULTS directory and NEVER the checkpoint path, on all four
#         arms, and malformed selectors are rejected at config time rather than becoming a
#         state_<garbage>.pt FileNotFoundError minutes into a GPU allocation.
#
#   sbatch gates_mix_visual.sh gb12     # just the U12 gate (~seconds, no GPU work)
#
# 🔴 Run `bone` before the first DiT training job. G-B6 is the one that matters most: a
# half-applied token bump trains fine and reads the WRONG positions.
#
# Exit code is non-zero if ANY gate fails, so this is safe to chain ahead of training.
#
# NOTE (fix_2): G6 is a RUNTIME test — it drives p_sample_loop at K=1 with a spy projector
# and counts project() calls. Its `fm` leg is EXPECTED to print a loud
# "KNOWN UPSTREAM DEFECT" banner: at K=1 with threshold=0.5 the Gen7 guard
# `0 >= 0.5` is False, so the DPCC projection never runs. That is a Gen7/Gen6V4 defect,
# NOT a Gen14 regression, so it does not fail the gate (failing would block the
# pipeline's --dependency=afterok chain). Read the banner; do not ignore it.
GATE="${1:-all}"
echo "[ gates ] running: $GATE"

python mix_visual_aligning_test/gates_mix_visual.py --gate "$GATE"

echo "Job completed successfully."
