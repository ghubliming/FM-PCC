#!/bin/bash
#SBATCH --job-name=train_mix_visual_aligning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

# ─── Gen14: pick the ML engine arm ──────────────────────────────────────
# $1 = engine  (diffusion | fm | mf | af)   default: fm  (the Gen7 reference arm)
# $2 = seed(s) (optional)              default: $MIX_SEEDS (6 7 8 9 10)
#
#   sbatch train_mix_visual_aligning.sh mf          # MeanFlow arm, all default seeds
#   sbatch train_mix_visual_aligning.sh af 7        # alpha-Flow arm, seed 7 only
#   sbatch train_mix_visual_aligning.sh mf "6 7"    # two seeds, SEQUENTIALLY in this one job
#
# Each arm writes to its OWN checkpoint tree (mix_visual_aligning_<engine>/...), so the
# four can train concurrently without touching each other or the Gen6V4/Gen7 originals.
#
# ⚠ SEQUENTIAL vs FAN-OUT. Seeds listed here run one after another INSIDE this single job,
#   against a 24 h wall. Visual aligning trains a ResNet-18 pair alongside the U-Net, so one
#   seed at 1e5 steps is already a large fraction of that wall — 5 sequential seeds will not
#   fit. Use mix_visual_aligning_pipeline.sh instead: it submits ONE JOB PER SEED, so each
#   seed gets its own 24 h budget and they run in parallel. Pass a list here only when you
#   deliberately want them serialised (e.g. to hold a single GPU allocation).
ENGINE="${1:-fm}"
SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
case "$ENGINE" in
    diffusion|fm|mf|af) ;;
    ddpm) ENGINE=diffusion; echo "[ engine ] NOTE: 'ddpm' is a deprecated alias for 'diffusion' (Gen14 U5)" ;;
    *) echo "[ train ] ERROR: unknown engine '$ENGINE' (want: diffusion | fm | mf | af)"; exit 1 ;;
esac
echo "[ train ] engine=$ENGINE  seeds='$SEEDS'"
if [ "$(echo $SEEDS | wc -w)" -gt 1 ]; then
    echo "[ train ] WARNING: $(echo $SEEDS | wc -w) seeds will run SEQUENTIALLY in this one job"
    echo "[ train ]          against the 24 h wall. Prefer mix_visual_aligning_pipeline.sh,"
    echo "[ train ]          which submits one job per seed."
fi

# $SEEDS is intentionally unquoted: it must word-split into separate --seeds arguments.
python mix_visual_aligning_test/train_mix_visual_aligning.py \
    --engine "$ENGINE" \
    --seeds $SEEDS \
    --use-wandb \
    --wandb-project FM-PCC-visual-aligning-gen14

echo "Job completed successfully."
