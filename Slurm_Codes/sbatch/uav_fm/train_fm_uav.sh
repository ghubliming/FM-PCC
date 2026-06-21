#!/bin/bash
#SBATCH --job-name=uav_fm_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
set -e

# ---- Args: $1=scene (all|empty|corridor|s_curve|pillars, def all)  $2=seed (def 5) ----
SCENE="${1:-all}"
SEED="${2:-5}"

# ---- Pro-logging ----
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "SCENE: $SCENE   SEED: $SEED"
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

cd "$REPO"
echo "[ uav_fm_train ] python FM_v3_uav_test/train_fm_uav.py --scene $SCENE --seed $SEED"
python FM_v3_uav_test/train_fm_uav.py --scene "$SCENE" --seed "$SEED"
echo "Job completed successfully."
