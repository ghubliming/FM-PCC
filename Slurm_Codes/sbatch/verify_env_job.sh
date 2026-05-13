#!/bin/bash
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=2           # Minimal CPUs for verification
#SBATCH --mem=8G                    # Minimal memory
#SBATCH --gres=gpu:1                # Request 1 GPU to verify CUDA access
#SBATCH --time=00:10:00             # 10 minute limit
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
# Point to the conda.sh in your miniconda installation
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 3) Set Environment Variables (The "Colab Way")
export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"

# Rendering variables for MuJoCo on headless remote nodes
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"

# 4) Run Full Verification (Adapted from Colab Step 9)
echo "Running FM-PCC Environment Verification..."
python - <<'PY'
import importlib
import sys

pkgs = [
    'torch', 'numpy', 'scipy', 'gym', 'gymnasium', 'gymnasium_robotics',
    'minari', 'wandb', 'mujoco', 'diffusers', 'transformers'
]

ok = True
print("\n--- Package Import Check ---")
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', 'unknown')
        print(f'{p:20s} {v}')
    except Exception as e:
        ok = False
        print(f'{p:20s} NOT IMPORTABLE ({type(e).__name__}: {e})')

import numpy, torch
print("\n--- System Check ---")
print('Numpy Version:  ', numpy.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:     ', torch.cuda.get_device_name(0))

major = int(numpy.__version__.split('.')[0])
if major >= 2:
    ok = False
    print('ERROR: numpy 2.x detected, expected 1.26.4 for this workflow')

if not ok:
    print("\n❌ VERIFICATION FAILED: Environment is inconsistent or missing packages.")
    sys.exit(2)
else:
    print("\n✅ VERIFICATION SUCCESSFUL: Environment is ready for training.")
PY
