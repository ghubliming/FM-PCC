#!/bin/bash
#SBATCH --job-name=TEMPLATE_JOB        # CHANGE THIS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # ADJUST CORES
#SBATCH --mem=32G                      # ADJUST RAM
#SBATCH --gres=gpu:1                   # ADJUST GPU
#SBATCH --time=24:00:00                # ADJUST TIME LIMIT
#SBATCH --partition=gpu-1-student      # CHANGE PARTITION IF NEEDED

# Exit on any error
set -e

# ------------------------------------------------------------------------------
# 1) LOGGING SETUP (Optional but recommended)
# ------------------------------------------------------------------------------
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then
    # Creates a symlink to the current log for easy 'tail -f'
    ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log
fi

echo "================================================================================"
echo "JOB START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "NODE:      $(hostname)"
echo "================================================================================"

# Trap for JOB END
function on_exit {
    echo "================================================================================"
    echo "JOB END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# ------------------------------------------------------------------------------
# 2) ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Export project-specific paths
export PYTHONPATH="$REPO:$REPO/d3il:$PYTHONPATH"

# ------------------------------------------------------------------------------
# 3) EXECUTION
# ------------------------------------------------------------------------------
cd "$REPO"

# REPLACE WITH YOUR COMMAND
# python your_script.py --args...

echo "Job completed successfully."
