#!/bin/bash
#SBATCH --job-name=uav_mix_gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-1-student
set -e

# Gen15 gates — run this FIRST, before any training. Minutes, not hours.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh cuda "G3 G4 G6"
# Args: $1=device (cpu|cuda) [cuda]   $2=gates (quoted, space-sep) [all]
#
# G6 (per-plan wall clock) is only meaningful on cuda — on cpu the absolute numbers say
# nothing about whether an arm meets the 33 Hz control deadline.
# G1 (fm parity vs Gen11) needs two real checkpoints and is SKIPPED here; run it by hand:
#   python mix_uav_test/gates_mix_uav.py --gates G1 \
#       --gen11-savepath logs/UAV_FM/uav-corridor/flow_matching_v3_uav/<...>/6 \
#       --gen15-savepath logs/UAV_MIX/uav-corridor/mix_uav_fm/<...>/6
DEVICE="${1:-cuda}"
GATES="${2:-}"

CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)  |  $SLURM_JOB_NAME  |  ID $SLURM_JOB_ID  |  NODE $(hostname)"
echo "DEVICE: $DEVICE   GATES: ${GATES:-<all>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU"
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
export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLBACKEND="agg"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

cd "$REPO"
echo "[ uav_mix_gates ] python mix_uav_test/gates_mix_uav.py --device $DEVICE ${GATES:+--gates $GATES}"
python mix_uav_test/gates_mix_uav.py --device "$DEVICE" ${GATES:+--gates $GATES}
echo "Job completed successfully. All requested gates passed."
