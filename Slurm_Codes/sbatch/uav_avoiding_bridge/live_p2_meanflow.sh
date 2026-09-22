#!/bin/bash
#SBATCH --job-name=u18_p2_mf        # Gen15 U18 P2/L1 — MeanFM K1 (bbunet, seed 6) IN THE LOOP with the quadrotor plant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00             # MeanFM K1, 2 variants x 3 geometries x 2 episodes: minutes
#SBATCH --partition=gpu-1-student

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  PENDING_20260923 §2.4, L1: the selected D3IL-avoiding configuration (MeanFM, nfe 1, U-Net) live with the drone as
#  the plant — `diffuser` + `dpcc-t-tightened`, seed 6, 3 geometries, 2 episodes (12 flights), tag p23pv2live.
#  Compare with turbo T1 (tag p23pv2turbo, same source cell) per (geometry, episode).
#      GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p2_meanflow.sh
#  Results: logs/avoiding-d3il/plans/flow_matching_v3_meanflow/<train bbunet>/H8_K1_..._msgp23pv2live/6/results/halfspace_<geo>/
# ──────────────────────────────────────────────────────────────────────────────────────────────
set -eo pipefail
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID | grep -oP 'StdOut=\K\S+')
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)   NAME: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   NODE: $(hostname)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "JOB END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

FMPCC_ROOT="$HOME/FMPCC"; REPO="$FMPCC_ROOT/FM-PCC"; CONDA_DIR="$HOME/miniconda3"; CONDA_ENV_NAME="FMPCC"
source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV_NAME"
export FMPCC="$REPO"; export D3IL_ROOT="$FMPCC/d3il"; export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"
export MPLBACKEND="agg"
export MUJOCO_GL="disable"          # the plant never renders; the Panda env is not built in uav mode
cd "$REPO"

GO="${GO:-0}"
TAG="${TAG:-p23pv2live}"
export FMPCC_AVOIDING_PLANT="uav"
export FMPCC_RUN_MSG="$TAG"
CFG="${CFG:-config/meanflow_projection_eval_u18_live.yaml}"
export MF_BACKBONE="${MF_BACKBONE:-unet}"     # the bbunet checkpoints of T1 (architecture-matched arm)
export MF_HORIZON="${MF_HORIZON:-8}"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-36}"
export FMPCC_AVOID_UAV_HZ="${HZ:-1}"
export FMPCC_AVOID_UAV_FF="${FF:-0}"
export FMPCC_AVOID_UAV_VMAX="${VMAX:-1.0}"
export FMPCC_AVOID_UAV_REPLAY="${REPLAY:-clock}"
export FMPCC_AVOID_UAV_SIDECAR_DIR="$REPO/logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/$TAG"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"
K="${K:-1}"; SEED="${SEED:-6}"
python uav_avoiding_bridge/scene.py
echo "[ u18 P2-L1 ] plant=uav tag=$TAG yaml=$CFG backbone=$MF_BACKBONE scale=$FMPCC_AVOID_UAV_SCALE hz=$FMPCC_AVOID_UAV_HZ ff=$FMPCC_AVOID_UAV_FF vmax=$FMPCC_AVOID_UAV_VMAX replay=$FMPCC_AVOID_UAV_REPLAY K=$K seed=$SEED"
if [ "$GO" != "1" ]; then echo "[ u18 P2-L1 ] dry-run: nothing launched. Re-submit with GO=1."; exit 0; fi
python FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py --flow-steps "$K" --seed "$SEED" --config "$CFG"
echo "[ u18 P2-L1 ] done"
