#!/bin/bash
#SBATCH --job-name=u18_p2_dpcc      # Gen15 U18 P2/L2 — diffusion K20 (DPCC baseline, seed 6) IN THE LOOP with the quadrotor plant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00             # diffusion K20, 1 variant x 3 geometries x 2 episodes x ~70 steps x ~0.5 s: minutes
#SBATCH --partition=gpu-1-student

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  PENDING_20260923 §2.4, L2: the diffusion baseline at its own budget and best rule (K20, `dpcc-c-tightened`) live
#  with the drone as the plant — seed 6, 3 geometries, 2 episodes (6 flights), tag p23pv2live. Compare with turbo T4.
#      GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p2_dpcc.sh
#  Results: logs/avoiding-d3il/plans/diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5_msgp23pv2live/6/results/halfspace_<geo>/
# ──────────────────────────────────────────────────────────────────────────────────────────────
set -eo pipefail
# ⛔ SUPERSEDED 2026-09-23 (author: R39 = live_p23_pillars.sh, tag p23uavpv2live). Default TAG p23pv2live lacks
# "uav", so the plant factory refuses it; and the MeanFM K1 cell would collide with R39. Kept for the record only.
echo "[ u18 ] SUPERSEDED: use Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh"; exit 3
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
export FMPCC_PROJ_CFG="${FMPCC_PROJ_CFG:-config/projection_eval_u18_live_dpcc.yaml}"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-36}"
export FMPCC_AVOID_UAV_HZ="${HZ:-1}"
export FMPCC_AVOID_UAV_FF="${FF:-0}"
export FMPCC_AVOID_UAV_VMAX="${VMAX:-1.0}"
export FMPCC_AVOID_UAV_REPLAY="${REPLAY:-clock}"
export FMPCC_AVOID_UAV_SIDECAR_DIR="$REPO/logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/$TAG"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"
SEED="${SEED:-6}"      # K is fixed at 20 by the trained diffusion schedule
python uav_avoiding_bridge/scene.py
echo "[ u18 P2-L2 ] plant=uav tag=$TAG yaml=$FMPCC_PROJ_CFG scale=$FMPCC_AVOID_UAV_SCALE hz=$FMPCC_AVOID_UAV_HZ ff=$FMPCC_AVOID_UAV_FF vmax=$FMPCC_AVOID_UAV_VMAX replay=$FMPCC_AVOID_UAV_REPLAY K=20 seed=$SEED"
if [ "$GO" != "1" ]; then echo "[ u18 P2-L2 ] dry-run: nothing launched. Re-submit with GO=1."; exit 0; fi
python scripts/eval.py --seed "$SEED"
echo "[ u18 P2-L2 ] done"
