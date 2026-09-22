#!/bin/bash
#SBATCH --job-name=u18_live_l1      # Gen15 U18 — Mode L check: the FM planner IN THE LOOP with the quadrotor plant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00             # FM K20, 2 variants x 3 geometries x 20 episodes x ~70 steps x ~0.6 s ≈ 1.5 h
#SBATCH --partition=gpu-1-student

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  The closed-loop counterpart of the turbo pilot cell: same model (FM K20, seed 6), same 3 geometries, same two
#  variants (diffuser, dpcc-r-tightened), 20 trials with the SAME torch trial seeds as the Panda's msg20trials cell —
#  but the plant is the quadrotor (FMPCC_AVOIDING_PLANT=uav, 36x, clock 1 Hz, no feed-forward, v_max 1 m/s), so the
#  planner sees the DRONE's position every step. Compare with uav_avoiding_bridge/compare_live_turbo.py.
#      GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_l1.sh
#  Results (avoiding tree, Mode L writes where the eval writes; the tag keeps them apart):
#      logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/<train>/H8_K20_..._msguavpv2s36live20/6/results/halfspace_<geo>/
#  Plant sidecar: logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/uavpv2s36live20/uav_plant_records_*.json
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
TAG="${TAG:-uavpv2s36live20}"
export FMPCC_AVOIDING_PLANT="uav"
export FMPCC_RUN_MSG="$TAG"
export FMPCC_PROJ_CFG="${FMPCC_PROJ_CFG:-config/projection_eval_u18_live.yaml}"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-36}"
export FMPCC_AVOID_UAV_HZ="${HZ:-1}"
export FMPCC_AVOID_UAV_FF="${FF:-0}"
export FMPCC_AVOID_UAV_VMAX="${VMAX:-1.0}"
export FMPCC_AVOID_UAV_REPLAY="${REPLAY:-clock}"
export FMPCC_AVOID_UAV_SIDECAR_DIR="$REPO/logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/$TAG"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"
K="${K:-20}"; SEED="${SEED:-6}"
python uav_avoiding_bridge/scene.py
echo "[ u18 L1 ] plant=uav tag=$TAG yaml=$FMPCC_PROJ_CFG scale=$FMPCC_AVOID_UAV_SCALE hz=$FMPCC_AVOID_UAV_HZ ff=$FMPCC_AVOID_UAV_FF vmax=$FMPCC_AVOID_UAV_VMAX replay=$FMPCC_AVOID_UAV_REPLAY K=$K seed=$SEED"
if [ "$GO" != "1" ]; then echo "[ u18 L1 ] dry-run: nothing launched. Re-submit with GO=1."; exit 0; fi
python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py --flow-steps "$K" --seed "$SEED"
echo "[ u18 L1 ] done"
