#!/bin/bash
#SBATCH --job-name=u18_p23_live     # Gen15 U18 / R39 — UAV-pillars: the Ch 6 table cells evaluated LIVE with the quadrotor plant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00             # one engine x one K: 5 seeds x 3 geometries x 2 episodes x 2 variants = 60 flights; expect < 2 h
#SBATCH --partition=gpu-1-student
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  R39 (PENDING_20260923 §2, rev 23-09 "real eval, no turbo"): Chapter 6 tab:uav-pillars-raw + fig:uav-pillars-paths.
#  The avoiding planner runs IN THE LOOP with the quadrotor as the plant (Mode L): it plans from the vehicle's state
#  at every control step. Same protocol as the table cells: seeds 6-10, 3 geometries, 2 episodes, variants
#  diffuser + dpcc-t-tightened.
#      GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh mf 1     # MeanFM K1
#      args: $1 = mf | af    $2 = K (1 | 2)        submitter: Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh
#  Cells (the table cells of Ch 6, 19-09 corpus):
#      mf: flow_matching_v3_meanflow/…_bbunet_…/H8_K<k>_Meuler_T0.5_A0.5_B1_…MeanFlowODE           (MF_BACKBONE=unet)
#      af: flow_matching_v3_alphaflow/…_bbunet_…_ae0.2_…/H8_K<k>_…AlphaFlowODE_msgdpccproto         (AF_BONE=unet AF_ALPHA_END=0.2 AF_EPOCH=latest)
#  Results: the same trees with the eval folder suffix _msg${TAG} (TAG default p23uavpv2live — MUST contain "uav":
#  uav_avoiding_bridge/factory.py refuses otherwise, the live results share the Panda layout).
#  Plant sidecars (world paths, track_err, contacts): logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/$TAG/<engine>_K<k>/
# ──────────────────────────────────────────────────────────────────────────────────────────────
set -eo pipefail
CURRENT_LOG=$(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -oP 'StdOut=\K\S+' || true)
if [ -n "$CURRENT_LOG" ]; then ln -snf "$CURRENT_LOG" Slurm_Codes/logs/latest.log; fi
echo "================================================================================"
echo "JOB START: $(date)   NAME: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   NODE: $(hostname)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repo')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo "================================================================================"
function on_exit { echo "================================================================================"; echo "JOB END:   $(date)"; echo "================================================================================"; }
trap on_exit EXIT

ENGINE="${1:-mf}"
KS="${2:-1}"
case "$ENGINE" in mf|af) ;; *) echo "[ R39 ] engine must be mf|af (got '$ENGINE')"; exit 2 ;; esac
GO="${GO:-0}"
TAG="${TAG:-p23uavpv2live}"
case "$(echo "$TAG" | tr 'A-Z' 'a-z')" in *uav*) ;; *) echo "[ R39 ] TAG='$TAG' must contain 'uav' (factory.py refuses otherwise)"; exit 2 ;; esac
SEEDS="${SEEDS:-6 7 8 9 10}"

FMPCC_ROOT="$HOME/FMPCC"; REPO="$FMPCC_ROOT/FM-PCC"; CONDA_DIR="$HOME/miniconda3"; CONDA_ENV_NAME="FMPCC"
source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV_NAME"
export FMPCC="$REPO"; export D3IL_ROOT="$FMPCC/d3il"; export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"
export MPLBACKEND="agg"
export MUJOCO_GL="disable"          # the plant never renders; the Panda env is not built in uav mode
cd "$REPO"

# ── the plant (U18 fix3/fix4 settings, as validated by the live check 26077) ──
export FMPCC_AVOIDING_PLANT="uav"
export FMPCC_RUN_MSG="$TAG"
export FMPCC_AVOID_UAV_SCALE="${SCALE:-36}"
export FMPCC_AVOID_UAV_HZ="${HZ:-1}"
export FMPCC_AVOID_UAV_FF="${FF:-0}"
export FMPCC_AVOID_UAV_VMAX="${VMAX:-1.0}"
export FMPCC_AVOID_UAV_REPLAY="${REPLAY:-clock}"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"      # four candidate plans, as the table cells

python uav_avoiding_bridge/scene.py                  # regenerate the scaled arena (asserts frame <-> scene)

echo "[ R39 ] engine=$ENGINE K=[$KS] seeds=[$SEEDS] tag=$TAG plant=uav scale=$FMPCC_AVOID_UAV_SCALE hz=$FMPCC_AVOID_UAV_HZ ff=$FMPCC_AVOID_UAV_FF vmax=$FMPCC_AVOID_UAV_VMAX replay=$FMPCC_AVOID_UAV_REPLAY"
for K in $KS; do
    export FMPCC_AVOID_UAV_SIDECAR_DIR="$REPO/logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/$TAG/${ENGINE}_K${K}"
    if [ "$ENGINE" = "mf" ]; then
        export MF_BACKBONE="${MF_BACKBONE:-unet}"; export MF_HORIZON="${MF_HORIZON:-8}"
        CFG="${CFG:-config/meanflow_projection_eval_u18_live.yaml}"
        echo "--------------------------------------------------------------------------------"
        echo "[ R39 ] MeanFM K=$K backbone=$MF_BACKBONE cfg=$CFG seeds=[$SEEDS] (one call per seed)"
        [ "$GO" = "1" ] || { echo "[ R39 ] dry-run: nothing launched. Re-submit with GO=1."; continue; }
        for S in $SEEDS; do
            python FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py --flow-steps "$K" --seed "$S" --config "$CFG"
        done
    else
        export AF_BONE="${AF_BONE:-unet}"; export AF_ALPHA_END="${AF_ALPHA_END:-0.2}"; export AF_EPOCH="${AF_EPOCH:-latest}"
        export AF_SEEDS="$SEEDS"; export AF_NTRIALS="${AF_NTRIALS:-2}"
        CFG="${CFG:-config/alphaflow_projection_eval_u18_live.yaml}"
        echo "--------------------------------------------------------------------------------"
        echo "[ R39 ] CI-MeanFM K=$K bone=$AF_BONE alpha_end=$AF_ALPHA_END epoch=$AF_EPOCH cfg=$CFG seeds=[$AF_SEEDS] n_trials=$AF_NTRIALS"
        [ "$GO" = "1" ] || { echo "[ R39 ] dry-run: nothing launched. Re-submit with GO=1."; continue; }
        python FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py --flow-steps "$K" --config "$CFG"
    fi
done
echo "[ R39 ] done"
