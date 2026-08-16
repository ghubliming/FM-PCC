#!/bin/bash
#SBATCH --job-name=mf_hf_eval
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit
#SBATCH --partition=gpu-1-student   # Updated from sinfo output

# Gen3v6 U3 — UNIFIED eval: DPCC arms + HardFlow arm (arm C) on the mean-flow checkpoint.
# The HardFlow arm queries the mean-flow u-head at h=0 (u(x,t,0)=v), so the projection math is
# IDENTICAL to Gen12's (see logs_in_develop/Gen3v6_MeanFlow/U3). Reads the Gen3v6-dedicated
# config/meanflow_projection_eval.yaml (NOT the shared projection_eval.yaml).

# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
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
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# 3) Set Environment Variables
export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$GYM_AV:$PYTHONPATH"

# Rendering variables for MuJoCo on headless remote nodes
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

# W&B Login (Colab-style from key file)
if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
    export WANDB_MODE="online"
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

# 4) HardFlow-arm knobs (all optional; defaults come from the YAML `hardflow:` block).
#    HFFM_BATCH          candidate fan (mpc); 1 = faithful (‑r/‑c/‑t collapse), 4 = DPCC‑parity
#    HFFM_ACT_THRESHOLD  fraction of late steps the NLP is active (0.5 == DPCC threshold 0.5)
#    HFFM_FLOW_STEPS     matched K for EVERY arm (overrides plan-block flow_steps).
#                        🔵 U9: when set it PINS this job to that single K; leave it unset to
#                        get the whole {1,2,5,10,20} grid in one job (see §5).
export HFFM_BATCH="${HFFM_BATCH:-1}"
export HFFM_ACT_THRESHOLD="${HFFM_ACT_THRESHOLD:-0.5}"
# export HFFM_FLOW_STEPS=2   # uncomment to force a specific matched K
echo "[ hardflow ] HFFM_BATCH=$HFFM_BATCH  HFFM_ACT_THRESHOLD=$HFFM_ACT_THRESHOLD  HFFM_FLOW_STEPS=${HFFM_FLOW_STEPS:-<plan flow_steps>}"

# ── H8+8 (U10) knobs — all optional, all defaulting to the historic behaviour ──────────
#   MF_HORIZON        checkpoint horizon; MUST equal what the checkpoint was TRAINED at
#                     (the eval aborts on a mismatch — horizon is not a sampling knob).
#   MF_BACKBONE       unet|dit|mf_dit; MUST equal the trained backbone.
#   MF_REPLAN_STEPS   actions executed per plan. 1 = replan every env step (default, every
#                     result to date). 8 = HardFlow's own H16 cadence. Must be < horizon.
#                     A value != 1 auto-tags the results path (FMPCC_RUN_MSG=r<N>).
echo "[ h8+8 ] MF_HORIZON=${MF_HORIZON:-8 (default)}  MF_BACKBONE=${MF_BACKBONE:-mf_dit (default)}  MF_REPLAN_STEPS=${MF_REPLAN_STEPS:-1 (default)}"

# 5) 🔵 U9 MATCHED-K AUTO-EVAL — ⚠️ MATCHED BUDGET OR NOTHING (PLAN §7 / fix_7.3 §9).
#    fix_4's K sweep was four separate submits, hand-typed with HFFM_FLOW_STEPS=1/2/5/20
#    (CHANGELOG_Gen3v6_fix_4 §"post-fix sweep"). One forgotten resubmit and the decisive
#    matched-K control silently does not exist — which is exactly how the Gen13 claim died.
#    The grid is now a loop in here, as it has been in the Gen3v7 sibling from day one.
#    Each K writes its OWN results dir ('K' token in args_to_watch_fmv3_hf_plan), no collisions.
#
#    HFFM_FLOW_STEPS still wins when set, so every documented single-K command keeps working:
#      HFFM_FLOW_STEPS=2 HFFM_BATCH=4 ./Slurm_Codes/submit.sh <this script>   -> K=2 only
#    Otherwise the whole grid runs; override it with MF_FLOW_STEPS="1 2".
cd "$REPO"
if [ -n "$HFFM_FLOW_STEPS" ]; then
    FLOW_STEPS_GRID="$HFFM_FLOW_STEPS"
    echo "[ eval ] HFFM_FLOW_STEPS is set -> single-K run: $FLOW_STEPS_GRID"
else
    FLOW_STEPS_GRID="${MF_FLOW_STEPS:-1 2 5 10 20}"
    echo "[ eval ] NFE budgets to evaluate: $FLOW_STEPS_GRID"
fi

for K in $FLOW_STEPS_GRID; do
    echo "================================================================================"
    echo "[ eval ] K = $K   ($(date))"
    echo "================================================================================"
    python FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py --flow-steps "$K"
done

echo "Evaluation completed successfully."
