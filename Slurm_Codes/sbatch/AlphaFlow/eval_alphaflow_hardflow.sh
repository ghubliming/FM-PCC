#!/bin/bash
#SBATCH --job-name=af_hf_eval
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit
#SBATCH --partition=gpu-1-student   # Updated from sinfo output

# Gen3v7 U3 — UNIFIED eval: DPCC arms + HardFlow arm (arm C) on the α-Flow checkpoint.
# The HardFlow arm queries the α-Flow u-head at h=0 (u(x,t,0)=v), so the projection math is
# IDENTICAL to Gen12's (see logs_in_develop/Gen3v7_AlphaFlow/U3). α-Flow trains that h=0 anchor
# directly (af_ratio_fm=0.5), so the identity is better supported here than in Gen3v6.
# Reads the Gen3v7-dedicated config/alphaflow_projection_eval.yaml (NOT the shared
# projection_eval.yaml, which ~50 files across every generation also read).

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
#    HFFM_BATCH          candidate fan (mpc) for the -r/-c/-t arms.
#                        🔴 B4_PARITY (2026-08-20): DEFAULT IS NOW 4, was 1. arms A/B run
#                        plan-block batch_size=4 and BOTH arms loop serially over candidates
#                        around their CPU solve, so a 1-vs-4 fan is a 4x compute discount for
#                        arm C that reads as a HardFlow SPEEDUP. Every historic result whose
#                        folder carries the `B1` token was produced by the old default here.
#                        Bare `hardflow_new` is pinned to 1 by resolve_hf_batch_size()
#                        regardless of this value — that arm IS the faithful batch-1 control.
#                        Set HFFM_BATCH=1 only to deliberately reproduce an old B1 run.
#    FMPCC_MPC_BATCH     candidate fan (mpc) for arms A/B (`diffuser`, `dpcc-*`) — the SECOND,
#                        independent fan. It was a hardcoded 4 in the plan block of
#                        config/avoiding-d3il.py and therefore unsettable; HFFM_BATCH above has
#                        only ever moved arm C. Keep the two EQUAL unless the mismatch IS the
#                        experiment: both arms loop SERIALLY over candidates around their CPU
#                        solve (projection.py scipy SLSQP / hardflow_projection.py IPOPT), so an
#                        unequal fan scales one arm's projection wall-time and voids the timing
#                        comparison — the B4_PARITY confound.
#                        FMPCC_MPC_BATCH=1 HFFM_BATCH=1 -> a single candidate in EVERY arm, i.e.
#                        MPC candidate selection switched OFF (dpcc-r/-c/-t and
#                        hardflow_new-r/-c/-t then all collapse to index 0 — do not run the trio).
#                        A value != 4 auto-tags the results path (FMPCC_RUN_MSG=mpc<N>) because
#                        batch_size is not one of the folder-name tokens.
#    HFFM_ACT_THRESHOLD  fraction of late steps the NLP is active (0.5 == DPCC threshold 0.5)
#    HFFM_FLOW_STEPS     matched K for EVERY arm (drives plan-block flow_steps_v3 AND flow_steps)
# ✅ Gen3v7 was ALREADY on 4 here — it never produced a B1 run. Kept as-is; only the
# rationale above was rewritten to match the repo-wide B4_PARITY rule.
export HFFM_BATCH="${HFFM_BATCH:-4}"
export HFFM_ACT_THRESHOLD="${HFFM_ACT_THRESHOLD:-0.5}"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"
# export HFFM_FLOW_STEPS=2   # uncomment to force a specific matched K
echo "[ hardflow ] HFFM_BATCH=$HFFM_BATCH (arm C)  FMPCC_MPC_BATCH=$FMPCC_MPC_BATCH (arms A/B)  HFFM_ACT_THRESHOLD=$HFFM_ACT_THRESHOLD  HFFM_FLOW_STEPS=${HFFM_FLOW_STEPS:-<plan flow_steps>}"

# 5) Pre-flight gates. H1 pins the h=0 query, H3 the sigma=1.0 init noise, H4 that the h=0
#    anchor was actually trained (af_ratio_fm > 0). All three fail SILENTLY at run time if
#    broken, so they run BEFORE the eval and `set -e` aborts the job on a non-zero exit.
cd "$REPO"
python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py

# 6) Run the unified Gen3v7 evaluation (reads config/alphaflow_projection_eval.yaml by default).
# "$@" forwards submit.sh script args to the eval, e.g. --config <a pruned projection yaml>
python FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py "$@"

echo "Evaluation completed successfully."
