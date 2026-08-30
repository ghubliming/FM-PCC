#!/bin/bash
#SBATCH --job-name=hffm_eval
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                   # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=24:00:00             # Time limit (~2x expected; arm C runs K NLP solves per plan)
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
    # Slurm job id -> W&B run tag (searchable/filterable alongside the run name)
    if [ -n "$SLURM_JOB_ID" ]; then export WANDB_TAGS="slurm-$SLURM_JOB_ID"; fi
fi

cd "$REPO"

# 4) Gen12 evaluation: arms A (unguided) / B (DPCC projection) / C (hardflow_new),
#    all three in ONE process per K so seeds, env resets and the checkpoint are shared.
#
# Config split (edit -> submit):
#   - which model : plan_fm_v3_hardflow block in config/avoiding-d3il.py (checkpoint_dir/loadpath)
#   - seeds / arms / geometry / arm-C tuning : config/hardflow_projection_eval.yaml
#
# ⚠️ MATCHED BUDGET (PLAN §5): each K in the grid is applied to ALL arms at once, so A/B/C are
# always compared at equal K. Each K writes its own results dir (K<K>_n<n>), so nothing overwrites
# and the sweep also confirms the K-override works (K2/K5/K10 dirs appear).
#
# K: DEFAULT is a SINGLE run at the plan block's `flow_steps` (config/avoiding-d3il.py).
# To SWEEP, set HFFM_FLOW_STEPS to a space-separated list — each K is applied to ALL arms
# (matched budget, PLAN §5) and writes its own results dir (K<K>_n<n>), so nothing overwrites:
#   HFFM_FLOW_STEPS="2 5 10"  sbatch ...
# Re-run a finished directory: FORCE_OVERWRITE=1 sbatch ...   (PLAN §3.6)
#
# 🔴 B4_PARITY (2026-08-20) — arm C's candidate fan. This script never exported HFFM_BATCH,
# so it inherits config/hardflow_projection_eval.yaml `hardflow.batch_size`, which is NOW 4
# (was 1). 4 == the plan block's batch_size, i.e. the fan arms A/B already use — required,
# because both arms loop serially over candidates around their CPU solve and a 1-vs-4 fan
# silently hands arm C a 4x compute discount. Bare `hardflow_new` stays at 1 regardless
# (resolve_hf_batch_size), so the faithful upstream control is still one variant away.
# Set HFFM_BATCH=1 only to deliberately reproduce an old `B1` run.
#
# 🔵 FMPCC_MPC_BATCH — the SECOND, independent candidate fan: arms A/B (`diffuser`, `dpcc-*`),
# i.e. the plan block's `batch_size`. It was a hardcoded 4 and therefore unsettable, while
# HFFM_BATCH above has only ever moved arm C. Keep the two EQUAL unless the mismatch IS the
# experiment. `FMPCC_MPC_BATCH=1 HFFM_BATCH=1` gives a single candidate in EVERY arm — MPC
# candidate SELECTION switched off, which is what isolates how much of DPCC's success rate
# comes from the selection rule rather than from the projector. At a fan of 1 the -r/-c/-t
# variants all collapse to index 0 in both arms, so run ONE of them, not the trio.
# A value != 4 auto-tags the eval-name (`..._msgmpc<N>`), because the existing `mpc<N>` token
# in hf_paths.eval_name describes arm C only.
export HFFM_BATCH="${HFFM_BATCH:-4}"
export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"
echo "[ hardflow ] HFFM_BATCH=$HFFM_BATCH (arm-C fan for -r/-c/-t; bare hardflow_new is always 1)  FMPCC_MPC_BATCH=$FMPCC_MPC_BATCH (arms A/B fan)"

# ── [SolverSwap 2026-08-28] HFFM_SOLVERS — the SAME K under BOTH NLP backends ────────────
# Arm C's NLP backend is a PROCESS-level default (FMPCC_HF_NLP_BACKEND ->
# hardflow_projection.resolve_nlp_backend), so one python process = one backend. To measure
# IPOPT and SLSQP on the SAME node, in the SAME job, back to back, this loops the eval once
# per backend for every K.
#
#   HFFM_SOLVERS=""             (default) ONE pass on the shipped default -> slsqp
#   HFFM_SOLVERS="ipopt slsqp"  TWO passes per K -> the A/B the swap actually needs
#
# The passes do NOT collide and nothing is overwritten: an ipopt pass writes `hardflow_new-*`,
# an slsqp pass writes `hardflow_sls-*` (hardflow_projection.artifact_variant_label). The
# SHARED arms (`diffuser`, `dpcc-*`) are run by whichever pass goes FIRST and then SKIPPED by
# the second (the `already exists` guard in the eval), so they cost one run, are measured once,
# and BOTH hardflow arms are compared against the identical DPCC row.
#
# Order matters for reading the log, not for correctness: put ipopt first so the expensive arm
# and the DPCC baseline land in the same pass.
# ── HFFM_ACT_THRESHOLD — arm C's projection budget ───────────────────────────────────────
# 🔴 THRESHOLD PARITY. This is the SAME quantity as DPCC's `diffusion_timestep_threshold`
# (0.5 in config/hardflow_projection_eval.yaml): the fraction of the late trajectory over
# which the NLP is active. HIGHER = MORE projection. If the two differ, arm C and arms A/B
# are running different projection budgets and their wall-clock comparison is void — which
# is exactly what happened up to job 25222, where the yaml shipped 1.0 against DPCC's 0.5
# and handed arm C ~2x the work (see the DA in logs_in_develop/aggregated_hf_nlp_backend/).
# The yaml default is now 0.5, i.e. matched. Override per run, or sweep:
#   HFFM_ACT_THRESHOLD=0.25            single value, overrides the yaml
#   HFFM_ACT_THRESHOLDS="0.1 0.25 0.5" sweep; each value is its own results dir (thres<A>)
# Unset (default) = whatever the yaml says. The value is a token in the results dir name
# (`K<K>_thres<A>_mpc<B>_n<n>`, hf_paths.eval_name), so no two settings can collide.
HFFM_ACT_THRESHOLDS="${HFFM_ACT_THRESHOLDS:-${HFFM_ACT_THRESHOLD:-}}"
if [ -n "$HFFM_ACT_THRESHOLDS" ]; then
    for A in $HFFM_ACT_THRESHOLDS; do
        case "$A" in
            ''|*[!0-9.]*|*.*.*)
                echo "[ eval ] FATAL: HFFM_ACT_THRESHOLD entry '$A' is not a number in [0,1] (list: '$HFFM_ACT_THRESHOLDS')" >&2
                exit 2 ;;
        esac
    done
    _n_a=$(set -- $HFFM_ACT_THRESHOLDS; echo $#)
    if [ "$_n_a" -gt 1 ]; then
        echo "[ hardflow ] activation_threshold sweep: $HFFM_ACT_THRESHOLDS   (DPCC diffusion_timestep_threshold = 0.5; MATCH IT unless the mismatch IS the experiment)"
    else
        echo "[ hardflow ] activation_threshold = $HFFM_ACT_THRESHOLDS (override)   (DPCC diffusion_timestep_threshold = 0.5; MATCH IT unless the mismatch IS the experiment)"
    fi
else
    echo "[ hardflow ] activation_threshold: from config/hardflow_projection_eval.yaml (0.5 = DPCC parity).  Set HFFM_ACT_THRESHOLD=<A> or HFFM_ACT_THRESHOLDS=\"...\" to override."
fi

HFFM_SOLVERS="${HFFM_SOLVERS:-}"
if [ -n "$HFFM_SOLVERS" ]; then
    echo "[ hardflow ] NLP backend sweep: $HFFM_SOLVERS  (one process per backend; ipopt -> hardflow_new-*, slsqp -> hardflow_sls-*)"
else
    echo "[ hardflow ] NLP backend: code default (slsqp).  Set HFFM_SOLVERS=\"ipopt slsqp\" for the A/B."
fi

# One eval invocation per backend. $1 = K ("" -> the plan block's flow_steps); rest passed through.
#
# 🔴 K_ENV_SCALAR (2026-08-30) — HFFM_FLOW_STEPS had TWO readers that disagreed. THIS script treats
# it as a space-separated LIST and loops it, handing each K down as `--flow-steps $K`. But the
# variable stayed exported, and config/avoiding-d3il.py reads the SAME name as a scalar
# `int(os.environ.get('HFFM_FLOW_STEPS', 2))` at MODULE-IMPORT time — before argparse can apply
# --flow-steps. A single value reads identically both ways, which is why every single-K job ever
# submitted worked; the multi-K sweep documented at the top of this file killed job 25161 in 5 s:
#     ValueError: invalid literal for int() with base 10: '10 20'
# Fix: the list is snapshotted into HFFM_K_LIST and HFFM_FLOW_STEPS is UNSET (below), then each
# child gets it back as the SCALAR for its own K. Env and CLI now always agree, so the
# config-derived exp_name ('_K{K}_') always names the K that was actually evaluated.
run_eval () {
    local K="$1"; shift
    # env is assembled as: all -u options FIRST, then assignments (GNU env parses options
    # before NAME=VALUE). Each child therefore sees a scalar for every knob, never a list.
    local unsets=() sets=() kargs=()
    if [ -n "$K" ]; then
        kargs=(--flow-steps "$K")
        sets+=("HFFM_FLOW_STEPS=$K")
    else
        # No override: the plan block's flow_steps decides. STRIP the variable rather than
        # pass it through empty — int('') is a ValueError too.
        unsets+=(-u HFFM_FLOW_STEPS)
    fi
    local alist="${HFFM_ACT_THRESHOLDS:-__cfg__}"
    for A in $alist; do
        local aunset=() aset=()
        if [ "$A" = "__cfg__" ]; then
            aunset=(-u HFFM_ACT_THRESHOLD)
        else
            aset=("HFFM_ACT_THRESHOLD=$A")
        fi
        if [ -z "$HFFM_SOLVERS" ]; then
            echo "[ eval ] activation_threshold = ${A/__cfg__/yaml}   ($(date))"
            env "${unsets[@]}" "${aunset[@]}" "${sets[@]}" "${aset[@]}" \
                python FM_v3_hardflow_test/eval_FM_v3_hardflow.py "${kargs[@]}" "$@"
            continue
        fi
        for S in $HFFM_SOLVERS; do
            echo "--------------------------------------------------------------------------------"
            echo "[ eval ] NLP backend = $S   K = ${K:-plan-block}   A = ${A/__cfg__/yaml}   ($(date))"
            echo "--------------------------------------------------------------------------------"
            env "${unsets[@]}" "${aunset[@]}" "${sets[@]}" "${aset[@]}" \
                "FMPCC_HF_NLP_BACKEND=$S" \
                python FM_v3_hardflow_test/eval_FM_v3_hardflow.py "${kargs[@]}" "$@"
        done
    done
}
if [ -n "${HFFM_FLOW_STEPS:-}" ]; then
    # K_ENV_SCALAR (see run_eval): snapshot the list, then take the multi-value variable OUT of the
    # environment so no child can import a config with a non-integer HFFM_FLOW_STEPS. Validate up
    # front — a typo'd grid should fail in 1 s here, not after the first K has already burned hours.
    HFFM_K_LIST="$HFFM_FLOW_STEPS"
    unset HFFM_FLOW_STEPS
    for K in $HFFM_K_LIST; do
        case "$K" in
            ''|*[!0-9]*)
                echo "[ eval ] FATAL: HFFM_FLOW_STEPS entry '$K' is not a positive integer (list: '$HFFM_K_LIST')" >&2
                exit 2 ;;
        esac
    done
    echo "[ eval ] K sweep: $HFFM_K_LIST   FORCE_OVERWRITE=${FORCE_OVERWRITE:-0}"
    for K in $HFFM_K_LIST; do
        echo "================================================================================"
        echo "[ eval ] K = $K   ($(date))"
        echo "================================================================================"
        run_eval "$K" "$@"
    done
else
    echo "[ eval ] single run, K from plan block   FORCE_OVERWRITE=${FORCE_OVERWRITE:-0}"
    run_eval "" "$@"
fi

echo "Evaluation completed successfully."
