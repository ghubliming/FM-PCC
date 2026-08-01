#!/bin/bash
#SBATCH --job-name=eval_mix_visual_aligning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student

set -e

# ─── Job Metadata ───────────────────────────────────────────────────────
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (no GPU info available)"
echo "GIT REV:   $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "================================================================================"

# ─── Environment Setup ──────────────────────────────────────────────────
FMPCC_ROOT="$HOME/FMPCC"
REPO="$FMPCC_ROOT/FM-PCC"
CONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="FMPCC"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export FMPCC="$REPO"
export D3IL_ROOT="$FMPCC/d3il"
export D3IL_ENV_ROOT="$D3IL_ROOT/environments/d3il"
export PYTHONPATH="$FMPCC:$D3IL_ROOT:$D3IL_ENV_ROOT:$PYTHONPATH"

# Headless rendering
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
# Fix_11b (sync from UAV): force UNBUFFERED stdout/stderr. Under SLURM python's stdout is a
# file (not a TTY) → block-buffered (~4-8KB), so progress prints sit in the buffer and never
# reach the .log until it fills or the process exits — the log looks frozen while eval runs
# fine, and a SIGKILL at the --time limit loses the buffered trail. Equivalent to `python -u`;
# an env var so it also covers any child python. See
# logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_11/CHANGELOG_fix11b_unbuffered_stdout.md.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
    echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
    exit 1
fi

cd "$REPO"

# ─── Run Evaluation (Gen14 Visual-Mix-ML) ───────────────────────────────
# Uses config/visual_aligning_eval.yaml for seed/variant configuration (SHARED by all four
# arms — same env, same constraints, same MPC pool; only the generative engine differs).
# Model loaded via experiment='plan_mix_visual_aligning_<engine>'.
# Results: logs/aligning-d3il-visual/plans/mix_visual_aligning_<engine>/<exp>/results/<seed>/
#
# Args: $1=engine (ddpm|fm|mf|af, default fm)  $2=seed(s) (optional)  $3=record_mode (default all)
# $2 blank    -> $MIX_SEEDS (default "6 7 8 9 10"), run sequentially in this one job.
# $2 = "6"    -> that seed only. Use for per-seed Slurm fan-out:
#   sbatch eval_mix_visual_aligning.sh mf 6
#   sbatch eval_mix_visual_aligning.sh mf 7
# $2 = "6 7"  -> those seeds, sequentially.
#
# NOTE: the seed list is passed on the COMMAND LINE, not read from
# config/visual_aligning_eval.yaml. That yaml is SHARED with the Gen6V4 and Gen7 evals and
# still says `seeds: [6]`; raising it there would drag those generations along with Gen14.
# Everything else (variants, constraints, n_contexts) still comes from the shared yaml, which
# is the point — same benchmark for every arm, only the seed set is Gen14's own.
#
# --engine MUST match the arm the checkpoint was trained with; the eval script asserts on
# it (engine_registry.assert_engine_matches) rather than dying later inside load_state_dict.
ENGINE="${1:-fm}"
case "$ENGINE" in
    ddpm|fm|mf|af) ;;
    *) echo "[ eval ] ERROR: unknown engine '$ENGINE' (want: ddpm | fm | mf | af)"; exit 1 ;;
esac
echo "[ eval ] engine=$ENGINE"

SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
echo "[ eval ] seeds='$SEEDS'"

RECORD_MODE="${3:-all}"
echo "[ eval ] Recording mode set to: $RECORD_MODE"

# $SEEDS is intentionally unquoted: it must word-split into separate --seeds arguments.
python mix_visual_aligning_test/eval_mix_visual_aligning.py \
    --engine "$ENGINE" --seeds $SEEDS --record "$RECORD_MODE" --eval-on-train

echo "Job completed successfully."
