#!/bin/bash
#SBATCH --job-name=hffm_solverbench
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=16G                   # Total memory
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=02:00:00             # Time limit (~2x expected; the bench is CPU-only and takes minutes)
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

# 4) Gen12 solver bench -- HardFlow's IPOPT vs DPCC's SLSQP on the SAME NLP.
#    No checkpoint, no dataset, no env: geometry comes from
#    config/hardflow_projection_eval.yaml, references are synthetic.
#    Nothing in the production sampler is modified -- both solvers are built
#    side by side in-process and timed.
#
#    Knobs (env vars, all optional):
#      HFFM_BENCH_REPS      timed solves per regime            (default 50)
#      HFFM_BENCH_HORIZON   planning horizon H                 (default 8)
#      HFFM_BENCH_REF       endpoint | iterate | both          (default both)
#      HFFM_BENCH_RUNS      how many times to repeat the whole
#                           bench, for run-to-run spread       (default 3)
#      HFFM_BENCH_TAG       label for the output directory     (default $SLURM_JOB_ID)
REPS="${HFFM_BENCH_REPS:-50}"
HORIZON="${HFFM_BENCH_HORIZON:-8}"
REF="${HFFM_BENCH_REF:-both}"
RUNS="${HFFM_BENCH_RUNS:-3}"
TAG="${HFFM_BENCH_TAG:-${SLURM_JOB_ID:-manual}}"

OUT_DIR="$REPO/logs/solver_bench/$TAG"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/solver_bench.csv"

echo "[ bench ] reps=$REPS horizon=$HORIZON ref=$REF runs=$RUNS"
echo "[ bench ] out -> $OUT_DIR"

# Repeat the whole bench RUNS times with different seeds. One job, several
# independent measurements -- that is what makes the ratio trustworthy rather
# than a single noisy sample on a shared node.
STATUS=0
for r in $(seq 1 "$RUNS"); do
    echo ""
    echo "################ bench run $r / $RUNS  (seed=$r) ################"
    python FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py \
        --reps "$REPS" \
        --horizon "$HORIZON" \
        --ref "$REF" \
        --seed "$r" \
        --csv "$CSV" \
        --json "$OUT_DIR/run_${r}.json" || STATUS=$?
done

echo ""
echo "================================================================================"
echo "[ bench ] all runs done. Aggregate CSV: $CSV"
echo "================================================================================"

# A non-zero exit means a projector returned INFEASIBLE output on some solve.
# That is a correctness bug and outranks anything in the timing table.
exit $STATUS
