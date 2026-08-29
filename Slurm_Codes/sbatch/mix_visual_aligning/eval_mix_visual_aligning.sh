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
# Args: $1=engine (diffusion|fm|mf|af, default fm)  $2=seed(s) (optional)  $3=record_mode (default all)
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
    diffusion|fm|mf|af) ;;
    ddpm) ENGINE=diffusion; echo "[ engine ] NOTE: 'ddpm' is a deprecated alias for 'diffusion' (Gen14 U5)" ;;
    *) echo "[ eval ] ERROR: unknown engine '$ENGINE' (want: diffusion | fm | mf | af)"; exit 1 ;;
esac
echo "[ eval ] engine=$ENGINE"

SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
echo "[ eval ] seeds='$SEEDS'"

RECORD_MODE="${3:-all}"
echo "[ eval ] Recording mode set to: $RECORD_MODE"

# ── Gen14 Fix_9 ── FiLM backbone. Each Gen14 arm has its own knob, MIX_FILM_MODE_<ENGINE>,
# with a bare MIX_FILM_MODE as the all-arms fallback (default v1). Accepted either way:
#   MIX_FILM_MODE_MF=v2 ./Slurm_Codes/submit.sh <this script> mf 6 all
#   MIX_FILM_MODE=v2    ./Slurm_Codes/submit.sh <this script> mf 6 all
#
# 🔴 MUST match the checkpoint being evaluated: film_mode is baked into diffusion_loadpath
# ('..._film{mode}_E<arm>'), so evaluating a v2 checkpoint at v1 resolves to a directory that
# may not exist. The backbone itself is always rebuilt from the train-time model_config.pkl,
# so the architecture cannot diverge from the weights — the failure mode here is a
# wrong/missing PATH, not wrong math. eval_mix_visual_aligning.py also prints the mode it
# read out of the pkl, and warns if the eval config disagrees.
#
# Same narrowing as the train script: resolve for THIS arm, re-publish arm-specifically,
# drop the broadcast form so the other three arm blocks resolve 'v1' on import.
ENGINE_UC=$(echo "$ENGINE" | tr '[:lower:]' '[:upper:]')
eval "FILM_MODE=\${MIX_FILM_MODE_${ENGINE_UC}:-\${MIX_FILM_MODE:-v1}}"
case "$FILM_MODE" in
    v1|v2) ;;
    *) echo "[ eval ] ERROR: FiLM mode '$FILM_MODE' is not known (want: v1 | v2)"; exit 1 ;;
esac
unset MIX_FILM_MODE
export "MIX_FILM_MODE_${ENGINE_UC}=$FILM_MODE"
echo "[ eval ] film_mode = $FILM_MODE  (MIX_FILM_MODE_${ENGINE_UC}; must match the checkpoint)"

# ── Gen14 U8 ── ML BONE. Same knob as the train script; MUST match the checkpoint.
#   MIX_BONE_MF=mf_dit ./Slurm_Codes/submit.sh <this script> mf 6 all
#
# 🔴 The bone is baked into diffusion_loadpath ('..._B{bone}_E<arm>'), and a DiT path
# carries NO '_film..' fragment, so evaluating a DiT checkpoint without this set resolves to
# the U-Net directory (or to nothing). The backbone itself is always rebuilt from the
# train-time model_config.pkl, so the architecture cannot diverge from the weights — the
# failure mode is a wrong/missing PATH, not wrong math. eval_mix_visual_aligning.py prints
# the bone it read out of the pkl and warns if the eval config disagrees.
eval "ML_BONE=\${MIX_BONE_${ENGINE_UC}:-\${MIX_BONE:-unet}}"
case "$ENGINE" in
    mf) VALID_BONES="unet mf_dit dit" ;;
    af) VALID_BONES="unet sit dit" ;;
    *)  VALID_BONES="unet" ;;
esac
if ! echo " $VALID_BONES " | grep -q " $ML_BONE "; then
    echo "[ eval ] ERROR: ml_bone '$ML_BONE' is not valid for engine '$ENGINE' (want: $VALID_BONES)"
    exit 1
fi
unset MIX_BONE
export "MIX_BONE_${ENGINE_UC}=$ML_BONE"
echo "[ eval ] ml_bone = $ML_BONE  (MIX_BONE_${ENGINE_UC}; must match the checkpoint)"

# ── Gen14 U10 ── alpha-FLOW SCHEDULE (af arm only). Like film_mode and ml_bone above, a
# non-default schedule is a CHECKPOINT-PATH key ('_AF<tag>', config: _mix_af_alpha_keys),
# so this job MUST see the same MIX_AF_ALPHA_* env the training job saw. If it does not,
# diffusion_loadpath resolves to the DEFAULT (alpha annealed to 0) tree — which either dies
# on a missing checkpoint or, worse, silently evaluates the wrong model under the right name.
# The pipeline exports these for you; set them by hand only when running this script alone.
if [ "$ENGINE" = "af" ]; then
    _AF_A=""
    for _v in MIX_AF_ALPHA_SCHED MIX_AF_ALPHA_INIT MIX_AF_ALPHA_END MIX_AF_ALPHA_CLAMP MIX_AF_ALPHA_GAMMA; do
        eval "_val=\${$_v:-}"
        if [ -n "$_val" ]; then _AF_A="$_AF_A $_v=$_val"; fi
    done
    if [ -n "$_AF_A" ]; then
        echo "[ eval ] alpha schedule:$_AF_A  (must match the checkpoint)"
    else
        echo "[ eval ] alpha schedule = SHIPPED DEFAULT -> resolving the alpha->0 checkpoint tree."
        echo "[ eval ]   ⚠  If you trained with MIX_AF_ALPHA_*, set the SAME values here or this"
        echo "[ eval ]      job will look in the wrong directory."
    fi
fi

# ── Gen14 U6 ── $4 = NFE override (flow_steps_v3), fm/mf/af only. Blank -> config default
# (mf/af: 2, fm: 100). Changes BOTH the sampler and the results folder, so a sweep lands in
# sibling H8_K<N>_... directories instead of overwriting. Also changes the projection budget:
# the sampler projects on every step from int((1 - T) * K) to the end, i.e. 1 SLSQP solve per
# replan at K=2 vs 50 at K=100 with T=0.5. Not valid for the diffusion arm (n_diffusion_steps).
#   sbatch eval_mix_visual_aligning.sh mf 6 all 2     # K=2
#   sbatch eval_mix_visual_aligning.sh mf 6 all       # config default
FLOW_STEPS="${4:-}"
if [ -n "$FLOW_STEPS" ]; then
    if [ "$ENGINE" = "diffusion" ]; then
        echo "[ eval ] ERROR: NFE override (\$4) does not apply to engine 'diffusion' —"
        echo "         its NFE key is n_diffusion_steps, set in the config block."
        exit 1
    fi
    FLOW_ARG="--flow-steps $FLOW_STEPS"
    echo "[ eval ] NFE override: flow_steps_v3 = $FLOW_STEPS"
else
    FLOW_ARG=""
    echo "[ eval ] NFE: config default for engine=$ENGINE"
fi

# $SEEDS and $FLOW_ARG are intentionally unquoted: $SEEDS must word-split into separate
# --seeds arguments, and $FLOW_ARG must vanish entirely when empty.
python mix_visual_aligning_test/eval_mix_visual_aligning.py \
    --engine "$ENGINE" --seeds $SEEDS --record "$RECORD_MODE" --eval-on-train $FLOW_ARG

echo "Job completed successfully."
