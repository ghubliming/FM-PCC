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

# ── Gen14 U11 ── MIX_PROJ_T: PROJECTION THRESHOLD (arm B, DPCC) — SWEEPABLE ───────────────
# T = the fraction of the LATE ODE over which the projector runs. The sampler projects on
# every step from int((1 - T) * K) to the end, so solves/replan ~= T*K.
#
#   T=0.5  K=100 -> 50 solves   (shipped; ~15 s/step, 50 h for 30 rollouts, NEVER finished)
#   T=0.1  K=100 -> 10 solves
#   T=0.05 K=100 ->  5 solves
#
# 🔴 SWEEP FORM. MIX_PROJ_T takes a SPACE-SEPARATED LIST and this job runs one full eval
#    pass per value, sequentially, in the same allocation:
#      MIX_PROJ_T="0.1 0.05"  -> two passes, T=0.1 then T=0.05
#    Each pass writes its OWN results dir (T is a plan path key), so the passes cannot
#    collide with each other or with the existing T0.5 run. `set -e` is relaxed around the
#    loop so a failure in pass 1 does not silently discard pass 2 — each pass reports its
#    own exit status and the job fails at the end if any pass failed.
#
# ✅ Cannot overwrite anything: results land in H8_K<K>_Meuler_T<T>_..., and the CHECKPOINT
#    path is untouched (T is eval-only) — same weights, no retraining.
# 🔴 Set it here, NOT in config/visual_aligning_eval.yaml: that YAML is read once at config
#    import and feeds every block in the file, so an edit re-points every later eval.
#
# HardFlow (arm C) INHERITS this value when `hardflow.activation_threshold: null` (the
# default), so arms B and C stay matched. HFFM_ACT_THRESHOLD overrides arm C alone.
PROJ_T_LIST="${MIX_PROJ_T:-}"
if [ -n "$PROJ_T_LIST" ]; then
    for _t in $PROJ_T_LIST; do
        if ! awk -v t="$_t" 'BEGIN{exit !(t+0==t && t>=0 && t<=1)}' </dev/null; then
            echo "[ eval ] ERROR: MIX_PROJ_T entry '$_t' must be a number in [0, 1] (a FRACTION"
            echo "         of the late ODE, not a step count)."
            exit 1
        fi
    done
    echo "[ eval ] projection threshold sweep: T = $PROJ_T_LIST   (config default 0.5)"
    for _t in $PROJ_T_LIST; do
        if [ -n "$FLOW_STEPS" ]; then
            awk -v t="$_t" -v k="$FLOW_STEPS" 'BEGIN{
                n = k - int((1-t)*k); if (n < 1) n = 1;
                printf "[ eval ]   T=%-6s -> %2d projector call(s)/replan  -> dir H8_K%d_Meuler_T%s_...\n", t, n, k, t }'
        else
            echo "[ eval ]   T=$_t -> dir H8_K<K>_Meuler_T${_t}_..."
        fi
    done
else
    echo "[ eval ] projection threshold T = config default (0.5), single pass"
    if [ -n "$FLOW_STEPS" ] && [ "$FLOW_STEPS" -ge 50 ] 2>/dev/null; then
        echo "[ eval ]   ⚠  WARNING: K=$FLOW_STEPS at T=0.5 means ~$((FLOW_STEPS/2)) SLSQP solves per replan."
        echo "[ eval ]      Every K>=50 projected cell in this tree has hit the 24 h wall and"
        echo "[ eval ]      truncated (mf K100 died at 11/30, needing 50 h). Set MIX_PROJ_T=0.1"
        echo "[ eval ]      or 0.05 unless you have deliberately raised --time."
    fi
fi

# ── Gen14 U11 ── arm C on/off and its NLP backend, for this job only ─────────────────────
# HFFM_VARIANTS   enables arm C without editing config/visual_aligning_eval.yaml
#                 (shipped `hardflow_variants: []` = arm C OFF).
# FMPCC_HF_NLP_BACKEND  slsqp (default) | ipopt. On slsqp the artifacts are renamed
#                 hardflow_new-* -> hardflow_sls-* (hardflow_projection.artifact_variant_label),
#                 so an SLSQP run can never overwrite the IPOPT corpus.
if [ -n "${HFFM_VARIANTS:-}" ]; then
    echo "[ eval ] arm C ENABLED: HFFM_VARIANTS='$HFFM_VARIANTS'"
    echo "[ eval ]   NLP backend = ${FMPCC_HF_NLP_BACKEND:-slsqp (default)}"
    if [ "${FMPCC_HF_NLP_BACKEND:-slsqp}" = "slsqp" ]; then
        echo "[ eval ]   -> artifacts written as hardflow_sls-*  (IPOPT corpus untouched)"
    else
        echo "[ eval ]   ⚠  ipopt writes hardflow_new-* — the SAME names as the existing corpus."
        echo "[ eval ]      Different T means a different results dir, so this is safe here,"
        echo "[ eval ]      but do not re-run ipopt at T=0.5 without FORCE/overwrite intent."
    fi
    if [ -n "${HFFM_ACT_THRESHOLD:-}" ]; then
        echo "[ eval ]   arm C threshold OVERRIDE = $HFFM_ACT_THRESHOLD (arms B and C deliberately UNMATCHED)"
    else
        echo "[ eval ]   arm C threshold inherits arm B's T (arms B and C matched)"
    fi
else
    echo "[ eval ] arm C (HardFlow) OFF — set HFFM_VARIANTS to enable"
fi

# $SEEDS and $FLOW_ARG are intentionally unquoted: $SEEDS must word-split into separate
# --seeds arguments, and $FLOW_ARG must vanish entirely when empty.
run_eval () {                      # $1 = threshold ("" -> config default)
    local targ=()
    if [ -n "$1" ]; then targ=(--proj-threshold "$1"); fi
    python mix_visual_aligning_test/eval_mix_visual_aligning.py \
        --engine "$ENGINE" --seeds $SEEDS --record "$RECORD_MODE" --eval-on-train \
        $FLOW_ARG "${targ[@]}"
}

# Relax `set -e` around the sweep so one failing threshold does not throw away the others;
# the exit status is re-raised at the end so the job still fails loudly.
FAILED=""
if [ -n "$PROJ_T_LIST" ]; then
    set +e
    for T in $PROJ_T_LIST; do
        echo "================================================================================"
        echo "[ eval ] PASS  T = $T   ($(date))"
        echo "================================================================================"
        # 🔴 unset the env form: with --proj-threshold given, a lingering MIX_PROJ_T list
        # ("0.1 0.05") would fail the eval's float() and kill the pass. The CLI flag is the
        # single source of truth inside the loop.
        MIX_PROJ_T= run_eval "$T"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[ eval ] ❌ PASS T=$T FAILED (exit $rc)"
            FAILED="$FAILED T=$T"
        else
            echo "[ eval ] ✅ PASS T=$T done"
        fi
    done
    set -e
else
    run_eval ""
fi

if [ -n "$FAILED" ]; then
    echo "[ eval ] one or more passes failed:$FAILED"
    exit 1
fi
echo "Job completed successfully."
