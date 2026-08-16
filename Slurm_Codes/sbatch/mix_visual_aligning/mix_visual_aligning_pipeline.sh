#!/bin/bash
#SBATCH --job-name=mix_visual_aligning_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

# Exit on error
set -e

# ------------------------------------------------------------------------------
# PRO-LOGGING SETUP
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "PIPELINE START: $(date)"
echo "JOB ID:    $SLURM_JOB_ID"
echo "================================================================================"

# Trap for PIPELINE END
function on_exit {
    echo "================================================================================"
    echo "PIPELINE END:   $(date)"
    echo "================================================================================"
}
trap on_exit EXIT

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION (Smart Unified Session)
# ------------------------------------------------------------------------------
# Inherit session metadata from submit.sh or fallback to current local time
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"

# Sub-jobs share the SAME timestamp as the pipeline manager for perfect grouping
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

# ==============================================================================
# Visual-Mix-ML (Gen14) Pipeline Master Script
# ==============================================================================
# Chains three Slurm jobs for ONE engine arm:
#   1. Gates      (gates_mix_visual.sh)   — fidelity + objective sanity, ~minutes
#   2. Training   (train_mix_visual_aligning.sh)  — only if the gates pass
#   3. Evaluation (eval_mix_visual_aligning.sh)   — only if training succeeds
#
# Usage:  sbatch mix_visual_aligning_pipeline.sh <engine> [seeds]
#           <engine> = diffusion | fm | mf | af      (default: fm, the Gen7 reference arm)
#           [seeds]  = space-separated seed list, QUOTED   (default: "6 7 8 9 10")
#
#           [K]      = NFE override for the eval stage (U6), fm/mf/af only
#
#   sbatch mix_visual_aligning_pipeline.sh mf                # all 5 default seeds
#   sbatch mix_visual_aligning_pipeline.sh mf 6              # seed 6 only (smoke run)
#   sbatch mix_visual_aligning_pipeline.sh af "6 7 8"        # a subset
#   sbatch mix_visual_aligning_pipeline.sh mf "6 7" 2        # NFE override (U6)
#   MIX_SEEDS="6 7" sbatch mix_visual_aligning_pipeline.sh mf   # via env instead
#
# The gates run FIRST on purpose: a broken copy or a dead alpha schedule is cheap to
# catch in one minute and expensive to discover after a 24-hour training run.
#
# FAN-OUT (Gen14 multi-seed): this submits ONE train job and ONE eval job PER SEED, all
# gated on a SINGLE shared gates job. Rationale: the gates are seed-independent (they check
# copy fidelity, the JVP, the alpha schedule and the projector — none of which touch a
# seed), so running them five times would be pure waste; whereas visual training is far too
# slow to serialise five seeds against one 24 h wall. Each seed therefore gets its own
# 24 h budget and they run in parallel, subject to queue capacity.
#
#   gates ──┬─> train(seed 6) ──> eval(seed 6)
#           ├─> train(seed 7) ──> eval(seed 7)
#           └─> ...
#
# Each eval depends only on ITS OWN train, so one seed dying does not block the others.
# ==============================================================================

SBATCH_DIR="Slurm_Codes/sbatch/mix_visual_aligning"
ENGINE="${1:-fm}"
SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
# ── Gen14 U6 ── $3 = NFE override, forwarded to every eval in the fan-out. Blank -> config
# default (mf/af: 2, fm: 20 since U7 2026-08-08). Training is unaffected: flow_steps_v3 is an
# inference-only key. engine=diffusion has no override — its K is n_diffusion_steps (20), a
# training key, so changing it means retraining.
#   sbatch mix_visual_aligning_pipeline.sh mf "6 7" 2
FLOW_STEPS="${3:-}"

case "$ENGINE" in
    diffusion|fm|mf|af) ;;
    ddpm) ENGINE=diffusion; echo "[ engine ] NOTE: 'ddpm' is a deprecated alias for 'diffusion' (Gen14 U5)" ;;
    *) echo "ERROR: unknown engine '$ENGINE' (want: diffusion | fm | mf | af)"; exit 1 ;;
esac

if [ -n "$FLOW_STEPS" ] && [ "$ENGINE" = "diffusion" ]; then
    echo "ERROR: NFE override (\$3) does not apply to engine 'diffusion' — its NFE key is"
    echo "       n_diffusion_steps, set in the config block."
    exit 1
fi

# ── Gen14 Fix_9 ── FiLM backbone. Each arm has its OWN knob, MIX_FILM_MODE_<ENGINE>, with a
# bare MIX_FILM_MODE as the all-arms fallback (default v1). Read the block comment above
# base['mix_visual_aligning_*'] in config/aligning-d3il-visual.py for what v1/v2 are; the
# short version is that v2 is a TRUE-FiLM backbone, it is an ARCHITECTURE key, and switching
# it REQUIRES A RETRAIN — which is why it is set here, on the pipeline that trains, and not
# on a standalone eval.
#
#   MIX_FILM_MODE=v2       ./Slurm_Codes/submit.sh <this script> mf "6"   # same thing;
#   MIX_FILM_MODE_MF=v2    ./Slurm_Codes/submit.sh <this script> mf "6"   # explicit form
#
# Either input works: this pipeline runs exactly ONE engine, so it resolves the mode for
# $ENGINE and re-exports it as the ARM-SPECIFIC variable. That narrowing matters — the child
# jobs import a config that defines all four arms, and exporting only MIX_FILM_MODE_<ENGINE>
# means the other three still resolve 'v1' inside that very process. The fm arm in
# particular is the Gen7 reference and must not be moved by an mf sweep.
#
# Validated here so a typo dies at submit time instead of after a GPU allocation, and
# exported EXPLICITLY onto each child job below rather than relying on inherited
# --export=ALL: a silently-dropped env var would train v1 while the operator believes it is
# running v2. (Even then the result would be visible — film_mode is a path key and both
# scripts print it — but "visible afterwards" is not the same as "cannot happen".)
ENGINE_UC=$(echo "$ENGINE" | tr '[:lower:]' '[:upper:]')
eval "FILM_MODE=\${MIX_FILM_MODE_${ENGINE_UC}:-\${MIX_FILM_MODE:-v1}}"
case "$FILM_MODE" in
    v1|v2) ;;
    *) echo "ERROR: FiLM mode '$FILM_MODE' is not known (want: v1 | v2). Set it via"
       echo "       MIX_FILM_MODE_${ENGINE_UC}=<mode> (this arm) or MIX_FILM_MODE=<mode> (all arms)."
       exit 1 ;;
esac
# Arm-specific ONLY. Any bare MIX_FILM_MODE on the submitting environment still rides along
# via ALL, but the arm-specific variable takes precedence in _film_mode(), so $ENGINE gets
# what was resolved here either way.
EXPORT_OPTS="--export=ALL,MIX_FILM_MODE_${ENGINE_UC}=$FILM_MODE"
if [ "$FILM_MODE" = "v2" ]; then
    echo "[ pipeline ] film_mode = v2 — TRUE FiLM backbone. This is a RETRAIN into a separate"
    echo "[ pipeline ]              '..._filmv2_E${ENGINE}' checkpoint tree; v1 runs are untouched."
    echo "[ pipeline ]              🔴 v2 has never executed a tensor op on any Gen14 arm — the"
    echo "[ pipeline ]              gates job below (G7) is the first thing that will."
else
    echo "[ pipeline ] film_mode = v1 (default, additive-bias FiLM)"
fi

echo "Launching Visual-Mix-ML (Gen14) Pipeline — engine=$ENGINE seeds='$SEEDS'${FLOW_STEPS:+ K=$FLOW_STEPS} film=$FILM_MODE ..."

# 1. Gates — ONE job, shared by every seed (seed-independent by construction).
# G7 builds ALL FOUR arms at v2 regardless of MIX_FILM_MODE, so the gates are film-mode
# independent just as they are seed independent — no need to fan them out per mode either.
GATE_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/gates_mix_visual.sh")
echo "Step 1: Gates submitted. Job ID: $GATE_ID"

# 2/3. One train->eval chain per seed. $SEEDS is unquoted so it word-splits.
# $EXPORT_OPTS carries MIX_FILM_MODE explicitly: train BUILDS the v1/v2 backbone and eval
# must REBUILD the matching one, so both ends of the chain need it.
for SEED in $SEEDS; do
    TRAIN_ID=$(sbatch --parsable $LOG_OPTS $EXPORT_OPTS --dependency=afterok:$GATE_ID \
        "${SBATCH_DIR}/train_mix_visual_aligning.sh" "$ENGINE" "$SEED")
    echo "  seed $SEED: train scheduled (afterok:$GATE_ID). Job ID: $TRAIN_ID"

    # $3 (record mode) must be passed positionally so $4 (NFE) lands in the right slot.
    EVAL_ID=$(sbatch --parsable $LOG_OPTS $EXPORT_OPTS --dependency=afterok:$TRAIN_ID \
        "${SBATCH_DIR}/eval_mix_visual_aligning.sh" "$ENGINE" "$SEED" all "$FLOW_STEPS")
    echo "  seed $SEED: eval  scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
done

echo "--------------------------------------------------------------------------------"
echo "Visual-Mix-ML (Gen14) Pipeline submitted — engine=$ENGINE seeds='$SEEDS'."
echo "Use 'squeue -u $USER' to monitor progress."
echo "A failed stage cancels its OWN downstream stage; other seeds are unaffected."
echo "================================================================================"
