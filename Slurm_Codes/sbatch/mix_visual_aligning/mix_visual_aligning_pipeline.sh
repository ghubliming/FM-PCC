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
#   sbatch mix_visual_aligning_pipeline.sh mf                # all 5 default seeds
#   sbatch mix_visual_aligning_pipeline.sh mf 6              # seed 6 only (smoke run)
#   sbatch mix_visual_aligning_pipeline.sh af "6 7 8"        # a subset
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

case "$ENGINE" in
    diffusion|fm|mf|af) ;;
    ddpm) ENGINE=diffusion; echo "[ engine ] NOTE: 'ddpm' is a deprecated alias for 'diffusion' (Gen14 U5)" ;;
    *) echo "ERROR: unknown engine '$ENGINE' (want: diffusion | fm | mf | af)"; exit 1 ;;
esac

echo "Launching Visual-Mix-ML (Gen14) Pipeline — engine=$ENGINE seeds='$SEEDS' ..."

# 1. Gates — ONE job, shared by every seed (seed-independent by construction).
GATE_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/gates_mix_visual.sh")
echo "Step 1: Gates submitted. Job ID: $GATE_ID"

# 2/3. One train->eval chain per seed. $SEEDS is unquoted so it word-splits.
for SEED in $SEEDS; do
    TRAIN_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$GATE_ID \
        "${SBATCH_DIR}/train_mix_visual_aligning.sh" "$ENGINE" "$SEED")
    echo "  seed $SEED: train scheduled (afterok:$GATE_ID). Job ID: $TRAIN_ID"

    EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID \
        "${SBATCH_DIR}/eval_mix_visual_aligning.sh" "$ENGINE" "$SEED")
    echo "  seed $SEED: eval  scheduled (afterok:$TRAIN_ID). Job ID: $EVAL_ID"
done

echo "--------------------------------------------------------------------------------"
echo "Visual-Mix-ML (Gen14) Pipeline submitted — engine=$ENGINE seeds='$SEEDS'."
echo "Use 'squeue -u $USER' to monitor progress."
echo "A failed stage cancels its OWN downstream stage; other seeds are unaffected."
echo "================================================================================"
