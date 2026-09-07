#!/bin/bash
#SBATCH --job-name=mix_visual_avoiding_ksweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
set -e

# Gen16's headline experiment: ONE checkpoint, ONE seed set, evaluated at several NFE budgets
# K. Thin submitter — one eval job per K, all in parallel; exits immediately.
#
# 🔴 MATCHED BUDGET OR NOTHING. The comparison Gen16 exists to make is "at equal K, which ML
#    objective holds success-and-constraints, and what does the freed wall-clock buy". Run
#    this once per engine with the SAME K list — never with per-arm K lists. Gen3v6's
#    fix_7.3 §9 records what a single hard-coded K cost: it made the decisive control
#    unrunnable and killed an entire generation's claim.
#
# 🔴 NOT VALID FOR THE `diffusion` ARM. Its NFE key is `n_diffusion_steps`, which is the DDPM
#    training chain length AND a checkpoint-path key — changing it needs a retrain, not an
#    eval flag. The eval refuses the override, so this script refuses the arm.
#
# K also sets the PROJECTION budget, which is the expensive half: the sampler projects on
# every step from int((1 - T) * K) to the end, so at T=0.5 a K=20 run does 10 SLSQP solves
# per replan and a K=2 run does 1. Expect the wall-clock to scale with K, not just the NFE.
#
# Each K writes its OWN results directory (`flow_steps_v3` is the 'K' token in
# args_to_watch_mix_visual_plan), so these jobs are independent and cannot overwrite one
# another.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_avoiding/eval_k_sweep.sh mf "6 7" "1 2 5 10 20"
#
# Args: $1=engine (fm|mf|af) [fm]   $2=seeds (quoted) ["6 7 8 9 10"]
#       $3=K list (quoted, space-separated) ["1 2 5 10 20"]
#       $4=hours per eval job [8 per seed]

ENGINE="${1:-fm}"
case "$ENGINE" in
    fm|mf|af) ;;
    diffusion|ddpm)
        echo "[ ERROR ] the diffusion arm has no eval-time K knob."
        echo "          n_diffusion_steps is the DDPM chain length and a checkpoint-path key;"
        echo "          changing it requires a RETRAIN. Sweep fm|mf|af instead."
        exit 1 ;;
    *) echo "[ ERROR ] engine must be fm|mf|af (got '$ENGINE')"; exit 1 ;;
esac

SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"
# ⚠️ [HFK1 2026-08-24] K=1 and K=2 are DEGENERATE for the HardFlow arm. HardFlow's guidance
#    lives only in ACTIVE NON-TERMINAL ODE steps; the terminal step has tau=1, which kills the
#    endpoint lookahead, snaps instead of nudging, and has no successor call to react. At K=1
#    the only step IS the terminal step, and at K=2 the shipped activation_threshold=0.5 floors
#    step 0 out. Both therefore run Pi_S(Euler sample) = sample-then-project, i.e. DPCC's
#    algorithm with IPOPT instead of SLSQP — NOT HardFlow. The sampler now prints
#    `[hardflow][DEGENERATE]` for these; keep the rows if you want the cheap one-shot-projection
#    comparison (it is matched-NFE since 2026-08-24), but do NOT label them HardFlow results.
#    Non-degenerate from K>=3 at the shipped A=0.5, or K>=2 at A=1.0 (1 genuine step each);
#    K>=5 at A=0.5 gives 2+, the first setting comparable to the paper's N=10 / A=0.5.
#    See logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md
#
# 🔴 [HFK1c 2026-08-30] THE ADVICE ABOVE ("keep the rows ... one-shot-projection comparison")
#    IS WITHDRAWN, and the eval now ENFORCES it. AUDIT_20260830 tested whether those rows
#    deliver the projector-only control they were kept for: 25 of 32 matched cells are
#    0.00 -> 0.00 floor effects, and in the 7 cells with signal HardFlow is WORSE in 5 —
#    K=1/2 varies the projector AND K at once, so the comparison runs where the sample is a
#    single Euler step and every arm floors. The clean instrument is A=0.0 at matched K.
#    The eval now DROPS the HardFlow variants at degenerate K (HF_DEGENERATE_SKIPPED.txt
#    sentinel); the DPCC/diffuser arms at those K still run. Knobs:
#      FMPCC_HF_ALLOW_DEGENERATE=1  run it anyway (only for the A=0.0 control at K>=5)
#      FMPCC_HF_MIN_GENUINE=2       also block THIN; 0 disables the guard
#    See logs_in_develop/aggregated_hardflow_lowK/AUDIT_20260830_*.md
export FMPCC_HF_ALLOW_DEGENERATE="${FMPCC_HF_ALLOW_DEGENERATE:-}"
export FMPCC_HF_MIN_GENUINE="${FMPCC_HF_MIN_GENUINE:-}"
KS="${3:-1 2 5 10 20}"

N_SEEDS=$(echo $SEEDS | wc -w)
EVAL_HOURS="${4:-$((N_SEEDS * 8))}"
if [ "$EVAL_HOURS" -gt 24 ]; then EVAL_HOURS=24; fi

EVAL="Slurm_Codes/sbatch/mix_visual_avoiding/eval_mix_visual_avoiding.sh"

echo "================================================================================"
echo "VISUAL-AVOIDING MIX K SWEEP (Gen16)  $(date)"
echo "  engine=$ENGINE  seeds=[$SEEDS]  K=[$KS]  wall=${EVAL_HOURS}h/job"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

n=0
for k in $KS; do
    # $3 (record mode) is passed positionally so $4 (the NFE override) lands correctly.
    ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS \
         "$EVAL" "$ENGINE" "$SEEDS" all "$k")
    echo "  eval   engine=$ENGINE  K=$k  -> Job $ID"
    n=$((n+1))
done

echo "--------------------------------------------------------------------------------"
echo "Submitted $n eval jobs (one per K)."
echo "Results: logs/avoiding-d3il-visual-mix/plans/mix_visual_avoiding_$ENGINE/<ckpt-id>/H8_K<k>_.../results/"
echo "Repeat for every arm with the SAME K list, then compare at matched K."
echo "================================================================================"
