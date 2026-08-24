#!/bin/bash
#SBATCH --job-name=uav_mix_ksweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student
set -e

# Gen15's headline experiment: the SAME scene/seed/variant evaluated at several NFE budgets K,
# for one arm. Thin submitter — one eval job per K, all in parallel; exits immediately.
#
# 🔴 MATCHED BUDGET OR NOTHING. The comparison Gen15 exists to make is "at equal K, which arm
# holds success and what does the freed wall-clock buy". Run this script once per engine with
# the SAME K list, never with per-arm K lists.
#
# Usage (from repo root):
#   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf corridor "6" "1 2 5 10 20"
# Args: $1=engine (fm|mf|af|diffusion) [fm]  $2=scene [all]  $3=seeds (quoted) ["6"]
#       $4=K list (quoted, space-sep) ["1 2 5 10 20"]  $5=n_trials [omit → yaml]  $6=projection [fm_only]
#       $7=record (none|gif|all) [none]
# NOTE record=gif/all renders one GIF per rollout (variants x trials of them) -> disk + job time.
# It does NOT contaminate the metrics: eval_mix_uav.py times ONLY the policy() call, and
# rendering happens outside that window. So a record=all run is timing-comparable to a
# record=none run.
ENGINE="${1:-fm}"
case "$ENGINE" in fm|mf|af|diffusion) ;; *) echo "[ ERROR ] engine must be fm|mf|af|diffusion (got '$ENGINE')"; exit 1 ;; esac
SCENE="${2:-all}"
SEEDS="${3:-6}"
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
KS="${4:-1 2 5 10 20}"
NTRIALS="${5:-}"
PROJ="${6:-fm_only}"
RECORD="${7:-none}"
EVAL="Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh"

N_SEEDS=$(echo $SEEDS | wc -w)
EVAL_HOURS=$((N_SEEDS * 8))

echo "================================================================================"
echo "UAV-MIX K SWEEP  $(date)   engine=$ENGINE  scene=$SCENE  seeds=[$SEEDS]  K=[$KS]  proj=$PROJ  record=$RECORD"
echo "================================================================================"

DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}; TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"; mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

n=0
for k in $KS; do
    # Each K writes its own output tree (…/E<engine>_K<k>_mpc<B>_<controller>_T<thresh>/…), so
    # these jobs are independent and cannot overwrite one another.
    ID=$(sbatch --parsable --time="${EVAL_HOURS}:00:00" $LOG_OPTS \
         "$EVAL" "$ENGINE" "$SCENE" "$SEEDS" "$NTRIALS" "$PROJ" "$RECORD" "$k")
    echo "  eval   engine=$ENGINE scene=$SCENE K=$k  → Job $ID"
    n=$((n+1))
done
echo "--------------------------------------------------------------------------------"
echo "Submitted $n eval jobs (one per K)."
echo "Results: logs/UAV_MIX/uav-<scene>/plans/mix_uav_$ENGINE/<train-id>/E${ENGINE}_K<k>_.../<seed>/<variant>/results.json"
echo "Repeat for every arm with the SAME K list, then compare at matched K."
echo "================================================================================"
