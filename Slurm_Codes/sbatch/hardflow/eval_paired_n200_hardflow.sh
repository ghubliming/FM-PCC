#!/bin/bash
#SBATCH --job-name=hf_paired_n200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 u_5 — THE DECISIVE PAIRED SAFETY RUN (INSIGHTS_Gen13_first_run.md §15).
#
# Runs BOTH arms back-to-back in ONE job, same seed, same env, same node:
#   Arm A: iMF  hardflow_new_imf, K=5, cp 4   (the candidate)
#   Arm B: FM   hardflow_new, ode_t_steps=10  (the baseline, RE-RUN at the same n)
#
# WHY BOTH: the frozen FM baseline is only 0/50, whose 95% CI upper bound is 7.1% —
# FM's true violation rate is NOT established either. Comparing a new n=200 iMF run
# against that would be an unfair, underpowered comparison. Both arms at identical n
# is the only way to get two rates with comparable confidence intervals.
#
# POWER: with a true 2% violation rate, n=200 has a 98% chance of revealing >=1
# violation (vs only 64% at n=50) — this is what converts "1 vs 0" into a real answer.
#
# --time: measured n=50 runtimes were iMF K5 3m44s and FM ~1.74x slower per plan
# => ~15 min + ~26 min = ~45 min expected at n=200. Requesting 2h per the standing
# 2x-safety-margin rule (24h partition cap).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_paired_n200_hardflow.sh
# Knobs:   N (default 200), IMF_K (default 5), IMF_CP (default 4), ARMS ("A B")
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

N="${N:-200}"
IMF_K="${IMF_K:-5}"
IMF_CP="${IMF_CP:-4}"
ARMS="${ARMS:-A B}"

echo "================================================================================"
echo "[ u_5 PAIRED ] n=$N per arm | arms: $ARMS | iMF K=$IMF_K cp=$IMF_CP"
echo "[ u_5 PAIRED ] frozen n=50 artifacts are preserved (new exp_names carry _n$N)"
echo "================================================================================"

# --- dynamics guard (both arms use --dynamics_constraint) ---------------------
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ u_5 ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ u_5 ] dynamics model present: $DYN"
fi

# --- required checkpoints ------------------------------------------------------
IMF_CKPT="logs/avoiding-v0/flow/H16_imf_100k/model_ema_${IMF_CP}.pth"
FM_CKPT="logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth"
for c in "$IMF_CKPT" "$FM_CKPT"; do
    [ -f "$c" ] || { echo "[ u_5 ] ERROR: missing checkpoint $c" >&2; exit 1; }
done
echo "[ u_5 ] both checkpoints present."

# ------------------------------------------------------------------ ARM A: iMF
if [[ " $ARMS " == *" A "* ]]; then
    echo "=============================================================="
    echo "[ u_5 ARM A ] iMF hardflow_new_imf  K=$IMF_K  n=$N"
    echo "=============================================================="
    RANDOM_REPEAT="$N" IMF_K="$IMF_K" IMF_CP="$IMF_CP" \
        bash run_scripts/eval_hardflow_new_imf.sh
fi

# ------------------------------------------------------------------- ARM B: FM
# NOTE: run/eval.py (pre-existing, un-editable) still uses the noisy tqdm run_env,
# which at n=200 would flood the job log. Its verbose output is therefore captured
# to a side file; only the tail + any errors are echoed here. The CSV is unaffected.
if [[ " $ARMS " == *" B "* ]]; then
    echo "=============================================================="
    echo "[ u_5 ARM B ] FM hardflow_new (baseline re-run)  n=$N"
    echo "=============================================================="
    FM_LOG="logs/avoiding-v0/eval/u5_armB_fm_verbose_${SLURM_JOB_ID:-local}.log"
    mkdir -p "$(dirname "$FM_LOG")"
    echo "[ u_5 ARM B ] verbose output -> $FM_LOG (kept out of the job log)"
    if RANDOM_REPEAT="$N" bash run_scripts/eval_hardflow_new_paired.sh > "$FM_LOG" 2>&1; then
        echo "[ u_5 ARM B ] completed. tail:"
        tail -5 "$FM_LOG"
    else
        echo "[ u_5 ARM B ] FAILED — last 40 lines:" >&2
        tail -40 "$FM_LOG" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------- summary
echo "=============================================================="
echo "[ u_5 ] done. result dirs:"
ls -d logs/avoiding-v0/eval/*_n${N} 2>/dev/null || echo "  (none found — check above)"
echo
echo "[ u_5 ] quick violation tallies (authoritative numbers are in the CSVs):"
for d in logs/avoiding-v0/eval/*_n${N}; do
    [ -f "$d/trajectories.csv" ] || continue
    python - "$d/trajectories.csv" << 'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
n = len(rows)
safe = sum(1 for r in rows if r['safety'].strip().lower() == 'true')
viol = sum(float(r['total_violations']) for r in rows)
fails = sum(int(r.get('nlp_failures') or 0) for r in rows)
print(f"    {sys.argv[1].split('/')[-2]:<45} n={n:<4} safe={100*safe/n:5.1f}%  "
      f"violations={viol:.0f}  nlp_failures={fails}")
PY
done
echo "=============================================================="
