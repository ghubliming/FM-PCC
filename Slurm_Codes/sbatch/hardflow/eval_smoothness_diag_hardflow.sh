#!/bin/bash
#SBATCH --job-name=hf_smooth_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 fix_7 — MPC planned-trajectory SMOOTHNESS diagnostic, 2x2 matrix, small n.
#
#   backbone {imf, fm}  x  guidance {unguided, guided}   at N episodes (default 5)
#
# Tests the hypothesis in Research/DISCUSSION_foresight_fan_and_smoothness_
# paradigms.md: HardFlow enforces dynamic feasibility as a HARD NLP constraint,
# so smoothness should be *manufactured by the projection* rather than inherited
# from the generative field. Prediction: unguided iMF is roughest (coarse field,
# 0.37/dim); both GUIDED variants collapse to similar, much lower roughness.
#
# n=5 is enough because each episode contributes ~6-7 replans => ~30-35 planned
# horizons per cell; roughness is a per-plan mean, not a rare-event rate (unlike
# the violation rate, which genuinely needed n=200).
#
# Outputs per cell: trajectories.csv (with `plan_roughness` column) + *_fan.png
# Runtime: ~1-3 min per cell => ~10 min total. 1h requested (2x margin rule).
#
# Submit: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_hardflow.sh
# Knobs:  N (episodes/cell), CELLS (subset, e.g. "imf:guided fm:guided")
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

N="${N:-5}"
CELLS="${CELLS:-imf:unguided imf:guided fm:unguided fm:guided}"

DYN="logs/avoiding-v0/dynamics/linear_model.npz"
[ -f "$DYN" ] || { echo "[ fix_7 ] fitting dynamics"; python run/fit_dynamics.py; }

for cell in $CELLS; do
    bk="${cell%%:*}"; gd="${cell##*:}"
    echo "=============================================================="
    echo "[ fix_7 ] cell: backbone=$bk guidance=$gd  n=$N"
    echo "=============================================================="
    BACKBONE="$bk" GUIDANCE="$gd" N="$N" bash run_scripts/eval_smoothness_diag.sh
done

echo "=============================================================="
echo "[ fix_7 ] SMOOTHNESS SUMMARY (plan_roughness: lower = smoother)"
echo "=============================================================="
for d in logs/avoiding-v0/eval/diag_smooth_*_n${N}; do
    [ -f "$d/trajectories.csv" ] || continue
    python - "$d/trajectories.csv" << 'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
def col(k):
    v=[float(x[k]) for x in rows if x.get(k) and x[k]==x[k] and x[k]!='nan']
    return v
r=col('plan_roughness'); rr=col('plan_roughness_raw')
safe=sum(1 for x in rows if x['safety'].strip().lower()=='true')
name=sys.argv[1].split('/')[-2]
if r:
    raw_s = f"raw={st.mean(rr):.3e}  ratio={st.mean(rr)/max(st.mean(r),1e-30):7.1f}x" if rr else "raw=n/a"
    print(f"  {name:<42} n={len(rows):<3} projected={st.mean(r):.3e}  {raw_s}  safe={100*safe/len(rows):.0f}%")
PY
done
echo "=============================================================="
