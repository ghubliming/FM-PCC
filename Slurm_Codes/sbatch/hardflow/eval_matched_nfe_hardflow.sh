#!/bin/bash
#SBATCH --job-name=hf_matched_nfe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 fix_7.3 — MATCHED-BUDGET TEST BATTERY (T1-T3).
#
# Purpose: settle whether Gen13's efficiency claim is attributable to iMF at all,
# or is merely "HardFlow with fewer steps". Every previous comparison ran iMF at
# K=5 against FM at K=10, so NFE and NLP-count were confounded with the backbone.
#
#   T1  FM at K = 1,2,5,10   (guided)   <- THE control: can FM just use fewer steps?
#   T2  iMF at K = 1,2,5,10  (guided)   <- reverse control: does iMF improve with budget?
#   T3  both unguided at K = 1,2,5,10   <- raw field quality vs budget, no NLP
#
# Together these give a full K-sweep for BOTH backbones, so the comparison can be
# read at EQUAL K (equal NFE and equal projection count) instead of K=5 vs K=10.
#
# n=20 per cell: enough to separate large safety differences (0% vs ~100%) without
# a 200-episode run. NOT enough to resolve the ~1.5pt residual gap — that is what
# the u_5 n=200 run is for.
#
# Runtime: 16 guided cells + 16 unguided ~ 60-90 min total. 4h requested (2x rule).
#
# Submit: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_matched_nfe_hardflow.sh
# Knobs:  N (default 20), KS ("1 2 5 10"), MODES ("guided unguided")
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

N="${N:-20}"
KS="${KS:-1 2 5 10}"
MODES="${MODES:-guided unguided}"

DYN="logs/avoiding-v0/dynamics/linear_model.npz"
[ -f "$DYN" ] || { echo "[ fix_7.3 ] fitting dynamics"; python run/fit_dynamics.py; }

for mode in $MODES; do
  for k in $KS; do
    for bk in imf fm; do
      echo "=============================================================="
      echo "[ fix_7.3 ] backbone=$bk  mode=$mode  K=$k  n=$N"
      echo "=============================================================="
      BACKBONE="$bk" GUIDANCE="$mode" N="$N" \
        IMF_K="$k" FM_K="$k" PLOT_FAN=1 \
        bash run_scripts/eval_smoothness_diag.sh || echo "[ fix_7.3 ] cell FAILED, continuing"
    done
  done
done

# ---------------------------------------------------------------- summary
echo "=============================================================="
echo "[ fix_7.3 ] MATCHED-BUDGET SUMMARY  (compare ACROSS backbones at EQUAL K)"
echo "=============================================================="
printf "  %-34s %5s %8s %8s %14s %12s\n" cell n safe% steps roughness s/plan
for d in logs/avoiding-v0/eval/diag_smooth_*_n${N}; do
    [ -f "$d/trajectories.csv" ] || continue
    python - "$d/trajectories.csv" << 'PY'
import csv, sys, statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
def m(k):
    v=[float(r[k]) for r in rows if r.get(k) and r[k] not in ('','nan')]
    return st.mean(v) if v else float('nan')
safe=sum(1 for r in rows if r['safety'].strip().lower()=='true')
name=sys.argv[1].split('/')[-2].replace('diag_smooth_','')
print(f"  {name:<34} {len(rows):>5} {100*safe/len(rows):>7.0f}% {m('steps'):>8.1f} "
      f"{m('plan_roughness'):>14.3e} {m('average_computation_time'):>12.4f}")
PY
done

# ------------------------------------------------- T4: direct seam test
echo "=============================================================="
echo "[ fix_7.3 ] T4 — x1 terminal-prediction accuracy vs tau (the seam itself)"
echo "=============================================================="
DIRS=""
for k in $KS; do
  for bk in imf fm; do
    d="logs/avoiding-v0/eval/diag_smooth_${bk}_guided_K${k}_n${N}"
    [ -d "$d" ] && DIRS="$DIRS $d"
  done
done
if [ -n "$DIRS" ]; then
    python run/analyze_x1_accuracy.py --dirs $DIRS \
        --out logs/avoiding-v0/eval/x1_accuracy_matched.png \
        || echo "[ fix_7.3 ] T4 failed (see above)"
fi
echo "[ fix_7.3 ] done."
