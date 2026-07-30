#!/bin/bash
#SBATCH --job-name=hf_ml_smooth_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U13 — MPC foresight-fan / smoothness diagnostic MATRIX for Mix-ML.
#
# Default matrix: {mf, af} x {hfproj} x K{1,2}  (the exact ask: "raw and
# projected traj smooth on K1,2 mf and af"). GUIDANCE=hfproj is what actually
# gives BOTH numbers (plan_roughness = projected, plan_roughness_raw = raw
# warmstart-before-NLP) per replanned horizon — see eval_smoothness_diag_ml.sh.
#
# Outputs per cell (family-first, U12.2 tree):
#   logs/avoiding-v0/eval/<ml_type>/<run>/smooth_<guidance>_K<k>_n<n>/
#     trajectories.csv (plan_roughness, plan_roughness_raw)  + *_fan.png/.npz
#
# Submit: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_ml_hardflow.sh
# Knobs:  N (episodes/cell, default 5)
#         CELLS (default "mf:hfproj:1 mf:hfproj:2 af:hfproj:1 af:hfproj:2")
#           format <ml_type>:<guidance>:<K>, e.g. add "imf:hfproj:2" for the
#           frozen-iMF reference cell, or ":raw:" cells for the no-NLP field.
#         ML_EXP_NAME / ML_CP per-cell override (else default <type>/H16_ml_<type>_100k cp4)
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

N="${N:-5}"
CELLS="${CELLS:-mf:hfproj:1 mf:hfproj:2 af:hfproj:1 af:hfproj:2}"

DYN="logs/avoiding-v0/dynamics/linear_model.npz"
[ -f "$DYN" ] || { echo "[ U13-smooth ] fitting dynamics"; python run/fit_dynamics.py; }

CELL_DIRS=()
for cell in $CELLS; do
    ml_type="${cell%%:*}"; rest="${cell#*:}"; gd="${rest%%:*}"; k="${rest##*:}"
    echo "=============================================================="
    echo "[ U13-smooth ] cell: ml_type=$ml_type guidance=$gd K=$k n=$N"
    echo "=============================================================="
    ML_TYPE="$ml_type" GUIDANCE="$gd" ML_K="$k" N="$N" bash run_scripts/eval_smoothness_diag_ml.sh
    flow_exp_name="${ML_EXP_NAME:-${ml_type}/H16_ml_${ml_type}_100k}"
    CELL_DIRS+=("logs/avoiding-v0/eval/${flow_exp_name}/smooth_${gd}_K${k}_n${N}")
done

echo "=============================================================="
echo "[ U13-smooth ] SMOOTHNESS SUMMARY  (roughness: lower = smoother; raw = pre-NLP warmstart)"
echo "=============================================================="
for d in "${CELL_DIRS[@]}"; do
    [ -f "$d/trajectories.csv" ] || { echo "  MISSING: $d (run failed or not evaluated)"; continue; }
    python - "$d/trajectories.csv" "$d" << 'PY'
import csv, sys, statistics as st
csv_path, dirpath = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path)))
def col(k):
    return [float(x[k]) for x in rows if x.get(k) and x[k] == x[k] and x[k] != 'nan']
proj, raw = col('plan_roughness'), col('plan_roughness_raw')
safe = sum(1 for x in rows if x.get('safety', '').strip().lower() == 'true')
name = '/'.join(dirpath.split('/')[-3:])   # <ml_type>/<run>/<smooth_arm>
if proj:
    raw_s = (f"raw={st.mean(raw):.3e}  ratio={st.mean(raw)/max(st.mean(proj),1e-30):7.1f}x"
              if raw else "raw=n/a (GUIDANCE=raw has no pre-NLP plan to compare)")
    print(f"  {name:<48} n={len(rows):<3} projected={st.mean(proj):.3e}  {raw_s}  safe={100*safe/len(rows):.0f}%")
else:
    print(f"  {name:<48} (no plan_roughness data)")
PY
done
echo "=============================================================="
echo "[ U13-smooth ] fan images (visual raw-vs-projected): find logs/avoiding-v0/eval -name '*_fan.png' -path '*smooth_*'"
