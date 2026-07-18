#!/bin/bash
#SBATCH --job-name=hf_imf_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — evaluate the iMF backbone on avoiding (unguided + seam-swapped HardFlow).
# Writes per-method trajectories.csv (incl. NFE columns) into
# FM-PCC/logs/hardflow/avoiding-v0/eval/H16_imf_<method>_K<k>/.
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_imf_hardflow.sh
# Knobs (env): IMF_METHODS ("original hardflow_new"), IMF_KS ("1 2"), IMF_CP (4).
# Requires: trained iMF checkpoint (train_imf_hardflow.sh) + fitted dynamics.
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

IMF_METHODS="${IMF_METHODS:-original hardflow_new}"
IMF_KS="${IMF_KS:-1 2}"

# dynamics guard (same as eval_hardflow.sh — hardflow_new_imf needs it)
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-IMF-EVAL ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-IMF-EVAL ] dynamics model present: $DYN"
fi

for method in $IMF_METHODS; do
    script="run_scripts/eval_${method}_imf.sh"
    if [ ! -f "$script" ]; then
        echo "[ HF-IMF-EVAL ] SKIP '$method' — $script not found." >&2
        continue
    fi
    for k in $IMF_KS; do
        echo "=============================================================="
        echo "[ HF-IMF-EVAL ] method=$method K=$k -> bash $script"
        echo "=============================================================="
        IMF_K="$k" bash "$script"
    done
done

echo "[ HF-IMF-EVAL ] done. results under logs/avoiding-v0/eval/ :"
ls -la logs/avoiding-v0/eval/ 2>/dev/null | grep -i imf || echo "  (none — check log above)"
