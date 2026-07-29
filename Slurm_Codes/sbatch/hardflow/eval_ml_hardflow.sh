#!/bin/bash
#SBATCH --job-name=hf_ml_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U12 — evaluate a Mix-ML run (imf|mf|af) on avoiding, RAW + HARDFLOW-proj.
# Clean sibling of eval_imf_hardflow.sh. Two differences that matter:
#   1. NAMING — dirs are named after the OBJECTIVE, not "imf", and grouped one
#      folder per training run:  logs/<env>/eval/<ML_EXP_NAME>/<method>_K<k>[_n<n>]/
#      (method ∈ {raw, hfproj}).  eval_${method}_ml.sh does the tidy naming.
#   2. NO PNG FLOOD — HF_EVAL_SAVE_PNG=0 by default, so the per-episode *_real.png
#      (thousands at n=200×methods×K; it filled the disk and crashed AF-K2 in U11)
#      is skipped. Set HF_EVAL_SAVE_PNG=1 to restore the renders.
#
# Submit (usually via ml_pipeline_hardflow.sh; standalone for re-eval):
#   ML_EXP_NAME=H16_ml_mf_100k ML_CP=4 ML_KS="1 2" RANDOM_REPEAT=200 \
#     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh
# Knobs (env): ML_METHODS ("raw hfproj"), ML_KS ("1 2"), ML_CP (4),
#              RANDOM_REPEAT, HF_EVAL_SAVE_PNG (0), IMF_PLOT_FAN.
# The frozen eval_imf_hardflow.sh (iMF baseline) is untouched.
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

ML_EXP_NAME="${ML_EXP_NAME:-${IMF_EXP_NAME:-H16_ml_imf_100k}}"
ML_METHODS="${ML_METHODS:-raw hfproj}"
ML_KS="${ML_KS:-${IMF_KS:-1 2}}"

# Default OFF: skip the unconditional per-episode *_real.png (disk-flood guard).
export HF_EVAL_SAVE_PNG="${HF_EVAL_SAVE_PNG:-0}"

# dynamics guard (hfproj needs the fitted linear model, same as eval_hardflow.sh)
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-ML-EVAL ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-ML-EVAL ] dynamics model present: $DYN"
fi

echo "[ HF-ML-EVAL ] run='$ML_EXP_NAME' methods='$ML_METHODS' K='$ML_KS' cp=${ML_CP:-4} n=${RANDOM_REPEAT:-50} save_png=$HF_EVAL_SAVE_PNG"

for method in $ML_METHODS; do
    script="run_scripts/eval_${method}_ml.sh"
    if [ ! -f "$script" ]; then
        echo "[ HF-ML-EVAL ] SKIP '$method' — $script not found." >&2
        continue
    fi
    for k in $ML_KS; do
        echo "=============================================================="
        echo "[ HF-ML-EVAL ] method=$method K=$k -> bash $script"
        echo "=============================================================="
        ML_K="$k" ML_EXP_NAME="$ML_EXP_NAME" bash "$script"
    done
done

echo "[ HF-ML-EVAL ] done. tidy results tree under logs/avoiding-v0/eval/$ML_EXP_NAME/ :"
find "logs/avoiding-v0/eval/$ML_EXP_NAME" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort \
    || echo "  (none — check log above)"
