#!/bin/bash
#SBATCH --job-name=hf_imf_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — one-job iMF pipeline: gates -> (train if no checkpoint) -> fit_dynamics
# -> eval E1-E4 matrix. Mirrors sbatch/hardflow/hardflow_pipeline.sh (the FM
# pipeline). Optional convenience; the separate train_imf_hardflow.sh /
# eval_imf_hardflow.sh give finer control (e.g. re-eval without retraining).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
# Skip training (reuse an existing checkpoint): SKIP_TRAIN=1 ./Slurm_Codes/submit.sh ...
# Knobs: N_TRAIN_STEPS, IMF_DATA_PROPORTION, IMF_P_STD (train);
#        IMF_METHODS ("original hardflow_new"), IMF_KS ("1 2"), IMF_CP (4) (eval)
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

IMF_METHODS="${IMF_METHODS:-original hardflow_new}"
IMF_KS="${IMF_KS:-1 2}"
CKPT="logs/avoiding-v0/flow/H16_imf_100k/model_ema_4.pth"

# --- 0. Gates FIRST — abort before any GPU training if they fail --------------
echo "[ HF-IMF-PIPE ] gates first: python run/imf_gates.py"
python run/imf_gates.py || { echo "[ HF-IMF-PIPE ] GATES FAILED — aborting."; exit 1; }

# --- 1. Train the iMF backbone (unless a checkpoint exists or SKIP_TRAIN=1) ---
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "[ HF-IMF-PIPE ] SKIP_TRAIN=1 — expecting an existing iMF checkpoint."
elif [ -f "$CKPT" ]; then
    echo "[ HF-IMF-PIPE ] checkpoint present ($CKPT) — skipping training."
else
    echo "[ HF-IMF-PIPE ] no checkpoint -> bash run_scripts/train_imf.sh"
    bash run_scripts/train_imf.sh
fi

# --- 2. Fit dynamics once if absent (hardflow_new_imf needs it) ---------------
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-IMF-PIPE ] dynamics missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-IMF-PIPE ] dynamics model present: $DYN"
fi

# --- 3. Eval the E1-E4 matrix (methods x K) via the iMF run scripts -----------
for method in $IMF_METHODS; do
    script="run_scripts/eval_${method}_imf.sh"
    [ -f "$script" ] || { echo "[ HF-IMF-PIPE ] SKIP '$method' — $script not found." >&2; continue; }
    for k in $IMF_KS; do
        echo "[ HF-IMF-PIPE ] method=$method K=$k -> bash $script"
        IMF_K="$k" bash "$script"
    done
done

echo "[ HF-IMF-PIPE ] done. results:"
ls -la logs/avoiding-v0/eval/ 2>/dev/null | grep -i imf || echo "  (none — check the run log above)"
