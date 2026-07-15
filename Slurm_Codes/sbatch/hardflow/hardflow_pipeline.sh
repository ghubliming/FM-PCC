#!/bin/bash
#SBATCH --job-name=hf_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# One-job pipeline: (train if no checkpoint) -> fit_dynamics -> eval.
# Mirrors sbatch/iMF/imf_pipeline.sh. Optional convenience; the separate
# train_hardflow.sh / eval_hardflow.sh give finer control.
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh
# Skip training (use downloaded .pth):  SKIP_TRAIN=1 ./Slurm_Codes/submit.sh .../hardflow_pipeline.sh
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

METHODS="${METHODS:-hardflow_new original}"
CKPT="logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth"

# --- 1. Train the FM backbone (unless a checkpoint exists or SKIP_TRAIN=1) -----
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "[ HF-PIPE ] SKIP_TRAIN=1 — expecting an existing/downloaded checkpoint."
elif [ -f "$CKPT" ]; then
    echo "[ HF-PIPE ] checkpoint present ($CKPT) — skipping training."
else
    echo "[ HF-PIPE ] no checkpoint -> bash run_scripts/train.sh"
    bash run_scripts/train.sh
fi

# --- 2. Fit dynamics once if absent -------------------------------------------
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-PIPE ] dynamics missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
fi

# --- 3. Eval each method via HardFlow's own scripts ---------------------------
for method in $METHODS; do
    script="run_scripts/eval_${method}.sh"
    [ -f "$script" ] || { echo "[ HF-PIPE ] SKIP '$method' — $script not found." >&2; continue; }
    echo "[ HF-PIPE ] method=$method -> bash $script"
    bash "$script"
done

echo "[ HF-PIPE ] done. results:"
ls -la logs/avoiding-v0/eval/ 2>/dev/null || echo "  (none — check the run log above)"
