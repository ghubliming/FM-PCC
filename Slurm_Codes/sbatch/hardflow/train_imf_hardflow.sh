#!/bin/bash
#SBATCH --job-name=hf_imf_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — train the iMF backbone inside HardFlow (avoiding, H16, 100k steps).
# Runs gates G0+G1 FIRST and aborts if they fail (convention/sign safety).
# Produces logs/hardflow/avoiding-v0/flow/H16_imf_100k/model_ema_{0..4}.pth.
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
# Sweep knobs via env: N_TRAIN_STEPS, IMF_DATA_PROPORTION, IMF_P_STD.
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

echo "[ HF-IMF-TRAIN ] gates first: python run/imf_gates.py"
python run/imf_gates.py || { echo "[ HF-IMF-TRAIN ] GATES FAILED — aborting."; exit 1; }

echo "[ HF-IMF-TRAIN ] bash run_scripts/train_imf.sh"
bash run_scripts/train_imf.sh

echo "[ HF-IMF-TRAIN ] done. checkpoints:"
# U9.2: honour IMF_EXP_NAME (same rule as run_scripts/train_imf.sh) — the old
# hardcoded H16_imf_<steps>k made this listing print "(none)" for named runs.
_exp="${IMF_EXP_NAME:-H16_imf_$(( ${N_TRAIN_STEPS:-100000} / 1000 ))k}"
ls -la "logs/avoiding-v0/flow/${_exp}/" 2>/dev/null || echo "  (none — check log above)"
