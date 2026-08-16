#!/bin/bash
#SBATCH --job-name=hf_ml_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U11/U12.2 — train a selectable MLbone (imf|mf|af) inside HardFlow.
# Runs the shared iMF gates FIRST (they validate the frozen TemporalImfUnet +
# convention + sampler that ALL three MLbones ride on) and aborts if they fail.
# Produces logs/hardflow/avoiding-v0/flow/<ml_type>/H16_ml_<ml_type>_<steps>k/model_ema_*.pth
# (U12.2: nested one folder per family first — imf/ mf/ af/).
#
# Submit (via the pipeline is preferred):
#   ML_TYPE=mf ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_ml_hardflow.sh
# Knobs: ML_TYPE (imf|mf|af), N_TRAIN_STEPS, ML_EXP_NAME, IMF_LR, IMF_GRAD_CLIP,
#        MF_DATA_PROPORTION/MF_P_STD, AF_ALPHA_*/AF_RATIO_FM, USE_WANDB, FORCE_OVERWRITE.
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

# Shared-convention safety: these gates test the frozen iMF matcher + TemporalImfUnet
# + sampler, which MF/AF also depend on. Always cheap; abort if the shared base broke.
echo "[ HF-ML-TRAIN ] shared gates first: python run/imf_gates.py"
python run/imf_gates.py || { echo "[ HF-ML-TRAIN ] GATES FAILED — aborting."; exit 1; }

echo "[ HF-ML-TRAIN ] ML_TYPE=${ML_TYPE:-imf}  ->  bash run_scripts/train_ml.sh"
bash run_scripts/train_ml.sh

echo "[ HF-ML-TRAIN ] done. checkpoints:"
_ml="${ML_TYPE:-imf}"
_exp="${ML_EXP_NAME:-${_ml}/H16_ml_${_ml}_$(( ${N_TRAIN_STEPS:-100000} / 1000 ))k}"
ls -la "logs/avoiding-v0/flow/${_exp}/" 2>/dev/null || echo "  (none — check log above)"
