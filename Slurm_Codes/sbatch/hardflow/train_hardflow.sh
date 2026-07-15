#!/bin/bash
#SBATCH --job-name=hf_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Train the HardFlow FM backbone (CFM, H16, 1e6 steps). Produces the checkpoint
# logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth (lands in FM-PCC via bridge).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_hardflow.sh
# Skip entirely if you download the released .pth instead (Path B).
#
# Env: activates the FMPCC CLONE (CONDA_ENV_NAME, default hardflow_clone).
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

# Reuse HardFlow's OWN training script — paper hyper-params stay baked in,
# nothing duplicated. It writes to ./logs (symlinked into FM-PCC).
echo "[ HF-TRAIN ] bash run_scripts/train.sh"
bash run_scripts/train.sh

echo "[ HF-TRAIN ] done. checkpoint(s):"
ls -la logs/avoiding-v0/flow/H16_1e6steps/ 2>/dev/null || echo "  (none — check the run log above)"
