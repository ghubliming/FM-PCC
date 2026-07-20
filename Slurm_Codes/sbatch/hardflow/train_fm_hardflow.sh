#!/bin/bash
#SBATCH --job-name=hf_fm_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U9 — train the FM (HardFlow) backbone OURSELVES, with W&B + CSV curves.
#
# The replication used the authors' downloaded 1e6-step checkpoint because
# run/train.py crashes without tensorboard (job 23559, 4 s). run/train_fm.py
# fixes that (try-import) and adds W&B + metrics.csv, with IDENTICAL training
# math — so this finally produces an FM loss curve comparable to Gen13's iMF one.
#
# NOTE 1e6 steps at FM's speed will likely EXCEED the 24 h cap. Start with a
# reduced budget matched to iMF for a fair curve comparison:
#     N_TRAIN_STEPS=100000 ./Slurm_Codes/submit.sh .../train_fm_hardflow.sh
#
# Submit: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_fm_hardflow.sh
# Knobs:  N_TRAIN_STEPS, FM_EXP_NAME, WANDB_PROJECT, USE_WANDB=0
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

echo "[ HF-FM-TRAIN ] bash run_scripts/train_fm_wandb.sh"
bash run_scripts/train_fm_wandb.sh

echo "[ HF-FM-TRAIN ] done. artifacts:"
ls -la logs/avoiding-v0/flow/${FM_EXP_NAME:-H16_fm_$(( ${N_TRAIN_STEPS:-1000001} / 1000 ))k}/ 2>/dev/null || echo "  (none — check log)"
