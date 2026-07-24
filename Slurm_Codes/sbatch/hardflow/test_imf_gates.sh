#!/bin/bash
#SBATCH --job-name=hf_imf_gates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # minimal — gates are CPU-only, tiny MLP
#SBATCH --mem=8G                    # minimal
#SBATCH --gres=gpu:1                # gpu-1-student requires a gres request;
                                     # the job itself never touches the GPU
#SBATCH --time=00:10:00             # gates finish in well under a minute
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 — run ONLY the iMF gates (G0 shapes + G1 sign/convention check) on a
# compute node, NOT interactively on the login node ("lobby"). Deliberately
# separate from train_imf_hardflow.sh (which requests 24h/32G for the real
# training run) — this is a <1-minute sanity check, sized like
# sbatch/verify_env_job.sh's "minimal verification" pattern.
#
# No extra packages needed: hardflow_clone already has torch/numpy/einops/
# matplotlib (inherited from the FMPCC/DPCC clone) — confirmed sufficient by a
# local CPU-only run on 2026-07-18 (all 4 gates passed).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/test_imf_gates.sh
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

echo "[ HF-IMF-GATES ] python run/imf_gates.py"
python run/imf_gates.py
