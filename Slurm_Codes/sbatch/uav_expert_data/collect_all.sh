#!/bin/bash
# Convenience wrapper — submits one collect job per scene in parallel.
#
# Usage (run from repo root):
#   ./Slurm_Codes/sbatch/uav_expert_data/collect_all.sh [n_trials] [gain]
#
# Examples:
#   ./Slurm_Codes/sbatch/uav_expert_data/collect_all.sh 500
#   ./Slurm_Codes/sbatch/uav_expert_data/collect_all.sh 500 pid_high_gain
#
# Submits 4 independent SLURM jobs (parallel execution, one per scene).

N_TRIALS="${1:-500}"
GAIN="${2:-pid_default}"

for scene in empty corridor s_curve pillars; do
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh "$scene" "$N_TRIALS" "$GAIN"
done
