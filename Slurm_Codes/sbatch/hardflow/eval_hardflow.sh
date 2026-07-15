#!/bin/bash
#SBATCH --job-name=hf_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Evaluate HardFlow on the "avoiding" task and write per-method trajectories.csv
# into FM-PCC/logs/hardflow/avoiding-v0/eval/<exp>/ (via the bridge symlink).
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh
# Override methods:  METHODS="original" ./Slurm_Codes/submit.sh .../eval_hardflow.sh
#   (submit.sh forwards env; or edit the default below.)
#
# Requires: a trained/downloaded checkpoint present at
#   logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth  (run train_hardflow.sh first).
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

# l4casadi-free method set by default. Add oc_flow / gradient_guidance freely.
# Add hardflow / projection / projection_relaxed ONLY after an l4casadi build.
METHODS="${METHODS:-hardflow_new original}"

# --- Fit the linear dynamics model ONCE if absent -----------------------------
# eval.py:517 loads logs/avoiding-v0/dynamics/linear_model.npz for --dynamics_constraint
# methods (e.g. hardflow_new) and SILENTLY proceeds without it (degrades). Guard here.
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-EVAL ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-EVAL ] dynamics model present: $DYN"
fi

# --- Run each method via HardFlow's OWN eval scripts (paper params baked in) ---
for method in $METHODS; do
    script="run_scripts/eval_${method}.sh"
    if [ ! -f "$script" ]; then
        echo "[ HF-EVAL ] SKIP '$method' — $script not found." >&2
        continue
    fi
    echo "=============================================================="
    echo "[ HF-EVAL ] method=$method -> bash $script"
    echo "=============================================================="
    bash "$script"
done

echo "[ HF-EVAL ] done. results under logs/avoiding-v0/eval/ :"
ls -la logs/avoiding-v0/eval/ 2>/dev/null || echo "  (none — check the run log above)"
