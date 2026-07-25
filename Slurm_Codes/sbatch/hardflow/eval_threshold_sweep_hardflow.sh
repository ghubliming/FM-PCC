#!/bin/bash
#SBATCH --job-name=hf_u10_thres
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 U10 — late-activation THRESHOLD sweep on the ORIGINAL HardFlow (FM backbone).
#
# Runs `hardflow_new` with the original TemporalUnet FM checkpoint (H16_1e6steps),
# sweeping the per-step-NLP activation threshold to compare threshold ON vs the
# full-step (every-step) baseline — a clean HF-vs-HF ablation on HardFlow's own env
# (avoiding-v0, H16). See logs_in_develop/Gen13/U_10/PLAN_Gen13_U10_*.md.
#
#   threshold 0.0 -> every step (full-step baseline)
#   threshold 0.5 -> last half (terminal always solved)
#   threshold 1.0 -> terminal-only NLP (~ post-hoc projection)
#
# The terminal step is ALWAYS solved (safety guarantee, paper Prop.). Each threshold
# writes its own eval dir (exp_name ..._thres<t>), so nothing overwrites.
#
# Submit:  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_threshold_sweep_hardflow.sh
# Override grid:  HF_THRES_GRID="0.0 0.5 0.75 1.0" ./Slurm_Codes/submit.sh .../eval_threshold_sweep_hardflow.sh
#
# Requires the FM checkpoint at logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth
# (run Slurm_Codes/sbatch/hardflow/train_hardflow.sh first if absent).
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

# --- Fit the linear dynamics model ONCE if absent (hardflow_new uses it) -------
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-U10 ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-U10 ] dynamics model present: $DYN"
fi

# --- Threshold sweep on original HF (hardflow_new) -----------------------------
THRES_GRID="${HF_THRES_GRID:-0.0 0.5 1.0}"
echo "[ HF-U10 ] threshold grid: $THRES_GRID  (FM backbone, hardflow_new)"
for thr in $THRES_GRID; do
    echo "=============================================================="
    echo "[ HF-U10 ] activation_threshold = $thr   ($(date))"
    echo "=============================================================="
    HF_ACT_THRESHOLD="$thr" bash run_scripts/eval_hardflow_new.sh
done

echo "[ HF-U10 ] done. results under logs/avoiding-v0/eval/ :"
ls -la logs/avoiding-v0/eval/ 2>/dev/null | grep thres || echo "  (check the run log above)"
