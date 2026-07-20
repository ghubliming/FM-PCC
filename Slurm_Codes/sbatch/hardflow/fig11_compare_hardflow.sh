#!/bin/bash
#SBATCH --job-name=hf_fig11_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-1-student
# ==============================================================================
# Gen13 u_8 — paper-Fig.11-style comparison figure: iMF vs original FM.
#
# Runs BOTH guided backbones at a small, representative n (default 3) with the
# foresight capture ON, then assembles a two-panel figure showing ONE
# representative planning instance per method — HardFlow Fig. 11's framing:
#   "Since the policy replans in a receding-horizon manner, we show one
#    representative planning instance during execution."
#
# WHY n=3 IS ENOUGH: this is a QUALITATIVE figure, not a statistic. One planning
# instance is what the paper shows; n=3 just gives a couple of episodes to pick a
# clean representative from. (Contrast: the n=200 safety run measured a rare-event
# RATE, which genuinely needed the samples.)
#
# Runtime: ~2-4 min. 1h requested per the 2x-margin rule.
#
# Submit: ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/fig11_compare_hardflow.sh
# Knobs:  N (episodes, default 3), RUN_ID (episode to plot, default 0),
#         PLAN_IDX (replan instant; default = middle of the episode)
# ==============================================================================
source Slurm_Codes/sbatch/hardflow/_hardflow_common.sh

N="${N:-3}"
RUN_ID="${RUN_ID:-0}"

DYN="logs/avoiding-v0/dynamics/linear_model.npz"
[ -f "$DYN" ] || { echo "[ u_8 ] fitting dynamics"; python run/fit_dynamics.py; }

# --- 1. both guided backbones, fan capture ON (writes *_fan.png AND *_fan.npz) --
for bk in imf fm; do
    echo "=============================================================="
    echo "[ u_8 ] capturing planning instances: backbone=$bk  n=$N"
    echo "=============================================================="
    BACKBONE="$bk" GUIDANCE="guided" N="$N" PLOT_FAN=1 \
        bash run_scripts/eval_smoothness_diag.sh
done

# --- 2. assemble the two-panel Fig.11-style comparison (pure post-processing) ---
IMF_DIR="logs/avoiding-v0/eval/diag_smooth_imf_guided_K${IMF_K:-5}_n${N}"
FM_DIR="logs/avoiding-v0/eval/diag_smooth_fm_guided_K10_n${N}"
OUT="logs/avoiding-v0/eval/fig11_imf_vs_fm_run${RUN_ID}.png"

echo "=============================================================="
echo "[ u_8 ] assembling comparison figure -> $OUT"
echo "=============================================================="
python run/make_fig11_comparison.py \
    --imf_dir "$IMF_DIR" \
    --fm_dir "$FM_DIR" \
    --run_id "$RUN_ID" \
    ${PLAN_IDX:+--plan_idx "$PLAN_IDX"} \
    --out "$OUT"

# --- 3. u_8.2: the REAL Fig.11 structure — 2xN ODE-step grid per method ------
GRID_OUT="logs/avoiding-v0/eval/fig11_ode_grid_run${RUN_ID}.png"
echo "=============================================================="
echo "[ u_8.2 ] assembling ODE-step grid (4 rows: iMF x_tau/x1, FM x_tau/x1)"
echo "=============================================================="
python run/make_fig11_ode_grid.py \
    --dir "$IMF_DIR" --dir2 "$FM_DIR" --both \
    --run_id "$RUN_ID" \
    ${PLAN_IDX:+--plan_idx "$PLAN_IDX"} \
    --n_cols "${N_COLS:-6}" \
    --out "$GRID_OUT" || echo "[ u_8.2 ] grid failed (see above)"

echo "[ u_8 ] done."
ls -la "$OUT" "$GRID_OUT" 2>/dev/null || echo "  (figures not produced — check above)"
echo "  per-method fans also available: $IMF_DIR/*_fan.png , $FM_DIR/*_fan.png"
