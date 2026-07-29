#!/bin/bash
# Gen13 U13 — MPC foresight-fan / smoothness diagnostic for Mix-ML (imf|mf|af).
#
# Extends fix_7's eval_smoothness_diag.sh (which only knew backbone imf/fm) to
# the Mix-ML objectives: MF and AF checkpoints are TemporalImfUnet under the
# hood (same as iMF, per U11's objective-agnostic-eval decision), so this just
# points --backbone imf at a different --flow_exp_name and reuses the SAME
# roughness/fan instrumentation in run/eval_imf.py verbatim (plan_roughness,
# plan_roughness_raw, *_fan.png/.npz) — nothing new to validate there.
#
# ONE hfproj (guided) run gives BOTH numbers per replanned horizon:
#   plan_roughness      = roughness of the PROJECTED (post-NLP) plan
#   plan_roughness_raw  = roughness of the RAW warmstart plan BEFORE the NLP
#                         (populated via policy.raw_plan() — only present when
#                         guidance=hfproj; a `raw` (unguided) run has no NLP to
#                         compare against, so plan_roughness_raw stays NaN there).
# => for "is the raw vs projected trajectory smooth", run GUIDANCE=hfproj.
#
#   ML_TYPE=imf|mf|af   GUIDANCE=raw|hfproj (default hfproj)   ML_K=1|2 (default 2)
#   N=<episodes, default 5>   ML_EXP_NAME/ML_CP (override which checkpoint)
#
# Small n is enough: each episode contributes ~6-7 replans (fix_7 precedent),
# so n=5 => ~30-35 planned horizons per cell; roughness is a per-plan mean.
start_time=$(date +%s)
export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"; state_dim=4; action_dim=2; horizon=16
N="${N:-5}"
ml_type="${ML_TYPE:-mf}"
guidance="${GUIDANCE:-hfproj}"
k_steps="${ML_K:-2}"
PLOT_FAN="${PLOT_FAN:-1}"          # fans ON by default -- this IS the fan diagnostic

flow_exp_name="${ML_EXP_NAME:-${IMF_EXP_NAME:-${ml_type}/H16_ml_${ml_type}_100k}}"
flow_cp="${ML_CP:-${IMF_CP:-4}}"

case "$guidance" in
    hfproj) gm="hardflow_new_imf" ;;
    raw)    gm="original_imf" ;;
    *) echo "[ smooth_diag_ml ] ERROR: GUIDANCE must be raw|hfproj, got '$guidance'" >&2; exit 1 ;;
esac

# U12.2-style nesting: family-first via flow_exp_name, then a smooth_* sub-arm
# (kept separate from the decisive raw_K*/hfproj_K* n=200 dirs — this is a
# small-n diagnostic, not a safety-rate measurement).
exp_name="${flow_exp_name}/smooth_${guidance}_K${k_steps}_n${N}"
fan_flag=""; [ "$PLOT_FAN" = "1" ] && fan_flag="--imf_plot_fan"

echo "=== [Gen13 U13 smooth_diag] ml_type=$ml_type guidance=$gm K=$k_steps n=$N run=$flow_exp_name ==="
echo "=== -> eval/${exp_name} ==="

args=(
  --device cuda:0 --seed 0
  --random_repeat "$N" --exp_name "$exp_name" --env "$env"
  --state_dim "$state_dim" --action_dim "$action_dim" --horizon "$horizon"
  --backbone imf
  --flow_exp_name "$flow_exp_name" --flow_cp "$flow_cp"
  --ode_t_steps "$k_steps"
  --constraint "novel" --controller "rh" --replan_steps 8
  --no-render --guidance_method "$gm"
)
if [ "$guidance" = "hfproj" ]; then
  args+=(
    --warmstart_batch 1
    --value_objective "distance" --value_objective_scale 0.1
    --value_constraint_scale 10.0
    --cost "distance" --cost_scale 2500.0 --hardflow_cost_scale 100.0
    --obstacle_margin 0.02 --dynamics_constraint
    --hardflow_activation "all"
    --solver_print_level "${SOLVER_PRINT_LEVEL:-0}"
  )
fi

# Small-n fan diagnostic: renders ARE wanted here, so HF_EVAL_SAVE_PNG is left
# at its default (1) unless the caller overrides it.
python run/eval_imf.py "${args[@]}" $fan_flag

end_time=$(date +%s); elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
