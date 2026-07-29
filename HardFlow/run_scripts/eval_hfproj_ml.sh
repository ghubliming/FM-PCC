#!/bin/bash
# Gen13 U12 — Mix-ML eval, HARDFLOW-PROJECTED arm (the real constrained method).
# Clean sibling of the frozen run_scripts/eval_hardflow_new_imf.sh: SAME python
# call and SAME --guidance_method hardflow_new_imf (code contract, untouched), but
# named after the OBJECTIVE (mf/af/imf), not "imf", into a tidy per-run folder:
#     logs/<env>/eval/<ML_EXP_NAME>/hfproj_K<k>[_n<n>]/
# U12.2: ML_EXP_NAME is normally "<ml_type>/<run>" (e.g. "af/H16_ml_af_100k"), so
# this nests family-first for free: eval/af/H16_ml_af_100k/hfproj_K2_n200/.
# The frozen eval_hardflow_new_imf.sh is left byte-identical (iMF baseline path).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2
horizon=16

flow_exp_name="${ML_EXP_NAME:-${IMF_EXP_NAME:-imf/H16_ml_imf_100k}}"
flow_cp="${ML_CP:-${IMF_CP:-4}}"
k_steps="${ML_K:-${IMF_K:-2}}"
random_repeat="${RANDOM_REPEAT:-50}"

controller="rh"
replan_steps=8

warmstart_batch=1
value_objective="distance"
value_objective_scale=0.1
value_constraint_scale=10.0

constraint="novel"
obstacle_margin=0.02
cost="distance"
cost_scale=2500.0
hardflow_cost_scale=100.0
hardflow_activation="all"
solver_print_level="${SOLVER_PRINT_LEVEL:-0}"

# ── tidy nested output name: <run>/hfproj_K<k>[_n<n>] ─────────────────────────
sub="hfproj_K${k_steps}"
[ "$random_repeat" != "50" ] && sub="${sub}_n${random_repeat}"
exp_name="${flow_exp_name}/${sub}"

fan_flag=""
[ "${IMF_PLOT_FAN:-0}" = "1" ] && fan_flag="--imf_plot_fan"

echo "=== Gen13 U12 Mix-ML HARDFLOW-proj | run=${flow_exp_name} K=${k_steps} n=${random_repeat} -> eval/${exp_name} ==="

python run/eval_imf.py \
	--device cuda:0 \
	--seed 0 \
	--random_repeat "$random_repeat" \
	--exp_name "$exp_name" \
	--env "$env" \
	--state_dim "$state_dim" \
	--action_dim "$action_dim" \
	--horizon "$horizon" \
	--flow_exp_name "$flow_exp_name" \
	--flow_cp "$flow_cp" \
	--ode_t_steps "$k_steps" \
	--warmstart_batch "$warmstart_batch" \
	--value_objective "$value_objective" \
	--value_objective_scale "$value_objective_scale" \
	--value_constraint_scale "$value_constraint_scale" \
	--solver_print_level "$solver_print_level" \
	--constraint "$constraint" \
	--cost "$cost" \
	--cost_scale "$cost_scale" \
	--hardflow_cost_scale "$hardflow_cost_scale" \
	--obstacle_margin "$obstacle_margin" \
	--dynamics_constraint \
	--controller "$controller" \
	--replan_steps "$replan_steps" \
	--hardflow_activation "$hardflow_activation" \
	--no-render \
	--guidance_method hardflow_new_imf \
	$fan_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
