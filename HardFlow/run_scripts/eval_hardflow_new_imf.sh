#!/bin/bash
# Gen13 — HardFlow (l4casadi-free) with the iMF backbone: the SEAM-swapped
# constrained sampler (sibling of eval_hardflow_new.sh; paper params kept).
# K (solver steps / NLP count) == ode_t_steps; K=2 default (Gen13 headline E3).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2

horizon=16
flow_exp_name="H16_imf_100k"
flow_cp="${IMF_CP:-4}"
k_steps="${IMF_K:-2}"

random_repeat=50
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
solver_print_level=5

exp_name="H16_imf_hardflow_new_K${k_steps}"

echo "=== Gen13 iMF HardFlow(new) on ${env}, horizon=${horizon}, K=${k_steps} ==="

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
	--guidance_method hardflow_new_imf

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
