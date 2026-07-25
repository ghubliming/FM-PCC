#!/bin/bash
# Evaluate HardFlow (l4casadi-free variant) on the "avoiding" task.
# Same surrogate problem as HardFlow but evaluates the reference flow with
# PyTorch directly instead of through the l4casadi bridge.
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2

flow_type="cfm"
horizon=16
flow_cp=20
ode_t_steps=10

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

# Gen13 U10: continuous late-activation threshold. Default -1.0 = DISABLED (falls
# back to hardflow_activation="all" -> every step; behaviour unchanged). Set
# HF_ACT_THRESHOLD to sweep, e.g. HF_ACT_THRESHOLD=0.5 (last half, terminal always
# solved) or 1.0 (terminal-only NLP). Encoded in exp_name so runs never collide.
hardflow_activation_threshold="${HF_ACT_THRESHOLD:--1.0}"

exp_name="H${horizon}_1e6steps_hardflow_new_${ode_t_steps}steps"
if awk "BEGIN{exit !($hardflow_activation_threshold >= 0)}"; then
    exp_name="${exp_name}_thres${hardflow_activation_threshold}"
fi

echo "=== Running HardFlow (new) on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps}, thres=${hardflow_activation_threshold} ==="

python run/eval.py \
	--device cuda:0 \
	--seed 0 \
	--random_repeat "$random_repeat" \
	--exp_name "$exp_name" \
	--env "$env" \
	--state_dim "$state_dim" \
	--action_dim "$action_dim" \
	--horizon "$horizon" \
	--flow_exp_name "H${horizon}_1e6steps" \
	--flow_cp "$flow_cp" \
	--flow_matching_type "$flow_type" \
	--ode_t_steps "$ode_t_steps" \
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
	--hardflow_activation_threshold "$hardflow_activation_threshold" \
	--no-render \
	--guidance_method hardflow_new

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
