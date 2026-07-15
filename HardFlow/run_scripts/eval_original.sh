#!/bin/bash
# Evaluate the Original (no-guidance) baseline on the "avoiding" task.
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

constraint="novel"

exp_name="H${horizon}_1e6steps_original_${ode_t_steps}steps"

echo "=== Running Original on ${env}, horizon=${horizon}, ode_t_steps=${ode_t_steps} ==="

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
	--constraint "$constraint" \
	--controller "$controller" \
	--replan_steps "$replan_steps" \
	--no-render \
	--guidance_method original

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
