#!/bin/bash
# Gen13 — unguided iMF eval on avoiding (sibling of eval_original.sh).
# K (NFE) == ode_t_steps; K=2 default (paper regime). Override: IMF_K=1 ...
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

constraint="novel"

exp_name="H16_imf_original_K${k_steps}"

echo "=== Gen13 iMF Original on ${env}, horizon=${horizon}, K=${k_steps} ==="

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
	--constraint "$constraint" \
	--controller "$controller" \
	--replan_steps "$replan_steps" \
	--no-render \
	--guidance_method original_imf

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
