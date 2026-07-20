#!/bin/bash
# Gen13 — unguided iMF eval on avoiding (sibling of eval_original.sh).
# K (NFE) == ode_t_steps; K=2 default (paper regime). Override: IMF_K=1 ...
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2

horizon=16
# U9: which TRAINING to evaluate. Must match IMF_EXP_NAME used at train time,
# otherwise eval silently loads the wrong checkpoint.
flow_exp_name="${IMF_EXP_NAME:-H16_imf_100k}"
flow_cp="${IMF_CP:-4}"
k_steps="${IMF_K:-2}"

# u_5: env-overridable (default 50 keeps earlier invocations byte-identical).
random_repeat="${RANDOM_REPEAT:-50}"
controller="rh"
replan_steps=8

constraint="novel"

exp_name="H16_imf_original_K${k_steps}"
# U9: tag the eval output with its TRAINING SOURCE so results from different
# trainings never collide. Legacy default (H16_imf_100k) keeps the old name.
[ "$flow_exp_name" != "H16_imf_100k" ] && exp_name="${exp_name}_from_${flow_exp_name}"
# u_5: suffix only when n != 50, so frozen result dirs are never overwritten.
[ "$random_repeat" != "50" ] && exp_name="${exp_name}_n${random_repeat}"

# u_5(B): MPC foresight-fan diagnostic. DEFAULT OFF -- set IMF_PLOT_FAN=1 for a
# small diagnostic run. Off => byte-identical behaviour to before (no capture,
# no plotting), so the decisive paired n=200 safety run is unaffected.
fan_flag=""
[ "${IMF_PLOT_FAN:-0}" = "1" ] && fan_flag="--imf_plot_fan"

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
	--guidance_method original_imf \
	$fan_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
