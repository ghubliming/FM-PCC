#!/bin/bash
# Gen13 U12 — Mix-ML eval, RAW / unguided arm (no projection).
# Clean sibling of the frozen run_scripts/eval_original_imf.sh: SAME python call
# and SAME --guidance_method original_imf (code contract, untouched), but it names
# the run after the OBJECTIVE it actually evaluated (mf/af/imf), not "imf", and
# writes into a tidy per-run folder:
#     logs/<env>/eval/<ML_EXP_NAME>/raw_K<k>[_n<n>]/
# The frozen eval_original_imf.sh is left byte-identical (iMF baseline path).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2
horizon=16

# WHICH training to evaluate. ML_EXP_NAME is the Mix-ML run (e.g. H16_ml_mf_100k);
# IMF_EXP_NAME kept as a fallback so the pipeline's existing wiring still works.
flow_exp_name="${ML_EXP_NAME:-${IMF_EXP_NAME:-H16_ml_imf_100k}}"
flow_cp="${ML_CP:-${IMF_CP:-4}}"
k_steps="${ML_K:-${IMF_K:-2}}"
random_repeat="${RANDOM_REPEAT:-50}"

controller="rh"
replan_steps=8
constraint="novel"

# ── tidy nested output name: <run>/raw_K<k>[_n<n>] ────────────────────────────
sub="raw_K${k_steps}"
[ "$random_repeat" != "50" ] && sub="${sub}_n${random_repeat}"
exp_name="${flow_exp_name}/${sub}"

fan_flag=""
[ "${IMF_PLOT_FAN:-0}" = "1" ] && fan_flag="--imf_plot_fan"

echo "=== Gen13 U12 Mix-ML RAW/unguided | run=${flow_exp_name} K=${k_steps} n=${random_repeat} -> eval/${exp_name} ==="

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
