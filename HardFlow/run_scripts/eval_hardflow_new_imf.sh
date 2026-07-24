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
# U9: which TRAINING to evaluate. Must match IMF_EXP_NAME used at train time,
# otherwise eval silently loads the wrong checkpoint.
flow_exp_name="${IMF_EXP_NAME:-H16_imf_100k}"
flow_cp="${IMF_CP:-4}"
k_steps="${IMF_K:-2}"

# u_5: env-overridable for the paired n=200 safety run (default 50 keeps every
# earlier K1/K2/K4/K5 invocation byte-identical).
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
# fix_4: IPOPT silenced (was 5). Level 5 emitted ~45 lines PER NLP solve — 31k of
# the 70k lines in the first run's eval log — of which only "Constraint violation"
# (~1e-16, i.e. always healthy) and solve failures carried signal. Failures are
# now reported by our own code instead: a loud "[ eval_imf ] WARNING: NLP solve
# failed" line, plus nlp_solves/nlp_failures columns in trajectories.csv and a
# per-episode `nlp=` field. Raise back to 5 only when debugging the solver itself.
solver_print_level="${SOLVER_PRINT_LEVEL:-0}"

exp_name="H16_imf_hardflow_new_K${k_steps}"
# U9: tag the eval output with its TRAINING SOURCE so results from different
# trainings never collide. Legacy default (H16_imf_100k) keeps the old name.
[ "$flow_exp_name" != "H16_imf_100k" ] && exp_name="${exp_name}_from_${flow_exp_name}"
# u_5: suffix only when n != 50, so the frozen K1/K2/K4/K5 result dirs are never
# overwritten and the n=200 arm lands in its own directory.
[ "$random_repeat" != "50" ] && exp_name="${exp_name}_n${random_repeat}"

# u_5(B): MPC foresight-fan diagnostic. DEFAULT OFF -- set IMF_PLOT_FAN=1 for a
# small diagnostic run. Off => byte-identical behaviour to before (no capture,
# no plotting), so the decisive paired n=200 safety run is unaffected.
fan_flag=""
[ "${IMF_PLOT_FAN:-0}" = "1" ] && fan_flag="--imf_plot_fan"

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
	--guidance_method hardflow_new_imf \
	$fan_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
