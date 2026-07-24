#!/bin/bash
# Gen13 u_5 — FM baseline arm for the PAIRED n=200 safety comparison.
#
# WHY THIS FILE EXISTS: `eval_hardflow_new.sh` hardcodes `random_repeat=50` and is
# PRE-EXISTING HardFlow code, protected by Gen13's additive-only rule — it must not
# be edited. This is an additive Gen13-owned sibling: an EXACT copy of that script
# with only three deliberate changes, so the FM arm can be re-run at a different n
# without touching the frozen baseline or its artifacts.
#
#   1. random_repeat  -> env-overridable  (RANDOM_REPEAT, default 200)
#   2. exp_name       -> gains an `_n<N>` suffix, so the frozen n=50 results in
#                        H16_1e6steps_hardflow_new_10steps/ are NEVER overwritten
#   3. solver_print_level -> 0 (was 5). Purely cosmetic: IPOPT's ~45 lines PER SOLVE
#                        would be ~4x worse at n=200. Does NOT affect the solution or
#                        any CSV value. Override with SOLVER_PRINT_LEVEL=5 to debug.
#
# EVERY numerical parameter below is otherwise byte-identical to eval_hardflow_new.sh
# (horizon 16, cp 20, ode_t_steps 10, seed 0, replan 8, constraint novel, all scales)
# so arm B is a faithful re-run of the frozen baseline at larger n.
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2

flow_type="cfm"
horizon=16
flow_cp=20
ode_t_steps=10

random_repeat="${RANDOM_REPEAT:-200}"
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

exp_name="H${horizon}_1e6steps_hardflow_new_${ode_t_steps}steps_n${random_repeat}"

echo "=== [u_5 arm B] FM HardFlow(new) on ${env}, h=${horizon}, ode_t_steps=${ode_t_steps}, n=${random_repeat} ==="
echo "=== exp_name: ${exp_name}  (frozen n=50 baseline is NOT touched) ==="

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
	--no-render \
	--guidance_method hardflow_new

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
