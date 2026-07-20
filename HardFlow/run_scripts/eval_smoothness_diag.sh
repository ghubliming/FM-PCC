#!/bin/bash
# Gen13 fix_7 — MPC planned-trajectory SMOOTHNESS diagnostic (2x2 matrix, small n).
#
# Runs ONE cell of the matrix; the sbatch loops all four. Both backbones go
# through run/eval_imf.py, so the roughness metric and the foresight fan are
# computed identically for FM and iMF (apples-to-apples by construction).
#
#   BACKBONE=imf|fm   GUIDANCE=<unguided|guided>   N=<episodes, default 5>
#
# The 2x2 tests the DISCUSSION hypothesis (Research/DISCUSSION_foresight_fan_
# and_smoothness_paradigms.md): HardFlow's NLP enforces dynamic feasibility as a
# HARD constraint, so it should manufacture smoothness even from a coarse field.
# Expectation: unguided iMF roughest; both guided variants much smoother and
# similar to each other.
start_time=$(date +%s)
export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"; state_dim=4; action_dim=2; horizon=16
N="${N:-5}"
BACKBONE="${BACKBONE:-imf}"
GUIDANCE="${GUIDANCE:-guided}"
PLOT_FAN="${PLOT_FAN:-1}"          # fans ON for this diagnostic (tiny n)

if [ "$BACKBONE" = "fm" ]; then
    # fix_7.3: FM_K env-overridable so FM can be run at iMF's budget (matched-NFE control)
    flow_exp_name="H16_1e6steps"; flow_cp="20"; k_steps="${FM_K:-10}"   # 10 = FM native
    [ "$GUIDANCE" = "guided" ] && gm="hardflow_new" || gm="original"
else
    flow_exp_name="H16_imf_100k"; flow_cp="${IMF_CP:-4}"; k_steps="${IMF_K:-5}"
    [ "$GUIDANCE" = "guided" ] && gm="hardflow_new_imf" || gm="original_imf"
fi

exp_name="diag_smooth_${BACKBONE}_${GUIDANCE}_K${k_steps}_n${N}"
fan_flag=""; [ "$PLOT_FAN" = "1" ] && fan_flag="--imf_plot_fan"

echo "=== [fix_7 smoothness] backbone=$BACKBONE guidance=$gm K=$k_steps n=$N ==="
echo "=== exp_name: $exp_name ==="

args=(
  --device cuda:0 --seed 0
  --random_repeat "$N" --exp_name "$exp_name" --env "$env"
  --state_dim "$state_dim" --action_dim "$action_dim" --horizon "$horizon"
  --backbone "$BACKBONE"
  --flow_exp_name "$flow_exp_name" --flow_cp "$flow_cp"
  --ode_t_steps "$k_steps"
  --constraint "novel" --controller "rh" --replan_steps 8
  --no-render --guidance_method "$gm"
)
# guided runs need the NLP machinery + the same paper cost/value settings
if [ "$GUIDANCE" = "guided" ]; then
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

python run/eval_imf.py "${args[@]}" $fan_flag

end_time=$(date +%s); elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
