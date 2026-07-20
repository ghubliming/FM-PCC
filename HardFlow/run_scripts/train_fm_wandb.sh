#!/bin/bash
# Gen13 U9 — FM (HardFlow) training with W&B + CSV logging.
# Additive sibling of train.sh: SAME training params (H16, 1e6 steps, batch 32,
# lr 2e-4, ema 0.995, save_freq 50000) — only the entry point and logging differ.
start_time=$(date +%s)
export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"; state_dim=4; action_dim=2; horizon=16
n_train_steps="${N_TRAIN_STEPS:-1000001}"
# U9 SAFETY: budget-tagged name, and NEVER "H16_1e6steps" (that is the authors'
# downloaded checkpoint backing the whole replication — must not be touched).
steps_tag="$(( n_train_steps / 1000 ))k"
exp_name="${FM_EXP_NAME:-H${horizon}_fm_${steps_tag}}"

if [ "$exp_name" = "H${horizon}_1e6steps" ]; then
    echo "[ train_fm ] ABORT: refusing to write to the downloaded baseline checkpoint dir." >&2
    exit 1
fi
final_cp=$(( n_train_steps / 50000 ))
if [ -f "./logs/avoiding-v0/flow/${exp_name}/model_ema_${final_cp}.pth" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
    echo "[ train_fm ] ABORT: ./logs/avoiding-v0/flow/${exp_name}/ already holds a finished run." >&2
    echo "              Use FM_EXP_NAME=<new_name> or FORCE_OVERWRITE=1." >&2
    exit 1
fi

wandb_flag=""
[ "${USE_WANDB:-1}" = "1" ] && wandb_flag="--use_wandb"

echo "=== [U9] FM train: env=$env horizon=$horizon steps=$n_train_steps exp=$exp_name ==="

python run/train_fm.py \
	--device cuda:0 \
	--log_folder ./logs \
	--exp_name "$exp_name" \
	--env "$env" \
	--horizon "$horizon" \
	--state_dim "$state_dim" \
	--action_dim "$action_dim" \
	--n_train_steps "$n_train_steps" \
	--save_freq 50000 \
	--batch_size 32 \
	--learning_rate 2e-4 \
	--ema_decay 0.995 \
	--flow_matching_type cfm \
	--wandb_project "${WANDB_PROJECT:-FMPCC-HF-iMF}" \
	$wandb_flag

end_time=$(date +%s); elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
