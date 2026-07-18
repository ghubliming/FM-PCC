#!/bin/bash
# Gen13 — train the iMF backbone on avoiding (additive sibling of train.sh).
# Produces logs/avoiding-v0/flow/H16_imf_100k/model_ema_{0..4}.pth (cp 4 = final).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2
horizon=16

exp_name="H16_imf_100k"

# Gen13 D5/D6 defaults (Gen3v4-informed); override via env vars if sweeping.
n_train_steps="${N_TRAIN_STEPS:-100000}"
data_proportion="${IMF_DATA_PROPORTION:-0.25}"
p_std="${IMF_P_STD:-1.4}"

echo "=== Gen13 iMF train: env=$env horizon=$horizon steps=$n_train_steps ==="

python run/train_imf.py \
	--device cuda:0 \
	--log_folder ./logs \
	--exp_name "$exp_name" \
	--env "$env" \
	--horizon "$horizon" \
	--state_dim "$state_dim" \
	--action_dim "$action_dim" \
	--n_train_steps "$n_train_steps" \
	--save_freq 25000 \
	--batch_size 32 \
	--learning_rate 2e-4 \
	--ema_decay 0.995 \
	--imf_data_proportion "$data_proportion" \
	--imf_p_std "$p_std"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
