#!/bin/bash
# Gen13 — train the iMF backbone on avoiding (additive sibling of train.sh).
# Produces logs/avoiding-v0/flow/H16_imf_100k/model_ema_{0..4}.pth (cp 4 = final).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2
horizon=16


# Gen13 D5/D6 defaults (Gen3v4-informed); override via env vars if sweeping.
n_train_steps="${N_TRAIN_STEPS:-100000}"
data_proportion="${IMF_DATA_PROPORTION:-0.25}"
p_std="${IMF_P_STD:-1.4}"
# U9.2: LR/clip are now knobs. Default 2e-4 kept for reproducibility of the
# existing runs, but ~2e-5 is the like-for-like value (see imf_config.py).
lr="${IMF_LR:-2e-4}"
grad_clip="${IMF_GRAD_CLIP:-1.0}"

# U9 SAFETY: exp_name encodes the STEP BUDGET so different budgets can never
# collide. 100000 -> "H16_imf_100k" (identical to the existing run, backward
# compatible); 300000 -> "H16_imf_300k" (a NEW dir, existing artifacts safe).
steps_tag="$(( ${N_TRAIN_STEPS:-100000} / 1000 ))k"
exp_name="${IMF_EXP_NAME:-H16_imf_${steps_tag}}"

# U9 SAFETY: refuse to clobber a COMPLETED training (save_config overwrites
# silently). The existing H16_imf_100k checkpoint backs every Gen13 result.
final_cp=$(( n_train_steps / 25000 ))
if [ -f "./logs/avoiding-v0/flow/${exp_name}/model_ema_${final_cp}.pth" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
    echo "[ train_imf ] ABORT: ./logs/avoiding-v0/flow/${exp_name}/ already holds a finished run." >&2
    echo "               Use IMF_EXP_NAME=<new_name>, a different N_TRAIN_STEPS, or FORCE_OVERWRITE=1." >&2
    exit 1
fi

# U9: W&B on by default (set USE_WANDB=0 to disable). Falls back to CSV if init fails.
wandb_flag=""
[ "${USE_WANDB:-1}" = "1" ] && wandb_flag="--use_wandb"

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
	--learning_rate "$lr" \
	--grad_clip "$grad_clip" \
	--ema_decay 0.995 \
	--imf_data_proportion "$data_proportion" \
	--imf_p_std "$p_std" \
	--wandb_project "${WANDB_PROJECT:-FMPCC-HF-iMF}" \
	$wandb_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
