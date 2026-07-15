start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

flow_list=(
	"cfm"
)

env_list=(
	"avoiding-v0"
)

horizon_list=(
	16
)

for flow_type in "${flow_list[@]}"; do
	flow_prefix=""

	for env in "${env_list[@]}"; do

		state_dim=4
		action_dim=2

		for horizon in "${horizon_list[@]}"; do

			exp_name="${flow_prefix}H${horizon}_1e6steps"

			echo "=== Running: flow_type=$flow_type, env=$env, horizon=$horizon ==="

			python run/train.py \
				--device cuda:0 \
				--log_folder ./logs \
				--exp_name "$exp_name" \
				--env "${env}" \
				--horizon "$horizon" \
				--state_dim "$state_dim" \
				--action_dim "$action_dim" \
				--n_train_steps 1000001 \
				--save_freq 50000 \
				--batch_size 32 \
				--learning_rate 2e-4 \
				--ema_decay 0.995 \
				--flow_matching_type "$flow_type"

		done
	done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

hours=$((elapsed / 3600))
remainder=$((elapsed % 3600))
minutes=$((remainder / 60))
seconds=$((remainder % 60))

printf "Total runtime: %d hours, %d minutes, %d seconds\n" \
	"$hours" "$minutes" "$seconds"
