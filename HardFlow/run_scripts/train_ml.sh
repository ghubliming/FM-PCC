#!/bin/bash
# Gen13 U11/U12.2 — train a selectable MLbone (imf|mf|af) on avoiding.
# Additive sibling of train_imf.sh; produces
#   logs/avoiding-v0/flow/<ml_type>/H16_ml_<ml_type>_<steps>k/model_ema_{0..N}.pth
# U12.2: nested one folder per FAMILY first (imf/ mf/ af/), so `ls logs/.../flow/`
# shows the three objectives instead of a flat pile of runs. The `ml_` prefix in
# the run name still guarantees NO collision with the frozen H16_imf_* iMF runs.
# Override ML_EXP_NAME to use any exact path/name (no auto family-prefix applied).
start_time=$(date +%s)

export D4RL_SUPPRESS_IMPORT_ERROR=1

env="avoiding-v0"
state_dim=4
action_dim=2
horizon=16

ml_type="${ML_TYPE:-imf}"
n_train_steps="${N_TRAIN_STEPS:-100000}"
# LR/clip are shared knobs across families (kept named IMF_* for continuity with
# the existing pipeline env-var vocabulary).
lr="${IMF_LR:-2e-4}"
grad_clip="${IMF_GRAD_CLIP:-1.0}"

# U12.2: default exp_name nests under <ml_type>/ so the family is the FIRST
# folder level; the step budget still disambiguates within it. An explicit
# ML_EXP_NAME override is used VERBATIM (no auto family-prefix) — the caller
# owns the full path then.
steps_tag="$(( n_train_steps / 1000 ))k"
run_name="H16_ml_${ml_type}_${steps_tag}"
exp_name="${ML_EXP_NAME:-${ml_type}/${run_name}}"

# refuse to clobber a COMPLETED run (save_config overwrites silently).
final_cp=$(( n_train_steps / 25000 ))
if [ -f "./logs/avoiding-v0/flow/${exp_name}/model_ema_${final_cp}.pth" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
    echo "[ train_ml ] ABORT: ./logs/avoiding-v0/flow/${exp_name}/ already holds a finished run." >&2
    echo "               Use ML_EXP_NAME=<new_name>, a different N_TRAIN_STEPS, or FORCE_OVERWRITE=1." >&2
    exit 1
fi

# W&B on by default (USE_WANDB=0 to disable). Falls back to CSV if init fails.
wandb_flag=""
[ "${USE_WANDB:-1}" = "1" ] && wandb_flag="--use_wandb"

# ── family-specific knob forwarding (each block adjustable, separated) ─────────
extra=()
case "$ml_type" in
    imf)
        extra+=( --imf_data_proportion "${IMF_DATA_PROPORTION:-0.25}"
                 --imf_p_std "${IMF_P_STD:-1.4}" )
        ;;
    mf)
        extra+=( --mf_data_proportion "${MF_DATA_PROPORTION:-0.25}"
                 --mf_p_std "${MF_P_STD:-1.4}" )
        ;;
    af)
        # 🔴 the α anneal MUST span the ACTUAL budget: end_step = n_train_steps.
        extra+=( --af_alpha_end_step "$n_train_steps"
                 --af_alpha_scheduler "${AF_ALPHA_SCHEDULER:-sigmoid}"
                 --af_alpha_init "${AF_ALPHA_INIT:-1.0}"
                 --af_alpha_end "${AF_ALPHA_END:-0.0}"
                 --af_alpha_gamma "${AF_ALPHA_GAMMA:-25.0}"
                 --af_ratio_fm "${AF_RATIO_FM:-0.5}" )
        ;;
    *)
        echo "[ train_ml ] ERROR: unknown ML_TYPE='$ml_type' (expected imf|mf|af)" >&2
        exit 1
        ;;
esac

echo "=== Gen13 U11 Mix-ML train: ml_type=$ml_type env=$env horizon=$horizon steps=$n_train_steps ==="
echo "    exp_name=$exp_name  lr=$lr  grad_clip=$grad_clip"

python run/train_ml.py \
	--device cuda:0 \
	--log_folder ./logs \
	--exp_name "$exp_name" \
	--ml_type "$ml_type" \
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
	--wandb_project "${WANDB_PROJECT:-FMPCC-HF-Mix-ML}" \
	"${extra[@]}" \
	$wandb_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total runtime: %dh %dm %ds\n" \
	$((elapsed / 3600)) $(( (elapsed % 3600) / 60 )) $((elapsed % 60))
