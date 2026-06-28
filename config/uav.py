"""UAV state-only Flow-Matching config (Gen11 Epoch 6).

Mirrors `config/avoiding-d3il.py` 1:1 (single file, experiment blocks, `watch` exp_name).
The train/eval scripts set `exp='uav'` → `config='config.uav'` and pick the block
`flow_matching_v3_uav`, exactly like the source picks `flow_matching_v3_ode_selectable`
from `config.avoiding-d3il`.

Scene selection is NOT a separate config: the `--scene` CLI flag sets the dataset string
to `uav-<scene>` (e.g. `uav-all`, `uav-empty`), which both selects the data branch in
`datasets/d4rl.py:sequence_dataset` and segregates the output path
(`logs/UAV_FM/uav-<scene>/<exp_name>/<seed>/`).

UAV schema (authoritative): obs=9 [p_des|p|v], action=3 Δp_des, transition=12, H=8.
Dims are derived from the data at runtime (train script:
`transition_dim = observation_dim + action_dim`), so they are not hard-set here.
"""

from diffuser.utils import watch

# minimal exp-name label (folder name); mirrors the source's args_to_watch pattern
args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    # U4: conditioning frame → folder fragment 'condp_des' / 'condreal_p' so the two
    # modes' checkpoints save in PARALLEL dirs and never collide. Read by BOTH train
    # (dataset build) and eval (rollout obs/integration + which checkpoint to load).
    ('cond_mode', 'cond'),
]

# All UAV-FM outputs live under logs/UAV_FM/ (NOT scattered at the top of logs/).
# savepath = <logbase>/<dataset>/<exp_name>/<seed> = logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/.../<seed>
# NOTE: the dataset string stays 'uav-<scene>' — the data loader keys on it
# (flow_matcher_v3_uav/datasets/d4rl.py: env.startswith('uav')); only the logbase moved.
logbase = 'logs/UAV_FM'

base = {
    'flow_matching_v3_uav': {
        # UAV fork of `flow_matching_v3_ode_selectable` — same FM-ODE engine + backbone.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,

        # U4 — conditioning frame (Gen11 E6/E7; see logs_in_develop/.../U4_cond/PLAN_U4_cond_real_p.md).
        #   'p_des'  (default) → obs=[p_des|p|v] (9D), action=Δp_des, transition=12 — current behavior.
        #   'real_p' (opt-in)  → obs=[p|v] (6D),       action=Δp,     transition=9  — plan in real
        #                        position so the command can't run away from the lagging drone.
        # Read by BOTH train (dataset) and eval (rollout); baked into the folder path via
        # args_to_watch ('cond_mode','cond'). 'real_p' requires a fresh retrain (different dims).
        # ⚠ Adding it to the path renames future 'p_des' dirs to '..._condp_des'; a pre-U4
        #   checkpoint must have its folder renamed to add '_condp_des' (no retrain) to be found.
        'cond_mode': 'p_des',

        # v3 SafeFlow-style time sampling parameters (unchanged from source).
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # dataset — generic SequenceDataset; the UAV branch lives in datasets/d4rl.py.
        'loader': 'datasets.SequenceDataset',
        # SafeLimitsNormalizer, NOT LimitsNormalizer: some scenes (e.g. pillars) have a
        # constant feature column (zero range) → plain LimitsNormalizer does (x-min)/(max-min)
        # = 0/0 = NaN, which poisons the whole net (all losses NaN from epoch 0). The Safe
        # variant widens only the constant dim and is identical to Limits when no dim is constant.
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 750,        # s_curve up to 22s*33Hz=726 steps; 750 avoids truncating its tail
        'include_returns': False,      # UAV has no reward signal — skip returns machinery
        'returns_scale': 400,          # unused when include_returns=False (kept for API parity)
        'discount': 0.99,
        'dataset_root': 'data/uav_fm/v1',   # documentation; loader honours UAV_FM_DATA_ROOT

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_uav/',
        'exp_name': watch(args_to_watch),

        # training (unchanged from source)
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
        # NOTE: Epoch-7 PCC/MPC eval config (projection_variants, constraint_types, dt,
        # batch_size, placeholders) lives in config/uav_eval.yaml — loaded by eval_fm_uav.py,
        # mirroring config/visual_aligning_eval.yaml. NOT here.
    },
}
