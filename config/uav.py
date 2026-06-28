"""UAV state-only Flow-Matching config (Gen11 Epoch 6/7).

Mirrors `config/avoiding-d3il.py` 1:1 (single file, BOTH training + plan/eval blocks,
`watch` exp_name, yaml import at top for projection threshold).

The train/eval scripts set `exp='uav'` → `config='config.uav'` and pick the block
`flow_matching_v3_uav` (train) or `plan_flow_matching_v3_uav` (eval), exactly like
`avoiding-d3il.py` splits `flow_matching_v3_ode_selectable` / `plan_fm_v3_ode_selectable`.

Scene selection is NOT a separate config: the `--scene` CLI flag sets the dataset string
to `uav-<scene>` (e.g. `uav-all`, `uav-empty`), which both selects the data branch in
`datasets/d4rl.py:sequence_dataset` and segregates the output path
(`logs/UAV_FM/uav-<scene>/<exp_name>/<seed>/`).

UAV schema (authoritative): obs=9 [p_des|p|v], action=3 Δp_des, transition=12, H=8.
Dims are derived from the data at runtime (train script:
`transition_dim = observation_dim + action_dim`), so they are not hard-set here.

Config split (mirrors avoiding-d3il.py pattern):
  config/uav.py              — training block + plan/eval block  ← this file
  config/uav_projection.yaml — projection-only (variants, constraints, geometry)
"""

import yaml
from diffuser.utils import watch

# Read projection threshold from YAML (same pattern as avoiding-d3il.py reading
# from config/projection_eval.yaml). The plan block below uses _yaml_threshold so
# that both sources are always in sync.
with open('config/uav_projection.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError(
        "CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/uav_projection.yaml"
    )
_yaml_threshold = _proj_config['diffusion_timestep_threshold']

# ── Per-scene episode buffer sizes (steps at DATASET_HZ=33 Hz) ───────────────
# train_fm_uav.py resolves max_path_length from this dict when scene != 'all',
# avoiding large per-episode padding for short scenes.
# Derivation (from generator.py _build_traj_and_init):
#   empty    : dur = max(4.0, dist/0.4), max dist ≈ 5.1 m → max ≈ 12.8s → 421 steps → 450 buffer
#   corridor : dur ∈ [6, 10] s → 330 steps max → 360 buffer
#   s_curve  : dur ∈ [16, 22] s → 726 steps max → 750 buffer
#   pillars  : dur ∈ [10, 16] s → 528 steps max → 560 buffer
#   all      : pooled, bounded by s_curve → 750
MAX_PATH_LENGTH_PER_SCENE = {
    'empty':    450,
    'corridor': 360,
    's_curve':  750,
    'pillars':  560,
    'all':      750,
}

# ── Exp-name label (folder name) — shared by BOTH training and plan blocks ───
# The same args_to_watch drives both so the plan folder mirrors the training folder.
args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
]

# All UAV-FM outputs live under logs/UAV_FM/ (NOT scattered at the top of logs/).
# savepath = <logbase>/<dataset>/<exp_name>/<seed> = logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/.../<seed>
# NOTE: the dataset string stays 'uav-<scene>' — the data loader keys on it
# (flow_matcher_v3_uav/datasets/d4rl.py: env.startswith('uav')); only the logbase moved.
logbase = 'logs/UAV_FM'

base = {
    # ── Training block ────────────────────────────────────────────────────────
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
        'max_path_length': 750,   # fallback (scene='all', bounded by s_curve); per-scene: see MAX_PATH_LENGTH_PER_SCENE
        'include_returns': False,  # UAV has no reward signal — skip returns machinery
        'returns_scale': 400,      # unused when include_returns=False (kept for API parity)
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
    },

    # ── Plan / eval block ─────────────────────────────────────────────────────
    # Loaded by eval_fm_uav.py via parse_args(experiment='plan_flow_matching_v3_uav').
    # Mirrors the plan_fm_v3_ode_selectable block in config/avoiding-d3il.py.
    # WHAT BELONGS HERE:  eval control scalars (batch_size, thresholds, logging flags,
    #                     U4 grounding knobs) and checkpoint loading pointers.
    # WHAT DOES NOT:      projection variants / constraint geometry → uav_projection.yaml.
    'plan_flow_matching_v3_uav': {
        # ── Model identity (must match the trained checkpoint) ────────────────
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # ── Checkpoint loading ────────────────────────────────────────────────
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v3_uav/',
        'exp_name': watch(args_to_watch),
        'diffusion_loadpath': 'f:flow_matching_v3_uav/H{horizon}_D{diffusion}',
        'diffusion_epoch': 'latest',

        # ── Eval control ──────────────────────────────────────────────────────
        'batch_size': 4,                     # MPC candidate-fan size
        'diffusion_timestep_threshold': _yaml_threshold,   # from uav_projection.yaml
        'write_to_file': True,
        'behavior_log': True,                # write per-step .log files; False = stats only
        'control_hz': 33,                    # override FM outer-loop rate; default = DATASET_HZ

        # ── fix_5 anchor-p (eval-only, no retrain) ───────────────────────────
        # anchor_to_p=True: p_des = p_real + action each step.
        #   Dynamics constraint rebinds to real p (dims 6,7,8) so the projector
        #   geo-calibrates the action from actual drone position, not drifted p_des.
        #   When False (default): free-running Euler (p_des += action). See:
        #   logs_in_develop/.../U4_cond/fix_5_train_eval_cond_p_redesign_patch/
        'anchor_to_p': False,

        # ── Standard plan fields ──────────────────────────────────────────────
        'device': 'cuda',
        'seed': 0,
        'preprocess_fns': [],
        'returns_condition': False,
    },
}
