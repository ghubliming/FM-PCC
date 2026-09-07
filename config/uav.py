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

# Path discriminator — dimension-based (Fix_4 + Fix_4b):
#
#   Only cond_mode determines the checkpoint path; controller is runtime-only and
#   NEVER appears in the path (all controllers share the same trained weights).
#
#   cond_mode='p_des'    → no suffix   → 'flow_matching_v3_uav/H{h}_D{d}'       (12D, E7 compat)
#   cond_mode='pos_only' → '_9D'       → 'flow_matching_v3_uav/H{h}_D{d}_9D'    (9D, vel dropped)
#
#   Why dimension not cond_mode string? Clearer at a glance; if a new cond_mode
#   still produces 9D it maps to the same checkpoint without renaming.
#
#   Bug fixes (Fix_4):
#     - prefix included so savepath gets 'flow_matching_v3_uav/' parent folder
#       (utils.Parser: savepath = logbase/dataset/exp_name/seed — prefix NOT joined
#        separately; watch() built it into exp_name via ('prefix','') entry)
#     - _ctrl{controller} suffix removed (controller ≠ model weights)
_COND_MODE_DIM = {
    'p_des':    None,   # 12D default — no suffix (E7 backward compat)
    'pos_only': '9D',   # 9D — velocity dropped from obs
}

def _uav_exp_name(args):
    prefix    = getattr(args, 'prefix', '')
    name      = f'H{args.horizon}_D{args.diffusion}'
    cond_mode = getattr(args, 'cond_mode', 'p_des')
    dim_tag   = _COND_MODE_DIM.get(cond_mode, f'cm{cond_mode}')  # fallback for unknown modes
    if dim_tag:
        name += f'_{dim_tag}'
    # Replicate watch()'s join: [prefix, name] → '_'.join → '/_'→'/' cleanup
    parts = [p for p in [prefix, name] if p]
    return '_'.join(parts).replace('/_', '/')

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

        # E8 (optional) — observation layout + tracker. Defaults preserve E7 exactly.
        #   cond_mode='p_des'    → obs=[p_des|p|v] 9D, transition 12D (E7 default).
        #   cond_mode='pos_only' → obs=[p_des|p] 6D, transition 9D (velocity dropped).
        #   controller='pid'        → cascaded PID, v_des=action/dt_fm (E7 default, continuous).
        #   controller='pid_stopgo' → cascaded PID, v_des=0 (U2, strict stop-and-go).
        #   controller='pid_const_v'→ cascaded PID, v_des=unit(action)*v_des_magnitude (U3, constant speed).
        #   controller='mjpc'       → MJX predictive-sampling tracker (E8 U6, cluster-only; pip install "jax[cuda12]" mujoco-mjx).
        # All are path discriminators via _uav_exp_name (non-default → folder suffix).
        'cond_mode': 'pos_only',
        'controller': 'pid_stopgo',

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
        # E8: conditional exp_name (default = watch(args_to_watch) output 'H{h}_D{d}';
        # non-default cond_mode/controller append a suffix — see _uav_exp_name above).
        'exp_name': _uav_exp_name,

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

        # E8 (optional) — MUST match the training block so build_experiment resolves the
        # same savepath. cond_mode sets the obs layout the eval feeds the FM; controller
        # selects the inner-loop tracker and segregates the eval output folder.
        # controller='pid_stopgo'  (U2): v_des=0 at eval only — same checkpoint as pid with same cond_mode.
        # controller='pid_const_v' (U3): v_des=unit(action)*v_des_magnitude — speed auto-derived from dataset
        #   (mean(|action|)×DATASET_HZ ≡ mean(action/dt_fm)); no manual knob needed.
        'cond_mode': 'pos_only',
        'controller': 'pid_stopgo',

        # ── Checkpoint loading ────────────────────────────────────────────────
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v3_uav/',
        # E8: conditional exp_name (same helper as training → train/eval paths agree).
        'exp_name': _uav_exp_name,
        # NOTE: eval_fm_uav.build_experiment loads the model from the TRAIN block's savepath
        # (via exp_name), so this diffusion_loadpath is vestigial for that script; kept for
        # API parity / external tooling. Default path only — the live load is exp_name-driven.
        'diffusion_loadpath': 'f:flow_matching_v3_uav/H{horizon}_D{diffusion}',
        'diffusion_epoch': 'best',

        # ── Eval control ──────────────────────────────────────────────────────
        # flow_steps_v3: ODE Euler steps at inference (eval only — never in train folder name).
        # Without this, FlowMatchingODE falls back to n_timesteps=1000 (extremely slow).
        # aligning-d3il-visual uses 16 (default) or 100; 20 is a sensible UAV default.
        # Appears in the eval output path as 'K{n}' via _uav_eval_tag.
        'flow_steps_v3': 20,
        # mpc_batch_size: MPC candidate-fan size (eval only).
        # Mirrors aligning-d3il-visual.py naming (mpc_batch_size vs train batch_size).
        # Appears in the eval output path as 'mpc{B}' (e.g. mpc4).
        # diffuser variant always uses mpc=1 effectively (random selection, no competition).
        'mpc_batch_size': 4,
        'diffusion_timestep_threshold': _yaml_threshold,   # from uav_projection.yaml

        'behavior_log': True,                # write per-step .log files; False = stats only
        'control_hz': 33,                    # override FM outer-loop rate; default = DATASET_HZ

        # ── MJX predictive-sampling knobs (controller='mjpc' only, U6) ───────
        # mjx_n_samples: parallel candidate trajectories per improve_policy() call.
        # mjx_horizon:   planning window in seconds → converted to MuJoCo steps in MJPCTracker.
        # mjx_n_improve: improve_policy iterations per compute() call. More = smoother,
        #                each adds ~1 ms on GPU. 5 rounds × 16 samples = 80 traj evals.
        # mjx_vel_weight: velocity-penalty weight in MJX cost. Mirrors PID Kd damping —
        #                 without it the planner charges at the setpoint at full thrust.
        'mjx_n_samples':  16,
        'mjx_horizon':    0.3,
        'mjx_n_improve':  5,
        'mjx_vel_weight': 0.1,

        # ── Standard plan fields ──────────────────────────────────────────────
        'device': 'cuda',
        'seed': 0,
        'preprocess_fns': [],
        'returns_condition': False,
    },
}
