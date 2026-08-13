"""
Gen9 Epoch 2 — Visual Avoiding config (Single Camera, 6-D trajectory).

Mirrors `config/aligning-d3il-visual.py`'s structure for the avoiding task.
Contains 4 top-level entries:
  - `visual_avoiding_dpcc`       — DPCC training
  - `fm_visual_avoiding`         — FM training
  - `plan_visual_avoiding_dpcc`  — DPCC inference
  - `plan_fm_visual_avoiding`    — FM inference

The two plan_* entries carry `constraint_list` of 6 fixed `sphere_outside`
obstacles sourced from `d3il/.../objects/avoiding_objects.py:get_obj_xy_list()`.

Trajectory layout (2-D analogue of aligning's 9-D):
    x[t] = [ dx  dy | des_x  des_y | c_x  c_y ]
             act(2)   des_xy(2)     c_xy(2)
Single camera: bp-cam only (no inhand-cam — avoiding has no grasping).
"""

from diffuser.utils import watch
import os
import yaml

# Read the threshold dynamically from the visual-avoiding eval YAML, abort if not found
with open('config/visual_avoiding_eval.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError(
        "CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/visual_avoiding_eval.yaml"
    )

_yaml_threshold = _proj_config['diffusion_timestep_threshold']

# ─────────────────────────── args_to_watch helpers ───────────────────────────

args_to_watch_dpcc_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),
]

args_to_watch_dpcc_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('max_episode_length', 'steps'),
    ('mpc_batch_size', 'mpc'),
]

args_to_watch_fm_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),
    ('film_mode', 'film'),   # skipped by watch() if key absent → v1 paths unchanged
]

args_to_watch_fm_visual_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('mpc_batch_size', 'mpc'),
    ('film_mode', 'film'),   # skipped by watch() if key absent → v1 paths unchanged
]

logbase = 'logs'

# ── CUSTOM RUN MESSAGE (2026-08-13) — opt-in results-path tag, EVAL ONLY ──────────────────
# Mirror of config/avoiding-d3il.py (see the long note there). Lets a re-run of the SAME config
# at a different budget write to its OWN results folder instead of overwriting the old numbers.
# Empty (the default) => paths are byte-identical to before.
#   FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/<eval>.sh
# 🔴 PLAN BLOCKS ONLY — never let this reach a training exp_name (it would move every
#    checkpoint folder and break every diffusion_loadpath).
def _sanitize_msg(text):
    """Filesystem-safe, stable token: keep [A-Za-z0-9._-], collapse everything else to '-'."""
    raw = str(text if text is not None else '').strip()
    if not raw:
        return ''
    out = ''.join(ch if (ch.isalnum() or ch in '._-') else '-' for ch in raw)
    while '--' in out:
        out = out.replace('--', '-')
    return out.strip('-._')[:40]


custom_msg = _sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
if custom_msg:
    print(f'[ config/avoiding-d3il-visual ] custom_msg="{custom_msg}" -> results dirs end in "_msg{custom_msg}"')


def _msg_suffix(args):
    """'_msg<token>' for a plan block that carries a non-empty custom_msg, else ''."""
    msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
    return f'_msg{msg}' if msg else ''


def watch_plan(args_to_watch_list):
    """watch(), plus the plan block's custom_msg suffix. Use in PLAN blocks ONLY."""
    _fn = watch(args_to_watch_list)
    return lambda args: _fn(args) + _msg_suffix(args)


# ─────────────────────────── The 6 fixed obstacles ───────────────────────────
# Positions sourced verbatim from
# d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py:68-82
# (`get_obj_xy_list()` — same return list across every env reset). z=0 since the
# avoiding plane is 2-D; sphere check is on c_xy slice (indices 4-5 of the 6-D traj).
# Radius 0.04 m = obstacle geom radius 0.025 m + safety margin 0.015 m
# (cylinder size from avoiding_objects.py geom definitions).

_AVOIDING_OBSTACLES = [
    ('sphere_outside', [0.500, -0.10, 0.0], 0.04),
    ('sphere_outside', [0.425,  0.08, 0.0], 0.04),
    ('sphere_outside', [0.575,  0.08, 0.0], 0.04),
    ('sphere_outside', [0.350,  0.26, 0.0], 0.04),
    ('sphere_outside', [0.500,  0.26, 0.0], 0.04),
    ('sphere_outside', [0.650,  0.26, 0.0], 0.04),
]

# ═══════════════════════════════ Config dict ═══════════════════════════════

base = {

    # ─── 1. Visual-DPCC AVOIDING (training) ───
    'visual_avoiding_dpcc': {
        'model':            'diffuser_visual_avoiding.models.visual_unet.VisualUNet',
        'diffusion':        'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'action_dim':       2,           # 2-D plane velocity [dx, dy]
        'obs_dim':          4,           # [des_xy(2), c_xy(2)] — §12 audit Option C
        'if_vision':        True,
        'horizon':          8,
        'n_diffusion_steps': 100,
        'action_weight':    10,
        'loss_type':        'l2',
        'dim':              32,
        'dim_mults':        (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,
        'max_path_length':  200,         # Avoiding episodes max ~106 steps; 200 is a safe ceiling
        'logbase':          logbase,
        'prefix':           'visual_avoiding_dpcc/',
        'exp_name':         watch(args_to_watch_dpcc_train),
        'batch_size':       64,
        'learning_rate':    2e-4,
        'ema_decay':        0.995,
        'n_steps_per_epoch': 1000,
        'n_train_steps':    1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device':           'cuda',
        'seed':             0,
    },

    # ─── 2. FM Visual AVOIDING (training) ───
    # film_mode: 'v1' (default) = additive-bias FiLM
    #            'v2'           = true FiLM per-block γ/β  (U5, opt-in)
    # watch() appends '_filmv1' / '_filmv2' → separate checkpoint folders per mode.
    # To switch: change film_mode here AND in plan_fm_visual_avoiding. That's it.
    'fm_visual_avoiding': {
        'model':            'fm_visual_avoiding.models.visual_unet.VisualUNet',
        'diffusion':        'fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching',
        'action_dim':       2,
        'obs_dim':          4,
        'if_vision':        True,
        'horizon':          8,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3':  1.0,
        'action_weight':    1,
        'loss_type':        'l2',
        'dim':              32,
        'dim_mults':        (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,
        'max_path_length':  200,
        'logbase':          logbase,
        'prefix':           'fm_visual_avoiding/',
        'exp_name':         watch(args_to_watch_fm_visual_train),
        'batch_size':       64,
        'learning_rate':    2e-4,
        'ema_decay':        0.995,
        'n_steps_per_epoch': 1000,
        'n_train_steps':    1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device':           'cuda',
        'seed':             0,
        'film_mode':        'v1',
    },

    # ─── 3. Visual-DPCC AVOIDING (planning) ───
    'plan_visual_avoiding_dpcc': {
        'horizon':          8,
        'n_diffusion_steps': 100,
        'max_episode_length': 200,
        'max_path_length':  200,
        'action_weight':    10,
        'window_size':      1,
        'obs_seq_len':      1,
        'if_vision':        True,
        'mpc_batch_size':   1,
        'train_batch_size': 64,
        'preprocess_fns':   [],
        'device':           'cuda',
        'seed':             0,
        'loadbase':         None,
        'logbase':          logbase,
        'prefix': (
            'f:plans/visual_avoiding_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}/'
        ),
        'exp_name':         watch_plan(args_to_watch_dpcc_plan),
        'custom_msg':       custom_msg,   # '' => path unchanged; else '..._msg<value>'
        'diffusion':        'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        'predict_epsilon':  True,
        'diffusion_timestep_threshold': _yaml_threshold,
        # U3-C1: must be True — False lets VisualGaussianDiffusion run unclamped, compounding
        # x_recon errors across K denoising steps → exploded trajectories at eval.
        # NOTE: this .py value is NEVER read at eval — the frozen diffusion_config.pkl wins.
        # For in-flight checkpoints run: python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
        'clip_denoised':    True,
        'diffusion_loadpath': (
            'f:visual_avoiding_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}'
        ),
        'diffusion_epoch':  'best',
        'verbose':          False,
        'suffix':           '0',
        # 6 fixed obstacles → sphere_outside projector constraints
        'constraint_list':  list(_AVOIDING_OBSTACLES),
    },

    # ─── 4. FM Visual AVOIDING (planning) ───
    'plan_fm_visual_avoiding': {
        'horizon':          8,
        'flow_steps_v3':    100,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3':  'euler',
        'ode_solver_rtol_v3':    None,
        'ode_solver_atol_v3':    None,
        'ode_solver_step_size_v3': None,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3':  1.0,
        'max_episode_length': 200,
        'max_path_length':  200,
        'action_weight':    1,
        'window_size':      1,
        'obs_seq_len':      1,
        'if_vision':        True,
        'mpc_batch_size':   4,
        'train_batch_size': 64,
        'preprocess_fns':   [],
        'device':           'cuda',
        'seed':             0,
        'loadbase':         None,
        'logbase':          logbase,
        'prefix': (
            'f:plans/fm_visual_avoiding/'
            'H{horizon}_K{flow_steps_v3}_M{ode_solver_method_v3}_T{diffusion_timestep_threshold}_D{diffusion}/'
        ),
        'exp_name':         watch_plan(args_to_watch_fm_visual_plan),
        'custom_msg':       custom_msg,   # '' => path unchanged; else '..._msg<value>'
        'diffusion':        'fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching',
        'returns_condition': False,
        'predict_epsilon':  True,
        'diffusion_timestep_threshold': _yaml_threshold,
        'clip_denoised':    False,
        'diffusion_loadpath': (
            'f:fm_visual_avoiding/'
            'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_film{film_mode}'
        ),
        'diffusion_epoch':  'best',
        'verbose':          False,
        'suffix':           '0',
        'constraint_list':  list(_AVOIDING_OBSTACLES),
        'film_mode':        'v1',
    },

}
