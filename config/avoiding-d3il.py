from diffuser.utils import watch
import yaml
import os

# 🔴 FIX_9_CFG_PROVENANCE (2026-08-07) — the results-folder name MUST describe the run that
# produced it. This module is imported lazily by utils.Parser.read_config
# (`importlib.import_module`) during Parser().parse_args(), i.e. AFTER the eval script has
# resolved its own projection yaml and the HFFM_*/DPCC_THRESHOLD overrides. Those scripts
# publish what they resolved into the environment; the tokens below are built from it:
#   FMPCC_PROJ_CFG        -> the projection yaml the eval ACTUALLY loaded (also snapshotted)
#   FMPCC_DPCC_THRESHOLD  -> resolved DPCC diffusion_timestep_threshold   ('T' token)
#   HFFM_ACT_THRESHOLD    -> resolved HardFlow NLP activation threshold   ('A' token)
#   HFFM_BATCH            -> resolved HardFlow candidate-fan size         ('B' token)
# With none of them set this falls back to exactly the old behaviour, so train jobs and every
# generation still on config/projection_eval.yaml are unaffected.
# Why it exists: job 24334 was written to a path saying `T1` — read from a file that eval never
# opens — while the Projector was gated at the meanflow yaml's 0.5.
# Full writeup: logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/CHANGELOG_Fix_9_config_provenance.md
_PROJ_CFG = os.environ.get('FMPCC_PROJ_CFG', 'config/projection_eval.yaml')


def _num(value):
    """Keep folder tokens stable: 1.0 -> 1 (so 'T1', not 'T1.0'), 0.5 -> 0.5."""
    f = float(value)
    return int(f) if f == int(f) else f


# Read the threshold dynamically from the YAML config, abort if not found
with open(_PROJ_CFG, 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError(f"CRITICAL: 'diffusion_timestep_threshold' MUST be defined in {_PROJ_CFG}")

# 🔴 FIX_9_CFG_PROVENANCE — the eval's RESOLVED value wins over the file: it is the number the
# Projector was actually handed. Absent (train jobs, other generations) => the file, unchanged.
_env_threshold = os.environ.get('FMPCC_DPCC_THRESHOLD')
_yaml_threshold = (_num(_env_threshold) if _env_threshold is not None
                   else _proj_config['diffusion_timestep_threshold'])

# 🔴 FIX_9_CFG_PROVENANCE — HardFlow (arm C) knobs, consumed by args_to_watch_fmv3_hf_plan.
# The eval publishes these ALREADY RESOLVED (the 'all'/'late' aliases mapped and the yaml
# fallbacks applied), so resolve_activation_threshold() in hardflow_projection.py stays the one
# and only resolver and this module never has to reimplement it. The defaults below apply only
# when no eval published them — in which case no HardFlow arm is running anyway.
_hf_act_threshold = _num(os.environ.get('HFFM_ACT_THRESHOLD', 1.0))
_hf_batch_size = int(os.environ.get('HFFM_BATCH', 1))

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion', 'D'),
]

args_to_watch_dpcc_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
]

args_to_watch_dpcc_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
]

args_to_watch_v3 = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('diffusion', 'D'),
]

args_to_watch_fmv3_ode_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
]

args_to_watch_fmv3_ode_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
]

# 🔴 FIX_9_CFG_PROVENANCE — HardFlow-arm plan blocks ONLY (Gen3v6 MeanFlow, Gen3v7 AlphaFlow).
# Identical to args_to_watch_fmv3_ode_plan plus the two arm-C knobs that change results but used
# to be invisible in the path:  A = NLP activation threshold,  B = candidate-fan (mpc) size.
# Before this, `HFFM_ACT_THRESHOLD=0.5` and `=1.0` wrote to the SAME directory and silently
# overwrote each other — Gen12 worked around it by hand-tagging folders
# (logs_in_develop/Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md).
# Deliberately NOT added to the shared ode_plan list: plan_fm_v3_ode_selectable / _drifting /
# _imeanflow have no HardFlow arm and must keep their existing result paths.
args_to_watch_fmv3_hf_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('hf_act_threshold', 'A'),
    ('hf_batch_size', 'B'),
    ('diffusion', 'D'),
]

args_to_watch_fmv3_imf_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('imf_objective', 'obj'),   # encodes training objective in folder name (fm_equivalent vs meanflow_jvp)
    ('imf_backbone', 'bb'),     # U6: encodes backbone (unet vs dit) so checkpoints never collide
    ('t_schedule', 'ts'),       # U7: time-schedule selector (logit_normal | beta | uniform)
]

# ── Gen3v6 (MeanFlow baseline) ────────────────────────────────────────────────────────
# ⭐ `dp` (meanflow_data_proportion) is in the folder name ON PURPOSE. POST_U10_II §1.1
# documents a live overwrite hazard: four knobs changed between two Gen3v4 runs and NONE
# was in args_to_watch, so both runs wrote to a byte-identical folder and silently
# clobbered each other. Any knob you intend to sweep MUST appear here.
# 🔴 plan_fm_v3_meanflow's `diffusion_loadpath` must reproduce this list token-for-token.
args_to_watch_fmv3_mf_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
    ('mf_objective', 'obj'),
    ('imf_backbone', 'bb'),               # key name kept: the backbone classes are inherited
    ('t_schedule', 'ts'),
    ('meanflow_data_proportion', 'dp'),   # first-class ablation axis in this generation
]

# ── Gen3v7 (α-Flow) ───────────────────────────────────────────────────────────────────
# ⭐ EVERY α knob is in the folder name ON PURPOSE. POST_U10_II §1.1 documents a live
# overwrite hazard where four un-watched knobs let two different runs write to a
# byte-identical directory and silently clobber each other. α-Flow has more sweepable
# knobs than any previous generation, so this matters more here than anywhere.
# 🔴 plan_fm_v3_alphaflow's `diffusion_loadpath` must reproduce this list token-for-token.
args_to_watch_fmv3_af_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
    ('imf_backbone', 'bb'),          # key name kept: the backbone classes are inherited
    ('t_schedule', 'ts'),
    ('af_alpha_init', 'ai'),         # 1.0  — α at step 0 (1.0 ⇒ starts as pure FM)
    ('af_alpha_end', 'ae'),          # 0.0  — α at the end (0.0 ⇒ ends as MeanFlow)
    ('af_alpha_gamma', 'ag'),        # 25.0 — sigmoid sharpness
    ('af_ratio_fm', 'rf'),           # 0.5  — fraction of the batch forced to h=0 (FM anchors)
]

logbase = 'logs'

# ── CUSTOM RUN MESSAGE (2026-08-13) — opt-in results-path tag, EVAL ONLY ──────────────────
# Lets a re-run of the SAME config at a different budget (e.g. n_trials 2 -> 20) write to its
# OWN results folder instead of overwriting the old numbers. Empty (the default) => every path
# is byte-identical to before, so no existing folder is renamed, moved or overwritten.
#
# Three ways to set it, in order of convenience:
#   1. env, at submit time — no git edit (submit.sh passes --export=ALL):
#        FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/<eval>.sh
#   2. edit the `custom_msg` default below — applies to every arm at once, git-tracked.
#   3. a literal inside ONE plan block — `'custom_msg': 'ode_only_rerun',` — to tag a single arm.
#
# 🔴 PLAN BLOCKS ONLY. This must never reach a TRAINING exp_name: that would move every
#    checkpoint folder and break every diffusion_loadpath. Two traps make this easy to get
#    wrong, which is why the token is applied by a WRAPPER (watch_plan) instead of by editing
#    the args_to_watch_* lists:
#      · args_to_watch_v3  is shared by the train block 'flow_matching_v3' AND the plan blocks
#        'plan_fm_v3' / 'plan_fm_v3_hardflow'.
#      · args_to_watch     is shared by 'flow_matching' / 'flow_matching_unet_v2' /
#        'flow_matching_v2' and their plan_* counterparts.
#    Second line of defence: watch_plan reads args.custom_msg, and no train block defines that
#    key, so a misapplied wrapper still renders the empty token.
#
# Why an eval ARG and not a projection-yaml key: the message describes THIS eval run, not the
# constraint projection, so it belongs next to diffusion_epoch / suffix. It also avoids a real
# footgun — scripts/eval.py and eval_flow_matching_v3_ode_selectable.py never publish
# FMPCC_PROJ_CFG, so a yaml-sourced message would have been read from the right file only by
# coincidence. As a config key it is also captured in args.json + the config snapshot, so the
# folder tag is provable from the run's own artifacts.
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
    print(f'[ config/avoiding-d3il ] custom_msg="{custom_msg}" -> results dirs end in "_msg{custom_msg}"')


def _msg_suffix(args):
    """'_msg<token>' for a plan block that carries a non-empty custom_msg, else ''."""
    msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
    return f'_msg{msg}' if msg else ''


def watch_plan(args_to_watch_list):
    """watch(), plus the plan block's custom_msg suffix. Use in PLAN blocks ONLY."""
    _fn = watch(args_to_watch_list)
    return lambda args: _fn(args) + _msg_suffix(args)


base = {
    'diffusion': {
        ## model
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,            
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,        

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,      # longest: 106
        'include_returns': True,
        'returns_scale': 400,   # Determined using rewards from the dataset
        'discount': 0.99,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/',
        'exp_name': watch(args_to_watch_dpcc_train),

        ## training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,            # Overwritten
    },

    'flow_matching': {
        # FM version: same as 'diffusion' but uses FM implementation
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.diffusion.GaussianDiffusion',  # Here is full long path, it distinguishes from the diffusion model, name in folder is longer
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching/',
        'exp_name': watch(args_to_watch),

        # training
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

    'flow_matching_unet_v2': {
        # FM_Unet_v2: uses Flow_matcher_U_Net_v2 backbone
        # TODO: Update model parameters here when U-Net structure is modified
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_unet_v2/',
        'exp_name': watch(args_to_watch),

        # training
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

    'flow_matching_v2': {
        # Flow matcher v2 copied from flow_matching_unet_v2 with SafeFlowMPC-style time sampling
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1, # DPCC is 10
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # v2 SafeFlowMPC-style time sampling parameters (exactly two)
        'time_beta_alpha_v2': 1.5,
        'time_beta_beta_v2': 1.0,

        # v2 ODE/VF decoupling parameters
        'vf_time_bins_v2': 20,
        'ode_inference_steps_v2': 10, # DPCC is 20

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v2/',
        'exp_name': watch(args_to_watch),

        # training
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

    'flow_matching_v3': {
        # Flow matcher v3: SafeFlow-style continuous-time query semantics.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        # 'n_diffusion_steps': 20, # this old parameter is not used in v3
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # v3 rollout step control.
        'flow_steps_v3': 10,
        # Compatibility alias for existing code paths/tools.
        'ode_inference_steps_v3': 10,

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3/',
        'exp_name': watch(args_to_watch_v3),

        # training
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

    'flow_matching_v3_ode_selectable': {
        # Copied-folder FM-v3 variant with config-selectable ODE backend/method.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        # 'n_diffusion_steps': 20, # this old parameter is not used in v3
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        # 'dynamic_loss': False, # DEAD code (legacy DDPM relic, unused in FMv3)
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        # 'test_ret': 0.9, # DEAD code (inference-only parameter)

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # v3 rollout step control.
        # 'flow_steps_v3': 10, # DEAD code (inference-only parameter)
        # 'ode_inference_steps_v3': 10, # Dead code

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_ode_selectable/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),

        # training
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

    'flow_matching_v3_drifting': {
        # Drift-augmented Flow Matcher v3: combines FM ODE with drift loss guidance.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingDrifting',
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

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # FM-D Drift Augmentation Parameters (Locked 3 params)
        'use_drift_augmentation': True,            # bool: enable FM-D mode
        'drift_loss_weight': 0.1,                  # float: lambda in drift field equation
        'drift_loss_type': 'embedding_nn',          # str: "embedding_nn" | "adversarial" | "mmd"

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_drifting/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),

        # training
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

    'flow_matching_v3_imeanflow': {
        # iMeanFlow: Improved Mean Flows for trajectory generation
        # Dual-velocity: u (mean field) + v (instantaneous deviation)
        # Reuses official iMF repo logic: github.com/Lyy-iiis/imeanflow
        
        ## model & engine (REAL iMF from official repo)
        'model': 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine',
        'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
        'horizon': 8,
        
        ## iMF architecture (matches official repo)
        # 🔴 FIX_8_UNET_WIDTH (2026-08-05) — THIS KEY IS THE UNET CHANNEL WIDTH.
        # Its only consumer anywhere is `dim=freq_dim` in models/*_trajectory_model.py,
        # and Flow_matcher_U_Net_v2 uses that one argument for BOTH the channel width
        # (:106) and the time-embed width (:110). At 256 the backbone was 253.0 M params
        # (channels 256/512/1024/2048) against the DPCC/FMv3ODE baseline's 3.97 M at 32
        # — a 63.8x capacity error on 96 demonstrations, which silently confounded every
        # imf_backbone='unet' run. DiT / SiT / mf_dit ignore this key (they size from
        # dit_hidden_size). Full audit: logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
        'freq_dim': 32,
        'depth': 8,
        'num_heads': 4,
        'mlp_dim': 256,
        'time_dim': 256,
        'dropout_rate': 0.1,
        
        ## stable iMF training (FMv3ODE-style main loss + small aux residual)
        'u_loss_weight': 1.0,               # Main flow velocity weight
        'v_loss_weight': 0.1,               # Auxiliary residual weight
        'loss_schedule': 'balanced',        # Keep training stable from step 1
        'warmup_epochs': 0,
        'transition_epochs': 0,
        'loss_type': 'l2',
        'predict_epsilon': True,

        ## U4/U10 — training objective selector (see logs_in_develop/Gen3v4_imf/U4, U10)
        # 'fm_equivalent': legacy finite-diff target = FM baseline arm.
        # 'meanflow_jvp' : ORIGINAL MeanFlow (analytic-v JVP tangent, unguided) — NOT faithful iMF.
        # 'imf_official' : U10 FAITHFUL improved-MeanFlow — predicted-v_c tangent + guided v_g +
        #                  cond_drop + 2×logit-normal + official loss. THE "one last shot".
        #                  See U10/PLAN_faithful_imf_replication.md. Requires dit backbone.
        'imf_objective': 'imf_official',     # was: 'meanflow_jvp' (kept as A/B arm; auto-separates by prefix)
        'meanflow_r_equals_t_frac': 0.25,    # legacy meanflow_jvp only (imf_official uses meanflow_data_proportion)
        'meanflow_adaptive_p': 0.5,          # legacy meanflow_jvp only (imf_official hard-codes official p=1.0)
        'meanflow_adaptive_c': 1e-3,         # legacy meanflow_jvp only (imf_official hard-codes official eps=0.01)
        'meanflow_aux_weight': 0.05,         # legacy meanflow_jvp only (imf_official uses official loss_u+loss_v)
        ## U10 imf_official knobs (faithful iMF)
        'meanflow_cfg_smax': 7.0,            # TRAIN CFG scale ceiling s_max (official default 7.0) — decoupled from eval ω
        'meanflow_data_proportion': 0.5,     # 50% FM anchors (official data_proportion)
        'meanflow_class_dropout_prob': 0.1,  # cond_drop prob — trains the null token (official 0.1)

        ## U5 Phase 1 — real-iMF on UNet (DEFAULT = ON). See Gen3v4_imf/U5.
        ## Safe-core fallback if a run misbehaves: set interval_cfg=False + meanflow_cfg_omega=0.0.
        'dual_head': True,           # was: False  — v-head shares the backbone (official u/v split)
        'interval_cfg': True,        # was: False  — condition backbone on (omega, t_min, t_max)
        ## Fix2 (U6) — CFG is now per-sample randomized at train time (official iMF algorithm),
        ## not a fixed constant. meanflow_cfg_omega is now the sampling ceiling s_max: each sample
        ## draws ω~power-law(0,s_max] and (t_min,t_max)~U(0,0.5)xU(0.5,1) independently every step
        ## (FM-anchor samples get the full [0,1] interval, i.e. no CFG restriction). See
        ## Gen3v4_imf/U6/Fix2_CFG&EMA/CHANGELOG.md.
        'meanflow_cfg_omega': 4.0,   # was: 0.0    — train-time sampling ceiling s_max for ω
        'meanflow_cfg_t_min': 0.4,   # unused when interval_cfg sampling is active (kept for fallback)
        'meanflow_cfg_t_max': 0.6,   # unused when interval_cfg sampling is active (kept for fallback)
        'meanflow_cfg_beta': 1.0,    # power-law shape for ω sampling; 1.0 = official iMF default (log-uniform)

        ## U6 — backbone selector. 'unet' (default) = U5 behaviour, byte-for-byte.
        ## 'dit' = faithful official-iMF transformer (IMFDiTTrajectory). Folder name carries
        ## _bb{imf_backbone}, so unet/dit checkpoints live in separate dirs (no collision).
        ## NOTE: plan block's imf_backbone + dit_* MUST match (state_dict depends on them).
        'imf_backbone': 'dit',       # 'unet' | 'dit' — U10 imf_official REQUIRES dit (unet no-ops cond_drop)
        'dit_depth': 8,              # total transformer blocks (DiT-only)
        'dit_hidden_size': 256,      # token width (DiT-only) — keep small for H=8
        'dit_num_heads': 4,          # attention heads (DiT-only)
        'dit_aux_head_depth': 2,     # private blocks per u/v head (DiT-only)
        'dit_patch_size': 1,         # trajectory steps per token; must divide horizon
        'dit_condition_on_t': False, # official recipe conditions only on h=t−r

        ## dataset (inherited from FMv3ODE)
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,
        'use_padding': True,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        
        ## training (from FMv3ODE baseline)
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'gradient_clip': 1.0,
        'ema_decay': 0.995,
        'action_weight': 10,
        'loss_discount': 1.0,              # BUG-02 fix: explicit uniform trajectory weighting
        'gradient_accumulate_every': 2,    # BUG-03 fix: match FMv3ODE effective batch size
        ## U9: was missing here — validation ran only via the train script's silent
        ## getattr(..., 0.9) fallback, so the split never appeared in the W&B run config.
        'train_test_split': 0.9,
        
        ## ODE inference — NFE is set in the plan block, not here
        # 'ode_inference_steps_v3': 10,  # dead in training; set flow_steps_v3 in plan block
        ## U7: time-schedule. 'logit_normal' = canonical iMF default (reference imf.py, DEFAULT).
        ## 'beta' = legacy 1-Beta(α,β); set α=β=1 for uniform. NOTE: plan block must match.
        't_schedule': 'logit_normal',     # U7 DEFAULT — was: 'beta' (implicit, pre-U7)
        'p_mean': -0.4,                   # logit-normal P_mean (sigmoid median ≈ 0.40)
        'p_std': 1.0,                     # logit-normal P_std
        ## Beta params kept for backward-compat / ablation ('beta' schedule only — ignored otherwise).
        'time_beta_alpha_v3': 1.0,        # was: 1.5
        'time_beta_beta_v3': 1.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_imeanflow/',
        'exp_name': watch(args_to_watch_fmv3_imf_train),
    },

    'flow_matching_v3_meanflow': {
        # ── Gen3v6: faithful MeanFlow (arXiv 2505.13447) baseline ──────────────────────
        # Copy-modify sibling of 'flow_matching_v3_imeanflow'. The ONLY scientific
        # difference is the JVP z-tangent: ANALYTIC v = x1 − x0 here vs iMF's PREDICTED
        # v_c there. Everything architectural is held identical so the A/B is controlled.
        # See logs_in_develop/Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md.

        ## model & engine
        'model': 'flow_matcher_v3_meanflow.models.MeanFlowEngine',
        'diffusion': 'flow_matcher_v3_meanflow.models.MeanFlowODE',
        'horizon': 8,

        ## architecture sizing (UNet arm; DiT sizing is the dit_* block below)
        # 🔴 FIX_8_UNET_WIDTH (2026-08-05) — THIS KEY IS THE UNET CHANNEL WIDTH.
        # Its only consumer anywhere is `dim=freq_dim` in models/*_trajectory_model.py,
        # and Flow_matcher_U_Net_v2 uses that one argument for BOTH the channel width
        # (:106) and the time-embed width (:110). At 256 the backbone was 253.0 M params
        # (channels 256/512/1024/2048) against the DPCC/FMv3ODE baseline's 3.97 M at 32
        # — a 63.8x capacity error on 96 demonstrations, which silently confounded every
        # imf_backbone='unet' run. DiT / SiT / mf_dit ignore this key (they size from
        # dit_hidden_size). Full audit: logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
        'freq_dim': 32,
        'depth': 8,
        'num_heads': 4,
        'mlp_dim': 256,
        'time_dim': 256,
        'dropout_rate': 0.1,

        ## legacy loss-mixing knobs — INERT in Gen3v6 (u and v are on equal footing,
        ## FIX-4), kept because the trainer/parser plumbing reads them.
        'u_loss_weight': 1.0,
        'v_loss_weight': 1.0,
        'loss_schedule': 'balanced',
        'warmup_epochs': 0,
        'transition_epochs': 0,
        'loss_type': 'l2',
        'predict_epsilon': True,

        ## ── Gen3v6 objective (PLAN §3.5) ──────────────────────────────────────────
        'mf_objective': 'meanflow',        # only value; folder-name slot for future arms
        'meanflow_data_proportion': 0.5,   # fraction forced to r==t (FM anchors) — official
        'mf_adp_p': 1.0,                   # official adaptive-loss exponent
        'mf_adp_eps': 0.01,                # official adaptive-loss epsilon

        ## ── architecture flags ────────────────────────────────────────────────────
        'dual_head': True,           # FIX-4: the v head carries a FULL loss, not a stabiliser
        'interval_cfg': False,       # 🔴 no CFG in Gen3v6. On the UNet arm this changes the
                                     # state_dict, so Gen3v6 checkpoints are NOT interchangeable
                                     # with Gen3v4's — intended, and why the folders are siblings.
                                     # On the DiT arm the ω/interval tokens still exist but are
                                     # fed a constant default (guidance off) ⇒ inert.

        ## backbone selector. MUST match the plan block (state_dict + loadpath depend on it).
        ## valid: 'unet' (DPCC U-Net) | 'dit' (iMF DiT) | 'mf_dit' (U2: official-MeanFlow DiT).
        'imf_backbone': 'mf_dit',    # U2 default: MeanFlow's own DiT (was 'dit'); use 'dit'/'unet' for A/B
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,
        'dit_patch_size': 1,
        'dit_condition_on_t': False, # official conditions on h only — KEEP FALSE (audit §2.3)

        ## dataset (inherited from FMv3ODE / Gen3v4)
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,
        'use_padding': True,
        'condition_dropout': 0.25,
        'condition_guidance_w': 0.0,  # returns-CFG output mix OFF (Gen3v6 has no guidance)

        ## training
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'gradient_clip': 1.0,        # 🔴 ACTUALLY APPLIED in Gen3v6 (dead key in Gen3v4/Gen13 —
                                     # POST_U10_III §4.1). flow_matcher_v3_meanflow/utils/training.py
                                     # clips before optimizer.step() and logs the pre-clip norm.
        'ema_decay': 0.995,
        'action_weight': 10,         # kept for folder naming + utils; NOT applied to the loss (FIX-3)
        'loss_discount': 1.0,        # same: kept, not applied to the loss (FIX-3)
        'gradient_accumulate_every': 2,
        ## ⚠️ TRAP (POST_U10_III §4.2): this is a WINDOW-level split. At H=8 adjacent windows
        ## share 7 of 8 frames, so loss_test is effectively a train loss. Gen3v6 INHERITS the
        ## leak — label every val number in the results MD as leaking, or implement an
        ## episode-level split before claiming generalisation.
        'train_test_split': 0.9,

        ## time schedule — MUST match the plan block (it is in diffusion_loadpath).
        't_schedule': 'logit_normal',
        'p_mean': -0.4,              # official convention; NEGATED inside the τ sampler
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,   # 'beta' ablation arm only — ignored otherwise
        'time_beta_beta_v3': 1.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_meanflow/',
        'exp_name': watch(args_to_watch_fmv3_mf_train),
    },

    'flow_matching_v3_alphaflow': {
        # ── Gen3v7: α-Flow (arXiv 2510.20771, snap-research @ b0fef77) ─────────────────
        # Copy-modify sibling of 'flow_matching_v3_meanflow'. The scientific difference is
        # the TARGET: MeanFlow regresses u to sg(v + h·du/dr) (a JVP of the network itself),
        # α-Flow regresses it to sg(α·v + (1−α)·u_next) — a SELF-BOOTSTRAPPED, no-grad,
        # derivative-free target — with α annealed 1 → 0, so training is a homotopy from
        # plain flow matching (α=1) to MeanFlow (α=0).
        # Everything architectural is held identical to Gen3v4/Gen3v6 so the three-way A/B
        # is controlled. See logs_in_develop/Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md.

        ## model & engine
        'model': 'flow_matcher_v3_alphaflow.models.AlphaFlowEngine',
        'diffusion': 'flow_matcher_v3_alphaflow.models.AlphaFlowODE',
        'horizon': 8,

        ## architecture sizing (UNet arm; DiT sizing is the dit_* block below)
        # 🔴 FIX_8_UNET_WIDTH (2026-08-05) — THIS KEY IS THE UNET CHANNEL WIDTH.
        # Its only consumer anywhere is `dim=freq_dim` in models/*_trajectory_model.py,
        # and Flow_matcher_U_Net_v2 uses that one argument for BOTH the channel width
        # (:106) and the time-embed width (:110). At 256 the backbone was 253.0 M params
        # (channels 256/512/1024/2048) against the DPCC/FMv3ODE baseline's 3.97 M at 32
        # — a 63.8x capacity error on 96 demonstrations, which silently confounded every
        # imf_backbone='unet' run. DiT / SiT / mf_dit ignore this key (they size from
        # dit_hidden_size). Full audit: logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
        'freq_dim': 32,
        'depth': 8,
        'num_heads': 4,
        'mlp_dim': 256,
        'time_dim': 256,
        'dropout_rate': 0.1,

        ## legacy loss-mixing knobs — INERT (u and v are on equal footing), kept because
        ## the trainer/parser plumbing reads them.
        'u_loss_weight': 1.0,
        'v_loss_weight': 1.0,
        'loss_schedule': 'balanced',
        'warmup_epochs': 0,
        'transition_epochs': 0,
        'loss_type': 'l2',
        'predict_epsilon': True,

        ## ── Gen3v7 α schedule (PLAN §3.6; upstream experiments-alphaflow.yaml:155,
        ##    RESCALED to OUR budget) ────────────────────────────────────────────────
        'af_alpha_scheduler': 'sigmoid',
        'af_alpha_init': 1.0,
        'af_alpha_end': 0.0,
        'af_alpha_init_step': 0,
        # 🔴🔴 THE #1 SILENT FAILURE OF THIS GENERATION (PLAN §11 trap 1). MUST equal
        # 'n_train_steps' below. Upstream anneals over 400000 steps; copying that verbatim
        # here leaves α pinned at ~1.0 for all 100k of our steps — i.e. you trained plain
        # flow matching and called it α-Flow, and nothing in the logs would say so.
        # AlphaFlowODE.__init__ raises if these two disagree; the train script also prints
        # the whole α curve before the first step. If you change n_train_steps, change this.
        'af_alpha_end_step': 100000,
        'af_alpha_gamma': 25.0,
        # snap-to-exact-0/1 guard. Without it α becomes a tiny-but-nonzero number and every
        # sample takes the discrete branch with dt≈0 ⇒ a degenerate near-identity target.
        'af_alpha_clamp': 0.005,

        'af_ratio_fm': 0.5,      # FM anchors (h=0). Upstream ships {0.25,0.5,0.75}; Gen3v4/
                                 # Gen3v6 use 0.5, so 0.5 keeps the A/B controlled.
        'af_clamp_utgt': 4.0,    # upstream clamp_utgt — no prior generation here clamps
        # ⚠️ α-Flow's adaptive_loss_weight_eps. DELIBERATELY ≠ MeanFlow/iMF's 0.01 —
        # different method, different constant. Do NOT "harmonise" it (PLAN §11 trap 7).
        'af_adp_eps': 1e-3,

        ## ── architecture flags ────────────────────────────────────────────────────
        'dual_head': True,           # the v head carries a FULL loss, not a stabiliser
        'interval_cfg': False,       # 🔴 no CFG in Gen3v7 — mirrors α-Flow's own non-cfg
                                     # `alphaflow-sigmoid-latentspace-B-2` config and keeps
                                     # the comparison to Gen3v6 clean.

        ## backbone selector. MUST match the plan block (state_dict + loadpath depend on it).
        ## valid: 'unet' (DPCC U-Net) | 'dit' (iMF DiT) | 'sit' (U2: α-Flow's own SiT).
        'imf_backbone': 'sit',       # U2 default: α-Flow's own SiT (was 'dit'); use 'dit'/'unet' for A/B
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,
        'dit_patch_size': 1,
        'dit_condition_on_t': False, # official conditions on h only — KEEP FALSE (audit §2.3)

        ## dataset (inherited from FMv3ODE / Gen3v4 / Gen3v6)
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,
        'use_padding': True,
        'condition_dropout': 0.25,
        'condition_guidance_w': 0.0,  # returns-CFG output mix OFF (Gen3v7 has no guidance)

        ## training
        'n_train_steps': 100000,     # 🔴 keep in sync with af_alpha_end_step above
        'batch_size': 32,
        'learning_rate': 5e-4,
        'gradient_clip': 1.0,        # 🔴 ACTUALLY APPLIED (dead key in Gen3v4/Gen13 —
                                     # POST_U10_III §4.1). Matters MORE here: the discrete
                                     # branch has no JVP and should be calmer, so a
                                     # surviving spike is diagnostic, not background noise.
        'ema_decay': 0.995,
        'action_weight': 10,         # kept for folder naming + utils; NOT applied to the loss
        'loss_discount': 1.0,        # same: kept, not applied to the loss
        'gradient_accumulate_every': 2,
        ## ⚠️ TRAP (POST_U10_III §4.2): this is a WINDOW-level split. At H=8 adjacent windows
        ## share 7 of 8 frames, so loss_test is effectively a train loss. Gen3v7 INHERITS the
        ## leak — label every val number in the results MD as leaking, or implement an
        ## episode-level split before claiming generalisation.
        'train_test_split': 0.9,

        ## time schedule — MUST match the plan block (it is in diffusion_loadpath).
        ## α-Flow's own distrib_t_t_next_mf is minmax over two logit_norm(-0.4, 1.0) draws,
        ## identical to what Gen3v6 already does, so nothing changed here.
        't_schedule': 'logit_normal',
        'p_mean': -0.4,              # official convention; NEGATED inside the τ sampler
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,   # 'beta' ablation arm only — ignored otherwise
        'time_beta_beta_v3': 1.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_alphaflow/',
        'exp_name': watch(args_to_watch_fmv3_af_train),
    },

    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/diffusion/',
        'exp_name': lambda args: f"plans/diffusion/H{args.horizon}_K{args.n_diffusion_steps}_D{args.diffusion}_aw{args.action_weight}/" + watch([
            ('horizon', 'H'),
            ('n_diffusion_steps', 'K'),
            ('diffusion_timestep_threshold', 'T'),
            ('diffusion', 'D')
        ])(args) + _msg_suffix(args),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## diffusion model
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        'action_weight': 10,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
    
    'plan_fm': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching/',
        'exp_name': watch_plan(args_to_watch),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_unet_v2': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_unet_v2/',
        'exp_name': watch_plan(args_to_watch),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching unet v2 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_unet_v2/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v2': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v2/',
        'exp_name': watch_plan(args_to_watch),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching v2 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'vf_time_bins_v2': 20,
        'ode_inference_steps_v2': 10,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v2/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v3/',
        'exp_name': watch_plan(args_to_watch_v3),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching v3 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'flow_steps_v3': 10,
        'ode_inference_steps_v3': 10,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v3/H{horizon}_K{flow_steps_v3}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    # ── Gen12 (HardFlow → FMv3ODE) ────────────────────────────────────────────
    # Eval-ONLY block. Gen12 trains nothing (PLAN §1): it reuses a pre-trained
    # checkpoint. HardFlow's `hardflow_new` sampler is only valid for a single-time
    # velocity field v = f(x, t), so Gen12 loads the **FMv3ODE** model
    # (`FlowMatchingODE`) SPECIFICALLY — not iMF / MeanFlow (two-time u(z,τ,h)) nor
    # anything else. This block is the single control entry (path + eval budget);
    # config/hardflow_projection_eval.yaml holds the experiment knobs (seeds, arms,
    # constraint geometry, arm-C tuning).
    #
    # LOADING: copied from `plan_fm_v3_ode_selectable` — the same RELATIVE templated
    # loadpath, so no machine-specific absolute path. With action_weight=10 it renders
    # to logs/avoiding-d3il/flow_matching_v3_ode_selectable/
    #        H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/<seed>/  — the trained
    # FMv3ODE checkpoint. The pickle's own FlowMatchingODE class loads natively.
    'plan_fm_v3_hardflow': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,            # arms A/B; arm C uses hardflow.batch_size (default 1)
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        # ⭐ Eval K — matched across ALL arms (PLAN §5). The SAMPLING budget, applied to
        # the model AFTER loading; recorded in the results dir name (K<flow_steps>_n<n>).
        # CLI `--flow-steps N` overrides this.
        'flow_steps': 10,
        # Optional: a direct absolute path to the checkpoint parent dir, overriding the
        # templated loadpath below. Default None -> use the FMv3ODE loadpath (recommended,
        # machine-independent).
        'checkpoint_dir': None,

        ## serialization (results path)
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v3_hardflow/',
        'exp_name': watch_plan(args_to_watch_v3),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## FMv3ODE model + loadpath — copied from plan_fm_v3_ode_selectable
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        'action_weight': 10,        # renders '_aw10' -> matches the trained checkpoint
        'time_beta_alpha_v3': 1.5,  # renders '_a1.5'
        'time_beta_beta_v3': 1.0,   # renders '_b1.0'
        'flow_steps_v3': 10,        # K in the results exp_name (not the loadpath)
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'max_path_length': 150,     # read by fit_dynamics_fmv3's default .npz path

        'diffusion_loadpath': 'f:flow_matching_v3_ode_selectable/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_ode_selectable': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_ode_selectable/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch_plan(args_to_watch_fmv3_ode_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching v3 model
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        'action_weight': 1,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        # 'n_diffusion_steps': 20, # DEAD code (mathematically irrelevant for FM flow)
        'flow_steps_v3': 10,
        # 'ode_inference_steps_v3': 10, # DEAD code (compatibility alias for flow_steps_v3)
        # Available backend options: legacy_euler, torchdiffeq.
        'ode_solver_backend_v3': 'legacy_euler',
        # Available method options (torchdiffeq backend):
        # dopri8, dopri5, bosh3, fehlberg2, adaptive_heun,
        # euler, midpoint, heun2, heun3, rk4,
        # explicit_adams, implicit_adams, fixed_adams, scipy_solver.
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'returns_condition': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        # 'predict_epsilon': True, # DEAD code (not used in inference velocity prediction)
        # 'dynamic_loss': False, # DEAD code (legacy DDPM relic, unused in FMv3)

        ## loading
        # 'diffusion_loadpath': 'f:flow_matching_v3_ode_selectable/H{horizon}_K{flow_steps_v3}_D{diffusion}',
        'diffusion_loadpath': 'f:flow_matching_v3_ode_selectable/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',
        # 'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}', # DEAD code (Value functions not used in FMv3 sampling)

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_drifting': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_drifting/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch_plan(args_to_watch_fmv3_ode_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching v3 drifting model
        'diffusion': 'models.diffusion.FlowMatchingDrifting',
        'horizon': 8,
        'action_weight': 1,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        'flow_steps_v3': 10,
        # Available backend options: legacy_euler, torchdiffeq.
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        
        # FM-D Drift Augmentation Parameters (Locked 3 params)
        'use_drift_augmentation': True,            # bool: enable FM-D mode during inference
        'drift_loss_weight': 0.1,                  # float: lambda in drift field equation
        'drift_loss_type': 'embedding_nn',          # str: "embedding_nn" | "adversarial" | "mmd"

        'returns_condition': False,
        'diffusion_timestep_threshold': _yaml_threshold,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v3_drifting/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_imeanflow': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_imeanflow/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}_ts{t_schedule}/',
        'exp_name': watch_plan(args_to_watch_fmv3_ode_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching v3 imeanflow model
        'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
        'horizon': 8,
        'action_weight': 10,
        'u_loss_weight': 1.0,
        'v_loss_weight': 0.1,
        'flow_steps_v3': 2,           # was: 10  — low-NFE real-iMF (use 4 to de-risk first, then 1–2)
        ## U7: MUST match training block (in diffusion_loadpath).
        't_schedule': 'logit_normal',     # U7 DEFAULT — set 'beta' to load pre-U7 checkpoints
        'p_mean': -0.4,
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,        # ignored when t_schedule='logit_normal'; kept for beta ablation
        'time_beta_beta_v3': 1.0,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'diffusion_timestep_threshold': _yaml_threshold,
        'imf_objective': 'imf_official',    # U10 — MUST match training (in diffusion_loadpath)

        ## U5 Phase 1 — must match the trained checkpoint's flags (architecture + CFG)
        'dual_head': True,           # was: False  — MUST equal training (else state_dict mismatch)
        'interval_cfg': True,        # was: False  — MUST equal training
        ## U10 imf_official — CFG is a NET INPUT baked into the trained weights (official iMF), NOT
        ## an output-space mix. At eval, feed a CONSTANT operating point (ω, t_min, t_max); the net
        ## applies the interval internally. ω=1.0 ⇒ guidance OFF (w_arg=1−1/1=0). Set ω∈(1,1+smax]
        ## + an interval to guide. condition_guidance_w stays 0 (returns-CFG path is deleted for
        ## this objective). See U10/PLAN_faithful_imf_replication.md §W8.
        'meanflow_cfg_omega': 1.0,   # EVAL operating point — 1.0 = guidance OFF (was 0.0 legacy-off)
        'meanflow_cfg_t_min': 0.0,   # eval guidance interval (official s-convention), inert while ω=1
        'meanflow_cfg_t_max': 1.0,
        'condition_guidance_w': 0.0, # returns-CFG output-mix OFF (the real neutralizer; sampling knob)
        ## returns_condition is an IDENTITY key (config-override-pkl fix_1 keeps the pkl value to
        ## protect the state_dict). It is fictional/inert for this gen anyway — returns never reach
        ## the backbone, and condition_guidance_w=0 already disables the returns-CFG path. Kept =True
        ## to MATCH the pkl (include_returns=True) so it does not false-warn every eval.
        'returns_condition': True,   # match pkl; inert (neutralized by condition_guidance_w=0)
        ## Train-time-only knobs — MUST equal training so config-overrides-pkl is a no-op (unused at eval)
        'meanflow_cfg_smax': 7.0,
        'meanflow_data_proportion': 0.5,
        'meanflow_class_dropout_prob': 0.1,

        ## EMA switch. Official iMF samples with EMA weights (imeanflow/utils/sample_util.py ema=True).
        ## U10: default True for imf_official (few-step MeanFlow is EMA-sensitive). False = dpcc-legacy raw.
        'eval_use_ema': True,        # was: False — imf-ema (official). Set False for the raw-weights A/B.

        ## U6 — backbone selector. MUST equal the trained checkpoint (state_dict + loadpath).
        'imf_backbone': 'dit',       # 'unet' | 'dit' — U10 imf_official REQUIRES dit (unet no-ops cond_drop)
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,
        'dit_patch_size': 1,
        'dit_condition_on_t': False,

        ## loading — path must match args_to_watch_fmv3_imf_train exactly (incl. _bb{imf_backbone})
        'diffusion_loadpath': 'f:flow_matching_v3_imeanflow/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}_ts{t_schedule}',
        'diffusion_epoch': 'best',
    },

    'plan_fm_v3_meanflow': {
        # ── Gen3v6 evaluation block. Every ARCHITECTURE key below MUST equal the
        # 'flow_matching_v3_meanflow' training block, or the state_dict load fails
        # (trap #6). Only sampling knobs (flow_steps_v3, solver, threshold) may differ.
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_meanflow/' +
                  'H{horizon}_D{diffusion}_aw{action_weight}_obj{mf_objective}_bb{imf_backbone}_ts{t_schedule}_dp{meanflow_data_proportion}/',
        # 🔴 FIX_9_CFG_PROVENANCE — was args_to_watch_fmv3_ode_plan; the _hf_ list adds the A/B tokens.
        'exp_name': watch_plan(args_to_watch_fmv3_hf_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## MeanFlow model
        'diffusion': 'flow_matcher_v3_meanflow.models.MeanFlowODE',
        'horizon': 8,
        'action_weight': 10,
        'u_loss_weight': 1.0,
        'v_loss_weight': 1.0,
        ## ⚠️ MATCHED-BUDGET OR NOTHING (PLAN §7 / fix_7.3 §9): every MeanFlow-vs-X table
        ## must be at equal K. Sweep flow_steps_v3 ∈ {1, 2, 5, 10}; never compare
        ## MeanFlow@K=5 against FM@K=10.
        ## Gen3v6 U3 — matched-K for ALL arms. Both K knobs read HFFM_FLOW_STEPS (default 2) so a
        ## K-sweep is `HFFM_FLOW_STEPS=<K> ./submit.sh …`. Because K flows through the CONFIG (not a
        ## post-load model patch), args.flow_steps_v3 → exp_name '_K{K}_' → each K writes its OWN
        ## results dir (no cross-K overwrite). flow_steps_v3 drives arms A/B (native sampler K);
        ## flow_steps drives arm C (HardFlow Euler K); kept EQUAL (matched-budget, PLAN §7).
        'flow_steps_v3': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
        'flow_steps': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
        ## MUST match training (both are in diffusion_loadpath)
        'mf_objective': 'meanflow',
        'meanflow_data_proportion': 0.5,
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,        # ignored when t_schedule='logit_normal'
        'time_beta_beta_v3': 1.0,
        ## train-time-only knobs — kept equal to training so config-overrides-pkl is a no-op
        'mf_adp_p': 1.0,
        'mf_adp_eps': 0.01,

        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'diffusion_timestep_threshold': _yaml_threshold,
        # 🔴 FIX_9_CFG_PROVENANCE — arm-C knobs, present so they reach the results path as the
        # A/B tokens. Resolved by the eval script (see the FMPCC_PROJ_CFG header); inert for the
        # DPCC-only arms, which is the price of a path that never lies about what ran.
        'hf_act_threshold': _hf_act_threshold,
        'hf_batch_size': _hf_batch_size,

        ## architecture — MUST equal the trained checkpoint
        'dual_head': True,
        'interval_cfg': False,
        ## valid: 'unet' | 'dit' | 'mf_dit' (U2) — MUST equal the train block's value.
        'imf_backbone': 'mf_dit',
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,      # iMF 'dit' only (ignored by 'mf_dit'/'unet')
        'dit_patch_size': 1,
        'dit_condition_on_t': False,  # iMF 'dit' only (ignored by 'mf_dit'/'unet')

        ## Gen3v6 has NO interval-CFG: there is no eval-time guidance operating point.
        ## condition_guidance_w=0 keeps the DPCC returns-CFG output mix off as well.
        'condition_guidance_w': 0.0,
        ## returns_condition is an IDENTITY key (config-override-pkl fix_1 keeps the pkl
        ## value to protect the state_dict). Fictional/inert for this gen — returns never
        ## reach the backbone. Kept =True to MATCH the pkl (include_returns=True) so it
        ## does not false-warn on every eval.
        'returns_condition': True,

        ## Few-step MeanFlow is EMA-sensitive and the official recipe samples with EMA.
        'eval_use_ema': True,        # set False for the raw-weights A/B

        ## loading — 🔴 must reproduce args_to_watch_fmv3_mf_train token-for-token,
        ## including _dp{meanflow_data_proportion}, or eval silently finds no checkpoint.
        'diffusion_loadpath': 'f:flow_matching_v3_meanflow/' +
                  'H{horizon}_D{diffusion}_aw{action_weight}_obj{mf_objective}_bb{imf_backbone}_ts{t_schedule}_dp{meanflow_data_proportion}',
        'diffusion_epoch': 'best',
    },

    'plan_fm_v3_alphaflow': {
        # ── Gen3v7 evaluation block. Every ARCHITECTURE key below MUST equal the
        # 'flow_matching_v3_alphaflow' training block, or the state_dict load fails.
        # Only sampling knobs (flow_steps_v3, solver, threshold) may differ.
        #
        # ✅ α is TRAINING-ONLY. It does not appear at inference and the sampler is
        # unchanged (x += dt·u) — which is exactly what makes Gen3v4 / Gen3v6 / Gen3v7
        # comparable at matched K. The af_* keys below are carried only so the
        # config-overrides-pkl reconciliation stays a silent no-op.
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_alphaflow/' +
                  'H{horizon}_D{diffusion}_aw{action_weight}_bb{imf_backbone}_ts{t_schedule}'
                  '_ai{af_alpha_init}_ae{af_alpha_end}_ag{af_alpha_gamma}_rf{af_ratio_fm}/',
        # 🔴 FIX_9_CFG_PROVENANCE — was args_to_watch_fmv3_ode_plan; the _hf_ list adds the A/B tokens.
        'exp_name': watch_plan(args_to_watch_fmv3_hf_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## α-Flow model
        'diffusion': 'flow_matcher_v3_alphaflow.models.AlphaFlowODE',
        'horizon': 8,
        'action_weight': 10,
        'u_loss_weight': 1.0,
        'v_loss_weight': 1.0,
        ## ⚠️ MATCHED-BUDGET OR NOTHING (PLAN §8 / fix_7.3 §9): every α-Flow-vs-X table must
        ## be at equal K. Sweep flow_steps_v3 ∈ {1, 2, 5, 10}; never compare
        ## α-Flow@K=5 against FM@K=10. The comparator is FM @ K=2 → 100% safe, 0.1894 s/plan.
        ## Gen3v7 U3 — matched-K for ALL arms (mirrors plan_fm_v3_meanflow L1252-1253). Both K
        ## knobs read HFFM_FLOW_STEPS (default 2), so a K-sweep is
        ## `HFFM_FLOW_STEPS=<K> ./submit.sh …` and `--flow-steps <K>` patches both (the eval
        ## driver's CLI override sets flow_steps too). Because K flows through the CONFIG rather
        ## than a post-load model patch, args.flow_steps_v3 → exp_name '_K{K}_' → each K writes
        ## its OWN results dir and no two budgets can overwrite each other.
        ##   flow_steps_v3 → arms A/B (the native α-Flow sampler's K)
        ##   flow_steps    → arm C   (the HardFlow Euler K)
        ## Kept EQUAL. Diverging them silently voids every arm-B-vs-arm-C comparison.
        'flow_steps_v3': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
        'flow_steps': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
        ## MUST match training (these four are in diffusion_loadpath)
        'af_alpha_init': 1.0,
        'af_alpha_end': 0.0,
        'af_alpha_gamma': 25.0,
        'af_ratio_fm': 0.5,
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,        # ignored when t_schedule='logit_normal'
        'time_beta_beta_v3': 1.0,
        ## train-time-only knobs — kept equal to training so config-overrides-pkl is a no-op
        'af_alpha_scheduler': 'sigmoid',
        'af_alpha_init_step': 0,
        'af_alpha_end_step': 100000,
        'af_alpha_clamp': 0.005,
        'af_clamp_utgt': 4.0,
        'af_adp_eps': 1e-3,

        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'diffusion_timestep_threshold': _yaml_threshold,
        # 🔴 FIX_9_CFG_PROVENANCE — arm-C knobs, present so they reach the results path as the
        # A/B tokens. Resolved by the eval script (see the FMPCC_PROJ_CFG header); inert for the
        # DPCC-only arms, which is the price of a path that never lies about what ran.
        'hf_act_threshold': _hf_act_threshold,
        'hf_batch_size': _hf_batch_size,

        ## architecture — MUST equal the trained checkpoint
        'dual_head': True,
        'interval_cfg': False,
        ## valid: 'unet' | 'dit' | 'sit' (U2) — MUST equal the train block's value.
        'imf_backbone': 'sit',
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,      # iMF 'dit' only (ignored by 'sit'/'unet')
        'dit_patch_size': 1,
        'dit_condition_on_t': False,  # iMF 'dit' only (ignored by 'sit'/'unet')

        ## Gen3v7 has NO interval-CFG: there is no eval-time guidance operating point.
        ## condition_guidance_w=0 keeps the DPCC returns-CFG output mix off as well.
        'condition_guidance_w': 0.0,
        ## returns_condition is an IDENTITY key (config-override-pkl fix_1 keeps the pkl
        ## value to protect the state_dict). Fictional/inert for this gen — returns never
        ## reach the backbone. Kept =True to MATCH the pkl (include_returns=True) so it
        ## does not false-warn on every eval.
        'returns_condition': True,

        ## Few-step MeanFlow-family models are EMA-sensitive; the official recipes use EMA.
        'eval_use_ema': True,        # set False for the raw-weights A/B

        ## loading — 🔴 must reproduce args_to_watch_fmv3_af_train token-for-token
        ## (H, D, aw, bb, ts, ai, ae, ag, rf) or eval silently finds no checkpoint.
        'diffusion_loadpath': 'f:flow_matching_v3_alphaflow/' +
                  'H{horizon}_D{diffusion}_aw{action_weight}_bb{imf_backbone}_ts{t_schedule}'
                  '_ai{af_alpha_init}_ae{af_alpha_end}_ag{af_alpha_gamma}_rf{af_ratio_fm}',
        'diffusion_epoch': 'best',
    },

    ## ── Hyperparameter Tuning Blocks ──────────────────────────────────
    ## These use the ORIGINAL flow_matcher model (UNet1DTemporalCondModel).
    ## Duplicate this pair (train + plan) for each tuning experiment.
    ## CRITICAL: Always use a unique 'prefix' to avoid overwriting data.
    ## See: logs_in_develop/guiding_hyperpara_tuning/hyperparameter_tuning_guide.md

    'flow_matching_hp_tune': {
        # HP Tune 1: example tuning run — same model, different hyperparams
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization — UNIQUE PREFIX for this tuning run
        'logbase': logbase,
        'prefix': 'flow_matching_hp_tune1/',
        'exp_name': watch(args_to_watch),

        # training — MODIFY THESE for your tuning experiment
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

    'plan_fm_hp_tune': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization — MUST match the training prefix
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_hp_tune1/',
        'exp_name': watch_plan(args_to_watch),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

        ## flow matching model (same as base flow_matching)
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading — points to the hp_tune training folder
        'diffusion_loadpath': 'f:flow_matching_hp_tune1/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
}
