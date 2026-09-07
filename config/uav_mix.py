"""Gen15 (UAV Mix-ML) config — UAV pipeline with a selectable ML engine.

Modelled on `config/uav.py` (Gen11), which stays BYTE-IDENTICAL: this generation shares no
mutable file with Gen11. Nothing here is imported from `config/uav.py` and nothing there is
appended to. See logs_in_develop/Gen15/init/PLAN_Gen15_uav_mix_ml.md §5 G5.

    engine='fm'  → Gen11  FlowMatchingODE   — the incumbent, reproduced for parity gate G1
    engine='mf'  → Gen3v6 MeanFlowODE       — MeanFlow (arXiv 2505.13447)
    engine='af'  → Gen3v7 AlphaFlowODE      — alpha-Flow (arXiv 2510.20771)

Blocks: `mix_uav_<engine>` (train) and `plan_mix_uav_<engine>` (eval), resolved by
mix_uav/models/engine_registry.py:experiment_name(). The train/eval scripts set
`exp = 'uav_mix'` → `config = 'config.uav_mix'`.

Scene selection is NOT a separate config: the `--scene` CLI flag sets the dataset string to
`uav-<scene>`, which both selects the data branch in `datasets/d4rl.py` and segregates the
output path (`logs/UAV_MIX/uav-<scene>/<exp_name>/<seed>/`).

UAV schema (authoritative, unchanged from Gen11 E8): cond_mode='pos_only' → obs=[p_des|p] 6-D,
action=Δp_des 3-D, transition=9-D, H=8. Dims are derived from the data at runtime, so they are
not hard-set here.

⚠️ DUPLICATED CONSTANTS. `MAX_PATH_LENGTH_PER_SCENE`, `_COND_MODE_DIM` and `args_to_watch`
below are COPIES of the ones in `config/uav.py`, not imports. That is deliberate (isolation
over DRY — the repo's copy-modify convention), and it means a future re-tune of a per-scene
max_path_length in Gen11 does NOT propagate here. Do not "fix" this by importing.
"""

import os                       # Gen15 U6 — the UAV_MIX_* env knobs below
import yaml
from diffuser.utils import watch

# Read projection threshold from the SHARED yaml. This is the ONE artifact Gen11 and Gen15
# have in common, and it is shared ON PURPOSE: identical constraints are what make the two
# generations' numbers comparable at all. It is read-only — nothing here writes it. Per-run
# provenance is preserved anyway: utils/setup.py:snapshot_configs copies the loaded yaml into
# every run's config_snapshot_uav/ folder.
with open('config/uav_projection.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError(
        "CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/uav_projection.yaml"
    )
_yaml_threshold = _proj_config['diffusion_timestep_threshold']

# ── Per-scene episode buffer sizes (steps at DATASET_HZ=33 Hz) — copy of Gen11's ────────────
MAX_PATH_LENGTH_PER_SCENE = {
    'empty':    450,
    'corridor': 360,
    's_curve':  750,
    'pillars':  560,
    'all':      750,
}

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
]

_COND_MODE_DIM = {
    'p_des':    None,   # 12D — no suffix (Gen11 E7 backward compat)
    'pos_only': '9D',   # 9D — velocity dropped from obs
}

# ── Path discriminator ──────────────────────────────────────────────────────────────────────
# Gen15's OWN helper. Gen11's `_uav_exp_name` is NOT imported, NOT edited, NOT called.
#
#   <prefix>H{horizon}_D{diffusion}[_9D][<engine tokens>]
#
# The engine tokens are the part Gen11's helper lacks and Gen15 needs. `D{diffusion}` is the
# raw engine class path, so the three arms already separate — but two `mf` runs differing only
# in `meanflow_data_proportion` (a first-class ablation axis in Gen3v6, folder-tagged there as
# `dp`) or in `imf_backbone` would land in the SAME checkpoint directory and silently overwrite
# each other. The token list per engine lives in the registry (`exp_name_tokens`) so the config
# and the dispatch table can never disagree.
#
# `fm` declares NO tokens, which keeps its path Gen11-shaped (modulo prefix/logbase) so parity
# gate G1 compares like with like.


def _fmt(value):
    """Compact, filesystem-safe rendering of a token value (0.5 -> '0.5', 1.0 -> '1')."""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _uav_mix_exp_name(args):
    prefix = getattr(args, 'prefix', '')
    name = f'H{args.horizon}_D{args.diffusion}'

    cond_mode = getattr(args, 'cond_mode', 'pos_only')
    dim_tag = _COND_MODE_DIM.get(cond_mode, f'cm{cond_mode}')
    if dim_tag:
        name += f'_{dim_tag}'

    # Engine identity tokens (registry-driven; empty for `fm`).
    from mix_uav.models import engine_registry
    engine = getattr(args, 'engine', engine_registry.DEFAULT_ENGINE)
    for key, label in engine_registry.get(engine)['exp_name_tokens']:
        if hasattr(args, key):
            name += f'_{label}{_fmt(getattr(args, key))}'

    parts = [p for p in [prefix, name] if p]
    return '_'.join(parts).replace('/_', '/')


# All Gen15 outputs live under logs/UAV_MIX/ — a DIFFERENT root from Gen11's logs/UAV_FM/.
# Isolation at the top of the path, not only at the `prefix` segment: a shared root is one bad
# prefix away from overwriting a Gen11 checkpoint.
logbase = 'logs/UAV_MIX'


# ═══ Gen15 U6 — the three env knobs this generation shipped without ═════════════════════
#
# Before U6 `config/uav_mix.py` read NO environment variable at all. Three consequences,
# all of them silent:
#
#  1. 🔴 `af_alpha_end: 0.0` was unreachable. The sigmoid + `af_alpha_clamp=0.005` snap alpha
#     to EXACTLY 0 from ~71.2% of the budget on, and af_diffusion.py:568 routes `alpha <= 0`
#     into Gen3v6's `_p_losses_meanflow` body UNMODIFIED. So every Gen15 `af` checkpoint ever
#     trained DEPLOYS A MEANFLOW MODEL wearing an alpha-Flow folder name — jobs 25135-25138
#     (pillars) included. The bootstrapped objective has never trained a single weight we
#     evaluated. Gen3v7 hit the identical defect and fixed it with AF_ALPHA_END
#     (config/avoiding-d3il.py:114-129, commit beb7f26c); this is the same fix, Gen15 spelling.
#
#  2. 🔴 The `af` arm was pinned to the SiT backbone, which is NOT parameter-matched to the
#     4.0 M U-Net the `fm`/`mf` arms run (dit_hidden_size=256, dit_depth=8 => ~9.4 M by the
#     18*depth*d^2 rule). Every published Gen15 af-vs-mf row therefore confounds OBJECTIVE
#     with BACKBONE and PARAMETER COUNT. U6 flips the default to 'unet' — which is what
#     _TWO_TIME_BACKBONE's own comment says this generation locked itself to — and keeps
#     'sit' one env var away, so the appendix arm is preserved, not deleted.
#
#  3. The checkpoint the eval deploys was fixed at 'best'. On the af arm `best` is chosen on
#     a test_loss that scales with alpha, so it structurally prefers a MID-CURRICULUM model
#     (Gen3v7 DA 2026-09-01 §3.1/§4.2 — `latest` vs `best` was the whole difference between
#     0/2 and 2/2 goals at K=1). Pairing knob 1 with 'best' floors alpha and then discards
#     the checkpoint the floor produced.
#
# ✅ PATH SAFETY IS ALREADY GUARANTEED for knobs 1 and 2: `('af_alpha_end', 'ae')` and
#    `('imf_backbone', 'bb')` are UNCONDITIONAL exp_name_tokens
#    (mix_uav/models/engine_registry.py:316-317) and `_uav_mix_exp_name` renders them for
#    every run. So each value already trains into its OWN directory — no overwrite, no
#    --auto-resume collision. Gate G2 asserts exactly this across the knob cross-product.
#    Knob 3 is eval-only and gets its folder token in `_uav_eval_tag` (eval_mix_uav.py).
#
# 🔴 THE DEFAULT FLIP IS NOT PATH-NEUTRAL, AND THAT IS THE POINT. The af checkpoint path
#    carries `_bbsit` today and `_bbunet` from now on. Existing SiT checkpoints are NOT
#    touched, NOT overwritten and NOT deleted — they simply are no longer what a bare
#    `... af` invocation resolves to. Reach them again with UAV_MIX_BONE_AF=sit. A default
#    `af` eval WILL fail on a missing checkpoint until the U-Net arm has been trained; that
#    is a loud, correct failure, and far better than the silent confound it replaces.


def _env_or_none(name):
    """Env value, or None when unset **or blank**.

    🔴 BLANK MUST MEAN UNSET. Shell `VAR= cmd` does not unset VAR — it exports the EMPTY
    STRING, so a bare `os.environ.get(name) is not None` is True and `float('')` kills the
    job at config import. Gen14 learned this the expensive way (job 25215); Gen15 inherits
    the rule rather than the bug.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


# Must equal engine_registry.get('af')['backbones']. Not imported here: this module is read
# at config-import time and must not pull in the model package. The U6 gate asserts the two
# agree, so drift is caught rather than assumed away.
_UAV_AF_BONES = ('unet', 'dit', 'sit')
_UAV_AF_BONE_DEFAULT = 'unet'


def _uav_af_bone():
    """The af arm's ML backbone. UAV_MIX_BONE_AF, default 'unet' (U6 — was 'sit')."""
    raw = _env_or_none('UAV_MIX_BONE_AF')
    if raw is None:
        return _UAV_AF_BONE_DEFAULT
    if raw not in _UAV_AF_BONES:
        raise ValueError(
            f"CRITICAL: UAV_MIX_BONE_AF='{raw}' is not one of {list(_UAV_AF_BONES)}. "
            f"('mf_dit' belongs to the mf arm — a different class with different provenance; "
            f"accepting it here would train a model whose folder names another architecture.)")
    return raw


def _uav_af_alpha_end():
    """Terminal alpha of the af schedule. UAV_MIX_AF_ALPHA_END, default 0.0 (= MeanFlow).

    >0 floors alpha so the bootstrapped branch stays live to the last step, i.e. the deployed
    model is genuinely alpha-Flow. Upstream's own switch for the same intent is
    `discrete_training: true` (aux_repo/alphaflow/src/training/loss.py:421-426), which floors
    alpha at clamp_value instead of snapping to 0 — so this is a supported deviation, not an
    invented knob. ⚠️ It IS a deviation: every upstream recipe ends at 0, because alpha-Flow
    is a CURRICULUM whose destination is MeanFlow. Say so in any write-up.

    🔴 Must land below 1 - af_alpha_clamp and above af_alpha_clamp, or _get_ratio snaps it to
    an endpoint and the folder name lies about what trained.
    """
    raw = _env_or_none('UAV_MIX_AF_ALPHA_END')
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"CRITICAL: UAV_MIX_AF_ALPHA_END='{raw}' is not a float.")
    if not (0.0 <= val <= 1.0):
        raise ValueError(
            f"CRITICAL: UAV_MIX_AF_ALPHA_END={val} must lie in [0, 1]; "
            f"alpha=1 is pure flow matching, alpha=0 is MeanFlow.")
    _clamp = 0.005                      # af_alpha_clamp, below
    if 0.0 < val < _clamp:
        raise ValueError(
            f"CRITICAL: UAV_MIX_AF_ALPHA_END={val} is below af_alpha_clamp={_clamp}, so "
            f"_get_ratio snaps it to exactly 0.0 and the arm ends on the MeanFlow target "
            f"while the folder still reads '_ae{val:g}'. Raise alpha, or lower the clamp.")
    return val


_UAV_EPOCH_DEFAULT = 'best'


def _uav_epoch(raw=None):
    """Which checkpoint the eval deploys. UAV_MIX_EPOCH: 'best' | 'latest' | <step>.

    Returns the value for `diffusion_epoch`. `raw=None` reads the environment; the eval
    script passes its --epoch through here too, so the CLI and the env form can never
    disagree about what is legal. Rejecting a typo HERE keeps it from becoming a
    'state_<garbage>.pt' FileNotFoundError minutes into a GPU allocation.
    """
    if raw is None:
        raw = _env_or_none('UAV_MIX_EPOCH')
    if raw is None:
        return _UAV_EPOCH_DEFAULT
    val = str(raw).strip()
    if val in ('best', 'latest'):
        return val
    if not val.isdigit():
        raise ValueError(
            f"CRITICAL: UAV_MIX_EPOCH/--epoch='{val}' is not 'best', 'latest', or a "
            f"non-negative step number.")
    return int(val)


_AF_BONE      = _uav_af_bone()
_AF_ALPHA_END = _uav_af_alpha_end()
_UAV_EPOCH    = _uav_epoch()


# ── Shared UAV task/data settings — identical on all three arms by construction ─────────────
# Anything an engine must NOT change lives here. Copied verbatim from Gen11's
# `flow_matching_v3_uav` block; the per-engine blocks below spread this dict first and then
# add only their objective knobs.
_UAV_TASK = {
    'horizon': 8,
    'loss_type': 'l2',
    'loss_discount': 1.0,
    'returns_condition': False,
    'action_weight': 1,
    'predict_epsilon': True,

    # E8 — observation layout + tracker. cond_mode is a path discriminator; controller is
    # runtime-only (all controllers share the same trained weights).
    'cond_mode': 'pos_only',
    'controller': 'pid_stopgo',

    # dataset — generic SequenceDataset; the UAV branch lives in datasets/d4rl.py.
    'loader': 'datasets.SequenceDataset',
    # SafeLimitsNormalizer, NOT LimitsNormalizer: some scenes (e.g. pillars) have a constant
    # feature column (zero range) → plain LimitsNormalizer does (x-min)/(max-min) = 0/0 = NaN,
    # which poisons the whole net (all losses NaN from epoch 0).
    'normalizer': 'SafeLimitsNormalizer',
    'preprocess_fns': [],
    'clip_denoised': False,
    'use_padding': True,
    'max_path_length': 750,   # fallback (scene='all'); per-scene: MAX_PATH_LENGTH_PER_SCENE
    'include_returns': False,  # UAV has no reward signal
    'returns_scale': 400,      # unused when include_returns=False (API parity)
    'discount': 0.99,
    'dataset_root': 'data/uav_fm/v1',   # documentation; loader honours UAV_FM_DATA_ROOT

    # serialization
    'logbase': logbase,
    'exp_name': _uav_mix_exp_name,

    # training budget — identical across arms so the comparison is compute-matched.
    'n_steps_per_epoch': 1000,
    'n_train_steps': 100000,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'gradient_accumulate_every': 2,
    'ema_decay': 0.995,
    'train_test_split': 0.9,
    'device': 'cuda',
    'seed': 0,
}

# ── Shared eval settings ────────────────────────────────────────────────────────────────────
_UAV_PLAN = {
    'horizon': 8,
    'cond_mode': 'pos_only',
    'controller': 'pid_stopgo',

    'loadbase': None,
    'logbase': logbase,
    'exp_name': _uav_mix_exp_name,
    # ── Gen15 U6 ── UAV_MIX_EPOCH. Eval-only: it picks among files already in the checkpoint
    # tree, so the CHECKPOINT path is untouched and no retrain is implied. The RESULTS folder
    # gains an '_EP<sel>' token (eval_mix_uav.py:_uav_eval_tag) when non-default, so a
    # 'latest' pass lands beside the 'best' one instead of silently overwriting it.
    'diffusion_epoch': _UAV_EPOCH,

    # 🔴 K (NFE budget) — the primary experimental axis of Gen15 (PLAN §7.3).
    #
    # ⚠️ In Gen11 this key was inert in BOTH directions and the bug is inherited by anyone who
    # copies that plumbing: (1) `load_diffusion(override_args=...)` receives the TRAIN block's
    # args, which have no `flow_steps_v3`, so the pkl's training value (10) was kept and the
    # plan block's 20 never reached the sampler; (2) `_uav_eval_tag` read `flow_steps_v3` off
    # the YAML-derived cfg dict, which never had it, so every folder was labelled `K20`.
    # Result: Gen11 evals sampled at K=10 inside folders named K20.
    # eval_mix_uav.py closes both paths — it injects this value into the cfg dict AND calls
    # engine_registry.apply_nfe() on the loaded model, then prints the resolved K.
    'flow_steps_v3': 20,
    'mpc_batch_size': 4,
    'diffusion_timestep_threshold': _yaml_threshold,

    'behavior_log': True,
    'control_hz': 33,

    # ── Gen15 U2 — HardFlow arm (arm C) ──────────────────────────────────────────────────────
    # A different guidance MECHANISM, not another DPCC projection variant: DPCC generates then
    # projects; HardFlow solves a prox-NLP inside every ODE step (`hardflow_new`, the only
    # portable upstream mode — the other three compile the U-Net into the NLP via l4casadi).
    #
    # 🔴 These variants live HERE and not in `config/uav_projection.yaml`, which is shared
    # read-only with Gen11. Putting them in the yaml would make Gen11's next eval try to run an
    # arm it has no code for. The CONSTRAINTS still come from the shared yaml — that is the half
    # that must match for a DPCC-vs-HardFlow comparison to be valid at all.
    #
    # Set to [] (or export UAV_MIX_HF_OFF=1) to run a DPCC-only eval.
    # 🔴 B4_PARITY (2026-08-20) — was ['hardflow_new', 'hardflow_new-c', 'hardflow_new-t'].
    # Arm C runs at `mpc_batch_size` (4), the same fan as arm B — and at B>1 bare
    # `hardflow_new` and `hardflow_new-r` are BYTE-IDENTICAL (both select index 0), so the
    # bare entry was the random arm under a name that promises upstream's batch-1 setting.
    # Renamed to '-r': identical compute, identical numbers, honest name. Bare
    # `hardflow_new` is still available and now resolves to B=1 (the faithful control) via
    # mix_uav/sampling/hardflow_projection.py::resolve_hf_batch_size — add it back here if
    # you want that arm, but know it is a DIFFERENT experiment, not a rename.
    # ── U5 (2026-08-27) — ACTIVE LIST. Pre-U5 value, kept for rollback (do not delete):
    #   'hardflow_variants': ['hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t'],
    #
    # Three changes:
    #  1. Bare `hardflow_new` is BACK. B4_PARITY removed it because at B=4 it was byte-identical
    #     to `-r`; it is re-added deliberately as the B=1 arm. `resolve_hf_batch_size` pins the
    #     bare name to 1 and gives every `-r`/`-c`/`-t` name the run-level `mpc_batch_size` (4).
    #     So this list now spans BOTH fan sizes, which is the point: B=1 is upstream-faithful
    #     HardFlow, B=4 is the DPCC-matched fan. They are different experiments, not a rename.
    #  2. `-geo_free` siblings for the three selection arms — HardFlow's prox-NLP built from
    #     dynamics + action bounds ONLY, no walls/pillars/box. The arm-C mirror of the
    #     `dpcc-*-geo_free` rows added to config/uav_projection.yaml in the same update, so the
    #     two arms stay constraint-matched row-for-row (the Gen12 port's first design rule).
    #  3. DPCC's own fan is untouched — arm B reads `mpc_batch_size` (4) directly and is never
    #     routed through `resolve_hf_batch_size`.
    #
    # Batch sizes resolved at runtime (mix_uav/sampling/hardflow_projection.py):
    #   hardflow_new                → B=1     hardflow_new-r/-c/-t            → B=4
    #   hardflow_new-r/-c/-t-geo_free → B=4   (U5 patch strips the toggle suffix first)
    'hardflow_variants': [
        'hardflow_new',                                                    # B=1, full stack
        'hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t',              # B=4, full stack
        'hardflow_new-r-geo_free',                                         # B=4, geometry OFF
        'hardflow_new-c-geo_free',
        'hardflow_new-t-geo_free',
    ],
    'hardflow': {
        # `dynamics_mode='deriv'` needs NO fitted linear-dynamics .npz — it writes the UAV's own
        # x[t+1] = x[t] + dt*dx[t] rows straight into the NLP. This is why HardFlow is portable
        # to UAV at all; the init plan §1.9 blocker applies only to 'linear_fit', which would
        # need an A/B/c refit in UAV normalizer units (the Gen12 warning) and is NOT used.
        'dynamics_mode': 'deriv',
        # Fraction of the LATE trajectory over which the NLP is active — same polarity as DPCC's
        # diffusion_timestep_threshold (higher = more projection), so 0.5 here == 0.5 there and
        # the two arms are threshold-matched by construction.
        'activation_threshold': 0.5,
        'reg_scale': 1.0,
        'candidate_cost': 'prox',      # 'prox' | 'control'
        'ipopt_print_level': 0,        # keep IPOPT silent in batch logs
        'casadi_print_time': False,
    },

    # MJX predictive-sampling knobs (controller='mjpc' only).
    'mjx_n_samples':  16,
    'mjx_horizon':    0.3,
    'mjx_n_improve':  5,
    'mjx_vel_weight': 0.1,

    'device': 'cuda',
    'seed': 0,
    'preprocess_fns': [],
    'returns_condition': False,
}

# ── Two-time backbone settings (mf + af) ────────────────────────────────────────────────────
# 🔴 FIX_8_UNET_WIDTH (Gen3v6): on the 'unet' backbone `freq_dim` IS the channel width — it is
# passed straight through as the U-Net's `dim`. 32 => 3.97 M params, 256 => 253 M. Gen11 trains
# at dim=32, so 32 is what makes the three arms parameter-identical (gate G3 asserts it).
# NEVER raise freq_dim to "improve the time embedding".
#
# `imf_backbone` is the ML-backbone selector, and it is what makes this generation "Mix-ML"
# in the architecture sense too:
#     mf : 'unet' | 'dit' | 'mf_dit'
#     af : 'unet' | 'dit' | 'sit'
#     fm : 'unet' only (the FM lineage never grew a transformer backbone)
# LOCKED to 'unet' for the headline comparison — with the backbone fixed, the arms differ only
# in objective and sampler, which is the only way the three-way result means anything
# (PLAN §6). The DiT/SiT arms are a deferred appendix: flip this key, and the `bb` token in
# exp_name keeps their checkpoints in separate directories automatically.
_TWO_TIME_BACKBONE = {
    'freq_dim': 32,
    'depth': 8,
    'num_heads': 4,
    'mlp_dim': 256,
    'time_dim': 256,
    'dropout_rate': 0.1,
    'dual_head': True,       # the v head carries a FULL loss, not a stabiliser (Gen3v6 FIX-4)
    'interval_cfg': False,   # neither Gen3v6 nor Gen3v7 has interval-CFG
    'imf_backbone': 'unet',  # ← the ML-backbone switch (see above)
    'dit_depth': 8,
    'dit_hidden_size': 256,
    'dit_num_heads': 4,
    'dit_aux_head_depth': 2,
    'dit_patch_size': 1,
    'dit_condition_on_t': False,
    # 🔴 ACTUALLY APPLIED by utils/training_twotime.py before optimizer.step() (dead key in
    # Gen3v4/Gen13, which logged 65-500x loss spikes).
    'gradient_clip': 1.0,
    # Two-time losses: u and v heads weighted equally; no CFG anywhere in this family.
    'u_loss_weight': 1.0,
    'v_loss_weight': 1.0,
    'loss_schedule': 'balanced',
    'warmup_epochs': 0,
    'transition_epochs': 0,
    'condition_guidance_w': 0.0,
    # Official logit-normal time sampling (both Gen3v6 and Gen3v7).
    't_schedule': 'logit_normal',
    'p_mean': -0.4,
    'p_std': 1.0,
    'time_beta_alpha_v3': 1.0,
    'time_beta_beta_v3': 1.0,
}


base = {

    # ══ arm: fm — Gen11 Flow Matching, reproduced verbatim ═══════════════════════════════════
    'mix_uav_fm': {
        **_UAV_TASK,
        'engine': 'fm',
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'prefix': 'mix_uav_fm/',

        # U-Net sizing — identical to Gen11's flow_matching_v3_uav block.
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,

        # v3 SafeFlow-style time sampling (unchanged from Gen11).
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
    },

    'plan_mix_uav_fm': {
        **_UAV_PLAN,
        'engine': 'fm',
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'prefix': 'plans/mix_uav_fm/',
        'diffusion_loadpath': 'f:mix_uav_fm/H{horizon}_D{diffusion}',
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
    },

    # ══ arm: diffusion — the DPCC baseline (DDPM), U3 ════════════════════════════════════════
    # 🔴 WHY THIS ARM EXISTS: on avoiding-d3il the DPCC `GaussianDiffusion` model IS the Target
    # every DA is measured against. On UAV it had never been trained, so Gen15's claim was capped
    # at "vs naive FM". This block closes that gap.
    #
    # 🔴 action_weight = 1, NOT DPCC's 10. Deliberate deviation: the paper baseline uses aw10,
    # but every other Gen15 arm (and all of Gen11) uses aw1, and Gen15's question is "which
    # OBJECTIVE wins on this task", not "reproduce DPCC's hyperparameters". Keeping aw1 makes
    # this arm comparable to fm/mf/af (same task config, same backbone, same budget). It is
    # therefore NOT a like-for-like reproduction of the avoiding-d3il Target row — say so in any
    # write-up, and train a second aw10 variant if that comparison is ever wanted (the `_aw`
    # token is not in exp_name, so change action_weight AND add a token before doing that).
    'mix_uav_diffusion': {
        **_UAV_TASK,
        'engine': 'diffusion',
        'model': 'models.unet1d_ddpm_cond.UNet1DTemporalCondModel',
        'diffusion': 'models.ddpm_diffusion.GaussianDiffusion',
        'prefix': 'mix_uav_diffusion/',

        # 🔴 K, and it is a TRAINING-time property here (the beta schedule is built from it).
        # It is in exp_name as `_K{n}` so two budgets cannot share a checkpoint folder. A K
        # sweep on this arm = separate training runs (as on avoiding-d3il).
        'n_diffusion_steps': 20,

        # U-Net sizing — identical to the `fm` arm so the four-way stays architecture-matched.
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
    },

    'plan_mix_uav_diffusion': {
        **_UAV_PLAN,
        'engine': 'diffusion',
        'diffusion': 'models.ddpm_diffusion.GaussianDiffusion',
        'prefix': 'plans/mix_uav_diffusion/',
        'diffusion_loadpath': 'f:mix_uav_diffusion/H{horizon}_D{diffusion}',
        # ⚠️ MUST match the training block — exp_name reads it to rebuild the same savepath.
        'n_diffusion_steps': 20,
        # HardFlow is unavailable on this arm (no velocity field); the eval drops it and says so.
        'hardflow_variants': [],
    },

    # ══ arm: mf — Gen3v6 MeanFlow ════════════════════════════════════════════════════════════
    'mix_uav_mf': {
        **_UAV_TASK,
        **_TWO_TIME_BACKBONE,
        'engine': 'mf',
        'model': 'models.mf_engine.MeanFlowEngine',
        'diffusion': 'models.mf_diffusion.MeanFlowODE',
        'prefix': 'mix_uav_mf/',

        # ── Gen3v6 objective knobs ───────────────────────────────────────────────────────────
        'mf_objective': 'meanflow',        # only implemented value; folder-name slot for arms
        'meanflow_data_proportion': 0.5,   # fraction of the batch forced to r==t (FM anchors)
        'mf_adp_p': 1.0,                   # official adaptive-loss exponent
        'mf_adp_eps': 0.01,                # official adaptive-loss epsilon (≠ af_adp_eps!)
    },

    'plan_mix_uav_mf': {
        **_UAV_PLAN,
        'engine': 'mf',
        'diffusion': 'models.mf_diffusion.MeanFlowODE',
        'prefix': 'plans/mix_uav_mf/',
        'diffusion_loadpath': 'f:mix_uav_mf/H{horizon}_D{diffusion}',
        # ⚠️ MUST match the training block token-for-token: `_uav_mix_exp_name` reads these to
        # rebuild the SAME savepath the checkpoint was written to.
        'meanflow_data_proportion': 0.5,
        'imf_backbone': 'unet',
    },

    # ══ arm: af — Gen3v7 alpha-Flow ══════════════════════════════════════════════════════════
    'mix_uav_af': {
        **_UAV_TASK,
        **_TWO_TIME_BACKBONE,
        # ── Gen15 U6 (2026-09-03) ── THE DEFAULT IS NOW 'unet', RESOLVED FROM UAV_MIX_BONE_AF.
        #
        # It used to be a hard 'sit'. 'sit' is alpha-Flow's OWN backbone (af_sit_trajectory.py,
        # Gen3v7 U2) and it sizes from dit_hidden_size/dit_depth, NOT from freq_dim — so at
        # 256/8 it is ~9.4 M against the fm/mf arms' 4.0 M U-Net. Every af-vs-mf row Gen15 has
        # published therefore moves OBJECTIVE, BACKBONE and PARAMETER COUNT together, which is
        # the one thing `architecture-matched-beat-is-the-strong-claim` says must never happen
        # in a headline. _TWO_TIME_BACKBONE's own comment already says this generation is
        # "LOCKED to 'unet' for the headline comparison"; the af arm was the exception.
        #
        # 'sit' is KEPT, not deleted — it is the deferred appendix arm (PLAN §6) and it is one
        # env var away:  UAV_MIX_BONE_AF=sit
        #
        # ✅ 'bb' is an UNCONDITIONAL exp_name token, so 'unet' and 'sit' train into separate
        #    trees and neither can overwrite the other (gate G2 asserts the cross-product).
        # 🔴 The flip DOES move the default path: `_bbsit` -> `_bbunet`. Existing SiT
        #    checkpoints survive untouched but are no longer what a bare `af` run resolves to,
        #    so a default af EVAL fails on a missing checkpoint until the U-Net arm is trained.
        # ⚠️ plan_mix_uav_af below MUST carry the same value or eval rebuilds a different savepath.
        'imf_backbone': _AF_BONE,
        'engine': 'af',
        'model': 'models.af_engine.AlphaFlowEngine',
        'diffusion': 'models.af_diffusion.AlphaFlowODE',
        'prefix': 'mix_uav_af/',

        # ── Gen3v7 alpha-schedule knobs ──────────────────────────────────────────────────────
        'af_alpha_scheduler': 'sigmoid',
        'af_alpha_init': 1.0,        # alpha at step 0   (1.0 ⇒ start as pure flow matching)
        # ── Gen15 U6 ── UAV_MIX_AF_ALPHA_END. 0.0 (the default, and every upstream recipe)
        # anneals alpha to EXACTLY zero, which routes the whole tail into af_diffusion.py:568 —
        # Gen3v6's MeanFlow JVP body, unmodified. A run that ends at 0.0 DEPLOYS A MEANFLOW
        # MODEL. That is upstream's design (alpha-Flow is a curriculum whose destination IS
        # MeanFlow), but it means the bootstrapped objective is untested on this task.
        #   UAV_MIX_AF_ALPHA_END=0.2 -> alpha floors at 0.2, the discrete branch stays live to
        #                               the last step, and the deployed model is genuinely AF.
        # 🔴 PATH SAFETY: 'ae' is an UNCONDITIONAL exp_name token, so every value already lands
        #    in its own '_ae<val>' tree. Keep this in sync with the PLAN block or eval finds no
        #    weights. Verify with `train/discrete_frac` > 0 at the final epochs — if it is 0.0
        #    the run was MeanFlow whatever the folder says.
        'af_alpha_end': _AF_ALPHA_END,
        'af_alpha_init_step': 0,
        # 🔴 MUST equal n_train_steps — AlphaFlowODE.__init__ hard-asserts it. Upstream's
        # 400000 would hold alpha≈1 for our whole run, i.e. train plain flow matching under an
        # alpha-Flow folder name (Gen3v7 PLAN §11 trap 1). Keep in sync with _UAV_TASK above.
        'af_alpha_end_step': 100000,
        'af_alpha_gamma': 25.0,      # sigmoid sharpness
        'af_alpha_clamp': 0.005,     # snap to exactly 0.0 / 1.0 near the ends
        'af_ratio_fm': 0.5,          # fraction of the batch forced to r==t (h=0)
        'af_clamp_utgt': 4.0,        # upstream clamp_utgt
        # ⚠️ DELIBERATELY ≠ MeanFlow's 0.01 — different method, different constant.
        # Do NOT harmonise (Gen3v7 PLAN §11 trap 7).
        'af_adp_eps': 1e-3,
    },

    'plan_mix_uav_af': {
        **_UAV_PLAN,
        'engine': 'af',
        'diffusion': 'models.af_diffusion.AlphaFlowODE',
        'prefix': 'plans/mix_uav_af/',
        'diffusion_loadpath': 'f:mix_uav_af/H{horizon}_D{diffusion}',
        # ⚠️ MUST match the training block (see the mf plan block's note). Both of these are
        # env-resolved as of U6 and read the SAME module-level value the train block does, so
        # a plan/train mismatch is now unrepresentable rather than merely warned about.
        'af_alpha_init': 1.0,
        'af_alpha_end': _AF_ALPHA_END,
        'imf_backbone': _AF_BONE,
    },
}
