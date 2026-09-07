"""
Gen16 — VISUAL-MIX-ML for AVOIDING: one frame, four config-activated ML engines.

Model folder: mix_visual_avoiding/      Test folder: mix_visual_avoiding_test/
Plan:         logs_in_develop/Gen16/init/PLAN_Gen16_visual_avoiding_mix_ml.md

    engine=diffusion  <- Gen6V4  visual DDPM       VisualGaussianDiffusion
    engine=fm         <- Gen7    visual FM ODE     VisualFlowMatching      [reference arm]
    engine=mf         <- Gen3v6  MeanFlow          VisualMeanFlow
    engine=af         <- Gen3v7  alpha-Flow        VisualAlphaFlow

Task/data/dims are FROZEN across all four arms — 6-D single-camera visual avoiding
(action_dim=2, obs_dim=4, horizon=8), declared once in
`mix_visual_avoiding/models/visual_spec.py` — and the backbone defaults to the VisualUNet
stack, so the four-way comparison is architecture-controlled: only objective + sampler vary.

Trajectory layout:
    x[t] = [ dx  dy | des_x des_y | c_x  c_y ]
             act(2)   des_xy(2)     c_xy(2)
Single camera: bp-cam only (avoiding has no grasping, so the in-hand cam sees nothing).

────────────────────────────────────────────────────────────────────────────────────────
🔴 THIS MODULE IS A COPY, NOT AN IMPORT.

The helper machinery below (`_mix_loadpath`, `_mix_train_block`, `_mix_plan_block`,
`_film_mode`, `_ml_bone`, `_mix_bone_keys`) is copy-modified from Gen14's block in
`config/aligning-d3il-visual.py`, following the same rule Gen15 used for `config/uav_mix.py`:
copy, do not import. Consequences, both intended:

  * `config/avoiding-d3il-visual.py` (Gen9) and `config/aligning-d3il-visual.py` (Gen14)
    are BYTE-UNTOUCHED. Every existing checkpoint and results path in those generations
    resolves exactly as before.
  * A change to Gen14's helpers does NOT propagate here. That is the point — Gen16 must not
    move because someone re-tuned the aligning task.

⚠️ Gen16 = DPCC math (arms A/B) + the HardFlow SAMPLER as arm C. Gen13 (HF_Mix_ML) is built
   ON HardFlow and is a different mechanism. NEVER pool their results.
────────────────────────────────────────────────────────────────────────────────────────
"""

import os

import yaml

from diffuser.utils import watch

# ── the eval yaml this generation reads ───────────────────────────────────────────────
# 🔴 FIX_9_CFG_PROVENANCE (Gen3v6) — the eval publishes the yaml it ACTUALLY opened via
# FMPCC_PROJ_CFG before importing this module, so a `--config` override or a threshold sweep
# reaches the results-folder tokens. Falling back to the default is correct for a TRAINING
# run, which imports this module with no eval in sight.
_CFG_PATH = os.environ.get('FMPCC_PROJ_CFG') or 'config/visual_avoiding_mix_eval.yaml'
with open(_CFG_PATH, 'r') as _f:
    _proj_config = yaml.safe_load(_f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError(
        f"CRITICAL: 'diffusion_timestep_threshold' MUST be defined in {_CFG_PATH}")

# The eval republishes the RESOLVED threshold (env override applied) before importing this
# module, so the folder name is built from what actually runs — not from the yaml default the
# run may have overridden. DPCC_THRESHOLD is arm B's knob; HFFM_ACT_THRESHOLD is arm C's, and
# they are SEPARATE by design (a threshold sweep needs them independent).
_yaml_threshold = float(os.environ.get(
    'FMPCC_DPCC_THRESHOLD', _proj_config['diffusion_timestep_threshold']))
_hf_act_threshold = float(os.environ.get('HFFM_ACT_THRESHOLD', 1.0))
_hf_batch_size    = int(os.environ.get('HFFM_BATCH', 4))

logbase = 'logs'

# Single source of truth for the training budget. 🔴 The alpha-Flow anneal MUST span the
# ACTUAL budget: af_alpha_end_step and af_n_train_steps are BOTH derived from this one name,
# and af_diffusion.py asserts they agree. Never write the number twice.
_MIX_FULL_N_TRAIN_STEPS = int(1e5)
_MIX_N_TRAIN_STEPS = int(float(os.environ.get('MIX_TRAIN_STEPS', _MIX_FULL_N_TRAIN_STEPS)))

# 🔴 A SHORTENED RUN MUST NOT LOOK LIKE A FULL ONE. Budget is an identity key: a 50k-step
# checkpoint and a 100k-step checkpoint are different models, and the whole point of the
# generation-sibling layout is that two different models never share a directory. So when
# MIX_TRAIN_STEPS cuts the budget, `train_budget` joins args_to_watch_mix_visual_train and
# the checkpoint folder gains a trailing '_TB50pct'.
#
# At the FULL budget the key is absent entirely (watch() skips keys the block does not
# define), so every path that exists today is byte-identical to what it was. Nothing that
# already trained needs re-pathing.
#
# Because the tag is derived from the shared watch list, the eval's diffusion_loadpath picks
# it up automatically via _mix_loadpath -- but ONLY if the eval job sees the same
# MIX_TRAIN_STEPS. The pipeline exports it explicitly for exactly that reason.
def _budget_tag():
    if _MIX_N_TRAIN_STEPS >= _MIX_FULL_N_TRAIN_STEPS:
        return None
    pct = 100.0 * _MIX_N_TRAIN_STEPS / _MIX_FULL_N_TRAIN_STEPS
    # Integer percents stay clean ('50pct'); anything else falls back to the raw step count
    # so an odd budget is still unambiguous rather than silently rounded into a collision.
    return f'{int(round(pct))}pct' if abs(pct - round(pct)) < 1e-9 else f'{_MIX_N_TRAIN_STEPS}steps'


_MIX_BUDGET_TAG = _budget_tag()

# The NFE budget the two-time arms evaluate at by default. Two-time models are the whole
# reason low-K is interesting (one u-head query spans an interval), and the Gen3v6/v7
# state-only lineage settled on K=2. Override per run with `--flow-steps N`.
_TWO_TIME_K = int(os.environ.get('MIX_FLOW_STEPS', 2))
# The single-time arms' default K. 20 matches the Gen6V4/Gen7 artefacts everyone compares
# against (their checkpoint and results folders are both `H8_K20_...`).
_SINGLE_TIME_K = 20


# ══════════════════════════════════════════════════════════════════════════════════════
# CUSTOM RUN MESSAGE — opt-in results-path tag, EVAL ONLY
# ══════════════════════════════════════════════════════════════════════════════════════
# Lets a re-run of the SAME config at a different budget write to its OWN results folder
# instead of overwriting the old numbers. Empty (the default) => paths are unchanged.
#   FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/<eval>.sh
# 🔴 PLAN BLOCKS ONLY — never let this reach a training exp_name; it would move every
#    checkpoint folder and break every diffusion_loadpath.

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
    print(f'[ config/avoiding-d3il-visual-mix ] custom_msg="{custom_msg}" '
          f'-> results dirs end in "_msg{custom_msg}"')


def _msg_suffix(args):
    """'_msg<token>' for a plan block carrying a non-empty custom_msg, else ''."""
    msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
    return f'_msg{msg}' if msg else ''


def watch_plan(args_to_watch_list):
    """watch(), plus the plan block's custom_msg suffix. Use in PLAN blocks ONLY."""
    _fn = watch(args_to_watch_list)
    return lambda args: _fn(args) + _msg_suffix(args)


# ══════════════════════════════════════════════════════════════════════════════════════
# The 6 fixed obstacles
# ══════════════════════════════════════════════════════════════════════════════════════
# Positions sourced verbatim from
# d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py
# (`get_obj_xy_list()` — the SAME list on every env reset, which is exactly why they are
# constraints here and not observation dims; see visual_spec.py's header). z=0 because the
# avoiding plane is 2-D; the sphere check runs on the c_xy slice (indices 4-5 of the 6-D
# trajectory). Radius 0.04 m = obstacle geom radius 0.025 m + 0.015 m safety margin.
_AVOIDING_OBSTACLES = [
    ('sphere_outside', [0.500, -0.10, 0.0], 0.04),
    ('sphere_outside', [0.425,  0.08, 0.0], 0.04),
    ('sphere_outside', [0.575,  0.08, 0.0], 0.04),
    ('sphere_outside', [0.350,  0.26, 0.0], 0.04),
    ('sphere_outside', [0.500,  0.26, 0.0], 0.04),
    ('sphere_outside', [0.650,  0.26, 0.0], 0.04),
]


# ══════════════════════════════════════════════════════════════════════════════════════
# watch lists
# ══════════════════════════════════════════════════════════════════════════════════════
# One list for all four arms. watch() skips keys a block does not define, so each arm's
# folder name carries exactly its own identity keys (e.g. 'K' only for diffusion, 'ts'/'afsch'
# only for the two-time arms).

args_to_watch_mix_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),      # diffusion only — fm/mf/af blocks do not define it
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),
    ('film_mode', 'film'),
    # 🔴 The ML-BONE key MUST be in this list: film_mode is a path key and the bone was not,
    # so a DiT run would overwrite the U-Net checkpoint in the same directory. Blocks that do
    # not define it are skipped by watch(), so U-Net paths carry no '_B..' fragment.
    ('ml_bone', 'B'),
    ('engine', 'E'),                 # the arm identity key
    ('t_schedule', 'ts'),            # mf/af only
    ('af_alpha_scheduler', 'afsch'), # af only
    # Appended LAST so a reduced-budget run reads as the full name plus a suffix, and so the
    # full-budget name is unchanged. Set only when the budget is cut (see _budget_tag).
    ('train_budget', 'TB'),
]

args_to_watch_mix_visual_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),      # diffusion only
    ('flow_steps_v3', 'K'),          # fm/mf/af only (mutually exclusive per arm)
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('hf_act_threshold', 'A'),       # arm C's gate — separate knob from arm B's T
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('mpc_batch_size', 'mpc'),
    ('film_mode', 'film'),
    ('ml_bone', 'B'),
    ('engine', 'E'),
]

# A few training identity keys live under a DIFFERENT name on the planning side, to keep the
# eval script's namespace unambiguous (Gen7 does the same rename by hand: `batch_size` ->
# `train_batch_size`, which also fixes the b{mpc} vs b{beta} clash). The VALUES must be equal;
# only the arg name differs. Extend this map, never the call sites.
_MIX_TRAIN_TO_PLAN_KEY = {
    'batch_size': 'train_batch_size',
}


def _mix_loadpath(watch_list, block, prefix, key_map=None):
    """Build a loadpath/prefix format string from the SAME watch list that builds exp_name.

    🔴 This is the fix for the oldest trap in this repo: a plan block whose
    diffusion_loadpath does not reproduce args_to_watch key-for-key resolves to a
    non-existent directory and the eval dies minutes into a GPU allocation. Deriving the
    string here makes that class of bug unrepresentable — there is only one list.

    Mirrors diffuser.utils.watch(): keys absent from `block` are skipped, entries joined with
    '_' after the trailing-slash prefix. `block` is always the TRAINING block (so fragments
    and order match what watch() emitted at train time); `key_map` renames the emitted
    {placeholder} to the name the CONSUMING block exposes.
    """
    key_map = key_map or {}
    parts = []
    for key, label in watch_list:
        if key == 'prefix' or key not in block:
            continue
        parts.append(f'{label}{{{key_map.get(key, key)}}}')
    return 'f:' + prefix + '_'.join(parts)


# ══════════════════════════════════════════════════════════════════════════════════════
# Parent blocks — the per-arm hyperparameters Gen16 inherits
# ══════════════════════════════════════════════════════════════════════════════════════
# Gen14 read these out of `base['visual_aligning_dpcc']` / `base['fm_visual_aligning']`,
# which live in the same module there. Gen16's equivalents are Gen9's blocks in
# config/avoiding-d3il-visual.py — a DIFFERENT module, which this one deliberately does not
# import (see the header). They are therefore reproduced here, values unchanged, with the
# `model` / `diffusion` class paths left out because every arm sets its own.
#
# 🔴 The DDPM arm's parent is the DPCC block, NOT the FM one: it must inherit Gen6V4's own
# hyperparameters (action_weight=10), or it is not the baseline it claims to be.

_PARENT_DPCC = {
    'action_dim':       2,           # 2-D plane velocity [dx, dy]  — visual_spec.ACTION_DIM
    'obs_dim':          4,           # [des_xy(2), c_xy(2)]         — visual_spec.OBS_DIM
    'if_vision':        True,
    'horizon':          8,
    'n_diffusion_steps': 100,
    'action_weight':    10,
    'loss_type':        'l2',
    'dim':              32,
    'dim_mults':        (1, 2, 4, 8),
    'condition_dropout': 0.1,
    'returns_condition': False,
    'max_path_length':  200,         # avoiding episodes max ~106 steps; 200 is a safe ceiling
    'logbase':          logbase,
    'batch_size':       64,
    'learning_rate':    2e-4,
    'ema_decay':        0.995,
    'n_steps_per_epoch': 1000,
    'n_train_steps':    _MIX_N_TRAIN_STEPS,
    'gradient_accumulate_every': 2,
    'train_test_split': 0.9,
    'device':           'cuda',
    'seed':             0,
}

_PARENT_FM = {
    **_PARENT_DPCC,
    'action_weight':      1,      # Gen7's value, not Gen6V4's
    'time_beta_alpha_v3': 1.5,
    'time_beta_beta_v3':  1.0,
    'film_mode':          'v1',
}
# n_diffusion_steps is a DDPM concept. Leaving it on the FM parent would put a meaningless
# 'K{n}' in every fm/mf/af CHECKPOINT path (it is in args_to_watch_mix_visual_train).
_PARENT_FM.pop('n_diffusion_steps')


# ── shared eval/planning knobs, identical for all four arms ───────────────────────────
# The ROLLOUT must be byte-identical across arms: same env, same episode budget, same MPC
# candidate pool, same constraint YAML. Only the generative engine differs — that is the
# whole point of the generation.
_mix_plan_common = {
    'horizon':          8,
    'max_episode_length': 200,
    'max_path_length':  200,
    'if_vision':        True,
    # 🔴 ONE candidate-fan number, two names. `mpc_batch_size` is the results-path token
    # (Gen9/Gen7 convention); `batch_size` is what the eval driver reads for the DPCC-arm fan
    # (state-only avoiding lineage convention). `_mix_plan_block` derives the second from the
    # first so they cannot drift — a mismatch would mean the folder name advertises a fan the
    # run did not use, and arm-B-vs-arm-C timing comparisons would be void (B4_PARITY).
    'mpc_batch_size':   4,
    'preprocess_fns':   [],
    'device':           'cuda',
    'seed':             0,
    'loadbase':         None,
    'logbase':          logbase,
    'custom_msg':       custom_msg,   # '' => path unchanged; else '..._msg<value>'
    'returns_condition': False,
    'predict_epsilon':  True,
    'diffusion_timestep_threshold': _yaml_threshold,
    'hf_act_threshold': _hf_act_threshold,
    'diffusion_epoch':  'best',
    # Deploy the EMA weights. 🔴 Gen16's trainers select `state_best` on the EMA loss
    # (mix_visual_avoiding/utils/training.py::test), so flipping this to False WITHOUT
    # reverting that selection re-creates the Gen9 U4 Fix1 mismatch: the deployed weights
    # would not be the ones the checkpoint was chosen for.
    'eval_use_ema':     True,
    'verbose':          False,
    'suffix':           '0',
    # 6 fixed obstacles -> sphere_outside projector constraints.
    'constraint_list':  list(_AVOIDING_OBSTACLES),
    # ODE-solver knobs (dropped on the diffusion arm, which has no ODE).
    'ode_solver_backend_v3':   'legacy_euler',
    'ode_solver_method_v3':    'euler',
    'ode_solver_rtol_v3':      None,
    'ode_solver_atol_v3':      None,
    'ode_solver_step_size_v3': None,
    'clip_denoised':    False,
}

# Sentinel meaning "REMOVE this inherited key from the block". An override dict can add or
# replace, but the mf/af arms inherit `film_mode` from `_PARENT_FM`, and on a DiT bone the key
# must be GONE, not merely unset: watch() skips keys the args object lacks, so deleting it is
# what keeps '_film..' out of a transformer checkpoint path.
_DROP = object()


def _mix_train_block(engine, parent, overrides):
    """Assemble one training block: parent hyperparameters + Gen16 identity + arm overrides.

    Keys whose override value is `_DROP` are removed from the merged block entirely.
    """
    blk = {**parent, **overrides, 'engine': engine,
           'prefix': f'mix_visual_avoiding_{engine}/'}
    for k in [k for k, v in blk.items() if v is _DROP]:
        del blk[k]
    # Present ONLY on a reduced budget -- absent means "full 1e5", which is how every
    # existing path keeps its current name.
    if _MIX_BUDGET_TAG is not None:
        blk['train_budget'] = _MIX_BUDGET_TAG
    blk['exp_name'] = watch(args_to_watch_mix_visual_train)
    return blk


def _mix_plan_block(engine, train_blk, overrides, drop=()):
    """Assemble one planning block, deriving every path string from the watch lists.

    Mirrors Gen7's two-level scheme exactly:
      prefix   -> the CHECKPOINT identity (training keys)  = which model produced this
      exp_name -> the EVAL identity       (plan keys)      = how it was rolled out
    so results for one checkpoint under different K / solver / mpc land in sibling dirs.

    `drop` removes keys inherited from the shared template that do not apply to this arm
    (e.g. the ODE-solver keys on the DDPM arm), so they reach neither the folder name nor
    the engine constructor.
    """
    blk = {**_mix_plan_common, **overrides, 'engine': engine,
           'diffusion': train_blk['diffusion']}
    for k in drop:
        blk.pop(k, None)

    # Mirror every TRAINING identity value into the plan block under its planning-side name,
    # so the derived {placeholders} below always resolve. Values are copied from the training
    # block itself — a plan/train value mismatch is therefore impossible, which is the whole
    # failure mode Gen7's "MUST match exactly" comments warn about.
    for key, _label in args_to_watch_mix_visual_train:
        if key == 'prefix' or key not in train_blk:
            continue
        plan_key = _MIX_TRAIN_TO_PLAN_KEY.get(key, key)
        # Unconditional: for an IDENTITY key the training value is the only correct one.
        # Overriding it in a plan block is exactly the mistake this loop prevents.
        blk[plan_key] = train_blk[key]

    # 🔴 An identity key the TRAINING block does not have must not survive on the plan block.
    # `film_mode` arrives via _mix_plan_common's parents, so on a DiT bone it would otherwise
    # label the RESULTS folder '_filmv1_' for a model with no FiLM path at all — the eval-side
    # twin of the checkpoint-path lie. The mirror loop above only ADDS keys; this removes.
    for _identity_key, _ in args_to_watch_mix_visual_train:
        if _identity_key == 'prefix':
            continue
        if _identity_key not in train_blk:
            blk.pop(_MIX_TRAIN_TO_PLAN_KEY.get(_identity_key, _identity_key), None)

    # ONE fan, two names — see the note on `mpc_batch_size` in _mix_plan_common.
    blk['batch_size'] = blk['mpc_batch_size']

    # Training-key fragments -> the CHECKPOINT identity, re-pointed at this arm's plan
    # namespace. `[2:]` strips the 'f:' marker; the whole prefix carries one of its own.
    _ckpt_id = _mix_loadpath(
        args_to_watch_mix_visual_train, train_blk, '', _MIX_TRAIN_TO_PLAN_KEY)[2:]
    blk['prefix']   = f'f:plans/mix_visual_avoiding_{engine}/{_ckpt_id}/'
    blk['exp_name'] = watch_plan(args_to_watch_mix_visual_plan)

    # diffusion_loadpath must reproduce the TRAINING block's exp_name exactly, key for key.
    blk['diffusion_loadpath'] = _mix_loadpath(
        args_to_watch_mix_visual_train, train_blk,
        f'mix_visual_avoiding_{engine}/', _MIX_TRAIN_TO_PLAN_KEY)
    return blk


# ══════════════════════════════════════════════════════════════════════════════════════
# FiLM conditioning backbone — the `film_mode` knob, all four arms
# ══════════════════════════════════════════════════════════════════════════════════════
# Each arm has its OWN knob, resolved by _film_mode('<arm>'). DEFAULT 'v1' everywhere, which
# is what the parent blocks already supply — so with no env var set, nothing changes.
#
#     MIX_FILM_MODE_DIFFUSION   MIX_FILM_MODE_FM   MIX_FILM_MODE_MF   MIX_FILM_MODE_AF
#
# and a bare MIX_FILM_MODE as the all-arms fallback. Per-arm is the primary form because the
# arms are separate experiments with separate checkpoint trees: putting mf on v2 must not drag
# fm — the reference arm — along with it.
#
# ⚠️ Before reading a v1-vs-v2 curve: `W_f` is zero-initialised and under v2 the visual latent
#    reaches the network through `W_f` and nowhere else, so at step 0 a v2 model is exactly
#    v1-with-no-vision. Early-epoch curves are not comparable step-for-step.
_MIX_FILM_MODES = ('v1', 'v2')


def _film_mode(engine):
    """Resolve ONE arm's FiLM mode. Precedence, most specific first:

        1. MIX_FILM_MODE_<ENGINE>   e.g. MIX_FILM_MODE_MF=v2   — this arm only
        2. MIX_FILM_MODE            — every arm with no specific setting
        3. 'v1'                     — the default, identical to the inherited parent value

    Unknown values RAISE. A silent fallback to v1 would train the wrong architecture into a
    directory whose name claims otherwise — the one failure the path key exists to prevent.
    """
    for key in (f'MIX_FILM_MODE_{engine.upper()}', 'MIX_FILM_MODE'):
        val = os.environ.get(key)
        if val:
            if val not in _MIX_FILM_MODES:
                raise ValueError(
                    f"CRITICAL: {key}='{val}' is not a known FiLM mode "
                    f"(want one of {list(_MIX_FILM_MODES)}).")
            return val
    return 'v1'


# ══════════════════════════════════════════════════════════════════════════════════════
# The ML-BONE knob (generative backbone for the two-time arms)
# ══════════════════════════════════════════════════════════════════════════════════════
#   'unet'    VisualUNetTwoTime  — the Gen16 BASELINE. FiLM conditioning (film_mode v1/v2).
#   'mf_dit'  official MeanFlow DiT (adaLN-zero trunk)          — mf arm only
#   'sit'     alpha-Flow SiT      (adaLN-zero trunk)            — af arm only
#   'dit'     iMF DiT             (RoPE, in-context tokens)     — both arms
#
# On every DiT/SiT bone the visual latent enters as ONE PREPENDED TOKEN, never as adaLN
# modulation — that design point is already occupied by the U-Net's FiLM.
#
# 🔴 START ON `unet`. The architecture-matched claim is the strong one: the DPCC baseline is a
#    U-Net, so only the U-Net row is an unconfounded comparison. A DiT/SiT win is a secondary,
#    confounded result and must be reported as such, with backbone + parameter count carried
#    in every table.
#
# ⚠️ film_mode is a U-NET concept. On a DiT bone it is not merely unused — it would put a
#    lying '_filmv1_' fragment in the checkpoint path — so `_mix_bone_keys()` DELETES the key
#    for non-unet bones and watch() then skips it. Never re-add it by hand.
_MIX_ML_BONES = {
    'mf': ('unet', 'mf_dit', 'dit'),
    'af': ('unet', 'sit', 'dit'),
}


def _ml_bone(engine):
    """Resolve ONE arm's generative bone. Same precedence shape as _film_mode().

        1. MIX_BONE_<ENGINE>   e.g. MIX_BONE_MF=mf_dit   — this arm only
        2. MIX_BONE            — every two-time arm with no specific setting
        3. 'unet'              — the default and the baseline

    Unknown values RAISE, and so does a bone belonging to the OTHER arm ('sit' on mf,
    'mf_dit' on af) — those are separate classes with separate provenance, and silently
    accepting one would train a model whose folder names a different architecture.
    """
    allowed = _MIX_ML_BONES[engine]
    for key in (f'MIX_BONE_{engine.upper()}', 'MIX_BONE'):
        val = os.environ.get(key)
        if val:
            if val not in allowed:
                raise ValueError(
                    f"CRITICAL: {key}='{val}' is not a valid ML bone for the '{engine}' arm "
                    f"(want one of {list(allowed)}).")
            return val
    return 'unet'


def _mix_bone_keys(engine):
    """The bone-dependent block fragment: EITHER film_mode OR ml_bone + the DiT sizing.

    Exactly one conditioning-config family is present at a time, so a block can never carry
    both a FiLM mode and a DiT width.
    """
    bone = _ml_bone(engine)
    if bone == 'unet':
        # 🔴 ml_bone is DELIBERATELY ABSENT on the baseline bone. watch() skips keys a block
        # does not define, so the U-Net's exp_name / diffusion_loadpath carry no '_B..'
        # fragment and the DiT blocks (which DO define it) can never collide with them.
        return {'film_mode': _film_mode(engine)}
    # 🔴 dit_hidden_size=160 (not the state-only 256) is the PARAMETER-MATCHED width:
    # 18*depth*d^2 => 160/8 ~ 3.9 M vs the visual U-Net's ~4.0 M (dim=32). 256/8 is ~9.9 M,
    # i.e. 2.5x — and an unmatched backbone A/B is the defect that already forced one public
    # retraction in this project. Both are ARCHITECTURE keys: changing either needs a retrain.
    return {
        'ml_bone':         bone,
        'film_mode':       _DROP,   # inherited from _PARENT_FM — must be DELETED on a DiT
        'dit_hidden_size': 160,
        'dit_depth':       8,
        'dit_num_heads':   4,
        'dit_patch_size':  1,
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# The blocks
# ══════════════════════════════════════════════════════════════════════════════════════

base = {}

_P = 'mix_visual_avoiding.models.'

# ─── arm: diffusion (Gen6V4 DDPM) ──────────────────────────────────────────────────────
base['mix_visual_avoiding_diffusion'] = _mix_train_block('diffusion', _PARENT_DPCC, {
    'model':     _P + 'visual_unet.VisualUNet',
    'diffusion': _P + 'visual_gaussian_diffusion.VisualGaussianDiffusion',
    # 🔴 UNLIKE fm/mf/af, this is NOT an inference-only knob. n_diffusion_steps is the DDPM
    # chain length: it sets the TRAINING noise schedule AND is a checkpoint-path key. Changing
    # it REQUIRES A RETRAIN — the eval's --flow-steps override explicitly refuses this arm.
    # The plan block picks it up automatically via _mix_plan_block's mirror loop.
    'n_diffusion_steps': _SINGLE_TIME_K,
    'film_mode': _film_mode('diffusion'),
})

# ─── arm: fm (Gen7 Flow Matching) — THE REFERENCE ARM ──────────────────────────────────
base['mix_visual_avoiding_fm'] = _mix_train_block('fm', _PARENT_FM, {
    'model':     _P + 'visual_unet.VisualUNet',
    'diffusion': _P + 'visual_fm_diffusion.VisualFlowMatching',
    # 🔴 Running this arm at v2 makes it a NEW arm, not the Gen7-lineage reference. Having a
    # per-arm knob is what keeps a v2 sweep of mf/af from silently dragging it along.
    'film_mode': _film_mode('fm'),
})

# ─── arm: mf (Gen3v6 MeanFlow) ─────────────────────────────────────────────────────────
base['mix_visual_avoiding_mf'] = _mix_train_block('mf', _PARENT_FM, {
    'model':     _P + 'mf_engine.MeanFlowEngine',
    'diffusion': _P + 'visual_mf_diffusion.VisualMeanFlow',
    # Time schedule: 'logit_normal' is the official MeanFlow default and Gen3v6's.
    # 🔴 The sign convention is -p_mean (mf_diffusion.py). Using +p_mean puts the mass near
    # NOISE and looks *almost* fine. Do not "fix" it.
    't_schedule': 'logit_normal',
    'p_mean': -0.4,
    'p_std': 1.0,
    # Official MeanFlow objective constants (per-sample SUM, p=1, eps=0.01).
    'meanflow_data_proportion': 0.5,   # fraction of the batch forced to r==t (FM anchors)
    'mf_adp_p': 1.0,
    'mf_adp_eps': 0.01,
    # 🔴 Architecture flags from Gen3v6. dual_head=True: the v head SHARES the backbone trunk
    # and carries a full loss. False falls back to an orphan MLP on raw x and guts half the
    # objective. interval_cfg=False: no CFG in Gen3v6; on the UNet arm it changes the
    # state_dict, so flipping it makes checkpoints non-interchangeable.
    'dual_head': True,
    'interval_cfg': False,
    # Gen3v6/v7 trainer extras. 🔴 split_seed is INERT on this task: Gen16's dataset provides
    # `episode_split()`, so both trainers take the deterministic episode-level branch and no
    # RNG is consulted. Kept so the key means the same thing it does in Gen3v6.
    'gradient_clip': 1.0,
    'split_seed': 42,
    # Ablation knob. Default OFF: the vision encoder trains end-to-end, exactly as in
    # Gen6V4/Gen7. Pre-encoding the latent is what zeroes the JVP tangent — freezing is NOT
    # required for that, and turning this on changes what is learned.
    'mf_freeze_vision_encoder': False,
    # ⚠️ For an mf-vs-af comparison, move this arm and the af arm TOGETHER.
    **_mix_bone_keys('mf'),
})

# ─── arm: af (Gen3v7 alpha-Flow) ───────────────────────────────────────────────────────
base['mix_visual_avoiding_af'] = _mix_train_block('af', _PARENT_FM, {
    'model':     _P + 'af_engine.AlphaFlowEngine',
    'diffusion': _P + 'visual_af_diffusion.VisualAlphaFlow',
    't_schedule': 'logit_normal',
    'p_mean': -0.4,
    'p_std': 1.0,
    # alpha-Flow's OWN constants. ⚠️ af_adp_eps=1e-3 is DELIBERATELY != MeanFlow's 0.01
    # (af_diffusion.py forbids harmonising them) — different method, different constant.
    'af_ratio_fm': 0.5,
    'af_adp_eps': 1e-3,
    'af_clamp_utgt': 4.0,
    # 🔴 Same architecture flags as the mf arm — Gen3v7 ships them identically, and keeping
    # mf and af equal here is what makes the MeanFlow-vs-alpha-Flow comparison
    # architecture-controlled.
    'dual_head': True,
    'interval_cfg': False,
    # The alpha anneal: 1 -> 0 means training starts as plain flow matching and becomes
    # MeanFlow. 🔴 end_step is bound to _MIX_N_TRAIN_STEPS, the same name that sets
    # n_train_steps. The train script re-derives it and af_diffusion asserts on it.
    'af_alpha_scheduler': 'sigmoid',
    'af_alpha_init': 1.0,
    'af_alpha_end': 0.0,
    'af_alpha_init_step': 0,
    'af_alpha_end_step': _MIX_N_TRAIN_STEPS,
    'af_alpha_gamma': 25.0,
    'af_alpha_clamp': 0.005,
    'gradient_clip': 1.0,
    'split_seed': 42,
    'mf_freeze_vision_encoder': False,
    # ⚠️ The mf-vs-af comparison is architecture-controlled ONLY if both arms run the same
    # mode and the same bone. Independent knobs make that YOUR responsibility rather than the
    # config's: use the bare MIX_FILM_MODE / MIX_BONE to move the pair together.
    **_mix_bone_keys('af'),
})

# ─── planning / evaluation blocks (one per arm) ────────────────────────────────────────
# 🔴 `flow_steps` (arm C's HardFlow Euler K) is set EQUAL to `flow_steps_v3` (arms A/B's
# native sampler K) in every block. Matched budget or nothing: an arm-B-vs-arm-C table at
# different NFE is not a comparison. The eval's --flow-steps patches BOTH.

base['plan_mix_visual_avoiding_diffusion'] = _mix_plan_block(
    'diffusion', base['mix_visual_avoiding_diffusion'], {
        # DDPM's clamp: False lets VisualGaussianDiffusion run unclamped, compounding x_recon
        # errors across K denoising steps into exploded trajectories at eval (Gen9 U3-C1).
        'clip_denoised': True,
        # Arm C has no host on this arm (no velocity field); the eval skips those variants.
    },
    # DDPM has a real reverse chain, not an ODE: drop every continuous-time key so it reaches
    # neither the folder name nor the constructor. n_diffusion_steps (mirrored from the
    # training block) is this arm's K.
    drop=('flow_steps_v3', 'ode_solver_backend_v3', 'ode_solver_method_v3',
          'ode_solver_rtol_v3', 'ode_solver_atol_v3', 'ode_solver_step_size_v3',
          'time_beta_alpha_v3', 'time_beta_beta_v3', 'hf_act_threshold'))

base['plan_mix_visual_avoiding_fm'] = _mix_plan_block(
    'fm', base['mix_visual_avoiding_fm'], {
        # K=20 matches the Gen7 lineage's own eval folders. Inference-only and safe to set
        # here: flow_steps_v3 lives in args_to_watch_mix_visual_plan ONLY, so the mirror loop
        # cannot clobber it and diffusion_loadpath is unchanged — same checkpoint, new
        # H8_K20_... results folder. Sweep per run with `--flow-steps N`.
        # 🔴 K also sets the PROJECTION budget: the sampler projects from
        # int((1 - T) * K) to the end, so at T=0.5 K=20 -> 10 SLSQP solves per replan.
        'flow_steps_v3': _SINGLE_TIME_K,
        'flow_steps':    _SINGLE_TIME_K,
    })

base['plan_mix_visual_avoiding_mf'] = _mix_plan_block(
    'mf', base['mix_visual_avoiding_mf'], {
        # 🔴 K=2. A DDPM-parity K of 100 is WRONG for a two-time model: MeanFlow's entire
        # premise is that one u-head query spans a whole interval, and the Gen3v6/v7
        # state-only lineage evaluates at K=2. It also sets the projection budget, which is
        # the expensive half — at T=0.5, K=100 is 50 SLSQP solves per replan and K=2 is 1.
        'flow_steps_v3': _TWO_TIME_K,
        'flow_steps':    _TWO_TIME_K,
        # MUST match the training block — both are checkpoint-path keys.
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
    })

base['plan_mix_visual_avoiding_af'] = _mix_plan_block(
    'af', base['mix_visual_avoiding_af'], {
        # Kept EQUAL to mf's on purpose: NFE is an operating point, and an mf-vs-af
        # comparison at different K would confound the objective with the step budget.
        'flow_steps_v3': _TWO_TIME_K,
        'flow_steps':    _TWO_TIME_K,
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
        'af_alpha_scheduler': 'sigmoid',
    })
