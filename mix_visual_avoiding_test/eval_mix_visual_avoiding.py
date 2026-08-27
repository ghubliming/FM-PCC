# Visual-Mix-ML for AVOIDING (Gen16) — evaluation script.
#
# ONE rollout harness, FOUR engines, THREE guidance arms:
#
#   engines (--engine):  diffusion | fm | mf | af      (mix_visual_avoiding/models/engine_registry.py)
#   arms    (yaml `projection_variants`):
#       A  `diffuser`        unguided sampler, no constraint machinery
#       B  `dpcc-*`          DPCC Projector inside the sampler loop
#       C  `hardflow_new*`   HardFlow in-loop constrained sampler (fm/mf/af only)
#
# ─────────────────────────────────────────────────────────────────────────────────────
# LINEAGE — read this before editing.
#
# This file is NOT a copy of Gen14's aligning eval. Gen14 drives D3IL's `Aligning_Sim`
# through a ~700-line `VisualAgentWrapper`; the avoiding task is a plain gym loop and its
# whole lineage (Gen3v2 -> Gen3v6 -> Gen3v7 -> Gen12) drives it through a `Policy`.
#
# So the STRUCTURE here is `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` @ HEAD —
# the mature avoiding harness, with its K sweep, its receding-horizon cadence (U10), its
# run provenance (U10.1), its arm-C fan parity (B4_PARITY) and its RTRecorder — and the
# THREE things layered on top are:
#
#   1. `--engine` dispatch through `engine_registry`          (from Gen14)
#   2. visual policies instead of diffuser's `Policy`          (mix_visual_avoiding/sampling/policies.py)
#   3. the bp-cam frame captured per env step and fed to them  (from Gen9 Epoch 2)
#
# Everything else in the rollout loop is the avoiding lineage's, deliberately unchanged, so
# Gen16 numbers sit in the same frame as Gen3v6/Gen3v7's state-only numbers.
#
# ⚠️ Gen16 = DPCC math (arms A/B) + the HardFlow SAMPLER as arm C. Gen13 (HF_Mix_ML) is
#    built ON HardFlow and is a different mechanism. Never pool their results.
#
# Usage:
#   python -m mix_visual_avoiding_test.eval_mix_visual_avoiding --engine mf --seed 6
#   python -m mix_visual_avoiding_test.eval_mix_visual_avoiding --engine fm --flow-steps 2
#
# Output: logs/avoiding-d3il-visual-mix/plans/mix_visual_avoiding_<engine>/<exp>/results/<seed>/
import argparse
import os
import pickle
import sys
import time
from copy import copy

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import mix_visual_avoiding.utils as utils
from mix_visual_avoiding.models import visual_spec
# Gen16 — the arm dispatch table. Every engine-specific branch lives there, not here.
from mix_visual_avoiding.models.engine_registry import (
    ENGINE_INPUT_KEYS, canonical_engine, resolve, describe, nfe_of, assert_engine_matches,
)
from mix_visual_avoiding.sampling.projection import Projector
from mix_visual_avoiding.sampling.policies import (
    VisualNormalizer, VisualPolicy, VisualHardFlowPolicy,
)
# Arm C — HardFlow's `hardflow_new` constrained sampler (Gen14 U7 port).
from mix_visual_avoiding.sampling.hardflow_projection import (
    build_hardflow_sampler, resolve_activation_threshold, resolve_hf_batch_size,
    resolve_engine_hf,
    hardflow_step_budget,          # HFK1 (2026-08-24)
    # [SolverSwap] artifact naming — keeps an SLSQP run from overwriting IPOPT data.
    artifact_variant_label, resolve_nlp_backend,
)
from diffuser.utils import provenance   # U10.1 — env-override provenance (shared)
# REAL_TIME_RECORDING_UPDATE — per-step timing/digital-twin recorder
from realtime_recording.behavior_logger import RTRecorder

from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import (
    ObstacleAvoidanceEnv,
)

RT_CONTROL_HZ = 30   # assumed deployment loop rate; total_ms bundles encoder+ODE+projection


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ══════════════════════════════════════════════════════════════════════════════════════
# 1. CLI + the arm
# ══════════════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Gen16 visual-avoiding eval (4 engines, 3 arms).')
parser.add_argument('--seed', type=int, help='Run only this specific seed.')
parser.add_argument('--seeds', type=int, nargs='+',
                    help='Run this seed list (overrides the yaml, overridden by --seed).')
parser.add_argument('--aggregate-only', action='store_true',
                    help='Skip inference; only aggregate existing npz into all_seeds plots.')
# 🔵 MATCHED-K — K is a first-class CLI knob so a {1,2,5,10,20} grid is a loop in the sbatch
# rather than a config edit somebody has to remember. `flow_steps_v3` is a results-path token,
# so each K writes its OWN directory and no two budgets overwrite each other.
parser.add_argument('--flow-steps', type=int, default=None, metavar='K',
                    help='override the NFE budget K for this run (fm/mf/af only)')
# Gen16 — the arm selector. Default 'fm' == the Gen7-lineage reference arm.
parser.add_argument('--engine', type=str, default='fm', choices=list(ENGINE_INPUT_KEYS),
                    help='ML engine arm: diffusion | fm | mf | af')
parser.add_argument('--config', type=str, default=None,
                    help='eval yaml (default: config/visual_avoiding_mix_eval.yaml)')
args_cli, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

ENGINE      = canonical_engine(args_cli.engine)   # 'ddpm' -> 'diffusion', with a notice
ENGINE_SPEC = resolve(ENGINE)
EXPERIMENT  = f'plan_mix_visual_avoiding_{ENGINE}'
print(f'[ eval ] {describe(ENGINE)}')
print(f'[ eval ] config block: {EXPERIMENT}')
print(f'[ eval ] task spec: {visual_spec.LAYOUT}')

# ══════════════════════════════════════════════════════════════════════════════════════
# 2. YAML + the knobs that must be resolved BEFORE the config module is imported
# ══════════════════════════════════════════════════════════════════════════════════════
# 🔴 FIX_9_CFG_PROVENANCE (Gen3v6) — the resolved values are published into the environment
# HERE, before the first Parser().parse_args(), because that call is what imports
# config/<exp>.py and the config module builds the results-folder tokens from these. Doing
# it later produces a run whose path claims a threshold the Projector never used.

_cfg_path = args_cli.config or 'config/visual_avoiding_mix_eval.yaml'
with open(_cfg_path, 'r') as f:
    config = yaml.safe_load(f)
print(f'[ eval ] config yaml: {_cfg_path}')
os.environ['FMPCC_PROJ_CFG'] = _cfg_path

# Arm B threshold — settable per job the way the HFFM_* knobs already are.
diffusion_timestep_threshold = float(os.environ.get(
    'DPCC_THRESHOLD', config.get('diffusion_timestep_threshold', 0.5)))

# ── arm C (HardFlow) knobs, resolved once (verbatim Gen3v6/Gen12 semantics) ────────────
hardflow_cfg = config.get('hardflow', {})
hf_act_threshold = resolve_activation_threshold(
    os.environ.get('HFFM_ACT_THRESHOLD',
                   hardflow_cfg.get('activation_threshold',
                                    hardflow_cfg.get('activation', 1.0))))
# 🔴 B4_PARITY — the run-level arm-C fan defaults to the DPCC arms' `batch_size` (4), because
# both arms loop serially over candidates around a CPU solve and a mismatched fan makes the
# arm-B-vs-arm-C wall-clock comparison void. Bare `hardflow_new` is still pinned to 1 by
# resolve_hf_batch_size() as the faithful upstream batch-1 control.
hf_batch_size = int(os.environ.get('HFFM_BATCH', hardflow_cfg.get('batch_size', 4)))
hf_candidate_cost = hardflow_cfg.get('candidate_cost', 'prox')

os.environ['FMPCC_DPCC_THRESHOLD'] = '%g' % float(diffusion_timestep_threshold)
os.environ['HFFM_ACT_THRESHOLD']   = '%g' % float(hf_act_threshold)
os.environ['HFFM_BATCH']           = str(int(hf_batch_size))
print(f'[ eval ] resolved  dpcc_threshold={diffusion_timestep_threshold}  '
      f'hf_act_threshold={hf_act_threshold}  hf_batch={hf_batch_size}  '
      f'hf_candidate_cost={hf_candidate_cost}')

# ── receding-horizon cadence (U10) ────────────────────────────────────────────────────
# How many actions of each plan are executed before replanning. 1 == every env step gets a
# fresh plan, which is what every FM-PCC result to date used and remains the DEFAULT.
replan_steps = int(os.environ.get('MIX_REPLAN_STEPS', config.get('replan_steps', 1)))
if replan_steps < 1:
    raise ValueError(f'MIX_REPLAN_STEPS must be >= 1, got {replan_steps}')
# 🔴 PATH COLLISION GUARD — the cadence is NOT a results-folder token, so an r1 and an r8 run
# at the same K would clobber each other. A non-default cadence auto-tags itself through the
# existing custom-message slot. An explicit FMPCC_RUN_MSG always wins. Must be set BEFORE the
# first Parser().parse_args().
if replan_steps != 1 and not os.environ.get('FMPCC_RUN_MSG'):
    os.environ['FMPCC_RUN_MSG'] = f'r{replan_steps}'
    print(f'[ eval ] replan_steps={replan_steps} -> auto-tagged results path with '
          f'FMPCC_RUN_MSG=r{replan_steps} (set it yourself to override)')
print(f'[ eval ] replan_steps={replan_steps} '
      f'({"per-step replanning (historic default)" if replan_steps == 1 else "receding horizon"})')

exps  = config['exps']
seeds = config['seeds']
if args_cli.seeds is not None:
    seeds = [int(s) for s in args_cli.seeds]
    print(f'[ eval ] Overriding seeds from --seeds: {seeds}')
if args_cli.seed is not None:
    seeds = [args_cli.seed]
    print(f'[ eval ] Overriding seeds from --seed: {seeds}')

# ── K override ────────────────────────────────────────────────────────────────────────
# 🔴 The DDPM arm REFUSES it. `n_diffusion_steps` is the training chain length AND a
# checkpoint-path key, so changing it at eval does not change the operating point — it makes
# the loadpath resolve to a directory that does not exist, or worse, silently mismatches the
# noise schedule the weights were trained against. fm/mf/af's `flow_steps_v3` is genuinely
# inference-only, which is why only they accept the flag.
if args_cli.flow_steps is not None:
    if ENGINE == 'diffusion':
        raise SystemExit(
            f'\n[ eval ] 🔴 --flow-steps is not valid for the diffusion arm.\n'
            f'         `n_diffusion_steps` is the DDPM chain length: it sets the TRAINING\n'
            f'         noise schedule and is a checkpoint-path key. Changing K for this arm\n'
            f'         requires a retrain, not an eval flag. Use --engine fm|mf|af to sweep K.\n')
    # Patch the config MODULE's dict before any Parser reads it. utils.Parser.read_config does
    # `importlib.import_module(args.config)` and copies `base[experiment]` key by key, and
    # Python caches modules — so this is the intended data path, not a monkey-patch: exp_name,
    # savepath and the diffusion kwargs all follow automatically.
    import importlib
    _mod = importlib.import_module('config.avoiding-d3il-visual-mix')
    _blk = _mod.base[EXPERIMENT]
    _blk['flow_steps_v3'] = args_cli.flow_steps
    if 'ode_inference_steps_v3' in _blk:
        _blk['ode_inference_steps_v3'] = args_cli.flow_steps
    # MATCHED-K: `flow_steps` is arm C's Euler K. Patch it with the SAME value or
    # --flow-steps would move arms A/B only and every arm-B-vs-arm-C table would be
    # comparing different NFE budgets.
    _blk['flow_steps'] = args_cli.flow_steps
    print(f'[ eval ] Overriding flow_steps_v3 / flow_steps (K) to: {args_cli.flow_steps}')

projection_variants = config['projection_variants']
halfspace_variants  = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials            = config['n_trials']
plot_how_many       = config['plot_how_many']
constraint_types    = config['constraint_types']

_IMG_W, _IMG_H = visual_spec.IMG_SHAPE[2], visual_spec.IMG_SHAPE[1]


# ══════════════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint loading
# ══════════════════════════════════════════════════════════════════════════════════════

# CONFIG-OVERRIDES-PKL (fix_1, 2026-07-14): the pkl PRESERVES training-time params; the eval
# config is reconciled against it in TWO tiers:
#   - SAMPLING knobs (operating point, safe to change at eval): eval config OVERRIDES, [INFO].
#   - identity/architecture keys (must match the checkpoint): pkl value is KEPT to protect
#     the state_dict; a loud [WARNING] fires if the eval config disagrees.
_SAMPLING_OVERRIDE_KEYS = {
    'flow_steps_v3', 'ode_inference_steps_v3', 'ode_solver_backend_v3',
    'ode_solver_method_v3', 'ode_solver_rtol_v3', 'ode_solver_atol_v3',
    'ode_solver_step_size_v3', 'condition_guidance_w', 'clip_denoised',
    'diffusion_timestep_threshold',
}


def load_diffusion_with_override(*loadpath, target_class=None, epoch='best',
                                 device='cuda:0', override_args=None, engine=None):
    """Rebuild the checkpoint's model/engine/trainer and load the weights.

    Merged from Gen14's loader (the film/bone breadcrumbs and the engine assertion) and the
    avoiding lineage's (the two-tier config reconciler). The breadcrumbs matter here for the
    same reason they do in Gen14: the backbone is rebuilt from the TRAIN-time
    `model_config.pkl`, so architecture keys always come from the checkpoint — but the
    RESULTS FOLDER is named from the EVAL args, and a disagreement would label a directory
    with an architecture the weights do not have.
    """
    import inspect
    lp = os.path.join(*loadpath)
    print(f'\n[ eval loading ] {lp}\n')
    dataset_config   = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config     = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config   = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = lp

    # ── which ARM produced this checkpoint ────────────────────────────────────────────
    # Without this, loading arm A's weights into arm B surfaces as an opaque state_dict key
    # error minutes into a GPU allocation.
    _ckpt_engine = None
    _engine_holder = model_config._dict.get('vis_config', model_config._dict.get('config'))
    if _engine_holder is not None:
        _ckpt_engine = getattr(_engine_holder, 'engine', None)
    if engine is not None:
        assert_engine_matches(engine, _ckpt_engine)

    if target_class is not None:
        target_cls = utils.config.import_class(target_class)
        if (diffusion_config._class.__module__ + '.' + diffusion_config._class.__name__
                != target_cls.__module__ + '.' + target_cls.__name__):
            print(f'[ eval loading ] pickled diffusion class '
                  f'{diffusion_config._class.__module__}.{diffusion_config._class.__name__} '
                  f'!= config class {target_cls.__module__}.{target_cls.__name__}; '
                  f'using the config class.', file=sys.stderr)
            diffusion_config._class = target_cls
            valid = set(inspect.signature(target_cls.__init__).parameters)
            for k in [k for k in diffusion_config._dict if k not in valid]:
                print(f'[ eval loading ] dropping unexpected kwarg from pickle: {k!r}',
                      file=sys.stderr)
                del diffusion_config._dict[k]

    # ── ML BONE breadcrumb (Gen14 U8) ─────────────────────────────────────────────────
    _bone_pkl = model_config._dict.get('imf_backbone', 'unet') or 'unet'
    _is_dit = _bone_pkl != 'unet'
    print(f"[ eval loading ] ml_bone = {_bone_pkl} "
          f"({'VisualDiTTwoTime — visual latent as ONE PREPENDED TOKEN' if _is_dit else 'VisualUNet(TwoTime) — FiLM'}) "
          f"(from train-time model_config.pkl)")
    if override_args is not None:
        _bone_cfg = getattr(override_args, 'ml_bone', _bone_pkl) or _bone_pkl
        if _bone_cfg != _bone_pkl:
            print(f"[ config->pkl ] WARNING  ml_bone: train-pkl={_bone_pkl!r} vs "
                  f"eval-config={_bone_cfg!r} -- ARCHITECTURE key; KEEPING the train value. "
                  f"🔴 The results folder is named from the EVAL config, so it will read "
                  f"'B{_bone_cfg}' while the weights are '{_bone_pkl}'.")

    # ── FiLM breadcrumb (Gen14 Fix_9). Skipped on a transformer bone: FiLM is a U-Net
    # concept and printing the 'v1' default would assert an architecture the weights lack.
    if _is_dit:
        print("[ eval loading ] film_mode = n/a (transformer bone: no FiLM path exists)")
    else:
        _film_pkl = getattr(_engine_holder, 'film_mode', 'v1') or 'v1'
        print(f"[ eval loading ] film_mode = {_film_pkl} (from train-time model_config.pkl)")
        if override_args is not None:
            _film_cfg = getattr(override_args, 'film_mode', _film_pkl) or _film_pkl
            if _film_cfg != _film_pkl:
                print(f"[ config->pkl ] WARNING  film_mode: train-pkl={_film_pkl!r} vs "
                      f"eval-config={_film_cfg!r} -- ARCHITECTURE key; KEEPING the train value. "
                      f"🔴 The results folder will read 'film{_film_cfg}' while the weights "
                      f"are '{_film_pkl}'.")

    if override_args is not None:
        for _k in list(diffusion_config._dict.keys()):
            if not hasattr(override_args, _k):
                continue
            _new, _old = getattr(override_args, _k), diffusion_config._dict[_k]
            try:
                _same = bool(_new == _old)
            except Exception:
                _same = False
            if _same:
                continue
            if _k in _SAMPLING_OVERRIDE_KEYS:
                print(f"[ config->pkl ] INFO  {_k}: train={_old!r} -> eval={_new!r}  "
                      f"(sampling knob; applied)")
                diffusion_config._dict[_k] = _new
            else:
                print(f"[ config->pkl ] WARNING  {_k}: train-pkl={_old!r} vs "
                      f"eval-config={_new!r} -- identity/architecture key; KEEPING the train "
                      f"value to protect the checkpoint (fix the config, or retrain).")

    dataset   = dataset_config()
    model     = model_config().to(device)
    diffusion_config._dict.pop('model', None)   # prevent duplicate positional/kwarg
    diffusion = diffusion_config(model).to(device)
    trainer   = trainer_config(diffusion_model=diffusion, dataset=dataset)

    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    losses = utils.load_losses(*loadpath, 'losses.pkl')
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model,
                                     trainer.ema_model, trainer, epoch, losses)


def warn_pkl_config_mismatch(diffusion, args):
    """Surface frozen pkl values that the two-tier reconciler above cannot cover.

    Only DIFFERENTLY-NAMED keys remain pkl-authoritative (n_diffusion_steps <-> pkl
    'n_timesteps'); same-named keys were already reconciled.
    """
    import warnings
    checks = [
        ('horizon',           getattr(diffusion, 'horizon',       None), getattr(args, 'horizon',           None)),
        ('n_diffusion_steps', getattr(diffusion, 'n_timesteps',   None), getattr(args, 'n_diffusion_steps', None)),
        ('clip_denoised',     getattr(diffusion, 'clip_denoised', None), getattr(args, 'clip_denoised',     None)),
    ]
    print('\n[ eval pkl values ] (config-overrides-pkl active for same-named keys)')
    for key, pkl_v, cfg_v in checks:
        if pkl_v is None and cfg_v is None:
            continue
        mismatch = (pkl_v is not None and cfg_v is not None and pkl_v != cfg_v)
        tag = '  *** MISMATCH — patch pkl or retrain ***' if mismatch else ''
        print(f'    {key}: {pkl_v!r}  (config: {cfg_v!r}){tag}')
    print()
    for key, pkl_v, cfg_v in checks:
        if pkl_v is not None and cfg_v is not None and pkl_v != cfg_v:
            warnings.warn(f'[ pkl/config mismatch ] {key}: pkl={pkl_v!r}, config={cfg_v!r}. '
                          f'The pkl value is used at eval.', stacklevel=2)


# ══════════════════════════════════════════════════════════════════════════════════════
# 4. Main eval loop
# ══════════════════════════════════════════════════════════════════════════════════════

for exp in exps:
    for halfspace_variant in halfspace_variants:
        robot_name = exp.split('-')[0]

        if halfspace_variant == 'top-left-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][0]]
            obstacle_constraints  = [config['obstacle_constraints'][exp][3]]
        elif halfspace_variant == 'top-right-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][1]]
            obstacle_constraints  = [config['obstacle_constraints'][exp][4]]
        elif halfspace_variant == 'both-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][2],
                                     config['halfspace_constraints'][exp][3]]
            obstacle_constraints  = [config['obstacle_constraints'][exp][5]]
        else:
            polytopic_constraints = []
            obstacle_constraints  = []

        bounds              = config['bounds'][exp]
        ax_limits           = config['ax_limits'][exp]
        enlarge_constraints = config['enlarge_constraints'][robot_name]
        dt                  = config['dt'][robot_name]
        obs_indices         = config['observation_indices'][robot_name]
        act_indices         = config['action_indices'][robot_name]

        # `exp` ('avoiding-d3il') keys the YAML's constraint/bounds tables. The PYTHON
        # config that holds the four `plan_mix_visual_avoiding_*` blocks is a different
        # module — Gen16's own — so the two names are deliberately not the same string.
        class Parser(utils.Parser):
            dataset: str = 'avoiding-d3il-visual-mix'
            config:  str = 'config.avoiding-d3il-visual-mix'

        figs_all_seeds, axes_all_seeds = zip(*[
            plt.subplots(1, 1, figsize=(9, 10)) for _ in range(len(projection_variants))])
        figs_all_seeds = list(figs_all_seeds)
        axes_all_seeds = list(axes_all_seeds)

        for seed in seeds:
            args = Parser().parse_args(experiment=EXPERIMENT, seed=seed)

            fm_model   = None
            normalizer = None
            env        = None
            flow_steps = None

            if not args_cli.aggregate_only:
                fm_experiment = load_diffusion_with_override(
                    args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                    target_class=args.diffusion, epoch=args.diffusion_epoch,
                    device=args.device, override_args=args, engine=ENGINE)

                # ── weight source ────────────────────────────────────────────────────
                # DEFAULT TRUE on every arm, and Gen16's trainers select `state_best` on
                # the EMA loss to match (utils/training.py::test). Flipping this to False
                # WITHOUT reverting that selection re-creates the raw-vs-EMA mismatch Gen9
                # U4 Fix1 diagnosed: the deployed weights would not be the ones the
                # checkpoint was chosen for.
                use_ema = bool(getattr(args, 'eval_use_ema', True))
                fm_model = fm_experiment.ema if use_ema else fm_experiment.diffusion
                print(f'[ eval ] weight source: '
                      f'{"EMA (matches state_best selection)" if use_ema else "raw/live (dpcc-legacy)"}')
                warn_pkl_config_mismatch(fm_model, args)

                # ── G1 HORIZON GUARD — abort, do not warn ────────────────────────────
                # `horizon` is a TRAINING property (dataset windows + per-step loss
                # weights), so evaluating a checkpoint at another horizon is invalid. It is
                # also SILENT on the U-Net arm: ResidualTemporalBlock takes `horizon` and
                # never uses it — Conv1d + Linear are length-agnostic — so an H8 checkpoint
                # runs happily at H16 and returns a clean-looking, meaningless number.
                _ckpt_horizon = getattr(fm_model, 'horizon', None)
                if _ckpt_horizon is not None and int(_ckpt_horizon) != int(args.horizon):
                    raise SystemExit(
                        f'\n[ eval ] 🔴 HORIZON MISMATCH — checkpoint trained at '
                        f'horizon={int(_ckpt_horizon)}, eval configured for '
                        f'horizon={int(args.horizon)}.\n'
                        f'         Loaded: {args.diffusion_loadpath}\n'
                        f'         Horizon is a training property, not a sampling knob.\n')
                # G2: a plan cannot supply more actions than it holds (HardFlow asserts the
                # same, run/eval.py:380-382).
                if replan_steps >= int(args.horizon):
                    raise SystemExit(
                        f'\n[ eval ] 🔴 MIX_REPLAN_STEPS={replan_steps} must be < horizon='
                        f'{int(args.horizon)} — a plan cannot supply more actions than it has.\n')

                # ── the NFE budget, per arm ──────────────────────────────────────────
                # `nfe_of` reads this arm's OWN key: n_diffusion_steps for diffusion,
                # flow_steps_v3 for fm/mf/af. Anything that prints or path-names "K" goes
                # through it, so the DDPM arm can never be labelled with an ODE step count.
                flow_steps = int(nfe_of(ENGINE, args, default=20))
                if ENGINE != 'diffusion':
                    fm_model.flow_steps_v3 = flow_steps
                    fm_model.ode_inference_steps_v3 = flow_steps
                    for _k, _default in (('ode_solver_backend_v3', 'legacy_euler'),
                                         ('ode_solver_method_v3',  'euler'),
                                         ('ode_solver_rtol_v3',    None),
                                         ('ode_solver_atol_v3',    None),
                                         ('ode_solver_step_size_v3', None)):
                        setattr(fm_model, _k, getattr(args, _k, getattr(fm_model, _k, _default)))
                print(f'[ eval ] matched K = {flow_steps} for every arm '
                      f'(key: {ENGINE_SPEC["nfe_key"]}; savepath: {args.savepath})')

                # ── normalizers ─────────────────────────────────────────────────────
                # The visual train script pickles obs/act normalizers separately (the
                # visual dataset fits them separately). VisualNormalizer adapts them to the
                # single-object interface the Projector, the HardFlow builder and the
                # policies all read.
                ckpt_dir = os.path.join(args.loadbase, args.dataset,
                                        args.diffusion_loadpath, str(args.seed))
                with open(os.path.join(ckpt_dir, 'obs_normalizer.pkl'), 'rb') as f:
                    obs_normalizer = pickle.load(f)
                with open(os.path.join(ckpt_dir, 'act_normalizer.pkl'), 'rb') as f:
                    act_normalizer = pickle.load(f)
                normalizer = VisualNormalizer(obs_normalizer, act_normalizer)
                print(f'[ eval ] {normalizer}')

                # ── RUN PROVENANCE (U10.1) ──────────────────────────────────────────
                # The config snapshot copies the .py module verbatim, so an H8 and an H16
                # run produce identical bytes; and Parser.save writes args.json for TRAIN
                # only. Written here — the first point where every knob is final.
                provenance.write(
                    args.savepath, role='eval',
                    yaml_path=_cfg_path,
                    resolved={
                        'generation': 'Gen16',
                        'engine': ENGINE,
                        'engine_label': ENGINE_SPEC['label'],
                        'ml_bone': getattr(args, 'ml_bone', 'unet'),
                        'film_mode': getattr(args, 'film_mode', None),
                        'horizon': int(args.horizon),
                        'checkpoint_horizon': int(getattr(fm_model, 'horizon', -1)),
                        'nfe_key': ENGINE_SPEC['nfe_key'],
                        'flow_steps_K': int(flow_steps),
                        'replan_steps': int(replan_steps),
                        'dpcc_threshold': float(diffusion_timestep_threshold),
                        'hf_act_threshold': float(hf_act_threshold),
                        'hf_batch_size': int(hf_batch_size),
                        'hf_candidate_cost': hf_candidate_cost,
                        'eval_use_ema': bool(use_ema),
                        'diffusion_epoch': getattr(args, 'diffusion_epoch', None),
                        'diffusion_loadpath': getattr(args, 'diffusion_loadpath', None),
                        'exp_name': getattr(args, 'exp_name', None),
                        'seed': int(seed),
                        'seeds_in_yaml': seeds,
                        'n_trials': n_trials,
                        'projection_variants': projection_variants,
                        'constraint_types': constraint_types,
                        'halfspace_variants': halfspace_variants,
                        'max_episode_length': getattr(args, 'max_episode_length', None),
                        'batch_size_dpcc_arms': getattr(args, 'batch_size', None),
                        'visual_layout': visual_spec.LAYOUT,
                    })

                env = ObstacleAvoidanceEnv()
                env.start()

                # ── trajectory dims ─────────────────────────────────────────────────
                # Read off the loaded engine (not visual_spec) so a checkpoint/spec
                # disagreement shows up as a constraint-matrix shape error naming the two
                # numbers, rather than being papered over.
                trajectory_dim = fm_model.transition_dim - fm_model.goal_dim
                action_dim     = fm_model.action_dim
                fm_variant     = 'states_actions'
                if (trajectory_dim, action_dim) != (visual_spec.TRANSITION_DIM,
                                                    visual_spec.ACTION_DIM):
                    raise SystemExit(
                        f'[ eval ] 🔴 checkpoint dims (transition={trajectory_dim}, '
                        f'action={action_dim}) disagree with visual_spec '
                        f'(transition={visual_spec.TRANSITION_DIM}, '
                        f'action={visual_spec.ACTION_DIM}).')
                obs_indices_updated = {k: v + action_dim for k, v in obs_indices.items()}
                act_obs_indices = {**act_indices, **obs_indices_updated}

                # ── constraints (identical construction to the state-only lineage) ───
                constraint_list = []
                constraint_list_tightened = []
                constraint_list_polytopic_not_tightened = []
                if 'halfspace' in constraint_types:
                    for c in polytopic_constraints:
                        constraint_list.append(('ineq', utils.formulate_halfspace_constraints(
                            c, 0, trajectory_dim, act_obs_indices)))
                        constraint_list_tightened.append(('ineq', utils.formulate_halfspace_constraints(
                            c, enlarge_constraints, trajectory_dim, act_obs_indices)))
                        constraint_list_polytopic_not_tightened.append(('ineq',
                            utils.formulate_halfspace_constraints(c, 0, trajectory_dim, act_obs_indices)))
                if 'bounds' in constraint_types:
                    lower_bound, upper_bound = utils.formulate_bounds_constraints(
                        constraint_types, bounds, trajectory_dim, act_obs_indices)
                    constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])
                    constraint_list_tightened.extend([['lb', lower_bound], ['ub', upper_bound]])
                if 'obstacles' in constraint_types:
                    for co in obstacle_constraints:
                        idx = [act_obs_indices[co['dimensions'][0]],
                               act_obs_indices[co['dimensions'][1]]]
                        constraint_list.append([co['type'], idx, co['center'], co['radius']])
                        constraint_list_tightened.append(
                            [co['type'], idx, co['center'], co['radius'] + enlarge_constraints])
                constraint_list_without_prior           = copy(constraint_list)
                constraint_list_without_prior_tightened = copy(constraint_list_tightened)
                dynamics_constraints = []
                if 'dynamics' in constraint_types:
                    dynamics_constraints = utils.formulate_dynamics_constraints(
                        exp, act_obs_indices, action_dim)
                for c in dynamics_constraints:
                    constraint_list.append(c)
                    constraint_list_tightened.append(c)

            for variant_idx, variant in enumerate(projection_variants):
                save_path = (f'{args.savepath}/results/halfspace_{halfspace_variant}'
                             if 'avoiding' in exp else f'{args.savepath}/results')
                os.makedirs(save_path, exist_ok=True)

                # [SolverSwap] 🔴 The artifact name carries the NLP backend, so an SLSQP run
                # lands BESIDE the IPOPT corpus instead of overwriting it. Defined at the TOP
                # of the variant loop because the aggregate-only reader and the per-variant log
                # both need it before inference starts. Under 'ipopt' it is the old name
                # unchanged, so nothing already on disk moves.
                variant_out = artifact_variant_label(variant, resolve_nlp_backend())

                if args_cli.aggregate_only:
                    npz_path = os.path.join(save_path, f'{variant_out}.npz')
                    if not os.path.exists(npz_path):
                        print(f'[ eval ] skipping {variant} seed {seed}: no npz at {npz_path}')
                        continue
                    print(f'[ eval ] aggregating {variant} - seed {seed}')
                    data = np.load(npz_path, allow_pickle=True)
                    if 'obs_all' in data:
                        for i in range(min(len(data['obs_all']), plot_how_many)):
                            buf = np.array(data['obs_all'][i])
                            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                            axes_all_seeds[variant_idx].plot(
                                buf[:, obs_indices['x']], buf[:, obs_indices['y']],
                                colors[seed % len(colors)], linewidth=2)
                    continue

                log_file = open(os.path.join(save_path, f'eval_{variant_out}.log'), 'w')
                original_stdout = sys.stdout
                sys.stdout = Tee(sys.stdout, log_file)
                try:
                    is_hardflow = variant.startswith('hardflow')
                    print(f'---------------- {exp} | {halfspace_variant} | {variant} | '
                          f'engine={ENGINE} | seed={seed} | K={flow_steps} ----------------')

                    # ── arm-C availability ──────────────────────────────────────────
                    # `resolve_engine_hf` refuses 'diffusion': a DDPM reverse chain has no
                    # velocity field, so `hardflow_new` is not merely unsupported on it, it
                    # is undefined. Skipping the variant (rather than crashing) lets ONE
                    # yaml list every arm and every engine run against it.
                    if is_hardflow and ENGINE == 'diffusion':
                        print(f'[ eval ] SKIP {variant}: HardFlow has no host on the '
                              f'diffusion arm (no velocity field). '
                              f'Hosts are fm / mf / af.')
                        continue

                    # [Gen0fix2] threshold 0 => the activation gate fires only on the FINAL
                    # step, i.e. ONE projection after the last ODE step — the paper's
                    # definition of post-processing. Without it, `post_processing` inherits
                    # the normal schedule and duplicates `dpcc-r`.
                    threshold = 0.0 if 'post_processing' in variant else diffusion_timestep_threshold

                    gradient = 'gradient' in variant
                    if 'model_free' in variant and 'tightened' in variant:
                        constraints = constraint_list_without_prior_tightened
                    elif 'model_free' in variant:
                        constraints = constraint_list_without_prior
                    elif 'tightened' in variant:
                        constraints = constraint_list_tightened
                    else:
                        constraints = constraint_list

                    delta_t = dt
                    if   'dt0p25' in variant: delta_t = 0.25 * dt
                    elif 'dt0p5'  in variant: delta_t = 0.5 * dt
                    elif 'dt2p0'  in variant: delta_t = 2.0 * dt
                    elif 'dt4p0'  in variant: delta_t = 4.0 * dt

                    if is_hardflow:
                        # ── arm C ───────────────────────────────────────────────────
                        # 🔴 B4_PARITY — the candidate fan is resolved PER VARIANT:
                        #   `hardflow_new`         -> 1              faithful upstream control
                        #   `hardflow_new-r/-c/-t` -> hf_batch_size  (default 4 == arms A/B)
                        # Both arms loop SERIALLY over candidates around their CPU solve, so
                        # a smaller fan is a compute discount that reads as a speedup in
                        # every timing table. See logs_in_develop/HF_Batch_Parity/.
                        batch_size = resolve_hf_batch_size(variant, hf_batch_size)
                        if batch_size != args.batch_size:
                            print(f'[ hardflow ] ⚠️  arm-C fan B={batch_size} != DPCC-arm fan '
                                  f'B={args.batch_size} for {variant!r} — wall-clock is NOT '
                                  f'comparable across arms for this variant.')
                        # Strip '-tightened' FIRST so the selection suffix composes with the
                        # geometry (hardflow_new-c-tightened -> min-cost AND enlarged).
                        _sel_base = variant.replace('-tightened', '')
                        hf_selection = 'random'
                        if   _sel_base.endswith('-t'): hf_selection = 'temporal_consistency'
                        elif _sel_base.endswith('-c'): hf_selection = 'minimum_projection_cost'
                        # 🔴 `-c` IS NOT TRUSTWORTHY AT B>1 (open, not fixed here). Pooled
                        # over 750 arm-C cells that ran at B=4 on avoiding:
                        #   -r  S&C 0.707 / succ 0.917 /  67.7 steps /   0 timeouts
                        #   -t  S&C 0.707 / succ 0.883 /  71.2 steps /   5 timeouts
                        #   -c  S&C 0.443 / succ 0.540 / 138.5 steps / 370 timeouts (49%)
                        # `candidate_costs` is Σ‖x1_proj − x1_ref‖², so argmin picks the
                        # candidate the NLP barely touched — on avoiding that is the
                        # candidate that barely MOVES, which stalls the episode.
                        if hf_selection == 'minimum_projection_cost' and batch_size > 1:
                            print(f'[ hardflow ] 🔴 {variant}: `-c` selection at B={batch_size} '
                                  f'is a KNOWN-BAD arm (49% timeouts across 750 B=4 cells). '
                                  f'Reported for completeness; do not cite without '
                                  f're-checking. See logs_in_develop/HF_Batch_Parity/.')
                        _init_noise, _two_time = resolve_engine_hf(ENGINE)
                        print(f'[ hardflow ] host engine={ENGINE}  init_noise={_init_noise}  '
                              f'two_time={_two_time}')
                        _layout, _nlp, hf_sampler = build_hardflow_sampler(
                            model=fm_model, normalizer=normalizer, horizon=args.horizon,
                            transition_dim=trajectory_dim, action_dim=action_dim,
                            constraint_list=constraints, engine=ENGINE, dt=delta_t,
                            reg_scale=float(hardflow_cfg.get('reg_scale', 1.0)),
                            activation_threshold=hf_act_threshold,
                            dynamics_mode=hardflow_cfg.get('dynamics_mode', 'deriv'),
                            linear_dynamics=None,
                            print_level=int(hardflow_cfg.get('ipopt_print_level', 0)),
                            print_time=bool(hardflow_cfg.get('casadi_print_time', False)),
                            device=args.device, goal_dim=fm_model.goal_dim)
                        policy = VisualHardFlowPolicy(
                            model=fm_model, normalizer=normalizer, sampler=hf_sampler,
                            flow_steps=flow_steps, device=args.device,
                            trajectory_selection=hf_selection,
                            candidate_cost=hf_candidate_cost)
                    else:
                        # ── arms A / B ──────────────────────────────────────────────
                        batch_size = args.batch_size
                        projector = Projector(
                            horizon=args.horizon, transition_dim=trajectory_dim,
                            action_dim=action_dim, goal_dim=fm_model.goal_dim,
                            constraint_list=constraints, normalizer=normalizer,
                            gradient=gradient, gradient_weights=[1, 0.5, 2],
                            variant=fm_variant, dt=delta_t, cost_dims=None,
                            device=args.device, solver='scipy',
                            diffusion_timestep_threshold=threshold)
                        projector = None if variant == 'diffuser' else projector
                        trajectory_selection = 'random'
                        if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                        if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
                        policy = VisualPolicy(
                            model=fm_model, normalizer=normalizer, projector=projector,
                            device=args.device, trajectory_selection=trajectory_selection)

                    # U10 — hand the cadence to BOTH policy classes. They do not loop; they
                    # need it only so the -t selection compares plans at the right shift.
                    policy.replan_steps = replan_steps

                    fig, ax = plt.subplots(min(n_trials, plot_how_many), 6,
                                           figsize=(30, 5 * min(n_trials, plot_how_many)),
                                           squeeze=False)
                    fig.suptitle(f'{exp} - {ENGINE} - {variant}')
                    fig_all, ax_all = plt.subplots(
                        min(n_trials, plot_how_many), len(projection_variants),
                        figsize=(10 * len(projection_variants),
                                 10 * min(n_trials, plot_how_many)), squeeze=False)

                    save_samples_every = 1                        # npz keeps every step
                    plot_samples_every = max(1, args.horizon // 2)  # plot stays readable

                    n_success                 = np.zeros(n_trials)
                    n_success_and_constraints = np.zeros(n_trials)
                    n_steps                   = np.zeros(n_trials)
                    n_violations              = np.zeros(n_trials)
                    total_violations          = np.zeros(n_trials)
                    avg_time                  = np.zeros(n_trials)
                    collision_free_completed  = np.ones(n_trials)
                    pos_tracking_errors       = np.zeros((n_trials, args.max_episode_length - 1))
                    nfe_total = nlp_solves_total = nlp_failures_total = 0
                    # [SolverSwap] default for arms A/B, which run no NLP at all.
                    nlp_backend_used = 'n/a'
                    obs_all, act_all, sampled_trajectories_all = [], [], []

                    for i in range(n_trials):
                        torch.manual_seed(i)

                        obs     = env.reset()
                        action  = env.robot_state()[:2]
                        fixed_z = env.robot_state()[2:]
                        obs     = np.concatenate((action[:2], obs))   # 4D [des_xy | c_xy]

                        obs_buffer, action_buffer, sampled_trajectories = [], [], []
                        disable_projection = False
                        desired_next_pos = obs[obs_indices['x']:obs_indices['y'] + 1].copy()
                        # U10 — per-episode replan state. `plan_idx` counts how many actions
                        # of the current plan have been consumed; at replan_steps=1 the
                        # cache is refilled every step, so these are inert.
                        plan_actions = plan_obs = None
                        plan_idx = 0
                        samples = None

                        rt_rec = RTRecorder(
                            episode_id=f'{exp}_{ENGINE}_{variant}_seed{seed}_trial{i}',
                            variant=variant, scene=exp,
                            system=f'VisualAvoiding_Mix_{ENGINE}',
                            control_hz=RT_CONTROL_HZ,
                            batch_size=batch_size, horizon=args.horizon,
                            text_log=config.get('write_to_file', True))

                        for _ in range(args.max_episode_length):
                            violated_this_timestep = 0

                            if 'halfspace' in constraint_types:
                                for constraint in constraint_list_polytopic_not_tightened:
                                    if constraint[0] == 'ineq':
                                        c, d = constraint[1]
                                        obs_check = (obs[:-fm_model.goal_dim]
                                                     if fm_model.goal_dim > 0 else obs)
                                        if obs_check @ c[action_dim:] >= d:
                                            violated_this_timestep = 1
                                            total_violations[i] += obs_check @ c[action_dim:] - d
                                            collision_free_completed[i] = 0

                            if 'obstacles' in constraint_types:
                                for co in obstacle_constraints:
                                    _p = obs[[obs_indices['x'], obs_indices['y']]]
                                    if np.linalg.norm(_p - co['center']) < co['radius']:
                                        violated_this_timestep = 1
                                        total_violations[i] += (
                                            co['radius'] - np.linalg.norm(_p - co['center']))
                                        collision_free_completed[i] = 0

                            if _ > 0 and 'bounds' in constraint_types:
                                act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                                total_violations[i] += (
                                    np.sum(np.maximum(0, act_obs - upper_bound))
                                    + np.sum(np.maximum(0, lower_bound - act_obs)))

                            n_violations[i] += violated_this_timestep

                            # ── U10 RECEDING HORIZON ────────────────────────────────
                            # Replan when the cache is empty or exhausted; otherwise replay
                            # the next action of the plan already in hand. At
                            # replan_steps=1 this is true on EVERY step and nothing below
                            # changes — byte-identical to the pre-U10 loop.
                            _replanned = (plan_actions is None) or (plan_idx >= replan_steps)
                            if _replanned:
                                # The visual condition: one bp-cam frame, captured at THIS
                                # step, alongside the 4-D obs anchor. Gen9 Epoch 2's swap B.
                                bp_raw = env.bp_cam.get_image(width=_IMG_W, height=_IMG_H,
                                                              depth=False)
                                bp_image = bp_raw.transpose((2, 0, 1)).copy() / 255.

                                start = time.time()
                                action, samples = policy(
                                    conditions={0: obs, 'primary_img': bp_image},
                                    batch_size=batch_size, horizon=args.horizon,
                                    disable_projection=disable_projection)
                                _rt_total_ms = (time.time() - start) * 1e3
                                avg_time[i] += _rt_total_ms / 1e3
                                if is_hardflow:
                                    nlp_solves_total += policy.last_info.get('nlp_solves', 0)
                                    nlp_failures_total += policy.last_info.get('nlp_failures', 0)
                                # Cache the EXECUTED candidate's plan (the policy publishes
                                # it; `which_trajectory` is not visible here, so
                                # `samples.actions[0]` would be the wrong candidate under
                                # -c/-t selection).
                                plan_actions = getattr(policy, 'last_executed_actions', None)
                                plan_obs     = getattr(policy, 'last_executed_observations', None)
                                if replan_steps > 1 and plan_actions is None:
                                    raise SystemExit(
                                        f'\n[ eval ] 🔴 replan_steps={replan_steps} needs the '
                                        f'executed plan, but {type(policy).__name__} did not '
                                        f'publish `last_executed_actions`.\n')
                                plan_idx = 0
                            else:
                                # No compute this step: the plan was paid for when made.
                                _rt_total_ms = 0.0
                                action = plan_actions[plan_idx]
                            plan_idx += 1

                            rt_rec.step(t=_ / RT_CONTROL_HZ, total_ms=_rt_total_ms, obs=obs,
                                        action=action,
                                        pos=obs[[obs_indices['x'], obs_indices['y']]],
                                        proj_active=(variant != 'diffuser'
                                                     and not disable_projection
                                                     and _replanned),   # no plan, no projection
                                        contact=bool(violated_this_timestep), step_idx=_)

                            next_pos_des = action + obs[:2]
                            obs, rew, terminated, info = env.step(
                                np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                            success = info[1]
                            obs = np.concatenate((next_pos_des[:2], obs))

                            if _ >= 1:
                                pos_tracking_errors[i, _ - 1] = np.linalg.norm(
                                    obs[obs_indices['x']:obs_indices['y'] + 1] - desired_next_pos)
                            # U10 — the tracking reference is the NEXT state of the plan
                            # being executed. At replan_steps=1 that is the fresh plan's
                            # step 1, i.e. the original expression untouched.
                            if replan_steps == 1 or plan_obs is None:
                                desired_next_pos = samples.observations[
                                    0, 1, [obs_indices['x'], obs_indices['y']]]
                            else:
                                _ref_k = min(plan_idx, plan_obs.shape[0] - 1)
                                desired_next_pos = plan_obs[
                                    _ref_k, [obs_indices['x'], obs_indices['y']]]

                            if _ % save_samples_every == 0:
                                sampled_trajectories.append(samples.observations[:, :, :])

                            obs_buffer.append(obs)
                            action_buffer.append(action)
                            if success:
                                n_success[i] = 1
                            if (terminated or _ == args.max_episode_length - 1) and not success:
                                collision_free_completed[i] = 0
                            if success or terminated or _ == args.max_episode_length - 1:
                                n_steps[i] = _
                                avg_time[i] /= max(_, 1)
                                if success and collision_free_completed[i]:
                                    n_success_and_constraints[i] = 1
                                break

                        obs_all.append(np.array(obs_buffer))
                        act_all.append(np.array(action_buffer))
                        if config.get('write_to_file', True):
                            rt_rec.save(f'{save_path}/realtime_{variant}_trial{i}.log',
                                        behaviour={'success': int(n_success[i]),
                                                   'n_steps': int(n_steps[i]),
                                                   'violations': int(n_violations[i])})
                        sampled_trajectories_all.append(sampled_trajectories)

                        if i >= plot_how_many:
                            continue
                        plot_states = ['x', 'y', 'x_des', 'y_des']
                        for j in range(len(plot_states)):
                            if obs_indices.get(plot_states[j]) is not None:
                                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                                ax[i, j].set_title(plot_states[j])

                        for curr_ax in [ax[i, 4], ax_all[i, variant_idx]]:
                            curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']],
                                         np.array(obs_buffer)[:, obs_indices['y']], 'k')
                            curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']],
                                         np.array(obs_buffer)[0, obs_indices['y']], 'go',
                                         label='Start')
                            curr_ax.set_xlim(ax_limits[0])
                            curr_ax.set_ylim(ax_limits[1])

                        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                        axes_all_seeds[variant_idx].plot(
                            np.array(obs_buffer)[:, obs_indices['x']],
                            np.array(obs_buffer)[:, obs_indices['y']],
                            colors[seed % len(colors)], linewidth=2)

                        for __ in range(0, len(sampled_trajectories_all[i]), plot_samples_every):
                            # 🔴 iterate the LOCAL batch, not args.batch_size — arm C
                            # overrides it (fix_7), and asking for index 1 of a 1-row
                            # candidate array is an IndexError.
                            for ___ in range(min(batch_size, 4)):
                                for curr_ax in [ax[i, 5], ax_all[i, variant_idx]]:
                                    _tr = sampled_trajectories_all[i][__]
                                    curr_ax.plot(_tr[___, :args.horizon, obs_indices['x']],
                                                 _tr[___, :args.horizon, obs_indices['y']], 'b')
                                    curr_ax.plot(_tr[___, 0, obs_indices['x']],
                                                 _tr[___, 0, obs_indices['y']], 'go')
                        ax[i, 5].set_xlim(ax_limits[0])
                        ax[i, 5].set_ylim(ax_limits[1])

                        for curr_ax in [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]:
                            utils.plot_environment_constraints(exp, curr_ax)
                            if 'halfspace' in constraint_types:
                                utils.plot_halfspace_constraints(
                                    exp, polytopic_constraints, curr_ax, ax_limits)
                            if 'obstacles' in constraint_types:
                                for co in obstacle_constraints:
                                    curr_ax.add_patch(matplotlib.patches.Circle(
                                        co['center'], co['radius'], color='b', alpha=0.2))

                    print(f'Success rate: {np.mean(n_success):.3f}')
                    print(f'Constraints satisfied: {np.mean(collision_free_completed):.3f}')
                    print(f'Success rate (goal and constraints): '
                          f'{np.mean(n_success_and_constraints):.3f}')
                    print(f'Avg number of steps: '
                          f'{(np.mean(n_steps[n_success > 0]) if n_success.sum() > 0 else 0):.2f}'
                          f' +- {(np.std(n_steps[n_success > 0]) if n_success.sum() > 0 else 0):.2f}')
                    print(f'Avg number of constraint violations: {np.mean(n_violations):.2f}'
                          f' +- {np.std(n_violations):.2f}')
                    print(f'Avg total violation: {np.mean(total_violations):.3f}'
                          f' +- {np.std(total_violations):.3f}')
                    print(f'Average computation time per step: {np.mean(avg_time):.3f}')
                    if variant == 'diffuser':
                        print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')
                    if is_hardflow:
                        nfe_total = int(getattr(policy, 'nfe', 0))
                        # [SolverSwap] read off the live NLP object, not the config, so the
                        # recorded value is what actually ran (env override included).
                        nlp_backend_used = str(getattr(getattr(policy, 'nlp', None), 'nlp_backend', 'n/a'))
                        print(f'[hardflow] NFE={nfe_total}  NLP solves={nlp_solves_total}  '
                              f'NLP failures={nlp_failures_total}  batch(mpc)={batch_size}  '
                              f'act_threshold={hf_act_threshold}  '
                              # [SolverSwap] the solver is now selectable, so it must be IN the log.
                              f'nlp_backend={nlp_backend_used}')

                    if config.get('write_to_file', True):
                        _hf_budget = (hardflow_step_budget(flow_steps, hf_act_threshold)
                                        if is_hardflow and flow_steps else (0, 0))
                        np.savez(f'{save_path}/{variant_out}.npz',
                                 n_success=n_success,
                                 n_success_and_constraints=n_success_and_constraints,
                                 n_steps=n_steps,
                                 n_violations=n_violations,
                                 total_violations=total_violations,
                                 avg_time=avg_time,
                                 collision_free_completed=collision_free_completed,
                                 args=args,
                                 # Gen16 — the arm identity travels WITH the numbers, so a
                                 # Data_Analysis sweep can never pool two engines by mistake.
                                 engine=ENGINE,
                                 ml_bone=getattr(args, 'ml_bone', 'unet'),
                                 flow_steps_K=int(flow_steps),
                                 replan_steps=int(replan_steps),
                                 is_hardflow=bool(is_hardflow),
                                 nfe_total=int(nfe_total),
                                 nlp_solves_total=int(nlp_solves_total),
                                 nlp_failures_total=int(nlp_failures_total),
                                 # [SolverSwap] 'slsqp' (DPCC scipy) or 'ipopt' (original CasADi).
                                 # Present on EVERY row so a DA can never pool the two backends.
                                 nlp_backend=str(nlp_backend_used),
                                 # numeric twin: generic DA loaders coerce npz scalars to float, so the
                                 # string alone would land as NaN. 1.0 = DPCC scipy SLSQP, 0.0 = IPOPT.
                                 nlp_backend_slsqp=float(nlp_backend_used == 'slsqp'),
                                 hf_batch_size=int(batch_size),
                                 hf_act_threshold=float(hf_act_threshold),
                                 # HFK1 (2026-08-24) — the degeneracy verdict, RECORDED rather
                                 # than left for a DA to re-derive. A step is genuinely HardFlow
                                 # only if it is active AND non-terminal; at the terminal step
                                 # tau=1 kills the lookahead, the damping and the feedback. So
                                 # hf_n_genuine == 0 means this row is Pi_S(Euler sample) —
                                 # sample-then-project, == DPCC modulo solver — and must NOT be
                                 # reported as a HardFlow result. True at K=1 always, and at
                                 # K=2 under the shipped A=0.5.
                                 # See logs_in_develop/aggregated_hardflow_lowK/
                                 hf_n_active=int(_hf_budget[0]),
                                 hf_n_genuine=int(_hf_budget[1]),
                                 hf_degenerate=bool(is_hardflow and _hf_budget[1] == 0),
                                 obs_all=np.array(obs_all, dtype=object),
                                 act_all=np.array(act_all, dtype=object),
                                 sampled_trajectories_all=np.array(sampled_trajectories_all,
                                                                   dtype=object))

                    fig.savefig(f'{save_path}/{variant_out}.png')
                    plt.close(fig)
                    ax_all[0, variant_idx].set_title(variant)

                finally:
                    sys.stdout = original_stdout
                    log_file.close()

            if not args_cli.aggregate_only:
                fig_all.savefig(f'{save_path}/all.png')
                plt.close(fig_all)   # fix_7: the only figure never closed -> "More than 20
                                     # figures have been opened" once the variant list grew.
                env.close()

        # ── aggregate plots across seeds ──────────────────────────────────────────────
        path = f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}'
        os.makedirs(path, exist_ok=True)
        for variant_idx, (fig, axis) in enumerate(zip(figs_all_seeds, axes_all_seeds)):
            axis.set_xlim(ax_limits[0])
            axis.set_ylim(ax_limits[1])
            axis.set_facecolor([1, 1, 0.9])
            utils.plot_environment_constraints(exp, axis)
            if 'halfspace' in constraint_types:
                utils.plot_halfspace_constraints(exp, polytopic_constraints, axis, ax_limits,
                                                 enlarge_constraints=enlarge_constraints)
            if 'obstacles' in constraint_types:
                for co in obstacle_constraints:
                    axis.add_patch(matplotlib.patches.Circle(
                        co['center'], co['radius'], color='b', alpha=0.2))
                    axis.add_patch(matplotlib.patches.Circle(
                        co['center'], co['radius'] + enlarge_constraints,
                        color='b', alpha=0.1, linestyle='--'))
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.png', bbox_inches='tight')
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.pdf',
                        bbox_inches='tight', format='pdf')
            plt.close(fig)

print(f'[ eval ] Gen16 visual-avoiding eval finished — engine={ENGINE} '
      f'({ENGINE_SPEC["label"]}), seeds={seeds}.')
