# Gen3v6 MeanFlow evaluation — DPCC-projected planning on D3IL avoiding.
# Sibling of FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py; loads the
# 'plan_fm_v3_meanflow' config block. See
# logs_in_develop/Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md.
import time
import yaml
import os
import torch
from copy import copy
import minari
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import flow_matcher_v3_meanflow.utils as utils
from flow_matcher_v3_meanflow.sampling.policies import Policy
from diffuser.utils import provenance   # U10.1 — env-override provenance (shared)
from flow_matcher_v3_meanflow.sampling.projection import Projector
# Gen3v6 U3 — arm C: HardFlow in-loop constrained sampler (verbatim Gen12 port; queried at h=0).
from flow_matcher_v3_meanflow.sampling.hardflow_projection import (
    HardFlowPolicy, resolve_activation_threshold, resolve_hf_batch_size,
    hardflow_step_budget)          # HFK1 (2026-08-24)
# REAL_TIME_RECORDING_UPDATE — per-step timing/digital-twin recorder (see logs_in_develop/REALTIME_RECORDING)
from realtime_recording.behavior_logger import RTRecorder
RT_CONTROL_HZ = 30   # REAL_TIME_RECORDING_UPDATE — assumed deployment loop rate (budget=1000/hz ms); tune per target hardware
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
import sys
import argparse

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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Evaluation script with aggregation mode.')
parser.add_argument('--seed', type=int, help='Run only this specific seed.')
parser.add_argument('--aggregate-only', action='store_true', help='Skip inference, only aggregate existing results into all_seeds plots.')
# 🔵 U9 MATCHED-K AUTO-EVAL — ⚠️ MATCHED BUDGET OR NOTHING (PLAN §7; fix_7.3 §9 — one hard-coded
# k_steps=10 made the decisive control unrunnable and killed an entire generation's claim). K is a
# first-class CLI knob here so the {1,2,5,10} grid is a loop in the sbatch, not an HFFM_FLOW_STEPS
# edit someone has to remember. Ported verbatim from the Gen3v7 sibling
# (FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py), which had it from day one.
# `flow_steps_v3` is in args_to_watch_fmv3_hf_plan (label 'K'), so each K writes its OWN results
# directory and no two budgets can overwrite each other.
parser.add_argument('--flow-steps', type=int, default=None, metavar='K',
                    help='override flow_steps_v3 (NFE budget K) for this run')
args_cli, remaining_argv = parser.parse_known_args()
# Pass remaining args to Parser if needed
sys.argv = [sys.argv[0]] + remaining_argv

# Gen3v6 U3: repointed from the SHARED config/projection_eval.yaml to the Gen3v6-dedicated
# unified file (DPCC arms + HardFlow arm in one document). --config overrides it.
_default_cfg = 'config/meanflow_projection_eval.yaml'
_cfg_path = _default_cfg
if '--config' in remaining_argv:
    _cfg_path = remaining_argv[remaining_argv.index('--config') + 1]
with open(_cfg_path, 'r') as file:
    config = yaml.safe_load(file)
print(f'[ eval ] config: {_cfg_path}')

# 🔴 FIX_9_CFG_PROVENANCE (2026-08-07) — publish the yaml THIS eval loaded. config/avoiding-d3il.py
# builds the results-folder tokens from it and utils/setup.py snapshots it; both used to hard-code
# config/projection_eval.yaml, a file this eval never opens. That is why job 24334 landed in a path
# saying `T1` while the Projector was gated at 0.5. Must be set BEFORE the first
# Parser().parse_args() / importlib.import_module('config.…'), which is what imports that module.
os.environ['FMPCC_PROJ_CFG'] = _cfg_path

# 🔴 FIX_9_CFG_PROVENANCE — DPCC_THRESHOLD makes arm B's threshold settable per job, the way the
# HFFM_* knobs already are for arm C (runs are configured at submit time on the cluster, not in git).
diffusion_timestep_threshold = float(os.environ.get(
    'DPCC_THRESHOLD', config.get('diffusion_timestep_threshold', 0.5)))

# ── arm C (HardFlow) knobs, resolved once (verbatim Gen12 semantics) ──────────────────────
hardflow_cfg = config.get('hardflow', {})
hf_act_threshold = resolve_activation_threshold(
    os.environ.get('HFFM_ACT_THRESHOLD',
                   hardflow_cfg.get('activation_threshold',
                                    hardflow_cfg.get('activation', 1.0))))
# 🔴 B4_PARITY (2026-08-20) — the run-level arm-C fan. Default is 4 (was 1), i.e. the DPCC
# arms' `batch_size`, because both arms loop serially over candidates around their CPU solve
# and a mismatched fan makes arm-B-vs-arm-C wall-clock comparisons void. This is the fan the
# SELECTION variants (-r/-c/-t) get; bare `hardflow_new` is pinned to 1 by
# resolve_hf_batch_size(). A yaml with no `hardflow.batch_size` key now also lands on 4.
hf_batch_size = int(os.environ.get('HFFM_BATCH', hardflow_cfg.get('batch_size', 4)))
hf_candidate_cost = hardflow_cfg.get('candidate_cost', 'prox')

# 🔴 FIX_9_CFG_PROVENANCE — republish the RESOLVED values (aliases mapped, yaml fallbacks applied)
# so the results path is built from what actually runs. resolve_activation_threshold() stays the
# single resolver; the config module only reads the number back out. Arm B's and arm C's thresholds
# are SEPARATE knobs by design (a threshold sweep needs them independent) — this line is what makes
# a mismatch visible instead of silent, and what stops HFFM_ACT_THRESHOLD=0.5 and =1.0 from
# overwriting each other's results directory.
os.environ['FMPCC_DPCC_THRESHOLD'] = '%g' % float(diffusion_timestep_threshold)
os.environ['HFFM_ACT_THRESHOLD'] = '%g' % float(hf_act_threshold)
os.environ['HFFM_BATCH'] = str(int(hf_batch_size))
print(f'[ eval ] resolved  cfg={_cfg_path}  dpcc_threshold={diffusion_timestep_threshold}  '
      f'hf_act_threshold={hf_act_threshold}  hf_batch={hf_batch_size}  '
      f'hf_candidate_cost={hf_candidate_cost}')

# ── H8+8 (U10) — RECEDING-HORIZON CADENCE ────────────────────────────────────────────────
# How many actions of each plan are executed before replanning. 1 == every env step gets a
# fresh plan, which is what every FM-PCC result to date used and remains the DEFAULT.
# HardFlow's own eval runs `replan_steps = 8` at H16 (run/eval.py:390-397) — reproducing that
# planning structure is what this knob exists for. `replan_steps < horizon` is asserted below,
# once the checkpoint's horizon is known (HardFlow asserts the same, run/eval.py:380-382).
replan_steps = int(os.environ.get('MF_REPLAN_STEPS', config.get('replan_steps', 1)))
if replan_steps < 1:
    raise ValueError(f'MF_REPLAN_STEPS must be >= 1, got {replan_steps}')

# ── MPC CANDIDATE FAN, arms A/B — the SECOND fan, until now unsettable ───────────────────
# This eval has TWO independent candidate fans and only one of them used to be reachable:
#   arms A/B (`diffuser`, `dpcc-*`) -> args.batch_size, from the plan block. Was a hardcoded
#                                      4; config/avoiding-d3il.py now reads FMPCC_MPC_BATCH.
#   arm C    (`hardflow_new-*`)     -> hf_batch_size above (HFFM_BATCH / hardflow.batch_size),
#                                      with bare `hardflow_new` pinned to 1 by
#                                      resolve_hf_batch_size().
# Read here only to build the collision tag and to report it — the VALUE is consumed by the
# config module, which is why this has to happen before the first Parser().parse_args().
mpc_batch = int(os.environ.get('FMPCC_MPC_BATCH', 4))
if mpc_batch < 1:
    raise ValueError(f'FMPCC_MPC_BATCH must be >= 1, got {mpc_batch}')
if mpc_batch != hf_batch_size:
    print(f'[ eval ] ⚠️  arms A/B fan (mpc={mpc_batch}) != arm-C fan (HFFM_BATCH={hf_batch_size}) '
          f'-- both arms solve SERIALLY per candidate, so avg_time is not comparable across arms '
          f'(B4_PARITY). Set FMPCC_MPC_BATCH=HFFM_BATCH unless the mismatch is the experiment.')

# 🔴 PATH COLLISION GUARD — neither the replan cadence NOR the arms-A/B fan is a results-folder
# token, so an r1-vs-r8 or an mpc4-vs-mpc1 pair at the same K/A/T would write to the SAME
# directory and clobber each other (the hazard args_to_watch_fmv3_hf_plan exists to prevent).
# Promoting either to a real token would rename every historic path, so a non-default value
# auto-tags itself via the existing custom-message slot instead. The tags compose ('r8-mpc1').
# An explicit FMPCC_RUN_MSG always wins. Must be set BEFORE the first Parser().parse_args(),
# which is what imports config/<exp>.py.
_auto_tags = ([f'r{replan_steps}'] if replan_steps != 1 else []
              ) + ([f'mpc{mpc_batch}'] if mpc_batch != 4 else [])
if _auto_tags and not os.environ.get('FMPCC_RUN_MSG'):
    os.environ['FMPCC_RUN_MSG'] = '-'.join(_auto_tags)
    print(f'[ eval ] non-default {" + ".join(_auto_tags)} -> auto-tagged results path with '
          f'FMPCC_RUN_MSG={os.environ["FMPCC_RUN_MSG"]} (set it yourself to override)')
print(f'[ eval ] replan_steps={replan_steps} '
      f'({"per-step replanning (historic default)" if replan_steps == 1 else "receding horizon, HardFlow-style"})'
      f'  |  mpc fan: arms A/B={mpc_batch}, arm C={hf_batch_size}')

exps = config['exps']
seeds = config['seeds']
if args_cli.seed is not None:
    seeds = [args_cli.seed]
    print(f'[ eval ] Overriding seeds from config to: {seeds}')

if args_cli.flow_steps is not None:
    # 🔵 U9 — patch the config MODULE's dict before any Parser reads it. utils.Parser.read_config
    # does `importlib.import_module(args.config)` and copies `base[experiment]` key by key, and
    # Python caches modules — so this is the intended data path, not a monkey-patch: exp_name,
    # savepath and the diffusion kwargs all follow automatically.
    import importlib
    for _exp in exps:
        _mod = importlib.import_module('config.' + _exp)
        _blk = _mod.base['plan_fm_v3_meanflow']
        _blk['flow_steps_v3'] = args_cli.flow_steps
        if 'ode_inference_steps_v3' in _blk:
            _blk['ode_inference_steps_v3'] = args_cli.flow_steps
        # Gen3v6 U3 — matched-K: `flow_steps` is arm C's Euler K. Patch it with the SAME value
        # or --flow-steps would move arms A/B only and every arm-B-vs-arm-C table would be
        # comparing different NFE budgets. The env path (HFFM_FLOW_STEPS) sets both in the
        # plan block itself; this is the CLI path doing the same.
        _blk['flow_steps'] = args_cli.flow_steps
    print(f'[ eval ] Overriding flow_steps_v3 / flow_steps (K) from config to: {args_cli.flow_steps}')

projection_variants = config['projection_variants']
halfspace_variants = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials = config['n_trials']
plot_how_many = config['plot_how_many']
constraint_types = config['constraint_types']
# 🔴 FIX_9_CFG_PROVENANCE — diffusion_timestep_threshold and the hf_* knobs are resolved ABOVE,
# immediately after the yaml load. 🔵 U9 moved them there: the --flow-steps branch above calls
# importlib.import_module('config.' + exp), and Python caches modules, so publishing the resolved
# values here would be too late for the folder name on any --flow-steps run.
# Matched-K note: K is set for EVERY arm via HFFM_FLOW_STEPS (or --flow-steps), which the plan
# block reads into args.flow_steps_v3 / args.flow_steps — so it also drives the results-dir name
# (`_K{K}_`) and distinct-K runs never collide. Resolved per-seed below.

for exp in exps:
    for halfspace_variant in halfspace_variants:
        robot_name = exp.split('-')[0]
        if halfspace_variant == 'top-left-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][0]]
            obstacle_constraints = [config['obstacle_constraints'][exp][3]]
        elif halfspace_variant == 'top-right-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][1]]
            obstacle_constraints = [config['obstacle_constraints'][exp][4]]
        elif halfspace_variant == 'both-hard':
            polytopic_constraints = [config['halfspace_constraints'][exp][2], config['halfspace_constraints'][exp][3]]
            obstacle_constraints = [config['obstacle_constraints'][exp][5]]
        bounds = config['bounds'][exp]
        ax_limits = config['ax_limits'][exp]
        enlarge_constraints = config['enlarge_constraints'][robot_name]
        dt = config['dt'][robot_name]
        class Parser(utils.Parser):
            dataset: str = exp
            config: str = 'config.' + exp
        figs_all_seeds, axes_all_seeds = zip(*[plt.subplots(1, 1, figsize=(9, 10)) for _ in range(len(projection_variants))])
        figs_all_seeds = list(figs_all_seeds)
        axes_all_seeds = list(axes_all_seeds)
        for seed in seeds:
            args = Parser().parse_args(experiment='plan_fm_v3_meanflow', seed=seed)
            
            fm_model = None
            dataset = None
            env = None
            obs_indices = config['observation_indices'][robot_name]
            act_indices = config['action_indices'][robot_name]
            
            if not args_cli.aggregate_only:
                # Get model
                def load_diffusion_with_override(*loadpath, target_class=None, epoch='best', device='cuda:0', seed=None):
                    import os  # TODO: Clean up these inline imports later (os/sys are already imported globally, inspect can be moved to the top)
                    import sys
                    print(f'\n[ eval loading ] Intercepting load from {os.path.join(*loadpath)}\n')
                    dataset_config = utils.load_config(*loadpath, 'dataset_config.pkl')
                    model_config = utils.load_config(*loadpath, 'model_config.pkl')
                    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
                    trainer_config = utils.load_config(*loadpath, 'trainer_config.pkl')
                    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

                    if target_class is not None:
                        # Resolve target class using the current module's import logic
                        target_class_resolved = utils.config.import_class(target_class)
                        target_class_str = target_class_resolved.__module__ + '.' + target_class_resolved.__name__
                        pickled_class_str = diffusion_config._class.__module__ + '.' + diffusion_config._class.__name__
                        
                        if pickled_class_str != target_class_str:
                            print(f"\n=======================================================", file=sys.stderr)
                            print(f"[WARNING] Pickled diffusion class does not match existing d3il.py config!", file=sys.stderr)
                            print(f"Pickled config class: {pickled_class_str}", file=sys.stderr)
                            print(f"Existing d3il.py class: {target_class_str}", file=sys.stderr)
                            print(f"Overriding picked config with existing d3il.py config!", file=sys.stderr)
                            print(f"=======================================================\n", file=sys.stderr)
                            diffusion_config._class = target_class_resolved

                            # Safely filter _dict to only include arguments the new class accepts
                            import inspect
                            sig = inspect.signature(target_class_resolved.__init__)
                            valid_kwargs = set(sig.parameters.keys())
                            keys_to_remove = [k for k in diffusion_config._dict if k not in valid_kwargs]
                            for k in keys_to_remove:
                                print(f"[WARNING] Dropping unexpected kwarg from pickle: '{k}'", file=sys.stderr)
                                del diffusion_config._dict[k]

                    # CONFIG-OVERRIDES-PKL (fix_1, 2026-07-14): the pkl PRESERVES training-time params; the eval
                    # config is compared against it and reconciled in TWO tiers (see
                    # logs_in_develop/config_override_pkl/fix_1/):
                    #   - SAMPLING knobs (operating point, safe to change at eval): eval config OVERRIDES the pkl, [INFO].
                    #   - identity/architecture keys (must match the checkpoint): pkl value is KEPT to protect the
                    #     state_dict; a loud [WARNING] fires if the eval config disagrees.
                    _SAMPLING_OVERRIDE_KEYS = {
                        'flow_steps_v3', 'ode_inference_steps_v3', 'ode_solver_backend_v3',
                        'ode_solver_method_v3', 'ode_solver_rtol_v3', 'ode_solver_atol_v3',
                        'ode_solver_step_size_v3', 'condition_guidance_w', 'clip_denoised',
                        'diffusion_timestep_threshold',
                        # NOTE (Gen3v6): the meanflow_cfg_* keys are gone — this generation
                        # has no interval-CFG, so there is no eval-time guidance operating
                        # point to override.
                    }
                    for _k in list(diffusion_config._dict.keys()):
                        if not hasattr(args, _k):
                            continue
                        _new, _old = getattr(args, _k), diffusion_config._dict[_k]
                        try:
                            _same = bool(_new == _old)
                        except Exception:
                            _same = False
                        if _same:
                            continue
                        if _k in _SAMPLING_OVERRIDE_KEYS:
                            print(f"[ config->pkl ] INFO  {_k}: train={_old!r} -> eval={_new!r}  (sampling knob; applied)", file=sys.stderr)
                            diffusion_config._dict[_k] = _new
                        else:
                            print(f"[ config->pkl ] WARNING  {_k}: train-pkl={_old!r} vs eval-config={_new!r} -- "
                                  f"identity/architecture key; KEEPING the train value to protect the checkpoint "
                                  f"(fix the config to match the checkpoint, or retrain).", file=sys.stderr)

                    import inspect
                    print(f"\n[INFO] Instantiating Diffusion Model from:", file=sys.stderr)
                    print(f"       -> {inspect.getfile(diffusion_config._class)}\n", file=sys.stderr)

                    dataset = dataset_config()
                    model = model_config().to(device)
                    diffusion_config._dict.pop('model', None) # Prevent duplicate positional/kwarg
                    diffusion = diffusion_config(model).to(device)
                    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)

                    if epoch == 'latest':
                        epoch = utils.get_latest_epoch(loadpath)
                    trainer.load(epoch)
                    losses = utils.load_losses(*loadpath, 'losses.pkl')
                    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer.ema_model, trainer, epoch, losses)

                fm_experiment = load_diffusion_with_override(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), target_class=args.diffusion, epoch=args.diffusion_epoch, device=args.device)
                # eval_use_ema config switch. Gen3v6 DEFAULTS TO TRUE: few-step MeanFlow is
                # EMA-sensitive and the official recipe samples with EMA weights. Set False for
                # the raw-weights (dpcc-legacy) A/B. Both are loaded above regardless; this
                # only picks which is used.
                use_ema = bool(getattr(args, 'eval_use_ema', True))
                fm_model = fm_experiment.ema if use_ema else fm_experiment.diffusion
                print(f'[ eval ] weight source: {"EMA (official)" if use_ema else "raw/live (dpcc-legacy)"}')
                # ── H8+8 (U10) G1: HORIZON GUARD — abort, do not warn ────────────────────
                # `horizon` is a TRAINING property (dataset windows + per-step loss weights),
                # so evaluating a checkpoint at a horizon it was not trained on is invalid.
                # It is also SILENT on the UNet arm: ResidualTemporalBlock takes `horizon` and
                # never uses it (models/unet1d_temporal_cond.py:55-70) — the weights are Conv1d
                # + Linear, both length-agnostic — so an H8 checkpoint runs happily at H16 and
                # returns a clean-looking, meaningless number. ('mf_dit' would crash on its
                # learned pos_embed; 'dit' would silently extrapolate RoPE. Only the crash is
                # safe, and we do not rely on it.) The CONFIG-OVERRIDES-PKL reconciler above
                # only prints a WARNING for architecture keys and keeps the pkl value, which is
                # not enough: args.horizon still drives the Projector and the policy call.
                _ckpt_horizon = getattr(fm_model, 'horizon', None)
                if _ckpt_horizon is not None and int(_ckpt_horizon) != int(args.horizon):
                    raise SystemExit(
                        f'\n[ eval ] 🔴 HORIZON MISMATCH — checkpoint was trained at '
                        f'horizon={int(_ckpt_horizon)}, this eval is configured for '
                        f'horizon={int(args.horizon)}.\n'
                        f'         Loaded: {args.diffusion_loadpath}\n'
                        f'         A checkpoint can only be evaluated at its OWN horizon; '
                        f'horizon is a training property, not a sampling knob.\n'
                        f'         Set MF_HORIZON={int(_ckpt_horizon)}, or train a '
                        f'horizon={int(args.horizon)} checkpoint first.\n')
                # G2: HardFlow asserts replan_steps < horizon (run/eval.py:380-382); a plan
                # cannot supply more actions than it holds. Same check, same reason.
                if replan_steps >= int(args.horizon):
                    raise SystemExit(
                        f'\n[ eval ] 🔴 MF_REPLAN_STEPS={replan_steps} must be < horizon='
                        f'{int(args.horizon)} — a plan cannot supply more actions than it has.\n')
                # Apply plan-time solver selection after loading checkpoint config.
                fm_model.flow_steps_v3 = int(getattr(args, 'flow_steps_v3', getattr(fm_model, 'flow_steps_v3', 10)))
                fm_model.ode_inference_steps_v3 = int(getattr(args, 'ode_inference_steps_v3', getattr(fm_model, 'ode_inference_steps_v3', fm_model.flow_steps_v3)))
                fm_model.ode_solver_backend_v3 = getattr(args, 'ode_solver_backend_v3', getattr(fm_model, 'ode_solver_backend_v3', 'legacy_euler'))
                fm_model.ode_solver_method_v3 = getattr(args, 'ode_solver_method_v3', getattr(fm_model, 'ode_solver_method_v3', 'euler'))
                fm_model.ode_solver_rtol_v3 = getattr(args, 'ode_solver_rtol_v3', getattr(fm_model, 'ode_solver_rtol_v3', None))
                fm_model.ode_solver_atol_v3 = getattr(args, 'ode_solver_atol_v3', getattr(fm_model, 'ode_solver_atol_v3', None))
                fm_model.ode_solver_step_size_v3 = getattr(args, 'ode_solver_step_size_v3', getattr(fm_model, 'ode_solver_step_size_v3', None))
                # Gen3v6 U3 — matched-K. K arrives via the config (args.flow_steps / flow_steps_v3,
                # both reading HFFM_FLOW_STEPS), so args.savepath already encodes _K{K}_ (distinct-K
                # runs never collide). Force it onto the native sampler too — line 185's getattr can
                # otherwise pick up a stale checkpoint ode_inference_steps_v3 instead of this K.
                flow_steps = int(getattr(args, 'flow_steps', 0)) or int(getattr(fm_model, 'flow_steps_v3', 10))
                fm_model.ode_inference_steps_v3 = flow_steps
                fm_model.flow_steps_v3 = flow_steps
                print(f'[ eval ] matched K (flow_steps) = {flow_steps} for every arm (savepath: {args.savepath})')

                # ── U10.1 RUN PROVENANCE ──────────────────────────────────────────────
                # The config snapshot copies avoiding-d3il.py verbatim, so since U10 it
                # reads `'horizon': _mf_horizon` — identical bytes for an H8 and an H16
                # run. And `Parser.save` writes args.json for TRAIN only (setup.py:85),
                # so an eval otherwise records its resolved values nowhere. Written here
                # (not at parse time) because this is the first point where every knob is
                # final: K is matched onto the sampler and the checkpoint is loaded, so
                # `horizon` below is the CHECKPOINT's, already agreed with args by G1.
                provenance.write(
                    args.savepath, role='eval',
                    yaml_path=_cfg_path,
                    resolved={
                        'horizon': int(args.horizon),
                        'checkpoint_horizon': int(getattr(fm_model, 'horizon', -1)),
                        'imf_backbone': getattr(args, 'imf_backbone', None),
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
                    })

                dataset = fm_experiment.dataset
                if 'pointmaze' in exp or 'antmaze' in exp:
                    minari_dataset = minari.load_dataset(exp, download=True)
                    env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()
                elif 'avoiding' in exp:
                    env = ObstacleAvoidanceEnv()
                    env.start()
                if robot_name == 'pointmaze': env.env.env.env.point_env.frame_skip = 2
                if robot_name == 'antmaze': env.env.env.env.ant_env.frame_skip = 5
                
                if fm_model.__class__.__name__ in ['FlowMatchingIMF', 'MeanFlowODE']:
                    trajectory_dim = fm_model.observation_dim + fm_model.action_dim - fm_model.goal_dim if hasattr(fm_model, 'observation_dim') else fm_model.transition_dim - fm_model.goal_dim
                    action_dim = fm_model.action_dim
                    fm_variant = 'states_actions'
                    obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
                    act_obs_indices = {**act_indices, **obs_indices_updated}
                else:
                    trajectory_dim = fm_model.observation_dim - fm_model.goal_dim
                    action_dim = 0
                    fm_variant = 'states'
                    act_obs_indices = obs_indices
                constraint_list = []
                constraint_list_tightened = []
                constraint_list_polytopic_not_tightened = []
                if 'halfspace' in constraint_types:
                    for constraint in polytopic_constraints:
                        constraint_list.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
                        constraint_list_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices)))
                        constraint_list_polytopic_not_tightened.append(('ineq', utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
                if 'bounds' in constraint_types:
                    lower_bound, upper_bound = utils.formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices)
                    constraint_list.extend([['lb', lower_bound], ['ub', upper_bound]])
                    constraint_list_tightened.extend([['lb', lower_bound], ['ub', upper_bound]])
                if 'obstacles' in constraint_types:
                    for constr in obstacle_constraints:
                        constraint_list.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius']])
                        constraint_list_tightened.append([constr['type'], [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]], constr['center'], constr['radius'] + enlarge_constraints])
                constraint_list_without_prior = copy(constraint_list)
                constraint_list_without_prior_tightened = copy(constraint_list_tightened)
                dynamics_constraints = []
                if 'dynamics' in constraint_types: dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices, action_dim)
                for constraint in dynamics_constraints:
                    constraint_list.append(constraint)
                    constraint_list_tightened.append(constraint)
                env_seeds = config['env_seeds'][exp] if 'pointmaze-umaze' in exp else np.arange(100)

            for variant_idx, variant in enumerate(projection_variants):
                save_path = f'{args.savepath}/results/halfspace_{halfspace_variant}' if 'avoiding' in exp else f'{args.savepath}/results'
                os.makedirs(save_path, exist_ok=True)
                
                if args_cli.aggregate_only:
                    # LOAD DATA MODE
                    npz_path = os.path.join(save_path, f'{variant}.npz')
                    if not os.path.exists(npz_path):
                        print(f'[ eval ] skipping {variant} for seed {seed}, no results found at {npz_path}')
                        continue
                    print(f'[ eval ] Aggregating existing results for {variant} - seed {seed}')
                    data = np.load(npz_path, allow_pickle=True)
                    # Use saved obs_all for aggregation plot
                    if 'obs_all' in data:
                        obs_all = data['obs_all']
                        # Re-plot on aggregate axes
                        for i in range(min(len(obs_all), plot_how_many)):
                            obs_buffer = obs_all[i]
                            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                            axes_all_seeds[variant_idx].plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], colors[seed % len(colors)], linewidth=2)
                    continue

                # INFERENCE MODE
                log_file = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
                original_stdout = sys.stdout
                sys.stdout = Tee(sys.stdout, log_file)
                
                try:
                    is_hardflow = variant.startswith('hardflow')
                    print(f'------------------------Running {exp} - {halfspace_variant} - {variant} ({seed}) - K={flow_steps}----------------------------')
                    # [Gen0fix2] Restore the per-variant threshold deleted by upstream DPCC commit 7f09d3a.
                    # threshold 0 => the activation gate fires only on the FINAL step, i.e. ONE projection
                    # applied after the last denoising/ODE step -- the paper's definition of post-processing
                    # ("modifying them after the last denoising step, usually by solving an optimization
                    # problem"). Without it, `post_processing` inherits the normal schedule and is a
                    # byte-identical duplicate of `dpcc-r`.
                    threshold = 0.0 if 'post_processing' in variant else diffusion_timestep_threshold

                    gradient = True if 'gradient' in variant else False
                    if 'model_free' in variant and 'tightened' in variant:
                        constraints = constraint_list_without_prior_tightened
                    elif 'model_free' in variant and not 'tightened' in variant:
                        constraints = constraint_list_without_prior
                    elif not 'model_free' in variant and 'tightened' in variant:
                        constraints = constraint_list_tightened
                    else:
                        constraints = constraint_list
                    delta_t = dt
                    if 'dt0p25' in variant:
                        delta_t = 0.25 * dt
                    elif 'dt0p5' in variant:
                        delta_t = 0.5 * dt
                    elif 'dt2p0' in variant:
                        delta_t = 2.0 * dt
                    elif 'dt4p0' in variant:
                        delta_t = 4.0 * dt
                    if is_hardflow:
                        # ---------------- arm C (HardFlow) — verbatim Gen12 construction ----------------
                        # 🔴 B4_PARITY (2026-08-20) — the candidate fan is resolved PER VARIANT, not per run:
                        #   `hardflow_new`          -> 1              faithful upstream batch-1 control
                        #   `hardflow_new-r/-c/-t`  -> hf_batch_size  (default 4 == args.batch_size, arms A/B)
                        # Before this, EVERY arm-C variant took the yaml's `hardflow.batch_size` (which defaulted
                        # to 1) while arms A/B took args.batch_size (4). Both arms loop SERIALLY over candidates
                        # around their CPU solve, so that was a 4x compute discount for arm C — and it read as a
                        # HardFlow speedup in every timing table. See logs_in_develop/HF_Batch_Parity/.
                        batch_size = resolve_hf_batch_size(variant, hf_batch_size)
                        if batch_size != args.batch_size:
                            print(f'[ hardflow ] ⚠️  arm-C fan B={batch_size} != DPCC-arm fan B={args.batch_size} '
                                  f'for {variant!r} — wall-clock is NOT comparable across arms for this variant.')
                        # DPCC-parity selection from the variant suffix; strip '-tightened' FIRST so the
                        # selection suffix composes with the geometry (hardflow_new-c-tightened → min-cost
                        # AND enlarged). At batch_size==1 all of -r/-c/-t collapse to index 0.
                        _sel_base = variant.replace('-tightened', '')
                        hf_selection = 'random'
                        if _sel_base.endswith('-t'): hf_selection = 'temporal_consistency'
                        elif _sel_base.endswith('-c'): hf_selection = 'minimum_projection_cost'
                        # 🔴 B4_PARITY follow-up — `-c` IS NOT TRUSTWORTHY AT B>1 (open, not fixed here).
                        # Pooled over the five 08-11..08-19 avoiding batches, 750 arm-C cells that DID run at
                        # B=4:  -r  S&C 0.707 / succ 0.917 / 67.7 steps /   0 timeouts
                        #       -t  S&C 0.707 / succ 0.883 / 71.2 steps /   5 timeouts
                        #       -c  S&C 0.443 / succ 0.540 / 138.5 steps / 370 timeouts  (49%)
                        # `candidate_costs` is Σ_k ||x1_proj − x1_ref||², so argmin picks the candidate the NLP
                        # barely had to touch — on `avoiding` that is the candidate that barely MOVES, which
                        # stalls the episode. DPCC's own -c does not degenerate this way. Until the ranking key
                        # is fixed, treat arm-C `-c` numbers at B>1 as suspect. See logs_in_develop/HF_Batch_Parity/.
                        if hf_selection == 'minimum_projection_cost' and batch_size > 1:
                            print(f'[ hardflow ] 🔴 {variant}: `-c` selection at B={batch_size} is a KNOWN-BAD arm '
                                  f'(49% timeouts across 750 B=4 cells). Reported for completeness; do not cite '
                                  f'without re-checking. See logs_in_develop/HF_Batch_Parity/.')
                        policy = HardFlowPolicy(
                            model=fm_model, normalizer=dataset.normalizer, horizon=args.horizon,
                            transition_dim=trajectory_dim, action_dim=action_dim,
                            constraint_list=constraints, dt=delta_t, flow_steps=flow_steps,
                            preprocess_fns=args.preprocess_fns, test_ret=args.test_ret,
                            reg_scale=float(hardflow_cfg.get('reg_scale', 1.0)),
                            activation_threshold=hf_act_threshold,
                            trajectory_selection=hf_selection,
                            candidate_cost=hf_candidate_cost,
                            dynamics_mode=hardflow_cfg.get('dynamics_mode', 'deriv'),
                            linear_dynamics=None,
                            print_level=int(hardflow_cfg.get('ipopt_print_level', 0)),
                            print_time=bool(hardflow_cfg.get('casadi_print_time', False)),
                            # fix_4: arm C must start from the SAME noise law as arms A/B.
                            # Gen3v6's MeanFlow sampler is sigma=1.0 (mf_diffusion.py:204);
                            # the U3 port had inherited Gen12's 0.5. Stated explicitly here
                            # so a generation swap can't reintroduce it silently.
                            init_noise_scale=1.0,
                            device=args.device, goal_dim=fm_model.goal_dim)
                    else:
                        # ---------------- arms A / B (diffuser / DPCC) — unchanged ----------------
                        batch_size = args.batch_size
                        projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=fm_model.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, gradient=gradient, gradient_weights=[1, 0.5, 2], variant=fm_variant, dt=delta_t, cost_dims=None, device=args.device, solver='scipy',
                                                diffusion_timestep_threshold=threshold)   # [Gen0fix2] post_processing override
                        projector = None if variant == 'diffuser' else projector
                        trajectory_selection = 'random'
                        if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                        if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
                        policy = Policy(model=fm_model, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, test_ret=args.test_ret, projector=projector, trajectory_selection=trajectory_selection)
                    # 🔵 U10 — hand the cadence to BOTH policy classes (they expose the same
                    # attribute). The policy does not loop; it needs this only so the
                    # temporal-consistency (-t) selection compares plans at the right shift.
                    policy.replan_steps = replan_steps
                    fig, ax = plt.subplots(min(n_trials, plot_how_many), 6, figsize=(30, 5 * min(n_trials, plot_how_many)), squeeze=False)
                    fig.suptitle(f'{exp} - {variant}')
                    save_samples_every = 1  # fix_1: save full-resolution MPC foresight every step (was: args.horizon // 2)
                    plot_samples_every = max(1, args.horizon // 2)  # fix_1.2: keep PLOT readable - draw foresight fan only every H/2 steps (npz still saves every step)
                    sampled_trajectories_all = []
                    n_success = np.zeros(n_trials)
                    n_success_and_constraints = np.zeros(n_trials)
                    n_steps = np.zeros(n_trials)
                    n_violations = np.zeros(n_trials)
                    total_violations = np.zeros(n_trials)
                    avg_time = np.zeros(n_trials)
                    collision_free_completed = np.ones(n_trials)
                    pos_tracking_errors = np.zeros((n_trials, args.max_episode_length - 1))
                    # Gen3v6 U3 — HardFlow-arm metrics (0 for arms A/B).
                    nfe_total = 0
                    nlp_solves_total = 0
                    nlp_failures_total = 0

                    obs_all = []
                    act_all = []

                    fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)), squeeze=False)

                    for i in range(n_trials):
                        torch.manual_seed(i)
                        env_seed = env_seeds[i] if ('pointmaze-umaze' in exp) else i
                        if 'avoiding' in exp:
                            obs = env.reset()
                            action = env.robot_state()[:2]
                            fixed_z = env.robot_state()[2:]
                        else:
                            obs, _ = env.reset(seed=env_seed)
                        if 'pointmaze' in exp:
                            obs = np.concatenate((obs['observation'], obs['desired_goal']))
                        elif 'antmaze' in exp:
                            obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                        elif 'avoiding' in exp:
                            obs = np.concatenate((action[:2], obs))
                        obs_buffer = []
                        action_buffer = []
                        sampled_trajectories = []
                        disable_projection = False
                        # 🔵 U10 — per-episode replan state. `plan_idx` counts how many actions
                        # of the current plan have been consumed; at replan_steps=1 the cache is
                        # refilled on every step, so these are inert. Reset per trial so no plan
                        # ever leaks across an env reset.
                        plan_actions = None
                        plan_obs = None
                        plan_idx = 0
                        # REAL_TIME_RECORDING_UPDATE — one recorder per rollout episode.
                        rt_rec = RTRecorder(episode_id=f'{exp}_{variant}_seed{seed}_trial{i}',
                                            variant=variant, scene=exp,
                                            system='FMv3_MeanFlow',
                                            control_hz=RT_CONTROL_HZ,
                                            batch_size=batch_size, horizon=args.horizon,
                                            text_log=config.get('write_to_file', True))
                        for _ in range(args.max_episode_length):
                            violated_this_timestep = 0
                            if 'halfspace' in constraint_types:
                                for constraint in constraint_list_polytopic_not_tightened:
                                    if constraint[0] == 'ineq':
                                        c, d = constraint[1]
                                        obs_to_check = obs[:-fm_model.goal_dim] if fm_model.goal_dim > 0 else obs
                                        if obs_to_check @ c[action_dim:] >= d:
                                            violated_this_timestep = 1
                                            total_violations[i] += obs_to_check @ c[action_dim:] - d
                                            collision_free_completed[i] = 0
                            if 'obstacles' in constraint_types:
                                for constraint in obstacle_constraints:
                                    if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center']) < constraint['radius']:
                                        violated_this_timestep = 1
                                        total_violations[i] += constraint['radius'] - np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - constraint['center'])
                                        collision_free_completed[i] = 0
                            if _ > 0 and 'bounds' in constraint_types:
                                act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                                total_violations[i] += np.sum(np.maximum(0, act_obs - upper_bound)) + np.sum(np.maximum(0, lower_bound - act_obs))
                            n_violations[i] += violated_this_timestep
                            # ── 🔵 U10 RECEDING HORIZON ────────────────────────────────────
                            # Replan when the cache is empty or exhausted; otherwise replay the
                            # next action of the plan already in hand. At replan_steps=1 the
                            # condition is true on EVERY step, `action` comes straight from
                            # policy() and nothing below this block changes — byte-identical to
                            # the pre-U10 loop.
                            _replanned = (plan_actions is None) or (plan_idx >= replan_steps)
                            if _replanned:
                                start = time.time()
                                action, samples = policy(conditions={0: obs}, batch_size=batch_size, horizon=args.horizon, disable_projection=disable_projection)
                                _rt_total_ms = (time.time() - start) * 1e3   # REAL_TIME_RECORDING_UPDATE — bundled FM+projection wall-time
                                avg_time[i] += _rt_total_ms / 1e3
                                # Gen3v6 U3 — accumulate HardFlow-arm metrics (nlp solves/failures per plan).
                                if is_hardflow:
                                    nlp_solves_total += policy.last_info.get('nlp_solves', 0)
                                    nlp_failures_total += policy.last_info.get('nlp_failures', 0)
                                # Cache the EXECUTED candidate's plan (the policy publishes it;
                                # `which_trajectory` is not visible here, so `samples.actions[0]`
                                # would be the wrong candidate under -c/-t selection).
                                plan_actions = getattr(policy, 'last_executed_actions', None)
                                plan_obs = getattr(policy, 'last_executed_observations', None)
                                if replan_steps > 1 and plan_actions is None:
                                    raise SystemExit(
                                        f'\n[ eval ] 🔴 replan_steps={replan_steps} needs the executed plan, but '
                                        f'{type(policy).__name__} did not publish `last_executed_actions`.\n')
                                plan_idx = 0
                            else:
                                # No compute this step: the plan was paid for when it was made.
                                _rt_total_ms = 0.0
                                action = plan_actions[plan_idx]
                            plan_idx += 1
                            # REAL_TIME_RECORDING_UPDATE — record per-step timing (proj bundled inside policy()).
                            rt_rec.step(t=_ / RT_CONTROL_HZ, total_ms=_rt_total_ms, obs=obs,
                                        action=action, pos=obs[[obs_indices['x'], obs_indices['y']]],
                                        proj_active=(variant != 'diffuser' and not disable_projection and _replanned),   # 🔵 U10: no plan, no projection
                                        contact=bool(violated_this_timestep), step_idx=_)
                            if 'avoiding' in exp:
                                next_pos_des = action + obs[:2]
                                obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                                success = info[1]
                            else:
                                obs, rew, terminated, truncated, info = env.step(action)
                                success = info['success']
                            if 'pointmaze' in exp:
                                obs = np.concatenate((obs['observation'], obs['desired_goal']))
                            elif 'antmaze' in exp:
                                obs = np.concatenate((obs['achieved_goal'], obs['observation'], obs['desired_goal']))
                            elif 'avoiding' in exp:
                                obs = np.concatenate((next_pos_des[:2], obs))
                            if _ >= 1:
                                pos_tracking_errors[i, _-1] = np.linalg.norm(obs[obs_indices['x']:obs_indices['y']+1] - desired_next_pos)
                            # 🔵 U10 — the tracking reference is the NEXT state of the plan being
                            # executed. At replan_steps=1 that is the fresh plan's step 1, i.e.
                            # the original expression, untouched. Under replan>1 the plan is
                            # `plan_idx` steps in, so the reference walks along the cached plan
                            # instead of freezing on step 1 of a plan made several steps ago.
                            if replan_steps == 1 or plan_obs is None:
                                desired_next_pos = samples.observations[0, 1, [obs_indices['x'], obs_indices['y']]]
                            else:
                                _ref_k = min(plan_idx, plan_obs.shape[0] - 1)
                                desired_next_pos = plan_obs[_ref_k, [obs_indices['x'], obs_indices['y']]]
                            if _ % save_samples_every == 0:
                                sampled_trajectories.append(samples.observations[:, :, :])
                            obs_buffer.append(obs)
                            action_buffer.append(action)
                            if success: n_success[i] = 1
                            if (terminated or _ == args.max_episode_length - 1) and (not success): collision_free_completed[i] = 0
                            if success or terminated or _ == args.max_episode_length - 1:
                                n_steps[i] = _
                                avg_time[i] /= _
                                if success and collision_free_completed[i]: n_success_and_constraints[i] = 1
                                break
                        
                        obs_all.append(np.array(obs_buffer))
                        act_all.append(np.array(action_buffer))
                        # REAL_TIME_RECORDING_UPDATE — write per-episode realtime_<variant>_trial<i>.log + SUMMARY.
                        if config.get('write_to_file', True):
                            rt_rec.save(f'{save_path}/realtime_{variant}_trial{i}.log',
                                        behaviour={'success': int(n_success[i]),
                                                   'n_steps': int(n_steps[i]),
                                                   'violations': int(n_violations[i])})

                        sampled_trajectories_all.append(sampled_trajectories)
                        if i >= plot_how_many: continue
                        plot_states = ['x', 'y', 'x_des', 'y_des']
                        for j in range(len(plot_states)):
                            ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                            ax[i, j].set_title(plot_states[j])
                        axes = [ax[i, 4], ax_all[i, variant_idx]]
                        for curr_ax in axes:
                            curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], 'k')
                            curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']], np.array(obs_buffer)[0, obs_indices['y']], 'go', label='Start')
                            curr_ax.set_xlim(ax_limits[0])
                            curr_ax.set_ylim(ax_limits[1])
                        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                        axes_all_seeds[variant_idx].plot(np.array(obs_buffer)[:, obs_indices['x']], np.array(obs_buffer)[:, obs_indices['y']], colors[seed % len(colors)], linewidth=2)
                        axes = [ax[i, 5], ax_all[i, variant_idx]]
                        for __ in range(0, len(sampled_trajectories_all[i]), plot_samples_every):
                            # 🔴 fix_7 — iterate the LOCAL batch, not args.batch_size. Arm C
                            # overrides it (batch_size = hf_batch_size, :316) and the yaml
                            # default is 1, so with HFFM_BATCH unset this asked for index 1 of
                            # a 1-row candidate array -> IndexError. Arms A/B are unaffected
                            # (batch_size = args.batch_size, :345). Restores parity with the
                            # alphaflow/hardflow siblings, which already read `batch_size`.
                            for ___ in range(min(batch_size, 4)):
                                for curr_ax in axes:
                                    curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['y']], 'b')
                                    curr_ax.plot(sampled_trajectories_all[i][__][___, 0, obs_indices['x']], sampled_trajectories_all[i][__][___, 0, obs_indices['y']], 'go', label='Start')
                        ax[i, 5].set_xlim(ax_limits[0])
                        ax[i, 5].set_ylim(ax_limits[1])
                        axes = [ax[i, 4], ax[i, 5], ax_all[i, variant_idx]]
                        for curr_ax in axes:
                            utils.plot_environment_constraints(exp, curr_ax)
                            if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(exp, polytopic_constraints, curr_ax, ax_limits)
                            if 'obstacles' in constraint_types:
                                for constraint in obstacle_constraints:
                                    curr_ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
                    print(f'Success rate: {np.mean(n_success)}')
                    print(f'Constraints satisfied: {np.mean(collision_free_completed)}')
                    print(f'Success rate (goal and constraints): {np.mean(n_success_and_constraints)}')
                    print(f'Avg number of steps: {(np.mean(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f} +- {(np.std(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f}')
                    print(f'Avg number of constraint violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
                    print(f'Avg total violation: {np.mean(total_violations):.3f} +- {np.std(total_violations):.3f}')
                    print(f'Average computation time per step: {np.mean(avg_time):.3f}')
                    if variant == 'diffuser': print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')
                    # Gen3v6 U3 — HardFlow-arm metric summary (nfe accumulated on the sampler).
                    if is_hardflow:
                        nfe_total = int(getattr(policy, 'nfe', 0))
                        print(f'[hardflow] NFE={nfe_total}  NLP solves={nlp_solves_total}  '
                              f'NLP failures={nlp_failures_total}  batch(mpc)={batch_size}  '
                              f'act_threshold={hf_act_threshold}')

                    if config['write_to_file']:
                        _hf_budget = (hardflow_step_budget(flow_steps, hf_act_threshold)
                                        if is_hardflow and flow_steps else (0, 0))
                        np.savez(f'{save_path}/{variant}.npz',
                                 n_success=n_success,
                                 n_success_and_constraints=n_success_and_constraints,
                                 n_steps=n_steps,
                                 n_violations=n_violations,
                                 total_violations=total_violations,
                                 avg_time=avg_time,
                                 collision_free_completed=collision_free_completed,
                                 args=args,
                                 # Gen3v6 U3 — HardFlow-arm metrics (0 for arms A/B).
                                 is_hardflow=bool(is_hardflow),
                                 nfe_total=int(nfe_total),
                                 nlp_solves_total=int(nlp_solves_total),
                                 nlp_failures_total=int(nlp_failures_total),
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
                                 sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))
                    fig.savefig(f'{save_path}/{variant}.png')
                    plt.close(fig)
                    ax_all[0, variant_idx].set_title(variant)
                    
                finally:
                    sys.stdout = original_stdout
                    log_file.close()
            
            if not args_cli.aggregate_only:
                fig_all.savefig(f'{save_path}/all.png')
                plt.close(fig_all)   # fix_7: was the only figure never closed -> "More than 20
                                     # figures have been opened" once the variant list grew to 13.
                env.close()
        
        # Save aggregate plots for all seeds
        path = f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}'
        os.makedirs(path, exist_ok=True)
        for variant_idx, (fig, ax) in enumerate(zip(figs_all_seeds, axes_all_seeds)):
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
            ax.set_facecolor([1, 1, 0.9])
            utils.plot_environment_constraints(exp, ax)
            if 'halfspace' in constraint_types: utils.plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits, enlarge_constraints=enlarge_constraints)
            if 'obstacles' in constraint_types:
                for constraint in obstacle_constraints:
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'], color='b', alpha=0.2))
                    ax.add_patch(matplotlib.patches.Circle(constraint['center'], constraint['radius'] + enlarge_constraints, color='b', alpha=0.1, linestyle='--'))
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.png', bbox_inches='tight')
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.pdf', bbox_inches='tight', format='pdf')
            plt.close(fig)
        
        if not args_cli.aggregate_only:
            plt.show()
