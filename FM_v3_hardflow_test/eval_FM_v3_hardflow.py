# Gen12 — FM_v3 evaluation with a third guidance arm: HardFlow's `hardflow_new`.
#
# Copy-modify of FM_v3_test/eval_FM_v3.py. The MPC loop, the metrics and the
# plots are unchanged, so arms A/B/C are measured by identical code:
#
#   arm A  'diffuser'      unguided FMv3 ODE                    (field quality floor)
#   arm B  'dpcc-*'        DPCC Projector, post-hoc per step    (the incumbent)
#   arm C  'hardflow_new'  in-loop constrained sampling         (the contribution)
#
# PLAN §5: K is NOT a free axis — it sets both NFE and the number of NLP solves.
# `flow_steps` in the YAML overrides it for EVERY arm at once, so a matched-K
# comparison is the default rather than something the operator has to remember.
import argparse
import glob
import json
import os
import sys
import time
from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
import yaml

import flow_matcher_v3_hardflow.utils as utils
from flow_matcher_v3_hardflow.sampling.policies import Policy
from diffuser.utils import provenance   # U10.1 — env-override provenance (shared)
from flow_matcher_v3_hardflow.sampling.projection import Projector
from flow_matcher_v3_hardflow.sampling.hardflow_projection import (
    HardFlowPolicy, resolve_activation_threshold, resolve_hf_batch_size,
    hardflow_step_budget,          # HFK1 (2026-08-24)
    # [SolverSwap] artifact naming — keeps an SLSQP run from overwriting IPOPT data.
    artifact_variant_label, resolve_nlp_backend)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_paths  # noqa: E402  (fix_5 FMv3ODE-style output paths)
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

parser = argparse.ArgumentParser(description='Gen12 HardFlow-into-FMv3 evaluation.')
parser.add_argument('--seed', type=int, help='Run only this seed.')
parser.add_argument('--flow-steps', type=int, help='Override K for every arm.')
parser.add_argument('--config', default='config/hardflow_projection_eval.yaml')
args_cli, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

with open(args_cli.config, 'r') as file:
    config = yaml.safe_load(file)

exps = config['exps']
seeds = [args_cli.seed] if args_cli.seed is not None else config['seeds']
projection_variants = config['projection_variants']
halfspace_variants = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials = config['n_trials']
plot_how_many = config['plot_how_many']
constraint_types = config['constraint_types']
hardflow_cfg = config.get('hardflow', {})
FORCE_OVERWRITE = os.environ.get('FORCE_OVERWRITE', '0') == '1'
# ── arm C (hardflow_new) knobs, resolved once ────────────────────────────────
# U4 + fix_6: late-activation threshold in DPCC polarity (higher = MORE projection;
# 1.0 = every step, 0.5 = last half, 0.0 = terminal-only). Accepts a float in [0,1] or
# the alias 'all'(=1.0)/'late'(=0.5). HFFM_ACT_THRESHOLD env overrides for sweeps.
hf_act_threshold = resolve_activation_threshold(
    os.environ.get('HFFM_ACT_THRESHOLD',
                   hardflow_cfg.get('activation_threshold',
                                    hardflow_cfg.get('activation', 1.0))))
# U4.2: candidate fan + selection. batch_size>1 fans candidates; selection rule comes
# from the variant suffix (hardflow_new-c/-r/-t), like DPCC.
# 🔴 B4_PARITY (2026-08-20) — the run-level arm-C fan. Default is 4 (was 1), i.e. the DPCC
# arms' `batch_size`, because both arms loop serially over candidates around their CPU solve
# and a mismatched fan makes arm-B-vs-arm-C wall-clock comparisons void. This is the fan the
# SELECTION variants (-r/-c/-t) get; bare `hardflow_new` is pinned to 1 by
# resolve_hf_batch_size(). A yaml with no `hardflow.batch_size` key now also lands on 4.
hf_batch_size = int(os.environ.get('HFFM_BATCH', hardflow_cfg.get('batch_size', 4)))
hf_candidate_cost = hardflow_cfg.get('candidate_cost', 'prox')
# [Gen12fix8] DPCC threshold was ORPHANED CONFIG. `diffusion_timestep_threshold` exists in
# config/hardflow_projection_eval.yaml (copied verbatim from config/projection_eval.yaml) but
# was never read here, so arms A/B silently used Projector's hardcoded default of 0.5 — the
# YAML knob did nothing. The FMv3ODE sibling reads it correctly
# (FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py:54 -> Projector(...:242));
# Gen12's port dropped both lines. Restored here + passed to Projector below.
# Harmless so far (YAML said 0.5 == the default), but it blocks any theta != 0.5 sweep.
# Env override DPCC_THRESHOLD added for parity with HFFM_ACT_THRESHOLD on the arm-C side.
dpcc_threshold = float(os.environ.get('DPCC_THRESHOLD',
                                      config.get('diffusion_timestep_threshold', 0.5)))
# `checkpoint_dir` and `flow_steps` now live in the plan_fm_v3_hardflow block in
# config/avoiding-d3il.py (read from `args` inside the seed loop), so the eval has a
# single tidy control entry. CLI `--flow-steps N` still overrides the block's K.
flow_steps_cli = args_cli.flow_steps

# ── MPC CANDIDATE FAN, arms A/B — the SECOND fan, until now unsettable (Gen3v6 sync) ─────
# Gen12 has TWO independent candidate fans and only one of them used to be reachable:
#   arms A/B (`diffuser`, `dpcc-*`) -> args.batch_size, from the plan_fm_v3_hardflow block.
#                                      Was a hardcoded 4; config/avoiding-d3il.py now reads
#                                      FMPCC_MPC_BATCH.
#   arm C    (`hardflow_new-*`)     -> hf_batch_size above (HFFM_BATCH / hardflow.batch_size),
#                                      with bare `hardflow_new` pinned to 1 by
#                                      resolve_hf_batch_size().
# Read here to build the path tag and to report it — the VALUE is consumed by the config
# module, which is why this has to happen before the first Parser().parse_args().
mpc_batch = int(os.environ.get('FMPCC_MPC_BATCH', 4))
if mpc_batch < 1:
    raise ValueError(f'FMPCC_MPC_BATCH must be >= 1, got {mpc_batch}')
if mpc_batch != hf_batch_size:
    print(f'[ eval ] ⚠️  arms A/B fan (mpc={mpc_batch}) != arm-C fan (HFFM_BATCH={hf_batch_size}) '
          f'-- both arms solve SERIALLY per candidate, so avg_time is not comparable across arms '
          f'(B4_PARITY). Set FMPCC_MPC_BATCH=HFFM_BATCH unless the mismatch is the experiment.')
# 🔴 PATH COLLISION GUARD — hf_paths.eval_name()'s `mpc<N>` token has ALWAYS described arm C
# only (it is fed hf_batch_size); arms A/B were an invisible constant 4. So an mpc=1 arms-A/B
# run would land in the very same `K…_mpc1_n…` directory as the historic B1 runs, whose DPCC
# arms ran at 4 — different controllers, one folder. Gen12 builds savepath itself and never
# applies config's custom_msg, so the tag is applied to the eval-name below instead. A
# non-default fan auto-tags itself; an explicit FMPCC_RUN_MSG always wins.
if mpc_batch != 4 and not os.environ.get('FMPCC_RUN_MSG'):
    os.environ['FMPCC_RUN_MSG'] = f'mpc{mpc_batch}'
    print(f'[ eval ] non-default mpc fan -> auto-tagged results path with '
          f'FMPCC_RUN_MSG=mpc{mpc_batch} (set it yourself to override)')
run_msg = hf_paths.sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
print(f'[ eval ] mpc fan: arms A/B={mpc_batch}, arm C={hf_batch_size}'
      + (f'  |  run_msg={run_msg}' if run_msg else ''))


def load_diffusion_with_override(*loadpath, target_class=None, epoch='best', device='cuda:0'):
    """Gen3v2's interceptor: the pickled config names the class that TRAINED the
    checkpoint (`flow_matcher_v3.models.diffusion.GaussianDiffusion`). Gen12 is a
    sibling package, so without this the loaded object would come from the ORIGINAL
    folder and silently ignore every Gen12 edit."""
    import inspect
    print(f'\n[ eval loading ] Intercepting load from {os.path.join(*loadpath)}\n')
    dataset_config = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    if target_class is not None:
        target = utils.config.import_class(target_class)
        target_str = f'{target.__module__}.{target.__name__}'
        pickled_str = f'{diffusion_config._class.__module__}.{diffusion_config._class.__name__}'
        if pickled_str != target_str:
            print(f'[WARNING] Pickled diffusion class {pickled_str} != config class '
                  f'{target_str}; overriding with the config class.', file=sys.stderr)
            diffusion_config._class = target
            valid = set(inspect.signature(target.__init__).parameters.keys())
            for key in [k for k in diffusion_config._dict if k not in valid]:
                print(f"[WARNING] Dropping unexpected kwarg from pickle: '{key}'", file=sys.stderr)
                del diffusion_config._dict[key]

    print(f'[INFO] Instantiating diffusion model from {inspect.getfile(diffusion_config._class)}',
          file=sys.stderr)
    dataset = dataset_config()
    model = model_config().to(device)
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)
    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    losses = utils.load_losses(*loadpath, 'losses.pkl')
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, losses)


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
            args = Parser().parse_args(experiment='plan_fm_v3_hardflow', seed=seed)
            # checkpoint_dir + eval K come from the plan block (config/avoiding-d3il.py);
            # CLI --flow-steps overrides K.
            checkpoint_dir = getattr(args, 'checkpoint_dir', None)
            flow_steps_override = flow_steps_cli if flow_steps_cli is not None else getattr(args, 'flow_steps', None)
            # ── Which checkpoint to load ─────────────────────────────────────────────
            # Gen12 loads the FMv3ODE checkpoint (FlowMatchingODE). Both paths load the
            # pickle's OWN class natively (target_class=None): FlowMatchingODE lives in the
            # flow_matcher_v3_ode_selectable package, not here, so no override is possible
            # or wanted.
            #   checkpoint_dir None (default) -> templated FMv3ODE loadpath (relative)
            #   checkpoint_dir set            -> that absolute <checkpoint_dir>/<seed>/
            target_class = None
            if checkpoint_dir:
                loadpath_parts = (checkpoint_dir, str(seed))
            else:
                loadpath_parts = (args.loadbase, args.dataset, args.diffusion_loadpath, str(seed))
            seed_dir = os.path.join(*loadpath_parts)
            # Warn + skip on a missing/incorrect path instead of crashing the whole job.
            if not os.path.isdir(seed_dir) or not glob.glob(os.path.join(seed_dir, 'state_*.pt')):
                print('=' * 80, file=sys.stderr)
                print(f'[ WARNING ] No checkpoint for seed {seed} — looked in:', file=sys.stderr)
                print(f'            {seed_dir}', file=sys.stderr)
                if checkpoint_dir:
                    print(f'            checkpoint_dir = {checkpoint_dir}', file=sys.stderr)
                    print('            Fix `checkpoint_dir` in the plan_fm_v3_hardflow block', file=sys.stderr)
                    print('            (config/avoiding-d3il.py), or check the seed exists.', file=sys.stderr)
                else:
                    print('            checkpoint_dir is None and the templated loadpath was not', file=sys.stderr)
                    print('            found. Set `checkpoint_dir` in the plan_fm_v3_hardflow block', file=sys.stderr)
                    print('            in config/avoiding-d3il.py to a real path.', file=sys.stderr)
                print(f'            SKIPPING seed {seed}.', file=sys.stderr)
                print('=' * 80, file=sys.stderr)
                continue
            # Get model
            fm_experiment = load_diffusion_with_override(*loadpath_parts, target_class=target_class, epoch=args.diffusion_epoch, device=args.device)
            fm_model = fm_experiment.diffusion
            dataset = fm_experiment.dataset
            # Trust the checkpoint's own horizon (a direct-path model may differ from the
            # plan block's default of 8).
            args.horizon = fm_model.horizon
            # ⭐ PLAN §5: one knob sets K for all three arms. Gen13's central error was
            # comparing arms at different K; making it a single override removes the
            # chance of repeating it.
            if flow_steps_override is not None:
                fm_model.flow_steps_v3 = int(flow_steps_override)
                fm_model.ode_inference_steps_v3 = int(flow_steps_override)
            flow_steps = int(fm_model.flow_steps_v3)
            print(f'[ eval ] matched K (flow_steps_v3) = {flow_steps} for every arm')
            # fix_5: FMv3ODE-style path — <train-name>/<eval-name>/<seed>/results/...
            # The eval knobs (K, threshold, mpc, n) live in the EVAL-NAME folder, matching
            # flow_matching_v3_ode_selectable (…T0.5_D…_mpc4/6/results/…), NOT as a run_tag
            # buried under results/.
            _train_name = hf_paths.train_name(checkpoint_dir, args.diffusion_loadpath)
            _eval_name = hf_paths.eval_name(flow_steps, hf_act_threshold, hf_batch_size, n_trials)
            args.savepath = hf_paths.eval_root(args.logbase, args.dataset, _train_name, _eval_name, seed)
            os.makedirs(args.savepath, exist_ok=True)
            print(f'[ eval ] savepath: {args.savepath}')

            # ── U10.1 RUN PROVENANCE ──────────────────────────────────────────────────
            # Gen12 builds savepath itself via hf_paths (not Parser.mkdir), so it gets
            # neither args.json nor a config snapshot here — the env-resolved knobs
            # (HFFM_ACT_THRESHOLD, HFFM_BATCH, DPCC_THRESHOLD, FORCE_OVERWRITE) survived
            # only as the eval-name tokens. Written after savepath is final. Never fatal.
            provenance.write(
                args.savepath, role='eval',
                yaml_path=args_cli.config,
                resolved={
                    'horizon': int(getattr(args, 'horizon', -1)),
                    'flow_steps_K': int(flow_steps),
                    'hf_act_threshold': float(hf_act_threshold),
                    'hf_batch_size': int(hf_batch_size),
                    'mpc_batch_arms_ab': int(mpc_batch),
                    'run_msg': run_msg,
                    'dpcc_threshold': float(dpcc_threshold),
                    'force_overwrite': bool(FORCE_OVERWRITE),
                    'checkpoint_dir': checkpoint_dir,
                    'diffusion_loadpath': getattr(args, 'diffusion_loadpath', None),
                    'train_name': _train_name,
                    'eval_name': _eval_name,
                    'seed': int(seed),
                    'n_trials': n_trials,
                })
            if 'pointmaze' in exp or 'antmaze' in exp:
                minari_dataset = minari.load_dataset(exp, download=True)
                env = minari_dataset.recover_environment(eval_env=True) if 'pointmaze' in exp else minari_dataset.recover_environment()
            elif 'avoiding' in exp:
                env = ObstacleAvoidanceEnv()
                env.start()
            if robot_name == 'pointmaze': env.env.env.env.point_env.frame_skip = 2
            if robot_name == 'antmaze': env.env.env.env.ant_env.frame_skip = 5
            obs_indices = config['observation_indices'][robot_name]
            act_indices = config['action_indices'][robot_name]
            # Gen12 loads FMv3ODE (FlowMatchingODE), which plans actions (action_dim > 0).
            # Route by action_dim rather than a hard class-name check so the loaded
            # FlowMatchingODE takes the states_actions path.
            if getattr(fm_model, 'action_dim', 0) > 0:
                trajectory_dim = fm_model.transition_dim - fm_model.goal_dim
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

            # Arm C's optional HardFlow-faithful dynamics. Refusing to load a
            # mismatched .npz is the whole point of PLAN §3.1.
            linear_dynamics = None
            if hardflow_cfg.get('dynamics_mode', 'deriv') == 'linear_fit':
                # sys.path[0] is this script's directory when run as
                # `python FM_v3_hardflow_test/eval_FM_v3_hardflow.py`.
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from fit_dynamics_fmv3 import load_linear_dynamics, output_path
                dyn_path = hardflow_cfg.get('linear_dynamics_path') or output_path(
                    exp, args.horizon, getattr(args, 'max_path_length', 150))
                linear_dynamics = load_linear_dynamics(dyn_path, dataset.normalizer)
                print(f'[ eval ] arm C linear dynamics loaded and normalizer-checked: {dyn_path}')

            env_seeds = config['env_seeds'][exp] if 'pointmaze-umaze' in exp else np.arange(100)
            # squeeze=False: with a single variant or a single plotted trial the
            # default squeeze collapses the axes array and every `ax[i, j]` below
            # raises. Inherited bug; fixed here rather than carried.
            fig_all, ax_all = plt.subplots(min(n_trials, plot_how_many), len(projection_variants), figsize=(10 * len(projection_variants), 10 * min(n_trials, plot_how_many)), squeeze=False)
            save_path = None
            for variant_idx, variant in enumerate(projection_variants):
                is_hardflow = variant.startswith('hardflow')
                print(f'------------------------Running {exp} - {halfspace_variant} - {variant} ({seed}) - K={flow_steps}----------------------------')
                # [Gen0fix2] Restore the per-variant threshold deleted by upstream DPCC commit 7f09d3a.
                # threshold 0 => the activation gate fires only on the FINAL step, i.e. ONE projection
                # applied after the last denoising/ODE step -- the paper's definition of post-processing
                # ("modifying them after the last denoising step, usually by solving an optimization
                # problem"). Without it, `post_processing` inherits the normal schedule and is a
                # byte-identical duplicate of `dpcc-r`.
                threshold = 0.0 if 'post_processing' in variant else dpcc_threshold

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

                # ---- fix_5: FMv3ODE-style path (K/thres/mpc are in the EVAL-NAME folder
                # already, via args.savepath), so results is just results/halfspace_<hv>/.
                # PLAN §3.6: refuse to clobber a finished dir.
                save_path = (f'{args.savepath}/results/halfspace_{halfspace_variant}'
                             if 'avoiding' in exp else f'{args.savepath}/results')
                # [SolverSwap] 🔴 artifact name carries the backend so an SLSQP run lands BESIDE
                # the IPOPT corpus instead of overwriting it (the clobber guard below would
                # otherwise just skip, or FORCE_OVERWRITE would destroy it). Resolved from the
                # module because the policy does not exist yet; asserted against the live NLP
                # once it does. IPOPT keeps the old name exactly, so nothing on disk moves.
                nlp_backend_planned = resolve_nlp_backend()
                variant_out = artifact_variant_label(variant, nlp_backend_planned)
                npz_path = f'{save_path}/{variant_out}.npz'
                if os.path.exists(npz_path) and not FORCE_OVERWRITE:
                    print(f'[ eval ] {npz_path} already exists — skipping. '
                          'Set FORCE_OVERWRITE=1 to re-run it.')
                    continue
                os.makedirs(save_path, exist_ok=True)

                if is_hardflow:
                    # ---------------- arm C ----------------
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
                    # U4.2 + U5: DPCC-parity selection from the variant suffix. Strip the
                    # '-tightened' marker FIRST so the selection suffix composes with it —
                    # hardflow_new-c-tightened -> minimum_projection_cost AND enlarged
                    # constraints — exactly like DPCC parses 'dpcc-c' independently of
                    # 'tightened'. (The old endswith('-c') broke here, silently falling back
                    # to random for every -tightened variant.) At batch_size==1 all of
                    # -r/-c/-t collapse to index 0 (see HardFlowPolicy._select), so they are
                    # identical there and only diverge once the candidate fan is on (mpc>1).
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
                        linear_dynamics=linear_dynamics,
                        print_level=int(hardflow_cfg.get('ipopt_print_level', 0)),
                        print_time=bool(hardflow_cfg.get('casadi_print_time', False)),
                        device=args.device, goal_dim=fm_model.goal_dim)
                else:
                    # ---------------- arms A / B ----------------
                    batch_size = args.batch_size
                    # [Gen12fix8] diffusion_timestep_threshold=... restored (was omitted by the
                    # Gen12 port -> Projector fell back to its hardcoded 0.5 default).
                    projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=fm_model.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, gradient=gradient, gradient_weights=[1, 0.5, 2], variant=fm_variant, dt=delta_t, cost_dims=None, device=args.device, solver='scipy', diffusion_timestep_threshold=threshold)   # [Gen0fix2] post_processing override
                    projector = None if variant == 'diffuser' else projector
                    trajectory_selection = 'random'
                    if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                    if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
                    policy = Policy(model=fm_model, normalizer=dataset.normalizer, preprocess_fns=args.preprocess_fns, test_ret=args.test_ret, projector=projector, trajectory_selection=trajectory_selection)

                fig, ax = plt.subplots(min(n_trials, plot_how_many), 6, figsize=(30, 5 * min(n_trials, plot_how_many)), squeeze=False)
                fig.suptitle(f'{exp} - {variant} - K{flow_steps}')
                save_samples_every = 1
                plot_samples_every = max(1, args.horizon // 2)
                sampled_trajectories_all = []
                obs_all = []
                act_all = []
                n_success = np.zeros(n_trials)
                n_success_and_constraints = np.zeros(n_trials)
                n_steps = np.zeros(n_trials)
                n_violations = np.zeros(n_trials)
                total_violations = np.zeros(n_trials)
                avg_time = np.zeros(n_trials)
                collision_free_completed = np.ones(n_trials)
                pos_tracking_errors = np.zeros((n_trials, args.max_episode_length - 1))
                nfe_total = 0
                nlp_solves_total = 0
                nlp_failures_total = 0
                n_plan_calls = 0
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
                        start = time.time()
                        action, samples = policy(conditions={0: obs}, batch_size=batch_size, horizon=args.horizon, disable_projection=disable_projection)
                        avg_time[i] += time.time() - start
                        n_plan_calls += 1
                        if is_hardflow:
                            nlp_solves_total += policy.last_info.get('nlp_solves', 0)
                            nlp_failures_total += policy.last_info.get('nlp_failures', 0)
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
                        desired_next_pos = samples.observations[0, 1, [obs_indices['x'], obs_indices['y']]]
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
                    sampled_trajectories_all.append(sampled_trajectories)
                    obs_all.append(np.array(obs_buffer))
                    act_all.append(np.array(action_buffer))
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
                if is_hardflow:
                    # Counted for real inside the sampler: 2 evals per ODE step
                    # (reference + terminal prediction), per candidate.
                    nfe_total = policy.nfe
                else:
                    # Arms A/B evaluate the field once per ODE step, batched over
                    # candidates. Reported in the SAME units as arm C so the
                    # matched-K compute comparison in PLAN §5 is meaningful.
                    nfe_total = flow_steps * n_plan_calls * batch_size
                print(f'Success rate: {np.mean(n_success)}')
                print(f'Constraints satisfied: {np.mean(collision_free_completed)}')
                print(f'Success rate (goal and constraints): {np.mean(n_success_and_constraints)}')
                print(f'Avg number of steps: {(np.mean(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f} +- {(np.std(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f}')
                print(f'Avg number of constraint violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
                print(f'Avg total violation: {np.mean(total_violations):.3f} +- {np.std(total_violations):.3f}')
                print(f'Average computation time per step: {np.mean(avg_time):.3f}')
                # PLAN §5: compute must be reported alongside success, per arm.
                # U4/U4.2: also report the activation threshold and selection for arm C.
                # [SolverSwap] the solver is now selectable, so it must be IN the log.
                nlp_backend_used = str(getattr(getattr(policy, 'nlp', None), 'nlp_backend', 'n/a'))
                # [SolverSwap] the filename was chosen before the policy existed. If the two ever
                # disagree the run is about to write an SLSQP result into an IPOPT filename —
                # fail loudly rather than corrupt the corpus.
                assert not is_hardflow or nlp_backend_used == nlp_backend_planned, (
                    f'nlp_backend mismatch: artifact named for {nlp_backend_planned!r} but the '
                    f'NLP ran {nlp_backend_used!r} — refusing to mislabel {npz_path}')
                hf_report = (f'  act_thr={hf_act_threshold:g}  sel={hf_selection}'
                             f'  nlp_backend={nlp_backend_used}'
                             if is_hardflow else '')
                print(f'Compute: K={flow_steps}  batch={batch_size}  '
                      f'NFE={nfe_total}  NLP solves={nlp_solves_total}  '
                      f'NLP failures={nlp_failures_total}{hf_report}')
                if variant == 'diffuser': print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')
                if config['write_to_file']:
                    # HFK1 (2026-08-24) — record the degeneracy verdict instead of leaving a
                    # DA to re-derive it. hf_n_genuine == 0 => this row is Pi_S(Euler sample),
                    # i.e. sample-then-project (== DPCC modulo solver), NOT HardFlow. Always
                    # true at K=1; also at K=2 under the shipped A=0.5. See
                    # logs_in_develop/aggregated_hardflow_lowK/
                    _hf_budget = (hardflow_step_budget(flow_steps, hf_act_threshold)
                                  if is_hardflow and flow_steps else (0, 0))
                    np.savez(npz_path, n_success=n_success, n_success_and_constraints=n_success_and_constraints, n_steps=n_steps, n_violations=n_violations, total_violations=total_violations, avg_time=avg_time, collision_free_completed=collision_free_completed, args=args, obs_all=np.array(obs_all, dtype=object), act_all=np.array(act_all, dtype=object), sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object), flow_steps=flow_steps, batch_size=batch_size, nfe=nfe_total, nlp_solves=nlp_solves_total, nlp_failures=nlp_failures_total, nlp_backend=str(nlp_backend_used), nlp_backend_slsqp=float(nlp_backend_used == 'slsqp'), variant=variant, activation_threshold=hf_act_threshold, dpcc_threshold=dpcc_threshold, hf_n_active=int(_hf_budget[0]), hf_n_genuine=int(_hf_budget[1]), hf_degenerate=bool(is_hardflow and _hf_budget[1] == 0), trajectory_selection=(hf_selection if is_hardflow else 'n/a'), hardflow_cfg=json.dumps(hardflow_cfg))
                    # [Gen12fix8] dpcc_threshold recorded. The results dir name
                    # (hf_paths.eval_name) encodes only the HF activation threshold, so with
                    # DPCC's threshold now independently settable a run could otherwise be
                    # silently mislabeled (folder says thres0.1 while arms A/B ran at 0.5).
                    # Keep the two equal unless you deliberately want them to differ.
                fig.savefig(f'{save_path}/{variant_out}.png')
                plt.close(fig)
                ax_all[0, variant_idx].set_title(variant)
                env.close()
            if save_path is not None:
                fig_all.savefig(f'{save_path}/all.png')
        variant_idx = 0
        # fix_5: eval knobs already live in the eval-name folder; all_seeds sits beside
        # the per-seed dirs at <train>/<eval>/all_seeds/halfspace_<hv>/.
        path = f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}'
        os.makedirs(path, exist_ok=True)
        for fig, ax in zip(figs_all_seeds, axes_all_seeds):
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
            variant_idx += 1
