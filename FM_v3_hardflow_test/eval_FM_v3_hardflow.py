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
from flow_matcher_v3_hardflow.sampling.projection import Projector
from flow_matcher_v3_hardflow.sampling.hardflow_projection import (
    HardFlowPolicy, resolve_activation_threshold)
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
hf_batch_size = int(os.environ.get('HFFM_BATCH', hardflow_cfg.get('batch_size', 1)))
hf_candidate_cost = hardflow_cfg.get('candidate_cost', 'prox')
# `checkpoint_dir` and `flow_steps` now live in the plan_fm_v3_hardflow block in
# config/avoiding-d3il.py (read from `args` inside the seed loop), so the eval has a
# single tidy control entry. CLI `--flow-steps N` still overrides the block's K.
flow_steps_cli = args_cli.flow_steps


def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest', device='cuda:0'):
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
                npz_path = f'{save_path}/{variant}.npz'
                if os.path.exists(npz_path) and not FORCE_OVERWRITE:
                    print(f'[ eval ] {npz_path} already exists — skipping. '
                          'Set FORCE_OVERWRITE=1 to re-run it.')
                    continue
                os.makedirs(save_path, exist_ok=True)

                if is_hardflow:
                    # ---------------- arm C ----------------
                    batch_size = hf_batch_size
                    # U4.2: DPCC-parity selection from the variant suffix.
                    hf_selection = 'random'
                    if variant.endswith('-t'): hf_selection = 'temporal_consistency'
                    elif variant.endswith('-c'): hf_selection = 'minimum_projection_cost'
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
                    projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim, goal_dim=fm_model.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer, gradient=gradient, gradient_weights=[1, 0.5, 2], variant=fm_variant, dt=delta_t, cost_dims=None, device=args.device, solver='scipy')
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
                hf_report = (f'  act_thr={hf_act_threshold:g}  sel={hf_selection}'
                             if is_hardflow else '')
                print(f'Compute: K={flow_steps}  batch={batch_size}  '
                      f'NFE={nfe_total}  NLP solves={nlp_solves_total}  '
                      f'NLP failures={nlp_failures_total}{hf_report}')
                if variant == 'diffuser': print(f'Tracking error: {np.max(pos_tracking_errors):.3f}')
                if config['write_to_file']:
                    np.savez(npz_path, n_success=n_success, n_success_and_constraints=n_success_and_constraints, n_steps=n_steps, n_violations=n_violations, total_violations=total_violations, avg_time=avg_time, collision_free_completed=collision_free_completed, args=args, obs_all=np.array(obs_all, dtype=object), act_all=np.array(act_all, dtype=object), sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object), flow_steps=flow_steps, batch_size=batch_size, nfe=nfe_total, nlp_solves=nlp_solves_total, nlp_failures=nlp_failures_total, variant=variant, activation_threshold=hf_act_threshold, trajectory_selection=(hf_selection if is_hardflow else 'n/a'), hardflow_cfg=json.dumps(hardflow_cfg))
                fig.savefig(f'{save_path}/{variant}.png')
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
