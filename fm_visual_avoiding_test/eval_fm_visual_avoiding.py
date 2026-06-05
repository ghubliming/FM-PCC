# Visual-FM avoiding eval — U2 rebuild.
# Copy-modified from FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py.
# Three targeted swaps vs the state-only baseline:
#   A) fm_visual_avoiding.utils  instead of flow_matcher_v3_ode_selectable.utils
#   B) env.bp_cam per-step image  instead of (no camera in state-only eval)
#   C) VisualAgent.predict(bp_image, pred_xy, c_xy)  instead of policy(conditions={0:obs})
import cv2
import os
import pickle
import sys
import time
from copy import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import torch
import yaml

# Swap A — package
import fm_visual_avoiding.utils as utils
from fm_visual_avoiding.sampling.projection import Projector
from fm_visual_avoiding.models.visual_gaussian_diffusion import VisualFlowMatching

from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
import argparse

_IMG_W = _IMG_H = 96   # must match training resolution (ParityAvoidingDataset)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


# ── Minimal normalizer wrapper so Projector gets .normalizers dict ────────────

class ProjectorNormalizer:
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {'observations': obs_normalizer, 'actions': act_normalizer}


# ── Thin visual agent (~40 lines, replaces 700-line VisualAgentWrapper) ──────

class VisualAgent:
    """Single-step visual inference for VisualFlowMatching (avoiding, Gen9 Ep2).

    Visual obs: 4D [des_xy(2) | c_xy(2)] — matches ParityAvoidingDataset.
    Trajectory: 6D [act(2) | des_xy(2) | c_xy(2)].
    """
    def __init__(self, model, obs_normalizer, act_normalizer,
                 projector=None, device='cuda:0'):
        self.model          = model
        self.obs_normalizer = obs_normalizer
        self.act_normalizer = act_normalizer
        self.projector      = projector
        self.device         = device

    def predict(self, bp_image, pred_xy, c_xy):
        """
        bp_image : (3, 96, 96) float32 BGR→RGB normalised to [0,1]
        pred_xy  : (2,) current desired XY (obs[:2] in eval loop)
        c_xy     : (2,) actual robot XY from env.robot.current_c_pos[:2]
        Returns  : (2,) unnormalised action delta
        """
        obs_4d   = np.concatenate([pred_xy, c_xy]).astype(np.float32)
        obs_norm = self.obs_normalizer.normalize(
            obs_4d.reshape(1, -1)).astype(np.float32).squeeze(0)

        bp_t  = torch.from_numpy(bp_image.astype(np.float32)).to(self.device)
        obs_t = torch.from_numpy(obs_norm).to(self.device)

        bp_b  = bp_t.unsqueeze(0).unsqueeze(0)    # (1, 1, C, H, W)
        obs_b = obs_t.unsqueeze(0).unsqueeze(0)   # (1, 1, 4)

        cond = {0: (bp_b, obs_b)}
        self.model.eval()
        with torch.no_grad():
            if self.projector is not None:
                traj, _ = self.model(cond, projector=self.projector)
            else:
                traj, _ = self.model(cond)

        act_norm = traj[0, 0, :2].detach().cpu().numpy()
        action   = self.act_normalizer.unnormalize(
            act_norm.reshape(1, -1)).squeeze(0)
        # Extract planned robot c_xy positions over the horizon for col-5 visualisation.
        # traj shape: (1, H, 6) = [act_norm(2)|des_xy_norm(2)|c_xy_norm(2)].
        # Unnormalise the obs part (dims 2:6) then take c_xy (last 2 cols of 4D obs).
        obs_norm_traj = traj[0, :, 2:].detach().cpu().numpy()       # (H, 4) normalised
        obs_raw_traj  = self.obs_normalizer.unnormalize(obs_norm_traj)  # (H, 4) raw
        planned_xy    = obs_raw_traj[:, 2:4][np.newaxis, :, :]       # (1, H, 2) c_xy
        return action, planned_xy


# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--aggregate-only', action='store_true')
# Legacy SLURM args — silently consumed so they don't reach utils.Parser.parse_args()
parser.add_argument('--record', default='all')
parser.add_argument('--eval-on-train', action='store_true')
args_cli, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

with open('config/projection_eval.yaml') as f:
    config = yaml.safe_load(f)

exps               = config['exps']
seeds              = config['seeds']
if args_cli.seed is not None:
    seeds = [args_cli.seed]
    print(f'[ eval ] Overriding seeds to: {seeds}')

projection_variants   = config['projection_variants']
halfspace_variants    = config['avoiding_halfspace_variants'] if 'avoiding' in exps[0] else ['top-left']
n_trials              = config['n_trials']
plot_how_many         = config['plot_how_many']
constraint_types      = config['constraint_types']
diffusion_timestep_threshold = config.get('diffusion_timestep_threshold', 0.5)


def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest',
                                 device='cuda:0', seed=None):
    import inspect
    print(f'\n[ eval loading ] {os.path.join(*loadpath)}\n')
    dataset_config   = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config     = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config   = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    if target_class is not None:
        target_cls = utils.config.import_class(target_class)
        if (diffusion_config._class.__module__ + '.' + diffusion_config._class.__name__
                != target_cls.__module__ + '.' + target_cls.__name__):
            diffusion_config._class = target_cls
            valid = set(inspect.signature(target_cls.__init__).parameters)
            for k in [k for k in diffusion_config._dict if k not in valid]:
                del diffusion_config._dict[k]

    dataset  = dataset_config()
    model    = model_config().to(device)
    diffusion = diffusion_config(model).to(device)
    trainer  = trainer_config(diffusion_model=diffusion, dataset=dataset)

    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    losses = utils.load_losses(*loadpath, 'losses.pkl')
    return utils.DiffusionExperiment(dataset, trainer.model.model,
                                     trainer.model, trainer, epoch, losses)


# ── Main eval loop ────────────────────────────────────────────────────────────

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

        bounds             = config['bounds'][exp]
        ax_limits          = config['ax_limits'][exp]
        enlarge_constraints = config['enlarge_constraints'][robot_name]
        dt                 = config['dt'][robot_name]
        obs_indices        = config['observation_indices'][robot_name]
        act_indices        = config['action_indices'][robot_name]

        # Fix_1: exp ('avoiding-d3il') is used only for YAML constraint/bounds lookups.
        # The Python config that holds 'plan_fm_visual_avoiding' is avoiding-d3il-visual.
        class Parser(utils.Parser):
            dataset: str = 'avoiding-d3il-visual'
            config:  str = 'config.avoiding-d3il-visual'

        figs_all_seeds, axes_all_seeds = zip(*[
            plt.subplots(1, 1, figsize=(9, 10)) for _ in range(len(projection_variants))])
        figs_all_seeds  = list(figs_all_seeds)
        axes_all_seeds  = list(axes_all_seeds)

        for seed in seeds:
            args = Parser().parse_args(experiment='plan_fm_visual_avoiding', seed=seed)

            fm_model       = None
            obs_normalizer = None
            act_normalizer = None
            env            = None

            if not args_cli.aggregate_only:
                fm_experiment = load_diffusion_with_override(
                    args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                    target_class=args.diffusion, epoch=args.diffusion_epoch, device=args.device)
                fm_model = fm_experiment.diffusion

                # Load normalizers saved by train script
                ckpt_dir = os.path.join(args.loadbase, args.dataset,
                                        args.diffusion_loadpath, str(args.seed))
                with open(os.path.join(ckpt_dir, 'obs_normalizer.pkl'), 'rb') as f:
                    obs_normalizer = pickle.load(f)
                with open(os.path.join(ckpt_dir, 'act_normalizer.pkl'), 'rb') as f:
                    act_normalizer = pickle.load(f)

                # Swap B — use ObstacleAvoidanceEnv directly; bp_cam accessed per step
                env = ObstacleAvoidanceEnv()
                env.start()

                # Trajectory/action dims for VisualFlowMatching
                if fm_model.__class__.__name__ in ('FlowMatchingODE', 'VisualFlowMatching'):
                    trajectory_dim = fm_model.transition_dim - fm_model.goal_dim
                    action_dim     = fm_model.action_dim
                    fm_variant     = 'states_actions'
                    obs_indices_updated = {k: v + action_dim for k, v in obs_indices.items()}
                    act_obs_indices = {**act_indices, **obs_indices_updated}
                else:
                    trajectory_dim = fm_model.observation_dim - fm_model.goal_dim
                    action_dim     = 0
                    fm_variant     = 'states'
                    act_obs_indices = obs_indices

                # Constraints
                constraint_list                      = []
                constraint_list_tightened            = []
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
                    lb, ub = utils.formulate_bounds_constraints(
                        constraint_types, bounds, trajectory_dim, act_obs_indices)
                    constraint_list.extend([['lb', lb], ['ub', ub]])
                    constraint_list_tightened.extend([['lb', lb], ['ub', ub]])
                if 'obstacles' in constraint_types:
                    for co in obstacle_constraints:
                        idx = [act_obs_indices[co['dimensions'][0]],
                               act_obs_indices[co['dimensions'][1]]]
                        constraint_list.append([co['type'], idx, co['center'], co['radius']])
                        constraint_list_tightened.append([co['type'], idx, co['center'],
                                                          co['radius'] + enlarge_constraints])
                constraint_list_without_prior          = copy(constraint_list)
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

                if args_cli.aggregate_only:
                    npz_path = os.path.join(save_path, f'{variant}.npz')
                    if not os.path.exists(npz_path):
                        print(f'[ eval ] skipping {variant} seed {seed}: no npz at {npz_path}')
                        continue
                    data = np.load(npz_path, allow_pickle=True)
                    if 'obs_all' in data:
                        for i in range(min(len(data['obs_all']), plot_how_many)):
                            buf = data['obs_all'][i]
                            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                            axes_all_seeds[variant_idx].plot(
                                np.array(buf)[:, obs_indices['x']],
                                np.array(buf)[:, obs_indices['y']],
                                colors[seed % len(colors)], linewidth=2)
                    continue

                log_file = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
                original_stdout = sys.stdout
                sys.stdout = Tee(sys.stdout, log_file)
                try:
                    print(f'--- {exp} {halfspace_variant} {variant} seed={seed} ---')

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
                    if 'dt0p25' in variant: delta_t = 0.25 * dt
                    elif 'dt0p5' in variant: delta_t = 0.5 * dt
                    elif 'dt2p0' in variant: delta_t = 2.0 * dt
                    elif 'dt4p0' in variant: delta_t = 4.0 * dt

                    proj_normalizer = ProjectorNormalizer(obs_normalizer, act_normalizer)
                    projector = Projector(
                        horizon=args.horizon, transition_dim=trajectory_dim,
                        action_dim=action_dim, goal_dim=fm_model.goal_dim,
                        constraint_list=constraints, normalizer=proj_normalizer,
                        gradient=gradient, gradient_weights=[1, 0.5, 2],
                        variant=fm_variant, dt=delta_t, cost_dims=None,
                        device=args.device, solver='scipy',
                        diffusion_timestep_threshold=diffusion_timestep_threshold)
                    projector = None if variant == 'diffuser' else projector

                    trajectory_selection = 'random'
                    if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                    if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'

                    agent = VisualAgent(fm_model, obs_normalizer, act_normalizer,
                                        projector=projector, device=args.device)

                    fig, ax = plt.subplots(
                        min(n_trials, plot_how_many), 6,
                        figsize=(30, 5 * min(n_trials, plot_how_many)), squeeze=False)
                    fig.suptitle(f'{exp} - {variant}')
                    fig_all, ax_all = plt.subplots(
                        min(n_trials, plot_how_many), len(projection_variants),
                        figsize=(10 * len(projection_variants),
                                 10 * min(n_trials, plot_how_many)), squeeze=False)

                    save_samples_every          = args.horizon // 2
                    n_success                   = np.zeros(n_trials)
                    n_success_and_constraints   = np.zeros(n_trials)
                    n_steps                     = np.zeros(n_trials)
                    n_violations                = np.zeros(n_trials)
                    total_violations            = np.zeros(n_trials)
                    avg_time                    = np.zeros(n_trials)
                    collision_free_completed    = np.ones(n_trials)
                    pos_tracking_errors         = np.zeros((n_trials, args.max_episode_length - 1))
                    obs_all                     = []
                    act_all                     = []
                    sampled_trajectories_all    = []

                    for i in range(n_trials):
                        torch.manual_seed(i)

                        if 'avoiding' in exp:
                            obs     = env.reset()
                            action  = env.robot_state()[:2]
                            fixed_z = env.robot_state()[2:]
                            obs     = np.concatenate((action[:2], obs))

                        obs_buffer         = []
                        action_buffer      = []
                        sampled_trajectories = []
                        disable_projection = False
                        desired_next_pos   = obs[obs_indices['x']:obs_indices['y'] + 1].copy()

                        for _ in range(args.max_episode_length):
                            violated_this_timestep = 0

                            if 'halfspace' in constraint_types:
                                for constraint in constraint_list_polytopic_not_tightened:
                                    if constraint[0] == 'ineq':
                                        c, d = constraint[1]
                                        obs_check = obs[:-fm_model.goal_dim] if fm_model.goal_dim > 0 else obs
                                        if obs_check @ c[action_dim:] >= d:
                                            violated_this_timestep = 1
                                            total_violations[i] += obs_check @ c[action_dim:] - d
                                            collision_free_completed[i] = 0

                            if 'obstacles' in constraint_types:
                                for co in obstacle_constraints:
                                    if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]]
                                                      - co['center']) < co['radius']:
                                        violated_this_timestep = 1
                                        total_violations[i] += (co['radius']
                                            - np.linalg.norm(obs[[obs_indices['x'],
                                                                   obs_indices['y']]] - co['center']))
                                        collision_free_completed[i] = 0

                            if _ > 0 and 'bounds' in constraint_types:
                                act_obs = np.concatenate((action, obs)) if action_dim > 0 else obs
                                total_violations[i] += (np.sum(np.maximum(0, act_obs - ub))
                                                       + np.sum(np.maximum(0, lb - act_obs)))

                            n_violations[i] += violated_this_timestep

                            # Swap C — visual predict
                            start = time.time()
                            if 'avoiding' in exp:
                                bp_img_raw = env.bp_cam.get_image(depth=False)
                                bp_img_raw = cv2.resize(bp_img_raw, (_IMG_W, _IMG_H),
                                                        interpolation=cv2.INTER_AREA)
                                bp_image = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.
                                c_xy     = env.robot.current_c_pos[:2].copy()
                                action, traj_plan = agent.predict(bp_image, obs[:2].copy(), c_xy)
                            avg_time[i] += time.time() - start

                            if 'avoiding' in exp:
                                next_pos_des = action + obs[:2]
                                obs, rew, terminated, info = env.step(
                                    np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0))
                                success = info[1]
                                obs = np.concatenate((next_pos_des[:2], obs))

                            if _ >= 1:
                                pos_tracking_errors[i, _ - 1] = np.linalg.norm(
                                    obs[obs_indices['x']:obs_indices['y'] + 1] - desired_next_pos)
                            desired_next_pos = next_pos_des[:2].copy()

                            if _ % save_samples_every == 0:
                                sampled_trajectories.append(traj_plan)   # (1, H, 2) planned c_xy

                            obs_buffer.append(obs)
                            action_buffer.append(action)
                            if success: n_success[i] = 1
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
                        sampled_trajectories_all.append(sampled_trajectories)

                        if i >= plot_how_many:
                            continue
                        plot_states = ['x', 'y', 'x_des', 'y_des']
                        for j in range(min(len(plot_states), 4)):
                            if obs_indices.get(plot_states[j]) is not None:
                                ax[i, j].plot(np.array(obs_buffer)[:, obs_indices[plot_states[j]]])
                                ax[i, j].set_title(plot_states[j])

                        for curr_ax in [ax[i, 4], ax_all[i, variant_idx]]:
                            curr_ax.plot(np.array(obs_buffer)[:, obs_indices['x']],
                                         np.array(obs_buffer)[:, obs_indices['y']], 'k')
                            curr_ax.plot(np.array(obs_buffer)[0, obs_indices['x']],
                                         np.array(obs_buffer)[0, obs_indices['y']], 'go')
                            curr_ax.set_xlim(ax_limits[0])
                            curr_ax.set_ylim(ax_limits[1])

                        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                        axes_all_seeds[variant_idx].plot(
                            np.array(obs_buffer)[:, obs_indices['x']],
                            np.array(obs_buffer)[:, obs_indices['y']],
                            colors[seed % len(colors)], linewidth=2)

                        # Col 5: planned c_xy trajectory from FM model at subsampled steps
                        for traj_np in sampled_trajectories_all[i]:
                            if traj_np is None:
                                continue
                            for k in range(traj_np.shape[0]):
                                for curr_ax in [ax[i, 5], ax_all[i, variant_idx]]:
                                    curr_ax.plot(traj_np[k, :, 0], traj_np[k, :, 1],
                                                 'b', alpha=0.5)
                                    curr_ax.plot(traj_np[k, 0, 0], traj_np[k, 0, 1], 'go')
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
                    print(f'Success rate (goal + constraints): {np.mean(n_success_and_constraints):.3f}')
                    print(f'Avg steps: {(np.mean(n_steps[n_success > 0]) if n_success.sum() > 0 else 0):.2f}'
                          f' +- {(np.std(n_steps[n_success > 0]) if n_success.sum() > 0 else 0):.2f}')
                    print(f'Avg violations: {np.mean(n_violations):.2f} +- {np.std(n_violations):.2f}')
                    print(f'Avg total violation: {np.mean(total_violations):.3f}'
                          f' +- {np.std(total_violations):.3f}')
                    print(f'Avg compute time/step: {np.mean(avg_time):.3f}')

                    if config['write_to_file']:
                        np.savez(f'{save_path}/{variant}.npz',
                                 n_success=n_success,
                                 n_success_and_constraints=n_success_and_constraints,
                                 n_steps=n_steps, n_violations=n_violations,
                                 total_violations=total_violations, avg_time=avg_time,
                                 collision_free_completed=collision_free_completed,
                                 args=args,
                                 obs_all=np.array(obs_all, dtype=object),
                                 act_all=np.array(act_all, dtype=object))

                    fig.savefig(f'{save_path}/{variant}.png')
                    plt.close(fig)
                    ax_all[0, variant_idx].set_title(variant)

                finally:
                    sys.stdout = original_stdout
                    log_file.close()

            if not args_cli.aggregate_only:
                fig_all.savefig(f'{save_path}/all.png')
                env.close()

        path = f'{os.path.dirname(args.savepath)}/all_seeds/{halfspace_variant}'
        os.makedirs(path, exist_ok=True)
        for variant_idx, (fig, ax) in enumerate(zip(figs_all_seeds, axes_all_seeds)):
            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
            ax.set_facecolor([1, 1, 0.9])
            utils.plot_environment_constraints(exp, ax)
            if 'halfspace' in constraint_types:
                utils.plot_halfspace_constraints(exp, polytopic_constraints, ax,
                                                 ax_limits, enlarge_constraints=enlarge_constraints)
            if 'obstacles' in constraint_types:
                for co in obstacle_constraints:
                    ax.add_patch(matplotlib.patches.Circle(co['center'], co['radius'],
                                                           color='b', alpha=0.2))
                    ax.add_patch(matplotlib.patches.Circle(co['center'],
                                                           co['radius'] + enlarge_constraints,
                                                           color='b', alpha=0.1, linestyle='--'))
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.png', bbox_inches='tight')
            fig.savefig(f'{path}/{projection_variants[variant_idx]}.pdf',
                        bbox_inches='tight', format='pdf')
            plt.close(fig)

        if not args_cli.aggregate_only:
            plt.show()
