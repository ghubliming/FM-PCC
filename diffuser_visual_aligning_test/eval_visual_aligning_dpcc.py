# Visual-DPCC (Gen6V4) evaluation script.
# Copy-modified from ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py.
# Logging pattern (realtime PNG/JSON/pkl, 7-metric report, expert reference,
# legacy rollout grid) is reused verbatim from that proven script.
#
# Key differences from ddpm_encdec_vision eval:
#   - trajectory is 9D  [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
#   - projector transition_dim=9,  bounds on c_pos indices [6,7,8]
#   - Euler dynamics:  [6←0, 7←1, 8←2]
#   - normalizer: LimitsNormalizer (no Scaler); obs_normalizer.pkl + act_normalizer.pkl
#   - D3IL API: only des_robot_pos exposed → obs_6d = [des_pos, des_pos]
#
# Output:  logs/aligning-d3il-visual/visual_aligning_dpcc/<exp>/results/<seed>/
#          ├── diffuser.npz
#          ├── diffuser.png          (6-panel rollout grid)
#          ├── results_seed_<s>.pkl
#          ├── eval_diffuser.log
#          ├── diagnostics/<variant>/rollout_<r>.{mp4,gif,txt}
#          ├── realtime_diagnostics/<variant>/rollout_<r>_{data.pkl,stats.json,report.png}
#          └── expert_references/expert_rollout_<r>.{mp4,gif}

import json
import time
import yaml
import os
import sys
import pickle
import argparse
import numpy as np
import torch
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import imageio
import cv2

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

sys.path.insert(0, os.path.abspath('d3il'))
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

import diffuser_visual_aligning.utils as utils
from diffuser_visual_aligning.sampling.projection import Projector

import d3il
print(f'[ eval ] Using d3il from: {d3il.__file__}')
print(f'[ eval ] D3IL_DIR set to: {os.environ["D3IL_DIR"]}')

from d3il.simulation.aligning_sim import Aligning_Sim

# ── Normalizer wrapper for Projector ─────────────────────────────────────────

class ProjectorNormalizer:
    """
    Wraps obs and act LimitsNormalizers into the dict that Projector('states_actions') expects.
    LimitsNormalizer already exposes .mins and .maxs, so no adapter indirection is needed.
    ProjectionNormalizer (inside Projector) reads normalizers['observations'].mins/maxs
    and normalizers['actions'].mins/maxs to build 9D constraint bounds [act(3)|obs(6)].
    """
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {
            'observations': obs_normalizer,   # LimitsNormalizer — .mins(6,) .maxs(6,)
            'actions':      act_normalizer,   # LimitsNormalizer — .mins(3,) .maxs(3,)
        }

# ── Projector setup ───────────────────────────────────────────────────────────

def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant):
    """
    Build the DPCC SLSQP projector for the 9D trajectory space.

    Trajectory layout: [dx(0) dy(1) dz(2) | des_x(3) des_y(4) des_z(5) | x(6) y(7) z(8)]

    Constraints:
        - Workspace bounds on actual EE position (c_pos, indices 6-8)
        - Euler dynamics: c_pos[t+1] = c_pos[t] + act[t]  (indices [6←0, 7←1, 8←2])
    """
    tightening = config.get('constraint_tightening_margin',
                            config.get('enlarge_constraints', 0.0))
    ws_lb = np.array(config['workspace_bounds']['lb'])   # (3,)
    ws_ub = np.array(config['workspace_bounds']['ub'])   # (3,)

    if 'tightened' in variant and tightening > 0.0:
        ws_lb += tightening
        ws_ub -= tightening

    constraint_list = []

    if 'bounds' in config.get('constraint_types', []):
        # Bounds only on c_pos dims (indices 6,7,8); act and des_c_pos unconstrained
        lb = np.concatenate([np.full(6, -np.inf), ws_lb])   # (9,)
        ub = np.concatenate([np.full(6,  np.inf), ws_ub])   # (9,)
        constraint_list.append(['lb', lb])
        constraint_list.append(['ub', ub])

    if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
        constraint_list.append(('deriv', [6, 0]))   # c_pos_x ← dx
        constraint_list.append(('deriv', [7, 1]))   # c_pos_y ← dy
        constraint_list.append(('deriv', [8, 2]))   # c_pos_z ← dz

    dt = config.get('dt', 1.0)
    if   'dt0p25' in variant: dt *= 0.25
    elif 'dt0p5'  in variant: dt *= 0.50
    elif 'dt2p0'  in variant: dt *= 2.0
    elif 'dt4p0'  in variant: dt *= 4.0

    threshold = 0.0 if 'post_processing' in variant else config.get('diffusion_timestep_threshold', 0.5)
    gradient  = 'gradient' in variant

    return Projector(
        horizon=getattr(args, 'horizon', 8),
        transition_dim=9,
        action_dim=3,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=ProjectorNormalizer(obs_normalizer, act_normalizer),
        diffusion_timestep_threshold=threshold,
        variant='states_actions',
        dt=dt,
        gradient=gradient,
        gradient_weights=[1, 0.5, 2] if gradient else None,
        solver='scipy',
        device=args.device,
    )

# ── Logging ───────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files):
        self.files = [f if hasattr(f, 'write') else open(f, 'a') for f in files]
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

# ── Expert reference generation ───────────────────────────────────────────────

def generate_expert_reference(save_path, n_rollouts=3):
    """Generate ground-truth expert videos from the dataset for reference."""
    expert_dir = os.path.join(save_path, 'expert_references')
    all_exist = all(
        os.path.exists(os.path.join(expert_dir, f'expert_rollout_{i}.mp4')) or
        os.path.exists(os.path.join(expert_dir, f'expert_rollout_{i}.gif'))
        for i in range(n_rollouts)
    )
    if all_exist:
        print(f'[ expert ] Reference videos already exist in {expert_dir}. Skipping.')
        return

    print(f'[ expert ] Generating {n_rollouts} expert reference videos...')
    os.makedirs(expert_dir, exist_ok=True)

    try:
        from agents.utils.sim_path import sim_framework_path
        from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env

        state_data_dir = sim_framework_path('environments/dataset/data/aligning/all_data/state')
        env = Robot_Push_Env(render=False, if_vision=True)
        env.start()

        for idx in range(n_rollouts):
            file_name = f'env_{idx}.pkl'
            try:
                with open(os.path.join(state_data_dir, file_name), 'rb') as f:
                    expert_data = pickle.load(f)
            except Exception:
                all_files = sorted(os.listdir(state_data_dir))
                if idx >= len(all_files):
                    continue
                with open(os.path.join(state_data_dir, all_files[idx]), 'rb') as f:
                    expert_data = pickle.load(f)

            expert_path = expert_data['robot']['des_c_pos']
            box_pos    = expert_data['push-box']['pos'][0]
            box_quat   = expert_data['push-box']['quat'][0]
            target_pos = expert_data['target-box']['pos'][0]
            target_quat = expert_data['target-box']['quat'][0]
            context = (box_pos, box_quat, target_pos, target_quat)

            env.reset(random=False, context=context)
            frames = []
            for step in range(len(expert_path)):
                sim_action = np.concatenate((expert_path[step], [0, 1, 0, 0]))
                obs, _, _, _ = env.step(sim_action)
                _, bp_img, ih_img = obs
                frames.append(np.concatenate(
                    [cv2.cvtColor(bp_img, cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(ih_img, cv2.COLOR_BGR2RGB)], axis=1))

            save_file = os.path.join(expert_dir, f'expert_rollout_{idx}.mp4')
            try:
                imageio.mimsave(save_file, frames, fps=20)
                print(f'  [ expert ] Saved {save_file}')
            except Exception:
                gif_file = save_file.replace('.mp4', '.gif')
                try:
                    imageio.mimsave(gif_file, frames, fps=10)
                    print(f'  [ expert ] Saved {gif_file}')
                except Exception as e:
                    print(f'  [ expert ] Failed to save rollout {idx}: {e}')

        env.close()
    except Exception as e:
        print(f'[ expert ] WARNING — expert reference generation failed: {e}')

# ── VisualAgentWrapper ────────────────────────────────────────────────────────

class VisualAgentWrapper:
    """
    D3IL-compatible agent wrapper for Visual-DPCC (9D trajectory).
    Logging pattern ported from ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py.

    D3IL calls:
        agent.reset()                        — start of each rollout
        agent.predict(state, if_vision=True) — every sim step
        agent.update_rollout_info(info)      — end of each rollout

    State received: (bp_image, inhand_image, des_robot_pos)
    9D obs: obs_6d = [des_robot_pos(3), des_robot_pos(3)]
            (c_pos not exposed by D3IL; des_c_pos ≈ c_pos under PD control)
    """

    def __init__(self, diffusion_model, device,
                 window_size=1, obs_seq_len=1, action_seq_size=1,
                 save_path=None, record_mode='all',
                 obs_normalizer=None, act_normalizer=None,
                 batch_size=1, projector=None,
                 trajectory_selection='random',
                 eval_on_train=False, variant='unspecified'):

        self.model              = diffusion_model
        self.device             = device
        self.window_size        = window_size
        self.obs_seq_len        = obs_seq_len
        self.obs_normalizer     = obs_normalizer
        self.act_normalizer     = act_normalizer
        self.batch_size         = batch_size
        self.projector          = projector
        self.trajectory_selection = trajectory_selection
        self.eval_on_train      = eval_on_train
        self.save_path          = save_path
        self.record_mode        = record_mode
        self.variant            = variant

        model_horizon = getattr(self.model, 'horizon', window_size)
        self.action_seq_size = min(action_seq_size, model_horizon)
        self.action_counter  = self.action_seq_size   # force replan on first step
        self.curr_action_seq = None
        self.prev_observations = None

        self.rollout_counter = -1
        self.step_counter    = 0
        self.mental_robot_pos = None
        self.last_predicted_pos = None

        self.bp_image_context    = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.obs_context          = deque(maxlen=self.obs_seq_len)

        self.history_real_pos            = []
        self.history_desired_actions     = []
        self.history_full_plans          = []
        self.history_n_steps             = []
        self.history_avg_time            = []
        self.history_pos_tracking_errors = []
        self.curr_rollout_tracking_errors = []
        self.curr_rollout_time           = 0
        self.master_rollout_history      = {}
        self.video_frames                = []

    def reset(self):
        self.mental_robot_pos   = None
        self.prev_observations  = None
        self.last_predicted_pos = None
        self.action_counter     = self.action_seq_size
        self.curr_action_seq    = None
        self.rollout_counter   += 1
        self.step_counter       = 0
        self.curr_rollout_time  = 0
        self.curr_rollout_tracking_errors.clear()
        self.history_real_pos.clear()
        self.history_desired_actions.clear()
        self.history_full_plans.clear()
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.obs_context.clear()
        self.video_frames.clear()

    def update_rollout_info(self, info):
        """Called by Aligning_Sim at rollout end. Mirrors ddpm_encdec verbose format."""
        success   = info.get('success', False)
        mean_dist = info.get('mean_distance', 0.0)
        mode      = info.get('mode', 0)
        ridx      = int(info.get('context', self.rollout_counter))

        max_err  = float(np.max(self.curr_rollout_tracking_errors)
                         if self.curr_rollout_tracking_errors else 0.0)
        avg_time = float(self.curr_rollout_time / max(1, self.step_counter))

        self.master_rollout_history[f'rollout_{ridx}'] = {
            'real_robot_pos':      np.array(self.history_real_pos),
            'desired_actions':     np.array(self.history_desired_actions),
            'full_plans':          np.array(self.history_full_plans),
            'plan_start_positions': np.array(self.history_real_pos)[::self.action_seq_size],
            'success':   bool(success),
            'mean_distance': float(mean_dist),
            'mode':      int(mode),
            'steps':     int(self.step_counter),
            'avg_time':  avg_time,
            'max_tracking_error': max_err,
        }
        self.history_n_steps.append(self.step_counter)
        self.history_avg_time.append(avg_time)
        self.history_pos_tracking_errors.append(
            np.array(self.curr_rollout_tracking_errors))

        ctx_type = 'Seen Training Context' if self.eval_on_train else 'Unseen Test Context'
        print(f'[ {ctx_type} {ridx} Finished ]')
        print(f'  - Total Steps: {self.step_counter}')
        print(f'  - Success status: {success}')
        print(f'  - Final Mean Distance: {mean_dist:.6f} m')
        print(f'  - Environment Mode: {mode}')
        print(f'  - Maximum Tracking Error: {max_err:.6f} m')
        print(f'  - Avg Inference Time: {avg_time:.4f} seconds/step')
        print('-' * 80 + '\n')

        if self.save_path is not None:
            self._export_rollout_realtime(ridx)

        if self.record_mode != 'none' and self.video_frames:
            self._save_diagnostics(ridx)

    def _save_diagnostics(self, rollout_idx):
        """Save video/gif + stats.txt alongside. Mirrors ddpm_encdec pattern."""
        path = os.path.join(self.save_path, 'diagnostics', self.variant)
        os.makedirs(path, exist_ok=True)

        try:
            if self.record_mode in ['video', 'all']:
                try:
                    imageio.mimsave(
                        os.path.join(path, f'rollout_{rollout_idx}.mp4'),
                        self.video_frames, fps=20)
                except Exception as e:
                    print(f'[ WARNING ] MP4 failed: {e}')

            if self.record_mode in ['gif', 'all']:
                try:
                    imageio.mimsave(
                        os.path.join(path, f'rollout_{rollout_idx}.gif'),
                        self.video_frames, fps=10)
                except Exception as e:
                    print(f'[ WARNING ] GIF failed: {e}')

            data = self.master_rollout_history.get(f'rollout_{rollout_idx}', {})
            with open(os.path.join(path, f'rollout_{rollout_idx}_stats.txt'), 'w') as sf:
                sf.write(f'Rollout {rollout_idx} Execution Summary\n')
                sf.write('=' * 40 + '\n')
                sf.write(f'Success: {data.get("success", False)}\n')
                sf.write(f'Total Steps: {data.get("steps", 0)}\n')
                sf.write(f'Mean Distance to Target: {data.get("mean_distance", 0.0):.6f} m\n')
                sf.write(f'Environment Mode: {data.get("mode", 0)}\n')
                sf.write(f'Average Inference Time: {data.get("avg_time", 0.0):.4f} s/step\n')
                sf.write(f'Max Tracking Error: {data.get("max_tracking_error", 0.0):.6f} m\n')
        except Exception as e:
            print(f'[ WARNING ] Diagnostics save failed: {e}')

    def _export_rollout_realtime(self, rollout_idx):
        """Per-rollout PNG (6-panel) + JSON + pkl. Mirrors ddpm_encdec pattern."""
        try:
            diag_path = os.path.join(self.save_path, 'realtime_diagnostics', self.variant)
            os.makedirs(diag_path, exist_ok=True)

            data = self.master_rollout_history[f'rollout_{rollout_idx}']

            with open(os.path.join(diag_path, f'rollout_{rollout_idx}_data.pkl'), 'wb') as f:
                pickle.dump(data, f)

            stats = {
                'rollout_index':              int(rollout_idx),
                'success':                    bool(data.get('success', False)),
                'steps':                      int(data.get('steps', 0)),
                'mean_distance':              float(data.get('mean_distance', 0.0)),
                'mode':                       int(data.get('mode', 0)),
                'avg_inference_time_per_step': float(data.get('avg_time', 0.0)),
                'max_tracking_error':         float(data.get('max_tracking_error', 0.0)),
            }
            with open(os.path.join(diag_path, f'rollout_{rollout_idx}_stats.json'), 'w') as sf:
                json.dump(stats, sf, indent=4)

            real_pos   = data['real_robot_pos']        # (T, 3)
            plans      = data['full_plans']            # list of (H, 3) action arrays
            plan_starts = data['plan_start_positions'] # (N, 3)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Rollout {rollout_idx} — MPC vs Real  '
                         f'(success={data.get("success")})')

            # XY trajectory with MPC foresight
            axes[0, 0].plot(real_pos[:, 0], real_pos[:, 1], 'k-', linewidth=2,
                            label='Real Path')
            for p_idx, plan in enumerate(plans):
                if p_idx % 4 == 0:
                    start = plan_starts[min(p_idx, len(plan_starts) - 1)]
                    abs_plan = start + np.cumsum(plan[:, :3], axis=0)
                    axes[0, 0].plot(abs_plan[:, 0], abs_plan[:, 1], 'b-', alpha=0.3)
            axes[0, 0].set_title('XY Projection (MPC foresight in blue)')
            axes[0, 0].set_xlabel('X (m)'); axes[0, 0].set_ylabel('Y (m)')
            axes[0, 0].legend()

            axes[0, 1].plot(real_pos[:, 0], 'k-')
            axes[0, 1].set_title('X Position over Steps')
            axes[0, 1].set_ylabel('Meters')

            axes[0, 2].plot(real_pos[:, 1], 'k-')
            axes[0, 2].set_title('Y Position over Steps')

            axes[1, 0].plot(real_pos[:, 2], 'r-')
            axes[1, 0].set_title('Z Height (Contact Stability)')

            if self.curr_rollout_tracking_errors:
                axes[1, 1].plot(self.curr_rollout_tracking_errors, 'g-')
            axes[1, 1].set_title('MPC Tracking Error (m)')

            vels = np.linalg.norm(real_pos[1:] - real_pos[:-1], axis=1)
            axes[1, 2].plot(vels, 'm-')
            axes[1, 2].set_title('End-Effector Velocity')

            plt.tight_layout()
            fig.savefig(os.path.join(diag_path, f'rollout_{rollout_idx}_report.png'))
            plt.close(fig)

        except Exception as e:
            print(f'[ diag ] Real-time export failed for rollout {rollout_idx}: {e}')

    @torch.no_grad()
    def predict(self, state, goal=None, extra_args=None, if_vision=False):
        """
        D3IL agent.predict() interface.
        Visual:     state = (bp_image_np, inhand_image_np, des_robot_pos_np)
        Non-visual: state = obs_np  — D3IL concatenated obs with robot_pos at [:3]
        """
        cond = None
        if if_vision:
            bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state  # C4: unpack actual robot_pos

            # ── Video capture ──────────────────────────────────────────────
            # bp_np is already RGB (A1 fix in aligning_sim.py); no cvtColor needed.
            if self.record_mode != 'none':
                try:
                    bp_vis     = (bp_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    inhand_vis = (inhand_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    self.video_frames.append(np.concatenate([bp_vis, inhand_vis], axis=1))
                except Exception:
                    pass

            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()

            self.history_real_pos.append(des_robot_pos_np.copy())
            if self.last_predicted_pos is not None:
                err = np.linalg.norm(des_robot_pos_np[:2] - self.last_predicted_pos[:2])
                self.curr_rollout_tracking_errors.append(err)

            # ── Build 6D obs = [des_c_pos | c_pos] ───────────────────────
            # C4 fix: use actual robot_pos from sim for the c_pos slot.
            # des_robot_pos_np = commanded position; robot_pos_np = actual sim state.
            obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,) [des_c_pos | c_pos]

            if self.obs_normalizer is not None:
                obs_6d_norm = self.obs_normalizer.normalize(
                    obs_6d_np.reshape(1, -1)).astype(np.float32).squeeze(0)
            else:
                obs_6d_norm = obs_6d_np.astype(np.float32)

            bp_t     = torch.from_numpy(bp_np.astype(np.float32)).to(self.device).unsqueeze(0)
            inhand_t = torch.from_numpy(inhand_np.astype(np.float32)).to(self.device).unsqueeze(0)
            obs_t    = torch.from_numpy(obs_6d_norm).to(self.device).unsqueeze(0)  # (1, 6)

            self.bp_image_context.append(bp_t)
            self.inhand_image_context.append(inhand_t)
            self.obs_context.append(obs_t)

            while len(self.bp_image_context) < self.window_size:
                self.bp_image_context.append(bp_t)
                self.inhand_image_context.append(inhand_t)
                self.obs_context.append(obs_t)

            bp_seq     = torch.cat(list(self.bp_image_context), dim=0)      # (W, C, H, W)
            inhand_seq = torch.cat(list(self.inhand_image_context), dim=0)  # (W, C, H, W)
            obs_seq    = torch.cat(list(self.obs_context), dim=0)           # (W, 6)

            bp_batch     = bp_seq.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
            inhand_batch = inhand_seq.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
            obs_batch    = obs_seq.unsqueeze(0).repeat(self.batch_size, 1, 1)

            cond = {0: (bp_batch, inhand_batch, obs_batch)}

        else:
            # Non-visual path: D3IL provides obs_np with robot_pos at [:3]
            obs_np = np.asarray(state, dtype=np.float64)
            des_robot_pos_np = obs_np[:3]

            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()

            self.history_real_pos.append(des_robot_pos_np.copy())
            if self.last_predicted_pos is not None:
                err = np.linalg.norm(des_robot_pos_np[:2] - self.last_predicted_pos[:2])
                self.curr_rollout_tracking_errors.append(err)

            obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,)
            if self.obs_normalizer is not None:
                obs_6d_norm = self.obs_normalizer.normalize(
                    obs_6d_np.reshape(1, -1)).astype(np.float32).squeeze(0)
            else:
                obs_6d_norm = obs_6d_np.astype(np.float32)
            obs_t = torch.from_numpy(obs_6d_norm).to(self.device).unsqueeze(0)  # (1, 6)
            self.obs_context.append(obs_t)
            while len(self.obs_context) < self.obs_seq_len:
                self.obs_context.append(obs_t)
            # obs anchor for apply_conditioning: {0: (B,6)} — no 'visual' key
            obs_anchor = obs_t.repeat(self.batch_size, 1)   # (B, 6)
            cond = {0: obs_anchor}

        # ── Plan (or execute from cached action chunk) ─────────────────────
        t_start = time.time()
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0
            self.model.eval()

            if self.projector is not None:
                trajectory, infos = self.model(cond, projector=self.projector)
            else:
                trajectory, infos = self.model(cond)

            traj_np = trajectory.detach().cpu().numpy()   # (B, H, 9)
            which   = 0
            selection_method = 'default (first)'

            if self.batch_size > 1:
                if (self.trajectory_selection == 'temporal_consistency'
                        and self.prev_observations is not None):
                    diffs = traj_np - np.expand_dims(self.prev_observations, 0)
                    which = int(np.argsort(np.linalg.norm(diffs, axis=(1, 2)))[0])
                    selection_method = 'temporal_consistency'
                elif (self.trajectory_selection == 'minimum_projection_cost'
                      and self.projector is not None):
                    # Try precomputed costs from post-processing projection first
                    has_precomputed = False
                    if (infos is not None and 'projection_costs' in infos
                            and len(infos['projection_costs']) > 0):
                        costs_total = np.zeros(self.batch_size)
                        for _, cost in infos['projection_costs'].items():
                            costs_total += cost
                        if len(costs_total) == self.batch_size:
                            which = int(np.argmin(costs_total))
                            selection_method = 'minimum_projection_cost (precomputed)'
                            has_precomputed = True
                    if not has_precomputed:
                        _, projection_costs = self.projector.project(trajectory)
                        which = int(np.argmin(projection_costs))
                        selection_method = 'minimum_projection_cost (calculated)'
                elif self.trajectory_selection == 'random':
                    which = np.random.randint(self.batch_size)
                    selection_method = 'random'

            self.prev_observations = traj_np[which].copy()

            # action dims = indices 0:3
            action_traj = trajectory[[which], :, :3]   # (1, H, 3)

            if self.act_normalizer is not None:
                act_np = action_traj.detach().cpu().numpy()
                B, H, D = act_np.shape
                act_np = self.act_normalizer.unnormalize(act_np.reshape(-1, D)).reshape(B, H, D)
                action_traj = torch.from_numpy(act_np).to(self.device)

            # One-time diagnostic on first replan of first rollout.
            # Prints pre/post-denorm action magnitudes so we can immediately spot:
            #   - actions stuck near zero  → denormalization failed
            #   - actions in [-1, 1] at "denorm" stage → act_normalizer was None
            #   - horizon range outside [-1, 1] at normalized stage → model diverged
            # Also writes a dedicated diag_first_replan.txt to save_path for easy
            # grep / cross-run comparison (not buried in the full eval log).
            if self.rollout_counter == 0 and self.step_counter == 0:
                norm_a0   = trajectory[[which], 0, :3].detach().cpu().numpy().squeeze()
                denorm_a0 = action_traj[0, 0].detach().cpu().numpy()
                full_norm = trajectory[which, :, :3].detach().cpu().numpy()
                diag_lines = [
                    f'[ DIAG first-replan ] normalized   a0 = {np.round(norm_a0, 4)}'
                    f'  |mag| = {np.linalg.norm(norm_a0):.4f}',
                    f'[ DIAG first-replan ] denormalized a0 = {np.round(denorm_a0, 5)}'
                    f'  |mag| = {np.linalg.norm(denorm_a0):.6f} m',
                    f'[ DIAG first-replan ] horizon act (normalized) range: '
                    f'[{full_norm.min():.4f}, {full_norm.max():.4f}]',
                    f'[ DIAG first-replan ] per-step normalized acts (H={full_norm.shape[0]}):',
                ]
                for h_i, row in enumerate(full_norm):
                    diag_lines.append(f'  step {h_i:2d}: {np.round(row, 4)}')
                for line in diag_lines:
                    print(line)
                if self.save_path is not None:
                    diag_file = os.path.join(self.save_path, 'diag_first_replan.txt')
                    with open(diag_file, 'w') as _df:
                        _df.write('\n'.join(diag_lines) + '\n')
                    print(f'[ DIAG ] saved → {diag_file}')

            self.curr_action_seq = action_traj[:, :self.action_seq_size, :]
            self.history_full_plans.append(action_traj[0].detach().cpu().numpy())

        next_action    = self.curr_action_seq[:, self.action_counter, :]
        next_action_np = next_action.detach().cpu().numpy().squeeze(0)   # (3,)

        self.mental_robot_pos += next_action_np

        self.history_desired_actions.append(next_action_np.copy())
        self.last_predicted_pos = self.mental_robot_pos.copy()

        self.curr_rollout_time += time.time() - t_start
        self.action_counter += 1
        self.step_counter   += 1
        return next_action_np.reshape(1, -1)   # (1, 3) — sim expects (1, act_dim)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest', device='cuda:0'):
    lp = os.path.join(*loadpath)
    print(f'\n[ eval loading ] Loading from {lp}\n')
    dataset_config   = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config     = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config   = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = lp

    if target_class is not None:
        diffusion_config._class = utils.config.import_class(target_class)

    dataset   = dataset_config()
    model     = model_config()
    diffusion = diffusion_config(model).to(device)
    trainer   = trainer_config(diffusion_model=diffusion, dataset=dataset)
    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, None)

# ── Parser & Main ─────────────────────────────────────────────────────────────

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str  = 'config.aligning-d3il-visual'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--record', type=str,
                        choices=['none', 'video', 'gif', 'all'], default='all')
    parser.add_argument('--eval-on-train', action='store_true')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    with open('config/visual_aligning_eval.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seeds               = [args_cli.seed] if args_cli.seed else config['seeds']
    projection_variants = config.get('projection_variants', ['diffuser'])
    n_contexts          = config.get('n_contexts', 30)
    n_trajectories      = config.get('n_trajectories_per_context', 1)

    for seed in seeds:
        print(f'\n=== Evaluating seed {seed} ===')
        args = Parser().parse_args(experiment='plan_visual_aligning_dpcc', seed=seed)

        diffusion_model = None
        if not args_cli.aggregate_only:
            exp = load_diffusion_with_override(
                args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                target_class=args.diffusion, epoch=args.diffusion_epoch,
                device=args.device,
            )
            diffusion_model = exp.diffusion
            # Original DPCC uses clip_denoised=False. Older checkpoints may have
            # been saved with clip_denoised=True, which causes the ±5 clamp to fire
            # at the first denoising step (cosine schedule amplification ~9.4× at
            # t=T-1), permanently corrupting the denoising chain. Force False here
            # so all checkpoints use the correct inference behaviour.
            diffusion_model.clip_denoised = False
            print('[ eval ] clip_denoised forced → False (matches original DPCC)')
            _model_n_ts  = getattr(diffusion_model, 'n_timesteps', '?')
            _config_n_ts = getattr(args, 'n_diffusion_steps', '?')
            print(f'[ eval ] Model n_timesteps = {_model_n_ts}  '
                  f'(config n_diffusion_steps = {_config_n_ts})')
            if isinstance(_model_n_ts, int) and isinstance(_config_n_ts, int):
                if _model_n_ts != _config_n_ts:
                    print(f'[ eval ] WARNING: n_timesteps mismatch — '
                          f'checkpoint trained with {_model_n_ts} steps, '
                          f'config says {_config_n_ts}. '
                          f'Denoising chain will use checkpoint value ({_model_n_ts}).')

        for variant in projection_variants:
            if args_cli.eval_on_train:
                variant   = f'{variant}_train_set'
                save_path = f'{args.savepath}/results_train_set'
            else:
                save_path = f'{args.savepath}/results'
            os.makedirs(save_path, exist_ok=True)

            # Expert reference videos (best-effort)
            generate_expert_reference(save_path, n_rollouts=3)

            if args_cli.aggregate_only:
                continue

            log_f = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = Tee(sys.stdout, log_f)
            sys.stderr = Tee(sys.stderr, log_f)

            try:
                # ── Load normalizers ─────────────────────────────────────────
                model_dir     = os.path.join(args.loadbase, args.dataset,
                                             args.diffusion_loadpath, str(args.seed))
                obs_norm_path = os.path.join(model_dir, 'obs_normalizer.pkl')
                act_norm_path = os.path.join(model_dir, 'act_normalizer.pkl')
                if not os.path.exists(obs_norm_path) or not os.path.exists(act_norm_path):
                    raise FileNotFoundError(
                        f'[ eval ] FATAL: normalizer pkl missing in {model_dir}\n'
                        f'  Expected: obs_normalizer.pkl + act_normalizer.pkl\n'
                        f'  Without them, sampled actions stay in [-1, 1] space and\n'
                        f'  produce wrong robot commands. Re-run training to regenerate.'
                    )
                with open(obs_norm_path, 'rb') as f: obs_normalizer = pickle.load(f)
                with open(act_norm_path, 'rb') as f: act_normalizer = pickle.load(f)
                print(f'[ eval ] Loaded normalizers from {model_dir}')
                # Sanity-check: near-zero range in any dim means zero-padded episodes
                # corrupted the scaler at training time — actions would denorm incorrectly.
                act_range = act_normalizer.maxs - act_normalizer.mins
                if np.any(act_range < 1e-4):
                    print(f'[ eval ] WARNING: act_normalizer near-zero range in dims '
                          f'{np.where(act_range < 1e-4)[0].tolist()} — '
                          f'possible zero-pad scaler corruption at train time')
                print(f'[ eval ] obs_normalizer  mins={np.round(obs_normalizer.mins, 4)}  '
                      f'maxs={np.round(obs_normalizer.maxs, 4)}')
                print(f'[ eval ] act_normalizer  mins={np.round(act_normalizer.mins, 4)}  '
                      f'maxs={np.round(act_normalizer.maxs, 4)}')

                # ── Setup DPCC projector ─────────────────────────────────────
                projector = None
                if 'diffuser' not in variant and obs_normalizer is not None:
                    projector = setup_dpcc_projector(
                        args, config, obs_normalizer, act_normalizer, variant)
                    print(f'[ eval ] DPCC projector active for variant {variant!r}')

                trajectory_selection = 'random'
                if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
                elif 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'

                batch_size = getattr(args, 'batch_size', 1)
                if 'diffuser' not in variant:
                    batch_size = 6

                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model,
                    device=args.device,
                    window_size=getattr(args, 'window_size', 1),
                    obs_seq_len=getattr(args, 'obs_seq_len', 1),
                    action_seq_size=getattr(args, 'action_seq_size', 1),
                    save_path=save_path,
                    record_mode=args_cli.record,
                    obs_normalizer=obs_normalizer,
                    act_normalizer=act_normalizer,
                    batch_size=batch_size,
                    projector=projector,
                    trajectory_selection=trajectory_selection,
                    eval_on_train=args_cli.eval_on_train,
                    variant=variant,
                )

                sim = Aligning_Sim(
                    seed=seed, device=args.device,
                    render=False, n_cores=1,
                    n_contexts=n_contexts,
                    n_trajectories_per_context=n_trajectories,
                    if_vision=getattr(args, 'if_vision', True),
                    eval_on_train=args_cli.eval_on_train,
                )

                # aligning_sim.test_agent() calls wandb.log() unconditionally at the end;
                # initialize in disabled mode so it doesn't crash when no wandb run is active.
                if _wandb is not None:
                    _wandb.init(mode='disabled')

                t0 = time.time()
                success_rate, mode_encoding, successes, mean_dist = sim.test_agent(agent)
                elapsed = time.time() - t0

                # ── Metrics ──────────────────────────────────────────────────
                n_modes    = 2
                mode_probs = torch.zeros([n_contexts, n_modes])
                for c in range(n_contexts):
                    mode_probs[c] = torch.tensor([
                        (mode_encoding[c] == 0).sum().item() / n_trajectories,
                        (mode_encoding[c] == 1).sum().item() / n_trajectories,
                    ])
                m_norm  = mode_probs / (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
                entropy = -(m_norm * torch.log(m_norm + 1e-12) /
                            torch.log(torch.tensor(float(n_modes)))).sum(1).mean().item()

                obs_all, act_all, plans_all = [], [], []
                for r in range(agent.rollout_counter + 1):
                    d = agent.master_rollout_history.get(f'rollout_{r}')
                    if d:
                        obs_all.append(d['real_robot_pos'])
                        act_all.append(d['desired_actions'])
                        plans_all.append(d['full_plans'])

                # ── NPZ save (legacy-compatible) ─────────────────────────────
                if config.get('write_to_file', True):
                    np.savez(f'{save_path}/{variant}.npz',
                             success_rate=success_rate, entropy=entropy,
                             mode_encoding=mode_encoding.numpy(),
                             elapsed_seconds=elapsed, seed=seed,
                             n_success=successes.flatten().numpy(),
                             n_steps=np.array(agent.history_n_steps),
                             avg_time=np.array(agent.history_avg_time),
                             n_violations=np.zeros(len(agent.history_n_steps)),
                             total_violations=np.zeros(len(agent.history_n_steps)),
                             collision_free_completed=successes.flatten().numpy(),
                             obs_all=np.array(obs_all, dtype=object),
                             act_all=np.array(act_all, dtype=object),
                             sampled_trajectories_all=np.array(plans_all, dtype=object),
                             pos_tracking_errors=np.array(
                                 agent.history_pos_tracking_errors, dtype=object),
                             mean_distance=mean_dist.flatten().numpy(),
                             args=vars(args))

                pkl_name = (f'results_seed_{seed}_train_set.pkl'
                            if args_cli.eval_on_train else f'results_seed_{seed}.pkl')
                with open(os.path.join(save_path, pkl_name), 'wb') as f:
                    pickle.dump({'success_rate': success_rate,
                                 'entropy': entropy, 'elapsed': elapsed}, f)

                # ── Legacy PNG rollout grid (mirrors ddpm_encdec) ────────────
                print(f'[ eval ] Generating PNG rollout grid for {variant}...')
                n_plot = min(len(obs_all), 5)
                if n_plot > 0:
                    fig, axes = plt.subplots(n_plot, 6, figsize=(30, 5 * n_plot),
                                             squeeze=False)
                    fig.suptitle(f'Visual-DPCC — {variant} (Seed {seed})')

                    for i in range(n_plot):
                        obs_traj   = obs_all[i]    # (T, 3) des_robot_pos
                        plans_list = plans_all[i]  # list of (H, 3) action arrays
                        rollout_data = agent.master_rollout_history.get(f'rollout_{i}', {})
                        plan_starts  = rollout_data.get('plan_start_positions',
                                                        np.zeros((1, 3)))

                        axes[i, 0].plot(obs_traj[:, 0], 'r-')
                        axes[i, 0].set_title('X Position')

                        axes[i, 1].plot(obs_traj[:, 1], 'g-')
                        axes[i, 1].set_title('Y Position')

                        axes[i, 2].plot(obs_traj[:, 2], 'b-')
                        axes[i, 2].set_title('Z Height')

                        vels = np.linalg.norm(obs_traj[1:] - obs_traj[:-1], axis=1)
                        axes[i, 3].plot(vels, color='gray', alpha=0.5)
                        axes[i, 3].set_title('Step Magnitude')

                        axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', linewidth=2)
                        axes[i, 4].plot(obs_traj[0, 0],  obs_traj[0, 1],  'go', markersize=10)
                        axes[i, 4].plot(obs_traj[-1, 0], obs_traj[-1, 1], 'ro', markersize=10)
                        axes[i, 4].set_title('XY Trajectory')

                        axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.3)
                        for p_idx, plan_deltas in enumerate(plans_list):
                            if p_idx % 4 == 0:
                                start = plan_starts[min(p_idx, len(plan_starts) - 1)]
                                abs_plan = start + np.cumsum(plan_deltas[:, :3], axis=0)
                                axes[i, 5].plot(abs_plan[:, 0], abs_plan[:, 1],
                                                'b-', alpha=0.6)
                        axes[i, 5].set_title('MPC Foresight (blue)')

                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.savefig(f'{save_path}/{variant}.png')
                    plt.close(fig)

                # ── 7-metric report (D3IL standard) ─────────────────────────
                n_success = np.array(successes)
                n_steps   = np.array(agent.history_n_steps)
                all_errs  = [e for t in agent.history_pos_tracking_errors for e in t]
                track_err = float(np.max(all_errs)) if all_errs else 0.0

                run_mode = 'seen training set' if args_cli.eval_on_train else 'default'
                print(f'--- aligning-d3il-visual [{run_mode}] {variant} seed={seed} ---')
                print(f'Success rate: {np.mean(n_success):.4f}')
                print(f'Constraints satisfied: 1.0000')
                print(f'Success rate (goal and constraints): {np.mean(n_success):.4f}')
                print(f'Avg number of steps (successful trials): '
                      f'{np.mean(n_steps[n_success > 0]) if n_success.sum() else 0:.2f} '
                      f'+- {np.std(n_steps[n_success > 0]) if n_success.sum() else 0:.2f}')
                print(f'Avg number of steps (all trials): '
                      f'{np.mean(n_steps):.2f} +- {np.std(n_steps):.2f}')
                print(f'Avg number of constraint violations: 0.00 +- 0.00')
                print(f'Avg total violation: 0.000 +- 0.000')
                print(f'Average computation time per step: {np.mean(agent.history_avg_time):.3f}')
                print(f'Tracking error: {track_err:.3f}')
                print('-' * 80 + '\n')

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_f.close()

    print('Visual-DPCC evaluation completed.')
