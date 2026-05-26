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
#          ├── expert_references/expert_rollout_<r>.{mp4,gif}   (generated once before variant loop)
#          └── <variant>/
#              ├── <variant>.npz
#              ├── <variant>.png     (6-panel rollout grid)
#              ├── results_seed_<s>.pkl
#              ├── eval_<variant>.log
#              ├── diag_first_replan.txt
#              └── diagnostics/rollout_<r>.{mp4,gif,_data.pkl,_stats.json,_report.png,_mpc_foresight.png}

import gc
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
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection
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

def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant, is_tightened=False):
    """
    Build the DPCC SLSQP projector for the 9D trajectory space.

    Trajectory layout: [dx(0) dy(1) dz(2) | des_x(3) des_y(4) des_z(5) | x(6) y(7) z(8)]

    Constraints:
        - Workspace bounds on actual EE position (c_pos, indices 6-8)
        - Euler dynamics: c_pos[t+1] = c_pos[t] + act[t]  (indices [6←0, 7←1, 8←2])
        - Obstacle exclusion: sphere_outside on EE position dims (from obstacle_constraints)
    """
    # Named-dim map: yaml obstacle_constraints may use strings instead of raw indices.
    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'des_x': 3, 'des_y': 4, 'des_z': 5,
            'x': 6, 'y': 7, 'z': 8}

    constraint_list = []

    if 'bounds' in config.get('constraint_types', []):
        tightening = config.get('enlarge_constraints') or 0.0
        ws_lb = np.array(config['workspace_bounds']['lb'])   # (3,)
        ws_ub = np.array(config['workspace_bounds']['ub'])   # (3,)
        if is_tightened and tightening > 0.0:
            ws_lb += tightening   # lower bound rises — smaller box
            ws_ub -= tightening   # upper bound drops  — smaller box
        # Bounds only on c_pos dims (indices 6,7,8); act and des_c_pos unconstrained
        lb = np.concatenate([np.full(6, -np.inf), ws_lb])   # (9,)
        ub = np.concatenate([np.full(6,  np.inf), ws_ub])   # (9,)
        constraint_list.append(['lb', lb])
        constraint_list.append(['ub', ub])

    if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
        constraint_list.append(('deriv', [6, 0]))   # c_pos_x ← dx
        constraint_list.append(('deriv', [7, 1]))   # c_pos_y ← dy
        constraint_list.append(('deriv', [8, 2]))   # c_pos_z ← dz

    if 'halfspace' in config.get('constraint_types', []):
        # Oriented linear inequality constraints in the x-y plane (EE horizontal position).
        # Same formulation as original DPCC avoiding paper.
        # Each entry: [[x1,y1], [x2,y2], 'above'/'below'] — line through two points, keep the named side.
        # Tightening shifts the boundary inward by enlarge_constraints (metres).
        tightening = config.get('enlarge_constraints') or 0.0
        _hs_indices = {'x': _DIM['x'], 'y': _DIM['y']}   # EE x=6, y=7 in 9D trajectory
        for hs in config.get('halfspace_constraints', []):
            margin = tightening if is_tightened else 0.0
            C_row, d = utils.formulate_halfspace_constraints(hs, margin, 9, _hs_indices)
            constraint_list.append(('ineq', (C_row, d)))

    if 'obstacles' in config.get('constraint_types', []):
        tightening = config.get('enlarge_constraints') or 0.0
        for obs in config.get('obstacle_constraints', []):
            dims = [_DIM[d] if isinstance(d, str) else int(d) for d in obs['dimensions']]
            radius = obs['radius'] + (tightening if is_tightened else 0.0)  # larger exclusion when tightened
            constraint_list.append((obs['type'], dims, obs['center'], radius))

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
                 eval_on_train=False, variant='unspecified',
                 max_action_delta=None,
                 mpc_foresight_stride=6):

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
        self.history_all_candidates      = []   # Fix 8: (B,H,3) per replan step, all candidates
        self.history_selected_idx        = []   # Fix 8: which index was chosen per replan step
        self.history_n_steps             = []
        self.history_avg_time            = []
        self.history_rollout_mean_dist   = []   # Fix 9: mean_distance per rollout for summary
        self.history_pos_tracking_errors = []
        self.curr_rollout_tracking_errors = []
        self.curr_rollout_all_candidates  = []  # Fix 8/9: per-rollout accumulator (stores c_pos dims)
        self.curr_rollout_selected_idx    = []  # Fix 8: per-rollout accumulator
        self.curr_rollout_c_pos           = []  # Fix 9: actual robot position per step
        self.curr_context_info            = {}  # Fix 10: set by record_context_info each rollout
        self.history_context_info         = []  # Fix 10: per-rollout context records
        self.curr_rollout_time           = 0
        self.master_rollout_history      = {}
        self.video_frames                = []
        self.max_action_delta            = max_action_delta
        self.mpc_foresight_stride        = mpc_foresight_stride
        self.curr_rollout_act_magnitudes = []
        self.curr_rollout_dist_to_target = []
        self.curr_rollout_clamp_events   = []
        self.history_act_magnitudes      = []
        self.history_dist_to_target      = []
        self.history_clamp_events        = []
        self._replan_count               = 0

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
        self.curr_rollout_c_pos.clear()            # Fix 9
        self.curr_context_info = {}                # Fix 10
        self.history_real_pos.clear()
        self.history_desired_actions.clear()
        self.history_full_plans.clear()
        self.curr_rollout_all_candidates.clear()   # Fix 8
        self.curr_rollout_selected_idx.clear()     # Fix 8
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.obs_context.clear()
        self.video_frames.clear()
        self.curr_rollout_act_magnitudes.clear()
        self.curr_rollout_dist_to_target.clear()
        self.curr_rollout_clamp_events.clear()
        self._replan_count = 0

    def update_rollout_info(self, info):
        """Called by Aligning_Sim at rollout end. Mirrors ddpm_encdec verbose format."""
        success   = info.get('success', False)
        mean_dist = info.get('mean_distance', 0.0)
        mode      = info.get('mode', 0)
        ridx      = int(info.get('context', self.rollout_counter))

        max_phys_err = float(np.max(self.curr_rollout_tracking_errors)
                             if self.curr_rollout_tracking_errors else 0.0)
        avg_time = float(self.curr_rollout_time / max(1, self._replan_count))  # Fix 12: per-replan avg

        self.master_rollout_history[f'rollout_{ridx}'] = {
            'real_robot_pos':      np.array(self.history_real_pos),
            'c_pos_history':       np.array(self.curr_rollout_c_pos),         # Fix 9
            'desired_actions':     np.array(self.history_desired_actions),
            'full_plans':          np.array(self.history_full_plans),
            'all_candidates':      list(self.curr_rollout_all_candidates),    # Fix 8/9
            'selected_idx':        list(self.curr_rollout_selected_idx),      # Fix 8
            'success':   bool(success),
            'mean_distance': float(mean_dist),
            'mode':      int(mode),
            'steps':     int(self.step_counter),
            'avg_time':  avg_time,
            'max_physical_tracking_error': max_phys_err,                      # Fix 9
            'act_magnitudes':     list(self.curr_rollout_act_magnitudes),
            'dist_to_target':     list(self.curr_rollout_dist_to_target),
            'clamp_events':       list(self.curr_rollout_clamp_events),
            'context_info':       dict(self.curr_context_info),               # Fix 10
        }
        self.history_n_steps.append(self.step_counter)
        self.history_avg_time.append(avg_time)
        self.history_rollout_mean_dist.append(float(mean_dist))               # Fix 9
        self.history_pos_tracking_errors.append(
            np.array(self.curr_rollout_tracking_errors))
        self.history_all_candidates.append(list(self.curr_rollout_all_candidates))  # Fix 8
        self.history_selected_idx.append(list(self.curr_rollout_selected_idx))      # Fix 8
        self.history_act_magnitudes.append(list(self.curr_rollout_act_magnitudes))
        self.history_dist_to_target.append(list(self.curr_rollout_dist_to_target))
        self.history_clamp_events.append(list(self.curr_rollout_clamp_events))
        self.history_context_info.append(dict(self.curr_context_info))        # Fix 10

        ctx_type = 'Seen Training Context' if self.eval_on_train else 'Unseen Test Context'
        ci = self.curr_context_info
        print(f'[ {ctx_type} {ridx} Finished ]')
        if ci:                                                                 # Fix 10
            print(f'  - Context idx: {ci.get("context_idx")}')
            print(f'  - Box  init XY=({ci["box_init_xy"][0]:.3f}, {ci["box_init_xy"][1]:.3f})  '
                  f'angle={ci["box_init_angle_deg"]:.1f}°')
            print(f'  - Target   XY=({ci["target_xy"][0]:.3f}, {ci["target_xy"][1]:.3f})  '
                  f'angle={ci["target_angle_deg"]:.1f}°')
            print(f'  - Init XY dist (box→target): {ci["init_xy_dist"]:.4f} m')
        print(f'  - Total Steps: {self.step_counter}')
        print(f'  - Success status: {success}')
        print(f'  - Final Mean Distance: {mean_dist:.6f} m')
        print(f'  - Environment Mode: {mode}')
        print(f'  - Max Physical Tracking Error: {max_phys_err:.6f} m')       # Fix 9
        print(f'  - Avg Inference Time: {avg_time:.4f} seconds/replan')
        print(f'  - Clamp events: {len(self.curr_rollout_clamp_events)}')
        print('-' * 80 + '\n')

        if self.save_path is not None:
            self._export_rollout_realtime(ridx)   # Fix 9: handles PNG+JSON+pkl+video

    def record_step_info(self, info):
        """Called by Aligning_Sim after each env.step() — accumulates per-step mean_distance."""
        d = info.get('mean_distance')
        if d is not None:
            self.curr_rollout_dist_to_target.append(float(d))

    def record_context_info(self, context, context_idx):
        """Called by Aligning_Sim after reset — stores initial scene config. Fix 10."""
        pos, quat, target_pos, target_quat = context
        init_xy_dist = float(np.linalg.norm(
            np.array([pos[0], pos[1]]) - np.array([target_pos[0], target_pos[1]])
        ))
        self.curr_context_info = {
            'context_idx':        int(context_idx),
            'box_init_xy':        [float(pos[0]), float(pos[1])],
            'box_init_angle_deg': float(pos[2]),
            'target_xy':          [float(target_pos[0]), float(target_pos[1])],
            'target_angle_deg':   float(target_pos[2]),
            'init_xy_dist':       init_xy_dist,
        }

    # Fix 9: _save_diagnostics() removed — video/gif + JSON now consolidated in _export_rollout_realtime().

    def _export_rollout_realtime(self, rollout_idx):
        """Per-rollout PNG (9-panel) + JSON + pkl + video. Fix 9: consolidated into diagnostics/."""
        try:
            diag_path = os.path.join(self.save_path, 'diagnostics')
            os.makedirs(diag_path, exist_ok=True)

            data = self.master_rollout_history[f'rollout_{rollout_idx}']

            # ── Video / GIF (Fix 9: moved from _save_diagnostics) ─────────
            if self.record_mode != 'none' and self.video_frames:
                if self.record_mode in ['video', 'all']:
                    try:
                        imageio.mimsave(os.path.join(diag_path, f'rollout_{rollout_idx}.mp4'),
                                        self.video_frames, fps=20)
                    except Exception as e:
                        print(f'[ WARNING ] MP4 failed: {e}')
                if self.record_mode in ['gif', 'all']:
                    try:
                        imageio.mimsave(os.path.join(diag_path, f'rollout_{rollout_idx}.gif'),
                                        self.video_frames, fps=10)
                    except Exception as e:
                        print(f'[ WARNING ] GIF failed: {e}')

            with open(os.path.join(diag_path, f'rollout_{rollout_idx}_data.pkl'), 'wb') as f:
                pickle.dump(data, f)

            # Fix 9: JSON only (no .txt duplicate); Fix 10: context_info added
            stats = {
                'rollout_index':                  int(rollout_idx),
                'success':                        bool(data.get('success', False)),
                'steps':                          int(data.get('steps', 0)),
                'mean_distance':                  float(data.get('mean_distance', 0.0)),
                'mode':                           int(data.get('mode', 0)),
                'avg_inference_time_per_replan':  float(data.get('avg_time', 0.0)),  # Fix 12
                'max_physical_tracking_error':    float(data.get('max_physical_tracking_error', 0.0)),
                'context_info':                   data.get('context_info', {}),
            }
            with open(os.path.join(diag_path, f'rollout_{rollout_idx}_stats.json'), 'w') as sf:
                json.dump(stats, sf, indent=4)

            real_pos  = data['real_robot_pos']       # (T, 3) des_robot_pos
            c_pos_h   = data.get('c_pos_history', None)  # Fix 9: actual robot positions

            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Rollout {rollout_idx} — MPC vs Real  '
                         f'(success={data.get("success")})')

            # Row 0 — Spatial
            # Fix 9 I2: cands are unnormalized c_pos XY, no cumsum; green=selected, gray=others
            all_cands_list = data.get('all_candidates', [])
            sel_idx_list   = data.get('selected_idx',   [])
            axes[0, 0].plot(real_pos[:, 0], real_pos[:, 1], 'k-', linewidth=2, label='des path')
            for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                if step_i % 4 != 0:
                    continue
                for b in range(cands.shape[0]):
                    if b == sel:
                        axes[0, 0].plot(cands[b, :, 0], cands[b, :, 1],
                                        color='green', linewidth=0.8, alpha=0.9, zorder=5)
                    else:
                        axes[0, 0].plot(cands[b, :, 0], cands[b, :, 1],
                                        color='gray', linewidth=0.2, alpha=0.2)
            n_cands = all_cands_list[0].shape[0] if all_cands_list else 1
            axes[0, 0].set_title(f'XY — MPC foresight  (green=selected, {n_cands} candidates/step)')
            axes[0, 0].set_xlabel('X (m)'); axes[0, 0].set_ylabel('Y (m)')
            axes[0, 0].legend()

            # Fix 9 I6: overlay c_pos (red dashed) on X/Y position panels
            axes[0, 1].plot(real_pos[:, 0], 'k-', label='des')
            if c_pos_h is not None and len(c_pos_h):
                axes[0, 1].plot(c_pos_h[:, 0], 'r--', label='actual')
            axes[0, 1].set_title('X — des (black) vs actual (red)')
            axes[0, 1].set_ylabel('Meters')

            axes[0, 2].plot(real_pos[:, 1], 'k-', label='des')
            if c_pos_h is not None and len(c_pos_h):
                axes[0, 2].plot(c_pos_h[:, 1], 'r--', label='actual')
            axes[0, 2].set_title('Y — des (black) vs actual (red)')

            # Row 1 — Task Progress
            dist_curve = data.get('dist_to_target', [])
            if dist_curve:
                axes[1, 0].plot(dist_curve, 'r-')
                axes[1, 0].axhline(0, color='g', linestyle='--', alpha=0.5, label='target')
                axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_title('Distance to Target over Steps')
            axes[1, 0].set_ylabel('m')

            axes[1, 1].plot(real_pos[:, 2], 'k-', label='Z des')
            if c_pos_h is not None and len(c_pos_h):
                axes[1, 1].plot(np.array(c_pos_h)[:, 2], 'r--', alpha=0.7, label='Z actual')
                axes[1, 1].legend(fontsize=7)
            axes[1, 1].set_title('Z — des (black) vs actual (red)')
            axes[1, 1].set_ylabel('Meters')

            # Fix 9: physical tracking error |c_pos - des_c_pos|
            phys_errs = data.get('dist_to_target', [])  # fallback
            errs_from_hist = list(self.curr_rollout_tracking_errors)
            if errs_from_hist:
                axes[1, 2].plot(errs_from_hist, 'g-')
            axes[1, 2].set_title('Physical Tracking Error |c_pos - des| (m)')

            # Row 2 — Action Quality
            act_mags = data.get('act_magnitudes', [])
            if act_mags:
                axes[2, 0].plot(act_mags, 'c-')
            axes[2, 0].set_title('Action Magnitude per Step (m)')
            axes[2, 0].set_ylabel('m')

            vels = np.linalg.norm(real_pos[1:] - real_pos[:-1], axis=1)
            axes[2, 1].plot(vels, 'm-')
            axes[2, 1].set_title('End-Effector Velocity')

            clamp_events = data.get('clamp_events', [])
            if clamp_events:
                ce_steps = [e[0] for e in clamp_events]
                ce_mags  = [e[1] for e in clamp_events]
                axes[2, 2].scatter(ce_steps, ce_mags, c='r', s=20, zorder=3)
                if self.max_action_delta is not None:
                    axes[2, 2].axhline(self.max_action_delta, color='orange',
                                       linestyle='--', alpha=0.7,
                                       label=f'limit={self.max_action_delta}m')
                    axes[2, 2].legend(fontsize=8)
            axes[2, 2].set_title(f'Clamp Events ({len(clamp_events)} total)')
            axes[2, 2].set_xlabel('Step'); axes[2, 2].set_ylabel('Raw |a| (m)')

            plt.tight_layout()
            fig.savefig(os.path.join(diag_path, f'rollout_{rollout_idx}_report.png'))
            plt.close(fig)

            # ── Standalone high-res MPC decision-point plot: XY (left) + XYZ 3D (right) ──
            # U11: all candidates uniform green solid; replan decision-point dots (black);
            #      des=black solid, actual=red solid (no dashes); PNG@200DPI + SVG.
            if all_cands_list:
                from matplotlib.lines import Line2D as _Line2D
                _STRIDE   = self.mpc_foresight_stride   # U11.2: yaml-settable (mpc_foresight_stride)
                n_steps   = len(real_pos)
                n_replans = len(all_cands_list)
                spr       = max(1, n_steps // max(1, n_replans))  # env steps per replan
                c_arr = (np.array(c_pos_h)
                         if (c_pos_h is not None and len(c_pos_h)) else None)

                fig_mpc = plt.figure(figsize=(26, 11))
                fig_mpc.suptitle(
                    f'Rollout {rollout_idx} — MPC Decision Points  '
                    f'(success={data.get("success")},  {n_cands} candidates/step,  '
                    f'every {_STRIDE} replans shown)',
                    fontsize=13)
                ax_xy = fig_mpc.add_subplot(1, 2, 1)
                ax_3d = fig_mpc.add_subplot(1, 2, 2, projection='3d')

                # ── XY panel ─────────────────────────────────────────────────
                for step_i, (cands, _sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                    if step_i % _STRIDE != 0:
                        continue
                    env_step = min(step_i * spr, n_steps - 1)
                    anchor   = c_arr[env_step] if c_arr is not None else real_pos[env_step]
                    for b in range(cands.shape[0]):
                        ax_xy.plot(cands[b, :, 0], cands[b, :, 1],
                                   color='green', linewidth=0.6, alpha=0.7, zorder=4)
                    ax_xy.scatter([anchor[0]], [anchor[1]],
                                  color='black', s=30, zorder=8, linewidths=0)

                if c_arr is not None:
                    ax_xy.plot(c_arr[:, 0], c_arr[:, 1],
                               color='red', linewidth=1.2, zorder=9)
                ax_xy.plot(real_pos[:, 0], real_pos[:, 1],
                           color='black', linewidth=1.2, zorder=7)
                # U11.2: start / end markers
                _ref = c_arr if c_arr is not None else real_pos
                ax_xy.scatter([_ref[0, 0]],  [_ref[0, 1]],
                              color='lime', marker='*', s=180, zorder=12, linewidths=0)
                ax_xy.scatter([_ref[-1, 0]], [_ref[-1, 1]],
                              color='red',  marker='s', s=80,  zorder=12, linewidths=0)
                _lgd = [
                    _Line2D([0],[0], color='green', lw=0.8,
                            label=f'MPC candidates ({n_cands}/step)'),
                    _Line2D([0],[0], color='black', lw=1.2, label='des (commanded)'),
                    _Line2D([0],[0], color='red',   lw=1.2, label='actual (c_pos)'),
                    _Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                            markersize=7, label='replan decision point'),
                    _Line2D([0],[0], marker='*', color='w', markerfacecolor='lime',
                            markersize=10, label='start'),
                    _Line2D([0],[0], marker='s', color='w', markerfacecolor='red',
                            markersize=7,  label='end'),
                ]
                ax_xy.legend(handles=_lgd, fontsize=9)
                ax_xy.set_title(f'XY — MPC Decision Points  (every {_STRIDE} replans)',
                                fontsize=12)
                ax_xy.set_xlabel('X (m)', fontsize=11)
                ax_xy.set_ylabel('Y (m)', fontsize=11)
                ax_xy.set_aspect('equal', adjustable='datalim')
                ax_xy.grid(True, alpha=0.3)

                # ── 3D XYZ panel ──────────────────────────────────────────────
                for step_i, (cands, _sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                    if step_i % _STRIDE != 0:
                        continue
                    env_step = min(step_i * spr, n_steps - 1)
                    anchor   = c_arr[env_step] if c_arr is not None else real_pos[env_step]
                    for b in range(cands.shape[0]):
                        ax_3d.plot(cands[b, :, 0], cands[b, :, 1], cands[b, :, 2],
                                   color='green', linewidth=0.6, alpha=0.7)
                    ax_3d.scatter([anchor[0]], [anchor[1]], [anchor[2]],
                                  color='black', s=30)

                if c_arr is not None:
                    ax_3d.plot(c_arr[:, 0], c_arr[:, 1], c_arr[:, 2],
                               color='red', linewidth=1.2, label='actual (c_pos)')
                ax_3d.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
                           color='black', linewidth=1.2, label='des (commanded)')
                # U11.2: start / end markers
                ax_3d.scatter([_ref[0, 0]],  [_ref[0, 1]],  [_ref[0, 2]],
                              color='lime', marker='*', s=180, zorder=12)
                ax_3d.scatter([_ref[-1, 0]], [_ref[-1, 1]], [_ref[-1, 2]],
                              color='red',  marker='s', s=80,  zorder=12)
                ax_3d.set_title('XYZ — MPC Decision Points (3D)', fontsize=12)
                ax_3d.set_xlabel('X (m)', fontsize=9)
                ax_3d.set_ylabel('Y (m)', fontsize=9)
                ax_3d.set_zlabel('Z (m)', fontsize=9)
                ax_3d.legend(fontsize=9)

                fig_mpc.tight_layout()
                _mpc_base = os.path.join(diag_path, f'rollout_{rollout_idx}_mpc_foresight')
                # fig_mpc.savefig(f'{_mpc_base}.png', dpi=200, bbox_inches='tight')
                fig_mpc.savefig(f'{_mpc_base}.svg', bbox_inches='tight')
                plt.close(fig_mpc)

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
            # bp_np is (C,H,W) float32 in BGR order (fix11: no [::-1] flip, env returns BGR).
            # imageio.mimsave() expects RGB, so convert at capture time — same as expert GIF.
            if self.record_mode != 'none':
                try:
                    bp_vis     = cv2.cvtColor((bp_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    inhand_vis = cv2.cvtColor((inhand_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    frame = np.concatenate([bp_vis, inhand_vis], axis=1)
                    cv2.putText(frame, f's{self.step_counter}', (5, 18),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
                    self.video_frames.append(frame)
                except Exception:
                    pass

            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()

            self.history_real_pos.append(des_robot_pos_np.copy())
            self.curr_rollout_c_pos.append(robot_pos_np.copy())              # Fix 9
            # Fix 9: physical tracking error = |actual - commanded| (PD controller lag)
            phys_err = float(np.linalg.norm(robot_pos_np[:2] - des_robot_pos_np[:2]))
            self.curr_rollout_tracking_errors.append(phys_err)

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
            self.curr_rollout_c_pos.append(des_robot_pos_np.copy())          # Fix 9: no separate c_pos
            self.curr_rollout_tracking_errors.append(0.0)                    # Fix 9: no separate c_pos

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
        if self.action_counter == self.action_seq_size:
            t_replan = time.time()   # Fix 12: time only the replan call, not cached fetches
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
                else:
                    which = 0   # 'random' (DPCC default) = always index 0, deterministic
                    selection_method = 'random (index 0, DPCC semantics)'

            # Fix 9: store c_pos dims (6:9) = predicted actual positions, unnormalized
            cpos_norm = traj_np[:, :, 6:9]   # (B, H, 3) normalized predicted actual positions
            if self.obs_normalizer is not None:
                B_f, H_f = cpos_norm.shape[:2]
                dummy = np.zeros((B_f * H_f, 3), dtype=np.float32)
                obs6d = np.concatenate([dummy, cpos_norm.reshape(-1, 3).astype(np.float32)], axis=1)
                obs6d_un = self.obs_normalizer.unnormalize(obs6d)
                self.curr_rollout_all_candidates.append(obs6d_un[:, 3:].reshape(B_f, H_f, 3).copy())
            else:
                self.curr_rollout_all_candidates.append(cpos_norm.copy())
            self.curr_rollout_selected_idx.append(int(which))

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
                # obs_6d health (Issue 4)
                diag_lines += [
                    f'[ DIAG obs ] des_c_pos={np.round(obs_6d_np[:3], 4)}  '
                    f'c_pos={np.round(obs_6d_np[3:], 4)}',
                    f'[ DIAG obs ] obs_6d_norm={np.round(obs_6d_norm, 4)}',
                ]
                # image health (Issue 5) — visual only
                if if_vision:
                    bp_std = float(np.std(bp_np))
                    ih_std = float(np.std(inhand_np))
                    diag_lines += [
                        f'[ DIAG img ] bp_image   std={bp_std:.4f}  shape={bp_np.shape}',
                        f'[ DIAG img ] inhand_img std={ih_std:.4f}  shape={inhand_np.shape}',
                    ]
                    if bp_std < 0.01:
                        diag_lines.append('[ DIAG img ] WARNING: bp_image near-black — camera may not be rendering')
                    if ih_std < 0.01:
                        diag_lines.append('[ DIAG img ] WARNING: inhand_image near-black — camera may not be rendering')
                for line in diag_lines:
                    print(line)
                if self.save_path is not None:
                    diag_file = os.path.join(self.save_path, 'diag_first_replan.txt')
                    with open(diag_file, 'w') as _df:
                        _df.write('\n'.join(diag_lines) + '\n')
                    print(f'[ DIAG ] saved → {diag_file}')

            # Periodic DIAG every 50 replans (Issue 2)
            self._replan_count += 1
            if self._replan_count % 50 == 0:
                _pa0 = trajectory[[which], 0, :3].detach().cpu().numpy().squeeze()
                _da0 = action_traj[0, 0].detach().cpu().numpy()
                _dir = _pa0 / (np.linalg.norm(_pa0) + 1e-9)
                print(f'[ DIAG replan={self._replan_count} step={self.step_counter} ] '
                      f'norm|a0|={np.linalg.norm(_pa0):.3f}  '
                      f'denorm|a0|={np.linalg.norm(_da0):.2e} m  '
                      f'dir={np.round(_dir, 3)}')

            self.curr_action_seq = action_traj[:, :self.action_seq_size, :]
            self.history_full_plans.append(action_traj[0].detach().cpu().numpy())
            self.curr_rollout_time += time.time() - t_replan   # Fix 12: accumulate per-replan time

        next_action    = self.curr_action_seq[:, self.action_counter, :]
        next_action_np = next_action.detach().cpu().numpy().squeeze(0)   # (3,)
        self.curr_rollout_act_magnitudes.append(float(np.linalg.norm(next_action_np)))

        if self.max_action_delta is not None:
            raw_mag = np.linalg.norm(next_action_np)
            if raw_mag > self.max_action_delta:
                next_action_np = next_action_np * (self.max_action_delta / raw_mag)
                self.curr_rollout_clamp_events.append((self.step_counter, float(raw_mag)))
                if len(self.curr_rollout_clamp_events) <= 5 or self.step_counter % 50 == 0:
                    print(f'[ CLAMP step={self.step_counter} ] '
                          f'raw|a|={raw_mag:.4f} m → clamped to {self.max_action_delta} m')

        self.mental_robot_pos += next_action_np

        self.history_desired_actions.append(next_action_np.copy())
        self.last_predicted_pos = self.mental_robot_pos.copy()

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
            # Original DPCC always trains/evaluates with clip_denoised=False — the cosine schedule
            # amplifies x_0 prediction by ~9.4× at early timesteps, so clipping to ±1 corrupts the
            # denoising chain. Older checkpoints may have been saved with True; this override corrects them.
            # D1: config-driven (was hardcoded False). Default=False matches reference DPCC.
            # Set clip_denoised=True in plan_visual_aligning_dpcc config only to ablate.
            _clip_denoised = getattr(args, 'clip_denoised', False)
            diffusion_model.clip_denoised = _clip_denoised
            print(f'[ eval ] clip_denoised set → {_clip_denoised} (config-driven; original DPCC default: False)')
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

        # Expert reference videos generated ONCE before the variant loop.
        # Running env.start()/env.close() inside the loop leaves residual MuJoCo
        # global state (body counter, OpenGL context) that corrupts the first
        # variant's scene init and changes bp_image pixel statistics.
        # gc.collect + cuda.empty_cache after close forces cleanup before any
        # variant env is created.  (AUDIT-FIX-1 — KEY_fix_6/BUG_REPORT.md)
        _base_results = (f'{args.savepath}/results_train_set'
                         if args_cli.eval_on_train else f'{args.savepath}/results')
        os.makedirs(_base_results, exist_ok=True)
        generate_expert_reference(_base_results, n_rollouts=3)
        gc.collect()
        torch.cuda.empty_cache()

        # FIX-7: Reset MuJoCo global robot body counter after expert gen.
        # Robot_Push_Env.__init__() creates an MjRobot which increments
        # MjRobot.GLOBAL_MJ_ROBOT_COUNTER (0→1). env.close() does NOT decrement it.
        # Without this reset, the first variant's Robot_Push_Env gets robot_id=1
        # (body prefix "rb1") instead of robot_id=0 ("rb0"), compiling a different
        # MuJoCo XML where camera bodies are named "rb1_*" instead of "rb0_*".
        # The mismatch changes what the camera renders → bp_image std 0.1978 instead
        # of clean 0.2093 → model receives wrong visual input → wrong trajectory.
        try:
            from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot as _MjRobot
            _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
            print('[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)')
        except Exception as _e:
            print(f'[ expert ] WARNING: MjRobot counter reset failed: {_e}')
        # FIX-7.2: Clear the process-global render context cache so the variant's
        # cameras create fresh RenderContextOffscreen objects bound to variant_model
        # and variant_data. Without this, __RENDER_CTX_MAP in mj_render_singleton.py
        # holds stale contexts from expert gen (bound to expert_model/expert_data),
        # causing all variant renders to show the expert gen's robot pose instead of
        # the variant's → bp_image std 0.1978 (wrong) instead of 0.2093 (correct).
        try:
            from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
                reset_singleton as _reset_render_singleton,
            )
            _reset_render_singleton()
            print('[ expert ] Render singleton cache cleared (FIX-7.2)')
        except Exception as _e:
            print(f'[ expert ] WARNING: Render singleton reset failed: {_e}')
        # Also delete stale panda_tmp_rb*.xml left by expert gen to suppress noisy
        # mju_openResource warnings on subsequent env inits.
        import glob as _glob
        _mj_dir = os.path.join(
            os.environ.get('D3IL_DIR', 'd3il/environments/d3il'), 'models/mj/robot')
        for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
            try:
                os.remove(_stale)
            except OSError:
                pass

        # ── Geo constraint outer loop (UF-14) ────────────────────────────────
        # Build flat (geo_name, geo_config, base_variant) product so the inner
        # loop body needs no indentation change.
        _geo_specs = config.get('geo_constraint_variants', [
            {'name': 'combined_2',
             'constraint_types': config.get('constraint_types', ['bounds', 'dynamics']),
             'workspace_bounds': {'lb': [0.30, -0.35, 0.05], 'ub': [0.70, 0.35, 0.40]}}
        ])
        # enlarge_constraints: None when yaml sets null → no tightened twin generated
        _enlarge = config.get('enlarge_constraints')
        _run_items = []
        for _gs in _geo_specs:
            _gc = dict(config)
            _gc['constraint_types'] = _gs['constraint_types']
            if 'workspace_bounds'      in _gs: _gc['workspace_bounds']      = _gs['workspace_bounds']
            if 'obstacle_constraints'  in _gs: _gc['obstacle_constraints']  = _gs['obstacle_constraints']
            if 'halfspace_constraints' in _gs: _gc['halfspace_constraints'] = _gs['halfspace_constraints']
            _has_geo = any(t in _gs['constraint_types'] for t in ('bounds', 'halfspace', 'obstacles'))
            for _v in projection_variants:
                _run_items.append((_gs['name'], _gc, _v, False))
            # auto-generate tightened twin for entries with bounds/obstacles
            if _enlarge is not None and _has_geo:
                for _v in projection_variants:
                    _run_items.append((_gs['name'] + '-tightened', _gc, _v, True))

        for geo_name, geo_config, geo_variant, is_tightened in _run_items:
            if geo_variant == projection_variants[0]:
                print(f'\n[ geo ] ── Constraint variant: {geo_name}  '
                      f'types={geo_config["constraint_types"]} ──')
            variant = geo_variant
            if args_cli.eval_on_train:
                variant   = f'{variant}_train_set'
                save_path = f'{args.savepath}/results_train_set/{geo_name}/{variant}'
            else:
                save_path = f'{args.savepath}/results/{geo_name}/{variant}'
            os.makedirs(save_path, exist_ok=True)

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
                        args, geo_config, obs_normalizer, act_normalizer, variant, is_tightened)
                    print(f'[ eval ] DPCC projector active for variant {variant!r}')

                # Trajectory selection — exact DPCC eval.py logic (projection_eval.yaml dpcc-r/c/t).
                # 'random' = always index 0 (deterministic); same as DPCC Policy.__call__ semantics.
                trajectory_selection = 'random'
                if 'dpcc-t' in variant:
                    trajectory_selection = 'temporal_consistency'
                if 'dpcc-c' in variant:
                    trajectory_selection = 'minimum_projection_cost'

                # Fix 8: diffuser runs single sample (no projection, no candidate diversity).
                # All projected variants use args.mpc_batch_size from plan config (MPC candidate pool).
                if 'diffuser' in variant:
                    batch_size = 1
                else:
                    batch_size = getattr(args, 'mpc_batch_size', 4)

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
                    max_action_delta=geo_config.get('max_action_delta', None),
                    mpc_foresight_stride=geo_config.get('mpc_foresight_stride', 6),
                )

                _if_vision_config = getattr(args, 'if_vision', True)
                if_vision = _if_vision_config
                if not if_vision and args_cli.record != 'none':
                    if_vision = True
                    print('[ eval ] WARNING: config if_vision=False but record_mode is active → '
                          'auto-enabling visual mode so GIFs/videos are captured (UF-13).')

                sim = Aligning_Sim(
                    seed=seed, device=args.device,
                    render=False, n_cores=1,
                    n_contexts=n_contexts,
                    n_trajectories_per_context=n_trajectories,
                    if_vision=if_vision,
                    eval_on_train=args_cli.eval_on_train,
                    max_episode_length=getattr(args, 'max_episode_length', 400),
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
                if geo_config.get('write_to_file', True):
                    # U10.2: flatten context_info + clean tracking error into NPZ
                    # so DA code can load per-rollout arrays directly (same pattern as avoiding).
                    _ci = agent.history_context_info   # list of dicts, one per rollout
                    _max_phys = np.array([
                        float(np.max(e)) if len(e) else 0.0
                        for e in agent.history_pos_tracking_errors
                    ], dtype=np.float32)
                    _ctx_box_xy   = np.array([[c.get('box_init_xy',  [0.0, 0.0])[0],
                                               c.get('box_init_xy',  [0.0, 0.0])[1]]
                                              for c in _ci], dtype=np.float32)
                    _ctx_tgt_xy   = np.array([[c.get('target_xy',    [0.0, 0.0])[0],
                                               c.get('target_xy',    [0.0, 0.0])[1]]
                                              for c in _ci], dtype=np.float32)
                    _ctx_box_ang  = np.array([c.get('box_init_angle_deg', 0.0) for c in _ci], dtype=np.float32)
                    _ctx_tgt_ang  = np.array([c.get('target_angle_deg',   0.0) for c in _ci], dtype=np.float32)
                    _ctx_xy_dist  = np.array([c.get('init_xy_dist',       0.0) for c in _ci], dtype=np.float32)
                    np.savez(f'{save_path}/{variant}.npz',
                             success_rate=success_rate, entropy=entropy,
                             mode_encoding=mode_encoding.numpy(),
                             elapsed_seconds=elapsed, seed=seed,
                             n_success=successes.flatten().numpy(),
                             n_steps=np.array(agent.history_n_steps),
                             avg_time=np.array(agent.history_avg_time),
                             mean_distance=mean_dist.flatten().numpy(),
                             mean_dist_per_rollout=np.array(agent.history_rollout_mean_dist),
                             physical_tracking_errors=np.array(
                                 agent.history_pos_tracking_errors, dtype=object),
                             max_phys_error_per_rollout=_max_phys,
                             context_box_init_xy=_ctx_box_xy,
                             context_target_xy=_ctx_tgt_xy,
                             context_box_angle_deg=_ctx_box_ang,
                             context_target_angle_deg=_ctx_tgt_ang,
                             context_init_xy_dist=_ctx_xy_dist,
                             obs_all=np.array(obs_all, dtype=object),
                             act_all=np.array(act_all, dtype=object),
                             sampled_trajectories_all=np.array(plans_all, dtype=object),
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
                        c_pos_hist   = rollout_data.get('c_pos_history', None)  # Fix 9: actual positions

                        axes[i, 0].plot(obs_traj[:, 0], 'k-', label='des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 0].plot(c_pos_hist[:, 0], 'r--', label='actual')
                        axes[i, 0].set_title('X — des (black) vs actual (red)')

                        axes[i, 1].plot(obs_traj[:, 1], 'k-', label='des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 1].plot(c_pos_hist[:, 1], 'r--', label='actual')
                        axes[i, 1].set_title('Y — des (black) vs actual (red)')

                        axes[i, 2].plot(obs_traj[:, 2], 'k-', label='Z des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 2].plot(np.array(c_pos_hist)[:, 2], 'r--',
                                            alpha=0.7, label='Z actual')
                        axes[i, 2].set_title('Z — des (black) vs actual (red)')

                        vels = np.linalg.norm(obs_traj[1:] - obs_traj[:-1], axis=1)
                        axes[i, 3].plot(vels, color='gray', alpha=0.5)
                        axes[i, 3].set_title('Step Magnitude')

                        axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', linewidth=2)
                        axes[i, 4].plot(obs_traj[0, 0],  obs_traj[0, 1],  'go', markersize=10)
                        axes[i, 4].plot(obs_traj[-1, 0], obs_traj[-1, 1], 'ro', markersize=10)
                        axes[i, 4].set_title('XY Trajectory')

                        # Fix 9 I2: cands are unnormalized c_pos XY, no cumsum; green=selected
                        all_cands_list = rollout_data.get('all_candidates', [])
                        sel_idx_list   = rollout_data.get('selected_idx',   [])
                        axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.4)
                        for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                            if step_i % 4 != 0:
                                continue
                            for b in range(cands.shape[0]):
                                if b == sel:
                                    axes[i, 5].plot(cands[b, :, 0], cands[b, :, 1],
                                                    color='green', linewidth=0.8, alpha=0.9)
                                else:
                                    axes[i, 5].plot(cands[b, :, 0], cands[b, :, 1],
                                                    color='gray', linewidth=0.2, alpha=0.2)
                        n_c = all_cands_list[0].shape[0] if all_cands_list else 1
                        axes[i, 5].set_title(f'MPC Foresight ({n_c} candidates/step)')

                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.savefig(f'{save_path}/{variant}.png')
                    plt.close(fig)

                # ── Aligning eval summary ────────────────────────────────────
                n_success   = np.array(successes.flatten())
                n_steps     = np.array(agent.history_n_steps)
                dists       = np.array(agent.history_rollout_mean_dist)
                all_errs    = np.concatenate([e for e in agent.history_pos_tracking_errors
                                              if len(e)]) if agent.history_pos_tracking_errors else np.array([0.0])
                max_phys    = float(np.max(all_errs))
                mean_phys   = float(np.mean(all_errs))

                run_mode = 'seen training set' if args_cli.eval_on_train else 'default'
                print(f'--- aligning-d3il-visual [{run_mode}] {variant} seed={seed} ---')
                print(f'Success rate:              {np.mean(n_success):.4f}')
                print(f'Avg final mean distance:   {np.mean(dists):.4f} m  '
                      f'+- {np.std(dists):.4f} m')
                print(f'Min final mean distance:   {np.min(dists):.4f} m')
                print(f'Avg steps (successful):    '
                      f'{np.mean(n_steps[n_success > 0]) if n_success.sum() else 0:.2f}'
                      f' +- {np.std(n_steps[n_success > 0]) if n_success.sum() else 0:.2f}')
                print(f'Avg steps (all trials):    {np.mean(n_steps):.2f} +- {np.std(n_steps):.2f}')
                print(f'Physical tracking error:   mean={mean_phys:.4f} m  max={max_phys:.4f} m')
                print(f'Avg inference time/replan: {np.mean(agent.history_avg_time):.3f} s')
                print('-' * 80 + '\n')

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_f.close()
                # FIX-7 (per-variant): Reset MuJoCo robot body counter so next
                # variant's Robot_Push_Env gets robot_id=0 (rb0 body prefix),
                # matching the clean-process scene geometry.
                try:
                    _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
                except NameError:
                    pass
                # FIX-7.2 (per-variant): Clear render context cache so next variant
                # creates fresh RenderContextOffscreen objects.
                try:
                    _reset_render_singleton()
                except NameError:
                    pass
                for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
                    try:
                        os.remove(_stale)
                    except OSError:
                        pass

    print('Visual-DPCC evaluation completed.')
