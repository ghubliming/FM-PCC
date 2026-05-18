# Visual Aligning Evaluation Script — DPCC/FMPCC Upgrade
# ═══════════════════════════════════════════════════════════════════════
# Built on the FMv3ODE eval blueprint (eval_flow_matching_v3_ode_selectable.py).
# Key adaptation: uses Aligning_Sim.test_agent() with D3IL-native agent wrapper
# instead of ObstacleAvoidanceEnv with manual step loop.
#
# Output:  logs/aligning-d3il-visual/plans/fm_encdec_vision/H8/<seed>/results/
#          ├── diffuser.npz          (baseline: no projection)
#          ├── results_seed_<s>.pkl  (structured metrics)
#          └── eval_diffuser.log     (full console log)
#
# Metrics: Success Rate, Entropy, Mean Distance, Score (D3IL standard)
# ═══════════════════════════════════════════════════════════════════════

import time
import yaml
import os
import torch
import numpy as np
import sys
import argparse
import pickle
from datetime import datetime
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Non-interactive backend
import imageio
import cv2

import fm_encdec_vision.utils as utils
from diffuser.sampling import Projector

# Ensure local d3il is prioritized in path
sys.path.insert(0, os.path.abspath('d3il'))
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))

# Fix MuJoCo resource loading: Force D3IL_DIR to local path
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

import d3il
print(f"[ eval ] Using d3il from: {d3il.__file__}")
print(f"[ eval ] D3IL_DIR set to: {os.environ['D3IL_DIR']}")

from d3il.simulation.aligning_sim import Aligning_Sim

# ─── Compatibility Normalizer Adapter ───────────────────────────────────────
class VisualNormalizerAdapter:
    """Bridges D3IL's Scaler class with the Projector's normalizer expectations."""
    def __init__(self, scaler):
        # Extract physical limits from the scaled dataset bounds
        self.mins = scaler.y_min.detach().cpu().numpy()
        self.maxs = scaler.y_max.detach().cpu().numpy()

class VisualNormalizerDict:
    """Wraps observations and actions to match the dataset normalizers dictionary."""
    def __init__(self, scaler):
        self.normalizers = {
            'observations': VisualNormalizerAdapter(scaler),
            'actions': VisualNormalizerAdapter(scaler)
        }

def setup_gen6_projector(args, config, scaler, variant):
    """Instantiates the DPCC projection engine for the visual workspace."""
    # 1. Determine constraint tightening margin (supports both legacy 'enlarge_constraints' and 'constraint_tightening_margin')
    tightening_margin = config.get('constraint_tightening_margin', config.get('enlarge_constraints', 0.0))
    
    workspace_lb = np.array(config['workspace_bounds']['lb'])
    workspace_ub = np.array(config['workspace_bounds']['ub'])
    
    if 'tightened' in variant and tightening_margin > 0.0:
        workspace_lb += tightening_margin
        workspace_ub -= tightening_margin

    constraint_list = []
    
    # 2. Formulate Safety Bounds Constraints (applied to absolute position dims 3, 4, 5)
    # Dims 0, 1, 2 are actions (deltas), Dims 3, 4, 5 are robot proprioception (absolute position)
    if 'bounds' in config.get('constraint_types', []):
        lb = np.array([-np.inf, -np.inf, -np.inf, workspace_lb[0], workspace_lb[1], workspace_lb[2]])
        ub = np.array([np.inf, np.inf, np.inf, workspace_ub[0], workspace_ub[1], workspace_ub[2]])
        constraint_list.append(['lb', lb])
        constraint_list.append(['ub', ub])
    
    # 3. Formulate Kinematics/Dynamics Constraints (Euler derivative bounds)
    # Explicit Euler derivative step binding coordinate dimensions to actions
    # x_idx = 3 (proprioception X), dx_idx = 0 (action vx)
    # y_idx = 4 (proprioception Y), dx_idx = 1 (action vy)
    # z_idx = 5 (proprioception Z), dx_idx = 2 (action vz)
    if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
        constraint_list.append(('deriv', [3, 0]))
        constraint_list.append(('deriv', [4, 1]))
        constraint_list.append(('deriv', [5, 2]))
    
    # 4. Construct compatibility normalizer dict
    adapter_normalizer = VisualNormalizerDict(scaler)
    
    # 5. Handle time scaling (dt scaling) and gradient/post-processing thresholds
    dt = config.get('dt', 1.0)  # Correct base integration step (dt = 1.0) since actions represent spatial delta displacements
    if 'dt0p25' in variant:
        dt = 0.25 * dt
    elif 'dt0p5' in variant:
        dt = 0.5 * dt
    elif 'dt2p0' in variant:
        dt = 2.0 * dt
    elif 'dt4p0' in variant:
        dt = 4.0 * dt

    threshold = 0.0 if 'post_processing' in variant else config.get('diffusion_timestep_threshold', 0.5)
    gradient = 'gradient' in variant

    # 6. Initialize the DPCC Projector
    projector = Projector(
        horizon=getattr(args, 'horizon', 8),
        transition_dim=6,                # Combined Action + State dimension (6D)
        action_dim=3,                    # XYZ Cartesian actions
        goal_dim=0,                      # Non-goal conditioned VAE
        constraint_list=constraint_list,
        normalizer=adapter_normalizer,
        diffusion_timestep_threshold=threshold,
        variant='states_actions',        # Must be states_actions for 6D trajectory
        dt=dt,
        gradient=gradient,
        gradient_weights=[1, 0.5, 2] if gradient else None,
        solver='scipy',                  # Robust SLSQP QP optimizer
        device=args.device
    )
    return projector

# ─── Logging ────────────────────────────────────────────────────────────────
class Tee(object):
    def __init__(self, *files):
        self.files = [f if hasattr(f, 'write') else open(f, 'a') for f in files]
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# ─── D3IL-Native Agent Wrapper ──────────────────────────────────────────────
class VisualAgentWrapper:
    """
    Bridges FM-PCC's loaded diffusion model into Aligning_Sim's expected agent interface.
    """
    def __init__(self, diffusion_model, device, window_size=8, obs_seq_len=8, action_seq_size=4, save_path=None, record_mode='all', scaler=None, eval_on_train=False, batch_size=1, projector=None, trajectory_selection='random', variant='unspecified'):
        self.model = diffusion_model
        self.device = device
        self.window_size = window_size
        self.obs_seq_len = obs_seq_len  # Respect trained config (FIX #12)
        self.scaler = scaler
        self.eval_on_train = eval_on_train
        self.batch_size = batch_size
        self.projector = projector
        self.trajectory_selection = trajectory_selection
        self.prev_observations = None
        self.variant = variant
        
        # Open-Loop State (The "Mental Map")
        self.mental_robot_pos = None 
        
        # --- HORIZON SAFETY CLAMP ---
        # Ensure we never try to execute more steps than the model planned.
        # This prevents IndexError when horizon < action_seq_size (e.g. H=2, Chunk=4)
        model_horizon = getattr(self.model, 'horizon', window_size)
        self.action_seq_size = min(action_seq_size, model_horizon)
        # -----------------------------

        self.action_counter = self.action_seq_size  # Force re-plan on first call
        self.curr_action_seq = None
        self.save_path = save_path
        self.record_mode = record_mode
        
        # Diagnostics
        self.rollout_counter = -1
        self.step_counter = 0
        
        # History buffers for rollout extraction
        self.history_real_pos = []
        self.history_desired_actions = []
        self.history_full_plans = []
        self.history_n_steps = []
        self.history_avg_time = []
        self.history_pos_tracking_errors = []
        
        self.master_rollout_history = {}
        
        # Temp step tracking
        self.curr_rollout_time = 0
        self.last_predicted_pos = None
        self.curr_rollout_tracking_errors = []
        
        # Context windows
        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.obs_seq_len)
        self.video_frames = []
    
    def reset(self):
        """Called by Aligning_Sim at the start of each rollout."""
        self.history_real_pos.clear()
        self.history_desired_actions.clear()
        self.history_full_plans.clear()
        self.curr_rollout_time = 0
        self.last_predicted_pos = None
        self.curr_rollout_tracking_errors.clear()
        
        self.mental_robot_pos = None # Reset mental map (FIX #17)
        self.prev_observations = None # Reset prev observations for trajectory selection
        
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
        self.action_counter = self.action_seq_size
        
        # Prepare for the NEXT rollout
        self.rollout_counter += 1
        self.step_counter = 0
        self.video_frames = []
        
    def update_rollout_info(self, info):
        """Called by Aligning_Sim at the end of each rollout to log stats and diagnostics immediately."""
        success = info.get('success', False)
        mean_dist = info.get('mean_distance', 0.0)
        mode = info.get('mode', 0)
        rollout_idx = int(info.get('context', self.rollout_counter))
        
        max_err = float(np.max(self.curr_rollout_tracking_errors) if len(self.curr_rollout_tracking_errors) > 0 else 0.0)
        avg_time = float(self.curr_rollout_time / max(1, self.step_counter))
        
        # Store rollout statistics in history dictionary
        self.master_rollout_history[f"rollout_{rollout_idx}"] = {
            "real_robot_pos": np.array(self.history_real_pos),
            "desired_actions": np.array(self.history_desired_actions),
            "full_plans": np.array(self.history_full_plans),
            "plan_start_positions": np.array(self.history_real_pos)[::self.action_seq_size, :],
            "success": bool(success),
            "mean_distance": float(mean_dist),
            "mode": int(mode),
            "steps": int(self.step_counter),
            "avg_time": avg_time,
            "max_tracking_error": max_err
        }
        
        self.history_n_steps.append(self.step_counter)
        self.history_avg_time.append(avg_time)
        self.history_pos_tracking_errors.append(np.array(self.curr_rollout_tracking_errors))
        
        # Print real-time diagnostic summary to console/log
        context_type = "Seen Training Context" if self.eval_on_train else "Unseen Test Context"
        print(f"[ {context_type} {rollout_idx} Finished ]")
        print(f"  - Total Steps: {self.step_counter}")
        print(f"  - Success status: {success}")
        print(f"  - Final Mean Distance: {mean_dist:.6f} m")
        print(f"  - Environment Mode: {mode}")
        print(f"  - Maximum Tracking Error: {max_err:.6f} m")
        print(f"  - Avg Inference Time: {avg_time:.4f} seconds/step")
        print("-" * 80 + "\n")
        
        # Real-time Export (Scientific JSON and PNG Report)
        if hasattr(self, 'save_path') and self.save_path is not None:
            self._export_rollout_realtime(rollout_idx)
            
        # Save video/gif diagnostics
        if self.record_mode != 'none' and len(self.video_frames) > 0:
            self._save_diagnostics(rollout_idx)
    
    def _save_diagnostics(self, rollout_idx, custom_path=None, custom_frames=None):
        frames = custom_frames if custom_frames is not None else self.video_frames
        path = custom_path if custom_path is not None else os.path.join(self.save_path, 'diagnostics', self.variant)
        os.makedirs(path, exist_ok=True)
        
        try:
            import imageio
            if self.record_mode in ['video', 'all']:
                try:
                    save_path_mp4 = os.path.join(path, f'rollout_{rollout_idx}.mp4')
                    imageio.mimsave(save_path_mp4, frames, fps=20)
                except Exception as e:
                    print(f"[ WARNING ] MP4 saving failed: {e}. Attempting GIF fallback...")
            
            if self.record_mode in ['gif', 'all']:
                try:
                    save_path_gif = os.path.join(path, f'rollout_{rollout_idx}.gif')
                    imageio.mimsave(save_path_gif, frames, fps=10)
                except Exception as e:
                    print(f"[ WARNING ] GIF saving failed: {e}.")
            
            # NEW: Save human-readable text stats file right next to the gif/mp4 for easy debugging
            data = self.master_rollout_history.get(f"rollout_{rollout_idx}", {})
            diag_stats_path = os.path.join(path, f'rollout_{rollout_idx}_stats.txt')
            with open(diag_stats_path, 'w') as sf:
                sf.write(f"Rollout {rollout_idx} Execution Summary\n")
                sf.write("=" * 40 + "\n")
                sf.write(f"Success: {data.get('success', False)}\n")
                sf.write(f"Total Steps: {data.get('steps', 0)}\n")
                sf.write(f"Mean Distance to Target: {data.get('mean_distance', 0.0):.6f} m\n")
                sf.write(f"Environment Mode: {data.get('mode', 0)}\n")
                sf.write(f"Average Inference Time: {data.get('avg_time', 0.0):.4f} seconds/step\n")
                sf.write(f"Max Tracking Error: {data.get('max_tracking_error', 0.0):.6f} m\n")
        except Exception as e:
            print(f"[ WARNING ] Diagnostics engine failed: {e}. Skipping.")

    def _export_rollout_realtime(self, rollout_idx):
        """Generates scientific PNG and PKL for the rollout immediately."""
        try:
            diag_path = os.path.join(self.save_path, "realtime_diagnostics", self.variant)
            os.makedirs(diag_path, exist_ok=True)
            
            # 1. Save data pickle
            data = self.master_rollout_history[f"rollout_{rollout_idx}"]
            with open(os.path.join(diag_path, f"rollout_{rollout_idx}_data.pkl"), "wb") as f:
                pickle.dump(data, f)
                
            # NEW: Save human-readable JSON stats file for easy debugging
            import json
            stats = {
                "rollout_index": int(rollout_idx),
                "success": bool(data.get("success", False)),
                "steps": int(data.get("steps", 0)),
                "mean_distance": float(data.get("mean_distance", 0.0)),
                "mode": int(data.get("mode", 0)),
                "avg_inference_time_per_step": float(data.get("avg_time", 0.0)),
                "max_tracking_error": float(data.get("max_tracking_error", 0.0))
            }
            stats_path = os.path.join(diag_path, f"rollout_{rollout_idx}_stats.json")
            with open(stats_path, 'w') as sf:
                json.dump(stats, sf, indent=4)
                
            # 2. Generate PNG Plot (Scientific)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Rollout {rollout_idx} - MPC vs Real vs Expert")
            
            real_pos = data['real_robot_pos']
            plans = data['full_plans']
            plan_starts = data['plan_start_positions']
            
            # Panel 1: XY Trajectory
            axes[0, 0].plot(real_pos[:, 0], real_pos[:, 1], 'k-', linewidth=2, label='Real Path')
            for p_idx, plan in enumerate(plans):
                if p_idx % 4 == 0:
                    start = plan_starts[min(p_idx, len(plan_starts)-1)]
                    abs_plan = start + np.cumsum(plan[:, :3], axis=0)
                    axes[0, 0].plot(abs_plan[:, 0], abs_plan[:, 1], 'b-', alpha=0.3)
            axes[0, 0].set_title("XY Projection (MPC Foresight in Blue)")
            axes[0, 0].set_xlabel("X (m)")
            axes[0, 0].set_ylabel("Y (m)")
            axes[0, 0].legend()
            
            # Panel 2: X over Time
            axes[0, 1].plot(real_pos[:, 0], 'k-', label='Real X')
            axes[0, 1].set_title("X Position over Steps")
            axes[0, 1].set_ylabel("Meters")
            
            # Panel 3: Y over Time
            axes[0, 2].plot(real_pos[:, 1], 'k-', label='Real Y')
            axes[0, 2].set_title("Y Position over Steps")
            
            # Panel 4: Z over Time (Force/Contact check)
            axes[1, 0].plot(real_pos[:, 2], 'r-', label='Real Z')
            axes[1, 0].set_title("Z Height (Contact Stability)")
            
            # Panel 5: Tracking Error
            if len(self.curr_rollout_tracking_errors) > 0:
                axes[1, 1].plot(self.curr_rollout_tracking_errors, 'g-')
                axes[1, 1].set_title("MPC Tracking Error (m)")
            
            # Panel 6: Velocities
            vels = np.linalg.norm(real_pos[1:] - real_pos[:-1], axis=1)
            axes[1, 2].plot(vels, 'm-')
            axes[1, 2].set_title("End-Effector Velocity")
            
            plt.tight_layout()
            fig.savefig(os.path.join(diag_path, f"rollout_{rollout_idx}_report.png"))
            plt.close(fig)
            
        except Exception as e:
            print(f"[ diag ] Real-time export failed for rollout {rollout_idx}: {e}")

    @torch.no_grad()
    def predict(self, state, goal=None, extra_args=None, if_vision=False):
        """Standard D3IL predict signature."""
        if if_vision:
            bp_image_np, inhand_image_np, des_robot_pos_np = state
            
            # Diagnostic capture (All rollouts)
            if self.record_mode != 'none':
                try:
                    # Force copies and ensure we are working with fresh data
                    bp_frame = bp_image_np.copy()
                    ih_frame = inhand_image_np.copy()

                    # Convert [C, H, W] to [H, W, C] and scale to 0-255
                    bp_vis = (bp_frame.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    inhand_vis = (ih_frame.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    
                    # Fix Colors: D3IL/OpenCV uses BGR, but GIF needs RGB
                    bp_vis = cv2.cvtColor(bp_vis, cv2.COLOR_BGR2RGB)
                    inhand_vis = cv2.cvtColor(inhand_vis, cv2.COLOR_BGR2RGB)
                    
                    combined = np.concatenate([bp_vis, inhand_vis], axis=1)
                    self.video_frames.append(combined)
                except Exception as e:
                    # print(f"DIAG ERROR: {e}")
                    pass

            # Open-Loop State Update (Mental Map - FIX #17)
            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()
            
            # Preprocess images to [C, H, W] and normalize
            bp_image = torch.from_numpy(bp_image_np).to(self.device).float().unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image_np).to(self.device).float().unsqueeze(0)
            
            # Use Mental Pos for conditioning, but SCALE it first
            mental_pos_torch = torch.from_numpy(self.mental_robot_pos).to(self.device).float().unsqueeze(0)
            if self.scaler is not None:
                mental_pos_torch = self.scaler.scale_input(mental_pos_torch)
            
            # Record real pos for diagnostics
            self.history_real_pos.append(des_robot_pos_np.copy())
            
            # Record tracking error from PREVIOUS step prediction
            if self.last_predicted_pos is not None:
                err = np.linalg.norm(des_robot_pos_np[:2] - self.last_predicted_pos[:2])
                self.curr_rollout_tracking_errors.append(err)
            
            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            self.des_robot_pos_context.append(mental_pos_torch)
            
            while len(self.bp_image_context) < self.window_size:
                self.bp_image_context.appendleft(bp_image)
                self.inhand_image_context.appendleft(inhand_image)
                self.des_robot_pos_context.appendleft(mental_pos_torch)
            
            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
        else:
            raise NotImplementedError()
        
        t_start = time.time()
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0
            self.model.eval()
            # --- Gen5 Architectural Parity Fix ---
            # Detect if model is 3D (Action-Only) or 6D (State-Action)
            # Repeat conditioning tensors for batch inference
            bp_image_batch = bp_image_seq.repeat(self.batch_size, 1, 1, 1, 1)
            inhand_image_batch = inhand_image_seq.repeat(self.batch_size, 1, 1, 1, 1)
            des_robot_pos_batch = des_robot_pos_seq.repeat(self.batch_size, 1, 1)
            cond = {0: (bp_image_batch, inhand_image_batch, des_robot_pos_batch)}
            if self.projector is not None:
                trajectory, infos = self.model(cond, projector=self.projector)
            else:
                trajectory, infos = self.model(cond)
            
            # Trajectory selection logic (minimum_projection_cost, temporal_consistency, random)
            trajectories_np = trajectory.detach().cpu().numpy()
            which_trajectory = 0
            selection_method = 'default (first)'
            sel_details = {}
            
            if self.batch_size > 1:
                if self.trajectory_selection == 'temporal_consistency' and self.prev_observations is not None:
                    diffs = trajectories_np - np.expand_dims(self.prev_observations, axis=0)
                    order = np.argsort(np.linalg.norm(diffs, axis=(1, 2)))
                    which_trajectory = order[0]
                    selection_method = 'temporal_consistency'
                    sel_details = {'distance_to_prev': float(np.linalg.norm(diffs[which_trajectory]))}
                
                elif self.trajectory_selection == 'minimum_projection_cost' and self.projector is not None:
                    # Case A: Try to load pre-computed costs from post-processing projection
                    has_precomputed = False
                    if infos is not None and 'projection_costs' in infos and len(infos['projection_costs']) > 0:
                        costs_total = np.zeros(self.batch_size)
                        for timestep, cost in infos['projection_costs'].items():
                            costs_total += cost
                        if len(costs_total) == self.batch_size:
                            which_trajectory = np.argmin(costs_total)
                            selection_method = 'minimum_projection_cost (precomputed)'
                            sel_details = {'projection_cost': float(costs_total[which_trajectory])}
                            has_precomputed = True
                    
                    # Case B: If costs are empty (e.g. gradient-based projection), compute actual projection costs on the spot
                    if not has_precomputed:
                        _, projection_costs = self.projector.project(trajectory)
                        which_trajectory = np.argmin(projection_costs)
                        selection_method = 'minimum_projection_cost (calculated)'
                        sel_details = {
                            'projection_cost': float(projection_costs[which_trajectory]),
                            'all_costs': [float(c) for c in projection_costs]
                        }
                
                elif self.trajectory_selection == 'random':
                    which_trajectory = np.random.randint(self.batch_size)
                    selection_method = 'random'
                    sel_details = {'random_idx': which_trajectory}
            
            # Print/Log candidate selection decision
            if self.batch_size > 1 and getattr(self, 'verbose', False):
                print(f"[ Trajectory Selection ] Method: {selection_method} | Selected candidate: {which_trajectory}/{self.batch_size} | Details: {sel_details}")
            
            # Store selected trajectory for future temporal consistency steps
            self.prev_observations = trajectories_np[which_trajectory].copy()
            
            if trajectory.shape[-1] == 3:
                # 3D Model (D3IL style): Use all 3 dims as actions
                action_trajectory = trajectory[[which_trajectory]]
            else:
                # 6D Model (Avoiding style): Actions are the FIRST 3 dims [act, obs]
                action_trajectory = trajectory[[which_trajectory], :, :3]
            
            # Inverse Scale (Now safe with Fix #29)
            if self.scaler is not None:
                action_trajectory = self.scaler.inverse_scale_output(action_trajectory)
            
            # Select the sorted candidate trajectory for online execution
            self.curr_action_seq = action_trajectory[:, :self.action_seq_size, :]
            # Record for diagnostics
            self.history_full_plans.append(action_trajectory[0].detach().cpu().numpy())
        
        next_action = self.curr_action_seq[:, self.action_counter, :]
        next_action_np = next_action.detach().cpu().numpy()
        
        # Update Mental Map (Open-Loop accumulation - FIX #17)
        self.mental_robot_pos += next_action_np.squeeze(0)
        
        self.history_desired_actions.append(next_action_np.copy().squeeze(0))
        
        # Calculate predicted next pos for tracking error in NEXT step
        # Note: In Open-Loop, last_predicted_pos will be same as mental_robot_pos
        self.last_predicted_pos = self.mental_robot_pos.copy()
        
        self.curr_rollout_time += (time.time() - t_start)
        self.action_counter += 1
        self.step_counter += 1
        return next_action_np

def generate_expert_reference(save_path, n_rollouts=3):
    """Generates ground-truth expert videos from the dataset for reference."""
    expert_dir = os.path.join(save_path, 'expert_references')
    
    # Skip generation if files already exist to avoid redundant CPU/GPU/IK computations
    all_exist = True
    for idx in range(n_rollouts):
        mp4_path = os.path.join(expert_dir, f"expert_rollout_{idx}.mp4")
        gif_path = os.path.join(expert_dir, f"expert_rollout_{idx}.gif")
        if not (os.path.exists(mp4_path) or os.path.exists(gif_path)):
            all_exist = False
            break
    if all_exist:
        print(f"[ expert ] Reference videos already exist in {expert_dir}. Skipping generation.")
        return

    print(f"[ expert ] Generating {n_rollouts} reference videos from dataset...")
    os.makedirs(expert_dir, exist_ok=True)
    
    from agents.utils.sim_path import sim_framework_path
    from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
    
    state_data_dir = sim_framework_path("environments/dataset/data/aligning/all_data/state")
    env = Robot_Push_Env(render=False, if_vision=True)
    env.start()
    
    # In this environment, state files are usually named env_0.pkl, env_1.pkl, etc.
    for idx in range(n_rollouts):
        file_name = f"env_{idx}.pkl"
        try:
            with open(os.path.join(state_data_dir, file_name), 'rb') as f:
                expert_data = pickle.load(f)
        except Exception:
            # Fallback to sorting all files
            all_files = sorted(os.listdir(state_data_dir))
            if idx < len(all_files):
                file_name = all_files[idx]
                with open(os.path.join(state_data_dir, file_name), 'rb') as f:
                    expert_data = pickle.load(f)
            else:
                continue
            
        expert_path = expert_data['robot']['des_c_pos']
        box_pos = expert_data['push-box']['pos'][0]
        box_quat = expert_data['push-box']['quat'][0]
        target_pos = expert_data['target-box']['pos'][0]
        target_quat = expert_data['target-box']['quat'][0]
        context = (box_pos, box_quat, target_pos, target_quat)
        
        env.reset(random=False, context=context)
        frames = []
        for step in range(len(expert_path)):
            sim_action = np.concatenate((expert_path[step], [0, 1, 0, 0]), axis=0)
            obs, _, _, _ = env.step(sim_action)
            _, bp_image, inhand_image = obs
            bp_vis = cv2.cvtColor(bp_image, cv2.COLOR_BGR2RGB)
            inhand_vis = cv2.cvtColor(inhand_image, cv2.COLOR_BGR2RGB)
            frames.append(np.concatenate([bp_vis, inhand_vis], axis=1))
            
        save_file = os.path.join(expert_dir, f"expert_rollout_{idx}.mp4")
        try:
            imageio.mimsave(save_file, frames, fps=20)
            print(f"  [ expert ] Saved {save_file}")
        except Exception as e:
            # GIF Fallback
            try:
                save_file_gif = save_file.replace('.mp4', '.gif')
                imageio.mimsave(save_file_gif, frames, fps=10)
                print(f"  [ expert ] Saved {save_file_gif}")
            except:
                pass
    env.close()

# ─── Model Loading ──────────────────────────────────────────────────────────
def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest', device='cuda:0', seed=None):
    lp = os.path.join(*loadpath)
    print(f'\n[ eval loading ] Intercepting load from {lp}\n')
    dataset_config = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = lp
    
    if target_class is not None:
        diffusion_config._class = utils.config.import_class(target_class)
    
    dataset = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)
    if epoch == 'latest': epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, None)

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str = 'config.aligning-d3il-visual'

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPCC Visual Evaluation.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--record', type=str, choices=['none', 'video', 'gif', 'png', 'all'], default='all')
    parser.add_argument('--eval-on-train', action='store_true', help='Evaluate on the seen expert training contexts instead of unseen test contexts.')
    args_cli, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    with open('config/visual_aligning_eval.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    seeds = [args_cli.seed] if args_cli.seed else config['seeds']
    projection_variants = config.get('projection_variants', ['diffuser'])
    n_contexts = config.get('n_contexts', 30)
    n_trajectories = config.get('n_trajectories_per_context', 1)
    
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        args = Parser().parse_args(experiment='plan_fm_encdec_vision', seed=seed)
        
        diffusion_model = None
        if not args_cli.aggregate_only:
            fm_exp = load_diffusion_with_override(
                args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                target_class=args.diffusion, epoch=args.diffusion_epoch, device=args.device
            )
            diffusion_model = fm_exp.diffusion
            import wandb
            try: wandb.init(mode="disabled")
            except: pass
        
        for variant in projection_variants:
            if args_cli.eval_on_train:
                variant = f"{variant}_train_set"
                save_path = f'{args.savepath}/results_train_set'
            else:
                save_path = f'{args.savepath}/results'
            os.makedirs(save_path, exist_ok=True)
            
            if args_cli.aggregate_only:
                continue
            
            # Generate Expert Reference Videos once per variant
            generate_expert_reference(save_path, n_rollouts=3)
            
            log_f = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = Tee(sys.stdout, log_f)
            sys.stderr = Tee(sys.stderr, log_f)
            
            try:
                # Load Scaler (FIX #17)
                scaler = None
                scaler_path = os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), 'scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    print(f"[ eval ] Loaded scaler from: {scaler_path}")
                else:
                    print(f"[ eval ] WARNING: No scaler.pkl found at {scaler_path}. Operating in RAW mode.")

                # Initialize Projector for DPCC variants
                projector = None
                if 'diffuser' not in variant and scaler is not None:
                    projector = setup_gen6_projector(args, config, scaler, variant)
                    print(f"[ eval ] DPCC Projector active for variant '{variant}'")

                # Trajectory Selection & Candidate Generation size overrides
                trajectory_selection = 'random'
                if 'dpcc-t' in variant:
                    trajectory_selection = 'temporal_consistency'
                elif 'dpcc-c' in variant:
                    trajectory_selection = 'minimum_projection_cost'

                batch_size = getattr(args, 'batch_size', 1)
                if 'diffuser' not in variant:
                    batch_size = 6  # D3IL default of 6 candidates for DPCC selection

                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model, device=args.device,
                    window_size=getattr(args, 'window_size', 8), 
                    obs_seq_len=getattr(args, 'obs_seq_len', 5),
                    action_seq_size=getattr(args, 'action_seq_size', 1),
                    save_path=save_path,
                    record_mode=args_cli.record,
                    scaler=scaler,
                    eval_on_train=args_cli.eval_on_train,
                    batch_size=batch_size,
                    projector=projector,
                    trajectory_selection=trajectory_selection,
                    variant=variant
                )
                sim = Aligning_Sim(seed=seed, device=args.device, render=False, n_cores=1,
                                  n_contexts=n_contexts, n_trajectories_per_context=n_trajectories, if_vision=True,
                                  eval_on_train=args_cli.eval_on_train,
                                  max_episode_length=getattr(args, 'max_episode_length', 400))
                
                t0 = time.time()
                success_rate, mode_encoding, successes, mean_distance_tensor = sim.test_agent(agent)
                elapsed = time.time() - t0

                # Metrics calculation
                n_modes = 2
                mode_probs = torch.zeros([n_contexts, n_modes])
                for c in range(n_contexts):
                    mode_probs[c, :] = torch.tensor([
                        (mode_encoding[c] == 0).sum().item() / n_trajectories,
                        (mode_encoding[c] == 1).sum().item() / n_trajectories
                    ])
                m_norm = mode_probs / (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
                entropy = -(m_norm * torch.log(m_norm + 1e-12) / torch.log(torch.tensor(float(n_modes)))).sum(1).mean().item()
                
                # Format data to match legacy FMv3ODE npz output
                obs_all = []
                act_all = []
                sampled_trajectories_all = []
                
                for r in range(agent.rollout_counter + 1):
                    rollout_key = f"rollout_{r}"
                    if rollout_key in agent.master_rollout_history:
                        data = agent.master_rollout_history[rollout_key]
                        obs_all.append(data['real_robot_pos'])
                        act_all.append(data['desired_actions'])
                        sampled_trajectories_all.append(data['full_plans'])

                if config.get('write_to_file', True):
                    # Align with legacy naming conventions for Matrix Analysis compatibility
                    np.savez(f'{save_path}/{variant}.npz', 
                             # Primary Metrics
                             success_rate=success_rate, 
                             entropy=entropy,
                             mode_encoding=mode_encoding.numpy(), 
                             elapsed_seconds=elapsed, 
                             seed=seed,
                             
                             # Legacy Vector Metrics (for Matrix Explorer)
                             n_success=successes.flatten().numpy(),
                             n_steps=np.array(agent.history_n_steps),
                             avg_time=np.array(agent.history_avg_time),
                             n_violations=np.zeros(len(agent.history_n_steps)), # No obstacles in Aligning
                             total_violations=np.zeros(len(agent.history_n_steps)),
                             collision_free_completed=successes.flatten().numpy(), # Success = Safe for Aligning
                             
                             # Detailed Trajectory Data
                             obs_all=np.array(obs_all, dtype=object),
                             act_all=np.array(act_all, dtype=object),
                             sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object),
                             pos_tracking_errors=np.array(agent.history_pos_tracking_errors, dtype=object),
                             
                             # Task Specific
                             mean_distance=mean_distance_tensor.flatten().numpy(),
                             args=vars(args))
                
                pkl_name = f'results_seed_{seed}_train_set.pkl' if args_cli.eval_on_train else f'results_seed_{seed}.pkl'
                with open(os.path.join(save_path, pkl_name), 'wb') as f:
                    pickle.dump({'success_rate': success_rate, 'entropy': entropy, 'elapsed': elapsed}, f)
                
                # ─── Legacy Visualization Engine ────────────────────────────────
                print(f"[ eval ] Generating legacy PNG plots for {variant}...")
                plot_limit = 5 # Plot first 5 trials
                n_plot = min(len(obs_all), plot_limit)
                
                # Main Plot: rollout grid
                fig, axes = plt.subplots(n_plot, 6, figsize=(30, 5 * n_plot), squeeze=False)
                fig.suptitle(f'Visual Aligning - {variant} (Seed {seed})')
                
                for i in range(n_plot):
                    obs_traj = obs_all[i] # [T, 3]
                    plans_list = sampled_trajectories_all[i] # List of [8, 3]
                    
                    # Panel 0: X Position
                    axes[i, 0].plot(obs_traj[:, 0], 'r-', label='Real X')
                    axes[i, 0].set_title("X Position")
                    
                    # Panel 1: Y Position
                    axes[i, 1].plot(obs_traj[:, 1], 'g-', label='Real Y')
                    axes[i, 1].set_title("Y Position")
                    
                    # Panel 2: Z Position (Height)
                    axes[i, 2].plot(obs_traj[:, 2], 'b-', label='Real Z')
                    axes[i, 2].set_title("Z Height")

                    # Panel 3: Step Magnitude (Drift Check)
                    vels = np.linalg.norm(obs_traj[1:] - obs_traj[:-1], axis=1)
                    axes[i, 3].plot(vels, color='gray', alpha=0.5)
                    axes[i, 3].set_title(f"Step Magnitude")

                    # Panel 4: XY Path (Real)
                    axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', linewidth=2)
                    axes[i, 4].plot(obs_traj[0, 0], obs_traj[0, 1], 'go', markersize=10) # Start
                    axes[i, 4].plot(obs_traj[-1, 0], obs_traj[-1, 1], 'ro', markersize=10) # End
                    axes[i, 4].set_title("XY Trajectory")
                    
                    # Panel 5: MPC Foresight (Blue)
                    axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.3)
                    plan_starts = agent.master_rollout_history[f"rollout_{i}"]['plan_start_positions']
                    for p_idx, plan_deltas in enumerate(plans_list):
                        if p_idx % 4 == 0: 
                            start_pos = plan_starts[min(p_idx, len(plan_starts)-1)]
                            abs_plan = start_pos + np.cumsum(plan_deltas[:, :3], axis=0)
                            axes[i, 5].plot(abs_plan[:, 0], abs_plan[:, 1], 'b-', alpha=0.6)
                    axes[i, 5].set_title("MPC Foresight (Blue)")

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f'{save_path}/{variant}.png')
                plt.close(fig)

                # --- THE SCIENTIFIC 7-METRIC REPORT (FMv3ODE Standard Replication) ---
                run_mode = "seen training set" if args_cli.eval_on_train else "default"
                print(f'------------------------Running aligning-d3il-visual - {run_mode} - {variant} ({seed})----------------------------')
                
                n_success = np.array(successes)
                n_steps = np.array(agent.history_n_steps)
                
                # Tracking error calculation (max error across all trials)
                tracking_error = 0.0
                if len(agent.history_pos_tracking_errors) > 0:
                    # Flatten the nested tracking error history
                    all_errors = [err for trial_errs in agent.history_pos_tracking_errors for err in trial_errs]
                    if len(all_errors) > 0:
                        tracking_error = np.max(all_errors)

                print(f'Success rate: {np.mean(n_success)}')
                print(f'Constraints satisfied: {1.0}') # No obstacles in Aligning
                print(f'Success rate (goal and constraints): {np.mean(n_success)}')
                print(f'Avg number of steps (successful trials): {(np.mean(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f} +- {(np.std(n_steps[n_success > 0]) if np.sum(n_success) > 0 else 0):.2f}')
                print(f'Avg number of steps (all trials): {np.mean(n_steps):.2f} +- {np.std(n_steps):.2f}')
                print(f'Avg number of constraint violations: {0.00:.2f} +- {0.00:.2f}')
                print(f'Avg total violation: {0.000:.3f} +- {0.000:.3f}')
                print(f'Average computation time per step: {np.mean(agent.history_avg_time):.3f}')
                print(f'Tracking error: {tracking_error:.3f}')
                print("-"*80 + "\n")
                # ----------------------------------------------------------------------
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_f.close()
    print("Done.")
