# Visual Aligning Evaluation Script — DPCC/FMPCC Upgrade
# ═══════════════════════════════════════════════════════════════════════
# Built on the FMv3ODE eval blueprint (eval_flow_matching_v3_ode_selectable.py).
# Key adaptation: uses Aligning_Sim.test_agent() with D3IL-native agent wrapper
# instead of ObstacleAvoidanceEnv with manual step loop.
#
# Output:  logs/aligning-d3il-visual/plans/ddpm_encdec_vision/H8/<seed>/results/
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

import ddpm_encdec_vision.utils as utils

# Ensure local d3il is prioritized in path
sys.path.insert(0, os.path.abspath('d3il'))
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))

# Fix MuJoCo resource loading: Force D3IL_DIR to local path
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

import d3il
print(f"[ eval ] Using d3il from: {d3il.__file__}")
print(f"[ eval ] D3IL_DIR set to: {os.environ['D3IL_DIR']}")

from d3il.simulation.aligning_sim import Aligning_Sim

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
    def __init__(self, diffusion_model, device, window_size=8, obs_seq_len=5, action_seq_size=4, save_path=None, record_mode='all'):
        self.model = diffusion_model
        self.device = device
        self.window_size = window_size
        self.obs_seq_len = obs_seq_len
        
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
        self.des_robot_pos_context = deque(maxlen=self.window_size)
        self.video_frames = []
    
    def reset(self):
        """Called by Aligning_Sim at the start of each rollout."""
        # Flush previous rollout data
        if self.rollout_counter >= 0:
            self.master_rollout_history[f"rollout_{self.rollout_counter}"] = {
                "real_robot_pos": np.array(self.history_real_pos),
                "desired_actions": np.array(self.history_desired_actions),
                "full_plans": np.array(self.history_full_plans),
                "plan_start_positions": np.array(self.history_real_pos)[::self.action_seq_size, :]
            }
            self.history_n_steps.append(self.step_counter)
            self.history_avg_time.append(self.curr_rollout_time / max(1, self.step_counter))
            self.history_pos_tracking_errors.append(np.array(self.curr_rollout_tracking_errors))
            
        self.history_real_pos.clear()
        self.history_desired_actions.clear()
        self.history_full_plans.clear()
        self.curr_rollout_time = 0
        self.last_predicted_pos = None
        self.curr_rollout_tracking_errors.clear()
        
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
        self.action_counter = self.action_seq_size
        self.rollout_counter += 1
        self.step_counter = 0
        
        # Save diagnostics from previous rollout
        if self.record_mode != 'none' and len(self.video_frames) > 0:
            self._save_diagnostics(self.rollout_counter - 1)
        self.video_frames = []
    
    def _save_diagnostics(self, rollout_idx, custom_path=None, custom_frames=None):
        frames = custom_frames if custom_frames is not None else self.video_frames
        path = custom_path if custom_path is not None else os.path.join(self.save_path, 'diagnostics')
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
        except Exception as e:
            print(f"[ WARNING ] Diagnostics engine failed: {e}. Skipping.")

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

            # Preprocess images to [C, H, W] and normalize
            bp_image = torch.from_numpy(bp_image_np).to(self.device).float().unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image_np).to(self.device).float().unsqueeze(0)
            des_robot_pos = torch.from_numpy(des_robot_pos_np).to(self.device).float().unsqueeze(0)
            
            # Record real pos
            self.history_real_pos.append(des_robot_pos_np.copy())
            
            # Record tracking error from PREVIOUS step prediction
            if self.last_predicted_pos is not None:
                err = np.linalg.norm(des_robot_pos_np[:2] - self.last_predicted_pos[:2])
                self.curr_rollout_tracking_errors.append(err)
            
            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            self.des_robot_pos_context.append(des_robot_pos)
            
            while len(self.bp_image_context) < self.window_size:
                self.bp_image_context.appendleft(bp_image)
                self.inhand_image_context.appendleft(inhand_image)
                self.des_robot_pos_context.appendleft(des_robot_pos)
            
            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
        else:
            raise NotImplementedError()
        
        t_start = time.time()
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0
            self.model.eval()
            cond = {0: (bp_image_seq, inhand_image_seq, des_robot_pos_seq)}
            trajectory, _ = self.model(cond)
            self.curr_action_seq = trajectory[:, :self.action_seq_size, :3]
            # Record full plan
            self.history_full_plans.append(trajectory.detach().cpu().numpy().squeeze(0))
        
        next_action = self.curr_action_seq[:, self.action_counter, :]
        next_action_np = next_action.detach().cpu().numpy()
        self.history_desired_actions.append(next_action_np.copy().squeeze(0))
        
        # Calculate predicted next pos for tracking error in NEXT step
        self.last_predicted_pos = next_action_np.squeeze(0) + des_robot_pos_np
        
        self.curr_rollout_time += (time.time() - t_start)
        self.action_counter += 1
        self.step_counter += 1
        return next_action_np

def generate_expert_reference(save_path, n_rollouts=3):
    """Generates ground-truth expert videos from the dataset for reference."""
    print(f"[ expert ] Generating {n_rollouts} reference videos from dataset...")
    expert_dir = os.path.join(save_path, 'expert_references')
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
        args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=seed)
        
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
            save_path = f'{args.savepath}/results'
            os.makedirs(save_path, exist_ok=True)
            
            if args_cli.aggregate_only:
                continue
            
            # Generate Expert Reference Videos once per variant
            generate_expert_reference(save_path, n_rollouts=3)
            
            log_f = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            old_stdout = sys.stdout
            sys.stdout = Tee(sys.stdout, log_f)
            
            try:
                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model, device=args.device,
                    window_size=getattr(args, 'window_size', 8), 
                    obs_seq_len=getattr(args, 'obs_seq_len', 5),
                    action_seq_size=getattr(args, 'action_seq_size', 1),
                    save_path=save_path,
                    record_mode=args_cli.record
                )
                sim = Aligning_Sim(seed=seed, device=args.device, render=False, n_cores=1,
                                  n_contexts=n_contexts, n_trajectories_per_context=n_trajectories, if_vision=True)
                
                t0 = time.time()
                success_rate, mode_encoding, successes, mean_distance_tensor = sim.test_agent(agent)
                elapsed = time.time() - t0
                
                # Flush last rollout
                if agent.record_mode != 'none' and len(agent.video_frames) > 0:
                    agent._save_diagnostics(agent.rollout_counter)
                
                if agent.rollout_counter >= 0:
                    agent.master_rollout_history[f"rollout_{agent.rollout_counter}"] = {
                        "real_robot_pos": np.array(agent.history_real_pos),
                        "desired_actions": np.array(agent.history_desired_actions),
                        "full_plans": np.array(agent.history_full_plans),
                        "plan_start_positions": np.array(agent.history_real_pos)[::agent.action_seq_size, :]
                    }
                    agent.history_n_steps.append(agent.step_counter)
                    agent.history_avg_time.append(agent.curr_rollout_time / max(1, agent.step_counter))
                    agent.history_pos_tracking_errors.append(np.array(agent.curr_rollout_tracking_errors))

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
                
                with open(os.path.join(save_path, f'results_seed_{seed}.pkl'), 'wb') as f:
                    pickle.dump({'success_rate': success_rate, 'entropy': entropy, 'elapsed': elapsed}, f)
                
                # ─── Legacy Visualization Engine ────────────────────────────────
                print(f"[ eval ] Generating legacy PNG plots for {variant}...")
                plot_limit = 5 # Plot first 5 trials
                n_plot = min(len(obs_all), plot_limit)
                
                # Main Plot: rollout grid
                fig, axes = plt.subplots(n_plot, 6, figsize=(30, 5 * n_plot), squeeze=False)
                fig.suptitle(f'Visual Aligning - {variant} (Seed {seed})')
                
                # Aggregate Plot: all paths overlaid
                fig_all, ax_all = plt.subplots(figsize=(10, 10))
                ax_all.set_title(f'Aggregate Paths - {variant}')
                
                for i in range(n_plot):
                    obs_traj = obs_all[i] # [T, 3]
                    plans_list = sampled_trajectories_all[i] # List of [8, 3]
                    
                    # Col 3: Plan Delta Stats (Magnitude)
                    axes[i, 3].plot(np.linalg.norm(obs_traj[1:] - obs_traj[:-1], axis=1), color='gray', alpha=0.5)
                    axes[i, 3].set_title(f"Step Magnitude")

                    # Col 4: XY Path (Real)
                    axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', label='Real Path')
                    axes[i, 4].plot(obs_traj[0, 0], obs_traj[0, 1], 'go', label='Start')
                    axes[i, 4].plot(obs_traj[-1, 0], obs_traj[-1, 1], 'ro', label='End')
                    axes[i, 4].set_title("XY Path")
                    axes[i, 4].legend()
                    
                    # Col 5: XY Path + Sampled Plans (FIXED: Absolute conversion)
                    axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.5, label='Real')
                    # Plot plans (blue lines)
                    plan_starts = agent.master_rollout_history[f"rollout_{i}"]['plan_start_positions']
                    for p_idx, plan_deltas in enumerate(plans_list):
                        if p_idx % 4 == 0: # Sparse plotting for clarity
                            start_pos = plan_starts[min(p_idx, len(plan_starts)-1)]
                            # Convert relative deltas to absolute trajectory
                            abs_plan = start_pos + np.cumsum(plan_deltas[:, :3], axis=0)
                            axes[i, 5].plot(abs_plan[:, 0], abs_plan[:, 1], 'b-', alpha=0.4)
                    axes[i, 5].set_title("Foresight (Blue) vs Real (Black)")

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f'{save_path}/{variant}.png')
                plt.close(fig)
                
                fig_all.savefig(f'{save_path}/all.png')
                plt.close(fig_all)

                # --- THE SCIENTIFIC 7-METRIC REPORT (FMv3ODE Standard) ---
                print("\n" + "-"*80)
                print(f" FINAL SCIENTIFIC REPORT: {variant} (Seed {seed})")
                print("-"*80)
                print(f" Success rate:                         {success_rate:>6.2f}")
                print(f" Constraints satisfied:                {1.0:>6.2f} (No obstacles)")
                print(f" Success rate (goal and constraints):   {success_rate:>6.2f}")
                print(f" Avg number of steps:                  {np.mean(agent.history_n_steps):>6.2f} +- 0.00")
                print(f" Avg number of constraint violations:   {0.0:>6.2f} +- 0.00")
                print(f" Avg total violation:                  {0.000:>6.3f} +- 0.000")
                print(f" Average computation time per step:    {np.mean(agent.history_avg_time):>6.3f}")
                print("-"*80 + "\n")
                # ------------------------------------------------------
            finally:
                sys.stdout = old_stdout
                log_f.close()
    print("Done.")
