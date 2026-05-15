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
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size  # Force re-plan on first call
        self.curr_action_seq = None
        self.save_path = save_path
        self.record_mode = record_mode
        
        # Diagnostics
        self.rollout_counter = -1
        self.step_counter = 0
        
        # Context windows
        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)
        self.video_frames = []
    
    def reset(self):
        """Called by Aligning_Sim at the start of each rollout."""
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
    
    def _save_diagnostics(self, rollout_idx):
        try:
            import imageio
            diag_dir = os.path.join(self.save_path, 'diagnostics')
            os.makedirs(diag_dir, exist_ok=True)
            
            try:
                if self.record_mode in ['all', 'video']:
                    path = os.path.join(diag_dir, f"rollout_{rollout_idx}.mp4")
                    imageio.mimsave(path, self.video_frames, fps=20)
                elif self.record_mode == 'gif':
                    raise ValueError("Force GIF")
            except Exception as e:
                if self.record_mode in ['all', 'gif']:
                    try:
                        path = os.path.join(diag_dir, f"rollout_{rollout_idx}.gif")
                        imageio.mimsave(path, self.video_frames, fps=20)
                    except Exception as e2:
                        if self.record_mode in ['all', 'png']:
                            self._save_png_sequence(diag_dir, rollout_idx)
                elif self.record_mode == 'png':
                    self._save_png_sequence(diag_dir, rollout_idx)
        except Exception as e:
            print(f"[ WARNING ] Diagnostic recording failed: {e}. Moving on.")

    def _save_png_sequence(self, diag_dir, rollout_idx):
        import cv2
        frame_dir = os.path.join(diag_dir, f"rollout_{rollout_idx}_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(self.video_frames):
            cv2.imwrite(os.path.join(frame_dir, f"f_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    @torch.no_grad()
    def predict(self, state, goal=None, extra_args=None, if_vision=False):
        if if_vision:
            bp_image_np, inhand_image_np, des_robot_pos_np = state
            
            # Diagnostic capture (every 10th rollout)
            if self.record_mode != 'none' and (self.rollout_counter % 10 == 0):
                try:
                    import cv2
                    bp_vis = (bp_image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                    inhand_vis = (inhand_image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                    bp_vis = cv2.cvtColor(bp_vis, cv2.COLOR_BGR2RGB)
                    inhand_vis = cv2.cvtColor(inhand_vis, cv2.COLOR_BGR2RGB)
                    combined = np.concatenate([bp_vis, inhand_vis], axis=1)
                    self.video_frames.append(combined)
                except Exception:
                    pass

            bp_image = torch.from_numpy(bp_image_np).to(self.device).float().unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image_np).to(self.device).float().unsqueeze(0)
            des_robot_pos = torch.from_numpy(des_robot_pos_np).to(self.device).float().unsqueeze(0)
            
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
        
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0
            self.model.eval()
            cond = {0: (bp_image_seq, inhand_image_seq, des_robot_pos_seq)}
            trajectory, _ = self.model(cond)
            self.curr_action_seq = trajectory[:, :self.action_seq_size, :3]
        
        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        self.step_counter += 1
        return next_action.detach().cpu().numpy()

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
            
            log_f = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            old_stdout = sys.stdout
            sys.stdout = Tee(sys.stdout, log_f)
            
            try:
                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model, device=args.device,
                    window_size=getattr(args, 'horizon', 8), save_path=save_path,
                    record_mode=args_cli.record
                )
                sim = Aligning_Sim(seed=seed, device=args.device, render=False, n_cores=1,
                                  n_contexts=n_contexts, n_trajectories_per_context=n_trajectories, if_vision=True)
                
                t0 = time.time()
                success_rate, mode_encoding = sim.test_agent(agent)
                elapsed = time.time() - t0
                
                # Flush last rollout
                if agent.record_mode != 'none' and len(agent.video_frames) > 0:
                    agent._save_diagnostics(agent.rollout_counter)

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
                
                if config.get('write_to_file', True):
                    np.savez(f'{save_path}/{variant}.npz', success_rate=success_rate, entropy=entropy,
                             mode_encoding=mode_encoding.numpy(), elapsed_seconds=elapsed, seed=seed)
                
                with open(os.path.join(save_path, f'results_seed_{seed}.pkl'), 'wb') as f:
                    pickle.dump({'success_rate': success_rate, 'entropy': entropy, 'elapsed': elapsed}, f)
                
                print(f'[ eval ] Success Rate: {success_rate:.4f} | Entropy: {entropy:.4f}')
            finally:
                sys.stdout = old_stdout
                log_f.close()
    print("Done.")
