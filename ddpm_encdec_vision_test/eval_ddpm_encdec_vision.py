# Visual Aligning Evaluation Script
# Uses D3IL's native DiffusionAgent directly — no FM-PCC Policy wrapper needed.
# This is a MANIPULATION task (push block to target), NOT a trajectory-planning avoiding task.
#
# D3IL Aligning Output Metrics:
#   - Success Rate: fraction of rollouts where the block reached the target
#   - Entropy: multimodality measure (push from inside vs outside)
#   - Mean Distance: average distance from block to target at end of rollout
#   - Score: 0.5 * (success_rate + entropy)
#
# These are the STANDARD D3IL metrics, identical to what the original codebase produces.

import os
import sys
import time
import yaml
import torch
import numpy as np
import argparse
import pickle
import json
import logging
from datetime import datetime
from omegaconf import OmegaConf
import hydra

# Ensure d3il is in path
sys.path.append(os.path.abspath('d3il'))
sys.path.append(os.path.abspath('d3il/environments/d3il'))

from d3il.simulation.aligning_sim import Aligning_Sim
from d3il.agents.ddpm_encdec_vision_agent import DiffusionAgent
from d3il.agents.utils.scaler import Scaler

log = logging.getLogger(__name__)

class Tee(object):
    """Mirrors stdout to a log file."""
    def __init__(self, *files):
        self.files = [f if hasattr(f, 'write') else open(f, 'a') for f in files]
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def build_agent_from_config(weights_dir, device='cuda'):
    """
    Reconstruct the D3IL DiffusionAgent from the Hydra config files
    saved during training, then load the best checkpoint weights.
    
    This mirrors exactly what D3IL does natively:
      1. hydra.utils.instantiate(cfg.agents) -> DiffusionAgent
      2. agent.load_pretrained_model(weights_dir)
    """
    # Load the saved Hydra config
    config_path = os.path.join(weights_dir, '.hydra', 'config.yaml')
    if not os.path.exists(config_path):
        # Fallback: try the overrides approach
        config_path = os.path.join(weights_dir, 'config.yaml')
    
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        print(f"[ eval ] Loaded Hydra config from: {config_path}")
    else:
        # Manual config construction from aligning_vision_config defaults
        print(f"[ eval ] No saved config found at {weights_dir}, using default aligning_vision_config")
        cfg = OmegaConf.load('d3il/configs/aligning_vision_config.yaml')
    
    # Override device
    OmegaConf.update(cfg, "device", device, force_add=True)
    
    # Instantiate the agent
    agent = hydra.utils.instantiate(cfg.agents)
    
    # Load the best checkpoint
    sv_name = "eval_best_ddpm.pth"
    if not os.path.exists(os.path.join(weights_dir, sv_name)):
        sv_name = "last_ddpm.pth"
        print(f"[ eval ] Best checkpoint not found, falling back to: {sv_name}")
    
    agent.load_pretrained_model(weights_dir, sv_name=sv_name)
    agent.model.eval()
    print(f"[ eval ] Loaded weights from: {os.path.join(weights_dir, sv_name)}")
    
    return agent


def build_agent_manual(weights_dir, device='cuda'):
    """
    Manual agent construction when Hydra config is not available.
    Uses the standard aligning_vision_config.yaml parameters.
    """
    from agents.models.diffusion.ema import ExponentialMovingAverage
    
    # Standard D3IL aligning vision parameters
    window_size = 8
    obs_dim = 3
    action_dim = 3
    obs_seq_len = 5
    action_seq_size = 4
    
    # Build the obs_encoder
    shape_meta = {
        "obs": {
            "agentview_image": {"shape": [3, 96, 96], "type": "rgb"},
            "in_hand_image": {"shape": [3, 96, 96], "type": "rgb"}
        }
    }
    
    obs_encoder_cfg = OmegaConf.create({
        "_target_": "agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder",
        "shape_meta": shape_meta,
        "rgb_model": {
            "_target_": "agents.models.vision.model_getter.get_resnet",
            "input_shape": [3, 96, 96],
            "output_size": 64
        },
        "resize_shape": None,
        "random_crop": False,
        "use_group_norm": True,
        "share_rgb_model": False,
        "imagenet_norm": True
    })
    
    # Build the diffusion inner model (DiffusionEncDec)
    inner_model_cfg = OmegaConf.create({
        "_target_": "agents.models.diffusion.diffusion_models.DiffusionEncDec",
        "_recursive_": False,
        "state_dim": 128,
        "action_dim": action_dim,
        "goal_conditioned": False,
        "goal_seq_len": 10,
        "obs_seq_len": obs_seq_len,
        "action_seq_len": action_seq_size,
        "embed_pdrob": 0,
        "embed_dim": 64,
        "device": device,
        "linear_output": True,
        "encoder": {
            "_target_": "agents.models.act.act_vae.TransformerEncoder",
            "embed_dim": 64,
            "n_heads": 4,
            "n_layers": 2,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "bias": False,
            "block_size": window_size + 1
        },
        "decoder": {
            "_target_": "agents.models.act.act_vae.TransformerDecoder",
            "embed_dim": 64,
            "cross_embed": 64,
            "n_heads": 4,
            "n_layers": 4,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "bias": False,
            "block_size": window_size + 1
        }
    })
    
    # Build the Diffusion wrapper
    diffusion_cfg = OmegaConf.create({
        "_target_": "agents.models.diffusion.diffusion_policy.Diffusion",
        "_recursive_": False,
        "state_dim": 128,
        "action_dim": action_dim,
        "beta_schedule": "cosine",
        "n_timesteps": 16,
        "loss_type": "l2",
        "clip_denoised": True,
        "predict_epsilon": True,
        "device": device,
        "diffusion_x": False,
        "diffusion_x_M": 10,
        "model": inner_model_cfg
    })
    
    # Build the DiffusionPolicy
    from agents.ddpm_encdec_vision_agent import DiffusionPolicy
    
    obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(device)
    diffusion_model = hydra.utils.instantiate(diffusion_cfg).to(device)
    
    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    torch.nn.Module.__init__(policy)
    policy.visual_input = True
    policy.obs_encoder = obs_encoder
    policy.model = diffusion_model
    policy = policy.to(device)
    
    # Build a minimal scaler from training dataset
    from environments.dataset.aligning_dataset import Aligning_Dataset
    from agents.utils.sim_path import sim_framework_path
    
    train_data_path = "environments/dataset/data/aligning/train_files.pkl"
    dataset = Aligning_Dataset(
        data_directory=train_data_path,
        device="cpu",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_len_data=512,
        window_size=window_size,
    )
    scaler = Scaler(
        dataset.get_all_observations(),
        dataset.get_all_actions(),
        True,  # scale_data
        device,
    )
    
    # Set action bounds for the diffusion sampler
    policy.model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(device)
    policy.model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(device)
    
    # Build a minimal DiffusionAgent-like wrapper
    class MinimalAgent:
        """Mimics DiffusionAgent's predict() interface for Aligning_Sim."""
        def __init__(self, policy, scaler, device, window_size, obs_seq_len, action_seq_size):
            from collections import deque
            self.policy = policy
            self.scaler = scaler
            self.device = device
            self.window_size = window_size
            self.obs_seq_len = obs_seq_len
            self.action_seq_size = action_seq_size
            self.action_counter = self.action_seq_size
            self.curr_action_seq = None
            
            self.bp_image_context = deque(maxlen=self.window_size)
            self.inhand_image_context = deque(maxlen=self.window_size)
            self.des_robot_pos_context = deque(maxlen=self.window_size)
            
            # EMA support
            from agents.models.diffusion.ema import ExponentialMovingAverage
            self.use_ema = True
            self.ema_helper = ExponentialMovingAverage(self.policy.parameters(), 0.995, self.device)
        
        def reset(self):
            self.bp_image_context.clear()
            self.inhand_image_context.clear()
            self.des_robot_pos_context.clear()
            self.action_counter = self.action_seq_size
        
        @torch.no_grad()
        def predict(self, state, goal=None, extra_args=None, if_vision=False):
            if if_vision:
                bp_image, inhand_image, des_robot_pos = state
                
                bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0)
                inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0)
                des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)
                
                des_robot_pos = self.scaler.scale_input(des_robot_pos)
                
                self.bp_image_context.append(bp_image)
                self.inhand_image_context.append(inhand_image)
                self.des_robot_pos_context.append(des_robot_pos)
                
                bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
                inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
                des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
                
                input_state = (bp_image_seq, inhand_image_seq, des_robot_pos_seq)
            else:
                obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                obs = self.scaler.scale_input(obs)
                input_state = obs.unsqueeze(1)
            
            if self.action_counter == self.action_seq_size:
                self.action_counter = 0
                
                self.policy.eval()
                model_pred = self.policy(input_state, goal)
                model_pred = self.scaler.inverse_scale_output(model_pred)
                self.curr_action_seq = model_pred
            
            next_action = self.curr_action_seq[:, self.action_counter, :]
            self.action_counter += 1
            return next_action.detach().cpu().numpy()
        
        def load_pretrained_model(self, weights_path, sv_name):
            state_dict = torch.load(os.path.join(weights_path, sv_name), map_location=self.device)
            self.policy.load_state_dict(state_dict)
            # Reinitialize EMA with loaded weights
            from agents.models.diffusion.ema import ExponentialMovingAverage
            self.ema_helper = ExponentialMovingAverage(self.policy.parameters(), 0.995, self.device)
            print(f"[ eval ] Loaded weights: {os.path.join(weights_path, sv_name)}")
    
    agent = MinimalAgent(policy, scaler, device, window_size, obs_seq_len, action_seq_size)
    
    # Load weights
    sv_name = "eval_best_ddpm.pth"
    if not os.path.exists(os.path.join(weights_dir, sv_name)):
        sv_name = "last_ddpm.pth"
        print(f"[ eval ] Best checkpoint not found, falling back to: {sv_name}")
    
    agent.load_pretrained_model(weights_dir, sv_name=sv_name)
    
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D3IL Visual Aligning Evaluation.')
    parser.add_argument('--weights-dir', type=str, required=True,
                        help='Path to the D3IL training run directory containing eval_best_ddpm.pth')
    parser.add_argument('--seed', type=int, default=42, help='Evaluation seed.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference.')
    parser.add_argument('--n-contexts', type=int, default=30, help='Number of test contexts.')
    parser.add_argument('--n-trajectories', type=int, default=1, help='Trajectories per context.')
    parser.add_argument('--n-cores', type=int, default=1, help='Number of parallel workers.')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Where to save results. Defaults to <weights-dir>/eval_results/')
    args = parser.parse_args()
    
    # Output directory
    output_dir = args.output_dir or os.path.join(args.weights_dir, 'eval_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Tee logger
    log_file_path = os.path.join(output_dir, f'eval_seed_{args.seed}.log')
    log_file = open(log_file_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        print(f"[ eval ] Visual Aligning Evaluation")
        print(f"[ eval ] Weights: {args.weights_dir}")
        print(f"[ eval ] Seed: {args.seed}")
        print(f"[ eval ] Output: {output_dir}")
        print(f"[ eval ] Log: {log_file_path}")
        print(f"[ eval ] Contexts: {args.n_contexts} | Trajectories/ctx: {args.n_trajectories}")
        print()
        
        # ── 1. Build Agent ──────────────────────────────────────────────────
        t0 = time.time()
        
        # Try Hydra config first, fall back to manual construction
        try:
            agent = build_agent_from_config(args.weights_dir, device=args.device)
        except Exception as e:
            print(f"[ eval ] Hydra instantiation failed: {e}")
            print(f"[ eval ] Falling back to manual agent construction...")
            agent = build_agent_manual(args.weights_dir, device=args.device)
        
        print(f"[ eval ] Agent loaded in {time.time() - t0:.1f}s")
        
        # ── 2. Setup Simulation ──────────────────────────────────────────────
        # Use a dummy wandb to prevent crashes if wandb is not configured
        import wandb
        try:
            wandb.init(mode="disabled")
        except Exception:
            pass
        
        print(f"[ eval ] Initializing Aligning_Sim (vision=True)")
        sim = Aligning_Sim(
            seed=args.seed,
            device=args.device,
            render=False,
            n_cores=args.n_cores,
            n_contexts=args.n_contexts,
            n_trajectories_per_context=args.n_trajectories,
            if_vision=True
        )
        
        # ── 3. Run Evaluation ────────────────────────────────────────────────
        t1 = time.time()
        print(f"[ eval ] Starting rollouts...")
        success_rate, mode_encoding = sim.test_agent(agent)
        elapsed = time.time() - t1
        
        # ── 4. Save Results ──────────────────────────────────────────────────
        # The Aligning_Sim.test_agent already prints success_rate, entropy, and
        # mean_distance to stdout (which our Tee captures). We also save a 
        # structured result file for the Performance Scorecard.
        results = {
            'success_rate': success_rate,
            'mode_encoding': mode_encoding.numpy(),
            'seed': args.seed,
            'weights_dir': args.weights_dir,
            'n_contexts': args.n_contexts,
            'n_trajectories_per_context': args.n_trajectories,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }
        
        res_file = os.path.join(output_dir, f'results_seed_{args.seed}.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n[ eval ] ═══════════════════════════════════════════════")
        print(f"[ eval ] Results saved to: {res_file}")
        print(f"[ eval ] Evaluation completed in {elapsed:.1f}s")
        print(f"[ eval ] ═══════════════════════════════════════════════")
        
    finally:
        sys.stdout = original_stdout
        log_file.close()
