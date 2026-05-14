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

# Ensure d3il is in path
sys.path.append(os.path.abspath('d3il'))
sys.path.append(os.path.abspath('d3il/environments/d3il'))

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
    
    Replicates D3IL's DiffusionAgent.predict() exactly:
      1. Maintains a sliding-window context of past frames (deque, maxlen=window_size)
      2. Applies the training-time Scaler for input normalization
      3. Runs the D3IL DiffusionPolicy (obs_encoder + diffusion_model)
      4. Returns action chunks (action_seq_size=4 steps per inference call)
    """
    def __init__(self, diffusion_model, scaler, device, window_size=8, obs_seq_len=5, action_seq_size=4):
        self.model = diffusion_model
        self.scaler = scaler
        self.device = device
        self.window_size = window_size
        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size  # Force re-plan on first call
        self.curr_action_seq = None
        
        # Context windows — match D3IL's DiffusionAgent exactly
        self.bp_image_context = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.des_robot_pos_context = deque(maxlen=self.window_size)
    
    def reset(self):
        """Called by Aligning_Sim at the start of each rollout."""
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()
        self.action_counter = self.action_seq_size
    
    @torch.no_grad()
    def predict(self, state, goal=None, extra_args=None, if_vision=False):
        """
        Aligning_Sim calls:  agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=True)
        
        Returns: np.array of shape [1, 3] (velocity delta for the end-effector)
        """
        if if_vision:
            bp_image, inhand_image, des_robot_pos = state
            
            # Convert numpy → tensor
            bp_image = torch.from_numpy(bp_image).to(self.device).float().unsqueeze(0)
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().unsqueeze(0)
            des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)
            
            # Apply training-time normalization
            des_robot_pos_scaled = self.scaler.scale_input(des_robot_pos)
            
            # Append to sliding window
            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            self.des_robot_pos_context.append(des_robot_pos_scaled)
            
            # Stack context window → [1, T, C, H, W] and [1, T, 3]
            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
        else:
            raise NotImplementedError("Non-vision mode not supported for VisualAgentWrapper")
        
        # Action chunking: only re-plan every action_seq_size steps
        if self.action_counter == self.action_seq_size:
            self.action_counter = 0
            self.model.eval()
            
            # VisualGaussianDiffusion.forward() expects:
            #   cond = {0: (bp_imgs, inhand_imgs, pos)}
            # It internally encodes the visual features via VisualUNet
            cond = {0: (bp_image_seq, inhand_image_seq, des_robot_pos_seq)}
            
            # Returns: ([batch, horizon, transition_dim], infos)
            trajectory, infos = self.model(cond)
            
            # Extract actions from trajectory: first action_dim=3 columns
            action_dim = 3
            actions = trajectory[:, :self.action_seq_size, :action_dim]
            
            # Inverse-scale the output
            actions = self.scaler.inverse_scale_output(actions)
            self.curr_action_seq = actions
        
        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()


# ─── Model Loading (FM-PCC Config Pickle System) ───────────────────────────
def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest', device='cuda:0', seed=None):
    """
    Replicated from FMv3ODE eval: loads model from FM-PCC pickle configs.
    Returns the loaded diffusion model and dataset.
    """
    import inspect
    
    lp = os.path.join(*loadpath)
    print(f'\n[ eval loading ] Intercepting load from {lp}\n')
    
    dataset_config = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = lp
    
    if target_class is not None:
        target_class_resolved = utils.config.import_class(target_class)
        target_class_str = target_class_resolved.__module__ + '.' + target_class_resolved.__name__
        pickled_class_str = diffusion_config._class.__module__ + '.' + diffusion_config._class.__name__
        
        if pickled_class_str != target_class_str:
            print(f"\n=======================================================", file=sys.stderr)
            print(f"[WARNING] Pickled diffusion class does not match config!", file=sys.stderr)
            print(f"Pickled config class: {pickled_class_str}", file=sys.stderr)
            print(f"Existing config class: {target_class_str}", file=sys.stderr)
            print(f"Overriding pickled config with existing config!", file=sys.stderr)
            print(f"=======================================================\n", file=sys.stderr)
            diffusion_config._class = target_class_resolved
            
            sig = inspect.signature(target_class_resolved.__init__)
            valid_kwargs = set(sig.parameters.keys())
            keys_to_remove = [k for k in diffusion_config._dict if k not in valid_kwargs]
            for k in keys_to_remove:
                print(f"[WARNING] Dropping unexpected kwarg from pickle: '{k}'", file=sys.stderr)
                del diffusion_config._dict[k]
    
    print(f"\n[INFO] Instantiating Diffusion Model from:", file=sys.stderr)
    print(f"       -> {inspect.getfile(diffusion_config._class)}\n", file=sys.stderr)
    
    dataset = dataset_config()
    model = model_config()
    diffusion_config._dict.pop('model', None)
    
    sig = inspect.signature(diffusion_config._class.__init__)
    valid_kwargs = set(sig.parameters.keys())
    keys_to_remove = [k for k in diffusion_config._dict if k not in valid_kwargs]
    for k in keys_to_remove:
        print(f"[WARNING] Dropping unexpected kwarg from pickle: '{k}'", file=sys.stderr)
        del diffusion_config._dict[k]
    
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)
    
    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    
    trainer.load(epoch)
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, None)


# ─── Scaler Construction ───────────────────────────────────────────────────
def build_scaler(device='cuda'):
    """
    Build the D3IL Scaler from the aligning training dataset.
    This is the SAME scaler used during training — critical for correct normalization.
    Uses Aligning_Img_Dataset (vision variant) which extracts obs_dim=3 (robot_des_pos).
    """
    from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
    from d3il.agents.utils.scaler import Scaler
    
    dataset = Aligning_Img_Dataset(
        data_directory='environments/dataset/data/aligning/train_files.pkl',
        device='cpu',
        obs_dim=3,
        action_dim=3,
        max_len_data=512,
        window_size=8,
    )
    scaler = Scaler(
        dataset.get_all_observations(),
        dataset.get_all_actions(),
        True,  # scale_data=True (matches training)
        device,
    )
    return scaler


# ─── Parser ─────────────────────────────────────────────────────────────────
class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str = 'config.aligning-d3il-visual'


# ═════════════════════════════════════════════════════════════════════════════
# Main — mirrors eval_flow_matching_v3_ode_selectable.py structure exactly
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='DPCC Visual Aligning Evaluation.')
    parser.add_argument('--seed', type=int, help='Run only this specific seed.')
    parser.add_argument('--aggregate-only', action='store_true', help='Skip inference, only aggregate existing results.')
    args_cli, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # --- YAML Config Loading (FMv3ODE Standard) ---
    with open('config/visual_aligning_eval.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    exp = 'aligning-d3il-visual'
    seeds = config['seeds']
    if args_cli.seed is not None:
        seeds = [args_cli.seed]
        print(f'[ eval ] Overriding seeds from config to: {seeds}')
    
    # Visual Aligning uses a single "diffuser" variant (no projection yet)
    # but the loop structure supports adding projection variants later
    projection_variants = config.get('projection_variants', ['diffuser'])
    n_contexts = config.get('n_contexts', 30)
    n_trajectories = config.get('n_trajectories_per_context', 1)
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Evaluating seed {seed}...")
        print(f"{'='*60}")
        
        args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=seed)
        
        diffusion_model = None
        scaler = None
        
        if not args_cli.aggregate_only:
            # ── Model Loading (FM-PCC Pickle Config System) ──────────────
            fm_experiment = load_diffusion_with_override(
                args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                target_class=args.diffusion, epoch=args.diffusion_epoch, device=args.device
            )
            diffusion_model = fm_experiment.diffusion
            
            # ── Build Scaler ─────────────────────────────────────────────
            scaler = build_scaler(device=args.device)
            
            # Set action bounds on the diffusion model's inner sampler
            try:
                if hasattr(diffusion_model, 'model') and hasattr(diffusion_model.model, 'backbone'):
                    pass  # VisualUNet doesn't need action bounds directly
                diffusion_model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(args.device)
                diffusion_model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(args.device)
            except Exception as e:
                print(f"[ eval ] Warning: Could not set action bounds: {e}")
            
            # ── Initialize Simulation ────────────────────────────────────
            # Disable wandb to prevent crashes
            import wandb
            try:
                wandb.init(mode="disabled")
            except Exception:
                pass
        
        for variant_idx, variant in enumerate(projection_variants):
            # ── Save Path (FM-PCC standard: nested under model dir) ──────
            save_path = f'{args.savepath}/results'
            os.makedirs(save_path, exist_ok=True)
            
            if args_cli.aggregate_only:
                npz_path = os.path.join(save_path, f'{variant}.npz')
                if not os.path.exists(npz_path):
                    print(f'[ eval ] Skipping {variant} for seed {seed}, no results found at {npz_path}')
                    continue
                print(f'[ eval ] Aggregating existing results for {variant} - seed {seed}')
                data = np.load(npz_path, allow_pickle=True)
                print(f"         Success Rate: {data['success_rate']:.4f}")
                print(f"         Entropy:      {data['entropy']:.4f}")
                print(f"         Mean Distance:{data['mean_distance']:.4f}")
                continue
            
            # ── Inference Mode ───────────────────────────────────────────
            log_file = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            original_stdout = sys.stdout
            sys.stdout = Tee(sys.stdout, log_file)
            
            try:
                print(f'─── Running {exp} - {variant} (seed {seed}) ───')
                print(f'[ eval ] Save path: {save_path}')
                
                # Build the D3IL-native agent wrapper
                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model,
                    scaler=scaler,
                    device=args.device,
                    window_size=getattr(args, 'horizon', 8),
                    obs_seq_len=5,
                    action_seq_size=4,
                )
                
                # Initialize Simulation
                print(f"[ eval ] Initializing Aligning_Sim (vision=True)")
                sim = Aligning_Sim(
                    seed=seed,
                    device=args.device,
                    render=False,
                    n_cores=1,
                    n_contexts=n_contexts,
                    n_trajectories_per_context=n_trajectories,
                    if_vision=True
                )
                
                # Run evaluation through D3IL's native test_agent
                t0 = time.time()
                success_rate, mode_encoding = sim.test_agent(agent)
                elapsed = time.time() - t0
                
                # Compute entropy (replicating Aligning_Sim's internal calculation)
                n_modes = 2
                successes = (mode_encoding >= 0).float()  # approximate
                mode_probs = torch.zeros([n_contexts, n_modes])
                for c in range(n_contexts):
                    mode_probs[c, :] = torch.tensor([
                        (mode_encoding[c] == 0).sum().item() / n_trajectories,
                        (mode_encoding[c] == 1).sum().item() / n_trajectories
                    ])
                mode_probs_norm = mode_probs / (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
                entropy = -(mode_probs_norm * torch.log(mode_probs_norm + 1e-12) / 
                           torch.log(torch.tensor(float(n_modes)))).sum(1).mean().item()
                
                # Save results as .npz (FMv3ODE standard)
                if config.get('write_to_file', True):
                    np.savez(
                        f'{save_path}/{variant}.npz',
                        success_rate=success_rate,
                        entropy=entropy,
                        mode_encoding=mode_encoding.numpy(),
                        mean_distance=0.0,  # captured by Aligning_Sim's stdout
                        elapsed_seconds=elapsed,
                        n_contexts=n_contexts,
                        n_trajectories_per_context=n_trajectories,
                        seed=seed,
                        variant=variant,
                    )
                
                # Also save structured .pkl (for Performance Scorecard)
                results = {
                    'success_rate': success_rate,
                    'entropy': entropy,
                    'mode_encoding': mode_encoding.numpy(),
                    'score': 0.5 * (success_rate + entropy),
                    'elapsed_seconds': elapsed,
                    'timestamp': datetime.now().isoformat(),
                    'args': vars(args),
                }
                res_file = os.path.join(save_path, f'results_seed_{seed}.pkl')
                with open(res_file, 'wb') as f:
                    pickle.dump(results, f)
                
                print(f'\n[ eval ] ═══════════════════════════════════════════════')
                print(f'[ eval ] Variant: {variant}')
                print(f'[ eval ] Success Rate: {success_rate:.4f}')
                print(f'[ eval ] Entropy:      {entropy:.4f}')
                print(f'[ eval ] Score:         {0.5 * (success_rate + entropy):.4f}')
                print(f'[ eval ] Elapsed:       {elapsed:.1f}s')
                print(f'[ eval ] Results:       {res_file}')
                print(f'[ eval ] ═══════════════════════════════════════════════')
            
            finally:
                sys.stdout = original_stdout
                log_file.close()
    
    print(f"\n[ eval ] All seeds completed.")
