# FM_v3 ODE-selectable version of eval_ddpm_encdec_vision.py
import time
import yaml
import os
import torch
import numpy as np
import argparse
import pickle
import json
from datetime import datetime
import sys

import ddpm_encdec_vision.utils as utils
from flow_matcher_v3_ode_selectable.sampling.policies import Policy

# Ensure d3il is in path
sys.path.append(os.path.abspath('d3il'))
sys.path.append(os.path.abspath('d3il/environments/d3il'))
from d3il.simulation.aligning_sim import Aligning_Sim

class IdentityNormalizer:
    """Pass-through normalizer for vision pipeline."""
    def normalize(self, x, *args, **kwargs): return x
    def unnormalize(self, x, *args, **kwargs): return x

class VisualAgentWrapper:
    """Bridges FM-PCC Policy into Aligning_Sim."""
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
    
    def reset(self):
        pass
        
    def predict(self, obs, if_vision=True):
        bp_image, inhand_image, des_robot_pos = obs
        
        # Format inputs [1, 1, C, H, W]
        bp_tensor = torch.tensor(bp_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        inhand_tensor = torch.tensor(inhand_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        pos_tensor = torch.tensor(des_robot_pos).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        # Condition matches VisualModel input
        cond = (bp_tensor, inhand_tensor, pos_tensor)
        
        # Temporarily bypass Policy._format_conditions which crashes when trying to apply to_torch/einops to tuples
        original_format = self.policy._format_conditions
        self.policy._format_conditions = lambda conditions, batch_size: conditions
        
        # Policy call
        action, samples = self.policy(conditions={0: cond}, batch_size=1)
        
        # Restore formatting function
        self.policy._format_conditions = original_format
        
        # action is [batch, action_dim], e.g., [1, 3]
        return action.detach().cpu().numpy()

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

def load_diffusion_with_override(loadbase, dataset, diffusion_loadpath, seed, target_class=None, epoch='best', device='cuda'):
    """Replicated from FMv3ODE: Loads vision model with full metadata support."""
    # If the path is relative, resolve it using the standard hierarchy
    if diffusion_loadpath.startswith('f:'):
        diffusion_loadpath = diffusion_loadpath[2:]
        loadpath = os.path.join(loadbase, dataset, diffusion_loadpath, seed)
    elif not os.path.isabs(diffusion_loadpath):
        loadpath = os.path.join(loadbase, dataset, diffusion_loadpath, seed)
    else:
        loadpath = diffusion_loadpath

    print(f'[ eval ] Loading Vision Diffusion model: {loadpath} | Epoch: {epoch}')
    
    # Load configs
    dataset_config = utils.load_config(loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(loadpath, 'trainer_config.pkl')
    
    dataset_obj = dataset_config()
    model_obj = model_config()
    diffusion_config._dict.pop('model', None) # Prevent duplicate positional/kwarg
    
    # Safely filter _dict to only include arguments the class accepts
    import inspect
    sig = inspect.signature(diffusion_config._class.__init__)
    valid_kwargs = set(sig.parameters.keys())
    keys_to_remove = [k for k in diffusion_config._dict if k not in valid_kwargs]
    for k in keys_to_remove:
        print(f"[WARNING] Dropping unexpected kwarg from pickle: '{k}'", file=sys.stderr)
        del diffusion_config._dict[k]
        
    diffusion_obj = diffusion_config(model_obj).to(device)
    trainer_obj = trainer_config(diffusion_model=diffusion_obj, dataset=dataset_obj)
    
    if epoch == 'latest':
        epoch = utils.get_latest_epoch([loadpath])
    
    trainer_obj.load(epoch)
    return diffusion_obj

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str = 'config.aligning-d3il-visual'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full PCC-Bone Evaluation for Visual Aligning.')
    parser.add_argument('--seed', type=int, help='Run only this specific seed.')
    parser.add_argument('--aggregate-only', action='store_true', help='Skip inference, only aggregate existing results.')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # --- YAML Config Loading (FMv3ODE Standard) ---
    with open('config/projection_eval.yaml', 'r') as file:
        config_yaml = yaml.safe_load(file)

    seeds = config_yaml['seeds']
    if args_cli.seed is not None:
        seeds = [args_cli.seed]
        print(f'[ eval ] Overriding seeds from config to: {seeds}')

    # For Visual Aligning, we usually only have one 'exp' and no projection variants
    # but we follow the loop structure for parity.
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        
        args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=seed)
        
        # 1. Logging setup (PCC Bone Replication)
        if args.diffusion_loadpath.startswith('f:'):
            formatted_path = args.diffusion_loadpath[2:].format(**vars(args))
            load_dir = os.path.join(args.loadbase, args.dataset, formatted_path, str(args.seed))
        elif not os.path.isabs(args.diffusion_loadpath):
            load_dir = os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed))
        else:
            load_dir = args.diffusion_loadpath

        # Standard nested result directory
        res_path = os.path.join(load_dir, args.exp_name)
        os.makedirs(res_path, exist_ok=True)
        
        # Tee logger
        log_file_path = os.path.join(res_path, f'eval_seed_{args.seed}.log')
        log_file = open(log_file_path, 'w')
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)
        
        try:
            print(f'[ eval ] Log saved to: {log_file_path}')

            # 2. Model Loading
            diffusion = load_diffusion_with_override(
                args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), 
                target_class=args.diffusion, epoch='best', device=args.device
            )
            
            # Policy instantiation
            policy = Policy(
                model=diffusion,
                normalizer=IdentityNormalizer(), # Vision uses pass-through
                preprocess_fns=[],
                test_ret=0
            )

            # 3. Simulation Environment
            print("[ eval ] Initializing Aligning_Sim (Vision=True)")
            sim = Aligning_Sim(
                seed=args.seed,
                device=args.device,
                render=False,
                n_cores=1,
                n_contexts=30,
                n_trajectories_per_context=1,
                if_vision=True
            )
            
            agent = VisualAgentWrapper(policy, args.device)
            
            # 4. Run Evaluation
            if not args_cli.aggregate_only:
                print(f"[ eval ] Starting simulation for seed {args.seed}...")
                success_rate, mode_encoding = sim.test_agent(agent)
                
                # Save results (PCC Bone)
                res = {
                    'success_rate': success_rate,
                    'mode_encoding': mode_encoding,
                    'timestamp': datetime.now().isoformat(),
                    'args': vars(args)
                }
                res_file = os.path.join(res_path, f'results_seed_{args.seed}.pkl')
                with open(res_file, 'wb') as f:
                    pickle.dump(res, f)
                print(f"[ eval ] Results saved to: {res_file}")
                print(f"[ eval ] Success Rate: {success_rate:.4f}")
            else:
                print("[ eval ] Aggregate-only mode active. Skipping simulation.")

        finally:
            sys.stdout = original_stdout
            log_file.close()

    print("\n[ eval ] All seeds completed.")
