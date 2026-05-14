# FM_v3 ODE-selectable version of eval_ddpm_encdec_vision.py
import os
import sys
import torch
import numpy as np
import argparse
import pickle
import json
import glob
from datetime import datetime

import ddpm_encdec_vision.utils as utils
from flow_matcher_v3_ode_selectable.sampling.policies import Policy

# Ensure d3il is in path
sys.path.append(os.path.abspath('d3il'))
sys.path.append(os.path.abspath('d3il/environments/d3il'))
from d3il.simulation.aligning_sim import Aligning_Sim

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
        
        # Policy call
        action, samples = self.policy(conditions={0: cond}, batch_size=1)
        
        return action.detach().cpu().numpy()[0]

def load_diffusion_with_override(loadbase, dataset, diffusion_loadpath, seed, target_class=None, epoch='best', device='cuda'):
    """Replicated from FMv3ODE: Loads vision model with full metadata support."""
    if diffusion_loadpath.startswith('f:'):
        # Resolve 'f:' prefix using dataset and seed
        diffusion_loadpath = diffusion_loadpath[2:]
        # Note: We assume the user has the 'horizon' etc. in the path string to be formatted
        # But for the vision refactor, we simplify to the standard path
        loadpath = os.path.join(loadbase, diffusion_loadpath, seed)
    else:
        loadpath = diffusion_loadpath

    print(f'[ eval ] Loading Vision Diffusion model: {loadpath} | Epoch: {epoch}')
    
    # Load configs
    dataset_config = utils.load_config(loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(loadpath, 'trainer_config.pkl')
    
    # Instantiate components
    dataset_obj = dataset_config()
    model_obj = model_config()
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
    parser.add_argument('--seed', type=int, default=5, help='Seed')
    parser.add_argument('--epoch', type=str, default='best', help='Epoch')
    parser.add_argument('--aggregate-only', action='store_true', help='Only aggregate existing results.')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=args_cli.seed)
    
    # 1. Logging setup (PCC Bone Replication)
    if args.diffusion_loadpath.startswith('f:'):
        formatted_path = args.diffusion_loadpath[2:].format(**vars(args))
        load_dir = os.path.join(args.loadbase, formatted_path, str(args.seed))
    else:
        load_dir = args.diffusion_loadpath

    # Standard nested result directory
    res_path = os.path.join(load_dir, args.exp_name)
    os.makedirs(res_path, exist_ok=True)
    
    # Tee logger
    log_file = os.path.join(res_path, f'eval_seed_{args.seed}.log')
    sys.stdout = utils.Tee(sys.stdout, log_file)
    print(f'[ eval ] Log saved to: {log_file}')

    # 2. Model Loading
    diffusion = load_diffusion_with_override(
        args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed), 
        target_class=args.diffusion, epoch=args_cli.epoch, device=args.device
    )
    
    # 3. Policy Construction
    policy = Policy(model=diffusion, normalizer=None) 
    
    # 4. Simulation Environment
    print("[ eval ] Initializing Aligning_Sim (Vision=True)")
    sim = Aligning_Sim(
        seed=args_cli.seed,
        device=args.device,
        render=False,
        n_cores=1,
        n_contexts=30,
        n_trajectories_per_context=1,
        if_vision=True
    )
    
    agent = VisualAgentWrapper(policy, args.device)
    
    # 5. Run Evaluation (or aggregate only)
    if not args_cli.aggregate_only:
        print(f"[ eval ] Starting evaluation for seed {args.seed}...")
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
        # Logic for multi-seed aggregation could be added here to match FMv3ODE fully
