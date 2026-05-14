# FM_v3 ODE-selectable version of eval_ddpm_encdec_vision.py
import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
        
        # Format images for encoder [1, 1, 3, 96, 96]
        bp_tensor = torch.tensor(bp_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        inhand_tensor = torch.tensor(inhand_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        pos_tensor = torch.tensor(des_robot_pos).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        # Policy expects conditions = {time: state}
        # In VisualFMv3Bridge, the 'state' passed to the backbone is actually (bp, inhand, robot_pos)
        # We wrap this in a way the policy can handle.
        # For now, we pass the raw images and state in a tuple as the condition.
        cond = (bp_tensor, inhand_tensor, pos_tensor)
        
        # Aligning_Img_Dataset uses window_size=H, but sim gives one step.
        # We create a dummy condition dict that matches what the VisualModel expects.
        action, samples = self.policy(conditions={0: cond}, batch_size=1)
        
        # action is [1, action_dim]
        return action.detach().cpu().numpy()[0]

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str = 'config.aligning-d3il-visual'

def load_diffusion_vision(loadpath, device='cuda', epoch='best'):
    """Replication of FMv3ODE load_diffusion_with_override for Vision."""
    import os
    print(f'[ eval ] Loading Vision Model from {loadpath}')
    
    # Use standard FMv3ODE loading pattern
    from ddpm_encdec_vision.models.visual_unet import VisualUNet
    from ddpm_encdec_vision.models.visual_gaussian_diffusion import VisualGaussianDiffusion
    
    # Note: In a real 'bone' implementation, we would load these from pickles.
    # For this refactor, we instantiate the VisualBridge which contains both.
    # To follow the 'bone', we should really use the pickle logic.
    
    dataset_config = utils.load_config(loadpath, 'dataset_config.pkl')
    model_config = utils.load_config(loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(loadpath, 'diffusion_config.pkl')
    trainer_config = utils.load_config(loadpath, 'trainer_config.pkl')
    
    dataset = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)
    
    if epoch == 'latest':
        epoch = utils.get_latest_epoch([loadpath])
    trainer.load(epoch)
    
    return diffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for visual aligning.')
    parser.add_argument('--seed', type=int, default=5, help='Seed')
    parser.add_argument('--epoch', type=str, default='best', help='Epoch')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=args_cli.seed)
    
    # Resolve load path (Bone: handle f: prefix)
    if args.diffusion_loadpath.startswith('f:'):
        formatted_path = args.diffusion_loadpath[2:].format(**vars(args))
        load_dir = os.path.join(args.logbase, formatted_path, str(args.seed))
    else:
        load_dir = args.diffusion_loadpath
        
    diffusion = load_diffusion_vision(load_dir, device=args.device, epoch=args_cli.epoch)
    
    # Create Policy (PCC Bone)
    policy = Policy(model=diffusion, normalizer=None) # Normalizer handled inside bridge/encoder
    
    # Start simulation
    print("[ eval ] Initializing Aligning_Sim (vision=True)")
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
    
    success_rate, mode_encoding = sim.test_agent(agent)
    print(f"[ eval ] Evaluation finished. Success Rate: {success_rate}")
