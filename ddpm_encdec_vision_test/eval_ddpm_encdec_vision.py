import os
import sys
import torch
import numpy as np
import argparse

import ddpm_encdec_vision.utils as utils
from ddpm_encdec_vision.models.d3il_visual_bridge import VisualDiffusionBridge

sys.path.append(os.path.abspath('d3il'))
sys.path.append(os.path.abspath('d3il/environments/d3il'))
from d3il.simulation.aligning_sim import Aligning_Sim

class VisualAgentWrapper:
    def __init__(self, bridge, device):
        self.bridge = bridge
        self.device = device
        self.bridge.eval()
    
    def reset(self):
        pass
        
    def predict(self, obs, if_vision=True):
        bp_image, inhand_image, des_robot_pos = obs
        
        # Aligning_Sim provides:
        # bp_image: (3, 96, 96) ndarray in [0, 1]
        # inhand_image: (3, 96, 96) ndarray in [0, 1]
        # des_robot_pos: (3,) ndarray
        bp_tensor = torch.tensor(bp_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        inhand_tensor = torch.tensor(inhand_image).unsqueeze(0).unsqueeze(0).to(self.device).float()
        pos_tensor = torch.tensor(des_robot_pos).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            visual_emb = self.bridge.encode_visual(bp_tensor, inhand_tensor, state=pos_tensor)
            pred = self.bridge.predict(visual_emb)
        
        # pred could be [1, action_seq_size, 3], but Aligning_Sim applies pred_action[0]
        # which expects the sequence or just the next action.
        return pred.detach().cpu().numpy()[0]

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str = 'config.aligning-d3il-visual'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for visual aligning.')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    args = Parser().parse_args(experiment='plan_ddpm_encdec_vision', seed=args_cli.seed)
    
    device = args.device
    bridge = VisualDiffusionBridge(args).to(device)
    
    # Resolve load path
    if args.diffusion_loadpath.startswith('f:'):
        # Format string resolution (FM-PCC convention)
        path = args.diffusion_loadpath[2:]
        formatted_path = path.format(**vars(args))
        load_dir = os.path.join(args.logbase, formatted_path)
    else:
        load_dir = args.diffusion_loadpath
        
    checkpoint_path = os.path.join(load_dir, f'state_{args.diffusion_epoch}.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"[ eval ] Loading {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=device)
        bridge.load_state_dict(data['model'])
    else:
        print(f"[ eval ] WARNING: No checkpoint found at {checkpoint_path}")

    # Start simulation
    print("[ eval ] Initializing Aligning_Sim (vision=True)")
    sim = Aligning_Sim(
        seed=args.seed,
        device=device,
        render=False,
        n_cores=1,  # Keep it simple for now
        n_contexts=30,
        n_trajectories_per_context=1,
        if_vision=True
    )
    
    agent = VisualAgentWrapper(bridge, device)
    
    success_rate, mode_encoding = sim.test_agent(agent)
    print(f"[ eval ] Evaluation finished. Success Rate: {success_rate}")
