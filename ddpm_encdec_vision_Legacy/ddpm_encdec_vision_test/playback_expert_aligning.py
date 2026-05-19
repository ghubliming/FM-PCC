# Expert Replay Utility — Visual Aligning (Fix 9 Extension)
# ═══════════════════════════════════════════════════════════════════════
# This script replays ground-truth expert demonstrations from the D3IL 
# dataset and records them as MP4 videos. Use this to compare your model's 
# "Desired" trajectory against the true "Expert" behavior.
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import torch
import numpy as np
import pickle
import imageio
import cv2
from tqdm import tqdm

# Ensure local d3il is prioritized
sys.path.insert(0, os.path.abspath('d3il'))
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

from d3il.simulation.aligning_sim import Aligning_Sim
from d3il.simulation.base_sim import sim_framework_path

def playback_expert(n_rollouts=5, output_dir='logs/aligning-d3il-visual/expert_diagnostics'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test context list
    context_list_path = sim_framework_path("environments/dataset/data/aligning/test_contexts.pkl")
    with open(context_list_path, 'rb') as f:
        state_files = pickle.load(f)
    
    # Path to raw state data
    state_data_dir = sim_framework_path("environments/dataset/data/aligning/all_data/state")
    
    # Initialize Sim
    sim = Aligning_Sim(seed=0, device='cpu', render=False, n_cores=1, if_vision=True)
    
    # We need access to the underlying env to set state manually
    from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
    env = Robot_Push_Env(render=False, if_vision=True)
    env.start()
    
    print(f"[ expert ] Replaying {n_rollouts} demonstrations...")
    
    for idx in range(min(n_rollouts, len(state_files))):
        file_name = state_files[idx]
        with open(os.path.join(state_data_dir, file_name), 'rb') as f:
            expert_data = pickle.load(f)
            
        # Extract expert robot positions
        expert_path = expert_data['robot']['des_c_pos'] # [T, 3]
        
        # Extract environment initial state
        # Context is just the box/target positions
        box_pos = expert_data['push-box']['pos'][0]
        box_quat = expert_data['push-box']['quat'][0]
        target_pos = expert_data['target-box']['pos'][0]
        target_quat = expert_data['target-box']['quat'][0]
        
        context = {
            'push-box': {'pos': box_pos, 'quat': box_quat},
            'target-box': {'pos': target_pos, 'quat': target_quat}
        }
        
        # Reset env with expert context
        obs = env.reset(random=False, context=context)
        video_frames = []
        
        print(f"  -> Replaying {file_name} (Length: {len(expert_path)})")
        
        for step in range(len(expert_path)):
            # Expert action is simply the next desired position
            target_pos_step = expert_path[step]
            
            # Step simulator
            # Format: [x, y, z, qx, qy, qz, qw]
            sim_action = np.concatenate((target_pos_step, [0, 1, 0, 0]), axis=0)
            obs, reward, done, info = env.step(sim_action)
            
            # Record frames
            _, bp_image, inhand_image = obs
            bp_vis = cv2.cvtColor(bp_image, cv2.COLOR_BGR2RGB)
            inhand_vis = cv2.cvtColor(inhand_image, cv2.COLOR_BGR2RGB)
            combined = np.concatenate([bp_vis, inhand_vis], axis=1)
            video_frames.append(combined)
            
        # Save video
        save_path = os.path.join(output_dir, f"expert_{idx}.mp4")
        imageio.mimsave(save_path, video_frames, fps=20)
        print(f"  [ DONE ] Saved to {save_path}")

    env.close()

if __name__ == "__main__":
    playback_expert()
