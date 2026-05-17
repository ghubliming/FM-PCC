from diffuser.utils import watch
import yaml
import os

# Read the threshold dynamically from the YAML config, abort if not found
with open('config/projection_eval.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError("CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/projection_eval.yaml")

_yaml_threshold = _proj_config['diffusion_timestep_threshold']

#------------------------ base ------------------------#

logbase = 'logs'

base = {
    'ddpm_encdec_vision': {
        'model': 'ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge',
        'horizon': 8, # 'horizon': 10 # ddpm act
        'window_size': 8, # 'window_size': 16 # ddpm act
        'obs_dim': 128,
        'action_dim': 3,
        'visual_input': True,
        'obs_seq_len': 5, # 'obs_seq_len': 1 # ddpm act
        'action_seq_size': 4, # 'action_seq_size': 10 # ddpm act
        'n_diffusion_steps': 16, # 'n_diffusion_steps': 100 # ddpm act
        'loss_type': 'l2',
        
        # dataset
        'loader': 'ignored',
        'max_path_length': 256,

        # serialization
        'logbase': logbase,
        'prefix': 'ddpm_encdec_vision/',
        'exp_name': watch([('prefix', ''), ('horizon', 'H')]),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5, # 'n_train_steps': 5e5 # ddpm act
        'batch_size': 8, # 'batch_size': 64 # ddpm act
        'learning_rate': 1e-4, # 'learning_rate': 5e-4 # ddpm act
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_ddpm_encdec_vision': {
        'policy': 'sampling.Policy',
        'max_episode_length': 400,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,
        
        # serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/ddpm_encdec_vision/',
        'exp_name': watch([('prefix', ''), ('horizon', 'H')]),
        
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'horizon': 8, # 'horizon': 10 # ddpm act
        'n_diffusion_steps': 16, # 'n_diffusion_steps': 100 # ddpm act
        
        'diffusion_loadpath': 'f:ddpm_encdec_vision/H{horizon}',
        'value_loadpath': 'f:values/H{horizon}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
}