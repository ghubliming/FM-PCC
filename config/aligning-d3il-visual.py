from diffuser.utils import watch
import yaml
import os

# Read the threshold dynamically from the YAML config, abort if not found
with open('config/visual_aligning_eval.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError("CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/visual_aligning_eval.yaml")

_yaml_threshold = _proj_config['diffusion_timestep_threshold']

args_to_watch_dpcc_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
]

args_to_watch_dpcc_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
]

logbase = 'logs'

base = {
    'ddpm_encdec_vision': {
        ## model
        'model': 'ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge',
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'horizon': 8, # D3IL default: 8
        'window_size': 8, # D3IL default: 8
        'obs_dim': 128,
        'action_dim': 3,
        'visual_input': True,
        'obs_seq_len': 5, # D3IL default: 5
        'action_seq_size': 4, # D3IL default: 4
        'n_diffusion_steps': 16, # D3IL default: 16
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 10,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,
        
        # dataset
        'loader': 'ignored',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 512, # D3IL default: 512
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'ddpm_encdec_vision/',
        'exp_name': watch(args_to_watch_dpcc_train),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5, # Original active setting (D3IL baseline trains for epoch: 4)
        'batch_size': 64, # D3IL default: 64
        'learning_rate': 5e-4, # D3IL default: 5e-4
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_ddpm_encdec_vision': {
        'policy': 'sampling.Policy',
        'max_episode_length': 1000, # More than learned steps (512) to allow closed-loop recovery
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,
        # serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}/',
        'exp_name': watch(args_to_watch_dpcc_plan),
        
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'horizon': 8, # D3IL default: 8
        'n_diffusion_steps': 16, # D3IL default: 16
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        'action_weight': 10,
        
        'diffusion_loadpath': 'f:ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
}