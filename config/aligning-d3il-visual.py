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
    ('max_path_length', 'steps'),
]

args_to_watch_dpcc_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('max_episode_length', 'steps'),
]

logbase = 'logs'

base = {
    'ddpm_encdec_vision': {
        ## model
        'model': 'ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge',
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion', # DUMMY: Imported directly in train script.
        'horizon': 8, # KEY: Must be a multiple of 8 (e.g. 8). Controls the UNet temporal padding block size. (D3IL default: 8)
        'window_size': 8, # KEY: Must match horizon. Sets VAE temporal context boundaries. (D3IL default: 8)
        'obs_dim': 128, # DUMMY: Ignored; visual features are encoded to 128 under the hood.
        'action_dim': 3,
        'visual_input': True, # DUMMY: Always True for the visual training backbone.
        'obs_seq_len': 5, # KEY: Length of sequential observation history frames (D3IL default: 5).
        'action_seq_size': 4, # KEY: Sequential chunk size of decoded action predictions (D3IL default: 4).
        'n_diffusion_steps': 16, # DANGER: If changed, training and eval step counts must match. Mismatches lead to out-of-distribution noise scaling and frozen robot rollouts. (D3IL default: 16)
        'loss_type': 'l2', # KEY: Reconstruction loss type. Must remain 'l2' (MSE) or 'l1' (MAE) to avoid training crashes.
        'loss_discount': 1.0, # DUMMY: Gaussian diffusion loss is not discounted temporally.
        'returns_condition': False, # DANGER: Must remain False! Setting to True expects external reward conditioning which visual tasks lack, leading to runtime failures.
        'action_weight': 10, # KEY: Prioritizes immediate next physical action loss over state prediction loss (default: 10).
        'dim': 32, # KEY: Sets core channel depth of UNet layers. Must match pre-trained checkpoint configuration.
        'dim_mults': (1, 2, 4, 8), # KEY: Downsampling/upsampling block multiplier. Must match pre-trained checkpoint configuration.
        'predict_epsilon': True, # DUMMY: Hardcoded True in VisualGaussianDiffusion.
        'dynamic_loss': False, # DUMMY: Unused; L2 loss function is used instead.
        'hidden_dim': 256, # DUMMY: CNN backbone (UNet) used instead of MLP.
        'attention': False, # DUMMY: CNN kernels handle spatial filtering instead.
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2, # DUMMY: Guidance scale is evaluated dynamically.
        'test_ret': 0.9, # DUMMY: Returns conditioning is disabled (returns_condition=False).
        
        # dataset
        'loader': 'ignored', # DUMMY: Ignored; custom Aligning_Img_Dataset is imported directly.
        'normalizer': 'LimitsNormalizer', # DUMMY: Scaler stats (scaler.pkl) computed and saved dynamically instead.
        'preprocess_fns': [],
        'clip_denoised': False, # DUMMY: The engine explicitly forces clip_denoised=True.
        'use_padding': True, # DUMMY: The dataset loader padding is hardcoded to True.
        'max_path_length': 512, # KEY: Max training sequence trajectory step length (D3IL default: 512).
        'include_returns': True, # DUMMY: Dataset loading handles returns internally.
        'returns_scale': 400, # DUMMY: Unused.
        'discount': 0.99, # DUMMY: Unused.
 
        # serialization
        'logbase': logbase,
        'prefix': 'ddpm_encdec_vision/',
        'exp_name': watch(args_to_watch_dpcc_train),
 
        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5, # Original active setting (D3IL baseline trains for epoch: 4)
        'batch_size': 64, # DANGER: Exceeding 64 runs extremely high risk of GPU OOM memory limits due to parallel camera ResNet passes. (D3IL default: 64)
        'learning_rate': 5e-4, # DANGER: Tuned at 5e-4. Setting above 1e-3 causes ResNet gradient explosion and NaN loss generation. (D3IL default: 5e-4)
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_ddpm_encdec_vision': {
        'policy': 'sampling.Policy',
        'max_episode_length': 1000, # More than learned steps (512) to allow closed-loop recovery
        'max_path_length': 512, # D3IL default: 512
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,
        # serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
        'exp_name': watch(args_to_watch_dpcc_plan),
        
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'horizon': 8, # D3IL default: 8
        'n_diffusion_steps': 16, # D3IL default: 16
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        'action_weight': 10,
        
        'diffusion_loadpath': 'f:ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
}