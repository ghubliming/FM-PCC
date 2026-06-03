from diffuser.utils import watch
import yaml

# Read the threshold dynamically from the YAML config, abort if not found
with open('config/projection_eval.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)

if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError("CRITICAL: 'diffusion_timestep_threshold' MUST be defined in config/projection_eval.yaml")

_yaml_threshold = _proj_config['diffusion_timestep_threshold']

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion', 'D'),
]

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

args_to_watch_v3 = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('diffusion', 'D'),
]

args_to_watch_fmv3_ode_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
]

args_to_watch_fmv3_ode_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
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

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,      # longest: 106
        'include_returns': True,
        'returns_scale': 400,   # Determined using rewards from the dataset
        'discount': 0.99,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/',
        'exp_name': watch(args_to_watch_dpcc_train),

        ## training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,            # Overwritten
    },

    'flow_matching': {
        # FM version: same as 'diffusion' but uses FM implementation
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.diffusion.GaussianDiffusion',  # Here is full long path, it distinguishes from the diffusion model, name in folder is longer
        'horizon': 8,
        'n_diffusion_steps': 20,
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
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_unet_v2': {
        # FM_Unet_v2: uses Flow_matcher_U_Net_v2 backbone
        # TODO: Update model parameters here when U-Net structure is modified
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
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
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_unet_v2/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_v2': {
        # Flow matcher v2 copied from flow_matching_unet_v2 with SafeFlowMPC-style time sampling
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1, # DPCC is 10
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # v2 SafeFlowMPC-style time sampling parameters (exactly two)
        'time_beta_alpha_v2': 1.5,
        'time_beta_beta_v2': 1.0,

        # v2 ODE/VF decoupling parameters
        'vf_time_bins_v2': 20,
        'ode_inference_steps_v2': 10, # DPCC is 20

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v2/',
        'exp_name': watch(args_to_watch),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_v3': {
        # Flow matcher v3: SafeFlow-style continuous-time query semantics.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        # 'n_diffusion_steps': 20, # this old parameter is not used in v3
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'dynamic_loss': False,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        'test_ret': 0.9,

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # v3 rollout step control.
        'flow_steps_v3': 10,
        # Compatibility alias for existing code paths/tools.
        'ode_inference_steps_v3': 10,

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3/',
        'exp_name': watch(args_to_watch_v3),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_v3_ode_selectable': {
        # Copied-folder FM-v3 variant with config-selectable ODE backend/method.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        # 'n_diffusion_steps': 20, # this old parameter is not used in v3
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        # 'dynamic_loss': False, # DEAD code (legacy DDPM relic, unused in FMv3)
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        # 'test_ret': 0.9, # DEAD code (inference-only parameter)

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # v3 rollout step control.
        # 'flow_steps_v3': 10, # DEAD code (inference-only parameter)
        # 'ode_inference_steps_v3': 10, # Dead code

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_ode_selectable/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_v3_drifting': {
        # Drift-augmented Flow Matcher v3: combines FM ODE with drift loss guidance.
        'model': 'models.Flow_matcher_U_Net_v2',
        'diffusion': 'models.diffusion.FlowMatchingDrifting',
        'horizon': 8,
        'loss_type': 'l2',
        'loss_discount': 1.0,
        'returns_condition': False,
        'action_weight': 1,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'hidden_dim': 256,
        'attention': False,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,

        # v3 SafeFlow-style time sampling parameters.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,

        # FM-D Drift Augmentation Parameters (Locked 3 params)
        'use_drift_augmentation': True,            # bool: enable FM-D mode
        'drift_loss_weight': 0.1,                  # float: lambda in drift field equation
        'drift_loss_type': 'embedding_nn',          # str: "embedding_nn" | "adversarial" | "mmd"

        # dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_drifting/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),

        # training
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'flow_matching_v3_imeanflow': {
        # iMeanFlow: Improved Mean Flows for trajectory generation
        # Dual-velocity: u (mean field) + v (instantaneous deviation)
        # Reuses official iMF repo logic: github.com/Lyy-iiis/imeanflow
        
        ## model & engine (REAL iMF from official repo)
        'model': 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine',
        'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
        'horizon': 8,
        
        ## iMF architecture (matches official repo)
        'freq_dim': 256,
        'depth': 8,
        'num_heads': 4,
        'mlp_dim': 256,
        'time_dim': 256,
        'dropout_rate': 0.1,
        
        ## stable iMF training (FMv3ODE-style main loss + small aux residual)
        'u_loss_weight': 1.0,               # Main flow velocity weight
        'v_loss_weight': 0.1,               # Auxiliary residual weight
        'loss_schedule': 'balanced',        # Keep training stable from step 1
        'warmup_epochs': 0,
        'transition_epochs': 0,
        'loss_type': 'l2',
        'predict_epsilon': True,
        
        ## dataset (inherited from FMv3ODE)
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,
        'use_padding': True,
        'condition_dropout': 0.25,
        'condition_guidance_w': 1.2,
        
        ## training (from FMv3ODE baseline)
        'n_train_steps': 100000,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'gradient_clip': 1.0,
        'ema_decay': 0.995,
        'action_weight': 10,
        'loss_discount': 1.0,              # BUG-02 fix: explicit uniform trajectory weighting
        'gradient_accumulate_every': 2,    # BUG-03 fix: match FMv3ODE effective batch size
        
        ## ODE inference (match FMv3ODE-style deterministic rollout)
        'ode_inference_steps_v3': 10,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        
        ## serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_imeanflow/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),
    },

    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/diffusion/',
        'exp_name': lambda args: f"plans/diffusion/H{args.horizon}_K{args.n_diffusion_steps}_D{args.diffusion}_aw{args.action_weight}/" + watch([
            ('horizon', 'H'),
            ('n_diffusion_steps', 'K'),
            ('diffusion_timestep_threshold', 'T'),
            ('diffusion', 'D')
        ])(args),

        ## diffusion model
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        'action_weight': 10,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },
    
    'plan_fm': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching/',
        'exp_name': watch(args_to_watch),

        ## flow matching model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_unet_v2': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_unet_v2/',
        'exp_name': watch(args_to_watch),

        ## flow matching unet v2 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_unet_v2/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v2': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v2/',
        'exp_name': watch(args_to_watch),

        ## flow matching v2 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'vf_time_bins_v2': 20,
        'ode_inference_steps_v2': 10,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v2/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_v3/',
        'exp_name': watch(args_to_watch_v3),

        ## flow matching v3 model
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'flow_steps_v3': 10,
        'ode_inference_steps_v3': 10,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v3/H{horizon}_K{flow_steps_v3}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_ode_selectable': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_ode_selectable/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch(args_to_watch_fmv3_ode_plan),

        ## flow matching v3 model
        'diffusion': 'models.diffusion.FlowMatchingODE',
        'horizon': 8,
        'action_weight': 1,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        # 'n_diffusion_steps': 20, # DEAD code (mathematically irrelevant for FM flow)
        'flow_steps_v3': 10,
        # 'ode_inference_steps_v3': 10, # DEAD code (compatibility alias for flow_steps_v3)
        # Available backend options: legacy_euler, torchdiffeq.
        'ode_solver_backend_v3': 'legacy_euler',
        # Available method options (torchdiffeq backend):
        # dopri8, dopri5, bosh3, fehlberg2, adaptive_heun,
        # euler, midpoint, heun2, heun3, rk4,
        # explicit_adams, implicit_adams, fixed_adams, scipy_solver.
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'returns_condition': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        # 'predict_epsilon': True, # DEAD code (not used in inference velocity prediction)
        # 'dynamic_loss': False, # DEAD code (legacy DDPM relic, unused in FMv3)

        ## loading
        # 'diffusion_loadpath': 'f:flow_matching_v3_ode_selectable/H{horizon}_K{flow_steps_v3}_D{diffusion}',
        'diffusion_loadpath': 'f:flow_matching_v3_ode_selectable/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',
        # 'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}', # DEAD code (Value functions not used in FMv3 sampling)

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_drifting': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_drifting/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch(args_to_watch_fmv3_ode_plan),

        ## flow matching v3 drifting model
        'diffusion': 'models.diffusion.FlowMatchingDrifting',
        'horizon': 8,
        'action_weight': 1,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        'flow_steps_v3': 10,
        # Available backend options: legacy_euler, torchdiffeq.
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        
        # FM-D Drift Augmentation Parameters (Locked 3 params)
        'use_drift_augmentation': True,            # bool: enable FM-D mode during inference
        'drift_loss_weight': 0.1,                  # float: lambda in drift field equation
        'drift_loss_type': 'embedding_nn',          # str: "embedding_nn" | "adversarial" | "mmd"

        'returns_condition': False,
        'diffusion_timestep_threshold': _yaml_threshold,

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v3_drifting/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    'plan_fm_v3_imeanflow': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/flow_matching_v3_imeanflow/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch(args_to_watch_fmv3_ode_plan),

        ## flow matching v3 imeanflow model
        'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
        'horizon': 8,
        'action_weight': 10,
        'u_loss_weight': 1.0,
        'v_loss_weight': 0.1,
        'flow_steps_v3': 10,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'diffusion_timestep_threshold': _yaml_threshold,   # encodes T in path so threshold sweeps don't overwrite

        ## loading
        'diffusion_loadpath': 'f:flow_matching_v3_imeanflow/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',
        'diffusion_epoch': 'best',
    },

    ## ── Hyperparameter Tuning Blocks ──────────────────────────────────
    ## These use the ORIGINAL flow_matcher model (UNet1DTemporalCondModel).
    ## Duplicate this pair (train + plan) for each tuning experiment.
    ## CRITICAL: Always use a unique 'prefix' to avoid overwriting data.
    ## See: logs_in_develop/guiding_hyperpara_tuning/hyperparameter_tuning_guide.md

    'flow_matching_hp_tune': {
        # HP Tune 1: example tuning run — same model, different hyperparams
        'model': 'models.UNet1DTemporalCondModel',
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
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
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 150,
        'include_returns': True,
        'returns_scale': 400,
        'discount': 0.99,

        # serialization — UNIQUE PREFIX for this tuning run
        'logbase': logbase,
        'prefix': 'flow_matching_hp_tune1/',
        'exp_name': watch(args_to_watch),

        # training — MODIFY THESE for your tuning experiment
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_fm_hp_tune': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,

        ## serialization — MUST match the training prefix
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/flow_matching_hp_tune1/',
        'exp_name': watch(args_to_watch),

        ## flow matching model (same as base flow_matching)
        'diffusion': 'models.diffusion.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading — points to the hp_tune training folder
        'diffusion_loadpath': 'f:flow_matching_hp_tune1/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',

        'diffusion_epoch': 'best',      # 'latest'

        'verbose': False,
        'suffix': '0',
    },

    # =====================================================================
    # Gen9 Epoch 2 — Visual Avoiding (Single Camera, 6-D trajectory)
    # Added 2026-06-03 per logs_in_develop/Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/
    # PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md (audit-corrected per §12).
    # =====================================================================
    #
    # Trajectory layout (2-D analogue of aligning's 9-D):
    #   x[t] = [ dx  dy | des_x  des_y | c_x  c_y ]
    #            act(2)   des_xy(2)     c_xy(2)
    # Single camera: bp-cam only (no inhand-cam — avoiding has no grasping).
    # The 6 fixed obstacles enter as projector `sphere_outside` constraints
    # in the plan_* configs (NOT as obs vector entries — they are env constants).
    # Obstacle positions sourced from d3il/.../objects/avoiding_objects.py:68-82.
    # Obstacle radius = 0.025 m (cylinder geom) + 0.015 m safety margin = 0.04 m.

    'visual_avoiding_dpcc': {
        'model':            'diffuser_visual_avoiding.models.visual_unet.VisualUNet',
        'diffusion':        'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'action_dim':       2,           # 2-D plane velocity [dx, dy]
        'obs_dim':          4,           # [des_xy(2), c_xy(2)] — §12 audit Option C
        'if_vision':        True,
        'horizon':          8,
        'n_diffusion_steps': 100,
        'action_weight':    10,
        'loss_type':        'l2',
        'dim':              32,
        'dim_mults':        (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,
        'max_path_length':  200,         # Avoiding episodes max ~106 steps; 200 is a safe ceiling
        'batch_size':       64,
        'learning_rate':    2e-4,
        'ema_decay':        0.995,
        'n_steps_per_epoch': 1000,
        'n_train_steps':    1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'logbase':          logbase,
        'prefix':           'visual_avoiding_dpcc/',
        'exp_name':         watch(args_to_watch_dpcc_train),
        'device':           'cuda',
        'seed':             0,
    },

    'fm_visual_avoiding': {
        'model':            'fm_visual_avoiding.models.visual_unet.VisualUNet',
        'diffusion':        'fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching',
        'action_dim':       2,
        'obs_dim':          4,
        'if_vision':        True,
        'horizon':          8,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3':  1.0,
        'action_weight':    1,
        'loss_type':        'l2',
        'dim':              32,
        'dim_mults':        (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,
        'max_path_length':  200,
        'batch_size':       64,
        'learning_rate':    2e-4,
        'ema_decay':        0.995,
        'n_steps_per_epoch': 1000,
        'n_train_steps':    1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'logbase':          logbase,
        'prefix':           'fm_visual_avoiding/',
        'exp_name':         watch(args_to_watch_fmv3_ode_train),
        'device':           'cuda',
        'seed':             0,
    },

    # The 6 fixed obstacle positions — sourced from avoiding_objects.py:68-82.
    # Same layout for both DPCC and FM planning configs below.
    # Format expected by ObstacleConstraints: ('sphere_outside', center_xyz, radius).
    # We use z=0 since the avoiding plane is 2-D; only x,y matter for the sphere check.

    'plan_visual_avoiding_dpcc': {
        'horizon':          8,
        'n_diffusion_steps': 100,
        'max_episode_length': 200,
        'max_path_length':  200,
        'action_weight':    10,
        'window_size':      1,
        'obs_seq_len':      1,
        'if_vision':        True,
        'mpc_batch_size':   1,
        'train_batch_size': 64,
        'preprocess_fns':   [],
        'device':           'cuda',
        'seed':             0,
        'loadbase':         None,
        'logbase':          logbase,
        'prefix': (
            'f:plans/visual_avoiding_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}/'
        ),
        'exp_name':         watch(args_to_watch_dpcc_plan),
        'diffusion':        'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        'predict_epsilon':  True,
        'diffusion_timestep_threshold': _yaml_threshold,
        'clip_denoised':    False,
        'diffusion_loadpath': (
            'f:visual_avoiding_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}'
        ),
        'diffusion_epoch':  'best',
        'verbose':          False,
        'suffix':           '0',
        # Projector constraint list — 6 fixed obstacles + workspace bounds.
        # Each tuple is (kind, params...) consumed by Projector.__init__.
        # 'sphere_outside': avoid a ball around (cx, cy, cz=0) with radius R, applied to c_xy slice.
        'constraint_list': [
            ('sphere_outside', [0.500, -0.10, 0.0], 0.04),
            ('sphere_outside', [0.425,  0.08, 0.0], 0.04),
            ('sphere_outside', [0.575,  0.08, 0.0], 0.04),
            ('sphere_outside', [0.350,  0.26, 0.0], 0.04),
            ('sphere_outside', [0.500,  0.26, 0.0], 0.04),
            ('sphere_outside', [0.650,  0.26, 0.0], 0.04),
        ],
    },

    'plan_fm_visual_avoiding': {
        'horizon':          8,
        'flow_steps_v3':    100,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3':  'euler',
        'ode_solver_rtol_v3':    None,
        'ode_solver_atol_v3':    None,
        'ode_solver_step_size_v3': None,
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3':  1.0,
        'max_episode_length': 200,
        'max_path_length':  200,
        'action_weight':    1,
        'window_size':      1,
        'obs_seq_len':      1,
        'if_vision':        True,
        'mpc_batch_size':   4,
        'train_batch_size': 64,
        'preprocess_fns':   [],
        'device':           'cuda',
        'seed':             0,
        'loadbase':         None,
        'logbase':          logbase,
        'prefix': (
            'f:plans/fm_visual_avoiding/'
            'H{horizon}_K{flow_steps_v3}_M{ode_solver_method_v3}_T{diffusion_timestep_threshold}_D{diffusion}/'
        ),
        'exp_name':         watch(args_to_watch_fmv3_ode_plan),
        'diffusion':        'fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching',
        'returns_condition': False,
        'predict_epsilon':  True,
        'diffusion_timestep_threshold': _yaml_threshold,
        'clip_denoised':    False,
        'diffusion_loadpath': (
            'f:fm_visual_avoiding/'
            'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}'
        ),
        'diffusion_epoch':  'best',
        'verbose':          False,
        'suffix':           '0',
        'constraint_list': [
            ('sphere_outside', [0.500, -0.10, 0.0], 0.04),
            ('sphere_outside', [0.425,  0.08, 0.0], 0.04),
            ('sphere_outside', [0.575,  0.08, 0.0], 0.04),
            ('sphere_outside', [0.350,  0.26, 0.0], 0.04),
            ('sphere_outside', [0.500,  0.26, 0.0], 0.04),
            ('sphere_outside', [0.650,  0.26, 0.0], 0.04),
        ],
    },
}