from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
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
        'exp_name': watch(args_to_watch),

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

        # v2 SafeFlowMPC-style time sampling parameters (exactly two)
        'time_beta_alpha_v2': 1.5,
        'time_beta_beta_v2': 1.0,

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
        'exp_name': watch(args_to_watch),

        ## diffusion model
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
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
}