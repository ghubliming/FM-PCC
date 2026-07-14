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

args_to_watch_fmv3_imf_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('imf_objective', 'obj'),   # encodes training objective in folder name (fm_equivalent vs meanflow_jvp)
    ('imf_backbone', 'bb'),     # U6: encodes backbone (unet vs dit) so checkpoints never collide
    ('t_schedule', 'ts'),       # U7: time-schedule selector (logit_normal | beta | uniform)
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

        ## U4/U10 — training objective selector (see logs_in_develop/Gen3v4_imf/U4, U10)
        # 'fm_equivalent': legacy finite-diff target = FM baseline arm.
        # 'meanflow_jvp' : ORIGINAL MeanFlow (analytic-v JVP tangent, unguided) — NOT faithful iMF.
        # 'imf_official' : U10 FAITHFUL improved-MeanFlow — predicted-v_c tangent + guided v_g +
        #                  cond_drop + 2×logit-normal + official loss. THE "one last shot".
        #                  See U10/PLAN_faithful_imf_replication.md. Requires dit backbone.
        'imf_objective': 'imf_official',     # was: 'meanflow_jvp' (kept as A/B arm; auto-separates by prefix)
        'meanflow_r_equals_t_frac': 0.25,    # legacy meanflow_jvp only (imf_official uses meanflow_data_proportion)
        'meanflow_adaptive_p': 0.5,          # legacy meanflow_jvp only (imf_official hard-codes official p=1.0)
        'meanflow_adaptive_c': 1e-3,         # legacy meanflow_jvp only (imf_official hard-codes official eps=0.01)
        'meanflow_aux_weight': 0.05,         # legacy meanflow_jvp only (imf_official uses official loss_u+loss_v)
        ## U10 imf_official knobs (faithful iMF)
        'meanflow_cfg_smax': 7.0,            # TRAIN CFG scale ceiling s_max (official default 7.0) — decoupled from eval ω
        'meanflow_data_proportion': 0.5,     # 50% FM anchors (official data_proportion)
        'meanflow_class_dropout_prob': 0.1,  # cond_drop prob — trains the null token (official 0.1)

        ## U5 Phase 1 — real-iMF on UNet (DEFAULT = ON). See Gen3v4_imf/U5.
        ## Safe-core fallback if a run misbehaves: set interval_cfg=False + meanflow_cfg_omega=0.0.
        'dual_head': True,           # was: False  — v-head shares the backbone (official u/v split)
        'interval_cfg': True,        # was: False  — condition backbone on (omega, t_min, t_max)
        ## Fix2 (U6) — CFG is now per-sample randomized at train time (official iMF algorithm),
        ## not a fixed constant. meanflow_cfg_omega is now the sampling ceiling s_max: each sample
        ## draws ω~power-law(0,s_max] and (t_min,t_max)~U(0,0.5)xU(0.5,1) independently every step
        ## (FM-anchor samples get the full [0,1] interval, i.e. no CFG restriction). See
        ## Gen3v4_imf/U6/Fix2_CFG&EMA/CHANGELOG.md.
        'meanflow_cfg_omega': 4.0,   # was: 0.0    — train-time sampling ceiling s_max for ω
        'meanflow_cfg_t_min': 0.4,   # unused when interval_cfg sampling is active (kept for fallback)
        'meanflow_cfg_t_max': 0.6,   # unused when interval_cfg sampling is active (kept for fallback)
        'meanflow_cfg_beta': 1.0,    # power-law shape for ω sampling; 1.0 = official iMF default (log-uniform)

        ## U6 — backbone selector. 'unet' (default) = U5 behaviour, byte-for-byte.
        ## 'dit' = faithful official-iMF transformer (IMFDiTTrajectory). Folder name carries
        ## _bb{imf_backbone}, so unet/dit checkpoints live in separate dirs (no collision).
        ## NOTE: plan block's imf_backbone + dit_* MUST match (state_dict depends on them).
        'imf_backbone': 'dit',       # 'unet' | 'dit' — U10 imf_official REQUIRES dit (unet no-ops cond_drop)
        'dit_depth': 8,              # total transformer blocks (DiT-only)
        'dit_hidden_size': 256,      # token width (DiT-only) — keep small for H=8
        'dit_num_heads': 4,          # attention heads (DiT-only)
        'dit_aux_head_depth': 2,     # private blocks per u/v head (DiT-only)
        'dit_patch_size': 1,         # trajectory steps per token; must divide horizon
        'dit_condition_on_t': False, # official recipe conditions only on h=t−r

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
        ## U9: was missing here — validation ran only via the train script's silent
        ## getattr(..., 0.9) fallback, so the split never appeared in the W&B run config.
        'train_test_split': 0.9,
        
        ## ODE inference — NFE is set in the plan block, not here
        # 'ode_inference_steps_v3': 10,  # dead in training; set flow_steps_v3 in plan block
        ## U7: time-schedule. 'logit_normal' = canonical iMF default (reference imf.py, DEFAULT).
        ## 'beta' = legacy 1-Beta(α,β); set α=β=1 for uniform. NOTE: plan block must match.
        't_schedule': 'logit_normal',     # U7 DEFAULT — was: 'beta' (implicit, pre-U7)
        'p_mean': -0.4,                   # logit-normal P_mean (sigmoid median ≈ 0.40)
        'p_std': 1.0,                     # logit-normal P_std
        ## Beta params kept for backward-compat / ablation ('beta' schedule only — ignored otherwise).
        'time_beta_alpha_v3': 1.0,        # was: 1.5
        'time_beta_beta_v3': 1.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'flow_matching_v3_imeanflow/',
        'exp_name': watch(args_to_watch_fmv3_imf_train),
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
        'prefix': 'f:plans/flow_matching_v3_imeanflow/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}_ts{t_schedule}/',
        'exp_name': watch(args_to_watch_fmv3_ode_plan),

        ## flow matching v3 imeanflow model
        'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
        'horizon': 8,
        'action_weight': 10,
        'u_loss_weight': 1.0,
        'v_loss_weight': 0.1,
        'flow_steps_v3': 2,           # was: 10  — low-NFE real-iMF (use 4 to de-risk first, then 1–2)
        ## U7: MUST match training block (in diffusion_loadpath).
        't_schedule': 'logit_normal',     # U7 DEFAULT — set 'beta' to load pre-U7 checkpoints
        'p_mean': -0.4,
        'p_std': 1.0,
        'time_beta_alpha_v3': 1.0,        # ignored when t_schedule='logit_normal'; kept for beta ablation
        'time_beta_beta_v3': 1.0,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        'diffusion_timestep_threshold': _yaml_threshold,
        'imf_objective': 'imf_official',    # U10 — MUST match training (in diffusion_loadpath)

        ## U5 Phase 1 — must match the trained checkpoint's flags (architecture + CFG)
        'dual_head': True,           # was: False  — MUST equal training (else state_dict mismatch)
        'interval_cfg': True,        # was: False  — MUST equal training
        ## U10 imf_official — CFG is a NET INPUT baked into the trained weights (official iMF), NOT
        ## an output-space mix. At eval, feed a CONSTANT operating point (ω, t_min, t_max); the net
        ## applies the interval internally. ω=1.0 ⇒ guidance OFF (w_arg=1−1/1=0). Set ω∈(1,1+smax]
        ## + an interval to guide. condition_guidance_w stays 0 (returns-CFG path is deleted for
        ## this objective). See U10/PLAN_faithful_imf_replication.md §W8.
        'meanflow_cfg_omega': 1.0,   # EVAL operating point — 1.0 = guidance OFF (was 0.0 legacy-off)
        'meanflow_cfg_t_min': 0.0,   # eval guidance interval (official s-convention), inert while ω=1
        'meanflow_cfg_t_max': 1.0,
        'condition_guidance_w': 0.0, # returns-CFG output-mix OFF (the real neutralizer; sampling knob)
        ## returns_condition is an IDENTITY key (config-override-pkl fix_1 keeps the pkl value to
        ## protect the state_dict). It is fictional/inert for this gen anyway — returns never reach
        ## the backbone, and condition_guidance_w=0 already disables the returns-CFG path. Kept =True
        ## to MATCH the pkl (include_returns=True) so it does not false-warn every eval.
        'returns_condition': True,   # match pkl; inert (neutralized by condition_guidance_w=0)
        ## Train-time-only knobs — MUST equal training so config-overrides-pkl is a no-op (unused at eval)
        'meanflow_cfg_smax': 7.0,
        'meanflow_data_proportion': 0.5,
        'meanflow_class_dropout_prob': 0.1,

        ## EMA switch. Official iMF samples with EMA weights (imeanflow/utils/sample_util.py ema=True).
        ## U10: default True for imf_official (few-step MeanFlow is EMA-sensitive). False = dpcc-legacy raw.
        'eval_use_ema': True,        # was: False — imf-ema (official). Set False for the raw-weights A/B.

        ## U6 — backbone selector. MUST equal the trained checkpoint (state_dict + loadpath).
        'imf_backbone': 'dit',       # 'unet' | 'dit' — U10 imf_official REQUIRES dit (unet no-ops cond_drop)
        'dit_depth': 8,
        'dit_hidden_size': 256,
        'dit_num_heads': 4,
        'dit_aux_head_depth': 2,
        'dit_patch_size': 1,
        'dit_condition_on_t': False,

        ## loading — path must match args_to_watch_fmv3_imf_train exactly (incl. _bb{imf_backbone})
        'diffusion_loadpath': 'f:flow_matching_v3_imeanflow/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}_obj{imf_objective}_bb{imf_backbone}_ts{t_schedule}',
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
}
