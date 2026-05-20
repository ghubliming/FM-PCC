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
    ('if_vision', 'V'),
    # NOTE: In visual_aligning_dpcc, max_path_length is max_n_episodes (not rollout steps).
    ('max_path_length', 'steps'),
]

args_to_watch_dpcc_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    # NOTE: max_episode_length is forwarded to Robot_Push_Env(max_steps_per_episode=...).
    #       max_path_length is a loadpath key only (checkpoint directory name fragment).
    ('max_episode_length', 'steps'),
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
    'ddpm_encdec_vision': {
        # ======================================================================================
        # 🔑 KEY MODEL BACKBONE PARAMETERS (Active configurations that shape networks)
        # ======================================================================================
        'model': 'ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge', # Bridge container model
        'action_dim': 3,            # Dimension of spatial actions (D3IL default: 3)
        
        # --- Temporal Sequence Dimensions ---
        # NOTE FOR RESEARCHERS:
        # - VAE Transformer Path: These are ACTIVE. Tied to window_size via: 5 + 4 - 1 = 8.
        # - U-Net Path: These are IGNORED during training/planning. U-Net loads contiguous
        #   state & action sequences at the full 'horizon' length (8) from the dataset.
        #   If mismatched during rollout (e.g. state context of 5 frames vs image context of 8),
        #   VisualUNet's internal safety lock (FIX #12) automatically stretches/repeats state history to 8.
        #   For absolute U-Net parity, set obs_seq_len = 8 and action_seq_size = 8.
        'obs_seq_len': 5,           # Length of historical observation frames (D3IL default: 5)
        'action_seq_size': 4,       # Sequential chunk size of decoded action predictions (D3IL default: 4)
        
        # --- Backbone-Specific Sequence Lengths ---
        # 1. U-Net Horizon: Only active for the U-Net Backbone. Must be a multiple of 8 (e.g. 8)
        #    to satisfy the 3-stage downsampling block divisions inside UNet1DTemporalCondModel.
        'horizon': 8,               # Target sequence length. [Active for U-Net / Ignored by VAE Transformer] (D3IL baseline default: N/A)
        
        # 2. VAE Window Size: Only active for VAE Transformer. Mathematically locked to:
        #    window_size = obs_seq_len + action_seq_size - 1 = 8.
        #    If window_size does not match (5 + 4 - 1 = 8), the Transformer positional embedding table
        #    size (seq_size) will mismatch the token sequence length, causing a runtime tensor crash.
        'window_size': 8,           # Context window size. [Active for VAE Transformer / Ignored by U-Net] (D3IL VAE default: 8)
        
        # --- Denoising & Optimization parameters ---
        'n_diffusion_steps': 16,    # Number of denoising timesteps. Must match exactly in training and eval. (D3IL baseline default: 16)
        'action_weight': 10,        # Prioritizes immediate next physical action loss over state prediction. (D3IL baseline default: N/A)
        'loss_type': 'l2',          # L2 (MSE) reconstruction loss type. Must remain 'l2' for DDPM scheduling. (D3IL baseline default: 'l2')
        
        # --- Architectural Channel Widths ---
        # For U-Net: 'dim' controls convolutional block size. Set 'dim: 32' for smaller GPU footprints, 
        #            'dim: 128' or '256' for deep feature extraction.
        # For Transformer: 'embed_dim' controls token size. Set 'hidden_dim: 256' inside 
        #                  the transformer adapter for stable attention learning.
        'dim': 32,                  # Core channel depth of U-Net convolutional layers. (D3IL baseline default: N/A)
        'dim_mults': (1, 2, 4, 8),  # Down/up temporal block multiplier for U-Net. (D3IL baseline default: N/A)
        'hidden_dim': 256,          # Token embedding size for VAE Transformer. (D3IL VAE default: 64)
        
        # --- Training Dropout and Regularization ---
        # - PCC/DPCC CFG: Set to 0.25 (vs D3IL 0.1). 25% dropout trains a strong unconditional prior
        #   needed for stable Classifier-Free Guidance (CFG, w=1.2) to prevent rollout drift.
        #   D3IL uses 0.1 because it bypasses guided planning to maximize conditional exposure.
        'condition_dropout': 0.25,  # Dropout on conditional image embeddings. (D3IL baseline default: 0.1)
        
        
        # ======================================================================================
        # 🪆 DUMMY / UNUSED CONFIGURATION PARAMETERS (Kept for compatibility with legacy parser)
        # ======================================================================================
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'obs_dim': 128,             # Ignored; visual features are encoded to 128D under the hood (D3IL baseline default: 128)
        'visual_input': True,       # Always True for visual encoders (D3IL baseline default: True)
        'loss_discount': 1.0,       # Gaussian diffusion is not temporally discounted (D3IL baseline default: N/A)
        'returns_condition': False, # Must remain False; visual tasks lack return-reward tokens (D3IL baseline default: False)
        'predict_epsilon': True,    # Hardcoded True in the diffusion engine (D3IL baseline default: True)
        'dynamic_loss': False,      # Unused; standard MSE loss operates (D3IL baseline default: False)
        'attention': False,         # Ignored; CNN kernels handle spatial filtering for U-Net (D3IL baseline default: N/A)
        'condition_guidance_w': 1.2,# Guidance scale is evaluated dynamically during rollout (D3IL baseline default: N/A)
        'test_ret': 0.9,            # Unused (returns_condition=False) (D3IL baseline default: N/A)
        
        
        # ======================================================================================
        # 📊 DATASET LOADERS & PATH CONSTRAINTS
        # ======================================================================================
        'max_path_length': 512,     # Max training path length (D3IL default: 512)
        
        # --- Unused Legacy Dataset Keys ---
        'loader': 'ignored',        # Ignored; custom Aligning_Img_Dataset is imported directly
        'normalizer': 'LimitsNormalizer', # Scaler stats are computed and saved dynamically (D3IL baseline default: LimitsNormalizer)
        'preprocess_fns': [],
        'clip_denoised': False,     # The engine explicitly forces clip_denoised=True (D3IL baseline default: True)
        'use_padding': True,        # The dataset loader padding is hardcoded to True
        'include_returns': True,    # Dataset loading handles returns internally
        'returns_scale': 400,       # Unused
        'discount': 0.99,           # Unused (D3IL baseline default: 0.99)
        
        
        # ======================================================================================
        # 💾 SERIALIZATION & EXPERIMENT LOGS
        # ======================================================================================
        'logbase': logbase,
        'prefix': 'ddpm_encdec_vision/',
        'exp_name': watch(args_to_watch_dpcc_train),
        
        
        # ======================================================================================
        # 🏋️‍♂️ ACTIVE TRAINING HYPERPARAMETERS (Tuning bounds for optimal convergence)
        # ======================================================================================
        # Batch Size: Keep at 64. Higher batch sizes risk GPU OOM due to parallel camera ResNet passes.
        'batch_size': 64,           # Training batch size. (D3IL baseline default: 64)
        
        # Learning Rate (lr) Tuning Bounds:
        # - For U-Net: Set lr to 2e-4. Setting above 5e-4 causes convolutional gradient explosion.
        # - For Transformer: Set lr to 5e-4 (D3IL default). Stable attention allows higher base rates.
        'learning_rate': 5e-4,      # Target base learning rate. (D3IL baseline default: 5e-4)
        
        # Parameter Smoothing (EMA):
        # - For U-Net: Highly crucial for physical path continuity. Keep ema_decay at 0.995.
        # - For Transformer VAE: Standard parameters work without EMA, but decay = 0.995 stabilizes.
        'ema_decay': 0.995,         # Exponential Moving Average parameter decay. (D3IL baseline default: 0.995)
        
        # Training Steps:
        'n_steps_per_epoch': 1000,  # Epoch steps limit
        'n_train_steps': 1e5,       # Total training steps (D3IL baseline trains for epoch: 4)
        'gradient_accumulate_every': 2, # Gradient accumulation steps
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_ddpm_encdec_vision': {
        # ======================================================================================
        # 🎮 INFERENCE PLANNING AND MULTI-THREAD SIMULATOR CONSTRAINTS
        # ======================================================================================
        'horizon': 8,               # Target planning horizon (D3IL default: 8)
        'window_size': 8,           # Explicitly define for evaluation rollout buffer (D3IL default: 8)
        'n_diffusion_steps': 16,    # Denoising inference steps (D3IL default: 16)
        'max_episode_length': 1000, # Allowed rollout steps (more than learned steps to allow recovery)
        'max_path_length': 512,     # Physical rollout execution threshold (D3IL default: 512)
        'action_weight': 10,
        
        'policy': 'sampling.Policy',
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
        'exp_name': watch(args_to_watch_dpcc_plan),
        
        'diffusion': 'ddpm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        
        'diffusion_loadpath': 'f:ddpm_encdec_vision/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}',
        'value_loadpath': 'f:values/H{horizon}_K{n_diffusion_steps}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },

    'visual_aligning_dpcc': {
        # ======================================================================================
        # 🔑 KEY MODEL BACKBONE PARAMETERS
        # Gen6V4 — Visual-DPCC: 9D trajectory [act(3) | des_c_pos(3) | c_pos(3)]
        # DPCC SLSQP projector enforces workspace bounds on c_pos (indices 6-8).
        # ======================================================================================
        'model': 'diffuser_visual_aligning.models.visual_unet.VisualUNet',
        'diffusion': 'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'action_dim': 3,            # 3D velocity: [dx, dy, dz]
        'obs_dim': 6,               # 6D obs: [des_c_pos(3), c_pos(3)] — MUST be 6, never 3 or 128
        'if_vision': True,
        'horizon': 8,               # Must be divisible by 8 for U-Net stride-2 downsampling (padded internally)
        # n_diffusion_steps = denoising chain length T (d3il vision baseline: 4; encdec baseline: 16; state-only: 50).
        # 100 gives high sample quality but 25× inference cost vs d3il vision default.
        # MUST stay identical in visual_aligning_dpcc and plan_visual_aligning_dpcc —
        # it is embedded in the checkpoint directory name (K{n_diffusion_steps}) via args_to_watch_dpcc_train.
        # If you retrain with a different K, update both blocks together.
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_type': 'l2',
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,

        # ======================================================================================
        # 📊 DATASET
        # ParityAligningDataset loads 9D trajectories from raw pkl files.
        # max_path_length serves TWO roles here (not a per-trajectory step cap):
        #   1. Passed as max_n_episodes to ParityAligningDataset — a soft ceiling.
        #      The aligning dataset has 900 episodes, so min(900, 1000)=900: all episodes
        #      load and the cap is never hit. Raise if you add more demos; lower to subsample.
        #   2. Embedded in the checkpoint directory name via args_to_watch_dpcc_train
        #      (as 'steps1000'). plan_visual_aligning_dpcc.max_path_length MUST match
        #      exactly, or diffusion_loadpath resolves to a non-existent directory.
        # ======================================================================================
        'max_path_length': 1000,

        # ======================================================================================
        # 💾 SERIALIZATION & EXPERIMENT LOGS
        # ======================================================================================
        'logbase': logbase,
        'prefix': 'visual_aligning_dpcc/',
        'exp_name': watch(args_to_watch_dpcc_train),

        # ======================================================================================
        # 🏋️‍♂️ TRAINING HYPERPARAMETERS
        # ======================================================================================
        'batch_size': 32,             # d3il vision baseline: 64
        'learning_rate': 2e-4,
        'ema_decay': 0.995,
        'n_steps_per_epoch': 1000,
        # d3il trains for epoch=4 (epoch-based). We use steps-based training.
        # 5e5 steps @ batch=32 / gradient_accumulate=2 ≈ effective 333 optimizer steps/epoch-equivalent.
        'n_train_steps': 5e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_visual_aligning_dpcc': {
        # ======================================================================================
        # 🎮 INFERENCE PLANNING AND MULTI-THREAD SIMULATOR CONSTRAINTS
        # ======================================================================================
        'horizon': 8,
        # MUST equal visual_aligning_dpcc.n_diffusion_steps — both are embedded as K{n_diffusion_steps} in the
        # checkpoint directory name. A mismatch here produces a FileNotFoundError (wrong K in loadpath).
        'n_diffusion_steps': 100,
        # D3IL Robot_Push_Env default is 400 (hardcoded, proven stable for the aligning task).
        # Fix 10 wired this field so it now actually reaches the env. Start at 400 (proven baseline).
        # Increase only after confirming the model benefits from a longer rollout budget.
        'max_episode_length': 400,
        'max_path_length': 1000,   # MUST match visual_aligning_dpcc.max_path_length (fix_1.3): loadpath key only
        'action_weight': 10,
        # window_size=1 / obs_seq_len=1 must match training: ParityAligningDataset
        # provides single-frame images per sample, so the model is trained on T_win=1.
        # Using window_size>1 at eval would mean-pool multiple frames and shift the
        # FiLM conditioning distribution away from what the model learned.
        'window_size': 1,
        'obs_seq_len': 1,
        'if_vision': True,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'loadbase': None,
        'logbase': logbase,
        'prefix': (
            'f:plans/visual_aligning_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}/'
        ),
        'exp_name': watch(args_to_watch_dpcc_plan),
        'diffusion': 'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        'predict_epsilon': True,
        'diffusion_timestep_threshold': _yaml_threshold,
        'diffusion_loadpath': (
            'f:visual_aligning_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}'
        ),
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },

    'fm_encdec_vision': {
        # ======================================================================================
        # 🔑 KEY MODEL BACKBONE PARAMETERS (Active configurations that shape networks)
        # ======================================================================================
        'model': 'fm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge', # Bridge container model
        'action_dim': 3,            # Dimension of spatial actions (D3IL default: 3)
        
        # --- Temporal Sequence Dimensions ---
        # NOTE FOR RESEARCHERS:
        # - VAE Transformer Path: These are ACTIVE. Tied to window_size via: 5 + 4 - 1 = 8.
        # - U-Net Path: These are IGNORED during training/planning. U-Net loads contiguous
        #   state & action sequences at the full 'horizon' length (8) from the dataset.
        #   If mismatched during rollout (e.g. state context of 5 frames vs image context of 8),
        #   VisualUNet's internal safety lock (FIX #12) automatically stretches/repeats state history to 8.
        #   For absolute U-Net parity, set obs_seq_len = 8 and action_seq_size = 8.
        'obs_seq_len': 5,           # Length of historical observation frames (D3IL default: 5)
        'action_seq_size': 4,       # Sequential chunk size of decoded action predictions (D3IL default: 4)
        
        # --- Backbone-Specific Sequence Lengths ---
        # 1. U-Net Horizon: Only active for the U-Net Backbone. Must be a multiple of 8 (e.g. 8)
        #    to satisfy the 3-stage downsampling block divisions inside UNet1DTemporalCondModel.
        'horizon': 8,               # Target sequence length. [Active for U-Net / Ignored by VAE Transformer] (D3IL baseline default: N/A)
        
        # 2. VAE Window Size: Only active for VAE Transformer. Mathematically locked to:
        #    window_size = obs_seq_len + action_seq_size - 1 = 8.
        #    If window_size does not match (5 + 4 - 1 = 8), the Transformer positional embedding table
        #    size (seq_size) will mismatch the token sequence length, causing a runtime tensor crash.
        'window_size': 8,           # Context window size. [Active for VAE Transformer / Ignored by U-Net] (D3IL VAE default: 8)
        
        # --- Continuous Time Flow Matching (FMv3ODE) Parameters ---
        # Note: ODE integration steps and solver options are planning-only parameters and are omitted here.
        'time_beta_alpha_v3': 1.5,  # Alpha parameter for continuous-time Beta distribution sampling (Gen7 default: 1.5)
        'time_beta_beta_v3': 1.0,   # Beta parameter for continuous-time Beta distribution sampling (Gen7 default: 1.0)
        
        # --- Denoising & Optimization parameters ---
        'n_diffusion_steps': 16,    # Number of legacy timesteps. Must match exactly in training and eval. (D3IL baseline default: 16)
        'action_weight': 10,        # Prioritizes immediate next physical action loss over state prediction. (D3IL baseline default: N/A)
        'loss_type': 'l2',          # L2 (MSE) reconstruction loss type. Must remain 'l2' for continuous-time target velocity modeling.
        
        # --- Architectural Channel Widths ---
        # For U-Net: 'dim' controls convolutional block size. Set 'dim: 32' for smaller GPU footprints, 
        #            'dim: 128' or '256' for deep feature extraction.
        # For Transformer: 'embed_dim' controls token size. Set 'hidden_dim: 256' inside 
        #                  the transformer adapter for stable attention learning.
        'dim': 32,                  # Core channel depth of U-Net convolutional layers. (D3IL baseline default: N/A)
        'dim_mults': (1, 2, 4, 8),  # Down/up temporal block multiplier for U-Net. (D3IL baseline default: N/A)
        'hidden_dim': 256,          # Token embedding size for VAE Transformer. (D3IL VAE default: 64)
        
        # --- Training Dropout and Regularization ---
        # - PCC/DPCC CFG: Set to 0.25 (vs D3IL 0.1). 25% dropout trains a strong unconditional prior
        #   needed for stable Classifier-Free Guidance (CFG, w=1.2) to prevent rollout drift.
        #   D3IL uses 0.1 because it bypasses guided planning to maximize conditional exposure.
        'condition_dropout': 0.25,  # Dropout on conditional image embeddings. (D3IL baseline default: 0.1)
        
        # ======================================================================================
        # 🪆 DUMMY / UNUSED CONFIGURATION PARAMETERS (Kept for compatibility with legacy parser)
        # ======================================================================================
        'diffusion': 'fm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'obs_dim': 128,             # Ignored; visual features are encoded to 128D under the hood (D3IL baseline default: 128)
        'visual_input': True,       # Always True for visual encoders (D3IL baseline default: True)
        'loss_discount': 1.0,       # Gaussian diffusion is not temporally discounted (D3IL baseline default: N/A)
        'returns_condition': False, # Must remain False; visual tasks lack return-reward tokens (D3IL baseline default: False)
        'predict_epsilon': True,    # Hardcoded True in the diffusion engine (D3IL baseline default: True)
        'dynamic_loss': False,      # Unused; standard MSE loss operates (D3IL baseline default: False)
        'attention': False,         # Ignored; CNN kernels handle spatial filtering for U-Net (D3IL baseline default: N/A)
        'condition_guidance_w': 1.2,# Guidance scale is evaluated dynamically during rollout (D3IL baseline default: N/A)
        'test_ret': 0.9,            # Unused (returns_condition=False) (D3IL baseline default: N/A)
        
        # ======================================================================================
        # 📊 DATASET LOADERS & PATH CONSTRAINTS
        # ======================================================================================
        'max_path_length': 512,     # Max training path length (D3IL default: 512)
        
        # --- Unused Legacy Dataset Keys ---
        'loader': 'ignored',        # Ignored; custom Aligning_Img_Dataset is imported directly
        'normalizer': 'LimitsNormalizer', # Scaler stats are computed and saved dynamically (D3IL baseline default: LimitsNormalizer)
        'preprocess_fns': [],
        'clip_denoised': False,     # The engine explicitly forces clip_denoised=True (D3IL baseline default: True)
        'use_padding': True,        # The dataset loader padding is hardcoded to True
        'include_returns': True,    # Dataset loading handles returns internally
        'returns_scale': 400,       # Unused
        'discount': 0.99,           # Unused (D3IL baseline default: 0.99)
        
        # ======================================================================================
        # 💾 SERIALIZATION & EXPERIMENT LOGS
        # ======================================================================================
        'logbase': logbase,
        'prefix': 'fm_encdec_vision/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),
        
        # ======================================================================================
        # 🏋️‍♂️ ACTIVE TRAINING HYPERPARAMETERS (Tuning bounds for optimal convergence)
        # ======================================================================================
        # Batch Size: Keep at 64. Higher batch sizes risk GPU OOM due to parallel camera ResNet passes.
        'batch_size': 64,           # Training batch size. (D3IL baseline default: 64)
        
        # Learning Rate (lr) Tuning Bounds:
        # - For U-Net: Set lr to 2e-4. Setting above 5e-4 causes convolutional gradient explosion.
        # - For Transformer: Set lr to 5e-4 (D3IL default). Stable attention allows higher base rates.
        'learning_rate': 2e-4,      # Target base learning rate. (D3IL baseline default: 5e-4)
        
        # Parameter Smoothing (EMA):
        # - For U-Net: Highly crucial for physical path continuity. Keep ema_decay at 0.995.
        # - For Transformer VAE: Standard parameters work without EMA, but decay = 0.995 stabilizes.
        'ema_decay': 0.995,         # Exponential Moving Average parameter decay. (D3IL baseline default: 0.995)
        
        # Training Steps:
        'n_steps_per_epoch': 1000,  # Epoch steps limit
        'n_train_steps': 1e5,       # Total training steps (D3IL baseline trains for epoch: 4)
        'gradient_accumulate_every': 2, # Gradient accumulation steps
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_fm_encdec_vision': {
        # ======================================================================================
        # 🎮 INFERENCE PLANNING AND MULTI-THREAD SIMULATOR CONSTRAINTS
        # ======================================================================================
        'horizon': 8,               # Target planning horizon (D3IL default: 8)
        'window_size': 8,           # Explicitly define for evaluation rollout buffer (D3IL default: 8)
        'max_episode_length': 1000, # Allowed rollout steps (more than learned steps to allow recovery)
        'max_path_length': 512,     # Physical rollout execution threshold (D3IL default: 512)
        'action_weight': 10,
        
        # --- Continuous Time Flow Matching Parameters ---
        'time_beta_alpha_v3': 1.5,  # Alpha parameter for Beta prior continuous-time integration
        'time_beta_beta_v3': 1.0,   # Beta parameter for Beta prior continuous-time integration
        'flow_steps_v3': 16,        # Continuous time ODE solver path steps (Gen7 default: 16)
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
        
        'policy': 'sampling.Policy',
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'test_ret': 0,
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'f:plans/fm_encdec_vision/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
        'exp_name': watch(args_to_watch_fmv3_ode_plan),
        
        'diffusion': 'fm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        'diffusion_timestep_threshold': _yaml_threshold,
        
        'diffusion_loadpath': 'f:fm_encdec_vision/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',
        'value_loadpath': 'f:values/H{horizon}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
}

# ─── Gen6 State-Only Non-Visual Configuration Appends ────────────────────────
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 2,
    'obs_dim': 20,
    'if_vision': False,
    'prefix': 'ddpm_encdec_vision_nonvisual/',
}

base['plan_ddpm_encdec_vision_nonvisual'] = {
    **base['plan_ddpm_encdec_vision'],
    'prefix': 'f:plans/ddpm_encdec_vision_nonvisual/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
    'diffusion_loadpath': 'f:ddpm_encdec_vision_nonvisual/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}',
}
