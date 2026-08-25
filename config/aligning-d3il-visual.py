from diffuser.utils import watch
import yaml
import os
import sys                     # Gen14 U9 — stderr notes from _mix_u9_keys()

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
    ('batch_size', 'bs'),           # training mini-batch size (bs32/bs64) — distinct from MPC candidate pool
    # FiLM_V2: emits 'filmv1'/'filmv2' so v1 and v2 checkpoints save in PARALLEL dirs.
    # watch() skips this for blocks that don't define film_mode (e.g. ddpm_encdec_vision).
    ('film_mode', 'film'),
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
    ('mpc_batch_size', 'mpc'),          # MPC candidate pool size in plan folder name (mpc4)
    # FiLM_V2: separates v1/v2 EVAL result folders.
    ('film_mode', 'film'),
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

# Gen7 Visual-FM (fm_visual_aligning) — FM-style naming (Beta params, not DDPM K)
args_to_watch_fm_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),     # Beta distribution α — FM training dist identity key
    ('time_beta_beta_v3', 'b'),      # Beta distribution β — FM training dist identity key
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),            # training mini-batch size (bs32/bs64) — distinct from MPC candidate pool
    # FiLM_V2: emits 'filmv1'/'filmv2' so v1 and v2 checkpoints save in PARALLEL dirs.
    ('film_mode', 'film'),
]

args_to_watch_fm_visual_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),   # identifies solver variant (euler/rk4/dopri5)
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('mpc_batch_size', 'mpc'),       # MPC candidate pool size in plan folder name (mpc4)
    # FiLM_V2: separates v1/v2 EVAL result folders.
    ('film_mode', 'film'),
]

args_to_watch_imf_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),
    ('imf_objective', 'obj'),   # encodes training objective in folder name
    ('t_schedule', 'ts'),       # U7: time-schedule selector (logit_normal | beta | uniform)
]

logbase = 'logs'

# ── CUSTOM RUN MESSAGE (2026-08-13) — opt-in results-path tag, EVAL ONLY ──────────────────
# Mirror of config/avoiding-d3il.py (see the long note there). Lets a re-run of the SAME config
# at a different budget write to its OWN results folder instead of overwriting the old numbers.
# Empty (the default) => paths are byte-identical to before.
#   FMPCC_RUN_MSG=20trials ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/<eval>.sh
# ⚠️ Visual-aligning has no n_trials — it rolls out `n_contexts` contexts
#    (config/visual_aligning_eval.yaml). "More trials" here means raising n_contexts; the tag
#    itself is budget-agnostic and works the same either way.
# 🔴 PLAN BLOCKS ONLY — never let this reach a training exp_name (it would move every
#    checkpoint folder and break every diffusion_loadpath).
def _sanitize_msg(text):
    """Filesystem-safe, stable token: keep [A-Za-z0-9._-], collapse everything else to '-'."""
    raw = str(text if text is not None else '').strip()
    if not raw:
        return ''
    out = ''.join(ch if (ch.isalnum() or ch in '._-') else '-' for ch in raw)
    while '--' in out:
        out = out.replace('--', '-')
    return out.strip('-._')[:40]


custom_msg = _sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
if custom_msg:
    print(f'[ config/aligning-d3il-visual ] custom_msg="{custom_msg}" -> results dirs end in "_msg{custom_msg}"')


def _msg_suffix(args):
    """'_msg<token>' for a plan block that carries a non-empty custom_msg, else ''."""
    msg = _sanitize_msg(getattr(args, 'custom_msg', ''))
    return f'_msg{msg}' if msg else ''


def watch_plan(args_to_watch_list):
    """watch(), plus the plan block's custom_msg suffix. Use in PLAN blocks ONLY."""
    _fn = watch(args_to_watch_list)
    return lambda args: _fn(args) + _msg_suffix(args)


base = {

    # ══════════════════════════════════════════════════════════════════════════════
    # TRAINING CONFIGS
    # ══════════════════════════════════════════════════════════════════════════════

    # ─── 1. DDPM ENCDEC VISION (Training) ───
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

    # ─── 2. FM ENCDEC VISION (Training) ───
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

    # ─── 3. DIFFUSER VISUAL ALIGNING (Training) ───
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

        # FiLM backbone selector (FiLM_V2 upgrade, 2026-06-27). OPTIONAL knob.
        #   'v1' (default) → UNet1DTemporalCondModel (current Fake-FiLM additive-bias).
        #   'v2'           → UNet1DTemporalFiLMModel (True FiLM: per-block γ scale + β shift).
        # film_mode is now a PATH key (via args_to_watch_dpcc_train → '..._filmv1'/'..._filmv2'),
        # so v1 and v2 checkpoints/results save in PARALLEL dirs automatically — just flip this
        # value, no prefix edit needed. v2 is OPT-IN and requires fresh training.
        # ⚠ Existing pre-2026-06-27 checkpoints were saved WITHOUT the '_filmv1' suffix; to eval
        #   them either rename their folder to add '_filmv1' or retrain.
        'film_mode': 'v1',

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
        'batch_size': 64,             # d3il vision baseline: 64
        'learning_rate': 2e-4,
        'ema_decay': 0.995,
        'n_steps_per_epoch': 1000,
        # d3il trains for epoch=4 (epoch-based). We use steps-based training.
        # 5e5 steps @ batch=32 / gradient_accumulate=2 ≈ effective 333 optimizer steps/epoch-equivalent.
        'n_train_steps': 1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    # ─── 4. FM VISUAL ALIGNING (Training) ───
    'fm_visual_aligning': {
        # ======================================================================================
        # 🔑 KEY MODEL BACKBONE PARAMETERS
        # Gen7 — Visual-FM: 9D trajectory [act(3) | des_c_pos(3) | c_pos(3)]
        # FM ODE engine: continuous-time Beta sampling, deterministic Euler forward ODE 0→1.
        # DPCC SLSQP projector enforces workspace bounds on c_pos (indices 6-8).
        # ======================================================================================
        'model': 'fm_visual_aligning.models.visual_unet.VisualUNet',
        'diffusion': 'fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching',
        'action_dim': 3,            # 3D velocity: [dx, dy, dz]
        'obs_dim': 6,               # 6D obs: [des_c_pos(3), c_pos(3)] — MUST be 6, never 3 or 128
        'if_vision': True,
        'horizon': 8,               # Must be divisible by 8 for U-Net stride-2 downsampling (padded internally)
        # FM uses continuous time — there is no discrete denoising chain.
        # n_diffusion_steps is DEAD here; kept only as a fallback buffer-size hint in the
        # GaussianDiffusion constructor (does not affect training dynamics or checkpoint naming).
        # 'n_diffusion_steps': 100, # DEAD (FM continuous time — see above)
        #
        # time_beta_alpha_v3 / time_beta_beta_v3: Beta(α, β) distribution for sampling
        # training timesteps t ~ Beta(α, β), then inverted to t = 1 - t to concentrate
        # samples near t=1 (data side) where velocity learning is most informative.
        # These ARE the training identity keys — embedded in the checkpoint directory name
        # as 'a{alpha}_b{beta}' via args_to_watch_fm_visual_train. If you change either
        # value, update the matching plan_fm_visual_aligning block and retrain from scratch.
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        # flow_steps_v3 / ode_solver_* are INFERENCE-ONLY — they live in plan_fm_visual_aligning.
        # 'flow_steps_v3': 100,             # DEAD in training
        # 'ode_solver_backend_v3': ...,     # DEAD in training
        # 'ode_solver_method_v3': ...,      # DEAD in training
        'action_weight': 1,
        'loss_type': 'l2',
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'condition_dropout': 0.1,
        'returns_condition': False,

        # FiLM backbone selector (FiLM_V2 upgrade, 2026-06-27). OPTIONAL knob.
        #   'v1' (default) → UNet1DTemporalCondModel (current Fake-FiLM additive-bias).
        #   'v2'           → UNet1DTemporalFiLMModel (True FiLM: per-block γ scale + β shift).
        # film_mode is now a PATH key (via args_to_watch_fm_visual_train → '..._filmv1'/'..._filmv2'),
        # so v1 and v2 checkpoints/results save in PARALLEL dirs automatically — just flip this
        # value, no prefix edit needed. v2 is OPT-IN and requires fresh training.
        # ⚠ Existing pre-2026-06-27 checkpoints were saved WITHOUT the '_filmv1' suffix; to eval
        #   them either rename their folder to add '_filmv1' or retrain.
        'film_mode': 'v1',

        # ======================================================================================
        # 📊 DATASET
        # ParityAligningDataset loads 9D trajectories from raw pkl files.
        # max_path_length serves TWO roles here (not a per-trajectory step cap):
        #   1. Passed as max_n_episodes to ParityAligningDataset — a soft ceiling.
        #      The aligning dataset has 900 episodes, so min(900, 1000)=900: all episodes
        #      load and the cap is never hit. Raise if you add more demos; lower to subsample.
        #   2. Embedded in the checkpoint directory name via args_to_watch_fm_visual_train
        #      (as 'steps1000'). plan_fm_visual_aligning.max_path_length MUST match
        #      exactly, or diffusion_loadpath resolves to a non-existent directory.
        # ======================================================================================
        'max_path_length': 1000,

        # ======================================================================================
        # 💾 SERIALIZATION & EXPERIMENT LOGS
        # ======================================================================================
        'logbase': logbase,
        'prefix': 'fm_visual_aligning/',
        'exp_name': watch(args_to_watch_fm_visual_train),

        # ======================================================================================
        # 🏋️‍♂️ TRAINING HYPERPARAMETERS
        # ======================================================================================
        'batch_size': 64,             # d3il vision baseline: 64
        'learning_rate': 2e-4,
        'ema_decay': 0.995,
        'n_steps_per_epoch': 1000,
        # d3il trains for epoch=4 (epoch-based). We use steps-based training.
        # 5e5 steps @ batch=32 / gradient_accumulate=2 ≈ effective 333 optimizer steps/epoch-equivalent.
        'n_train_steps': 1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    # ══════════════════════════════════════════════════════════════════════════════
    # PLANNING CONFIGS (EVALUATION / MPC)
    # ══════════════════════════════════════════════════════════════════════════════

    # ─── 1. DDPM ENCDEC VISION (Planning) ───
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
        'exp_name': watch_plan(args_to_watch_dpcc_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

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

    # ─── 2. FM ENCDEC VISION (Planning) ───
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
        'exp_name': watch_plan(args_to_watch_fmv3_ode_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'

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

    # ─── 3. DIFFUSER VISUAL ALIGNING (Planning) ───
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
        # mpc_batch_size: MPC candidate pool — all B trajectories compete under trajectory_selection.
        # DPCC reference (avoiding-d3il.py) uses batch_size=4. diffuser variant overrides to 1 in eval script.
        # Distinct from train batch_size (32) — renamed to avoid path ambiguity.
        'mpc_batch_size': 1,
        # train_batch_size: reflects visual_aligning_dpcc.batch_size (32) for diffusion_loadpath construction.
        # MUST match the training config exactly or diffusion_loadpath resolves to a non-existent directory.
        'train_batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,
        'loadbase': None,
        'logbase': logbase,
        'prefix': (
            'f:plans/visual_aligning_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_film{film_mode}/'
        ),
        'exp_name': watch_plan(args_to_watch_dpcc_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'
        'diffusion': 'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'returns_condition': False,
        # MUST match the TRAINING block's film_mode — the architecture must agree with the
        # checkpoint, and film_mode is baked into diffusion_loadpath ('..._film{film_mode}'),
        # so this auto-resolves to the v1 or v2 checkpoint dir. Flip both blocks together.
        'film_mode': 'v1',
        'predict_epsilon': True,
        'diffusion_timestep_threshold': _yaml_threshold,
        # D1: original DPCC always evaluates with clip_denoised=False (cosine schedule ~9.4× amplification
        # at early steps corrupts the chain if clamped). Must be False; set True only to ablate.
        'clip_denoised': False,
        'diffusion_loadpath': (
            'f:visual_aligning_dpcc/'
            'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_film{film_mode}'
        ),
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },

    # ─── 4. FM VISUAL ALIGNING (Planning) ───
    'plan_fm_visual_aligning': {
        # ======================================================================================
        # 🎮 INFERENCE PLANNING AND MULTI-THREAD SIMULATOR CONSTRAINTS
        # ======================================================================================
        'horizon': 8,
        # flow_steps_v3: number of Euler ODE steps integrating the velocity field 0→1.
        # More steps = smoother trajectories, higher inference cost. 100 matches the
        # visual_aligning_dpcc baseline (K=100 denoising steps) for fair comparison.
        'flow_steps_v3': 100,
        # Available backend options: 'legacy_euler', 'torchdiffeq'.
        'ode_solver_backend_v3': 'legacy_euler',
        # Available method options (torchdiffeq backend only):
        # dopri8, dopri5, bosh3, fehlberg2, adaptive_heun,
        # euler, midpoint, heun2, heun3, rk4,
        # explicit_adams, implicit_adams, fixed_adams, scipy_solver.
        'ode_solver_method_v3': 'euler',
        # Tolerance / step-size params — unused by legacy_euler; set to None.
        # Must be present: Config passes ALL keys to the constructor, and
        # VisualGaussianDiffusion.__init__ intercepts them to avoid TypeError.
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        # Beta params MUST match fm_visual_aligning training block exactly —
        # they are embedded in diffusion_loadpath as 'a{alpha}_b{beta}'.
        # A mismatch here produces a FileNotFoundError (wrong a/b in loadpath).
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        # D3IL Robot_Push_Env default is 400 (hardcoded, proven stable for the aligning task).
        # Fix 10 wired this field so it now actually reaches the env. Start at 400 (proven baseline).
        # Increase only after confirming the model benefits from a longer rollout budget.
        'max_episode_length': 400,
        'max_path_length': 1000,    # MUST match fm_visual_aligning.max_path_length: loadpath key only
        'action_weight': 1,
        # window_size=1 / obs_seq_len=1 must match training: ParityAligningDataset
        # provides single-frame images per sample, so the model is trained on T_win=1.
        # Using window_size>1 at eval would mean-pool multiple frames and shift the
        # FiLM conditioning distribution away from what the model learned.
        'window_size': 1,
        'obs_seq_len': 1,
        'if_vision': True,
        # mpc_batch_size: MPC candidate pool — all B trajectories compete under trajectory_selection.
        # DPCC reference (avoiding-d3il.py) uses batch_size=4. diffuser variant overrides to 1 in eval script.
        # Distinct from train batch_size (32) — renamed to avoid path ambiguity (also fixes b{mpc} vs b{beta} clash).
        'mpc_batch_size': 4,
        # train_batch_size: reflects fm_visual_aligning.batch_size (32) for diffusion_loadpath construction.
        # MUST match the training config exactly or diffusion_loadpath resolves to a non-existent directory.
        'train_batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,

        # ======================================================================================
        # 💾 SERIALIZATION & MODEL LOADING
        # ======================================================================================
        'loadbase': None,
        'logbase': logbase,
        'prefix': (
            'f:plans/fm_visual_aligning/'
            'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_film{film_mode}/'
        ),
        'exp_name': watch_plan(args_to_watch_fm_visual_plan),
        'custom_msg': custom_msg,   # '' => path unchanged; else '..._msg<value>'
        'diffusion': 'fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching',
        'returns_condition': False,
        # MUST match the TRAINING block's film_mode — the architecture must agree with the
        # checkpoint, and film_mode is baked into diffusion_loadpath ('..._film{film_mode}'),
        # so this auto-resolves to the v1 or v2 checkpoint dir. Flip both blocks together.
        'film_mode': 'v1',
        'predict_epsilon': True,
        'diffusion_timestep_threshold': _yaml_threshold,
        # D1: FM ODE does not use a denoising schedule, so clip_denoised has no effect on the chain.
        # Kept False (same as reference DPCC default) for consistency. Set True only to ablate.
        'clip_denoised': False,
        # diffusion_loadpath resolves the training checkpoint directory.
        # The a/b/aw/V/steps/bs fragments MUST match fm_visual_aligning's exp_name exactly
        # (generated by args_to_watch_fm_visual_train). Any mismatch → FileNotFoundError.
        'diffusion_loadpath': (
            'f:fm_visual_aligning/'
            'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
            '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_film{film_mode}'
        ),
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
}

# ─── Gen6 State-Only Non-Visual Configuration Appends ────────────────────────
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 3,   # UF-17: 3D velocity [dx,dy,dz] — matches visual path and DPCC principle
    'obs_dim': 20,     # full state: des_c_pos(3)+c_pos(3)+box(3)+box_q(4)+tgt(3)+tgt_q(4)
    'if_vision': False,
    'prefix': 'ddpm_encdec_vision_nonvisual/',
}

base['plan_ddpm_encdec_vision_nonvisual'] = {
    **base['plan_ddpm_encdec_vision'],
    'prefix': 'f:plans/ddpm_encdec_vision_nonvisual/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
    'diffusion_loadpath': 'f:ddpm_encdec_vision_nonvisual/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}',
}

# ─── Gen8 iMF Visual Aligning (Training) ─────────────────────────────────────
# Same task/data/dims as Gen7 fm_visual_aligning (9D, dual-cam, D3IL aligning).
# Engine swap: FlowMatchingODE → iMeanFlowODE (mean-flow target, h-conditioning, u/v dual heads).
# Dims unchanged: action_dim=3, obs_dim=6, transition_dim=9.
# u_loss_weight / v_loss_weight / loss_schedule: iMF-specific; not in fm_visual_aligning.
base['imf_visual_aligning'] = {
    **base['fm_visual_aligning'],
    'model':     'imf_visual_aligning.models.imf_engine.iMeanFlowEngine',
    'diffusion': 'imf_visual_aligning.models.visual_imf_diffusion.VisualIMF',
    'prefix':    'imf_visual_aligning/',
    'exp_name':  watch(args_to_watch_imf_visual_train),
    # iMF-specific loss mixing weights (inherited by VisualIMF from iMeanFlowODE)
    'u_loss_weight':  1.0,   # weight on main u-head (mean-flow target)
    'v_loss_weight':  0.1,   # weight on aux v-head (instantaneous velocity)
    'loss_schedule':  'balanced',
    # U3/imfv2 — training objective selector (see logs_in_develop/Gen8/Epoch_1/U3).
    # 'fm_equivalent': legacy finite-diff = FM baseline arm.
    # 'meanflow_jvp' : real MeanFlow-Identity (JVP). NFE is set in the plan block, not here.
    'imf_objective': 'fm_equivalent',
    'meanflow_r_equals_t_frac': 0.25,
    'meanflow_adaptive_p': 0.5,
    'meanflow_adaptive_c': 1e-3,
    'meanflow_aux_weight': 0.0,
    # U7: time-schedule. 'logit_normal' = canonical iMF default (reference imf.py, DEFAULT).
    # 'beta' = legacy 1-Beta(α,β). NOTE: plan block diffusion_loadpath must match.
    't_schedule': 'logit_normal',     # U7 DEFAULT
    'p_mean': -0.4,                   # logit-normal P_mean (sigmoid median ≈ 0.40)
    'p_std': 1.0,                     # logit-normal P_std
    # iMF guardrails against E4 spike (Gen3v4 lesson):
    # lower lr (2e-4 already set via fm_visual_aligning inherit) + smaller action_weight
    'action_weight': 1,
    'learning_rate': 2e-4,
}

# ─── Gen8 iMF Visual Aligning (Planning/Eval) ────────────────────────────────
base['plan_imf_visual_aligning'] = {
    **base['plan_fm_visual_aligning'],
    'diffusion': 'imf_visual_aligning.models.visual_imf_diffusion.VisualIMF',
    'imf_objective': 'fm_equivalent',   # must match training block to resolve diffusion_loadpath
    # U7: MUST match training block. Set 'beta' to load pre-U7 checkpoints.
    't_schedule': 'logit_normal',       # U7 DEFAULT
    'p_mean': -0.4,
    'p_std': 1.0,
    'prefix': (
        'f:plans/imf_visual_aligning/'
        'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
        '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_obj{imf_objective}_ts{t_schedule}/'
    ),
    # diffusion_loadpath MUST match args_to_watch_imf_visual_train exactly
    'diffusion_loadpath': (
        'f:imf_visual_aligning/'
        'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
        '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}_obj{imf_objective}_ts{t_schedule}'
    ),
}


# ══════════════════════════════════════════════════════════════════════════════════════
# Gen14 — VISUAL MIX-ML: one frame, four config-activated ML engines
# ══════════════════════════════════════════════════════════════════════════════════════
# Model folder: mix_visual_aligning/   Test folder: mix_visual_aligning_test/
# Plan:         logs_in_develop/Gen14/init/PLAN_Gen14_visual_mix_ml.md
#
#   engine=diffusion  <- Gen6V4  (visual_aligning_dpcc)      VisualGaussianDiffusion
#   engine=fm    <- Gen7    (fm_visual_aligning)        VisualFlowMatching     [reference arm]
#   engine=mf    <- Gen3v6  (flow_matcher_v3_meanflow)  VisualMeanFlow
#   engine=af    <- Gen3v7  (flow_matcher_v3_alphaflow) VisualAlphaFlow
#
# Task/data/dims are FROZEN across all four arms (9D visual aligning, action_dim=3,
# obs_dim=6, horizon=8) and the backbone is locked to the VisualUNet stack, so the
# four-way comparison is architecture-controlled: only objective + sampler vary.
#
# ⚠️ Gen14 = DPCC math. Gen13 (HF_Mix_ML) = HardFlow math. NEVER mix their results.
#
# The EXISTING blocks above are untouched: Gen6V4 and Gen7 keep their own checkpoints
# and paths, and Gen14 writes under mix_visual_aligning/ exclusively.
# ──────────────────────────────────────────────────────────────────────────────────────

# Single source of truth for the training budget. 🔴 The alpha-Flow anneal MUST span the
# ACTUAL budget (PLAN §6.2a): af_alpha_end_step and af_n_train_steps are BOTH derived from
# this one name, and af_diffusion.py asserts they agree. Never write the number twice.
_MIX_FULL_N_TRAIN_STEPS = int(1e5)
_MIX_N_TRAIN_STEPS = int(float(os.environ.get('MIX_TRAIN_STEPS', _MIX_FULL_N_TRAIN_STEPS)))

# 🔴 A SHORTENED RUN MUST NOT LOOK LIKE A FULL ONE. Budget is an identity key: a 50k-step
# checkpoint and a 100k-step checkpoint are different models, and two different models must
# never share a directory. When MIX_TRAIN_STEPS cuts the budget, `train_budget` joins
# args_to_watch_mix_visual_train and the checkpoint folder gains a trailing '_TB50pct'.
# At the FULL budget the key is absent (watch() skips undefined keys), so every path that
# exists today keeps its current name. Mirrored from Gen16's avoiding config.
def _budget_tag():
    if _MIX_N_TRAIN_STEPS >= _MIX_FULL_N_TRAIN_STEPS:
        return None
    pct = 100.0 * _MIX_N_TRAIN_STEPS / _MIX_FULL_N_TRAIN_STEPS
    return f'{int(round(pct))}pct' if abs(pct - round(pct)) < 1e-9 else f'{_MIX_N_TRAIN_STEPS}steps'


_MIX_BUDGET_TAG = _budget_tag()

# One watch list for all four arms. watch() skips keys a block does not define, so each
# arm's folder name carries exactly its own identity keys (e.g. 'K' only for diffusion,
# 'ts'/'af' only for the two-time arms).
args_to_watch_mix_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),      # diffusion only — FM/MF/AF blocks do not define it
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
    ('batch_size', 'bs'),
    ('film_mode', 'film'),
    # 🔴 Gen14 U8 — the ML-BONE key. MUST be in this list: film_mode is a path key and the
    # bone was not, so a DiT run would have overwritten the U-Net checkpoint in the same
    # directory (the trap CHANGELOG_Gen14_U5...md:208 flagged a year in advance). Blocks that
    # do not define it are skipped by watch(), so every existing U-Net path is unchanged.
    ('ml_bone', 'B'),
    ('engine', 'E'),                 # ← Gen14 arm identity key
    ('t_schedule', 'ts'),            # mf/af only
    ('af_alpha_scheduler', 'afsch'), # af only
    # ── Gen14 U9 ── perception / conditioning identity keys. Each is emitted by
    # _mix_u9_keys() ONLY when it differs from the pre-U9 value, and watch() skips keys a
    # block does not define — so every path that exists today keeps its exact current name
    # and a U9 run lands in its own tree. This is the same absent-when-default trick
    # ml_bone, film_mode and train_budget already use; it is what stops U9 from orphaning
    # every U8 checkpoint and results folder.
    ('vis_pretrained', 'VP'),
    ('vis_lr_scale',   'VLR'),
    ('vis_cond_mode',  'VC'),
    # Appended LAST so a reduced-budget run reads as the full name plus a suffix, and the
    # full-budget name is unchanged. Set only when the budget is cut (see _budget_tag).
    ('train_budget', 'TB'),
]

args_to_watch_mix_visual_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),      # diffusion only
    ('flow_steps_v3', 'K'),          # fm/mf/af only  (the two are mutually exclusive per arm)
    ('ode_solver_method_v3', 'M'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('mpc_batch_size', 'mpc'),
    ('film_mode', 'film'),
    ('ml_bone', 'B'),                # Gen14 U8 — see the training list
    ('engine', 'E'),
]


# A few training identity keys live under a DIFFERENT name on the planning side, to keep
# the eval script's namespace unambiguous. Gen7 does the same rename by hand
# (`batch_size` -> `train_batch_size`, "also fixes b{mpc} vs b{beta} clash"). The VALUES
# must be equal; only the arg name differs. Extend this map, never the call sites.
_MIX_TRAIN_TO_PLAN_KEY = {
    'batch_size': 'train_batch_size',
}


def _mix_loadpath(watch_list, block, prefix, key_map=None):
    """Build a loadpath/prefix format string from the SAME watch list that builds exp_name.

    🔴 This is the fix for the oldest trap in this repo: a plan block whose
    diffusion_loadpath does not reproduce args_to_watch key-for-key resolves to a
    non-existent directory and the eval dies minutes into a GPU allocation. Deriving the
    string here makes that class of bug unrepresentable — there is only one list.

    Mirrors diffuser.utils.watch(): keys absent from `block` are skipped, and the entries
    are joined with '_' after the trailing-slash prefix.

    `block` is the block that DEFINES the identity (always the TRAINING block, so the
    fragments and their order match what watch() emitted at train time). `key_map`
    renames the emitted {placeholder} to the name the CONSUMING block exposes — the
    placeholder is resolved later by eval_fstrings against the consuming args object,
    so it must name a key that object actually has.
    """
    key_map = key_map or {}
    parts = []
    for key, label in watch_list:
        if key == 'prefix' or key not in block:
            continue
        parts.append(f'{label}{{{key_map.get(key, key)}}}')
    return 'f:' + prefix + '_'.join(parts)


# ─── shared eval/planning knobs, identical for all four arms ────────────────────────────
# Inherited from plan_fm_visual_aligning so the ROLLOUT is byte-identical across arms:
# same env, same episode budget, same MPC candidate pool, same constraint YAML. Only the
# generative engine differs — that is the whole point of the generation.
_mix_plan_common = {
    k: v for k, v in base['plan_fm_visual_aligning'].items()
    if k not in ('prefix', 'exp_name', 'diffusion', 'diffusion_loadpath')
}


# Gen14 U8 — sentinel meaning "REMOVE this inherited key from the block".
# An override dict can add or replace, but the mf/af arms inherit `film_mode` from their
# parent (`fm_visual_aligning`), and on a DiT bone the key must be GONE, not merely unset:
# diffuser's watch() skips keys the args object lacks (utils/setup.py:25), so deleting it is
# what keeps '_film..' out of a transformer checkpoint path.
_DROP = object()


def _mix_train_block(engine, parent, overrides):
    """Assemble one training block: parent arm's config + Gen14 identity + arm overrides.

    Keys whose override value is `_DROP` are removed from the merged block entirely.
    """
    blk = {**base[parent], **overrides, 'engine': engine,
           'prefix': f'mix_visual_aligning_{engine}/'}
    for k in [k for k, v in blk.items() if v is _DROP]:
        del blk[k]
    # Present ONLY on a reduced budget -- absent means "full 1e5".
    if _MIX_BUDGET_TAG is not None:
        blk['train_budget'] = _MIX_BUDGET_TAG
    blk['exp_name'] = watch(args_to_watch_mix_visual_train)
    return blk


def _mix_plan_block(engine, train_blk, overrides, drop=()):
    """Assemble one planning block, deriving every path string from the watch lists.

    Mirrors Gen7's two-level scheme exactly:
      prefix   -> the CHECKPOINT identity (training keys)  = which model produced this
      exp_name -> the EVAL identity       (plan keys)      = how it was rolled out
    so results for one checkpoint under different K / solver / mpc land in sibling dirs.

    `drop` removes keys inherited from the FM plan template that do not apply to this arm
    (e.g. the ODE-solver keys on the DDPM arm), so they never reach the folder name or the
    engine constructor.
    """
    blk = {**_mix_plan_common, **overrides, 'engine': engine,
           'diffusion': train_blk['diffusion']}
    for k in drop:
        blk.pop(k, None)

    # Mirror every TRAINING identity value into the plan block under its planning-side
    # name, so the derived {placeholders} below always resolve. Values are copied from the
    # training block itself — a plan/train value mismatch is therefore impossible, which is
    # the whole failure mode Gen7's "MUST match exactly" comments warn about.
    for key, _label in args_to_watch_mix_visual_train:
        if key == 'prefix' or key not in train_blk:
            continue
        plan_key = _MIX_TRAIN_TO_PLAN_KEY.get(key, key)
        # Unconditional: for an IDENTITY key the training value is the only correct one.
        # Overriding it in a plan block is exactly the mistake this loop prevents.
        blk[plan_key] = train_blk[key]

    # 🔴 Gen14 U8 — an identity key the TRAINING block does not have must not survive on the
    # plan block. `film_mode` arrives here via _mix_plan_common (copied from the FM plan
    # template), so on a DiT bone it would otherwise label the RESULTS folder '_filmv1_' for a
    # model that has no FiLM path at all — the eval-side twin of the checkpoint-path lie.
    # The mirror loop above only ADDS keys; this is the matching removal.
    for _identity_key, _ in args_to_watch_mix_visual_train:
        if _identity_key == 'prefix':
            continue
        if _identity_key not in train_blk:
            blk.pop(_MIX_TRAIN_TO_PLAN_KEY.get(_identity_key, _identity_key), None)

    # Training-key fragments -> the CHECKPOINT identity, re-pointed at this arm's plan
    # namespace. `[2:]` strips the 'f:' marker; the whole prefix carries one of its own.
    _ckpt_id = _mix_loadpath(
        args_to_watch_mix_visual_train, train_blk, '', _MIX_TRAIN_TO_PLAN_KEY)[2:]
    blk['prefix']   = f'f:plans/mix_visual_aligning_{engine}/{_ckpt_id}/'
    # custom_msg is inherited via _mix_plan_common (copied from plan_fm_visual_aligning),
    # so the four mix arms carry the same tag as every other plan block.
    blk['exp_name'] = watch_plan(args_to_watch_mix_visual_plan)

    # diffusion_loadpath must reproduce the TRAINING block's exp_name exactly, key for key.
    blk['diffusion_loadpath'] = _mix_loadpath(
        args_to_watch_mix_visual_train, train_blk,
        f'mix_visual_aligning_{engine}/', _MIX_TRAIN_TO_PLAN_KEY)
    return blk


# ─── FiLM conditioning backbone — the `film_mode` knob, all four arms ──────────────────
# Each of the four mix candidates has its OWN film_mode entry, resolved by its OWN knob via
# _film_mode('<arm>'). The DEFAULT is 'v1' on every arm — identical to what the parent blocks
# already supplied (visual_aligning_dpcc:377 and fm_visual_aligning:461 both say 'v1') — so
# with no env var set, nothing changes anywhere.
#
#     MIX_FILM_MODE_DIFFUSION   MIX_FILM_MODE_FM   MIX_FILM_MODE_MF   MIX_FILM_MODE_AF
#
# and a bare MIX_FILM_MODE as the all-arms fallback. Per-arm is the primary form because the
# arms are separate experiments with separate checkpoint trees: putting mf on v2 must not
# drag fm — the Gen7 reference arm — along with it.
#
# Selecting v2 is a LAUNCH-TIME choice, not a config edit. The pipeline derives the arm-
# specific variable from its own $1, so the ordinary case needs no thought:
#
#     MIX_FILM_MODE=v2 ./Slurm_Codes/submit.sh \
#         Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf "6"
#
# ...submits the mf arm at v2 and exports MIX_FILM_MODE_MF=v2 onto the child jobs, leaving
# the other three arms resolving 'v1' even inside that job's own config import.
#
# This mirrors the env-var pattern already used for the HardFlow knobs in
# config/avoiding-d3il.py:1334 (HFFM_FLOW_STEPS) and exists because there is NO --film-mode
# CLI override anywhere in the repo: diffuser's generic add_extras is dead (the call is
# commented out at mix_visual_aligning/utils/setup.py:77, and the Tap parser that supplied
# `extra_args` was dropped upstream). film_mode is also a TRAINING key, so the eval-side
# dict-mutation trick that gives --flow-steps its override (U6) cannot work here — the
# checkpoint path has to move with the value.
#
# 🔴 SCOPE: these env vars are read by the four Gen14 arms ONLY. The Gen6V4/Gen7 parent
#    blocks keep their own hardcoded 'v1' (lines 377 / 461) and are unaffected, so a stray
#    MIX_FILM_MODE in the environment can never move a Gen6V4 or Gen7 run.
#
# ✅ A propagation failure is LOUD, not silent: film_mode is a path key, so the savepath
#    reads '..._filmv1_E..' vs '..._filmv2_E..', and both the train and eval scripts print
#    the resolved mode (Gen14 Fix_9). Check the log line before trusting a v2 run.
#
#   'v1'  (default)  Additive-bias "Fake FiLM". The visual latent is concatenated into the
#                    time embedding, so conditioning reaches each residual block as a bias.
#   'v2'              TRUE FiLM: per-block gamma scale + beta shift. Same
#                    `FiLMResidualTemporalBlock` class object on every arm — diffusion/fm
#                    reach it via Gen7's `unet1d_temporal_film.py`, mf/af via
#                    `unet1d_twotime_film.py` (Gen14 U5), which imports that block rather
#                    than reimplementing it and keeps `h_mlp` so the MeanFlow / alpha-Flow
#                    forward-mode JVP survives the multiplicative gate.
#
# 🔴 This is a TRAINING key, not an inference knob — SWITCHING TO v2 REQUIRES A RETRAIN.
#    v1 and v2 state_dicts are NOT interchangeable: `embed_dim` is `dim + cond_embed_dim`
#    under v1 but `dim` under v2, and `film_proj` has no v1 counterpart. v2 costs ~+1.0M
#    backbone parameters. There is no --film-mode CLI override anywhere in the repo
#    (diffuser's generic add_extras is disabled — mix_visual_aligning/utils/setup.py:77),
#    so this config entry is the only switch.
#
# ✅ Safe to flip one arm at a time: film_mode is in args_to_watch_mix_visual_train, so v1
#    and v2 checkpoints land in PARALLEL trees ('..._bs64_filmv1_E..' / '..._filmv2_E..')
#    and no existing run is overwritten. Set it in the TRAINING block only —
#    _mix_plan_block's training-key mirror loop copies it into the plan block, and setting
#    it there too is exactly the double-set that loop exists to prevent.
#
# ⚠️ Before reading a v1-vs-v2 curve: `W_f` is zero-initialised and under v2 the visual
#    latent reaches the network through `W_f` and nowhere else, so at step 0 a v2 model is
#    exactly v1-with-no-vision. Early-epoch curves are not comparable step-for-step
#    (logs_in_develop/Gen14/U5/CHANGELOG_Gen14_U5_engine_rename_and_twotime_filmv2.md §2.8).
#
# 🔴 v2 has never executed a tensor op on any Gen14 arm. Gate G7 builds all four arms at
#    v2 and asserts film_proj is live and one mf loss step is finite under the JVP; it runs
#    by default (gates_mix_visual.sh -> --gate all). Do not bypass the gate job on a v2 run.
_MIX_FILM_MODES = ('v1', 'v2')


def _film_mode(engine):
    """Resolve ONE arm's FiLM mode. Each of the four mix candidates has its own knob.

    Precedence, most specific first:
        1. MIX_FILM_MODE_<ENGINE>   e.g. MIX_FILM_MODE_MF=v2   — this arm only
        2. MIX_FILM_MODE            — every arm that has no specific setting
        3. 'v1'                     — the default, identical to the inherited parent value

    Per-arm is the primary form: the arms are separate experiments with separate checkpoint
    trees, so "mf at v2" must be expressible without also moving af, fm and diffusion. The
    bare MIX_FILM_MODE remains as a convenience for sweeping every arm at once.

    Unknown values RAISE. A silent fallback to v1 would train the wrong architecture into a
    directory whose name claims otherwise — the one failure the path key exists to prevent.
    """
    specific_key = f'MIX_FILM_MODE_{engine.upper()}'
    for key in (specific_key, 'MIX_FILM_MODE'):
        val = os.environ.get(key)
        if val:
            if val not in _MIX_FILM_MODES:
                raise ValueError(
                    f"CRITICAL: {key}='{val}' is not a known FiLM mode "
                    f"(want one of {list(_MIX_FILM_MODES)}).")
            return val
    return 'v1'

# ══════════════════════════════════════════════════════════════════════════════════════
# Gen14 U8 — the ML-BONE knob (generative backbone for the two-time arms)
# ══════════════════════════════════════════════════════════════════════════════════════
# Plan:     logs_in_develop/Gen14/U8/PLAN_Gen14_U8_visual_dit_bone.md
# Decision: logs_in_develop/Gen14/U8/DECISION_Gen14_U8_injection_choice.md
#
#   'unet'    VisualUNetTwoTime  — the Gen14 baseline. FiLM conditioning (film_mode v1/v2).
#   'mf_dit'  official MeanFlow DiT (adaLN-zero trunk)          — mf arm only
#   'sit'     alpha-Flow SiT      (adaLN-zero trunk)            — af arm only
#   'dit'     iMF DiT             (RoPE, in-context tokens)     — both arms
#
# On every DiT/SiT bone the 128-D visual latent enters as ONE PREPENDED TOKEN, never as
# adaLN modulation — that design point is already occupied by the U-Net's FiLM, and
# diffusion_policy (the upstream of this repo's vision encoder) tokenises for its
# transformer and reserves FiLM for its U-Net. See the DECISION doc §2.
#
# ⚠️ film_mode is a U-NET concept. On a DiT bone it is not merely unused, it would put a
#    lying '_filmv1_' fragment in the checkpoint path — so `_mix_bone_keys()` DELETES the
#    key for non-unet bones and watch() then skips it. Never re-add it by hand.
#
# ✅ Safe to flip one arm at a time: ml_bone is in args_to_watch_mix_visual_train, so each
#    bone trains into its own '..._B{bone}_E..' tree and no existing run is overwritten.
_MIX_ML_BONES = {
    'mf': ('unet', 'mf_dit', 'dit'),
    'af': ('unet', 'sit', 'dit'),
}


def _ml_bone(engine):
    """Resolve ONE arm's generative bone. Same precedence shape as _film_mode().

        1. MIX_BONE_<ENGINE>   e.g. MIX_BONE_MF=mf_dit   — this arm only
        2. MIX_BONE            — every two-time arm that has no specific setting
        3. 'unet'              — the default, identical to the pre-U8 behaviour

    Unknown values RAISE, and so does a bone that belongs to the OTHER arm ('sit' on mf,
    'mf_dit' on af) — those are separate classes with separate provenance, and silently
    accepting one would train a model whose folder name names a different architecture.
    """
    allowed = _MIX_ML_BONES[engine]
    for key in (f'MIX_BONE_{engine.upper()}', 'MIX_BONE'):
        val = os.environ.get(key)
        if val:
            if val not in allowed:
                raise ValueError(
                    f"CRITICAL: {key}='{val}' is not a valid ML bone for the '{engine}' arm "
                    f"(want one of {list(allowed)}).")
            return val
    return 'unet'


def _mix_bone_keys(engine):
    """The bone-dependent block fragment: ml_bone plus EITHER film_mode OR the DiT sizing.

    Returns a dict to splat into the arm's training block. Exactly one conditioning-config
    family is present at a time, so a block can never carry both a FiLM mode and a DiT width.
    """
    bone = _ml_bone(engine)
    if bone == 'unet':
        # 🔴 ml_bone is DELIBERATELY ABSENT on the baseline bone. watch() skips keys a block
        # does not define, so the U-Net's exp_name / diffusion_loadpath stay BYTE-IDENTICAL to
        # every pre-U8 Gen14 run — no existing checkpoint or results folder is orphaned. The
        # DiT blocks below DO define it, so they carry a '_B{bone}' fragment the U-Net lacks
        # and the two can never collide. Same trick n_diffusion_steps and film_mode already use.
        return {'film_mode': _film_mode(engine)}
    # DiT/SiT bone: film_mode is deliberately ABSENT (see the warning above).
    # 🔴 dit_hidden_size=160 (not the state-only 256) is the PARAMETER-MATCHED width:
    # 18*depth*d^2 => 160/8 ~ 3.9 M vs the visual U-Net's ~4.0 M (dim=32). 256/8 is ~9.9 M,
    # i.e. 2.5x, and an unmatched backbone A/B is exactly the Fix_8 defect that already
    # forced one public retraction (PLAN §1.2(c), §8). Both are ARCHITECTURE keys — changing
    # either requires a retrain, and neither is in the watch list, so treat a change as
    # needing a new bone name rather than silently overwriting.
    return {
        'ml_bone':         bone,
        'film_mode':       _DROP,   # 🔴 inherited from fm_visual_aligning — must be DELETED
        'dit_hidden_size': 160,
        'dit_depth':       8,
        'dit_num_heads':   4,
        'dit_patch_size':  1,
    }


# ═══ Gen14 U9 — perception-first knobs ═══════════════════════════════════════════════
# Three ML-side flags. NONE of them touches the projector, the constraint YAML, the MPC
# candidate fan, the horizon or the observation window — U9 is deliberately comparable to
# U8 cell for cell (PLAN §3).
#
#   MIX_VIS_PRETRAINED=1        ImageNet init of the dual ResNet-18 (default 0 = random,
#                               i.e. every pre-U9 run). Only the WEIGHTS change: the
#                               architecture, LATENT_DIM=128 and cond_dim are untouched, so
#                               every U8 gate stays valid exactly as written.
#   MIX_VIS_LR_SCALE=0.1        encoder LR = train_lr * scale (default 1.0 = pre-U9).
#                               0.0 is the hard-freeze extreme.
#   MIX_VIS_COND=adaln          where the latent enters the transformer (default 'token'
#                               = U8, bit-identical). adaLN bones only (mf_dit / sit).
#
# 🔴 EVERY key is emitted ONLY when non-default. At the defaults _mix_u9_keys() returns {},
#    watch() skips all three, and the checkpoint path is character-for-character the U8 one
#    — which is what makes the R0 reproduction run in PLAN §6 a real check rather than a
#    new tree that trivially "passes".
_U9_DEFAULTS = {'vis_pretrained': False, 'vis_lr_scale': 1.0, 'vis_cond_mode': 'token'}


def _mix_u9_keys(engine):
    """U9 knobs for one arm, present only where they differ from the pre-U9 behaviour."""
    def _env(name):
        return os.environ.get(f'{name}_{engine.upper()}') or os.environ.get(name)

    out = {}

    raw = _env('MIX_VIS_PRETRAINED')
    if raw is not None:
        if raw.strip().lower() not in ('0', '1', 'true', 'false'):
            raise ValueError(
                f"CRITICAL: MIX_VIS_PRETRAINED='{raw}' must be 0|1|true|false.")
        val = raw.strip().lower() in ('1', 'true')
        if val != _U9_DEFAULTS['vis_pretrained']:
            out['vis_pretrained'] = val

    raw = _env('MIX_VIS_LR_SCALE')
    if raw is not None:
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f"CRITICAL: MIX_VIS_LR_SCALE='{raw}' is not a float.")
        if val < 0.0:
            raise ValueError(f"CRITICAL: MIX_VIS_LR_SCALE={val} must be >= 0.")
        if val != _U9_DEFAULTS['vis_lr_scale']:
            # Rendered as-is ('VLR0.1'): dotted fragments are already the house style
            # here (the plan paths carry 'T0.5'), so no p-substitution is introduced.
            out['vis_lr_scale'] = val

    raw = _env('MIX_VIS_COND')
    if raw is not None:
        if raw not in ('token', 'adaln', 'both'):
            raise ValueError(
                f"CRITICAL: MIX_VIS_COND='{raw}' is not one of token|adaln|both.")
        if raw != _U9_DEFAULTS['vis_cond_mode']:
            # 🔴 DROP, DO NOT RAISE (jobs 25034 / 25038, 2026-08-25).
            #
            # This module defines ALL FOUR arms on import, and a bare MIX_VIS_COND rides
            # along to every one of them. Raising here made the whole config UNIMPORTABLE
            # for any arm whose bone is not adaLN -- which is most of them, most of the
            # time -- and it took down G-B7 twice: once on 'mf', then on 'af'. A config
            # module that cannot be imported is a worse failure than the one being guarded.
            #
            # What actually needs preventing is a LYING PATH KEY: a '_VCadaln' fragment on
            # a checkpoint whose bone has no adaLN pathway and ignored the setting. Dropping
            # the key prevents exactly that, with no collateral damage -- the same treatment
            # film_mode gets on a DiT bone (_DROP in _mix_bone_keys).
            #
            # The HARD failure lives where the arm is actually known, which is where it
            # belongs and where it still fires:
            #   * Slurm_Codes/.../train_mix_visual_aligning.sh and the pipeline -- at SUBMIT
            #     time, before a GPU is allocated;
            #   * VisualDiTTwoTime.__init__ -- at build time, as the backstop.
            bone = _ml_bone(engine)
            if bone not in ('mf_dit', 'sit'):
                print(f"[ config ] NOTE: MIX_VIS_COND='{raw}' ignored for the '{engine}' arm "
                      f"(ml_bone='{bone}' has no adaLN pathway); the key is dropped so no "
                      f"'_VC{raw}' fragment lands in a path that did not use it. Use "
                      f"MIX_VIS_COND_{engine.upper()} to target one arm.", file=sys.stderr)
            else:
                out['vis_cond_mode'] = raw

    return out


# ─── arm: diffusion (Gen6V4) ────────────────────────────────────────────────────────────────
# Parent is visual_aligning_dpcc, NOT fm_visual_aligning: the DDPM arm must inherit
# Gen6V4's own hyperparameters (action_weight=10), otherwise it is not the Gen6V4 baseline
# it claims to be.
base['mix_visual_aligning_diffusion'] = _mix_train_block('diffusion', 'visual_aligning_dpcc', {
    'model':     'mix_visual_aligning.models.visual_unet.VisualUNet',
    'diffusion': 'mix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
    # ── U7 2026-08-08 ── K=20, matching the ARCHIVED Gen6V4 run (its checkpoint and results
    # folders are both `H8_K20_...`). The parent block says 100; that value post-dates the
    # Gen6V4 artefacts everyone actually compares against (commit 2c87cb70), so inheriting it
    # made the arm a 5×-NFE variant of Gen6V4 rather than Gen6V4 — see
    # logs_in_develop/Gen14/U7/DA_20260808_gen14_diffu_fm_arms.md §1.2.
    # 🔴 UNLIKE the fm/mf/af arms, this is NOT an inference-only knob. n_diffusion_steps is the
    # DDPM chain length: it sets the training noise schedule AND is a checkpoint-path key in
    # args_to_watch_mix_visual_train ('K{n_diffusion_steps}'). Changing it here therefore
    # REQUIRES A RETRAIN — the eval's --flow-steps override explicitly refuses this arm
    # (mix_visual_aligning_test/eval_mix_visual_aligning.py:2444). The plan block picks 20 up
    # automatically via _mix_plan_block's training-key mirror loop; do not set it there too.
    # Overriding here and not in visual_aligning_dpcc keeps Gen6V4's own blocks untouched.
    'n_diffusion_steps': 20,
    # FiLM backbone — this arm's OWN knob. Default 'v1' == the inherited value;
    # MIX_FILM_MODE_DIFFUSION=v2 selects TRUE FiLM for this arm alone (retrain required).
    # Backbone: VisualUNet -> unet1d_temporal_film.
    'film_mode': _film_mode('diffusion'),
})

# ─── arm: fm (Gen7) — THE REFERENCE ARM ────────────────────────────────────────────────
# Must remain a pure re-pointing of fm_visual_aligning. Do not tune anything here: gate G1
# compares this arm against Gen7 and expects bit-identical training.
base['mix_visual_aligning_fm'] = _mix_train_block('fm', 'fm_visual_aligning', {
    'model':     'mix_visual_aligning.models.visual_unet.VisualUNet',
    'diffusion': 'mix_visual_aligning.models.visual_fm_diffusion.VisualFlowMatching',
    # FiLM backbone — this arm's OWN knob (MIX_FILM_MODE_FM). This is NOT tuning: the
    # default 'v1' is byte-for-byte the value fm_visual_aligning:461 already supplied, so
    # G1's Gen7 training-parity check is unaffected. 🔴 Running this arm at v2 WOULD break
    # that parity by design — the fm arm is the Gen7 reference, so a v2 fm run is a NEW arm,
    # not the reference. Do not compare it to Gen7 and call it a reproduction. Having a
    # per-arm knob is what keeps a v2 sweep of mf/af from silently dragging this arm along.
    'film_mode': _film_mode('fm'),
})

# ─── arm: mf (Gen3v6 MeanFlow) ─────────────────────────────────────────────────────────
base['mix_visual_aligning_mf'] = _mix_train_block('mf', 'fm_visual_aligning', {
    'model':     'mix_visual_aligning.models.mf_engine.MeanFlowEngine',
    'diffusion': 'mix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow',
    # Time schedule: 'logit_normal' is the official MeanFlow default and the Gen3v6 default.
    # 🔴 The sign convention is -p_mean (mf_diffusion.py:360). Using +p_mean puts the mass
    # near NOISE and looks *almost* fine. Do not "fix" it.
    't_schedule': 'logit_normal',
    'p_mean': -0.4,
    'p_std': 1.0,
    # Official MeanFlow objective constants (per-sample SUM, p=1, eps=0.01).
    'meanflow_data_proportion': 0.5,   # fraction of the batch forced to r==t (FM anchors)
    'mf_adp_p': 1.0,
    'mf_adp_eps': 0.01,
    # 🔴 Architecture flags — copied from Gen3v6 (config/avoiding-d3il.py:635-636).
    # dual_head=True: the v head SHARES the backbone trunk and carries a full loss
    # (FIX-4). False falls back to an orphan MLP on raw x and guts half the objective.
    # interval_cfg=False: no CFG in Gen3v6. On the UNet arm this flag changes the
    # state_dict, so flipping it makes checkpoints non-interchangeable.
    'dual_head': True,
    'interval_cfg': False,
    # Gen3v6/v7 trainer extras — a DEAD key in the Gen7 trainer, actually applied here.
    'gradient_clip': 1.0,
    'split_seed': 42,
    # PLAN §6.1 ablation knob. Default OFF: the vision encoder trains end-to-end, exactly
    # as in Gen6V4/Gen7. Pre-encoding the latent is what zeroes the JVP tangent — freezing
    # is NOT required for that, and turning this on changes what is learned.
    'mf_freeze_vision_encoder': False,
    # FiLM backbone — this arm's OWN knob. Default 'v1' == the inherited value;
    # MIX_FILM_MODE_MF=v2 selects TRUE FiLM (retrain required). Backbone: VisualUNetTwoTime
    # -> unet1d_twotime_film.Flow_matcher_U_Net_v2_FiLM, which retains h_mlp (U5).
    # ⚠️ For an mf-vs-af comparison, move this arm and the af arm TOGETHER (see af block).
    # Gen14 U8: _mix_bone_keys emits ml_bone + film_mode on the U-Net bone, or ml_bone +
    # the DiT sizing (film_mode ABSENT) on a transformer bone. MIX_BONE_MF=mf_dit|dit.
    **_mix_bone_keys('mf'),
    # Gen14 U9 — empty dict at the defaults, so this line is inert on every
    # pre-U9 configuration and the path string is unchanged.
    **_mix_u9_keys('mf'),
})

# ─── arm: af (Gen3v7 alpha-Flow) ───────────────────────────────────────────────────────
base['mix_visual_aligning_af'] = _mix_train_block('af', 'fm_visual_aligning', {
    'model':     'mix_visual_aligning.models.af_engine.AlphaFlowEngine',
    'diffusion': 'mix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow',
    't_schedule': 'logit_normal',
    'p_mean': -0.4,
    'p_std': 1.0,
    # alpha-Flow's OWN constants. ⚠️ af_adp_eps=1e-3 is DELIBERATELY != MeanFlow's 0.01
    # (af_diffusion.py:97) — different method, different constant. Do NOT harmonise them.
    'af_ratio_fm': 0.5,
    'af_adp_eps': 1e-3,
    'af_clamp_utgt': 4.0,
    # 🔴 Same architecture flags as the mf arm — Gen3v7 ships them identically
    # (config/avoiding-d3il.py:754-755). Keeping mf and af equal here is also what
    # makes the MeanFlow-vs-alpha-Flow comparison architecture-controlled.
    'dual_head': True,
    'interval_cfg': False,
    # The alpha anneal: 1 -> 0 means training starts as plain flow matching and becomes
    # MeanFlow. 🔴 end_step is bound to _MIX_N_TRAIN_STEPS, the same name that sets
    # n_train_steps below. The train script re-derives it and af_diffusion asserts on it.
    'af_alpha_scheduler': 'sigmoid',
    'af_alpha_init': 1.0,
    'af_alpha_end': 0.0,
    'af_alpha_init_step': 0,
    'af_alpha_end_step': _MIX_N_TRAIN_STEPS,
    'af_alpha_gamma': 25.0,
    'af_alpha_clamp': 0.005,
    'n_train_steps': _MIX_N_TRAIN_STEPS,
    'gradient_clip': 1.0,
    'split_seed': 42,
    'mf_freeze_vision_encoder': False,
    # FiLM backbone — this arm's OWN knob (MIX_FILM_MODE_AF), independent of mf's.
    # ⚠️ The mf-vs-af comparison is architecture-controlled ONLY if both arms run the same
    # mode. Independent knobs make that YOUR responsibility rather than the config's: set
    # MIX_FILM_MODE=v2 (the all-arms form) when you want the pair moved together, or set
    # MIX_FILM_MODE_MF and MIX_FILM_MODE_AF to the same value. Comparing mf@v2 against
    # af@v1 confounds the objective with the conditioning route.
    # Gen14 U8: same bone treatment as the mf arm. MIX_BONE_AF=sit|dit. The mf-vs-af
    # comparison is architecture-controlled only if BOTH arms sit on the same bone — use
    # the bare MIX_BONE to move them together.
    **_mix_bone_keys('af'),
    # Gen14 U9 — empty dict at the defaults, so this line is inert on every
    # pre-U9 configuration and the path string is unchanged.
    **_mix_u9_keys('af'),
})

# ─── planning / evaluation blocks (one per arm) ────────────────────────────────────────
base['plan_mix_visual_aligning_diffusion'] = _mix_plan_block(
    'diffusion', base['mix_visual_aligning_diffusion'], {},
    # DDPM has a real reverse chain, not an ODE: drop every continuous-time key inherited
    # from the FM plan template so it reaches neither the folder name nor the constructor.
    # n_diffusion_steps (copied from the training block) is this arm's K.
    drop=('flow_steps_v3', 'ode_solver_backend_v3', 'ode_solver_method_v3',
          'ode_solver_rtol_v3', 'ode_solver_atol_v3', 'ode_solver_step_size_v3',
          'time_beta_alpha_v3', 'time_beta_beta_v3'))

base['plan_mix_visual_aligning_fm'] = _mix_plan_block(
    'fm', base['mix_visual_aligning_fm'], {
        # ── U7 2026-08-08 ── K=20, matching Gen7's own eval folders (`H8_K20_Meuler_...`).
        # Overrides the 100 inherited from plan_fm_visual_aligning via _mix_plan_common — that
        # 100 is the DDPM-parity number and it made the reference arm a 5×-NFE variant of Gen7,
        # which is not what "compare this arm against Gen7" means. At K=100 this arm diverged
        # (peak tracking error > 1 m) in 77% of unprojected rollouts against Gen7's 20–27%:
        # logs_in_develop/Gen14/U7/DA_20260808_gen14_diffu_fm_arms.md §3.
        # Inference-only and safe to set here, exactly as for the mf/af arms below:
        # flow_steps_v3 lives in args_to_watch_mix_visual_plan ONLY, so the training-key mirror
        # loop cannot clobber it and diffusion_loadpath is unchanged — same checkpoint, new
        # H8_K20_... results folder. Sweep per run with `--flow-steps N`; no config edit needed.
        # 🔴 This ALSO halves-then-some the projection budget: the sampler projects from
        # int((1 - diffusion_timestep_threshold) * K) to the end, so at T=0.5 K=100 -> 50 SLSQP
        # solves per replan and K=20 -> 10. That is the half that made the K=100 eval miss the
        # 24 h cap after 2 of 32 items.
        'flow_steps_v3': 20,
    })

base['plan_mix_visual_aligning_mf'] = _mix_plan_block(
    'mf', base['mix_visual_aligning_mf'], {
        # ── U6 ── NFE. Overrides the 100 inherited from plan_fm_visual_aligning via
        # _mix_plan_common. 100 is the DDPM-parity number and is WRONG for a two-time
        # model: MeanFlow's entire premise is that one u-head query spans a whole
        # interval, and the Gen3v6/v7 state-only lineage evaluates at K=2 (see
        # logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md).
        # 🔴 This ALSO sets the projection budget, which is the expensive half. The sampler
        # projects on every step from int((1 - diffusion_timestep_threshold) * K) to the end
        # (mf_diffusion.py:284), so at T=0.5:  K=100 -> 50 SLSQP solves per replan,
        # K=2 -> 1. That is why the first dpcc-r sweep cost ~15 s/replan and died at 11/30
        # rollouts against the 24 h cap (logs_in_develop/Gen14/U5/DA_20260804_*.md §6).
        # Safe to set here: flow_steps_v3 is in args_to_watch_mix_visual_plan ONLY, never in
        # ..._train, so _mix_plan_block's training-key mirror loop cannot clobber it and
        # diffusion_loadpath is unchanged — same checkpoint, new H8_K2_... results folder.
        # Override per run with `--flow-steps N` (U6); no config edit needed to sweep.
        'flow_steps_v3': 2,
        # MUST match the training block — both are checkpoint-path keys.
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
    })

base['plan_mix_visual_aligning_af'] = _mix_plan_block(
    'af', base['mix_visual_aligning_af'], {
        # ── U6 ── see the mf block above for the full rationale. Kept equal to mf's on
        # purpose: NFE is an operating point, and an mf-vs-af comparison at different K
        # would confound the objective with the step budget.
        'flow_steps_v3': 2,
        't_schedule': 'logit_normal',
        'p_mean': -0.4,
        'p_std': 1.0,
        'af_alpha_scheduler': 'sigmoid',
    })
