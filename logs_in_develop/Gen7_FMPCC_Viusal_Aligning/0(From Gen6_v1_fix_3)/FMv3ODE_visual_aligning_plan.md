# ⚡ Gen7 Flow Matching Upgrade: FMv3ODE Integration Plan
## ═══════════════════════════════════════════════════════════════════════
> [!IMPORTANT]
> This plan establishes a clean scientific framework to integrate **Flow Matching (FMv3ODE)** into the Visual-Aligning workspace.
>
> We achieve this by duplicating the legacy DDPM folders into new dedicated Flow Matching sibling directories, adapting the model classes to inherit from the `flow_matcher_v3_ode_selectable` engine, adding matching configuration profiles directly inside `config/aligning-d3il-visual.py`, and preparing a complete duplicate Slurm suite. All legacy paths and directories remain fully functional.

---

## 📂 Step 1: Duplicate DDPM Folders (Completed)

We copy the core model package and test suites:
1. `ddpm_encdec_vision` ➔ `fm_encdec_vision`
2. `ddpm_encdec_vision_test` ➔ `fm_encdec_vision_test`

```bash
cp -r ddpm_encdec_vision fm_encdec_vision
cp -r ddpm_encdec_vision_test fm_encdec_vision_test
```

---

## 🧠 Step 2: Code Modifications inside Sibling Folders

### 1. Retaining Backbone in `fm_encdec_vision/models/visual_unet.py`
We retain the same temporal U-Net backbone (`UNet1DTemporalCondModel` from `diffuser.models.unet1d_temporal_cond`) rather than switching to `Flow_matcher_U_Net_v2`. This is structurally crucial because it allows the model to leverage the FiLM projection mechanism (`use_cond_projection=True`), which projects visual features directly into the temporal convolutions. The continuous-time Flow Matching engine (`flow_matcher_v3_ode_selectable.models.diffusion.GaussianDiffusion`) is fully compatible with this backbone class.

```python
# 2. Instantiate Backbone (Standard DDPM UNet)
from diffuser.models.unet1d_temporal_cond import UNet1DTemporalCondModel
backbone_class = UNet1DTemporalCondModel
```

### 2. Rewiring Continuous-Time Engine in `fm_encdec_vision/models/visual_gaussian_diffusion.py`
We rewrite the class `VisualGaussianDiffusion` to inherit from the Flow Matching base `GaussianDiffusion` instead of diffuser's standard DDPM `GaussianDiffusion`:

```python
import torch
# Sourced from the robust FMv3ODE selectable package
from flow_matcher_v3_ode_selectable.models.diffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides Flow Matching GaussianDiffusion to handle vision-specific batch format,
    continuous-time Beta distribution sampling, and image token conditioning.
    """
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Flow Matching continuous-time training loss:
        Linear interpolation path x_t = (1-t)*x_base + t*x_start.
        """
        # Trajectory x: [batch, horizon, transition_dim] (act: 3D, obs: 3D)
        x = torch.cat([act, obs], dim=-1)
        
        # Condition dict containing image tokens and snapping key
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),
            0: obs[:, 0]  # Snapping boundaries
        }
        
        # Draw continuous time t from Beta(alpha, beta) distribution
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        t = beta_dist.sample((batch_size,))
        t = 1.0 - t  # D3IL standard continuous shift
        
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        """
        Inference: Triggers the Flow Matching ODE/Euler integration loop (p_sample_loop).
        Handles vision-specific tuple unpacking and batch repetition.
        """
        # 1. Unpack tuples passed from Aligning_Sim.test_agent()
        if 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, pos = cond[0]
            
            # Create a clean 'visual' cond for VisualUNet
            visual_cond = (bp_imgs, inhand_imgs, pos)
            # Snapping context at t=0 (last frame in history)
            snapping_cond = {0: pos[:, -1]} 
            
            new_cond = snapping_cond.copy()
            new_cond['visual'] = visual_cond
        else:
            new_cond = cond

        # Forward integration: t=0 → t=1 (noise → action plan)
        return super().forward(new_cond, *args, **kwargs)
```

### 3. Update Package Imports in Executables (`fm_encdec_vision_test/`)
We replace all legacy imports targeting `ddpm_encdec_vision` with the new sibling package `fm_encdec_vision`:
- In `fm_encdec_vision_test/train_fm_encdec_vision.py`:
  ```python
  import fm_encdec_vision.utils as utils
  # ... and ensure the default dataset matches 'aligning-d3il-visual'
  ```
- In `fm_encdec_vision_test/eval_fm_encdec_vision.py`:
  ```python
  import fm_encdec_vision.utils as utils
  ```
- In `fm_encdec_vision_test/load_results_fm_encdec_vision.py`:
  ```python
  import fm_encdec_vision.utils as utils
  ```

---

## 🛠️ Step 3: Append Configuration Keys (`config/aligning-d3il-visual.py`)

We add the active Flow Matching watch lists (`args_to_watch_fmv3_ode_train` and `args_to_watch_fmv3_ode_plan`) and their respective configuration profiles `fm_encdec_vision` and `plan_fm_encdec_vision` inside `config/aligning-d3il-visual.py`:

```python
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

base = {
    ...
    'fm_encdec_vision': {
        # ======================================================================================
        # 🔑 KEY FLOW MATCHING MODEL PARAMETERS
        # ======================================================================================
        'model': 'fm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge',
        'diffusion': 'fm_encdec_vision.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
        'action_dim': 3,
        'obs_seq_len': 5,
        'action_seq_size': 4,
        'horizon': 8,
        'window_size': 8,
        
        # --- Continuous Time Flow Matching Parameters ---
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        'flow_steps_v3': 16,
        'ode_inference_steps_v3': 16,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        
        'n_diffusion_steps': 16,
        'action_weight': 10,
        'loss_type': 'l2',
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'condition_dropout': 0.25,
        
        'obs_dim': 128,
        'visual_input': True,
        'loss_discount': 1.0,
        'returns_condition': False,
        'predict_epsilon': True,
        'dynamic_loss': False,
        
        # --- Dataset loader ---
        'max_path_length': 512,
        'loader': 'ignored',
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'include_returns': True,
        
        # --- Serialization ---
        'logbase': logbase,
        'prefix': 'fm_encdec_vision/',
        'exp_name': watch(args_to_watch_fmv3_ode_train),
        
        # --- Optimization ---
        'batch_size': 64,
        'learning_rate': 2e-4,
        'ema_decay': 0.995,
        'n_steps_per_epoch': 1000,
        'n_train_steps': 1e5,
        'gradient_accumulate_every': 2,
        'train_test_split': 0.9,
        'device': 'cuda',
        'seed': 0,
    },

    'plan_fm_encdec_vision': {
        # ======================================================================================
        # 🎮 INFERENCE PLANNING AND SOLVER SETTINGS
        # ======================================================================================
        'horizon': 8,
        'window_size': 8,
        'max_episode_length': 1000,
        'max_path_length': 512,
        'action_weight': 10,
        
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
        
        # Continuous-Time solver parameters
        'time_beta_alpha_v3': 1.5,
        'time_beta_beta_v3': 1.0,
        'flow_steps_v3': 16,
        'ode_solver_backend_v3': 'legacy_euler',
        'ode_solver_method_v3': 'euler',
        'ode_solver_rtol_v3': None,
        'ode_solver_atol_v3': None,
        'ode_solver_step_size_v3': None,
        
        'diffusion_loadpath': 'f:fm_encdec_vision/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}',
        'value_loadpath': 'f:values/H{horizon}',
        'diffusion_epoch': 'best',
        'verbose': False,
        'suffix': '0',
    },
```
```

---

## 🚀 Step 4: Duplicate & Adapt Slurm Batch Codes (`Slurm_Codes/sbatch/Visual_Aligning/`)

We duplicate the legacy Slurm submission scripts under `/workspaces/FM-PCC/Slurm_Codes/sbatch/Visual_Aligning/` to target Flow Matching:

### 1. `visual_aligning_pipeline_fm.sh`
```bash
#!/bin/bash
#SBATCH --job-name=visual_pipeline_fm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

set -e

SBATCH_DIR="Slurm_Codes/sbatch/Visual_Aligning"
DATE=${SUBMIT_DATE:-$(date +%Y-%m-%d)}
TIME=${SUBMIT_TIME:-$(date +%H_%M_%S)}
LOG_DIR="Slurm_Codes/logs/$DATE"
mkdir -p "$LOG_DIR"
LOG_OPTS="--output=$LOG_DIR/${TIME}_%x_%j.log --error=$LOG_DIR/${TIME}_%x_%j.log"

TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_visual_aligning_fm.sh")
echo "Step 1: Flow Matching Training submitted. Job ID: $TRAIN_ID"

EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_visual_aligning_fm.sh")
echo "Step 2: Flow Matching Evaluation scheduled. Job ID: $EVAL_ID"
```

### 2. `train_visual_aligning_fm.sh`
```bash
#!/bin/bash
#SBATCH --job-name=train_visual_fm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1
#SBATCH --gres=gpu:1

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python fm_encdec_vision_test/train_fm_encdec_vision.py --seeds 6 --use-wandb --wandb-project FMPCC-visual-aligning
```

### 3. `eval_visual_aligning_fm.sh`
```bash
#!/bin/bash
#SBATCH --job-name=eval_visual_fm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-1
#SBATCH --gres=gpu:1

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python fm_encdec_vision_test/eval_fm_encdec_vision.py --seed 6
```

### 4. `load_results_visual_aligning_fm.sh`
```bash
#!/bin/bash
#SBATCH --job-name=load_results_visual_fm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-1-student

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python fm_encdec_vision_test/load_results_fm_encdec_vision.py
```
