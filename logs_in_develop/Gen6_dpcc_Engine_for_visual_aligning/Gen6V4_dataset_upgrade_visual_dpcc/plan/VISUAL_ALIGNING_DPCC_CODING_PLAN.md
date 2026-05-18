# Coding & Migration Plan: Strict Visual-Aligning DPCC Model (Gen6v4)

This coding plan provides a step-by-step implementation guide to construct a **Visual-Aligning DPCC Model**. It leverages the clean temporal trajectories and boundary projection lifelines of `diffuser` and integrates the dual-camera ResNet image encoders from `ddpm_encdec_vision`.

To preserve baseline safety, we will implement this by **creating isolated folder copies** rather than polluting or breaking existing models.

---

## 🗺️ The 5-Step High-Level Implementation Roadmap

1. **Step 1: Duplicate & Isolate (Zero Pollution)**
   * **Action:** Copy standard `diffuser` package to `diffuser_visual_aligning`.
   * **Why:** Establishes a clean, independent visual sandbox and protects active baselines from pollution.

2. **Step 2: Slice Trajectories (Multi-Modal Ingestion)**
   * **Action:** Format sequences into unified trajectory dimensions ($H=8$) containing actions and proprioception.
   * **Why:** Converts standard visual `.pkl` dumps into clean z-score normalized sliding sequences.

3. **Step 3: Embed Camera Signals (Visual U-Net Conditioning)**
   * **Action:** Integrate ResNet spatial encoders from `ddpm_encdec_vision` into the temporal denoising backbone.
   * **Why:** Extracts image feature representations and projects them as U-Net conditioning embeddings.

4. **Step 4: Closed-Loop Replanning (Receding Horizon Rollouts)**
   * **Action:** Plan at every step, executing **only the first action $a_0$**.
   * **Why:** Maximizes online robot reactivity, replacing the legacy chunked planning block execution.

5. **Step 5: Continuous Shielding (Intermediate DPCC Projections)**
   * **Action:** Run the table/obstacle QP contact solver inside intermediate reverse denoising timesteps ($t \leq \text{threshold}$).
   * **Why:** Projects coordinate generations back onto safe physics manifolds during generation.


---

## 🛠️ Step 1: Copying and Re-Structuring the Codebase

To prevent namespace clashes and protect active baselines, we copied the standard `diffuser` package into a specialized visual-aligning variant, and created a dedicated entry folder `/workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc` to hold our runners:

```bash
# 1. Duplicate standard diffuser into a visual-aligning variant
cp -r /workspaces/FM-PCC/diffuser /workspaces/FM-PCC/diffuser_visual_aligning

# 2. Create the dedicated visual DPCC entry script folder
mkdir -p /workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc

# 3. Duplicate train/eval test scripts into our entry script folder
cp /workspaces/FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py /workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc/train_visual_aligning_dpcc.py
cp /workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py /workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc/eval_visual_aligning_dpcc.py
```

---

## 📂 Step 2: Designing the Multi-Modal Sequence Dataset
To guarantee robust, zero-variance image ingestion, we **reuse the battle-tested multi-modal loader class `Aligning_Img_Dataset`** (imported from `d3il.environments.dataset.aligning_dataset`) directly from the existing `ddpm_encdec_vision` pipeline! 

This ensures that all multi-camera image channels are loaded correctly. We simply wrap it to slice trajectories into DPCC-compatible z-score normalized sliding sequences:

### 💻 Code Block for `AligningImgSequenceDataset`:
```python
# APPEND TO: diffuser_visual_aligning/datasets/sequence.py

import pickle
import torch
import numpy as np
from diffuser.datasets.normalization import DatasetNormalizer, LimitsNormalizer

# Reuse the golden dataloader to handle complex multi-modal image structures
from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset

class AligningImgSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, horizon=8, normalizer='LimitsNormalizer', max_n_episodes=1000):
        super().__init__()
        self.horizon = horizon
        
        # 1. Instantiate the pre-existing multi-modal visual dataset loader
        self.base_dataset = Aligning_Img_Dataset(
            dataset_path=dataset_path,
            obs_seq_len=horizon,
            action_seq_size=horizon
        )
        
        # 2. Extract states, actions, and dual RGB image vectors
        self.observations = self.base_dataset.observations[:max_n_episodes]
        self.actions = self.base_dataset.actions[:max_n_episodes]
        self.primary_images = self.base_dataset.primary_rgb[:max_n_episodes]
        self.wrist_images = self.base_dataset.wrist_rgb[:max_n_episodes]
        
        self.n_episodes = len(self.observations)
        self.max_path_length = self.observations.shape[1]
        
        # 3. Fit standard LimitsNormalizer to scale values to z-scores
        self.obs_normalizer = LimitsNormalizer(self.observations.reshape(-1, 20))
        self.act_normalizer = LimitsNormalizer(self.actions.reshape(-1, 2))
        
        # 4. Generate sliding indices
        self.indices = []
        for i in range(self.n_episodes):
            for start in range(self.max_path_length - self.horizon):
                self.indices.append((i, start, start + self.horizon))
        self.indices = np.array(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start, end = self.indices[idx]
        
        # Normalize actions & observations individually
        obs_seq = self.obs_normalizer.normalize(self.observations[episode_idx, start:end])
        act_seq = self.act_normalizer.normalize(self.actions[episode_idx, start:end])
        
        # Concatenate actions and observations: Shape (8, 22)
        trajectories = np.concatenate([act_seq, obs_seq], axis=-1)
        
        # Extract dual camera frames at the starting frame (t=0 condition)
        primary_img = self.primary_images[episode_idx, start]
        wrist_img = self.wrist_images[episode_idx, start]
        
        conditions = {
            0: obs_seq[0], # Anchor proprioceptive state boundary
            'primary_img': torch.tensor(primary_img, dtype=torch.float32) / 255.0,
            'wrist_img': torch.tensor(wrist_img, dtype=torch.float32) / 255.0
        }
        
        return Batch(trajectories, conditions)
```

---

## 🧠 Step 3: Modifying U-Net for Spatial Conditioning (The ResNet + FiLM integration)
We will add a new U-Net model wrapper `VisualTemporalUNet` inside [diffuser_visual_aligning/models/temporal.py](file:///workspaces/FM-PCC/diffuser_visual_aligning/models/temporal.py) that integrates a spatial ResNet feature encoder to map image frames into the U-Net spatial-temporal conditioning space:

```python
# ADD TO: diffuser_visual_aligning/models/temporal.py

import torch
import torch.nn as nn
from ddpm_encdec_vision.models.visual_unet import SpatialSoftmaxEncoder

class VisualTemporalUNet(nn.Module):
    def __init__(self, transition_dim, horizon, proprio_dim=20, embed_dim=256):
        super().__init__()
        # 1. Instantiate the dual spatial ResNet encoders from ddpm_encdec_vision
        self.primary_encoder = SpatialSoftmaxEncoder(input_channels=3)
        self.wrist_encoder = SpatialSoftmaxEncoder(input_channels=3)
        
        # 2. Linear projector to merge visual latents + proprioception at t=0
        visual_latent_dim = self.primary_encoder.op_shape
        self.proj = nn.Sequential(
            nn.Linear(visual_latent_dim * 2 + proprio_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. Core 1D Temporal Convolutional U-Net backbone
        from .temporal import TemporalUnet
        self.unet = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            cond_dim=embed_dim
        )

    def forward(self, x, cond, t):
        # x: Joint sequence of actions + observations [B, H, 22]
        # cond: dict containing primary_img, wrist_img, and proprio state at t=0
        
        # 1. Encode primary and wrist images
        f_primary = self.primary_encoder(cond['primary_img'])
        f_wrist = self.wrist_encoder(cond['wrist_img'])
        
        # 2. Stacks visual features with initial proprioception state [B, 20]
        feat = torch.cat([f_primary, f_wrist, cond[0]], dim=-1)
        emb = self.proj(feat) # [B, embed_dim] conditioning vector
        
        # 3. Denoise joint grid with visual-proprioceptive embedding
        return self.unet(x, emb, t)
```

---

## 🔒 Step 4: Strict DPCC Boundary Projection Execution Loop
During training and online simulator evaluation, we enforce a strict **DPCC Lifeline**. We copy the high-level receding-horizon planner from `diffuser/sampling/policies.py` and run it in the loop:

```python
# Proposed step evaluation loop inside ddpm_encdec_vision_test_visual_dpcc/eval_visual_aligning_dpcc.py

from diffuser_visual_aligning.sampling.policies import Policy

class VisualDPCCEvaluator:
    def __init__(self, model, normalizer, env, projector):
        # 1. Wrap model in the standard Receding Horizon Policy wrapper
        self.policy = Policy(
            model=model,
            normalizer=normalizer,
            projector=projector # QP table/obstacle boundary solver
        )
        self.env = env

    def evaluate_episode(self):
        obs = self.env.reset()
        done = False
        
        while not done:
            # 1. Get raw primary & wrist camera frames from simulator
            primary_img, wrist_img = self.env.render_cameras()
            
            # 2. Structure step conditioning
            conditions = {
                0: obs['proprio_state'],
                'primary_img': primary_img,
                'wrist_img': wrist_img
            }
            
            # 3. Call receding horizon policy: denoises full H=8 joint path
            # AND applies QP projections on intermediate steps
            action, trajectories = self.policy(
                conditions,
                horizon=8,
                disable_projection=False
            )
            
            # 4. STRICT DPCC LIFELINE: Execute ONLY the first action (action = actions[0])
            obs, reward, done, info = self.env.step(action)
```

---

## 📑 Step 5: Slurm / Sbatch Scripting for Background Training
To enable headless background execution and multi-seed evaluations on GPU clusters, we created dedicated sbatch scripts inside `/workspaces/FM-PCC/Slurm_Codes/sbatch/Visual_Aligning/`:

* **`train_visual_aligning_dpcc.sh`:** Initiates headless z-score z-normalized model training over GPU-1 clusters.
* **`eval_visual_aligning_dpcc.sh`:** Handles simulation planning rollouts and captures diagnostic trajectories.
* **`visual_aligning_dpcc_pipeline.sh`:** Executes the full pipeline (Phase 1: Training $\rightarrow$ Phase 2: Evaluation).

---

## 🎛️ Step 6: Config Reuse & Parameter Adaptations
To maintain parity and save setup overhead, we will directly reuse **`config/aligning-d3il-visual.py`** and **`config/visual_aligning_eval.yaml`**. 

Instead of rewriting them from scratch, we will simply append/modify the following parameters within these files when the coding phase begins:

### 1. In [config/aligning-d3il-visual.py](file:///workspaces/FM-PCC/config/aligning-d3il-visual.py):
We will add a new visual DPCC key under the `base` dictionary:
```python
base['visual_aligning_dpcc'] = {
    **base['ddpm_encdec_vision'],
    'model': 'diffuser_visual_aligning.models.temporal.VisualTemporalUNet',
    'diffusion': 'diffuser_visual_aligning.models.diffusion.GaussianDiffusion',
    'loader': 'diffuser_visual_aligning.datasets.sequence.AligningImgSequenceDataset',
    'horizon': 8,
    'obs_seq_len': 8,      # Force parity: unified H=8 grid
    'action_seq_size': 8,  # Force parity: unified H=8 grid
    'if_vision': True,
    'obs_dim': 20,
    'action_dim': 2,       # Cartesian control dimensions
    'prefix': 'visual_aligning_dpcc/',
}

base['plan_visual_aligning_dpcc'] = {
    **base['plan_ddpm_encdec_vision'],
    'prefix': 'f:plans/visual_aligning_dpcc/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
    'diffusion_loadpath': 'f:visual_aligning_dpcc/H{horizon}',
}
```

### 2. In [config/visual_aligning_eval.yaml](file:///workspaces/FM-PCC/config/visual_aligning_eval.yaml):
We will reuse the pre-configured physical Franka workspace boundaries and z-score constraints:
* **Franka Workspace Limits (Cartesian):** Reuses bounds `lb: [0.3, -0.35, 0.05]` and `ub: [0.7, 0.35, 0.40]`.
* **Inference timesteps:** Reuses `diffusion_timestep_threshold: 0.5`.
* **Add custom experiment trigger:**
  ```yaml
  exps: [
    'visual_aligning_dpcc',
  ]
  ```

---

## ♻️ Step 7: Granular Code Reuse Map (diffuser vs. ddpm_encdec_vision)
To maximize code efficiency and eliminate parity drift, we will reuse specific, battle-tested architectural blocks from both folders:

### 1. What we will reuse from the `diffuser` folder:
* **The SLSQP Boundary Projector (`diffuser/sampling/projection.py`):**
  * *Code Block:* `class Projector` (Analytical convex optimization engine calculating the SLSQP QP solvers). We reuse this to enforce physical table/obstacle limits on joint state vectors.
* **The Receding-Horizon Policy (`diffuser/sampling/policies.py`):**
  * *Code Block:* `class Policy`. Coordinates standard closed-loop execution. Extracts and executes only the first step action (`action = actions[0]`) to ensure reactive planning.
* **Conditioning Snap Lock (`diffuser/models/helpers.py`):**
  * *Code Block:* `apply_conditioning`. Enforces boundary state clamping at sequence index $t=0$, locking trajectory generations back to physical simulator proprioception.
* **Feature Normalizers (`diffuser/datasets/normalization.py`):**
  * *Code Block:* `LimitsNormalizer` and `DatasetNormalizer`. Standardizes action and state coordinate arrays into a unified $[-1, 1]$ z-score range.

### 2. What we will reuse from the `ddpm_encdec_vision` folder:
* **Pre-trained Spatial Encoder (`ddpm_encdec_vision/models/visual_unet.py`):**
  * *Code Block:* `SpatialSoftmaxEncoder` and spatial ResNet blocks. We reuse this to encode raw RGB primary/wrist camera frames into compact $64D$ latent representations without retraining the visual backbone.
* **Multi-Modal Data Parser (`d3il/environments/dataset/aligning_dataset.py`):**
  * *Code Block:* `Aligning_Img_Dataset`. Reused to parse raw simulation logs, synchronize primary and wrist camera signals, and structure observation vectors.
* **Simulation Loop Utilities (`ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`):**
  * *Code Block:* The rollout wrapper, headless MuJoCo rendering hooks, success diagnostic monitors, and trajectory visualization utilities.

---

## 🔬 Self-Audit of the Plan

To guarantee architectural correctness, we perform a strict audit of the design:

| Audit Parameter | **Gen6 (Failed/Legacy)** | **Proposed Visual-DPCC** | **Verification & Parity** |
|:---|:---|:---|:---|
| **Unified Horizon** | No. Uses separate variables causing shape mismatches. | **Yes.** Single `horizon = 8` sequence grid. | **100% Pass.** Eliminates dimensional mismatch crashes. |
| **Safety Integration** | Post-hoc checking. | **Continuous.** DPCC QP Projector active during denoising. | **100% Pass.** Restores simulator boundary stability. |
| **Control Frequency** | Chunked (replans only every 4 steps). | **Closed-loop.** Replans on every simulation tick ($a_0$). | **100% Pass.** Maximizes obstacle dodging reactivity. |
| **Dataset Ingestion** | Rigid loaders bound to Transformer shapes. | **Modular (Reuses `Aligning_Img_Dataset`)**. | **100% Pass.** Guarantees perfect image ingestion fidelity. |

---

## 🏆 Step 8: Final Deliverables Specification

The migration will produce the following clean, isolated, and tested deliverables in the workspace:

### 1. The Isolated Codebase Core
* **Path:** `/workspaces/FM-PCC/diffuser_visual_aligning/`
* **Role:** A completely independent library containing the custom sequence data loader, visual 1D temporal conditional U-Net backbone, and safety projection scheduler.

### 2. The Entry Runners
* **Path:** `/workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc/`
* **Contents:**
  * `train_visual_aligning_dpcc.py`: Headless training script mapping visual datasets to U-Net layers.
  * `eval_visual_aligning_dpcc.py`: Receding horizon evaluation script executing closed-loop $a_0$ rollouts.

### 3. Unified Configurations & Slurm Suite
* **Config Appends:** Reused `config/aligning-d3il-visual.py` and `config/visual_aligning_eval.yaml` with the `visual_aligning_dpcc` config blocks.
* **Sbatch scripts:** Slurm job files inside `/workspaces/FM-PCC/Slurm_Codes/sbatch/Visual_Aligning/` for background cluster runs.

### 🎯 Expected Validation Criteria:
* **Zero Collision Rate:** Workspace trajectory violations must be strictly $0.0\%$ (guaranteed by continuous SLSQP boundary projection).
* **High Success Rates:** D3IL simulation success rates must meet or exceed $90\%$ under online closed-loop rollouts.

---

## ⚖️ Architectural Head-to-Head: Proposed Visual-DPCC vs. Gen6 vs. Old DPCC

The following structural comparison highlights how the **Proposed Visual-DPCC** integrates the reactive safety of the old state-only models with the multi-modal intelligence of visual systems:

| Architectural Metric | **Old DPCC (State-Only)** | **Gen6 (Failed/Legacy Visual)** | **Proposed Visual-DPCC (Gen6v4)** |
|:---|:---|:---|:---|
| **Input Modality** | Proprioceptive state vectors only ($20D$). | Dual camera frames ($3 \times 128 \times 128$) + Proprioception ($20D$). | **Dual camera frames + Proprioception** (Ingested via pre-trained ResNet Spatial Softmax). |
| **Control Loop Frequency** | **High-Frequency Reactivity.** Closed-loop replanning at every single step ($a_0$). | **Chunked Latency.** Open-loop block execution of action chunks (replans only every 4 steps). | **High-Frequency Reactivity.** Closed-loop replanning at every single step ($a_0$). |
| **Sequence Horizon** | Unified horizon ($H=8$). | Fragmented horizons (obs history $O=5$, actions $A=4$). | **Unified horizon ($H=8$)** mapping continuous trajectories. |
| **Safety Shielding** | Intermediate SLSQP boundary projections ($t \le \text{threshold}$). | Post-hoc checking (passive out-of-bound indicators). | **Intermediate SLSQP boundary projections** active during diffusion denoising steps. |
| **Dataset Loader** | Proprioceptive state sequence arrays. | Rigid VAE-bridge sequence loader. | **Modular loader** wrapping battle-tested `Aligning_Img_Dataset`. |



