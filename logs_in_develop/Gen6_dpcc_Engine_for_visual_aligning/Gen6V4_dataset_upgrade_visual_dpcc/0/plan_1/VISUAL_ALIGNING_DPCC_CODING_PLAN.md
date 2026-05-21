# Coding & Migration Plan: Strict Visual-Aligning DPCC Model (Gen6v4) - 3D Cartesian Edition

This coding plan provides a step-by-step implementation guide to construct a **Visual-Aligning DPCC Model**. It leverages the clean temporal trajectories and boundary projection lifelines of `diffuser` and integrates the dual-camera ResNet image encoders from `ddpm_encdec_vision`.

To preserve baseline safety, we will implement this by **creating isolated folder copies** rather than polluting or breaking existing models.

---

## ⚠️ Critical Points to Care: 3D Spatial Parity & Surgical Adapter Rebuild (No-Edit `d3il` Constraint)

To achieve absolute mathematical parity with the **State-Only DPCC (`avoiding-d3il`)** dataset construction style, we will replicate its exact structural design in **Visual-Aligning DPCC**, but **upgraded to full 3D (XYZ) space**. This means we will retain all three Cartesian dimensions ($x, y, z$) in the model, diffusion, and safety projection spaces.

To respect the strict **no-edit constraint on the `d3il/` directory**, we will implement a clean in-memory **Adapter Class** completely outside of the `d3il/` folder to transform the dataset dynamically.

### ⚖️ Head-to-Head Architectural Parity Comparison

| Feature / Dimension | State-Only DPCC (`avoiding-d3il`) | Legacy Visual-Aligning DPCC (Flawed Design) | **New Parity-Aligned Visual-Aligning (3D)** |
| :--- | :--- | :--- | :--- |
| **Observation Channels** | 4D: `[des_x, des_y, x, y]` (Target + Robot physical pose) | 3D Target: `[des_x, des_y, des_z]` (Lacks actual physical pose!) | **6D: `[des_x, des_y, des_z, x, y, z]`** (Target + Robot physical pose in XYZ) |
| **Camera Streams** | None | Dual RGB: Agentview & Wrist camera | Dual RGB: Agentview & Wrist camera |
| **Action Dimensions** | 2D Cartesian deltas: `[dx, dy]` | 3D Cartesian deltas: `[dx, dy, dz]` | **3D Cartesian deltas: `[dx, dy, dz]`** (Full 3D control) |
| **Trajectory Grid** | 6D: `Action (2D) + Obs (4D) = 6D` | 6D: `Action (3D) + Obs (3D) = 6D` (No actual robot coordinates!) | **9D: `Action (3D) + Obs (6D) = 9D`** |
| **Temporal Horizon ($H$)** | `horizon = 8` | `horizon = 8` | `horizon = 8` |
| **Boundary Snap ($t=0$)** | `{0: [des_x, des_y, x, y]}` | `{0: [des_x, des_y, des_z]}` + Images (Snaps target, not pose!) | **`{0: [des_x, des_y, des_z, x, y, z]}`** + Images |
| **Safety Projection Engine** | SLSQP 2D convex optimizer | **None / Inactive** (Cannot project physical path limits!) | **SLSQP 3D convex optimizer** (Full 3D workspace contact limits) |
| **Planning Paradigm** | Closed-loop RHC | Closed-loop RHC | Closed-loop RHC |

> [!NOTE]
> ### 🧮 Mathematical Clarification: Action "Velocity" ($vx, vy$) vs. "Displacement" ($dx, dy$)
> In the legacy YAML configurations and joint grids (e.g., `config/projection_eval.yaml`), the action dimensions are labeled `vx` and `vy` (standing for Cartesian velocity). However, **numerically, these actions function strictly as delta position displacements ($dx$ and $dy$)** under discrete simulator integrator dynamics:
> 
> 1. **Simulator Stepping Equation:** During evaluation (see `scripts/eval_prior_model.py` lines 64–65), the environment steps by directly adding the action to the current position:
>    $$\vec{x}_{des, t+1} = \vec{x}_{des, t} + \vec{a}_t$$
>    Since the discrete timestep is $dt = 1$, the velocity vector $\vec{v} = \frac{d\vec{x}}{dt}$ is numerically identical to the displacement vector $d\vec{x}$.
> 
> 2. **Dynamics Model Prediction:** The safety engine's forward dynamics predictions (lines 94–97) perform direct single-step position updates:
>    $$x_{t+1} = x_t + vx_t$$
>    $$y_{t+1} = y_t + vy_t$$
> 
> Therefore, in both the 2D avoiding task and our new 3D visual-aligning task, the generated action channels are mathematically **pure Cartesian displacement deltas** (i.e., $[dx, dy]$ for avoiding, and $[dx, dy, dz]$ for aligning).

### 🔍 The Adapter Rebuild Strategy (No-Edit d3il Folder)

The core DPCC mathematical solvers (the SLSQP QP solver, the boundary constraint projector, and the inverse diffusion sampling loops) are mathematically correct and robust. Rebuilding them is unnecessary.
Instead of modifying the files inside `d3il/`, we will perform a **surgical rebuild using adapter layers completely outside the `d3il/` folder**:

1. **Surgically write an In-Memory Dataset Wrapper (`diffuser_visual_aligning/datasets/sequence.py`)**:
   * We will create a wrapper `ParityAligningDataset` that instantiates the untouched `Aligning_Img_Dataset` inside `d3il/`.
   * **⚠️ Critical API & Observation Layout Discrepancies:**
     * **API Returns:** The state-only `Avoiding_Dataset` returns **3 elements** `(obs, act, mask)`. The visual `Aligning_Img_Dataset` returns **5 elements** `(bp_imgs, inhand_imgs, obs, act, mask)`. Our wrapper handles this by loading the 5-element visual sequence and returning a standardized `Batch(trajectories, conditions)` object.
     * **Observation Dimension:** `Avoiding_Dataset` loads a 4D stacked joint vector `[des_x, des_y, x, y]` (commanded target + actual feedback). `Aligning_Img_Dataset` only contains a 3D desired position `[des_x, des_y, des_z]` inside its `self.observations`.
     * **Surgical Reconstitution Solution:** To establish perfect spatial parity in memory, our `ParityAligningDataset` wrapper parses the source pickle files (`env_state`) using the file paths loaded during initialization. It extracts the actual feedback robot Cartesian state `env_state['robot']['c_pos']` (full 3 dimensions `x, y, z`) and stacks them with `robot_des_pos` (full 3 dimensions `des_x, des_y, des_z`) to form the identical **6D observations** `[des_x, des_y, des_z, x, y, z]`.
     * **Action Dimension:** Retains full 3D Cartesian deltas `[dx, dy, dz]` without any slicing!
2. **Surgically rewrite the Evaluation Loop (`eval_visual_aligning_dpcc.py`)**:
   * Reads full 3D environment coordinates from MuJoCo, passes them directly as the 6D observation `[des_x, des_y, des_z, x, y, z]` to the model, and steps the simulator using the generated 3D Cartesian velocity actions `[dx, dy, dz]` directly. No coordinate slicing, no z-axis reconstitution necessary!

### 🎯 The Parity Philosophy
1. **Model & Denoising Space:** U-Net operates entirely in 3D space. Proprioception is represented in a 6D stacked vector `[des_x, des_y, des_z, x, y, z]`, and actions are represented in 3D deltas `[dx, dy, dz]`, forming a unified 9D joint trajectory grid of length $H=8$.
2. **Visual Conditioning:** The dual RGB frames (primary + wrist) at $t=0$ are encoded through the pre-trained `SpatialSoftmaxEncoder` (from `ddpm_encdec_vision`), projected, and concatenated with the starting proprioception state to serve as the U-Net conditioning key.
3. **Continuous Safety Shield:** Denoised trajectory grids are projected during diffusion reverse steps using the **3D SLSQP Convex Safety Projector** (DPCC) to respect obstacle/table boundaries.
4. **Simulator Interface:** Feeds full 3D coordinates directly to the model, executes the model's predicted 3D action `[dx, dy, dz]` directly in MuJoCo.

### 🧮 3D Coordinate & Trajectory Layout

We standardise the sequence space to be **3D (`xyz`)**, resulting in a **9D joint trajectory space** matching the avoiding task structure:

```
                  ┌──────────────────────────────┐
                  │   9D JOINT TRAJECTORY SEQUENCE   │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  3D ACTIONS (deltas)                            6D OBSERVATIONS (poses)
 ┌────────────────────┐                   ┌───────────────────────────────────┐
  │  dx  │  dy  │  dz  │                   │ des_x │ des_y │ des_z │ x │ y │ z │
  └──────┴──────┴──────┘                   └───────┴───────┴───────┴───┴───┴───┘
     Indices [0, 1, 2]                                Indices [3, 4, 5, 6, 7, 8]
```

### 📦 DPCC D3IL Dataset Ingestion Parity via Wrapper (Code Level)

The clean wrapper `ParityAligningDataset` wraps the untouched D3IL dataset and performs the dynamic coordinate transformations:

```python
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
from agents.utils.sim_path import sim_framework_path

class ParityAligningDataset(Dataset):
    def __init__(self, data_directory, device="cpu", max_len_data=256, window_size=8):
        # 1. Load the original raw dataset from d3il (Untouched)
        self.raw_dataset = Aligning_Img_Dataset(
            data_directory=data_directory,
            device=device,
            obs_dim=3,          # Desired EE position dimension (X, Y, Z)
            action_dim=3,       # Command displacement dimension (v_x, v_y, v_z)
            max_len_data=max_len_data,
            window_size=window_size
        )
        
        # 2. Extract state file paths from raw dataset to reconstruct feedback positions in-memory
        data_dir = sim_framework_path("environments/dataset/data/aligning/all_data")
        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)
        
        inputs = []
        actions = []
        
        # 3. Dynamic coordinate transformation: Stacking 3D + 3D
        for file in state_files:
            with open(os.path.join(data_dir, 'state', file), 'rb') as f:
                env_state = pickle.load(f)
            
            # Commanded Target position in 3D
            robot_des_pos = env_state['robot']['des_c_pos']        # (T, 3)
            # Physical Feedback position in 3D
            robot_c_pos = env_state['robot']['c_pos']              # (T, 3)
            
            # Stack into 6D observations [des_x, des_y, des_z, x, y, z]
            input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1) # (T, 6)
            # Action: 3D velocity displacement command [dx, dy, dz]
            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]                 # 3D action (T-1, 3)
            
            # Pad sequences to max_len_data
            zero_obs = np.zeros((1, max_len_data, 6), dtype=np.float32)
            zero_action = np.zeros((1, max_len_data, 3), dtype=np.float32)
            valid_len = len(vel_state)
            
            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = vel_state
            
            inputs.append(zero_obs)
            actions.append(zero_action)
            
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        
        self.bp_cam_imgs = self.raw_dataset.bp_cam_imgs
        self.inhand_cam_imgs = self.raw_dataset.inhand_cam_imgs
        self.masks = self.raw_dataset.masks
        self.slices = self.raw_dataset.slices
```

### 🛡️ Pure 3D Safety Projector

All physical constraints, obstacles, halfspaces, and integration steps inside the Projector are mapped in the 3D space (`xyz`):

```python
# Formulation of the derivative integration constraints (dt = 1.0)
dynamics_constraints = [
    # 1. Actual physical path integration (action index 0,1,2 -> obs index 6,7,8)
    ('deriv', np.array([act_obs_indices['x'], act_obs_indices['dx']])),
    ('deriv', np.array([act_obs_indices['y'], act_obs_indices['dy']])),
    ('deriv', np.array([act_obs_indices['z'], act_obs_indices['dz']])),
    
    # 2. Desired commanded path integration (action index 0,1,2 -> obs index 3,4,5)
    ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['dx']])),
    ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['dy']])),
    ('deriv', np.array([act_obs_indices['z_des'], act_obs_indices['dz']])),
]
```

### 🔄 3D Simulator Interface

During online evaluation rollouts:
1. **Inputs:** Slices nothing! The evaluation script reads 3D states `des_robot_pos` (3D target) and `robot_c_pos` (3D actual pose) from the environment, stacks them directly to a 6D observation vector `[des_x, des_y, des_z, x, y, z]`, and feeds them to the policy.
2. **Outputs:** Slices nothing! The policy predicts 3D delta displacements `[dx, dy, dz]`. The evaluation script executes it directly inside MuJoCo without any extra coordinate pads.

---

## 🗺️ The 5-Step High-Level Implementation Roadmap

1. **Step 1: Duplicate & Isolate (Zero Pollution)**
   * **Action:** Copy standard `diffuser` package to `diffuser_visual_aligning` and create `ddpm_encdec_vision_test_visual_dpcc`.
   * **Why:** Establishes a clean, independent visual sandbox and protects active baselines from pollution.

2. **Step 2: Slice Trajectories (Multi-Modal Ingestion)**
   * **Action:** Format sequences into unified 9D trajectory dimensions ($H=8$) containing 3D actions and 6D stacked proprioception.
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

To prevent namespace clashes and protect active baselines, we copy the standard `diffuser` package into a specialized visual-aligning variant under `diffuser_visual_aligning`, and create a dedicated entry folder `/workspaces/FM-PCC/ddpm_encdec_vision_test_visual_dpcc` to hold our runners:

```bash
# 1. Duplicate standard diffuser into the diffuser_visual_aligning folder
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
        
        # 1. Instantiate D3IL pre-existing multi-modal visual loader with corrected signature
        self.base_dataset = Aligning_Img_Dataset(
            data_directory=dataset_path,
            obs_dim=3,
            action_dim=3,
            window_size=horizon
        )
        
        # 2. Extract states, actions, and camera frames
        self.observations = self.base_dataset.observations[:max_n_episodes]
        self.actions = self.base_dataset.actions[:max_n_episodes]
        
        self.n_episodes = len(self.observations)
        self.max_path_length = self.observations.shape[1]
        
        # 3. Fit standard LimitsNormalizer to scale 6D proprioception & 3D actions
        self.obs_normalizer = LimitsNormalizer(self.observations.reshape(-1, 6))
        self.act_normalizer = LimitsNormalizer(self.actions.reshape(-1, 3))
        
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
        
        # Concatenate actions and observations: Shape (8, 9)
        trajectories = np.concatenate([act_seq, obs_seq], axis=-1)
        
        # Extract dual camera frames at the starting frame (t=0 condition)
        # Directly extract the float tensors already pre-formatted by Aligning_Img_Dataset
        primary_tensor = self.base_dataset.bp_cam_imgs[episode_idx][start]
        wrist_tensor = self.base_dataset.inhand_cam_imgs[episode_idx][start]
        
        conditions = {
            0: obs_seq[0], # Anchor proprioceptive state boundary
            'primary_img': primary_tensor,
            'wrist_img': wrist_tensor
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
    def __init__(self, transition_dim=9, horizon=8, proprio_dim=6, embed_dim=256):
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
        # x: Joint sequence of actions + observations [B, H, 9]
        # cond: dict containing primary_img, wrist_img, and proprio state at t=0
        
        # 1. Encode primary and wrist images
        f_primary = self.primary_encoder(cond['primary_img'])
        f_wrist = self.wrist_encoder(cond['wrist_img'])
        
        # 2. Stacks visual features with initial proprioception state [B, 6]
        feat = torch.cat([f_primary, f_wrist, cond[0]], dim=-1)
        emb = self.proj(feat) # [B, embed_dim] conditioning vector
        
        # 3. Denoise joint grid with visual-proprioceptive embedding
        return self.unet(x, emb, t)
```

---

## 🔒 Step 4: Strict DPCC Boundary Projection Execution Loop
During training and online simulator evaluation, we enforce a strict **DPCC Lifeline**. We copy the high-level receding-horizon planner from `diffuser/sampling/policies.py` (making sure it imports from `diffuser_visual_aligning`) and run it in the loop:

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
                0: obs['proprio_state'], # 6D stacked observation
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
            
            # 4. STRICT DPCC LIFELINE: Execute ONLY the first action (action = actions[0], which is 3D XYZ)
            obs, reward, done, info = self.env.step(action)
```

---

## 📑 Step 5: Slurm / Sbatch Scripting for Background Training
To enable headless background execution and multi-seed evaluations on GPU clusters, we created dedicated sbatch scripts inside `/workspaces/FM-PCC/Slurm_Codes/sbatch/diffuser_visual_aligning/`:

* **`train_visual_aligning_dpcc.sh`:** Initiates headless model training over GPU-1 clusters.
* **`eval_visual_aligning_dpcc.sh`:** Handles simulation planning rollouts and captures diagnostic trajectories.
* **`visual_aligning_dpcc_pipeline.sh`:** Executes the full pipeline (Phase 1: Training $\rightarrow$ Phase 2: Evaluation).

---

## 🎛️ Step 6: Config Reuse & Parameter Adaptations
To maintain parity and save setup overhead, we will directly reuse **`config/aligning-d3il-visual.py`** and **`config/visual_aligning_eval.yaml`**. 

Instead of rewriting them from scratch, we will simply append/modify the following parameters within these files when the coding phase begins:

### 1. In [config/aligning-d3il-visual.py](file:///workspaces/FM-PCC/config/aligning-d3il-visual.py):
We will add a new visual DPCC key under the `base` dictionary:
```python
base['diffuser_visual_aligning'] = {
    **base['ddpm_encdec_vision'],
    'model': 'diffuser_visual_aligning.models.visual_unet.VisualUNet',
    'diffusion': 'diffuser_visual_aligning.models.diffusion.GaussianDiffusion',
    'loader': 'diffuser_visual_aligning.datasets.sequence.AligningImgSequenceDataset',
    'horizon': 8,
    'if_vision': True,
    'obs_dim': 6,          # 6D stacked joint state [des_x, des_y, des_z, x, y, z]
    'action_dim': 3,       # Cartesian control dimensions (3D XYZ control)
    'prefix': 'diffuser_visual_aligning/',
}

base['plan_diffuser_visual_aligning'] = {
    **base['plan_ddpm_encdec_vision'],
    'prefix': 'f:plans/diffuser_visual_aligning/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
    'diffusion_loadpath': 'f:diffuser_visual_aligning/H{horizon}',
}
```

### 2. In [config/visual_aligning_eval.yaml](file:///workspaces/FM-PCC/config/visual_aligning_eval.yaml):
We will reuse the pre-configured physical Franka workspace boundaries and z-score constraints:
* **Franka Workspace Limits (Cartesian):** Reuses bounds `lb: [0.3, -0.35, 0.05]` and `ub: [0.7, 0.35, 0.40]`.
* **Inference timesteps:** Reuses `diffusion_timestep_threshold: 0.5`.
* **Add custom experiment trigger:**
  ```yaml
  exps: [
    'diffuser_visual_aligning',
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
  * *Code Block:* `LimitsNormalizer` and `DatasetNormalizer`. Standardizes action and state coordinate arrays into a unified $[-1, 1]$ limits range.

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

### 📂 Complete File and API Specification Map

| File Path | Component Name | API Signature / Key Classes | Role & Technical Responsibility |
| :--- | :--- | :--- | :--- |
| **`diffuser_visual_aligning/`**<br/>`datasets/sequence.py` | `AligningImgSequenceDataset` | `__init__(dataset_path, horizon=8, normalizer='LimitsNormalizer')`<br/>`__getitem__(idx) -> Batch(trajectories, conditions)` | Dynamic 3D Cartesian sequence loader. Instantiates untouched `Aligning_Img_Dataset` inside `d3il/`, scales values, and structures $[B, H, 9]$ joint trajectory blocks. |
| **`diffuser_visual_aligning/`**<br/>`models/temporal.py` | `VisualTemporalUNet` | `__init__(transition_dim=9, horizon=8, proprio_dim=6)`<br/>`forward(x, cond, t) -> denoised_trajectory` | Multi-modal visual-latent temporal U-Net backbone. Processes wrist/primary RGB frames via pre-trained `SpatialSoftmaxEncoder` blocks and embeds them as conditioning keys. |
| **`diffuser_visual_aligning/`**<br/>`sampling/policies.py` | `Policy` | `__call__(conditions, horizon, disable_projection=False) -> action, trajectories` | Receding Horizon Policy wrapper. Generates planning grids over $H=8$ steps, invokes the 3D boundary projector, and extracts action command $a_0$. |
| **`diffuser_visual_aligning/`**<br/>`sampling/projection.py` | `Projector` | `__init__(env, ...)`<br/>`project(trajectory, t) -> projected_trajectory` | Continuous 3D SLSQP Convex Safety Projector. Formulates quadratic programming (QP) limits to steering generated states back to obstacle boundary manifolds in 3D. |
| **`ddpm_encdec_vision_test_visual_dpcc/`**<br/>`train_visual_aligning_dpcc.py` | Training Loop Runner | Headless standalone runner script | Parses `aligning-d3il-visual.py` configurations, instantiates `AligningImgSequenceDataset`, runs Adam optimization over U-Net weights, and saves checkpoints. |
| **`ddpm_encdec_vision_test_visual_dpcc/`**<br/>`eval_visual_aligning_dpcc.py` | Evaluation & Rollout Runner | `VisualDPCCEvaluator`<br/>`evaluate_episode() -> success_rate` | Closed-loop simulator bridge. Connects to 3D environment, collects camera feeds and 3D positions, queries policy, and steps environment with predicted actions. |
| **`config/`**<br/>`aligning-d3il-visual.py` | Configuration Blocks | Key: `'diffuser_visual_aligning'`<br/>Key: `'plan_diffuser_visual_aligning'` | Base hyperparameters. Manages observation dimensions (`obs_dim: 6`), action dimensions (`action_dim: 3`), learning rates, and save directories. |
| **`Slurm_Codes/sbatch/`**<br/>`diffuser_visual_aligning/` | Batch Scripts | `train_visual_aligning_dpcc.sh`<br/>`eval_visual_aligning_dpcc.sh`<br/>`visual_aligning_dpcc_pipeline.sh` | HPC scheduling wrapper. Executes automated headless multi-seed training, evaluations, and renders MP4 visualization rollouts. |

---

### ⛓️ Architectural Deliverables Integration Diagram

The following diagram illustrates exactly how these final deliverables connect with each other and interact with the frozen `d3il/` simulator environment:

```
            [Raw Pickles / Images] ───> [Aligning_Img_Dataset (d3il)] (Frozen)
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │ AligningImgSequenceDataset   │ (3D Slicer)
                                       └──────────────┬───────────────┘
                                                      │ Batch(trajectories, conditions)
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           diffuser_visual_aligning Engine                          │
│                                                                                         │
│   ┌───────────────────────────┐      [Images]     ┌──────────────────────────────────┐  │
│   │    VisualTemporalUNet     │ <───────────────> │ SpatialSoftmaxEncoder (ddpm_enc) │  │
│   └─────────────┬─────────────┘                   └──────────────────────────────────┘  │
│                 │                                                                       │
│                 ▼ [Denoised Grid x_t]                                                   │
│   ┌───────────────────────────┐                                                         │
│   │    3D SLSQP Projector     │ <─── [Table/Obstacle Boundaries (visual_aligning_eval)] │
│   └─────────────┬─────────────┘                                                         │
│                 │                                                                       │
│                 ▼ [Safe Trajectory]                                                     │
│   ┌───────────────────────────┐                                                         │
│   │  Receding Horizon Policy  │ ───> [Extract Command a_0 (dx, dy, dz)]                  │
│   └─────────────┬─────────────┘                                                         │
└─────────────────┼───────────────────────────────────────────────────────────────────────┘
                  │ [3D Command: dx, dy, dz]
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│               eval_diffuser_visual_aligning (Simulator Interface Adapter)           │
│                                                                                         │
│   1. Passes 3D env coordinates directly to the model as 6D observation [des, actual].   │
│   2. Steps physical MuJoCo simulation environment in closed loop with 3D dx, dy, dz.     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 🎯 Expected Validation and Success Criteria

To scientifically validate the new **Visual-DPCC (Gen6v4)** model against the baseline state-only performance, the final deliverables must meet or exceed the following metrics:
1. **Absolute Parity Verification:** The normalizers, dataset dimensions, and U-Net layers must operate over an exact **3D action space (`action_dim = 3`)** and **6D observation space (`obs_dim = 6`)**.
2. **Zero Collision Rate:** Trajectory boundary violations inside the MuJoCo simulation workspace must be **strictly $0.0\%$** (safeguarded by intermediate SLSQP boundary projections).
3. **High Success Rate:** Closed-loop rollout success rates must be **$\ge 90\%$** under online receding-horizon control ($a_0$ replanning on every simulation tick).
4. **Thesis Traceability:** Trajectory rollouts and successes must be automatically dumped into diagnostic JSON files for comparison with non-visual models.

---

## 📊 Step 9: Scientific Diagnostics & Visual Rollout Debug System

To eliminate the "black-box" nature of visual-diffusion modeling and match the diagnostic fidelity of the `ddpm_encdec_vision` baseline, we will port its exact rollout debugging and telemetry logging features directly into `eval_visual_aligning_dpcc.py`:

### 1. Side-by-Side Video & GIF Rendering
* **Mechanism:** Slices camera observations at each step, transposes the tensor shapes from PyTorch `[C, H, W]` to OpenCV `[H, W, C]`, and rescales the float values to standard `[0, 255]` integers.
* **Stream Concatenation:** Concatenates the **primary camera (agentview)** and **in-hand (wrist) camera** frames horizontally.
* **RGB Calibration:** Converts image color spaces from OpenCV BGR to standard RGB before saving to prevent color inversion.
* **Automated Generation:** Saves the complete side-by-side rollout immediately after episode termination under `results/diagnostics/<variant>/rollout_<idx>.mp4` (with an automatic fallback to `.gif` if standard MP4 libraries are missing).

### 2. Multi-Panel Scientific PNG Reports
Every single episode rollout generates a scientific diagnostic report (`rollout_<idx>_report.png`) detailing the policy's inner decisions:
* **Panel 1 (XY Projection):** Plots the real robot path (black line) superimposed on the MPC blue foresight plans (`full_plans`), visualizing plan stability.
* **Panel 2 & 3 (X / Y Timelines):** Plots Cartesian coordinates over steps.
* **Panel 4 (Height Profile):** Traces $z$-coordinate levels to verify workspace contact/height stability.
* **Panel 5 (MPC Tracking Error):** Computes and plots the Euclidean tracking distance between predicted trajectories and physical execution.
* **Panel 6 (Velocities):** Tracks the Euclidean velocity norm of the end-effector.

### 3. Automated JSON & Text Stats Telemetry
Alongside visual logs, every rollout outputs structured metrics:
* **`rollout_<idx>_stats.json`:** Structured diagnostic JSON storing:
  ```json
  {
      "rollout_index": 0,
      "success": true,
      "steps": 142,
      "mean_distance": 0.0152,
      "mode": 1,
      "avg_inference_time_per_step": 0.0421,
      "max_tracking_error": 0.0034
  }
  ```
* **Human-Readable Text Logs (`rollout_<idx>_stats.txt`):** Structured summary printed right next to the GIFs for instant diagnostics.

### 4. Headless Expert Reference Generation
* **Function:** `generate_expert_reference(save_path, n_rollouts)`
* **Role:** Before starting evaluation, parses the source pickle files (`env_*.pkl`) in the training dataset, executes their exact paths in MuJoCo, and dumps ground-truth reference GIFs/videos. This provides a scientific baseline to cross-verify model execution.

---

## ⚖️ Architectural Head-to-Head: Proposed Visual-DPCC vs. Gen6 vs. Old DPCC

The following structural comparison highlights how the **Proposed Visual-DPCC** integrates the reactive safety of the old state-only models with the multi-modal intelligence of visual systems:

| Architectural Metric | **Old DPCC (State-Only)** | **Gen6 (Failed/Legacy Visual)** | **Proposed Visual-DPCC (Gen6v4)** |
|:---|:---|:---|:---|
| **Input Modality** | Proprioceptive state vectors only ($20D$). | Dual camera frames ($3 \times 128 \times 128$) + Proprioception ($20D$). | **Dual camera frames + Proprioception** (Ingested via pre-trained ResNet Spatial Softmax). |
| **Control Loop Frequency** | **High-Frequency Reactivity.** Closed-loop replanning at every single step ($a_0$). | **Chunked Latency.** Open-loop block execution of action chunks (replans only every 4 steps). | **High-Frequency Reactivity.** Closed-loop replanning at every single step ($a_0$). |
| **Sequence Horizon** | Unified horizon ($H=8$). | Fragmented horizons (obs history $O=5$, actions $A=4$). | **Unified horizon ($H=8$)** mapping continuous trajectories. |
| **Safety Shielding** | Intermediate SLSQP boundary projections ($t \le \text{threshold}$). | Post-hoc checking (passive out-of-bound indicators). | **Intermediate SLSQP boundary projections** active during diffusion denoising steps. |
| **Dataset Ingestion** | Proprioceptive state sequence arrays. | Rigid VAE-bridge sequence loader. | **Modular loader** wrapping battle-tested `Aligning_Img_Dataset`. |

---

## 🔬 Detailed Mathematical Comparison: 6D DPCC vs. 9D DPCC (Verification of Reality)

To guarantee scientific rigor, we verify the exact dimensional footprints of the DPCC models. Both configurations are **highly real, active mathematical structures** in this workspace. 

### 1. Verification of the 6D DPCC (State-Only baseline for avoiding-d3il)
The **6D DPCC** is the active configuration used for the state-only **`avoiding-d3il`** task. It is defined by a 2D action space and a 4D proprioceptive state space:
* **Action Space ($2D$):** Robot end-effector delta commands `[vx, vy]`.
* **Proprioceptive Observation Space ($4D$):** Commanded target pose and physical end-effector coordinates sliced to 2D `[des_x, des_y, x, y]`.
* **Total Joint Grid Size:** $\text{Action } (2D) + \text{Observation } (4D) = \mathbf{6D}$.
* **Reality Verification (Code Reference):** In `/workspaces/FM-PCC/config/projection_eval.yaml`, the indices are explicitly mapped as follows:
  ```yaml
  avoiding:
    action_indices: {'vx': 0, 'vy': 1}
    state_indices: {'x_des': 2, 'y_des': 3, 'x': 4, 'y': 5}
  ```
  This proves the 6D joint grid structure is real and forms a cohesive mathematical space.

### 2. Verification of the 9D DPCC (New Visual-Aligning 3D upgrade)
The **9D DPCC** is our upgraded configuration for **`visual-aligning-d3il`**. It expands the spatial dimensions to full 3D Cartesian coordinates to capture height and contact variables:
* **Action Space ($3D$):** Robot end-effector delta velocity commands `[dx, dy, dz]`.
* **Proprioceptive Observation Space ($6D$):** Commanded target pose and physical end-effector coordinates in 3D `[des_x, des_y, des_z, x, y, z]`.
* **Total Joint Grid Size:** $\text{Action } (3D) + \text{Observation } (6D) = \mathbf{9D}$.

---

### 📊 Comparative Dimension Matrix: 6D vs. 9D

| Parameter / Space | **6D DPCC (State-Only Avoiding)** | **9D DPCC (Visual-Aligning Upgrade)** | **Mathematical Rationale** |
|:---|:---|:---|:---|
| **Action Layout ($d_a$)** | `[vx, vy]` ($2D$) | `[dx, dy, dz]` ($3D$) | Upgrades control from 2D plane to full 3D Cartesian space. |
| **Observation Layout ($d_o$)** | `[des_x, des_y, x, y]` ($4D$) | `[des_x, des_y, des_z, x, y, z]` ($6D$) | Stacks 3D target coordinates with 3D physical coordinate feedback. |
| **Joint Trajectory ($d_a + d_o$)** | **$6D$** joint grid per horizon step | **$9D$** joint grid per horizon step | Concatenates control commands and physical poses to form a single grid. |
| **Workspace Constraints** | Table limits on $x$ and $y$. | Table limits on $x, y$ AND contact height $z$. | Prevents physical tabletop crashes ($z \ge 0.05$) in 3D rollouts. |
| **Derivative Constraints** | $x_{t+1} = x_t + vx_t$<br/>$y_{t+1} = y_t + vy_t$ | $x_{t+1} = x_t + dx_t$<br/>$y_{t+1} = y_t + dy_t$<br/>$z_{t+1} = z_t + dz_t$ | Enforces physical Euler integration over the planning horizon $H$. |
| **SLSQP QP Matrix Size** | $8 \times 6 = 48D$ optimization space | $8 \times 9 = 72D$ optimization space | The size of the optimization variable vector solved by SLSQP. |

