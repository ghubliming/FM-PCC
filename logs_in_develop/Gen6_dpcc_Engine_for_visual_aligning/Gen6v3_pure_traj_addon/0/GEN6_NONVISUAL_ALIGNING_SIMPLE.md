# Gen6 Non-Visual Aligning Mode (Comprehensive Implementation Plan)

## Overview & Objective
Enable the Gen6/FM-PCC framework to execute in **non-visual mode** (state-only) on the D3IL aligning task. This serves to establish a rigorous performance baseline for comparison against the visual-backbone Gen6 model, verifying that the Gen6 UNet1D and diffusion scheduling cleanly reduce to the state-only FMv3ODE architecture.

---

## 🔑 Crucial Insight: Reconciling the 17D vs. 20D Mismatch
Based on our unified analysis, D3IL’s state-only aligning environment exposes a hidden complexity:
* **Training Dataset (20D observation):** Tracks two distinct position channels: `robot_des_pos` (3D desired command) and `robot_c_pos` (3D actual TCP pos), plus boxes and target coordinates (3+3+3+4+3+4 = 20D).
* **Simulator Runtime (17D observation):** The MuJoCo environment's `get_observation()` returns a 17D vector (missing the `robot_des_pos` channel).

### The MPC Solution:
To achieve exact semantic parity and avoid a massive distribution mismatch, our rollout agent must replicate D3IL's native evaluation-time padding pattern:
1. Maintain a running desired position vector (`self.mental_robot_pos`, starting at the initial 3D TCP position).
2. At every rollout step, prepend the 3D desired position vector to the 17D environment observation: `obs = np.concatenate((self.mental_robot_pos, state))`, constructing a 20D observation.
3. Feed this 20D observation sequence into the state-only model.
4. Execute the predicted 2D actions and accumulate them into `self.mental_robot_pos[:2]`.

---

## Step-by-Step Implementation Plan

### Step 1: Add Config Block
**File:** [config/aligning-d3il-visual.py](file:///workspaces/FM-PCC/config/aligning-d3il-visual.py)

Add the state-only configuration profile to the config file (correcting the old mistaken location in `avoiding-d3il.py`):
```python
'ddpm_encdec_vision_nonvisual': {
    'model': 'ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge',
    'dataset': 'aligning-d3il',
    'if_vision': False,                  # ← Core state-only flag
    'obs_dim': 20,                       # ← 20D state vector (instead of 3D proprioception)
    'action_dim': 2,                     # ← 2D velocity action (vx, vy)
    'horizon': 8,
    'obs_seq_len': 5,
    'action_seq_size': 4,
    'n_diffusion_steps': 16,
    'action_weight': 10,
    'learning_rate': 2e-4,
    'batch_size': 32,
    'n_train_steps': 50000,
    'device': 'cuda',
}
```

---

### Step 2: Dataset Loading and State Dimension Calibration
**File:** [ddpm_encdec_vision_test/train_ddpm_encdec_vision.py](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py)

Conditionally load the state-only `Aligning_Dataset` instead of `Aligning_Img_Dataset` when `if_vision = False`, and adjust the `obs_dim` passed to the builder:
```python
# --- Dynamic Dataset Selection ---
if_vision = args.get('if_vision', True)

if if_vision:
    from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
    dataset_cls = Aligning_Img_Dataset
    obs_dim = 3  # Proprioception-only
else:
    from d3il.environments.dataset.aligning_dataset import Aligning_Dataset
    dataset_cls = Aligning_Dataset
    obs_dim = 20  # Full 20D State

dataset_config = utils.Config(
    dataset_cls,
    savepath=(args.savepath, 'dataset_config.pkl'),
    data_directory='environments/dataset/data/aligning/train_files.pkl',
    device='cpu',
    obs_dim=obs_dim,
    action_dim=args.action_dim,
    window_size=args.horizon,
    max_len_data=args.max_path_length
)
dataset = dataset_config()
```

---

### Step 3: Dynamic Backbone Generalization
**File:** [ddpm_encdec_vision/models/visual_unet.py](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_unet.py)

1. Generalize the `transition_dim` dynamically from `config.action_dim + 3` to `config.action_dim + config.obs_dim` (supporting both the 6D visual proprioceptive trajectory and 22D non-visual state-action trajectory).
2. Skip vision encoder initialization and forward logic when `if_vision = False`.

```python
class VisualUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.if_vision = getattr(config, "if_vision", True)
        
        # 1. Conditionally Instantiate Vision Encoder
        if self.if_vision:
            # (Standard MultiImageObsEncoder setup...)
            self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
            latent_dim = 128
        else:
            self.obs_encoder = None
            latent_dim = 0  # No image embeddings
        
        # 2. Dynamic transition_dim and safe horizon setup
        self.target_horizon = config.horizon
        self.padded_horizon = ((self.target_horizon + 7) // 8) * 8
        
        # transition_dim is action + full observation channels
        transition_dim = config.action_dim + config.obs_dim

        self.backbone = backbone_class(
            horizon=self.padded_horizon,
            transition_dim=transition_dim, 
            cond_dim=latent_dim,
            dim=getattr(config, "dim", 128),
            dim_mults=getattr(config, "dim_mults", (1, 2, 4, 8)),
            returns_condition=getattr(config, "returns_condition", False),
            condition_dropout=getattr(config, "condition_dropout", 0.1),
            use_cond_projection=self.if_vision, # True only for visual
        ).to(self.device)

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False):
        # Apply padding
        B, T, D = x.shape
        if T < self.padded_horizon:
            pad_len = self.padded_horizon - T
            x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
        
        if self.if_vision:
            bp_imgs, inhand_imgs, state = cond['visual'] if isinstance(cond, dict) else cond
            visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state=state)
            if T < self.padded_horizon:
                visual_emb = torch.cat([visual_emb, torch.zeros(B, pad_len, visual_emb.shape[-1], device=visual_emb.device)], dim=1)
        else:
            visual_emb = None  # No external conditioning tensor
            
        out = self.backbone(x, visual_emb, t, returns=returns, use_dropout=use_dropout, force_dropout=force_dropout)
        return out[:, :T, :]
```

---

### Step 4: Engine Training/Inference Separation
**File:** [ddpm_encdec_vision/models/visual_gaussian_diffusion.py](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py)

Bypass image unpacking and create state-only inpainting conditions:
```python
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        # Trajectory x: [B, T, action + obs]
        x = torch.cat([act, obs], dim=-1)
        
        if getattr(self.model, 'if_vision', True):
            cond = {
                'visual': (bp_imgs, inhand_imgs, obs),
                0: obs[:, 0]
            }
        else:
            # State-only condition: first frame state for snapping
            cond = {0: obs[:, 0]}
        
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        if getattr(self.model, 'if_vision', True):
            # (Standard visual tuple unpacking...)
            new_cond = ...
        else:
            # Bypassed: cond is already state-only dict: {0: state}
            new_cond = cond
            
        return super().forward(new_cond, *args, **kwargs)
```

---

### Step 5: Simulator Adapter & Rollout Loop Implementation
**File:** [ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)

1. Make `if_vision` dynamic in the evaluation script to adapt based on model configuration:
   ```python
   sim_vision = getattr(diffusion_model.model, 'if_vision', True)
   sim = Aligning_Sim(..., if_vision=sim_vision)
   ```
2. Implement the missing `else` branch of `VisualAgentWrapper.predict()` to resolve the 17D -> 20D mismatch:
   ```python
   def predict(self, state, goal=None, extra_args=None, if_vision=False):
       if if_vision:
           # (Standard image extraction and visual MPC planning...)
           ...
       else:
           # ─── STATE-ONLY INFERENCE PATH ───
           # 1. 17D to 20D Mismatch Adapter: Prepend desired pos
           if self.mental_robot_pos is None:
               self.mental_robot_pos = state[:3].copy()  # Starts at actual TCP pos
           
           # Prepend 3D desired position to 17D state -> 20D
           obs_20d = np.concatenate((self.mental_robot_pos, state))
           
           # Record real position for diagnostics
           self.history_real_pos.append(state[:3].copy())
           if self.last_predicted_pos is not None:
               err = np.linalg.norm(state[:2] - self.last_predicted_pos[:2])
               self.curr_rollout_tracking_errors.append(err)
               
           # 2. Sequence conditioning assembly
           obs_torch = torch.from_numpy(obs_20d).to(self.device).float().unsqueeze(0)
           if self.scaler is not None:
               obs_torch = self.scaler.scale_input(obs_torch)
               
           self.des_robot_pos_context.append(obs_torch)
           while len(self.des_robot_pos_context) < self.window_size:
               self.des_robot_pos_context.appendleft(obs_torch)
               
           obs_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
           cond = {0: obs_seq}  # Snapping at t=0
           
           # 3. Trigger Diffusion MPC Planning
           if self.action_counter == self.action_seq_size:
               self.action_counter = 0
               self.model.eval()
               
               # Replicate batch execution if DPCC selection is active
               cond_batch = {0: cond[0].repeat(self.batch_size, 1, 1)}
               trajectory, infos = self.model(cond_batch)
               
               # Inverse scale and select trajectory
               action_trajectory = trajectory[[0], :, :self.model.action_dim] # [1, H, 2]
               if self.scaler is not None:
                   action_trajectory = self.scaler.inverse_scale_output(action_trajectory)
               
               self.curr_action_seq = action_trajectory[:, :self.action_seq_size, :]
               self.history_full_plans.append(action_trajectory[0].detach().cpu().numpy())
           
           # 4. Extract action step and update desired position
           next_action = self.curr_action_seq[:, self.action_counter, :]
           next_action_np = next_action.detach().cpu().numpy()
           
           self.mental_robot_pos[:2] += next_action_np.squeeze(0)  # vx, vy step
           self.history_desired_actions.append(next_action_np.copy().squeeze(0))
           self.last_predicted_pos = self.mental_robot_pos.copy()
           
           self.action_counter += 1
           self.step_counter += 1
           return next_action_np
   ```

---

## Validation Checklist
- [ ] Config loads correctly from `config/aligning-d3il-visual.py`.
- [ ] `Aligning_Dataset` correctly yields `obs_dim=20` state sequences without image loading overhead.
- [ ] `VisualUNet` dynamically scales `transition_dim` to 22 (20D observation + 2D action) and skips multi-camera feature extraction.
- [ ] `VisualGaussianDiffusion` loss handles flat tensor conditioning.
- [ ] The `VisualAgentWrapper` prepends desired positions to convert 17D sim step observations to 20D inputs, eliminating distribution mismatches during rollout.
