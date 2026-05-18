# D3IL DDPM-ACT Baseline vs. Gen6 Visual-Aligning Implementation
## Detailed Architectural & Trajectory Dimensionality Analysis

---

## EXECUTIVE SUMMARY

The **D3IL DDPM-ACT baseline** (in `/workspaces/d3il/`) and **Gen6 (in `/workspaces/FM-PCC/ddpm_encdec_vision_test/`)** both implement stochastic diffusion models for visual-conditioned behavior cloning, but with **fundamentally different trajectory representations**:

| **Aspect** | **D3IL Baseline** | **Gen6** |
|---|---|---|
| **Trajectory Format** | Separate action and state sequences | **Combined 6D [action, state] at each timestep** |
| **Action Dim** | 3D (Δx, Δy, Δz velocities) | 3D (Δx, Δy, Δz velocities) |
| **State Dim** | 7D (full proprioception) | 3D (only EE position) |
| **Training Loss** | MSE on action only | **MSE on concatenated [action, state]** |
| **Subject to Constraints** | None (vanilla diffusion) | **DPCC projection on 6D trajectory** |
| **Backbone** | Transformer (Act-VAE) | U-Net (VisualUNet) |

---

## PART 1: D3IL DDPM-ACT BASELINE ARCHITECTURE

### 1.1 Dataset & Preparation

**File**: [d3il/agents/act_vision_agent.py](d3il/agents/act_vision_agent.py) (lines 190-240)

Raw trajectories are collected as:
```python
# Raw dataset format
state_raw.shape = [B, T, 7]         # Full proprioceptive state (7D)
action_raw.shape = [B, T, 3]        # Cartesian deltas (3D)
```

**Proprioceptive State (7D)**: 
```
[robot_ee_x, robot_ee_y, robot_ee_z,      # EE Cartesian position (3D)
 robot_gripper_q1, robot_gripper_q2,      # Joint angles (2D)
 robot_gripper_q3, robot_gripper_q4]      # Additional joints (2D)
```

**Actions (3D)**:
```
[Δx_command, Δy_command, Δz_command]      # Cartesian velocity targets sent to EE controller
```

### 1.2 Scaler/Normalizer

**File**: [d3il/agents/utils/scaler.py](d3il/agents/utils/scaler.py) (lines 1-135)

```python
class Scaler:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, scale_data: bool, device: str):
        # x_data is STATES [dataset_size, 7]
        # y_data is ACTIONS [dataset_size, 3]
        
        self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)      # Shape: [7]
        self.x_std = torch.from_numpy(x_data.std(0)).to(device)        # Shape: [7]
        self.y_mean = torch.from_numpy(y_data.mean(0)).to(device)      # Shape: [3]
        self.y_std = torch.from_numpy(y_data.std(0)).to(device)        # Shape: [3]
        
        # Z-AXIS CRITICAL: For flat table tasks, y_std[2] ≈ 0.0
        # Scaler ADDS epsilon to prevent division by zero:
        self.y_std_safe = self.y_std + 1e-12 * ones_like(self.y_std)
        
        # Bounds used by diffusion sampler for action clipping
        if self.scale_data:
            self.y_bounds[0, :] = (y_data.min(0) - y_data.mean(0)) / (y_data.std(0) + 1e-12)
            self.y_bounds[1, :] = (y_data.max(0) - y_data.mean(0)) / (y_data.std(0) + 1e-12)
```

**Normalization Formula (scaled mode)**:
```
x_scaled = (x - x_mean) / (x_std + 1e-12)
y_scaled = (y - y_mean) / (y_std + 1e-12)
```

**Inverse (for control commands)**:
```
y_original = y_scaled * (y_std + 1e-12) + y_mean
```

### 1.3 Vision Encoder

**File**: [d3il/agents/act_vision_agent.py](d3il/agents/act_vision_agent.py) (lines 26-88)

```python
class ActPolicy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig, visual_input: bool = False):
        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.model = hydra.utils.instantiate(model).to(device)
    
    def forward(self, state, action):
        if self.visual_input:
            agentview_image, in_hand_image, state = state  # Unpack visual tuple
            B, T, C, H, W = agentview_image.size()
            
            # Reshape to [B*T, C, H, W] for batch encoding
            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            state = state.view(B * T, -1)  # [B*T, 7]
            
            # Encode both images + proprioceptive state
            obs_dict = {
                "agentview_image": agentview_image,     # Multi-view RGB
                "in_hand_image": in_hand_image,         # Egocentric camera
                "robot_ee_pos": state                   # Low-dim prop state
            }
            
            obs = self.obs_encoder(obs_dict)            # Returns [B*T, feature_dim]
            obs = obs.view(B, T, -1)                    # Reshape back to [B, T, feature_dim]
```

**Encoder Output**: `obs_encoded.shape = [B, T, obs_encoded_dim]` where `obs_encoded_dim` is typically 64-256 depending on the vision encoder architecture.

### 1.4 Core Diffusion Model Definition

**File**: [d3il/agents/models/diffusion/diffusion_policy.py](d3il/agents/models/diffusion/diffusion_policy.py) (lines 1-100)

```python
class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim: int,          # = obs_encoded_dim (e.g., 128)
        action_dim: int,         # = 3
        model: DictConfig,       # Neural network architecture
        beta_schedule: str,      # 'cosine', 'linear', or 'vp'
        n_timesteps: int,        # = 16 (diffusion steps)
        loss_type: str,          # = 'l2'
        clip_denoised: bool,     # = True
        predict_epsilon=True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.state_dim = state_dim          # Encoded observation dimension
        self.action_dim = action_dim        # = 3 (XYZ actions only)
        
        # Beta schedule: noise variance at each diffusion step
        if beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(n_timesteps)
        # ... [alpha, alpha_cumprod, etc. calculations]
        
        # Core denoising network: predicts noise from noisy action + observation condition
        self.model = hydra.utils.instantiate(model)  # E.g., DiffusionMLPNetwork
        
        self.loss_fn = Losses[loss_type]()  # L2 MSE loss
```

**Key Point**: The diffusion model operates ONLY on **action dimension (3D)**, not on states. States are used as conditioning.

### 1.5 Loss Function & Training

**File**: [d3il/agents/models/diffusion/diffusion_policy.py](d3il/agents/models/diffusion/diffusion_policy.py) (lines 230-260)

```python
def p_losses(self, x_start: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, t: torch.Tensor, weights=1.0):
    """
    Args:
        x_start: action targets [B, 3] or [B, T, 3]
        state:   encoded observations [B, obs_dim] or [B, T, obs_dim]
        goal:    goal state [B, obs_dim] or [B, T, obs_dim]
        t:       random timestep [B]
    """
    # Sample random noise from N(0, I)
    noise = torch.randn_like(x_start)  # Same shape as x_start [B, 3] or [B, T, 3]
    
    # Forward diffusion: add noise at timestep t
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    # x_noisy = sqrt(alpha_cumprod[t]) * x_start + sqrt(1 - alpha_cumprod[t]) * noise
    
    # Denoising network predicts noise
    x_recon = self.model(x_noisy, t, state, goal)  # Predicts noise [B, 3] or [B, T, 3]
    
    # Loss: MSE between predicted noise and actual noise (or between x_0 and x_recon)
    if self.predict_epsilon:
        loss = self.loss_fn(x_recon, noise, weights)  # Predict noise
    else:
        loss = self.loss_fn(x_recon, x_start, weights)  # Predict x_0 directly
    
    return loss

def loss(self, x: torch.Tensor, state: torch.Tensor, goal: Optional[torch.Tensor] = None):
    batch_size = len(x)
    t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
    return self.p_losses(x, state, goal, t)
```

**Loss Computation**:
```
L = || ε_pred(x_t, t, state, goal) - ε_true ||²  [predicted noise vs. true noise]
```

### 1.6 Forward Pass (Inference) - Sampling

**File**: [d3il/agents/models/diffusion/diffusion_policy.py](d3il/agents/models/diffusion/diffusion_policy.py) (lines 150-200)

```python
def p_sample_loop(self, state, goal, shape, verbose=False, return_diffusion=False):
    """
    Reverse diffusion: generate actions by iteratively denoising.
    
    Args:
        state: encoded observation [B, obs_dim]
        goal:  goal state [B, obs_dim]
        shape: target action shape [B, action_dim] = [B, 3]
    """
    batch_size = shape[0]
    
    # Start from pure Gaussian noise
    x = torch.randn(shape, device=self.device)  # [B, 3]
    
    if return_diffusion:
        diffusion = [x]
    
    # Iteratively denoise for n_timesteps (16) steps
    for i in reversed(range(0, self.n_timesteps)):
        # Create timestep batch
        timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
        
        # Denoise using the model
        x = self.p_sample(x, timesteps, state, goal)
        # Computes: x_recon from model
        #           posterior_mean = (beta * sqrt(alpha_cumprod_prev) * x_0 + 
        #                             (1 - alpha_cumprod_prev) * sqrt(alpha) * x_t) / (1 - alpha_cumprod)
        #           x = posterior_mean + sqrt(posterior_variance) * z
        
        if return_diffusion:
            diffusion.append(x)
    
    # Clip actions to valid range
    action = x.clamp_(self.min_action, self.max_action)
    
    return action

def sample(self, state, goal, *args, **kwargs):
    batch_size = state.shape[0]
    if len(state.shape) == 3:
        shape = (batch_size, self.model.action_seq_len, self.action_dim)
    else:
        shape = (batch_size, self.action_dim)
    action = self.p_sample_loop(state, goal, shape, *args, **kwargs)
    return action.clamp_(self.min_action, self.max_action)
```

**Reverse Diffusion Process**:
```
x_0 = pure Gaussian noise [B, 3]
For t = T-1, ..., 0:
    ε_t = model(x_t, t, state, goal)
    x_t-1 = (x_t - β_t / sqrt(1 - ᾱ_t) * ε_t) / sqrt(α_t) + σ_t * z_t
```

### 1.7 Agent Training Loop

**File**: [d3il/agents/act_vision_agent.py](d3il/agents/act_vision_agent.py) (lines 190-250) and [d3il/agents/ddpm_vision_agent.py](d3il/agents/ddpm_vision_agent.py) (lines 100-200)

```python
class DiffusionAgent(BaseAgent):
    def train_vision_agent(self):
        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, obs, action, mask = data  # Raw data
            
            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)
            
            # SCALE the data
            obs = self.scaler.scale_input(obs)        # [B, T, 7] → normalized
            action = self.scaler.scale_output(action)  # [B, T, 3] → normalized
            
            # Create state tuple for vision encoder
            state = (bp_imgs, inhand_imgs, obs)
            
            # Encode observations
            state_embedding = self.model.get_embedding(state)  # [B, T, obs_encoded_dim]
            
            # Training step on scaled action and encoded state
            batch_loss = self.train_step(state_embedding, action)
    
    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        self.model.train()
        
        # Forward pass: compute diffusion loss
        # self.model is the Diffusion instance
        loss = self.model.loss(actions, state, goal)  # [B, T, 3] and [B, T, obs_dim]
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## PART 2: GEN6 VISUAL-ALIGNING IMPLEMENTATION

### 2.1 Dataset & Preparation (Same as D3IL)

**File**: [FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) (lines 50-150)

```python
# Raw dataset shapes identical to D3IL
state_raw.shape = [B, T, 7]         # Full proprioception
action_raw.shape = [B, T, 3]        # Cartesian deltas

# But Gen6 ONLY uses 3D proprioceptive subset (EE position)
state_subset = state_raw[:, :, :3]  # [B, T, 3] - only EE position
```

**Key Difference**: Gen6 **concatenates action and state into a 6D combined trajectory**:
```python
x = torch.cat([action, state_subset], dim=-1)  # [B, T, 6]
# Now x = [Δx, Δy, Δz, x_ee, y_ee, z_ee]
```

### 2.2 Scaler/Normalizer (Adapted for 6D)

**File**: [FM-PCC/ddpm_encdec_vision/utils/scaler.py](FM-PCC/ddpm_encdec_vision/utils/scaler.py) (lines 1-150)

```python
class Scaler:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, scale_data: bool, device: str):
        # CRITICAL: Gen6 treats concatenated [action, state] as a single entity
        # x_data = concatenated [Δx, Δy, Δz, x_ee, y_ee, z_ee] [dataset_size, 6]
        # y_data is NOT USED (only actions in the concatenated vector)
        
        self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)      # Shape: [6]
        self.x_std = torch.from_numpy(x_data.std(0)).to(device)        # Shape: [6]
        
        # Z-AXIS CRITICAL: For flat table tasks with no vertical motion:
        # x_std[5] (z-position std) ≈ 0.0
        # x_std[2] (z-action std) ≈ 0.0
        
        # Safe std to prevent division by zero
        self.x_std_safe = self.x_std + 1e-2  # LARGER epsilon than D3IL's 1e-12
        
        # Bounds for both actions and states
        self.x_bounds[0, :] = (x_data.min(0) - x_data.mean(0)) / (x_data.std(0) + 1e-2)
        self.x_bounds[1, :] = (x_data.max(0) - x_data.mean(0)) / (x_data.std(0) + 1e-2)
        
        # Bounds shape: [2, 6]
        # bounds[0, :] = [min_Δx_scaled, min_Δy_scaled, min_Δz_scaled, min_x_scaled, min_y_scaled, min_z_scaled]
        # bounds[1, :] = [max_Δx_scaled, max_Δy_scaled, max_Δz_scaled, max_x_scaled, max_y_scaled, max_z_scaled]
```

**For Z-axis flat-table example** (NO vertical motion in data):
```
Raw data:
  z_action_data = [0.0001, -0.0002, 0.00015, ...]     (micro-vibrations)
  z_position_data = [0.6, 0.6, 0.6, ...]              (constant height)
  
  z_action_std_raw = 0.0001
  z_position_std_raw = 0.0 (or 1e-7)
  
With safe epsilon:
  z_action_std_safe = max(0.0001, 1e-2) = 0.01
  z_position_std_safe = max(0.0, 1e-2) = 0.01
  
Scaled bounds:
  bounds[0, 2] = (0.00015 - mean) / 0.01 ≈ -0.1 to -1.0
  bounds[1, 2] = (0.00010 - mean) / 0.01 ≈ 0.1 to 1.0
  bounds[0, 5] = (0.6 - 0.6) / 0.01 = 0.0  
  bounds[1, 5] = (0.6 - 0.6) / 0.01 = 0.0
```

### 2.3 Vision Encoder (Same as D3IL)

Identical to [D3IL](#13-vision-encoder).

### 2.4 Core Diffusion Model - VisualGaussianDiffusion

**File**: [FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py](FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py) (lines 1-80)

```python
class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Diffuses over concatenated 6D trajectory [action, state].
    Overrides parent p_mean_variance to apply DPCC projection.
    """
    
    def __init__(self, ...):
        super().__init__(...)
        self.action_dim = 3      # First 3 dims are actions
        # implicit: state_dim = 3 (last 3 dims are position)
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Training loss on COMBINED 6D trajectory.
        
        Args:
            act: [B, T, 3]      Cartesian deltas
            obs: [B, T, 3]      EE positions (scaled)
        """
        # COMBINE into 6D trajectory
        x = torch.cat([act, obs], dim=-1)  # [B, T, 6]
        
        # Condition dict for VisualUNet
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),  # Vision conditioning
            0: obs[:, 0]                             # First-frame state (for snapping)
        }
        
        # Standard DDPM loss on 6D combined vector
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)  # Loss on 6D trajectory
    
    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        """
        Denoising step with OPTIONAL DPCC projection.
        
        Args:
            x: noisy 6D trajectory [B, T, 6]
            cond: conditioning dict with vision + first-frame state
            t: timestep
            projector: DPCC projector (optional)
        """
        # Get noise prediction from U-Net
        epsilon = self.model(x, cond, t)  # Predicts noise on [B, T, 6]
        
        # Convert noise to reconstruction of x_0
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        
        if self.clip_denoised:
            # CRITICAL: Only clamp action dimensions [0:3]
            # NEVER clamp state dimensions [3:6]
            x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)  # Actions only
            # State dimensions remain unclamped!
        
        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        
        # DPCC Projection (if enabled and not in training)
        if projector is not None and projector.gradient:
            # Extract trajectory without goal dimension
            if self.goal_dim > 0:
                trajectory = x_recon[:, :, :-self.goal_dim]  # [B, T, 6]
            else:
                trajectory = x_recon  # [B, T, 6]
            
            # Compute projection gradient (solves QP to enforce constraints)
            grad = projector.compute_gradient(trajectory, constraints)  # [B, T, 6]
            
            # Inject gradient into denoising mean
            model_mean = model_mean + grad  # Move toward feasible region
        
        return model_mean, posterior_variance, posterior_log_variance
```

**Key Difference from D3IL**:
- D3IL loss operates on **3D actions only**
- Gen6 loss operates on **6D [actions, states] concatenated**
- Gen6 OPTIONALLY applies DPCC projection during denoising loop

### 2.5 Loss Function & Training

**File**: [FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py](FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py) (similar to D3IL but with 6D loss)

```python
def p_losses(self, x_start: torch.Tensor, cond: Dict, t: torch.Tensor):
    """
    Args:
        x_start: combined 6D trajectory [B, T, 6]
                 = [Δx, Δy, Δz, x_ee, y_ee, z_ee]
        cond:    conditioning dict {visual: ..., 0: initial_state}
        t:       random timestep [B]
    """
    # Sample random noise
    noise = torch.randn_like(x_start)  # [B, T, 6]
    
    # Forward diffusion
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    
    # Model predicts noise on FULL 6D space
    x_recon = self.model(x_noisy, cond, t)  # [B, T, 6]
    
    # Loss: MSE on 6D trajectory
    if self.predict_epsilon:
        loss = F.mse_loss(x_recon, noise)  # Predicts noise on [B, T, 6]
    else:
        loss = F.mse_loss(x_recon, x_start)  # Predicts x_0 on [B, T, 6]
    
    return loss
```

**Loss Computation (Gen6)**:
```
L = || ε_pred(x_t[6D], t, cond) - ε_true ||²
  = sum over [Δx, Δy, Δz, x_ee, y_ee, z_ee] dimensions
```

### 2.6 Forward Pass (Inference) with Projection

**File**: [FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py](FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py) (lines 40-90) and inherited from [FM-PCC/diffuser/models/diffusion.py](FM-PCC/diffuser/models/diffusion.py)

```python
def p_sample_loop(self, shape, cond, projector=None, constraints=None, verbose=False):
    """
    Reverse diffusion with OPTIONAL DPCC projection.
    
    Args:
        shape: [B, T, 6] - batch size, horizon, transition dim
        cond: {visual: ..., 0: initial_state, ...}
        projector: DPCC Projector instance (optional)
        constraints: constraint specs for projection
    """
    batch_size = shape[0]
    
    # Start from Gaussian noise on 6D space
    x = torch.randn(shape, device=self.device)  # [B, T, 6]
    
    # Apply initial state snapping (constraint conditioning)
    x = apply_conditioning(x, cond, action_dim=self.action_dim, ...)
    
    # Iteratively denoise
    for i in reversed(range(0, self.n_timesteps)):
        timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
        
        # Denoise with optional projection
        x = self.p_sample(
            x, timesteps, cond, 
            returns=None, 
            projector=projector,      # DPCC projector
            constraints=constraints   # QP constraints
        )
        
        # Re-apply conditioning (snap initial state again)
        x = apply_conditioning(x, cond, action_dim=self.action_dim, ...)
    
    # Clamp action dimensions [0:3] only
    x[..., :self.action_dim].clamp_(-5.0, 5.0)
    
    return x  # [B, T, 6]
```

**Reverse Diffusion with Projection**:
```
x_0 = Gaussian noise [B, T, 6]
For t = T-1, ..., 0:
    ε_t = model(x_t, t, cond)
    x_t-1_denoised = (x_t - β_t / sqrt(1 - ᾱ_t) * ε_t) / sqrt(α_t) + σ_t * z_t
    
    IF projector is active:
        # Solve QP to find nearest feasible trajectory
        x_t-1 = x_t-1_denoised + ∇_proj(x_t-1_denoised, constraints)
```

### 2.7 DPCC Projector Setup

**File**: [FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) (lines 63-130)

```python
def setup_gen6_projector(args, config, scaler, variant):
    """
    Instantiate DPCC projection engine.
    """
    workspace_lb = np.array(config['workspace_bounds']['lb'])  # [0.3, -0.5, 0.0]
    workspace_ub = np.array(config['workspace_bounds']['ub'])  # [0.8, 0.5, 0.7]
    
    constraint_list = []
    
    # Bounds constraints on POSITION ONLY (dims 3, 4, 5)
    if 'bounds' in config.get('constraint_types', []):
        lb = np.array([-np.inf, -np.inf, -np.inf, workspace_lb[0], workspace_lb[1], workspace_lb[2]])
        ub = np.array([np.inf, np.inf, np.inf, workspace_ub[0], workspace_ub[1], workspace_ub[2]])
        constraint_list.append(['lb', lb])
        constraint_list.append(['ub', ub])
        
        # Mathematically interpreted as:
        # -∞ ≤ Δx ≤ +∞  (no action velocity bounds)
        # -∞ ≤ Δy ≤ +∞
        # -∞ ≤ Δz ≤ +∞
        # 0.3 ≤ x_ee ≤ 0.8    (workspace bounds)
        # -0.5 ≤ y_ee ≤ 0.5
        # 0.0 ≤ z_ee ≤ 0.7
    
    # Dynamics constraints: action → state mapping
    if 'dynamics' in config.get('constraint_types', []):
        # Explicit Euler: x_t+1 = x_t + action * dt
        constraint_list.append(('deriv', [3, 0]))  # dim3 (x_ee) driven by dim0 (Δx)
        constraint_list.append(('deriv', [4, 1]))  # dim4 (y_ee) driven by dim1 (Δy)
        constraint_list.append(('deriv', [5, 2]))  # dim5 (z_ee) driven by dim2 (Δz)
    
    # Initialize projector
    projector = Projector(
        horizon=args.horizon,                       # = 8
        transition_dim=6,                          # [action, state]
        action_dim=3,                              # First 3 dims
        constraint_list=constraint_list,
        normalizer=VisualNormalizerDict(scaler),   # Uses scaler bounds
        variant='states_actions',                  # Must be this for 6D
        dt=config.get('dt', 0.1),
        gradient=('gradient' in variant),          # Enable gradient injection
        solver='scipy',                            # QP solver
        device=args.device
    )
    
    return projector
```

**Projector QP Formulation** (from [diffuser/sampling.py](FM-PCC/diffuser/sampling.py)):
```
minimize    (1/2) * ||τ - τ_raw||_Q^2
subject to  lb ≤ τ ≤ ub                    (bounds on state positions)
            τ_t+1[3:] = τ_t[3:] + τ_t[0:3] * dt  (dynamics)
```

---

## PART 3: DETAILED TRAJECTORY DIMENSIONALITY COMPARISON

### 3.1 Training Data Pipeline

| Stage | **D3IL** | **Gen6** | Notes |
|---|---|---|---|
| Raw from dataset | action: [B, T, 3], state: [B, T, 7] | action: [B, T, 3], state: [B, T, 7] | Identical data source |
| State subset | Full 7D | **3D (EE only)** | Gen6 discards joint angles |
| Concatenation | Separate sequences | **x = [action, state] → [B, T, 6]** | Gen6 marries them |
| Scaler input | x: [B, T, 7], y: [B, T, 3] | x: [B, T, 6], y: unused | Different shapes fed to Scaler |
| Scaler output | x_mean: [7], x_std: [7], y_mean: [3], y_std: [3] | **x_mean: [6], x_std: [6]** | Different normalization stats |

### 3.2 Loss Function

| Aspect | **D3IL** | **Gen6** | Notes |
|---|---|---|---|
| Input tensor | action [B, 3] or [B, T, 3] | **x [B, T, 6]** | D3IL uses action only |
| Noise vector | ε ~ N(0, I), shape [B, 3] | **ε ~ N(0, I), shape [B, T, 6]** | Different dimensionality |
| Forward diffusion | x_noisy = α̂ * action + √(1−α̂) * ε | **x_noisy = α̂ * x + √(1−α̂) * ε** | Both follow DDPM theory |
| Model prediction | ε_pred = model(x_t, t, state, goal) → [B, 3] | **ε_pred = model(x_t, t, cond) → [B, T, 6]** | U-Net handles 6D |
| Loss computation | MSE(ε_pred, ε_true) on [B, 3] | **MSE(ε_pred, ε_true) on [B, T, 6]** | Each formulation correct for its trajectory |

### 3.3 Inference Sampling

| Aspect | **D3IL** | **Gen6** | Notes |
|---|---|---|---|
| Initialization | x_0 = Gaussian(0, I) [B, 3] | **x_0 = Gaussian(0, I) [B, T, 6]** | Noise matches diffusion target |
| Denoising loop | 16 steps, t = T-1...0 | **16 steps, t = T-1...0** | Same number of steps |
| At each step t | x_{t-1} = (x_t - β/√(1−ᾱ) * ε_t) / √α + σ*z | **Same formula on 6D** | Identical mathematics |
| Output | action_pred [B, 3] | **trajectory [B, T, 6]** | Different output interpretation |
| Post-processing | Clip to [-1, 1] (scaled bounds) | **Clip action dims [0:3] only** | Asymmetric clipping in Gen6 |
| Final action | action_unscaled = action_scaled * y_std + y_mean | **action_unscaled = traj[0:3] * scaler.x_std[0:3] + scaler.x_mean[0:3]** | Different inverse scaling |

### 3.4 Z-Axis Problem: Concrete Example

**Scenario**: Flat-table manipulation task. Robot only moves in XY plane.

**Dataset Statistics**:
```
action_z_data = [0.0001, -0.0002, 0.00015, ...]  # Micro-vibrations, mean ≈ 0
position_z_data = [0.6, 0.6, 0.6, ...]            # Height constant, mean ≈ 0.6, std ≈ 0
```

#### D3IL Baseline Processing

```python
# Scaler setup
y_mean[2] = 0.0                    # action_z mean
y_std[2] = 0.0001                  # action_z std (micro-vibrations)
y_std_safe[2] = max(0.0001, 1e-12) = 0.0001

# Scaled action_z bounds
y_bounds[0, 2] = (0.0 - 0.0) / 0.0001 = 0.0      # min
y_bounds[1, 2] = (0.0 - 0.0) / 0.0001 = 0.0      # max
# (Effectively zero bounds because data variance is tiny)

# During inference clipping
action_z_scale ∈ [-1, 1]  (raw diffusion output range)
action_z_clip = clamp(action_z_scale, 0.0, 0.0) = 0.0  # ALWAYS ZERO
action_z_unscaled = 0.0 * 0.0001 + 0.0 = 0.0    # ALWAYS ZERO
```

**Result**: D3IL learns to predict **100% zero Z-actions**, which is correct for the flat-table task.

#### Gen6 Processing

```python
# Gen6 concatenates [action_z, position_z]
x_data[..., 2] = action_z_data ≈ 0.0001 std
x_data[..., 5] = position_z_data ≈ 0.0 std

x_mean = [... 0.0 (action), ... 0.6 (position)]
x_std = [... 0.0001 (action), ... 0.0 (position)]
x_std_safe = [..., max(0.0001, 1e-2), ..., max(0.0, 1e-2)]
           = [..., 0.01, ..., 0.01]  # INFLATED from 0.0001 and 0.0!

# Scaled bounds
x_bounds[0, 5] = (0.6 - 0.6) / 0.01 = 0.0
x_bounds[1, 5] = (0.6 - 0.6) / 0.01 = 0.0
x_bounds[0, 2] = (0.0 - 0.0) / 0.01 = 0.0
x_bounds[1, 2] = (0.0 - 0.0) / 0.01 = 0.0
# (Still zero bounds due to perfect constraint satisfaction in data)

# But during inference:
# U-Net generates 6D noise: [Δx, Δy, Δz_action, x_ee, y_ee, z_ee]
# Random z-action noise: z_act_scale ≈ 0.5 (from Gaussian)
# Random z-position noise: z_pos_scale ≈ -0.3

# Clipping (ONLY actions [0:3] are clipped)
z_act_clipped = clamp(0.5, -5.0, 5.0) = 0.5
z_pos_clipped = -0.3  # NOT CLIPPED (state dimension)

# Unscaling
z_act_unscaled = 0.5 * 0.01 + 0.0 = 0.005         # Small but nonzero!
z_pos_unscaled = -0.3 * 0.01 + 0.6 = 0.597        # Moved 0.003m down!

# Projection then corrects, but creates coupled noise feedback
```

**Result**: Gen6 generates **spurious Z-actions** due to large epsilon scaling, then projects them out, creating **iterative coupling** between action and position denoising.

---

## PART 4: CRITICAL ARCHITECTURAL DIFFERENCES

### 4.1 State-Action Decoupling (Theoretical)

**D3IL Baseline**:
- Actions are the **unique decision variables** being optimized
- States are **observations/conditioning** (read-only)
- Diffusion operates on action space only
- States constrain but do not degrade

**Gen6**:
- Combined `[action, state]` = **single joint vector** being denoised
- Both treated symmetrically in diffusion
- U-Net sees them as coupled latent features
- Assumption: both were **independently optimized** in training (FALSE)

**Mathematical Issue**:
Gen6 projects a **coupled latent trajectory** generated by:
```
p(τ | image) = p([a_0...a_H, s_0...s_H] | image)
```

But applies bounds as if:
```
τ = [a_0...a_H, s_0...s_H] were independently optimized
```

In reality, the U-Net's noise prediction for state dimensions is **driven by observation encoding**, not independent action-state pairs.

### 4.2 Epsilon Scaling & Zero-Variance Handling

| Aspect | **D3IL** | **Gen6** |
|---|---|---|
| Epsilon added to std | `1e-12` | `1e-2` |
| Rationale | Tiny offset for numerical stability | **Large buffer to prevent zero-division** |
| Impact on Z-axis | Preserves data variance (0.0001 stays 0.0001) | **Inflates zero-variance dims 100x** |
| Con | May cause NaN in rare cases | **Introduces artificial noise distribution mismatch** |
| Upside | Faithful to training data distribution | **Prevents projection cycling** |

### 4.3 Clamping Strategy

| Aspect | **D3IL** | **Gen6** |
|---|---|---|
| What's clamped | Action dimensions only | **Action dimensions [0:3] only** |
| Range | `[-1, 1]` (normalized bounds) | `[-5.0, 5.0]` (wider, safer) |
| State dimensions | Unclamped (implicit) | Explicitly unclamped |
| Rationale | Prevent action saturation | **Actions freely denoised; states follow via dynamics** |

### 4.4 Vision Encoding

Both D3IL and Gen6 share the **exact same** vision encoder:
- Multi-image RGB fusion (agentview + in-hand)
- Maps to shared latent space
- Conditioning both action and state predictions

**No architectural difference here.**

### 4.5 Inverse Scaling (Unscaling) for Control

**D3IL**:
```python
action_unscaled = action_scaled * y_std_safe + y_mean
                = action_scaled * (0.0001 + 1e-12) + 0.0
                ≈ action_scaled * 0.0001
```

**Gen6**:
```python
action_unscaled = traj[0:3] * x_std_safe[0:3] + x_mean[0:3]
                = action_scaled * (0.0001 + 1e-2) + 0.0
                ≈ action_scaled * 0.01  # 100x LARGER
```

**Impact**: Gen6 magnifies small diffusion noise by **100x** compared to D3IL.

---

## PART 5: TRAJECTORY SHAPES SUMMARY

### Input Shapes at Each Stage

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATA LOADING                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  state [B, T, 7]     action [B, T, 3]                        │
│ Gen6:  state [B, T, 7]     action [B, T, 3]                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  state unchanged     action unchanged                        │
│        Feed to vision encoder separately                           │
│                                                                    │
│ Gen6:  state[:, :, :3] → subset to EE position only              │
│        x = cat([action, state_subset], dim=-1) → [B, T, 6]       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SCALER FIT                                                          │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  scaler.fit(state=[B,T,7], action=[B,T,3])                  │
│        → mean/std: [7D] and [3D]                                   │
│                                                                    │
│ Gen6:  scaler.fit(x=[B,T,6], y=None)                              │
│        → mean/std: [6D] only                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ VISION ENCODING                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  obs_encoder(images + state_7d) → [B, T, obs_dim]           │
│        state_encoded [B, T, ~128]                                  │
│                                                                    │
│ Gen6:  obs_encoder(images + state_3d) → [B, T, obs_dim]           │
│        obs_encoded [B, T, ~128]                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING LOSS                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  input to diffusion: action [B, T, 3]                       │
│        cond: state_encoded [B, T, 128]                            │
│        loss: MSE(ε_pred[B,T,3], ε_true[B,T,3])                    │
│                                                                    │
│ Gen6:  input to diffusion: x [B, T, 6]                            │
│        cond: visual + obs_encoded                                 │
│        loss: MSE(ε_pred[B,T,6], ε_true[B,T,6])                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE SAMPLING                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  x_0 ~ N(0,I) [B, 3]                                         │
│        For t in range(16): x_{t-1} ~ denoise(x_t)                  │
│        output: action [B, 3]                                       │
│                                                                    │
│ Gen6:  x_0 ~ N(0,I) [B, T, 6]                                      │
│        For t in range(16): x_{t-1} ~ denoise(x_t)                  │
│        OPTIONAL: x_{t-1} ← project(x_{t-1), constraints)          │
│        output: trajectory [B, T, 6]                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ POST-PROCESSING & INVERSE SCALING                                   │
├─────────────────────────────────────────────────────────────────────┤
│ D3IL:  action_pred [B, 3] → clip → unscale                         │
│        action_cmd = action_pred * y_std_safe + y_mean              │
│                                                                    │
│ Gen6:  trajectory [B, T, 6] → extract [B, 3] → clip → unscale    │
│        action_cmd = traj[0:3] * x_std_safe[0:3] + x_mean[0:3]     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PART 6: KEY INSIGHTS FOR VISUAL ALIGNING ALIGNMENT

### Insight 1: The Dimensionality Mismatch Problem

Gen6's approach of combining action and state into a single 6D diffusion space creates a **semantic mismatch** with optimal control theory:

1. **In theory**: Actions are _controls_ (decision variables), states are _observations_ (state of the system).
2. **In Gen6**: Both are treated as _latent variables_ of equal importance, denoised jointly.
3. **The conflict**: The U-Net doesn't recognize that state prediction should depend on action prediction via dynamics. Instead, it learns a _coupled generative model_ where both are conditionally independent given the image.

### Insight 2: The Z-Axis Sensitivity

For flat-table tasks, Gen6's large epsilon (`1e-2`) inflates zero-variance dimensions. This causes:
1. Spurious noise generation in Z during diffusion
2. QP solver must correct it in projection loop
3. Coupling between action and state denoising creates iterative refinement (good?) or cycling (bad?)

### Insight 3: Scaler API Compatibility

Gen6 retrofits a 6D trajectory into D3IL's action/state scaler API, which expects separate tensors. The workaround (treating concatenated data as x-input only) works but obscures the asymmetry.

### Proposed Fix Direction

To properly implement state-action projection, the architecture should:
1. **Keep diffusion on action space only** (like D3IL)
2. **Use forward dynamics** to compute implied states: $s_{t+1} = s_t + a_t \cdot dt$
3. **Apply bounds constraints** to state sequence, not to diffusion input
4. **Project actions**, not concatenated vectors

This would recover the theoretical separation while preserving the DPCC safety guarantees.

---

## Appendix: File Locations Reference

### D3IL Baseline Files

| Component | File | Key Lines |
|---|---|---|
| Scaler | `d3il/agents/utils/scaler.py` | 1-135 |
| Vision Agent | `d3il/agents/act_vision_agent.py` | 26-450 |
| DDPM Policy | `d3il/agents/ddpm_vision_agent.py` | 79-300 |
| Diffusion Model | `d3il/agents/models/diffusion/diffusion_policy.py` | 1-300 |
| Diffusion Core | `d3il/agents/models/diffusion/diffusion_models.py` | 1-400 |
| Utils | `d3il/agents/models/diffusion/utils.py` | 1-250 |

### Gen6 Files

| Component | File | Key Lines |
|---|---|---|
| Visual Diffusion | `FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py` | 1-80 |
| Scaler | `FM-PCC/ddpm_encdec_vision/utils/scaler.py` | 1-150 |
| Evaluation | `FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | 50-300 |
| Projector Setup | `FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | 63-130 |
| Training | `FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py` | 1-200 |
