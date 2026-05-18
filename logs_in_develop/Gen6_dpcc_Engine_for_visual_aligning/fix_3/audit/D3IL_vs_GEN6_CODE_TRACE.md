# Exact Code Trace: Trajectory Dimensionality Through Both Systems

## A. D3IL Baseline - Complete Code Trace

### Step 1: Data Loading
**File**: `d3il/agents/act_vision_agent.py` lines 190-250

```python
# Training loop
for data in self.train_dataloader:
    bp_imgs, inhand_imgs, obs, action, mask = data
    
    # obs.shape = [B=4, T=8, 7]  ← Full proprioception
    # action.shape = [B=4, T=8, 3] ← Cartesian deltas
    # bp_imgs.shape = [B=4, T=8, 3, 224, 224]
    # inhand_imgs.shape = [B=4, T=8, 3, 224, 224]
```

**Concrete Shapes**:
```
Batch size: 4
Horizon: 8 timesteps
obs: torch.Size([4, 8, 7])  ← [robot_ee_x, robot_ee_y, robot_ee_z, q1, q2, q3, q4]
action: torch.Size([4, 8, 3]) ← [Δx_cmd, Δy_cmd, Δz_cmd]
```

### Step 2: Scaling
**File**: `d3il/agents/utils/scaler.py` lines 88-100

```python
obs_scaled = self.scaler.scale_input(obs)      # [B=4, T=8, 7]
action_scaled = self.scaler.scale_output(action)  # [B=4, T=8, 3]

# Inside scaler:
def scale_output(self, y):
    y = y.to(self.device)
    if self.scale_data:
        out = (y - self.y_mean) / (self.y_std + 1e-12 * ones(...))
        return out.to(torch.float32)
    else:
        return y.to(self.device)
        
# For flat-table Z-axis scenario:
# y_mean[2] = 0.0 (action_z mean)
# y_std[2] = 0.0001 (action_z std from vibrations)
# y_std_safe[2] = 0.0001 + 1e-12 ≈ 0.0001
# After scaling: action_scaled[..., 2] ∈ [-1.5, 1.5] (approximate range)
```

**After Scaling - Shapes**:
```
obs_scaled: torch.Size([4, 8, 7])   ← Normalized
action_scaled: torch.Size([4, 8, 3]) ← Normalized
```

### Step 3: Vision Encoding
**File**: `d3il/agents/act_vision_agent.py` lines 45-88

```python
state_tuple = (bp_imgs, inhand_imgs, obs_scaled)

state_embedding = self.model.get_embedding(state_tuple)

# Inside get_embedding():
def get_embedding(self, inputs):
    if self.visual_input:
        agentview_image, in_hand_image, state = inputs
        B, T, C, H, W = agentview_image.size()  # B=4, T=8, C=3, H=224, W=224
        
        agentview_image = agentview_image.view(B * T, C, H, W)     # [32, 3, 224, 224]
        in_hand_image = in_hand_image.view(B * T, C, H, W)         # [32, 3, 224, 224]
        state = state.view(B * T, -1)                               # [32, 7]
        
        obs_dict = {
            "agentview_image": agentview_image,      # [32, 3, 224, 224]
            "in_hand_image": in_hand_image,          # [32, 3, 224, 224]
            "robot_ee_pos": state                    # [32, 7]
        }
        
        obs = self.obs_encoder(obs_dict)             # [32, 128] (encoder output dim)
        obs = obs.view(B, T, -1)                     # [4, 8, 128]
    
    return obs
```

**After Vision Encoding - Shapes**:
```
state_embedding: torch.Size([4, 8, 128])  ← Encoded observation features
action_scaled (unchanged): torch.Size([4, 8, 3])
```

### Step 4: Diffusion Training
**File**: `d3il/agents/models/diffusion/diffusion_policy.py` lines 240-270

```python
# Training step
def train_step(self, state, actions):
    self.model.train()
    
    # state.shape = [4, 8, 128]    ← encoded observations (conditioning)
    # actions.shape = [4, 8, 3]    ← actions to denoise
    # goal.shape = [4, 8, 128] if goal-conditioned
    
    loss = self.model.loss(actions, state, goal=None)
    # ↓
    # Calls Diffusion.loss():
```

**Inside Diffusion.loss()**:
```python
def loss(self, x, state, goal=None):
    """
    Args:
        x: [4, 8, 3] ← ACTIONS ONLY, not states
        state: [4, 8, 128] ← conditioning
        goal: [4, 8, 128]
    """
    batch_size = len(x)  # = 4
    t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
    # t.shape = [4] with values in [0, 15]
    
    return self.p_losses(x, state, goal, t)  # Pass 3D actions, not states

def p_losses(self, x_start, state, goal, t):
    # x_start.shape = [4, 8, 3]  ← 3D ACTION trajectories
    
    noise = torch.randn_like(x_start)  # [4, 8, 3]
    
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    # x_noisy.shape = [4, 8, 3]
    
    # Model predicts noise in action space
    x_recon = self.model(x_noisy, t, state, goal)
    # x_recon.shape = [4, 8, 3] ← predicted noise
    
    loss = self.loss_fn(x_recon, noise)  # MSE on [4, 8, 3]
    return loss
```

**Inside DiffusionMLPNetwork.forward()**:
```python
def forward(self, x, t, state, goal):
    # x.shape = [4, 8, 3]
    # t.shape = [4]
    # state.shape = [4, 8, 128]
    # goal.shape = [4, 8, 128]
    
    t = self.temp_layers(t)  # [4, 24] → time embedding
    
    if len(state.shape) == 3:  # True
        t = einops.rearrange(t, 'batch dim -> batch 1 dim')  # [4, 1, 24]
        x = torch.cat([x, t, state, goal], dim=2)
        # Concatenation along feature dimension:
        # [4, 8, 3] + [4, 8, 24] + [4, 8, 128] + [4, 8, 128]
        # = [4, 8, 283]  ← Input to MLP
    
    out = self.layers(x)  # [4, 8, 283] → [4, 8, 3] (noise for actions)
    return out
```

**Training Loss Computation**:
```
L = MSE(predicted_noise[4,8,3], true_noise[4,8,3])
  = (1/96) * sum(
      (ε_pred[b,t,d] - ε_true[b,t,d])² 
      for b in [0,3], t in [0,7], d in [0,2]
    )
```

### Step 5: Inference Sampling
**File**: `d3il/agents/models/diffusion/diffusion_policy.py` lines 150-200

```python
def sample(self, state, goal):
    # state.shape = [4, 128]  ← Single encoded observation (current time only)
    # goal.shape = [4, 128]
    
    batch_size = state.shape[0]  # 4
    shape = (batch_size, self.action_dim)  # (4, 3)
    
    action = self.p_sample_loop(state, goal, shape)
    return action.clamp_(self.min_action, self.max_action)

def p_sample_loop(self, state, goal, shape):
    # shape = (4, 3)
    batch_size = shape[0]  # 4
    
    x = torch.randn(shape, device=self.device)  # [4, 3] pure Gaussian
    
    for i in reversed(range(0, self.n_timesteps)):  # i = 15, 14, ..., 0
        timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
        # timesteps.shape = [4]
        
        x = self.p_sample(x, timesteps, state, goal)
        # x.shape stays [4, 3]
    
    action = x.clamp_(self.min_action, self.max_action)
    return action  # [4, 3]
```

**Sampling Loop Iteration (t=3 example)**:
```
i = 3
timesteps = [3, 3, 3, 3]  # [4]

# Estimate posterior
x_recon = self.predict_start_from_noise(x, t=timesteps, noise=pred_noise)
# x_recon.shape = [4, 3]

model_mean, posterior_var, posterior_log_var = self.q_posterior(
    x_start=x_recon, x_t=x, t=timesteps
)
# model_mean.shape = [4, 3]
# posterior_log_var.shape = [4, 3]

# Sample from posterior
noise = torch.randn_like(x)  # [4, 3]
nonzero_mask = (1 - (timesteps == 0).float()).reshape(4, 1)  # [4, 1]
x = model_mean + nonzero_mask * (0.5 * posterior_log_var).exp() * noise
```

### Step 6: Unscaling & Control
**File**: `d3il/agents/utils/scaler.py` lines 108-112

```python
def inverse_scale_output(self, y):
    # y.shape = [4, 3] (sampled actions, still normalized)
    
    if self.scale_data:
        out = y * (self.y_std + 1e-12 * ones(...)) + self.y_mean
        return out
    
    # For flat-table Z-axis:
    # y[..., 2] * (0.0001 + 1e-12) + 0.0
    # ≈ y[..., 2] * 0.0001
    
    # If y[..., 2] from diffusion = 0.5:
    # action_z_cmd = 0.5 * 0.0001 = 0.00005  ← Essentially ZERO ✓
```

**Final Output**:
```
action_cmd.shape = [4, 3]
action_cmd values ≈ [0.05m/s, -0.03m/s, 0.00005m/s]  ← Z is ~0 ✓
    ↓
Sent to robot EE controller
```

---

## B. Gen6 Visual-Aligning - Complete Code Trace

### Step 1: Data Loading & Concatenation
**File**: `FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` and training code

```python
# Training loop (conceptual)
for data in dataloader:
    bp_imgs, inhand_imgs, obs, action, mask = data
    
    # obs.shape = [B=4, T=8, 7]     ← Full proprioception
    # action.shape = [B=4, T=8, 3]  ← Cartesian deltas
    # bp_imgs.shape = [B=4, T=8, 3, 224, 224]
    # inhand_imgs.shape = [B=4, T=8, 3, 224, 224]
    
    # 🔴 GEN6 CRITICAL: Extract only EE position (first 3 dims)
    obs_3d = obs[:, :, :3]  # [B=4, T=8, 3]
    
    # 🔴 GEN6 CRITICAL: Concatenate into 6D trajectory
    x = torch.cat([action, obs_3d], dim=-1)
    # x.shape = [B=4, T=8, 6] = [Δx, Δy, Δz, x_ee, y_ee, z_ee]
```

**After Concatenation - Shapes**:
```
x (full 6D): torch.Size([4, 8, 6])
├─ dims [0:3]: [Δx, Δy, Δz] (actions)
└─ dims [3:6]: [x_ee, y_ee, z_ee] (positions)
```

### Step 2: Scaling (Different from D3IL)
**File**: `FM-PCC/ddpm_encdec_vision/utils/scaler.py` lines 1-150

```python
# Gen6 scaler fits 6D concatenated data
def __init__(self, x_data, y_data=None, scale_data=True, device='cuda'):
    # x_data.shape = [total_samples, 6]  ← Combined [action, state]
    # y_data is IGNORED in Gen6
    
    self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)      # [6]
    self.x_std = torch.from_numpy(x_data.std(0)).to(device)        # [6]
    
    # CRITICAL: Larger epsilon
    self.x_std_safe = self.x_std + 1e-2 * torch.ones_like(self.x_std)
    
    # For flat-table scenario:
    # x_data[:, 2] = action_z = [0.0001, -0.0002, ...] vibrations
    # x_data[:, 5] = pos_z = [0.6, 0.6, ...] constant
    
    # self.x_mean = [0.0, 0.0, 0.0, 0.4, 0.0, 0.6]
    # self.x_std_raw = [0.1, 0.1, 0.0001, 0.05, 0.05, ~0.0]
    
    # 🔴 INFLATION: std + 1e-2
    # self.x_std_safe = [0.1, 0.1, 0.0101, 0.05, 0.05, 0.01]
    #                         ↑ Was 0.0001, now 0.01 (100x!)
```

**Bounds Calculation**:
```python
# Scaled bounds (in normalized space)
self.x_bounds[0, :] = (x_data.min(0) - x_data.mean(0)) / x_std_safe
self.x_bounds[1, :] = (x_data.max(0) - x_data.mean(0)) / x_std_safe

# For Z-axis:
# x_bounds[0, 2] = (0.0 - 0.0) / 0.01 = 0.0
# x_bounds[1, 2] = (0.0 - 0.0) / 0.01 = 0.0
# x_bounds[0, 5] = (0.6 - 0.6) / 0.01 = 0.0
# x_bounds[1, 5] = (0.6 - 0.6) / 0.01 = 0.0
```

**Scaling Function**:
```python
def scale_input(self, x):
    # x is the concatenated 6D trajectory [B, T, 6]
    
    x = x.to(self.device)
    if self.scale_data:
        out = (x - self.x_mean) / (self.x_std_safe)
        return out.to(torch.float32)
    else:
        return x.to(self.device)
```

**After Scaling - Shapes**:
```
x_scaled: torch.Size([4, 8, 6])
├─ dims [0:3]: action_scaled (from ~0.0 std data → normalized range)
└─ dims [3:6]: pos_scaled (from ~0.0 std data → normalized range)
```

### Step 3: Vision Encoding (Same as D3IL)
```python
# Same as D3IL
obs_encoded = vision_encoder(bp_imgs, inhand_imgs, obs_3d)  # [4, 8, 128]
```

### Step 4: Diffusion Training - 🔴 KEY DIFFERENCE
**File**: `FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py` lines 10-30

```python
def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
    """
    🔴 GEN6: Loss on 6D concatenated trajectory
    """
    # obs.shape = [4, 8, 3]  (only EE position)
    # act.shape = [4, 8, 3]  (actions)
    
    # 🔴 CONCATENATE into 6D
    x = torch.cat([act, obs], dim=-1)  # [4, 8, 6]
    # x = [Δx, Δy, Δz, x_ee, y_ee, z_ee]
    
    cond = {
        'visual': (bp_imgs, inhand_imgs, obs),
        0: obs[:, 0]  # First frame state [4, 3]
    }
    
    batch_size = len(x)  # 4
    t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
    # t.shape = [4]
    
    return self.p_losses(x, cond, t)

def p_losses(self, x_start, cond, t):
    # x_start.shape = [4, 8, 6]  🔴 6D, not 3D!
    
    noise = torch.randn_like(x_start)  # [4, 8, 6]
    
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    # x_noisy.shape = [4, 8, 6]
    
    # Model predicts noise in 6D space
    x_recon = self.model(x_noisy, cond, t)
    # x_recon.shape = [4, 8, 6]  🔴 PREDICTIONS FOR BOTH ACTION AND STATE!
    
    loss = F.mse_loss(x_recon, noise)  # MSE on [4, 8, 6]
    return loss
```

**Inside VisualUNet.forward()**:
```python
def forward(self, x, cond, t):
    # x.shape = [4, 8, 6]  ← Combined 6D trajectory
    # cond contains visual conditioning
    # t.shape = [4]
    
    # VisualUNet processes the full 6D trajectory
    # (different from D3IL's separate action/state conditioning)
    
    # 🔴 KEY: Model treats action dimensions [0:3] and state dimensions [3:6]
    #         as equally important features to denoise
    
    out = self.model(x, cond, t)  # [4, 8, 6] → predicted noise on 6D
    return out
```

**Training Loss Computation**:
```
L = MSE(predicted_noise[4,8,6], true_noise[4,8,6])
  = (1/192) * sum(
      (ε_pred[b,t,d] - ε_true[b,t,d])² 
      for b in [0,3], t in [0,7], d in [0,5]
    )
    
    🔴 INCLUDES prediction errors for BOTH action and state dimensions
       equally weighted!
```

### Step 5: Inference Sampling with Projection
**File**: `FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py` lines 50-90

```python
def forward(self, cond, projector=None):
    # cond is condition dict with visual + state info
    # projector is DPCC constraint enforcer (optional)
    
    # Calls p_sample_loop (inherited from parent)
    return self.conditional_sample(cond, projector=projector)

# In parent GaussianDiffusion:
def p_sample_loop(self, shape, cond, projector=None, constraints=None):
    # shape = (4, 8, 6)  🔴 6D shape for concatenated trajectory
    
    batch_size = shape[0]  # 4
    x = torch.randn(shape, device=self.device)  # [4, 8, 6] pure Gaussian
    
    # Initial conditioning snap
    x = apply_conditioning(x, cond, action_dim=3)
    # Forces x[b, 0, 3:6] = cond[0][b]  (snap initial state)
    
    for i in reversed(range(0, self.n_timesteps)):  # 16 to 0
        timesteps = torch.full((batch_size,), i, device=self.device).long()
        
        x = self.p_sample(
            x, timesteps, cond,
            projector=projector,
            constraints=constraints
        )
        
        # Re-apply initial state snap
        x = apply_conditioning(x, cond, action_dim=3)
    
    # Clamp action dimensions only
    x[..., :self.action_dim].clamp_(-5.0, 5.0)  # action dims [0:3]
    # State dims [3:6] are NOT clamped!
    
    return x  # [4, 8, 6]
```

**Denoising Step with Projection**:
```python
def p_mean_variance(self, x, cond, t, projector=None, constraints=None):
    # x.shape = [4, 8, 6]  (noisy 6D trajectory)
    # t.shape = [4]
    
    # Get noise prediction
    epsilon = self.model(x, cond, t)  # [4, 8, 6]
    
    # Convert to reconstruction
    x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
    # x_recon.shape = [4, 8, 6]
    
    if self.clip_denoised:
        # 🔴 ASYMMETRIC CLIPPING
        x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)  # Actions [0:3]
        # State dims [3:6] are NOT clamped!
    
    # Compute posterior
    model_mean, posterior_var, posterior_log_var = self.q_posterior(
        x_start=x_recon, x_t=x, t=t
    )  # All [4, 8, 6]
    
    # 🔴 OPTIONAL PROJECTION (safety)
    if projector is not None and projector.gradient:
        trajectory = x_recon[:, :, :-self.goal_dim]  # [4, 8, 6]
        
        # Solve QP: minimize ||τ - τ_raw|| subject to constraints
        grad = projector.compute_gradient(trajectory, constraints)
        # grad.shape = [4, 8, 6]
        
        # Inject gradient to move toward feasible region
        model_mean = model_mean + grad
    
    return model_mean, posterior_var, posterior_log_var
```

**Sampling Loop Iteration (t=3, with projection enabled)**:
```
i = 3
timesteps = [3, 3, 3, 3]

# Denoising
epsilon = model(x, cond, t)  # [4, 8, 6]
x_recon = predict_from_noise(x, epsilon)  # [4, 8, 6]

# 🔴 Clamp only action dims
x_recon[..., :3].clamp_(-5, 5)  # Actions
# State dims [3:6] unclamped

# Posterior
model_mean, posterior_log_var, ... = q_posterior(x_recon, x, t)

# 🔴 PROJECTION (corrects violations)
if projector:
    # At this point, x_recon might violate workspace bounds
    # e.g., x_recon[0, :, 5] (z-position) = -0.01 (below table)
    
    grad = projector.solve_qp(x_recon)  # [4, 8, 6]
    # grad[0, :, 5] might be [+0.01, +0.01, ...]  (push back up)
    
    model_mean = model_mean + grad
    # model_mean[0, :, 5] now corrected back toward 0.0

# Sample from posterior
z_noise = randn_like(x)  # [4, 8, 6]
nonzero_mask = (1 - (t == 0)) = 1.0
x = model_mean + sqrt(0.5 * posterior_log_var) * z_noise
# x.shape = [4, 8, 6]
```

### Step 6: Unscaling - 🔴 CRITICAL ISSUE
**File**: `FM-PCC/ddpm_encdec_vision/utils/scaler.py` lines 108-112

```python
def inverse_scale_input(self, x):
    # x.shape = [4, 8, 6]  (sampled trajectory, still scaled)
    
    if self.scale_data:
        out = x * (self.x_std_safe) + self.x_mean
        return out
    
    # For flat-table Z-axis:
    # x[..., 2] * (0.01) + 0.0      🔴 100x larger!
    # x[..., 5] * (0.01) + 0.6
    
    # If x[..., 2] from diffusion = 0.5:
    # action_z_cmd = 0.5 * 0.01 = 0.005  🔴 Not ~0!
    #
    # If x[..., 5] from diffusion = -0.5:
    # pos_z_result = -0.5 * 0.01 + 0.6 = 0.595  🔴 Moves down 0.005m!
```

**Final Output**:
```
traj_cmd.shape = [4, 8, 6]
traj_cmd[0, :, 0:3] = action predictions  [0.05m/s, -0.03m/s, 0.005m/s]  🔴 Z inflated!
traj_cmd[0, :, 3:6] = position state      [0.402m, 0.001m, 0.595m]  🔴 Z drifted down!
    ↓
Extract actions [0:3] and send to controller
    ↓
🔴 PROBLEM: Z-action is 100x too large, creating spurious vertical motion!
            (Then projected out, but creates coupling loop)
```

---

## C. Side-by-Side Comparison at Each Step

```
STEP          D3IL Baseline                  Gen6 Visual-Aligning
═════════════════════════════════════════════════════════════════════════

Entry         act:[B,T,3] s:[B,T,7]         act:[B,T,3] s:[B,T,7]
              (separate streams)             → cat → x:[B,T,6]

Scaler        x_mean:[7], x_std:[7]         x_mean:[6], x_std:[6]
              y_mean:[3], y_std:[3]          y unused
              eps=1e-12                      eps=1e-2 🔴

After         obs_enc:[B,T,128]             obs_enc:[B,T,128]
Vision        act:[B,T,3]                   x:[B,T,6]

Loss Mode     DIFFUSION ON ACTIONS ONLY     🔴 DIFFUSION ON 6D
              loss(a:[B,T,3], ...)          loss(x:[B,T,6], ...)

Model Out     ε:[B,T,3] (noise for a)      🔴 ε:[B,T,6] (noise for x)

Sample        x_0:[B,3]                     🔴 x_0:[B,T,6]
Init          for t loop: x:[B,3]           for t loop: x:[B,T,6]

Z-Unscale     a_z * 0.0001 ≈ ~0 ✓          🔴 a_z * 0.01 = inflated!

Projection    None                          🔴 DPCC (optional)

Final         action:[B,3]                  traj:[B,T,6] → extract [0:3]
Output        ✓ Correct                     🔴 Issues (#1, #2)
```

---

## Key Takeaways from Code Trace

1. **Scaler Epsilon Inflation**: Gen6's `1e-2` vs D3IL's `1e-12` creates **100x difference** for zero-variance dims
2. **Joint vs Separate**: D3IL learns P(a|s,img), Gen6 learns P([a,s]|img) — different problem formulation
3. **Asymmetric Clipping**: Only action dims [0:3] are clamped in Gen6, state [3:6] are unclamped
4. **Coupling**: Gen6's projection loop introduces feedback between denoising and constraint satisfaction
5. **Unscaling Mismatch**: Gen6 uses `x_std_safe` for both actions and states, magnifying action errors
