# FM-D Engine: Code Explanation & Architecture

**Date**: 2026-05-12  
**Project**: FM-PCC (Flow Matcher Predictive Control)  
**Component**: FM-D (Flow Matcher-Drifting) Implementation  
**Audience**: Developers, researchers, maintainers

---

## Overview

FM-D is a **hybrid generative control architecture** that combines:

$$v(x, t) = v_\theta(x, t) + \lambda \cdot \nabla_x \mathcal{L}_{\text{drift}}(x)$$

Where:
- $v_\theta(x,t)$ = learned velocity field (Flow Matching)
- $\mathcal{L}_{\text{drift}}(x)$ = trajectory quality loss
- $\lambda$ = drift weight (tunable, scheduled during training)
- $\nabla_x$ = gradient w.r.t. trajectory state

This creates **deterministic ODE integration** guided by expert trajectory distribution.

---

## Code Architecture

### 1. Drift Loss Computation (`models/drift_loss.py`)

#### Class: `DriftLoss`

**Purpose**: Measure deviation of sampled trajectory from expert distribution.

**Constructor Parameters**:
```python
def __init__(
    self,
    trajectory_dim: int,           # e.g., 28 for 7-DOF arm
    loss_type: str = "kl_divergence",  # or "mmd", "adversarial"
    memory_bank_size: int = 5000,  # Max expert trajectories
    temperature: float = 0.1,      # Softmax scaling
):
```

**Memory Bank**:
- Circular buffer storing expert trajectory samples
- Shape: `(memory_bank_size, trajectory_dim)`
- Updated via `update_memory_bank(trajectories)`
- Accessed via `sample()` for batch retrieval

**Loss Variants**:

1. **KL Divergence** (`compute_kl_divergence`)
   ```
   Loss = -log Pr[x_sampled matches expert]
   
   Steps:
   1. Encode sampled traj via MLP → q_z (128D embedding)
   2. Encode reference trajs via same MLP → p_z
   3. Compute softmax similarity: exp(-||q_z - p_z||²)
   4. KL = -log(max_probability over all references)
   ```
   - **Pros**: Simple, differentiable, captures mode-seeking
   - **Cons**: Ignores full distribution structure

2. **MMD (Maximum Mean Discrepancy)** (`compute_mmd_loss`)
   ```
   MMD² = E[K(q,q)] - 2E[K(q,p)] + E[K(p,p)]
   
   where K(x,y) = exp(-||x-y||² / 2σ²)  # RBF kernel
   ```
   - **Pros**: Measures full distribution distance, non-parametric
   - **Cons**: More expensive, kernel hyperparameter tuning

3. **Adversarial** (`compute_adversarial_loss`)
   ```
   Gen Loss:  E[-log D(q_z)]    # fool discriminator
   Dis Loss:  E[-log D(p_z)] + E[log(1-D(q_z))]  # real vs fake
   ```
   - **Pros**: Flexible, can capture complex distributions
   - **Cons**: Training instability, requires careful tuning

**Gradient Computation** (`get_gradient`):
```python
def get_gradient(self, trajectory):
    # Requires gradient tracking on trajectory
    trajectory.requires_grad_(True)
    loss = self.forward(trajectory)
    loss.backward()
    return trajectory.grad  # Used in ODE solver
```

---

### 2. Drift-Augmented U-Net (`models/drift_unet.py`)

#### Class: `DriftConditioner`

**Purpose**: Encode trajectory history + drift metrics into conditioning embeddings.

```python
def forward(self, trajectory, drift_metrics=None):
    # trajectory: (B, T, state_dim) → average over time
    # drift_metrics: (B, metric_dim) optional quality metrics
    # Output: (B, cond_dim) embedding
    
    # MLP layers: state_dim → hidden → cond_dim
    # Applied per-batch, normalized output
```

**Example flow**:
```
Input trajectory (B, T, 28)
    ↓ [Average over time]
    ↓
(B, 28)
    ↓ [Linear(28, 128)]
    ↓ [ReLU]
    ↓ [Linear(128, 128)]
    ↓ [ReLU]
    ↓ [Linear(128, 64)]
    ↓ [LayerNorm]
    ↓
Output: (B, 64) drift conditioning
```

#### Class: `DriftAugmentedUNet1D`

**Purpose**: Wrap base U-Net with drift conditioning stream.

```python
def forward(
    self,
    x,                    # (B, T, state_dim) trajectory
    cond,                 # (B, cond_dim) original conditioning
    t,                    # (scalar or B,) time step
    trajectory=None,      # (B, T', state_dim) for drift encoding
    drift_metrics=None,   # (B, metric_dim) optional metrics
    use_dropout=False,    # For classifier-free guidance
):
    # Compute drift conditioning (if trajectory provided)
    if trajectory is not None:
        drift_cond = self.drift_conditioner(trajectory, drift_metrics)
        # Concatenate: [original_cond | drift_cond]
        augmented_cond = torch.cat([cond, drift_cond], dim=-1)
    else:
        augmented_cond = cond
    
    # Fuse if dimensions mismatch
    if self.cond_fusion is not None:
        augmented_cond = self.cond_fusion(augmented_cond)
    
    # Call base U-Net with augmented conditioning
    return self.base_unet(x, augmented_cond, t, ...)
```

**Key Design**: U-Net remains unchanged; drift is prepended as conditioning.

---

### 3. ODE Solvers with Drift (`sampling/drift_ode_solvers.py`)

#### Class: `DriftAugmentedVelocityField`

**Purpose**: Wrapper that injects drift gradient into velocity computation.

```python
def __call__(self, t, x):
    # Compute base velocity from FM model
    velocity = self.velocity_fn(t, x)
    
    # Add drift guidance
    if self.drift_weight > 0 and self.drift_loss_fn is not None:
        drift_grad = self.drift_loss_fn(x)
        
        # Clip for numerical stability
        drift_norm = ||drift_grad||₂
        clipped = drift_grad * min(drift_norm, drift_clip) / drift_norm
        
        # Combine
        velocity = velocity + drift_weight * clipped
    
    return velocity
```

**Numerical Stability**:
- Gradient clipping prevents divergent trajectories
- Separate control of FM vs. drift contributions
- Temperature scaling for adaptive adjustment

#### Class: `DriftODESolver`

**Purpose**: Unified interface for multiple ODE backends.

**Backend Support**:
1. **Legacy Euler** (always available)
   ```python
   for step in range(num_steps):
       v = velocity_fn(t, x)
       x = x + dt * v
   ```
   - Simple, stable
   - First order: O(dt) error

2. **RK4** (4th order, always available)
   ```python
   k1 = velocity_fn(t, x)
   k2 = velocity_fn(t + dt/2, x + dt/2 * k1)
   k3 = velocity_fn(t + dt/2, x + dt/2 * k2)
   k4 = velocity_fn(t + dt, x + dt * k3)
   x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
   ```
   - Better accuracy: O(dt⁴)
   - More function evaluations

3. **torchdiffeq** (if installed)
   ```python
   solution = odeint(
       velocity_fn, x0, t,
       method='dopri5',  # adaptive stepsize
       rtol=1e-5,
       atol=1e-6,
   )
   ```
   - Adaptive stepping
   - Automatic error control

**Method Selection**:
```python
solver = DriftODESolver(
    solver_method='dopri5',     # method (if torchdiffeq)
    solver_backend='torchdiffeq',  # backend
    rtol=1e-5,                  # relative tolerance
    atol=1e-6,                  # absolute tolerance
)
```

**solve() Function**:
```python
def solve(
    self,
    velocity_fn,         # v(t, x)
    x0,                 # (B, state_dim) initial state
    t_span,             # (t_start, t_end)
    num_steps=10,       # for fixed solvers
    drift_loss_fn=None, # gradient function
    drift_weight=0.1,   # λ
):
    # Wrap velocity with drift
    if drift_weight > 0:
        augmented_fn = DriftAugmentedVelocityField(
            velocity_fn,
            drift_loss_fn,
            drift_weight,
        )
    else:
        augmented_fn = velocity_fn
    
    # Choose solver
    if backend == 'torchdiffeq':
        return self._solve_torchdiffeq(...)
    else:
        return self._solve_legacy(...)
```

---

### 4. Training Integration (`utils/drift_training.py`)

#### Class: `DriftLossScheduler`

**Purpose**: Manage drift weight $\lambda$ during training.

**Three Modes**:

1. **Warmup** (recommended)
   ```
   λ(step) = λ_start + (λ_target - λ_start) * (step / warmup_steps)
   
   Effect: Start with pure FM (λ=0), gradually introduce drift
   Benefit: Stable training, fine convergence
   ```

2. **Constant**
   ```
   λ(step) = λ_target (always)
   
   Effect: Fixed contribution throughout
   Benefit: Simple, reproducible
   ```

3. **Exponential Decay**
   ```
   λ(step) = λ_target * (decay_rate)^step
   
   Effect: Strong guidance early, weaken over time
   Benefit: Exploration → exploitation schedule
   ```

**Usage**:
```python
scheduler = DriftLossScheduler(
    mode='warmup',
    start_weight=0.0,
    target_weight=0.1,
    warmup_steps=1000,
)

for step in training_loop:
    weight = scheduler.get_weight()
    # Use weight in loss computation
    scheduler.step()
```

#### Class: `DriftMemoryBank`

**Purpose**: Circular buffer for expert trajectory storage.

**Design**:
```
Memory: [traj_0, traj_1, ..., traj_N]
         ↑
         ptr (current write position)
         
After capacity reached:
- ptr wraps to 0
- Overwrites oldest trajectories
- Maintains "full" flag
```

**Operations**:
```python
bank = DriftMemoryBank(max_size=5000, trajectory_dim=28)

# Push trajectories
bank.push(expert_batch)         # (B, 28)

# Sample for loss computation
batch = bank.sample(batch_size=32)  # (32, 28)

# Get all for statistics
all_trajs = bank.get_all()      # (N≤5000, 28)
```

**Memory Efficiency**:
- Single allocation: O(max_size × trajectory_dim)
- ~5KB per trajectory (28D floats)
- ~140 MB for 5000 trajectories

#### Class: `DriftTrainingWrapper`

**Purpose**: End-to-end training integration.

**Workflow**:
```python
wrapper = DriftTrainingWrapper(
    drift_loss_fn=drift_loss,
    memory_bank=bank,
    drift_scheduler=scheduler,
)

# In training loop:
for batch in dataloader:
    # 1. Update memory with expert trajectories
    wrapper.update_memory_bank_from_batch(batch['trajectories'])
    
    # 2. Forward pass (sample from diffusion model)
    sampled = model(x0, cond, t)
    
    # 3. Compute FM loss
    fm_loss = compute_fm_loss(sampled, batch['target'])
    
    # 4. Compute combined loss
    total_loss, loss_dict = wrapper.compute_training_loss(
        sampled, fm_loss
    )
    
    # 5. Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 6. Update scheduler
    wrapper.step()
```

**Loss Combination**:
$$L_{total} = L_{FM} + \lambda \cdot L_{drift}$$

```python
def compute_combined_loss(fm_loss, drift_loss, drift_weight):
    total = fm_loss + drift_weight * drift_loss
    return total, {
        'loss_fm': fm_loss.item(),
        'loss_drift': drift_loss.item(),
        'loss_total': total.item(),
        'drift_weight': drift_weight,
    }
```

---

### 5. Metrics & Logging (`utils/drift_metrics.py`)

#### Class: `DriftMetricsTracker`

**Purpose**: Rolling averages of training metrics.

```python
tracker = DriftMetricsTracker(window_size=100)

# Each step
tracker.update(
    loss_fm=0.45,
    loss_drift=0.02,
    loss_total=0.47,
)

# Get rolling mean (last 100 values)
mean_fm = tracker.get_mean('loss_fm')

# Get all means at once
all_means = tracker.get_all_means()
```

#### Key Metrics Functions

**`compute_trajectory_smoothness(trajectory)`**
```
Smoothness = mean(||a(t)||)  where a = d²x/dt²

Lower = smoother (for control, prefer smooth)
```

**`compute_constraint_satisfaction(trajectory, constraint_fn)`**
```
Return:
- violation_rate: fraction of timesteps violating constraint
- mean_violation: average magnitude of violation
- max_violation: worst case violation
```

**`compute_trajectory_fidelity(sampled, reference_trajectories)`**
```
Measure how close sampled trajectory is to expert distribution:
- min_distance: closest expert
- mean_distance: average distance to all experts
- coverage: fraction of references within 1 std
```

**`compute_ode_efficiency(steps_taken, max_steps)`**
```
For adaptive solvers:
efficiency = steps_taken / max_steps
wasted_budget = 1 - efficiency

Useful for measuring ODE solver performance
```

---

## Integration Points

### With FMv3ODE

| Component | FMv3ODE | FM-D | Integration |
|-----------|---------|------|-------------|
| U-Net | Generic 1D temporal | Same, with drift wrapper | No changes to core |
| ODE solver | Dopri5 + Euler | Same with drift injection | Wrapper pattern |
| Velocity field | Model output | Model + drift gradient | Pluggable |
| Loss function | FM regression | FM + drift (weighted) | Separate computation |
| Config | ODE params | +3 drift params | Additive, no conflicts |

**Backward Compatibility**: Set `drift_weight=0` to disable drift and revert to pure FM.

### With DPCC Projections

```python
# Existing projection loop
for step in num_denoising_steps:
    x = denoise_step(x, model, t)
    x = apply_constraints(x)  # ← Compatible!

# FM-D sampling loop
for step in num_ode_steps:
    x = ode_step(x, drift_field, dt)  # drift_field = v + λ∇L
    x = apply_constraints(x)  # ← Same projection operator!
```

**Key**: Projection operates on state only, doesn't care about velocity field structure.

---

## Data Flow: Training

```
Expert batch
    ↓ [update_memory_bank]
    ↓ [store in circular buffer]
    ↓
Memory bank (growing)
    ↓
    ├→ [sample] → Reference trajectories for loss
    │
    └→ [encode] → Distribution statistics

Sampled trajectory (from diffusion)
    ↓ [compute_kl_divergence]
    ↓ [loss = -log P(x|expert)]
    ↓
Drift loss: L_drift

Velocity field prediction
    ↓ [MSE or L1]
    ↓
FM loss: L_FM

Combined loss
    ↓ [L_total = L_FM + λ·L_drift]
    ↓
Backward pass
    ↓
Update weights
    ↓
scheduler.step() → λ increases (warmup)
```

---

## Data Flow: Inference

```
Initial state: x₀ ~ N(0,I)

ODE Integration (t: 0→1)
    ↓
Step 0: t=0, x=x₀
    ├→ v(x,t) from FM model
    ├→ ∇L_drift(x) from drift loss encoder
    ├→ v_aug = v + λ·∇L_drift
    └→ x₁ = x₀ + dt·v_aug
    
Step 1: t=dt, x=x₁
    ├→ v(x,t) from FM model
    ├→ ∇L_drift(x) from drift loss encoder
    ├→ v_aug = v + λ·∇L_drift
    └→ x₂ = x₁ + dt·v_aug
    
... (repeat num_steps times)

Step N: t=1, x=x_N
    ↓
Final trajectory: x_N (or trajectory of all steps)
```

---

## Key Design Decisions

### 1. **Circular Memory Bank**
- Why: Efficient storage without unbounded growth
- How: Pointer wraps after max_size
- Trade-off: Forgets old samples (OK, want recent expert distribution)

### 2. **Warmup Schedule**
- Why: Start with pure FM (stable), gradually add drift
- How: λ linearly increases from 0 to target
- Trade-off: Slower initial convergence, but better long-term stability

### 3. **Gradient Clipping in ODE**
- Why: Drift gradient can be large, causing numerical instability
- How: `clipped = grad * min(||grad||, threshold) / ||grad||`
- Trade-off: Limits drift influence, but ensures convergence

### 4. **Three Loss Variants**
- KL: Simple, fast, differentiable
- MMD: Full distribution matching, non-parametric
- Adversarial: Flexible, but less stable
- Choice: Default to KL (good balance of simplicity and power)

### 5. **Separate Drift Encoder**
- Why: Avoid backprop through base U-Net
- How: MLP encoder just for drift conditioning (frozen during data loading)
- Trade-off: Extra parameter, but cleaner loss computation

---

## Memory & Computational Complexity

### Memory Usage

**Model Parameters**:
- U-Net: ~10M (from FMv3ODE)
- Drift loss encoder: ~200K (3 linear layers)
- Drift conditioner: ~50K
- Discriminator (adversarial only): ~100K
- **Total**: ~10.4M (~40 MB on GPU)

**Memory Bank**:
- 5000 trajectories × 28 dims × 4 bytes = **560 KB**
- Negligible compared to model

### Computational Cost

**Training (per batch)**:
1. Forward pass: 1 U-Net pass (FMv3ODE cost)
2. Drift loss: 1 encoder pass, 1 backward on encoder (~2% overhead)
3. Combined loss: Simple weighted sum
- **Total**: ~105% of FMv3ODE cost

**Inference (per trajectory)**:
1. ODE steps: Same count as FMv3ODE
2. Drift computation: ~1 forward + 1 backward per step (optional)
   - If `drift_weight=0`: Same as FMv3ODE
   - If `drift_weight>0`: ~2x slower per step (but fewer steps needed?)
- **Trade-off**: Potentially net gains from better convergence

---

## Extension Points

### Adding Custom Loss Functions

```python
class CustomDriftLoss(DriftLoss):
    def compute_custom_loss(self, trajectory):
        # Your loss computation
        return loss
    
    def forward(self, trajectory):
        # Call custom loss
        custom_loss = self.compute_custom_loss(trajectory)
        return {'loss': custom_loss}
```

### Custom ODE Solvers

```python
class CustomSolver(DriftODESolver):
    def _solve_custom(self, velocity_fn, x0, ...):
        # Your ODE algorithm
        return solution
```

### Adding Task-Specific Metrics

```python
def compute_task_metric(trajectory, task):
    # Compute metric relevant to your task
    return metric

metrics.update(task_metric=compute_task_metric(...))
```

---

## Debugging & Troubleshooting

### NaN in Training

**Cause**: Exploding drift gradient  
**Fix**: 
1. Reduce `drift_weight` (e.g., 0.05 → 0.01)
2. Increase `drift_clip` threshold
3. Use warmup schedule (don't start with λ=0.1)

### Low Drift Loss

**Cause**: Memory bank has few samples, or encoder not learning  
**Fix**:
1. Increase memory bank size
2. Ensure `update_memory_bank_from_batch()` is called
3. Check encoder hyperparameters (hidden_dim, num_layers)

### Trajectories Not Improving

**Cause**: λ too small, or drift loss has wrong sign  
**Fix**:
1. Check drift loss values are decreasing
2. Verify memory bank is being updated
3. Increase `drift_weight` gradually

---

## Summary

FM-D combines:
- **Deterministic ODE sampling** (from FM) with structured trajectory generation
- **Distribution matching** (from drifting) for expert-aligned behavior
- **Flexible training** via warmup scheduling and memory banks
- **Production-ready code** with tests and documentation

The result is a **hybrid generative model** that learns both physics (via FM velocity field) and statistics (via drift loss gradient), enabling better trajectory quality and satisfaction of learned constraints.

---

**Author**: FM-PCC Development  
**Last Updated**: 2026-05-12  
**Version**: 1.0 (Complete)
