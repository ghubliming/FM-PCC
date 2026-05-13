# iMF-PCC Architecture Overview

**Date**: May 13, 2026  
**Version**: Gen3v4 Fix #3  
**Component**: iMeanFlow (iMF) ML Engine  
**Classification**: Real dual-velocity trajectory model (not theoretical)

---

## Architecture Layers

### Layer 1: Core Model (`imf_trajectory_model.py`)

**Purpose**: Dual-velocity prediction heads

```
Input Space
    ↓
[x: noisy trajectory] + [t: timestep] + [cond: optional]
    ↓
FMv3ODE U-Net Backbone (flow_matcher_u_net_v2)
    ↓ 
    ├─→ u-head: Mean velocity field [batch, seq_len, state_dim]
    └─→ v-head: Instantaneous deviation [batch, seq_len, state_dim]
    ↓
Output: (u, v) tuple
```

**Key Components**:
- **u_net**: `Flow_matcher_U_Net_v2` (reused from FMv3ODE)
  - Input: noisy trajectory, timestep, conditioning
  - Output: predicted mean velocity field
  - Architecture: 8 blocks, 4 attention heads, 256 hidden dim
  
- **v_head**: Lightweight MLP auxiliary head
  - Input: u_net intermediate features
  - Output: instantaneous velocity component
  - Architecture: Linear → ReLU → Linear (simple)

**Design Rationale**:
- u: Stable mean prediction (backbone does heavy lifting)
- v: Fast adaptive refinement (small head)
- Parallel execution: no sequential dependencies

---

### Layer 2: Engine (`imf_engine.py`)

**Purpose**: iMF inference and sampling API

```
Training Mode:
  forward_train(x_noisy, t, cond) → (u, v) predictions
  
Inference Mode:
  sample(batch_size, num_steps, schedule) → sampled trajectory
    ├─ Initialize z_t ~ N(0, I)
    ├─ For each ODE step:
    │   ├─ Get (u, v) predictions
    │   ├─ Compute velocity = u_weight * u + v_weight * v
    │   └─ z_t = z_t - h * velocity  (Euler step)
    └─ Return z_t
```

**Key Methods**:
- **u_fn()**: Return dual velocity (official iMF API)
- **sample()**: Generate trajectories (iMF sampling algorithm)
- **forward_train()**: Get predictions for loss computation

**Design Rationale**:
- Matches official iMF repo's `u_fn` signature (transferability)
- Supports configurable schedule ('balanced' or 'u_first')
- Compatible with FM-PCC Trainer interface

---

### Layer 3: Training Loss (`imf_losses.py`)

**Purpose**: Curriculum-based dual-loss computation

```
Loss Computation:

Phase 1 (epochs 0-30): u-only
  loss = 1.0 * MSE(u_pred, u_target) + 0.0 * MSE(v_pred, v_target)

Phase 2 (epochs 30-60): Curriculum blend
  progress = (epoch - 30) / 30
  u_scale = 1.0 - 0.5 * progress     # 1.0 → 0.5
  v_scale = 0.5 * progress           # 0.0 → 0.5
  loss = u_scale * u_loss + v_scale * v_loss

Phase 3 (epochs 60+): Balanced u+v
  loss = 0.5 * MSE(u_pred, u_target) + 0.5 * MSE(v_pred, v_target)
```

**Key Components**:
- **get_loss_weights()**: Curriculum schedule (from official iMF)
- **compute_losses()**: MSE for each component
- **forward()**: Full loss stack with metrics

**Design Rationale**:
- Official iMF paper's curriculum: u first (stable), then transition, then dual
- Early training stabilizes mean field (harder to learn)
- Late training refines with instantaneous component (residual)
- Smooth transition prevents divergence

---

### Layer 4: Diffusion Wrapper (`imf_diffusion.py`)

**Purpose**: FM-PCC training pipeline integration

```
FM-PCC Trainer Interface:
  
  diffusion.p_losses(x_start, t, cond, epoch) → (loss, metrics_dict)
    ↓
    ├─ 1. Add flow-matching noise
    │      x_noisy = (1 - t) * x_start + t * noise
    │
    ├─ 2. Get dual-velocity predictions
    │      u_pred, v_pred = model.forward_train(x_noisy, t, cond)
    │
    ├─ 3. Compute curriculum-weighted loss
    │      loss, metrics = imf_loss.forward(...)
    │
    └─ 4. Return loss + logging metrics
           return loss, {'u_loss': ..., 'v_loss': ..., 'u_weight': ..., ...}
```

**Key Methods**:
- **p_losses()**: Compute loss (matches Trainer signature)
- **sample()**: Generate trajectories at inference
- **forward_train()**: Predictions for loss

**Design Rationale**:
- Wrapper pattern: iMF model + iMF loss + FM-PCC interface
- No modifications to Trainer code needed
- Drop-in replacement for FMv3ODE's diffusion module

---

## Data Flow: Training Step

```
1. Data Loading (FM-PCC Trainer)
   Dataset → Batch [observations, actions, conditions, mask]
       ↓
2. Trajectory Assembly
   obs + actions → trajectory (dim: state_dim + action_dim)
       ↓
3. Noise Schedule (Flow Matching)
   t ~ U(0, 1)
   x_noisy = (1 - t) * trajectory + t * noise
       ↓
4. iMF Forward Pass
   iMFDiffusion.p_losses(x_noisy, t, cond, epoch)
   ├─ iMFTrajectoryModel(x_noisy, t, cond) → (u_pred, v_pred)
   └─ iMFTrainingLoss.forward(u_pred, v_pred, ..., epoch) → loss
       ↓
5. Curriculum Scheduling (inside iMFTrainingLoss)
   u_scale, v_scale = get_loss_weights(epoch)
       ↓
6. Loss Computation
   total_loss = u_scale * ||u_pred - u_target||² + v_scale * ||v_pred - v_target||²
       ↓
7. Backprop
   optimizer.zero_grad()
   total_loss.backward()
   optimizer.step()
       ↓
8. W&B Logging
   run.log({'train/u_loss': u_loss, 'train/v_loss': v_loss, ...})
```

---

## Data Flow: Inference Step

```
1. Load Checkpoint
   diffusion_exp = utils.load_diffusion(savepath, epoch='best')
   model = diffusion_exp.model (iMeanFlowEngine)
       ↓
2. Sampling Call
   sampled = diffusion.sample(batch_size=64, num_steps=1)
       ↓
3. iMF Sampling Loop
   z_t ~ N(0, I)
   for i in range(num_steps):
       t = t_steps[i]
       r = t_steps[i + 1]
       h = t - r
       
       u, v = model(z_t, t)  → iMFTrajectoryModel forward
       velocity = u_weight * u + v_weight * v
       z_t = z_t - h * velocity
       ↓
4. Output
   return z_t (sampled trajectory)
```

---

## Component Interaction

```
┌─────────────────────────────────────────────────────┐
│                 FM-PCC Trainer                      │
│  (handles data loading, optimization, checkpointing)│
└──────────────────────┬──────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────┐
│              iMFDiffusion (wrapper)                 │
│  • Bridges FM-PCC ↔ iMF                            │
│  • Implements p_losses() signature                  │
│  • Routes to iMFTrainingLoss                       │
└──────────────┬──────────────────────┬──────────────┘
               │                      │
               ↓                      ↓
    ┌─────────────────────┐  ┌──────────────────┐
    │ iMFTrajectoryModel  │  │ iMFTrainingLoss  │
    │ (architecture)      │  │ (curriculum)     │
    │                     │  │                  │
    │ • u_net (U-Net)     │  │ • get_loss_w()   │
    │ • v_head (MLP)      │  │ • compute_loss() │
    │ • forward() → (u,v) │  │ • forward()      │
    └─────────────────────┘  └──────────────────┘
               ↑
               │
    ┌──────────────────────────┐
    │   iMeanFlowEngine        │
    │  (inference interface)   │
    │                          │
    │ • u_fn() → (u, v)        │
    │ • sample() → trajectory  │
    │ • forward_train()        │
    └──────────────────────────┘
```

---

## Design Philosophy

### 1. **Minimal Modification**
- Reuses FMv3ODE's U-Net backbone (proven, stable)
- Adds only v-head auxiliary (lightweight)
- Everything else inherited from FM-PCC

### 2. **Official iMF Fidelity**
- `u_fn()` API: Direct from official iMF repo
- Curriculum schedule: Official iMF paper's method
- Dual-loss weighting: Official iMF training scheme

### 3. **Modular Separation**
- Model (prediction) ← iMFTrajectoryModel
- Loss (training) ← iMFTrainingLoss
- Integration (FM-PCC) ← iMFDiffusion
- Inference (sampling) ← iMeanFlowEngine

Each module has single responsibility; easy to test/debug independently.

### 4. **Configuration-Driven**
- 3 iMF-specific params: u_loss_weight, v_loss_weight, loss_schedule
- 30+ inherited FMv3ODE params: data loading, ODE solver, batch size, etc.
- All via `config/avoiding-d3il.py` block (no code changes needed)

---

## File Structure

```
flow_matcher_v3_imeanflow/
├── models/
│   ├── __init__.py
│   ├── imf_trajectory_model.py     ← Layer 1: Dual-head architecture
│   ├── imf_engine.py               ← Layer 2: iMF sampling API
│   ├── imf_losses.py               ← Layer 3: Curriculum training
│   ├── imf_diffusion.py            ← Layer 4: FM-PCC integration
│   ├── unet1d_temporal_cond.py     ← Reused from FMv3ODE (u-net backbone)
│   ├── diffusion.py                ← Kept for compatibility
│   └── ...
├── __init__.py
└── ...

FM_v3_imeanflow_test/
├── train_flow_matching_v3_imeanflow.py     ← Multi-seed training
├── eval_flow_matching_v3_imeanflow.py      ← Validation evaluation
└── load_results_flow_matching_v3_imeanflow.py  ← Results aggregation

config/
└── avoiding-d3il.py                        ← Updated config block

Slurm_Codes/sbatch/iMF/
├── train_imf.sh                    ← SLURM training job
├── eval_imf.sh                     ← SLURM evaluation job
└── load_results_imf.sh             ← SLURM results display
```

---

## Next Development Stages (Optional)

### Stage 1: Multi-NFE Support
- Variable `num_steps ∈ {1, 2, 4, 8}`
- Quality/speed tradeoff
- Runtime: 1 step (fast) → 8 steps (slow, accurate)

### Stage 2: Constraint Guidance
- Optional jvp_weight for collision avoidance
- Gradient penalty: `∇ collision ≤ threshold`
- Safeguards trajectory generation

### Stage 3: Comparative Analysis
- iMF vs FMv3ODE on same D3IL data
- Measure: accuracy improvement, training speed, sample quality
- Publication: method comparison paper

### Stage 4: Fine-Grained Curriculum
- Learnable schedule (instead of fixed epochs)
- Curriculum duration: calibrate per dataset
- Adaptive: adjust based on validation loss

---

## Summary

iMF-PCC is a **4-layer architecture** that implements Improved Mean Flows (from official repo) while preserving FM-PCC's training infrastructure. Each layer has clear responsibility; integration is clean via `iMFDiffusion` wrapper.

**Core Innovation**: Dual-velocity decomposition with curriculum learning  
**Implementation**: Official iMF patterns + FMv3ODE backbone + FM-PCC interface  
**Status**: Production-ready, ready for multi-seed training on D3IL data
