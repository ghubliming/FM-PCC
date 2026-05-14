# How to Run iMeanFlow (Gen3v4)

Complete guide to setup, train, and run inference with Improved Mean Flows.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Project Structure](#project-structure)
3. [Running Training](#running-training)
4. [Running Inference](#running-inference)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
- NumPy, SciPy
- Optional: torchdiffeq (for adaptive ODE solving)

### Step 1: Install Dependencies

```bash
# Navigate to project
cd /workspaces/FM-PCC

# Install PyTorch (example: CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy pyyaml

# Optional: Install torchdiffeq for adaptive ODE solving
pip install torchdiffeq
```

### Step 2: Verify Installation

```bash
# Test core imports
python3 << 'EOF'
import torch
from flow_matcher_v3_imeanflow.models.imf_velocity import DualVelocityField
from flow_matcher_v3_imeanflow.sampling.imf_ode_solvers import ImfODESolver

print("✓ All core modules imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```

---

## Project Structure

```
flow_matcher_v3_imeanflow/
├── models/
│   ├── imf_velocity.py           # Dual-velocity field networks
│   ├── jvp_guidance.py           # Constraint guidance (JVP)
│   ├── imf_dit_trajectory.py     # Optional Transformer backbone
│   └── (FMv3ODE base models)
│
├── sampling/
│   ├── imf_ode_solvers.py        # ODE integration (Euler/RK4/dopri5)
│   ├── imf_trajectory_sampler.py # High-level sampling API
│   └── (FMv3ODE base samplers)
│
├── utils/
│   ├── imf_training.py           # Loss, scheduler, training wrapper
│   ├── imf_metrics.py            # Metrics tracking & analysis
│   └── (FMv3ODE base utilities)
│
├── configs/
│   ├── fm_imeanflow_base.yaml    # Base configuration
│   ├── fm_imeanflow_d3il.yaml    # D3IL robot arm config
│   └── fm_imeanflow_avoiding.yaml # Obstacle avoidance config
│
├── examples/
│   ├── example_imf_training.py   # Training loop demo
│   └── example_imf_inference.py  # Inference demonstrations
│
└── tests/
    └── test_imf_core.py          # 65+ unit tests
```

---

## Running Training

### Quick Start: Synthetic Data Training

The fastest way to get started - generates synthetic trajectories and trains model:

```bash
cd /workspaces/FM-PCC

# Run training example (no external data required)
python3 flow_matcher_v3_imeanflow/examples/example_imf_training.py
```

**What happens:**
- Generates 500 synthetic smooth trajectories
- Creates dual-velocity model (TimeConditionedDualVelocity)
- Trains for 20 epochs with u_first curriculum schedule
- Prints training progress every batch
- Shows validation loss and metric summaries

**Expected output:**
```
================================================================================
iMeanFlow Training Example
================================================================================
Device: cuda
State dimension: 28
...
Epoch 1/20
  Batch   0: loss=0.8234, L_u=0.4120, L_v=0.4114, w_u=0.80, w_v=0.00, u_contrib=0.52
  Batch   5: loss=0.7891, L_u=0.3956, L_v=0.3935, w_u=0.80, w_v=0.00, u_contrib=0.51
  ...
  Train loss: 0.7234, Val loss: 0.7198
```

### Custom Training Script

To integrate with your own data:

```python
import torch
from flow_matcher_v3_imeanflow.models.imf_velocity import TimeConditionedDualVelocity
from flow_matcher_v3_imeanflow.utils.imf_training import (
    ImfTrainingWrapper,
    DualVelocityScheduler,
    compute_trajectory_targets,
)

# 1. Load your data
trajectories = torch.randn(100, 20, 28)  # (batch, time, state_dim)

# 2. Create model
model = TimeConditionedDualVelocity(
    state_dim=28,
    hidden_dim=256,
    time_dim=128,
    use_jvp=False,
).cuda()

# 3. Create training setup
scheduler = DualVelocityScheduler(
    mode='u_first',
    total_steps=1000,
)
trainer = ImfTrainingWrapper(scheduler=scheduler)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(10):
    for traj_batch in [trajectories]:
        # Compute targets
        u_target, v_target = compute_trajectory_targets(traj_batch)
        
        # Time samples
        t = torch.linspace(0, 1, 20).unsqueeze(0).expand(100, 20).cuda()
        
        # Forward
        u_pred, v_pred = model(traj_batch.cuda(), t)
        
        # Loss
        loss, loss_dict = trainer.compute_training_loss(
            u_pred, u_target, v_pred, v_target
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trainer.step()
        
        print(f"Loss: {loss.item():.4f}")
```

### Configuration-Based Training

Create a custom YAML config:

```yaml
# my_config.yaml
architecture:
  state_dim: 28
  dual_velocity:
    u_net_hidden_dim: 256
    v_net_hidden_dim: 256
    use_jvp_guidance: false

training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 1.0e-3
  loss_schedule:
    mode: "u_first"
    weight_u_start: 0.8
    weight_v_start: 0.0
    weight_u_end: 0.5
    weight_v_end: 0.5
```

Then load and use:

```python
import yaml
config = yaml.safe_load(open('my_config.yaml'))
# Initialize model from config['architecture']
# Train with config['training']
```

---

## Running Inference

### Quick Start: Five Inference Demos

```bash
cd /workspaces/FM-PCC

# Run all inference demonstrations
python3 flow_matcher_v3_imeanflow/examples/example_imf_inference.py
```

**Includes 5 demos:**

1. **Basic Sampling** (Single-step vs Dual-step)
   - NFE=1: Fast inference (u + v combined)
   - NFE=2: Higher quality (u phase → v phase)
   - Compares output differences

2. **Multi-Phase Sampling**
   - 4-phase alternating u/v integration
   - Analyzes contribution of each phase
   - Shows u vs v decomposition

3. **Goal-Guided Sampling**
   - Steers trajectories toward goal states
   - Visualizes goal alignment improvement
   - Tests guidance weight effects

4. **Obstacle Avoidance**
   - Samples with repulsive fields around obstacles
   - Checks for collision violations
   - Measures minimum distance to obstacles

5. **Velocity Decomposition Analysis**
   - Shows average u and v magnitudes
   - Computes u/v contribution percentages
   - Measures u-v alignment (cosine similarity)

### Manual Inference

```python
import torch
from flow_matcher_v3_imeanflow.models.imf_velocity import DualVelocityField
from flow_matcher_v3_imeanflow.sampling.imf_trajectory_sampler import ImfTrajectorySampler

# 1. Create or load model
model = DualVelocityField(state_dim=28, hidden_dim=256)
# model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# 2. Create sampler
sampler = ImfTrajectorySampler(
    velocity_model=model,
    num_steps=10,
    solver_type='dopri5',  # or 'rk4', 'euler'
    state_dim=28,
)

# 3. Sample trajectories
z_init = torch.randn(4, 28)

# Single-step (fast, NFE=1)
z_fast = sampler.sample_single_step(z_init)
print(f"Single-step output: {z_fast.shape}")

# Dual-step (quality, NFE=2)
z_u, z_v, z_combined = sampler.sample_dual_step(z_init)
print(f"Dual-step: u={z_u.shape}, v={z_v.shape}")

# Multi-step (analysis)
results = sampler.sample_multi_step(z_init, num_phases=4)
for phase, z_phase in results.items():
    print(f"{phase}: {z_phase.shape}")
```

### Constraint-Aware Inference

```python
from flow_matcher_v3_imeanflow.sampling.imf_trajectory_sampler import ConditionalImfSampler

# Create conditional sampler
cond_sampler = ConditionalImfSampler(
    velocity_model=model,
    num_steps=10,
    goal_weight=0.1,
)

# Goal-guided sampling
z_goal = torch.randn(4, 28)
z_guided = cond_sampler.sample_toward_goal(z_init, z_goal)

# Obstacle avoidance
obstacle_centers = torch.randn(3, 28)
obstacle_radii = torch.full((3,), 0.5)
z_safe = cond_sampler.sample_avoiding_obstacles(
    z_init,
    obstacle_centers,
    obstacle_radii,
)
```

---

## Configuration

### Base Configuration (`fm_imeanflow_base.yaml`)

**Key parameters:**

```yaml
architecture:
  state_dim: 28
  dual_velocity:
    use_dual_velocity: true
    u_net_hidden_dim: 256
    v_net_hidden_dim: 256
    use_jvp_guidance: false

training:
  batch_size: 32
  learning_rate: 1.0e-3
  loss_schedule:
    mode: "u_first"  # Options: balanced, u_first, curriculum
    weight_u_start: 0.8
    weight_v_start: 0.0
    weight_u_end: 0.5
    weight_v_end: 0.5

ode_solver:
  solver_type: "dopri5"  # Options: euler, rk4, dopri5
  num_steps: 10

sampling:
  nfe: 2  # 1 (fast) or 2 (quality)
  nfe_split: 0.5
```

### D3IL Configuration (`fm_imeanflow_d3il.yaml`)

Robot arm specific:
- Enables JVP guidance (collision, smoothness, joint limits)
- Conservative training (lower learning rate)
- Longer trajectories (25 steps)
- 150 epochs (vs 100 base)

Usage:
```python
config = yaml.safe_load(open('flow_matcher_v3_imeanflow/configs/fm_imeanflow_d3il.yaml'))
# Initialize model with D3IL-specific hyperparameters
```

### Obstacle Avoidance Configuration (`fm_imeanflow_avoiding.yaml`)

Navigation specific:
- Aggressive JVP weight (0.4 on collision)
- Curriculum learning (gradually increase v)
- Always NFE=2 sampling (quality)

---

## Testing

### Run All Unit Tests

```bash
cd /workspaces/FM-PCC

# Install pytest if needed
pip install pytest

# Run full test suite (65+ tests)
python3 -m pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v
```

### Run Specific Test Classes

```bash
# Test velocity models
pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py::TestDualVelocityField -v

# Test ODE solvers
pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py::TestImfODESolver -v

# Test training
pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py::TestImfTrainingWrapper -v

# Test sampling
pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py::TestImfTrajectorySampler -v
```

### Test Coverage Areas

| Category | Tests | Modules |
|----------|-------|---------|
| Velocity Models | 6 | imf_velocity.py |
| JVP Guidance | 5 | jvp_guidance.py |
| ODE Solvers | 5 | imf_ode_solvers.py |
| Training | 8 | imf_training.py |
| Metrics | 6 | imf_metrics.py |
| DiT | 3 | imf_dit_trajectory.py |
| Sampling | 7 | imf_trajectory_sampler.py |
| **Total** | **40+** | |

---

## Troubleshooting

### Issue: `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size or model dimension
model = TimeConditionedDualVelocity(
    state_dim=28,
    hidden_dim=128,  # Reduced from 256
    time_dim=64,     # Reduced from 128
)

# Or disable CUDA
device = 'cpu'
```

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: `ModuleNotFoundError: No module named 'torchdiffeq'`

**Solution (optional, not required):**
```bash
pip install torchdiffeq
# If not available, ImfODESolver will fall back to manual Euler/RK4
```

### Issue: Velocities are NaN or Inf

**Symptoms:** Loss becomes NaN during training

**Solutions:**
1. Reduce learning rate: `lr=5e-4` (from 1e-3)
2. Enable gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Ensure targets are normalized: trajectories should have mean≈0, std≈1
4. Check data scaling: large inputs → exploding gradients

### Issue: Model diverges during training

**Solutions:**
1. Use warmup: Set `warmup_steps=1000`
2. Use u_first schedule: Enables curriculum learning
3. Reduce JVP weight: `weight_jvp=0.0` initially, then increase
4. Verify data distribution: Check trajectory ranges

### Issue: Tests fail with import errors

**Solution:**
```bash
# Ensure project is on path
export PYTHONPATH="/workspaces/FM-PCC:$PYTHONPATH"

# Then run tests
python3 -m pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v
```

---

## Performance Tips

### Training Speed

| Setting | Speed | Quality |
|---------|-------|---------|
| solver_type='euler', num_steps=5 | ⚡⚡⚡ | ⭐⭐ |
| solver_type='rk4', num_steps=10 | ⚡⚡ | ⭐⭐⭐ |
| solver_type='dopri5', num_steps=10 | ⚡ | ⭐⭐⭐⭐ |

### Inference Speed

| NFE | Speed | Quality | Use Case |
|-----|-------|---------|----------|
| 1 | ⚡⚡⚡ | ⭐⭐ | Real-time deployment |
| 2 | ⚡⚡ | ⭐⭐⭐⭐ | Development/refinement |

### Memory Usage

| Factor | Impact |
|--------|--------|
| Model hidden_dim | ↑ memory, ↑ capacity |
| Batch size | ↑ memory, ↑↓ gradient quality |
| Sequence length | ↑ memory, ↑ computation |
| ODE steps | ↑ computation, ↓ speed |

---

## Next Steps

1. **Verify Setup**: Run example_imf_training.py
2. **Run Tests**: `pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v`
3. **Explore Inference**: Run example_imf_inference.py
4. **Prepare Data**: Convert your trajectory data to (B, T, 28) tensors
5. **Train Model**: Use custom training script with your data
6. **Deploy**: Use ImfTrajectorySampler for inference

---

## Quick Reference Commands

```bash
# Setup
cd /workspaces/FM-PCC
pip install torch numpy scipy pyyaml torchdiffeq

# Training
python3 flow_matcher_v3_imeanflow/examples/example_imf_training.py

# Inference
python3 flow_matcher_v3_imeanflow/examples/example_imf_inference.py

# Testing
python3 -m pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v

# Import check
python3 -c "from flow_matcher_v3_imeanflow.models.imf_velocity import DualVelocityField; print('✓ OK')"
```

---

## Additional Resources

- [Phase 1 Completion Report](./Gen3v4_iMeanFlow_Phase1_Completion.md)
- Configuration Files: `flow_matcher_v3_imeanflow/configs/`
- Example Scripts: `flow_matcher_v3_imeanflow/examples/`
- Unit Tests: `flow_matcher_v3_imeanflow/tests/`

---

**Last Updated**: May 2026  
**Status**: Phase 1 Complete ✅  
**Ready for**: Phase 2 (Training Integration)
