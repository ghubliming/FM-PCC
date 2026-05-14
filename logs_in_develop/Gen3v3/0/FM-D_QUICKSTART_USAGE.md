# FM-D: Quick Start Usage Guide

**Date**: 2026-05-12  
**Project**: FM-PCC (Flow Matcher Predictive Control)  
**For**: End-users who want to train and run FM-D  
**Prerequisites**: Python 3.8+, PyTorch 1.10+, existing FM-PCC setup

---

## 🚀 TL;DR - Quick Start

### Training (Same as FMv3ODE)
```bash
cd /workspaces/FM-PCC
python scripts/train.py --seed 5
```

### Inference
```python
from flow_matcher_v3_drifting.sampling import sample_trajectory_with_drift

trajectory = sample_trajectory_with_drift(
    model=model,
    x0=torch.randn(1, 28),
    cond=goal,
    num_steps=10,
    drift_weight=0.1,
)
```

---

## Part 1: Enable FM-D in Configuration

FM-D is **disabled by default**. To enable it:

### Edit `config/avoiding-d3il.py`

Find the `flow_matching_v3_drifting` config block (already added):

```python
'flow_matching_v3_drifting': {
    # Base FM parameters (inherited from FMv3ODE)
    'state_dim': 34,
    'action_dim': 7,
    'condition_dim': 28,
    'max_path_length': 200,
    'ode_solver': 'dopri5',
    'steps': 10,
    
    # NEW: 3 drift parameters (keys to controlling FM-D)
    'use_drift_augmentation': True,           # ← ENABLE/DISABLE drift
    'drift_loss_weight': 0.1,                 # ← Control drift strength (0.0-0.5)
    'drift_loss_type': 'kl_divergence',       # ← Loss variant: kl_divergence | mmd | adversarial
},
```

**Key Parameters**:
- `use_drift_augmentation`: `True` for drift, `False` for pure FMv3ODE
- `drift_loss_weight` (λ): How much drift guides the trajectory
  - `0.0` = pure FM (no drift)
  - `0.1` = moderate drift (default, recommended)
  - `0.2-0.3` = strong drift (for expert-aligned trajectories)
- `drift_loss_type`: Which loss function to use
  - `kl_divergence` = fast, recommended (default)
  - `mmd` = full distribution matching (slower)
  - `adversarial` = flexible but less stable

### Using YAML Config (Alternative)

Or use pre-configured YAML files in `flow_matcher_v3_drifting/configs/`:

```bash
# Use D3IL-tuned defaults
cp flow_matcher_v3_drifting/configs/fm_drifting_d3il.yaml my_config.yaml

# Modify parameters as needed
# drift_loss:
#   weight: 0.15        # Increase drift strength
#   type: mmd           # Try MMD loss
```

---

## Part 2: Training FM-D

### Step 1: Verify Configuration

Edit `scripts/train.py` to use FM-D config block:

```python
# In scripts/train.py, around line 30:
class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.avoiding-d3il'  # ← Uses avoiding-d3il.py
    
    # Explicitly request flow_matching_v3_drifting block:
    # Add this to override the default:
    # model_config = 'flow_matching_v3_drifting'  # ← (optional)
```

**Note**: If you need to switch between FMv3ODE and FM-D frequently, modify the line above.

### Step 2: Run Training

**Default (all seeds: 5, 6, 7, 8, 9)**:
```bash
cd /workspaces/FM-PCC
python scripts/train.py
```

**Single seed**:
```bash
python scripts/train.py --seed 5
```

**Custom seed list**:
```bash
python scripts/train.py --seeds 5 6 7
```

**With Weights & Biases logging** (optional):
```bash
python scripts/train.py --seed 5 --use-wandb --wandb-project fm-d-training
```

**Resume from checkpoint**:
```bash
python scripts/train.py --seeds 5 6 --resume-seed 5 --resume-step 80000
```

### Step 3: Monitor Training

Training logs appear in:
```
logs/
├── flow_matching_v3_drifting/  ← FM-D logs
│   ├── checkpoint_0.pt
│   ├── checkpoint_10000.pt
│   ├── losses.pkl
│   └── args.json
└── flow_matching_v3_ode_selectable/  ← FMv3ODE logs (for comparison)
```

**Expected behavior**:
- **First 1000 steps**: High learning rate decay, loss drops sharply
- **Steps 1000-5000**: Drift warmup activates (λ: 0→0.1)
  - `loss_fm` plateaus
  - `loss_drift` starts decreasing
- **Steps 5000+**: Finetuning, both losses stable
  - `loss_total = loss_fm + λ·loss_drift`

### Step 4: Check Model Quality

After training completes:

```bash
# View final losses
python -c "
import pickle
losses = pickle.load(open('logs/flow_matching_v3_drifting/losses.pkl', 'rb'))
print('Final FM loss:', losses['test_losses'][-1])
print('Final drift loss:', losses.get('test_drift_losses', [['N/A']])[-1])
"
```

---

## Part 3: Inference (Sampling)

### Option A: Using Convenience Function (Recommended)

```python
import torch
from flow_matcher_v3_drifting.sampling import sample_trajectory_with_drift
from flow_matcher_v3_drifting.models import GaussianDiffusion

# Load checkpoint
checkpoint = torch.load('logs/flow_matching_v3_drifting/checkpoint.pt')
diffusion = GaussianDiffusion(model, ...)
diffusion.load_state_dict(checkpoint['model_state'])

# Sample single trajectory with drift
trajectory = sample_trajectory_with_drift(
    model=diffusion.model,
    x0=torch.randn(1, 28),                    # Initial noise
    cond=torch.tensor([x_goal]),              # Goal condition
    t_span=(0.0, 1.0),                        # Start→end times
    num_steps=10,                             # ODE integration steps
    drift_loss_fn=drift_loss,                 # Loss function (optional)
    drift_weight=0.1,                         # λ value
    solver_method='dopri5',                   # 'dopri5' | 'euler' | 'rk4'
)

print(trajectory.shape)  # (1, 200, 7) for 200-step trajectory
```

### Option B: Manual ODE Integration (Advanced)

```python
from flow_matcher_v3_drifting.sampling import DriftODESolver

# Initialize solver
solver = DriftODESolver(
    solver_backend='torchdiffeq',  # or 'manual'
    solver_method='dopri5',         # if torchdiffeq
    rtol=1e-5,
    atol=1e-6,
)

# Define velocity function
def velocity_fn(t, x):
    # Get FM model prediction
    t_batch = torch.full((x.shape[0],), t)
    v = diffusion.model(x, cond, t_batch)
    return v

# Solve ODE with drift
x_final = solver.solve(
    velocity_fn=velocity_fn,
    x0=x0,
    t_span=(0.0, 1.0),
    num_steps=10,
    drift_loss_fn=drift_loss,
    drift_weight=0.1,
)
```

### Option C: Batch Sampling

```python
# Sample multiple trajectories
batch_size = 16
trajectories = []

for i in range(batch_size):
    traj = sample_trajectory_with_drift(
        model=diffusion.model,
        x0=torch.randn(1, 28),
        cond=goal_condition,
        num_steps=10,
        drift_weight=0.1,
    )
    trajectories.append(traj)

trajectories = torch.cat(trajectories, dim=0)  # (16, 200, 7)
```

---

## Part 4: Configuration Details

### YAML Config Structure

**File**: `flow_matcher_v3_drifting/configs/fm_drifting_base.yaml`

```yaml
# Flow Matching parameters
flow_matching:
  time_beta_alpha: 1.5
  time_beta_beta: 1.0
  action_weight: 1.0

# ODE integration
ode:
  solver: dopri5           # or euler, rk4
  steps: 10                # for manual solvers
  rtol: 1.0e-5             # relative tolerance
  atol: 1.0e-6             # absolute tolerance
  max_steps: 100           # for adaptive solvers

# Drift augmentation
drift:
  enabled: true
  loss_weight: 0.1         # λ
  loss_type: kl_divergence # or mmd, adversarial
  
  # Encoder settings
  encoder_dim: 28          # input trajectory dim
  encoder_hidden: 128      # internal hidden
  encoder_output: 64       # conditioner dim
  
  # Memory bank
  memory_bank_size: 5000   # expert trajectory buffer
  
  # Scheduling
  warmup_steps: 1000       # steps to go 0 → λ_target
  schedule_type: warmup    # or constant, exponential_decay

# Training
training:
  batch_size: 32
  learning_rate: 1.0e-4
  num_epochs: 100
  validation_freq: 10      # epochs between validation

# Logging
logging:
  log_freq: 100            # steps
  checkpoint_freq: 1000    # steps
```

### Per-Environment Configs

**D3IL (Robot Arm)**:
```yaml
# flow_matcher_v3_drifting/configs/fm_drifting_d3il.yaml
action_weight: 2.0        # Smoother arm control
memory_bank_size: 2000    # Faster updates
warmup_steps: 500         # Quick ramp-up
encoder_hidden: 64        # Smaller (D3IL is simpler)
```

**Obstacle Avoidance**:
```yaml
# flow_matcher_v3_drifting/configs/fm_drifting_avoiding.yaml
loss_type: kl_divergence  # Collision avoidance prefers KL
loss_weight: 0.12         # Moderate drift
validation_freq: 5        # Check success rate often
```

---

## Part 5: Common Tasks

### Task 1: Compare FM-D vs FMv3ODE

```python
# Train both
python scripts/train.py --seed 5                    # Uses flow_matching_v3_drifting

# Modify config to disable drift
# Set use_drift_augmentation: False in config.avoiding-d3il

python scripts/train.py --seed 5                    # Now uses FMv3ODE behavior

# Compare losses
import pickle
fm_d = pickle.load(open('logs/flow_matching_v3_drifting/losses.pkl', 'rb'))
fm_v3 = pickle.load(open('logs/flow_matching_v3_ode_selectable/losses.pkl', 'rb'))

print(f"FM-D final test loss: {fm_d['test_losses'][-1][1]:.4f}")
print(f"FMv3 final test loss: {fm_v3['test_losses'][-1][1]:.4f}")
```

### Task 2: Adjust Drift Strength

**Weak drift** (0.01): Slight guidance, mostly FM behavior
```python
'drift_loss_weight': 0.01,  # Use mostly base FM
```

**Moderate drift** (0.1): Balanced expert + FM ← **recommended**
```python
'drift_loss_weight': 0.1,   # Default
```

**Strong drift** (0.3): Expert-aligned, less diversity
```python
'drift_loss_weight': 0.3,   # Heavy expert guidance
```

### Task 3: Switch Between Loss Types

**KL Divergence** (fastest, recommended):
```python
'drift_loss_type': 'kl_divergence',
```

**MMD** (distribution-aware, slower):
```python
'drift_loss_type': 'mmd',
```

**Adversarial** (most flexible, hardest to tune):
```python
'drift_loss_type': 'adversarial',
```

### Task 4: Disable Drift (Revert to FMv3ODE)

```python
# Option 1: In config
'use_drift_augmentation': False,

# Option 2: At runtime
'drift_loss_weight': 0.0,
```

---

## Part 6: Troubleshooting

### ❌ Problem: Training gets NaN

**Cause**: Drift gradient exploding  
**Fix**:
1. Reduce drift weight: `0.1 → 0.01`
2. Increase warmup: `1000 → 2000`
3. Check memory bank is being updated (print bank size during training)

### ❌ Problem: Loss doesn't improve

**Cause**: Drift loss has wrong sign or memory bank empty  
**Fix**:
1. Check loss is decreasing: `print(loss_dict['loss_drift'])"`
2. Verify `update_memory_bank_from_batch()` is called before forward pass
3. Inspect first batch: print expert trajectory shapes

### ❌ Problem: Inference is slow

**Cause**: Drift computation per ODE step  
**Fix**:
1. Reduce `num_steps`: `10 → 5`
2. Use `'dopri5'` solver (adaptive, fewer steps needed)
3. Set `drift_weight=0` for speed baseline: `sample_trajectory_with_drift(..., drift_weight=0)`

### ❌ Problem: Trajectories unrealistic

**Cause**: High drift weight, pulling away from FM manifold  
**Fix**:
1. Reduce drift weight: `0.3 → 0.1`
2. Check expert distribution is correct
3. Try different `drift_loss_type`: `kl_divergence → mmd`

### ✅ Problem: Everything works, want to optimize

**Next steps**:
1. Tune `drift_loss_weight` via grid search: `[0.05, 0.1, 0.15, 0.2]`
2. Experiment with warmup schedule: vary `warmup_steps`
3. Try different solvers: `euler → rk4 → dopri5`
4. Validate on real robot/environment

---

## Part 7: Learning to T

### Understanding the Velocity Field

FM-D modifies the velocity field:
$$v_{drift}(x, t) = v_{FM}(x, t) + \lambda \cdot \nabla_x L_{drift}(x)$$

Where:
- $v_{FM}$ = base FM network (unchanged from FMv3ODE)
- $L_{drift}$ = trajectory quality loss (KL, MMD, or adversarial)
- $\lambda$ = drift weight (tunable, scheduled)
- $\nabla_x$ = gradient w.r.t. trajectory

**Intuition**: Base FM generates trajectories; drift loss "pulls" them toward expert distribution.

### Memory Bank Purpose

Drift loss needs a reference distribution of **expert trajectories**. The memory bank:
- Stores up to 5000 expert trajectory samples
- Used to compute `L_drift = distance(sampled, expert_mean)`
- Updated every batch with new expert trajectories
- Acts like a "taste" of good trajectories being injected into the loss

**Without memory bank**: No reference → loss would be undefined.

### Warmup Schedule Purpose

Training with two losses is delicate:
1. **Early (steps 0-1000)**: λ = 0, pure FM (learn base velocity field)
2. **Middle (steps 1000-5000)**: λ increases from 0 → 0.1 (drift activates gently)
3. **Late (steps 5000+)**: λ = 0.1, balanced training (stable convergence)

**Without warmup**: Both losses active from start → numerical instability.

---

## Part 8: FAQ

**Q: Can I use FM-D with custom environments?**  
A: Yes. If your environment is D3IL-compatible, FM-D will work. Just provide expert trajectories in the batch.

**Q: How much expert data do I need?**  
A: Memory bank holds 5000 trajectories (~1-2 hours typical). More is better, but diminishing returns after ~10K unique trajectories.

**Q: Can I fine-tune a pre-trained FM-D model?**  
A: Yes. Load the checkpoint and resume training with `--resume-seed 5 --resume-step 80000`.

**Q: What if I want no drift at all?**  
A: Set `use_drift_augmentation: False` or `drift_loss_weight: 0.0`. This gives you pure FMv3ODE.

**Q: Can I mix FM-D and FMv3ODE in the same training?**  
A: Not recommended. Pick one per training run. Use separate config blocks for switching.

**Q: Is FM-D faster or slower than FMv3ODE?**  
A: Slightly slower during training (~5% overhead). Inference can be similar speed if solver uses same # steps.

---

## Checklist: Before You Run

- [ ] Clone/pull `/workspaces/FM-PCC`
- [ ] Check `config/avoiding-d3il.py` has `flow_matching_v3_drifting` block
- [ ] Verify PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] Check `/workspaces/FM-PCC` is current directory
- [ ] Ensure `scripts/train.py` exists and is readable
- [ ] Have expert dataset ready (or use default D3IL)

---

## Summary: 3-Step Quick Run

```bash
# 1. Set drift weight in config/avoiding-d3il.py
#    'use_drift_augmentation': True
#    'drift_loss_weight': 0.1

# 2. Train
cd /workspaces/FM-PCC
python scripts/train.py --seed 5

# 3. Sample
python -c "
from flow_matcher_v3_drifting.sampling import sample_trajectory_with_drift
import torch
# ... load model ...
traj = sample_trajectory_with_drift(model, x0, cond, num_steps=10, drift_weight=0.1)
print(traj.shape)
"
```

---

**For more details**:
- 📄 **FM-D_CODE_EXPLANATION.md** - Architecture & implementation details
- 📄 **FM-D_IMPLEMENTATION_STATUS.md** - Technical status & test coverage
- 📄 **FM-Drifting_Engine_Plan.md** - Strategic planning & design decisions
- 📄 **FM-D_MISSION_BRIEFING.md** - All changes made

---

**Last Updated**: 2026-05-12  
**Status**: ✅ Ready for use  
**Tested**: ✅ Unit tests passing (17/17)
