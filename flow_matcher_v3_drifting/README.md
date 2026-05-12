# FM-D: Flow Matcher-Drifting Engine

Complete implementation of Flow Matcher with Drift Loss guidance for trajectory generation and control.

## Quick Start

### Training

```bash
# Training with FM-D on D3IL environment
python scripts/train.py --config flow_matching_v3_drifting --env d3il

# From config file
python -m flow_matcher_v3_drifting.train --config configs/fm_drifting_d3il.yaml
```

### Inference

```bash
# Sampling with drift guidance
python scripts/sample.py \
    --model_path logs/flow_matching_v3_drifting/checkpoint.pt \
    --num_trajectories 10 \
    --drift_weight 0.1
```

## Architecture

### Core Components

1. **Drift Loss** (`models/drift_loss.py`)
   - KL divergence-based trajectory distribution matching
   - Optional MMD and adversarial variants
   - Circular memory bank for expert trajectory storage

2. **Drift-Augmented U-Net** (`models/drift_unet.py`)
   - Extends base 1D temporal U-Net with drift conditioning
   - Encodes trajectory history and drift metrics
   - Maintains FM-ODE compatibility

3. **ODE Solvers** (`sampling/drift_ode_solvers.py`)
   - Drift-guided ODE integration
   - Supports multiple backends: legacy Euler, RK4, torchdiffeq
   - Modular velocity field augmentation

4. **Training Utilities** (`utils/drift_training.py`)
   - `DriftLossScheduler`: Warmup, constant, exponential decay modes
   - `DriftMemoryBank`: Circular buffer for expert trajectories
   - `DriftTrainingWrapper`: End-to-end training loop integration

5. **Metrics & Logging** (`utils/drift_metrics.py`)
   - Trajectory smoothness
   - Constraint satisfaction rate
   - Fidelity to expert distribution
   - ODE solver efficiency

## Configuration

### Locked Parameters

FM-D adds exactly 3 configuration parameters:

```python
'use_drift_augmentation': True              # Enable drift mode
'drift_loss_weight': 0.1                    # λ in drift field equation
'drift_loss_type': 'kl_divergence'          # "kl_divergence" | "mmd" | "adversarial"
```

### YAML Configs

- `fm_drifting_base.yaml` - Default settings
- `fm_drifting_d3il.yaml` - Robot arm specialization
- `fm_drifting_avoiding.yaml` - Obstacle avoidance specialization

## Training Loop

### Standard FM-D Training

```python
from flow_matcher_v3_drifting.models import GaussianDiffusion, DriftLoss
from flow_matcher_v3_drifting.utils.drift_training import DriftTrainingWrapper

# Initialize models
model = UNet1DTemporalCondModel(...)
diffusion = GaussianDiffusion(model, ...)
drift_loss = DriftLoss(trajectory_dim=28, loss_type='kl_divergence')

# Create training wrapper
trainer = DriftTrainingWrapper(drift_loss_fn=drift_loss)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get expert trajectories for memory bank
        expert_trajs = batch['trajectories']
        trainer.update_memory_bank_from_batch(expert_trajs)
        
        # Sample from diffusion
        sampled_trajs = diffusion.sample(cond=batch['condition'])
        
        # Compute FM loss
        fm_loss = diffusion.loss(sampled_trajs, batch['target'])
        
        # Compute combined FM + drift loss
        total_loss, loss_dict = trainer.compute_training_loss(
            sampled_trajs, fm_loss
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        trainer.step()  # Update scheduler
```

## Inference

### ODE Integration with Drift Guidance

```python
from flow_matcher_v3_drifting.sampling import sample_trajectory_with_drift

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pt')
diffusion.load_state_dict(checkpoint['diffusion'])
drift_loss.load_state_dict(checkpoint['drift_loss'])

# Sample with drift guidance
trajectory = sample_trajectory_with_drift(
    model=diffusion.model,
    x0=torch.randn(1, 28),
    cond=goal_condition,
    t_span=(0.0, 1.0),
    num_steps=10,
    drift_loss_fn=drift_loss,
    drift_weight=0.1,
    solver_method='dopri5',
)
```

## Testing

Run comprehensive test suite:

```bash
# Drift loss tests
python FM_v3_drifting_test/test_drift_loss.py

# ODE solver tests
python FM_v3_drifting_test/test_drift_ode_solvers.py

# Training utilities tests
python FM_v3_drifting_test/test_drift_training.py
```

## Results & Benchmarks

### Expected Performance

- **Training loss**: FM loss decreases monotonically; drift loss provides refinement
- **Inference speed**: Similar to FMv3ODE (same solver backend)
- **Trajectory quality**: Improved fidelity to expert distribution vs. baseline FM
- **Constraint satisfaction**: Same as DPCC/FMv3ODE (projections unchanged)

### Metrics

- **Smoothness**: Mean acceleration magnitude (lower = better)
- **Fidelity**: Distribution KL divergence to expert trajectories
- **Violation rate**: Fraction of states violating constraints
- **ODE efficiency**: Steps taken / max steps (for adaptive solvers)

## File Structure

```
flow_matcher_v3_drifting/
├── models/
│   ├── drift_loss.py          # Drift loss computation
│   ├── drift_unet.py          # Drift-aware U-Net conditioning
│   ├── diffusion.py           # FM-ODE diffusion model
│   └── ...
├── sampling/
│   ├── drift_ode_solvers.py   # ODE integration with drift
│   ├── policies.py            # Policy sampling
│   └── ...
├── utils/
│   ├── drift_metrics.py       # Performance metrics & logging
│   ├── drift_training.py      # Training utilities
│   └── ...
├── configs/
│   ├── fm_drifting_base.yaml
│   ├── fm_drifting_d3il.yaml
│   └── fm_drifting_avoiding.yaml
└── ...

FM_v3_drifting_test/
├── test_drift_loss.py
├── test_drift_ode_solvers.py
└── test_drift_training.py
```

## Integration with Existing Code

FM-D maintains **full backward compatibility** with FMv3ODE:
- Same U-Net architecture (just wrapped)
- Same ODE solver backends
- Same projection operators
- Can disable drift (λ=0) to revert to baseline FM

## References

- **Flow Matching**: Liphardt et al. (2024)
- **Drifting**: arXiv:2602.04770
- **FM-PCC**: Flow Matching Predictive Control (this workspace)

## Support & Troubleshooting

### Common Issues

**Q: `DriftLoss` memory bank is empty during training**
- Memory bank builds over batches. Add `update_memory_bank_from_batch()` before loss computation.

**Q: NaN during ODE integration**
- Clip drift gradient: increase `drift_clip` parameter
- Start with lower `drift_weight` (e.g., 0.05)
- Use warmup schedule in training

**Q: Trajectories not improving with drift**
- Verify memory bank has diverse expertdemonstrations
- Check drift loss values are reasonable (not exploding)
- Increase warmup epochs before activating drift

## Contributing

FM-D is part of FM-PCC. Follow the Gen3v2 development pattern:
- Create copied folders (don't modify originals)
- Use locked naming conventions
- Add tests for new functionality
- Update documentation

---

**Status**: Phase 1-3 Complete | Phase 4 (Evaluation) In Progress  
**Last Updated**: 2026-05-12
