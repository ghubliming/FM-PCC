# FM-D: Flow Matcher-Drifting Engine Plan

Date: 2026-05-12
Status: Architecture & Implementation Plan
Repo Reference: `/workspaces/FM-PCC` (FM-PCC) + `/workspaces/drifting` (Drifting)

---

## Executive Summary

This document describes the FM-D (Flow Matcher-Drifting) engine—a next-generation generative control architecture that combines:
- **Flow Matcher ODE** methodology from `FMv3ODE` (deterministic velocity field learning)
- **Drifting** generative dynamics from `/workspaces/drifting` (JAX-based one-step generation via drift trajectories)

The result is a hybrid framework capable of learning implicit policy trajectories through drift-augmented flow matching with ODE-based deterministic sampling.

---

## 1) Goals & Vision

### Primary Objectives

1. **Unify FM and Drifting paradigms**: Merge ODE-based deterministic flow matching with drift-based generative dynamics
2. **One-step trajectory generation**: Leverage drifting's efficiency to produce valid trajectories in minimal forward steps
3. **Safety compliance**: Maintain projection-based constraint enforcement from DPCC/FM-PCC lineage
4. **Modular architecture**: Keep clear separation between generative engine, sampling strategy, and control loop

### Success Criteria

- [ ] Module properly inherits from `FMv3ODE` structure
- [ ] Drifting loss (`drift_loss.py`) integrated into training loop
- [ ] ODE solver produces valid control sequences in < N steps
- [ ] Safety projections apply to drift-augmented trajectories
- [ ] Configuration system supports both FM and drifting hyperparameters
- [ ] Inference achieves target FID/trajectory quality metrics

---

## 2) Architecture Overview

### 2.1 Folder Structure (Locked)

```
flow_matcher_v3_drifting/          (NEW - parallel to flow_matcher_v3_ode_selectable)
├── __init__.py
├── setup.py
├── models/
│   ├── __init__.py
│   ├── drift_unet.py               (Drifting-augmented U-Net for velocity field)
│   ├── drift_loss.py               (Gradient computation for drift trajectories)
│   ├── diffusion.py                (adapted from FMv3ODE, drift-aware)
│   ├── helpers.py                  (shared utilities)
│   ├── mlp.py                      (from FMv3ODE baseline MLP)
│   └── unet1d_temporal_cond.py     (temporal conditioning via drifting)
├── sampling/
│   ├── __init__.py
│   ├── policies.py                 (drift-augmented policy sampling)
│   ├── projection.py               (constraint projection w/ drift state)
│   └── drift_ode_solvers.py        (ODE integrators for drift paths)
├── datasets/
│   ├── __init__.py
│   ├── buffer.py                   (trajectory replay buffer + drift trajectories)
│   └── drift_trajectory_augmentation.py (augmentation pipeline)
├── utils/
│   ├── __init__.py
│   ├── config.py                   (drift+FM unified config)
│   ├── training.py                 (epoch loop with drift loss)
│   ├── logger.py                   (metrics: drift divergence, ODE steps)
│   ├── drift_metrics.py            (FID, drift fidelity, trajectory quality)
│   └── serialization.py            (checkpoint save/load)
├── configs/
│   ├── fm_drifting_base.yaml       (default FM-D config)
│   ├── fm_drifting_d3il.yaml       (D3IL environment specialization)
│   └── fm_drifting_avoiding.yaml   (obstacle avoidance specialization)
└── notebooks/
    ├── fm_drifting_demo.ipynb      (inference examples)
    └── drift_loss_visualization.ipynb (drift path analysis)

FM_v3_drifting_test/               (NEW - test suite for FM-D)
├── test_drift_integration.py
├── test_drift_loss_backward.py
├── test_projection_drift_compat.py
├── eval_drift_trajectories.py
└── benchmark_drift_vs_fm_ode.py
```

### 2.2 Parallel Folder Copies (Locked Naming)

Per the `Gen3v2` pattern, **do not modify original folders**:

1. **Source**: `flow_matcher_v3_ode_selectable` → **Copy**: `flow_matcher_v3_drifting`
2. **Source**: `FM_v3_ode_selectable_test` → **Copy**: `FM_v3_drifting_test`
3. **Config injection point**: `config/avoiding-d3il.py` (add 3 new parameters, see §3.3)

---

## 3) Core Technical Concepts

### 3.1 Flow Matching + Drift Integration

**Standard FM ODE evolution:**
$$\frac{dx}{dt} = v_\theta(x_t, t | \text{condition})$$

**FM-D with drift augmentation:**
$$\frac{dx}{dt} = v_\theta(x_t, t | \text{condition}) + \lambda \cdot \nabla_x \mathcal{L}_{\text{drift}}(x_t)$$

where:
- $v_\theta$ = learned velocity field (from FM training, deterministic)
- $\mathcal{L}_{\text{drift}}$ = drift loss measuring trajectory quality (from `/workspaces/drifting`)
- $\lambda$ = drift weighting hyper-parameter (learned or fixed)

**Key distinction from standard drifting**:
- Drifting typically learns $\mathcal{L}_{\text{drift}}$ as a generative objective; FM-D uses it as a *regularization* term during trajectory rollout
- ODE integration remains deterministic (no stochasticity in sampling phase)
- Drift loss provides implicit trajectory shaping without explicit state constraints

### 3.2 Drift Loss Computation

Adapted from `/workspaces/drifting/drift_loss.py`:

```python
# Pseudo-code structure
def compute_drift_loss(trajectory, condition, target_distribution):
    """
    Measure deviation of sampled trajectory from learned distribution.
    
    Inputs:
      trajectory: (T, state_dim) tensor of rollout states
      condition: (cond_dim,) goal/context info
      target_distribution: learned expert distribution
      
    Returns:
      loss: scalar, gradient magnitude guiding ODE integration
    """
    # 1. Encode trajectory via memory bank or learned encoder
    traj_encoding = encoder(trajectory, condition)
    
    # 2. Compare against target distribution learned from expert demos
    kl_div = kl_divergence(traj_encoding, target_distribution)
    
    # 3. Optionally: adversarial refinement
    # discriminator_logit = discriminator(traj_encoding)
    
    return kl_div  # or combined adversarial loss
```

### 3.3 ODE Integration with Drift Guidance

**Drift-augmented sampler pseudocode:**

```python
def sample_with_drift_guidance(x0, t_span, v_theta, drift_loss_fn, lambda_drift):
    """ODE solver with drift guidance."""
    
    def drift_field(x, t):
        velocity = v_theta(x, t)  # from FM network
        drift_grad = grad(drift_loss_fn)(x)  # backprop through loss
        return velocity + lambda_drift * drift_grad
    
    # Use standard ODE solver (RK45, Dopri5, etc.)
    solution = odint(drift_field, x0, t_span, method='rk45')
    return solution
```

---

## 4) Configuration & Hyperparameters

### 4.1 New Config Parameters (config/avoiding-d3il.py)

Add exactly 3 mandatory parameters:

```python
# Drift engine enablement
USE_DRIFT_AUGMENTATION = True                  # bool: enable FM-D mode

# Drift loss weighting
DRIFT_LOSS_WEIGHT = 0.1                        # float: λ in drift field equation

# Drift loss variant
DRIFT_LOSS_TYPE = "kl_divergence"              # str: "kl_divergence" | "adversarial" | "mmd"
```

### 4.2 YAML Config Files

**fm_drifting_base.yaml** (template):
```yaml
# Flow Matcher core
model:
  name: "flow_matcher_v3_drifting"
  input_dim: 28  # example: 7 action dims * 4 steps
  hidden_dim: 256
  num_layers: 4
  
# ODE integration
sampling:
  ode_solver: "dopri5"          # integrator choice
  t_span: [0.0, 1.0]            # reverse time integration
  method: "flow_matching"       # vs "diffusion"
  
# Drift augmentation
drift:
  enabled: true
  loss_weight: 0.1
  loss_type: "kl_divergence"
  memory_bank_size: 5000
  
# Training
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1e-4
  drift_warmup_epochs: 10  # train FM first, then activate drift
  
# Logging
logging:
  log_drift_metrics: true
  save_drift_visualizations: true
```

---

## 5) Implementation Phases

### Phase 1: Foundation (Core Structure)

**Objective**: Establish folder layout and baseline FM-D architecture

**Deliverables**:
- [ ] Copy `flow_matcher_v3_ode_selectable` → `flow_matcher_v3_drifting`
- [ ] Copy `FM_v3_ode_selectable_test` → `FM_v3_drifting_test`
- [ ] Implement `models/drift_unet.py` (U-Net with drift-aware state conditioning)
- [ ] Implement `models/drift_loss.py` (loss computation, backward pass)
- [ ] Update `config/avoiding-d3il.py` with 3 new parameters

**Files to create**:
- `flow_matcher_v3_drifting/models/drift_unet.py`
- `flow_matcher_v3_drifting/models/drift_loss.py`
- `flow_matcher_v3_drifting/sampling/drift_ode_solvers.py`
- config YAML templates

**Testing**:
- Unit test: drift loss backward pass
- Unit test: ODE integrator accepts drift field
- Integration test: FM-D loads in training loop

---

### Phase 2: Training Loop Integration

**Objective**: Full training support with drift loss

**Deliverables**:
- [ ] Update `utils/training.py` to compute drift loss during backward pass
- [ ] Implement drift warmup schedule (train FM first)
- [ ] Extended logger with drift metrics (divergence, trajectory smoothness)
- [ ] Memory bank for storing drift reference trajectories

**New functions**:
- `train_with_drift_loss()` in `utils/training.py`
- `DriftMetricsLogger` class in `utils/logger.py`
- `DriftMemoryBank` in `datasets/buffer.py`

**Testing**:
- [ ] Training convergence test (drift enabled/disabled)
- [ ] Gradient flow verification (drift loss backprop)
- [ ] Memory bank correctness (trajectory storage/retrieval)

---

### Phase 3: Sampling & Projection

**Objective**: Inference with drift guidance + constraint safety

**Deliverables**:
- [ ] Implement `sampling/policies.py` (drift-guided policy sampling)
- [ ] Update `sampling/projection.py` to handle drift state evolution
- [ ] ODE solver integration with projections (alternating: ode_step → project → ode_step)

**Key functions**:
- `sample_policy_with_drift()` in `policies.py`
- `project_drift_state()` in `projection.py` (adapts projection to maintain drift gradients)

**Testing**:
- [ ] Constraint satisfaction with drift (obstacle avoidance)
- [ ] ODE step stability (numerical errors, order of operations)
- [ ] Comparison: FM-ODE vs. FM-D convergence to goal

---

### Phase 4: Evaluation & Benchmarking

**Objective**: Compare FM-D against FMv3ODE and drifting baselines

**Deliverables**:
- [ ] `FM_v3_drifting_test/benchmark_drift_vs_fm_ode.py`
- [ ] Metrics: FID, trajectory smoothness, constraint satisfaction rate
- [ ] Visualization: drift loss landscape, sampled trajectory overlays
- [ ] Notebook: `notebooks/fm_drifting_demo.ipynb` (inference examples)

**Evaluation scenarios**:
1. **Image generation** (drifting-style): FID on standard benchmarks
2. **Trajectory control** (FM-PCC-style): success rate on D3IL tasks
3. **Hybrid**: trajectory quality + constraint satisfaction trade-off

**Benchmarking targets**:
- FM-D inference time vs. FMv3ODE (should be similar or faster due to one-step nature)
- FM-D trajectory fidelity vs. expert demonstrations (measured via drift loss)
- FM-D constraint violation rate vs. DPCC (should match or improve)

---

## 6) Integration Points

### 6.1 With FMv3ODE

| Component | FMv3ODE | FM-D | Integration Strategy |
|-----------|---------|------|----------------------|
| U-Net architecture | Generic 1D temporal | Drift-conditioned | Fork u-net, add drift encoding stream |
| ODE solver | Dopri5 + Euler options | Dopri5 + drift guidance | Wrap solver, inject drift field |
| Sampling loop | Deterministic rollout | ODE + drift projection | Extend `policies.py` |
| Loss function | FM regression loss | FM loss + drift loss (weighted) | Multi-loss trainer |
| Config space | `ode_solver_backend_v3`, `ode_solver_method_v3` | Add `drift_loss_weight`, `drift_loss_type` | Config extension |

### 6.2 With Drifting (/workspaces/drifting)

| Component | Drifting | FM-D Adaptation | Notes |
|-----------|----------|-----------------|-------|
| `drift_loss.py` | JAX generative loss | PyTorch/TF for control | Port gradient computation |
| `memory_bank.py` | Reference distribution | Trajectory buffer | Adapt for sequence data |
| Generator network | MAE/ConvNext | 1D U-Net temporal | Different domain (image → trajectory) |
| Training loop | JAX/Flax | PyTorch/TF (aligned with FM-PCC) | Port loss, keep solver logic |
| Inference | One-step generation | ODE integration steps | Keep deterministic, multi-step for stability |

### 6.3 With DPCC/Projections

**Compatibility**: Full backward compatibility with existing projection code

```python
# Existing DPCC projection loop (unchanged)
for step in range(num_denoising_steps):
    x = denoise_step(x, model, t)
    x = apply_constraints(x)  # projection

# FM-D projection loop (compatible extension)
for step in range(num_ode_steps):
    x = ode_step(x, drift_field, dt)  # drift_field includes FM + drift loss
    x = apply_constraints(x)  # same projection code
```

---

## 7) Dependencies & Requirements

### 7.1 New Packages

| Package | Version | Purpose | Source Repo |
|---------|---------|---------|------------|
| `jax` | ≥0.4.0 | Drift loss computation (gradient tracing) | from `/workspaces/drifting` |
| `flax` | ≥0.7.0 | Optional JAX-based drift encoder | from `/workspaces/drifting` |
| `torch` / `tensorflow` | existing | Primary training framework | FM-PCC baseline |

*Note*: If drift loss is ported to pure PyTorch/TF, JAX is optional.

### 7.2 File Dependencies

**Must-have from FMv3ODE**:
- `flow_matcher_v3_ode_selectable/models/unet1d_temporal_cond.py`
- `flow_matcher_v3_ode_selectable/sampling/projection.py`
- `flow_matcher_v3_ode_selectable/utils/training.py`

**Must-have from drifting**:
- `/workspaces/drifting/drift_loss.py` (port to control domain)
- `/workspaces/drifting/memory_bank.py` (adapt for trajectories)

**Existing FM-PCC infra**:
- `config/avoiding-d3il.py` (parameter injection)
- `d3il/` environment wrappers
- `diffuser/` dataset utilities (optional)

---

## 8) Known Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Drift loss instability during ODE integration | High | Implement loss clipping; start with small λ (0.01); warmup schedule |
| Memory overhead (drift bank + model weights) | Medium | Implement circular buffer; quantize reference trajectories |
| ODE solver stepping + projection conflicts | Medium | Validate constraint satisfaction per step; document calling order |
| JAX ↔ PyTorch compatibility (if mixed) | Medium | Port drift loss to PyTorch only; avoid JAX in training loop |
| Different trajectory distributions (expert vs. drift) | High | Ablation study: compare FM loss alone vs. FM + drift loss |

---

## 9) Success Metrics & Validation Checklist

### 9.1 Build Validation

- [ ] `flow_matcher_v3_drifting/` loads without errors
- [ ] Config parser accepts all 3 new drift parameters
- [ ] ODE solver produces valid trajectories (shape, dtype correct)
- [ ] Drift loss backpropagates (gradient shape matches network params)
- [ ] Training loop runs end-to-end (1 epoch, no NaN)

### 9.2 Functional Validation

- [ ] FM-D trajectories satisfy constraints (collision-free, dynamics-valid)
- [ ] Drift loss improves over baseline FM-ODE (measured by test set loss)
- [ ] Inference runs in <T seconds per trajectory (vs FMv3ODE benchmark)
- [ ] Memory bank correctly stores and retrieves reference trajectories

### 9.3 Evaluation Validation

- [ ] FID score on image generation tasks (if applicable)
- [ ] Success rate on D3IL control tasks (vs DPCC and FMv3ODE)
- [ ] Ablation: λ sweep shows improvement over λ=0 (baseline FM)
- [ ] Visualizations: drift loss landscape and sampled paths align

---

## 10) Timeline & Milestones

| Phase | Duration | Milestones |
|-------|----------|-----------|
| **Phase 1: Foundation** | 1-2 weeks | Folder structure, drift_unet.py, drift_loss.py, config update |
| **Phase 2: Training** | 2-3 weeks | Loss integration, memory bank, logging, first clean training run |
| **Phase 3: Sampling** | 1-2 weeks | Projection compat, ODE+projection loop, inference validation |
| **Phase 4: Evaluation** | 1-2 weeks | Benchmarking, ablations, notebook demo, documentation |
| **Total** | ~6-9 weeks | Production-ready FM-D engine |

---

## 11) Documentation & References

### Code Documentation
- [ ] Docstrings for all public functions (drift_unet, drift_loss, sampling)
- [ ] Inline comments explaining drift field math
- [ ] Type hints on all function signatures

### User-Facing Guides
- [ ] `FM-D_QUICKSTART.md` (how to run inference)
- [ ] `FM-D_CONFIG_GUIDE.md` (drift parameter tuning)
- [ ] `FM-D_ADVANCED.md` (custom loss functions, memory bank tuning)

### Papers & Theory
- Link to FMv3ODE paper/notes
- Link to Drifting paper (arXiv:2602.04770)
- Derivation of FM-D hybrid objective

---

## 12) Appendix: FMv3ODE vs FM-D Comparison Table

| Property | FMv3ODE | FM-D |
|----------|---------|------|
| **Generative model** | Flow Matching (FM) | FM + Drift Loss |
| **Sampling** | ODE integration (deterministic) | ODE + drift guidance (deterministic) |
| **Training loss** | FM regression to velocity field | FM + weighted drift loss |
| **Inference steps** | ~10-50 ODE steps (tunable) | ~1-20 steps (drift accelerates) |
| **Constraint handling** | Projection-based | Projection-based (compatible) |
| **Memory footprint** | Model weights + batch buffer | Model + drift memory bank |
| **Convergence** | Guaranteed (smooth FM field) | Similar (drift is regularizer) |
| **Code complexity** | Baseline | +30-40% (drift components) |
| **Potential gains** | Stable, proven | Faster convergence, better fidelity |

---

## 13) Next Steps

1. **Immediate** (next meeting): Review this plan, approve folder structure
2. **Week 1**: Phase 1 foundation (folder copies, drift_unet.py skeleton)
3. **Week 2-3**: Phase 2 training loop (backward compat testing)
4. **Week 4**: Phase 3 sampling + projection validation
5. **Week 5-6**: Phase 4 evaluation + benchmarking

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-12  
**Author**: Planning Document  
**Status**: Ready for Review
