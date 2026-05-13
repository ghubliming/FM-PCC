# FM-PCC Gen3v4: Improved Mean Flows for Trajectory Control

**Date**: 2026-05-13  
**Status**: 🎯 **PLANNING PHASE**  
**Base Foundation**: FMv3ODE (battle-tested, production-ready)  
**Development Pattern**: Learning from FM-D (Phases 1-4 methodology)  
**Target Launch**: Phase-based implementation (Phases 1-4)

---

## Executive Summary

**Gen3v4 Vision**: Build on **proven FMv3ODE foundation** + integrate **Improved Mean Flows** (`imeanflow`) architecture to achieve:

1. ✅ **Faster trajectory generation** - Single-step (NFE=1) or dual-step (NFE=2) sampling
2. ✅ **Better velocity field learning** - Decoupled u (average) and v (instantaneous) velocity prediction
3. ✅ **Superior trajectory quality** - Improved loss objectives beyond simple L2/L1
4. ✅ **DiT backbone option** - Transformer-based architecture for scalability
5. ✅ **De-risked implementation** - Start from FMv3ODE (NOT FM-D), inherit its stability

**Key Insight**: 
- **FMv3ODE** = proven baseline (no drift, no experimental components)
- **FM-D** = showed us HOW to extend FM-PCC (4-phase methodology)
- **Gen3v4** = Apply FM-D's development lifecycle to FMv3ODE + iMeanFlow innovations

**Philosophy**: Copy FMv3ODE's tested architecture. Learn extension patterns from FM-D. Add imeanflow's velocity field innovations. Keep it simple, keep it stable.

---

## Inheritance Strategy: From FMv3ODE, NOT FM-D

### Why Start from FMv3ODE (Not FM-D)?

| Aspect | FM-D | FMv3ODE | Gen3v4 |
|--------|------|---------|--------|
| **Foundation** | FMv3ODE + drift | Core framework | ✅ Our base |
| **Tested** | New (adapted code) | Battle-tested | Safe to build on |
| **Complexity** | Higher (drift loss) | Moderate | Controlled |
| **Risk** | Medium (new + adaptation) | Low | Low (proven) |
| **Stability** | Good (for FM) | Excellent | Inherits excellent |

**Decision**: Copy `flow_matcher_v3_ode_selectable/` directly → `flow_matcher_v3_imeanflow/`

### What We Learn From FM-D (The Methodology)

FM-D showed us a **proven 4-phase development pattern**:
- ✅ Phase 1: Foundation (copy folder, new modules, config injection)
- ✅ Phase 2: Training utilities (loss, scheduler, metrics)
- ✅ Phase 3: Sampling & inference (ODE solvers, APIs)
- ✅ Phase 4: Testing & documentation (tests, examples, guides)

**Gen3v4 will follow the EXACT SAME PATTERN** but starting from FMv3ODE.

### Optional Future: FM-D + iMeanFlow

Later (Gen3v5?), could combine:
- Gen3v4 iMeanFlow (u/v split on FMv3ODE)
- Gen3v3 Drift Loss (expert-aligned trajectories)

But NOT in Gen3v4 - keep focus on iMeanFlow innovations.

---

## What is imeanflow?

### Paper: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"

**Core Contribution**: Decouples velocity field into two components:

$$z_t = z_r + \int_r^t \underbrace{u(\tau)}_{\text{average}} d\tau + \int_r^t \underbrace{v(\tau)}_{\text{instantaneous}}d\tau$$

Where:
- **u(τ)** = average velocity (global trajectory direction)
- **v(τ)** = instantaneous velocity (local refinement, includes JVP term)
- **JVP** = Jacobian-Vector Product (second-order derivative guidance)

### Why It Matters for Trajectories

Traditional FM learns single velocity: $v_{FM}(x,t)$.  
Problems:
- Single velocity must balance global direction AND local correction
- Difficult to learn both simultaneously
- Causes suboptimal solution quality

imeanflow separates concerns:
- **u** learns "where to go?" (global trajectory shape)
- **v** learns "how to refine?" (local corrections, including constraints)

### Current Status (Image Generation)

| Metric | NFE | FID | IS |
|--------|-----|-----|-----|
| iMF-B/2 | 1 | 3.32 | 255.7 |
| iMF-M/2 | 1 | 2.26 | 258.3 |
| iMF-L/2 | 1 | 1.83 | 275.7 |
| iMF-XL/2 | 1 | 1.72 | 279.9 |
| iMF-XL/2 | 2 | 1.54 | 288.9 |

**Key**: State-of-the-art with 1-2 function evaluations (vs. 50-100 for diffusion).

---

## Strategic Position: FMv3ODE → Gen3v4

### FMv3ODE (Proven Baseline)
- ✅ Single velocity field: $v_{FM}(x,t)$
- ✅ ODE integration (dopri5, Euler, RK4)
- ✅ Battle-tested, production-ready
- ✅ ~50 ODE steps for good quality

**Limitation**: Monolithic velocity (doesn't separate global + local).

### Gen3v4 (iMeanFlow on Proven Base)
- ✅ Dual velocity: u (global) + v (local)
- ✅ Same ODE integration options
- ✅ Inherit FMv3ODE's stability
- ✅ Learn u + v with 1-2 steps for excellent quality

**Advantage**: 
- Cleaner separation of concerns
- Faster convergence
- 25-50× fewer ODE steps needed

---

## Architecture Overview

### Phase 1: Foundation (Weeks 1-2)

**Objective**: Create production-ready baseline from FMv3ODE.

#### 1.1 Folder Structure

```
flow_matcher_v3_imeanflow/           ← COPY from FMv3ODE (NOT FM-D!)
├── models/
│   ├── diffusion.py                 ← COPY from FMv3ODE
│   ├── unet1d_temporal_cond.py      ← COPY from FMv3ODE
│   ├── helpers.py                   ← COPY from FMv3ODE
│   ├── imf_velocity.py              ← NEW: u (avg) + v (inst) split
│   ├── imf_dit_trajectory.py        ← NEW: DiT backbone for trajectories
│   └── jvp_guidance.py              ← NEW: Jacobian-Vector Product module
├── sampling/
│   ├── ode_solvers.py               ← COPY from FMv3ODE
│   ├── imf_ode_solvers.py           ← NEW: Single-step + dual-step sampling
│   └── imf_trajectory_sampler.py    ← NEW: High-level sampling API
├── utils/
│   ├── training.py                  ← COPY from FMv3ODE
│   ├── imf_training.py              ← NEW: Training loop (u + v losses)
│   └── imf_metrics.py               ← NEW: Metrics for dual-velocity
├── configs/
│   ├── imf_trajectory_base.yaml     ← NEW: Default iMF config
│   ├── imf_trajectory_d3il.yaml     ← NEW: D3IL specialization
│   └── imf_trajectory_avoiding.yaml ← NEW: Avoidance specialization
└── examples/
    ├── example_imf_training.py      ← NEW: Training walkthrough
    └── example_imf_inference.py     ← NEW: Sampling walkthrough
```

**Key Difference from FM-D**: 
- FM-D added drift modules to FMv3ODE
- Gen3v4 adds iMeanFlow modules to FMv3ODE
- Both follow same 4-phase pattern

#### 1.2 Test Folder

```
FM_v3_imeanflow_test/
├── train_flow_matching_v3_ode_selectable.py  ← COPY base (rename for clarity)
├── test_imf_velocity.py          ← NEW: 4 tests
├── test_imf_dit.py               ← NEW: 5 tests  
├── test_jvp_guidance.py          ← NEW: 3 tests
├── test_imf_ode_solvers.py       ← NEW: 5 tests
└── test_imf_training.py          ← NEW: 4 tests
```

#### 1.3 Key Modules

**`models/imf_velocity.py`** (150 lines)
```python
class DualVelocityField(nn.Module):
    """Separate u (average) and v (instantaneous) prediction."""
    
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.u_net = MLP(state_dim → hidden_dim → state_dim)  # Global
        self.v_net = MLP(state_dim → hidden_dim → state_dim)  # Local
        self.jvp_encoder = JVPModule(state_dim, hidden_dim)   # 2nd-order

    def forward(self, x, t, cond):
        u = self.u_net(x, t, cond)           # Average velocity
        v_base = self.v_net(x, t, cond)      # Instantaneous (1st-order)
        v_jvp = self.jvp_encoder(x, v_base)  # JVP refinement
        return u, v_base + v_jvp              # Combined v
```

**`models/imf_dit_trajectory.py`** (200 lines)
```python
class ImfDiT(nn.Module):
    """DiT backbone for dual-velocity trajectory prediction."""
    
    def __init__(self, trajectory_dim, num_layers, num_heads):
        super().__init__()
        # Multi-head self-attention on trajectory tokens
        # Separate output heads for u and v
        
    def forward(self, x, t, cond):
        # x: (B, T, state_dim) trajectory sequence
        # t: (B,) time step
        # cond: (B, cond_dim) condition
        # Returns: (u, v) both (B, T, state_dim)
```

**`models/jvp_guidance.py`** (100 lines)
```python
class JVPGuidance(nn.Module):
    """Jacobian-Vector Product for constraint satisfaction."""
    
    def forward(self, x, v_base, constraint_fn=None):
        # Compute JV where J = ∇_x constraint(x)
        # Add to v_base for constraint-aware refinement
        with torch.enable_grad():
            J = torch.autograd.jacobian(constraint_fn, x)  # Jacobian
            jvp = torch.einsum('...ij,j->i', J, v_base)   # Product
        return jvp
```

#### 1.4 Config Injection

Edit `config/avoiding-d3il.py`:

```python
'flow_matching_v3_imeanflow': {
    # Base FM parameters (inherited from FMv3ODE)
    'state_dim': 34,
    'action_dim': 7,
    'condition_dim': 28,
    'max_path_length': 200,
    'ode_solver': 'dopri5',
    'steps': 10,
    
    # NEW: Dual velocity (iMF) parameters
    'use_dual_velocity': True,          # Enable u + v split
    'u_loss_weight': 0.5,               # Weight of u prediction
    'v_loss_weight': 0.5,               # Weight of v prediction
    'jvp_weight': 0.1,                  # JVP guidance strength
    'jvp_constraint': 'collision_free', # or None, 'smooth', etc.
    
    # DiT architecture (alternative to U-Net)
    'backbone': 'unet',                 # Default: 'unet' or 'dit'
    'dit_depth': 12,                    # Transformer layers (if dit)
    'dit_hidden': 768,                  # Hidden dimension (if dit)
    'dit_heads': 12,                    # Attention heads (if dit)
}
```

---

### Phase 2: Training Utilities (Weeks 2-3)

**Objective**: Implement training loop for dual-velocity learning.

#### 2.1 Loss Function

$$\mathcal{L}_{total} = \alpha \mathcal{L}_u + \beta \mathcal{L}_v + \gamma \mathcal{L}_{JVP}$$

Where:
- $\mathcal{L}_u = ||u_{pred} - u_{expert}||^2$ (average velocity MSE)
- $\mathcal{L}_v = ||v_{pred} - v_{expert}||^2$ (instantaneous velocity MSE)
- $\mathcal{L}_{JVP}$ = constraint satisfaction loss (optional)

**`utils/imf_training.py`** (250 lines)
```python
class DualVelocityLoss(nn.Module):
    """Combined u + v + JVP loss."""
    
    def forward(self, u_pred, u_target, v_pred, v_target, jvp_penalty=None):
        loss_u = F.mse_loss(u_pred, u_target)
        loss_v = F.mse_loss(v_pred, v_target)
        loss_jvp = self.compute_jvp_loss(v_pred) if jvp_penalty else 0
        
        return (
            self.weight_u * loss_u +
            self.weight_v * loss_v +
            self.weight_jvp * loss_jvp
        )

class ImfTrainingWrapper:
    """End-to-end training for iMF trajectories."""
    
    def __init__(self, model, scheduler=None):
        self.model = model
        self.scheduler = scheduler
    
    def compute_training_loss(self, batch):
        # Forward pass
        u_pred, v_pred = self.model(batch['x'], batch['t'], batch['cond'])
        
        # Extract expert u, v from target trajectories
        u_expert = extract_average_velocity(batch['target'])
        v_expert = extract_instantaneous_velocity(batch['target'])
        
        # Compute combined loss
        loss = DualVelocityLoss(
            u_pred, u_expert,
            v_pred, v_expert,
        )
        
        return loss, {'loss_u': ..., 'loss_v': ..., 'loss_total': ...}
```

#### 2.2 Metrics

**`utils/imf_metrics.py`** (200 lines)
```python
class ImfMetricsTracker:
    """Track u, v, and combined trajectory metrics."""
    
    def update(self, u_pred, u_expert, v_pred, v_expert):
        self.u_mse = F.mse_loss(u_pred, u_expert)
        self.v_mse = F.mse_loss(v_pred, v_expert)
        self.combined_error = self.u_mse + self.v_mse
        
        # Trajectory-level metrics
        self.trajectory_smoothness = compute_smoothness(u_pred + v_pred)
        self.velocity_decomposition = self.analyze_uv_split()
```

---

### Phase 3: Sampling & Inference (Weeks 3-4)

**Objective**: Single-step and dual-step trajectory generation.

#### 3.1 ODE Integration

**Single-step (NFE=1)**:
```python
z_final = z_0 + ∫₀¹ u(τ) dτ + ∫₀¹ v(τ) dτ
# Entire trajectory in one ODE evaluation
```

**Dual-step (NFE=2)**:
```python
# Step 1: Refine with u (average)
z_mid = z_0 + ∫₀^0.5 u(τ) dτ

# Step 2: Refine with v (instantaneous)
z_final = z_mid + ∫₀.5^1 v(τ) dτ
```

**`sampling/imf_ode_solvers.py`** (200 lines)
```python
class ImfODESolver:
    """Single and dual-step iMF sampling."""
    
    def sample_single_step(self, model, x0, cond, t_span=(0, 1)):
        """NFE=1: Integrate u + v directly."""
        
        def combined_velocity(t, x):
            u, v = model(x, t, cond)
            return u + v
        
        x_final = solve_ode(combined_velocity, x0, t_span)
        return x_final
    
    def sample_dual_step(self, model, x0, cond, t_split=0.5):
        """NFE=2: Refine trajectory in two stages."""
        
        # Stage 1: Global direction (u-dominated)
        def u_velocity(t, x):
            u, _ = model(x, t, cond)
            return u
        
        x_mid = solve_ode(u_velocity, x0, (0, t_split))
        
        # Stage 2: Local refinement (v-dominated)
        def v_velocity(t, x):
            _, v = model(x, t, cond)
            return v
        
        x_final = solve_ode(v_velocity, x_mid, (t_split, 1.0))
        return x_final
```

#### 3.2 Trajectory Sampler

**`sampling/imf_trajectory_sampler.py`** (150 lines)
```python
def sample_trajectory_imf(
    model,
    x0,
    cond,
    nfe=1,              # 1 or 2
    solver='dopri5',
):
    """High-level API matching FMv3ODE interface."""
    
    if nfe == 1:
        return solver.sample_single_step(model, x0, cond)
    elif nfe == 2:
        return solver.sample_dual_step(model, x0, cond)
```

---

### Phase 4: Testing & Documentation (Week 4)

**Objective**: Comprehensive testing & user guides.

#### 4.1 Test Suite
- `test_imf_velocity.py` (100 lines) - u/v split correctness
- `test_imf_dit.py` (150 lines) - DiT backbone validation
- `test_jvp_guidance.py` (80 lines) - Constraint satisfaction
- `test_imf_ode_solvers.py` (150 lines) - Single/dual-step sampling
- `test_imf_training.py` (120 lines) - Training loop integration

**Total Tests**: 21 tests (matching FM-D pattern)

#### 4.2 Documentation
- `README.md` - Architecture overview
- `QUICKSTART.md` - Training & inference guide
- `CODE_EXPLANATION.md` - Technical deep dive
- `MISSION_BRIEFING.md` - All changes made

---

## Implementation Phases Timeline

| Phase | Duration | Deliverables | Status |
|-------|----------|--------------|--------|
| **Phase 1** | Weeks 1-2 | Folder copy from FMv3ODE, u/v split, DiT, config | ⏳ To Start |
| **Phase 2** | Weeks 2-3 | Training loop, dual loss, metrics | ⏳ To Start |
| **Phase 3** | Weeks 3-4 | Single/dual-step sampling, ODE solvers, API | ⏳ To Start |
| **Phase 4** | Week 4 | Tests (21 tests), examples, documentation | ⏳ To Start |
| **Phase 5** | Weeks 5-6 | Evaluation & benchmarking (optional) | 🎯 Future |

---

## Integration Points

### With FMv3ODE
- ✅ 100% backward compatible (copy folder, no modifications)
- ✅ Same ODE solvers, training infrastructure
- ✅ Same config parsing, validation
- ✅ Inherits all FMv3ODE's battle-tested code

### With Existing FM-PCC
- ✅ Single training script: `scripts/train.py`
- ✅ Config block: `'flow_matching_v3_imeanflow'`
- ✅ Inference: `sample_trajectory_imf()` function
- ✅ Works with existing data pipelines

### With imeanflow Torch Repo
- ✅ Adapt DiT backbone from `/workspaces/imeanflow/models/imfDiT.py`
- ✅ Port JVP guidance concepts from original paper
- ✅ Keep inference-only approach (match repo style)

### Optional Future: Gen3v3 (FM-D)
- Later (Gen3v5?): Could combine iMeanFlow + drift loss
- For now: Keep Gen3v4 focused on u/v split innovations
- No dependency on FM-D code

---

## Key Design Decisions

### 1. **Copy from FMv3ODE, NOT FM-D**
- ✅ FMv3ODE is proven, stable, battle-tested
- ✅ FM-D added complexity (drift loss adaptation)
- ✅ Gen3v4 starts clean, inherits reliability
- ✅ Can add drift loss later if needed (Gen3v5)

### 2. **Follow FM-D's Development Pattern (Not Code)**
- ✅ Same 4-phase structure (Foundation → Training → Sampling → Testing)
- ✅ Same 21-test target, documentation style
- ✅ Same config injection pattern
- ✅ Different implementation (iMeanFlow vs drift)

### 3. **Dual Velocity Over Monolithic**
- ✅ u (average) learns global shape
- ✅ v (instantaneous) learns local refinement
- ✅ Easier to train, faster convergence
- ✅ Separates concerns cleanly

### 4. **DiT as Optional Backbone**
- ✅ Keep U-Net as default (proven)
- ✅ Offer DiT as upgrade option
- ✅ Switch via config: `'backbone': 'dit'` or `'unet'`

### 5. **NFE=1 vs NFE=2 Runtime**
- ✅ Single-step: Fast, real-time (matches imeanflow promise)
- ✅ Dual-step: Slower but higher quality
- ✅ User choice: Set `nfe` parameter

### 6. **Backward Compatibility**
- ✅ Disable dual-velocity: `'use_dual_velocity': False` → pure FM
- ✅ Disable JVP: `'jvp_weight': 0` → standard v learning
- ✅ Can revert to FMv3ODE behavior entirely
- ✅ Never breaks existing infrastructure

---

## Success Metrics & Validation

### Quantitative
1. **Trajectory Quality**: Lower MSE to expert trajectories (vs FMv3ODE)
2. **Sampling Speed**: NFE=1-2 vs. traditional 50-step solvers
3. **Constraint Satisfaction**: Collision-free rate, smoothness
4. **Training Convergence**: Loss curves (u, v independent tracking)

### Qualitative
1. ✅ u learns interpretable global direction
2. ✅ v learns localized refinement (especially near constraints)
3. ✅ Dual-step sampling visibly improves over single-step

### Comparison Points
- **FMv3ODE**: 50 ODE steps, monolithic velocity
- **Gen3v4 NFE=1**: 1 function evaluation, dual velocity
- **Gen3v4 NFE=2**: 2 function evaluations, dual velocity

Expected: 20-30× speedup with better or comparable quality.

---

## Known Challenges & Mitigation

| Challenge | Root Cause | Mitigation |
|-----------|-----------|-----------|
| Training instability | Both u and v active early | Initialize v as copy of u, gradually decouple |
| JVP computation overhead | Jacobian expensive | Approximate or cache, optional component |
| DiT scaling on trajectories | Quadratic attention | Use linear attention or local windowing |
| Extracting u, v from targets | No ground truth | Fit polynomial, Use FD, or learn from data |

---

## File Inventory (Preview)

### New Files to Create
```
flow_matcher_v3_imeanflow/
├── models/
│   ├── imf_velocity.py          (150 lines) ← NEW
│   ├── imf_dit_trajectory.py    (200 lines) ← NEW
│   └── jvp_guidance.py          (100 lines) ← NEW
├── sampling/
│   ├── imf_ode_solvers.py       (200 lines) ← NEW
│   └── imf_trajectory_sampler.py (150 lines) ← NEW
├── utils/
│   ├── imf_training.py          (250 lines) ← NEW
│   └── imf_metrics.py           (200 lines) ← NEW
├── configs/
│   ├── imf_trajectory_base.yaml ← NEW
│   ├── imf_trajectory_d3il.yaml ← NEW
│   └── imf_trajectory_avoiding.yaml ← NEW
└── examples/
    ├── example_imf_training.py  ← NEW
    └── example_imf_inference.py ← NEW
```

### Copied Files (from FMv3ODE)
```
flow_matcher_v3_imeanflow/
├── models/
│   ├── diffusion.py             ← COPY
│   ├── unet1d_temporal_cond.py  ← COPY
│   ├── helpers.py               ← COPY
│   └── mlp.py                   ← COPY
├── sampling/
│   ├── ode_solvers.py           ← COPY
│   └── ode_inference.py         ← COPY
├── utils/
│   ├── training.py              ← COPY
│   └── arrays.py                ← COPY
└── datasets/                    ← COPY
```

### Tests to Create
```
FM_v3_imeanflow_test/
├── test_imf_velocity.py         (100 lines, 4 tests) ← NEW
├── test_imf_dit.py              (150 lines, 5 tests) ← NEW
├── test_jvp_guidance.py         (80 lines, 3 tests) ← NEW
├── test_imf_ode_solvers.py      (150 lines, 5 tests) ← NEW
└── test_imf_training.py         (120 lines, 4 tests) ← NEW
```

**Total**: ~1,400 lines of new code + 21 tests (copying ~500 lines from FMv3ODE)

---

## Success Checklist

Phase-by-phase completion criteria:

### Phase 1: Foundation ✅
- [ ] `flow_matcher_v3_imeanflow/` created (copy FMv3ODE, NOT FM-D)
- [ ] Core models copied unchanged: diffusion.py, unet1d, helpers.py
- [ ] `models/imf_velocity.py` fully implemented
- [ ] `models/imf_dit_trajectory.py` backbone integrated
- [ ] `models/jvp_guidance.py` constraint guidance working
- [ ] Config block added to `avoiding-d3il.py`
- [ ] All FMv3ODE tests still pass on new folder

### Phase 2: Training ✅
- [ ] Dual-velocity loss correctly computes u and v MSE
- [ ] `DualVelocityLoss` integrates u, v, JVP terms
- [ ] `ImfTrainingWrapper` coordinates training
- [ ] `DualVelocityMetricsTracker` logs u, v separately
- [ ] Training loop runs without numerical issues

### Phase 3: Sampling ✅
- [ ] Single-step sampling (NFE=1) working
- [ ] Dual-step sampling (NFE=2) working
- [ ] `sample_trajectory_imf()` API functional
- [ ] ODE solver selection working (dopri5 | euler | rk4)
- [ ] Sampling speed benchmarked vs FMv3ODE

### Phase 4: Testing & Docs ✅
- [ ] All 21 unit tests passing
- [ ] Training example script runs
- [ ] Inference example script produces trajectories
- [ ] README with architecture overview
- [ ] QUICKSTART with commands
- [ ] CODE_EXPLANATION with implementation details
- [ ] This plan + MISSION_BRIEFING documenting all changes

---

## Comparison: FMv3ODE vs Gen3v4

| Aspect | FMv3ODE | Gen3v4 |
|--------|---------|--------|
| **Velocity** | Single: v | Dual: u + v |
| **Loss** | FM only | u + v + JVP |
| **Sampling** | ODE (50 steps typical) | NFE=1-2 |
| **Speed** | Baseline | 25-50× faster |
| **Backbone** | U-Net | U-Net or DiT |
| **Stability** | Excellent | Inherits excellent |
| **Complexity** | Standard | Moderate increase |
| **Training Time** | Baseline | ~1.5× (more to learn) |
| **Quality** | Proven | Expected improvement |

---

## Next Steps (Ready for Phase 1)

1. ✅ Review this plan (you're here)
2. ⏳ **Approve architecture** - Confirm u/v split, DiT option, JVP guidance
3. ⏳ **Create Phase 1 Mission Briefing** - Detailed engineering checklist
4. ⏳ **Begin implementation** - Start with copying `flow_matcher_v3_ode_selectable/` → `flow_matcher_v3_imeanflow/`

---

## References

**Original imeanflow Paper**:
- "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
- Authors: Lyy-iiis et al. | 2025
- arXiv: [2512.02012](https://arxiv.org/abs/2512.02012)

**iMF PyTorch Repo**:
- GitHub: https://github.com/Lyy-iiis/imeanflow

**FM-D Development Pattern** (our methodology):
- See `/workspaces/FM-PCC/logs_in_develop/Gen3v3/FM-D_*` files

**FMv3ODE Baseline** (our foundation):
- See `/workspaces/FM-PCC/flow_matcher_v3_ode_selectable/`

---

**Prepared by**: FM-PCC Development  
**Document Creation**: 2026-05-13  
**Status**: 🎯 Ready for Phase 1 Development  
**Key Principle**: Copy FMv3ODE (proven), learn from FM-D (methodology), add iMeanFlow (innovation)  
**Next Document**: Gen3v4 Mission Briefing (Phase 1 Detailed Engineering Tasks)
