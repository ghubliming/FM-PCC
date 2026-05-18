# Quick Reference: FMv3ODE vs Gen6 DPCC
## Constraint Projection Architecture Comparison

---

## 1. API Signatures Side-by-Side

### FMv3ODE (Original)
```python
# From flow_matcher_v3_ode_selectable/sampling/projection.py

Projector(
    horizon=8,
    transition_dim=6,        # 3D action + 3D state
    action_dim=3,
    goal_dim=0,
    constraint_list=[...],
    variant='states_actions',
    dt=0.02
)
```

### Gen6 DPCC (Inherited)
```python
# From fm_encdec_vision_test/eval_fm_encdec_vision.py

projector = Projector(
    horizon=args.horizon,
    transition_dim=6,        # Same 6D structure
    action_dim=action_dim,    
    goal_dim=0,
    constraint_list=constraints,
    variant=fm_variant,
    normalizer=scaler,
    dt=dt
)
```

**Difference:** None in the Projector class itself. Gen6 only changes the constraint bounds.

---

## 2. The "6D Vector" Explained

### Data Layout
```
z = [s_0^T s_1^T ... s_{H-1}^T]^T  (flattened trajectory)

where each s_t = [v_x, v_y, v_z, x, y, z]^T  (6D)
                  [0    1    2    3  4  5]      (indices)
                  [<- action -> <-- state -->]
```

### How Constraints Index This Vector

| Constraint Type | Indices Used | Meaning |
|---|---|---|
| **Action bounds** | [0, 1, 2] | Keep velocities bounded |
| **Position bounds** | [3, 4, 5] | Keep position in workspace |
| **Dynamics** | (0↔3), (1↔4), (2↔5) | Euler integration |
| **Obstacles** | [3, 4] only | Avoid circular regions at (x, y) |

### Which Dimensions Get What

```python
# Full constraint list for avoiding task:
constraint_list = [
    ('lb', [-0.01, -0.01, -0.01,  -∞,  -∞,  -∞]),     # Action lower bounds
    ('ub', [ 0.01,  0.01,  0.01,  +∞,  +∞,  +∞]),     # Action upper bounds
    ('deriv', [3, 0]),   # x[t+1] = x[t] + dt*v_x[t]
    ('deriv', [4, 1]),   # y[t+1] = y[t] + dt*v_y[t]
    ('deriv', [5, 2]),   # z[t+1] = z[t] + dt*v_z[t]
    ('sphere_outside', [3, 4], [0.4, 0.08], 0.06),  # Obstacle at (x,y)
]
```

---

## 3. Constraint Classes Architecture

### SafetyConstraints (Bounds + Linear)
```
┌─────────────────────────────────┐
│  Bound/Linear Constraints       │
│  ├─ Type: 'lb', 'ub', 'ineq'   │
│  ├─ Applied: All timesteps      │
│  └─ Form: C·z ≤ d              │
└─────────────────────────────────┘
     ↓
     Builds C ∈ ℝ^(H×6H)  (constraint matrix)
     Builds d ∈ ℝ^H       (constraint RHS)
```

### DynamicConstraints (Euler)
```
┌─────────────────────────────────┐
│  Dynamics Constraints           │
│  ├─ Type: 'deriv'               │
│  ├─ Applied: All timesteps      │
│  └─ Form: A·z = b              │
└─────────────────────────────────┘
     ↓
     Builds A ∈ ℝ^(H-1 × 6H)  (Euler matrix)
     Fixes initial state s₀
```

### ObstacleConstraints (Quadratic)
```
┌─────────────────────────────────┐
│  Obstacle Constraints           │
│  ├─ Type: 'sphere_outside'      │
│  ├─ Applied: t=1..H-1           │
│  └─ Form: s^T P s + q^T s ≤ v  │
└─────────────────────────────────┘
     ↓
     Pre-computes P, q, v
     Applied via scipy.minimize callback
```

---

## 4. Implementation Checklist: FMv3ODE Features in Gen6

| Feature | FMv3ODE | Gen6 | Status |
|---|---|---|---|
| Projector class | ✅ | ✅ | **Identical** |
| SafetyConstraints | ✅ | ✅ | **Identical** |
| DynamicConstraints | ✅ | ✅ | **Identical** |
| ObstacleConstraints | ✅ | ✅ | **Identical** |
| Action bounds source | FMv3ODE config bounds | Workspace [workspace_lb, workspace_ub] | **Modified** |
| State bounds source | FMv3ODE config bounds | Same (unbounded) | **Modified** |
| Obstacle avoidance | ✅ Quadratic | ✅ Quadratic | **Identical** |
| Gradient refinement | ✅ Optional | ✅ Optional | **Identical** |
| Trajectory selection | Multiple variants | Added 'temporal_consistency' | **Extended** |

---

## 5. Key Code Locations

### FMv3ODE Reference Implementation
- **Projector API:** [flow_matcher_v3_ode_selectable/sampling/projection.py#46-100](flow_matcher_v3_ode_selectable/sampling/projection.py#46-100)
- **SafetyConstraints:** [flow_matcher_v3_ode_selectable/sampling/projection.py#249-350](flow_matcher_v3_ode_selectable/sampling/projection.py#249-350)
- **DynamicConstraints:** [flow_matcher_v3_ode_selectable/sampling/projection.py#361-410](flow_matcher_v3_ode_selectable/sampling/projection.py#361-410)
- **ObstacleConstraints:** [flow_matcher_v3_ode_selectable/sampling/projection.py#427-470](flow_matcher_v3_ode_selectable/sampling/projection.py#427-470)
- **Evaluation:** [FM_v3_test/eval_FM_v3.py#50-250](FM_v3_test/eval_FM_v3.py#50-250)
- **Config:** [config/projection_eval.yaml](config/projection_eval.yaml)

### Gen6 Adaptation
- **Setup:** [fm_encdec_vision_test/eval_fm_encdec_vision.py#80-115](fm_encdec_vision_test/eval_fm_encdec_vision.py#80-115)
- **Bound differences:** Lines 95-105 (workspace_lb/ub instead of action bounds)
- **Projector instantiation:** Line 121 (same API as FMv3ODE)

---

## 6. Mathematical Notation Quick Reference

### Projector Optimization Problem
$$\min_{z \in \mathbb{R}^{6H}} \frac{1}{2} z^T Q z + r^T z$$

**Subject to:**
- **Dynamics** (Euler): $Az = b$, where $A \in \mathbb{R}^{(H-1) \times 6H}$
- **Linear constraints** (bounds, halfspace): $Cz \leq d$, where $C \in \mathbb{R}^{K \times 6H}$
- **Quadratic constraints** (obstacles): $z_t^T P z_t + q^T z_t \leq v$ for $t=1..H-1$

### Cost Function
$$Q = \text{eye}(6H) \quad \text{(L2 norm: minimize deviation from diffusion sample)}$$
$$r = -z_{\text{diffusion}}^T Q \quad \text{(warm start from noise)}$$

### Obstacle Constraint Matrices (Unnormalized Case)
$$P = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ & \ddots \\ && 0 \end{bmatrix}$$
$$q = [-2c_x, -2c_y, 0, ...]^T$$
$$v = r^2 - c_x^2 - c_y^2$$

**Interpretation:** $(x - c_x)^2 + (y - c_y)^2 \leq r^2$ ✓ Circle constraint

---

## 7. Common Pitfalls When Modifying Constraints

### ❌ DON'T: Change transition_dim without updating indices
```python
# WRONG
transition_dim = 5  # 2D action + 3D state
constraint_list = [('deriv', [3, 0])]  # Index 3 no longer maps to position!
```

### ✅ DO: Keep consistent mapping
```python
# CORRECT
action_dim = 2
obs_dim = 3
transition_dim = action_dim + obs_dim  # = 5

# Position is at index action_dim = 2 (not 3)
constraint_list = [('deriv', [2, 0])]  # x[t+1] = x[t] + dt*v_x[t]
```

### ❌ DON'T: Apply bounds to action dimensions without understanding skip_initial_state
```python
# WRONG - Action bounds get skipped at t=0
skip_initial_state = True
action_dim = 3
bound = [... -inf, -inf, -inf, ub_x, ub_y, ub_z]  # Action bounds will be skipped!
```

### ✅ DO: Use skip_initial_state correctly
```python
# CORRECT
# Action bounds (dims 0-2) have NO skip_initial_state check → applied at all t
# State bounds (dims 3-5) have skip_initial_state check → applied at t=1..H-1
if self.skip_initial_state and dim >= self.action_dim:
    skip_t0 = True  # Only skip for state dimensions
```

### ❌ DON'T: Forget to update act_obs_indices when changing dimensions
```python
# WRONG - Old indices still assume 3D action space
act_obs_indices['x'] = 3  # But now transition_dim = 5, so x should be at index 2
```

### ✅ DO: Regenerate indices dynamically
```python
# CORRECT
action_dim = 2
obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
act_obs_indices = {**act_indices, **obs_indices_updated}
```

---

## 8. Testing Checklist for Constraint Modifications

Before deploying constraint changes:

- [ ] Verify `transition_dim` = actual vector size in sampled trajectories
- [ ] Verify `action_dim` correctly identifies action vs. state dimensions
- [ ] Check that dynamics constraints link correct dimension pairs
- [ ] Confirm obstacle constraints use state dimensions only (not action dims)
- [ ] Test that bounds on actions are applied at t=0 (not skipped)
- [ ] Test that bounds on states are applied at t=1..H-1 (initial state fixed)
- [ ] Validate normalizer dimensions match transition_dim
- [ ] Run simple 1-obstacle test case to verify quadratic constraint correctness
- [ ] Check gradient computation in `compute_gradient()` matches constraint application
- [ ] Verify trajectory selection metric (projection cost, temporal consistency, etc.)

---

## 9. Summary: Why Gen6 Works

### Inherited from FMv3ODE ✅
1. Unified 6D constraint formulation
2. Proper handling of state-action coupling via Explicit Euler
3. Quadratic obstacle constraint formulation matching Euclidean geometry
4. Normalizer-aware constraint denormalization
5. Multi-sample projection with trajectory selection

### Modified by Gen6 ✅
1. Changed bound sources from action limits to workspace bounds
2. Added temporal consistency trajectory selection metric
3. Adapted normalizer handling for vision-conditioned model

### Net Result
✅ **Gen6 inherits a robust, well-tested constraint projection engine from FMv3ODE and only modifies constraint sourcing, not the underlying architecture.**

This is a **design pattern worth replicating:**
- Core optimization algorithm stays stable
- Only plug in different constraint specifications
- Enables rapid iteration without rewriting projection solver

---

## Document Cross-References

1. **[FMv3ODE_Direct_Answers.md](FMv3ODE_Direct_Answers.md)** — Direct answers to 5 original questions
2. **[FMv3ODE_Constraint_Projection_Analysis.md](FMv3ODE_Constraint_Projection_Analysis.md)** — Detailed mathematical formulation
3. **[FMv3ODE_Obstacle_Avoidance_Formulation.md](FMv3ODE_Obstacle_Avoidance_Formulation.md)** — Obstacle constraint deep dive
4. **[Quick_Reference.md](Quick_Reference.md)** — This document
