# FMv3ODE Constraint Projection Analysis
## Understanding the Origin of the "6D [action, obs]" Representation

**Date:** 2026-05-18  
**Purpose:** Trace the constraint projection design through FMv3ODE to Gen6 DPCC

---

## 1. FMv3ODE Constraint Projection Architecture

### 1.1 Projector Signature

All FMv3ODE variants use a unified Projector API:

```python
class Projector:
    def __init__(self, 
        horizon,           # Planning horizon
        transition_dim,    # Total dimension of [action, state] or [state] only
        action_dim=0,      # Number of action dimensions (default: 0)
        goal_dim=0,        # Number of goal/reference dimensions
        constraint_list=[], # List of constraints to apply
        normalizer=None,   # Data normalizer for normalization/denormalization
        variant='states',  # 'states' or 'states_actions' (determines what's in transition_dim)
        dt=0.1,           # Time step for derivative constraints
        ...
    )
```

**Key Parameters:**
- `transition_dim`: The **total dimension** passed to the projector
- `action_dim`: How many of the first `transition_dim` dimensions are actions
- `variant`: Controls how the normalizer is constructed (includes actions or not)

### 1.2 The "6D [action, obs]" Approach in FMv3ODE

Looking at the FMv3 test evaluation code ([FM_v3_test/eval_FM_v3.py](FM_v3_test/eval_FM_v3.py#L66)):

```python
if fm_model.__class__.__name__ == 'GaussianDiffusion':
    trajectory_dim = fm_model.transition_dim - fm_model.goal_dim
    action_dim = fm_model.action_dim
    fm_variant = 'states_actions'
    obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
    act_obs_indices = {**act_indices, **obs_indices_updated}
```

**Critical Finding:**
- `transition_dim` in the **model** = action_dim + obs_dim (e.g., 3D actions + 3D obs = 6D)
- When passed to Projector, this 6D dimension is used directly
- The Projector **internally interprets this as [action_0, action_1, action_2, obs_0, obs_1, obs_2]**

### 1.3 Constraint Formulation in FMv3ODE

#### SafetyConstraints (Bounds & Linear Constraints)

[flow_matcher_v3_ode_selectable/sampling/projection.py#L249-L350](flow_matcher_v3_ode_selectable/sampling/projection.py#L249-L350):

```python
class SafetyConstraints(Constraints):
    def __init__(self, skip_initial_state=True, action_dim=0, ...):
        self.skip_initial_state = skip_initial_state
        self.action_dim = action_dim
        
    def build_matrices(self, constraint_list=None):
        # For bounds ('lb'/'ub' constraints):
        # - Iterates over ALL dimensions in bound vector (0 to transition_dim)
        # - Applies bounds at EVERY timestep (t=0 to t=horizon-1)
        # - BUT: skips t=0 for dimensions where dim >= action_dim
        #   (initial state dimensions beyond actions are fixed)
        
        if self.skip_initial_state and dim >= self.action_dim:
            mat_append = mat_append[1:]  # Remove t=0 row
            vec_append = vec_append[1:]
```

**Bounds Application Logic:**
- **Action dimensions (0 to action_dim-1):** Bounds applied at **ALL** timesteps including t=0
- **State dimensions (action_dim to transition_dim-1):** Bounds applied at t=1 to t=H-1 (initial state fixed)

#### From the FMv3ODE eval config ([config/projection_eval.yaml](config/projection_eval.yaml#L73-L77)):

```yaml
bounds: {
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
    ...
  ],
}
```

**Interpretation:**
- Bounds are **only specified for actions (vx, vy)**
- These map to dimensions [0, 1] in the 6D representation
- Bounds are applied to **all timesteps** (including t=0)

#### DynamicConstraints (Explicit Euler)

[flow_matcher_v3_ode_selectable/sampling/projection.py#L361-L410]:

```python
class DynamicConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        # For ('deriv', [x_idx, dx_idx]) constraints:
        # Creates equality constraints: x[t+1] = x[t] + dt * dx[t]
        # Also enforces x[0] = s_0 (current state) via skip_initial_state
        
        # Example: ('deriv', [3, 0])  # x_position couples to vx action
        # Produces: x[t+1] - x[t] - dt*vx[t] = 0
```

**Key Feature:** The dynamics constraints **cross-link dimensions** - actions influence how states evolve.

#### ObstacleConstraints (Quadratic/Sphere Constraints)

[flow_matcher_v3_ode_selectable/sampling/projection.py#L427-L470]:

```python
class ObstacleConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        # For sphere constraints: s^T P s + q^T s <= v
        # Where s is the full state-action vector at timestep t
        
        # P is diagonal for axis-aligned spheres
        # P[dim, dim] = delta_s^2 / 4 (where delta_s = s_max - s_min)
        # q[dim] = delta_s^2/2 + delta_s*(s_min - center[dim])
        # v = radius^2 - (normalization_adjustment)
```

**Critical Point:** 
- Obstacle constraints are applied to **state dimensions only** (not actions)
- The dimensions specified in the constraint (e.g., ['x', 'y']) are mapped via `act_obs_indices`
- In the 6D representation: x maps to dimension 3, y maps to dimension 4 (after the 3D action space)

---

## 2. How the "6D [action, obs]" Representation Works in FMv3ODE

### 2.1 Data Layout in Projector

When `transition_dim=6` and `action_dim=3`:

```
Dimension indices in transition vector s:
[0, 1, 2,      3, 4, 5]
[vx, vy, vz,   x, y, z]
[<- actions -> <-- state -->]
```

### 2.2 Constraint Index Mapping

From evaluation code ([FM_v3_test/eval_FM_v3.py#L70-72](FM_v3_test/eval_FM_v3.py#L70-72)):

```python
obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
act_obs_indices = {**act_indices, **obs_indices_updated}
```

**Result:** 
- `act_obs_indices['vx'] = 0`, `act_obs_indices['vy'] = 1`, `act_obs_indices['vz'] = 2`
- `act_obs_indices['x'] = 3`, `act_obs_indices['y'] = 4`, `act_obs_indices['z'] = 5`

### 2.3 Constraint List Construction

From [config/projection_eval.yaml](config/projection_eval.yaml):

```python
# Dynamic constraints bind position to velocity
dynamic_constraints = [
    ('deriv', [act_obs_indices['x'], act_obs_indices['vx']]),  # (3, 0)
    ('deriv', [act_obs_indices['y'], act_obs_indices['vy']]),  # (4, 1)
    ('deriv', [act_obs_indices['z'], act_obs_indices['vz']]),  # (5, 2)
]

# Bounds on actions
constraint_list.append(['lb', lower_bound])  # 6D vector
constraint_list.append(['ub', upper_bound])  # 6D vector

# Obstacle avoidance on state positions
for constr in obstacle_constraints:
    constraint_list.append([
        constr['type'],                           # 'sphere_outside'
        [act_obs_indices[constr['dimensions'][0]], 
         act_obs_indices[constr['dimensions'][1]]],  # (3, 4) for x, y
        constr['center'],
        constr['radius']
    ])
```

---

## 3. Constraint Application Summary

### 3.1 What Gets Bounded?

| Constraint Type | Dimensions | Applied Where | Applied When | Implementation |
|---|---|---|---|---|
| **Bounds ('lb'/'ub')** | All (0-5) | At every timestep | Specified in bound vector | SafetyConstraints |
| **Dynamics ('deriv')** | Crosstalk (0↔3, 1↔4, 2↔5) | All timesteps t=0..H-1 | As equality constraints (A·z = b) | DynamicConstraints |
| **Obstacles ('sphere')** | State only (3, 4) | Timesteps t=1..H-1 | As quadratic constraints | ObstacleConstraints |
| **Halfspace ('ineq')** | State only (3, 4) | Timesteps t=1..H-1 | As linear inequalities | SafetyConstraints |

### 3.2 Key Design Decisions

1. **Action bounds are global** - Applied to all timesteps
2. **State bounds are causal** - Not applied to t=0 (initial state is observed)
3. **Dynamics cross-dimension** - Actions at t directly affect state at t+1 via Euler
4. **Obstacle avoidance on states** - Ignores action dimensions, focuses on position

---

## 4. Why the "6D [action, obs]" Design?

### 4.1 Advantages of Concatenating Actions and States

1. **Unified constraint formulation:** A single `transition_dim=6` vector can represent all constraints
2. **Simple index mapping:** `act_obs_indices` dictionary provides clear dimensional semantics
3. **Batch efficiency:** Single optimization problem over all dimensions
4. **Explicit Euler integration:** Actions directly couple to state derivatives

### 4.2 Alternative Approaches NOT Used

- **Separate action/state spaces:** Would require constraint coordination
- **Action-only planning:** Cannot enforce state bounds directly
- **State-only planning:** Missing explicit action constraints

---

## 5. Comparison: FMv3ODE vs Gen6 DPCC

### 5.1 Inherited from FMv3ODE

✅ **6D concatenation** - Gen6 adopted this directly  
✅ **SafetyConstraints** - Same logic for bounds  
✅ **DynamicConstraints** - Same Euler coupling  
✅ **ObstacleConstraints** - Same quadratic obstacle formulation  
✅ **Normalizer-based denormalization** - Uses ProjectionNormalizer  

### 5.2 Specific to Gen6 DPCC

From [fm_encdec_vision_test/eval_fm_encdec_vision.py#L80-L115](fm_encdec_vision_test/eval_fm_encdec_vision.py#L80-L115):

```python
def setup_gen6_projector(args, config, scaler, variant):
    # Gen6 explicitly sets:
    lb = np.array([-np.inf, -np.inf, -np.inf,  # No bounds on vx, vy, vz
                   workspace_lb[0], workspace_lb[1], workspace_lb[2]])  # Bounds on x, y, z
    ub = np.array([np.inf, np.inf, np.inf,  # No bounds on vx, vy, vz
                   workspace_ub[0], workspace_ub[1], workspace_ub[2]])  # Bounds on x, y, z
    
    # vis_aligning uses workspace bounds instead of action bounds
```

**Key Difference:** Gen6 shifts from **action bounds** to **workspace bounds**, maintaining the 6D structure.

---

## 6. Conclusion

### The "6D [action, obs]" representation is **inherited from FMv3ODE**, not invented by Gen6.

**Origin Tracing:**
1. **FMv2/FM_v3** use `transition_dim = obs_dim` (states only, `action_dim=0`)
2. **FMv3ODE** introduced `transition_dim = action_dim + obs_dim` with `variant='states_actions'`
   - This enables explicit Euler: `x[t+1] = x[t] + dt * a[t]`
   - Simplifies constraint formulation to a **single unified vector**
3. **Gen6 DPCC** adopted this 6D design wholesale
   - Changed bound sources: action bounds → workspace bounds
   - Kept the underlying constraint projection engine

### Design Pattern: "Concatenation for Constraint Simplicity"
The 6D approach trades semantic clarity (actions and states are different things) for **mathematical simplicity** - one unified Projector class that handles all constraint types simultaneously.

This is optimal for quadratic programming-based optimization but differs from traditional MPC approaches that handle constraints on actions and states separately.

---

## References

| Component | File | Key Lines |
|-----------|------|-----------|
| Projector API | [flow_matcher_v3_ode_selectable/sampling/projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | 46-100 |
| SafetyConstraints | [flow_matcher_v3_ode_selectable/sampling/projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | 249-350 |
| DynamicConstraints | [flow_matcher_v3_ode_selectable/sampling/projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | 361-410 |
| ObstacleConstraints | [flow_matcher_v3_ode_selectable/sampling/projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | 427-470 |
| FMv3 Eval | [FM_v3_test/eval_FM_v3.py](FM_v3_test/eval_FM_v3.py) | 50-250 |
| FMv3 Config | [config/projection_eval.yaml](config/projection_eval.yaml) | 1-100 |
| Gen6 Setup | [fm_encdec_vision_test/eval_fm_encdec_vision.py](fm_encdec_vision_test/eval_fm_encdec_vision.py) | 80-115 |
