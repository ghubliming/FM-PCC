# Direct Answers to Initial Questions
## FMv3ODE Implementation Analysis

**Created:** 2026-05-18  
**Based on:** Code analysis of FMv3ODE, FMv3, and Gen6 implementations

---

## Question 1: How does FMv3ODE formulate constraint projection with state-action dimensions?

### Answer

FMv3ODE uses a **unified vector representation** of the constraint projection problem:

```python
# From flow_matcher_v3_ode_selectable/sampling/projection.py (lines 46-100)

class Projector:
    def __init__(self, horizon, transition_dim, action_dim=0, goal_dim=0, 
                 constraint_list=[], normalizer=None, variant='states', ...):
        
        self.transition_dim = transition_dim  # Total: action_dim + obs_dim
        self.action_dim = action_dim          # How many of these are actions
```

**Formulation:**
- **Combined vector:** $z = [s_0, s_1, ..., s_{H-1}]$ where each $s_t$ is 6D
- **Optimization problem:**
  $$\min_{z} \frac{1}{2} z^T Q z + r^T z$$
  $$\text{subject to: } Az = b \quad \text{(dynamics)}$$
  $$Cz \leq d \quad \text{(bounds, halfspace)}$$
  $$s_t^T P s_t + q^T s_t \leq v \quad \text{(obstacles)}$$

- **Dimensions in z:** Actions occupy indices [0, 1, 2], states occupy indices [3, 4, 5]

**Key insight:** State-action coupling is achieved by:
1. **Explicit Euler constraints** linking dimension 3 to dimension 0 (position to velocity)
2. **Single flattened optimization** over all dimensions simultaneously

---

## Question 2: Does FMv3ODE use a 6D [action, obs] representation or different?

### Answer

**YES, FMv3ODE DOES use 6D [action, obs] representation.**

### Evidence from FMv3 test evaluation:

```python
# From FM_v3_test/eval_FM_v3.py (lines 66-72)

if fm_model.__class__.__name__ == 'GaussianDiffusion':
    trajectory_dim = fm_model.transition_dim - fm_model.goal_dim
    action_dim = fm_model.action_dim
    fm_variant = 'states_actions'
    obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
    act_obs_indices = {**act_indices, **obs_indices_updated}
```

### The 6D structure:

| Dimension | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|----|-----|-----|
| **Type** | Action | Action | Action | State | State | State |
| **Semantic** | $v_x$ | $v_y$ | $v_z$ | $x$ | $y$ | $z$ |

### Projection setup:

```python
projector = Projector(
    horizon=8,
    transition_dim=6,        # 3D actions + 3D states = 6D
    action_dim=3,            # First 3 dimensions are actions
    goal_dim=0,
    constraint_list=constraints,
    normalizer=dataset.normalizer,
    variant='states_actions',  # ← Signals concatenated representation
    dt=0.02,
    device='cuda'
)
```

**Comparison matrix:**

| Version | Uses 6D? | transition_dim Composition | variant |
|---------|----------|---------------------------|---------|
| **FMv2** | ❌ | obs_dim only | 'states' |
| **FMv3** | ❌ | obs_dim only | 'states' |
| **FMv3ODE** | ✅ | action_dim + obs_dim | 'states_actions' |
| **Gen6 DPCC** | ✅ | Same as FMv3ODE | 'states_actions' |

---

## Question 3: How does FMv3ODE handle the "transition_dim" parameter?

### Answer

The `transition_dim` is **split across two purposes:**

### 3.1 In the Projector Constructor

```python
class Projector:
    def __init__(self, horizon, transition_dim, action_dim=0, ...):
        self.transition_dim = transition_dim  # Total size: 6D
        self.action_dim = action_dim          # Subset: 3D
```

**Internal interpretation:**
- First `action_dim` dimensions = actions (e.g., [0, 1, 2])
- Remaining `(transition_dim - action_dim)` dimensions = observations (e.g., [3, 4, 5])

### 3.2 In SafetyConstraints

```python
# From flow_matcher_v3_ode_selectable/sampling/projection.py (lines 249-350)

class SafetyConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        for constraint in constraint_list:
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):  # ← Iterate over all 6 dimensions
                    # ...
                    if self.skip_initial_state and dim >= self.action_dim:
                        # For state dimensions (dim >= 3), skip t=0
                        mat_append = mat_append[1:]  # Remove time 0
```

**Decision logic:**
- **Action bounds (dim < action_dim):** Applied at all times t=0..H-1
- **State bounds (dim >= action_dim):** Applied only at t=1..H-1 (initial state fixed)

### 3.3 In DynamicConstraints

```python
# From flow_matcher_v3_ode_selectable/sampling/projection.py (lines 361-410)

class DynamicConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        # Example: ('deriv', [3, 0])  means: x[t+1] = x[t] + dt * vx[t]
        x_idx = 3  # State dimension (position)
        dx_idx = 0  # Action dimension (velocity)
        
        # Constraint matrix entry:
        mat_append[t, t * transition_dim + x_idx] = 1      # x[t] coefficient
        mat_append[t, t * transition_dim + dx_idx] = self.dt  # dt * vx[t] coefficient
        mat_append[t, (t+1) * transition_dim + x_idx] = -1    # -x[t+1] coefficient
        # Result: x[t+1] - x[t] - dt*vx[t] = 0
```

**Coupling mechanism:** Actions (low indices) directly affect state evolution (high indices) via explicit Euler.

### 3.4 In ObstacleConstraints

```python
# From flow_matcher_v3_ode_selectable/sampling/projection.py (lines 427-470)

class ObstacleConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        P = np.zeros((self.transition_dim, self.transition_dim))  # 6×6 matrix
        q = np.zeros(self.transition_dim)  # 6D vector
        
        for dim in dims:  # Only dimensions in the constraint (e.g., [3, 4] for x,y)
            P[dim, dim] = ...
            q[dim] = ...
```

**Key point:** Only state dimensions (3, 4) get nonzero entries in P and q; action dimensions (0, 1, 2) remain zero.

---

## Question 4: Constraint formulation—are bounds applied to both actions and states?

### Answer

**Depends on the constraint type:**

### 4.1 Bounds Constraints ('lb'/'ub')

**Configuration-dependent:**

```yaml
# From config/projection_eval.yaml (lines 73-77)
bounds:
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
  ]
```

**Interpretation:**
- Bounds specified **only for actions** (vx, vy)
- These map to dimensions [0, 1] in the 6D representation

**Application:**
```python
# All timesteps t=0..H-1 get these bounds
if self.skip_initial_state and dim >= self.action_dim:
    # This condition is FALSE for action dimensions (0, 1 < 3)
    # So action bounds are NOT skipped at t=0
```

**Result:** Action bounds are enforced at **all timesteps including t=0**.

### 4.2 Dynamic Constraints ('deriv')

**Both action and state:**

```python
# From config/projection_eval.yaml (implied from formulate_dynamics_constraints)
dynamic_constraints = [
    ('deriv', [act_obs_indices['x'], act_obs_indices['vx']]),  # (3, 0)
    ('deriv', [act_obs_indices['y'], act_obs_indices['vy']]),  # (4, 1)
    ('deriv', [act_obs_indices['z'], act_obs_indices['vz']]),  # (5, 2)
]
```

**Constraint:** Actions **drive** state evolution via Explicit Euler
$$x[t+1] - x[t] - \Delta t \cdot v_x[t] = 0$$

### 4.3 Obstacle Constraints ('sphere_outside')

**State dimensions only:**

```python
# From config/projection_eval.yaml (lines 60-75)
obstacle_constraints:
  'avoiding-d3il': [
    {'type': 'sphere_outside', 'dimensions': ['x', 'y'], 
     'center': [0.4, 0.08], 'radius': 0.06},
  ]
```

**Mapping:**
- 'x' → dimension 3 (state position)
- 'y' → dimension 4 (state position)
- **Actions (0, 1, 2) NOT included**

**Constraint:** Keep state position outside obstacle
$$(x - c_x)^2 + (y - c_y)^2 \geq r^2$$

### Summary Table

| Constraint Type | Actions | States | Timesteps |
|---|---|---|---|
| **Bounds** | ✅ Bounded | ❌ Unbounded | t=0..H-1 |
| **Dynamics** | ✅ Couples to | ✅ Follows from | t=0..H-1 |
| **Obstacles** | ❌ Ignored | ✅ Avoided | t=1..H-1 |
| **Halfspace** | ❌ Ignored | ✅ Polytopic | t=1..H-1 |

---

## Question 5: How does the original working baseline in FMv3ODE obstacle avoidance work?

### Answer

**Three-stage pipeline: Sample → Project → Select**

### 5.1 Stage 1: Diffusion Sampling

```python
# From FM_v3_test/eval_FM_v3.py (lines 150-170)
policy = Policy(
    model=fm_model,
    normalizer=dataset.normalizer,
    projector=projector,
    trajectory_selection='minimum_projection_cost'  # or other variants
)

# Call policy to generate samples
action, samples = policy(
    conditions={0: obs},        # Condition on current observation
    batch_size=4,               # Generate 4 trajectory hypotheses
    horizon=8,                  # 8-step trajectories
    disable_projection=False    # Enable constraint projection
)
```

**Output:** 4 different 8-step trajectories, each with random noise from diffusion.

### 5.2 Stage 2: Constraint Projection

**For each of the 4 samples:**

```python
# Pseudo-code for projector.project(trajectory)
# From flow_matcher_v3_ode_selectable/sampling/projection.py (lines 70-130)

def project(self, trajectory):
    """
    Solve: min 0.5 * z^T Q z + r^T z
           s.t. A z = b         (dynamics)
                C z <= d        (bounds, halfspace)
                q^T P q <= v    (obstacles)
    """
    
    # Use scipy.minimize with SLSQP method
    res = minimize(
        fun=cost_fun,
        x0=trajectory,           # Start from diffusion sample
        constraints=constraints,  # Dynamics + bounds + obstacles
        method='SLSQP',
        jac=jac_cost_fun,
        tol=1e-6
    )
    return projected_trajectory
```

**Why this works:**
1. **Warm start:** Diffusion sample is already relatively good (close to expert distribution)
2. **Minimal correction:** Cost function $\frac{1}{2}||z - z_{\text{diffusion}}||^2$ keeps solution close to original
3. **Constraint satisfaction:** SLSQP enforces all constraints exactly at convergence

### 5.3 Stage 3: Trajectory Selection

```python
# From evaluation (implied from variants)

trajectory_selection = 'minimum_projection_cost'  # Or 'temporal_consistency'

if trajectory_selection == 'minimum_projection_cost':
    # Choose the projection with smallest cost
    best_idx = argmin([projection_cost_i for i in range(batch_size)])
    selected_trajectory = projected_trajectories[best_idx]
    
elif trajectory_selection == 'temporal_consistency':
    # Reward trajectories that change slowly over time
    smoothness = sum(||s[t+1] - s[t]||^2 for t in range(H-1))
    best_idx = argmin(smoothness)
    selected_trajectory = projected_trajectories[best_idx]
```

### 5.4 Full Evaluation Loop

```python
# From FM_v3_test/eval_FM_v3.py (lines 160-220)

for episode in range(n_trials):
    obs = env.reset()
    
    for timestep in range(max_episode_length):
        # 1. Validate current state
        if is_in_obstacle(obs):
            collision_free = False
            break
        
        # 2. Generate action via policy (sample → project → select)
        action, samples = policy(
            conditions={0: obs},
            batch_size=4,
            horizon=8,
            disable_projection=False
        )
        
        # 3. Execute in environment
        obs_next = env.step(action)  # Use first planned action
        
        # 4. Check success
        if obs_next == goal:
            success = True
            break
        
        obs = obs_next
    
    # 5. Record metrics
    if success and collision_free:
        success_and_constraints += 1
```

### 5.5 Success Metrics

```python
# From FM_v3_test/eval_FM_v3.py (computed metrics)

n_success[i]                    # Did agent reach goal?
collision_free_completed[i]     # Did agent avoid all obstacles?
n_success_and_constraints[i]    # BOTH success AND collision-free?
total_violations[i]             # Sum of constraint violation magnitudes
n_violations[i]                 # Count of constraint violations
avg_time[i]                     # Computation time per step
```

### Why This Baseline Works

| Factor | Mechanism | Result |
|--------|-----------|--------|
| **Multiple hypotheses** | Batch size = 4 | Probability at least one is feasible |
| **Proper normalization** | Constraints in [-1, 1] space | Match diffusion training distribution |
| **Gradient-based search** | SLSQP optimization | Fine-tune infeasible samples to feasible |
| **Causal dynamics** | Explicit Euler coupling | Actions naturally influence state trajectory |
| **Current state validation** | Check before policy call | Never in impossibly infeasible state |
| **Warm starting** | Use diffusion as initialization | Few SLSQP iterations needed |

---

## Conclusion: Is the "6D [action, obs]" Inherited from FMv3ODE?

### ✅ YES, DEFINITIVELY

**Evidence chain:**
1. **FMv3** (v3_test) uses `transition_dim = obs_dim` only, `action_dim = 0`
2. **FMv3ODE** (v3_ode_selectable) introduces `transition_dim = action_dim + obs_dim`, `variant='states_actions'`
3. **Gen6 DPCC** (fm_encdec_vision_test) uses identical Projector API with same 6D configuration

**The 6D design is NOT Gen6-specific; it's a FMv3ODE design choice adopted downstream.**

The three key decisions that make 6D work:
1. ✅ **Index mapping:** Actions at [0,1,2], states at [3,4,5]
2. ✅ **SafetyConstraints:** Smart handling of action vs. state bounds
3. ✅ **DynamicConstraints:** Explicit Euler couples dimensions within 6D vector

All three are inherited unchanged by Gen6.
