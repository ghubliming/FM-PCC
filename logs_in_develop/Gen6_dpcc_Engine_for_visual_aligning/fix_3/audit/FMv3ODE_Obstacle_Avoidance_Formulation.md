# FMv3ODE Obstacle Avoidance: Constraint Design & Formulation

**Date:** 2026-05-18  
**Focus:** How obstacle constraints are formulated, optimized, and why they work

---

## 1. FMv3ODE Obstacle Avoidance Formulation

### 1.1 Constraint Mathematical Form

**Goal:** Keep planned trajectory **outside** circular obstacle regions

**Quadratic Constraint Class:**
$$s_t^T P s_t + q^T s_t - v \leq 0$$

Where:
- $s_t$ = full state-action vector at timestep $t$ (6D: [vx, vy, vz, x, y, z])
- $P$ = diagonal matrix (only for dimensions in obstacle constraint)
- $q$ = linear offset vector
- $v$ = threshold value derived from radius

**Source Code:** [flow_matcher_v3_ode_selectable/sampling/projection.py#L427-470](flow_matcher_v3_ode_selectable/sampling/projection.py#L427-470)

### 1.2 Obstacle Constraints Build Process

```python
class ObstacleConstraints(Constraints):
    def build_matrices(self, constraint_list=None):
        self.P_list = []
        self.q_list = []
        self.v_list = []
        
        for constraint in constraint_list:
            type = constraint[0]              # 'sphere_outside' or 'sphere_inside'
            dims = constraint[1]              # [3, 4] for (x, y) coordinates
            center = constraint[2]            # Obstacle center [cx, cy]
            radius = constraint[3]            # Obstacle radius r
            
            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius ** 2
            
            # Build P and q for each dimension in the constraint
            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    
                    # Normalized space formulation
                    P[dim, dim] = delta_s ** 2 / 4
                    q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + \
                         (s_min - center[dim_counter]) ** 2
```

### 1.3 Normalized vs. Unnormalized Obstacle Constraints

#### Unnormalized Form (No Normalizer)

**Input:**  
- Position state: $s = [x, y]$ (unnormalized)  
- Obstacle center: $[c_x, c_y]$  
- Radius: $r$

**Constraint Matrix:**
$$P = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad q = \begin{bmatrix} -2c_x \\ -2c_y \end{bmatrix}, \quad v = r^2 - c_x^2 - c_y^2$$

**Interpretation:** Standard Euclidean distance
$$(x - c_x)^2 + (y - c_y)^2 \leq r^2$$

#### Normalized Form (With Normalizer)

**Transform:** $s_n = \frac{2(s - s_{min})}{s_{max} - s_{min}} - 1$ (maps to [-1, 1])

**Inverse:** $s = \frac{(s_n + 1)(s_{max} - s_{min})}{2} + s_{min}$

**Constraint in Normalized Space:**
$$s_n^T P s_n + q^T s_n \leq v$$

Where:
$$P_{dim,dim} = \frac{\Delta s^2}{4}, \quad \Delta s = s_{max} - s_{min}$$

$$q_{dim} = \frac{\Delta s^2}{2} + \Delta s(s_{min} - c_{dim})$$

$$v = r^2 - \sum_{dim} \left[ \frac{\Delta s^2}{4} + \Delta s(s_{min} - c_{dim}) + (s_{min} - c_{dim})^2 \right]$$

**Why normalization?** Constraints are expressed in the model's normalized state space [-1, 1], matching the trained diffusion model's distribution.

### 1.4 "Sphere Inside" vs. "Sphere Outside"

```python
if type == 'sphere_outside':
    P = -P
    q = -q
    v = -v
```

**Sphere Outside (avoiding obstacles):**
- Constraint: $s^T P s + q^T s \leq v$ (as written above)
- Interpretation: $(x - c_x)^2 + (y - c_y)^2 \geq r^2$ (stay outside)

**Sphere Inside (reaching target regions):**
- P, q, v negated: $-s^T P s - q^T s \leq -v$
- Equivalent to: $(x - c_x)^2 + (y - c_y)^2 \leq r^2$ (stay inside)

---

## 2. Obstacle Constraint Application in Optimization

### 2.1 The Projection Optimization Problem

[flow_matcher_v3_ode_selectable/sampling/projection.py#L70-120](flow_matcher_v3_ode_selectable/sampling/projection.py#L70-120)

```python
def project(self, trajectory, constraints=None):
    """
    Solve: minimize 0.5 * z^T Q z + r^T z
           subject to: A z = b              (dynamics)
                       C z <= d             (bounds, halfspace)
                       s_t^T P_k s_t + q_k^T s_t <= v_k  (obstacles)
    
    where z = [s_0, s_1, ..., s_{H-1}] is the flattened trajectory
    """
    
    # Build obstacle constraints for all timesteps
    constraints = ()
    for constraint_idx in range(len(self.obstacle_constraints.P_list)):
        P = self.obstacle_constraints.P_list[constraint_idx]
        q = self.obstacle_constraints.q_list[constraint_idx]
        v = self.obstacle_constraints.v_list[constraint_idx]
        
        for t in range(1, self.horizon):  # ← Skip t=0 (current state observed)
            start_idx = t * self.transition_dim
            end_idx = (t + 1) * self.transition_dim
            
            # Quadratic function: -s_t^T P s_t - q^T s_t + v
            constraints += ({
                'type': 'ineq',
                'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: \
                    -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - \
                     q @ x[start_idx: end_idx] + v,
                'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: \
                    np.concatenate([
                        np.zeros(start_idx),
                        -2 * P @ x[start_idx: end_idx] - q,  # Gradient
                        np.zeros(len(x) - end_idx)
                    ])
            },)
```

### 2.2 Why t=1 to t=H-1?

```python
for t in range(1, self.horizon):  # Not 0!
```

**Reason:** 
- $s_0$ = current observed state (boundary condition, not a decision variable)
- $s_{\text{now}}$ is already outside obstacles (enforced in evaluation loop)
- Optimization focuses on **future** states $s_1, ..., s_{H-1}$

---

## 3. How the Original Working Baseline Operates

### 3.1 FMv3ODE Obstacle Avoidance Pipeline

[FM_v3_test/eval_FM_v3.py#L150-220](FM_v3_test/eval_FM_v3.py#L150-220)

```python
for _ in range(args.max_episode_length):
    # 1. Check if current state violates constraints
    if 'obstacles' in constraint_types:
        for constraint in obstacle_constraints:
            if np.linalg.norm(obs[[obs_indices['x'], obs_indices['y']]] - 
                            constraint['center']) < constraint['radius']:
                violated_this_timestep = 1
                collision_free_completed[i] = 0  # Failure flag
    
    # 2. Sample trajectory from diffusion + projection
    action, samples = policy(
        conditions={0: obs},
        batch_size=args.batch_size,
        horizon=args.horizon,
        disable_projection=disable_projection
    )
    
    # 3. Take action in environment
    obs, rew, terminated, truncated, info = env.step(action)
    
    # 4. Check success
    if success: n_success[i] = 1
    if success and collision_free_completed[i]: n_success_and_constraints[i] = 1
```

### 3.2 Policy Class Workflow

[flow_matcher_v3_ode_selectable/sampling/policies.py](flow_matcher_v3_ode_selectable/sampling/policies.py) (referenced in eval):

**Expected Flow:**
1. **Diffusion sampling:** Generate $K$ trajectory hypotheses (batch_size samples)
2. **Constraint projection:** Project each trajectory to satisfy obstacles
3. **Trajectory selection:** Choose best trajectory by:
   - `'diffuser'`: No projection, random sample
   - `'dpcc-c'`: Minimum projection cost
   - `'dpcc-t'`: Temporal consistency
   - `'dpcc-r'`: Post-hoc repair (reoptimize violated constraints)

### 3.3 Configuration for Avoiding Task

[config/projection_eval.yaml#L1-80](config/projection_eval.yaml#L1-80)

```yaml
exps: ['avoiding-d3il']
constraint_types: ['halfspace', 'obstacles', 'dynamics', 'bounds']

obstacle_constraints:
  'avoiding-d3il': [
    {'type': 'sphere_outside', 'dimensions': ['x', 'y'], 
     'center': [0.4, 0.08], 'radius': 0.06},
    {'type': 'sphere_outside', 'dimensions': ['x', 'y'], 
     'center': [0.6, 0.08], 'radius': 0.06},
  ]

bounds:
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
  ]
```

**Baseline Settings:**
- **2 obstacles** with radius 0.06
- **Action bounds** [-0.01, 0.01] for velocity
- **Halfspace constraints** for polytopic obstacles
- **Dynamics** linking actions to state evolution

---

## 4. Gradient-Based Obstacle Constraint Refinement

### 4.1 Gradient Computation

[flow_matcher_v3_ode_selectable/sampling/projection.py#L480-510](flow_matcher_v3_ode_selectable/sampling/projection.py#L480-510)

```python
def compute_gradient(self, trajectory, constraints=None):
    """
    Compute weighted gradients for obstacle constraint violations
    """
    grad3 = np.zeros_like(trajectory_np)  # Initialize zero gradient
    
    for constraint_idx in range(len(self.obstacle_constraints.P_list)):
        P = self.obstacle_constraints.P_list[constraint_idx]
        q = self.obstacle_constraints.q_list[constraint_idx]
        v = self.obstacle_constraints.v_list[constraint_idx]
        
        for t in range(1, self.horizon):
            start_idx = t * self.transition_dim
            end_idx = (t + 1) * self.transition_dim
            
            for i in range(trajectory.shape[0]):
                # Check if constraint is violated
                violation = trajectory_np[i, start_idx: end_idx] @ P @ \
                           trajectory_np[i, start_idx: end_idx] + \
                           q @ trajectory_np[i, start_idx: end_idx] - v
                
                if violation > 0:  # Violated!
                    # Compute gradient at violating state
                    grad3[i, start_idx: end_idx] -= \
                        2 * P @ trajectory_np[i, start_idx: end_idx] + q
```

**Gradient Form:**
$$\nabla_{s_t} (s_t^T P s_t + q^T s_t) = 2 P s_t + q$$

**Direction:** Points away from constraint satisfaction toward feasible region.

---

## 5. Why the FMv3ODE Baseline Works

### 5.1 Key Success Factors

| Factor | Mechanism | Impact |
|--------|-----------|--------|
| **Proper Normalization** | Constraints expressed in [-1, 1] space matching diffusion training | Realistic obstacle geometry in model space |
| **Skip t=0 Constraint** | Current state observed & validated before policy call | No infeasible initial conditions |
| **Quadratic Obstacle Form** | $(x - c_x)^2 + (y - c_y)^2 \leq r^2$ is exact Euclidean geometry | Geometrically precise avoidance |
| **Gradient Refinement** | Optional trajectory repair via gradient descent | Can fix near-miss constraint violations |
| **Trajectory Batch Selection** | Multiple samples → choose best by projection cost | Redundancy + optimization work together |
| **Action Bounds** | Limit velocity changes | Prevents unrealistic aggressive maneuvers |
| **Explicit Euler Dynamics** | State evolution tied to action dimensions | Physically grounded constraints |

### 5.2 Baseline Performance Metrics

From evaluation code:
- **Success rate:** $P(\text{goal reached})$
- **Collision-free completion:** $P(\text{no constraint violations})$
- **Success + constraints:** $P(\text{goal AND no violations})$
- **Constraint violation count:** Total per episode
- **Avg projection cost:** Measure of how much correction was needed

---

## 6. Potential Issues with Original Design

### 6.1 Known Limitations

1. **Non-convex optimization**
   - Quadratic constraints are **non-convex** for obstacle avoidance
   - SLSQP solver may find local minima
   - No guarantee of global optimality

2. **Batch coupling**
   - Projection solves independently for each sample
   - No cross-sample information sharing

3. **Gradient computation efficiency**
   - Evaluates constraint violation per sample per timestep
   - O(batch_size × horizon) operations

4. **Temporal consistency**
   - No explicit smoothness penalty on trajectory
   - Can produce jerky plans

### 6.2 How Gen6 Addressed These

- Added **'dpcc-t'** (temporal consistency) trajectory selection
- Introduced **cost function weighting** for different constraint types
- Enabled **'post_processing'** trajectory repair

---

## 7. Code Cross-Reference Summary

| Aspect | Implementation File | Key Function | Lines |
|--------|---|---|---|
| Obstacle matrix build | [projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | `ObstacleConstraints.build_matrices()` | 427-470 |
| Constraint application | [projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | `Projector.project()` | 100-150 |
| Gradient correction | [projection.py](flow_matcher_v3_ode_selectable/sampling/projection.py) | `Projector.compute_gradient()` | 480-510 |
| Runtime validation | [eval_FM_v3.py](FM_v3_test/eval_FM_v3.py) | Main loop | 180-220 |
| Configuration | [projection_eval.yaml](config/projection_eval.yaml) | Obstacle definitions | 60-75 |
| Policy interface | [policies.py](flow_matcher_v3_ode_selectable/sampling/policies.py) | Policy class | (inferred) |

---

## Conclusion

The **FMv3ODE obstacle avoidance mechanism is a well-designed quadratic programming formulation** that:

1. ✅ Precisely models circular obstacles via $(x-c_x)^2 + (y-c_y)^2 \leq r^2$
2. ✅ Properly normalizes to match diffusion model training space
3. ✅ Integrates constraints with state-action dynamics via Euler discretization
4. ✅ Provides gradient-based refinement for near-miss violations
5. ✅ Leverages batch sampling for redundancy

**Gen6 inherits this entire infrastructure unchanged**, only modifying the specific constraint bounds (from action limits to workspace limits) and adding trajectory selection heuristics.
