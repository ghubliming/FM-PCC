# Gen6 DPCC Engine - Problems & Recommendations

---

## 🔴 Priority 1: Incomplete Candidate Selection Feature

### Problem
- **Location**: [eval_ddpm_encdec_vision.py](../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py#L674-L676) + [eval_fm_encdec_vision.py](../../fm_encdec_vision_test/eval_fm_encdec_vision.py#L674-L676)
- **What**: Code hardcodes `batch_size = 6` (line 674-676), generating 6 independent trajectory candidates
- **Why**: Computational waste — all 6 are generated but only the first is used by default
- **Root Cause**: Selection logic for `'minimum_projection_cost'` is gated but cost dictionary never populated

### Why It's a Problem
1. **Computational**: 6x diffusion forward passes per action, but only 1 trajectory selected
2. **Incomplete Feature**: The infrastructure for trajectory selection exists but isn't functional
3. **No Diagnostic Info**: No logging of which trajectory was selected or why

### Recommended Fix
```python
# Option A: Complete Cost-Based Selection
# Ensure Projector.project() returns cost metrics in infos dict
# Then validation in eval_ddpm_encdec_vision.py:200-210 will work

# Option B: Make Temporal Consistency Primary Selection
if self.batch_size > 1:
    # Use temporal_consistency as reliable fallback (already works)
    if self.trajectory_selection == 'temporal_consistency' and self.prev_observations is not None:
        diffs = trajectories_np - np.expand_dims(self.prev_observations, axis=0)
        order = np.argsort(np.linalg.norm(diffs, axis=(1, 2)))
        which_trajectory = order[0]  # ✅ This is working
    else:
        which_trajectory = 0  # First trajectory fallback
else:
    which_trajectory = 0

# Add logging:
    logging.info(f"Selected trajectory {which_trajectory} via {self.trajectory_selection}")
```

---

## 🟡 Priority 2: Missing Diagnostic Instrumentation

### Problem
- **Location**: [eval_ddpm_encdec_vision.py](../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py#L464-L520) (get_action method)
- **What**: No logging for trajectory selection decisions during rollout
- **Why**: Can't diagnose whether trajectories are being selected well or identify selection failures

### Why It's a Problem
1. **Debugging**: When trajectory selection uses cost metrics, silent failures leave no trace
2. **Analysis**: Can't measure trajectory quality or identify when cost-based selection triggered
3. **Data**: No metrics to verify multi-candidate strategy is effective

### Recommended Fix
```python
# In get_action() method, add logging after selection:

if self.batch_size > 1:
    selection_method = 'first_only'  # default
    sel_details = {}
    
    if (self.trajectory_selection == 'temporal_consistency' and 
        self.prev_observations is not None):
        diffs = trajectories_np - np.expand_dims(self.prev_observations, axis=0)
        order = np.argsort(np.linalg.norm(diffs, axis=(1, 2)))
        which_trajectory = order[0]
        selection_method = 'temporal_consistency'
        sel_details = {'distance_to_prev': float(np.linalg.norm(diffs[which_trajectory]))}
        
    elif (self.trajectory_selection == 'minimum_projection_cost' and 
          self.projector is not None and infos is not None and 
          'projection_costs' in infos):
        costs_total = np.zeros(self.batch_size)
        for timestep, cost in infos['projection_costs'].items():
            costs_total += cost
        if len(costs_total) == self.batch_size:
            which_trajectory = np.argmin(costs_total)
            selection_method = 'minimum_projection_cost'
            sel_details = {'projection_cost': float(costs_total[which_trajectory])}
    
    # Log selection decision
    self.logger.info(f"Trajectory selection: method={selection_method}, "
                     f"idx={which_trajectory}/{self.batch_size}, "
                     f"details={sel_details}")
else:
    which_trajectory = 0
```

---

## 🟠 Priority 3: State-Space Documentation Gap

### Problem
- **Location**: [diffuser/sampling/projection.py](../../diffuser/sampling/projection.py) (entire file)
- **What**: No docstring explaining whether constraints operate in scaled or unscaled coordinates
- **Why**: Future maintainers won't know if Euler constraints should use scaled or raw space

### Why It's a Problem
1. **Maintainability**: Ambiguous coordinate frame assumptions
2. **Correctness Risk**: If someone modifies constraint logic, they may use wrong coordinate space
3. **Integration**: Unclear contract between diffusion model (scaled) and projector (?)

### Recommended Fix
```python
# Add module-level docstring to projection.py:

"""
Differentiable Projective Control Constraint (DPCC) Engine

COORDINATE FRAME CONVENTION
----------------------------
All inputs and outputs use SCALED coordinates (matching training data scale).

Input:
  τ_raw ∈ ℝ^(H × d)  : Trajectory from diffusion model (SCALED via scaler)
  
Output:
  τ_proj ∈ ℝ^(H × d) : Constrained trajectory (same scale as input, SCALED)

Constraints Applied:
  - Workspace bounds: Applied in SCALED space (matching constraint_list specification)
  - Euler dynamics: x[t+1] = x[t] + dt * vx[t]  (SCALED space, dims match input)
  - Obstacle avoidance: In SCALED visual feature space (if used)

Why Scaled?
  Training data scaled via Z-score normalization (scaler.py)
  Constraints specified as scaled bounds (e.g., [-3, +3] std)
  Projector must maintain scaling for QP solver numerical stability

State-Action Structure (6D case):
  dims [0, 1, 2]: Action (Cartesian velocity, scaled)
  dims [3, 4, 5]: Observation (Absolute EE position, scaled)
  
Assumptions Maintained:
  ✓ Input τ_raw is already scaled
  ✓ Bounds in constraint_list are pre-scaled
  ✓ QP solver operates in scaled space
  ✓ Output τ_proj needs no additional scaling
"""
```

---

## 🟠 Priority 3b: Parameter Naming Clarity

### Problem
- **Location**: [eval_fm_encdec_vision.py](../../fm_encdec_vision_test/eval_fm_encdec_vision.py#L70)
- **What**: Parameter called `enlarge_constraints` but actually shrinks the workspace
- **Why**: Semantically reversed name causes confusion

### Why It's a Problem
```python
enlarge_constraints = config.get('enlarge_constraints', 0.0)

if 'tightened' in variant and enlarge_constraints > 0.0:
    workspace_lb += enlarge_constraints     # Actually shrinks!
    workspace_ub -= enlarge_constraints     # Actually shrinks!
```

- Config parameter name contradicts its actual effect (enlarges → shrinks)
- Maintainers may misinterpret the parameter's purpose

### Recommended Fix
```yaml
# In config/aligning-d3il-visual.yaml:
constraint_tightening_margin: 0.05  # meters to contract workspace bounds
```

```python
# In eval_fm_encdec_vision.py:
tightening_margin = config.get('constraint_tightening_margin', 0.0)

if 'tightened' in variant and tightening_margin > 0.0:
    workspace_lb += tightening_margin    # Now semantically clear: tighten = shrink bounds
    workspace_ub -= tightening_margin
```

---

## Summary Table

| Priority | Problem | Impact | Fix Effort |
|----------|---------|--------|-----------|
| **P1** | Incomplete candidate selection | Computational waste (6x cost for 1 output) | Medium (complete selection logic) |
| **P2** | No diagnostic logging | Can't debug trajectory selection | Low (add logging) |
| **P3** | State-space documentation gap | Risk of future maintainability issues | Low (add docstrings) |
| **P3b** | Parameter naming reversed | Confusion during configuration | Low (rename parameter) |

---

**All architecture is correct.** These are implementation completeness and clarity issues, not correctness bugs.
