# Constraint Satisfaction / Violation Metrics — Reference Guide

**Introduced**: UF-16.3 (2026-05-27)
**Applies to**: `eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`
**Changelog**: [CHANGELOG_UF16_3.md](CHANGELOG_UF16_3.md)

---

## Overview

Two measurement levels are distinguished throughout this guide:

| Level | Symbol prefix | What is checked | Data source |
|---|---|---|---|
| **Execution** | `exec_*` | Actual EE positions the robot visited during execution | `c_pos_history (T, 3)` in metres |
| **Planning** | `plan_*` | Post-projection planned trajectories at each MPC replan step | `all_candidates (B, H, 3)` unnormalised |

Execution metrics answer *"was the real trajectory safe?"*.
Planning metrics answer *"did the projector succeed in making planned trajectories feasible?"*.

---

## Coordinate frame and units

All metrics are computed in the **physical Franka EE Cartesian frame** (metres):

| Axis | Direction | Range (nominal scene) |
|---|---|---|
| `x` | Forward from robot base | 0.30 – 0.70 m |
| `y` | Lateral (left = positive) | −0.35 – 0.35 m |
| `z` | Vertical (up) | 0.05 – 0.40 m |

Tightening (`enlarge_constraints`) shifts constraint boundaries inward by the
configured margin (default 0.01 m).  All metrics respect this margin — bounds are
tightened `lb += enlarge` / `ub -= enlarge`, obstacles grow by `enlarge`.

---

## Execution metrics (`exec_*`)

These are computed per rollout by `check_trajectory_constraints()` over the full
`c_pos_history (T, 3)` trajectory.

### Core counts

#### `exec_n_steps` — int
Total trajectory length T.  Denominator for all rate metrics.

#### `exec_n_violated_steps` — int
Number of timesteps where **any** active constraint is violated.
```
violated[t] = bounds_violated[t] OR halfspace_violated[t] OR obstacle_violated[t]
```

#### `exec_constraint_sat_rate` — float ∈ [0, 1]
```
1 - exec_n_violated_steps / exec_n_steps
```
The primary safety KPI.  1.0 = perfectly constraint-satisfying trajectory.
Analogous to `success_rate` but measures geometric safety, not task completion.

#### `exec_zero_violation_rollout` — bool
True if `exec_n_violated_steps == 0`.  Useful as a binary safety flag — a rollout
either never left the feasible region or it did.  Aggregate: `exec_zero_violation_rollouts / n_rollouts`.

---

### Per-constraint-type violation counts

#### `exec_bounds_viol_count` — int
Steps where `c_pos[t]` is outside the workspace bounding box `[lb, ub]` (after
tightening).  Checks all three dimensions simultaneously; a step is counted if
*any* dimension violates.

#### `exec_halfspace_viol_count` — int
Steps where `c_pos[t, xy]` is on the infeasible side of any halfspace boundary.
For each halfspace, the signed distance from the boundary is computed:
```
sd[t] = n · (c_pos[t, :2] − p0)
```
where `n` is the unit normal pointing to the feasible side.  `sd[t] < 0` → violated.

#### `exec_obstacle_viol_count` — int
Steps where `c_pos[t]` is **inside** any obstacle exclusion sphere:
```
penetration[t] = max(0,  radius − ||c_pos[t, obs_dims] − center||)
```
Penetration > 0 → violated.  For 2D obstacles (`dimensions: ['x','y']`), only the
xy-plane distance is checked; z is unconstrained.

---

### Violation magnitudes

#### `exec_max_bounds_viol_m` — float (metres)
Maximum distance outside the workspace bounds across all timesteps and dimensions:
```
max over t,i of max(0, lb[i] − c_pos[t,i], c_pos[t,i] − ub[i])
```
0.0 if bounds are never violated.  Non-zero tells you how far the robot escaped.

#### `exec_max_halfspace_viol_m` — float (metres)
Maximum signed-distance violation magnitude across all halfspace constraints and
timesteps.  Geometrically: how far (in metres) was the EE on the wrong side of the
boundary line at the worst timestep.

#### `exec_max_obstacle_penetration_m` — float (metres)
Maximum penetration depth into any obstacle sphere.  Equal to
`max(0, radius − dist_to_center)`.  For a sphere of radius 0.06 m, a value of
0.02 m means the EE got within 0.04 m of the obstacle centre.

---

### Safety margin (innovative)

#### `exec_constraint_margin_mean_m` — float (metres)
Mean **minimum distance to the nearest active constraint boundary** at all
non-violated timesteps.  Computed as:
```
margin[t] = min(
    min_i(c_pos[t,i] − lb[i], ub[i] − c_pos[t,i]),   # bounds
    sd[t] for each halfspace,                            # halfspace
    dist_to_obs[t] − radius  for each obstacle           # obstacle
)
mean_margin = mean(margin[t] for t where not violated[t])
```
**Interpretation**: a larger margin means the robot consistently moved through the
*interior* of the feasible region, far from boundaries.  A small margin (even with
`exec_constraint_sat_rate = 1.0`) means the robot was operating right at the edge —
fragile in the face of any perturbation.

Compare across variants:
- `diffuser` (no projection) → low margin or zero (random walk near boundaries)
- `dpcc-r/c/t` → higher margin (projector pushes trajectories away from walls)
- `combined_4` vs `no_constraint` → quantifies how much safety the constraint set adds

---

### Temporal structure (innovative)

#### `exec_first_violation_step` — int (or -1)
The timestep index of the **first** constraint violation in the rollout.
-1 if no violations occurred.

**Interpretation**: a small value means the policy violates immediately (constraint
is structurally incompatible with the initial condition).  A large value means the
robot starts safe and drifts into violation over time — likely a slow divergence or
a context where the target is near a constraint boundary.

Aggregated as: mean first-violation step over rollouts that had any violation
(`n_rollouts_with_violation` reported separately).

#### `exec_longest_safe_streak` — int (steps)
Longest consecutive sequence of timesteps with no constraint violation.
```
longest_safe_streak = max length of any True-run in ~violated[t]
```
**Interpretation**: complements `exec_constraint_sat_rate` by revealing violation
*structure*.  Two rollouts can both have `sat_rate = 0.85` but very different
behaviour:
- Short streaks (e.g. max 20 steps): violations are scattered throughout — policy
  consistently exceeds constraints.
- One long safe streak (e.g. 300 steps) + a few at the end: policy is safe for most
  of the episode, then fails near the goal.

---

### Dynamics consistency (innovative)

#### `exec_dynamics_consistency_error_mean` / `_max` — float (metres)
Measures how well the **real executed trajectory** conforms to the Euler dynamics
model that the DPCC projector enforces:
```
expected_pos[t+1] = c_pos[t] + act[t]
error[t]          = ||c_pos[t+1] − expected_pos[t+1]||
```
`act[t]` is the commanded action (`history_desired_actions`).  `c_pos[t]` is the
actual robot position from the simulator.

**Why it matters**: the projector constrains the *planned* trajectory to satisfy
`c_pos[t+1] = c_pos[t] + act[t]`.  If the real robot's physics deviate from this
model, the projector's guarantee does not transfer to execution.  A high dynamics
consistency error means:
- The robot controller does not faithfully integrate the commanded delta (controller
  dynamics mismatch).
- Or: the action was clamped / modified between planning and execution.

Typical healthy range: < 0.005 m/step.  Values > 0.01 m/step suggest the Euler
model is a poor approximation of the real robot at the current time scale.

---

## Planning metrics (`plan_*`)

Computed at each MPC replan step by `_check_planned_violations()` on the latest
unnormalised candidate set.

### `plan_post_viol_rate_mean` / `_max` — float ∈ [0, 1]
Fraction of `(sample, horizon_step)` pairs in the planned candidates that still
violate any constraint **after projection** has been applied.

```
plan_post_viol_rate = violated_pairs / (B × H)
```
where B = batch size (number of MPC candidates), H = horizon length.

**Interpretation**: this should be ~0 when the SLSQP projector succeeds.  Non-zero
values mean at least some planned trajectories were not brought into the feasible
region — the solver either did not converge (`status=8`, tight constraints) or the
initial point was too far from feasibility.

Compare with `exec_constraint_sat_rate`: if `plan_post_viol_rate` is high but
`exec_constraint_sat_rate` is also high, the executed action sequence (index 0 or
selected by min-cost) happened to be feasible even though the batch had violations.
If both are high, projection is failing and execution is unsafe.

### `plan_n_replan_steps` — int
Number of MPC replan calls in this rollout.  Denominator for understanding
`plan_post_viol_rate_mean`.

---

## Output files

### `diagnostics/rollout_{idx}_stats.json`
Per-rollout JSON.  `constraint_metrics` key added alongside existing fields.
```json
{
  "rollout_index": 0,
  "success": true,
  "mean_distance": 0.0213,
  ...
  "constraint_metrics": {
    "exec_n_steps": 312,
    "exec_n_violated_steps": 14,
    "exec_constraint_sat_rate": 0.9551,
    "exec_zero_violation_rollout": false,
    "exec_bounds_viol_count": 10,
    "exec_halfspace_viol_count": 4,
    "exec_obstacle_viol_count": 0,
    "exec_max_bounds_viol_m": 0.0183,
    "exec_max_halfspace_viol_m": 0.0071,
    "exec_max_obstacle_penetration_m": 0.0,
    "exec_constraint_margin_mean_m": 0.0391,
    "exec_first_violation_step": 47,
    "exec_longest_safe_streak": 183,
    "exec_dynamics_consistency_error_mean": 0.0021,
    "exec_dynamics_consistency_error_max": 0.0088,
    "plan_post_viol_rate_mean": 0.0074,
    "plan_post_viol_rate_max": 0.0625,
    "plan_n_replan_steps": 39
  }
}
```

### `{variant}/constraint_metrics.json`
Cross-rollout aggregate JSON written once per variant at the end of evaluation.
Structure: mean ± std for every metric, plus `per_rollout` list of individual dicts.
```json
{
  "variant": "dpcc-c",
  "geo_name": "combined_4",
  "seed": 6,
  "n_rollouts": 9,
  "exec_constraint_sat_rate": {"mean": 0.891, "std": 0.074},
  "exec_n_violated_steps":    {"mean": 17.3,  "std": 8.1},
  ...
  "exec_zero_violation_rollouts": 2,
  "per_rollout": [...]
}
```

---

## Interpreting metric combinations

| Pattern | Likely cause |
|---|---|
| High `exec_sat_rate` + low `plan_post_viol_rate` | Projector working well; execution safe |
| High `exec_sat_rate` + high `plan_post_viol_rate` | SLSQP failing but selected candidate happens to be safe by chance |
| Low `exec_sat_rate` + low `plan_post_viol_rate` | Planned trajectories are feasible but robot tracking error pushes execution outside bounds |
| Low `exec_sat_rate` + high `plan_post_viol_rate` | Projector failing; unsafe plans executed |
| Large `exec_dynamics_consistency_error` + any `exec_sat_rate` | Robot controller deviates from Euler model; projector's guarantee does not transfer |
| High `exec_constraint_margin_mean_m` | Robot moving well inside feasible region; robust to perturbations |
| Small `exec_longest_safe_streak` | Violations scattered throughout episode; structural issue with policy/constraints |
| Large `exec_first_violation_step` + few rollouts with violation | Policy safe initially; fails near task completion (goal is close to a boundary) |

---

## Quick comparison table template

Use this layout when comparing variants in a paper or report:

| Metric | diffuser | dpcc-r | dpcc-c | dpcc-t | gradient |
|---|---|---|---|---|---|
| `exec_constraint_sat_rate` (↑) | | | | | |
| `exec_zero_violation_rollouts` (↑) | | | | | |
| `exec_max_bounds_viol_m` (↓) | | | | | |
| `exec_constraint_margin_mean_m` (↑) | | | | | |
| `exec_longest_safe_streak` (↑) | | | | | |
| `exec_dynamics_consistency_error_mean` (↓) | | | | | |
| `plan_post_viol_rate_mean` (↓) | | | | | |

Arrow convention: ↑ = higher is better, ↓ = lower is better.
`diffuser` (no projection) serves as the safety baseline (expected worst).
