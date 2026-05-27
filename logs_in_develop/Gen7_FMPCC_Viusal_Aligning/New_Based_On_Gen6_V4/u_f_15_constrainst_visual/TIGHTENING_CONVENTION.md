# Tightening Convention — `enlarge_constraints` Reference

**Written**: 2026-05-27
**Related**: UF-15 (constraint visualisation), UF-16.3 (constraint metrics)

---

## TL;DR — fundamental behaviour

| | Planning (MPC projector) | Execution metrics (`exec_*`) |
|---|---|---|
| Nominal run | solves against **nominal** feasible region | checked vs nominal |
| Tightened run | solves against **smaller** feasible region (bounds −δ, obs +δ, hs shifted in) | checked vs **nominal** (same baseline) |

**Harder to plan, expected to satisfy nominal constraints more easily.**
The projector must keep the trajectory inside a δ-smaller box — a harder optimisation problem.
But because the trajectory never gets within δ of the nominal boundary, execution noise of up
to δ metres cannot push it outside the nominal constraint.  Measured against the nominal
boundary (as the original DPCC paper does), the tightened variant should show a *better*
`exec_constraint_sat_rate` than the nominal variant.

---

## Purpose

`enlarge_constraints` is a single scalar δ (metres, e.g. 0.01) that controls how much
the geometric constraints are tightened when the **tightened twin** variant runs
(`is_tightened=True`).

Tightening provides a **safety margin**: a trajectory that satisfies the tightened
constraint is guaranteed to satisfy the nominal constraint even with a δ-metre
perturbation.  The tightened twin is auto-generated alongside every base geo entry
that contains `bounds` or `obstacles` in `constraint_types`.

---

## Effect per constraint type

### Bounds

```
Nominal:   lb  ≤  c_pos  ≤  ub
Tightened: lb+δ ≤  c_pos  ≤  ub−δ
```

The workspace bounding box **shrinks** by δ on every side.  The feasible region
becomes a smaller box.  Harder for the robot to satisfy.

Code path: `setup_dpcc_projector` → `ws_lb += tightening; ws_ub -= tightening`

### Obstacles

```
Nominal:   ‖c_pos_xy − center‖ ≥ r
Tightened: ‖c_pos_xy − center‖ ≥ r + δ
```

The exclusion sphere **grows** by δ.  More space is forbidden.  Harder for the robot
to satisfy.

Code path: `radius = obs['radius'] + (tightening if is_tightened else 0.0)`

### Halfspace

```
Nominal:   n · (c_pos_xy − p₀) ≥ 0        (n = unit normal toward feasible side)
Tightened: n · (c_pos_xy − p₀_shifted) ≥ 0
           where p₀_shifted = p₀ + δ·n
```

The boundary line is **shifted by δ into the feasible half-plane**.  The feasible
region shrinks — less of the xy-workspace is available.  Harder for the robot to
satisfy.

Code path: `formulate_halfspace_constraints(hs, margin=tightening, ...)` in
`constraints_helpers.py` shifts both boundary points by `+δ·n`.

Visualisation confirms this: in the tightened `constraint_overview.png` the darkorange
halfspace boundary line moves toward the feasible side compared to the non-tightened
overview.

### Dynamics

```
Nominal:   c_pos[t+1] = c_pos[t] + act[t]
Tightened: unchanged
```

The dynamics constraint is an **equality** — there is no geometric boundary to move
inward.  δ is not applied.  The `is_tightened` flag has no effect on dynamics.

---

## Is one parameter enough?

**Yes**, for the current setup.  A single δ is applied uniformly to all geometric
constraint types (bounds, obstacle, halfspace) and is geometrically consistent: each
type shrinks the feasible region by approximately δ metres at its boundary.

If per-type margins are ever needed (e.g. 0.02 m on the obstacle but only 0.005 m
on the bounds), the yaml could be extended with separate keys:

```yaml
enlarge_bounds:     0.005
enlarge_obstacle:   0.020
enlarge_halfspace:  0.010
```

But this requires code changes in `setup_dpcc_projector`, `plot_geo_constraints`,
and the UF-16.3 check functions.  Not needed now.

---

## What "tightened" looks like in visualisation

| Panel | Non-tightened | Tightened |
|---|---|---|
| 3D box | workspace bounds box | **smaller** box (all walls closer in) |
| 3D obstacle sphere | sphere at radius r | **larger** sphere (r+δ) |
| 3D halfspace plane | darkorange rect at nominal boundary | plane shifted **toward feasible side** |
| XY bounds | blue rectangle | **smaller** rectangle |
| XY obstacle | tomato circle | **larger** circle |
| XY halfspace | orange infeasible fill | orange fill **wider** (boundary moved in) |
| XZ obstacle | dashed circle + band | **larger** circle |
| XZ halfspace | orange band x-extent | same x-extent, boundary at shifted y |

---

## User observation confirmed

> "the tightened makes obstacle bigger and halfspace less usable space"

**This is correct and NOT inverted.**

- Obstacle bigger (r+δ) = larger exclusion zone = **tighter** ✓
- Halfspace less usable space (boundary pushed into feasible half-plane) = **tighter** ✓

Both effects are the intended DPCC tightening behavior.

---

## Fixes applied to UF-16.3 metric checks (2026-05-27)

### Fix 1 — halfspace sign in `_check_planned_violations`

```python
# WRONG — shifts boundary AWAY from feasible region → looser check for tightened
x1 -= enlarge * nx;  y1 -= enlarge * ny

# CORRECT — shifts boundary INTO feasible region → tighter check
x1 += enlarge * nx;  y1 += enlarge * ny
```

`(nx, ny)` is the feasible-side normal.  Subtracting it moved the boundary in the
infeasible direction.  Fix applied to `_check_planned_violations` in both eval files.

### Fix 2 — exec metrics always use nominal boundary (matches original DPCC paper)

`check_trajectory_constraints` is now always called with `enlarge=0.0` regardless of
`is_tightened`.  Previously it was called with `enlarge=δ` for tightened runs, which
evaluated the tightened trajectory against a *harder* standard than the nominal run —
making the cross-variant comparison unfair and hiding the benefit of tightening.

The original DPCC `eval.py` uses `constraint_list_polytopic_not_tightened` and
un-enlarged obstacle radii for violation checking in all variants.  Our code now
matches that convention.  The tightened variant is expected to show better
`exec_constraint_sat_rate` because its trajectories have a δ buffer over the nominal
boundary that absorbs execution noise.

---

## Summary of all tightening code paths

| File | Function | Bounds | Obstacle | Halfspace | enlarge used? |
|---|---|---|---|---|---|
| `eval_visual_aligning_dpcc.py` | `setup_dpcc_projector` | `lb+=δ, ub-=δ` ✓ | `r+=δ` ✓ | via `formulate_halfspace_constraints(margin=δ)` ✓ | δ (projector) |
| `eval_visual_aligning_dpcc.py` | `plot_geo_constraints` | `lb+=δ, ub-=δ` ✓ | `r+=δ` ✓ | `p += δ·n` ✓ | δ (visualisation) |
| `eval_visual_aligning_dpcc.py` | `_hs_xy_draw` | — | — | `x1 += δ·nx` ✓ | δ (visualisation) |
| `eval_visual_aligning_dpcc.py` | `check_trajectory_constraints` | always `enlarge=0` | always `enlarge=0` | always `enlarge=0` | **0 always** (nominal) |
| `eval_visual_aligning_dpcc.py` | `_check_planned_violations` | `lb+=δ, ub-=δ` ✓ | `r+=δ` ✓ | `x1 += δ·nx` ✓ | δ (projector check) |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | all of the above | same ✓ | same ✓ | same ✓ | same ✓ |

**Why `check_trajectory_constraints` always uses `enlarge=0`**: matches original DPCC paper `eval.py`,
which checks all variants (nominal and tightened) against nominal constraint boundaries so that
`exec_constraint_sat_rate` is a fair cross-variant comparison.  The tightened variant is expected to
show *better* sat_rate because its trajectories have a δ buffer over the nominal boundaries.

**Why `_check_planned_violations` uses δ for tightened**: this metric answers "did the projector
enforce the constraints it was *given*?" — the projector was given the tighter boundary, so the
check must use the same boundary.  A non-zero `plan_post_viol_rate` means SLSQP did not converge
to the tighter feasible region.
