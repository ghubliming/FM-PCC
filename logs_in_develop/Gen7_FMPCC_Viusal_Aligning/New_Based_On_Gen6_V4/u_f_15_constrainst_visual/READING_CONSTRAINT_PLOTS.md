# Reading the Constraint Visualisation Plots

**Scope**: `constraint_overview.png` (UF-15.1) and `rollout_N_mpc_foresight.svg` overlay (UF-15.2)  
**Date**: 2026-05-27  
**Branch**: `update_into_FM`

---

## 1. Overview: Two separate outputs

| Output file | When generated | What it shows |
|---|---|---|
| `results/{geo_name}/constraint_overview.png` | Once per geo entry, before any trajectories run | Pure geometry: workspace box, halfspace line, obstacle — **no trajectory data** |
| `results/{geo_name}/{variant}/diagnostics/rollout_N_mpc_foresight.svg` | Once per rollout | Constraint geometry overlaid **on top of** the actual EE trajectories and MPC candidates |

The two outputs serve different purposes. `constraint_overview.png` is a **pre-run sanity check** — confirm the geometry is correct before committing to a long run. The foresight SVG is the **performance check** — see whether trajectories actually respected the constraints.

---

## 2. Coordinate frame

All plots use the robot EE (end-effector) world frame:

| Axis | Physical direction | Typical range |
|---|---|---|
| **x** | Table depth (robot forward/backward) | 0.30 – 0.70 m |
| **y** | Lateral (robot left/right) | −0.35 – 0.35 m |
| **z** | Vertical height above table | 0.05 – 0.40 m |

Trajectory dimension layout in the 9D state vector:
```
[  dx   dy   dz  |  des_x  des_y  des_z  |  x    y    z  ]
  [0]  [1]  [2]     [3]    [4]    [5]      [6]  [7]  [8]
```
The 2D XY panel plots `x` (dim 6) vs `y` (dim 7).  
The 3D panel plots all three: `x` (6), `y` (7), `z` (8).

---

## 3. Constraint types and their visual encoding

### 3.1 Workspace bounds — steelblue

**What it is**: an axis-aligned box `[lb_x, ub_x] × [lb_y, ub_y] × [lb_z, ub_z]` that the EE must stay inside.  
**Configured by**: `workspace_bounds.lb` / `.ub` in the yaml geo entry.  
**2D variant**: set `lb[2] = -.inf`, `ub[2] = .inf` to leave z unconstrained (used in `bounds_only_1`).  
**3D variant**: explicit z floor/ceiling (used in `combined_2`, `combined_4`).

| Panel | Element | Color | Style |
|---|---|---|---|
| XY top-down | Filled rectangle | steelblue | alpha=0.10, edgecolor steelblue, lw=1.5 (overview); alpha=0.10 behind trajectories (SVG) |
| XZ side | Filled rectangle | steelblue | alpha=0.10 (overview only) |
| 3D | Wireframe box (12 edges) | steelblue | alpha=0.45, lw=1.0 — 8 corners connected |

**How to read**: anything outside the steelblue rectangle/box is a constraint violation. In the foresight SVG the rectangle is deliberately faint (alpha=0.10) so trajectories remain readable; the boundary edge (steelblue line) is the hard limit.

**Inf z clamping**: when z bounds are `±inf` (2D config), the 3D and XZ panels cannot draw an infinite box. Display-only clamp is applied: `z_lo = 0.0`, `z_hi = 0.50`. This does **not** affect the actual constraint — it is purely for visual rendering.

---

### 3.2 Halfspace — darkorange

**What it is**: a half-plane constraint defined by a line through two points. The EE must stay on the feasible side of the line.  
**Configured by**: `halfspace_constraints` list in the yaml geo entry.  
**Format per entry**: `[[x1, y1], [x2, y2], 'above' | 'below']`

The two points define the **boundary line** (infinite in both directions, clipped to the display viewport). The side keyword says which half of the plane the EE is allowed to be in.

#### How the normal is computed

```
direction vector:  d = (x2-x1, y2-y1)

'above':  normal n = (-dy,  dx)   ← CCW rotation of d (points "left" when walking d→)
'below':  normal n = ( dy, -dx)   ← CW  rotation of d (points "right" when walking d→)
```

The normal vector **always points toward the feasible region** — the side the EE must stay on.

#### What you see

| Element | Color | Meaning |
|---|---|---|
| Straight line across the viewport | darkorange | Boundary of the halfspace — EE must not cross this |
| Arrow from line midpoint | darkorange | Points toward the **feasible** (allowed) region |
| Label "feasible" | darkorange | Confirms which side is allowed |

**How to read**: the darkorange line divides the XY plane. The arrow tells you which side the EE is supposed to be on. If a trajectory crosses to the other side of the line, it violates the halfspace constraint.

#### Viewport clipping (Cohen-Sutherland)

The line is infinite in theory but drawn only inside the visible area. The clipping logic:
```
tx = [t at xlim[0], t at xlim[1]]   # where the line crosses x boundaries
ty = [t at ylim[0], t at ylim[1]]   # where the line crosses y boundaries
t_lo = max(min(tx), min(ty))         # enter both slabs
t_hi = min(max(tx), max(ty))         # exit both slabs
```
This ensures both endpoints are inside the display box. Only drawn in 2D panels (XY, XZ); the 3D panel does not render halfspace lines.

---

### 3.3 Obstacle — tomato (red)

**What it is**: a spherical exclusion zone centred at a 3D point with a given radius. The EE must stay **outside** this sphere.  
**Configured by**: `obstacle_constraints` list.  
**Format per entry**: `type: sphere_outside`, `dimensions: ['x','y']` or `['x','y','z']`, `center: [cx, cy, cz]`, `radius: r`.

| Panel | Element | Color | Style |
|---|---|---|---|
| XY top-down | Filled circle | tomato | alpha=0.15, edgecolor tomato, lw=1.5 |
| XY top-down | Centre cross marker | red | `r+`, ms=6 — marks exact centre |
| XZ side | Filled circle (only if `'z' in dimensions`) | tomato | alpha=0.15 |
| 3D | Sphere surface mesh | tomato | alpha=0.25 (overview only) |

**How to read**: the tomato circle is the exclusion zone — the EE should never be inside it. The cross at the centre marks the measured obstacle position. If a trajectory passes through the circle, it is inside the obstacle.

**2D obstacles** (`dimensions: ['x','y']`): the 3D view centres the sphere at the vertical midpoint of the workspace for display only (`cz = (lb_z + ub_z) / 2`). The actual constraint only enforces the x-y projection.

---

### 3.4 Dynamics constraint — no geometry

`constraint_types` may include `'dynamics'`. This encodes the Euler integration link:
```
c_pos[t+1] = c_pos[t] + act[t]
```
There is no geometric boundary to draw. When `dynamics` is the **only** active type, both the XY and XZ axes in `constraint_overview.png` show the text label:
```
no geometric constraints
```
and no coloured shapes appear. The figure is still generated (for completeness).

---

## 4. `constraint_overview.png` — the 3-panel standalone figure

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│                     │                     │                     │
│   3D view           │   XY top-down       │   XZ side           │
│   (3D projection)   │   (x horizontal,    │   (x horizontal,    │
│                     │    y vertical)      │    z vertical)      │
│                     │                     │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Figure title**: `{geo_name} [tightened]  |  types: [dynamics, bounds, ...]`  
- `[tightened]` appears only for the `-tightened` geo variant.  
- `types` lists all active constraint types.

**Footer** (only when `'dynamics'` is in constraint_types):
> `Dynamics: c_pos[t+1] = c_pos[t] + act[t]  (Euler link — no geometric shape)`

### 4.1 Panel 1 — 3D view

Shows the full 3D workspace. Camera is at a fixed elevation and azimuth (elev=20°, azim=-60° default).

| Element | Condition |
|---|---|
| Steelblue wireframe box + semi-transparent faces | `'bounds'` in constraint_types |
| Tomato sphere surface | `'obstacles'` in constraint_types |
| No halfspace shape | Halfspace is 2D only — not rendered in 3D panel |

The wireframe has 12 edges (all box edges). The semi-transparent faces use `Poly3DCollection` with `alpha=0.08` so the interior remains visible.

### 4.2 Panel 2 — XY top-down

Bird's-eye view of the table plane. This is the **primary diagnostic panel** because both the aligning task and the DPCC projector operate mainly in x-y.

| Element | Condition |
|---|---|
| Steelblue rectangle | `'bounds'` in constraint_types |
| Darkorange line + arrow + "feasible" label | `'halfspace'` in constraint_types |
| Tomato circle + cross | `'obstacles'` in constraint_types |

Axes are set from `ws_lb[:2]` / `ws_ub[:2]` ± 0.05 m margin. Fixed limits ensure the halfspace clipping is correct.

### 4.3 Panel 3 — XZ side

Side view showing height profile.

| Element | Condition |
|---|---|
| Steelblue rectangle | `'bounds'` in constraint_types |
| Grey dashed lines + z annotations | Always — marks floor (lb_z) and ceiling (ub_z) |
| Tomato circle | `'obstacles'` in constraint_types **and** `'z' in dimensions` |

For 2D obstacles (no z), the XZ panel shows no obstacle circle — the sphere only constrains x-y.

---

## 5. `rollout_N_mpc_foresight.svg` — constraint overlay

The foresight SVG is generated per rollout during eval. It has **two panels** side by side: XY (left) and 3D (right). UF-15.2 adds constraint overlays behind the trajectory data (all at `zorder=1`, below trajectory lines).

```
┌───────────────────────┬───────────────────────┐
│  XY top-down          │  3D view               │
│  (with trajectories)  │  (with trajectories)   │
│  + constraint overlay │  + constraint overlay  │
└───────────────────────┴───────────────────────┘
```

### 5.1 XY panel overlay (left)

Drawn after `ax_xy.grid(True, alpha=0.3)`, before trajectory lines:

| Layer | What | Color / style |
|---|---|---|
| z=1 (bottom) | Workspace bounds rectangle | steelblue, facecolor alpha=0.10, edgecolor lw=1.5 |
| z=1 | Obstacle exclusion circle | tomato, facecolor alpha=0.15, edgecolor lw=1.5 |
| z=2 | Obstacle centre cross | red `r+` marker, ms=6 |
| z=3 | Halfspace boundary line + arrow | darkorange, lw=1.5 |

Trajectory lines are drawn at higher zorder by default, so they always appear in front of the constraint shapes.

**Xlim/ylim for halfspace clipping**: taken from `workspace_bounds.lb[:2]` / `.ub[:2]` ± 0.05 m. Falls back to `(0.20, 0.80)` × `(-0.45, 0.45)` if no bounds are configured.

### 5.2 3D panel overlay (right)

Drawn after `ax_3d.legend(fontsize=9)`:

| Element | Color / style | Condition |
|---|---|---|
| Wireframe box (12 edges) | steelblue, alpha=0.45, lw=1.0 | `'bounds'` in constraint_types |
| No obstacle or halfspace shape | — | 3D panel is wireframe-only in the foresight SVG |

The 3D panel does not re-draw obstacles or halfspace in the foresight view (no `Poly3DCollection` sphere, no halfspace plane) — the wireframe alone makes the workspace boundary visible without obscuring the trajectory.

---

## 6. Tightened variants

When `enlarge_constraints` is non-null in the yaml and a geo entry has bounds or obstacles, a `-tightened` twin is automatically generated (e.g. `combined_4-tightened/`).

`is_tightened=True` activates these adjustments in all visualisation functions:

| Constraint type | Effect of tightening |
|---|---|
| Bounds | Each wall shifts **inward** by `enlarge` m: `lb += enlarge`, `ub -= enlarge` |
| Halfspace | Boundary line shifts **inward** (toward infeasible side) by `enlarge` m: `x1 += enlarge*nx`, `y1 += enlarge*ny` |
| Obstacle | Radius grows by `enlarge` m: drawn at `radius + enlarge` |

This means the tightened visualisation shows a **smaller feasible region** — the constraint is more conservative. The tightened twin is a test of robustness: if trajectories still respect the tightened constraints, they have a safety margin over the nominal ones.

---

## 7. What "correct" looks like — reading checklist

When you open a foresight SVG, check:

1. **Bounds rectangle present?** If `combined_4` is active and no steelblue rectangle appears, the `'bounds'` type is not in `constraint_types` or `workspace_bounds` is missing from the yaml.

2. **Halfspace arrow direction correct?** The arrow should point toward the region where most of the trajectories are. If the arrow points into a region where trajectories are sparse or absent, either the side keyword or the two points in the yaml are wrong.

3. **Trajectories inside the rectangle?** EE positions (x,y) should lie within the steelblue box. Excursions outside indicate violation.

4. **Trajectories outside the tomato circle?** Trajectories that pass through the obstacle circle are violations.

5. **Trajectories on the correct side of the orange line?** All trajectory points should be on the side the arrow points toward.

6. **Axes not zoomed out to strange range?** If the foresight SVG shows x/y ranges like 0.0–1.2, the halfspace clipping bug was active (old outer-extreme code). Fixed in UF-15.3 — axes should stay within ≈ ±0.05 m of the workspace bounds.

---

## 8. Yaml → plot mapping (quick reference)

```yaml
geo_constraint_variants:
  - name: combined_4
    constraint_types: ['dynamics', 'bounds', 'halfspace', 'obstacles']
    workspace_bounds:
      lb: [0.30, -0.35, 0.05]   # → steelblue rect left/bottom/floor
      ub: [0.70,  0.35, 0.40]   # → steelblue rect right/top/ceiling
    halfspace_constraints:
      - [[0.30, -0.05], [0.70, 0.05], 'above']
        #  pt1 = (0.30,-0.05)     → line start (before clipping)
        #  pt2 = (0.70, 0.05)     → line end   (before clipping)
        #  'above' → normal = (-dy, dx) = (-0.10, 0.40) normalised
        #           → arrow points upward-left (positive-y side)
    obstacle_constraints:
      - type: sphere_outside
        dimensions: ['x', 'y']   # 2D: only x-y projected in XY panel
        center: [0.50, 0.00]      # → cross marker at (0.50, 0.00)
        radius: 0.06              # → circle radius 0.06 m (+ enlarge if tightened)
```

---

## 9. Color legend summary

| Color | Shape | Constraint |
|---|---|---|
| **steelblue** | Rectangle (2D) / Wireframe box (3D) | Workspace bounds — EE must stay **inside** |
| **darkorange** | Line + arrow | Halfspace boundary — EE must stay on **arrow side** |
| **tomato / red** | Circle (2D) / Sphere surface (3D) | Obstacle exclusion — EE must stay **outside** |
| grey dashed | Horizontal lines (XZ panel only) | Floor and ceiling z annotations |
