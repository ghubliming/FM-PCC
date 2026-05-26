# How to Control Geo Constraint Configs in visual_aligning_eval.yaml

**File**: `config/visual_aligning_eval.yaml`  
**Section**: `geo_constraint_variants`

---

## Disable a config (skip it in eval)

Comment out the entry:

```yaml
geo_constraint_variants:
  - name: no_constraint
    constraint_types: []
  # - name: bounds_only_1        ← commented out = will not run
  #   constraint_types: ['bounds']
  - name: bounds_dynamics_1
    constraint_types: ['bounds', 'dynamics']
```

---

## Enable the _2 placeholder (parameter tuning sweep)

Uncomment the `_2` block and edit the values:

```yaml
  - name: bounds_dynamics_2
    constraint_types: ['bounds', 'dynamics']
    workspace_bounds:
      lb: [0.25, -0.40, 0.05]   # ← your new bounds
      ub: [0.75,  0.40, 0.45]
    enlarge_constraints: 0.02   # ← your new tightening margin
```

---

## Add a new config

Append a new entry with a unique name:

```yaml
  - name: bounds_only_tight
    constraint_types: ['bounds']
    workspace_bounds:
      lb: [0.35, -0.30, 0.08]
      ub: [0.65,  0.30, 0.35]
    enlarge_constraints: 0.02
```

---

## Rules

| Key | Behaviour |
|---|---|
| `workspace_bounds` absent | Inherits top-level `workspace_bounds` default |
| `enlarge_constraints` absent | Inherits top-level `enlarge_constraints` default |
| `constraint_types: []` | Projector is a no-op — identical to raw FM (use for baseline) |
| `constraint_types: ['bounds']` | Workspace box enforced on EE position only |
| `constraint_types: ['bounds', 'dynamics']` | Box + Euler integration consistency |
| `'dynamics'` without `'bounds'` | Valid but unusual — deriv links with no position ceiling |
| `'obstacles'` | Requires `obstacle_constraints` list in the same geo entry |
| `name` must be unique | Each active entry writes to `results/{name}/` — duplicates overwrite |

---

## Why `no_constraint` has no index

`constraint_types: []` has no tunable parameters, so a `_2` variant would be identical. Only configs with physical parameters (`workspace_bounds`, `enlarge_constraints`) get the `_1`/`_2` index scheme.

---

## Add an obstacle constraint

Enable by uncommenting `obstacle_2d_1` or `obstacle_3d_1` in the yaml, then tuning:

```yaml
  - name: obstacle_2d_1
    constraint_types: ['bounds', 'dynamics', 'obstacles']
    obstacle_constraints:
      - type: sphere_outside
        dimensions: ['x', 'y']      # 2D cylinder projection (x-y plane, no z constraint)
        center: [0.50, 0.00]        # metres — centre of workspace; tune to scene
        radius: 0.06                # metres — tune to physical obstacle size
```

Or the 3D sphere variant (stricter — EE must stay outside a true sphere):

```yaml
  - name: obstacle_3d_1
    constraint_types: ['bounds', 'dynamics', 'obstacles']
    obstacle_constraints:
      - type: sphere_outside
        dimensions: ['x', 'y', 'z']
        center: [0.50, 0.00, 0.12]  # metres — tune z to actual obstacle height
        radius: 0.06
```

Named dims `'x'`, `'y'`, `'z'` map to trajectory indices 6, 7, 8 (EE Cartesian position). You can also use `'dx'`/`'dy'`/`'dz'` (indices 0-2) or `'des_x'`/`'des_y'`/`'des_z'` (indices 3-5) for action/desired-position constraints, though that is unusual. Multiple obstacles are supported — just add more list entries under `obstacle_constraints`.

---

## Classic DPCC constraint set (avoiding paper) vs visual aligning

**Classic DPCC** (`config/projection_eval.yaml`) runs all four types:

```yaml
constraint_types: ['halfspace', 'obstacles', 'dynamics', 'bounds']
```

| Constraint type | Avoiding (published) | Visual aligning | Reason |
|---|---|---|---|
| `halfspace` | ✅ triangular obstacle walls (top-left / top-right / both) | ❌ not applicable | Aligning workspace is open table — no diagonal wall obstacles |
| `obstacles` | ✅ spherical obstacles (6 cylinder positions) | ❌ not applicable | No simulated spheres in aligning scene |
| `bounds` | ✅ 2D xy position limits | ✅ 3D xyz Cartesian box (`workspace_bounds`) | Same concept, extended to 3D |
| `dynamics` | ✅ Euler link x[t+1]=x[t]+vx[t], y[t+1]=y[t]+vy[t] | ✅ Euler link c_pos[t+1]=c_pos[t]+act[t] (3D) | Same concept, extended to 3D |

**`bounds_dynamics_1` is the closest equivalent to classic DPCC for the aligning task** — it applies every constraint type that is physically meaningful for this environment.

### How to replicate the DPCC-equivalent constraint set

No code change needed. `bounds_dynamics_1` already matches:

```yaml
  - name: bounds_dynamics_1
    constraint_types: ['bounds', 'dynamics']
    # workspace_bounds inherited from top-level defaults:
    #   lb: [0.3, -0.35, 0.05]   (x forward, y lateral, z vertical)
    #   ub: [0.7,  0.35, 0.40]
    # enlarge_constraints: 0.01  (applied when variant name contains 'tightened')
```

The `halfspace` and `obstacles` types are absent because the aligning task has no such geometric features. Adding them to `constraint_types` would have no effect — `setup_dpcc_projector()` only reads `'bounds'` and `'dynamics'` from the list; unknown types are silently ignored.
