# Study — Which trajectory dims does each DPCC constraint bind to?

**Question:** In the original DPCC (avoiding-task lineage), do the **geometric** constraints
(halfspace, obstacles) apply to `p_des`, to `p`, and/or to the action? And is the Gen7
**visual-aligning** projector faithfully the same as DPCC-avoiding?

**Short answer:**
- **Geometric constraints (halfspace, obstacles) bind to ACTUAL position `p` ONLY** — never
  `p_des`, never the action.
- **`bounds`** in avoiding bind to the **action** (a *velocity* limit, `vx,vy`) — not position.
- **`dynamics`** bind **both** `p` and `p_des`, each Euler-linked to the action.
- **Visual-aligning is faithful** to DPCC-avoiding for halfspace / obstacles / dynamics. It
  **deliberately diverges on `bounds`**: it repurposes them from a velocity limit into a
  Cartesian workspace box on `p`. That divergence is an intentional, documented adaptation
  (an arm needs a physical workspace box), not a bug.

---

## 1. The avoiding transition tensor and its index map

`config/projection_eval.yaml`:
```yaml
dt: { 'avoiding': 1 }          # a = [delta_x, delta_y], NOT [vx, vy] → Euler dt=1
observation_indices: { 'avoiding': {'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3} }
action_indices:      { 'avoiding': {'vx': 0, 'vy': 1} }
```

`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` L158-163 shifts the obs
indices past the action block and merges them:
```python
if fm_model.__class__.__name__ == 'FlowMatchingODE':          # states_actions variant
    action_dim = fm_model.action_dim                          # = 2 for avoiding
    obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
    act_obs_indices = {**act_indices, **obs_indices_updated}
```

Resolving the arithmetic (`+action_dim = +2` on every obs index):

```
act_obs_indices = { vx:0, vy:1,      # action  (the two Δ / velocity dims)
                    x_des:2, y_des:3, # p_des   (desired/commanded position)
                    x:4, y:5 }        # p       (actual realized position)
```

So the flattened per-step trajectory is:
```
[ vx(0) vy(1) | x_des(2) y_des(3) | x(4) y(5) ]
  └ action ─┘   └── p_des ──────┘   └── p ──┘
```
This is the canonical DPCC `[action | state]` layout, where `state = [p_des, p]`.

---

## 2. Where each constraint family lands (quoted)

### 2.1 halfspace → actual `p` only (dims 4,5)

Build call, eval L172-176:
```python
if 'halfspace' in constraint_types:
    for constraint in polytopic_constraints:
        constraint_list.append(('ineq',
            utils.formulate_halfspace_constraints(constraint, 0, trajectory_dim, act_obs_indices)))
```

`flow_matcher_v3_uav/utils/constraints_helpers.py` L4-20 — note it reads **only** `['x']`
and `['y']` out of the index map:
```python
def formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices):
    m = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
    ...
    C_row = np.zeros(trajectory_dim)
    if constraint[2] == 'below':
        C_row[act_obs_indices['x']] = -m      # act_obs_indices['x'] == 4  → ACTUAL p
        C_row[act_obs_indices['y']] = 1        # act_obs_indices['y'] == 5  → ACTUAL p
    elif constraint[2] == 'above':
        C_row[act_obs_indices['x']] = m        # 4
        C_row[act_obs_indices['y']] = -1       # 5
        d *= -1
    return C_row, d
```
`act_obs_indices['x'] = 4`, `['y'] = 5` → the halfspace row is nonzero **only on actual
position `p`**. `x_des`/`y_des` (2,3) and the action (0,1) get zero coefficients.

### 2.2 obstacles → actual `p` only (dims 4,5)

Build call, eval L181-184:
```python
if 'obstacles' in constraint_types:
    for constr in obstacle_constraints:
        constraint_list.append([constr['type'],
            [act_obs_indices[constr['dimensions'][0]], act_obs_indices[constr['dimensions'][1]]],
            constr['center'], constr['radius']])
```
The config always names position dims:
```yaml
obstacle_constraints: { 'avoiding-d3il': [
    {'type': 'sphere_outside', 'dimensions': ['x', 'y'], 'center': [0.4, 0.08], 'radius': 0.06},
    ... ] }
```
`dimensions:['x','y']` → `[act_obs_indices['x'], act_obs_indices['y']] = [4,5]` → the
quadratic exclusion `(x−cx)² + (y−cy)² ≥ r²` is written **only on actual `p`**. The
quadratic is assembled in `ObstacleConstraints.build_matrices` (`projection.py` L442-459),
placing `P[dim,dim]`, `q[dim]` at exactly those dims and flipping sign for `sphere_outside`.

### 2.3 bounds → the ACTION (velocity), not position (dims 0,1)

Config, `projection_eval.yaml`:
```yaml
bounds: {   # need to be within the limits of the dataset due to the normalization
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
    ... ] }
```
`formulate_bounds_constraints` (`constraints_helpers.py` L22-32) writes lb/ub at
`act_obs_indices[dim]` for each named dim; `['vx','vy']` → dims **0,1** = the **action**.
So in avoiding, `bounds` is a **velocity/action limit** ("keep actions inside the dataset's
normalized range"), *not* a spatial box on position. This is a different role for the
`bounds` family than "workspace box."

### 2.4 dynamics → BOTH `p` and `p_des`, each linked to the action

`formulate_dynamics_constraints` (`constraints_helpers.py` L47-54):
```python
if 'avoiding' in exp and action_dim > 0:
    dynamic_constraints = [
        ('deriv', np.array([act_obs_indices['x'],     act_obs_indices['vx']])),   # [4, 0]
        ('deriv', np.array([act_obs_indices['y'],     act_obs_indices['vy']])),   # [5, 1]
        ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),   # [2, 0]
        ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),   # [3, 1]
    ]
```
Four `deriv` rows: actual `x`(4)←`vx`(0), `y`(5)←`vy`(1) **and** desired `x_des`(2)←`vx`(0),
`y_des`(3)←`vy`(1). Each `deriv` is the explicit-Euler link built in
`DynamicConstraints.build_matrices` (`projection.py` L382-392):
```python
mat_append[i, i * transition_dim + x_idx]        =  1        # x[t]
mat_append[i, i * transition_dim + dx_idx]       =  self.dt  # + dt * (Δ/vel)[t]
mat_append[i, (i+1) * transition_dim + x_idx]    = -1        # = x[t+1]
```
i.e. `x[t+1] = x[t] + dt·(action)[t]`, written for **both** the actual and the desired
position channel against the **same** action.

---

## 3. Summary table

| Family | Reads which named dims | Resolved indices | Channel it constrains |
|---|---|---|---|
| **halfspace** | `['x'], ['y']` | 4, 5 | **actual `p` only** |
| **obstacles** | `dimensions:['x','y']` | 4, 5 | **actual `p` only** |
| **bounds** | `['vx','vy']` | 0, 1 | **the action** (velocity limit) |
| **dynamics** | `[x,vx],[y,vy],[x_des,vx],[y_des,vy]` | (4,0)(5,1)(2,0)(3,1) | **both `p` and `p_des`**, ← action |

**The action itself is never given a *geometric* constraint.** It appears only as (a) the
`dx` term inside `deriv` (the Euler link), and (b) the target of the velocity-limit `bounds`
in avoiding. There is no obstacle/halfspace on the action.

---

## 4. Why it is designed this way — and is it right?

**Geometric constraints on actual `p` (not `p_des`, not action): correct, and deliberate.**
- Physical safety is a property of *where the robot actually is*. A pillar is cleared by the
  realized position `p`, not by the setpoint `p_des`. So the exclusion/halfspace must sit on
  `p`. ✔ sound.
- Why not *also* put a copy on `p_des`? Because `p` and `p_des` are **rigidly coupled** by
  the four `deriv` rows (§2.4): both are the running Euler integral of the **same** action
  from the same initial state. Once you constrain `p`, the projector cannot satisfy the
  obstacle row while leaving `p_des` free — the dynamics rows drag `p_des` along. A second
  geometric copy on `p_des` would be **redundant** (and, if margins differed, could make the
  QP infeasible). So `p`-only is not a gap; it is the minimal sufficient set given the
  dynamics coupling. ✔ good design.
- Why not on the action? A geometric (position) constraint on a *velocity/Δ* dimension is a
  category error — the action lives in a different space. The action is constrained where it
  makes sense: magnitude limits via `bounds` (§2.3) and the Euler consistency via `deriv`. ✔.

**`bounds` on velocity in avoiding: right for its purpose.** The config comment says it
plainly — *"need to be within the limits of the dataset due to the normalization."* It is a
guard that keeps sampled actions inside the trained/normalized range, not a spatial safety
box. Correct for the avoiding task's needs.

**One caveat for lagging plants (the UAV).** On a perfect-tracking system (the D3IL point /
arm) `p ≈ p_des`, so `p`-only geometric constraints and the coupling are airtight. On a
drone `p` *lags* `p_des`, so the *executed* setpoint could momentarily sit a margin outside
what the plan's `p` cleared. DPCC's native answer is **constraint tightening** (shrink the
geometry by `enlarge_constraints` for `-tightened` variants), not adding `p_des` rows — this
is exactly what E9 §4.3 sizes from measured tracking error. So the canonical `p`-only design
is kept; the drone's lag is absorbed the DPCC way.

---

## 5. Is Gen7 visual-aligning faithfully the same as DPCC-avoiding?

Visual-aligning transition (3D): `[dx dy dz(0,1,2) | des_x des_y des_z(3,4,5) | x y z(6,7,8)]`,
`_DIM = {'dx':0,'dy':1,'dz':2,'des_x':3,'des_y':4,'des_z':5,'x':6,'y':7,'z':8}`
(`fm_visual_aligning_test/eval_fm_visual_aligning.py` `setup_dpcc_projector`).

| Family | DPCC-avoiding | Visual-aligning | Faithful? |
|---|---|---|---|
| **halfspace** | actual `p` (4,5) | `_hs_indices={'x':6,'y':7}` → actual `p` (6,7) | ✅ same (2D→3D dims) |
| **obstacles** | actual `p` (4,5) | `dimensions` mapped via `_DIM['x'..]` → 6,7(,8) | ✅ same |
| **dynamics** | both `p_des`&`p` ← action (4 rows) | both `des`&actual ← action, **6 rows** `(3←0)(4←1)(5←2)(6←0)(7←1)(8←2)` | ✅ same pattern, 3D-extended (DC_FIX) |
| **bounds** | **action** velocity limit (`vx,vy`, dims 0,1) | **`workspace_bounds`** Cartesian box on actual `p` (6,7,8) | ⚠️ **repurposed** |

**Verdict:** For the **geometric** families (halfspace, obstacles) and **dynamics**,
visual-aligning is **faithfully identical** to DPCC-avoiding — same channel (actual `p`),
same coupling, only lifted 2D→3D. The one **intentional divergence** is `bounds`:
- avoiding: `bounds` = *velocity/action* limit (dataset-normalization guard).
- visual-aligning: the old velocity `bounds` are explicitly retired
  (`visual_aligning_eval.yaml`: *"NOT NEEDED: Replaced by 'workspace_bounds' below, which
  enforces safe physical Cartesian ranges"*) and `bounds` now means a **Cartesian workspace
  box on actual `p` (6,7,8)**.

That change is a **justified adaptation, not a deviation from principle**: an arm/drone needs
a physical workspace envelope (a role avoiding never used `bounds` for), and it is still
applied to actual `p`, consistent with the "geometry on `p`" rule. It only means "the
`bounds` family means different things in the two tasks" — worth stating explicitly so nobody
assumes byte-identical `bounds` semantics across them.

**Bottom line:** visual-aligning did **not** get the geometric binding wrong — it matches
DPCC-avoiding (geometry on actual `p` only). E9 (UAV) follows the same rule; the only
task-specific choice is whether `bounds` is used as a workspace box (as in visual-aligning /
E9) or a velocity limit (as in avoiding) — a config choice, not a change to where geometry
binds.

---

## 6. Files quoted

- `config/projection_eval.yaml` — avoiding indices, halfspace/obstacle/bounds config.
- `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` L158-188 — index merge + constraint build.
- `flow_matcher_v3_uav/utils/constraints_helpers.py` L4-54 — `formulate_halfspace_constraints`,
  `formulate_bounds_constraints`, `formulate_dynamics_constraints`.
- `flow_matcher_v3_uav/sampling/projection.py` L382-392 (`DynamicConstraints` Euler rows),
  L442-459 (`ObstacleConstraints` quadratic).
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` `setup_dpcc_projector` — `_DIM`,
  `_hs_indices`, dynamics rows.
- `config/visual_aligning_eval.yaml` — `workspace_bounds` note (bounds repurposed).
