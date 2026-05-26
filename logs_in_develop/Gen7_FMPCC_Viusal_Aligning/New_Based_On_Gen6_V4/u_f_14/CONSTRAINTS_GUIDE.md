# Geo Constraint Configuration Guide — visual_aligning_eval.yaml

**File**: `config/visual_aligning_eval.yaml`  
**Section**: `geo_constraint_variants`

---

## Redundant runs — what is actually equivalent

Understanding which (geo, variant) combinations produce identical output avoids
running the same experiment multiple times and wasting Slurm time.

### `diffuser` — identical across ALL geo configs

The raw diffusion baseline. No projector is created when `'diffuser' in variant`
(code path: `if 'diffuser' not in variant: projector = setup_dpcc_projector(...)`).
The trajectory output is independent of `geo_constraint_variants`.

**Run `diffuser` under exactly one geo entry** (e.g., `no_constraint`). Comment it out
from all others, or accept the waste and just look at the `no_constraint` result.

### All variants under `no_constraint` — all equivalent to `diffuser`

`constraint_types: []` → `constraint_list` is empty → projector is a mathematical
no-op (unconstrained optimisation returns the input). Every projection variant
(`dpcc-r`, `dpcc-c`, `dpcc-t`, `gradient`, `post_processing`, `model_free`) produces
the same output as raw diffusion.

**Practical note**: run `no_constraint/diffuser` as the baseline. All other variants
under `no_constraint` are redundant — they consume compute for zero information gain.

### `model_free` — only affected by `'bounds'`/`'halfspace'`/`'obstacles'`, never by `'dynamics'`

The `model_free` variant has an explicit guard in `setup_dpcc_projector`:
```python
if 'dynamics' in constraint_types and 'model_free' not in variant:
    ...  # dynamics constraint — skipped for model_free
```

Dynamics is always skipped regardless of the geo entry's `constraint_types`. Halfspace, bounds, and obstacles are **not** guarded — they apply to `model_free` normally. So:

| Geo config | `model_free` constraint list | Equivalent to |
|---|---|---|
| `no_constraint` | empty | `diffuser` |
| `dynamics_only` | empty (dynamics skipped) | `no_constraint/model_free` |
| `bounds_only_1` | bounds only | — (unique result) |
| `halfspace_only_1` | halfspace only | — (unique result) |
| `combined_2` | bounds only (dynamics skipped) | `bounds_only_1/model_free` |
| `combined_4` | bounds + halfspace + obstacles (dynamics skipped) | — (unique result) |

**Unique results**: `bounds_only_1/model_free`, `halfspace_only_1/model_free`, `combined_4/model_free`.

### `dpcc-c-dt*` — only meaningful under `combined_2`

`dpcc-c-dt0p25/dt0p5/dt2p0/dt4p0` scale the **dynamics constraint** coefficient.
Under any geo entry without `'dynamics'` in `constraint_types`, there is no dynamics
constraint to scale:

| Geo config | `dpcc-c-dt0p25/dt0p5/dt2p0/dt4p0` result |
|---|---|
| `no_constraint` | all four == `diffuser` (empty constraints) |
| `dynamics_only` | all four differ from each other and from `dpcc-c` — **meaningful** |
| `bounds_only_1` | all four == `dpcc-c` under `bounds_only_1` (no dynamics to scale) |
| `combined_2` | all four differ — **meaningful**, this is the intended sweep |

**Run `dpcc-c-dt*` under `combined_2` only** (and its `-tightened` twin).

### `post_processing` — NOT always dynamics-enabled

`post_processing` respects `constraint_types` exactly like `dpcc-*` and `gradient`.
The only difference is projection timing (applied once post-denoising vs per-step).
Under `bounds_only_1` it has bounds only; under `combined_2` it has bounds + dynamics.

The `model_free` rule (skip dynamics) does **not** apply to `post_processing`.

---

## Which geo entry is sufficient for paper results?

| Goal | Minimum geo entries needed |
|---|---|
| Final paper numbers (DPCC-equivalent) | `combined_2` (+ tightened twin) |
| Ablation: which constraint type contributes what | `no_constraint` + `bounds_only_1` + `dynamics_only` + `combined_2` |
| Obstacle study | `obstacle_only_1/2` + `combined_3` (once geometry is measured) |

`combined_2` alone gives the full DPCC-equivalent result. The ablation entries
(`bounds_only_1`, `dynamics_only`) isolate each constraint type's contribution
and are needed for the analysis figures, not for the headline metric.

---

## Tightening control (`enlarge_constraints`)

`enlarge_constraints` is a **global** parameter (top-level yaml, follows original DPCC logic):

```yaml
enlarge_constraints: 0.01   # non-null → auto-generate *-tightened sibling folders
# enlarge_constraints: null  # disable all tightening — no -tightened folders created
```

When non-null, the geo loop **automatically** creates a `{geo_name}-tightened` run for every
active geo entry whose `constraint_types` includes `'bounds'`, `'halfspace'`, or `'obstacles'`.
`no_constraint` and `dynamics_only` are never affected (nothing to tighten).

Tightening applied (matches original DPCC):
- **Bounds**: `ws_lb += enlarge`, `ws_ub -= enlarge` — smaller workspace box
- **Halfspace**: boundary line shifted inward by `enlarge` — same as DPCC halfspace tightening
- **Obstacles**: `radius += enlarge` — larger exclusion sphere

Do **not** put `enlarge_constraints` inside individual geo entries — it is global only.

---

## Output folder structure

Active entries with `enlarge_constraints: 0.01`, default yaml (`bounds_only_1`, `halfspace_only_1`, `combined_2`):

```
results/no_constraint/dpcc-r/             ← no tightened twin (no bounds/halfspace/obstacles)
results/dynamics_only/dpcc-r/             ← no tightened twin
results/bounds_only_1/dpcc-r/             ← normal  (2D: x-y only)
results/bounds_only_1-tightened/dpcc-r/   ← auto-generated tightened twin
results/halfspace_only_1/dpcc-r/          ← normal  (2D diagonal line)
results/halfspace_only_1-tightened/dpcc-r/← auto-generated tightened twin
results/combined_2/dpcc-r/               ← normal  (3D bounds + dynamics)
results/combined_2-tightened/dpcc-r/     ← auto-generated tightened twin
```

Set `enlarge_constraints: null` to suppress all `*-tightened` folders.

---

## Disable a geo entry (skip it in eval)

Comment out the entry:

```yaml
geo_constraint_variants:
  - name: no_constraint
    constraint_types: []
  # - name: bounds_only_1        ← commented out = will not run
  #   constraint_types: ['bounds']
  - name: combined_2
    constraint_types: ['dynamics', 'bounds']
    workspace_bounds:
      lb: [0.30, -0.35, 0.05]
      ub: [0.70,  0.35, 0.40]
```

---

## Enable `bounds_only_2` (3D bounds)

`bounds_only_1` is 2D (x-y only, z unconstrained). `bounds_only_2` adds z bounds (floor + ceiling):

```yaml
  - name: bounds_only_2
    constraint_types: ['bounds']
    workspace_bounds:
      lb: [0.30, -0.35, 0.05]   # 3D: x-y-z all constrained (floor=0.05)
      ub: [0.70,  0.35, 0.40]   # 3D: ceiling=0.40 stops upward EE escape
```

Both will run and each gets a `-tightened` sibling (if `enlarge_constraints` set).

---

## Add a new geo entry

Append with a unique name:

```yaml
  - name: bounds_narrow
    constraint_types: ['bounds']
    workspace_bounds:
      lb: [0.35, -0.25, 0.08]
      ub: [0.65,  0.25, 0.35]
```

---

## Rules

| Key | Behaviour |
|---|---|
| `constraint_types: []` | Projector is a no-op — raw FM output (baseline) |
| `constraint_types: ['bounds']` | Workspace box on EE position only |
| `constraint_types: ['dynamics']` | Euler link only — no position ceiling |
| `constraint_types: ['dynamics', 'bounds']` | Box + Euler integration (DPCC-equivalent) |
| `'halfspace'` | Requires `halfspace_constraints` list; operates in EE x-y plane |
| `'obstacles'` | Requires `obstacle_constraints` list in the same geo entry |
| `workspace_bounds` absent | Required when `'bounds'` in constraint_types — no global fallback |
| `enlarge_constraints` in geo entry | **Wrong** — must be global only |
| `name` must be unique | Each active entry writes to `results/{name}/` — duplicates overwrite |
| `_has_geo` (bounds, halfspace, or obstacles) | True → tightened twin created; False → runs once only |

---

## Why `no_constraint` has no index

`constraint_types: []` has no tunable parameters — a `_2` variant would be identical.
Only configs with physical parameters (`workspace_bounds`) get the `_1`/`_2` index scheme.

---

## Halfspace vs bounds — they are NOT the same

A common misconception: **bounds ≠ halfspace**.

| | Bounds | Halfspace |
|---|---|---|
| Constraint form | `lb_i <= x_i <= ub_i` per dimension | `C @ x <= d` — arbitrary hyperplane |
| Boundary shape | axis-aligned box faces | oblique/diagonal plane at any angle |
| Example | EE x must stay in [0.30, 0.70] | EE must stay to the left of a diagonal line |

A workspace box IS 6 axis-aligned halfspaces — but halfspace adds oblique planes that
bounds cannot express. Use halfspace when a physical boundary in the scene is diagonal.

## Add a halfspace constraint

Uncomment `halfspace_only_1` or `halfspace_only_2` in the yaml and tune:

```yaml
  - name: halfspace_only_1
    constraint_types: ['halfspace']
    halfspace_constraints:
      - [[0.35, -0.35], [0.65, 0.35], 'above']
      # Format: [[x1, y1], [x2, y2], 'above'/'below']  (metres; x=forward, y=lateral)
      # 'above'/'below' = which side of the line is the feasible region
```

Tightening (`enlarge_constraints`): shifts each halfspace boundary inward — identical
to original DPCC halfspace tightening. Multiple constraints: add more list entries.

---

## Add an obstacle constraint

Uncomment `obstacle_only_1` in the yaml and tune centre/radius to the real scene:

```yaml
  - name: obstacle_only_1
    constraint_types: ['obstacles']
    obstacle_constraints:
      - type: sphere_outside
        dimensions: ['x', 'y']      # 2D cylinder (x-y plane, no z constraint) — avoiding-paper style
        center: [0.50, 0.00]        # metres — tune to measured obstacle position
        radius: 0.06                # metres — tune to measured obstacle radius
```

Or the 3D sphere variant (stricter — EE must stay outside a true sphere):

```yaml
  - name: obstacle_only_2
    constraint_types: ['obstacles']
    obstacle_constraints:
      - type: sphere_outside
        dimensions: ['x', 'y', 'z']
        center: [0.50, 0.00, 0.12]  # metres — z ≈ table surface + block height
        radius: 0.06
```

Named dims `'x'`, `'y'`, `'z'` map to trajectory indices 6, 7, 8 (EE Cartesian position).
`'dx'`/`'dy'`/`'dz'` → 0-2 (action deltas), `'des_x'`/`'des_y'`/`'des_z'` → 3-5.
Multiple obstacles: add more entries under `obstacle_constraints`.

---

## Classic DPCC (avoiding paper) vs visual aligning

**Classic DPCC** (`config/projection_eval.yaml`):

```yaml
constraint_types: ['halfspace', 'obstacles', 'dynamics', 'bounds']
enlarge_constraints: {'avoiding': 0.025}
```

| Constraint type | Avoiding (published) | Visual aligning | Reason |
|---|---|---|---|
| `halfspace` | ✅ triangular wall obstacles | ⏸ implemented, disabled | No known oblique physical boundary yet |
| `obstacles` | ✅ spherical exclusion zones | ⏸ implemented, disabled | Geometry not yet measured |
| `bounds` | ✅ 2D xy velocity/position limits | ✅ 3D xyz workspace box | Extended to 3D Cartesian |
| `dynamics` | ✅ Euler link 2D | ✅ Euler link 3D | Extended to 3D |

All four constraint types are now **fully implemented** for visual aligning.

**`combined_2`** (`['dynamics', 'bounds']`) is the current DPCC-equivalent.  
**`combined_3`** (`['dynamics', 'bounds', 'obstacles']`) — once obstacle geometry is measured.  
A `combined_4` (`['dynamics', 'bounds', 'halfspace', 'obstacles']`) would be the exact full match to the avoiding paper.

### How to replicate DPCC-equivalent results

`combined_2` is already active in the default yaml — no changes needed:

```yaml
  - name: combined_2
    constraint_types: ['dynamics', 'bounds']
    workspace_bounds:
      lb: [0.30, -0.35, 0.05]   # x forward, y lateral, z vertical (metres)
      ub: [0.70,  0.35, 0.40]
```

With `enlarge_constraints: 0.01` set globally, the eval also produces
`results/combined_2-tightened/` — the tightened-constraint counterpart to
`dpcc-c-tightened` in the original paper's Table 1.
