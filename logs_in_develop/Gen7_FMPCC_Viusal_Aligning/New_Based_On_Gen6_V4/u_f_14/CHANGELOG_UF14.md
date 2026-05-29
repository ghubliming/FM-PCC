# UF-14: Geo Constraint Sweep — Outer Loop over Geometric Constraint Configurations

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`, `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`  
**Guide**: [CONSTRAINTS_GUIDE.md](CONSTRAINTS_GUIDE.md) — constraint config how-to + redundant-run analysis

---

## Motivation

Prior to this update, `constraint_types` was a single top-level yaml key applied uniformly to every evaluation run. This made it impossible to compare DPCC projection under different geometric constraint configurations without manually editing the yaml between runs.

The avoiding paper (published) produces results under multiple obstacle configurations via an outer `halfspace_variants` loop, generating separate output subtrees per configuration. UF-14 brings the same structure to visual aligning: a `geo_constraint_variants` outer loop produces a separate `results/{name}/` subtree for each constraint configuration, so all configurations run in a single job and are directly comparable.

---

## Output path structure

**Before UF-14:**
```
results/diffuser/
results/dpcc-r/
results/dpcc-c/
...
```

**After UF-14:**
```
results/no_constraint/diffuser/
results/no_constraint/dpcc-r/
results/dynamics_only/diffuser/
results/bounds_only_1/diffuser/
results/combined_2/diffuser/
...
```

Old runs without `geo_constraint_variants` in the yaml are **not affected** — the eval script falls back to a single `bounds_dynamics_1` iteration using the top-level `constraint_types`, producing the same path structure as before.

---

## Geo constraint naming scheme

| Tier | Name pattern | Index | Active by default |
|---|---|---|---|
| Baseline | `no_constraint` | none | ✅ |
| Single-type ablation | `dynamics_only` | none | ✅ |
| Single-type ablation | `bounds_only_1/2` | `_1/_2` | ✅ `_1` / commented `_2` |
| Single-type ablation | `obstacle_only_1/2` | `_1/_2` | ❌ both commented (geometry not measured) |
| Combination | `combined_1/2/3` | N | ✅ `combined_2` / ❌ obstacle ones commented |

- **No index** — no tunable geometric parameters (`no_constraint`, `dynamics_only`)
- **`_1/_2`** — parameter variants: `_1` = current baseline, `_2` = tuning placeholder with all parameters listed inline
- **`combined_N`** — flexible numbered slots; content defined entirely by `constraint_types` (and `obstacle_constraints` if needed)

### Active config set

| Name | `constraint_types` | Notes |
|---|---|---|
| `no_constraint` | `[]` | Raw FM baseline |
| `dynamics_only` | `['dynamics']` | Euler link only |
| `bounds_only_1` | `['bounds']` | Workspace box only |
| `combined_2` | `['dynamics', 'bounds']` | DPCC-equivalent for this task |

### Commented-out configs (disabled)

| Name | `constraint_types` | Reason |
|---|---|---|
| `bounds_only_2` | `['bounds']` | Parameter tuning placeholder |
| `obstacle_only_1` | `['obstacles']` | Obstacle geometry not measured |
| `obstacle_only_2` | `['obstacles']` | 3D sphere variant placeholder |
| `combined_1` | `['dynamics', 'obstacles']` | Obstacle geometry not measured |
| `combined_3` | `['dynamics', 'bounds', 'obstacles']` | Obstacle geometry not measured |

---

## Classic DPCC vs visual aligning constraint mapping

Classic DPCC (`config/projection_eval.yaml`): `['halfspace', 'obstacles', 'dynamics', 'bounds']`

| Type | Avoiding (published) | Visual aligning | Reason |
|---|---|---|---|
| `halfspace` | ✅ triangular walls | ❌ | Open table, no diagonal walls |
| `obstacles` | ✅ spherical obstacles | ⏸ designed, disabled | Geometry not measured yet |
| `bounds` | ✅ 2D xy limits | ✅ 3D xyz box | Extended to 3D |
| `dynamics` | ✅ Euler link 2D | ✅ Euler link 3D | Extended to 3D |

`combined_2` (`['dynamics', 'bounds']`) is the current DPCC-equivalent for this task.
`combined_3` (`['dynamics', 'bounds', 'obstacles']`) will be the full equivalent once obstacle geometry is confirmed.

---

---

## Revision A — Final fixes applied after initial implementation (2026-05-26)

Three follow-up fixes applied after the original UF-14 implementation, plus one significant redesign of the tightening mechanism.

### Fix 1 — `ws_lb`/`ws_ub` crash for no-bounds entries

`setup_dpcc_projector` read `config['workspace_bounds']` unconditionally before checking `constraint_types`. Entries without `'bounds'` (e.g., `no_constraint`, `dynamics_only`) would crash with `KeyError`. Fixed: moved `ws_lb`/`ws_ub` reading inside the `if 'bounds' in constraint_types:` block.

### Fix 2 — `enlarge_constraints` as global top-level (original DPCC logic)

Original DPCC (`scripts/eval.py`) reads `enlarge_constraints` once globally before any loop. It controls tightening for the full run — not per-experiment. UF-14 initially placed `enlarge_constraints: 0.01` inside individual geo entries (`bounds_only_1`, `combined_2`), which was wrong. Fixed: moved to top-level global. Per-geo entries no longer carry `enlarge_constraints`; `dict(config)` inheritance propagates the global automatically.

### Fix 3 — Fallback `_geo_specs` cleaned up

Removed `'enlarge_constraints': 0.01` from the fallback `_geo_specs` in both eval scripts. The global value is inherited via `dict(config)`.

---

## Redesign — Geo-level tightened auto-generation (2026-05-26)

**Motivation**: In UF-14's initial design, `-tightened` was a projection-variant suffix (`dpcc-r-tightened`, `gradient-tightened`, etc.), requiring every projection variant to be listed twice. Tightening is a property of the constraint setup, not of the projection method — it belongs at the geo level.

**New design**: the geo loop auto-generates a `{geo_name}-tightened` sibling for every geo entry whose `constraint_types` includes `'bounds'` or `'obstacles'`, when `enlarge_constraints` is non-null.

### Output path change

**Before redesign (UF-14 initial):**
```
results/bounds_only_1/dpcc-r/
results/bounds_only_1/dpcc-r-tightened/      ← tightening on projection variant
results/bounds_only_1/dpcc-c-tightened-dt0p25/
```

**After redesign:**
```
results/bounds_only_1/dpcc-r/                ← normal geo
results/bounds_only_1-tightened/dpcc-r/      ← tightened geo (auto-generated)
results/bounds_only_1-tightened/dpcc-c-dt0p25/
```

### Entries NOT affected
`no_constraint` and `dynamics_only` have no `'bounds'` or `'obstacles'` in `constraint_types` — `_has_geo` is False, so no tightened twin is created. They always run once only.

### Disable tightening
Set `enlarge_constraints: null` in the yaml. `_enlarge` becomes Python `None` → no tightened twins, no `-tightened` folders.

### Obstacle tightening — matches original DPCC
`radius + enlarge_constraints` (larger exclusion sphere). Bounds tightening: `ws_lb += enlarge`, `ws_ub -= enlarge` (smaller workspace box, our 3D extension).

### `projection_variants` cleanup
All `-tightened` variants removed. `dpcc-c-tightened-dt*` renamed to `dpcc-c-dt*` (tightened folder auto-generated at geo level).

---

## Changed Files (final state after all revisions)

### `config/visual_aligning_eval.yaml`

- 3-tier `geo_constraint_variants`: Baseline / Single-type ablation / Combinations.
- Active entries: `no_constraint`, `dynamics_only`, `bounds_only_1`, `combined_2`.
- Commented-out entries: `bounds_only_2`, `obstacle_only_1`, `obstacle_only_2`, `combined_1`, `combined_3`.
- Top-level `constraint_types: ['bounds', 'dynamics']` retained as fallback.
- Top-level `enlarge_constraints: 0.01` (global — null disables all tightening).
- `projection_variants`: all `-tightened` entries removed; `dpcc-c-tightened-dt*` → `dpcc-c-dt*`.

### `fm_visual_aligning_test/eval_fm_visual_aligning.py`

- `setup_dpcc_projector(... variant, is_tightened=False)` — `is_tightened` flag replaces `'tightened' in variant` check.
  - Bounds: `ws_lb += enlarge` / `ws_ub -= enlarge` when `is_tightened`.
  - Obstacles: `radius + enlarge` when `is_tightened` (matches original DPCC).
  - `ws_lb`/`ws_ub` read inside `if 'bounds'` block only — no crash for no-bounds entries.
- Geo loop (`_run_items`): 4-tuple `(geo_name, geo_config, geo_variant, is_tightened)`.
  - `_enlarge = config.get('enlarge_constraints')` — None when yaml null.
  - For each geo entry: normal pass always; tightened pass only when `_enlarge is not None and _has_geo`.
  - `_has_geo = any(t in constraint_types for t in ('bounds', 'obstacles'))`.
  - `obstacle_constraints` now transferred from geo spec to `_gc`.
  - Fallback `_geo_specs` no longer carries `enlarge_constraints` (global inherited via `dict(config)`).
- `setup_dpcc_projector` call: passes `is_tightened`.

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Identical changes as FM eval above.

---

## Revision B — Halfspace, 2D/3D bounds scheme, constraint loading analysis (2026-05-26)

### B1 — Halfspace constraint support (2D)

`halfspace` is now a first-class constraint type in both eval scripts, on par with `bounds` and `obstacles`.

**`setup_dpcc_projector` halfspace block** (after dynamics, before obstacles):
```python
if 'halfspace' in config.get('constraint_types', []):
    tightening = config.get('enlarge_constraints') or 0.0
    _hs_indices = {'x': _DIM['x'], 'y': _DIM['y']}   # EE x=6, y=7 in 9D trajectory
    for hs in config.get('halfspace_constraints', []):
        margin = tightening if is_tightened else 0.0
        C_row, d = utils.formulate_halfspace_constraints(hs, margin, 9, _hs_indices)
        constraint_list.append(('ineq', (C_row, d)))
```

`utils.formulate_halfspace_constraints` was already implemented in
`fm_visual_aligning/utils/constraints_helpers.py` (exported via `__init__.py`).
The Projector already supports `('ineq', (C_row, d))` format (projection.py line 58).
Tightening shifts the halfspace boundary inward by `enlarge_constraints` — matches original DPCC halfspace tightening.

**`_has_geo`** updated: `any(t in constraint_types for t in ('bounds', 'halfspace', 'obstacles'))` — halfspace now triggers tightened-twin generation.

**Geo loop**: `halfspace_constraints` transferred from geo spec to `_gc`, alongside `obstacle_constraints` and `workspace_bounds`.

**3D halfspace** — NOT implemented. 3D requires a plane normal + offset (not a two-point 2D line). A `halfspace_only_2` placeholder is in the yaml but disabled — implement before enabling.

### B2 — 2D/3D bounds scheme for all standalone ablation configs

Single-type ablation configs default to 2D (x-y only, z=±∞). The full 3D version is the `_2` variant:

| Entry | Bounds | z |
|---|---|---|
| `bounds_only_1` | `lb: [0.30,-0.35,-.inf]`, `ub: [0.70,0.35,.inf]` | unconstrained (2D) |
| `bounds_only_2` | `lb: [0.30,-0.35,0.05]`, `ub: [0.70,0.35,0.40]` | floor+ceiling (3D) |
| `halfspace_only_1` | 2D line in x-y plane | z unconstrained |
| `halfspace_only_2` | 3D plane (pending) | — |
| `combined_2` | `lb: [0.30,-0.35,0.05]`, `ub: [0.70,0.35,0.40]` | 3D intentionally |

`combined_2` keeps 3D bounds (floor/ceiling active) — the combination config is the full physical model. Ablation configs use 2D to isolate x-y constraint effect only.

2D = 3D with z=±∞. PyYAML parses `-.inf` / `.inf` as Python `float('-inf')` / `float('inf')`. The projector handles inf bounds correctly (no z constraint applied).

### B3 — New geo entries added to yaml

- `halfspace_only_1` — 2D halfspace, active for debugging (verifies halfspace code path)
- `halfspace_only_2` — 3D halfspace, commented (pending implementation)
- `combined_4` — `['dynamics', 'bounds', 'halfspace', 'obstacles']`, exact match to original avoiding paper when obstacle geometry is measured

**Current yaml state**: `combined_4` is the only active entry (all others commented). Obstacle geometry in `combined_4` uses placeholder values — not real measurements. For a valid DPCC-equivalent run, comment out `combined_4` and uncomment `combined_2`.

### B4 — Investigation: original DPCC constraint loading vs our design

See `UF14_investigation_constraint_loading.md` for full analysis. Summary:

**Original DPCC (brittle)**:
- `scripts/eval.py` selects constraints by hard-coded integer indices into flat YAML lists (`[0]`, `[3]`, `[4]`...).
- Commenting out one YAML list item shifts all subsequent indices — silently loads wrong constraints.
- `bounds` dict defined with duplicate lower/upper entries for the same dimensions, creating redundant constraints fed to the solver for every variant.

**Our design (safe)**:
- Each `geo_constraint_variants` entry is a named dict carrying its own complete constraint specification inline.
- The geo loop iterates ALL items in `halfspace_constraints` / `obstacle_constraints` with no index picking.
- A single `workspace_bounds` lb/ub replaces the redundant bounds list.
- YAML entries can be freely commented/uncommented without affecting other entries' indexing.

No code changes required — our existing design already avoids both issues identified in the investigation.
