# UF-16: Debug / Test Mode — Relaxed Constraints + Silent Projector

**Date**: 2026-05-27  
**Branch**: `update_into_FM`  
**Scope**: `config/visual_aligning_eval.yaml`, `diffuser_visual_aligning/sampling/projection.py`, `fm_visual_aligning/sampling/projection.py`

---

## Motivation

Running `combined_4` (dynamics + bounds + halfspace + obstacles) with the nominal constraint geometry produced two problems during early debug runs:

1. **Constraints too tight for initial testing** — the workspace bounds cut close to the real EE motion range and the halfspace divided the workspace roughly in half, so the SLSQP projector was heavily active on nearly every sample.  Trajectories were heavily modified before any real policy behaviour could be assessed.

2. **Console unreadable** — the per-sample projector diagnostic print (added in Fix 9.3) fired on every sample whenever the projector was active, producing hundreds of lines like:
   ```
   [ projector ] sample 0: SLSQP delta=1.526379 success=False nit=11 status=8
   ```
   These drowned all other eval output (rollout counters, timing, diagnostics).

UF-16 addresses both issues as explicit **test/debug settings**, preserving the original values in commented-out lines for easy restoration.

---

## UF-16.1 — Relaxed constraints in `visual_aligning_eval.yaml`

### Workspace bounds — `combined_4`

The blue bounding box is widened so the EE rarely hits a wall during normal motion:

| | `lb` | `ub` |
|---|---|---|
| **Original** (nominal) | `[0.30, -0.35, 0.05]` | `[0.70, 0.35, 0.40]` |
| **Relaxed** (test) | `[0.20, -0.45, 0.02]` | `[0.80, 0.45, 0.50]` |

Changes: ±0.10 m wider in x and y; z floor lowered by 0.03 m; z ceiling raised to 0.50 m.  The original lines are preserved as comments directly below.

### Halfspace — `combined_4`

The halfspace line is pushed to the lower edge of the (relaxed) workspace so only a ~0.08 m strip at the very bottom of the y range is forbidden:

| | Line points | Effect |
|---|---|---|
| **Original** (moderate) | `[[0.30,-0.05],[0.70,0.05],'above']` | Line at y ≈ 0; forbids lower ~50% of y range |
| **Relaxed** (test) | `[[0.20,-0.38],[0.80,-0.30],'above']` | Line at y ≈ −0.34; forbids only a ~0.08 m strip at workspace bottom |

The original line is preserved as a comment directly below.

### How to restore nominal constraints

Swap the active/commented lines inside the `combined_4` entry in `visual_aligning_eval.yaml`:

```yaml
# comment out relaxed lines:
#   lb: [0.20, -0.45, 0.02]
#   ub: [0.80,  0.45, 0.50]
#   - [[0.20, -0.38], [0.80, -0.30], 'above']

# uncomment original lines:
    lb: [0.30, -0.35, 0.05]
    ub: [0.70,  0.35, 0.40]
    - [[0.30, -0.05], [0.70, 0.05], 'above']
```

---

## UF-16.2 — `active_geo_variants` selector in yaml

### Problem

Choosing which geo constraint configurations to run required manually commenting and uncommenting large `- name:` blocks inside `geo_constraint_variants`.  With 11 entries, a single-experiment test meant scrolling through ~150 lines of yaml and toggling dozens of lines — error-prone and slow.

### Solution

Added a top-level `active_geo_variants` key in the geo constraint section of `config/visual_aligning_eval.yaml` (placed immediately after `enlarge_constraints`):

```yaml
active_geo_variants: [combined_4]
# null → run all defined entries
# Any subset: [no_constraint, dynamics_only, combined_4]
```

The eval scripts (`eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`) filter `_geo_specs` against this list before building `_run_items`:

```python
_active_names = config.get('active_geo_variants')
if _active_names is not None:
    _active_set = set(_active_names)
    _geo_specs  = [gs for gs in _geo_specs if gs['name'] in _active_set]
    print(f'\n[ geo ] active_geo_variants: {[gs["name"] for gs in _geo_specs]}')
```

`null` is fully backwards-compatible — all entries run as before.

### All geo entries uncommented

All ready entry definitions in `geo_constraint_variants` were uncommented so any combination can be activated without touching the definitions:

| Entry | Status |
|---|---|
| `no_constraint` | active (always was) |
| `dynamics_only` | active (always was) |
| `bounds_only_1` | **uncommented** |
| `bounds_only_2` | **uncommented** |
| `obstacle_only_1` | **uncommented** |
| `obstacle_only_2` | **uncommented** |
| `halfspace_only_1` | **uncommented** |
| `combined_1` | **uncommented** |
| `combined_2` | **uncommented** |
| `combined_3` | **uncommented** |
| `combined_4` | active (always was) |
| `halfspace_only_2` | stays commented — 3D `normal`/`offset` format not yet implemented in code |

### Dead code removed

The top-level `constraint_types: ['bounds', 'dynamics']` fallback line was removed.  It was only used when `geo_constraint_variants` was absent — dead since the geo loop was introduced.

### `_has_geo` fix

`_has_geo` was checking `_gs['constraint_types']` (per-entry definition, before tightened twin generation).  Changed to `_gc['constraint_types']` (the effective value) so tightened twins are correctly recognised as having geometry.

---

## UF-16.3 — Suppressed SLSQP per-sample print

### Files changed

- `diffuser_visual_aligning/sampling/projection.py` (line ~204)
- `fm_visual_aligning/sampling/projection.py` (line ~204)

### What was suppressed

The Fix 9.3 diagnostic block that printed one line per batch sample whenever the projector modified a trajectory:

```python
# Fix 9.3: log when SLSQP meaningfully modifies the trajectory
delta = np.linalg.norm(sol_np[i] - trajectory_np[i])
if delta > 1e-4:
    print(f'[ projector ] sample {i}: SLSQP delta={delta:.6f} '
          f'success={res.success} nit={res.nit} status={res.status}')
```

This fires on every sample that has an active constraint — typically all B samples in every replanning step when `combined_4` is active — producing O(B × T / stride) lines per rollout.

### How to re-enable for debugging

Uncomment the four lines (they are left in place as comments). To reduce volume while still catching real failures, add a `success=False` guard:

```python
delta = np.linalg.norm(sol_np[i] - trajectory_np[i])
if delta > 1e-4 and not res.success:
    print(f'[ projector ] sample {i}: SLSQP delta={delta:.6f} '
          f'success={res.success} nit={res.nit} status={res.status}')
```

`status=8` ("Positive directional derivative for linesearch") is the most common failure mode when constraints are tight and the initial point is far from feasible — expected during the current test phase with relaxed constraints not yet tuned to the real scene.

---

## Changed files summary

| File | Change |
|---|---|
| `config/visual_aligning_eval.yaml` | UF-16.1: Relaxed `workspace_bounds` and `halfspace_constraints` in `combined_4`; originals commented out |
| `config/visual_aligning_eval.yaml` | UF-16.2: Added `active_geo_variants` selector; uncommented all ready geo entries; removed dead `constraint_types` fallback |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | UF-16.2: Filter `_geo_specs` by `active_geo_variants`; fix `_has_geo` to use `_gc['constraint_types']` |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | UF-16.2: Same as DPCC eval |
| `diffuser_visual_aligning/sampling/projection.py` | UF-16.3: Commented out Fix 9.3 per-sample SLSQP print |
| `fm_visual_aligning/sampling/projection.py` | UF-16.3: Same |
