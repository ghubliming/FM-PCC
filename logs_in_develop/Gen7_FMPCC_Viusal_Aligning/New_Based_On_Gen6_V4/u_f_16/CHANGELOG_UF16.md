# UF-16: Full Changelog (16.1 – 16.7)

**Branch**: `update_into_FM`  
**Dates**: 2026-05-27 (UF-16.1–16.6) · 2026-05-28 (UF-16.7)  
**Scope**: `config/visual_aligning_eval.yaml`, `d3il/simulation/aligning_sim.py`,  
&emsp;&emsp;&emsp;&emsp;`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`,  
&emsp;&emsp;&emsp;&emsp;`fm_visual_aligning_test/eval_fm_visual_aligning.py`,  
&emsp;&emsp;&emsp;&emsp;`diffuser_visual_aligning/sampling/projection.py`,  
&emsp;&emsp;&emsp;&emsp;`fm_visual_aligning/sampling/projection.py`

---

## UF-16.1 — Relaxed constraints in `visual_aligning_eval.yaml`

### Motivation

Nominal `combined_4` constraints were too tight for initial debug runs: the SLSQP
projector was heavily active on nearly every sample, so trajectories were heavily
modified before any real policy behaviour could be assessed.

### Workspace bounds — `combined_4`

| | `lb` | `ub` |
|---|---|---|
| **Original** (nominal) | `[0.30, -0.35, 0.05]` | `[0.70, 0.35, 0.40]` |
| **Relaxed** (test/debug) | `[0.20, -0.45, 0.02]` | `[0.80, 0.45, 0.50]` |

±0.10 m wider in x and y; z floor −0.03 m; z ceiling +0.10 m.
Original lines preserved as comments directly below each relaxed line.

### Halfspace — `combined_4`

| | Line points | Effect |
|---|---|---|
| **Original** | `[[0.30,-0.05],[0.70,0.05],'above']` | Line at y ≈ 0; forbids lower ~50% of y range |
| **Relaxed** | `[[0.20,-0.38],[0.80,-0.30],'above']` | Line at y ≈ −0.34; forbids only ~0.08 m strip at workspace bottom |

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

## UF-16.2 — `active_geo_variants` selector in YAML

### Problem

Choosing which geo constraint configurations to run required manually commenting and
uncommenting large `- name:` blocks — error-prone and slow across 11 entries.

### Solution

Added `active_geo_variants` key in `config/visual_aligning_eval.yaml`:

```yaml
active_geo_variants: [combined_4]
# null → run all defined entries
# Any subset: [no_constraint, dynamics_only, combined_4]
```

Both eval scripts filter `_geo_specs` against this list before building `_run_items`:

```python
_active_names = config.get('active_geo_variants')
if _active_names is not None:
    _active_set = set(_active_names)
    _geo_specs  = [gs for gs in _geo_specs if gs['name'] in _active_set]
    print(f'\n[ geo ] active_geo_variants: {[gs["name"] for gs in _geo_specs]}')
```

`null` is fully backwards-compatible — all entries run as before.

### All geo entries uncommented

All ready entry definitions in `geo_constraint_variants` were uncommented so any
combination can be activated directly via the selector list:

| Entry | Status before | Status after |
|---|---|---|
| `no_constraint` | active | active |
| `dynamics_only` | active | active |
| `bounds_only_1` | commented | **uncommented** |
| `bounds_only_2` | commented | **uncommented** |
| `obstacle_only_1` | commented | **uncommented** |
| `obstacle_only_2` | commented | **uncommented** |
| `halfspace_only_1` | commented | **uncommented** |
| `combined_1` | commented | **uncommented** |
| `combined_2` | commented | **uncommented** |
| `combined_3` | commented | **uncommented** |
| `combined_4` | active | active |
| `halfspace_only_2` | commented | **stays commented** — 3D normal/offset format not yet implemented |

### Other changes

- Dead top-level `constraint_types: ['bounds', 'dynamics']` fallback removed (unused
  since `geo_constraint_variants` was introduced).
- `_has_geo` fixed: was checking `_gs['constraint_types']` (pre-twin-generation), changed
  to `_gc['constraint_types']` (effective value) so tightened twins are correctly recognised.

---

## UF-16.3 — Constraint Satisfaction / Violation Metrics

**Metric reference**: [CONSTRAINT_METRICS.md](CONSTRAINT_METRICS.md)

### Motivation

Before UF-16.3 the eval pipeline reported task-level metrics only. UF-16.3 adds
quantitative constraint compliance measurement at two levels:

| Level | Question answered |
|---|---|
| **Execution** | Did the real executed `c_pos_history` stay inside bounds / halfspace / obstacles? |
| **Planning** | Did the post-projection planned trajectories satisfy constraints? (non-zero → SLSQP did not fully converge) |

### New module-level functions (both eval files)

**`check_trajectory_constraints(c_pos_traj, act_traj, geo_config, enlarge)`**  
Vectorised NumPy check of a `(T, 3)` actual EE trajectory. Returns 15 `exec_*` metrics
(JSON-serialisable). Works in physical metres; handles `±inf` bounds; `enlarge` applies
the same tightening margin as the projector.

**`_check_planned_violations(cands_xyz, geo_config, enlarge)`**  
Checks `(B, H, 3)` unnormalised planned candidates (post-projection) against bounds +
halfspace + obstacles. Returns fraction of `(sample, horizon_step)` pairs still violating.

### `VisualAgentWrapper` additions

| Location | Change |
|---|---|
| `__init__` | `history_constraint_metrics = []`, `_plan_post_viol_rates = []` |
| `reset()` | Both lists cleared |
| `predict()` (after unnorm) | Calls `_check_planned_violations`, appends to `_plan_post_viol_rates` |
| `update_rollout_info()` | Calls `check_trajectory_constraints` at rollout end; stores in `master_rollout_history` and `history_constraint_metrics` |
| `_export_rollout_realtime()` | Adds `constraint_metrics` to per-rollout JSON; prints 3-line summary |
| Eval summary block | Prints aggregate table (mean ± std for all metrics); saves `constraint_metrics.json` |

### Console output (per rollout)

```
  [ constraints ] sat=0.923  violated=12steps  (bounds=8 hs=4 obs=0)
    first_viol_step=47  longest_safe=183  margin=0.0312m  dyn_err=0.0021m
    plan_post_viol_rate=0.0082  zero_viol=False
```

### New output file per variant

`{variant}/constraint_metrics.json` — cross-rollout aggregate with mean ± std for every
metric and per-rollout list.

### Post-release fixes (2026-05-27)

**Fix 1 — halfspace sign in `_check_planned_violations`**  
Changed `x1 -= enlarge * nx` → `x1 += enlarge * nx` (and y). Previous sign moved the
halfspace boundary in the infeasible direction for tightened runs, making the check
evaluate a looser halfspace than the projector actually enforced.

**Fix 2 — exec metrics always check against nominal boundary**  
`check_trajectory_constraints` now called with `enlarge=0.0` for **all** variants,
matching the original DPCC `eval.py` convention. This makes cross-variant comparison fair
(tightened variant expected to show better `exec_constraint_sat_rate` because its
planned trajectories have a δ buffer over the nominal boundary).

| Function | `enlarge` used | Question answered |
|---|---|---|
| `check_trajectory_constraints` | always `0` | Was the real trajectory safe vs nominal? |
| `_check_planned_violations` | δ for tightened, 0 for nominal | Did the projector succeed at the boundary it was given? |

---

## UF-16.4 — Final box position + angle logging

### Motivation

The eval console showed box **init** and target XY/angle but not where the box **ended up**
after the rollout — impossible to tell at a glance whether the policy moved the box toward
or away from the target.

### Changes

**`d3il/simulation/aligning_sim.py`** — after the rollout while-loop:
```python
_fbox_pos  = env.scene.get_obj_pos(env.push_box)
_fbox_quat = env.scene.get_obj_quat(env.push_box)
agent.update_rollout_info({**info, 'context': context,
                            'final_box_pos':  _fbox_pos,
                            'final_box_quat': _fbox_quat})
```

**Both eval files** — `update_rollout_info()`:
```python
_final_angle_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
_final_xy_dist   = np.sqrt((_fx - target_x)**2 + (_fy - target_y)**2)
curr_context_info.update({
    'final_box_xy':        [_fx, _fy],
    'final_box_angle_deg': _final_angle_deg,
    'final_xy_dist':       _final_xy_dist,
})
```

These flow automatically into per-rollout console print, `rollout_N_stats.json`, and
`history_context_info`.

New console line:
```
  - Box  final XY=(0.501,  0.330)  angle=-57.1°  (dist_to_target: 0.0043 m)
```

**Angle formula**: for pure Z-rotation quats `[cos(θ/2), 0, 0, sin(θ/2)]`, the standard
formula `arctan2(2*(w*z + x*y), 1 - 2*(y²+z²))` reduces exactly to θ — verified exact
for this convention.

---

## UF-16.5 — Success / Mode / Angle explainer

New document: `SUCCESS_MODE_ANGLE_EXPLAINER.md` (kept in this folder).

Documents the three most-misinterpreted eval outputs:

| Field | Source | Meaning |
|---|---|---|
| `Success status` | `_check_early_termination()` | pos ≤ 1.8 cm AND rot ≤ 8.6° at the SAME step |
| `Environment Mode` | `check_mode()` | 0 = robot ≤ 5.1 cm from box at last step (in contact); 1 = moved away. Mode 0 is expected for a push task |
| `Final Mean Distance` | `0.5*(3D_pos_m + rot/π)` | Mixed-unit D3IL metric; success boundary ≈ 0.033 |
| `Box final angle` | Z-euler from quat | Exact for pure Z rotations |

No bugs found — logic is correct as-is.

---

## UF-16.6 — Dual-boundary visualization for tightened variants

### Motivation

MPC decision-point plots for tightened variants (`dpcc-t`) showed only the **tightened
planning boundary**. The **nominal boundary** (what execution metrics check against) was
invisible, making the safety margin δ imperceptible visually.

### Design

Two constraint layers per shape for tightened runs:

| Layer | XY style | 3D style | Meaning |
|---|---|---|---|
| Nominal (`enlarge=0`) | Solid edge + light fill | Solid wireframe / filled plane / filled sphere | Real evaluation boundary |
| Planning (`enlarge=δ`) | Dashed edge, no fill | Dashed wireframe / quad edges / wireframe circles | What SLSQP projected against |

Single layer for non-tightened runs — unchanged appearance.

### Key changes

`_hs_xy_draw(ax, hs, enlarge, xlim, ylim, dashed=False)` — new `dashed` param; skips
fill polygon and annotation when `True`.

`_c_layers` variable drives both XY and 3D overlay blocks:
```python
_c_layers = [(0.0, False)]
if self.is_tightened and _enlarge > 0:
    _c_layers.append((_enlarge, True))
```

Legend addition (tightened only):
```python
solid  → 'nominal constraint (eval boundary)'
dashed → 'planning constraint (δ=X.XXXm inside)'
```

---

## UF-16.7 — YAML comment corrections

`config/visual_aligning_eval.yaml` — outdated comments corrected:

| Location | Before | After |
|---|---|---|
| `n_trials` comment | hardcoded `n_contexts: 30` | `n_contexts` (live key reference) |
| `active_geo_variants` list | missing `combined_5` | added `\| combined_5` |
| `bounds_only_2` description | `commented until needed` | `both entries defined and selectable` |
| `halfspace_only_1` description | `ACTIVE for debugging` | `defined; selectable for debugging` |
| Combined slots header | `Obstacle-involving slots disabled until…` | `use PLACEHOLDER geometry until…` |
| `combined_2` | `[active — DPCC-equivalent]` | `[defined; not currently active — see active_geo_variants]` |
| `combined_4` | `[disabled — full DPCC equiv]` | `[defined; not currently active]` + UF-16.1 relaxed-constraints note |
| `combined_5` | bare two-line description | full description with `[CURRENTLY ACTIVE]` tag |

---

## Consolidated changed files

| File | Changes |
|---|---|
| `config/visual_aligning_eval.yaml` | UF-16.1: relaxed `combined_4` bounds/halfspace; UF-16.2: `active_geo_variants`, uncommented entries, removed dead fallback; UF-16.7: comment corrections |
| `d3il/simulation/aligning_sim.py` | UF-16.4: read final box pos+quat, pass into info dict |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | UF-16.2: filter + `_has_geo` fix; UF-16.3: constraint metrics functions + integration; UF-16.4: final box logging; UF-16.6: dual-boundary vis |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Same as DPCC eval for all sub-items |
| `diffuser_visual_aligning/sampling/projection.py` | UF-16.3: Fix 9.3 per-sample SLSQP print suppressed |
| `fm_visual_aligning/sampling/projection.py` | UF-16.3: Same |
| `u_f_15_constrainst_visual/TIGHTENING_CONVENTION.md` | UF-16.3 Fix 2: TL;DR table added; sign bug section expanded |
