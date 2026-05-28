# UF-16 Applied to Gen6V4 (DPCC) — Full Changelog (16.1 – 16.7)

**Branch**: `update_into_FM`  
**Dates**: 2026-05-27 (UF-16.1–16.6) · 2026-05-28 (UF-16.7)  
**Source (Gen7 canonical)**: [`logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16/CHANGELOG_UF16.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16/CHANGELOG_UF16.md)  
**Metric reference**: [`u_f_16/CONSTRAINT_METRICS.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16/CONSTRAINT_METRICS.md)  
**Scope**: `config/visual_aligning_eval.yaml`, `d3il/simulation/aligning_sim.py`,  
&emsp;&emsp;&emsp;&emsp;`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`,  
&emsp;&emsp;&emsp;&emsp;`diffuser_visual_aligning/sampling/projection.py`

> **No DPCC-specific divergence** from the FM version across all sub-items.
> All changes are shared config/infrastructure applied identically to both eval stacks.
> For full derivation and design rationale, see the Gen7 source MD above.

---

## UF-16.1 — Relaxed constraints in `visual_aligning_eval.yaml`

Nominal `combined_4` constraints were too tight for initial debug runs: SLSQP projector
was active on every sample, making it impossible to assess raw policy behaviour.

**Workspace bounds** widened by ±0.10 m in x/y, z floor −0.03 m, z ceiling +0.10 m:

| | `lb` | `ub` |
|---|---|---|
| Original (nominal) | `[0.30, -0.35, 0.05]` | `[0.70, 0.35, 0.40]` |
| Relaxed (debug) | `[0.20, -0.45, 0.02]` | `[0.80, 0.45, 0.50]` |

**Halfspace** pushed to near the lower-y workspace edge (only ~0.08 m strip forbidden):
- Original: `[[0.30,-0.05],[0.70,0.05],'above']` — cuts ~50% of y range
- Relaxed:  `[[0.20,-0.38],[0.80,-0.30],'above']` — minimal restriction

Original values preserved as YAML comments directly below each relaxed line.

---

## UF-16.2 — `active_geo_variants` selector in YAML

Added `active_geo_variants` list key in `config/visual_aligning_eval.yaml`.
`null` = run all entries (backwards-compatible). Any subset runs only those named variants.

Both eval scripts filter `_geo_specs` against the list before building `_run_items`:
```python
_active_names = config.get('active_geo_variants')
if _active_names is not None:
    _active_set = set(_active_names)
    _geo_specs  = [gs for gs in _geo_specs if gs['name'] in _active_set]
```

All ready `geo_constraint_variants` entries uncommented so any can be activated via
the selector list. Dead top-level `constraint_types` fallback removed. `_has_geo` fixed:
was checking `_gs['constraint_types']` (pre-twin), changed to `_gc['constraint_types']`
(effective value) so tightened twins are correctly recognised.

`halfspace_only_2` kept commented — 3D normal/offset format not yet implemented.

---

## UF-16.3 — Constraint Satisfaction / Violation Metrics

### SLSQP per-sample print suppressed

`diffuser_visual_aligning/sampling/projection.py` — Fix 9.3 diagnostic block that
printed one line per sample per replanning step commented out. Produced O(B × T / stride)
lines per rollout, flooding the console. Lines kept as comments for easy re-enable.

### New helper functions

**`check_trajectory_constraints(c_pos_traj, act_traj, geo_config, enlarge)`**  
Vectorised NumPy check of a `(T, 3)` actual EE trajectory. Returns 15 `exec_*` metrics
(JSON-serialisable). Works in physical metres; handles `±inf` bounds.

**`_check_planned_violations(cands_xyz, geo_config, enlarge)`**  
Checks `(B, H, 3)` planned candidates (post-projection) against bounds + halfspace +
obstacles. Returns fraction of `(sample, horizon_step)` pairs still violating.

### `VisualAgentWrapper` additions

| Location | Change |
|---|---|
| `__init__` | `history_constraint_metrics = []`, `_plan_post_viol_rates = []` |
| `reset()` | Both lists cleared |
| `predict()` | Calls `_check_planned_violations` after unnorm; appends to `_plan_post_viol_rates` |
| `update_rollout_info()` | Calls `check_trajectory_constraints` at rollout end |
| `_export_rollout_realtime()` | Adds `constraint_metrics` to per-rollout JSON; prints 3-line summary |
| Eval summary block | Prints aggregate table; saves `constraint_metrics.json` |

### Post-release fixes (2026-05-27)

**Fix 1 — halfspace sign in `_check_planned_violations`**: changed
`x1 -= enlarge * nx` → `x1 += enlarge * nx`. Previous sign made the planned violation
check evaluate a looser halfspace than the projector actually enforced.

**Fix 2 — exec metrics always check against nominal boundary**: `check_trajectory_constraints`
called with `enlarge=0.0` for all variants. Tightened variant expected to show better
`exec_constraint_sat_rate` because its planned trajectories already have a δ buffer.

---

## UF-16.4 — Final box position + angle logging

`d3il/simulation/aligning_sim.py`: after the rollout loop, reads live MuJoCo box
pos+quat and passes as `final_box_pos` / `final_box_quat` into `update_rollout_info`.

`eval_visual_aligning_dpcc.py` — `update_rollout_info()`: extracts final XY, converts
quat to Z-angle (exact for pure Z-rotation), computes 2D dist to target. Three new keys
in `curr_context_info`: `final_box_xy`, `final_box_angle_deg`, `final_xy_dist`.

New console line:
```
  - Box  final XY=(0.501,  0.330)  angle=-57.1°  (dist_to_target: 0.0043 m)
```

---

## UF-16.5 — Success / Mode / Angle explainer

New document: [`u_f_16/SUCCESS_MODE_ANGLE_EXPLAINER.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16/SUCCESS_MODE_ANGLE_EXPLAINER.md)

Documents success criteria (pos ≤ 1.8 cm AND rot ≤ 8.6°), mode semantics (0 = robot
in contact at last step), `mean_distance` formula, angle formula verification.
No bugs found — logic is correct as-is.

---

## UF-16.6 — Dual-boundary visualization for tightened variants

`eval_visual_aligning_dpcc.py`:

- `_hs_xy_draw(ax, hs, enlarge, xlim, ylim, dashed=False)` — new `dashed` param; skips
  fill polygon and annotation when `True`.
- `_c_layers = [(0.0, False)]` for non-tightened; `[(0.0, False), (δ, True)]` for
  tightened. Both XY and 3D overlay blocks loop over `_c_layers`.
- Tightened runs draw two layers per shape: solid nominal (eval boundary) + dashed
  planning boundary. Non-tightened: single layer, unchanged appearance.
- Legend addition (tightened only): solid = `nominal constraint (eval boundary)`;
  dashed = `planning constraint (δ=X.XXXm inside)`.

---

## UF-16.7 — YAML comment corrections

`config/visual_aligning_eval.yaml` outdated comments corrected. Key corrections:
`n_contexts` reference de-hardcoded; `combined_5` added to available entries list;
active/inactive tags updated to reflect `active_geo_variants: [combined_5]`;
`combined_4` annotated with UF-16.1 relaxed-constraints note; `combined_5` given full
`[CURRENTLY ACTIVE]` description.

---

## Consolidated changed files

| File | Changes |
|---|---|
| `config/visual_aligning_eval.yaml` | UF-16.1: relaxed `combined_4`; UF-16.2: `active_geo_variants`, uncommented entries; UF-16.7: comment corrections |
| `d3il/simulation/aligning_sim.py` | UF-16.4: read final box pos+quat |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | UF-16.2: filter + `_has_geo` fix; UF-16.3: constraint metrics; UF-16.4: final box logging; UF-16.6: dual-boundary vis |
| `diffuser_visual_aligning/sampling/projection.py` | UF-16.3: per-sample print suppressed |
