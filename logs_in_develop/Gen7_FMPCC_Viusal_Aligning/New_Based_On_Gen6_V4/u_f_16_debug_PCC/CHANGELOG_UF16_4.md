# UF-16 Session 2 — Output Logging, Visualization & Documentation

**Date**: 2026-05-28  
**Branch**: `update_into_FM`  
**Scope**: `d3il/simulation/aligning_sim.py`, `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`,
           `fm_visual_aligning_test/eval_fm_visual_aligning.py`, `config/visual_aligning_eval.yaml`

---

## UF-16.4 — Final box position + angle logging

### Motivation

The eval console printed box **init** XY/angle and target XY/angle but not where the box
**ended up** after the rollout.  This made it impossible to tell at a glance whether the
policy moved the box toward the target or pushed it away.

### Changes

**`d3il/simulation/aligning_sim.py`** (line ~144)

After the rollout while-loop, before calling `update_rollout_info`, the final MuJoCo
object state is read and appended to the info dict:

```python
_fbox_pos  = env.scene.get_obj_pos(env.push_box)   # [x, y, z] metres
_fbox_quat = env.scene.get_obj_quat(env.push_box)   # [w, x, y, z]
agent.update_rollout_info({**info, 'context': context,
                            'final_box_pos':  _fbox_pos,
                            'final_box_quat': _fbox_quat})
```

**Both eval files** — `update_rollout_info()` (before `master_rollout_history` dict is built)

Extracts final box state, converts quat to Z-angle (standard formula, verified exact for
pure Z-rotation), computes 2D distance to target, updates `curr_context_info`:

```python
_final_angle_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
_final_xy_dist   = np.sqrt((_fx - target_x)**2 + (_fy - target_y)**2)
curr_context_info.update({
    'final_box_xy':        [_fx, _fy],
    'final_box_angle_deg': _final_angle_deg,
    'final_xy_dist':       _final_xy_dist,
})
```

Because `curr_context_info` is updated before `master_rollout_history` is built and
`history_context_info` is appended, the three new fields flow automatically into:
- Per-rollout console print
- `diagnostics/rollout_N_stats.json` → `context_info` key
- `history_context_info` list (cross-rollout)

### New console line (after Init XY dist)

```
  - Box  final XY=(0.501,  0.330)  angle=-57.1°  (dist_to_target: 0.0043 m)
```

### New JSON fields (inside `context_info`)

```json
"final_box_xy":        [0.501, 0.330],
"final_box_angle_deg": -57.1,
"final_xy_dist":       0.0043
```

`final_xy_dist` is the **2D XY-only** distance from final box centre to target centre.
It complements `mean_distance` (which is `0.5*(3D_pos + rot/π)`, the combined D3IL metric).

### Angle formula verification

The aligning env samples box context as `[x, y, angle_deg]` and builds quaternions via
`euler2quat([0, 0, angle_deg * π/180])` (pure Z-rotation).  For this convention the
quat is `[cos(θ/2), 0, 0, sin(θ/2)]` and the standard formula
`arctan2(2*(w*z + x*y), 1 - 2*(y²+z²))` reduces to `arctan2(sin θ, cos θ) = θ` — exact.

---

## UF-16.5 — Success / Mode / Angle explainer

### New document

`logs_in_develop/…/u_f_16_debug_PCC/SUCCESS_MODE_ANGLE_EXPLAINER.md`

Documents the meaning and implementation of the three eval outputs that are most often
misinterpreted:

| Field | Source | Meaning |
|---|---|---|
| `Success status` | `_check_early_termination()` | pos ≤ 1.8 cm AND rot ≤ 8.6° at the SAME step — both must hold; `False` = hit 400-step limit |
| `Environment Mode` | `check_mode()` | 0 = robot ≤ 5.1 cm from box at last step (in contact); 1 = moved away.  Mode 0 is expected and normal for a push task |
| `Final Mean Distance` | `0.5*(3D_pos_m + rot/π)` | Mixed-unit D3IL metric; success boundary ≈ 0.033 |
| `Box final angle` | Z-euler from quat | Derived with standard formula — verified exact for pure Z rotations |

**No bugs found** — `done` and `success=True` are set on the same step; mode=0 at rollout
end is by design; angle formula is correct.

---

## UF-16.6 — Dual-boundary visualization for tightened variants

### Motivation

For tightened variants (`dpcc-t`), the MPC decision-point plots (XY and 3D) showed only
the **tightened planning boundary** (what SLSQP projected against).  The **nominal
boundary** (what execution metrics check against) was invisible.  This made it impossible
to see the δ safety margin visually.

### Design

For tightened runs: draw two constraint layers per shape.

| Layer | Style (XY) | Style (3D) | Meaning |
|---|---|---|---|
| Nominal (`enlarge=0`) | Solid edge + light fill | Solid wireframe / filled plane / filled sphere | Real evaluation boundary |
| Planning (`enlarge=δ`) | Dashed edge, no fill | Dashed wireframe / quad edges / wireframe circles | What DPCC projected against |

For non-tightened runs: single layer at `enlarge=0`, unchanged.

### `_hs_xy_draw` signature change

```python
def _hs_xy_draw(ax, hs, enlarge, xlim, ylim, dashed=False):
```

`dashed=True`: draws boundary as dashed line only — skips infeasible-side fill polygon
and feasible-arrow annotation.  Used for the inner planning layer.

### XY panel — `_c_layers` loop

```python
_c_layers = [(0.0, False)]
if self.is_tightened and _enlarge > 0:
    _c_layers.append((_enlarge, True))

for _cl_e, _cl_dash in _c_layers:
    # bounds: Rectangle — facecolor='none' + dashed edge for inner
    # halfspace: _hs_xy_draw(..., dashed=_cl_dash)
    # obstacles: Circle — facecolor='none' + dashed edge for inner
```

### 3D panel — same `_c_layers` loop

- **Bounds**: 12-edge wireframe loop with `linestyle='--'` for inner
- **Halfspace**: solid `Poly3DCollection` for nominal; 4 dashed edges of the quad for planning
- **Obstacles**: `plot_surface` sphere for nominal; equatorial + latitude wireframe circles (dashed) for planning

### Legend addition (XY panel, tightened only)

```python
_Line2D([0],[0], color='steelblue', lw=1.5, ls='-',  label='nominal constraint (eval boundary)')
_Line2D([0],[0], color='steelblue', lw=1.5, ls='--', label='planning constraint (δ=0.030m inside)')
```

### Files changed

| File | Change |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | `_hs_xy_draw` + XY overlay `_c_layers` loop + 3D all three shapes + legend |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Identical changes |

---

## UF-16.7 — YAML comment updates (`config/visual_aligning_eval.yaml`)

Outdated comments corrected; avoiding comments preserved untouched.

| Location | Before | After |
|---|---|---|
| `n_trials` comment | `n_contexts: 30` (stale hardcoded number) | `n_contexts` (refers to the live key) |
| `active_geo_variants` available list | missing `combined_5` | added `\| combined_5` |
| `bounds_only_2` description | `_2 = … commented until needed` | `both entries are defined and selectable` |
| `halfspace_only_1` description | `ACTIVE for debugging` | `defined; selectable for debugging` |
| Combined slots header | `Obstacle-involving slots disabled until…` | `use PLACEHOLDER geometry until…` |
| `combined_2` | `[active — DPCC-equivalent for this task]` | `[defined; not currently active — see active_geo_variants]` |
| `combined_4` | `[disabled — full DPCC equiv] / Enable once…` | `[defined; not currently active]` + UF-16.1 relaxed-constraints note |
| `combined_5` | bare two-line description | full description with `[CURRENTLY ACTIVE]` tag, '\' halfspace explanation, notes on bounds/obstacle parity |

---

## Changed files summary

| File | Change |
|---|---|
| `d3il/simulation/aligning_sim.py` | UF-16.4: read final box pos+quat, pass in info dict |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | UF-16.4: final box logging; UF-16.6: dual-boundary vis |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | UF-16.4: final box logging; UF-16.6: dual-boundary vis |
| `config/visual_aligning_eval.yaml` | UF-16.7: comment corrections |
| `logs_in_develop/…/u_f_16_debug_PCC/SUCCESS_MODE_ANGLE_EXPLAINER.md` | UF-16.5: new explainer doc |
