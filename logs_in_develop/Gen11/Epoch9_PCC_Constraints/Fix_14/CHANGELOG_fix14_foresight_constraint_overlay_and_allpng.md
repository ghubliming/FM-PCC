# Epoch 9 Fix_14 — MPC foresight SVG now shows the ENFORCED constraints; drop the redundant `all.png`

**Date:** 2026-07-10. UAV-only (Gen11 `FM_v3_uav_test`). Two independent visualization fixes the
user hit while inspecting the **s_curve / dpcc-r** rollouts:

1. **The MPC foresight SVG didn't draw the projector's obstacle/halfspace geometry**, so you
   could not tell whether the projector actually solved the corner constraints and re-routed the
   plan (the whole point of looking at dpcc-r on the s_curve corners).
2. **`all.png` was a byte-identical duplicate** of `<variant>.png` — redundant and confusing.

No behavior/metric change — both are pure plotting/output fixes.

---

## Fix 1 — draw the enforced constraint surfaces on the foresight SVG

### The bug
`write_mpc_foresight` (`FM_v3_uav_test/eval_artifacts.py`) drew:
- the green MPC candidate fan (`plans`), the black `p_des` path, the red actual-`p` path, and
- `_draw_obstacles(ax_xy, SCENE_OBSTACLES[scene])` — the **raw physical scene geometry** from
  `uav_expert_data_collect.generator.SCENE_OBSTACLES`.

It never drew the surfaces the **projector actually enforces**: the `halfspace_constraints`
walls (with s_curve's per-segment `x_active` switching), the `obstacle_constraints` balls, or the
`workspace_bounds` box — and none of them at the **inflated planning margin**
(`r_drone+margin_base [+enlarge]`). On s_curve the entire constraint set IS those halfspace walls
+ corner balls, so the SVG showed a green fan floating against *nothing to check it against*. You
could not see if dpcc-r's projection pushed the plan off the corner wall or not.

### The change (`eval_artifacts.py`)
Added `draw_projector_geometry(ax_xy, ax_xz, geo_config, variant)` and call it from
`write_mpc_foresight` (new optional args `geo_config=None, variant=''`). It **mirrors
`eval_fm_uav.plot_geo_constraints`** (same colours, same TRUE enforced margin) but paints onto
the foresight's existing axes:
- steelblue **workspace box** (XY rectangle + XZ floor/ceiling lines), shrunk by `margin`;
- darkorange **halfspace walls** with a feasible-side arrow, each **clipped to its live
  `x_active` x-range** via the ported `_fs_wall_xy` helper (so s_curve seg1/seg2 walls appear
  only where they are actually switched on), plus the `x∈[lo,hi]` label and an XZ x-span;
- tomato **obstacle balls** at `radius+margin` (XY filled, XZ dashed at mid-z).

It **respects the per-variant toggles** so the overlay is exactly what THIS variant's QP saw:
- `geo_free` in the variant name → geometric families are removed; instead of drawing stale
  surfaces it prints a red **"geo_free: geometry NOT enforced"** note (no misleading walls);
- `-tightened` → `margin += enlarge_constraints`, matching `setup_dpcc_projector`.

Raw `SCENE_OBSTACLES` are still drawn underneath, so the SVG now shows **both** the physical
collision truth (grey) and the planning surface the projector enforces (orange/tomato/steelblue)
— the gap between them being the inflation margin.

To avoid a circular import back into the `__main__` eval module, `_fs_normalize_halfspace` /
`_fs_wall_xy` are small local copies of `eval_fm_uav._normalize_halfspace` / its `_wall_xy`
(≈20 lines, pure geometry) — the same copy-modify pattern the repo already uses; `eval_artifacts`
stays self-contained (numpy + matplotlib only, importable in Docker).

### Caller (`eval_fm_uav.py:1337`)
```python
artifacts.write_mpc_foresight(diag_dir, i, r, scene,
                              geo_config=config, variant=variant)   # Fix_14
```
`config` (the merged per-scene geo entry, carrying `inflation`, `enlarge_constraints`,
`halfspace_constraints`, `obstacle_constraints`, `workspace_bounds`) and `variant` are already in
scope — same objects passed to `rollout_one(geo_config=config)` and `plot_geo_constraints`.

### What you'll now see for s_curve / dpcc-r
Both segment walls (each over its x-range) + the two crossover corner balls, drawn at margin
0.33, with the green candidate fan overlaid — so you can read directly whether the projector
solved the corner (fan bends off the wall into the feasible band) or failed (fan/actual path
crosses the orange line → the crash the U_13 s_curve investigation predicted at the crossover).

---

## Fix 2 — drop the redundant `all.png`

### The bug
`plot_overview` saved the SAME figure twice:
```python
fig.savefig(os.path.join(out_dir, f'{variant}.png'), dpi=130)
fig.savefig(os.path.join(out_dir, 'all.png'), dpi=130)   # aggregate alias (single-seed)
```
But `out_dir` is the **per-variant** folder `results/<geo_tag>/<variant>/` (`eval_fm_uav.py:1256`),
and the figure already overlays all trials of that one variant. So `all.png` was a
**byte-identical duplicate** of `<variant>.png`, not the "all variants, this seed" aggregate its
name implies (that would have to live one level up at `results/<geo_tag>/`). The legacy name came
from generations where `plot_overview` shared one `save_path` across variants; it doesn't hold in
the UAV per-variant layout.

### The change
Removed the `all.png` write (kept `<variant>.png`). Verified **nothing consumes it**: a repo-wide
`grep all.png` shows only (a) unrelated generations with a genuinely separate `fig_all` aggregate
figure, and (b) docs. No UAV analysis/aggregation script reads it. Updated the module docstring's
output list too. If a true across-variants aggregate is ever wanted, it should be written at the
`<geo_tag>/` level after the variant loop — flagged, not implemented (out of scope for a hot fix).

---

## Scope / not touched
- **UAV only.** `eval_artifacts.py` is the UAV eval's own writer; the s_curve scene and this
  candidate-fan foresight are UAV-specific. Visual-aligning (Gen7/Gen6V4) has its own inline
  plotting (`plot_geo_constraints` + its own foresight in `eval_fm_visual_aligning.py` /
  `eval_visual_aligning_dpcc.py`) and a different `all.png` mechanism (auto per geo entry), so
  there is nothing to sync here. If the visual-aligning foresight turns out to have the same
  "constraints not overlaid" gap, that's a separate follow-up — not bundled into Fix_14.
- No change to projection math, metrics, npz schema, or the projection-variant set (U8/U8b/U8c).

## Verification done here
- `python3 -m py_compile FM_v3_uav_test/eval_artifacts.py FM_v3_uav_test/eval_fm_uav.py` → **OK**.
- Confirmed `config` at the call site carries `inflation` (`{r_drone:0.31, margin_base:0.02}` →
  base margin 0.33), `halfspace_constraints`, `obstacle_constraints`, `workspace_bounds`,
  `enlarge_constraints` — the same object `plot_geo_constraints` consumes.
- Confirmed `all.png` has no code consumer (repo-wide grep).
- **Not run end-to-end** — Docker has no torch/MuJoCo; the actual SVG render must be regenerated
  on the cluster (re-run the s_curve eval, or just re-render from an existing `<variant>.npz` +
  `plans` via `write_mpc_foresight`). **Run on cluster** to confirm the overlay renders and the
  s_curve `x_active` clipping looks right.

## Files
- `FM_v3_uav_test/eval_artifacts.py` — new `draw_projector_geometry` + `_fs_wall_xy`/
  `_fs_normalize_halfspace`; `write_mpc_foresight` overlays enforced constraints & merges legend;
  `plot_overview` no longer writes `all.png`; docstrings updated.
- `FM_v3_uav_test/eval_fm_uav.py` — pass `geo_config=config, variant=variant` to
  `write_mpc_foresight`.
