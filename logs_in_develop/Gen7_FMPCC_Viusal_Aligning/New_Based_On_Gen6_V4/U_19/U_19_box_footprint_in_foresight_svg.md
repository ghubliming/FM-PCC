# U_19 — Draw the physical push-box footprint in the MPC-foresight SVG

**Scope:** Gen7 (`fm_visual_aligning_test/eval_fm_visual_aligning.py`) **synced to**
Gen6V4 (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`).

## Motivation

The per-rollout `rollout_<r>_mpc_foresight.svg` XY panel already drew the constraint
geometry (workspace bounds, halfspace, obstacle circle) and the EE trajectories/
candidates, but **never drew the aligning push-box itself**. Suspicion: the box may
overlap the obstacle we added. You cannot judge that without seeing the box.

This upgrade marks the box as a square on the XY panel of every rollout foresight SVG.

## What changed

In the standalone MPC decision-point figure, right after the constraint-geometry
overlay (`if _gc:` block) and before the 3D panel, a new `if _ci:` block draws the
push-box footprint on `ax_xy`:

- **Geometry.** The aligning push-box is a **0.10 × 0.10 m square** — MuJoCo half-extent
  `0.05` from `robot_push_box.xml` (`<geom ... size="0.05 0.05 0.01" type="box"/>`).
  Drawn as a rotated square via a small `_draw_box_xy(cx, cy, ang_deg, ...)` helper
  (4 corners at ±0.05, rotated by a 2×2 yaw matrix, translated to center).
- **Pose source.** `context_info` (`_ci`), already populated per rollout:
  - `box_init_xy` + `box_init_angle_deg` → **init pose** — solid saddlebrown, filled (α=0.18).
  - `final_box_xy` + `final_box_angle_deg` → **final pose** — dashed saddlebrown outline.
  - `target_xy` + `target_angle_deg` → **target pose** — solid goldenrod outline.
  - Yaw is a real angle in degrees: `box_space` samples `[x, y, angle ∈ [-90, 90]]`
    (aligning env `BlockContextManager`), confirmed in the D3IL env source.
- **Legend.** The three box entries are appended to `_lgd` and `ax_xy.legend(...)` is
  re-called so they show up.

Each pose is guarded (`if ... is not None`), so nothing breaks if a field is missing
(e.g. a rollout that ended before `final_box_*` was recorded).

## Notes / next step

- Only the **XY** panel is annotated (that is where box-vs-obstacle overlap is legible).
  The 3D panel is unchanged.
- This is a **diagnostics-only** change — no effect on planning, projection, or metrics.
- **Run on cluster** to regenerate the SVGs and confirm the box renders where expected;
  then check the original suspicion (box footprint overlapping the obstacle circle).
