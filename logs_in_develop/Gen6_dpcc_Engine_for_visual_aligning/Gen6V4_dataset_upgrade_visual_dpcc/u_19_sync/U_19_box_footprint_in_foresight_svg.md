# U_19 — Draw the physical push-box footprint in the MPC-foresight SVG

**Scope:** Gen7 (`fm_visual_aligning_test/eval_fm_visual_aligning.py`) **synced to**
Gen6V4 (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`).

**Commit:** `db4f7451` — "(Gen7 U19 / Gen6V4) feat: add visual push-box footprint to XY
foresight diagnostic plots in evaluation scripts".

**Change type:** diagnostics-only, **pure addition** (no existing lines deleted/edited;
no planning / projection / metric path touched).

## Motivation

The per-rollout `rollout_<r>_mpc_foresight.svg` XY panel already drew the constraint
geometry (workspace bounds, halfspace, obstacle circle) and the EE trajectories/
candidates, but **never drew the aligning push-box itself**. Suspicion: the box may
overlap the obstacle we added — impossible to judge without seeing the box.

## Files & functions touched (rollback map)

| File | Class → method | Location | Diff |
|---|---|---|---|
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | `VisualAgentWrapper._export_rollout_realtime` | new block at lines **1620–1663** (after the `if _gc:` constraint overlay, before `# ── 3D XYZ panel`) | +44 / −0 |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | `VisualAgentWrapper._export_rollout_realtime` | new block at lines **1626–1669** (same anchor) | +44 / −0 |

Both blocks are byte-identical. Nothing else in these two functions was modified; the
pre-existing `ax_xy.legend(handles=_lgd, ...)` is simply re-called at the end of the
new block so the box legend entries appear.

Also committed alongside (docs only, no code): this changelog in both
`.../U_19/` and `.../u_19_sync/`, and `logs_in_develop/MASTER_TEST_HISTORY.md` (+13).

## What the added block does

New `if _ci:` block inside the MPC decision-point figure, drawing the push-box on
`ax_xy` via a nested helper `_draw_box_xy(cx, cy, ang_deg, edgecolor, linestyle, fill)`:

- **Geometry.** Aligning push-box = **0.10 × 0.10 m square**, MuJoCo half-extent `0.05`
  from `robot_push_box.xml` (`<geom size="0.05 0.05 0.01" type="box"/>`). Helper builds
  4 corners at ±0.05, applies a 2×2 yaw rotation, translates to center → `Polygon`.
- **Pose source** = per-rollout `context_info` (`_ci`), already populated upstream:
  - `box_init_xy` + `box_init_angle_deg` → init pose — solid saddlebrown, filled (α=0.18)
  - `final_box_xy` + `final_box_angle_deg` → final pose — dashed saddlebrown outline
  - `target_xy` + `target_angle_deg` → target pose — solid goldenrod outline
  - Yaw is a real angle in degrees: `box_space` samples `[x, y, angle ∈ [-90, 90]]`
    (aligning env `BlockContextManager`), confirmed in the D3IL env source.
- Each pose is guarded (`if ... is not None`) so a missing field never raises (e.g. a
  rollout that ended before `final_box_*` was recorded).
- Local imports only (`matplotlib.patches as _mpa_box`); no new module-level imports.

## How to roll back

- **Full revert** (drops code + changelog + history edit): `git revert db4f7451`
  — clean, since it is HEAD and a pure addition.
- **Code-only manual revert:** delete the `if _ci:` block —
  lines **1620–1663** in `eval_fm_visual_aligning.py` and **1626–1669** in
  `eval_visual_aligning_dpcc.py` (from the `# U_19 (Gen7/Gen6V4): draw the physical
  push-box footprint` comment through the trailing `ax_xy.legend(handles=_lgd, fontsize=9)`).

## Notes / next step

- Only the **XY** panel is annotated (where box-vs-obstacle overlap is legible); the 3D
  panel is unchanged.
- The HF-iMF sibling (`imf_visual_aligning_test/eval_imf_visual_aligning.py`) has the
  identical foresight code and was **not** patched — sync it if desired.
- **Run on cluster** to regenerate the SVGs, then check the original suspicion (box
  footprint overlapping the obstacle circle).
