# Epoch 9 U2 — port the constraint-geometry schematic + GIF step counter (NOT faithfully recovering visual-aligning)

**Date:** 2026-07-04. Two gaps found and fixed in this changelog, both cases of the UAV eval
missing a visual-aligning diagnostic that had nothing to do with E9's constraint math itself —
pure observability parity gaps.

## Gap 1 — the constraint-geometry schematic

**Date:** 2026-07-04. Fixes a real gap: the E9 UAV pipeline had no equivalent of visual-aligning's
`constraint_overview.png` — the schematic showing where the environment's bounds/halfspace/
obstacle geometry actually is, generated once per geometry configuration before any rollout.
Confirmed missing by inspecting the exact file the user pointed to:
`.../<seed>/results_train_set/combined_5-tightened/constraint_overview.png`
(`fm_visual_aligning_test/eval_fm_visual_aligning.py::plot_geo_constraints`, called at L1864).
We had ported the *runtime* projector geometry (E9 init) but never the *diagnostic plot* of it —
this was a real "not faithfully recovered" gap, not a style choice.

## What was missing

`plot_geo_constraints(geo_name, geo_config, out_dir, is_tightened)`: a 3-panel figure
(3D wireframe | XY top-down | XZ side) drawing the workspace box, halfspace boundary +
feasible-side arrow, and obstacle sphere for one named geometry entry — saved once, before any
trajectory runs, at the geo-level results folder (`results/<geo_name>/constraint_overview.png`).
Nothing analogous existed for the UAV eval; the only visual artifacts were per-rollout
plots (`plot_overview`, MPC foresight), never a static "here is the enforced geometry" figure.

## The fix

### `FM_v3_uav_test/eval_fm_uav.py` — new `plot_geo_constraints(geo_name, config, out_dir, is_tightened=False, basename='constraint_overview')`
Ported and adapted from visual-aligning's function of the same name:
- Same 3-panel layout (3D wireframe / XY top-down / XZ side), same color convention
  (steelblue=bounds, darkorange=halfspace, tomato=obstacle).
- **Draws the TRUE enforced boundary**, not raw scene geometry: applies the same `margin`
  `setup_dpcc_projector` computes (`r_drone + margin_base` always-on, `+ enlarge_constraints`
  when tightened) — so the figure shows what the projector actually believes, matching the
  runtime behavior exactly (this is stricter than visual-aligning's original, which only had
  the tightening `enlarge`, not a separate always-on inflation term).
- **Handles the UAV-specific halfspace formats**: both the plain list `[p0,p1,side]` and the
  dict form `{line, side, x_active}` (s_curve's per-segment switching, U8/E9-init). A wall with
  `x_active` is drawn **only over its live x-range** — clipped and labeled
  (`x∈[lo,hi]`) — rather than as an infinite line, so the figure shows the actual non-convex
  active-set geometry, not a misleading full-length wall.
- **Saved as BOTH `.png` AND `.svg`** — visual-aligning's original only produced `.png`. Added
  the vector `.svg` output because a schematic meant for a paper/thesis figure should be
  scalable, not raster-only; `.png` is kept for quick viewing. Idempotent: skipped only when
  BOTH files already exist (matches the original's PNG-only idempotency, extended to both).

### `_run_variant` — call site (mirrors visual-aligning's `if geo_variant == projection_variants[0]: plot_geo_constraints(...)`)
Generated once per `geo_dir` (the E9-fix1 output folder), before any rollout, gated so it isn't
regenerated per variant. One real structural difference from visual-aligning required a design
call:

**Why `constraint_overview.png` vs `constraint_overview_tightened.png` (two files, not two
folders):** in visual-aligning, `-tightened` is baked into a **separate named geo entry**
with its own `results/<name>-tightened/` folder — e.g. the user's own example path,
`results_train_set/combined_5-tightened/`. In the UAV yaml (E9 init), `-tightened` is instead a
**per-projection-variant margin modifier** (`dpcc-c` vs `dpcc-c-tightened`) sharing the *same*
`geo_tag`/`geo_dir` as its base sibling — this matches the older DPCC-avoiding convention
(`variant` name alone controls `enlarge`), which is what E9 built on. Since both margins can
coexist under one `geo_dir`, the schematic call site detects whichever margins are actually
present in `config['projection_variants']` and writes `constraint_overview.png` (base) and,
if any `-tightened` variant is configured, `constraint_overview_tightened.png` — two files in
the same folder, each showing its own correctly-computed margin, rather than one plot silently
representing only one of the two enforced geometries.

### Verification (Gap 1)
Executed `plot_geo_constraints` directly (module loaded via `importlib` with the heavy
torch/mujoco-importing submodules stubbed out — matplotlib itself IS available in this
environment) against the real `config/uav_projection.yaml` geo entries:
- All 4 scenes (`empty`, `corridor`, `pillars`, `s_curve`) render without error, both base and
  tightened (for the 3 with `bounds` active), producing `.png` + `.svg` each.
- Visually inspected `pillars` (6 balls correctly placed at their XML-derived centers, envelope
  halfspace walls, workspace box) and `s_curve` (4 walls correctly clipped to their `x_active`
  ranges — non-convex geometry rendered exactly as the projector enforces it, not as 4 infinite
  lines) — both match the intended per-scene geometry from
  `../Plan/PLAN_E9_PCC_constraints.md` §3.
- `empty` renders the "no geometric constraints" placeholder text (its `constraint_types=[]`),
  confirming the unconstrained-baseline marking survives into the diagnostic plot too.
- `py_compile` clean on the full `eval_fm_uav.py`.
- Full rollout / SLSQP / cluster path still untested here (no torch/MuJoCo runtime) — this
  fix only concerns the plotting function, which was independently exercised.

### Files touched (Gap 1)
- `FM_v3_uav_test/eval_fm_uav.py` — new `plot_geo_constraints` function; call site added in
  `_run_variant` right after `out_dir`/`geo_dir` creation.

---

## Gap 2 — the GIF step-count overlay

The rollout GIFs visual-aligning produces burn a small `sK` step counter into the top-left
corner of every frame (`Aligning_Sim.capture_frame`):
```python
cv2.putText(frame, f's{self.step_counter}', (5, 18),
            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
```
This lets a viewer read off, frame-by-frame while scrubbing/pausing a GIF, exactly which
control step is on screen — useful for cross-referencing a visible event (a stall, a near-miss,
a projection kick) against the structured per-step log / plots, without re-deriving it from
frame index × `frame_stride`. The UAV eval's overhead-camera GIF frames
(`rollout_one`, `FM_v3_uav_test/eval_fm_uav.py`) had **no such overlay** — another observability
parity gap, same root cause as Gap 1 (a visual-aligning diagnostic never ported).

### Fix
In `rollout_one`'s frame-capture block (where `frames.append(_render_overhead(...))` already
lived, gated on `renderer is not None and k % frame_stride == 0`), overlay the current FM step
index `k` onto the frame **before** appending, in the exact same style (yellow, top-left,
`FONT_HERSHEY_PLAIN`, scale 1.2):
```python
frame = _render_overhead(mujoco, model, data, renderer)
import cv2
cv2.putText(frame, f's{k}', (5, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
frames.append(frame)
```
`cv2` is imported lazily inside the existing `try/except` (matching this file's lazy-import
convention for other optional/heavy deps — `mujoco`, `MJPCTracker`, `BehaviorLogger`, and
Gap 1's own `matplotlib`); a missing/broken cv2 falls into the pre-existing `except Exception`
handler and just stops frame capture for that rollout, same as any other render failure — it
does not crash the eval.

`k` (not a separately maintained counter) is used directly since `rollout_one`'s FM loop is
already indexed by `k` — no new state needed, unlike visual-aligning's `self.step_counter`
(which exists there because its frame capture is a callback invoked from inside the sim,
outside the loop that owns the index).

### Verification (Gap 2)
- `py_compile` clean on `eval_fm_uav.py`.
- `opencv-python==4.10.0.84` confirmed already listed in `requirements.txt` (visual-aligning's
  own dependency) — this is not a new dependency, just a previously-unused import in this file.
- `cv2` is not installed in this Docker AI-coding shell (no Python runtime here, by design —
  see `project_env` memory), so `cv2.putText` itself could not be executed in this session;
  the call matches visual-aligning's own working invocation exactly (same function, same
  argument order/types: `frame` uint8 (H,W,3) from `renderer.render().copy()`, same font/color/
  thickness), so runtime behavior is expected identical. **Confirm the rendered overlay
  visually on the cluster** on the next `--record` run (this is the one part of Gap 2 not
  independently exercised here).

### Files touched (Gap 2)
- `FM_v3_uav_test/eval_fm_uav.py` — `rollout_one`'s frame-capture block: added the `cv2`
  step-counter overlay before `frames.append(...)`.
