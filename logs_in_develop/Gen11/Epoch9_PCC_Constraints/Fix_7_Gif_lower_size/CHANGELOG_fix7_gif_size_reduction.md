# Epoch 9 Fix_7 — reduce UAV rollout GIF file size

**Date:** 2026-07-04. Answers: why are UAV GIFs so much bigger than the arm's, and lowers
UAV's GIF size/quality to close the gap.

## Root cause — resolution, not fps/stride

Both pipelines use the same `fps=10` and the same `frame_stride` cadence — the size gap is
almost entirely **pixels per frame**:

| | Arm (visual-aligning) | UAV (before this fix) |
|---|---|---|
| Camera resolution | **96×96** (`BPCageCam.__init__`, `gym_aligning/envs/aligning.py:39`) | **360×360** (`_make_overhead_renderer`) |
| Frame layout | bp_cam + inhand_cam concatenated ≈ 192×96 | single overhead view, 360×360 |
| Pixels/frame | ≈ 18,432 | ≈ 129,600 (**~7x more**) |

**Why the arm is small:** 96×96 isn't a size choice for the GIF — it's the arm's actual
**vision-policy input resolution**. The GIF reuses the same low-res frames the trained model
was fed (`capture_frame`'s docstring: "same shape/dtype/convention the visual `predict()`
receives"). Shrinking it isn't an option there — it would change what the policy sees.

**Why the UAV is big:** the UAV overhead render (`_make_overhead_renderer`) is a pure
debug/visualization artifact — the UAV policy is state-only (no vision conditioning at all,
`returns_condition=False`), so this render is **never fed to anything**. There was no reason
for it to be 360×360; that resolution was just never tuned down.

## Fix

### `FM_v3_uav_test/eval_fm_uav.py::_make_overhead_renderer`
```python
def _make_overhead_renderer(mujoco, model, res=200):   # was: res=360
```
360→200 is a ~3.2x reduction in pixel count (`(200/360)^2 ≈ 0.31`), while still legible for a
top-down overview of drone position vs. walls/pillars. No accuracy/training impact — this
render touches nothing but the GIF/PNG debug artifacts.

### `FM_v3_uav_test/eval_artifacts.py::save_rollout_gif`
```python
try:
    imageio.mimsave(path, frames, fps=fps, subrectangles=True, palettesize=128)
except TypeError:
    imageio.mimsave(path, frames, fps=fps)   # fallback if the backend rejects these kwargs
```
- `subrectangles=True`: encode only the pixels that changed between consecutive frames.
  MuJoCo's overhead view background (floor, walls, pillars) is static across an entire
  rollout — only the drone (and any candidate-trajectory overlay) moves — so this is a
  large, close-to-free win specifically for this kind of frame content.
- `palettesize=128`: halves the GIF color table from the default 256, a modest further
  reduction with negligible visible impact on a fairly simple overhead render (steelblue
  floor, colored geoms, one drone).
- Wrapped in `try/except TypeError` since I could not execute imageio in this environment
  (no Python runtime here — see project convention) to confirm the exact
  `imageio==2.34.1` GIF writer's accepted kwargs; the fallback guarantees the save can never
  fail outright even if these are unsupported by whatever writer/plugin resolves at runtime.

## What this does NOT touch
- `frame_stride` (still 2) and `fps` (still 10) — unchanged; resolution was the dominant
  factor, not frame count.
- The arm/visual-aligning GIF pipeline — untouched; its resolution is load-bearing (actual
  policy input), not a debug-only artifact, so there's no equivalent free win there.
- Any non-GIF artifact (PNG plots, npz, logs) — unaffected.

## Verification
- `py_compile` clean on both files.
- Confirmed via `grep` that the arm's 96×96 resolution is the vision-model's actual input
  size (`aligning_sim.py` feeds the same `bp_np`/`ih_np` arrays to both `agent.predict()` and
  `agent.capture_frame()`), not a GIF-specific choice — ruling out "just copy the arm's
  settings" as the fix, since the two resolutions serve different purposes.
- Could not execute `imageio.mimsave` directly (not installed in this Docker shell, cluster-only
  runtime) — the `subrectangles`/`palettesize` kwargs are applied defensively with a fallback
  for this reason. **Confirm the GIF still renders correctly and looks acceptable on the next
  cluster `--record gif` run** — this is the one part of Fix_7 not independently exercised here.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `_make_overhead_renderer` default resolution 360→200.
- `FM_v3_uav_test/eval_artifacts.py` — `save_rollout_gif` GIF encoding with palette/delta
  optimization + safe fallback.
