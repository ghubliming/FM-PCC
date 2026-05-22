# Gen7 Upgrade 10 — Context Info Logging

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py` · `d3il/simulation/aligning_sim.py`

---

## Motivation

Each aligning context randomises **both** box and target independently: XY position + Z-angle (±90°). Without recording this, a failed rollout is uninterpretable — was it a hard start (box far, angles misaligned) or a policy failure on an easy scene? `mean_distance` alone cannot answer this.

---

## Changes

### `aligning_sim.py`
After `env.reset(random=False, context=ctx_pool[context])`, call:
```python
if hasattr(agent, 'record_context_info'):
    agent.record_context_info(ctx_pool[context], int(context))
```
`hasattr` guard — backward-compatible with any agent that does not implement the method.

### `VisualAgentWrapper.__init__`
```python
self.curr_context_info    = {}   # Fix 10
self.history_context_info = []   # Fix 10
```

### `VisualAgentWrapper.reset()`
```python
self.curr_context_info = {}      # Fix 10
```

### New method `record_context_info(context, context_idx)`
Extracts from the context 4-tuple `(box_pos[3], box_quat[4], target_pos[3], target_quat[4])`:
- `context_idx` — index into train/test pkl array
- `box_init_xy` — `[pos[0], pos[1]]` metres
- `box_init_angle_deg` — `pos[2]` degrees (±90°); `pos[2]` is the angle parameter, **not Z height** (Z is always 0)
- `target_xy` — `[target_pos[0], target_pos[1]]` metres
- `target_angle_deg` — `target_pos[2]` degrees
- `init_xy_dist` — Euclidean XY distance box→target at rollout start

### `update_rollout_info()`
- Stores `'context_info': dict(self.curr_context_info)` in `master_rollout_history`
- Appends to `self.history_context_info`
- Prints context block before other stats:
  ```
  - Context idx: 7
  - Box  init XY=(0.523, -0.187)  angle=34.2°
  - Target   XY=(0.481, 0.274)  angle=-18.6°
  - Init XY dist (box→target): 0.4626 m
  ```

### Per-rollout JSON (`_export_rollout_realtime`)
`'context_info'` dict added to `rollout_N_stats.json`.

---

## What is NOT recorded
- Rotation distance at t=0 (requires `euler2quat` import — derivable post-hoc from angles)
- Box dimensions / four corners (derivable from center + angle + known geometry)
- Full quaternion arrays (stored in `master_rollout_history` pkl if needed)
