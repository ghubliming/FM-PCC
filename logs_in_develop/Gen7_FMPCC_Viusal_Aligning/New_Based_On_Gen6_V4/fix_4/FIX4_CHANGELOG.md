# Fix 4 — Changelog

**Date:** 2026-05-20
**Branch:** update_into_FM
**Files modified:**
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7)
- `config/visual_aligning_eval.yaml`

---

## Summary

Fix 4 adds ODE-explosion protection, richer per-step diagnostics, and a restructured
PNG dashboard. All items were applied identically to both Gen6V4 and Gen7 eval scripts
because both share the same `VisualAgentWrapper` architecture.

---

## Changes

### `VisualAgentWrapper.__init__`

| Addition | Description |
|---|---|
| `max_action_delta=None` param | Config-driven safety clamp threshold (metres) |
| `self.max_action_delta` | Stored for use in `predict()` |
| `self.curr_rollout_clamp_events` | Per-rollout list of `(step, raw_mag)` clamp events |
| `self.history_clamp_events` | Cross-rollout clamp history |

### `VisualAgentWrapper.reset()`

- Clears `curr_rollout_clamp_events` at the start of every rollout.
- Resets `_replan_count` counter.

### `VisualAgentWrapper.update_rollout_info()`

- Saves `curr_rollout_clamp_events` to `master_rollout_history`.
- Appends to `history_clamp_events`.
- Prints `  - Clamp events: N` in rollout summary.

### `VisualAgentWrapper.record_step_info()`  *(new method)*

- Receives the `info` dict from `env.step()` each step.
- Appends `info.get('mean_distance', np.nan)` to `curr_rollout_dist_to_target`.

### `VisualAgentWrapper.predict()` — direction-preserving clamp  *(core change)*

After logging the raw action magnitude, applies a direction-preserving rescale:

```python
if self.max_action_delta is not None:
    raw_mag = np.linalg.norm(next_action_np)
    if raw_mag > self.max_action_delta:
        next_action_np = next_action_np * (self.max_action_delta / raw_mag)
        self.curr_rollout_clamp_events.append((self.step_counter, float(raw_mag)))
        if len(self.curr_rollout_clamp_events) <= 5 or self.step_counter % 50 == 0:
            print(f'[ CLAMP step={self.step_counter} ] raw|a|={raw_mag:.4f} m → clamped to {self.max_action_delta} m')
```

The raw magnitude is recorded **before** clamping so the Act Magnitude panel captures
the true ODE output for diagnostic purposes. The direction vector is preserved (unlike
per-axis `np.clip` which distorts trajectory direction).

### `VisualAgentWrapper.predict()` — additional diagnostics

- `[ DIAG obs ]`: logs `des_c_pos`, `c_pos`, `obs_6d_norm` at first replan.
- `[ DIAG img ]`: logs birdseye/inhand image std + shape; warns if std < 0.01.
- `[ DIAG replan=N ]`: every 50 replans logs `norm|a0|`, `denorm|a0|`, direction unit vector.
- GIF step counter overlay: `s{N}` yellow text confirms left-panel-frozen is scene-static.

### `VisualAgentWrapper._export_rollout_realtime()` — 3×3 PNG grid

Grid restructured from 2×4 → 3×3 (figsize 18×15):

| Position | Panel |
|---|---|
| `[0,0]` | XY trajectory + MPC foresight |
| `[0,1]` | X position over steps |
| `[0,2]` | Y position over steps |
| `[1,0]` | Distance to Target over steps |
| `[1,1]` | Z height over steps |
| `[1,2]` | Tracking error (desired vs actual) |
| `[2,0]` | Action magnitude per step (raw, pre-clamp) |
| `[2,1]` | End-effector velocity per step |
| `[2,2]` | Clamp events scatter (step vs raw magnitude) |

### `VisualAgentWrapper(...)` instantiation in main block

Added `max_action_delta=config.get('max_action_delta', None)` to both eval scripts'
`VisualAgentWrapper(...)` call so the YAML value flows through at runtime.

### `config/visual_aligning_eval.yaml`

Added at end of file:

```yaml
# Safety clamp — max physical delta per step (metres). null = disabled.
max_action_delta: 0.01
```

`0.01 m ≈ 45× the healthy Gen6V4 step size` (0.000224 m), stopping ODE explosions
while allowing normal fast-approach actions. Set to `null` to disable.

---

## Rationale

The Gen6V4 diagnostic showed a healthy first replan (normalized `|a0|` = 0.24,
physical step ≈ 0.22 mm) but `Success: False` with a final distance of 0.091 m.
Two hypotheses:
1. ODE divergence mid-episode causing `LimitsNormalizer` saturation.
2. Policy correct but too slow (0.22 mm/step × 400 steps = 88 mm maximum travel).

The clamp prevents case 1 from masking case 2 in diagnostic output. The new PNG
panels (Distance-to-Target + Clamp-Events scatter) distinguish the two hypotheses
visually without requiring a retraining run.
