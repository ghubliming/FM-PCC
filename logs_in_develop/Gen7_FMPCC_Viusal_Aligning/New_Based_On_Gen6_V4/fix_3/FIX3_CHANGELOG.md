# Fix 3 Changelog — Eval Diagnostic Enrichment

**Date:** 2026-05-20
**Branch:** update_into_FM
**Plan reference:** FIX3_PLAN.md

Applied to:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7 — identical changes)
- `d3il/simulation/aligning_sim.py` (shared sim loop)

No model weights or config values changed. No retraining required.

---

## Change 1 — `aligning_sim.py`: per-step `mean_distance` forwarding to agent (Issue 1)

**File:** `d3il/simulation/aligning_sim.py`

After every `env.step()` call (both visual and non-visual loops), added a
`record_step_info(info)` hook call guarded by `hasattr`:

```python
# Visual loop (was line ~102):
obs, reward, done, info = env.step(pred_action)
if hasattr(agent, 'record_step_info'):
    agent.record_step_info(info)

# Non-visual loop (was line ~130):
obs, reward, done, info = env.step(pred_action)
if hasattr(agent, 'record_step_info'):
    agent.record_step_info(info)
```

`hasattr` guard means old agents (no `record_step_info`) are unaffected.

---

## Change 2 — `VisualAgentWrapper`: new state and `record_step_info()` method (Issue 1)

**Files:** both eval scripts

Added five new instance variables in `__init__` and cleared/reset in `reset()`:

```python
self.curr_rollout_act_magnitudes = []   # per-step action |mag| this rollout
self.curr_rollout_dist_to_target = []   # per-step mean_distance this rollout
self.history_act_magnitudes      = []   # across all rollouts
self.history_dist_to_target      = []   # across all rollouts
self._replan_count               = 0    # replans fired this rollout
```

New method `record_step_info()`:
```python
def record_step_info(self, info):
    d = info.get('mean_distance')
    if d is not None:
        self.curr_rollout_dist_to_target.append(float(d))
```

In `update_rollout_info()`, the two new lists are saved into `master_rollout_history`
and appended to the across-rollout history lists.

---

## Change 3 — `_export_rollout_realtime`: 2×3 → 2×4 report PNG (Issues 1 + 3)

**Files:** both eval scripts

Expanded from 6 panels (`figsize=(18,10)`) to 8 panels (`figsize=(24,10)`):

```
Before (2×3)                          After (2×4)
─────────────────────────────         ──────────────────────────────────────
[0,0] XY + MPC foresight              [0,0] XY + MPC foresight
[0,1] X position over steps           [0,1] X position over steps
[0,2] Y position over steps           [0,2] Y position over steps
                                       [0,3] Distance to Target over Steps  ← NEW
[1,0] Z height                        [1,0] Z height
[1,1] MPC tracking error              [1,1] MPC tracking error
[1,2] End-effector velocity           [1,2] Action Magnitude per Step (m)  ← NEW
                                       [1,3] End-effector velocity           ← moved
```

Panel `[0,3]` plots `dist_to_target[]` with a green dashed target line at y=0.
Panel `[1,2]` plots `act_magnitudes[]` per step.
Velocity is retained at `[1,3]`.

---

## Change 4 — GIF step-counter overlay (Issue 6)

**Files:** both eval scripts

In the video-capture block inside `predict()`:

```python
# Before:
self.video_frames.append(np.concatenate([bp_vis, inhand_vis], axis=1))

# After:
frame = np.concatenate([bp_vis, inhand_vis], axis=1)
cv2.putText(frame, f's{self.step_counter}', (5, 18),
            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
self.video_frames.append(frame)
```

Yellow step number in top-left corner. Immediately confirms whether the "frozen"
left panel is a scene-static issue or a genuine rendering failure.

---

## Change 5 — First-replan DIAG: obs_6d + image health (Issues 4 + 5)

**Files:** both eval scripts

Extended the existing `if self.rollout_counter == 0 and self.step_counter == 0:` block.
The new lines are appended to `diag_lines` so they also appear in `diag_first_replan.txt`.

**obs_6d (Issue 4)** — always printed:
```
[ DIAG obs ] des_c_pos=[x, y, z]  c_pos=[x, y, z]
[ DIAG obs ] obs_6d_norm=[...]
```
Confirms the 6D obs is built correctly: `des_c_pos` and `c_pos` should differ after
the first real step; identical values → both slots wired to the same source.

**Image health (Issue 5)** — visual path only:
```
[ DIAG img ] bp_image   std=0.XXXX  shape=(3, H, W)
[ DIAG img ] inhand_img std=0.XXXX  shape=(3, H, W)
```
`std < 0.01` triggers a WARNING line. Near-black images (broken camera render)
would otherwise produce silent garbage conditioning.

---

## Change 6 — Periodic DIAG every 50 replans (Issue 2)

**Files:** both eval scripts

Immediately after the first-replan DIAG block, inside the planning gate
`if self.action_counter == self.action_seq_size:`:

```python
self._replan_count += 1
if self._replan_count % 50 == 0:
    _pa0 = trajectory[[which], 0, :3].detach().cpu().numpy().squeeze()
    _da0 = action_traj[0, 0].detach().cpu().numpy()
    _dir = _pa0 / (np.linalg.norm(_pa0) + 1e-9)
    print(f'[ DIAG replan={self._replan_count} step={self.step_counter} ] '
          f'norm|a0|={np.linalg.norm(_pa0):.3f}  '
          f'denorm|a0|={np.linalg.norm(_da0):.2e} m  '
          f'dir={np.round(_dir, 3)}')
```

One log line per 50 replans. Detects mid-episode model degradation (norm growing),
direction oscillation, or action magnitude collapse — none of which are visible from
the first-replan DIAG alone.

---

## Change 7 — Action magnitude tracked per step (Issue 3)

**Files:** both eval scripts

After `next_action_np` is computed at the bottom of `predict()`:
```python
self.curr_rollout_act_magnitudes.append(float(np.linalg.norm(next_action_np)))
```

Feeds the `[1,2]` panel in the report PNG and is saved to `rollout_data.pkl`.

---

## Files Changed

| File | Issues |
|---|---|
| `d3il/simulation/aligning_sim.py` | 1 (per-step hook) |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 1, 2, 3, 4, 5, 6 |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | 1, 2, 3, 4, 5, 6 |

---

## What each new output tells you

| New output | How to read it |
|---|---|
| GIF step counter `s0`, `s1`… | If number increases but left panel looks frozen → scene is genuinely static at that scale, not a render bug |
| `[ DIAG obs ]` at step 0 | `des_c_pos == c_pos` → both slots wired to the same source |
| `[ DIAG img ] std` | `< 0.01` → camera render broken, model has no visual signal |
| `[ DIAG replan=50 ]` lines | `norm|a0|` growing → model diverging mid-episode; `dir` sign-flipping → oscillation |
| PNG `[0,3]` Distance to Target | Flat from step 0 → robot never pushes box; decreasing → progress but too slow; oscillating → policy stuck at local minimum |
| PNG `[1,2]` Action Magnitude | Near-zero → scaler broken; oscillating → policy indecisive; ramping → controller building momentum |
