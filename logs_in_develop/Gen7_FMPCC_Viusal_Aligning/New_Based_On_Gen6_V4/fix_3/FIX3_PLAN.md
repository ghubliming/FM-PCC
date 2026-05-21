# Fix 3 — Eval Diagnostic Enrichment

**Date:** 2026-05-20
**Branch:** update_into_FM
**Applies to:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4, immediate)
            and `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7, mirror same changes)

**Trigger:** Promising Gen6V4 run (Mode 0, healthy first-replan DIAG) still fails.
GIF is the only visual tool but is unreadable — left panel frozen, right panel camera
moving without obvious task relevance. Cannot diagnose WHY the policy reaches the box
but fails to complete alignment.

---

## What the current outputs CAN tell you

| Output | What it tells you | What it CANNOT tell you |
|---|---|---|
| `[ DIAG first-replan ]` | Are model weights sane? Is normalizer working? (step 0 only) | Whether the policy degrades later in the episode |
| `Environment Mode` (end) | Did the robot reach the box by end? | When it reached, whether it ever made contact, or if it was near but not touching |
| `Final Mean Distance` (end) | Box-to-target gap at episode end | Distance trajectory — was it decreasing, flat, or oscillating? |
| `Avg Inference Time` | Inference speed sanity | Nothing about policy quality |
| 6-panel PNG `[0,0]` XY + MPC foresight | Robot path shape; whether plans point in consistent direction | Box path; whether plans ever pointed AT the box |
| 6-panel PNG `[0,1][0,2]` X/Y time-series | Position components over time | Whether position changes are task-progress or oscillation |
| 6-panel PNG `[1,0]` Z height | Height stability (contact safety) | Nothing about alignment progress |
| 6-panel PNG `[1,1]` tracking error | How well robot follows its own plans (MPC quality) | Nothing about task success |
| 6-panel PNG `[1,2]` velocity | Whether robot is moving at all | Whether velocity correlates with progress toward alignment goal |
| GIF (left = agentview, right = wrist) | Rough sanity that frames are changing | Left frozen = robot not visibly moving in wide view; right chaotic = arm swinging without purpose |

---

## Why the GIF looks the way it does

### Left panel (agentview / bp camera) — frozen

The agentview camera is **fixed** to the scene, not the arm.
Healthy physical step magnitude ≈ `0.000224 m` (0.22 mm). Over 400 steps that is at most
~90 mm of total end-effector travel. From a wide-angle fixed camera this looks nearly static.

It is NOT a rendering bug. It means the robot is making very small delta actions.
This is physically plausible for a precision alignment task — but makes the GIF useless
for diagnosing whether the robot is moving purposefully or oscillating.

**How to confirm**: check `[0,1]` and `[0,2]` panels of the 6-panel PNG. If position
changes are < 5 mm over the episode, the "frozen" GIF is expected behavior.

### Right panel (wrist / inhand camera) — "moving nonsensically"

The wrist camera is rigidly attached to the end-effector. Even 0.22 mm of arm motion
rotates the inhand view significantly. Combined with the policy making small oscillatory
corrections near the box, the wrist view looks like erratic jitter.

**What it likely means:** the robot IS physically near the box (Mode 0 confirms this)
but is oscillating without committing to a consistent push direction. The wrist camera
is the only thing that "moves" — it just doesn't look purposeful.

---

## Root diagnostic question we cannot answer yet

> **Is the robot oscillating near the box, or is it making consistent progress that is
> just too slow to complete in 400 steps?**

The distance-to-box at episode end (`0.091 m`) tells us the final gap but not the
trajectory. We need the distance curve over the episode. That is the single most
important missing diagnostic.

---

## Issues to fix (priority order)

---

### Issue 1 — No distance-to-box time series  ★ HIGHEST PRIORITY

**What's missing:** `mean_distance` is returned in `info` from `env.step()` every step
(see `aligning.py::check_mode()`), but the agent only receives it via `update_rollout_info()`
at the very end of the rollout. There is no distance curve.

**Why it matters:** A flat distance curve means the robot oscillates but doesn't progress.
A decreasing curve means the policy is working but too slow. These require different fixes.

**Where `mean_distance` lives per step:**
In `d3il/simulation/aligning_sim.py`, `env.step(action)` returns `obs, reward, done, info`
where `info['mean_distance']` is live. This is available but never forwarded to the agent.

**Fix:** Two parts:
1. In `aligning_sim.py` — after `env.step()`, call a new `agent.record_step_info(info)` hook
   (or pass info into the existing predict call). The simplest change: pass `info` as a
   keyword argument when calling `agent.step_hook(info)` each step.
2. In the eval agent — add `self.history_mean_distance = []` and append per step.
   Add this as panel `[1,0]` in the report PNG (replacing Z height, which is less useful
   for alignment diagnosis), titled **"Distance to Target over Steps"**.

**Alternate (zero-sim-change) approach:** Since `check_mode()` uses only `box_pos`,
`target_pos`, and `robot_pos` — all of which are in the obs — we can re-derive
`mean_distance` inside the agent from the 9D trajectory obs slots if the env doesn't
expose box_pos separately. But this is approximate. Better to read it from `info`.

---

### Issue 2 — DIAG printed only at step 0 of rollout 0

**What's missing:** After the first replan, we have no visibility into how action
magnitude evolves. The policy might produce healthy actions at step 0 then degrade
(OOM activations in later layers, conditioning saturation, etc.).

**Fix:** Add a periodic DIAG print every `N_DIAG_STEPS = 50` replans (not steps):
```
[ DIAG step=50 ] norm|a0|=0.231  denorm|a0|=2.1e-4 m  plan_dir=[-0.02, 0.89, -0.23]
[ DIAG step=100] norm|a0|=0.198  ...
```
Only log normalized magnitude, denormalized magnitude, and the direction unit vector.
One line per replan checkpoint — does not flood the log.

**File:** Add inside the `if self.action_counter == self.action_seq_size:` block in `predict()`.

---

### Issue 3 — Action magnitude not in 6-panel PNG

**What's missing:** Panel `[1,2]` currently shows end-effector velocity (= L2 norm of
consecutive positions). This is a coarse proxy for action magnitude but mixes position
history with action size.

**Fix:** Add `self.history_act_magnitudes = []` and append
`np.linalg.norm(next_action_np)` each step. Replace panel `[1,2]` with a plot of
per-step action magnitude. This immediately distinguishes:
- Stuck near zero → actions too small / scaler broken
- Oscillating → sign-flipping actions, policy indecisive
- Ramping up → policy building momentum toward goal

Keep the velocity panel only if there is a free slot (add a 7th panel: 3×3 grid minus
two, or expand to `2×4`).

---

### Issue 4 — obs_6d construction not logged

**What's missing:** The agent builds `obs_6d = [des_c_pos | c_pos]` (6D) from the
`(des_robot_pos_np, robot_pos_np)` pair. If these are swapped, identical, or wrong units,
the model receives a broken conditioning signal — but no log shows the raw values.

**Fix:** At `rollout_counter == 0 and step_counter == 0` (alongside the existing
first-replan DIAG), print:
```
[ DIAG obs ] des_c_pos = [x, y, z]   c_pos = [x, y, z]
[ DIAG obs ] obs_6d_raw = [...]   obs_6d_normalized = [...]
```
Lets you immediately verify the obs construction and normalizer range.

---

### Issue 5 — Image health not verified

**What's missing:** If a camera fails to render (returns black frame or repeats
the same frame), the model's visual conditioning is broken. There is no check.

**Fix:** At first replan, print:
```
[ DIAG img ] bp   std={std:.4f}  shape={shape}
[ DIAG img ] inhand std={std:.4f}  shape={shape}
```
`std < 0.01` on a float32 image means near-black/white — broken render.
`std` of both cameras identical to 4 decimal places across steps → frozen feed.

---

### Issue 6 — GIF has no text overlay — impossible to tell if frames update

**What's missing:** The left panel looks frozen. Without a frame counter we can't
distinguish "image is genuinely not changing" from "scene looks static because robot
moves < 1 mm per frame."

**Fix:** Overlay step number on each GIF frame using `cv2.putText` before appending
to `self.video_frames`:
```python
frame = np.concatenate([bp_vis, inhand_vis], axis=1)
cv2.putText(frame, f's{self.step_counter}', (5, 18),
            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
self.video_frames.append(frame)
```
Cost: negligible. Payoff: immediately tells you whether left panel is updating.

---

## Files to change

| File | Issues addressed |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 2, 3, 4, 5, 6 (all agent-side) |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Same — mirror identical changes |
| `d3il/simulation/aligning_sim.py` | Issue 1 — add per-step info forwarding to agent |

---

## Implementation order

1. **Issue 6** — GIF text overlay. One line. Immediate payoff — confirms if left panel is actually frozen or just scene-static.
2. **Issue 3** — Action magnitude panel. Replaces one existing panel in the PNG.
3. **Issues 4 + 5** — obs_6d + image std log. Both are additions to the existing first-replan DIAG block (~10 lines each).
4. **Issue 2** — Periodic DIAG every 50 replans. ~15 lines inside predict().
5. **Issue 1** — Per-step distance from `aligning_sim.py` + new PNG panel. Requires touching the sim file — do last since it's the most invasive.

---

## What success looks like after fix_3

After these changes, a failing run should reveal one of these clear failure patterns:

| What the new data shows | Diagnosis |
|---|---|
| Distance curve flat from step 0 | Robot near box but never contacts/pushes it — action scale too small or collision geometry wrong |
| Distance curve decreasing then plateauing | Policy makes progress but gets stuck at a local minimum near the target |
| Distance curve oscillating | Policy indecisive — replans contradict each other. Check plan direction consistency in DIAG. |
| Action magnitude near zero | Scaler broken or model outputting near-zero. Check obs_6d DIAG. |
| Image std near zero | Camera render broken — model has no visual conditioning signal |
| obs_6d_raw shows identical des/c pos | des_c_pos and c_pos slots are identical — obs construction bug |
