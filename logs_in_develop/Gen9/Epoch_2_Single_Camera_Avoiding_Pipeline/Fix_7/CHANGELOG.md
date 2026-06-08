# Gen9 Epoch 2 — Fix-7: Post-Rollout Export & Plotting Crashes (Z-axis + GIF shape)

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs_6.3` — Slurm job 21165
**Parent**: [`../Fix_6/CHANGELOG.md`](../Fix_6/CHANGELOG.md)

---

## 1. Context — rollouts completed successfully

Job 21165 was the **first clean end-to-end run**:
- All 3 rollouts finished (no crash during inference)
- Results: 1/3 success rate (Context 0: ✅ 54 steps; Contexts 1–2: ❌ 15 and 14 steps)
- Denormalized a0 magnitude: ~9 mm/step (healthy; within act_normalizer range ±12.5 mm)
- bp_image std: 0.21 (camera rendering correctly)
- Constraint satisfaction: 1.000 for all rollouts (no_constraint variant → trivially satisfied)
- Dynamics consistency error: ~5 mm/step (reasonable)

The crashes below all occurred **after** the rollout loop, during per-rollout export and end-of-eval plotting.

---

## 2. Bugs

### 2.1 GIF failed: `all input arrays must have the same shape`

**Symptom**: Every rollout printed `[ WARNING ] GIF failed: all input arrays must have the same shape`.

**Root cause**: `video_frames` contained a mix of two shapes:
- `(96, 96, 3)` — appended by `predict()` visual path on each step (single-cam frame)
- `(96, 192, 3)` — appended by `capture_frame(bp_image, bp_image)` in `Avoiding_Sim.eval_agent`, which concatenates bp + inhand side-by-side along axis=1

`Avoiding_Sim.eval_agent` called `agent.capture_frame(bp_image, bp_image)` in the visual loop. But `predict()` **already** appends frames to `video_frames` for visual mode — it's the primary GIF capture path. `capture_frame` is meant for the **non-visual** path (where the model never sees images, so the sim provides them separately). Using both doubled the frames and produced mixed shapes.

**Fix — `d3il/simulation/avoiding_sim.py`**: disabled `capture_frame` call in the visual loop with a `if False and ...` guard + explanatory comment:
```python
# Visual mode: predict() already appends frames to agent.video_frames.
# Do NOT also call capture_frame — it concatenates bp+inhand side-by-side
# producing (96,192,3) frames that mismatch predict()'s (96,96,3).
if False and hasattr(agent, 'capture_frame'):
    ...
```

### 2.2 `_export_rollout_realtime` crash: `index 2 is out of bounds for axis 1 with size 2`

**Symptom**: `[ diag ] Real-time export failed for rollout N: index 2 is out of bounds for axis 1 with size 2` after every rollout.

**Root cause**: `_export_rollout_realtime` Z panel plots `real_pos[:, 2]` (commanded z-position over steps) and `c_pos_hist[:, 2]` (actual z-position). For avoiding, `real_pos` and `c_pos_hist` contain 2D `(T, 2)` positions — index 2 doesn't exist.

**Fix — both eval scripts**:
```python
# Before:
axes[1, 1].plot(real_pos[:, 2], 'k-', label='Z des')
axes[1, 1].plot(c_pos_hist[:, 2], 'r--', ...)
axes[1, 1].set_title('Z — des (black) vs actual (red)')

# After:
if real_pos.shape[1] > 2:
    axes[1, 1].plot(real_pos[:, 2], 'k-', label='Z des')
    ...
    axes[1, 1].set_title('Z — des (black) vs actual (red)')
else:
    axes[1, 1].set_visible(False)  # avoiding is 2D; no Z axis
```

### 2.3 PNG rollout grid crash: `IndexError: index 2 is out of bounds for axis 1 with size 2`

**Symptom**: Final `[ eval ] Generating PNG rollout grid` step crashed at `axes[i, 2].plot(obs_traj[:, 2], 'k-', label='Z des')`.

**Root cause**: Two aligning-era 3D hardcodes in the end-of-eval PNG grid loop:
1. Z-panel per rollout row: `obs_traj[:, 2]`, `c_pos_hist[:, 2]`
2. 3D XYZ panel: `cands[b, :, 2]`, `real_pos[:, 2]`, `c_arr[:, 2]`, scatter `[:, 2]` etc.

For avoiding, both `obs_traj` and `c_pos_hist` are `(T, 2)`.

**Fix — both eval scripts**:
- Z-panel per row: guarded with `if obs_traj.shape[1] > 2: ... else: axes[i, 2].set_visible(False)`
- 3D XYZ panel: entire block wrapped in `if real_pos.shape[1] < 3: ax_3d.set_visible(False) else: <original 3D code>`

---

## 3. Files touched

```
M  d3il/simulation/avoiding_sim.py                             (disable capture_frame in visual loop)
M  fm_visual_avoiding_test/eval_fm_visual_avoiding.py          (Z panel + 3D panel guards)
M  diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py  (Z panel + 3D panel guards)
```

---

## 4. Verification

| Check | Result |
|---|---|
| AST parse on all 3 files | ✅ |
| `capture_frame` in visual loop disabled | ✅ |
| `_export_rollout_realtime` Z panel guarded `shape[1] > 2` | ✅ FM + DPCC |
| PNG grid Z panel guarded `shape[1] > 2` | ✅ FM + DPCC |
| PNG grid 3D panel wrapped `shape[1] < 3` → `set_visible(False)` | ✅ FM + DPCC |

**Cluster-side expectation**: all 3 rollouts complete → per-rollout JSON/PNG export succeeds → end-of-eval PNG grid saves → NPZ results file written.

---

## 5. Remaining known issues after Fix-7

| Issue | Severity | Note |
|---|---|---|
| MP4 save fails: `imageio[ffmpeg]` not installed on cluster | Non-blocking | Falls back to GIF automatically; install `pip install imageio[ffmpeg]` if MP4 needed |
| DIAG obs labels say `des_c_pos` / `c_pos` (aligning names) | Cosmetic | Values correct; just confusing label in log — `obs_4d_np[:3]` and `[3:]` printed as aligning 3+3 split |
| `Entropy = -0.0` | Expected | Only 1 rollout per context; no mode diversity to measure |

---

## 6. First eval results summary (no_constraint / diffuser / seed 6 / train set)

| Rollout | Success | Steps | Mean Dist to Goal |
|---------|---------|-------|-------------------|
| Context 0 | ✅ | 54 | 0.306 m |
| Context 1 | ❌ | 15 | 0.558 m |
| Context 2 | ❌ | 14 | 0.564 m |
| **Overall** | **33.3%** | — | **0.476 m** |

Model at step 5000 (early training). Results expected to improve with more training steps.

---

## 7. Cross-references

| Document | Content |
|---|---|
| [`../Fix_6/CHANGELOG.md`](../Fix_6/CHANGELOG.md) | Previous fix (camera resize + inhand_np DIAG) |
| `fm_visual_avoiding/models/visual_gaussian_diffusion.py` | `VisualFlowMatching.forward` — confirmed working |
| `fm_visual_avoiding/models/visual_unet.py` | `VisualUNet.encode_visual` — confirmed working (ResNet encodes 96×96 RGB) |
