# Gen11 Epoch 5 Fix_1 — `AttributeError: 'Renderer' object has no attribute 'close'`

**Date**: 2026-06-05  
**Status**: ✅ Fixed  
**Triggered by**: Job 21291 — `collect_camera_images.sh` (WS-A, all scenes)  
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md)

---

## Symptom

WS-A camera collection crashed after completing corridor (216/216 episodes) but before
starting the next scene:

```
AttributeError: 'Renderer' object has no attribute 'close'
Exception ignored in: <function GLContext.__del__ ...>
EGLError(err = EGL_NOT_INITIALIZED, ...)
```

Because the sbatch script uses `set -e`, the job terminated immediately. Empty, pillars,
and s_curve scenes were never processed.

---

## Root cause

`collect_camera_images.py` called `renderer.close()` after each scene's batch. The
cluster's `mujoco` Python package version does not expose `.close()` on `mujoco.Renderer`
— cleanup is handled via `__del__` / garbage collection instead. The method simply does
not exist, hence `AttributeError`.

The secondary `EGLError` in the traceback is a GC warning caused by the interrupted
cleanup and does not affect saved image data.

---

## Fix

`uav_expert_data_collect/collect_camera_images.py` line 273:

```python
# Before:
renderer.close()

# After:
if hasattr(renderer, 'close'):
    renderer.close()
```

The guard makes the call a no-op on versions without `.close()` while preserving correct
behaviour on versions that do have it.

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/collect_camera_images.py` | `renderer.close()` guarded with `hasattr` check |

---

## Re-run note

Corridor images (216 episodes) were fully saved before the crash. On resubmit,
`skip_existing=True` skips corridor automatically and picks up from empty → pillars → s_curve.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh
```
