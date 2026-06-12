# Gen11 Epoch 5 Fix_2 — WS-B `renderer.close()` missing on cluster mujoco

**Date**: 2026-06-05  
**Status**: ✅ Fixed  
**Triggered by**: Job 21297 — `generate_gifs.sh` (WS-B, all scenes)  
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md)

---

## Symptom

Same crash as Fix_1 but in `generate_trajectory_gifs.py`:

```
AttributeError: 'Renderer' object has no attribute 'close'
```

Crashed after corridor (216/216 GIFs saved) before empty/pillars/s_curve.

---

## Root cause

Identical to Fix_1 — `renderer.close()` does not exist on the cluster's `mujoco` version.
The same fix was not applied to `generate_trajectory_gifs.py` at the time of Fix_1 because
only `collect_camera_images.py` was checked.

---

## Fix

`uav_expert_data_collect/generate_trajectory_gifs.py` line 263:

```python
# Before:
renderer.close()

# After:
if hasattr(renderer, 'close'):
    renderer.close()
```

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/generate_trajectory_gifs.py` | `renderer.close()` guarded with `hasattr` check |

---

## Re-run note

Corridor GIFs (216 episodes) are saved. Resubmit — `skip_existing=True` skips corridor
and continues with empty → pillars → s_curve.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh
```
