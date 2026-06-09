# Gen11 Epoch 5 — U3 Changelog: Physics Replay GIFs

**Date:** 2026-06-09  
**Status:** Coded — ready for SLURM smoke test  
**Plan:** `PLAN.md`

---

## New files

### 1. `uav_expert_data_collect/generate_physics_gifs.py`

Physics replay GIF generator (Epoch 5 WS-D). Re-runs the exact original PID
simulation using `mj_step` (real physics, not `mj_forward` state injection)
and records frames with contact overlays.

**Key design decisions:**

| Decision | Detail |
|---|---|
| Seed recovery | `int(ep_id.split('_')[-1])` — 7-digit zero-padded seed is always the last `_`-token |
| Traj reproduction | `_build_traj_and_init(scene, homotopy, rng)` + `_make_pid(model, controller)` — already standalone in `generator.py` (no changes needed there) |
| Frame capture rate | `dataset_stride * frame_stride` physics steps; `dataset_stride = round(1 / (dt * 33))` aligns with 33 Hz collect rate |
| Contact detection | `_is_obstacle_contact()` from `generator.py` — same logic as original collection |
| Contact overlay | Red 4px border on both panels + `CONTACT` text when `ncon > 0` (non-floor contacts) |
| Proximity bar | Bottom strip of bp panel: green (>0.30 m) → orange (>0.15 m) → red (≤0.15 m); range 0–0.5 m |
| Obstacle distance | `_nearest_obstacle_dist()` — standard box SDF for walls, 2-D radial SDF for cylinders (pillars) |
| Renderer close | `if hasattr(renderer, 'close'): renderer.close()` — E5 Fix2 pattern |
| Output path | `gifs_physics/{scene}/{homotopy}/{ep_id}_physics.gif` |
| Skip existing | `--skip-existing` on by default; `--no-skip` to regenerate |
| CONTACT logging | `tqdm.write` prints episode_id + contact step count for any episode with contacts |

**Functions added:**
- `parse_args()` — CLI
- `discover_episodes(data_dir, scene_filter)` — same walk pattern as `generate_trajectory_gifs.py`
- `_nearest_obstacle_dist(pos, obstacles)` — box SDF + cylinder SDF over `ep['obstacles']`
- `_render_overhead(model, data, renderer)` — bird's-eye free camera
- `_render_track(model, data, renderer, cam_id)` — FPV named camera
- `_burn_text(frame, text, pos, ...)` — cv2 text overlay helper
- `_apply_contact_border(frame, width)` — red border on contact frames
- `_build_frame(...)` — assembles bp + track panels with all overlays
- `physics_replay_frames(model, data, renderer, episode, res, frame_stride)` — core loop
- `main()` — per-scene model load, tqdm progress, error collection, summary

---

### 2. `Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh`

SBATCH script for WS-D. Mirrors `generate_gifs.sh` structure.

**EGL setup (identical to `generate_gifs.sh` Group B pattern):**
```bash
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MPLBACKEND="agg"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```
`MUJOCO_GL=disabled` is NOT used here — `mujoco.Renderer` requires EGL.
The 3-line pinning block prevents EGL defaulting to GPU 0 (IT violation).

**Args:** `$1`=max_episodes, `$2`=scene, `$3`=mp4, `$4`=frame_stride (default 3)

---

## No changes to existing files

`generator.py` — `_build_traj_and_init` and `_make_pid` were already standalone
functions (lines 108–170). No refactoring needed.

---

## What Change C (comparison GIF stitcher) is not implemented

Change C from the plan (side-by-side state-injection vs physics GIF) is deferred.
It requires both E5 U2 GIFs and U3 physics GIFs to exist first. Implement after
smoke test confirms U3 output looks correct.

---

## Smoke test command

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh 3 pillars
```

Expected output in `logs/uav_expert_data/gifs_physics/pillars/L_L_L/` (or whichever
homotopy the first 3 episodes are). Pillar episodes may show `CONTACT` overlays for
the rejected 27/500 — the 473 clean ones should show green proximity bar throughout.

---

## GPU device-pinning check

`generate_physics_gifs.sh` uses `--gres=gpu:1`. EGL device is pinned via
`MUJOCO_EGL_DEVICE_ID=$ALLOCATED_GPU`. To verify the renderer opens the **allocated**
GPU (not GPU 0 by default), use the self-contained debug block already in the script
(line after `EGL device:` echo, before `Build args` section).

**How to use:**
1. Uncomment the `# python3 -c "..."` line in `generate_physics_gifs.sh`
2. Submit the smoke-test job
3. In the SLURM output look for:
   ```
   [ EGL-CHECK ] EGL_DEVICE=<N>  CUDA=<N>
   [ EGL-CHECK ] renderD fds open (after Renderer): ['/dev/dri/renderD<128+N>']
   ```
4. `renderD128 + MUJOCO_EGL_DEVICE_ID` should equal the device shown (e.g. EGL_DEVICE=2 → renderD130)
5. Comment the line back out

**Why not `lsof /dev/dri/renderD*`:** that command must run on the compute node while the
job is live — impossible from the login node or WSL. The embedded check runs inside the
job itself and reads `/proc/{pid}/fd` symlinks directly (no `lsof` binary needed).
