# UF-13: Non-Visual Mode GIF Investigation

**Date:** 2026-05-26  
**Scope:** `fm_visual_aligning_pipeline.sh` → `eval_fm_visual_aligning.py` running with `config/aligning-d3il-visual.py` non-visual mode  
**Status:** Investigation Complete

---

## 1. Root Cause: Why No GIF in Non-Visual Mode

### The Causal Chain

The GIF generation depends on `self.video_frames` being populated. Frame capture **only** happens inside the `if if_vision:` branch of `VisualAgentWrapper.predict()`:

```
eval_fm_visual_aligning.py
├── predict(state, if_vision=False)  ← non-visual call
│   ├── if if_vision:                ← SKIPPED
│   │   └── self.video_frames.append(frame)  ← never reached
│   └── else:                        ← non-visual path enters here
│       └── obs_np = state           ← pure numeric, no images
│
└── _export_rollout_realtime()
    └── if self.video_frames:        ← empty list → GIF skipped silently
```

**Code evidence:**

| File | Line | What happens |
|------|------|--------------|
| `eval_fm_visual_aligning.py` | L653 | `if if_vision:` gates all frame capture |
| `eval_fm_visual_aligning.py` | L659-666 | `bp_vis`, `inhand_vis` → `frame` → `self.video_frames.append(frame)` |
| `eval_fm_visual_aligning.py` | L713-737 | `else:` branch — receives only `obs_np` (numeric array), no images at all |
| `eval_fm_visual_aligning.py` | L419 | `if self.record_mode != 'none' and self.video_frames:` — empty list is falsy → GIF block skipped |

**Why the non-visual path has no images:**

In non-visual mode, `Robot_Push_Env.__init__()` creates the Scene with `RenderMode.BLIND` (`aligning.py:142`), and `get_observation()` (`aligning.py:205-234`) returns only a flat numeric state vector `[robot_pos | box_pos | box_quat | target_pos | target_quat]` — no camera images are produced.

The `Aligning_Sim` class (`aligning_sim.py:88-122`) mirrors this: the non-visual `else:` loop (L123-138) never touches `bp_image` / `inhand_image`. There are simply no pixel arrays to capture.

**Summary:** Non-visual mode = `RenderMode.BLIND` = no camera rendering = no frames = no GIF. This is by design.

---

## 2. Can We Reuse D3IL Code to Render MuJoCo GIFs Without the Visual Camera?

### Yes — via `RenderContextOffscreen` + `BPCageCam`

The D3IL MuJoCo stack has a fully functional **offscreen rendering** pipeline that works independently of whether the env was created in visual or non-visual mode:

| Component | File | Key API |
|-----------|------|---------|
| `RenderContextOffscreen` | `d3il_sim/sims/mj_beta/mj_utils/mj_renderer.py:250` | Creates an EGL/OSMesa offscreen context |
| `render()` singleton | `d3il_sim/sims/mj_beta/mj_utils/mj_render_singleton.py:33-56` | `render(cam_name, w, h, model, data, depth, seg)` → pixel array |
| `MjCamera.get_image()` | `d3il_sim/sims/mj_beta/MjCamera.py:72-109` | Calls `render()` singleton → returns RGB np array |
| `BPCageCam` | `envs/gym_aligning_env/gym_aligning/envs/aligning.py:34-53` | Bird's-eye cage camera (96×96), always attached to scene |
| `Scene.RenderMode.OFFSCREEN` | `d3il_sim/core/Scene.py:22` | Third mode between BLIND and HUMAN — headless GPU rendering |

**Critical insight:** Even in BLIND mode, the `BPCageCam` and `inhand_cam` objects are still **added to the scene** (`aligning.py:174-180`). The cameras exist in the MuJoCo model XML — the `RenderMode.BLIND` flag only prevents the _viewer window_ from opening. The offscreen rendering path (`mj_render_singleton.py`) operates independently of the viewer.

However, there is a caveat: when Scene is BLIND, `MjScene.render()` is a no-op (it only calls `viewer.render()` in HUMAN mode — `MjScene.py:164,189,202`). This means that **calling `cam.get_image()` in BLIND mode should still work** because `get_image()` → `_get_img_data()` → `render()` singleton creates its own `RenderContextOffscreen` per camera name, bypassing the viewer entirely.

> [!IMPORTANT]
> The offscreen singleton creates a `RenderContextOffscreen` that owns its own OpenGL context. It does NOT depend on `Scene.RenderMode`. As long as `MUJOCO_GL=egl` (or `osmesa`) is set and the MuJoCo `model`/`data` handles are valid, `cam.get_image()` will produce pixels even in BLIND mode.

---

## 3. Can We Generate GIFs From Existing Non-Visual Outputs Without Re-Running Eval?

### What existing outputs contain

After a non-visual eval run, each variant's results directory contains:

| Output File | Content | Usable for replay? |
|-------------|---------|---------------------|
| `rollout_X_data.pkl` | `real_robot_pos` (T,3), `desired_actions` (T,3), `c_pos_history` (T,3), `all_candidates`, `context_info` | ✅ Has commanded positions per step + context init |
| `{variant}.npz` | `obs_all`, `act_all`, `context_*` arrays | ✅ Has per-rollout trajectories |
| `results_seed_X.pkl` | Summary metrics only | ❌ No trajectory data |
| `rollout_X_stats.json` | Per-rollout metrics | ❌ No trajectory data |
| `diag_first_replan.txt` | First-step action diagnostics | ❌ |

**Key data for replay:**

- `context_info` → `(box_pos, box_quat, target_pos, target_quat)` → needed to reset environment to same initial configuration
- `real_robot_pos` → `des_c_pos` per step → the commanded trajectory (what the agent sent to the env)
- `desired_actions` → per-step velocity actions → directly replayable as `env.step()` inputs

### Standalone Replay Script Design

A standalone script can:
1. Load `rollout_X_data.pkl` from existing results
2. Create a `Robot_Push_Env(render=False, if_vision=True)` — this creates a BLIND scene but with cameras attached
3. Reset env with the stored `context_info`
4. Replay the stored `desired_actions` sequence via `env.step()`
5. After each step, call `env.bp_cam.get_image(depth=False)` + `env.inhand_cam.get_image(depth=False)` to capture offscreen frames
6. Assemble frames into GIF/MP4 via `imageio.mimsave()`

```python
# Pseudocode — standalone replay GIF generator
import pickle, numpy as np, imageio, cv2
from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env

def replay_to_gif(pkl_path, output_gif, fps=10):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    ctx = data['context_info']
    context = (
        [ctx['box_init_xy'][0], ctx['box_init_xy'][1], ctx['box_init_angle_deg']],
        # ... reconstruct quat from angle ...
        [ctx['target_xy'][0], ctx['target_xy'][1], ctx['target_angle_deg']],
        # ... reconstruct quat from angle ...
    )
    
    actions = data['desired_actions']  # (T, 3) velocity commands
    real_pos = data['real_robot_pos']  # (T, 3) des_c_pos
    
    env = Robot_Push_Env(render=False, if_vision=True)  # BLIND + cameras
    env.start()
    env.reset(random=False, context=context)
    
    frames = []
    des_pos = real_pos[0].copy()
    for t in range(len(actions)):
        action_7d = np.concatenate([des_pos + actions[t], [0, 1, 0, 0]])
        des_pos = action_7d[:3]
        obs, _, done, _ = env.step(action_7d)
        
        # Offscreen capture
        bp_img = env.bp_cam.get_image(depth=False)
        ih_img = env.inhand_cam.get_image(depth=False)
        frame = np.concatenate([
            cv2.cvtColor(bp_img, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(ih_img, cv2.COLOR_RGB2BGR)
        ], axis=1)
        frames.append(frame)
        if done:
            break
    
    imageio.mimsave(output_gif, frames, fps=fps)
    env.close()
```

### Feasibility Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| **Context reconstruction** | ⚠️ Partial | `context_info` stores `box_init_xy` + angle in degrees, and `target_xy` + angle. The original context tuple format is `(pos_3d, quat_4d, target_pos_3d, target_quat_4d)`. The angle→quat conversion uses `euler2quat([0, 0, angle_rad])` from D3IL utils. This is reconstructable. |
| **Action replay fidelity** | ✅ High | `desired_actions` are the denormalized velocity commands. Combined with `real_robot_pos[0]` as initial position, the replay should closely match the original run. Small MuJoCo physics non-determinism may cause micro-drift, but visually indistinguishable. |
| **Offscreen rendering** | ✅ Works | Requires `MUJOCO_GL=egl` + GPU or `MUJOCO_GL=osmesa` (CPU fallback). The Slurm eval script already sets `MUJOCO_GL=egl`. |
| **Camera availability** | ✅ Present | `Robot_Push_Env` always creates `bp_cam` and `inhand_cam` regardless of `if_vision` flag. Setting `if_vision=True` in the replay env ensures `get_observation()` returns images. |
| **No model/weights needed** | ✅ | Pure replay — no diffusion model, no normalizers, no config parsing needed. |

> [!WARNING]
> **Context reconstruction caveat:** The pkl stores `box_init_angle_deg` and `target_angle_deg` as scalar degrees. The original context is `(pos, quat, target_pos, target_quat)` where `pos = [x, y, angle_deg]` and `quat = euler2quat([0, 0, angle_deg * pi / 180])`. The Z coordinate of `pos` is hardcoded to 0.0 in `BlockContextManager.set_context()` (`aligning.py:114`). So full reconstruction is: `pos = [box_xy[0], box_xy[1], box_angle_deg]`, `quat = euler2quat([0, 0, box_angle_deg * pi / 180])`.

> [!TIP]
> **Simpler alternative:** Instead of replaying actions through physics, you could also use the expert reference generation pattern (already in `eval_fm_visual_aligning.py:146-212`) as a template. That function creates `Robot_Push_Env(render=False, if_vision=True)` and replays a position trajectory with `env.step(sim_action)` while capturing frames from camera observations — exactly what we need.

---

## 4. Cost Comparison: Standalone Replay vs Re-Run Eval

### What each path actually does

| Step | Standalone Replay | Re-Run Eval (visual mode) |
|------|------------------|---------------------------|
| Load model weights | ❌ Skip | ✅ ~5-10s (one-time) |
| MuJoCo env creation | ✅ Required | ✅ Required |
| MuJoCo physics stepping | ✅ ~400 steps × 30 contexts | ✅ ~400 steps × 30 contexts |
| Offscreen camera render (2 cams/step) | ✅ ~24,000 renders | ✅ ~24,000 renders |
| Model inference (GPU forward pass) | ❌ Skip | ✅ ~400 replans × 30 contexts |
| Frame assembly + GIF/MP4 save | ✅ imageio.mimsave | ✅ imageio.mimsave |
| Diagnostic outputs (PNG/JSON/pkl) | ❌ Skip (or reimplement) | ✅ Already built-in |
| Context reconstruction from pkl | ⚠️ Manual (angle→quat) | ❌ Not needed |
| Physics fidelity | ⚠️ Approximate (replay drift) | ✅ Exact |

### Time breakdown (approximate, per 30-context eval)

| Component | Estimated Time | Shared? |
|-----------|---------------|---------|
| MuJoCo env init + scene setup | ~5-10s | Both |
| Physics stepping (400 steps × 30 ctx × 35 substeps) | ~15-30 min | Both |
| Offscreen rendering (2 × 96×96 per step) | ~10-20 min | Both |
| Model inference (FM ODE, B=4 candidates, 100 flow steps) | ~30-90 min | **Eval only** |
| I/O (GIF save, pkl, JSON, PNG plots) | ~2-5 min | Both |

### Verdict

The standalone replay script saves **only the model inference time** — but it still pays the full MuJoCo + offscreen rendering cost, which is the same order of magnitude. Meanwhile:

- It requires engineering effort to write and debug (context reconstruction, action replay loop, GIF assembly)
- It introduces physics fidelity risk (MuJoCo non-determinism → replayed trajectories won't be bit-identical)
- It loses all the built-in diagnostic outputs (PNG reports, JSON stats, MPC foresight plots)
- The model inference cost is the only thing saved, and on a GPU cluster node it's not the dominant wall-clock bottleneck relative to total env stepping

> [!IMPORTANT]
> **Conclusion: No reason to write a standalone replay script.** The correct approach is to simply re-run eval in visual mode (`if_vision=True`). The compute cost is comparable, the engineering effort is zero, and you get exact physics + full diagnostics + GIFs for free. The eval script already handles everything.
