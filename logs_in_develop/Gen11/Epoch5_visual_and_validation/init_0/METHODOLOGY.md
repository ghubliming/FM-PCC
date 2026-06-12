# Gen11 Epoch 5 — Visual Collection & Validation: Methodology

**Date**: 2026-06-06  
**Status**: In progress — WS-A corridor done; WS-B Fix_2 resubmit pending  
**Maximum fix index**: Fix_2  
**PLAN**: [`EPOCH5_PLAN.md`](EPOCH5_PLAN.md)  
**Predecessor**: [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md)

---

## 0. Epoch 5 in one sentence

**Replay the 1769 Epoch 4 state pickles through MuJoCo offscreen rendering to produce
camera images (WS-A) and inspection GIFs (WS-B), while independently running a mini-FM
sanity gate (WS-C) to confirm data correctness before Epoch 6 training.**

---

## 1. The two-stage architecture — why the stages are separate

Epoch 4 (Stage 1) ran a physics simulation to record `(p, v, p_des)` at each step and
saved them as pickle files.  Camera rendering was deliberately excluded from Stage 1.

**Why?** Offscreen rendering (EGL) requires a GPU and is ~10× slower per step than headless
physics.  If rendering were coupled to the physics loop, every re-run (to fix a bug,
change resolution, add a camera) would require re-flying all 1769 trajectories.  By
separating the stages, Stage 2 can replay the *same* stored states repeatedly with
different render parameters at zero physics cost.

**The coupling point**: both stages share the state vector `obs[t] = [p(3), v(3)]`.  Stage
2 reads this from the pickle and injects it directly into MuJoCo, bypassing the PID
entirely.

---

## 2. State injection — the physics of "replaying without re-flying"

At each timestep `t`, Stage 2 sets:

```python
data.qpos[:3] = obs[t, :3]           # 3D position — world frame
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion — forced identity (level)
data.qvel[:3]  = obs[t, 3:6]         # linear velocity — world frame
data.qvel[3:6] = 0.0                  # angular velocity — forced zero
mujoco.mj_forward(model, data)        # kinematics update, no physics step
```

**`mj_forward` vs `mj_step`**: `mj_step` advances the simulation by one timestep using the
full ODE solver (contact dynamics, constraint resolution, rotor forces).  `mj_forward`
only propagates the current `qpos`/`qvel` through the kinematic chain to update positions
of all bodies and cameras — it does *not* step time and does not integrate any dynamics.
The result is that every body (drone hull, rotors, attached cameras) is placed exactly
where `qpos` specifies, ready for rendering.

Code: `collect_camera_images.py:replay_and_capture()` lines 166–177 (WS-A),
`generate_trajectory_gifs.py:replay_to_frames()` lines 154–163 (WS-B).

### 2.1 Identity quaternion — design choice and limitation

Both WS-A and WS-B force the drone to a level attitude `q = [1, 0, 0, 0]` (zero
roll, zero pitch, zero yaw).  In real flight, the drone tilts to accelerate: to move forward
at 0.7 m/s it pitches ~10° nose-down.  Those actual quaternions are **not stored** in the
Epoch 4 pickle (only `p` and `v` are in `obs`).

**Why force level?** Two reasons:
1. The Epoch 4 schema deliberately excluded attitude to keep the state space minimal (9D =
   `[Δp_des(3) ‖ p(3), v(3)]`).  Attitude is not needed for the FM-PCC state observation.
2. The visual FM model (Epoch 6) will be trained on these rendered images — consistently
   level renders avoid the model having to learn to disentangle attitude-induced visual
   effects from position-induced ones.

**Limitation**: Forcing level attitude means the rendered drone appears upright even when
the recorded velocity implies it was tilted.  This can make wall-adjacent positions look
more dramatic in GIFs than they were during the actual flight (see
`INVESTIGATION_wall_contact_gifs.md`).

---

## 3. WS-A — Camera image collection

### 3.1 Physical meaning

WS-A produces the *pixel observations* that the visual FM-PCC policy will condition on
during inference.  Each training sample for Epoch 6 will be a tuple
`(bp_image[t], state[t], action[t])`, where `bp_image[t]` is the overhead camera frame
captured at the same timestep as `state[t]`.

**Why overhead (bird's-eye)?** The bird's-eye view gives the policy a map-like observation:
it can see both the drone and the obstacles from above.  This is the same rationale used
in Gen9 visual avoiding (Gen9 `camera_image_from_state/CHANGELOG.md`).

### 3.2 Two camera streams

| Stream | Type | What it captures | Code |
|---|---|---|---|
| `bp-cam` | Virtual free camera (`MJCAMERA_FREE`) | Overhead view: drone + obstacles + floor | `render_frame_overhead()` line 125 |
| `track-cam` | Body-mounted (`mode=trackcom`, XML `<camera name="track">`) | First-person: drone's forward view | `render_frame()` line 118 |

The `bp-cam` is a *programmatic* free camera — its `lookat` is set to `data.qpos[:3]`
(the drone's current position) at every frame, so it always centres on the drone.
`distance=5.0`, `elevation=−90°` gives a straight-down overhead view.

The `track-cam` is a physical camera body in `quadrotor_modified.xml`.  `mj_name2id`
resolves it by name at the start of each episode.

### 3.3 Color channel convention

MuJoCo renderer returns **RGB** arrays.  WS-A converts to **BGR** before writing PNG
with `cv2.imwrite` (`collect_camera_images.py` lines 185–186):

```python
bp_bgr    = cv2.cvtColor(bp_img,    cv2.COLOR_RGB2BGR)
track_bgr = cv2.cvtColor(track_img, cv2.COLOR_RGB2BGR)
```

Downstream loaders (`cv2.imread`) return BGR by default, so reading these PNGs will
already be in the expected BGR channel order.  This mirrors the Gen9 colour convention
(Gen7 Fix 18.6.1).

### 3.4 Output layout

```
logs/uav_expert_data/
  images/bp-cam/{scene}/{homotopy}/{episode_id}/
    0.png, 1.png, …, (T-1).png
  images/track-cam/{scene}/{homotopy}/{episode_id}/
    0.png, 1.png, …, (T-1).png
```

One PNG per timestep per camera per episode.  Total: 1769 episodes × ~300 steps mean ×
2 cameras ≈ **~1 million images**.

### 3.5 Skip logic

`--skip-existing` (on by default) skips an episode if both `bp-cam` and `track-cam`
output directories exist and are non-empty.  This makes the job restartable after cluster
failures (`collect_camera_images.py` lines 252–255).

**Current status (Fix_2)**: Fix_1 (WS-A) guarded `renderer.close()` with `hasattr` —
the cluster's MuJoCo version lacked this method, crashing after corridor (216 episodes).
After Fix_1 the WS-A job resumed and completed remaining scenes.

---

## 4. WS-B — GIF generation

### 4.1 Physical meaning

WS-B produces human-readable videos of each expert trajectory for *visual quality
inspection*.  GIFs are not consumed by training — they answer the question: "do the stored
states correspond to a UAV actually flying the intended route without hitting obstacles?"

### 4.2 Implementation: Option B1 (render-from-state)

WS-B uses the same state injection loop as WS-A (§2) but instead of saving individual
PNGs, it assembles frames into a GIF:

```
for each timestep t:
    inject state → mj_forward → render bp-cam + track-cam
    stitch side-by-side: [bp (96×96) | track (96×96)] → 192×96 frame
    append to frame list
imageio.mimsave(episode.gif, frames, fps=10)
```

**Frame stitching**: the two views are `np.concatenate([bp_rgb, track_rgb], axis=1)`
(horizontal stack).  The left panel is the overhead map view; the right panel is the
first-person view from the drone's body camera.

Code: `generate_trajectory_gifs.py:replay_to_frames()` lines 136–182.

### 4.3 Text overlay

Each frame gets `cv2.putText` overlays:
- Left panel: `{scene}/{homotopy} t={t}/{T}` — scene identity + timestep counter.
- Right panel: `FPV` — camera label.

These are purely for human inspection.  The overlay burns directly into the pixel array
using OpenCV's antialiased font renderer; the GIF frames are then passed to `imageio` as
RGB arrays (WS-B converts BGR overlay back to RGB before appending).

Code: `generate_trajectory_gifs.py` `_burn_overlay()` lines 128–133, applied at lines
170–176.

### 4.4 Known issue — "UAV hitting wall" in corridor GIFs

When inspecting corridor GIFs (Fix_2 run), episodes appear to show the drone clipping
the wall.  This is a combination of two effects:

1. **Contact filter allows brief contact** (up to 2% of steps → up to 4 steps in a
   200-step episode).  The drone's recorded position in the pickle was genuinely at or
   inside the wall surface during those steps.  The GIF faithfully replays this.

2. **Identity quaternion exaggerates visual contact**.  The drone was actually tilted
   during high-speed flight near the wall; forcing level attitude in the GIF places the
   full upright drone hull at the wall-adjacent COM position, making the contact appear
   more severe.

Full analysis: [`INVESTIGATION_wall_contact_gifs.md`](INVESTIGATION_wall_contact_gifs.md)

### 4.5 Fix history (WS-B)

| Fix | Issue | Resolution |
|---|---|---|
| Fix_1 | `renderer.close()` AttributeError in WS-A after corridor | `hasattr` guard in `collect_camera_images.py` |
| Fix_2 | Same crash in WS-B after corridor (216 GIFs done) | Same `hasattr` guard in `generate_trajectory_gifs.py:263` |

After Fix_2 the WS-B job was resubmitted with `--skip-existing` (corridor 216 GIFs kept).
Empty / pillars / s_curve (1333 episodes) are pending SLURM completion.

---

## 5. WS-C — Mini-FM sanity gate

### 5.1 Purpose

Before investing GPU-days training a full FM-PCC model (Epoch 6), verify that the data
pipeline — schema, action convention, normalisation, dataloader — is internally consistent.

**Pass criterion**: A tiny FM trained on ≤100 empty-scene episodes achieves < 0.1 m RMS
position error on held-out empty-scene episodes.  Empty scene is used because there are
no obstacles; the FM only needs to interpolate between start and end positions.  If this
simple case fails, there is a fundamental data/architecture bug.

### 5.2 What it tests

| Pass condition | What it proves |
|---|---|
| Dataloader produces `(B, H=8, D=9)` tensors | Schema and `actions[3] ‖ obs[6]` ordering are correct |
| Loss decreases monotonically | Normalisation and action convention are consistent |
| RMS < 0.1 m on held-out | The FM velocity field has learned the trajectory manifold, not just noise |

### 5.3 Status

WS-C is not yet executed.  It is the immediate gate before Epoch 6 training begins.
See `EPOCH5_PLAN.md §4` for full pass/fail criteria and failure diagnosis table.

---

## 6. Cluster execution pattern

All three workstreams run on SLURM with `MUJOCO_GL=egl` (headless EGL for offscreen GPU
rendering).  The pattern is:

```
MUJOCO_GL=egl python uav_expert_data_collect/{script}.py [args]
```

All jobs use `--skip-existing` so they are idempotent after crashes — the job can be
resubmitted without losing completed work.

SLURM wrappers:
- `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh` — WS-A
- `Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh` — WS-B

---

## 7. What Epoch 5 does NOT do

- ❌ FM-PCC training on the UAV dataset — Epoch 6
- ❌ Visual encoder training / fine-tuning — Epoch 6
- ❌ DPCC safety-filter integration — Epoch 6 architecture
- ❌ Domain randomisation (texture / lighting variation) — Epoch 6+
- ❌ DAgger on-policy correction — Epoch 7+
- ❌ Multi-controller variants (`pid_high_gain` / `pid_low_gain`) — not collected in E4

---

## 8. Cross-references

| Document | Content |
|---|---|
| [`EPOCH5_PLAN.md`](EPOCH5_PLAN.md) | Full plan, workstream specs, risk register |
| [`INVESTIGATION_wall_contact_gifs.md`](INVESTIGATION_wall_contact_gifs.md) | "UAV hitting wall" root cause analysis |
| [`Fix_1/CHANGELOG.md`](Fix_1/CHANGELOG.md) | WS-A `renderer.close()` AttributeError fix |
| [`Fix_2/CHANGELOG.md`](Fix_2/CHANGELOG.md) | WS-B same fix; corridor 216 GIFs preserved |
| [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md) | How the Epoch 4 state pickles were generated |
| [`../Epoch4_expert_data/CLOSURE.md`](../Epoch4_expert_data/CLOSURE.md) | Final dataset stats + fix history |
