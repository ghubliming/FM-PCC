# Gen11 Epoch 5 — U2 Upgrade Plan

**Date**: 2026-06-07  
**Status**: Planning — not yet executed  
**Predecessor**: E4 U2 (obs 6D→9D, dataset verified 1769 episodes, obs=(T,9)=[p_des(3),p(3),v(3)])  
**Driving docs**:
- [`../../Epoch4_expert_data/U2/PLAN.md`](../../Epoch4_expert_data/U2/PLAN.md) — E4 U2 template
- [`../../Epoch4_expert_data/U2/CLOSURE.md`](../../Epoch4_expert_data/U2/CLOSURE.md) — final 9D obs schema
- [`../../Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md`](../../Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md) — WS-A/B unaffected claim (conditional on correct column extraction)
- [`../METHODOLOGY.md`](../METHODOLOGY.md) — current E5 state injection code

---

## Why U2 is needed

Four issues discovered after E4 U2 closed:

| # | Issue | Type | Trigger |
|---|---|---|---|
| A | WS-A / WS-B state injection reads wrong obs columns after 9D widening | Code bug | E4 U2 changed obs layout |
| B | WS-C mini-FM config still references 9D tensor (D=9) not 12D | Config error | E4 U2 widened FM tensor |
| C | `track-cam` uses `mode="trackcom"` (chase cam) instead of body-frame FPV | Design error | Never analysed — chosen because it "Already exists" |
| D | Drone is forced to identity quaternion in all renders — FPV is misleading, sim-to-real gap will be large | Architecture gap | Quaternion not stored in E4 pickles |

**A and B must be fixed before any re-run.**  
**C and D are design corrections** — C is a one-line XML fix; D requires an E4 U3 re-collection to store quaternion then a WS-A/B re-render.

---

## Change A — State injection column fix (WS-A + WS-B)

### Root cause

After E4 U2, obs layout is:

```
obs[t, :3]  = p_des   ← commanded position  (NOT what state injection needs)
obs[t, 3:6] = p       ← actual position      ← correct column for qpos
obs[t, 6:9] = v       ← velocity             ← correct column for qvel
```

Both scripts hardcoded the pre-U2 6D layout and were never updated.  The drone is
currently rendered at `p_des` instead of `p` — a systematic offset of up to ~3 cm
(PID lag).  This is the **same class of bug** as E4 stats_validator Fix_1.

### Fix — `collect_camera_images.py`

**File**: `uav_expert_data_collect/collect_camera_images.py`

```python
# BEFORE (line 152, 168-169):
obs = episode['obs']        # (T, 6) = [p(3), v(3)]
...
p = obs[t, :3]
v = obs[t, 3:6]

# AFTER:
obs = episode['obs']        # (T, 9) = [p_des(3), p(3), v(3)]  U2: p_des prepended
...
p = obs[t, 3:6]   # U2: p shifted to columns 3:6
v = obs[t, 6:9]   # U2: v shifted to columns 6:9
```

### Fix — `generate_trajectory_gifs.py`

**File**: `uav_expert_data_collect/generate_trajectory_gifs.py`

```python
# BEFORE (line 142, 159-162):
obs = episode['obs']  # (T, 6)
...
data.qpos[:3] = obs[t, :3]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = obs[t, 3:6]
data.qvel[3:6] = 0.0

# AFTER:
obs = episode['obs']  # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
...
data.qpos[:3] = obs[t, 3:6]   # U2: p at columns 3:6
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = obs[t, 6:9]   # U2: v at columns 6:9
data.qvel[3:6] = 0.0
```

### Re-collection assessment

**WS-A (camera images)**: collected 2026-06-06, one day *before* E4 U2 replaced the
pickles.  At that time obs was 6D and `obs[:, :3]` was correctly p.  Existing images are
valid.  After Change D re-renders with attitude, all WS-A images will be replaced anyway.

**WS-B (GIFs)**: Fix_2 resubmit was pending as of 2026-06-06; E4 U2 pickles landed
2026-06-07.  Timing overlap is uncertain — some GIFs may have been generated against the
new 9D pickles at wrong position.  Check corridor GIFs after applying Change A; re-run
with `--no-skip` if drone appears systematically shifted toward the wall.

---

## Change B — WS-C mini-FM gate: update tensor dim

WS-C has not yet run.  The E4 U2 FM tensor is now 12D (`action_dim=3 + obs_dim=9`).

### Config update

| Location | Old | New |
|---|---|---|
| `mini_fm_sanity.py` `obs_dim` | `6` | `9` |
| `mini_fm_sanity.py` `transition_dim` | `9` | `12` |
| `EPOCH5_PLAN.md` §4.2 config row | `D=9` | `D=12` |
| `METHODOLOGY.md` §5.2 pass condition | `(B, H=8, D=9)` | `(B, H=8, D=12)` |

With 9D obs, WS-C also implicitly verifies that `p_des` sits correctly at `obs[:, :3]` and
that the 12D tensor flows through the dataloader.  Pass criterion unchanged: RMS < 0.1 m
on held-out empty-scene episodes.

---

## Change C — Fix `track-cam` to true body-frame FPV

### Why it is wrong

The E5 CHANGELOG records the entire rationale for choosing `track-cam`:

> `"track"` camera from `quadrotor_modified.xml` (line 35) — **Already exists**: `mode="trackcom"`, body-mounted on X2

No comparison with D3IL `inhand_cam` was made.  The E5 audit confirmed it "correct"
without questioning the mode.  `mode="trackcom"` in MuJoCo means position follows the
body COM but orientation is **world-frame fixed** — a chase cam, not a body-frame cam.

### Comparison

| Property | D3IL `inhand_cam` | Current `track-cam` | After Change C |
|---|---|---|---|
| Position frame | Body — moves + rotates with wrist | Translates with COM only | Body — moves + rotates with drone |
| Orientation when drone pitches | Pitches with gripper | World-fixed (no rotation) | Pitches with drone ✅ |
| Analog of inhand cam? | ✅ Yes | ❌ No — chase cam | ✅ Yes |

### Fix

**File**: `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml`, line 35.

```xml
<!-- BEFORE — chase cam -->
<camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>

<!-- AFTER — body-frame FPV (mode omitted = "fixed" inside a body element) -->
<camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2"/>
```

### Effect now vs after Change D

With forced identity quaternion (current): zero visual difference — body never rotates so
chase-cam and body-cam are identical.  Change C becomes visually meaningful only after
Change D removes the forced-level constraint.  Apply Change C now to have the correct
semantics in place before Change D lands.

---

## Change D — Attitude-aware rendering (E4 U3 + WS-A/B re-render)

### The problem

Every frame in WS-A and WS-B renders the drone level (`qpos[3:7] = [1,0,0,0]`).  In real
flight the drone tilts to accelerate: at 0.7 m/s corridor speed the pitch angle is ~10°
nose-down.  For the FPV (body-frame) camera that is a visible and semantically important
difference — the camera's viewing direction rotates with the drone.

**Two concrete consequences of keeping forced-level:**

1. **Change C is semantically inert** — the FPV body-frame fix has zero visual effect as
   long as the body is forced level.
2. **Sim-to-real gap** — the visual FM trains on level-drone images.  At inference on a
   real drone (or in a physics-accurate sim), the drone is tilted.  The model has never
   seen that view.

Deferring this to Epoch 7 means re-collecting E4 **and** re-rendering all of WS-A/B a
second time.  Fixing it now costs one extra E4 re-collection but avoids the second full
re-render later.

### What is needed (E4 U3)

The quaternion is computed during the E4 physics simulation but was never saved to the
pickle.  To fix this, E4 U3 must:

1. In `generator.py` step dict: add `'q': np.asarray(data.qpos[3:7])` alongside `p`, `v`, `p_des`.
2. In `dataset_writer.py`: write `q = np.array([s['q'] for s in steps], dtype=np.float32)` as a new `(T, 4)` field in the pickle.
3. Re-collect all 4 scenes (same as E4 U2 re-collection — ~2 hours, 4 SLURM jobs).

### What changes in WS-A/B after E4 U3

State injection uses actual quaternion instead of identity:

```python
# BEFORE (forced level):
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

# AFTER (attitude-aware):
data.qpos[3:7] = ep['q'][t]   # actual quaternion from physics rollout
```

Both `collect_camera_images.py` and `generate_trajectory_gifs.py` need this change in
their state injection loops (same lines as Change A).

WS-A images and WS-B GIFs must be fully re-rendered after E4 U3.

### Alternative: estimate quaternion from velocity (no re-collection)

Pitch can be estimated from acceleration: `pitch ≈ arctan2(ax, g)` where
`ax = (v[t] - v[t-1]) / dt`.  This avoids re-collecting E4 but:
- gives pitch only (no roll, no yaw)
- amplifies noise via finite difference (same problem as DPCC_OBS_DEVIATION §"9D incl a")
- is wrong during turns or while the PID attitude controller is settling

**Recommendation: re-collect (E4 U3).** The estimate is too noisy and incomplete to be
useful for training data.

### bp-cam — realistic for sim only

`bp-cam` (overhead, elevation=−90°) is a sim-only training convenience — no real UAV has
a cage-mounted overhead camera.  With attitude rendering, the FPV cam becomes the only
realistic camera stream:

| Epoch | bp-cam role |
|---|---|
| E5–E6 | Primary training camera (collect both streams, bp-cam used for training) |
| E7+ sim-to-real | **Drop bp-cam**; train and evaluate on onboard FPV cam only |

---

## Execution — all code first, then run

### Phase 1 — Apply all code and config fixes (no cluster jobs yet)

Do all of the following before submitting any SLURM job:

| # | Fix | File | What to change |
|---|---|---|---|
| A1 | WS-A obs columns | `collect_camera_images.py` | line 152 comment; lines 168-169 (`[:3]→[3:6]`, `[3:6]→[6:9]`) |
| A2 | WS-B obs columns | `generate_trajectory_gifs.py` | line 142 comment; lines 159, 161 (`[:3]→[3:6]`, `[3:6]→[6:9]`) |
| B | WS-C tensor dim | `mini_fm_sanity.py` | `obs_dim=9`, `transition_dim=12` |
| C | XML camera mode | `quadrotor_modified.xml` line 35 | remove `mode="trackcom"` |
| D-prep | E4 step dict | `generator.py` step append | add `'q': data.qpos[3:7]` |
| D-prep | E4 pickle schema | `dataset_writer.py` | write `q = (T, 4)` field |
| D-prep | WS-A quat injection | `collect_camera_images.py` | `qpos[3:7] = ep['q'][t]` (same location as A1) |
| D-prep | WS-B quat injection | `generate_trajectory_gifs.py` | `qpos[3:7] = obs_q[t]` (same location as A2) |

### Phase 2 — Re-collect and re-render (SLURM)

After all Phase 1 edits are committed:

**Step 1 — E4 U3 re-collection** (adds quaternion to pickles):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
```

Expected: ~2 hours, same episode counts as E4 U2 (1769 total).

**Step 2 — WS-A re-render** (attitude-aware images, full re-collect with `--no-skip`):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh
```

**Step 3 — WS-B re-render** (attitude-aware GIFs, full re-collect with `--no-skip`):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh
```

### Phase 3 — Test and verify

After Phase 2 completes:

**WS-A verification**:
```python
import os, pickle
ep = pickle.load(open('logs/uav_expert_data/corridor/C/<ep>.pkl', 'rb'))
print(ep['obs'].shape)    # expect (T, 9)
print(ep['q'].shape)      # expect (T, 4)  ← new field
```
Spot-check a FPV frame: at mid-trajectory the drone should appear slightly nose-down, not
perfectly level.

**WS-B verification**: Open a corridor L or R GIF — the FPV panel should show the drone
tilted when moving fast near a wall, not rigidly level at all speeds.

**WS-C run**:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/mini_fm_sanity.sh
```
Pass: RMS < 0.1 m; tensor shape `(B, H=8, D=12)` confirmed; no NaN.

**XML verify**:
```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml')
print('ncam =', m.ncam)
"
```

---

## Summary of all changes

| Change | Type | Files touched | Re-run? |
|---|---|---|---|
| A1 — WS-A obs columns | Code bug | `collect_camera_images.py` | Via Phase 2 WS-A re-render |
| A2 — WS-B obs columns | Code bug | `generate_trajectory_gifs.py` | Via Phase 2 WS-B re-render |
| B — WS-C tensor dim | Config | `mini_fm_sanity.py` | First WS-C run (not yet run) |
| C — XML camera body-frame | Design fix | `quadrotor_modified.xml` | No extra run; takes effect in Phase 2 re-render |
| D — Quaternion in E4 pickles + attitude render | Architecture | `generator.py`, `dataset_writer.py`, `collect_camera_images.py`, `generate_trajectory_gifs.py` | ✅ E4 re-collect + full WS-A/B re-render |

---

## What E5 U2 does NOT address

- ✅ **Quaternion in FM obs** — quaternion is deliberately absent from the FM observation. The FM is a position-space planner (`[p_des, p, v]`); attitude is a low-level PID concern. Follows Gen11 design (Gen11 Path §step 5) and the D3IL avoiding pattern. Not a gap — by design.
- ❌ **WS-A image re-collection** — existing images valid (pre-U2 collection); fix code before any future re-run
- ❌ **bp-cam removal** — still both streams collected in E5/E6; flagged for E7+ sim-to-real transition
- ❌ **FM training** — Epoch 6 scope
- ❌ **Domain randomisation** (texture, lighting) — Epoch 6+

---

## Cross-references

| Document | Content |
|---|---|
| [`../../Epoch4_expert_data/U2/CLOSURE.md`](../../Epoch4_expert_data/U2/CLOSURE.md) | Why E4 obs is now 9D; FM tensor 12D |
| [`../../Epoch4_expert_data/U2/Fix_1/CHANGELOG.md`](../../Epoch4_expert_data/U2/Fix_1/CHANGELOG.md) | stats_validator Fix_3 — same column-shift pattern as Change A |
| [`../../Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md`](../../Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md) | WS-A/B unaffected claim; FiLM reusability; noise amplification from finite-differencing |
| [`../METHODOLOGY.md`](../METHODOLOGY.md) | §2 state injection; §3.2 camera streams; §5 WS-C spec |
| [`../EPOCH5_PLAN.md`](../EPOCH5_PLAN.md) | WS-C pass criteria (D=9 → update to D=12) |
| [`../CHANGELOG.md`](../CHANGELOG.md) | "Already exists" rationale for track-cam (the decision that Change C corrects) |
| `quadrotor_modified.xml` line 35 | `<camera name="track" ... mode="trackcom"/>` — the line Change C edits |
