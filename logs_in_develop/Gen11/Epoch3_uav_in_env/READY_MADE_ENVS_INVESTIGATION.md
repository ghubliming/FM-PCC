# Investigation — Are there ready-to-use UAV environments in the workspace?

**Date**: 2026-05-31
**Audience**: future Gen11 / Epoch 4+ developers deciding whether to keep hand-writing scenes or vendor existing ones.
**TL;DR**: **Partially yes for `mujoco_mpc`, no for `UAV-Flow`** (it's a UE4 simulator, not MuJoCo). Our hand-written scenes from Epoch 3 are still appropriate for the DPCC obstacle-avoidance line of work — but **MJPC's `task.xml` + `gates.xml` is a free pre-built racing env we could vendor if we want quick-win "drone in a designed scene"** with zero environment-design effort.

---

## 1. `UAV-Flow` — not a MuJoCo project, but useful for trajectory data

### What it is
- **NeurIPS 2025** benchmark for instruction-conditioned UAV imitation learning.
- Built on **UnrealZoo Gym** (UnrealCV wrapper around Unreal Engine 4).
- Eval env package: `UAV-Flow/UAV-Flow-Eval/gym_unrealcv/`.

### What's "ready" in there
- **Pre-packaged UE4 scenes**: `DowntownWest` (campus), `SuburbNeighborhood_Day`, `SuburbNeighborhood_Night`, `Demo_Roof` (`UAV-Flow-Eval/gym_unrealcv/envs/setting/{Track,Navigation}/`).
- Scene configs are JSON references to a **Windows-only UE4 binary** (`Collection_WinNoEditor_0424_25.zip`, ~GB-scale download).
- Eval harness `batch_run_act_all.py` runs trajectories against these scenes and scores them with nDTW.

### Why we can't drop it into FM-PCC
- **Wrong simulator.** UE4 / UnrealCV ≠ MuJoCo. There's no XML, no `mj_step`, no shared physics interface.
- **Windows binary requirement.** Our Slurm cluster is Linux; UnrealZoo's pre-packaged binary doesn't run there without a Wine + GPU passthrough setup we don't have.
- **No physics fidelity for control.** UnrealZoo gives photorealistic visual eval, but the underlying flight model is not the same as our Skydio X2 MuJoCo dynamics — controller transfer would not be apples-to-apples.

### What IS useful from UAV-Flow
- **`UAV-Flow-Sim` trajectory dataset** (HuggingFace: `wangxiangyu0814/UAV-Flow-Sim`) — simulated UAV trajectories.
- **`UAV-Flow` real-world trajectory dataset** (HuggingFace: `wangxiangyu0814/UAV-Flow`) — real flights.
- Either can be **mined for trajectory statistics** (typical horizon, sampling rate, distance per episode, velocity/accel distributions) per the original `path_temp_initial.md` step 3 ("Mirror UAV‑Flow trajectory statistics").
- Their **OpenVLA-UAV checkpoint** could serve as a baseline policy to compare FM-PCC against — but not as an env.

### Verdict for env work
**Skip for env, mine for trajectories.** Add to Epoch 4 backlog: when we need expert demos or trajectory-shape priors, pull from `UAV-Flow-Sim`. For now, no env benefit.

---

## 2. `mujoco_mpc` — does have a pre-built quadrotor env, but with caveats

### What we already use (Epoch 1)
- `quadrotor_modified.xml` (X2 model only, no scene) — vendored.
- `gates.xml` (8 racing gates as `<body>`/`<geom>`) — vendored but **unused** in Epoch 3.

### What we did NOT vendor and **could vendor**

#### 2.1 `mjpc/tasks/quadrotor/task.xml` — the real racing env
This file is **a ready-made scene**: floor, 11 waypoints (mocap goal + 10 ghost markers), `<include>` of the X2 model, `<include>` of `gates.xml`. Plus a `<keyframe>` block with all 11 waypoint positions for replay. Plus a "track" camera tied to the X2 body.

```xml
<!-- skeleton of mjpc/tasks/quadrotor/task.xml -->
<mujoco model="Quadrotor Racing">
  <include file="../common.xml"/>          ← MJPC's standard scene assets
  <include file="quadrotor_modified.xml"/> ← the X2 we already have
  <include file="gates.xml"/>              ← the 8 gates

  <worldbody>
    <geom name="floor" type="plane" .../>
    <body name="goal" mocap="true" pos="1.2 0.0 0.75">...</body>
    <body name="wp1" pos="1.2 0.0 0.75">...</body>
    ... 10 more waypoints ...
  </worldbody>

  <custom>
    ... 14 MJPC planner-specific numeric parameters ...   ← STRIP THIS
  </custom>
</mujoco>
```

**The `<custom>` block is MJPC-specific** (predictive-sampling planner config). Strip those ~15 lines and the rest is a standard MuJoCo scene that loads anywhere.

#### 2.2 `mjpc/tasks/common.xml` — reusable scene assets
Headlight, blue/grey grid materials, skybox gradient texture, standard colour palette. Better-looking than what we wrote by hand in `scene_empty.xml`. Drop-in replacement for our `<asset>` and `<visual>` blocks.

#### 2.3 `mjpc/tasks/quadrotor/quadrotor.cc` — task-Python translation source
Defines the residual + waypoint-transition function in C++. Already documented in [`../Epoch1_UAV_model/MUJOCO_MPC_UAV_MATH.md`](../Epoch1_UAV_model/MUJOCO_MPC_UAV_MATH.md) §3. Useful as the **canonical spec** when Epoch 4 needs to port it to Python (waypoint advance, residual computation for any cost-based work).

### Trade-offs of vendoring MJPC's racing env vs. our hand-written scenes

| Aspect | MJPC `task.xml` (vendored) | Our Epoch 3 scenes |
|---|---|---|
| Effort to add | 30 min (`cp` + strip `<custom>`) | Already done |
| Scene complexity | Realistic racing course (8 gates, 11 waypoints) | Simple primitives (boxes, cylinders) |
| Obstacle geometry for DPCC | Hard: gates are 4 capsule segments each → ~32 geoms → many halfspaces | Easy: each scene has 2-6 primitive geoms → ~6-24 halfspaces |
| Visual demo value | High (looks like real drone racing) | Low (functional but austere) |
| Reproduces MJPC's published demo | Yes, exactly | No |
| Aligned with FM-PCC obstacle work | No — gates aren't "avoid", they're "fly through" | Yes — corridors / pillars are unambiguous obstacles |
| Maintenance burden | Vendor sync if MJPC updates | None (it's ours) |

---

## 3. Recommendation for future dev

### When to use **our Epoch 3 scenes**
- Default. DPCC obstacle-avoidance work (Epoch 4+) wants simple halfspace-able geometry. Corridor and pillar field are easier to express as `Z_f^t = {x : a^T x ≤ b}` halfspaces than MJPC's capsule-gate constructs.
- All FM-PCC work targeting **avoidance** rather than gate-passing.

### When to vendor **MJPC's `task.xml`**
- "Drone racing" demos where the visual story matters (papers, slide decks).
- Comparing our FM-PCC stack to MJPC's published predictive-sampling baseline on the same scene (apples-to-apples).
- Quick-win to get a more visually-impressive GIF than our hand-written scenes provide.

If we vendor it, target path:
```
d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_mjpc_racing.xml
```
Steps:
1. `cp mujoco_mpc/mjpc/tasks/quadrotor/task.xml → scene_mjpc_racing.xml`
2. Update `<include file="...">` paths to point at our local `quadrotor_modified.xml` and `gates.xml`.
3. Update `<include file="../common.xml">` either to point at MJPC's vendored common.xml (also `cp`'d) or replace with our minimal asset/visual block.
4. **Delete the `<custom>` block** (~15 lines, MJPC-planner-specific).
5. Add a corresponding `task=racing` branch in `uav_env_test/run_env.py` that uses MJPC's waypoint sequence as the trajectory (already in the file's `<keyframe>` block).

Effort: ~30 minutes. Risk: low (we already verified MJPC's `quadrotor.xml.patch` applies cleanly in Epoch 1).

### When to mine **UAV-Flow datasets**
- Epoch 4+ when we need expert trajectories for FM training.
- Pull from `huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim`.
- Analyse for: typical episode length, dt, velocity/accel distributions, altitude profile.
- Use as either (a) trajectory-statistics target for a synthetic expert in MuJoCo, or (b) pretrain corpus for an FM policy before MuJoCo fine-tuning.
- **Do not** try to use UAV-Flow's UE4 eval env; it doesn't fit our toolchain.

---

## 4. What's NOT useful in either repo

- **No MuJoCo UAV envs in any other workspace repo.** Checked: `d3il`, `diffuser`, `dpcc`, `drifting`, `drifting_policy`, `imeanflow`, `SafeFlowMPC` — none ship a quadrotor scene. The X2 in `mujoco_mpc` is the only MuJoCo-format UAV in the workspace.
- **No DPCC-compatible halfspace exporter exists yet** — neither MJPC nor UAV-Flow provides one. We'll need to write it ourselves in Epoch 4 (per PLAN.md §3 deferred `obstacles.py` stub).

---

## 5. One-line summary

For the obstacle-avoidance line we're pursuing: **keep our Epoch 3 scenes.** If a future epoch wants a flashier visual demo or apples-to-apples comparison with MJPC's baseline, **`cp mjpc/tasks/quadrotor/task.xml` + strip `<custom>` block** gives us a pre-designed racing course in 30 minutes. UAV-Flow gives us **trajectories only**, not envs.

---

## 6. Were the E3 UAV collisions fixed in Epoch 4 and 5?

E3 produced two failure cases in the obstacle scenes.  This section records whether they
were resolved in subsequent epochs.

### s_curve — 41% contact steps ✅ Fixed in E4

**Root cause**: `s_curve_path` chained `traverse_line` segments with `v_des = 0` at each
internal waypoint junction (t=5 s, t=10 s).  At those moments the drone entered the
`Kp_omega = 10` limit-cycle regime — saturating the angular-rate controller →
violent oscillation → drone wedged into the corridor walls for ~6 s of each 15 s episode.

**Fixed by two independent mechanisms before any E4 data was collected:**

1. **`Kp_omega` correction** (`EPOCH4_EXECUTION_PLAN.md` Decision 2):  
   `flight_controller.py` updated `[10.0, 10.0, 2.0]` → `[2.5, 2.5, 1.0]`.  
   Eliminates the saturation condition at its root.

2. **Trajectory redesign** (E4 Fix_5.1):  
   `s_curve_scene_path()` rewritten from piecewise-stops to a **3-segment
   proportional-duration `traverse_line`** — each segment's time budget is proportional
   to its Euclidean length, so no segment forces `v = 0` at an internal joint.  The
   critical diagonal gap crossing (1.89 m) gets 5.18 s instead of ~3.2 s → peak speed
   drops from 1.17 m/s to 0.57 m/s.

E4 s_curve final rejection rate: **28.8%** (under an 8% per-scene contact threshold
introduced in Fix_4 to accept narrow end-face grazes).

### pillars weave — 2.9% contact / 0.922 m RMS ✅ Accepted

**Root cause**: 0.25g lateral demand → ~14° tilt → phase lag in y → 29 grazing contacts
over 10 s.  The drone always passed on the correct side of each pillar and reached the
endpoint (final error 0.062 m).  This was a tracking-lag result, not a control failure.

E4 uses the same `weave()` trajectory factory with homotopy-specific amplitude parameters.
E4 pillar rejection: **4.6%** (477 episodes saved) — similar behaviour, within tolerance.
Pillar grazing under high lateral demand is considered acceptable training data (realistic
near-miss behaviour for a visual FM policy).

### corridor — 0 contact in E3 → ⚠️ New concern in E4/E5

E3 corridor was clean (0.023 m RMS, 0 contact steps).  No E3 problem to fix.

In E4, corridor data was collected with a **2% contact threshold** (up to 4 contact steps
per 200-step episode allowed).  E5 WS-B GIFs revealed that some accepted episodes
genuinely show the drone clipping the wall at speed.  This is not a regression from E3 —
it is a threshold policy decision made in E4 that the investigation exposed.

The **E4 U2 upgrade plan** (`../Epoch4_expert_data/U2/PLAN.md`) tightens the corridor
threshold from 2% → 1% as Change B of the next re-collection.

### Resolution status

| E3 problem | Fixed in E4? | Fixed in E5? | Mechanism |
|---|---|---|---|
| `s_curve` 41% contact (limit-cycle at v=0) | ✅ Yes | N/A — E5 is replay-only | `Kp_omega` [10→2.5] + proportional-duration trajectory (Fix_5.1) |
| `pillars` 2.9% contact / 0.922 m RMS | ✅ Accepted | N/A | Tracking lag within tolerance; weave factory kept |
| `corridor` (E3 was clean) | N/A | ⚠️ New concern found | E4 2% threshold allows brief contact; U2 tightens to 1% |

Full root-cause analysis: [`METHODOLOGY.md`](METHODOLOGY.md) §Test results.  
Full E5 corridor GIF investigation: [`../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`](../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md).  
U2 re-collection plan: [`../Epoch4_expert_data/U2/PLAN.md`](../Epoch4_expert_data/U2/PLAN.md).
