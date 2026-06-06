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
