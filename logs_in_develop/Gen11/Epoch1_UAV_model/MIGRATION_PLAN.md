# Gen11 — Migrating the Skydio X2 Quadrotor (UAV) from `mujoco_mpc` into FM-PCC / D3IL

**Date**: 2026-05-30
**Status**: Plan only — no files moved yet.
**Source repo**: `/workspaces/mujoco_mpc/mjpc/tasks/quadrotor/`
**Target repo**: `/workspaces/FM-PCC/d3il/environments/d3il/`
**Reference**: [`notes.md`](notes.md) — model overview from prior investigation

---

## 1. Goal and Scope

**Goal:** Get the Skydio X2 quadrotor model running inside FM-PCC as a
**D3IL-style gym env** (`ObstacleAvoidanceEnv`-shaped) so that downstream
FM-PCC training / eval pipelines can target it the same way they target
aligning, avoiding, pushing, etc.

**In scope:**
- MuJoCo XML model + mesh + texture (the actual physics).
- Racing-gate / waypoint *geometry* (positions + visual markers).
- A new D3IL env class (`QuadrotorRacingEnv`) with a Python action contract
  matching the X2 actuator set.
- A minimal hover / waypoint controller for sanity checks.

**Out of scope (MJPC-specific, not portable):**
- `quadrotor.cc` / `quadrotor.h` — MJPC C++ residuals and transition
  function. Replaced by a Python equivalent inside the new env class.
- `custom { numeric ...}` block in `task.xml` — MJPC predictive-sampling
  planner parameters. FM-PCC has its own planner; these have no meaning
  here.
- The `.xml.patch` file — applied inline once during migration; no need to
  carry the patch format itself.

---

## 2. Source Inventory and Per-File Disposition

| Source (under `mujoco_mpc/mjpc/tasks/quadrotor/`) | What it is | Disposition |
|---|---|---|
| `task.xml` | Top-level MJPC task: includes common.xml, quadrotor model, gates; defines goal mocap, 11 waypoints, sensors, keyframes, MJPC custom planner params | **Split**: keep waypoint + gate world bodies + sensor block; drop `<custom>` MJPC planner block; drop `<include common.xml>` (FM-PCC has its own scene); fold include of quadrotor model |
| `quadrotor.xml.patch` | Diff against the Menagerie Skydio X2 base XML: adds quat init, drops MJPC-only sensors and keyframes | **Apply once**, commit the result as `quadrotor_modified.xml`. Discard the patch file. |
| `gates.xml` | 8 racing gates as static `<body><geom>` groups | **Copy verbatim** into the D3IL model tree |
| `quadrotor.cc` / `.h` | C++ residual + waypoint-transition logic for MJPC sampling planner | **Reimplement in Python** inside the new env class. Residual maps to a per-step cost; transition maps to a Python waypoint pointer that advances when ‖pos − wp‖ < 0.5 m. ~30 lines of Python. |
| (referenced) Menagerie Skydio X2 source: `X2_lowpoly.obj`, `X2_lowpoly_texture_SpinningProps_1024.png` | Mesh + texture | **Fetch from MuJoCo Menagerie** (`google-deepmind/mujoco_menagerie/skydio_x2/`), copy mesh + texture into the D3IL asset tree. The X2 XML in MJPC refers to them; we need the actual files. |

---

## 3. Target Placement in FM-PCC / D3IL Tree

Mirroring D3IL's existing per-task layout (avoiding, aligning, …):

```
d3il/environments/d3il/
├── envs/
│   └── gym_quadrotor_env/                     ← NEW package, mirrors gym_avoiding_env
│       ├── __init__.py
│       ├── setup.py
│       └── gym_quadrotor/
│           ├── __init__.py
│           └── envs/
│               ├── __init__.py
│               ├── quadrotor.py               ← env class (QuadrotorRacingEnv)
│               └── objects/
│                   ├── __init__.py
│                   └── quadrotor_objects.py   ← waypoints / gates / goal mocap factory
├── models/
│   └── mj/
│       ├── robot/
│       │   └── quadrotor/                     ← NEW
│       │       ├── quadrotor_modified.xml     ← from .patch + Menagerie base
│       │       ├── X2_lowpoly.obj             ← from Menagerie
│       │       └── X2_lowpoly_texture_SpinningProps_1024.png
│       └── common-objects/
│           └── quadrotor_gates/               ← NEW (or under workspace/)
│               └── gates.xml
└── d3il_sim/
    └── sims/
        └── mj_beta/
            └── MjQuadrotor.py                 ← NEW (see §4)
```

**Conventions retained from D3IL:**
- Per-task env is a pip-installable sub-package with its own `setup.py`.
- All XML lives under `models/mj/...`.
- Sim wrappers go under `d3il_sim/sims/mj_beta/`.

---

## 4. D3IL-Side Adaptations (the non-trivial part)

D3IL's plumbing is built around **manipulator** semantics (Panda arm,
Cartesian controller, in-hand camera, gripper width). The quadrotor breaks
several of those assumptions. Three adaptations needed:

### 4.1 New robot wrapper: `MjQuadrotor`

`MjRobot` (`d3il_sim/sims/mj_beta/MjRobot.py:62`) hardcodes a Panda arm with
`cartesianPosQuatTrackingController`, gripper, in-hand camera, etc. None of
that applies to a free-floating X2.

**Decision: bypass `MjRobot` entirely, write a thin `MjQuadrotor` class** that
exposes only what `GymEnvWrapper` actually needs from a "robot":
- `current_c_pos`, `current_c_quat` (read from sensors)
- `current_c_vel`, `current_c_avel`
- `beam_to_joint_pos()` for resets
- A `set_thrusts(ctrl4)` method (replaces the cartesian controller)

We do **not** subclass `MjRobot` because too much of its surface assumes a
Panda XML structure (joint names, IK chains, gripper). A small standalone
class is cleaner.

### 4.2 Controller swap

Aligning/avoiding envs use `cartesianPosQuatTrackingController` and pass a
7-D target pose to `env.step()`. The quadrotor takes a **4-D thrust vector**
directly into `data.ctrl[:4]`.

Two options:
- **(a) Direct thrust action** (what MJPC does). `env.step(np.array([t1, t2, t3, t4]))` → writes to `data.ctrl[:4]`, advances n_substeps. Lowest-level, lets a policy learn raw motor commands.
- **(b) Wrap an attitude / position controller** in Python (e.g. a cascaded PID) so `env.step(target_pos_3d)` flies the drone. Higher-level, matches the D3IL "cartesian-position-tracking" interface.

**Recommendation: ship (a) first**, optionally add (b) as a wrapper later.
(a) is what existing FM-PCC trajectory models would consume directly
(velocity / position targets are not naturally the X2's action space).

### 4.3 GymEnvWrapper subclass

`GymEnvWrapper.step()` (`d3il_sim/gyms/gym_env_wrapper.py:45`) takes
`(action, gripper_width=None, desired_vel=None, desired_acc=None)` and calls
`self.controller.execute_action(...)`. For the quadrotor we don't go through
a controller; we write directly to `data.ctrl`.

Cleanest path: **override `step()` in `QuadrotorRacingEnv`** to:
1. write `action` (4-D thrust) into `data.ctrl[:4]`
2. advance `n_substeps` via `self.scene.next_step()` (same as base)
3. read sensors, return `(obs, reward, done, info)`

Don't try to fit through `GymEnvWrapper`'s controller path — it doesn't
generalise here.

---

## 5. The Python Port of `quadrotor.cc` (Residual + Transition)

Both fit in ~30 lines. Drop these onto `QuadrotorRacingEnv`:

```python
GOAL_REACHED_DIST = 0.5   # matches mju_norm3 threshold in quadrotor.cc:76

def _residual(self):
    pos     = self._sensor('position')
    linvel  = self._sensor('linear_velocity')
    angvel  = self._sensor('angular_velocity')
    goal    = self._mocap_pos('goal')
    thrust_hover = ((self.model.body_mass[0] + self.model.body_mass[1])
                    * np.linalg.norm(self.model.opt.gravity) / 4.0)
    return np.concatenate([
        pos - goal,                                # 3
        linvel,                                    # 3
        angvel,                                    # 3
        self.data.ctrl[:4] - thrust_hover,         # 4 (hover-deviation cost)
    ])  # → 13-D residual; FM-PCC sums to scalar cost in trainer if needed

def _advance_waypoint(self):
    pos = self._sensor('position')
    goal = self._mocap_pos('goal')
    if np.linalg.norm(pos - goal) <= GOAL_REACHED_DIST:
        self.current_wp = (self.current_wp + 1) % self.n_waypoints
        self._set_mocap_pos('goal', self.waypoints[self.current_wp])
```

Call `_advance_waypoint()` at the end of every `step()`. Optionally surface
`_residual()` as part of `info` so trainers can use it as a per-step cost.

---

## 6. Phased Migration

Three checkpoints, each independently testable:

### Phase A — Static model loads in MuJoCo (no D3IL plumbing)

**Deliverable:** A standalone Python script in `temp/uav_smoketest.py`
that does `mujoco.MjModel.from_xml_path(...)` on the migrated
`quadrotor_modified.xml`, steps the sim for 1 second, prints final pose.

**Why first:** Validates that the XML + mesh + texture all resolve under
D3IL's path conventions before any D3IL classes get involved.

**Outputs:**
- `models/mj/robot/quadrotor/quadrotor_modified.xml` placed and loadable.
- Mesh + texture files in place.
- `gates.xml` placed (optional inclusion — verify standalone first).

### Phase B — `QuadrotorRacingEnv` constructs and resets

**Deliverable:** `env = QuadrotorRacingEnv(render=False); env.reset()`
returns a valid initial obs without errors.

**Includes:**
- `MjQuadrotor` wrapper.
- `quadrotor_objects.py` (waypoint factory, mocap goal).
- `QuadrotorRacingEnv.__init__()` using `MjFactory` + scene composition.
- A no-op `step(np.zeros(4))` that advances `n_substeps`.

**Smoke check:** `python -c "from ... import QuadrotorRacingEnv; e =
QuadrotorRacingEnv(); e.reset(); e.step(np.zeros(4)); print('ok')"` runs to
completion.

### Phase C — Hover and waypoint-track in Python

**Deliverable:** Open-loop hover + closed-loop next-waypoint controller
demo. Drone holds altitude for 5 s, then chases waypoints 1→11→loop.
Validates that the C++ residual / transition logic is correctly
reimplemented in Python.

**Includes:**
- `_residual()` and `_advance_waypoint()` (§5).
- Simple cascaded PID (height + horizontal) as the demo controller. Lives
  inside the smoketest, **not** inside the env class — env stays
  controller-agnostic.

**Smoke output:** GIF / MP4 of the X2 visiting all 11 waypoints under the
PID controller.

---

## 7. Open Questions Before Starting

These shape the implementation; would prefer answers before Phase B.

| # | Question | Why it matters |
|---|---|---|
| Q1 | Should the env's action be **4-D raw thrust** (MJPC-style) or **3-D position target** (D3IL-style)? | Determines the controller-in-the-loop story. Recommendation in §4.2 is 4-D thrust; happy to flip if FM-PCC trajectory models prefer the position interface. |
| Q2 | Should we keep the **8 racing gates** (`gates.xml`) or strip down to just waypoints? | Gates add visual clutter and collision geoms; nice for racing demos, irrelevant for hover / point-to-point. Recommend keeping geometry but `contype=0` so it's purely visual unless explicitly enabled. |
| Q3 | Should the env hold its own **camera** (D3IL-style bp-cam + inhand-cam) for vision-conditioned training, or no cameras for now? | bp-cam (third-person tracking the drone) is easy. "inhand" for a quadrotor would be a forward-facing FPV cam — possible (Menagerie X2 includes a `track` camera site) but more work. Recommend bp-cam-only Phase B, defer FPV. |
| Q4 | Does FM-PCC's existing **DPCC projector / SLSQP** need any quadrotor-specific constraint plumbing in this migration? | Probably no for Phase A-C (just get the env running). Defer to a later "Gen11 Epoch 2" once we know what trajectories look like. |
| Q5 | Do we want to vendor the Menagerie Skydio X2 source files (mesh + texture) directly into the repo, or hold them as a download step? | Mesh is ~few hundred KB, texture is ~1 MB. Recommend vendoring — keeps the repo self-contained, matches how aligning/pushing mesh assets are already vendored under `models/mj/`. |

---

## 8. What This Migration Will NOT Do

- Train any controller. Phase C ships a hand-tuned PID for sanity, not a
  learned policy.
- Touch any of `fm_visual_aligning/`, `diffuser_visual_aligning/`,
  `config/`, etc. The migration is **strictly** under `d3il/...` until the
  env is verified to work standalone.
- Modify D3IL's existing manipulation envs. The avoiding / aligning /
  pushing pipelines stay bit-for-bit identical.
- Bring across MJPC's predictive sampling planner, residual class
  hierarchy, or task abstraction. Those belong to MJPC's runtime, not to
  FM-PCC.
- Set up any FM-PCC training/eval scripts for the quadrotor. That's
  Epoch 2, after the env is verified.

---

## 9. Estimated Effort

Rough order-of-magnitude (assuming Q1-Q5 answered):

| Phase | Effort | Risk |
|---|---|---|
| A — XML + assets land + load | ~30 min | Low. Asset-path / texture-path mistakes are the only common failure. |
| B — Env class constructs + steps | ~2 h | Medium. `MjQuadrotor` minimal-surface design + bypassing `GymEnvWrapper.step()` is the main novelty. |
| C — PID + waypoint demo | ~1 h | Low once B works. Mostly tuning Kp/Kd numbers. |
| **Total to a hover-and-track demo** | **~3-4 h** | Bounded. |

---

## 10. Recommended Order of Operations (When You Greenlight)

1. Answer Q1-Q5 (or accept the recommendations).
2. Phase A — copy XML, mesh, texture; smoketest with raw MuJoCo.
3. Phase B — wire D3IL env class.
4. Phase C — PID + waypoint demo, produce GIF.
5. Write a changelog under `logs_in_develop/Gen11/Epoch1_UAV_model/CHANGELOG.md`
   summarising what landed and where.

After that, Gen11 Epoch 2 can plan how (or whether) to wire the quadrotor
into FM-PCC's trajectory training / DPCC projection stack — but **that
decision shouldn't be made until we have an env to look at.**

---

## 11. Mission-Begin Preparation (Epoch 1, model-only scope)

This section describes the **exact CLI operations** that will perform
Epoch 1. Scope is narrowed to *model files only* per user direction —
no env class, no D3IL plumbing changes, no Python port of residuals.
The §4–§5 work (env wrapper, residual port) and §6 Phase B/C are
**deferred** out of Epoch 1.

### 11.1 Hard rules for this execution

To prevent LLM "rebuild from memory" hallucinations on the model XML:

1. **No file content will be LLM-generated.** Every byte of XML, mesh,
   and texture comes from a `cp`, `git clone`, or `patch` invocation.
2. **The `Edit` tool is allowed only for small textual adjustments**
   (e.g. asset path prefixes, namespace declarations) on copied files
   — never for synthesising or paraphrasing model content.
3. **All copies use the canonical upstream source.** Menagerie's
   `skydio_x2/x2.xml` is the base, MJPC's `quadrotor.xml.patch` is the
   diff, MJPC's `gates.xml` is verbatim. No memory-reconstructed XML.
4. **The migration is reversible.** A single `rm -rf` of the target
   directory undoes everything in Epoch 1.

### 11.2 Source locations (confirmed by inspection)

- MJPC quadrotor sources (already on disk):
  `/workspaces/mujoco_mpc/mjpc/tasks/quadrotor/{gates.xml, quadrotor.xml.patch}`
- Menagerie X2 (must be cloned — confirmed via
  `mujoco_mpc/CMakeLists.txt:170-179` and `mjpc/tasks/CMakeLists.txt:109-117`):
  - `mujoco_menagerie/skydio_x2/x2.xml` → renamed to `quadrotor.xml`
    by MJPC's build, then patched in place.
  - `mujoco_menagerie/skydio_x2/assets/` → mesh `X2_lowpoly.obj` +
    texture `X2_lowpoly_texture_SpinningProps_1024.png`.

### 11.3 Target layout (Epoch 1, model-only)

```
d3il/environments/d3il/models/mj/robot/quadrotor/
├── quadrotor.xml                       ← cp from menagerie/skydio_x2/x2.xml (verbatim)
├── quadrotor_modified.xml              ← produced by `patch` from quadrotor.xml + .patch
├── gates.xml                           ← cp from mujoco_mpc verbatim
├── assets/
│   ├── X2_lowpoly.obj                  ← cp from menagerie/skydio_x2/assets
│   └── X2_lowpoly_texture_SpinningProps_1024.png
└── LICENSE-skydio_x2.txt               ← cp from menagerie/skydio_x2/LICENSE if present
```

No env package created, no `MjQuadrotor`, no `MjFactory` registration in
Epoch 1.

### 11.4 Execution sequence (CLI-only, in order)

The following commands will be executed in this order. Each is a pure
file operation — no content synthesis.

**Step 0 — verify Menagerie isn't already cached locally.**
```
find / -maxdepth 7 -type d -name 'mujoco_menagerie' 2>/dev/null
```
If a copy exists, use it. Otherwise proceed to Step 1.

**Step 1 — sparse-clone Menagerie's `skydio_x2/` subtree only**
(avoids pulling the full ~1 GB Menagerie repo for one robot):
```
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git \
    /tmp/mujoco_menagerie
cd /tmp/mujoco_menagerie && git sparse-checkout set skydio_x2
```
Expected result: `/tmp/mujoco_menagerie/skydio_x2/` contains `x2.xml`,
`assets/`, `LICENSE`, `README.md`. Total ~few MB.

**Step 2 — make the target directory.**
```
mkdir -p /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/assets
```

**Step 3 — copy the verbatim files.**
```
cp /tmp/mujoco_menagerie/skydio_x2/x2.xml \
   /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor.xml

cp /tmp/mujoco_menagerie/skydio_x2/assets/*.obj \
   /tmp/mujoco_menagerie/skydio_x2/assets/*.png \
   /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/assets/

cp /workspaces/mujoco_mpc/mjpc/tasks/quadrotor/gates.xml \
   /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/

cp /tmp/mujoco_menagerie/skydio_x2/LICENSE \
   /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/LICENSE-skydio_x2.txt
```

**Step 4 — apply MJPC's patch with the `patch` CLI tool** (this is
exactly what MJPC's CMakeLists.txt:115-117 does):
```
cd /workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/
patch -o quadrotor_modified.xml quadrotor.xml \
    < /workspaces/mujoco_mpc/mjpc/tasks/quadrotor/quadrotor.xml.patch
```
Expected result: `quadrotor_modified.xml` exists alongside `quadrotor.xml`,
differing from it only by the changes shown in the `.patch` (quat init +
sensor/keyframe removal).

**Step 5 — smoke load via raw MuJoCo** (no LLM-generated code; one-liner):
```
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path(
    '/workspaces/FM-PCC/d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml')
d = mujoco.MjData(m)
for _ in range(100): mujoco.mj_step(m, d)
print('OK — nq=%d nv=%d nu=%d, body z after 0.1s = %.3f m' % (m.nq, m.nv, m.nu, d.qpos[2]))
"
```
Expected output: something like `nq=7 nv=6 nu=4, body z after 0.1s = 0.052 m`
(drone falls under gravity from 0.1 m start, no thrust → about 5 cm below
the start position).

### 11.5 Allowed `Edit`-tool modifications (small, surgical)

If — and only if — Step 5 fails on an asset-path lookup error, the
following narrow Edit operations are permitted:

| Allowed edit | Why |
|---|---|
| Adjust `<mesh file="...">` and `<texture file="...">` relative paths in `quadrotor.xml` / `quadrotor_modified.xml` if Menagerie's base uses bare `X2_lowpoly.obj` but our layout puts them under `assets/` | Mirrors MJPC's runtime layout where assets live alongside the XML |
| Add a `<compiler meshdir="assets" texturedir="assets"/>` directive if it isn't already there | Standard MuJoCo asset-root mechanism — single line, fully scoped |

**Not permitted:** any other content edit. If something else looks
wrong, stop and investigate; do not rewrite.

### 11.6 Verification checklist (post-execution)

- [ ] `models/mj/robot/quadrotor/` directory exists with the 5 expected files
- [ ] `diff quadrotor.xml /tmp/mujoco_menagerie/skydio_x2/x2.xml` produces
      empty output (we did not LLM-edit the base)
- [ ] `diff` between `quadrotor.xml` and `quadrotor_modified.xml` matches the
      MJPC patch's intent (quat added, MJPC-redundant sensor/keyframe blocks
      removed)
- [ ] Step 5 smoke-load runs with `nq=7 nv=6 nu=4`
- [ ] No other directory under `d3il/` was modified

### 11.7 What does NOT happen in Epoch 1

- No `gym_quadrotor_env/` package created.
- No `MjQuadrotor` wrapper.
- No Python port of the residual / waypoint logic.
- No PID / hover demo.
- No `config/` edit, no `Slurm_Codes/` script.
- No FM-PCC training or evaluation script touches.

Those all belong to a later epoch — they are documented in §4–§6 above
but are explicitly **deferred** to Epoch 2+.

### 11.8 Epoch 1 deliverables

- 5 files placed under `d3il/environments/d3il/models/mj/robot/quadrotor/`
  (`quadrotor.xml`, `quadrotor_modified.xml`, `gates.xml`,
  `assets/X2_lowpoly.obj`, `assets/X2_lowpoly_texture_SpinningProps_1024.png`,
  `LICENSE-skydio_x2.txt`).
- One smoke-load verification line printed.
- One concise changelog at
  `logs_in_develop/Gen11/Epoch1_UAV_model/CHANGELOG.md` listing the
  five files, the source provenance of each, and the smoke-test result.

That is the entirety of Epoch 1. Ready to execute on greenlight.
