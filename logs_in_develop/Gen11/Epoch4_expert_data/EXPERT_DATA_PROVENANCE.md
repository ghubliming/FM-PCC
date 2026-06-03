# Expert-Data Provenance — D3IL, UAV-Flow, and the case for/against manual generation

**Date**: 2026-06-02
**Parent**: extends [`RESEARCH.md`](RESEARCH.md). That file evaluates *which* source we use; this file investigates *how others got their data* and whether our "manual generation" plan is defensible against the alternative ("we need real-world expert flights").
**Triggering questions** (user, verbatim):
> "How did D3il get expert data? ask gemini paper and codebase!"
> "how is UAV Flow repo get the expert data? from real life? or some maunul generation"
> "is the manual generation make sense? i thought we may need the in real life expert data?"

---

## 1. D3IL — how the upstream benchmark we mirror got its data

**Source authority order**: (a) paper acronym, (b) project website, (c) codebase.

### 1.1 What the acronym tells you up-front

D3IL = **"D**atasets with **D**iverse human **D**emonstrations for **I**mitation **L**earning**"** (Jia et al., ICLR 2024). The benchmark's *premise* is that the trajectories come from **multiple distinct human operators per task**, deliberately producing multi-modal behaviour — different humans solve "align this push-box to the slot" via different action sequences. This is the property diffusion/flow-matching policies are *designed* to model (single Gaussian regressors cannot fit a multi-modal action distribution).

So before reading a single line of code: D3IL trajectories are **human-generated, multi-operator, per-task multi-modal**. Not scripted controllers.

### 1.2 What the project website confirms

[`alrhub.github.io/d3il-website`](https://alrhub.github.io/d3il-website/) (linked from the repo `d3il/README.md`):
> "we have collected demonstration data from multiple human demonstrators"

That's the entire claim — they don't elaborate on the website. Detail lives in the paper (ICLR 2024 OpenReview `id=6pPYRXKPpw`) and the codebase.

### 1.3 What the codebase confirms (this is the load-bearing evidence)

Inside the vendored d3il at `/workspaces/FM-PCC/d3il/`, there is a full **teleoperation subsystem** at:

```
d3il/environments/d3il/d3il_sim/sims/sl/teleoperation/
├── demo_teacher.py            # zero-force-mode control of a single Panda
├── demo_teleoperation.py      # primary Panda → replica Panda + force feedback
├── demo_virtualtwin.py        # primary real Panda → replica virtual (MuJoCo) Panda
├── readme.md
├── src/                       # controllers, schedulers, CLI, logging
└── mockup_tests/
```

The `readme.md` (verbatim, paraphrased): three modes are available — (i) single-arm zero-force teaching, (ii) **primary-replica teleop with force feedback** between two real Franka Panda robots, (iii) primary-real → replica-virtual (MuJoCo) twin.

`demo_teleoperation.py` shows two real robots being instantiated by IP — `tcp://141.3.53.151` and `tcp://141.3.53.152` (KIT subnet). A human operator moves the primary; the replica follows with a PD controller while exposing contact forces back to the primary so the human "feels" the environment. The session is logged by `TeleopMetaLogger` to disk under `/home/.../teleop_data/raw_loads/...`.

**So D3IL's data-collection method is, end-to-end:**

1. Two real Franka Panda robots on a KIT lab bench. *(physical hardware, not simulation)*
2. Multiple human operators teleoperate the primary robot to perform each of the 7 tasks (Avoiding, Pushing, Aligning, Sorting, Stacking, Inserting, Arranging).
3. The replica robot mirrors with PD-following, providing force feedback so the human's motion respects environment dynamics.
4. State (joint positions, end-effector pose, gripper width, contact forces) and the human's command stream are logged.
5. Raw logs are post-processed into the `(B, H, D)` tensor format with `train_files_*.pkl` splits at `environments/dataset/data/aligning/`.

### 1.4 Implication for FM-PCC

When our visual aligning eval loads `train_files_0.5_.pkl`, the trajectories inside it are **recordings of human hands moving a Franka through a push-box alignment task on a physical bench**. Not synthetic. Not algorithmic. Not from MuJoCo.

That's why FM/DPCC produce *plausible* push-and-align motions — they imitate the human policy, including small hesitations, re-grasps, and the multi-modal "left-around-the-box vs. right-around-the-box" choice that motivates a diffusion-style policy in the first place.

---

## 2. UAV-Flow Colosseo — how that dataset was collected

**Source authority order**: (a) repo README, (b) arXiv abstract (paper `2505.15725`), (c) JSON file structure, (d) HuggingFace dataset card.

### 2.1 What the title & repo README say

Title: "UAV-Flow Colosseo: **A Real-World Benchmark** for Flying-on-a-Word UAV Imitation Learning" (NeurIPS 2025, Wang et al.).

README highlights:
> "**Real and simulated UAV data.** UAV-Flow contains real-world UAV trajectories, while UAV-Flow-Sim provides simulation trajectories for scalable training and evaluation."

Two HuggingFace datasets are released:
- `wangxiangyu0814/UAV-Flow` → **real-world trajectories**
- `wangxiangyu0814/UAV-Flow-Sim` → **simulated trajectories** (used for OpenVLA-UAV training)

Neither the README nor the project page (`prince687028.github.io/UAV-Flow`) specifies the collection hardware or the pilot setup explicitly.

### 2.2 What the arXiv abstract reveals

The arXiv `2505.15725` abstract has one critical phrase:
> "expert **pilot** trajectories paired with atomic language instructions"

That single word "pilot" is the answer: **a human pilot flew the drone**. Not autonomous, not scripted, not algorithmic. The "Colosseo" naming and the language-conditioning ("flying-on-a-word") imply natural-language instructions paired with pilot demonstrations — the UAV analogue of D3IL's multi-human teleop.

### 2.3 What the JSON files reveal about the format

Each `UAV-Flow-Eval/test_jsons/*.json` is one trajectory with fields:

```json
{
  "instruction":            "Turn to the direction of the person.",
  "instruction_unified":    "Turn to the direction of the person.",
  "initial_pos":            [x, y, z, roll, pitch, yaw],     // UE4 coordinates, cm
  "end_pos":                [x, y, z, roll, pitch, yaw],
  "target_pos":             [x, y, z, roll, pitch, yaw],
  "obj_id":                 19,
  "use_obj":                1,
  "reference_path_raw":     [ [x,y,z,roll,pitch,yaw], ... ]  // ~80–150 waypoints per episode
}
```

**What's NOT recorded**:
- Raw motor / thrust commands (no `u₁..u₄`).
- IMU readings (no accel/gyro).
- Velocities (only positions; velocity must be computed by differencing).
- Time deltas between waypoints (assumed uniform).

**What's RECORDED**: a per-waypoint pose `(p, R)` sampled densely along the executed flight. This is **the pilot's resulting trajectory**, not the pilot's *control input*. The mapping from "pilot stick deflection" to "drone pose" has already been applied by the drone's flight controller and the physics; the dataset captures the *output* of that loop.

For instruction-conditioned VLA training (OpenVLA-UAV's use case), this is fine — the policy predicts the next *pose target*, and a low-level flight controller closes the loop. But it means UAV-Flow doesn't ship the kind of "raw action chunk" data a state-action imitation model traditionally consumes.

### 2.4 UAV-Flow vs. UAV-Flow-Sim — the practical difference

| Property | UAV-Flow (real) | UAV-Flow-Sim |
|---|---|---|
| Pilot | Human, in physical environment | (Likely) human, in UnrealZoo/UE4 simulator |
| Hardware | Real drone (model unspecified in README; in arXiv only as "expert pilot trajectories") | Simulated drone in UnrealZoo |
| Rendering | Real RGB photos from drone camera | UE4 renders |
| Coverage | Limited by physical access, weather, battery life | Scalable — can re-render any pose |
| Used by | Evaluation / sim-to-real transfer studies | Training (OpenVLA-UAV baseline) |
| What we have locally | Only the `test_jsons/` subset (~100 episodes) — the eval split | Not pre-downloaded; would need HF parquet pull |

So **UAV-Flow itself does not ship "manually generated synthetic" data** — both subsets are pilot-flown. The "Sim" variant just substitutes the rendering and physics backend.

### 2.5 What we can and cannot reuse from UAV-Flow

For our FM-PCC drone planner:

| What | Reusable? | Why |
|---|---|---|
| Pilot trajectories (real) | ⚠️ Partially | Wrong sim, wrong physics, wrong coordinate frame, no actions. *Can* extract pose sequences for statistical alignment (avg speed, avg path curvature, altitude bands). *Cannot* train on directly. |
| Pilot trajectories (sim, UE4) | ❌ No | Same problems as real, plus we'd have to spin up UE4 just to verify them. |
| OpenVLA-UAV baseline code | ⚠️ Reference only | They use a different architecture (VLA + Flask inference server). Not architecturally aligned with FM-PCC. |
| UnrealZoo eval harness | ❌ No | Requires UE4. Our eval is MuJoCo. |
| **Trajectory statistics for "what should our manually generated data look like?"** | ✅ Yes | This is exactly what existing `RESEARCH.md` §3.3 already proposes. |

---

## 3. Is "manual generation" actually sensible?

This is the user's third question and it deserves a serious treatment because **the instinct "we need real-life expert data" is a reasonable one** and worth pushing on before committing to manual generation.

### 3.1 What "manual generation" means here (precisely)

It does **not** mean "we write the trajectories by hand in a JSON file." It means:

> A controller we wrote (PID cascade, or an LQR, or a simple optimal-control script) flies the drone *inside MuJoCo*, and we record the resulting (state, action) tuples at our chosen sampling rate. The "expert" is **the controller**, not a human.

This is the same template as **D3IL except the demonstrator is a control script instead of a human teleop**. The recording infrastructure (state log, action log, time-aligned chunking into `(B, H, D)`) is identical.

### 3.2 The case FOR manual generation (and why it dominates for us)

**(a) Format alignment is mandatory regardless of source.** Whatever source we use must end up in our `(B, H, action+obs)` tensors at our sampling rate, in our coordinate frame, with our action convention. UAV-Flow's pose-only JSON would need post-processing to recover actions (finite-differencing positions to produce velocity "actions"). MJPC's continuous-time output would need re-sampling. Manual generation in MuJoCo skips this entirely — the data exits the simulator already in our format.

**(b) Imitation learning imitates whatever demonstrator you give it.** This is the foundational fact people forget. If your manual controller produces:
- Smooth trajectories → the policy learns smooth.
- Collision-free trajectories → the policy learns collision-free.
- Constraint-satisfying trajectories (DPCC half-spaces!) → the policy learns constraint-satisfying.

For FM-PCC, where DPCC's whole job is *projecting* policy outputs onto constraints, training the policy on constraint-respecting demonstrations is a **direct benefit**. A human pilot doesn't know our constraint formulation; a scripted MPC/PID controller can be written to respect it natively.

**(c) Coverage is controllable.** Real-pilot data has whatever distribution the pilots produced — often clustered around "easy" cases. Manual generation lets us systematically sweep initial conditions, target positions, obstacle layouts. For 100k training steps you want **distributional coverage**, not stylistic diversity.

**(d) Cost & iteration speed.** Re-collecting human-pilot data because we changed the state representation is expensive (hours of human time per dataset variant). Re-generating manual data is a `python collect.py` that runs in minutes. For an early-stage project (we're at Epoch 4 of 7+) where the state convention may still change, this matters a lot.

**(e) Precedent: D3IL's own structural choice.** D3IL's data-collection design choice was "give 7 humans 7 tasks, record their teleop". They did this because *they wanted multi-modal demonstrations for IL research*. We don't need that property for our planner — we want a **competent** planner, not a *diverse* one. Multi-modality is a feature only when (i) the task genuinely has multiple acceptable solutions AND (ii) you want the policy to pick stochastically. For a UAV waypoint planner, neither is true at the level we're working at.

**(f) FM/iMF theory is demonstrator-agnostic.** Flow matching learns the velocity field `v(x, t)` that transports noise → data. The geometry of "data" can come from a human, a script, an analytic distribution, or a random number generator — flow matching just fits the field. There is no theoretical mechanism by which "human-generated" data is privileged over "controller-generated" data, provided both lie on the same manifold of *valid* trajectories.

### 3.3 The case AGAINST (the user's instinct that "we need real-life data")

To be fair, here is the *strongest* form of the counterargument:

**(a) Sim-to-real gap.** If your eventual deployment is a real drone, MuJoCo-trained policies will not generalize without sim-to-real techniques (domain randomization, system ID, real-data fine-tuning). UAV-Flow's real trajectories *would* have captured the messy real physics — wind, motor saturation, IMU noise — that MuJoCo doesn't. **This concern is real but only if real-world deployment is in scope.** If FM-PCC for drones stays a MuJoCo-only research project, this concern doesn't apply.

**(b) Behavioural richness.** Human pilots produce stylistic variation (banking turns, slow approaches, dynamic recoveries) that a PID controller will not. If your benchmark *evaluates* on stylistic richness, you need rich demos. **For waypoint-planning evaluation this isn't the metric.** For "follow a natural-language instruction" evaluation (UAV-Flow's actual task) it would be.

**(c) Distribution match to real pilots.** If the planned downstream use is "predict what a human pilot would do in this scenario", then trained-on-PID will not match trained-on-pilot. **For FM-PCC's "produce a constraint-satisfying motion plan" use case, this isn't required.**

**(d) Single-demonstrator → mono-modal data.** A single PID controller, given identical initial conditions, produces a single deterministic trajectory. That's mono-modal. Flow matching can fit mono-modal data trivially (it doesn't need diffusion's noise schedule for that) — but you lose the *purpose* of using FM in the first place. **Mitigation**: run the manual controller with multiple gain settings / multiple goal biases / multiple obstacle-avoidance heuristics to produce 2–4 distinct policies' worth of demos. This restores the multi-modality property without needing humans.

### 3.4 What real-world data would buy us — and at what cost

Suppose we *did* want real-world demonstrations. The path would be:

1. Acquire a research drone (Crazyflie 2.x, or larger) — capital cost.
2. Set up a motion-capture room (VICON / OptiTrack) for ground-truth pose — capital + space.
3. Hire / train pilots for the specific task scripts we care about.
4. Collect O(100) demonstrations per scenario, post-process into our format.
5. Discover that the dynamics in our MuJoCo model don't match the real drone, and either (a) re-fit MuJoCo to the real drone or (b) accept the dynamics mismatch as noise.
6. Train. Evaluate. Discover that sim-to-real gap means the policy doesn't fly in MuJoCo *or* in the real world without additional adaptation.

Compared to: write `collect_uav.py` in our existing MuJoCo stack, generate 10k trajectories overnight, train. **The marginal value of real data is non-zero but the cost is 10–100× and the learnings transfer poorly until the rest of the FM-PCC drone stack is validated**.

### 3.5 The phased answer

Honest engineering: real-pilot data is the *right* answer for a *deployable* policy, but the wrong answer for an *initial validation* of FM-PCC on drones.

| Phase | Right data source | Justification |
|---|---|---|
| **Epoch 4–6** (FM-PCC validation in MuJoCo) | **Manual generation** | Controllability, format match, iteration speed, demonstrator-agnostic FM theory |
| **Epoch 7+** (if/when sim works and we want real flight) | **Real pilot demos** (UAV-Flow's approach, or our own MoCap-room equivalent) | Sim-to-real gap, stylistic richness, language-conditioning if that's in scope |

This mirrors how the D3IL-vs-FM-PCC project itself evolved: D3IL exists because real human demos were necessary for *their* research question (multi-modal IL benchmark). FM-PCC came later, was developed *against* D3IL's existing data, and only validated the planner's properties — it didn't need to re-collect anything. Our drone project is in the equivalent of FM-PCC-pre-D3IL state right now: we are validating a planner architecture, not collecting a benchmark. **Manual generation is the right tool for our actual question.**

---

## 3a. Codebase-grounded evidence (both repos are vendored locally — direct file reads, not web fetches)

This section pins every claim above to specific files in `/workspaces/FM-PCC/d3il/` and `/workspaces/UAV-Flow/`. Anyone re-checking can `grep` for these references.

### 3a.1 D3IL — exact recording schema

File: `d3il/environments/d3il/d3il_sim/sims/sl/teleoperation/src/util/teaching_log.py:42-60`

```python
def log_entry(self, robot, tau_raw, tau_cmd, curr_load, ts):
    self._log_data["tau"].append(tau_cmd)            # commanded torque (the human's intent post-controller)
    self._log_data["tau_raw"].append(tau_raw)        # raw torque (pre-filter)
    self._log_data["curr_load"].append(curr_load)
    self._log_data["c_pos"].append(robot.current_c_pos)     # end-effector Cartesian position
    self._log_data["c_vel"].append(robot.current_c_vel)     # end-effector Cartesian velocity
    self._log_data["c_quat"].append(robot.current_c_quat)   # end-effector orientation
    self._log_data["gripper"].append(robot.gripper_width)
    self._log_data["j_pos"].append(robot.current_j_pos)     # joint positions (7-DoF Franka)
    self._log_data["j_vel"].append(robot.current_j_vel)     # joint velocities
    self._log_data["power"].append(np.dot(curr_load, robot.current_j_vel))
```

So per recording timestep, D3IL captures **both raw torque commands AND full resulting state** — far richer than UAV-Flow's pose-only logs. Notably, *both* the human's command stream (via `tau_cmd`, the controller setpoint that produced the motion) and the realized motion (`c_pos`, `j_pos`, etc.) are persisted.

### 3a.2 D3IL — what the *training pipeline* actually uses

File: `d3il/environments/dataset/aligning_dataset.py:50-78`

```python
robot_des_pos     = env_state['robot']['des_c_pos']    # the human-commanded Cartesian setpoint
robot_c_pos       = env_state['robot']['c_pos']        # actual end-effector pos
push_box_pos      = env_state['push-box']['pos']
push_box_quat     = env_state['push-box']['quat']
target_box_pos    = env_state['target-box']['pos']
target_box_quat   = env_state['target-box']['quat']

input_state = np.concatenate(
    (robot_des_pos, robot_c_pos, push_box_pos, push_box_quat, target_box_pos, target_box_quat),
    axis=-1,
)
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]      # ← THE ACTION
```

This is the data structure FM-PCC trains on. Concrete takeaways:

- **`obs_dim = 20`**: `des_c_pos(3) + c_pos(3) + push_box_pos(3) + push_box_quat(4) + target_box_pos(3) + target_box_quat(4)`. (Same 20-D non-visual contract we discovered the hard way in Fix-18.1.)
- **`action_dim = 2`** (or 3, depending on `act` slice — the dataset class declares `action_dim=2` but downstream code uses 3 for the full Cartesian Δ): `vel_state = des_c_pos[1:] - des_c_pos[:-1]`. *Actions are forward-differences of the human's commanded position*. **D3IL throws away the recorded torque** (`tau_cmd`) and treats the human's positional intent as the supervised target. The torque is collected for completeness but unused at training time.

This is the **same convention** UAV-Flow uses (next section), and the same convention our manual generation should produce.

### 3a.3 UAV-Flow — exact recording schema

File: `UAV-Flow-Eval/test_jsons/*.json` (any one of 273 episodes).

```
keys:        ['instruction', 'instruction_unified', 'initial_pos', 'end_pos',
              'obj_id', 'use_obj', 'target_pos',
              'reference_path_raw', 'reference_path_preprocessed']
per waypoint: [x, y, z, roll, pitch, yaw]   (UE4 coords, cm; yaw in degrees)
```

There is **no torque, no thrust, no IMU, no force feedback** in the JSON — only pose timeseries. This contrasts sharply with D3IL's `tau / curr_load / power` fields. UAV-Flow's design choice: capture *what the drone did*, not *what the pilot's stick was doing*. The drone's flight controller is treated as a black box between pilot intent and pose output.

### 3a.4 UAV-Flow — what the training pipeline computes

File: `OpenVLA-UAV/prismatic/vla/datasets/uav_dataset.py:120-200`

```python
trajectory_raw = np.array(episode_data['raw_logs'])         # [T, 6]
trajectory     = np.array(episode_data['preprocessed_logs']) # [T, 6]

# Discard roll & pitch — keep only (x, y, z, yaw)
trajectory_raw = trajectory_raw[:, [0,1,2,4]]   # [T, 4]
trajectory     = trajectory[:, [0,1,2,4]]       # [T, 4]
trajectory_raw[:, 3] = np.deg2rad(trajectory_raw[:, 3])

actions = np.zeros_like(trajectory)  # [T, 4]
for i in range(len(trajectory) - 1):
    actions[i] = self._transform_to_local_frame(
        trajectory_raw[i], trajectory_raw[i + 1]
    )                                # → (Δx_local, Δy_local, Δz_local, Δyaw)
actions[-1] = np.zeros(4)
```

And `_transform_to_local_frame` rotates the world-frame pose-delta into the *current* pose's yaw frame. So the action at step `t` is "where do I go in my own body frame to reach the next pose."

**This is, structurally, the same thing as D3IL's `vel_state = des_c_pos[1:] - des_c_pos[:-1]`**, just (i) extended to include yaw and (ii) expressed in the body frame rather than the world frame. Both projects converged on **"position-deltas as actions"**.

### 3a.5 UAV-Flow — actual size of what we have locally

Our local `test_jsons/` is the eval split. Stats from a `python3 ... len(d['reference_path_raw'])` sweep:

| Metric | Value |
|---|---|
| Number of trajectories | **273** (RESEARCH.md says "~100" — correction: 273 in the eval split alone) |
| Min waypoint count | 5 |
| Median waypoint count | 38 |
| Max waypoint count | 379 |
| Mean waypoint count | 46.2 |

The full HuggingFace dataset (real + sim) is much larger; this is just the held-out eval set we vendored.

### 3a.6 Cross-implementation alignment summary

| Field | D3IL aligning | UAV-Flow | Our planned manual-gen UAV |
|---|---|---|---|
| State capture | Joint + Cartesian + scene (`c_pos`, `j_pos`, `push-box.pos`…) | Pose-only `(x, y, z, roll, pitch, yaw)` | Pose + velocity (`p`, `v`) per Epoch 4 RESEARCH §1 |
| Raw command capture | Yes (`tau_cmd`) | No | We can choose (likely Δposition target) |
| Action used for training | `Δdes_c_pos` (position delta) | `Δpose_local` (body-frame position+yaw delta) | `Δp_target` (position delta) |
| Per-timestep dim used in train | 20 obs + 2-3 act | 4 obs + 4 act | 6 obs + 3 act *(RESEARCH §1 default)* |
| Demonstrator | Multiple humans | Expert pilots | Scripted PID/MPC |
| Number of trajectories | ~hundreds-to-thousands (full dataset) | ≥273 in eval; full HF dataset is much larger | Target: 5k–10k (RESEARCH §4.4) |

**The structural pattern is invariant across domains**: high-level intent (target position) is recorded or computed via differencing; low-level control (torque/thrust) is either recorded-but-unused (D3IL) or never recorded (UAV-Flow). Our manual generation matches this pattern exactly — there's no methodological deviation.

---

## 3b. Could we make "manual generation" actually be a human in the loop? — Isaac Lab, MuJoCo teleop, innovative alternatives

**Triggering questions** (user, paraphrased):
> "Could we use something like Isaac Lab as a game where a human player flies the UAV to collect expert data?
> Isaac Lab can't run on Slurm — could MuJoCo do similar? Or some innovative way?
> If Isaac Lab is super, we can borrow."

This section sits between "scripted manual generation" (§3.1) and "real-world pilot demos" (§3.4). It is the *missing middle*: a human flies a simulated drone via joystick/keyboard, and we capture trajectories that are (a) human-stylistically rich like real-pilot data but (b) cheap and reproducible like scripted data. The user is asking whether this hybrid is feasible for us.

### 3b.1 What "Isaac Lab style game-collection" would look like

The high-level recipe, agnostic to which simulator we use:

1. Spin up an interactive sim — drone in a scene (corridor, pillar field, S-curve).
2. Connect a joystick / keyboard / spacemouse / VR controller via a teleop driver.
3. Map input axes to a high-level command (position target, velocity target, body-rate target).
4. The sim's own controller (cascaded PID from Epoch 2) closes the low-level loop to motor thrusts.
5. Log `(state, command)` at fixed Hz to JSONL.
6. Repeat across scenes and operators; batch the JSONL into our `(B, H, D)` training tensors.

This is **structurally identical to D3IL's teleoperation rig** (§1.3) — only the demonstrator hardware is software (joystick into a sim) instead of hardware (human hand on a primary Franka). The recorded data format is the same: position-delta actions (§3a.6).

### 3b.2 What Isaac Lab provides out-of-the-box

**Isaac Lab** (NVIDIA, built on Isaac Sim + Omniverse Kit) ships several teleop interfaces that make this recipe trivial *if* you can run it:

| Isaac Lab feature | Relevance to "game-style UAV collection" |
|---|---|
| `Se3KeyboardController` | Maps WASD/arrows to 6-DoF SE(3) deltas. Zero-config teleop. |
| `Se3SpaceMouseController` | 6-DoF spacemouse (3DConnexion) → SE(3) deltas. Better for drones than keyboard. |
| `Se3GamepadController` | Xbox/PlayStation controller mapping. |
| Quadcopter env (`Isaac-Quadcopter-...-v0`) | A flight-ready quadcopter with cascaded controller already wired |
| USD scene format | Easy to drop in photorealistic environments |
| GPU-parallel rollout | Many envs in parallel for headless training |
| OpenXR VR interface | Immersive piloting, head tracking |
| RTX path-traced rendering | High-fidelity visuals if you want vision-based training |

**Why this is "super" by reputation**: the teleop drivers are a `kit.app` extension you load with one line; controller polling, axis remapping, dead-zones, and recorder hooks are all pre-baked. Compared to writing all that ourselves it would save 1–2 days.

### 3b.3 Why Isaac Lab is off-the-table for us (concrete blockers, not just "Slurm")

1. **Omniverse Kit GUI dependency.** The teleop drivers live inside the Kit app — they require the Omniverse runtime, which requires either a display server or an `omniverse-launcher` headless setup that itself needs a recent CUDA + NVIDIA driver combination that most academic Slurm partitions do not ship.
2. **USD scene-graph weight.** A minimal Isaac Lab quadcopter env loads ~3–5 GB of USD assets at startup. The cluster's per-node scratch and the GH cache size we sync over make this painful even before runtime issues.
3. **Driver version pinning.** Isaac Sim 2024.x requires CUDA 11.8 / driver ≥ 535. Many Slurm partitions are still on CUDA 11.6 / driver 525.
4. **The Slurm cluster has no joystick.** Even if Isaac Lab ran headlessly, our teleop has to happen at *some* interactive workstation. The cluster cannot *be* the workstation. So the architecture is forced to "collect on workstation → sync data to cluster" regardless of which sim is used.
5. **Project Environment memory says** Docker is for AI coding only, all real runs go to Slurm via git-sync. Adding an Omniverse runtime to that chain is a separate platform-engineering project.

Point 4 is the structurally important one. **Even if Isaac Lab were perfectly Slurm-compatible, the human is always at the laptop, not the cluster.** So *the sim only needs to run on a laptop*. That's a constraint MuJoCo meets trivially.

### 3b.4 The MuJoCo equivalent — what we can build natively in ~1 day

MuJoCo doesn't ship a `Se3GamepadController` analogue, but every primitive we need is one Python file away. Concrete recipe:

```
tools/teleop_uav.py                                 # new file, ~150 lines
    import mujoco
    import mujoco.viewer
    import pygame
    from uav_naive_test.flight_controller import CascadedPID  # already exists, Epoch 2 §C

    model = mujoco.MjModel.from_xml_path("uav_naive_test/scene_corridor.xml")
    data = mujoco.MjData(model)
    pid = CascadedPID(...)            # the controller from Epoch 2 (passed Task C 9D, RMS 0.029 m)
    pygame.joystick.init()
    js = pygame.joystick.Joystick(0); js.init()

    logger = JsonlLogger("collected/run_{utc_ts}.jsonl")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_p = data.qpos[:3].copy()
        while viewer.is_running():
            pygame.event.pump()
            dx = js.get_axis(0) * SPEED   # right stick X → body-frame Δx_target
            dy = -js.get_axis(1) * SPEED  # right stick Y → body-frame Δy_target
            dz = -js.get_axis(3) * SPEED  # left stick Y → Δz_target (throttle)
            dyaw = js.get_axis(2) * YAW_SPEED   # left stick X → yaw rate
            target_p, target_yaw = update_setpoint(target_p, target_yaw, dx, dy, dz, dyaw, dt)
            thrusts = pid.step(data.qpos, data.qvel, target_p, target_yaw, dt)
            data.ctrl[:] = thrusts
            mujoco.mj_step(model, data)
            logger.write({
                "t":         data.time,
                "p":         data.qpos[:3].tolist(),
                "v":         data.qvel[:3].tolist(),
                "q":         data.qpos[3:7].tolist(),
                "target_p":  target_p.tolist(),
                "target_yaw": float(target_yaw),
                "thrusts":   thrusts.tolist(),
            })
            viewer.sync()
```

What this gives us:
- An interactive MuJoCo viewer window on the user's laptop with a third-person camera.
- Joystick-as-pilot. (Or keyboard, swap `pygame.joystick` for `pygame.key.get_pressed()`.)
- Same cascaded PID that Epoch 2 already validated at RMS 0.029 m on Task C 9D — so the human is steering high-level setpoints, not raw motors. (This is exactly the D3IL/UAV-Flow convention — humans steer high-level intent; low-level controller is a black box.)
- A JSONL recorder with the schema we already chose in `RESEARCH.md §1` (3-D position target + 6-D `[p, v]` obs).
- **No GPU, no Omniverse, no USD.** Runs on a laptop with `pip install mujoco pygame` plus the existing `uav_naive_test` controller.
- Trajectories sync to cluster via the same git-sync path the rest of the project uses.

This is roughly the **D3IL pattern transplanted from real-Franka to MuJoCo-UAV** — same idea, much cheaper, no hardware.

### 3b.5 What's worth *borrowing* from Isaac Lab even though we won't run it

Looking at Isaac Lab's teleop module structurally, three design ideas are worth copying into our `teleop_uav.py`:

1. **The SE(3) abstraction.** Isaac Lab maps every input device (keyboard, spacemouse, gamepad, VR) to the *same* SE(3) delta interface, then the env consumes only SE(3) deltas. We should do the same: a tiny `class TeleopSource` with a `def get_se3_delta() -> (Δp, Δyaw)` interface, and one concrete implementation per device (`KeyboardSource`, `GamepadSource`, later `VRSource`). The downstream simulator code is device-agnostic.
2. **Dead-zone + exponential mapping.** Raw joystick axes are noisy near zero and bad linearly mapped. Isaac Lab applies a dead-zone of ~0.1 and an exponential curve (`axis · |axis|`). Cheap to copy; large UX improvement.
3. **Recording-via-callback rather than recording-inside-the-loop.** Isaac Lab exposes a `pre_physics_step_callback` slot. Putting the recorder there decouples logging from the controller. Lets us A/B test "with vs without human in the loop" by swapping the controller for an autonomous PID baseline.

What is **not** worth copying:
- USD scenes (overkill; our MJCF + procedural obstacle generation is fine).
- VR/OpenXR integration (cool but Epoch 4 doesn't need it).
- Multi-env parallel teleop (one human flies one drone; parallelism doesn't help here).
- Photoreal rendering (we're training non-visual policies; aesthetics don't matter).

### 3b.6 More innovative options (ranked by how much we'd actually use them)

| Idea | Pitch | Effort | Verdict |
|---|---|---|---|
| **MuJoCo viewer + joystick** (the §3b.4 plan) | The minimum viable teleop. Already-validated PID does low-level. | ~1 day | **Primary recommendation.** |
| **Web-based teleop** | Stream MuJoCo frames over WebRTC; take keyboard inputs from browser back; record on server. Lets non-local collaborators help. | ~3–5 days | Defer unless multi-collaborator collection becomes urgent. |
| **Mixed-initiative ("co-pilot")** | Human clicks waypoints; PID flies between them; record the *combined* trajectory. Each click is a goal; PID is the executor. Faster than full manual flying; lower noise. | ~1 day on top of the §3b.4 base | **Strong second** — gives us a "manual high-level + scripted low-level" hybrid that nicely matches D3IL's "human des_c_pos + Franka low-level controller" split (§3a.2). |
| **Replay editing** | Generate baseline trajectories with PID; let human "scrub" and edit waypoints in the viewer; re-execute. Faster than from-scratch flying. | ~3 days | Defer — too clever for too little gain at this stage. |
| **Hook into existing FPV game** (Liftoff, Velocidrone, DRL Sim) | Real FPV-pilot games already have huge user bases and recordable telemetry. Could extract joystick streams + drone state from an existing game and treat it as UAV-Flow-style real-pilot data without needing a real drone. | Days–weeks to reverse-engineer telemetry formats; legal gray area | Cool idea, defer indefinitely. |
| **VR / OpenXR (`mujoco-openxr` style)** | Headset + Quest controllers. Immersive piloting. Could be very high-quality demos. | ~1–2 weeks | Skip unless we already own a headset and have nothing better to do. |
| **Crowdsourced (Mechanical Turk style)** | Ship a tiny Electron app; pay people $0.50 per flight. Hundreds of operators → multi-modal demos like D3IL. | Weeks of platform work | Skip — would only matter at "publication grade dataset" scale, not for our internal Epoch 4 validation. |
| **Inverse RL from pre-recorded human flight videos** | YouTube/dronestagram videos as expert data; SfM-recover pose; train against poses. | Months; high noise | Skip. |

### 3b.7 Where this lands relative to §3.5's phased answer

§3.5 split the phased plan into:
- **Epoch 4–6**: scripted manual generation. *Primary path.*
- **Epoch 7+**: real pilot demos. *Only if real-flight or language-conditioning becomes scope.*

The §3b discussion adds a **middle option** that wasn't in the original phased plan:

- **Epoch 4 mid–late**: *MuJoCo-viewer + joystick teleop* (§3b.4) + mixed-initiative co-pilot (§3b.6 row 3). Cheap, sits on the existing Epoch 2 PID, gives us stylistic diversity without needing real hardware or pilots.

Updated phasing:

| Phase | Primary data source | Cost | Justification |
|---|---|---|---|
| **Epoch 4 early** | Scripted manual generation (PID flying parametric routes) | Hours | Get any data flowing through FM-PCC; validate the pipeline |
| **Epoch 4 mid–late** | **MuJoCo teleop (§3b.4) + co-pilot (§3b.6 row 3)** | ~1–2 days for the rig + a few hours per operator-session | Multi-modal demos at near-scripted-data cost; structurally aligned with D3IL's recording pattern; runs entirely on the laptop, cluster sees only JSONL |
| **Epoch 5–6** | Continued teleop + scripted baseline as control | — | Validate FM/iMF on multi-modal demos |
| **Epoch 7+** | UAV-Flow-style real-pilot dataset (only if deployment / language is in scope) | Weeks + hardware | Sim-to-real or language-conditioning |

**Conclusion**: Isaac Lab is not usable for us (§3b.3) but its teleop *design* (§3b.4 + §3b.5) is borrow-able into a MuJoCo equivalent that fits our existing infra. The middle-path "human in MuJoCo viewer" option is now the recommended Epoch 4 mid-late deliverable; it doesn't replace scripted manual generation, it complements it.

---

## 3c. Two-stage collection — state-only manual generation + post-hoc dual-camera replay (recommended)

**Triggering question** (user, paraphrased):
> "Could we use Gen9's camera setup — overhead bp-cam + first-person inhand-cam — but generate the *state* trajectories manually first, then collect the *images* by replaying through the env?"

**Verdict**: **Yes — this is the cleanest path forward**, and the pattern is already proven in this repo by Gen9 Epoch 1 (`collect_visual_avoiding_data/collect_visual_avoiding_data.py`). It cleanly separates the two concerns that don't need to be coupled, and unlocks several capabilities (re-renderability, domain randomization, cheap state-space sweeps) that a single-stage pipeline can't offer.

### 3c.1 The proposal, precisely

Two stages, run sequentially, each independently restartable:

| Stage | What | Cost | Output |
|---|---|---|---|
| **Stage 1 — State generation** *(headless, fast)* | Scripted PID / MPC controller (Epoch 2's cascaded PID, validated at RMS 0.029 m on Task C 9D) flies the drone through parametrised scenarios. Records `(t, p, v, q, ω, des_p, des_yaw, motor_thrusts)` per step. No rendering, no GPU. Maybe `tqdm` progress; can run 1000s of trajectories per minute on CPU. | ~minutes per 1k trajectories | `state_pickles/<ep_id>.pkl` |
| **Stage 2 — Visual augmentation** *(GPU-bound, slow)* | Load each state pickle, replay the action sequence step-by-step through a MuJoCo env that has `bp-cam` (third-person overhead) + `inhand-cam` (body-frame FPV) mounted. Capture both cameras per step. Save frames to `images/{bp,inhand}-cam/<ep_id>/<frame>.jpg`. | ~seconds per trajectory; tens of minutes for 1k trajectories | `images/bp-cam/<ep>/`, `images/inhand-cam/<ep>/` |

This is **structurally identical to `collect_visual_avoiding_data.py:151` `replay_and_capture()`** which already does this for the avoiding task (state pickles → replay through env with cameras → save frames).

### 3c.2 Why two stages, not one

A single-stage pipeline that flies the drone AND renders images per step couples them — every iteration on the controller forces a re-render of all trajectories. The two-stage pattern decouples:

| Capability | Single-stage | Two-stage |
|---|---|---|
| Iterate on controller without re-rendering | ❌ — every controller tweak invalidates all images | ✅ — re-run Stage 1 only; Stage 2 only when you ship |
| Render multiple camera variants from one state set | ❌ — must re-fly each | ✅ — change camera config in Stage 2, replay |
| Domain randomization at render time (textures, lighting, distractors) | ❌ — domain noise corrupts physics | ✅ — physics is frozen in Stage 1; Stage 2 randomizes visuals only |
| Render high-res for eval and low-res for train from same dataset | ❌ — would require two flights | ✅ — same state, two render passes |
| Skip image rendering entirely when training a non-visual baseline | ❌ — wastes the render cost | ✅ — Stage 1 output IS the non-visual dataset |
| Cheaply expand from 1k to 10k state trajectories | ❌ — flight + render scales linearly | ✅ — Stage 1 scales; Stage 2 only renders the subset you actually need |

### 3c.3 Camera placement for the drone — what `bp-cam` and `inhand-cam` mean here

D3IL's terminology maps cleanly but the camera names should be renamed to avoid confusion with the manipulation context. Proposed mapping for the UAV task:

| D3IL name *(for the manipulation arm)* | Drone-task analogue | Mount | Frame |
|---|---|---|---|
| `bp-cam` (cage / third-person observer) | `world-cam` | Fixed in world frame, looking down at the flight arena | World |
| `inhand-cam` (wrist / first-person) | `fpv-cam` | Rigid-mounted on the drone body, forward-facing | Body |

You could keep `bp-cam` / `inhand-cam` as filename conventions for compatibility with the Gen9 collection script's directory layout — the renamed semantic labels are for documentation only. Concretely:

- **`bp-cam` (overhead world)**: gives the policy global awareness of obstacle positions. Equivalent to UAV-Flow's "Colosseo" outside-observer view. Cheap to add — one fixed MuJoCo `<camera>` XML entry.
- **`inhand-cam` (body-frame FPV)**: gives the policy ego-view; matches what a human FPV pilot sees. Equivalent to UAV-Flow's recorded camera frames (which are drone-body-mounted real cam). One MuJoCo `<camera>` entry parented to the drone body, with `pos="0 0.05 0"` (just forward of CoM) and `xyaxes` set for forward-facing orientation.

### 3c.4 Determinism — the only real risk

For Stage 2's replay to land at the same scene state Stage 1 produced, the env must be deterministic: same `(qpos_0, qvel_0)` initial condition + same action sequence + same RNG seed → same trajectory. MuJoCo satisfies this in general but with two caveats:

1. **Floating-point drift over long episodes.** ~100-step episodes are well within the safe regime; ~1000-step episodes may show ε-level divergence after the long replay. Mitigation: at each Stage 2 step, *also* check the env's reported state against the Stage 1 pickle; if drift exceeds a threshold, log a warning. Don't try to "correct" — that breaks reproducibility.
2. **Camera-rendering side-effects.** `mjr_readPixels` and `mujoco.mj_step` should not interact, but on some MuJoCo+GPU combinations the render backend can perturb the offscreen context. Mitigation: render *after* `mj_step`, never *before* the next step is computed.

Both Gen9 Epoch 1's `replay_and_capture()` and Gen6V4's visual aligning collection survived this in production, so the pattern is empirically safe.

### 3c.5 How this slots into the §3.5 phased answer

The two-stage pattern is **how Stage 1 of the §3.5 phasing actually gets executed**. Updated phasing:

| Phase | Stage 1 (state) | Stage 2 (images) |
|---|---|---|
| Epoch 4 early | Scripted PID flies parametric routes → state pickles | Skip (validate non-visual FM-PCC first) |
| **Epoch 4 mid–late** | MuJoCo-viewer teleop (§3b.4) records `des_p` + state | **Render dual-cam images via `replay_and_capture`** |
| Epoch 5–6 | Either Stage-1 source feeds the same Stage 2 | Re-render with domain randomization for sim-to-real prep |
| Epoch 7+ | UAV-Flow real-pilot data (if scope) | Already has real cam frames — Stage 2 is moot |

### 3c.6 Concrete deliverables to build (when this becomes Epoch 4 mid–late work)

1. **`tools/generate_state_trajectories.py`** *(new)* — Stage 1. Wraps `uav_naive_test/flight_controller.py` (Epoch 2 cascaded PID) + a scene-config iterator. Outputs `state_pickles/<ep_id>.pkl` with the same key structure D3IL uses (`['robot']['des_c_pos']`, etc., adapted: maybe `['drone']['des_p']`, `['drone']['p']`).
2. **`tools/collect_visual_uav_data.py`** *(new)* — Stage 2. Direct mirror of `collect_visual_avoiding_data/collect_visual_avoiding_data.py:151` (`replay_and_capture`), adapted for the UAV env. Reads `state_pickles/`, replays, saves `images/{bp,inhand}-cam/`.
3. **MuJoCo MJCF additions** — add two `<camera>` entries to the UAV scene XMLs: one fixed overhead (`world-cam`), one body-mounted forward (`fpv-cam`).
4. **`config/uav-d3il-visual.py`** *(new, when Gen11 hits "visual UAV" epoch)* — mirrors `config/avoiding-d3il-visual.py` with UAV-appropriate `obs_dim` (probably 6 for `[p(3), v(3)]`) and `action_dim` (3 for position-target deltas). 6 obstacle sphere constraints stay the same pattern.

None of this is in Epoch 4's *current* scope — Epoch 4's RESEARCH.md aims to land Stage 1 (state-only generation) first. But the design above lets Stage 2 be added later as a separate epoch without re-doing Stage 1.

### 3c.7 What this idea is NOT solving

Worth being explicit so the scope doesn't creep:

- **Not a teleop system.** §3b covers human-in-the-loop teleop; §3c is about post-hoc image rendering on top of *any* state source (scripted OR teleop). The two compose: Stage 1 can be either.
- **Not a sim-to-real bridge.** Renders are still MuJoCo-quality. If you need photoreal images for sim-to-real, that's a Stage 2 *enhancement* (different rendering backend) — out of Epoch 4 scope.
- **Not multi-agent.** One drone, one Stage 1, one Stage 2. Multi-drone teleop / multi-agent rendering is a separate research thread.
- **Not a substitute for real data.** §3.4's "if you want a deployable drone, you need real pilots" argument stands. §3c is for *training-time* visual data in MuJoCo, not for the final eval.

### 3c.8 One-line summary

The Gen9 Epoch 1 pattern — **manual state generation → post-hoc dual-camera replay** — is the cleanest deliverable shape for Epoch 4's visual UAV pipeline. It separates state from pixels, lets us iterate on either in isolation, supports cheap multi-camera and domain-randomization variants, and reuses an already-validated `replay_and_capture()` template byte-for-byte. **Adopt this as the Epoch 4 mid-to-late stage architecture; promote `tools/generate_state_trajectories.py` + `tools/collect_visual_uav_data.py` to the Epoch 4 deliverables list.**

---

## 3d. The "manual vs automated" spectrum — what does *manual generation* actually mean?

**Triggering question** (user, paraphrased):
> "I thought 'manual generation' meant *we draw a line by hand* showing how the UAV crosses the obstacles in the abstract-geometry env. But your earlier writing sounded like an automated PID. Which is it? Add a section on how manual/auto ideas relate."

The user caught a real ambiguity. Earlier sections used "manual generation" loosely — sometimes meaning "scripted controller" (automated), sometimes implying "human draws a path" (truly manual). They are *different* points on a spectrum, and the right one for our use case isn't obvious without laying out all six options.

### 3d.1 The spectrum (six approaches, from most-manual to most-automated)

| # | Approach | Who decides the path? | Who executes? | Human time per traj | Style diversity | Throughput |
|---|---|---|---|---|---|---|
| **A** | **Hand-drawn waypoint sketch** | Human draws line in 2-D plot / GUI | Spline + tracking PID | ~30-60 s | High (creative) | ~60-120/hr |
| **B** | **Waypoint-and-go** | Human clicks ~5 sparse waypoints | PID navigates between | ~10-15 s | Medium (route choice) | ~250/hr |
| **C** | **Joystick teleop** *(§3b.4)* | Human flies in real-time | PID closes inner loop | ~episode-duration | High (continuous control) | ~30/hr |
| **D** | **Parametrised scripted** | Generator script (`(start, end, "go-left-of-obs3")` table) | PID | 0 (one-time script write) | Medium (parameter-sweep based) | ~10k/hr |
| **E** | **Sample-based planner** (RRT, A*) | Algorithm finds any collision-free path | PID tracks the result | 0 | Low (algorithm bias) | ~1k/hr |
| **F** | **Optimal-control / MPC planner** | Cost function (smoothness + collision) | The planner IS the controller (already produces controls) | 0 | Very low (one solution per IC) | ~100s/hr |

"Manual" in the strictest sense = **A** (literal hand-drawn line). "Manual" in the loose sense I used earlier = **D** (parametrised scripted) — a human writes the GENERATOR by hand, but the trajectories themselves come out automated.

### 3d.2 What earlier sections were *actually* recommending

| Earlier section | Actually meant | Spectrum row |
|---|---|---|
| §3.1 "manual generation means a controller we wrote flies the drone inside MuJoCo" | Parametrised scripted | **D** |
| §3.2(b) "policy learns whatever demonstrator you give it" — examples all assumed PID/MPC | Scripted, mostly | **D / F** |
| §3b.4 "MuJoCo teleop with joystick" | Joystick teleop | **C** |
| §3b.6 "co-pilot mode: human clicks waypoints, PID flies between" | Waypoint-and-go | **B** |
| §3c "manual state generation" | Parametrised scripted | **D** |

So my earlier "manual generation" was almost always **D** (scripted), occasionally **B** or **C**. **Never A** (literal hand-drawn lines), which is what your question implied. The discrepancy is the ambiguity worth resolving.

### 3d.3 Should we use A (literal hand-drawn lines)?

**Probably no, but A has a defensible niche.**

**The case AGAINST A as the primary path:**

1. **Throughput.** FM-PCC training on D3IL-scale problems needs ~1000–10000 trajectories. Even at 60 trajectories per hour, A demands 17-170 hours of human drawing time — a week of work just to make a dataset that the script can produce overnight.
2. **Floor-quality issue.** Hand-drawn paths are *not* dynamically feasible by construction. A line drawn on a 2-D plot may have curvature radii that violate the drone's max-acceleration constraint. The system would either reject the path or "snap" it to feasibility, in which case the human's intent doesn't survive the post-processing anyway — at which point you're back to D.
3. **Coverage.** A human draws what feels natural; they will not systematically cover the state space the way a parameter sweep does. For learning a *planner* (which is what FM-PCC is), you need distributional coverage, not stylistic richness.
4. **The thing A would buy** (genuinely novel paths around obstacles, captured human intuition for "what looks like a good route") is what **C (teleop)** gives you at higher throughput with feedback-loop quality control built in.

**The case FOR A as a small supplement:**

1. **Seed paths for hard scenarios.** If the script's auto-generated paths miss a topologically interesting route (e.g., "duck under and pop up between obstacles 4 and 5"), a single hand-drawn path is a cheap way to *seed* the dataset with that mode. The script can then generate variations around the seed.
2. **Demonstrating intent.** For papers / presentations: showing a hand-drawn path that the policy then replicates is much more compelling than "the script generated this." Useful for figures, not for training data scale.
3. **Sanity check.** Drawing 5–10 paths by hand and feeding them through the same Stage-1 → Stage-2 pipeline as the scripted paths catches schema mismatches early.

So: **A as 0.1% of the dataset** (10 seed paths in a 10k-trajectory corpus) makes sense; A as the bulk does not.

### 3d.4 Recommendation: layered combination

Don't pick one row of the spectrum — combine them. Updated phasing (refinement of §3.5 + §3c.5):

| Phase | Primary source | Volume | What it contributes |
|---|---|---|---|
| Epoch 4 early | **D** (scripted, parametrised by start/end pairs and "go-left/right-of-obs-N" labels) | ~1k-10k | Bulk coverage of obstacle-avoidance patterns |
| Epoch 4 mid | **F** (MPC planner as oracle, generates feasible+optimal) | ~500-1k | High-quality "expert" trajectories for the model to imitate; useful upper-bound benchmark |
| Epoch 4 late | **C** (joystick teleop, §3b.4) | ~100-200 | Stylistic richness, multi-modal demos (the FM/diffusion sweet spot) |
| **Optional augmentation** | **A** (hand-drawn) | ~10-50 | Seed paths for hard topological cases; figures for paper |
| Out of scope (Epoch 4) | **B** (waypoint-click) | n/a | Useful for §3c-style co-pilot mode in later epochs |

Why this layering: rows D and F give you **distributional coverage and feasibility floors**, row C adds **multi-modality** (the property that justifies using diffusion/FM in the first place), and row A adds **interpretability seeds** for paper figures. Each layer has a different cost-to-benefit ratio; mixing them is strictly better than any single source.

### 3d.5 What "abstract-geometry env" already gives us for free

The user's mental model of an "abstract-geometry env we created" is the right one: it's a MuJoCo XML with N obstacle cylinders, a planar workspace, and a kinematic drone. That env supports *every* row of the spectrum:

- For **A**: tools/draw_path.py opens a matplotlib top-down view of the env, captures mouse clicks, fits a B-spline, hands the spline to the PID as a reference trajectory.
- For **B**: same tool, fewer clicks, no spline (PID handles the straight-line segments).
- For **C**: §3b.4's `tools/teleop_uav.py`.
- For **D**: a generator script that iterates over `(start, end) ∈ start_set × end_set` and labels paths with "(L, R, R, L, L, R)" indicating which side of each obstacle to pass on.
- For **E**: drop in [`OMPL`](https://ompl.kavrakilab.org/) or the standard library's [`networkx.astar_path`](https://networkx.org) on a grid discretization.
- For **F**: MJPC / drake's iLQR / a Python iLQR. Higher integration cost.

The env itself is invariant; only the *trajectory source* changes. The Stage-1 / Stage-2 split from §3c means the env is also invariant w.r.t. whether you're producing state-only or visual data.

### 3d.6 The honest one-liner I should have used earlier

When earlier sections said "manual generation," what was meant was:

> **"A trajectory generator that we (humans) wrote in Python, parametrised by start/end conditions and high-level route hints, executed by a deterministic controller in MuJoCo."**

Which is "manual" in the sense that *we authored the generator*, not in the sense that *we drew each path by hand*. The terminology was ambiguous; **D** (parametrised scripted) is the correct technical name and what this document now uses consistently in subsequent sections.

### 3d.7 One-line summary of §3d

The "manual vs automated" axis is a six-row spectrum from **A** (hand-drawn lines, 30–60 s of human time per trajectory) through **D** (parametrised scripted, ~0 human time per trajectory) to **F** (full MPC oracle). Earlier sections used "manual" to mean **D**, not **A**. The recommended Epoch 4 architecture is a *layered combination* — bulk **D** for coverage, a slice of **F** for feasibility upper-bound demos, a smaller slice of **C** (teleop) for stylistic multi-modality, and optionally a handful of **A** seeds for hard topological cases and paper figures. The "abstract-geometry env" we built supports all six paths via the same Stage 1 / Stage 2 pipeline (§3c).

---

## 4. The cross-domain pattern

If you abstract over D3IL, UAV-Flow, and our drone work, the choice of expert demonstrator follows a single principle:

> **The demonstrator must be a competent solver of the task you want the policy to learn — and whatever properties the demonstrator has, the policy will inherit.**

| Project | Task | Competent solver chosen | Why |
|---|---|---|---|
| D3IL | Manipulation, 7 tasks, multi-modal IL | Multiple human teleoperators on real Franka | Multi-modality requires multiple solvers; humans available; multi-modal IL is the research point |
| UAV-Flow | Language-conditioned UAV flight | Single expert pilot per drone | Language-conditioning + real-flight evaluation is the research point; only pilots can follow free-form instructions |
| FM-PCC (D3IL tasks) | Constraint-aware motion planning | D3IL's human teleop (reused as-is) | We don't need new data; the IL property is orthogonal to the constraint-projection property we're studying |
| **FM-PCC (drones, Epochs 4–7)** | Constraint-aware UAV motion planning | **A PID / MPC controller we write** | We need a competent constraint-respecting demonstrator. Real pilots don't know our constraint formulation; a script that we write does. We are not studying multi-modality or language-following |

So the answer to "is manual generation sensible?" is **yes for what we're actually doing**, and the seemingly-conflicting answer "we need real expert data" is **right for a different project** (deployable drone, language-conditioned IL, multi-modal style transfer). Different research goal → different data source.

---

## 5. Concrete deltas to RESEARCH.md (what should change in the parent doc)

After writing this file, the parent `RESEARCH.md` is still correct in its bottom-line recommendation but could be tightened:

| RESEARCH.md section | Suggested edit |
|---|---|
| §3.1 (UAV-Flow inventory) | Add the phrase "expert pilot trajectories" sourced from the arXiv abstract — clarifies provenance. |
| §3.3 (What UAV-Flow IS useful for) | Add: "statistical targets for our manual-controller tuning (e.g., match UAV-Flow's avg flight speed, avg path curvature, altitude bands)". |
| §4 (Manual generation) | Reference this file's §3 for the *justification*; current §4 mostly describes the *mechanism*. |
| New §4.6 (or footnote) | Note that mono-modal manual data is OK for our use case but if we ever benchmark on D3IL-style multi-modality, we'd need multiple controllers / multiple gain sets. |

Not applied here — left for the next pass on `RESEARCH.md` itself, or for whichever epoch document supersedes it.

---

## 6. Cross-references

- [`RESEARCH.md`](RESEARCH.md) — primary epoch document; this file extends its §3 (UAV-Flow) and §4 (manual generation) with provenance evidence and a defense of the manual-generation choice.
- [`../Epoch1_UAV_model/MIGRATION_PLAN.md`](../Epoch1_UAV_model/MIGRATION_PLAN.md) — UAV model choice rationale.
- [`../Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md`](../Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md) — cascaded PID validation; the candidate "manual demonstrator".
- D3IL paper: ICLR 2024, OpenReview `id=6pPYRXKPpw`; codebase teleop at `d3il/environments/d3il/d3il_sim/sims/sl/teleoperation/`.
- UAV-Flow paper: NeurIPS 2025, arXiv `2505.15725`; codebase at `/workspaces/UAV-Flow/`.

---

## 7. One-line summaries

- **D3IL got expert data from** multiple human teleoperators driving real Franka Panda robots in a KIT lab via the primary→replica teleop rig at `d3il/.../teleoperation/`. `TeachingLog` captures `tau_cmd, c_pos, c_vel, c_quat, j_pos, j_vel, gripper, power` per step — both raw torque commands AND full state. But the *training* dataloader (`aligning_dataset.py`) discards torque and uses **`vel = des_c_pos[1:] − des_c_pos[:-1]`** as the action — i.e., forward differences of the human's commanded Cartesian position.
- **UAV-Flow got expert data from** expert pilots flying real drones (UAV-Flow) and simulated drones in UnrealZoo (UAV-Flow-Sim). The recorded JSON is pose-only — `reference_path_raw` is `[T, 6]` of `(x, y, z, roll, pitch, yaw)` — no torque, no IMU. The OpenVLA-UAV loader (`uav_dataset.py:_process_episode`) computes **actions on-the-fly as local-frame pose deltas** `Δ(x, y, z, yaw)`. Structurally identical to D3IL's "action = forward-difference of position intent" convention, just in body frame.
- **Manual generation is sensible for FM-PCC-on-drones** because (a) we are validating a constraint-aware planner architecture, not building a deployable real-world flight policy or studying language-conditioning; (b) *both* D3IL and UAV-Flow's actual training-time action format is "position-delta", which a PID/MPC controller in MuJoCo can produce natively without any human-in-the-loop; (c) FM/iMF theory is demonstrator-agnostic — it fits whatever manifold the data lies on. The instinct "we need real expert data" is correct *for a different project* (deployable drone, language-conditioned IL, multi-modal style transfer); none of those is Epoch 4's scope.
- **"Manual generation" is a six-row spectrum, not one technique.** Rows A→F: hand-drawn waypoints → click-and-go → joystick teleop → **parametrised scripted (what earlier sections actually meant)** → sample-based planner → optimal-control oracle. Earlier writing was loose; **D (scripted)** is the load-bearing primary source, with optional **C (teleop)** for multi-modality, **F (MPC oracle)** for a feasibility upper bound, and **A (hand-drawn)** at ≤0.1% of dataset only for paper-figure seeds. The abstract-geometry env supports all six. *(See §3d for the full breakdown.)*
- **Gen9's two-stage collection pattern (state-then-images) is the recommended Epoch 4 architecture.** Stage 1 = scripted PID / MPC (or teleop, per §3b) flies the drone and records state-only pickles. Stage 2 = `replay_and_capture()`-style replay through the env with `bp-cam` (overhead world view) + `inhand-cam` / `fpv-cam` (body-frame FPV) mounted, saving frames per timestep. Decouples controller iteration from rendering, enables domain randomization at Stage 2 without re-flying, and reuses the already-validated `collect_visual_avoiding_data.py:replay_and_capture` template (Gen9 Epoch 1). Concrete deliverables: `tools/generate_state_trajectories.py` (Stage 1) + `tools/collect_visual_uav_data.py` (Stage 2, mirror of Gen9 script) + two `<camera>` MJCF entries (world + body-mounted forward).
- **An "Isaac-Lab-style game" for human-piloted demos is feasible but should be built in MuJoCo, not Isaac Lab.** Isaac Lab's teleop drivers (`Se3KeyboardController`, `Se3GamepadController`, etc.) require Omniverse Kit + recent CUDA/driver stack that our Slurm cluster doesn't ship and shouldn't try to ship — *and human teleop has to happen at a laptop, not the cluster, regardless of which sim is used*. The MuJoCo equivalent is a ~150-line `tools/teleop_uav.py` wrapping `mujoco.viewer.launch_passive` + `pygame.joystick` + the **already-validated Epoch 2 cascaded PID** (RMS 0.029 m on Task C 9D). Recommended Epoch 4 mid–late deliverable: this rig + a "co-pilot" mode (human clicks waypoints, PID flies between) so a human session yields multi-modal demos at near-scripted cost. Isaac Lab is useful as a *design reference* (SE(3) abstraction, dead-zone/exponential axis mapping, callback-based recording) — those design choices port directly into our MuJoCo rig.
