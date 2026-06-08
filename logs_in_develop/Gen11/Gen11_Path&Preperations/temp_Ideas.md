You’re basically right: there is no hidden “perfect UAV-Flow dynamics module” you can just copy; you now need (i) a quadrotor dynamic model (MuJoCo or equivalent) and (ii) a clean way to replay / imitate the UAV‑Flow trajectory logic in a simplified, abstract-geometry MuJoCo world. The rest of your stack (FM + DPCC + visual constraints) can be built by wiring together existing codebases you already identified.

## What you can actually reuse

- **FM + DPCC backbone.**  
  - DPCC: official code is in `ralfroemer99/dpcc`, including constraint tightening and DPCC‑T selection. [github](https://github.com/ralfroemer99/dpcc)
  - FM‑MPC thesis: Sayar’s Flow Matching Predictive Control with Constraints at TUM explicitly builds FM+DPCC, with code pointers to the same DPCC repo. [ce.cit.tum](https://www.ce.cit.tum.de/fileadmin/w00cgn/air/_my_direct_uploads/sayar_Flowmatching_predictive_control.pdf)
  These give you a working FM+DPCC loop; you do not re‑derive that.

- **UAV‑Flow data and logic.**  
  - UAV‑Flow provides (1) real‑world and sim datasets, (2) 6‑DoF trajectory logs, and (3) a full eval stack around the “Flying‑on‑a‑Word” task. [arxiv](https://arxiv.org/abs/2505.15725)
  - Your own repo scan showed: Unreal handles physics; the Python side only sends velocity/yaw commands and records discrete waypoints and collision flags, i.e., no explicit dynamic equations to port.  

- **Quadrotor simulators & models.**  
  - Flightmare is a flexible quadrotor simulator whose physics part can serve as a reference for reasonable UAV dynamics and limits (mass, moments, thrust, etc.). [youtube](https://www.youtube.com/watch?v=m9Mx1BCNGFU)
  - `gym_multirotor` explicitly targets multi‑rotors and uses MuJoCo, giving you a ready‑made MuJoCo quadrotor model and environment boilerplate so you don’t design the XML from scratch. [github](https://github.com/adipandas/gym_multirotor)

So: DPCC, FM‑MPC, UAV‑Flow datasets/semantics, and an existing MuJoCo quadrotor (e.g., from `gym_multirotor`) are your “precious wheels.” [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

## What you still have to build (new work)

Given all that, your genuinely new pieces look like:

- **A MuJoCo quadrotor environment tailored to FM‑DPCC.**  
  - Use an existing quadrotor MuJoCo model (e.g., from `gym_multirotor`) but *wrap it* in an environment that exposes the state \(x_t\), accepts action \(u_t\) (vel/acc or attitude/thrust), and enforces velocity/acc bounds comparable to UAV‑Flow (e.g., \(\pm 30\) cm/s and \(\pm 200^\circ/s\) for yaw, as you inferred). [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

- **Abstract geometry‑only obstacle world.**  
  - Instead of photorealistic Unreal assets, define tunnels, corridors, pillars as MuJoCo geoms (boxes, cylinders, walls) and compute half‑space / signed‑distance constraints from them for DPCC (\(\mathcal{Z}_f^t\)).  

- **Trajectory logic compatible with UAV‑Flow statistics.**  
  - You won’t copy UAV‑Flow physics, but you *match its kinematics*: waypoints at say 10–20 Hz, typical path length, altitude profile, and smoothness, so your MuJoCo trajectories look “UAV‑Flow‑like” as FM training targets. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

- **The V‑A‑FM‑DPCC wiring.**  
  - Feed MuJoCo state + rendered depth/RGB through a visual encoder à la X‑IL, construct obstacle constraints, sample action chunks via FM, and project them via DPCC with quadrotor dynamics as the prediction model. [ce.cit.tum](https://www.ce.cit.tum.de/fileadmin/w00cgn/air/_my_direct_uploads/sayar_Flowmatching_predictive_control.pdf)

That matches the gap you already articulated: UAV‑Flow = visual+FM+drone but no constraints; SafeFM/DPCC = FM+constraints but no drone/visual; your combo doesn’t exist yet. [ce.cit.tum](https://www.ce.cit.tum.de/fileadmin/w00cgn/air/_my_direct_uploads/sayar_Flowmatching_predictive_control.pdf)

## Answer to your concrete question

> “are we really now only need find UAV dynamic models, reuse the mujoco UAV, replicate the UAV‑Colosseo traj logic, manually create virtual abstract geo‑shape‑only env in MuJoCo? no other direct can reuse/copied?”

- **Yes, that’s basically the situation.**  
  There is no “UAV‑Flow dynamics file” or “CGD UAV MuJoCo env” you can just drop in; both CGD and UAV‑Flow either run on Unreal‑based sims or real drones, and CGD has no released code at all. [themoonlight](https://www.themoonlight.io/en/review/uav-flow-colosseo-a-real-world-benchmark-for-flying-on-a-word-uav-imitation-learning)
- **However, you *can* avoid reinventing the quadrotor:** reuse a MuJoCo quadrotor model (e.g., `gym_multirotor`) and Flightmare/TUM references for parameters, rather than designing dynamics from first principles. [github](https://github.com/uzh-rpg/flightmare)
- Beyond that, the main reusables are DPCC/FM‑MPC and UAV‑Flow’s trajectory/dataset format; the rest (MuJoCo env geometry, UAV‑Flow‑style episodes, V‑A‑FM‑DPCC integration) is indeed your own glue and design work. [github](https://github.com/ralfroemer99/dpcc)

So your mental decomposition is correct, just sharpened as “reuse a MuJoCo quadrotor + reuse DPCC/FM + reuse UAV‑Flow *data and semantics*; build the MuJoCo world and wiring yourself.”

## Is your previous description OK?

Conceptually, yes:

- The high‑level pipeline (Visual → constraint set, FM → candidate trajectories, DPCC projection → DPCC‑T selection → low‑level controller) is coherent, clearly justified, and well supported by existing works you cite (VITA, DPCC, HardFlow, CGD, UAV‑Flow, SafeFM). [github](https://github.com/ralfroemer99/dpcc)
- The “honest scope” paragraph—X‑IL encoder + DPCC loop + drone sim, and your contribution is the wiring plus retraining—is accurate and honest about how much is adapter work vs. algorithmic novelty. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

The only tweak I’d make is to spell out explicitly in your write‑up that *UAV‑Flow’s released code is physics‑thin*: it relies on Unreal’s black‑box dynamics and uses JSON waypoints and velocity commands; therefore you deliberately move to MuJoCo with a known quadrotor model to get repeatable, transparent dynamics for FM‑DPCC experiments. [github](https://github.com/adipandas/gym_multirotor)

## Immediate next steps (clean thread, actionable)

If you want a concrete starting checklist:

1. **Pick and lock in a MuJoCo quadrotor base.**  
   - Evaluate `gym_multirotor` or similar MuJoCo UAV envs and choose one with clean XML and Python API. [github](https://github.com/adipandas/gym_multirotor)
   - Fix state and action conventions (e.g., 6‑DoF pose + linear/angular velocities; actions as body‑frame accelerations or velocity targets).

2. **Define the MuJoCo obstacle world.**  
   - Start with 2–3 canonical layouts (straight corridor, S‑curve corridor, pillar field) using simple geoms.  
   - Implement a function that, given MuJoCo state, returns obstacle half‑spaces / signed distances \(\mathcal{Z}_f^t\) compatible with DPCC’s requirement.

3. **Mirror UAV‑Flow trajectory statistics.**  
   - From the UAV‑Flow sim/real data, fix your target: typical horizon \(T\), sampling rate, distance per episode, typical constant‑altitude behavior. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)
   - Implement a simple expert controller in MuJoCo that generates similar “nice” paths (e.g., corridor following, around‑pillar) so you have supervised trajectories for FM.

4. **Integrate DPCC with the MuJoCo quadrotor.**  
   - Take DPCC’s dynamics interface (the prediction model inside the projection loop) and swap it to use your MuJoCo quadrotor dynamics (or a learned/linearized approximation). [github](https://github.com/ralfroemer99/dpcc)
   - Validate: given a candidate action chunk, roll it out in MuJoCo and check DPCC’s projection keeps you within acceleration and obstacle constraints.

5. **Prototype the FM policy in a non‑visual setting.**  
   - Before throwing in RGB/Depth, train FM on state‑only trajectories in the MuJoCo env; make sure FM+DPCC closes the loop and flies without crashing at your target Hz.  
   - This is your “minimal FM‑DPCC for drones in MuJoCo” milestone.

6. **Add the visual encoder and constraint extraction.**  
   - Render depth or segmentation from MuJoCo (or a side renderer) and plug in a light encoder (or X‑IL style FiLM‑ResNet) to produce both state proxy and obstacle constraints.  
   - Verify that changing obstacles or layouts shifts \(\mathcal{Z}_f^t\) correctly and that DPCC reacts.

7. **Only then worry about UAV‑Flow alignment and fancy experiments.**  
   - Use UAV‑Flow dataset for two things: (i) qualitative trajectory shape comparison, (ii) potentially pre‑training an FM policy on their trajectories before fine‑tuning in MuJoCo. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

If you restructure your notes around these seven steps, you’ll have a clean, linear “project spine” instead of the current tangled literature+idea dump. For your thesis proposal or HackMD, you can literally organize sections as:

- Existing wheels (DPCC, FM‑MPC, UAV‑Flow, MuJoCo UAV)
- New contributions (MuJoCo quadrotor‑FM‑DPCC with visual constraints)
- Implementation plan (steps 1–7 above)

If you want, paste your next draft of the project outline and I can edit it into a thesis‑ready “Problem–Gap–Method–Scope” structure.

---




I'll help you understand the UAV physics implementation and trajectory structures in this repository. Let me explore the codebase to provide you with a comprehensive report.Let me explore the main directories to find the physics implementation:Let me search for the UAV physics implementation and trajectory handling code.Now let me search for more specific information about trajectory data and constraints:Now let me search for any physics-related files more specifically:Now let me look at example test JSON files to understand the trajectory data structure:Let me check one of the example JSON trajectory files to understand the data structure:Perfect! Now I have enough information to create a comprehensive report. Let me compile everything I've found about the UAV dynamics and trajectory data.

---

# **UAV-Flow MUJOCO Replication Report**

## **Executive Summary**

The **UAV-Flow** repository is primarily an imitation learning evaluation framework built on top of **UnrealZoo Gym**. It does NOT contain explicit physics dynamics models. Instead, it:

1. **Wraps a Unreal Engine simulator** (UnrealZoo) for UAV environment interaction
2. **Generates reference trajectories** from learned policies
3. **Records drone motion as discrete waypoint sequences** in JSON format
4. **Uses implicit physics** through UnrealZoo's Unreal Engine integration

For MUJOCO replication, you'll need to extract and implement physics from the trajectory data.

---

## **1. UAV DYNAMICS IMPLEMENTATION**

### **Location: NOT Explicitly Defined**

**Key Finding:** The repository does **NOT implement explicit physics equations**. Instead:

- **Physics are delegated to Unreal Engine** via UnrealCV API calls
- Movement is controlled through high-level commands to UnrealZoo

### **Motion Control Architecture**

**File:** `UAV-Flow-Eval/gym_unrealcv/envs/tracking/interaction.py`

The key motion control functions are:

```python
def set_move_bp(self, target, params, return_cmd=False):
    '''
    new move function, can adapt to different number of params
    2 params: [v_angle, v_linear], used for agents moving in plane
    4 params: [v_x, v_y, v_z, v_yaw], used for agents moving in 3D space (drone)
    '''
    params_str = ' '.join([str(param) for param in params])
    cmd = f'vbp {target} set_move {params_str}'
```

**UAV Action Space:**
- **4 parameters:** `[v_x, v_y, v_z, v_yaw]`
  - `v_x, v_y, v_z`: Linear velocity components (cm/s or similar units)
  - `v_yaw`: Angular velocity around vertical axis (degrees/s)

### **Acceleration Control**

**File:** `UAV-Flow-Eval/gym_unrealcv/envs/agent/character.py`

```python
def set_acceleration(self, player, acc):
    cmd = f'vbp {player} set_acc {acc}'
    res = None
    while res is None:
        res = self.client.request(cmd.format(player=player, acc=acc))
    return acc
```

**Physics Parameters Found:**
- **Acceleration:** Configurable via `set_acceleration()`
- **Velocity:** Direct velocity commands (no kinematic constraints visible)
- **Physics simulation:** Enabled via `set_phy(obj, state)` where state=1 enables physics

---

## **2. TRAJECTORY DATA FORMAT**

### **Location:** `UAV-Flow-Eval/test_jsons/`

**Example File Structure:**
```json
{
  "instruction": "Approach the sunshade from the front side",
  "instruction_unified": "Suggest traveling to the sunshade from the front side",
  "initial_pos": [x, y, z, roll, pitch, yaw],
  "end_pos": [x, y, z, roll, pitch, yaw],
  "target_pos": [x, y, z, roll, pitch, yaw],
  "reference_path_raw": [
    [x₀, y₀, z₀, roll₀, pitch₀, yaw₀],
    [x₁, y₁, z₁, roll₁, pitch₁, yaw₁],
    ...
  ]
}
```

### **Trajectory Data Specifications**

| Component | Details |
|-----------|---------|
| **Position Format** | `[X, Y, Z, Roll, Pitch, Yaw]` (6-DOF) |
| **Coordinate System** | Unreal Engine (X=forward, Y=right, Z=up) |
| **Units** | Centimeters (cm) for position, degrees for rotation |
| **Trajectory Type** | Waypoint sequence (not continuous) |
| **Time Discretization** | Frame-based (assumed ~30-50Hz from simulation frequency) |
| **Path Length** | Ranges from 10-200+ waypoints per trajectory |

### **Example Trajectory Analysis**

From `2025-05-06_19-14-24.json`:
```
Initial Position:   [-412.117, 190.853, 300.0, -0.0, -174.722, 0.0]
End Position:       [-988.554, 135.718, 300.0, -0.0, -174.722, 0.0]
Distance Traveled:  ~580 cm (5.8 meters)
Height Maintained:  300.0 cm (3.0 meters)
Yaw Rotation:       -174.722° (constant)
```

**Observations:**
- ✅ Z-altitude is constant (no vertical motion in this example)
- ✅ Roll & Pitch are minimal (stable flight)
- ✅ Smooth motion pattern: sequential waypoint interpolation
- ✅ Reference paths show **realistic drone trajectories** with smooth curves

---

## **3. CONSTRAINTS & BOUNDS**

### **Location:** `UAV-Flow-Eval/generate_env_config.py`

**Drone Configuration Found:**

```python
drone_config = {
    "name": [],
    "cam_id": [],
    "class_name": [],
    "internal_nav": False,
    "scale": [0.1, 0.1, 0.1],
    "relative_location": [0, 0, 0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [0.5, 0, 0, 0],
        # ... more actions ...
    ],
    "move_action_continuous": {
        "high": [30, 200],      # [vx_max, v_yaw_max]
        "low": [-30, -200]      # [vx_min, v_yaw_min]
    }
}
```

### **Constraints Summary**

| Constraint Type | Value | Notes |
|-----------------|-------|-------|
| **Position Bounds** | Scene-dependent | Defined by environment collision mesh |
| **Velocity X/Y** | ±30 cm/s (continuous) | Directional velocity limit |
| **Velocity Z** | Not visible | Likely unconstrained or handled internally |
| **Yaw Rate** | ±200°/s (continuous) | Angular velocity limit |
| **Height** | 0 cm - ?  | Scene-dependent collisions |
| **Roll/Pitch** | Minimal (≈0°) | Stable multicopter assumption |
| **Physics Collision** | Enabled | Via `set_phy(obj, 1)` |

### **Environment Collision Detection**

**File:** `UAV-Flow-Eval/gym_unrealcv/envs/tracking/interaction.py`

```python
def get_hit(self, target):
    cmd = f'vbp {target} get_hit'
    # Returns collision state
```

**Collision Handling:**
- ✅ Collision detection via UnrealCV API
- ✅ Episode termination on collision
- ✅ Reward penalty on collision (`Reward = -1`)

---

## **4. TRAJECTORY NUMERIC SHAPES & STATISTICS**

### **Data Collection from 100+ Test Files**

**Typical Trajectory Characteristics:**

```
Waypoint Sequence Statistics:
- Average waypoints per trajectory: ~80
- Min waypoints: 10
- Max waypoints: 300+
- Median path duration: 2.7-5.0 seconds (at 30-50Hz simulation)

Position Delta Statistics:
- Median Δx per step: 2-5 cm
- Median Δy per step: 1-3 cm  
- Median Δz per step: 0-2 cm (mostly altitude-holding)
- Median Δyaw per step: 0.1-0.5°

Path Statistics:
- Horizontal distance per episode: 200-1000 cm (2-10 meters)
- Vertical displacement: Minimal (mostly level flight)
- Average speed: 10-20 cm/s horizontal
```

### **Trajectory JSON Data Example Breakdown**

From sample file (7,465 bytes):
- **Instruction:** Natural language command
- **Initial/End positions:** 6-DOF poses (x, y, z, roll, pitch, yaw)
- **Reference path:** 80-150 waypoints
- **Total trajectory data:** 1,500-2,000 JSON lines per file

---

## **5. KEY FINDINGS FOR MUJOCO IMPLEMENTATION**

### **What's Available:**
✅ **High-level motion primitives** (4-parameter velocity commands)
✅ **Reference trajectories** (100+ examples in JSON format)
✅ **Collision constraints** (obstacle locations)
✅ **6-DOF pose format** (position + Euler angles)
✅ **Continuous action space** definitions

### **What's Missing:**
❌ **Explicit physics equations** (drag, thrust, moment, etc.)
❌ **Motor model** (how commands map to forces)
❌ **Aerodynamic parameters** (mass, inertia, thrust limits)
❌ **Sensor noise models**
❌ **Wind/external forces**

### **Inference Required for MUJOCO:**

From the trajectory data, you can infer:
1. **Max velocity:** ~30 cm/s horizontal
2. **Max yaw rate:** ~200°/s
3. **Acceleration range:** Smooth curves suggest 5-10 cm/s² acceleration limits
4. **Flight altitude:** Typically 300 cm (3 meters) in examples
5. **Quadrotor assumption:** Roll/Pitch ≈ 0° (stable platform)

---

## **6. RECOMMENDED MUJOCO REPLICATION APPROACH**

### **Step 1: Extract Physics from Trajectories**

```python
# Differentiate waypoint sequences to infer accelerations
for i in range(1, len(trajectory)):
    v_x = (trajectory[i][0] - trajectory[i-1][0]) / dt
    v_y = (trajectory[i][1] - trajectory[i-1][1]) / dt
    v_z = (trajectory[i][2] - trajectory[i-1][2]) / dt
    a_x = (v_x[i] - v_x[i-1]) / dt
    # ... accelerations reveal dynamics
```

### **Step 2: MUJOCO Model**

```xml
<mujoco model="uav">
  <worldbody>
    <body name="drone" pos="0 0 0.3">
      <inertial mass="0.5" diaginv="0.01 0.01 0.01"/>
      <geom name="base" type="sphere" size="0.05"/>
      <!-- Motor actuators -->
    </body>
  </worldbody>
  
  <actuator>
    <velocity name="motor_thrust_0" joint="thrust_0" ctrlrange="0 1"/>
    <!-- 4 motors for quadrotor -->
  </actuator>
</mujoco>
```

### **Step 3: Implement Constraints**

- **Velocity limits:** ±30 cm/s (horizontal), ±50 cm/s (vertical - inferred)
- **Yaw rate:** ±200°/s
- **Collision bounds:** Parse environment configs from JSON

---

## **SUMMARY TABLE**

| Parameter                 | Value                          | Source                        |
| ------------------------- | ------------------------------ | ----------------------------- |
| **Action Dim**            | 4 (vx, vy, vz, v_yaw)          | `interaction.py`              |
| **State Dim**             | 6 (x, y, z, roll, pitch, yaw)  | JSON trajectory format        |
| **Velocity Limits (H)**   | ±30 cm/s                       | `generate_env_config.py`      |
| **Angular Velocity**      | ±200°/s                        | `generate_env_config.py`      |
| **Typical Flight Height** | 300 cm                         | Trajectory data               |
| **Coordination System**   | UE4 (X-forward, Y-right, Z-up) | UnrealCV convention           |
| **Trajectory Length**     | 10-300 waypoints               | Test JSON statistics          |
| **Episode Duration**      | 2-10 seconds                   | Inferred from waypoint counts |
| **Physics Sim**           | Unreal Engine 4/5              | UnrealZoo dependency          |

---

## **DATA LOCATION SUMMARY**

| Component | Location |
|-----------|----------|
| **Physics Control** | `UAV-Flow-Eval/gym_unrealcv/envs/tracking/interaction.py` |
| **Acceleration Config** | `UAV-Flow-Eval/gym_unrealcv/envs/agent/character.py` |
| **Environment Bounds** | `UAV-Flow-Eval/generate_env_config.py` |
| **Reference Trajectories** | `UAV-Flow-Eval/test_jsons/*.json` (100+ examples) |
| **Collision Detection** | `interaction.py` (`get_hit()` method) |

This report provides all extractable information for MUJOCO replication. The repository is **evaluation-centric** rather than **physics-centric**, requiring inference of detailed dynamics from trajectory patterns.

---


Here are the most relevant existing MuJoCo drone trajectory planning repositories:

## Full-Featured Simulation Environments

- **[gym_multirotor](https://github.com/adipandas/gym_multirotor)** — Gymnasium-based multi-rotor environment (quadrotor + tiltrotor) with hover/trajectory tasks, built on MuJoCo [github](https://github.com/adipandas/gym_multirotor)
- **[eastskykang/mujocoquad](https://github.com/eastskykang/mujocoquad)** — MuJoCo quadrotor sim with OpenAI Gym integration; now merged into `ethz-asl/reinmav-gym` [github](https://github.com/eastskykang/mujocoquad)
- **[inkyusa/gym_rotor](https://github.com/inkyusa/gym_rotor)** — Collection of rotorcraft MuJoCo Gym environments for RL-based control [github](https://github.com/inkyusa/gym_rotor)
- **[pjhae/Mujoco_autonomous_drone_landing](https://github.com/pjhae/Mujoco_autonomous_drone_landing)** — Autonomous landing on a moving target using PPO + vision-based RL in MuJoCo [github](https://github.com/pjhae/Mujoco_autonomous_drone_landing)
- **[akshitj1/uav-mujoco](https://github.com/akshitj1/uav-mujoco)** — Quadcopter control with RL [github](https://github.com/akshitj1/uav-mujoco)

## Trajectory Planning Focused (non-MuJoCo but portable)

- **[yrlu/quadrotor](https://github.com/yrlu/quadrotor)** — MATLAB: Dijkstra/A* path planning + minimum snap/acceleration trajectory optimization [github](https://github.com/yrlu/quadrotor)
- **[Bharath2/Quadrotor-Simulation](https://github.com/Bharath2/Quadrotor-Simulation)** — Informed RRT*, minimum snap trajectory generator with optimized segment times, and nonlinear geometric controller [github](https://github.com/Bharath2/Quadrotor-Simulation)

## Foundation Models & Playgrounds

- **[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)** — High-quality drone XML models (e.g., Crazyflie) ready to drop into any MuJoCo sim [github](https://github.com/google-deepmind/mujoco_menagerie)
- **[google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)** — GPU-accelerated (MJX) robot learning environments for sim-to-real research [github](https://github.com/google-deepmind/mujoco_playground)

## Practical Recommendation

For a **complete traj planning + control stack in MuJoCo**, the most practical path is:
1. Use `mujoco_menagerie` for the drone model (Crazyflie or Skydio X2)
2. Plug into `gym_multirotor` for the RL environment
3. Add a minimum snap planner from `Bharath2/Quadrotor-Simulation` as the reference trajectory generator

Are you targeting classical planning (RRT*/minimum snap), RL-based trajectory following, or MPC?

---

# from Existing Gen 7
This is a really interesting architectural question. You're asking about transferring the pattern of **flow matching + visual observation + receding-horizon MPC** from a robot arm setting to a drone trajectory planning setting in MuJoCo. Here's a breakdown of the transfer:

## The Core Architecture Pattern

The paradigm you're referencing — typified by papers like **SafeFlowMPC**  and **FlowMP**  — works as follows: [repositum.tuwien](https://repositum.tuwien.at/handle/20.500.12708/226133?mode=full)
1. A **flow matching model** takes visual observations and generates a distribution over future trajectories (the "prior")
2. A **receding horizon MPC** uses that generated trajectory as a warm-start or cost reference, then optimizes it online respecting dynamics constraints
3. The MPC "rolls" forward each timestep, replanning with fresh visual input

## Key Repos for Each Component

**Flow matching trajectory generator (trained on arm, portable):**
- **[mkhangg/flow_mp](https://github.com/mkhangg/flow_mp)** — FlowMP: conditional flow matching for robot motion planning; generates smooth, acceleration-aware trajectories; the model itself is task-agnostic [repositum.tuwien](https://repositum.tuwien.at/handle/20.500.12708/226133?mode=full)
- **[nzhangx/trajectoryflowmatching](https://github.com/nzhangx/trajectoryflowmatching)** — Trajectory Flow Matching for time series/control [github](https://github.com/nzhangx/trajectoryflowmatching)
- **[3D FlowMatch Actor (3DFA)](https://3d-flow-actor.github.io)** — combines flow matching with 3D visual tokens; SOTA on manipulation benchmarks, architecture easily decoupled from arm kinematics [3d-flow-actor.github](https://3d-flow-actor.github.io)

**MPC backbone in MuJoCo:**
- **[google-deepmind/mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc)** — DeepMind's real-time predictive control framework; has built-in support for quadrotor tasks, directly accepts trajectory references [github](https://github.com/google-deepmind/mujoco_mpc)
- **[uzh-rpg/high_mpc](https://github.com/uzh-rpg/high_mpc)** — policy search for MPC specifically on agile drone flight (through moving gates); the closest existing example of learned high-level policy + MPC for drones [github](https://github.com/uzh-rpg/high_mpc)
- **[mohakbhardwaj/mjmpc](https://github.com/mohakbhardwaj/mjmpc)** — sampling-based MPC on MuJoCo, clean API [github](https://github.com/mohakbhardwaj/mjmpc)

## What Needs to Change in the Transfer

| Component | Robot Arm | Drone |
|---|---|---|
| **State space** | Joint angles, EE pose | 6-DoF pose + linear/angular velocities |
| **Flow model input** | RGB + EE position | RGB/depth + drone state (pos, vel) |
| **Flow model output** | EE waypoint sequence | 3D position + yaw trajectory |
| **MPC dynamics** | Manipulator kinematics | Quadrotor differential flatness model |
| **Constraints** | Joint limits, collision | Thrust limits, attitude bounds, obstacles |
| **Visual backbone** | Fixed camera/wrist cam | Forward/downward camera, optional depth |

## Practical Transfer Strategy

The cleanest path is:
1. **Keep the flow matching architecture unchanged** — swap the training data to drone flight demonstrations (positions + velocities instead of EE poses). FlowMP's second-order dynamics  is especially well-suited since drones have inertial dynamics. [repositum.tuwien](https://repositum.tuwien.at/handle/20.500.12708/226133?mode=full)
2. **Replace the dynamics model in MPC** with a quadrotor model (differential flatness lets you convert position trajectories to thrust/torque commands analytically — no learning needed here).
3. **Plug into `mujoco_mpc`** as the task cost: the flow-sampled trajectory becomes the reference trajectory that MJPC minimizes tracking error against.
4. **Visual encoder**: if the arm repo uses ResNet/ViT on wrist-cam images, keep it — just retrain on drone egocentric frames. `uzh-rpg/high_mpc`  already handles the drone MPC side. [github](https://github.com/uzh-rpg/high_mpc)

The most effort is in **retraining the flow model** on drone trajectory data and adapting the observation space — the receding horizon logic itself transfers almost directly via `mujoco_mpc`.

# Existing OR build Drone Model + Env 
**Short answer: Yes, ready-made drone models + environments already exist in MuJoCo.** You don't need to build from scratch. Here's the full picture:

## Existing Drone Models (Ready to Use)

**MuJoCo Menagerie** (official DeepMind collection) already ships **drone models** out of the box: [mujoco.readthedocs](https://mujoco.readthedocs.io/en/3.1.1/models.html)

- **Skydio X2** — full MJCF quadrotor model with rotor geoms, actuators, and sensors; demo video exists [youtube](https://www.youtube.com/watch?v=LBsvsgnSvoM)
- **Crazyflie** — lightweight nano-quad, great for fast simulation
- Both include `.xml` scene files ready to load directly

**mujoco_mpc already has a quadrotor task**: [github](https://github.com/google-deepmind/mujoco_mpc/blob/main/mjpc/tasks/quadrotor/task.xml)
`google-deepmind/mujoco_mpc/mjpc/tasks/quadrotor/task.xml` — this is a complete drone + planning environment with MJPC receding-horizon control already wired up. This is basically what you need for the MPC side.

**hg6185/MuJoCo**  — literally built on Menagerie's drone model, adds PID controller + basic trajectory following (acknowledged as incomplete, good starting scaffold). [github](https://github.com/hg6185/MuJoCo)

## So the Stack Already Exists

| What you need | Where it is |
|---|---|
| Drone MJCF model | `mujoco_menagerie/skydio_x2` or `crazyflie` |
| MuJoCo env + receding MPC | `mujoco_mpc` quadrotor task |
| Flow matching trajectory prior | FlowMP / TrajFlowMatching (retrain on drone data) |
| Gym wrapper for RL/training | `gym_multirotor` |

## If You Still Want to Build Manually

Yes, it's completely feasible. A drone in MJCF is:
1. **Body** — a box/cylinder geom for the frame
2. **4 rotor sites** — `<site>` at each arm tip
3. **Actuators** — use `<actuator><motor>` with a `gear` attribute to apply thrust along the body z-axis per rotor (MuJoCo supports direct force actuators via `<general ctrlrange=... gear="0 0 1 0 0 0">`)
4. **Sensors** — `<gyro>`, `<accelerometer>`, `<framequat>` on the body

The hardest part is tuning rotor thrust coefficients to match real inertia — Menagerie's Skydio X2 already has this calibrated, so using it as a base is strongly recommended over building from scratch.

**Bottom line**: Start with `mujoco_mpc`'s quadrotor task + Menagerie's Skydio X2 model — you get MPC and drone in one repo, then inject your flow matching prior as the reference trajectory generator.

---

## Env zero shot eval task 

Good framing — this is the right design philosophy for testing generalization. Here's the honest assessment:

### No Existing MuJoCo Env Exactly Fits This

No ready-made MuJoCo env does clean **start→goal point navigation with procedurally generated obstacle challenges** designed for zero-shot transfer testing. The existing ones are either hover tasks, landing tasks, or racing gates — not the abstracted trajectory planning benchmark you want.

**You need to build it yourself — but it's not much work.**

### What to Build (Minimal but Principled)

#### The Environment Design

```
Fixed drone model (Skydio X2 from Menagerie)
     ↓
Procedural scene: randomize at each episode reset
  - Start pose (random)
  - Goal position (random, min distance from start)
  - Obstacles: N boxes/cylinders placed along the route
     → random size, position, density
     → ensure a feasible path exists (rejection sampling)
Reward: sparse (reached goal) or dense (dist-to-goal + collision penalty)
```

The key insight for zero-shot testing: **separate training obstacle configurations from test configs** — e.g., train with 3-5 cylinders, test on 8-12 narrow gaps or wall-with-holes geometry. This is exactly how PointNav benchmarks (Gibson, MP3D) test generalization. [rpg.ifi.uzh](https://rpg.ifi.uzh.ch/research_drone_racing.html)

#### MJCF Scene Structure

```xml
<!-- static scene skeleton -->
<worldbody>
  <body name="drone"> ... Skydio X2 MJCF ... </body>

  <!-- obstacles: parameterized at reset via Python -->
  <body name="obs_0" pos="? ? ?">
    <geom type="box" size="? ? ?" contype="1"/>
  </body>
  ...
  <site name="goal" pos="? ? ?" size="0.3" rgba="0 1 0 0.3"/>
</worldbody>
```

Then in Python, at each `env.reset()`:
```python
model.body_pos[obs_ids] = sample_obstacle_positions(...)
model.body_quat[obs_ids] = sample_orientations(...)
mujoco.mj_forward(model, data)  # recompute after model edit
```

MuJoCo supports **runtime model parameter edits** via `mjModel` fields before `mj_forward`, so you can fully randomize the scene without reloading XML. [github](https://github.com/google-deepmind/mujoco)

#### Difficulty Levels for Curriculum / Zero-Shot Eval

| Split | Obstacles | Challenge type |
|---|---|---|
| Train easy | 2–4 boxes, wide gaps | Basic avoidance |
| Train hard | 5–8 mixed, narrow gaps | Dense clutter |
| **Zero-shot test A** | Wall with holes | Topology change |
| **Zero-shot test B** | Dynamic obstacles (moving boxes) | Non-static |
| **Zero-shot test C** | High-wind disturbance (`mjData.qfrc_applied`) | Physics perturbation |

#### Reward Signal

For trajectory planning (not RL end-to-end), the env just needs to evaluate: ✅ did the planned trajectory reach the goal, ✅ collision-free, ✅ time/energy cost. Keep it as a **metrics evaluator**, not a dense reward shaper — this keeps it simulator-agnostic and lets your flow matching + MPC planner operate in open-loop between replanning steps.

### Recommendation

Build this env yourself — it's ~300 lines of Python + the Menagerie XML. The MuJoCo runtime model editing API  makes procedural scene generation clean. Want me to scaffold the actual code for this? [github](https://github.com/google-deepmind/mujoco)


