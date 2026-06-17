# MjRobot.py — Code Logic, Math, and Critique

**File:** `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py`  
**Class:** `MjRobot(RobotBase, MjIncludeTemplate)`  
**Robot target:** Franka Emika Panda (7-DoF arm + 2-finger gripper)

---

## 0. Is This Code Used by Our Avoiding Task?

**Yes — directly.** `avoiding.py` imports `MjRobot` at line 9 and instantiates it:

```python
robot = MjRobot(
    scene,
    xml_path=d3il_path("./models/mj/robot/panda_rod_invisible.xml")
)
```

The rod-tip variant (`panda_rod_invisible.xml`) replaces the standard gripper with a passive rod — relevant for the pushing/avoiding geometry.

**What avoiding actually reads from MjRobot:**

| Field / Method | Where used | Source in MjRobot |
|---|---|---|
| `robot.current_c_pos[:2]` | Observation (x,y TCP) | `receiveState` → `_localize_cart_coords` |
| `robot.current_c_pos[0/1]` | Step reward + goal check | same |
| `robot.gotoCartPosQuatController` | Reset: plan trajectory to home pose | inherited Cartesian IK controller |
| `robot.gotoCartPositionAndQuat(...)` | Reset: execute planned trajectory | uses Jacobian at **current q** only |
| `robot.beam_to_joint_pos(qpos)` | Reset: teleport to joint config | `set_q` |
| `robot.init_qpos/init_tcp_pos/init_tcp_quat` | Reset bookkeeping | attributes set at init |

**Critical consequence for the bugs found in §3:**
- `current_c_quat_vel` **(Bug 3.1)** — **never read** by avoiding. The wrong quaternion derivative is computed every step and immediately discarded.
- Jacobian at arbitrary q **(Bug 3.2)** — **never triggered**. The Cartesian IK controller queries the Jacobian at the current simulation state only (no `q` argument), so the state-corruption bug is dormant.
- FK at arbitrary q **(Bug 3.3)** — **never triggered** for the same reason.

**Bottom line:** For the avoiding task, `MjRobot` is safe to use as-is. The three high-severity bugs are all in paths that avoiding never exercises. The live hot path (2D TCP position readout → Cartesian controller → joint teleportation) is correct.

---

## 1. Architecture Overview

`MjRobot` is the MuJoCo-specific concrete implementation of a robot in the d3il simulation stack. It sits at the intersection of three concerns:

```
RobotBase          ← abstract robot state machine (joint/cart state, controllers)
MjIncludeTemplate  ← XML template modification for multi-robot scene assembly
MjRobot            ← bridges MuJoCo data API ↔ RobotBase interface
```

The lifecycle is:

```
Scene construction:
  modify_template(et)   → rewrite XML namespaces, write temp file, collect joint_names
  (scene loads XML, MuJoCo allocates model/data)

Per-step execution:
  prepare_step()
    ├─ activeController.getControl(self)   → raw command
    ├─ preprocessCommand(command)          → gravity comp, clipping → self.uff
    ├─ data.ctrl[joint_act_indices] = uff  → write to MuJoCo actuator buffer
    └─ receiveState()                      → read back all state
```

State reads and model queries (FK, Jacobian) are separate on-demand operations.

---

## 2. Component-by-Component Logic + Math

### 2.1 Multi-Robot Namespacing (`add_id2model_key` + `modify_template`)

MuJoCo XML requires globally unique names. For multi-robot scenes, each robot's XML is post-processed to inject a robot-id into every named attribute.

**Naming convention:**
```
"tcp"          → "tcp_rb0"      (robot 0)
"panda_joint1" → "panda_rb1_joint1"  (robot 1)
```
The id is inserted at position 1 of the `_`-split:
```python
attrib_split = "panda_joint1".split("_")   # ["panda", "joint1"]
attrib_split.insert(1, "rb0")              # ["panda", "rb0", "joint1"]
"_".join(...)                              # "panda_rb0_joint1"
```
`modify_template` walks every XML node and rewrites 8 attribute types (`name`, `joint`, `class`, `body1/2`, etc.), then injects `base_position/orientation` into the root body and writes to a UUID-named temp file.

**Math:** none; pure string manipulation.

---

### 2.2 Index Initialization (`_init_jnt_indices`)

Maps symbolic joint names → MuJoCo integer indices for both position (`qpos`) and velocity/force (`qvel`, `qacc`) arrays, and for the actuator (`ctrl`) array.

MuJoCo stores q in a flat array; joints are addressed by their `jnt_qposadr`/`jnt_dofadr` offsets (they differ for ball joints but are equal for 1-DoF revolute joints, which all Panda joints are). Actuators are named `<joint_name>_act`.

---

### 2.3 Jacobian (`_getJacobian_internal`)

Returns **J ∈ ℝ^{6×7}** — the body Jacobian of the TCP:

```
J_p ∈ ℝ^{3×7}   positional rows:   ṗ_tcp = J_p · q̇
J_r ∈ ℝ^{3×7}   rotational rows:   ω_tcp = J_r · q̇
```

MuJoCo's `get_body_jacp(tcp)` returns a **3 × nv** matrix where nv = total model DoF. For the Panda model: nv = 9 (7 arm + 2 fingers). The Jacobian is extracted as:
```python
jac[:3, :] = jacp.reshape(3, -1)[:, -9:-2]   # arm DoFs only (skip 2 finger DoFs)
jac[3:, :] = jacr.reshape(3, -1)[:, -9:-2]
```

**Optional q-override:** saves current simulation state, sets `qpos` to the query joint config, calls `mj_kinematics` + `mj_comPos` to propagate FK, reads Jacobian, then restores state.

---

### 2.4 Forward Kinematics (`_getForwardKinematics_internal`)

Same save/set/compute/restore pattern as the Jacobian. Returns:
- `cart_pos ∈ ℝ^3` — TCP position in world frame
- `cart_or ∈ ℝ^4` — TCP orientation as unit quaternion (w, x, y, z)

---

### 2.5 State Readout (`receiveState`)

Called every timestep after writing to `ctrl`. Populates:

| Attribute | Content |
|---|---|
| `current_j_pos/vel` | Joint positions/velocities [7] |
| `current_c_pos_global/vel_global` | TCP Cartesian position/velocity in world frame |
| `current_c_quat_global` | TCP orientation quaternion in world frame |
| `current_c_quat_vel_global` | TCP quaternion time-derivative (see §3.1 for correctness) |
| `current_c_pos/vel/quat/quat_vel` | Same quantities in robot base frame |
| `current_fing_pos/vel` | Finger joint positions/velocities |
| `gripper_width` | Sum of both finger qpos values (total gap) |

**Quaternion kinematics (intended math):**

Given angular velocity ω ∈ ℝ^3 and unit quaternion q = (q_w, q_x, q_y, q_z):

```
dq/dt = 0.5 · q ⊗ [0, ω]   (Hamilton product with pure-quaternion ω)
```

or in the right-hand body-frame convention:
```
dq/dt = 0.5 · [0, ω] ⊗ q
```

**Frame localization:**

`_localize_cart_coords` (from `RobotBase`) computes:
```
p_local = R_base^T · (p_global − base_position)
q_local = q_base^{-1} ⊗ q_global
```

For **velocity**, the translation term drops out (time derivative of a constant), so only the rotation applies: `v_local = R_base^T · v_global`. The code cancels the internal subtraction via a hack: `_localize_cart_coords(v_global + base_position)` → `R_base^T · ((v_global + base_position) − base_position) = R_base^T · v_global`. Correct result, ugly mechanism.

---

### 2.6 Inverse Dynamics (`get_command_from_inverse_dynamics`)

**Full inverse dynamics** (`mj_calc_inv=True`):
```
τ = M(q)·q̈ + C(q,q̇)·q̇ + g(q)
```
Sets `qacc` to the desired acceleration, calls `mujoco.mj_inverse`, reads `qfrc_inverse`.

**Bias-only fallback** (`mj_calc_inv=False`):
```
τ_bias = C(q,q̇)·q̇ + g(q)    (no inertia term)
```
Returns `qfrc_bias` directly. This is the gravity + Coriolis term only — **not** inverse dynamics.

---

### 2.7 Gravity Compensation (`preprocessCommand`)

When `gravity_comp=True` (the default), the inherited `preprocessCommand` adds `qfrc_bias` as a feedforward term to the commanded torques. This gives zero-gravity-drift behaviour at the cost of ignoring the inertia term `M(q)·q̈` — acceptable when accelerations are small (slow motions, quasi-static planning).

---

### 2.8 State Teleportation (`set_q`)

Directly writes joint positions with zero velocity:
```python
qpos[joint_indices] = joint_pos
qvel[joint_vel_indices] = 0
scene.set_state(qpos, qvel)
```
Used for initialization and state resets. Correctly vectorized via index arrays.

---

## 3. Critique and Potential Pitfalls

*(Combines and extends points from the original review.)*

---

### 3.1 [BUG — Math] Quaternion Velocity is Computed Wrong

**Lines 158–161:**
```python
self.current_c_quat_vel_global[1:] = get_body_xvelr(...)  # ω = [ωx, ωy, ωz]
self.current_c_quat_vel_global *= 0.5 * self.current_c_quat_global
```

This is **element-wise multiplication** between two 4-vectors. The correct formula is a **Hamilton product** (quaternion multiplication):

```
dq/dt = 0.5 · q ⊗ [0, ωx, ωy, ωz]
```

The element-wise version produces the wrong quaternion at every timestep. Any downstream code using `current_c_quat_vel` or `current_c_quat_vel_global` (e.g., orientation rate control, trajectory derivatives) will silently compute garbage.

**Fix:** Use a proper quaternion product utility, e.g.:
```python
omega_quat = np.array([0, *get_body_xvelr(...)])
self.current_c_quat_vel_global = 0.5 * quat_multiply(self.current_c_quat_global, omega_quat)
```

---

### 3.2 [BUG — Acknowledged] Jacobian at Arbitrary q Corrupts Simulation State

**Lines 83–106:** The developer's own comment admits this and then leaves without an answer:

```python
# NOTE: Test followin workflow:
#       get jacobian for current simulation step
#       set qpos to a random position
#       call forward kinematics and compos
#       get new jacobian -> should be different to the one from before
#       reset simulation to the state before (with self.scene.sim.set_state
#       calculate again jacobian -> should be the same from the state before.
#       BUT: IT is not!!! why?
```

**The answer to "why?"** — `set_state` only writes `qpos` and `qvel` back into `mjData`. It does **not** recompute the cached derived quantities: body positions (`xpos`, `xquat`), body velocities, or Jacobians. Those only update when `mj_forward()` or `mj_step()` is called next. So after `set_state`, the Jacobian cached in `data` still reflects the last `mj_kinematics` call (the query configuration, not the restored state). The developer was missing one line after `set_state`:

```python
self.scene.sim.set_state(cur_sim_state)
mujoco.mj_forward(self.scene.model, self.scene.sim.data)  # ← missing: recompute derived quantities
```

Without it, `_getJacobian_internal(q=some_q)` leaves the simulation with stale cached kinematics that no longer match the actual `qpos`. Any subsequent step or sensor read computes against wrong body transforms until the next full `mj_step` propagates through.

Calling `_getJacobian_internal(q=some_q)` **silently corrupts** `scene.sim` state. Any code that queries the Jacobian at a non-current configuration (e.g., a gradient-based IK solver, a replanning step) will experience physics state drift with no error thrown.

**For avoiding:** this path is never triggered (see §0), so the bug is dormant.

---

### 3.3 [BUG — API Mismatch] FK Method Mixes mujoco-py and MuJoCo 3 APIs

**Lines 112–122** in `_getForwardKinematics_internal`:
```python
self.scene.data.qpos[qpos_idx] = q          # MuJoCo 3 ✓
mujoco.mj_kinematics(self.scene.model, self.scene.data)  # MuJoCo 3 ✓
cart_pos = self.scene.data.get_body_xpos(tcp_id)  # MuJoCo 3 ✓
self.scene.sim.set_state(cur_sim_state)     # mujoco-py ✗
```

`scene.sim` is a `mujoco-py` object; `scene.data` is a MuJoCo 3 object. Calling `sim.set_state` on a MuJoCo 3 data object will either crash or silently do nothing. This method is **broken** for query-at-q usage. The `_getJacobian_internal` equivalent correctly uses `scene.sim` throughout (older mujoco-py style), while this method was partially ported.

---

### 3.4 [PITFALL] `get_command_from_inverse_dynamics` is Misleadingly Named

**Lines 186–198:** The `mj_calc_inv=False` branch returns `qfrc_bias` (gravity + Coriolis) and calls it "inverse dynamics". This is **not** inverse dynamics — the inertia term `M(q)·q̈` is missing. Callers expecting full inverse dynamics will get incorrect feedforward commands, especially at high accelerations. The method name should be `get_gravity_coriolis_bias` when `mj_calc_inv=False`.

---

### 3.5 [PITFALL] Jacobian Column Slicing is Fragile Magic

**Lines 89–93:**
```python
[:, -9:-2]
```

Hard-codes the assumption that:
- total model DoF = exactly 9
- the 7 arm joints are at positions -9 through -3
- the 2 finger joints are at positions -2 and -1

If the XML model ever gains a floating base, a wrist attachment, or a different gripper (e.g., 3 fingers), the slice produces the wrong columns **silently** — no assertion, no name check. This should be replaced by: `[:, self.joint_vel_indices]` using the already-computed velocity DOF address list.

---

### 3.6 [PITFALL] Quaternion Local Velocity Doubly Wrong

**Lines 173–175:**
```python
_, self.current_c_quat_vel = self._localize_cart_coords(
    self.base_position, self.current_c_quat_vel_global
)
```

Two compounding issues:
1. `current_c_quat_vel_global` is already wrong (§3.1).
2. `_localize_cart_coords` applies quaternion frame rotation (`q_base^{-1} ⊗ q_vel_global`). This is only correct if `current_c_quat_vel_global` is itself a quaternion representing a finite rotation — but a quaternion time-derivative `dq/dt` is **not** a unit quaternion and cannot be transformed by simple quaternion product. The correct transform is `dq_local/dt = q_base^{-1} ⊗ dq_global/dt` (linear in q̇, so this does happen to be a linear operation — but the input is wrong).

---

### 3.7 [DESIGN] `receiveState` Does Not Vectorize Joint Reads

**Lines 140–145:**
```python
self.current_j_pos = np.array(
    [self.scene.data.joint(name).qpos.copy() for name in self.joint_names]
).squeeze()
```

Called every simulation timestep, this iterates over 7 joint names with Python overhead. `set_q` (line 264) already has the index arrays (`self.joint_indices`) — the correct pattern is:
```python
self.current_j_pos = self.scene.data.qpos[self.joint_indices].copy()
```
This is 5–10× faster and already used elsewhere in the same file.

---

### 3.8 [DESIGN] Global Robot Counter is Not Thread-Safe

**Lines 24, 58–59:**
```python
GLOBAL_MJ_ROBOT_COUNTER = 0  # class variable

self._mj_robot_id = MjRobot.GLOBAL_MJ_ROBOT_COUNTER
MjRobot.GLOBAL_MJ_ROBOT_COUNTER += 1
```

No lock. If two `MjRobot` objects are constructed concurrently (e.g., in a parallel env pool), both may read the same counter value. Since the robot ID drives XML namespace injection, two robots would share names → MuJoCo XML parse failure or silent aliasing. A `threading.Lock` on `GLOBAL_MJ_ROBOT_COUNTER` or using an `itertools.count` atomic iterator would fix this.

---

### 3.9 [DESIGN] Temp XML Files Never Cleaned Up

**Lines 330–341:** Each `modify_template` call writes a UUID-named XML file:
```
./models/mj/robot/panda_tmp_rb0_<uuid>.xml
```
There is no `__del__`, no context manager teardown, and no registry of created files. Long training runs with many environment resets will accumulate hundreds of stale XML files on disk.

---

### 3.10 [MINOR] Dead Debug Variable

**Line 147:**
```python
test = self.scene.data.body(tcp_name)
```
Assigned, never used. Left from a debugging session. A linter would flag this; CI with `ruff` or `pyflakes` would catch it. Non-critical but signals low code hygiene.

---

## 4. Summary Table

| # | Category | Location | Severity | Impact |
|---|---|---|---|---|
| 3.1 | Bug — Math | `receiveState` L158-161 | **High** | Wrong orientation rate state everywhere |
| 3.2 | Bug — Acknowledged | `_getJacobian_internal` q-override | **High** | Silent simulation state corruption |
| 3.3 | Bug — API | `_getForwardKinematics_internal` L121 | **High** | Method crashes / silently no-ops on q-override |
| 3.4 | Design | `get_command_from_inverse_dynamics` | Medium | Misleading API; wrong feedforward at high accelerations |
| 3.5 | Pitfall | Jacobian slice `[:, -9:-2]` | Medium | Wrong Jacobian columns if model changes |
| 3.6 | Pitfall | `current_c_quat_vel` L173-175 | Medium | Doubly wrong (bad input + questionable transform) |
| 3.7 | Design | `receiveState` joint reads | Low | Unnecessary Python loop at every sim step |
| 3.8 | Design | `GLOBAL_MJ_ROBOT_COUNTER` | Low | Race condition in parallel env construction |
| 3.9 | Design | `modify_template` temp files | Low | Disk bloat over long training runs |
| 3.10 | Minor | `test =` L147 | Negligible | Dead code |

---

## 5. Verdict

This is **functional research-grade boilerplate** — it correctly handles the core path (joint command write → state readout → FK/Jacobian at current config) well enough to run the d3il avoiding-d3il experiments. The bugs (3.1–3.3) only surface when you need:
- Orientation rate control (3.1)
- IK or replanning that queries kinematics at arbitrary q (3.2, 3.3)

For DPCC/FM-PCC usage (proprioceptive state → joint position commands via the `ModelBasedFeedforwardController`), **none of the high-severity bugs are in the hot path**. The joint position/velocity readout is correct; the Jacobian-at-current-q path (used by the tracking controller) is also correct (the bug only triggers on the optional `q` argument). The code is safe to use as-is for the planned experiments, with the caveat that `current_c_quat_vel` should not be trusted.
