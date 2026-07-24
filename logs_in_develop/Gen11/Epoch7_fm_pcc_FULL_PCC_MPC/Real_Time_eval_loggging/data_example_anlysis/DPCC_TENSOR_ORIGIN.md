# DPCC Tensor Origin: Is It Pure D3IL?

**Question**: Does the DPCC paper actually *design* its state/action tensor, or did it inherit the format from D3IL? How does the paper "legalize" calling position deltas "velocities" with dt=1 Euler dynamics?

**Short answer**: YES — the entire tensor is pure D3IL hardware data. DPCC found the data format and wrote a mathematical formalization (Euler + ts=1 + w_t) that is consistent with D3IL's implicit structure. It designed nothing; it rationalized everything.

---

## 1. Where the data comes from: D3IL hardware logger

D3IL uses a Franka Panda arm controlled by an IK position-tracking controller
(`cartesianPosQuatTrackingController`). The human expert moves the robot via gamepad. The
logger records TWO signals at every control step:

| Signal | What it is |
|---|---|
| `des_c_pos` | Desired Cartesian position — set by human gamepad via `setSetPoint()` |
| `c_pos` | Actual Cartesian position — from FK of measured joint angles |

This is not a mathematical design choice. The hardware simply records the commanded setpoint
AND the measured actual position. Both exist in the logger because the IK controller needs
to track the setpoint, and the human-in-the-loop needs both to understand tracking error.

**Control loop timing** (`avoiding.py:55`, `gym_env_wrapper.py:83-86`):

```
simulation dt  = 0.001 s   (MuJoCo physics step, SimFactory.py:39)
n_substeps     = 35         (policy steps execute 35 physics steps)
control period = 35 × 0.001 = 0.035 s  (~28.6 Hz)
```

So each "action step" in D3IL corresponds to 35ms of real simulation time.

---

## 2. D3IL dataset: position delta is called "vel_state"

`d3il/environments/dataset/avoiding_dataset.py:60`:
```python
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
```

`d3il/environments/dataset/aligning_dataset.py:79`:
```python
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
```

D3IL's OWN dataset code computes `Δdes_c_pos` and names it `vel_state`. Why "vel"?
Robotics convention: velocity = Δposition / Δt. With Δt = 35ms, `vel_state` is nominally
velocity in m/s — but D3IL **never divides by Δt**. The division was dropped, leaving the
raw position delta.

This is not a deliberate lie. It is engineering shorthand: in the control loop, if you apply
a velocity command for one step (35ms), the position change is `vel × 0.035`. If you just
label the position change per step "velocity" and implicitly assume dt=1 step, the math
works out numerically even though the unit (m vs m/s) is wrong.

**Dataset construction** (avoiding_dataset.py:55-68):

```python
robot_des_pos = env_state['robot']['des_c_pos'][:, :2]   # (T, 2) desired xy
robot_c_pos   = env_state['robot']['c_pos'][:, :2]       # (T, 2) actual xy

input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)  # (T, 4) obs
vel_state   = robot_des_pos[1:] - robot_des_pos[:-1]                 # (T-1, 2) action

zero_obs[0, :valid_len, :]    = input_state[:-1]   # obs at time t
zero_action[0, :valid_len, :] = vel_state          # action = Δdes_pos (t→t+1)
```

**State tensor**: `[des_xy | c_xy]` = 4D. This is literally the two raw logger signals.
**Action**: `vel_state = Δdes_xy` = 2D position delta.

---

## 3. D3IL eval loop: explicit delta integration

`d3il/simulation/avoiding_sim.py:89-143`:

```python
# initialize: desired pos starts at current robot pos
pred_xy = env.robot_state()[:2].copy()
fixed_z_q = np.concatenate([env.robot_state()[2:], [0, 1, 0, 0]])

# visual mode (line 106-111)
pred_delta = agent.predict((bp_image, pred_xy, c_xy), if_vision=True)
pred_xy    = pred_delta[0] + pred_xy                   # ← EXPLICIT INTEGRATION
full_action = np.concatenate([pred_xy, fixed_z_q])    # ← absolute 7D pos+quat
env.step(full_action)

# non-visual mode (line 137-143)
obs_concat  = np.concatenate([pred_xy, obs])           # obs = c_xy from env
pred_delta  = agent.predict(obs_concat)
pred_xy     = pred_delta[0] + obs_concat[:2]           # ← EXPLICIT INTEGRATION: new_des = old_des + delta
full_action = np.concatenate([pred_xy, fixed_z_q])
env.step(full_action)
```

**This is the key pattern**: D3IL's own eval loop performs the integration `new_des = old_des + Δ`.
The DDPM agent outputs a 2D position delta; the eval loop adds it to the running `pred_xy`;
and `env.step()` receives the ABSOLUTE desired position.

The controller (`IKControllers.py:96-106`):
```python
def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
    ...
    self.desired_c_pos = desired_pos   # sets ABSOLUTE desired position
```

So the D3IL chain is:
```
DDPM → Δdes_xy → eval loop (pred_xy += delta) → absolute pred_xy → setSetPoint → IK tracks it
```

---

## 4. Gym controllers: the merge conflict that reveals the ambiguity

`d3il/environments/d3il/d3il_sim/gyms/gym_controllers.py:118-130`:

```python
def set_action(self, action):
# <<<<<<< HEAD
    # self.desired_c_pos = np.array(self.robot.current_c_pos) + np.array(action[:3])
    self.desired_c_pos = np.array(action[:3])           # ← HEAD: action IS absolute position
    # ...
# =======
#     self.desired_c_pos = np.array(self.robot.des_c_pos) + np.array(action[:3])  # ← origin/controller_fixes: des + delta
# >>>>>>> origin/controller_fixes
```

And for the XY-only controller (`gym_controllers.py:173-174`):
```python
def set_action(self, action):
    self.desired_c_pos = np.array(self.robot.des_c_pos) + np.array(action[:2])  # ← des + delta
```

**There is a literal unresolved git merge conflict left in the codebase** showing that D3IL's
own developers argued about whether "action" means an absolute position OR a position delta.
The two branches chose differently. The `GymXYVelController` (XY plane only) integrates the
delta internally. `GymCartesianVelController` (3D) now sets absolute position (HEAD branch).

D3IL's codebase is internally inconsistent about this. The ambiguity is real. DPCC inherited
the ambiguity and resolved it with the accumulator pattern (`mental_robot_pos += action`).
FM-PCC UAV inherited the accumulator pattern (`p_des += act`).

---

## 5. How DPCC "legalizes" the velocity label and dt=1

DPCC paper (arXiv:2412.09342) §6.1:

> "The state st ∈ ℝ⁴ consists of the current and desired end-effector positions in the 2D
> plane, and the action at ∈ ℝ² contains the **desired Cartesian velocities**, which are
> sent to a low-level controller."

> "st+1 = st + [at^T, at^T]^T · ts + wt"     (ts = timestep = 1)

**The paper's move**:

| Step | What happens |
|---|---|
| 1 | D3IL already labeled position deltas as "vel_state" |
| 2 | DPCC inherits the label: "desired Cartesian velocities" |
| 3 | DPCC writes Euler: `s_{t+1} = s_t + [a,a]^T · ts` |
| 4 | Sets ts = 1.0 |
| 5 | Now: velocity × 1 = position delta numerically |
| 6 | The Euler formula is tautologically correct for `des_c_pos`: `des_c_pos[t+1] = des_c_pos[t] + a` |
| 7 | And *approximately* correct for `c_pos`: `c_pos[t+1] ≈ c_pos[t] + a + w_t` (with controller lag = w_t) |

**ts=1 is NOT the physical control period (35ms).** It is a normalization choice: the paper
absorbs the 0.035s control period into the action magnitude and declares ts=1. Numerically
this makes "velocity" (m/s × step) equal to "position delta" (m), which is fine because
the IK controller is POSITION-CONTROLLED anyway.

**Why ts=1 is internally consistent for des_c_pos:**

```
des_c_pos[t+1] = des_c_pos[t] + a_t    (exact — des_c_pos IS the Euler integrator by construction)
```

This requires NO approximation. `des_c_pos` is the desired position setpoint; the action IS
the increment. ts=1 because "action = increment per step" by definition of how the logger
records it. This is a tautology, not a physics model.

**Why ts=1 is an approximation for c_pos:**

```
c_pos[t+1] ≈ c_pos[t] + a_t + w_t     (approximate — inner controller has lag, overshoot, w_t ≠ 0)
```

The paper's quote: "We do not assume knowledge of the dynamics of the low-level controller"
is the honest acknowledgment that `w_t` is unknown. Including `des_c_pos` in state (alongside
`c_pos`) is justified precisely because `des_c_pos` IS the exact integrator that the
approximate `c_pos` is trying to track.

---

## 6. The full inheritance chain

```
Hardware logger
│  records: [des_c_pos | c_pos] at each control step (both signals always available)
│
├── D3IL dataset code
│   vel_state = des_c_pos[t+1] - des_c_pos[t]          # position delta, named "velocity"
│   obs       = [des_c_pos | c_pos]                     # 4D (avoiding) / 23D (aligning)
│   action    = vel_state = Δdes_c_pos                  # 2D (avoiding) / 3D (aligning)
│
├── D3IL eval loop (avoiding_sim.py)
│   pred_xy += agent.predict(obs)[:2]                   # EXPLICIT delta integration
│   env.step(pred_xy_absolute)                          # passes absolute pos to controller
│
├── D3IL gym_controllers.py
│   GymXYVelController: des_c_pos = robot.des_c_pos + action[:2]   # delta integration IN controller
│   GymCartesianVelController: des_c_pos = action[:3]              # HEAD branch: absolute (merge conflict!)
│
├── DPCC paper / codebase
│   obs:    [des_c_pos(3) | c_pos(3)]                  # 6D (reduced from 23D via FiLM)
│   action: Δdes_c_pos(3)                               # inherited from D3IL vel_state
│   tensor: [act(3) | des_c_pos(3) | c_pos(3)]        # 9D (Janner format: [action|obs])
│   paper:  "desired Cartesian velocities", ts=1       # formalization of D3IL's implicit structure
│   eval:   mental_robot_pos += action                  # IMPLICIT delta integration (accumulator)
│
└── FM-PCC UAV
    obs:    [p_des(3) | p(3) | v(3)]                   # 9D (added velocity, no FiLM)
    action: Δp_des(3)                                   # same pattern
    tensor: [act(3) | p_des(3) | p(3) | v(3)]         # 12D
    eval:   p_des += act                                # IMPLICIT delta integration (accumulator)
    config: dt=1.0 because "actions are position deltas, NOT velocities"  # admits the truth
```

---

## 7. What DPCC actually designed vs what it inherited

| Component | Origin |
|---|---|
| `[des_c_pos \| c_pos]` state format | **D3IL hardware** (logger records both) |
| action = Δdes_c_pos | **D3IL dataset code** (vel_state = position delta) |
| "velocity" label for action | **D3IL naming** (vel_state convention) |
| Delta integration in eval | **D3IL avoiding_sim.py:140** (pred_xy += delta) |
| ts=1 Euler formalization | **DPCC** (formalizing D3IL's implicit structure) |
| w_t model mismatch term | **DPCC** (honestly acknowledging c_pos approximation) |
| 23D → 6D obs reduction | **DPCC** (via FiLM; needed to drop goal/box from obs) |
| [act\|obs] trajectory tensor | **Janner** (DPCC grafted Janner's format onto D3IL data) |
| PCC projector | **DPCC novelty** |
| FiLM visual conditioning | **DPCC novelty** |

**DPCC designed**: dimension reduction via FiLM, PCC projector, Euler ts=1 formalization,
Janner trajectory graft. It did NOT design the state format, action format, "velocity"
label, or integration pattern. Those are all D3IL hardware data artifacts.

---

## 8. The self-referential loop: also pure D3IL

The most dangerous property — `des_c_pos` appears in the observation while being the
quantity the FM is computing — is also implicit in D3IL's own design:

```
D3IL avoiding_sim.py:137:   obs_concat = np.concatenate([pred_xy, obs])
                                                           ^^^^^^^^
                                                           This is pred_xy = old desired pos
                                                           NOT env.robot_state()[:2]
```

The non-visual obs_concat starts with `pred_xy` (the running desired position maintained by
the eval loop), not the raw environment observation. So D3IL's own eval loop already
feeds the agent's own accumulated desired position as part of the observation — the same
self-referential structure that FM-PCC inherited.

For visual mode (`avoiding_sim.py:107`): `agent.predict((bp_image, pred_xy.copy(), c_xy))` —
`pred_xy` (accumulated desired position) is passed explicitly as the second argument.

**D3IL was already self-referential.** DPCC didn't invent it; DPCC just formalized it inside
the eval class as `mental_robot_pos` rather than in the eval loop as `pred_xy`.

---

## 9. Summary: three things the DPCC paper "legalizes" and how

| Claim | D3IL source | DPCC "legalization" | Is it honest? |
|---|---|---|---|
| Action = "desired Cartesian velocity" | D3IL `vel_state` = Δdes_c_pos | Euler: velocity × ts = Δpos, ts=1 | Technically consistent; semantically misleading |
| dt=1 (timestep) | D3IL: no explicit dt in action (just Δpos per step) | Paper: absorb control period into action magnitude, declare ts=1 | Internally consistent; ts=1 is not the 35ms physical period |
| Conditioning on `des_c_pos` (not `c_pos`) | D3IL: hardware records both; eval loop uses `pred_xy` (desired) | Paper: "Get current state st" (one line, no detail) | Never explicitly stated; justified only by: "We do not assume knowledge of inner controller dynamics" |

The paper's §6.1 is three paragraphs that retroactively formalize what D3IL's hardware data
format and dataset code already implied. The Euler dynamics with ts=1 is not a model that
DPCC derived from physics — it is the exact formula that `vel_state = Δdes_c_pos` produces
when you write it as an integration step and set ts=1.

The format is D3IL. The formalization is DPCC. The confusion is inherited by everyone downstream.

---

*Generated 2026-06-26. Companion to WHY_FM_KEEPS_PLANNING.md (§10-§14). Code citations verified against current repo state.*
