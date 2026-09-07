# AUX — the UAV control stack: PID, MJPC, and the plan → setpoint → thrust chain

**Thesis home:** `sec:method:uav` · **TARGET** §5.2
**Primary dev logs:** `Gen11/Epoch8_UAV_Mjpc_thrust_control/` (`DESIGN_obs_action_loop.md`,
`DESIGN_dataset_pid_vs_mjpc.md`, `DESIGN_code_flow_and_math.md`, `PLAN_MJPC_Thrust_Control.md`,
`U2_PID_Stop_Go/`, `U3_v_des_Patch/`, `U6_rebuild_mujoco_MPC/`) ·
`Gen11/Epoch7_.../U4_cond/DESIGN_control_chain_arm_vs_UAV.md`

---

## 1. What the thesis must say

This is the section that answers the obvious referee question: **"a flow model does not fly a
quadrotor — what closes the loop?"** The answer is a three-layer chain, and the important structural
point is that **the generative model sits at the top of it and never sees a motor command**.

```
   FM / MeanFlow plan  ──►  projector (DPCC or HardFlow)  ──►  p_des setpoint
                                                                    │
                                                      cascaded PID (position → attitude → thrust)
                                                                    │
                                                          data.ctrl[:4]  ──►  MuJoCo X2
```

Three things must be stated explicitly or the UAV results cannot be read:

1. **The network's entire output is `Δp_des` — a 3-D position delta.** It never emits velocity,
   attitude, or rotor thrust. Everything below the setpoint is derived by classical control.
2. **`p_des` is a free-running accumulation of the model's own past outputs**, not an environment
   observation: `p_des ← init_pos`, then `p_des ← p_des + Δp_des` each step, with the *previous*
   `p_des` fed back into the observation `obs = [p_des | p | v]` (9-D). The plan is therefore an
   open integrator driven by the model — errors accumulate, which is exactly why the divergence
   aborts of Gen15 (Fix_16) mattered.
3. **This mirrors the manipulator chain rather than departing from it.** On `avoiding-d3il` the arm
   task does the same thing one layer down: FM emits `Δ(x,y)`, the eval adds it to the current
   observation to get an absolute Cartesian setpoint, and IK closes the loop inside every step. Both
   embodiments are **receding-horizon: the generative model replans every timestep.** Saying this
   once, for both embodiments, is what makes the transfer claim structural rather than anecdotal.

---

## 2. The two low-level controllers, and why both exist

| | **Cascaded PID** | **MJPC (MuJoCo MPC)** |
|---|---|---|
| role | the shipped controller — expert data collection *and* evaluation | an alternative thrust-level controller, explored in Epoch 8 |
| interface | `pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des) → u`, `data.ctrl[:4] = u` | full thrust-level optimisation against an MJPC task |
| rate | **100 Hz** control loop, one `mj_step` per iteration | per its own planner budget |
| status | ✅ used for every dataset and every reported UAV result | 🟡 built and studied; **not** the source of the reported data |

**The decision to record:** `DESIGN_dataset_pid_vs_mjpc.md` was written to answer whether the
dataset should be re-collected with MJPC. The conclusion was that it need not be — which is why every
UAV number in the thesis rests on PID-collected data. The thesis should state this as a deliberate
scoping decision with its reason, not leave MJPC as an unexplained loose end.

### Sub-threads worth a sentence each

- **`v_des` provenance** (`U3_v_des_Patch/PLAN_pid_const_v.md`) — the PID needs a velocity setpoint
  the plan does not supply. How `v_des` is synthesised (constant-speed assumption) is a real modelling
  choice and a candidate threat-to-validity.
- **Stop-and-go** (`U2_PID_Stop_Go/`) — a setpoint regime where the drone is allowed to arrest between
  waypoints. It appears in run tags as `pid_stopgo`; readers will see the token in folder names.
- **The ODE-step warning** (`Fix_4/WARNING_1000_ODE_steps.md`) — a documented failure mode when the
  sampler's step count and the control rate are mismatched.
- **Sim-vs-real latency** (`DESIGN_sim_vs_irl_latency.md`) — the honest framing for any timing claim.

---

## 3. 🔴 The `budget_ms` / 33 Hz trap

Folder tags and eval logs print a `budget=30.3ms → real_time_OVER×N` line. **This is a data-rate
artefact, not a real-time criterion** (TARGET §6.5): the budget is derived from the expert data's
sample rate, and the measured times include cluster-shared-GPU latency. It must never be reported as
a real-time pass/fail. If the thesis wants a latency claim on UAV it needs a separate,
properly-instrumented measurement.

---

## 4. Parameters of record

| item | value | source |
|---|---|---|
| control rate | 100 Hz, one `mj_step` per PID iteration | `DESIGN_dataset_pid_vs_mjpc.md` §1.2 |
| network output | `Δp_des` ∈ ℝ³ | `DESIGN_obs_action_loop.md` §1 |
| observation | `obs = [p_des(3) \| p(3) \| v(3)]`, 9-D | `DESIGN_obs_action_loop.md` §2 |
| setpoint recursion | `p_des ← p_des + Δp_des`, initialised to `init_pos` | `DESIGN_obs_action_loop.md` §2 |
| PID signature | `compute(p, q, v, om, p_des, v_des, a_des, yaw_des)` | `DESIGN_dataset_pid_vs_mjpc.md` §1.2 |
| actuation | `data.ctrl[:4]`, 4 rotors | ibid. |
| conditioning mode | `cond_mode='pos_only'` for the Gen15 UAV runs | eval logs; `DESIGN_dataset_pid_vs_mjpc.md` |

---

## 5. Holes — fill before writing

- [ ] **The PID gains are nowhere in the logs.** Position/velocity/attitude/rate gains, and the
      cascade structure (which loop runs at which rate) must be read out of the `CascadedPID`
      implementation and tabulated. Without this the controller is not reproducible.
- [ ] **`v_des` synthesis is described as a plan, not as shipped.** Confirm what the final code does.
- [ ] **MJPC's final status is ambiguous across logs** — `U6_rebuild_mujoco_MPC` suggests a rebuild
      landed. Establish whether MJPC is (a) abandoned, (b) a comparison controller, or (c) used
      anywhere in a reported number, and say so once.
- [ ] **Yaw handling** — `yaw_des` is in the PID signature but the trajectory layer's yaw policy is
      not documented.
- [ ] **The divergence-abort criterion** ("body z-axis · world z < 0 — the drone is upside down") is
      an eval-side safety gate, documented in Gen15, not here. Cross-reference; do not restate.
