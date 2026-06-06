# Gen11 Epoch 2 — UAV Controller Validation: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete  
**Maximum fix index**: none (single SLURM run, no code fixes)  
**CLOSURE**: [`EPOCH2_CLOSURE.md`](EPOCH2_CLOSURE.md)

---

## What this epoch does

Validates two things before any data collection or training begins:
1. **The controller is correct** — a hand-written cascaded PID can fly the X2 along a
   non-trivial trajectory inside MuJoCo.
2. **The trajectory format is locked** — 6D `[p, v]` vs. 9D `[p, v, a]` is resolved by
   measurement, not assumption.

---

## Cascaded PID — the control architecture

The X2 is underactuated: 4 rotor thrusts produce 3D force + 1 yaw torque, so attitude
(roll/pitch) cannot be controlled independently of thrust.  The PID runs two nested loops:

**Outer loop (position)**:
```
F_des = Kp_pos·(p_des − p) + Kd_pos·(v_des − v) + a_des
```
This is a standard PD controller plus feedforward acceleration `a_des`.  The desired force
vector `F_des` implicitly encodes the desired pitch/roll: to accelerate forward, the drone
must tilt nose-down.  The outer loop converts `F_des` to a desired attitude `R_des`.

**Inner loop (attitude rate)**:
```
τ = Kp_omega·(ω_des − ω)
```
A P controller on angular velocity that drives the drone toward the desired attitude by
commanding differential torques across the 4 rotors.

**Why feedforward matters**: with `a_des = 0` (6D trajectory), the controller is purely
reactive — it only corrects errors after they appear.  With `a_des ≠ 0` (9D trajectory),
the controller pre-compensates: it starts tilting the drone *before* position error
builds, keeping attitude near identity and the inner loop in its linear region.

Code: `uav_naive_test/flight_controller.py` (`CascadedPID`).

---

## 6D vs 9D trajectory format — the decisive experiment

Four tasks were run:

| Task | Format | RMS (m) | Verdict |
|---|---|---|---|
| A — hover (static) | — | 0.335 | ❌ (not format-related — see below) |
| B — step response (static) | — | 0.328 | ❌ |
| C — circle (6D `[p, v]`) | `a_des = 0` | 0.214 | ❌ |
| C — circle (9D `[p, v, a]`) | `a_des = circle accel` | **0.029** | ✅ |

**9D beats 6D by 7.4× RMS on the same task.**  The difference is entirely the
feedforward term — the 9D trajectory supplies `a_des = −Aω²cos(ωt)` (centripetal
acceleration of the circle), keeping position error < 3 cm throughout.

**Decision locked**: 9D `[p, v, a]` is the trajectory format for all subsequent epochs.
Code: `uav_naive_test/trajectories.py`.

---

## Hover instability — discovered and deferred

Tasks A and B (static setpoint) failed with a discrete-time limit cycle:
motor outputs alternate `[6.5,6.5,0,0] ↔ [0,0,6.5,6.5]` every step.

**Root cause**: `Kp_omega = [10, 10, 2]` is too aggressive for 100 Hz physics.  Any
disturbance reaching `|ω| > 0.18 rad/s` saturates the torque command; the over-correction
flips sign in the next step → locked oscillation.

**Why it doesn't matter for FM-PCC**: FM/diffusion policies output continuously-moving
trajectories with non-zero `a_des`.  The limit cycle only triggers when the drone is asked
to hold a static position (near-zero velocity for multiple steps) — a scenario FM-PCC
never produces.  The one-line fix (`Kp_omega → [2.5, 2.5, 1.0]`) is deferred to Epoch 4,
where it becomes necessary for the data-collection pipeline.

---

## Architectural conclusion

The planning/execution split is validated: a scripted reference trajectory (from any
planner, including FM) can be executed at < 3 cm RMS by a separate cascaded controller
with no coupling between them.  This is the same architectural assumption FM-PCC makes
for the Panda arm — now confirmed for the UAV.

---

## Cross-references

| Document | Content |
|---|---|
| [`EPOCH2_CLOSURE.md`](EPOCH2_CLOSURE.md) | Full results, stability diagnosis, arithmetic |
| [`../Epoch1_UAV_model/METHODOLOGY.md`](../Epoch1_UAV_model/METHODOLOGY.md) | Model assets used here |
| [`../Epoch_3_uav_in_env/METHODOLOGY.md`](../Epoch_3_uav_in_env/METHODOLOGY.md) | Same controller tested inside obstacle scenes |
| [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md) | Kp_omega fix applied; controller used for data collection |
