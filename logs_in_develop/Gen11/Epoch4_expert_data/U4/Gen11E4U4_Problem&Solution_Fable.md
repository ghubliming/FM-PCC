# Gen11E4U4 — Problem & Solution

**Date:** 2026-06-10 · **Scope:** s_curve expert-data collection — 100% rejection after hover-pause fix (F4)
**Inputs:** `U4/{EVAL.md, FIX_PLAN.md, CHANGELOG.md, F4_RESULTS.md}`, `temp/Gen11E4U3/F4/`, controller + trajectory + scene code audit
**Principle applied:** code/robotic-logic bugs first.

---

## Problem

F4 run (job 21400): s_curve **0 saved / 21 rejected → ABORT**, worse than F3's 90.5%. The U4 Fix A (1.0 s hover pauses at segment junctions + duration [18,24] s) did not help. Empty/corridor are clean; pillars works at 20.8% rejection (R_R_R recovered by Fix B).

## Why the hover fix could not work (wrong location)

FIX_PLAN's own tracking table (episode 0000010) shows the divergence starts **mid-Seg-B** (steps 310→381), during the cosine profile's acceleration→deceleration reversal around peak lateral velocity — **not at the junction**. `traverse_line` already has v=0, a=0 at both endpoints (`uav_env_test/trajectories.py:65-89`), so the junction was never the discontinuity. Hovering before the diagonal zeroes the entry velocity, which was already ~0. The 90.5%→100% delta on a 21-trial abort window is statistical noise — the fix changed nothing material.

## Actual root cause (robotic logic): attitude-loop lag → overshoot → motor saturation → altitude collapse

Physics check that rules out "trajectory too aggressive": Seg B commands peak accel ≈ 0.46 m/s² → required tilt ≈ 2.7° → cos-loss ≈ 0.1% thrust. A correctly tracking quad loses **no** altitude here. Yet the data shows a 0.40 m z-collapse. The mechanism, from `uav_env_test/flight_controller.py` (`CascadedPID`):

1. The attitude inner loop takes time to tilt; during that lag the drone falls behind the commanded y (err_y −0.185 m at step 330).
2. The position loop (`Kp_pos=4, Kd_pos=3`) adds up to ~1–2 m/s² corrective accel on top of feed-forward; when the cosine profile reverses sign, the accumulated momentum overshoots (+0.84 m swing at 1.41 m/s — far above the 0.6 m/s command).
3. Arresting the overshoot demands large tilt + large attitude torques. Per-motor command clips at **`u_max = 2.0 × u_hover`** (`flight_controller.py:61`). Clipping a saturated wrench corrupts the **collective thrust** component → z falls 0.3–0.45 m → with start z as low as 0.70 m and `Z_FLOOR_MARGIN=0.50`, the floor check rejects.
4. Same controller signature appears in pillars: 6.6 m/s end-of-episode spikes (Fix C observation). One root cause, two symptoms.

## Secondary findings

- **Rejection reason is not instrumented** (`generator.py run_trial`): both the `contact_frac > 0.08` and `min_z < 0.50` rejects `return None` indistinguishably. We cannot confirm from logs which check fired in F4. This is a real code gap that has cost multiple debug cycles.
- The hover points sit exactly on the wall end-face plane x=±0.5 with only 0.14 m lateral margin (corridor half-width 0.45 m inner, rotor reach 0.31 m); an overshooting entry into corridor 2 (y > +1.25) can also hit `seg2_wall_pos`. Risky placement even if not the trigger.

## Solution (for the implementing agent, in order)

1. **Instrument first (decisive, ~20 lines):** make `run_trial` return/record the reject reason + `min_z` + `max_contact_frac` + per-step motor-clip fraction; have `collect.py` print a reject histogram. One 20-trial smoke run then tells us exactly which check fires and at what trajectory time. Do this before any further trajectory tuning.
2. **Altitude headroom (1 line):** s_curve start `z ~ U(0.70, 1.10)` → `U(0.90, 1.30)` in `generator.py _build_traj_and_init`. The observed dip is 0.30–0.45 m; +0.20 m floor headroom converts most min_z rejects into accepts. (Walls are 1.5 m tall — still inside the corridor.)
3. **Slow the diagonal (1 line):** give Seg B a disproportionate time budget (e.g. weight `d_b` by 2× in the allocation, keeping total T). Halving peak accel/velocity quadratically shrinks the overshoot that triggers saturation.
4. **Controller fix (the principled one, benefits pillars too):** thrust-priority allocation in `CascadedPID.compute` — when `u = M_inv @ wrench` exceeds limits, scale down the torque components and preserve collective thrust before clipping (or simply raise `u_max` 2.0→2.6×hover). This removes the altitude-collapse mechanism itself, and should also kill the pillars end-of-episode speed spikes (Fix C closes for free).
5. **Remove or relocate the hover pauses** once 1–4 land: they don't address the mechanism and they park the drone at the wall end-face plane. If junction settling is still wanted, hover at x=∓0.7 (inside the corridor), not at ±0.5.

**Validation gate:** 20-trial smoke with (1)+(2) → expect rejection well under 60%; add (3)/(4) if still above 30%. Then full 500.
