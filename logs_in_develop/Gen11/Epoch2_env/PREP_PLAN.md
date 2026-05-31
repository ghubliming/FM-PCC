# Gen11 Epoch 2 — Naive Fly Test & Trajectory-Tracking Sanity Check

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Status**: Preparation only — no execution yet.
**Predecessor**: [`../Epoch1_UAV_model/CHANGELOG.md`](../Epoch1_UAV_model/CHANGELOG.md) (model assets placed, byte-identical to upstream)
**Roadmap context**: [`../path_temp_initial.md`](../path_temp_initial.md) — this epoch precedes path_temp step 2 (obstacle world) and step 3 (expert controller).

---

## 1. Purpose

Validate two things before any obstacle / expert / FM-PCC integration work begins:

1. **Does the X2 model from Epoch 1 actually load and step in MuJoCo on the cluster?** (Local Docker can't test this — it has no Python runtime by project convention.)
2. **Can a hand-specified trajectory be tracked by the X2 under a basic flight controller, with planning and control fully decoupled?** This validates the architectural hypothesis behind everything downstream.

If both pass, Epoch 3 (obstacles + DPCC halfspaces) and Epoch 4 (expert trajectories) become well-founded. If either fails, we discover it now with one Python file instead of after building three layers on top.

---

## 2. The Architectural Hypothesis Being Tested

The downstream plan is to treat the UAV exactly like FM-PCC's visual aligning treats the Panda manipulator:

| | Visual aligning (Panda) | UAV (Skydio X2) — proposed |
|---|---|---|
| Policy output | Trajectory of `[act_3D_velocity \| obs_6D]` (9-D) | Trajectory of `[act_3D_velocity \| obs_13D]` — or 9-D `[p, v, a]` |
| Action semantic | Cartesian velocity target | Cartesian velocity target (or position+velocity+accel) |
| Below the policy | `cartesianPosQuatTrackingController` (D3IL built-in: IK + joint torques) | **Flight controller (to be added)**: maps `(p_des, v_des, a_des) → (u1, u2, u3, u4)` thrust commands |
| Whose responsibility is dynamics? | Controller's — policy doesn't know joint torques | Controller's — policy doesn't know motor thrusts |

**The hypothesis:** with a working flight controller layered between the policy and the X2, the policy never needs to learn flight dynamics. It outputs position-space trajectories; the controller handles attitude, thrust allocation, and underactuation.

**Why this works mathematically (key fact):** Quadrotors are **differentially flat** systems (Mellinger & Kumar 2011, "Minimum Snap Trajectory Generation and Control for Quadrotors"). Given a smooth position trajectory `p(t)` with bounded derivatives up to snap (4th derivative), the entire state `(p, q, v, ω)` and the required motor thrusts `u(t)` can be **analytically** reconstructed. So planning in position-space and executing via an inner controller is not an approximation — it's exact.

This is the standard architecture in production drones (PX4, Crazyflie, every commercial autopilot). The "must design planning and control together" worry does **not** apply to quadrotors. It would apply if we were planning over raw thrust directly, which we are explicitly choosing not to.

---

## 3. Trajectory Format Decision (Important — Affects Epoch 3+)

The flight controller's input richness determines tracking quality. Four options ranked:

| Trajectory output per step | Dim | Pros | Cons | Verdict for Epoch 2 |
|---|---|---|---|---|
| Position only `[x, y, z]` | 3 | Simplest | Sharp velocity discontinuities → controller overshoots | ❌ Reject |
| Position + velocity `[p, v]` | 6 | Smooth tracking | Policy learns more channels | ✅ Acceptable |
| Position + velocity + accel `[p, v, a]` | 9 | Best tracking; exploits flatness directly | Highest channel count | ✅ **Recommended** — matches aligning's 9-D backbone exactly |
| Raw thrust `[u1..u4]` | 4 | Most expressive | Policy must learn flight dynamics; brittle | ❌ Reject (this is the "design together" worst-case) |

**Decision for Epoch 2:** test with both 6-D (`[p, v]`) and 9-D (`[p, v, a]`). The hand-specified test trajectory is small enough that authoring both formats is trivial. We don't lock in a format until we see how the controller tracks each.

---

## 4. Flight Controller Choice

The Menagerie X2 is bare physics — no controller ships with it (unlike D3IL's Panda which ships `cartesianPosQuatTrackingController`). We must write or vendor one. Options:

| Option | What it is | Effort | Tracking quality | Choice for Epoch 2 |
|---|---|---|---|---|
| Cascaded PID (PX4-style) | Position PID → velocity setpoint → attitude PID → rate PID → thrust mixing | ~150 lines Python | Good for moderate maneuvers | ✅ **Start here** — simplest, well-understood |
| Geometric SE(3) controller (Lee et al. 2010) | Single-step computation in rotation-matrix space, exact under flatness | ~100 lines Python | Excellent for aggressive maneuvers | Phase 2 if cascaded PID is limiting |
| Vendor from `gym-pybullet-drones` | Open-source PID/MPC implementations | ~30 min porting | Production-quality | Possible but adds dep |
| MJPC's own planner | Sampling MPC | Already in `mujoco_mpc` | Heavy | ❌ Not the right tool for trajectory tracking |

**Decision:** hand-write a **cascaded PID** in pure Python. ~150 lines, no new dependencies, fully readable. Tune gains by trial-and-error on a hover task. Live next to the test script, not in `d3il/` proper — this is a test fixture, not a library component (yet).

---

## 5. Naive Tasks for Epoch 2

Three tests of increasing aggression. All run on Slurm (Docker has no Python).

### Task A — Hover

- Initial state: drone at `(0, 0, 0.3)`, identity orientation, zero velocity.
- Trajectory: constant position target `(0, 0, 0.5)` for 5 seconds.
- Pass condition: terminal altitude within ±2 cm of target; max horizontal drift < 5 cm; no NaN states.
- What this tests: gravity compensation, vertical PID, controller stability at the trim point.

### Task B — Step to a setpoint

- Initial state: hovering at `(0, 0, 0.5)`.
- Trajectory: step to `(1.0, 0.0, 0.5)` at t=2s, hold.
- Pass condition: settling time < 3 s; overshoot < 20 cm; final position within ±5 cm.
- What this tests: position-error → attitude command path (the underactuation handling).

### Task C — Circular trajectory

- Initial state: hovering at `(0.5, 0, 0.75)`.
- Trajectory: 3 m circle at 0.75 m altitude, period 10 s, traced for 30 s.
- Pass condition: tracking error RMS < 10 cm; no contact with floor/ceiling.
- What this tests: continuous tracking of a non-trivial trajectory — the prototype of what FM policy outputs will look like.

For Task C, **author the trajectory in both 6-D and 9-D formats** (position+velocity, and position+velocity+accel). Compare tracking RMS. Lock the format choice for Epoch 3+ based on the result.

---

## 6. Deliverables

| Artifact | Location | Purpose |
|---|---|---|
| `uav_naive_test/flight_controller.py` | Cascaded PID, ~150 lines | The controller-under-the-policy fixture |
| `uav_naive_test/trajectories.py` | Hand-coded Tasks A/B/C | Test inputs |
| `uav_naive_test/run_naive.py` | Driver: loads X2, runs each task, saves logs + GIF | Smoke harness |
| `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | Submits the harness to a GPU node for offscreen rendering | Cluster runner |
| `logs/uav_naive/` | Output logs, plots, GIFs of each task | Evidence |
| `logs_in_develop/Gen11/Epoch2_env/CHANGELOG.md` | What was created, what passed/failed, controller gains chosen | Closure |

Nothing in `d3il/`, `config/`, `fm_visual_aligning/`, or `diffuser_visual_aligning/` is touched. Epoch 2 is fully under `temp/` + `logs_in_develop/Gen11/Epoch2_env/` + a single SLURM script.

---

## 7. Explicit Non-Goals

- **No learned policy.** Trajectories are hand-coded for Epoch 2.
- **No obstacles.** That's Epoch 3.
- **No expert demonstrator** (no UAV-Flow alignment, no rollout recording). That's Epoch 4.
- **No D3IL env class.** No `MjQuadrotor`, no `gym_quadrotor_env/`. We use raw `mujoco.MjModel` + a thin driver. Wrapping it into a D3IL-style env is a later decision; Epoch 2 doesn't commit to it.
- **No FM-PCC trajectory format integration.** The 6-D / 9-D test is to *inform* the format choice, not to lock the policy stack into it.
- **No DPCC projector touch.** Workspace bounds in 3D for a quadrotor are a separate plumbing question (Epoch 3+).

---

## 8. Pass / Fail Criteria for Epoch 2 as a Whole

Epoch 2 is a **success** if:

1. ✅ X2 model loads in MuJoCo on Slurm (raw `mj_step` smoke test passes — finally executes the Step 5 from Epoch 1 §11.4 that we deferred).
2. ✅ Cascaded PID achieves Task A pass condition (hover within ±2 cm).
3. ✅ Task B pass condition (step settles cleanly).
4. ✅ Task C pass condition (circular tracking RMS < 10 cm in at least one trajectory format).
5. ✅ A trajectory format (6-D or 9-D) is selected for Epoch 3 with a recorded rationale.

Epoch 2 is a **partial success** (proceed cautiously) if 1-3 pass but Task C tracking is poor in both formats — would indicate the cascaded PID is too weak and we need the SE(3) controller for Epoch 3.

Epoch 2 is a **failure** (stop, investigate) if step 1 fails. That would mean Epoch 1's verbatim assets are subtly broken in a way the static `diff` checks missed — possibly an asset-path issue under MuJoCo's runtime path resolution. Fix by going back to Epoch 1 and adding the `<compiler meshdir="assets" texturedir="assets"/>` directive (Epoch 1 §11.5 allowed edit).

---

## 9. Estimated Effort

| Step | Effort |
|---|---|
| Write cascaded PID + tune on hover | ~1.5 h |
| Author Tasks A/B/C trajectories | ~30 min |
| Driver + Slurm script + GIF rendering | ~1 h |
| Run on cluster, iterate on gains | ~1 h |
| Plots + changelog | ~30 min |
| **Total** | **~4-5 h** |

Effort budget assumes the cascaded PID works on the first or second tuning pass. If we need the SE(3) controller, add ~2 h.

---

## 10. Decision Points Before Launch

| # | Question | Default if not answered |
|---|---|---|
| D1 | Cascaded PID or SE(3) for Epoch 2? | Cascaded PID (simpler) |
| D2 | Render bp-cam-style third-person view for the GIF? | Yes — visual sanity is high-value, low-effort |
| D3 | Test with 6-D `[p,v]` AND 9-D `[p,v,a]` trajectories, or just one? | Both, for Task C only; commit to one for Epoch 3 |
| D4 | Live next to existing visual aligning sbatch scripts, or new dir? | New dir `Slurm_Codes/sbatch/uav_naive/` — keeps UAV cleanly separated |
| D5 | Output format for the trajectory test logs? | JSON + per-step CSV (mirrors existing FM-PCC diagnostic conventions) |

---

## 11. Bottom Line

Epoch 2 = **one Python harness + one SLURM script + three hand-coded tasks**, run once, producing a yes/no on the architectural hypothesis "we can plan in position-space and execute via a separate flight controller, just like aligning does."

If yes → Epochs 3-4-… are unblocked.
If no → we know now, before building three layers on top of an unflyable sim.

Ready to start drafting `uav_naive_test/flight_controller.py` on greenlight.
