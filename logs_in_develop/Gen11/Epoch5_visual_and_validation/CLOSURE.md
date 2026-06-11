# Gen11 Epoch 5 — Closure

**Date:** 2026-06-11  
**Status:** Complete — GIF generation done, dataset visually inspected

---

## What was delivered

| Workstream | Output | Status |
|------------|--------|--------|
| WS-B Trajectory GIFs | `gifs/{scene}/{homotopy}/*.gif` | ✅ Done |
| WS-D Physics GIFs | `gifs_physics/{scene}/{homotopy}/*_physics.gif` | ✅ Done |
| E4 dataset (1,952 episodes) | `logs/uav_expert_data/{scene}/` | ✅ Complete |

---

## Visual inspection findings

### ✅ No obstacle ghosting / pass-through

Drone no longer clips through or passes through walls and pillars in the physics GIFs. The E4 data collection fixes (U5–U7) — particularly the Z-route for s_curve and the exact-scale thrust allocation for pillars — produced clean expert trajectories that respect scene geometry throughout. The FPV camera fix (U3 Fix_1 + Fix_2) also confirmed the nose-mounted view shows correct obstacle proximity.

### ⚠️ Stop-and-go behavior in trajectory data

The expert trajectories exhibit visible stop-and-go: the drone decelerates to zero at every waypoint transition, pauses, then accelerates again. Two sources:

1. **Hover pauses** — explicit 1.0 s stationary holds at s_curve gap entrances (Hov 1, Hov 2). Added in U7 to give the PID controller time to stabilise attitude before the lateral crossing.

2. **Cosine velocity profile (`traverse_line`)** — every segment starts and ends at `v = 0` by construction. At segment boundaries the drone comes to a full stop before the next leg begins.

**This is by design for the expert data collection** — the goal was safe, rejection-free PID rollouts, not smooth continuous-velocity trajectories. The expert is conservative.

**Implication for downstream training:** The FM-PCC policy will learn this stop-and-go pattern as part of the expert mode of behaviour. Whether this is acceptable depends on the deployment requirement:

- If the task only requires obstacle avoidance (not smooth flight) → acceptable as-is.
- If smooth continuous flight is required → the trajectory generator needs to be redesigned (e.g. remove hover pauses, use a different velocity profile that maintains non-zero speed at waypoints, or use a spline-based planner).

This is an **open question for Epoch 6+**, not a blocking issue for E5.

---

## What is NOT done (out of E5 scope)

- WS-A camera image collection (`collect_camera_images.py`) — not run; not required before training
- WS-C mini-FM sanity gate (`mini_fm_sanity.py`) — not run; optional pre-training check
- Smooth trajectory redesign — deferred to Epoch 6+ if stop-and-go proves limiting
- Pillars homotopy top-up (L,R,L=99 / R,L,R=103 vs 125/125) — optional, not blocking

---

## Next step

Epoch 6 — FM-PCC training on the E4 dataset + E5 visual observations.
