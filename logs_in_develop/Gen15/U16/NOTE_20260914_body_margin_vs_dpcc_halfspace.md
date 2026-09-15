# NOTE — the UAV body margin vs DPCC's "slide on the halfspace"

**Date:** 2026-09-14 · **Gen:** 15 · **Scope:** discussion note, no code change

## 1. The question

In the corridor_v2 GIFs and plots, the drone slides along a line **0.31 m below** the drawn orange halfspace. In the
DPCC paper figure, the trajectory rides **on** the halfspace line. Is our code wrong?

## 2. Answer: no. It is the Gen11 E9 body inflation (crash prevention), on every UAV scene

The yaml halfspaces and obstacles are **physical surfaces**, i.e. wall faces and pillar radii. The planner offsets every
spatial surface by the drone's rotor reach, so the **rotors** clear the wall, not just the centre
(`config/uav_projection.yaml`, the `inflation` block around line 160). `r_drone = 0.31 m` is the rotor reach at yaw 0.
Fix_12 reduced it from 0.36.

| entries | planner margin | scorer margin (`_exec_constraint_violations`) |
|---|---|---|
| `pillars_hg`, `s_curve_hg`, `corridor_hg`, `corridor_v2_slide` (all campaign / paper runs) | 0.31 m (`planning_inflation`) | 0.31 m |
| `pillars`, `s_curve`, `corridor` (old pre-`_hg` entries) | 0.33 m (global `inflation`, 0.31 + 0.02) | 0.31 m |
| any `-tightened` variant | + 0.025 m (`enlarge_constraints`) | unchanged |

## 3. Same mechanism as DPCC, different value

- **DPCC** (`aux_repo/dpcc/diffuser/utils/constraints_helpers.py:4-9`, `scripts/eval.py:96-97`):
  `formulate_halfspace_constraints(line, enlarge_constraints, …)` shifts the line along its normal.
  - Normal variants pass **0**, because the avoiding end-effector is a point.
  - `-tightened` passes **0.025**.
- **Ours** (`mix_uav_test/eval_mix_uav.py`, `setup_dpcc_projector`, `margin = inflation_base + enlarge`):
  the same function, passed **0.31** (0.335 with `-tightened`).

So the drone **centre does slide on the halfspace**, exactly as in the DPCC figure, but on the line shifted by the body
margin (the red dashed line in the GIFs). The gap between the orange and red lines is the drone's half-width.

## 4. Consequences to keep in mind

1. **Figures:** draw the shifted (centre-limit) line, or caption the orange line as the wall with a 0.31 m body
   clearance. Otherwise readers see a violation of DPCC's picture that is not there.
2. **Scoring is body-aware too.** A "violation" means the drone centre is within 0.31 m of a surface, so it counts
   rotor overlap with the wall. MuJoCo contacts can be zero while violations are not.
3. **Avoiding vs UAV are not the same constraint semantics.**
   - Avoiding (point robot): margin 0, or 0.025 when tightened.
   - UAV (body): 0.31, or 0.335 when tightened.

   State this whenever results from both tasks appear together.
4. **Geometry sizing must include the body.** It is why the original corridor (0.90 m walls, 0.62 m drone) left only
   0.28 m of centre room, and why corridor_v2 was widened (`CHANGELOG_20260913_corridor_v2_wide_slide.md`).
5. **Not a margin issue, but related:** `-pdes` puts this same shifted surface on the setpoint instead of the lagging
   drone (`CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md`).

## 5. Open for discussion

- **Old vs `_hg` entries:** should the old entries' 0.33 planner margin (0.31 + `margin_base` 0.02) be documented as
  superseded? All current runs use `_hg` (0.31).
- **Paper figure:** show the physical wall and the centre limit together, as the U16 GIFs already do, or only the centre limit.
