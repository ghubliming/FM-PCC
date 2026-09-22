---
name: u17-pillars-xl-floor-result
description: "Gen15 U17 (pillars_xl, enlarged keep-out) is ABANDONED 2026-09-22 — all-zero grid, every projector takes the forbidden centre corridor; thesis FELL BACK to pillars_hg with its caveat; never use pillars_xl/xxl numbers or run them again"
metadata: 
  node_type: memory
  type: project
  originSessionId: 31c7ba41-aa81-4cab-b50e-d49e8f993a05
  modified: 2026-09-22T10:08:27.265Z
---

**CLOSED 2026-09-22 — U17 abandoned, fallback to `pillars_hg`.** UAV-pillars was re-evaluated with the
virtual keep-out enlarged (`pillars_xl`, R=0.35 → 0.66 m with rotor reach; Gen15 **U17**, not U7).
Full wave (94 cells × 10 flights, jobs 25970–25995) read **S&C = 0.00 and collision-free = 0.00 in every
cell, unprojected included**. Mechanism (per-flight classification): the corridor between the pillar rows
is physically 0.96 m wide but closed by the keep-out by only 6 cm/side; **every projector — DPCC
per-step and HardFlow endpoint — routes into that forbidden centre lane** (contact-free, line crossed,
~1 m from the goal point, 0.4 m inside the keep-out); per-step at K5 with r/c then hits pillars.
Trajectory PNGs draw physical 0.12 m pillars, so it *looks* fine; scorer/planner use 0.66.

**Author decision:** abandon U17 (closure `Gen15/U17/CLOSURE_20260922_U17_abandoned.md`), restore the
withheld `pillars_hg` section (`v3/withheld/20260918_uav_pillars_section.tex`) WITH its caveat (the
unprojected flows are already feasible there → projected rows measure preservation, not repair).
v3 was notified via `cross_draft/to_v3/FROM_DA_20260922_pillars_u17_abandoned_fallback_hg.md`.

**How to apply:**
- Never put a `pillars_xl`/`pillars_xxl` number in a thesis slot; never submit against those entries
  (yaml carries an ABANDONED banner). No nfe=3 rung, no diffusion K20 projected remainder (10.4 s/step).
- `pillars_grid.py` → pillars_hg/15-09; `uav_results.py` `PILLARS_EXCLUDED = False`; the 22-09 batch
  stays only as the post-mortem corpus (`DA_20260922_pillars_xl_wave.md`).
- "Score only the trajectory" ≠ a fix: the scorer already checks the centre point vs the inflated
  obstacle (= vehicle disc vs raw pillar, same set the projector solves); dropping the 0.31 makes the
  scene trivially feasible again.
- Scene-design lesson: check the PROJECTOR can reach the feasible set from the demo lane (U16 did);
  a lane closed by centimetres is what a local solver will take. Size disk from a projected cell.

Related: [[fmpcc-dev-logs-navigation]], [[da-requires-csv-never-from-logs]], [[hardflow-low-K-degeneracy]]
