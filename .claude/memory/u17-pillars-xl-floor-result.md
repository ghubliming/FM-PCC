---
name: u17-pillars-xl-floor-result
description: "Gen15 U17 pillars_xl wave (Sep 2026) came back as a FLOOR — 53/54 projected cells success=0.00; leading hypothesis is goal radius (0.30) < detour (0.15) + track_err (0.34–0.45), not the projector; 4 results.json files decide it; no more pillars compute until read"
metadata: 
  node_type: memory
  type: project
  originSessionId: 31c7ba41-aa81-4cab-b50e-d49e8f993a05
  modified: 2026-09-21T22:06:45.160Z
---

**State as of 2026-09-21.** The UAV-pillars scene was re-evaluated with the obstacles enlarged at
test time (`pillars_xl`, R=0.35, tag `u7xl`, Gen15 **U17** — not U7, which is the 2026-09-04
honest-geometry unit). Attempt 1 (jobs 25942–25963) died of a full disk; attempt 2 (drivers
25970–25978 → children 25984–25995) ran clean. Diffusion K20 (25951) walled at 2/7 variants.

**The result is a floor, not an ordering.** Unprojected rows reproduce `pillars_hg` (mf/fm 0.90 at
K=5) but are illegal by construction; **every projected row — per-step and endpoint, all three flow
models, K=1/2/5 — reads `success = 0.00`** (one cell 0.10). At K≤2 the signature is
`relaxed=1, safe=1, goal_reached=0`: line crossed, contact-free, goal missed laterally by >0.30 m.

**Leading hypothesis (unverified):** goal at (3.2, ±1.11) with radius 0.30; constraint forces
|y|≥1.26 and the models (trained only on straight |y|=1.11 lines) never return; 0.15 m of the radius
is spent on the detour and tracking error is 0.34–0.45 m → goal unreachable in expectation even for a
constraint-satisfying flight. Alternative: SLSQP non-convergence (HardFlow logs `[NLP-FAILURE]`).

**Why:** `pillars_hg` was degenerate one way (every plan already feasible); `pillars_xl` is degenerate
the other way. Neither orders the four models, so `sec:res:uav:pillars` cannot be restored from this
wave. This is the same class of scene-design trap U11–U13 fell into on the corridor.

**How to apply:**
- Do NOT schedule more pillars compute (nfe=3 rung, diffusion remainder, `pillars_xxl`) until the
  four `results.json` cells named in `SLURM_RUNBOOK_20260919_pillars_enlarged.md` §3e are read
  (`goal.dist`, `n_violations`, `min_z`/`contact_frac`, `projection_health`).
- If hypothesis 1 holds, the fix is at evaluation time (goal on the detour lane, or radius sized to
  detour + tracking error, as U16 did for the corridor slide) — then re-score or re-run.
- Timing column (`proj_ms` per K, diffusion 10.4 s/step) is usable regardless.
- Records: runbook §3b/§3e, `Gen15/U17/CHANGELOG` §8, `PENDING_20260922` §16 (answers filled).
- Sizing lesson: an unprojected cell (9.7 MiB) is a floor for disk, not an average — projected
  cells write far more.

Related: [[fmpcc-dev-logs-navigation]], [[slurm-sbatch-is-real-entrypoint]], [[da-requires-csv-never-from-logs]]
