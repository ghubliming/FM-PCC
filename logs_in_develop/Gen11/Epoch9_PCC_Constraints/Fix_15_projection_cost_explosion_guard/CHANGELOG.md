# Fix_15 — Projection cost-explosion guard (Gen11 UAV)

**Module:** `flow_matcher_v3_uav/sampling/projection.py` · status: **implemented (Guard 1), run on cluster to verify**
Problem analysis: see `PROBLEM_projection_cost_explosion.md` in this folder.

## Why
JOB 23265 (`eval_fm_uav`, pillars, seed 6) hit the 24 h SLURM limit and was killed mid-variant 15/20.
Two variants (`bounds_free`, `bounds_free-tightened`) ate ~17.5 h: SLSQP thrashes toward `maxiter=1000`
on the nonconvex obstacle QCQP when the box bounds are removed, so single solves took up to **112 s**
(p95) — ~3 700× the 30.3 ms (33 Hz) real-time tick. DPCC upstream has **no wall-clock guard** (only
`maxiter=1000`), so nothing stopped a variant from running for hours.

## What changed
Added **Guard 1: a per-solve wall-clock deadline** inside `Projector.project()`:
- SLSQP now gets a `callback=_deadline_cb` that raises `_SolveBudgetExceeded` once a solve exceeds
  `_PROJ_SOLVE_BUDGET_S` (default **2.0 s**, override via env `FMPCC_PROJ_SOLVE_BUDGET_S`, `<=0` disables).
- On overrun the sample **keeps its unprojected (FM) trajectory**, its `projection_cost` is set to
  `inf` (so trajectory-selection never prefers a runaway sample), and a loud greppable marker is
  printed to stdout (→ the real `.log`):
  `[ projector ] COST EXPLODED: SLSQP solve exceeded 2.0s budget (…s) — kept unprojected trajectory …`
- Running tally kept in `self._cost_exploded_count`.

## Guarantees / scope
- **Solver math untouched** (`maxiter`, `tol`, `Bounds`, constraints identical) — the guard is a pure
  watchdog; well-behaved solves return byte-identical results.
- No `__init__` or `eval_fm_uav.py` signature change (counter via `getattr`, budget via env) — drop-in.
- Callback fires between SLSQP iterations, so a solve may overrun the budget by at most one iteration.

## Not done (deferred, see PROBLEM MD §4)
Guard 2 (per-episode budget kill) and Guard 3 (per-variant abort + `status:cost_exploded` in
`results.json`) — Guard 1 already prevents the multi-hour runaway; add if solves still stack up.

## Verify on cluster
Re-run the pillars/seed-6 sweep; confirm `bounds_free*` now finish in minutes with `COST EXPLODED`
lines, and the full 20-variant job completes inside the time limit.

*Synced identically to Gen7 (`fm_visual_aligning`) and Gen6V4 (`diffuser_visual_aligning`).*
