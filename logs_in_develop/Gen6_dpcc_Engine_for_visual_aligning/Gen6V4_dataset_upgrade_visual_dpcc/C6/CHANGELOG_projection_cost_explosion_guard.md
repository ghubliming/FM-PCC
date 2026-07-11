# C6 — Projection cost-explosion guard (Gen6V4, synced from Gen11 Fix_15)

**Module:** `diffuser_visual_aligning/sampling/projection.py` · status: **implemented (Guard 1), run on cluster to verify**
Origin / full analysis: `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_15_projection_cost_explosion_guard/`.

## Why
`Projector.project()` here is the DPCC SLSQP projector (Gen6V4 is the diffusion engine on the `dpcc
base`), byte-identical to Gen7's. It shares the failure mode surfaced in the Gen11 UAV run: on a
nonconvex obstacle QCQP — especially a `bounds_free`-style set where the regularizing box bounds are
removed — SLSQP thrashes toward `maxiter=1000` and a single solve can exceed 100 s. DPCC upstream caps
only `maxiter`, **never wall time**, so a runaway variant can burn hours and be killed by the SLURM
time limit. Patched to keep Gen11 ↔ Gen7 ↔ Gen6V4 in sync.

## What changed
**Guard 1: per-solve wall-clock deadline** in `Projector.project()` (identical to Gen11/Gen7):
- `minimize(..., method='SLSQP', callback=_deadline_cb)`; the callback raises `_SolveBudgetExceeded`
  once a solve passes `_PROJ_SOLVE_BUDGET_S` (default **2.0 s**, env `FMPCC_PROJ_SOLVE_BUDGET_S`,
  `<=0` disables).
- On overrun: keep the **unprojected trajectory** for that sample, set `projection_cost = inf` (never
  selected), bump `self._cost_exploded_count`, and print a greppable `COST EXPLODED` line to stdout
  (→ real `.log`).

## Scope
- Solver math (`maxiter`, `tol`, `Bounds`, constraints) unchanged — pure watchdog; well-behaved
  solves are byte-identical. No `__init__` / eval-script signature change.

## Verify on cluster
Run a Gen6V4 visual-DPCC eval with an active obstacle/bounds-free constraint set; confirm `COST
EXPLODED` fires only on pathological solves and normal projection results are unaffected.
