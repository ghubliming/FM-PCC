# C6 — Projection cost-explosion guard (Gen7, synced from Gen11 Fix_15)

**Module:** `fm_visual_aligning/sampling/projection.py` · status: **implemented (Guard 1), run on cluster to verify**
Origin / full analysis: `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_15_projection_cost_explosion_guard/`.

## Why
`Projector.project()` here is a copy of the DPCC SLSQP projector, sharing the same failure mode found
in the Gen11 UAV run: on a nonconvex obstacle QCQP (especially a `bounds_free`-style constraint set
where the regularizing box bounds are dropped) SLSQP thrashes toward `maxiter=1000`, and a single
solve can take >100 s. DPCC upstream has **no wall-clock guard** — only `maxiter` — so a runaway
variant can burn hours and get killed by the SLURM time limit. Gen7 shares the code, so it shares the
risk; patched proactively to keep the sibling gens in sync (Gen11 ↔ Gen7 ↔ Gen6V4).

## What changed
**Guard 1: per-solve wall-clock deadline** in `Projector.project()` (identical to Gen11):
- `minimize(..., method='SLSQP', callback=_deadline_cb)` where the callback raises
  `_SolveBudgetExceeded` once a solve passes `_PROJ_SOLVE_BUDGET_S` (default **2.0 s**, env override
  `FMPCC_PROJ_SOLVE_BUDGET_S`, `<=0` disables).
- On overrun: keep the **unprojected trajectory** for that sample, set its `projection_cost` to `inf`
  (never selected), bump `self._cost_exploded_count`, and print a greppable `COST EXPLODED` line to
  stdout (→ real `.log`).

## Scope
- Solver math (`maxiter`, `tol`, `Bounds`, constraints) unchanged — pure watchdog; well-behaved
  solves are byte-identical. No `__init__` / eval-script signature change.

## Verify on cluster
Run a visual-aligning eval with an active obstacle/bounds-free constraint set; confirm `COST EXPLODED`
fires only on pathological solves and normal runs are unaffected.
