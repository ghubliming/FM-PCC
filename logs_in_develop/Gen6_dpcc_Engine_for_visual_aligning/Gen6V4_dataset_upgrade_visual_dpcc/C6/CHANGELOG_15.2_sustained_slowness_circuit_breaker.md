# C6 — Fix_15.2: sustained-slowness circuit breaker (Gen6V4, synced from Gen11)

**Module:** `diffuser_visual_aligning/sampling/projection.py` · status: **implemented, run on cluster to verify**
Full write-up: `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_15.2_sustained_slowness_circuit_breaker/`.

## Why (supersedes Fix_15 here)
Gen6V4's projector is byte-identical to Gen7's DPCC SLSQP projector and shares the same failure mode
(Gen7 JOB 23293 `dpcc-r × combined_5`; UAV JOB 23265 `bounds_free`). Fix_15's per-solve 2 s cap was the
wrong altitude — it killed rare-but-legitimate hard solves, turned projection into a silent no-op, and
still burned hours. The disease is *sustained* slowness, not any single slow solve.

## What changed (identical to Gen11/Gen7)
Self-contained in `Projector.project()`, **no eval-script changes**:
- **Generous 60 s per-solve backstop** (was 2 s) — only catches a truly non-terminating solve.
- **Sliding-window circuit breaker** over the last `WINDOW` project() calls: when ≥ `TRIP_FRAC` of a
  full window is "slow" (call > `SLOW_MS`) it **OPENS** and skips projection (~0 ms), printing
  `COST EXPLODED (sustained): …`; after `COOLDOWN` skips it HALF-OPENs and probes one real solve
  (fast ⇒ resume, slow ⇒ re-open). A lone spike never trips it; a new easy episode auto-recovers.
- Tunables (env, defaults): `FMPCC_PROJ_SOLVE_BACKSTOP_S=60`, `FMPCC_PROJ_SLOW_MS=1000`,
  `FMPCC_PROJ_CB_WINDOW=40` (0 disables), `FMPCC_PROJ_CB_TRIP_FRAC=0.9`, `FMPCC_PROJ_CB_COOLDOWN=40`.

## Scope
Solver math unchanged — pure watchdog; breaker state lazily initialized via `hasattr`, no
`__init__`/signature change (keeps the three sibling copies diff-identical).

## Verify on cluster
Run a Gen6V4 visual-DPCC eval with an active obstacle/bounds constraint set that reproduces the
non-convergence; confirm one early `COST EXPLODED (sustained)` line then near-zero projection time,
and that a healthy variant never trips.
