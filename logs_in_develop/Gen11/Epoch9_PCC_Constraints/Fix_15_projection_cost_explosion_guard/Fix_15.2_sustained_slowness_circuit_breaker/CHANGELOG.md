# Fix_15.2 — Sustained-slowness circuit breaker (supersedes Fix_15 per-solve cap)

**Gen11 / Epoch9 (PCC Constraints)** · status: **implemented, run on cluster to verify**
Supersedes: `Fix_15_projection_cost_explosion_guard/`. Motivated by JOB 23293 (Gen7
`dpcc-r × combined_5`) and JOB 23265 (UAV `bounds_free`).
Synced identically to **Gen7** (`fm_visual_aligning`) and **Gen6V4** (`diffuser_visual_aligning`) — see
their `C6/` changelogs.

## Why Fix_15 was the wrong altitude
Fix_15 hard-capped **every** SLSQP solve at 2 s. Two problems, both seen live in JOB 23293:
- **It punishes rare-but-legitimate hard solves.** A single 120 s solve among 400 otherwise-0.01 s
  steps is harmless — capping it is wrong.
- **It corrupts output.** Every capped solve fell back to the *unprojected* trajectory. On
  `dpcc-r × combined_5` it fired **6 470 times in one variant**, so the projection silently became a
  no-op *and still burned ~3.6 h* (6 470 × 2 s) — the job never reached item 3/24.

The signal that actually matters is **sustained** slowness: is *essentially every* step of an episode
blowing the budget? A per-solve cap can't see that; it only sees one solve at a time.

## What Fix_15.2 does — a circuit breaker over recent steps
All logic is self-contained in `Projector.project()` (`flow_matcher_v3_uav/sampling/projection.py`), so
it syncs identically across the three DPCC-derived projectors with **zero eval-script changes**.

1. **Generous per-solve backstop (60 s).** Replaces the 2 s cap. Only catches a genuinely
   non-terminating solve; normal slow solves finish and **keep their real projected result**.
2. **Sliding window over the last `WINDOW` project() calls (= replan steps).** Each call is timed;
   a call slower than `SLOW_MS` counts as one "slow step".
3. **Trip when sustained.** If ≥ `TRIP_FRAC` of a *full* window is slow, the breaker **OPENS**:
   subsequent calls skip projection entirely and return the unprojected trajectory (~0 ms), so a
   hopeless episode stops bleeding time. Prints one greppable
   `[ projector ] COST EXPLODED (sustained): N/WINDOW recent steps > SLOW_MS ms — OPENING …`.
4. **Auto-recovery (half-open probe).** After `COOLDOWN` skips it HALF-OPENs and runs one real solve:
   fast ⇒ **close** (a new, easy episode resumed — the sliding window needs no explicit episode
   boundary, so this works across trials/episodes automatically); still slow ⇒ re-open.

### Behavior by case
| scenario | Fix_15 (old) | Fix_15.2 (new) |
|---|---|---|
| one 120 s spike, rest fast | killed + result corrupted | **runs to completion, result kept** (1 slow / WINDOW ⇒ no trip) |
| every step ≥ 2 s | 3.6 h slow bleed, invalid output | **trips in ~WINDOW steps, then ~0 ms/step**, marked |
| bad episode → later easy episode | n/a | half-open probe **auto-resumes** projection |
| result integrity | fallback-to-unprojected on every cap | only skips while sustained-hopeless |

## Tunables (env-overridable, sane defaults)
| env var | default | meaning |
|---|---|---|
| `FMPCC_PROJ_SOLVE_BACKSTOP_S` | 60.0 | per-solve hang backstop (s); `<=0` disables |
| `FMPCC_PROJ_SLOW_MS` | 1000.0 | a project() call slower than this = one "slow step" |
| `FMPCC_PROJ_CB_WINDOW` | 40 | steps of recent history to judge; **0 disables the breaker** |
| `FMPCC_PROJ_CB_TRIP_FRAC` | 0.9 | fraction of a full window that must be slow to OPEN |
| `FMPCC_PROJ_CB_COOLDOWN` | 40 | OPEN skips before a HALF-OPEN probe |

Defaults: need 36 of the last 40 steps > 1 s to trip → a lone spike never trips; all-steps-slow trips
within ~40 steps (~1–2 min) then costs ~nothing. Healthy solves are sub-100 ms, so `SLOW_MS=1000` has
wide separation from normal behavior.

## Implementation notes
- Breaker state (`_cb_state`, `_cb_window`, `_cb_skips`, `_cb_trips`) is lazily initialized in
  `project()` via `hasattr` — **no `__init__` change**, keeps the three copies diff-identical.
- OPEN-state skip returns `projection_costs = inf` (never preferred by trajectory-selection) and sets
  `self.last_proj_ms = 0.0`.
- The 60 s backstop retains the SLSQP-`callback` raise mechanism from Fix_15 (`_SolveBudgetExceeded`).
- `WINDOW=0` disables the breaker (guarded so an empty deque can't trip spuriously).

## Verify on cluster
- Re-run JOB 23293's `dpcc-r × combined_5`: expect a single `COST EXPLODED (sustained)` line early,
  then near-zero projection time, and the job progressing past item 2/24 within the time limit.
- Re-run UAV `bounds_free*`: same — breaker opens, variant finishes fast, other variants no longer
  starved. Spot-check that a normal variant (e.g. `post_processing`) never trips.
- Fast triage knob if needed: lower `FMPCC_PROJ_SLOW_MS` / `FMPCC_PROJ_CB_WINDOW`.

## Follow-ups (unchanged from Fix_15)
The breaker is a *watchdog*, not a cure. The real defect is `dpcc-r × combined_5` (and UAV
`bounds_free`) projection **non-convergence** — warm-start `x0`, check constraint feasibility/scaling
of the QCQP. A variant that trips the breaker should be treated as "projection broken for this
geometry", not as a valid result.

*Run all validation on the cluster (i6-gpu-1); no Python executes in this container.*
