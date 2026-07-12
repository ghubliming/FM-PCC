# C6 — Fix_15.2: sustained-slowness circuit breaker (Gen7, synced from Gen11)

**Module:** `fm_visual_aligning/sampling/projection.py` · status: **implemented, run on cluster to verify**
Full write-up: `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_15.2_sustained_slowness_circuit_breaker/`.
Directly motivated by this gen's JOB 23293 (`dpcc-r × combined_5`), see
`REPORT_dpcc-r_combined5_cost_explosion_run23293.md`.

## Why (supersedes Fix_15 here)
Fix_15's per-solve 2 s cap was the wrong altitude: on JOB 23293 it fired **6 470 times** in a single
variant, made `dpcc-r` projection a silent no-op, **and still burned ~3.6 h** — the job never reached
item 3/24. A single slow solve is harmless; the disease is *sustained* slowness across the whole
episode.

## What changed (identical to Gen11)
Self-contained in `Projector.project()`, **no eval-script changes**:
- **Generous 60 s per-solve backstop** (was 2 s) — only catches a truly non-terminating solve; normal
  slow solves finish and keep their real result.
- **Sliding-window circuit breaker** over the last `WINDOW` project() calls (= replan steps): when
  ≥ `TRIP_FRAC` of a full window is "slow" (call > `SLOW_MS`), the breaker **OPENS** and projection is
  **skipped** (unprojected, ~0 ms) — prints `COST EXPLODED (sustained): …`. After `COOLDOWN` skips it
  HALF-OPENs and probes one real solve; fast ⇒ resume, slow ⇒ re-open. A lone spike never trips it; a
  new easy episode auto-recovers.
- Tunables (env, defaults): `FMPCC_PROJ_SOLVE_BACKSTOP_S=60`, `FMPCC_PROJ_SLOW_MS=1000`,
  `FMPCC_PROJ_CB_WINDOW=40` (0 disables), `FMPCC_PROJ_CB_TRIP_FRAC=0.9`, `FMPCC_PROJ_CB_COOLDOWN=40`.

## Scope
Solver math (`maxiter`, `tol`, `Bounds`, constraints) unchanged — pure watchdog; breaker state
lazily initialized via `hasattr`, no `__init__`/signature change (keeps the three sibling copies
diff-identical).

## Verify on cluster
Re-run `dpcc-r × combined_5`: expect one early `COST EXPLODED (sustained)` line, then near-zero
projection time, and the job progressing past item 2/24 within the time limit. Confirm a healthy
variant never trips.

## Follow-up — Fix_15.3 (artifact marking)
The breaker OPENS silently and returns the UNPROJECTED trajectory, so a tripped run still writes
all artifacts, previously unmarked. **Fix_15.3** now persists the trip into `eval_fm_visual_aligning.py`
outputs (npz `projection_cb_tripped`/`projection_cb_skipped_steps`, red banners on the foresight `.svg`
and rollout-grid `.png`, per-rollout `projection_health`, and a `PROJECTION_CB_TRIPPED.txt` sentinel in
the variant dir). Note this supersedes the "no eval-script changes" scope above for 15.3. Full write-up:
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_15_projection_cost_explosion_guard/Fix_15.3_circuit_breaker_artifact_marking/CHANGELOG.md`.
