# Fix_15.3 — Persist & mark the circuit-breaker trip in the eval artifacts

**Gen11 / Epoch9 (PCC Constraints)** · status: **implemented, run on cluster to verify**
Extends: `Fix_15.2_sustained_slowness_circuit_breaker/`. Synced identically to **Gen7**
(`fm_visual_aligning` / `fm_visual_aligning_test`) and **Gen6V4** (`diffuser_visual_aligning` /
`diffuser_visual_aligning_test`) — same C6 sync group as Fix_15.2.

## The gap Fix_15.2 left

Fix_15.2's circuit breaker does **not** terminate the run. When sustained SLSQP slowness is
detected it **OPENS** and *silently returns the UNPROJECTED trajectory* (~0 ms). So the
episode runs to completion and **all** artifacts (`.npz`, foresight `.svg`, overview/grid
`.png`, `.gif`, `eval_*.log`, per-rollout stats `.json`) are still written — but they carried
**no marker** that projection was abandoned. A tripped result was byte-indistinguishable from a
healthy one: the constraint metrics look real but were computed on an unprojected path.

So the answer to "do we get a warning / do the already-done steps still produce traj/svg/gifs?":
- **Console warning:** yes — `project()` already prints `COST EXPLODED (sustained): …` on trip
  (and `circuit breaker RECOVERED` / `solve backstop hit`). That was the *only* record.
- **Artifacts:** yes, they're still fully written (the run isn't killed) — but **previously
  unmarked**. Fix_15.3 marks them.

## What Fix_15.3 adds — the trip is now recorded everywhere the results live

**1. Projector emits a per-call signal** (`sampling/projection.py`, 3 synced copies):
`self.last_proj_skipped` is set `True` on every OPEN-state skip return and `False` on a real
solve (and on the no-constraint fast path in the visual-aligning copies). Cumulative counters
`_cb_trips` and `_cost_exploded_count` (backstop hits) already existed. No signature/`__init__`
change — keeps the three copies diff-clean.

**2. Eval loop counts skipped steps per rollout** and stores a `projection_health` group
(`cb_tripped`, `cb_skipped_steps`, `cb_trips`, `backstop_hits`):
- UAV: counted in `rollout_one`, added to the returned rollout dict (→ per-rollout
  `rollout_<i>_stats.json` via `save_rollout_stats`, which dumps the whole dict).
- Gen7/Gen6V4: counted in the agent replan block (`curr_rollout_cb_skipped_steps`), rolled into
  `history_cb_skipped_steps` / `history_cb_trips` and into each `master_rollout_history` record.

**3. Non-destructive markers on the saved artifacts** (filenames unchanged — DA pipeline,
`.partial.npz` sidecar, and the npz visualizer all keep working):
| artifact | marker |
|---|---|
| `<variant>.npz` (+ `.partial.npz`) | new arrays `projection_cb_tripped`, `projection_cb_skipped_steps` |
| foresight `.svg` | red banner: *"⚠ PROJECTION CIRCUIT-BREAKER TRIPPED — N step(s) UNPROJECTED … Fan NOT constraint-valid"* |
| overview / rollout-grid `.png` | red banner with the tripped-trial count |
| `eval_<variant>.log` (UAV) | `!!!` banner block + per-rollout `cb=TRIPPED(n)` marker |
| per-rollout `_stats.json` (UAV) / rollout record | `projection_health` group |
| **variant dir** | greppable sentinel `PROJECTION_CB_TRIPPED.txt` (scene/variant, tripped-trial list, total skipped steps, cause) |
| console | one `⚠ … TRIPPED on N/M rollouts … See PROJECTION_CB_TRIPPED.txt` line per tripped variant |
| GIF | marked via the sentinel only (frames are not re-encoded) |

The summary dict also carries a variant-level `projection_health` rollup
(`n_tripped_trials` / `total_skipped_steps` / `tripped_trials`).

## Files touched
- `flow_matcher_v3_uav/sampling/projection.py`, `fm_visual_aligning/sampling/projection.py`,
  `diffuser_visual_aligning/sampling/projection.py` — `last_proj_skipped` flag.
- `FM_v3_uav_test/eval_fm_uav.py` (+ `FM_v3_uav_test/eval_artifacts.py`) — UAV capture + markers.
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`,
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — aligning capture + markers.

## Notes / limits
- On `x_active`/`s_curve` scenes the projector is **rebuilt each replan step**, so its
  `_cb_trips` window resets and can't accumulate — but the per-step `last_proj_skipped` count is
  still exact, so `cb_tripped`/`cb_skipped_steps` remain reliable there.
- The two motivating jobs (`dpcc-r × combined_5`, UAV `bounds_free`) are build-once (no rebuild),
  so both `_cb_trips` and the skip count are accurate.
- Semantics unchanged: a tripped variant means *projection is broken for this geometry*; the
  markers make that unmissable instead of silent. The real fix is still upstream QCQP
  convergence (warm-start `x0`, constraint feasibility/scaling) — see Fix_15.2 follow-ups.

## Verify on cluster
Re-run a tripping variant and confirm: `PROJECTION_CB_TRIPPED.txt` appears in the variant dir;
`npz['projection_cb_tripped']` has 1s; the foresight `.svg` / grid `.png` show the red banner;
and (UAV) `eval_<variant>.log` shows the `!!!` block. A healthy variant produces none of these.

*Run all validation on the cluster (i6-gpu-1); no Python executes in this container.*
