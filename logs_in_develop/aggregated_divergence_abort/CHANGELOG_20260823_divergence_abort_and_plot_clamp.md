# Div_Abort — abort a lost rollout, log when/where/why, stop it destroying the plots

**Date:** 2026-08-23
**Scope:** AGGREGATED across two families / six live generations (hence `aggregated_*`, not a
per-Gen folder). UAV: **Gen15** `mix_uav_test/` + **Gen11** `FM_v3_uav_test/`. Visual aligning:
`fm_visual_aligning_test/`, `mix_visual_aligning_test/`, `imf_visual_aligning_test/`,
`diffuser_visual_aligning_test/` (+ the shared `d3il/simulation/aligning_sim.py` episode loop).
**Status:** code written and syntax-checked in Docker — **NOT run**. Needs a cluster run to
confirm the thresholds do not fire on healthy rollouts (see §6).

---

## 1. The report

> the svg, e.g. `.../pillars_bounds+dynamics+geo_bounds+obstacles/hardflow_new-c/diagnostics/rollout_0_mpc_foresight.svg`
> often gets destroyed if the UAV loses control. The plot focus becomes extremely small, most of
> it is the x/y/z hikes in the 2 subplots. If the UAV is flying like crazy, abort the sim and
> write into the logging data incl. when, where and why; in the svg add a marker as abort.

Confirmed and extended to the visual-aligning evals, which "also exploded sometimes" and needed
their own, individual setup rather than the UAV numbers.

## 2. Root cause

Two independent things compound:

1. **The commanded point is a free-running integrator with no absolute clamp.**
   * UAV (`rollout_one`): `p_des = p_des + action` every FM step. Only the projector's
     `bounds` family caps the per-step Δ, nothing caps the accumulated position.
   * Aligning (`aligning_sim.eval_agent`): `pred_action = agent.predict(...)[0] + des_robot_pos`.
     Only `max_action_delta` caps the per-step Δ.
   Once the policy emits a consistent direction, the command walks away — tens to hundreds of
   metres for the drone, right off the table for the arm — while the plant tumbles/saturates
   behind it. **Nothing in either loop stopped the episode**: it burned its full step budget.

2. **Every plot autoscales to the data.** One runaway trace therefore sets the axis limits, and
   the real flight/motion — a ~7 m arena, or a 0.6 × 0.9 m table — collapses into a couple of
   pixel rows next to the hike. That is exactly the "focus becomes extremely small" symptom.

Neither is a constraint-violation issue. Leaving the declared workspace box IS a normal,
measured violation (`_exec_constraint_violations` / `check_trajectory_constraints`) and must
stay measurable — so the new thresholds sit deliberately far outside every planning surface.

## 3. What changed — UAV (Gen15 `mix_uav_test`, Gen11 `FM_v3_uav_test`; identical siblings)

### 3.1 Divergence guard in the control loop — `eval_{mix_uav,fm_uav}.py`
New module block `Div_Abort` + `_divergence_arena()` + `_check_divergence()`. Checked once per
FM step, after the physics decimation, only while the goal is NOT already latched (a rollout
that reached the goal exits through the normal break and is never re-labelled an abort).

| reason tag | condition | default |
|---|---|---|
| `nan_state` | non-finite `p` / `v` / `p_des` | — |
| `out_of_arena` | `p` outside workspace box ⊕ slack (fallback `x,y ∈ ±15`, `z ∈ [-1,15]`) | slack **3.0 m** |
| `overspeed` | `\|v\|` above cap (expert cruise is 0.3–0.5 m/s) | **12 m/s** |
| `p_des_runaway` | `\|p_des − p\|` above cap | **5.0 m** |
| `inverted` | body z-axis · world z < 0 (upside down), from `qpos[3:7]` | — |

Env overrides: `FMPCC_UAV_DIVERGENCE_ABORT=0` (disable, exact old behaviour),
`FMPCC_UAV_DIV_SLACK_M`, `FMPCC_UAV_DIV_SPEED_MS`, `FMPCC_UAV_DIV_LEAD_M`.

On a trip the loop breaks and:
* **`physical.safe` is forced `False`.** Without this, a drone that flew off cleanly (no
  obstacle contact, `min_z` fine) still scored `safe=True`, and on `empty` — where
  `success == safe` — that was a **SUCCESS**. Now every success flag collapses to 0.
* **`n_fm_steps` is charged the FULL budget**, per the U_13 "a miss costs the full budget"
  convention — otherwise a lost flight would look like a *fast* one in `steps_mean`. The real
  executed count is kept as `divergence.executed_steps`.

### 3.2 Where when/where/why lands
* `results.json` / `rollout_<i>_stats.json` → new **`divergence`** group: `reason`, `detail`,
  `step`, `time_s`, `physics_step`, `executed_steps`, `p`, `p_des`, `v`, `speed`,
  `p_des_lead`, `arena_lb/ub`, `thresholds`.
* `<variant>.npz` → `divergence_aborted`, `divergence_step`, `divergence_reason`.
* `eval_<variant>.log` → `!!!` banner listing every aborted trial with position/velocity/why,
  an `ABORT(reason@step)` tag on the per-rollout line, and a `divergence_aborts : n/N` summary.
* `rollout_<episode>.log` (behaviour logger) → new `BehaviorLogger.note()` writes an inline
  `!!!` banner and an `EVENTS:` block in the summary; `behaviour.result` becomes `ABORT(reason)`.
* stdout → one `⚠ DIVERGENCE ABORT` line per trial, one rollup per variant.
* **`DIVERGENCE_ABORT.txt`** sentinel in the variant dir (mirrors `PROJECTION_CB_TRIPPED.txt`),
  so an affected variant is visible from the file tree without opening anything.
* `summary['divergence']` → `n_aborted_trials`, `aborted_trials`, `reasons`.

### 3.3 Plots — `eval_artifacts.py` (byte-identical in both siblings)
* New `view_window()` / `_outside_note()` / `geometry_anchors()`.
  **Rule:** the flown path (robust 2–98 percentile band) and the enforced geometry set the
  scale; `p_des` and the candidate fan may widen the window by at most **1.0 core span** per
  side. Excursions are still drawn (matplotlib clips them) and a red corner note says how many
  points of which series fell outside and how far the worst one reached — the clamp is never
  silent.
* `set_aspect('equal', adjustable='datalim')` → **`'box'`**: with `datalim` matplotlib
  re-expands the data limits to satisfy the aspect ratio and silently undoes the clamp.
* **Abort marker:** dark-red **✖** at the drone's last physical position on both panels, with a
  dotted leader to the `p_des` it was chasing (typically off-window — that IS the failure), an
  `ABORT step N / reason` callout, a dark-red banner naming when/where/why, and `✖ ABORTED at
  step N — reason` appended to the suptitle.
* `plot_overview` (`<variant>.png`) gets the same window rule, a ✖ per aborted trace and a
  variant-level banner.

## 4. What changed — visual aligning (4 evals + `d3il/simulation/aligning_sim.py`)

Same idea, **its own setup** — the arm has no velocity or attitude state exposed by D3IL, and
its workspace is ~1/10 the UAV's, so the UAV numbers would be meaningless here.

| reason tag | condition | default |
|---|---|---|
| `nan_state` | non-finite `des_c_pos` / `c_pos` | — |
| `des_out_of_arena` | commanded position outside the arena box | see below |
| `ee_out_of_arena` | actual TCP outside the arena box | see below |
| `des_runaway` | `\|des_c_pos − c_pos\|` **in XY** above cap | **0.25 m** |

Two deliberate design calls:
* **The lead check is XY-only**, matching the eval's own `max_physical_tracking_error`
  (computed on `[:2]`), so the threshold can be sanity-checked against real runs, and any
  steady-state z offset between commanded and realised TCP cannot trip it.
* **The arena is the UNION of `workspace_bounds ⊕ 0.30 m` and the physical-table box**
  (`x∈[-0.30,1.60]`, `y∈[±1.20]`, `z∈[-0.50,1.50]`), never just the declared box. The aligning
  geo variants deliberately SHRINK `workspace_bounds` for ablations (`geo_bounds_only_1/2`,
  relaxed vs tight `combined_*`) — a shrunken *planning* box must never become an abort
  trigger, because going outside it is precisely the violation the eval exists to measure.
  With every geo entry shipped today the table box is what actually bounds the guard.

Env overrides: `FMPCC_ALIGN_DIVERGENCE_ABORT=0`, `FMPCC_ALIGN_DIV_SLACK_M`,
`FMPCC_ALIGN_DIV_LEAD_M`.

**Mechanism.** The check runs in `predict()` after the per-step bookkeeping (so the trace keeps
a real position up to and including the abort step) and before any planning — no point spending
a replan's SLSQP solves on a runaway command. It sets `agent.abort_episode`; the EE holds
position for the single step it takes `Aligning_Sim.eval_agent` to notice the flag and `break`
out of `while not done`. The break was added to **both** branches (visual and non-visual) of
`d3il/simulation/aligning_sim.py`, behind `getattr(agent, 'abort_episode', False)` so every
agent without the attribute is unaffected. `info` from the step above is what the post-loop
metric assignment uses, so the rollout still records normally.

**Logging:** `divergence` group in `rollout_<r>_stats.json`; `divergence_aborted` /
`divergence_step` / `divergence_reason` in `<variant>.npz`; a per-rollout `⚠ DIVERGENCE ABORT
… — EXCLUDE FROM METRICS` block in the rollout summary print; a `DIVERGENCE_ABORT.txt` sentinel
per variant.

**Plots:** the same window rule (`align_view_window` / `align_outside_note` /
`align_geometry_anchors` — core = the ACTUAL arm path, fixed = enforced geometry + box/target
poses, extra = `des_c_pos` + candidate fan), applied to the XY panel and the 3-D panel
(`set_xlim3d/ylim3d/zlim3d`), plus the ✖ abort marker, leader line, callout and banner.
`adjustable='datalim'` → `'box'` here too.

**Note — env success is NOT overridden.** Unlike the UAV path there is no synthetic
`success == safe` branch to protect, so the D3IL env's own `success`/`mean_distance` are left
exactly as reported. An aborted rollout is simply a failure that covers fewer steps.

## 5. Files touched

```
mix_uav_test/eval_mix_uav.py                                (Gen15)
mix_uav_test/eval_artifacts.py
mix_uav_test/behavior_logger.py
FM_v3_uav_test/eval_fm_uav.py                               (Gen11 sibling — same edits)
FM_v3_uav_test/eval_artifacts.py                            (kept byte-identical to Gen15)
FM_v3_uav_test/behavior_logger.py
fm_visual_aligning_test/eval_fm_visual_aligning.py          (Gen7)
mix_visual_aligning_test/eval_mix_visual_aligning.py        (Gen14)
imf_visual_aligning_test/eval_imf_visual_aligning.py        (Gen3v4)
diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py  (Gen9 / DPCC baseline)
d3il/simulation/aligning_sim.py                             (shared episode loop, both branches)
```

No CLI flags added, no sbatch change needed — the guard is on by default and tuned through
environment variables only.

## 6. What still needs a cluster run

1. **False-positive check (the important one).** Run one healthy variant per family and confirm
   `DIVERGENCE_ABORT.txt` does not appear and `divergence_aborts : 0/N` in the UAV eval log.
   The aligning `des_runaway` threshold (0.25 m XY) is the one to watch — compare it against
   `outcome.max_physical_tracking_error` in the stats JSONs of a known-good run and raise
   `FMPCC_ALIGN_DIV_LEAD_M` if healthy rollouts come anywhere near it.
2. **True-positive check.** Re-run a variant known to produce destroyed SVGs (e.g. the
   `uav-pillars … hardflow_new-c` case in the report) and confirm the abort fires, the SVG is
   readable, and the ✖ + banner land where expected.
3. **Metric drift.** Aborted UAV trials are now scored as misses with `safe=False`; on `empty`
   this can lower a previously-inflated `success_rate`. That is a correction, not a regression —
   but any DA table that mixes pre- and post-Div_Abort UAV runs must say so.
4. **Not covered.** The aligning 9-panel `rollout_<r>_report.png` time-series still autoscales
   per signal; it was not part of the reported symptom and is left alone.
