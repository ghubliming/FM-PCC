# fix_2 — MPC fan / executed-dot off-by-one (recording-phase misalignment)

**Date:** 2026-07-12 · **Files:** `npz_analysis/npz_traj_visualizer/npz_traj_export.py`
(+ new `resnap_plan_steps.js` stopgap; existing scene.json/viewers hot-patched).

User (fix_1 follow-up): the MPC fan does **not** connect to the executed dots — the fan does not
start from the dot, and the next dot is not any of the receding-horizon waypoints. Asked to first
rule out the visualizer/extractor, then (if clean) suspect the imf code logic.

## Verdict: NOT an imf-logic / coordinate bug. It is a one-step recording-phase misalignment.

I dumped the exported `scene.json` numbers directly (node, since no Python in this container) and
compared each fan's `h=0` waypoint against the executed path:

- The plan's `h=0` lands **EXACTLY** on an executed sample — `dist == 0.0000` — so columns,
  normalization and coordinate frame are all correct. The extractor faithfully preserves the data,
  and the imf sampler's conditioning (`h=0 === current obs`) is intact. **No coordinate bug.**
- But `h=0` matches `executed[plan_step − 1]`, **not** `executed[plan_step]`. Uniform off-by-one:
  `plan_step` labelled `S` actually starts at executed index `S−1`, and `h=1 ≈ executed[S]`, etc.
  That is exactly the reported symptom (fan one dot ahead of its dot; "next dot" is really `h=1`).

### Root cause (in the eval, not the viewer)

`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` records with a phase offset:

```
action, samples = policy(conditions={0: obs}, ...)   # plan conditioned on the PRE-step obs
... env.step(...) advances obs ...
obs_buffer.append(obs)                                # stores the POST-step obs
```

The plan's `h=0` is the pre-step obs; `obs_buffer` stores the post-step obs (and the initial reset
obs is never stored). So the exported `executed` array is shifted **+1 control step** relative to
the plan's conditioning. The extractor then labelled snapshot `k` with `plan_step = k·save_every`,
one step ahead of the true anchor `k·save_every − 1` → the viewer drew every fan one dot off.

This is **eval-recording convention**, not a control/imf bug — the fan is a faithful receding-horizon
plan. (`eval_fm_uav.py` appends `obs_traj.append(obs)` *before* the state update → offset **0**, so a
blanket "−1" would have *broken* UAV. Confirmed: UAV fans match `executed[plan_step]` exactly.)

## Fix — data-driven offset detection in the extractor (no eval re-run needed)

`npz_traj_export.py`:
- `_nearest_executed()` — nearest executed sample to a fan's `h=0` within ±2 of the nominal step,
  ties broken toward the nominal step (a stalled robot has many executed points equal to `h=0`).
- `_recording_offset()` — takes the **mode** of `nominal − matched_index` over the fans whose `h=0`
  lands cleanly on the path (`residual < 1e-2`, i.e. `dist ≈ 0`). This recovers the eval's structural
  offset (1 for avoiding/imf, 0 for UAV) without hard-coding it. Applying **one** offset uniformly
  per trial keeps genuine plan *divergence* visible (see below) instead of snapping each fan onto
  whatever dot happens to be nearest.
- `snapshot_analytics()` refactored to two passes: detect the offset once, then shift every fan's
  executed anchor (and the tracking-error window) by it. `div_ref`/`explosion`/`spread`/`viol` are
  per-snapshot and unaffected.
- Sanity log per variant/trial: `recording offset=… off_path_frac=…`. The WARN fires only when
  **no** fan's `h=0` lands on the path (`off_path_frac ≈ 1`) — that would mean the offset is
  undetectable, i.e. a real coord/conditioning bug rather than a phase offset.

## Secondary finding (real, expected — surfaced by `off_path_frac`, not hidden)

- **avoiding, unprojected variants** (`diffuser`, `gradient`, `model_free`): `h=0` is on the path
  early but the raw plan **diverges** from the executed path late (`off_path_frac ≈ 0.64`). Correct
  — with no projection the unconstrained plan is free to drift. Projected variants (`dpcc-*`,
  `post_processing`) stay on-path the whole trial (`off_path_frac = 0.00`).
- **UAV s_curve**: even projected variants show high `off_path_frac` late — this is the
  projection-solver-gives-up scenario the E9 circuit-breaker/deadline work targets (the projected
  plan really does diverge once the solver bails). Offset is still cleanly detected as 0 from the
  early/mid fans; the divergence is faithfully shown, not papered over.

## Stopgap for existing exports (Docker has no Python)

`resnap_plan_steps.js` mirrors the same two-pass offset detection in Node and rewrites **only**
`plan_steps` in an existing `scene.json` / embedded viewer, so the fan-connection fix is viewable in
the browser **now** without a cluster re-run. Applied to all current `temp/**/_traj_viz/*` scenes.
Note: it corrects x-alignment only; the pre-fix `track_err` *values* keep their old magnitude until a
proper re-export.

## Verification (node, in-container)

- Patched imf scene, `dpcc-c` t0: `plan_steps [0,4,8,12,…] → [0,3,7,11,…]`; every fan's `h=0` now
  equals `executed[plan_step]` (dist `0.0000`, spot-checked fans 0/1/2/3/5/10). ✓
- Offset detected uniformly: **avoiding = 1**, **UAV = 0** across all variants. ✓
- Projected `off_path_frac = 0.00` (avoiding); unprojected ≈ 0.64; UAV divergence flagged but offset
  still 0. ✓
- `node --check resnap_plan_steps.js`: pass. Python not run locally (**run on cluster**).

## Addendum (2026-07-12, same day) — pillars export, trials, local Python

While preparing the new `temp/pillars_bounds+dynamics+geo_bounds+obstacles` UAV scene:

- **Local Python IS available for this tool.** The exporter needs only numpy (+ pyyaml for avoiding
  geometry). The on-PATH `python3` has no numpy, but **`/usr/local/bin/python3` has numpy 2.5.1 +
  pyyaml**, so the export runs in-container:
  `/usr/local/bin/python3 npz_analysis/npz_traj_visualizer/npz_traj_export.py <scene> --env uav`.
  (Heavy torch/MuJoCo/GPU/eval work still belongs on the cluster.)
- **Truncated npz caught before it could mislead.** First pillars pull was an interrupted transfer —
  every `.npz` cut at a 256 KiB boundary, missing the zip EOCD, so `np.load` → `BadZipFile`. Member
  scan showed the scalars + `obs_all` intact but `sampled_trajectories_all` (the 10.9 MB fan array)
  cut mid-stream (unrecoverable). Quick integrity check for a re-synced file:
  `tail -c 200000 <f>.npz | grep -qa $'PK\x05\x06' && echo OK || echo TRUNCATED`. After re-download
  all 14 files were complete and the export succeeded.
- **`--trials`: viewer already supports trial selection; no code fix needed.** The npz holds **10
  trials**; the viewer has a working Trial `<select>` (populated from the exported `trials` list). It
  only showed trial 0 because the first export used the **default `--trials 0`**. Re-ran with
  `--trials all` → all trials 0–9 × 14 variants now selectable. Cost: inlined viewer grew to ~27 MB
  (subset via `--trials 0,3,7` or thin fans via `--plan-every N` if load time bites).
- **Fresh export ⇒ no resnap.** Offset auto-detected as **0** (UAV convention) for all variants;
  fan `h=0` lands on `executed[plan_step]` (dist 0.0000, spot-checked `bounds_free`). The
  `resnap_plan_steps.js` stopgap is only for pre-fix `scene.json`.
- Reminder (unchanged): UAV env background still not drawn — pillar geometry export is the open TODO.

### Candidates checkbox — confusing "none = all" replaced with "checked = shown"

Viewer template (`npz_traj_visualizer.html`). Old behaviour (from fix_1): the candidate boxes 0–3
defaulted **unchecked**, and "none ticked ⇒ use ALL candidates" — so selecting nothing still drew the
full MPC fan, which reads as broken. Reworked to the obvious semantics:
- Candidate boxes are now **built per-scene from the actual candidate count** (fan `[0].length`, so it
  adapts to any batch size, not a hard-coded 0–3) and **all start checked**.
- **checked = shown**; unticking narrows the fan/mean/horizon-end to the selected candidates; unticking
  **all** hides the fan (guarded so `meanFan([])` can't run). Violation marks still use all candidates.
- Added a sidebar hint: *"checked = shown (all on by default; untick all to hide the fan)"*.
Verified: all `<script>` blocks parse (vm.Script). Pillars re-exported so its inlined viewer carries
the new UI. Other existing viewers keep the old candidate UI until re-exported.

## Not done / follow-ups

- **Re-export on the cluster** via the fixed `npz_traj_export.py` for fully-consistent analytics
  (`track_err` values, not just `plan_steps`). The stopgap only re-aligns x.
- Optional eval-side cleanup (out of scope here): make `eval_flow_matching_v3_imeanflow.py` store the
  initial reset obs and/or append the pre-step obs so avoiding matches UAV's offset-0 convention. The
  extractor now tolerates either, so this is cosmetic.
- Still open from fix_1's deferred list: symlog axis, edge-arrow clip markers, INSPECT fan popup
  table, view-state URL hash, per-layer opacity, UAV pillar geometry, visual-aligning env.
