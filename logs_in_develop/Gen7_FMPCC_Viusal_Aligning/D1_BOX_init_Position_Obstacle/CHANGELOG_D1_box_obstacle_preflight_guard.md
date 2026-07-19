# CHANGELOG — D1: box-init × obstacle pre-flight guard

**Date:** 2026-07-19
**Scope:** Gen6V4 VA-dpcc + Gen7 FM-VA eval scripts (sibling-synced, identical patch)
**Companion analysis:** [`RESEARCH_box_init_vs_obstacle_feasibility.md`](./RESEARCH_box_init_vs_obstacle_feasibility.md)
**Status:** written, syntax-checked locally. **Not executed — run on cluster.**

---

## 1. What was added

A naive **config-sanity pre-flight check**: before a rollout runs, test whether the configured
`obstacles` disc penetrates the *initial box footprint*. If it does by more than a tolerance, print a
loud `[ box-obstacle ]` block to the console/Slurm log, **abort the rollout** (hold position, skip all
planning), and flag it in the rollout JSON.

Modelled deliberately on the existing SLSQP wall-clock safeguard (`projection.py` `Fix_15.2`): detect a
pathological configuration → shout in the log → degrade safely. Never fail silently, never quietly
pollute the aggregate metrics.

### Why this is worth guarding (short version)
The obstacle constraint is a `sphere_outside` cylinder on the **EE position dims (6,7)** — it never sees
the box. So a box starting inside the disc is *not itself* a constraint violation, and it does **not**
make the projection infeasible on its own. But it makes the rollout **guaranteed futile**: to push the
box the EE must go where the box is, and that is precisely the region the projector forbids the EE from
entering. En route the EE gets driven into the sub-disc of radius `r − max_one_step` where the t=1
one-step-escape requirement makes the SLSQP feasible set **empty** — which today surfaces only as
unexplained solver thrash and circuit-breaker trips. See §3 of the research doc for the full derivation.

> **This guard does not repair the projection.** It declares the *context* unusable and says why. The
> actual fix for the underlying geometry is R1 in the research doc (move/delete the placeholder
> obstacle).

---

## 2. Files patched

Identical 5-hunk patch, +124 lines each:

| File | Generation |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | Gen6V4 VA-dpcc |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Gen7 FM-VA |

The two files are near-identical siblings (~9-line offset); all five anchors were verified to match
byte-for-byte before patching, so the generations stay in sync per the repo's copy-modify convention.

### Hunks

1. **Module level** (after `RT_CONTROL_HZ`) — env-tunable constants + two helpers
   `_box_obstacle_overlap()` and `_scan_box_obstacle_conflicts()`.
2. **`reset()`** — `self.ctx_box_obs_conflict = None` (cleared per rollout; `reset()` runs *before*
   `record_context_info()` in `Aligning_Sim.eval_agent`, so ordering is safe).
3. **`record_context_info()`** — runs the scan, prints the abort block, sets the flag, and writes
   `curr_context_info['box_obstacle_conflict']` (which already flows into the rollout record's
   `'context_info'` key → JSON).
4. **`predict()`** — early return of a zero action, inserted at the convergence point of the visual and
   non-visual branches (immediately before `# ── Plan (or execute from cached action chunk)`), so **all
   per-step bookkeeping still runs** and the rollout exports cleanly.
5. **`update_rollout_info()`** — one extra `!! BOX/OBSTACLE CONFLICT ... EXCLUDE FROM METRICS` line in
   the per-rollout summary.

---

## 3. The overlap test

Exact **square-vs-circle penetration depth**, not a bounding-radius approximation:

```python
th   = np.deg2rad(box_angle_deg)
c, s = np.cos(th), np.sin(th)
d    = obs_xy - box_xy
local   = np.array([c*d[0] + s*d[1], -s*d[0] + c*d[1]])   # obstacle centre in box frame
closest = np.clip(local, -half_side, half_side)           # closest point on the box rectangle
sep     = np.linalg.norm(local - closest)
overlap = max(0.0, radius - sep)
```

- Box footprint = the 0.10 × 0.10 m square from `robot_push_box.xml`
  (`geom size="0.05 0.05 0.01"` → half-side 0.05 m), rotated by the context yaw.
- `radius` includes the tightening margin when `is_tightened` (`enlarge_constraints`, 0.03 m).
- Only x-y is considered. Every configured obstacle is `sphere_outside` on `('x','y')` (optionally
  `'z'`); ignoring z is the conservative reading of an infinite cylinder.
- Gated on `'obstacles' in geo_config['constraint_types']` — a no-op for `no_constraint`,
  `geo_bounds_only_*`, `halfspace_only_1`, `combined_2`, and for any `geo_free` variant's config.

### Knobs

| Env var | Default | Meaning |
|---|---|---|
| `FMPCC_BOX_OBS_GUARD` | `1` | `0`/`false`/`no` disables the guard entirely |
| `FMPCC_BOX_OBS_MAX_OVERLAP_M` | `0.005` | penetration tolerated before aborting (m); `0` = abort on the faintest touch |
| `FMPCC_BOX_HALF_SIDE_M` | `0.05` | box half-extent, if the asset ever changes |

---

## 4. Numerical verification (local, pure-Python reimplementation)

`numpy` is unavailable in this container, so the formula was re-derived in stdlib `math` and checked
against the hand-computed figures in the research doc. **Agreement is exact.**

| Case (obstacle centre `(0.50, 0.00)`) | overlap | doc §2.2 says |
|---|---|---|
| worst-case box centre `(0.50, −0.10)` @ 45°, r=0.06 | **0.0307 m** | "up to 0.031 m" ✓ |
| worst-case box centre `(0.50, −0.10)` @ 45°, r=0.09 | **0.0607 m** | "up to 0.061 m" ✓ |
| worst-case box centre `(0.50, −0.10)` @ 0°, r=0.06 | 0.0100 m | — |
| real ctx 25 from job 23514 `(0.521, −0.145)` @ −51.6°, r=0.06 | **0.0000 m** (no abort) | — |
| real ctx 25, tightened r=0.09 | **0.0098 m** (aborts) | — |
| far corner `(0.40, −0.25)` @ 45°, r=0.06 | 0.0000 m | — |

### Expected abort rate under `combined_5` (grid sweep of the full context volume)
`box_space` = x∈[0.40,0.60], y∈[−0.25,−0.10], yaw∈[−90°,90°], tolerance 0.005 m:

| geometry | contexts aborted |
|---|---|
| nominal r = 0.06 | **4.4 %** |
| tightened r = 0.09 | **22.2 %** |

**Read this as a diagnostic, not as noise.** A 22 % abort rate on the tightened twin is the guard
reporting that the placeholder obstacle at `(0.50, 0.00)` is badly placed — it sits on the centreline of
the only corridor the task uses. That is exactly the finding the guard exists to make visible, and the
real remedy is R1 (move or delete the obstacle), not loosening the tolerance.

If a run needs the old behaviour for comparison, set `FMPCC_BOX_OBS_GUARD=0`.

---

## 5. Console output format

```
[ box-obstacle ] CONTEXT 25 ABORTED — box init (0.521, -0.145) @ -51.6° sits inside the EE obstacle.
[ box-obstacle ]   sphere_outside centre=(0.500, 0.000) r=0.090 m penetrates the box footprint by 0.0098 m (tolerance 0.0050 m).
[ box-obstacle ]   The EE must enter this disc to push the box, but the projector forbids exactly that —
                   the rollout is unwinnable and would drive SLSQP toward an empty feasible set.
                   Holding position for this rollout; it is flagged `box_obstacle_conflict` in the JSON.
                   Set FMPCC_BOX_OBS_GUARD=0 to run it anyway.
[ box-obstacle ] holding position for this rollout (context 25) — see abort notice above.
```

and in the per-rollout summary:

```
  - Init XY dist (box→target): 0.3760 m
  - !! BOX/OBSTACLE CONFLICT: aborted, held position (worst overlap 0.0098 m > 0.0050 m tolerance) — EXCLUDE FROM METRICS
```

JSON: `rollout_N['context_info']['box_obstacle_conflict']` =
`{context_idx, worst_overlap_m, tolerance_m, obstacles:[{type, center, radius, overlap_m}]}`.

---

## 6. Behavioural impact

- **Constraint families other than `obstacles`: zero change.** The scan returns `[]` immediately.
- **`no_constraint` / `geo_free*` variants: zero change.**
- **Aborted rollouts still produce a full record** — position trace, video frames, JSON — because the
  early return sits *after* all per-step bookkeeping. They are recorded as ordinary failures
  (`success=False`), distinguishable only by the `box_obstacle_conflict` key.
- **Aborted rollouts are fast**: no diffusion/FM inference and no SLSQP at all, versus ~0.65 s/replan
  previously. The episode still walks its full step budget in the sim, but the expensive part is gone.
- **Downstream aggregation is NOT auto-adjusted.** Success rates still count aborted rollouts in the
  denominator. Filtering on `context_info.box_obstacle_conflict` is left to the analysis layer
  deliberately — silently changing the denominator would be a worse surprise than an explicit flag.

---

## 7. Not done / follow-ups

Deliberately out of scope for D1, carried over from the research doc:

- **R1 — move or delete the placeholder obstacle.** The real fix. It has no physical counterpart in the
  MuJoCo scene (`get_obj_list()` returns before the `Sphere(...)` list — dead code) and sits dead-centre
  in the push corridor. This guard only makes the consequence visible.
- **R2 — check `res.success` in `projection.py`.** Independent, still open: an infeasible SLSQP solve is
  currently accepted as `res.x` with a *finite* cost, so `dpcc-c` selection can actively prefer it.
- **R3 — monotone-recession relaxation** of the obstacle constraint when the EE starts inside.
- **R4 — `Fix_15.2` window semantics**: the "(= replan steps)" comment is wrong; `project()` is called
  ~200×/replan (every denoising step below threshold), not once.

## 8. Verification status

- [x] Both files compile (`python3 -m py_compile`) — syntax clean.
- [x] Overlap formula independently re-derived and matched against the research doc's hand computations.
- [x] All five patch anchors confirmed unique and byte-identical across both generations before editing.
- [ ] **Not run.** No Python env in this container; no cluster job submitted. Needs a `combined_5` eval
      on i6-gpu-1 to confirm the guard fires and the rollout export path stays intact.

Suggested first check on cluster — cheap, exercises the guard densely via the tightened twin:

```
active_geo_variants: [combined_5]     # tightened sibling auto-generated (enlarge_constraints: 0.03)
projection_variants: ['dpcc-r']
n_contexts: 3
```

Expect `[ box-obstacle ]` blocks on roughly 1 in 5 tightened contexts, none on most nominal ones.
