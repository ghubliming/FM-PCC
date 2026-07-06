# C4 — PLAN: reorganize Gen7/Gen6V4's chaotic per-rollout JSON + NPZ, + new `success_relaxed` (mirrors UAV Fix_10/U7)

**Date:** 2026-07-06. Concept/plan only — **no code in this file**. Same motivation as
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_10_json_metrics/`: the metrics are correct
and complete, but the schema is chaotic — flat scalars mixed with two ad-hoc nested groups,
one literal duplicate metric, and a genuinely orthogonal "plan vs. executed" axis buried
inside one undifferentiated `constraint_metrics` blob.

## The concrete example (user-pasted, `rollout_0_stats.json`)
```json
{
  "rollout_index": 0, "success": false, "steps": 400, "mean_distance": 0.48,
  "mode": 1, "avg_inference_time_per_replan": 0.395, "max_physical_tracking_error": 0.448,
  "context_info": { "context_idx": 0, "box_init_xy": [...], "box_init_angle_deg": ...,
                     "target_xy": [...], "target_angle_deg": ..., "init_xy_dist": ...,
                     "final_box_xy": [...], "final_box_angle_deg": ..., "final_xy_dist": ... },
  "constraint_metrics": {
    "exec_n_steps": 400, "exec_n_violated_steps": 268, "exec_constraint_sat_rate": 0.33,
    "exec_zero_violation_rollout": false,
    "exec_bounds_viol_count": 268, "exec_halfspace_viol_count": 0, "exec_obstacle_viol_count": 0,
    "exec_max_bounds_viol_m": 0.58, "exec_max_halfspace_viol_m": 0.0, "exec_max_obstacle_penetration_m": 0.0,
    "exec_constraint_margin_mean_m": 0.109,
    "exec_first_violation_step": 132, "exec_longest_safe_streak": 132,
    "exec_dynamics_consistency_error_mean": 0.0066, "exec_dynamics_consistency_error_max": 0.054,
    "plan_post_viol_rate_mean": 0.3575, "plan_post_viol_rate_max": 1.0, "plan_n_replan_steps": 400
  }
}
```

## Where this is actually produced (traced in code, not guessed)
- `Aligning_Sim`-style agent class's rollout-end handler builds `master_rollout_history[...]`
  (the IN-MEMORY dict, also carrying heavy arrays like `real_robot_pos`/`full_plans`), calling
  `check_trajectory_constraints(...)` for the `exec_*` fields and appending
  `self._plan_post_viol_rates` (accumulated once per replan step from
  `_check_planned_violations(...)`, called during the rollout) for the `plan_*` fields.
- The **exported** `rollout_{idx}_stats.json` (what the user pasted) is a filtered/renamed
  subset of that in-memory dict — `fm_visual_aligning_test/eval_fm_visual_aligning.py` ~L1063-1074.
- A **separate** aggregate file, `constraint_metrics.json`, is written once per (variant,
  seed) with mean±std across all rollouts **and** the raw `per_rollout` list re-embedded
  (~L2228-2247) — i.e. the same per-rollout `constraint_metrics` dict exists in **two places on
  disk** (inside every `rollout_N_stats.json` **and** inside `constraint_metrics.json`'s
  `per_rollout` array) — not itself wrong (one is per-rollout detail, one is the aggregate
  view), but worth knowing before touching either.
- The **NPZ** (`<variant>.npz`, ~L2082-2102) currently has **zero** constraint-metric arrays —
  `exec_*`/`plan_*` were never persisted there, only `context_*` (already flattened into NPZ
  arrays, e.g. `context_box_init_xy`) and the basic `success`/`steps`/`mean_distance`/
  `avg_time`/`physical_tracking_errors`. Same gap class as UAV had pre-Fix_10 (JSON has more
  metrics than NPZ ever captured).
- **Confirmed dual-file, same as Patch_Constraints_C3 / Gen11E9U8_Sync**: this exact
  structure (byte-for-byte, same variable names, same line-number-ish layout) exists in BOTH
  `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7, FM engine) and
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4, DDPM engine) — any fix
  must be applied to both, identically, as every prior fix in this Epoch/thread has been.
- **Also present** (grepped, 8 matches) in `imf_visual_aligning_test/eval_imf_visual_aligning.py`
  — a third sibling, **not** named by the user's "Gen7/Gen6V4" scope. Flagged here for
  awareness (mirroring how the config-snapshot audit handled `imf_visual_aligning`), not
  included in the primary scope below unless the user asks for it too.

## What's actually chaotic (concrete findings, not vague "it's messy")

### 1. A literal duplicate: `steps` vs `constraint_metrics.exec_n_steps`
Both are `400` in the pasted example. `steps` (top-level, `self.step_counter`) and
`exec_n_steps` (inside `constraint_metrics`, from `check_trajectory_constraints`'s own `T =
len(pos)`) are computed from **different code paths** but represent the **same underlying
quantity** (this rollout's total step count) and will always coincide in practice (both derive
from the same rollout's `c_pos` history length). Two names, one fact — delete one, keep the
other as the single canonical source.

### 2. `constraint_metrics` conflates TWO genuinely orthogonal axes with no grouping
- **Executed-trajectory axis** (`exec_*`): checks the trajectory the robot **actually flew**
  against the declared geometric constraints. Sub-breakdown by **geometric family** (bounds /
  halfspace / obstacles) is present but flattened into 6 parallel sibling keys
  (`exec_bounds_viol_count`, `exec_halfspace_viol_count`, `exec_obstacle_viol_count`,
  `exec_max_bounds_viol_m`, `exec_max_halfspace_viol_m`, `exec_max_obstacle_penetration_m`) —
  should be one `by_family: {bounds: {...}, halfspace: {...}, obstacles: {...}}` structure.
- **Planned-trajectory axis** (`plan_*`): checks the FM's **post-projection candidate
  trajectories**, before execution — a genuinely different, earlier measurement point in the
  pipeline (`_check_planned_violations`, called once per replan during the rollout, not once
  at the end). This is actually a **richer** distinction than anything UAV's schema captures
  (UAV never separated plan-time vs. exec-time checking) — the reorg must **preserve** this
  distinction clearly as two sub-groups, not flatten it away chasing UAV's exact shape.
- These two axes currently sit as unmarked siblings in one dict with no visual/structural
  signal that `exec_*` and `plan_*` are measuring the plan vs. the execution of it.

### 3. Top-level scalars are an unsorted grab-bag
`rollout_index`, `success`, `steps`, `mean_distance`, `mode`, `avg_inference_time_per_replan`,
`max_physical_tracking_error` sit at the same flat level with no grouping — a mix of
identity (`rollout_index`, `mode`), outcome (`success`, `mean_distance`,
`max_physical_tracking_error`), and performance (`avg_inference_time_per_replan`) fields.

### 4. `max_physical_tracking_error` vs `exec_dynamics_consistency_error_*` — similar-sounding, different things (clarify, don't merge)
Both are "how far is actual from something" style metrics, but: `max_physical_tracking_error`
= |actual c_pos − desired c_pos| (tracking error — did the robot follow its own commanded
setpoint?); `exec_dynamics_consistency_error_*` = does the actual trajectory obey the Euler
consistency equation (`c_pos[t+1] = c_pos[t] + act[t]`)? — a model-consistency check, not a
tracking-fidelity check. **Not a duplicate** — but the near-identical framing ("error between
two close things") is exactly the kind of confusion Fix_10 flagged for UAV's `safe` vs.
`collision_free`. Keep both, but place them in a way that signals they're different rulers
(see proposed schema — one lives in `outcome`, the other inside `constraint.exec`).

### 5. NEW metric requested: `success_relaxed` — position-only, ignore angle (mirrors UAV's U7)

Traced the **strict** `success` all the way to its source — `_check_early_termination()` in
`d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py:333-351`:
```python
box_goal_pos_dist = np.linalg.norm(box_pos - target_pos)
box_goal_rot_dist = rotation_distance(box_quat, target_quat) / np.pi
if (box_goal_pos_dist <= self.pos_min_dist) and (box_goal_rot_dist <= self.rot_min_dist):
    ...  # success
```
`self.pos_min_dist = 0.018` (metres), `self.rot_min_dist = 0.048` (normalized rotation
distance) — set at `aligning.py:198-199`. **Strict `success` requires BOTH** position AND
rotation to be within their own threshold.

**User's ask:** a `success_relaxed` that checks **position only** — if the final position gets
within a range, it's relaxed-success, regardless of final angle. This is the direct analog of
UAV's U7 `success_relaxed` (crossed the finish line, regardless of exact final pose) —
same idea, applied to this task's own strict-success decomposition (pos AND rot → relax to
pos alone).

**The needed ingredient is already computed and stored, no new measurement required**: the
eval script's own rollout-end handler already computes `final_xy_dist` — literally
`sqrt((final_box_x - target_x)^2 + (final_box_y - target_y)^2)` — and stores it in
`context_info` (`fm_visual_aligning_test/eval_fm_visual_aligning.py` ~L899-913, the
`_final_xy_dist` block already read while building this plan). So:
```python
success_relaxed = bool(final_xy_dist <= pos_min_dist)   # same threshold, no angle term at all
```
**Implementation note — don't hardcode `0.018`.** Pull `pos_min_dist` from the live env
instance (e.g. `env.pos_min_dist`, whatever attribute path the eval script already has access
to at rollout-end) rather than copying the literal `0.018` into the eval script — the env is
the single source of truth for that threshold; hardcoding it a second place risks silent
drift if the env's threshold is ever tuned. Confirm the eval script actually holds an `env`
reference at the point `context_info`/`success` are finalized (very likely, since it already
reads `box_pos`/`target_pos`/`box_quat` off the sim to build `context_info` in the first
place) — open question #5 below.

**Deliberately not doing more than asked:** UAV's `success_relaxed` was `crossed_line AND
safe` (two conditions). This one is requested as **position-distance alone** — no safety/
constraint gate, no angle term. Honoring that literally; not bundling in `exec_constraint_sat_rate`
or anything else unless asked.

**Schema placement:** groups with strict `success` (was a lone top-level boolean; now a small
`success` object with both), same naming convention as UAV's `success.strict`/`success.relaxed`
— reusing that word choice because the concept genuinely is analogous here, not because of any
forced cross-pipeline identical-key-name goal (which was explicitly rejected for these two
codebases in the UAV Fix_10 discussion — this is just "same English word for the same idea,"
not schema-sharing).

### 6. NEW metric requested: first-contact / last-contact time+position (EE/gripper ↔ box)

**Verdict: cheap — do it.** Traced what's already available before proposing anything new:

**What "contact" would actually mean here — a real gap first.** Checked
`aligning.py`/`d3il_sim/sims/mj_beta/{MjScene,MjRobot}.py` for a genuine MuJoCo mesh-contact
query (the `data.contact`/`ncon` pattern UAV already uses via `gen._is_obstacle_contact`) —
**none exists** in this env's Python wrapper. What DOES already exist, computed every step
inside `check_mode()` (`aligning.py:294-318`) and returned in `info['mode']`:
```python
robot_box_dist = np.linalg.norm(box_pos[:2] - robot_pos[:2])
mode = 0 if robot_box_dist < self.robot_box_dist else 1   # self.robot_box_dist = 0.051 m
```
i.e. a **distance-proximity proxy** ("EE within 5.1cm of the box, in XY"), not literal mesh
contact. Two options, honestly costed:
- **Cheap (recommended): reuse the proximity proxy.** `record_step_info` (`eval_fm_visual_aligning.py:997`,
  called after every `env.step()`) already receives this exact `info` dict and today only
  captures `mean_distance` from it, discarding `mode`. Capturing it too is a **one-line
  addition** mirroring the existing pattern: `self.curr_rollout_mode_history.append(int(info.get('mode', 1)))`.
  At rollout-end, first/last-contact step = first/last index where `mode_history[i] == 0`;
  position at that step = `self.curr_rollout_c_pos[i]` (**already recorded**, no new capture
  needed). Zero new physics queries, zero new simulation instrumentation — purely reusing two
  arrays that already exist or need one already-established-pattern line to start existing.
- **Expensive (not recommended unless truly wanted): genuine MuJoCo mesh contact.** Would need
  the gripper's and box's exact geom IDs/names in the compiled MuJoCo model, then a per-step
  `data.contact[i].geom1/geom2` membership check (same shape as UAV's
  `_is_obstacle_contact`, but that helper doesn't exist for THIS env and there's no existing
  reference to copy — would be written from scratch). Real but non-trivial new work, and a
  genuinely different (stricter) signal than "within 5cm."
- **Recommendation: ship the cheap proximity-proxy version**, but label it honestly as
  proximity (`contact` meaning "within `robot_box_dist` threshold", not verified mesh
  contact) so nobody mistakes it for a physics-engine contact event later.

**Proposed schema** — new `contact` group (doesn't fit cleanly into `outcome`/`context`/
`timing`, it's its own axis):
```json
"contact": {
  "first_step": 45,  "first_pos_xy": [0.31, -0.08],
  "last_step":  210, "last_pos_xy":  [0.02,  0.36],
  "note": "proximity proxy (robot-box XY dist < 0.051m), not physical mesh contact"
}
```
`null`/`-1` for `first_step`/`last_step` if `mode` never reaches 0 during the rollout (never
got close) — must handle the empty-history case explicitly, not assume at least one contact
step exists.

**NPZ addition** (same additive spirit as the rest of this plan): `contact_first_step`,
`contact_last_step`, and the two position arrays (`contact_first_pos_xy`, `contact_last_pos_xy`,
shape `(n_rollouts, 2)`) — cheap, four more `np.array([...])` lines in the existing `np.savez(...)` call.

### 7. NEW plot request: mark first/last-contact position on the MPC-foresight SVG

**Verdict: cheap — do it, but only on the XY panel.** Traced the actual SVG (not guessed):
`fig_mpc` (`eval_fm_visual_aligning.py:1202-1449`, saved as
`rollout_{idx}_mpc_foresight.svg`) has an `ax_xy` top-down panel (`add_subplot(1,2,1)`,
~L1208-1266) that already plots the real/commanded path (`real_pos`/`c_arr`) the candidate
fan is drawn against. Marking the first/last-contact position there is **two more plot calls
using data already loaded at that point in the function** — e.g.
```python
if _first_contact_step is not None:
    _p = c_arr[_first_contact_step] if c_arr is not None else real_pos[_first_contact_step]
    ax_xy.scatter([_p[0]], [_p[1]], marker='*', s=140, color='blue', zorder=15, label='first contact')
if _last_contact_step is not None:
    _p = c_arr[_last_contact_step] if c_arr is not None else real_pos[_last_contact_step]
    ax_xy.scatter([_p[0]], [_p[1]], marker='X', s=140, color='purple', zorder=15, label='last contact')
```
**Not** proposing this for the 3D panel (`ax_3d`) too — one clear marked panel is enough;
duplicating onto the 3D view roughly doubles the plotting code for this feature with little
extra legibility (3D scatter markers are harder to read precisely than a 2D top-down one for
"where exactly did contact happen"). Flag as a follow-up if the 3D view turns out to matter in
practice, not a default part of this addition.

## Proposed reorganized schema (per-rollout JSON)
```json
{
  "rollout_index": 0,
  "mode": 1,
  "success": {
    "strict": false,
    "relaxed": true
  },
  "outcome": {
    "mean_distance": 0.48,
    "max_physical_tracking_error": 0.448
  },
  "timing": {
    "steps": 400,
    "avg_inference_time_per_replan": 0.395
  },
  "context": {
    "context_idx": 0,
    "box_init_xy": [0.454, -0.190], "box_init_angle_deg": -41.80,
    "target_xy": [0.526, 0.279],    "target_angle_deg": -53.45,
    "init_xy_dist": 0.474,
    "final_box_xy": [0.003, 0.388], "final_box_angle_deg": 23.14,
    "final_xy_dist": 0.534
  },
  "contact": {
    "first_step": 45,  "first_pos_xy": [0.31, -0.08],
    "last_step":  210, "last_pos_xy":  [0.02,  0.36],
    "note": "proximity proxy (robot-box XY dist < 0.051m), not physical mesh contact"
  },
  "constraint": {
    "exec": {
      "n_violated_steps": 268,
      "constraint_sat_rate": 0.33,
      "zero_violation_rollout": false,
      "by_family": {
        "bounds":    {"viol_count": 268, "max_viol_m": 0.58},
        "halfspace": {"viol_count": 0,   "max_viol_m": 0.0},
        "obstacles": {"viol_count": 0,   "max_viol_m": 0.0}
      },
      "margin_mean_m": 0.109,
      "first_violation_step": 132,
      "longest_safe_streak": 132,
      "dynamics_consistency_error": {"mean": 0.0066, "max": 0.0543}
    },
    "plan": {
      "post_viol_rate_mean": 0.3575,
      "post_viol_rate_max": 1.0,
      "n_replan_steps": 400
    }
  }
}
```
`exec_n_steps` is **deleted** (Finding #1) — `timing.steps` is the single canonical step count;
`check_trajectory_constraints`'s own `T = len(pos)` is used internally to build the array
lengths but no longer separately reported. (Open question below: confirm the two truly can
never diverge before deleting, not just assumed from one example.)

## NPZ additions (mirrors UAV Fix_10's "share metrics" principle)
Add per-rollout arrays for the constraint axis (currently **absent from NPZ entirely**,
existing only in the two JSON files), group-prefixed to match the JSON schema above:
```
constraint_exec_n_violated_steps, constraint_exec_sat_rate, constraint_exec_zero_violation,
constraint_exec_bounds_viol_count, constraint_exec_halfspace_viol_count, constraint_exec_obstacle_viol_count,
constraint_exec_max_bounds_viol_m, constraint_exec_max_halfspace_viol_m, constraint_exec_max_obstacle_penetration_m,
constraint_exec_margin_mean_m, constraint_exec_first_violation_step, constraint_exec_longest_safe_streak,
constraint_exec_dyn_err_mean, constraint_exec_dyn_err_max,
constraint_plan_post_viol_rate_mean, constraint_plan_post_viol_rate_max, constraint_plan_n_replan_steps,
outcome_max_physical_tracking_error,
success_strict, success_relaxed,
contact_first_step, contact_last_step, contact_first_pos_xy, contact_last_pos_xy,
```
(`context_*` arrays already exist in NPZ, untouched; `n_steps`/`mean_distance`/`avg_time`
already exist, untouched. `success` already exists as `n_success` — keep that array as the
strict signal (rename to `success_strict` for consistency, same as UAV Fix_10 did) and add
`success_relaxed` as a new array, computed the same way as the JSON field above. Every other
addition here — the constraint/outcome metrics — is new, same "additive, not just renamed"
spirit as UAV's `phys_*`/`goal_*` additions.)

## Files in scope (confirmed by direct grep, both files byte-for-byte identical structure)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` — rollout-end handler (`master_rollout_history`
  construction + `check_trajectory_constraints`/`_plan_post_viol_rates` call sites), the
  `rollout_{idx}_stats.json` export block, the `constraint_metrics.json` aggregate block, the
  `np.savez(...)` NPZ block.
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — identical touch points.
- `npz_analysis/analyze_npz.py` — `HEADLINE_KEYS` needs the new NPZ array names appended
  (additive only, same pattern as UAV Fix_10 — confirmed schema-generic elsewhere in this file
  already, per the earlier UAV plan's audit of this same tool).

## Open questions (need answers/verification before implementing, not guessed here)
1. **Can `steps` and `exec_n_steps` ever actually diverge?** Confirm by reading
   `check_trajectory_constraints`'s exact input (`c_pos_traj` — is it ever a truncated/sub-
   sampled slice of the full rollout, or always the complete `curr_rollout_c_pos`?) before
   deleting one as "always identical" — this plan asserts it from one example, not a proof.
2. **Does anything outside these two eval scripts read `rollout_N_stats.json` or
   `constraint_metrics.json` by their current flat/nested key names?** (e.g. a Data_Analysis
   pipeline, a notebook, `npz_analysis`). Needs a repo-wide grep for the specific field names
   (`exec_bounds_viol_count`, `plan_post_viol_rate_mean`, `avg_inference_time_per_replan`,
   etc.) before finalizing the rename — not assumed absent just because this plan didn't find
   one yet.
3. **Should `constraint_metrics.json` (the separate aggregate file) also restructure to match**
   (nested `exec`/`plan` groups with `by_family`), or is its current flat
   `{metric: {mean, std}}` shape acceptable as a distinct, aggregate-only convention? (My
   lean: yes, restructure it too, for the same reason UAV's `summary` block was restructured
   alongside its per-rollout dict — but flagging as a decision, not assuming.)
4. **Is `imf_visual_aligning_test/eval_imf_visual_aligning.py` (the third sibling found) in
   scope too, or explicitly deferred** since the user named only "Gen7/Gen6V4"?
5. **Where can the eval script read `pos_min_dist` from at rollout-end?** Confirm the exact
   attribute path (e.g. `env.pos_min_dist`, or via whatever wrapper object holds the sim) at
   the point `context_info`/`success` are finalized, so `success_relaxed` derives the threshold
   from the env instead of a second hardcoded `0.018`.
6. **Confirm `record_step_info` actually receives `info` on every step, every rollout type**
   (visual and non-visual/`capture_frame` branches both call `env.step()` — verify both paths
   route through `record_step_info` identically, not just the branch read so far) before
   relying on it for the new `mode`-history capture.
7. **Decide the never-touched edge case explicitly**: if `mode` never reaches 0 for an entire
   rollout (robot never got within 5.1cm of the box), `contact.first_step`/`last_step` are
   `null` and `first_pos_xy`/`last_pos_xy` absent — confirm downstream consumers (NPZ arrays
   especially, which want fixed-shape numeric arrays) handle this sentinel consistently (e.g.
   `-1` for the step index, `[nan, nan]` for position, rather than a ragged/optional array).

## What this plan does NOT propose
- No change to what's already measured — `check_trajectory_constraints`/
  `_check_planned_violations` themselves are untouched; this is presentation/organization
  only for everything except the one explicit addition below.
- **Three exceptions are genuinely NEW metrics/plot elements**, not reorganizations of
  existing ones — all added because the user asked, all assessed as cheap before being
  included:
  - `success_relaxed` (Finding #5) — new boolean, derived from an existing number
    (`final_xy_dist`) and the env's existing `pos_min_dist` threshold.
  - `contact.first_step`/`contact.last_step`/positions (Finding #6) — new per-step capture,
    but of a value (`info['mode']`) that was **already being computed and discarded** every
    step; genuine mesh-contact detection was considered and explicitly NOT chosen (costed as
    the expensive option) in favor of the existing proximity proxy.
  - First/last-contact markers on the MPC-foresight SVG's XY panel (Finding #7) — two plot
    calls using data (`c_arr`/`real_pos`) already loaded at that point in the function; NOT
    extended to the 3D panel (lower value for the added complexity).
- No merging of `max_physical_tracking_error` and `exec_dynamics_consistency_error_*` — they
  measure different things and both stay, just clearly separated (Finding #4).
- No decision yet on whether `constraint_metrics.json` gets deleted in favor of NPZ arrays —
  recommend keeping it (human-readable mean±std summary), just made consistent in naming.

## Suggested next step
Answer open questions 1–4 (mostly quick greps/reads, doable without cluster access), then
implement identically across both `eval_fm_visual_aligning.py` and
`eval_visual_aligning_dpcc.py`, mirroring exactly how `Fix_10_json_metrics` was executed for
UAV: nest the per-rollout dict, extend NPZ with the previously-missing arrays, update
`analyze_npz.py`'s `HEADLINE_KEYS` additively, verify with a synthetic-dict dry run (no
cluster needed, pure Python/numpy), then changelog.
