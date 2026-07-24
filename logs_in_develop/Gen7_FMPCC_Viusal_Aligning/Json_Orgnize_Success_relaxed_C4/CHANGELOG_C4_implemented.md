# C4 — implemented the grouped JSON/NPZ schema from `PLAN_C4_json_npz_reorganize_metrics.md`

**Date:** 2026-07-06. Implements the plan's locked decisions: reorganize Gen7 (FM) and Gen6V4
(DDPM/diffuser) visual-aligining's per-rollout `rollout_{idx}_stats.json` into grouped
`success`/`outcome`/`timing`/`context`/`contact`/`constraint` sections, add `success_relaxed`
(Finding #5) and a first/last-contact metric + SVG markers (Findings #6/#7), extend NPZ with
the constraint/contact/outcome arrays it was previously missing entirely, and restructure
`constraint_metrics.json`'s aggregate to match. Applied **identically** to both eval scripts,
same as every prior fix in this Epoch/thread.

## Open questions resolved before implementing (not guessed)

1. **`steps` vs `exec_n_steps` divergence** — traced `check_trajectory_constraints`'s
   `T = len(pos)` where `pos = self.curr_rollout_c_pos`; both `step_counter` (`+= 1`) and
   `curr_rollout_c_pos.append(...)` happen once per call to `predict()`, i.e. once per env
   step, in the same function scope — they can never diverge. `exec_n_steps` is now dropped
   from the exported JSON; `timing.steps` is the single canonical count.
2. **Other consumers of the old flat key names** — not exhaustively provable, but no read-back
   of `rollout_{idx}_stats.json` exists inside either eval script itself (grepped, both files:
   the JSON is write-only from this code's own perspective). Left as a residual risk for any
   external Data_Analysis / notebook consumer — flagging here since the plan's Q2 couldn't be
   fully closed without a repo-wide consumer audit outside this scope.
3. **`constraint_metrics.json` aggregate** — restructured too, into the same `exec`/`plan`/
   `by_family` nesting as the per-rollout JSON (both files, `_cm_summary` block).
4. **`imf_visual_aligining_test`** — left untouched, out of scope (user named only Gen7/Gen6V4).
   Its `update_rollout_info(info)` reads `info` with `.get(...)`, so the additive
   `pos_min_dist`/`rot_min_dist` keys now threaded through `aligning_sim.py` are silently
   ignored there — confirmed no breakage.
5. **`pos_min_dist` source** — threaded through the shared `d3il/simulation/aligning_sim.py`'s
   `agent.update_rollout_info({**info, ..., 'pos_min_dist': env.pos_min_dist, 'rot_min_dist':
   env.rot_min_dist})`, read live off the env instance rather than hardcoded a second place.
   A `0.018` fallback exists in both eval scripts' `update_rollout_info` only for robustness
   against an older/unpatched `aligning_sim.py`.
6. **`record_step_info` coverage across both step branches** — confirmed in
   `d3il/simulation/aligning_sim.py`: both the visual branch (`env.step()` at line ~106) and
   the non-visual branch (line ~136) call `agent.record_step_info(info)` identically, and
   `aligining.py`'s `step()` always returns `{'mode': mode, ...}` — so the new
   `curr_rollout_mode_history` capture is populated every step, every rollout type.
7. **Never-touched contact edge case** — handled explicitly: `-1`/`None` sentinel for
   `contact_first_step`/`contact_last_step` in JSON, `-1`/`[nan, nan]` in NPZ (fixed-shape,
   numeric-only arrays) when `mode` never reaches 0 for the whole rollout. Verified via
   synthetic dry run (see Verification).

## What changed, per file

### `d3il/simulation/aligning_sim.py` (shared by fm/diffuser/imf eval scripts)
- `eval_agent()`'s call to `agent.update_rollout_info({**info, ...})` now also passes
  `'pos_min_dist': env.pos_min_dist, 'rot_min_dist': env.rot_min_dist` — additive only, so
  `imf_visual_aligining_test` (which also uses this class) is unaffected.

### `fm_visual_aligning_test/eval_fm_visual_aligining.py` and
### `diffuser_visual_aligining_test/eval_visual_aligining_dpcc.py` (identical edits, both files)
- **New per-rollout state**: `curr_rollout_mode_history` (per-step proximity `mode`, reset each
  rollout) and 5 new `history_*` accumulators (`success_relaxed`, `contact_first_step`,
  `contact_last_step`, `contact_first_pos_xy`, `contact_last_pos_xy`).
- **`record_step_info`**: now also appends `int(info.get('mode', 1))` to
  `curr_rollout_mode_history` (one-line addition, reuses the existing per-step hook — no new
  physics queries).
- **`update_rollout_info`**: computes `success_relaxed = final_xy_dist <= pos_min_dist`
  (position-only, no angle term, per the plan's explicit "don't do more than asked") and
  first/last-contact step+XY-position from `curr_rollout_mode_history`/`curr_rollout_c_pos`.
  Both are stored as new additive keys in `master_rollout_history[ridx]` (the internal flat
  dict is otherwise untouched — plots, the `.pkl` dump, and every other existing consumer of
  that dict keep working unchanged).
- **New module-level helper `_nest_constraint_metrics(cm)`**: reshapes the flat
  `exec_*`/`plan_*` dict from `check_trajectory_constraints`/rollout-end merge into
  `{exec: {..., by_family: {bounds, halfspace, obstacles}}, plan: {...}}`. Used at both the
  per-rollout JSON export and the aggregate `constraint_metrics.json`'s `per_rollout` list, so
  both artifacts share one nesting convention. Purely a presentation reshape — nothing about
  what's measured changed.
- **`_export_rollout_realtime`**: the exported `stats` dict (→ `rollout_{idx}_stats.json`) is
  now:
  ```python
  {
    'rollout_index': ..., 'mode': ...,
    'success':    {'strict': ..., 'relaxed': ...},
    'outcome':    {'mean_distance': ..., 'max_physical_tracking_error': ...},
    'timing':     {'steps': ..., 'avg_inference_time_per_replan': ...},
    'context':    {...},               # was 'context_info', renamed for consistency
    'contact':    {'first_step', 'first_pos_xy', 'last_step', 'last_pos_xy', 'note'},
    'constraint': {'exec': {..., 'by_family': {...}}, 'plan': {...}},
  }
  ```
  `exec_n_steps` is gone (Finding #1). The print block right above the JSON dump was updated to
  read from the same nested `_nested_cm` dict instead of the old flat `_cm` keys.
- **MPC-foresight SVG (`ax_xy` panel only, not `ax_3d`)**: two new `scatter()` calls mark
  first-contact (blue `*`) and last-contact (purple `X`) positions when they exist, added
  right after the existing start/end markers, with matching legend entries.
- **NPZ (`np.savez(...)`)**: added, all additive (nothing removed/renamed away from what
  existed):
  - `success_strict` (same data as existing `n_success`, new schema-consistent name — `n_success`
    itself is kept, per the block's own "legacy-compatible" comment), `success_relaxed`.
  - `outcome_max_physical_tracking_error` (alias of the existing `max_phys_error_per_rollout`).
  - `contact_first_step`, `contact_last_step` (int32, `-1` sentinel),
    `contact_first_pos_xy`, `contact_last_pos_xy` (float32 `(n_rollouts, 2)`, `[nan, nan]`
    sentinel).
  - 17 `constraint_exec_*`/`constraint_plan_*` arrays extracted from
    `agent.history_constraint_metrics` (previously **zero** constraint-axis arrays existed in
    NPZ at all — same gap class UAV had pre-Fix_10).
- **`constraint_metrics.json` aggregate (`_cm_summary`)**: restructured to
  `{variant, geo_name, seed, n_rollouts, exec: {..., by_family: {...}}, plan: {...},
  per_rollout: [...]}`, where `per_rollout` is now the same nested shape as the per-rollout
  JSON's `constraint` field (via `_nest_constraint_metrics`), not the old flat per-rollout dicts.

### `npz_analysis/analyze_npz.py`
- `HEADLINE_KEYS` appended (additive only, existing avoiding/UAV names untouched):
  `outcome_max_physical_tracking_error`, `contact_first_step`, `contact_last_step`,
  `constraint_exec_n_violated_steps`, `constraint_exec_sat_rate`,
  `constraint_exec_zero_violation`, `constraint_exec_bounds_viol_count`,
  `constraint_exec_halfspace_viol_count`, `constraint_exec_obstacle_viol_count`,
  `constraint_plan_post_viol_rate_mean`. `success_strict`/`success_relaxed` were already present
  from the UAV Fix_10 block — not duplicated. `contact_first_pos_xy`/`contact_last_pos_xy`
  deliberately **not** added as headline scalars (they're `(n,2)` arrays; `per_trial_metrics`
  would flatten x/y together into one misleading column) — still saved in the NPZ, just not
  headlined, same treatment UAV's own positional arrays get.

## Verification
- `py_compile` clean on all 4 touched `.py` files.
- **Standalone dry run** (pure Python/numpy, no torch/MuJoCo needed — matches this
  environment's constraints) of the extracted success_relaxed/contact-detection logic:
  - Normal case (contact mid-rollout, steps 3–7): correct first/last step + XY position.
  - Never-contact case (`mode` always 1): `-1`/`None` sentinels, no crash.
  - Missing `final_xy_dist` case: `success_relaxed` safely `False`, no crash.
  - Full `stats` dict (including the never-contact `None` fields) round-trips through
    `json.dumps`/`json.loads` cleanly.
  - NPZ round-trip via `np.savez`/`np.load` with the `-1`/`[nan, nan]` sentinels: values
    survive exactly, `np.isnan(...)` correctly detects the never-contact slot.
- `_nest_constraint_metrics` dry-run against a synthetic `check_trajectory_constraints` output
  (loaded via `ast`-isolated exec, no heavy deps): correct `exec`/`plan`/`by_family` nesting,
  and correct `{'exec': {}, 'plan': {}}` for the empty-input case.
- Confirmed via grep: neither eval script reads its own exported `rollout_{idx}_stats.json`
  back (write-only from this code's perspective) — no in-process breakage risk from the
  schema change.
- Confirmed via grep: edit counts (`Json_Orgnize_C4` marker, `_nest_constraint_metrics`
  occurrences) are identical between the two mirrored files (17 and 4 respectively) — the
  dual-file edit stayed in sync.
- Confirmed `imf_visual_aligining_test`'s `update_rollout_info(info)` uses `info.get(...)`
  exclusively — the two new `aligning_sim.py` info-dict keys are silently ignored there, no
  breakage.
- Full SLSQP/rollout/cluster execution untested here (no torch/MuJoCo runtime) — this fix is
  entirely at the artifact-writing/reading layer, and every piece of that layer that could be
  exercised without torch/MuJoCo was exercised above.

## What did NOT change
- What's actually measured — `check_trajectory_constraints`, `_check_planned_violations`, and
  the env's own `_check_early_termination`/`check_mode` are untouched. This is a
  presentation/organization + two additive-metric change, not a re-derivation of existing
  numbers (same principle as UAV's Fix_10).
- The internal `master_rollout_history[ridx]` dict's existing flat keys (`success`, `steps`,
  `mean_distance`, `all_candidates`, `dist_to_target`, etc.) — left alone, since the 9-panel PNG
  plot, the `.pkl` dump, and the legacy PNG rollout grid all read them directly. Only the
  *exported* `stats.json` (and the `constraint_metrics.json` aggregate) were reorganized.
- `imf_visual_aligining_test` — out of scope per the user's explicit "Gen7/Gen6V4" framing.
- No `schema_version` field added — same reasoning as UAV Fix_10 (git commit hash is the
  existing provenance mechanism).

## Files touched
- `d3il/simulation/aligning_sim.py` — `eval_agent()`'s `update_rollout_info` call site.
- `fm_visual_aligning_test/eval_fm_visual_aligining.py` — `__init__`, `reset`,
  `record_step_info`, `update_rollout_info`, `_nest_constraint_metrics` (new),
  `_export_rollout_realtime`, the MPC-foresight SVG block, the NPZ `np.savez` block, the
  `constraint_metrics.json` aggregate block.
- `diffuser_visual_aligining_test/eval_visual_aligining_dpcc.py` — identical touch points.
- `npz_analysis/analyze_npz.py` — `HEADLINE_KEYS`.
