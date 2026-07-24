# Fix_11 (Gen7/Gen6V4) — progress logging for visual-aligining eval jobs

**Date:** 2026-07-06. Same complaint as UAV's `Epoch9_PCC_Constraints/Fix_11`: the per-rollout
debug block (`update_rollout_info`'s "Context N Finished" print) is good and stays untouched,
but nothing in either eval script says how many (geo, variant) combos or rollouts are left, or
how long they'll take — so a killed/timed-out job gives no indication of where it was. Applied
the same fix, identically to both `fm_visual_aligning_test/eval_fm_visual_aligining.py` (Gen7)
and `diffuser_visual_aligining_test/eval_visual_aligining_dpcc.py` (Gen6V4).

## Loop structure traced first (not guessed)

Unlike UAV's 4-deep nested loop, this eval script already flattens geo-variant × projection-
variant (+ auto tightened twins) into one list, `_run_items`, before the main loop:
```
for seed in seeds:                                          # bash-style outer loop, in Python
  _run_items = [(geo_name, geo_config, variant, is_tightened), ...]   # flat cartesian product
  for geo_name, geo_config, geo_variant, is_tightened in _run_items:
      agent = VisualAgentWrapper(...)                        # one agent per item
      sim.test_agent(agent)                                  # internally: n_contexts × n_trajectories rollouts
```
`sim.test_agent()` (`d3il/simulation/aligining_sim.py`) already calls
`agent.update_rollout_info(...)` once per rollout, which already prints a per-rollout debug
block — that's the "good" part the user referred to. What was missing: (1) an index/total for
the `_run_items` loop, (2) an index/total for the seed loop, (3) any progress *within* one
item's `n_contexts × n_trajectories` rollouts (the debug block prints the context index, but
never a total or elapsed/ETA).

## What changed, per file (identical edits, both `eval_fm_visual_aligining.py` and
## `eval_visual_aligining_dpcc.py`)

- **Seed loop**: `for seed in seeds:` → `for _seed_i, seed in enumerate(seeds):`, print changed
  from `=== Evaluating seed {seed} ===` to `=== Evaluating seed {i+1}/{N} (seed={seed}) ===`.
- **`_run_items` loop**: enumerated; prints `[ eval ] >>> item X/N: geo=... variant=...
  tightened=...` before each item starts — this is the line to grep after a killed job; the
  last one printed is the (geo, variant) combo that was running when it got cut.
- **`VisualAgentWrapper`**: new constructor param `total_rollouts=None` (stored as
  `self.total_rollouts`), plus `self._item_t0 = time.time()` set at construction — both added
  right next to the existing `self.is_tightened` line.
- **Construction call site**: passes `total_rollouts=n_contexts * n_trajectories` (both values
  were already local variables in scope at the call site, read from the eval yaml earlier in
  the same function).
- **`update_rollout_info`'s existing debug block**: one new line appended at the end (right
  before the `'-'*80` separator, right after "Clamp events"), left everything above it
  untouched:
  ```
  - Progress: rollout {done}/{total_rollouts}  ({elapsed:.1f}s elapsed this item, ~{eta:.1f}s to go)
  ```
  `done = self.rollout_counter + 1` — `rollout_counter` is 0-based and already incremented by
  this rollout's own `reset()` call, so it's the count of rollouts completed so far regardless
  of context/trajectory indexing (unlike the `ridx`/context index already printed above it,
  which can repeat across `n_trajectories_per_context > 1`).

## Verification
- `py_compile` clean on both eval scripts.
- Confirmed via grep: identical edit counts in both files (`Fix_11` marker: 6/6,
  `total_rollouts`: 7/7) — the dual-file mirror stayed in sync.
- Elapsed/ETA print-formatting logic is the same pattern already dry-run-verified for UAV's
  Fix_11 (monotonic elapsed increase, monotonic ETA decrease to ~0 at the last rollout) — not
  re-derived here since it's identical arithmetic.
- Confirmed `time` is already imported at module level in both files (no new import needed).
- Full cluster execution untested here (no torch/MuJoCo runtime in this environment) — this
  change is print-only plus one new constructor kwarg with a safe default (`None` → the
  progress line is simply skipped), so it cannot affect rollout behavior, artifacts, or return
  values.

## What did NOT change
- The existing per-rollout debug block content (context idx, steps, success, mean distance,
  mode, tracking error, inference time, clamp events) — untouched, per the user's explicit
  "the debug info which is good" framing.
- No logic, timing behavior, output artifacts (JSON/NPZ/GIF/SVG), or return values.
- Did not touch `imf_visual_aligining_test` — out of scope, same as the rest of this C4 thread
  (user named only Gen7/Gen6V4).

## Files touched
- `fm_visual_aligning_test/eval_fm_visual_aligining.py` — `VisualAgentWrapper.__init__`,
  `update_rollout_info`, the seed loop, the `_run_items` loop, the construction call site.
- `diffuser_visual_aligining_test/eval_visual_aligining_dpcc.py` — identical touch points.
