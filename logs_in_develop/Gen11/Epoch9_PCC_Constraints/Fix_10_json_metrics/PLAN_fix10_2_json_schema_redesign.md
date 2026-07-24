# Fix_10 (2/2) — PLAN: the per-rollout JSON schema is confusing, needs redesign

**Date:** 2026-07-06. Concept/plan only — **no code in this file**. Triggered by the same
pasted JSON as `fix10_1` (the `crossed_line` invariant fix), but a separate complaint: even
once `crossed_line` is correct, the JSON itself is hard to read because it mixes **two
unrelated measurement systems** under confusingly similar names, plus a 4-way success matrix
flattened into oddly-named booleans with no visual grouping.

## The concrete confusion, from the pasted example
```json
"safe": true,
"collision_free": false,
"n_violations": 107,
"total_violations": 5.305563941640017,
"goal_reached": true,
"success": true,
"success_relaxed": false,          <- fixed in fix10_1
"success_and_constraints": false,
"success_and_constraints_relaxed": false,
```
A reader's first reaction: *"safe=true but collision_free=false — which one is it?"* They
sound like they should agree. They don't have to — but nothing in the JSON says why.

## Root cause: two independent measurement axes, named as if they were one

| | What it measures | How | Fields |
|---|---|---|---|
| **Axis A — physical ground truth** | Did the drone **actually touch** a wall/obstacle mesh in MuJoCo? | Hard contact detection (`gen._is_obstacle_contact`, `n_hit`/`n_phys`) | `safe`, `contact_frac`, `min_z`, `final_z` |
| **Axis B — declared-constraint margin truth** | Did the **flown path** cross the projector's own inflated safety margin around geo_bounds/halfspace/obstacles — a SOFTER, larger buffer than physical contact? | `_exec_constraint_violations` checking `[p_des\|p\|v]` against the scene's declared `constraint_types` geometry ⊕ `r_drone` | `collision_free`, `n_violations`, `total_violations` |

These are **expected to disagree** in general — Axis B's margin is deliberately larger/softer
than Axis A's literal mesh contact (that's the whole point of a safety margin: violating it
should usually happen well before actual contact). `n_violations=107` with
`contact_frac=0.0022` in the pasted example is exactly this: the flown path spent many steps
inside the soft margin, but only briefly grazed actual mesh. **Not a bug** — but nothing in
the naming (`safe` vs `collision_free` — near-synonyms in plain English) signals that these
are different rulers, so every reader re-derives this confusion from scratch (as the user just
did, and as I had to re-explain earlier in this session when the same axis-confusion came up
around `dynamics_only`/`model_free` metrics).

**The success fields have the same root problem**, one level up — they're a 2×2 matrix
(strict-vs-relaxed goal-reach × with-vs-without Axis-B compliance) flattened into 4
independently-named booleans with no visual grouping:

| | goal-reach: strict | goal-reach: relaxed (crossed finish line) |
|---|---|---|
| **+ Axis-B compliance** | `success_and_constraints` | `success_and_constraints_relaxed` |
| **(no Axis-B requirement)** | `success` | `success_relaxed` |

Nothing about the four flat names conveys that they're one small matrix, not four unrelated
facts.

## What's NOT being proposed here
Not touching the underlying **measurements** — Axis A and Axis B are both correct, both
useful, and deliberately different (that distinction is real and worth keeping, per the
"which metric to trust" discussion earlier in this Epoch). This plan is about **presentation/
naming**, not re-deriving what gets computed.

## Redesign options (pick one; not mutually exclusive — could phase A→B)

### Option A — additive only, zero breaking changes (do this first, low risk)
Keep every existing flat key **exactly as-is** (no downstream consumer breaks — `save_npz`,
`write_eval_log`, `results.json`, `npz_analysis/analyze_npz.py` untouched). Add:
- A one-time, static **schema legend** (e.g. `docs/UAV_ROLLOUT_JSON_SCHEMA.md` or a docstring
  block at the top of `rollout_one`) explaining the two axes and the success matrix, once,
  in one place — so nobody has to re-derive it from code every time (as happened twice this
  session already).
- Two new, purely **additive** convenience fields per rollout: `axis_a_safe` /
  `axis_b_collision_free` as aliases of `safe`/`collision_free` (or, cheaper: just the legend,
  no new fields) — optional, low value beyond the doc.
- **Cost:** near-zero. **Benefit:** stops the "which one is real safety" confusion for anyone
  reading the doc once. Doesn't fix the visual flatness of the success matrix.

### Option B — grouped/prefixed flat keys (moderate migration, clearer long-term)
Rename (not restructure — still a flat dict, so NPZ/array-based consumers are minimally
affected) with axis-revealing prefixes:
```
safe                          → phys_safe
contact_frac                  → phys_contact_frac
min_z / final_z               → phys_min_z / phys_final_z
collision_free                → constraint_collision_free
n_violations                  → constraint_n_violations
total_violations              → constraint_total_violations
goal_reached / goal_dist      → goal_reached / goal_dist        (unchanged — already unambiguous)
crossed_line                  → goal_crossed_line
success                       → success_strict
success_relaxed               → success_relaxed                  (already named for the axis it varies)
success_and_constraints       → success_strict_and_constraints
success_and_constraints_relaxed → success_relaxed_and_constraints
```
**Cost:** every consumer that reads these exact key strings needs updating — confirmed touch
points: `rollout_one`'s return dict (`FM_v3_uav_test/eval_fm_uav.py`), `save_npz` and
`write_eval_log` (`FM_v3_uav_test/eval_artifacts.py`), `results.json`'s writer, and
`npz_analysis/analyze_npz.py` (which reads **aggregate** NPZ array names like `n_success`,
`n_violations`, `collision_free_completed` — a **different**, already-aggregated naming layer,
so check whether it needs matching renames or is insulated by already being one level removed).
**Benefit:** the axis is legible from the key name alone, permanently, everywhere the JSON is
read (dashboards, ad-hoc `jq`, a colleague's first look) — no legend lookup required.

### Option C — nested/structured JSON (clearest, most disruptive)
```json
{
  "physical":   {"safe": true, "contact_frac": 0.0022, "min_z": 0.83, "final_z": 0.95},
  "constraint": {"collision_free": false, "n_violations": 107, "total_violations": 5.31},
  "goal":       {"reached": true, "dist": 0.216, "crossed_line": true},
  "success":    {"strict": true, "relaxed": true, "strict_and_constraints": false, "relaxed_and_constraints": false}
}
```
**Cost:** highest — every flat-key consumer breaks (NPZ save specifically wants flat
scalar arrays per rollout, so `save_npz` would need explicit un-nesting logic, not a
find/replace). **Benefit:** structurally impossible to misread — the grouping IS the schema.
Best reserved for a `results.json`-only change (human/dashboard-facing), keeping the NPZ
arrays flat via Option B naming underneath (i.e. B and C aren't exclusive: NPZ stays flat+
prefixed, `results.json`/per-rollout logs go nested for readability).

## DECISION (2026-07-06, user): do both, together, UAV-only — no cross-pipeline goal

Superseded the "Option A first, defer B/C" staging above. Final call:
- **`results.json` / rollout dicts → Option C (nested groups)**, as scoped in the previous
  chat turn: `rollout_one`'s return dict and `_run_variant`'s `summary` dict both restructure
  into `physical` / `constraint` / `goal` / `success` / `timing` groups.
- **NPZ arrays get renamed too** — same leaf names, flat with a matching group prefix (true
  nesting isn't natural for `np.savez`), so the two artifacts describe the same metrics with
  the same names at two different aggregation levels (one rollout vs. across all trials of a
  variant) instead of drifting apart.
- **Explicitly NOT a goal:** matching avoiding/visual-aligining's NPZ naming. Those pipelines
  keep their own existing flat names (`n_violations`, `collision_free_completed`, etc.)
  unchanged — this is a UAV-only rename. `npz_analysis/analyze_npz.py` is schema-generic (its
  own docstring: "any 1-D numeric array is treated as a per-trial metric... new/renamed keys
  are picked up automatically") so it keeps working on every pipeline's npz regardless of UAV
  renaming its own; the only adjustment needed there is additive (see below), not a rewrite.
- **No `schema_version` field** — skipped as unnecessary machinery for what is a straight
  rename (repo convention: don't add abstraction for a problem that doesn't exist yet); the
  git commit hash already printed in every SLURM job log (`GIT REV: <hash>`) is the existing,
  sufficient provenance mechanism for "which schema produced this run," same as used elsewhere
  in this Epoch (config-snapshot provenance discussion).

## Final concrete schema (both artifacts, same leaf names)

**`results.json` / per-rollout dict** (nested):
```json
{
  "scene": "...", "homotopy": "...",
  "physical":   {"safe": true, "contact_frac": 0.0, "min_z": 0.0, "final_z": 0.0},
  "constraint": {"collision_free": false, "n_violations": 0, "total_violations": 0.0},
  "goal":       {"reached": true, "dist": 0.0, "crossed_line": true},
  "success":    {"strict": true, "relaxed": true, "strict_and_constraints": false, "relaxed_and_constraints": false},
  "timing":     {"fm_ms_mean": 0.0, "fm_ms_p95": 0.0, "proj_ms_mean": 0.0,
                 "total_ms_mean": 0.0, "total_ms_p95": 0.0, "total_over_budget": 0, "budget_ms": 0.0},
  "track_err_mean": 0.0
}
```
`summary` (inside `results.json`) mirrors the same groups with `_rate`/`_mean` suffixes inside
each group (e.g. `summary['physical']['safe_rate']`, `summary['success']['strict_rate']`)
instead of today's flat `safe_rate`/`success_rate`.

**NPZ** (flat, group-prefixed — same leaf names as the JSON groups above):
```
phys_safe, phys_contact_frac, phys_min_z, phys_final_z,
constraint_collision_free, constraint_n_violations, constraint_total_violations,
goal_reached, goal_dist, goal_crossed_line,
success_strict, success_relaxed, success_strict_and_constraints, success_relaxed_and_constraints,
n_steps, obs_all, act_all, sampled_trajectories_all, args
```
Mapping is mechanical: JSON `group.leaf` ↔ NPZ `group_leaf` (`physical.safe` ↔ `phys_safe`,
`constraint.n_violations` ↔ `constraint_n_violations`, `success.strict` ↔ `success_strict`).
`goal_reached`/`goal_dist` keep their current names in both (already unambiguous, no group
prefix needed) other than `crossed_line` → `goal_crossed_line` for consistency with its
siblings.

## `npz_analysis/analyze_npz.py` — the one place needing an additive touch
`HEADLINE_KEYS` (module-level list, "printed first when present") currently has the OLD flat
names (`n_violations`, `total_violations`, `n_success_and_constraints`, `collision_free_completed`)
for avoiding/visual-aligining's still-unchanged npz files. **Append** UAV's new prefixed names
alongside them (don't replace — avoiding/visual-aligining still produce the old names):
```python
HEADLINE_KEYS += ['phys_safe', 'constraint_n_violations', 'constraint_total_violations',
                  'success_strict', 'success_strict_and_constraints']
```
Everything else in `analyze_npz.py` needs no change — confirmed schema-generic (per its own
docstring) and no other hardcoded reference to the renamed keys exists in that file (grepped).

## Files to be touched — EXHAUSTIVE (grepped every `r.get(...)`/`r['...']` rollout-dict access
## across `eval_artifacts.py`; nothing left unaccounted for)
- `FM_v3_uav_test/eval_fm_uav.py` — `rollout_one`'s return dict (nest); `_run_variant`'s
  `summary` dict (nest, matching groups).
- `FM_v3_uav_test/eval_artifacts.py`:
  - `save_npz` — `r.get('success')`, `r.get('success_relaxed')`, `r.get('success_and_constraints')`,
    `r.get('n_violations')`, `r.get('total_violations')` all move to nested paths;
    `r.get('n_fm_steps')`/`obs_traj`/`act_traj`/`plans` are UNCHANGED (not part of any renamed
    group).
  - `plot_overview` — only reads `obs_traj`/`homotopy`, neither renamed — **no change needed**.
  - `write_eval_log` — `r.get('success')`, `r.get('success_relaxed')`, `r.get('contact_frac')`,
    `r.get('min_z')`, `r.get('goal_dist')` move to nested paths; `homotopy`/`track_err_mean`
    stay top-level, unchanged.
  - `save_rollout_stats` — **no change needed**, re-serializes whatever dict shape it's given.
- `FM_v3_uav_test/aggregate_scene_summaries.py` — `METRICS` list and the `summ.get(m)` access
  pattern need updating to the new nested `summary` groups (e.g. `success_rate` →
  `summary.get('success', {}).get('strict_rate')`).
- `npz_analysis/analyze_npz.py` — `HEADLINE_KEYS` gets the new names appended (additive only,
  old avoiding/visual-aligining names stay for their own npz outputs).

No other file in the repo reads these exact per-rollout/summary flat keys (confirmed earlier
in this Epoch when auditing `npz_analysis`/`aggregate_scene_summaries.py` for the config-
snapshot work) — this list is the complete blast radius.
