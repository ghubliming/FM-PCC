# Patch_Constraints_C3 — DANGEROUS FIX: split `bounds` into `geo_bounds` + restored `bounds`

**Date:** 2026-07-04. Resolves `PROBLEM_bounds_velocity_vs_geo.md` (same folder): visual-aligning
had silently dropped DPCC-avoiding's action-magnitude guard when it repurposed the `bounds`
constraint_types flag for a Cartesian position box. Applies to **both** live consumers of the
shared `config/visual_aligning_eval.yaml`:
- **Gen7**: `fm_visual_aligning_test/eval_fm_visual_aligning.py` (FM ODE engine)
- **Gen6V4**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (DDPM engine) — Gen7's
  own file header says its logging pattern is "reused verbatim from Gen6V4"; this script is
  that ancestor, still live (not archived), and reads the **same** yaml
  (`with open('config/visual_aligning_eval.yaml', 'r') as f:`) with its **own copy** of
  `setup_dpcc_projector` carrying the **identical** bug. Both needed the same fix.

**Marked "dangerous"** because `constraint_types: ['bounds', ...]` is a widely-referenced
string across two Python files (each with ~9 call sites: the projector, exec-metric checks,
plot overlays, and yaml-default fallbacks) and the shared yaml's `geo_constraint_variants`
list — and the fix changes the **currently active** entry (`combined_5`)'s enforced behavior,
not just an inactive one.

---

## The rename (both files, identical)

| Old | New | Meaning |
|---|---|---|
| `'bounds'` (old) | `'geo_bounds'` | Cartesian workspace box on ACTUAL position (dims 6,7,8), reads `config['workspace_bounds']`. This is what `'bounds'` used to mean. |
| `'bounds'` (new) | `'bounds'` (restored) | DPCC's TRUE meaning: an action-magnitude limit on dims 0,1,2 (dx,dy,dz). |

Every one of the ~9 call sites per file that gated the geo box on `'bounds'` was renamed to
`'geo_bounds'` (`setup_dpcc_projector`, `plot_geo_constraints`'s `has_bounds`,
`check_trajectory_constraints`, `_check_planned_violations`, the two rollout-plot geometry
overlays, the yaml-default fallback tuple, and the tightening-eligibility check). A plot
legend label was also relabeled `'bounds'` → `'geo bounds'` for clarity (cosmetic only).

## The restoration — faithful to DPCC, not copied from avoiding

New `if 'bounds' in constraint_types:` block in both `setup_dpcc_projector`s:
```python
if 'bounds' in config.get('constraint_types', []):
    ab = config.get('action_bounds', 'auto')
    if ab == 'auto':
        a_lb = np.asarray(act_normalizer.mins, dtype=float)   # dataset's own dx,dy,dz range
        a_ub = np.asarray(act_normalizer.maxs, dtype=float)
    elif ab is not None:
        a_lb = np.array(ab['lb'], dtype=float)                # explicit override
        a_ub = np.array(ab['ub'], dtype=float)
    else:
        a_lb = a_ub = None                                    # disabled
    if a_lb is not None:
        lb = np.concatenate([a_lb, np.full(trajectory_dim - 3, -np.inf)])
        ub = np.concatenate([a_ub, np.full(trajectory_dim - 3,  np.inf)])
        constraint_list.append(['lb', lb]); constraint_list.append(['ub', ub])
```
**Self-derived (`'auto'`, the default), not avoiding's hardcoded number.** Avoiding's
`['vx','vy']` bound (`{-0.01,0.01}` etc.) was fitted to avoiding's own dataset — a different
robot, different scale, different expert speed. Copying that number here would be arbitrary.
`act_normalizer.mins/.maxs` (`LimitsNormalizer`, `mins/maxs = X.min/max(axis=0)` over the
training data) computes exactly what avoiding's number was hand-approximating, correctly, for
*this* dataset — mirrors the UAV E9 Fix_3 pattern
(`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_3/`), applied back to the task where the
original problem was first found.

**Not tightened.** The action bound is excluded from the `-tightened` enlarge margin (it's a
dataset-range cap, not a spatial surface) — mirrored in the tightening-eligibility check
(`_has_geo`), which still only fires on `'geo_bounds'`/`'halfspace'`/`'obstacles'`.

## `config/visual_aligning_eval.yaml` changes

- New top-level `action_bounds: 'auto'` (settable: `'auto'` / explicit `{lb,ub}` / `null`).
- `bounds_only_1`/`bounds_only_2` → renamed `geo_bounds_only_1`/`geo_bounds_only_2`,
  `constraint_types: ['bounds']` → `['geo_bounds']` (these ablate the position box, not
  action limits — the rename makes that explicit).
- **New** `action_bounds_only` entry (`constraint_types: ['bounds']`) — restores DPCC
  Table-1 ablation parity: the true action-magnitude bound now has its own isolated test slot,
  which didn't exist before (the old `bounds_only_*` entries were actually geo-box ablations
  mislabeled as "bounds").
- `combined_2`/`combined_3`: `'bounds'` → `'geo_bounds'` (rename only — these never claimed to
  be "the full DPCC set", so no restoration needed).
- `combined_4`/`combined_5`: `'bounds'` → `'geo_bounds'` **plus** `'bounds'` added back
  (`constraint_types: ['dynamics','geo_bounds','halfspace','obstacles','bounds']`) — both
  explicitly claim to be "Full DPCC constraint set (matches avoiding paper...)", so they must
  carry the action-magnitude limit DPCC actually had. **`combined_5` is the CURRENTLY ACTIVE
  entry** (`active_geo_variants: [combined_5]`) — this is the one live-behavior change from
  this fix: the next run under `combined_5` will enforce a self-derived action-magnitude cap
  it previously did not.
- Comment blocks updated throughout (`enlarge_constraints` doc, the commented-out
  `bounds:`/`action_indices` blocks, the "All available entries" list) to describe the new
  split and point at this changelog.

## Why this belongs to both Gen7 and Gen6V4, and not just one

Both eval scripts are separate Python files but **share one yaml** and carry **structurally
identical** copies of `setup_dpcc_projector` and every downstream `'bounds'`-gated function
(confirmed line-for-line: same ~9 call sites, same variable names `_ct`/`_gc`/`ct`/`geo_config`
in both files). Fixing only the yaml would have silently broken Gen6V4 (its code would still
read the old `'bounds'`-means-geo-box logic against a yaml that no longer has any entry using
`'bounds'` for that purpose after the rename) or left Gen7 with the original conflation if only
Gen6V4's copy were patched. Both files needed the identical code change to stay consistent
with the one shared config.

## Verification
- `py_compile` clean on both `fm_visual_aligning_test/eval_fm_visual_aligning.py` and
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`.
- `yaml.safe_load` on `config/visual_aligning_eval.yaml`: confirmed `action_bounds: 'auto'`
  present; every `geo_constraint_variants` entry's `constraint_types` printed and checked
  against the intended rename/restoration (see table below).
- Simulated the exact branch logic (`'geo_bounds'`/`'bounds'` dispatch, `'auto'` self-derivation
  against a fake normalizer) for all 14 entries — every entry resolves correctly:

| Entry | geo box | action bound |
|---|---|---|
| `no_constraint`, `dynamics_only`, `obstacle_only_*`, `halfspace_only_1`, `combined_1` | — | — |
| `geo_bounds_only_1`, `geo_bounds_only_2` | ✅ | — |
| `action_bounds_only` | — | ✅ (auto) |
| `combined_2`, `combined_3` | ✅ | — |
| `combined_4`, `combined_5` (**active**) | ✅ | ✅ (auto) |

- Full SLSQP/rollout path untested here (needs torch/MuJoCo, cluster-only) — this fix is
  config/wiring-level and independently verifiable, as done above.

## Files touched
- `config/visual_aligning_eval.yaml` — `action_bounds` key added; `geo_constraint_variants`
  entries renamed/restored/added as above; explanatory comments updated.
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` — `setup_dpcc_projector` split +
  restored action-bound block; ~8 other `'bounds'` call sites renamed to `'geo_bounds'`.
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — identical set of changes
  (Gen6V4's live copy of the same vulnerable function).
