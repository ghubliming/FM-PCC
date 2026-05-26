# UF-14: Geo Constraint Sweep — Outer Loop over Geometric Constraint Configurations

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`, `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`

---

## Motivation

Prior to this update, `constraint_types` was a single top-level yaml key applied uniformly to every evaluation run. This made it impossible to compare DPCC projection under different geometric constraint configurations without manually editing the yaml between runs.

The avoiding paper (published) produces results under multiple obstacle configurations via an outer `halfspace_variants` loop, generating separate output subtrees per configuration. UF-14 brings the same structure to visual aligning: a `geo_constraint_variants` outer loop produces a separate `results/{name}/` subtree for each constraint configuration, so all configurations run in a single job and are directly comparable.

---

## Output path structure

**Before UF-14:**
```
results/diffuser/
results/dpcc-r/
results/dpcc-c/
...
```

**After UF-14:**
```
results/no_constraint/diffuser/
results/no_constraint/dpcc-r/
results/dynamics_only/diffuser/
results/bounds_only_1/diffuser/
results/combined_2/diffuser/
...
```

Old runs without `geo_constraint_variants` in the yaml are **not affected** — the eval script falls back to a single `bounds_dynamics_1` iteration using the top-level `constraint_types`, producing the same path structure as before.

---

## Geo constraint naming scheme

| Tier | Name pattern | Index | Active by default |
|---|---|---|---|
| Baseline | `no_constraint` | none | ✅ |
| Single-type ablation | `dynamics_only` | none | ✅ |
| Single-type ablation | `bounds_only_1/2` | `_1/_2` | ✅ `_1` / commented `_2` |
| Single-type ablation | `obstacle_only_1/2` | `_1/_2` | ❌ both commented (geometry not measured) |
| Combination | `combined_1/2/3` | N | ✅ `combined_2` / ❌ obstacle ones commented |

- **No index** — no tunable geometric parameters (`no_constraint`, `dynamics_only`)
- **`_1/_2`** — parameter variants: `_1` = current baseline, `_2` = tuning placeholder with all parameters listed inline
- **`combined_N`** — flexible numbered slots; content defined entirely by `constraint_types` (and `obstacle_constraints` if needed)

### Active config set

| Name | `constraint_types` | Notes |
|---|---|---|
| `no_constraint` | `[]` | Raw FM baseline |
| `dynamics_only` | `['dynamics']` | Euler link only |
| `bounds_only_1` | `['bounds']` | Workspace box only |
| `combined_2` | `['dynamics', 'bounds']` | DPCC-equivalent for this task |

### Commented-out configs (disabled)

| Name | `constraint_types` | Reason |
|---|---|---|
| `bounds_only_2` | `['bounds']` | Parameter tuning placeholder |
| `obstacle_only_1` | `['obstacles']` | Obstacle geometry not measured |
| `obstacle_only_2` | `['obstacles']` | 3D sphere variant placeholder |
| `combined_1` | `['dynamics', 'obstacles']` | Obstacle geometry not measured |
| `combined_3` | `['dynamics', 'bounds', 'obstacles']` | Obstacle geometry not measured |

---

## Classic DPCC vs visual aligning constraint mapping

Classic DPCC (`config/projection_eval.yaml`): `['halfspace', 'obstacles', 'dynamics', 'bounds']`

| Type | Avoiding (published) | Visual aligning | Reason |
|---|---|---|---|
| `halfspace` | ✅ triangular walls | ❌ | Open table, no diagonal walls |
| `obstacles` | ✅ spherical obstacles | ⏸ designed, disabled | Geometry not measured yet |
| `bounds` | ✅ 2D xy limits | ✅ 3D xyz box | Extended to 3D |
| `dynamics` | ✅ Euler link 2D | ✅ Euler link 3D | Extended to 3D |

`combined_2` (`['dynamics', 'bounds']`) is the current DPCC-equivalent for this task.
`combined_3` (`['dynamics', 'bounds', 'obstacles']`) will be the full equivalent once obstacle geometry is confirmed.

---

## Changed Files

### `config/visual_aligning_eval.yaml`

- Replaced old `geo_constraint_variants` block (flat list of `bounds_only_1`, `bounds_dynamics_1`, ad-hoc obstacle entries) with a structured 3-tier layout: Baseline / Single-type ablation / Combinations.
- Active entries: `no_constraint`, `dynamics_only`, `bounds_only_1`, `combined_2`.
- Commented-out entries: `bounds_only_2`, `obstacle_only_1`, `obstacle_only_2`, `combined_1`, `combined_3` — all with tunable parameters listed inline as edit targets.
- Top-level `constraint_types` retained as fallback.

### `fm_visual_aligning_test/eval_fm_visual_aligning.py`

- Added `_geo_specs` / `_run_items` product build before the variant loop (after MuJoCo cleanup block).
- `for variant in projection_variants:` → `for geo_name, geo_config, geo_variant in _run_items:` — no indentation change to the 280-line loop body.
- `save_path`: `results/{geo_name}/{variant}/` (and `results_train_set/{geo_name}/{variant}/`).
- Banner print `[ geo ] ── Constraint variant: {geo_name} ──` at the start of each geo group.
- 4 internal `config.get(...)` calls → `geo_config.get(...)`: `setup_dpcc_projector`, `max_action_delta`, `mpc_foresight_stride`, `write_to_file`.
- `setup_dpcc_projector`: added `_DIM` named-dimension map (`'x'`→6, `'y'`→7, `'z'`→8, etc.) and `'obstacles'` block that reads `obstacle_constraints` list from `geo_config` and appends `(type, dims, center, radius)` tuples to the projector constraint list.
- Docstring updated to include obstacle exclusion.

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Identical changes as FM eval above.
