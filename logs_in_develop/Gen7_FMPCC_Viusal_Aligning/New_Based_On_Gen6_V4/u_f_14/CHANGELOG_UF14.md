# UF-14: Geo Constraint Sweep — Outer Loop over Geometric Constraint Configurations

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`, `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`

---

## Motivation

Prior to this update, `constraint_types` was a single top-level yaml key that applied uniformly to every evaluation run. This made it impossible to compare DPCC projection behaviour under different geometric constraint configurations (bounds-only vs bounds+dynamics) without re-running with a manually edited yaml.

The avoiding paper (published) produces results under multiple obstacle halfspace configurations via an outer `halfspace_variants` loop, generating separate output subtrees per configuration. UF-14 brings the same structure to visual aligning: an outer `geo_constraint_variants` loop produces a separate `results/{geo_name}/` subtree for each constraint configuration, so all configs are produced in a single run and are directly comparable.

**Old output structure:**
```
results/
├── diffuser/
├── dpcc-r/
└── dpcc-c/
```

**New output structure:**
```
results/
├── bounds_only_1/
│   ├── diffuser/
│   ├── dpcc-r/
│   └── dpcc-c/
└── bounds_dynamics_1/
    ├── diffuser/
    ├── dpcc-r/
    └── dpcc-c/
```

Old runs (without `geo_constraint_variants` in the yaml) are **unaffected** — the eval script falls back to a single `bounds_dynamics_1` iteration using the top-level `constraint_types`, producing the same `results/{variant}/` path as before.

---

## Design

A flat `(geo_name, geo_config, base_variant)` product list is built before the variant loop. The loop variable changes from `variant` to `geo_name, geo_config, geo_variant` — no indentation change to the ~280-line loop body. This is structurally identical to the avoiding paper's `halfspace_variants` outer loop, adapted for the 3D robot workspace.

`_2` placeholder entries are commented out in the yaml with all tunable parameters explicitly listed (workspace_bounds, enlarge_constraints), ready to be uncommented and edited for parameter sensitivity sweeps.

---

## Changed Files

### `config/visual_aligning_eval.yaml`

- Replaced single `constraint_types: [...]` with `geo_constraint_variants` list.
- Active entries: `no_constraint` (raw FM, no index — no tunable parameters), `bounds_only_1` (bounds only), `bounds_dynamics_1` (bounds + dynamics).
- Commented-out placeholder entries: `bounds_only_2`, `bounds_dynamics_2` with all tunable parameters listed inline.
- Top-level `constraint_types` retained as fallback default for scripts that do not read `geo_constraint_variants`.

### `fm_visual_aligning_test/eval_fm_visual_aligning.py`

- Added `_geo_specs` / `_run_items` product build before the variant loop (immediately after the MuJoCo cleanup block).
- `for variant in projection_variants:` → `for geo_name, geo_config, geo_variant in _run_items:`
- `save_path` updated: `results/{geo_name}/{variant}/` (and `results_train_set/{geo_name}/{variant}/`).
- Banner print `[ geo ] ── Constraint variant: {geo_name} ──` printed once at the start of each geo group.
- 3 internal `config.get(...)` calls replaced with `geo_config.get(...)`:
  - `setup_dpcc_projector(args, config, ...)` → `geo_config`
  - `max_action_delta=config.get(...)` → `geo_config`
  - `mpc_foresight_stride=config.get(...)` → `geo_config`
  - `write_to_file` check → `geo_config`

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Identical changes as FM eval above.
