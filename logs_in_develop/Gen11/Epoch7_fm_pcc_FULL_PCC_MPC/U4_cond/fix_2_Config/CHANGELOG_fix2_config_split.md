# CHANGELOG — fix_2_Config: massive config file restructure to match avoiding-d3il.py pattern

**Date**: 2026-06-28
**Parent**: [../CHANGELOG_U4_cond.md](../CHANGELOG_U4_cond.md)

---

## Root cause: the structural bug

`config/avoiding-d3il.py` (the source we forked from) uses a **two-level split**:

| File | What it contains |
|---|---|
| `config/avoiding-d3il.py` | Training blocks **AND** plan/eval blocks (one file); reads threshold from yaml at top |
| `config/projection_eval.yaml` | **Projection-only**: variants, constraints, geometry (halfspaces, obstacles, bounds) |

Our UAV config broke this pattern catastrophically:

| File (before) | What was actually inside | What was WRONG |
|---|---|---|
| `config/uav.py` | Training block ONLY | **Missing the plan/eval block entirely** |
| `config/uav_eval.yaml` | Everything else dumped in one place: `batch_size`, `control_hz`, `behavior_log`, `reanchor_alpha`, `lead_gain`, `write_to_file` (eval params) **AND** `projection_variants`, `constraint_types`, `dt`, `enlarge_constraints`, obstacle/halfspace geometry (projection params) | **Wrong file, wrong split, not the pattern** |

Result: `eval_fm_uav.py` loaded ALL eval params from a YAML (anti-pattern), making the
eval config invisible to the `watch`/args machinery and impossible to version-control with
the checkpoint path.

---

## Fix: the two correct files

```
config/uav.py                  ← training block + plan/eval block   (one file, like avoiding-d3il.py)
config/uav_projection.yaml     ← projection-only                    (like projection_eval.yaml)
config/uav_eval.yaml           ← DEPRECATED stub (safe to delete)
```

---

## Files changed

### `config/uav.py` — two additions

**1. YAML import at top** (matches `avoiding-d3il.py` lines 1–12):
```python
import yaml
with open('config/uav_projection.yaml', 'r') as f:
    _proj_config = yaml.safe_load(f)
if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError("CRITICAL: ...")
_yaml_threshold = _proj_config['diffusion_timestep_threshold']
```
The plan block uses `_yaml_threshold` so the threshold is always in sync between yaml and plan block — same as `avoiding-d3il.py` does with `projection_eval.yaml`.

**2. `plan_flow_matching_v3_uav` block** added to `base`:

| Key | Value | Moved from |
|---|---|---|
| `batch_size` | 4 | `uav_eval.yaml` |
| `diffusion_timestep_threshold` | `_yaml_threshold` | `uav_eval.yaml` |
| `write_to_file` | True | `uav_eval.yaml` |
| `behavior_log` | True | `uav_eval.yaml` |
| `control_hz` | 33 | `uav_eval.yaml` |
| `reanchor_alpha` | 0.0 | `uav_eval.yaml` (U4) |
| `lead_gain` | 1.0 | `uav_eval.yaml` (U4) |
| `diffusion`, `horizon`, `cond_mode`, `time_beta_*` | match training block | NEW (standard plan fields) |
| `diffusion_loadpath` | `f:flow_matching_v3_uav/H{horizon}_D{diffusion}_cond{cond_mode}` | NEW |
| `diffusion_epoch` | `'latest'` | NEW |
| `prefix` | `'plans/flow_matching_v3_uav/'` | NEW |
| `exp_name` | `watch(args_to_watch)` (same as training) | NEW |

### `config/uav_projection.yaml` — NEW file

Contains ONLY projection params (everything that stays in the yaml):

| Key | Value |
|---|---|
| `diffusion_timestep_threshold` | 0.5 (source-of-truth; plan block reads it as `_yaml_threshold`) |
| `projection_variants` | all 13 variants (same as before) |
| `constraint_types` | `['dynamics']` |
| `dt` | 1.0 |
| `enlarge_constraints` | 0.025 |
| `workspace_bounds` | null (placeholder) |
| `halfspace_constraints` | [] (placeholder) |
| `obstacle_constraints` | [] (placeholder) |

### `config/uav_eval.yaml` — DEPRECATED stub

Replaced with a comment explaining where each param moved. Safe to delete on the cluster
once all jobs are updated.

### `FM_v3_uav_test/eval_fm_uav.py` — minimal surgical changes

**`load_pcc_config()` signature change**: now `load_pcc_config(scene, seed)`.

The function now loads from TWO sources and merges:
1. **Projection params** from `config/uav_projection.yaml` (variants, constraints, geometry)
2. **Eval control params** from `plan_flow_matching_v3_uav` block via `parse_args(experiment='plan_flow_matching_v3_uav', seed=seed)`

Merged into the same dict shape as before, so `_run_variant`, `rollout_one`, and
`setup_dpcc_projector` are **unchanged**.

**`eval_scene()` call site**: `load_pcc_config()` → `load_pcc_config(scene, args.seed)`.

**Comment in `_run_variant`**: updated reference from `uav_eval.yaml` to plan block.

---

## Boundary rule (permanent reference)

| Param type | Goes in | Example |
|---|---|---|
| Projection variants | `uav_projection.yaml` | `projection_variants: [dpcc-r, ...]` |
| Constraint geometry | `uav_projection.yaml` | `obstacle_constraints:`, `halfspace_constraints:` |
| Projection tuning | `uav_projection.yaml` | `dt:`, `enlarge_constraints:` |
| Policy threshold | `uav_projection.yaml` + plan block (`_yaml_threshold`) | `diffusion_timestep_threshold:` |
| Eval scalars | `plan_flow_matching_v3_uav` block in `uav.py` | `batch_size`, `control_hz`, `behavior_log` |
| U4 grounding knobs | `plan_flow_matching_v3_uav` block in `uav.py` | `reanchor_alpha`, `lead_gain` |
| Dataset/model params | `flow_matching_v3_uav` block in `uav.py` | `cond_mode`, `normalizer`, `max_path_length` |
| Scene-specific geometry | future `uav_projection.yaml` scene dict | (same pattern as `projection_eval.yaml`) |

---

## Verification

- `py_compile` passes: `config/uav.py`, `eval_fm_uav.py`, `train_fm_uav.py`
- `uav_projection.yaml` parses: `threshold=0.5`, `13 variants`
- `train_fm_uav.py` is **unchanged** (it only uses the training block `flow_matching_v3_uav`)

---

## How to revert

```bash
git checkout -- config/uav.py FM_v3_uav_test/eval_fm_uav.py config/uav_eval.yaml
rm config/uav_projection.yaml
```
