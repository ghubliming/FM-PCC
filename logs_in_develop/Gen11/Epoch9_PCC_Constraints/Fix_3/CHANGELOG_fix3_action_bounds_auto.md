# Epoch 9 Fix_3 — `action_bounds` self-derived (`'auto'`), not a copied/guessed number

**Date:** 2026-07-04. Resolves the question: can we just reuse DPCC-avoiding's `bounds`
value (`['vx','vy']` limits in `config/projection_eval.yaml`) for the UAV's action-magnitude
guard? **No** — but the underlying *method* transfers, and the codebase already computes
exactly what's needed to apply it correctly, so no cluster measurement is required either.

## Why avoiding's number can't be reused

Avoiding's `bounds` (`{'vx':[-0.01,0.01], 'vy':[0,0.01]}`, tightened `[-0.012,0.012]`) is not
a universal constant — it's a value fitted to **avoiding's own dataset**. Its own config
comment says so: *"need to be within the limits of the dataset due to the normalization."*
The UAV is a different robot, a different workspace scale (arena ~6-7 m vs. avoiding's ~0.5 m
tabletop), and a different expert flight speed. Copying the *number* would either clip
legitimate UAV steps (too tight) or do nothing (too loose) — there's no reason the two
datasets' action ranges should coincide.

## The fix — reuse the METHOD, not the number

`LimitsNormalizer` (`flow_matcher_v3_uav/datasets/normalization.py:96-97`) already computes
exactly the quantity avoiding's number was hand-approximating, correctly, per dataset:
```python
self.mins = X.min(axis=0)
self.maxs = X.max(axis=0)
```
fit directly from the training data at load time. This **is** "the limits of the dataset,"
computed by code instead of guessed by hand. The codebase already has this self-calibration
pattern in exactly one other place — `pid_const_v` in `_run_variant` derives its speed from
`dataset.fields.actions` the same way.

### `FM_v3_uav_test/eval_fm_uav.py::setup_dpcc_projector`
The action-bound block now reads `config.get('action_bounds', 'auto')` and branches three ways:
```python
ab = config.get('action_bounds', 'auto')
if ab is None:
    a_lb = a_ub = None                              # bound disabled
elif ab == 'auto':
    a_lb = np.asarray(act_normalizer.mins, dtype=float)   # dataset's own Δp_des min
    a_ub = np.asarray(act_normalizer.maxs, dtype=float)   # dataset's own Δp_des max
else:
    a_lb = np.array(ab['lb'], dtype=float)          # explicit hand-picked override
    a_ub = np.array(ab['ub'], dtype=float)
```
Everything downstream (padding to `trajectory_dim`, `['lb'/'ub']` rows) is unchanged.

### `config/uav_projection.yaml`
```yaml
action_bounds: 'auto'
```
replaces the old hardcoded placeholder `{lb: [-0.05,-0.05,-0.05], ub: [0.05,0.05,0.05]}`.
**Still user-settable** — three forms, documented inline in the yaml:
- `'auto'` (**default, recommended**) — self-derived from `act_normalizer.mins/.maxs`.
- `{lb: [...], ub: [...]}` — explicit override, e.g. to deliberately test a tighter/looser cap.
- `null` — disables the action-magnitude bound entirely (workspace box/halfspace/obstacles
  are unaffected — they're independent families).

## Why this fully resolves the earlier "unverified placeholder" risk

The prior placeholder (±0.05) needed a cluster-side measurement step before any sweep could
be trusted (flagged in `init/CHANGELOG_E9_PCC_constraints.md` and the U2 discussion). `'auto'`
removes that step entirely: the bound is computed from the same normalizer object
`setup_dpcc_projector` already receives as an argument, at eval time, from whichever
dataset/scene is actually loaded — there is nothing left to confirm by hand. This also means
the bound self-corrects if the dataset is ever re-collected or extended — no yaml edit needed.

(`workspace_bounds` and `r_drone`/`inflation` remain separately-flagged unverified geometry
numbers — this fix only closes the action-bounds gap.)

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- `config/uav_projection.yaml` parses; `action_bounds == 'auto'` confirmed via `yaml.safe_load`.
- Branch logic (`None` / `'auto'` / explicit dict) traced by hand for all three cases —
  correct in each. Cannot execute the `'auto'` branch itself here (needs a real
  `act_normalizer` from a loaded dataset — torch/MuJoCo runtime, cluster-only).

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `setup_dpcc_projector` action-bounds block + docstring.
- `config/uav_projection.yaml` — `action_bounds: 'auto'` + explanatory comment (replaces the
  hardcoded placeholder).
