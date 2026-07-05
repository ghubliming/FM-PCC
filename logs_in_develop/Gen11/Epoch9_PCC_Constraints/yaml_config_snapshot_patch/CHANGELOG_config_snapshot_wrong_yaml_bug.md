# Config-snapshot bug — every per-package `setup.py` had a hardcoded, wrong yaml path

**Date:** 2026-07-04. A cross-cutting bug, found in two independent places by the user
auditing run provenance, then confirmed to be a repo-wide pattern across several packages'
`utils/setup.py` copies.

## The bug (one line, repeated across packages)

Every package under this repo has its own copy of `utils/setup.py` (not shared code — each
package forked its own). `snapshot_configs()` is meant to copy that package's **own** live
eval yaml into `.../config_snapshot_<config_name>/` at eval time, so a run's provenance can be
audited later. In several copies, the yaml path was **hardcoded to a different package's
yaml** — almost certainly copy-paste residue from whichever package a given one was forked
from, never updated for the fork's own config file.

Symptom: a saved snapshot like
`.../config_snapshot_uav/projection_eval.yaml` (avoiding-d3il's yaml, inside a **UAV** run's
snapshot folder) or
`.../config_snapshot_avoiding-d3il-visual/visual_aligning_eval.yaml` (visual-**aligning's**
yaml, inside a visual-**avoiding** run's snapshot folder) — the snapshot exists, has a
plausible-looking filename, but is the **wrong file's contents**, making it useless (or
actively misleading) for reconstructing what config actually produced that run.

Both were found because the *y*aml a snapshot claimed to be didn't match the package that
produced it — always exists (`config/projection_eval.yaml` and the others all live in the same
shared `config/` directory) so `os.path.exists(yaml_path)` never caught the mismatch; the
snapshot just silently copied the wrong, but real, file every single time.

## Audit — every package's `setup.py`, checked for this exact pattern

| Package | Hardcoded `yaml_path` | Package's REAL eval yaml (confirmed by grep) | Status |
|---|---|---|---|
| `flow_matcher_v3_uav` | `config/projection_eval.yaml` | `config/uav_projection.yaml` | ❌ **WRONG → FIXED** |
| `fm_visual_avoiding` | `config/visual_aligning_eval.yaml` | `config/visual_avoiding_eval.yaml` | ❌ **WRONG → FIXED** |
| `diffuser_visual_avoiding` | `config/visual_aligning_eval.yaml` | `config/visual_avoiding_eval.yaml` | ❌ **WRONG → FIXED** |
| `fm_visual_aligning` | `config/visual_aligning_eval.yaml` | `config/visual_aligning_eval.yaml` | ✅ already correct |
| `imf_visual_aligning` | `config/visual_aligning_eval.yaml` | `config/visual_aligning_eval.yaml` | ✅ already correct |
| `diffuser` | `config/projection_eval.yaml` | `config/projection_eval.yaml` (avoiding-d3il DDPM) | ✅ already correct |
| `flow_matcher_v3_drifting` | `config/projection_eval.yaml` | `config/projection_eval.yaml` (avoiding-d3il family) | ✅ already correct |
| `flow_matcher_v3_imeanflow` | `config/projection_eval.yaml` | `config/projection_eval.yaml` (avoiding-d3il family) | ✅ already correct |
| `flow_matcher_v3_ode_selectable` | `config/projection_eval.yaml` | `config/projection_eval.yaml` (avoiding-d3il family) | ✅ already correct |
| `flow_matcher`, `flow_matcher_v2`, `flow_matcher_unet_v2`, `flow_matcher_v3` | — (no yaml step) | n/a | not applicable — no `snapshot_configs` yaml step at all (older packages, predate this feature) |
| `ddpm_encdec_vision`, `fm_encdec_vision` | dynamic (`'visual'`/`'aligning'` substring heuristic → picks `visual_aligning_eval.yaml` or `projection_eval.yaml`) | n/a | ⚠️ **not fixed — flagged below** |

## What was fixed

### `flow_matcher_v3_uav/utils/setup.py`
```python
yaml_path = 'config/projection_eval.yaml'   # WAS: avoiding-d3il's yaml
```
→
```python
yaml_path = 'config/uav_projection.yaml'    # NOW: UAV's own yaml
```
(destination filename inside the snapshot folder updated to match: `uav_projection.yaml`).
Full detail: `../Fix_6/CHANGELOG_fix6_multi_geo_variant_per_job.md` (addendum).

### `fm_visual_avoiding/utils/setup.py` and `diffuser_visual_avoiding/utils/setup.py`
```python
yaml_path = 'config/visual_aligning_eval.yaml'   # WAS: visual-ALIGNING's yaml
```
→
```python
yaml_path = 'config/visual_avoiding_eval.yaml'   # NOW: visual-avoiding's own yaml
```
Confirmed against the actual eval scripts each package runs:
`fm_visual_avoiding_test/eval_fm_visual_avoiding.py` and
`diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` both explicitly
`open('config/visual_avoiding_eval.yaml')` — neither ever reads `visual_aligning_eval.yaml`.
The `diffuser_visual_avoiding` copy's stale comment even named the wrong script
(`eval_visual_aligning_dpcc.py`), confirming it was a copy-paste fork of the visual-aligning
package's `setup.py` that never got its yaml reference updated.

## What was checked and found already correct

`fm_visual_aligning`/`imf_visual_aligning` (visual-aligning's own packages) correctly point at
`visual_aligning_eval.yaml` — no bug. `diffuser`, `flow_matcher_v3_drifting`,
`flow_matcher_v3_imeanflow`, `flow_matcher_v3_ode_selectable` all correctly point at
`projection_eval.yaml` — all four are members of the avoiding-d3il model family
(`config/avoiding-d3il.py`'s `base` dict lists all of them), for which that really is the right
yaml.

## What was NOT fixed — flagged, not acted on

`ddpm_encdec_vision`/`fm_encdec_vision` use a **heuristic** instead of a hardcoded path:
```python
if 'visual' in getattr(args, 'dataset', '') or 'visual' in getattr(args, 'config', '') or 'aligning' in getattr(args, 'dataset', ''):
    yaml_name = 'visual_aligning_eval.yaml'
else:
    yaml_name = 'projection_eval.yaml'
```
This has the **same class of bug**: it never checks for `'avoiding'`, so a visual-avoiding
`dataset`/`config` string (containing `'visual'` but not `'aligning'`) would still resolve to
`visual_aligning_eval.yaml` — wrong, for the same reason as the two fixes above. However, the
only caller found that could reach this path
(`fm_visual_avoiding_test (legacy_based_on_visual_aligning)/eval_fm_visual_avoiding.py`) lives
in a folder explicitly marked **legacy** — the active visual-avoiding pipeline
(`fm_visual_avoiding_test/eval_fm_visual_avoiding.py`, no `(legacy...)` suffix) reads its yaml
directly and doesn't appear to route through this heuristic's snapshot call. Left unfixed to
avoid touching a legacy/likely-unused code path without first confirming it's actually dead;
flag here for whoever next works in `encdec_vision` to resolve if that path turns out to still
be live.

## Verification
- `py_compile` clean on all three fixed files (`flow_matcher_v3_uav`, `fm_visual_avoiding`,
  `diffuser_visual_avoiding` `utils/setup.py`).
- Cross-checked every package's real eval script (`grep` for `yaml.safe_load`/`open('config/...')`)
  against its `setup.py`'s snapshot target — the table above is exhaustive over every package
  with a `setup.py` in the repo (excluding `Archived_Codes/`).
- Confirmed `config/projection_eval.yaml`, `config/visual_aligning_eval.yaml`, and
  `config/visual_avoiding_eval.yaml` are three distinct, independently-existing files (not
  aliases) — every mismatch above was a genuine wrong-file bug, not a naming coincidence.

## Files touched
- `flow_matcher_v3_uav/utils/setup.py`
- `fm_visual_avoiding/utils/setup.py`
- `diffuser_visual_avoiding/utils/setup.py`

## Practical consequence for existing runs
Any past run's `config_snapshot_<name>/` folder from these three packages contains the WRONG
yaml and cannot be used for provenance — reconstruct the actual config from the git commit
hash printed in the job's SLURM log header (`GIT REV: <hash>`) instead, for anything predating
this fix. Snapshots taken from here on will contain the correct file.
