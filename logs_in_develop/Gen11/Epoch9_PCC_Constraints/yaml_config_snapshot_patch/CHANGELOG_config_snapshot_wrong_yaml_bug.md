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

---

## Addendum — the UAV fix didn't actually take effect (found by the user re-checking)

**Reported symptom:** after the fix above, a UAV run's snapshot folder still showed the OLD
avoiding-d3il yaml, not `uav_projection.yaml`. The fix to `snapshot_configs` itself was
correct — but its caller in `FM_v3_uav_test/eval_fm_uav.py` (`_run_variant`) never invoked it:

```python
_snap_dir = os.path.join(seed_dir, f'config_snapshot_{parsed.config.split(".")[-1]}')
if not os.path.exists(_snap_dir):          # ← the bug
    ...
    utils.Parser().snapshot_configs(_snap_args)
```

**Root cause:** this guard checks the **filesystem**, not the process. Once
`config_snapshot_uav/` exists on disk — including from a run that predates the
`snapshot_configs` fix, when it still contained the wrong yaml — this check is `True`
forever, and `snapshot_configs` is never called again for that `seed_dir`. The fixed function
was correct; it just never got a chance to run for any seed_dir whose folder already existed.
**Why only UAV had this:** `flow_matcher_v3_uav/utils/setup.py::mkdir()` gates
`snapshot_configs` inside `if save:` ("Config snapshot only during training... Eval writes the
snapshot explicitly" — comment in that file), so the UAV eval script re-implements its own
snapshot call — with this bug. Checked every other eval script for the same custom
re-implementation (`grep` for `_snap_dir`/`os.path.exists(_snap_dir)` repo-wide) — UAV is the
**only** one; every other package's `mkdir()` calls `snapshot_configs` **unconditionally**
(not gated by `save` at all — see the visual_avoiding finding below), so they don't have this
specific bug.

### Fix
Replaced the filesystem-existence check with an in-memory, per-process guard:
```python
_SNAPSHOTTED_DIRS = set()   # module-level
...
if _snap_dir not in _SNAPSHOTTED_DIRS:
    ...
    utils.Parser().snapshot_configs(_snap_args)
    _SNAPSHOTTED_DIRS.add(_snap_dir)
```
Still avoids redundant re-copies within one job's variant/geo_tag loop (the original intent),
but a **fresh process** (i.e. every new job submission) always re-snapshots, and `shutil.copy`
naturally overwrites whatever stale content was there before.

### Leftover-clutter cleanup
The pre-fix bug wrote a file named `projection_eval.yaml`; the fixed code writes
`uav_projection.yaml` — a **different filename**, so simply re-running doesn't overwrite the
old one, it just adds the correct file alongside the stale one. Added an explicit cleanup:
after snapshotting, delete `config_snapshot_uav/projection_eval.yaml` if present — safe here
specifically because that filename is never legitimately correct content for a UAV run.

## Addendum 2 — audited visual_avoiding for the same class of bug (per user's follow-up ask)

Checked whether `fm_visual_avoiding`/`diffuser_visual_avoiding` (fixed for the wrong-yaml-path
bug earlier in this changelog) have the SAME "never re-runs" problem UAV had. **They don't** —
their `mkdir()` calls `self.snapshot_configs(args)` **unconditionally**, outside any `if save:`
guard, so it fires on every single eval run regardless of `os.path.exists`. Confirmed via
`parse_args`: `save = (experiment == 'train')`, so eval always passes `save=False`, but
`snapshot_configs` isn't gated by that flag at all for these packages — only the separate
`args_resume_X.json` write is. **These self-heal automatically on the very next eval run.**

They do have the same **leftover-clutter** issue as UAV, though, for the same reason (fixed
destination filename changed from `visual_aligning_eval.yaml` to `visual_avoiding_eval.yaml`,
so the old file isn't overwritten, just left alongside the new correct one). Added the
equivalent cleanup to both `fm_visual_avoiding/utils/setup.py` and
`diffuser_visual_avoiding/utils/setup.py`: delete `visual_aligning_eval.yaml` from the
snapshot dir if present, after writing the correct `visual_avoiding_eval.yaml`.

`fm_visual_aligning`/`diffuser_visual_aligining`/`imf_visual_aligining` need no equivalent
check — they never had the wrong-yaml-path bug in the first place (confirmed in the original
audit above), so there's no stale/mismatched-filename file to clean up there.

### Verification
- `py_compile` clean on all 4 touched files
  (`FM_v3_uav_test/eval_fm_uav.py`, `flow_matcher_v3_uav/utils/setup.py`,
  `fm_visual_avoiding/utils/setup.py`, `diffuser_visual_avoiding/utils/setup.py`).
- Repo-wide `grep` for `_snap_dir`/`os.path.exists(_snap_dir)` confirms UAV is the only eval
  script with a custom, filesystem-guarded snapshot re-implementation.
- Confirmed (by reading `parse_args`/`mkdir` in `fm_visual_avoiding/utils/setup.py`) that
  `snapshot_configs` there is unconditional, unlike UAV's `if save:`-gated version.

### Files touched (this addendum)
- `FM_v3_uav_test/eval_fm_uav.py` — `_SNAPSHOTTED_DIRS` process-scoped guard replacing the
  `os.path.exists(_snap_dir)` filesystem check; stale `projection_eval.yaml` cleanup.
- `fm_visual_avoiding/utils/setup.py`, `diffuser_visual_avoiding/utils/setup.py` — stale
  `visual_aligning_eval.yaml` cleanup after the correct snapshot is written.
