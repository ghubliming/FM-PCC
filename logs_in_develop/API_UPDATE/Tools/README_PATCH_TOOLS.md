# API Rename Patch Tools

**Purpose:** Fix a legacy FM-PCC checkpoint folder that fails to load after the
`GaussianDiffusion` / `iMFDiffusion` / `VisualGaussianDiffusion` → new-name API rename.

**Root cause:** Python pickle stores class references as `(module_path, class_name)` pairs.
After renaming a class, `pickle.load()` raises `AttributeError: Can't get attribute 'OldName'`.
The checkpoint folder name also embeds the old class name (via the `{diffusion}` config
template) and must be corrected for the eval path-resolver to find the run.

---

## Files

| File | Purpose |
|---|---|
| `patch_legacy_checkpoints.py` | Python script — detects, patches, renames |
| `run_patch_legacy_checkpoints.sh` | SLURM job — edit `TARGET_PATH` and submit |
| `README_PATCH_TOOLS.md` | This file |

---

## Workflow

**Edit** `TARGET_PATH` in `run_patch_legacy_checkpoints.sh` to the old folder path, then submit:

```bash
sbatch logs_in_develop/API_UPDATE/Tools/run_patch_legacy_checkpoints.sh
```

The script does **nothing** if the folder name contains no legacy tokens (safe to re-run or
point at an already-correct path).

---

## What the Python script does

1. **Checks** the folder's basename for old class tokens.
   - If none found → prints "nothing to do" and exits immediately.
2. **Patches all `.pkl` config files** using a custom `RemapUnpickler` that intercepts
   old class references and redirects them to the new names. Re-saves only if changed.
3. **Patches `args.json`** files via string replacement of old dotted class-path strings.
4. **Renames the folder** to the corrected name.

`losses.pkl` and `state_*.pt` weight files are never touched.

---

## Token remap table

| Old token (in folder name / pkl / json) | New token |
|---|---|
| `...VisualGaussianDiffusion` | `...VisualFlowMatching` |
| `...fm_visual_aligning.models.diffusion.GaussianDiffusion` | `...FlowMatchingODE` |
| `...flow_matcher_v3_drifting.models.diffusion.GaussianDiffusion` | `...FlowMatchingDrifting` |
| `...flow_matcher_v3_drifting.models.diffusion.FlowMatchingODE` *(intermediate)* | `...FlowMatchingDrifting` |
| `...flow_matcher_v3_imeanflow.models.diffusion.GaussianDiffusion` | `...FlowMatchingIMF` |
| `...flow_matcher_v3_imeanflow.models.diffusion.FlowMatchingODE` *(intermediate)* | `...FlowMatchingIMF` |
| `...flow_matcher_v3_imeanflow.models.imf_diffusion.iMFDiffusion` | `...iMeanFlowODE` |
| `...flow_matcher_v3_ode_selectable.models.diffusion.GaussianDiffusion` | `...FlowMatchingODE` |

The intermediate `FlowMatchingODE` entries handle checkpoints trained between the first
rename pass (2026-05-25) and the second module-specific pass (2026-05-26).

---

## SLURM script variables

Open `run_patch_legacy_checkpoints.sh` and edit these two lines before submitting:

```bash
TARGET_PATH="FMPCC/FM-PCC/logs/aligning-d3il-visual/fm_visual_aligning/H8_D...OldName..."
EXTRA_FLAGS=""   # optional: --dry-run   --backup   or both
```

`TARGET_PATH` can be relative to `$HOME` or absolute.

---

## Flags

| Flag | Effect |
|---|---|
| `--dry-run` | Print every change without writing or renaming. |
| `--backup` | Save `<file>.pkl.bak` before overwriting each pkl. |
