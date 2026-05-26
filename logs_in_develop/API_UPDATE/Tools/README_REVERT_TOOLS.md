# API Rename Revert Tools

**Purpose:** Undo a mistaken run of the `patch_legacy_checkpoints.py` script and revert a legacy FM-PCC checkpoint folder back to its original `GaussianDiffusion` / `iMFDiffusion` / `VisualGaussianDiffusion` state.

**Root cause:** If a legacy folder was inadvertently patched using the patch tool, it could break historical logging or strict backward compatibility in some legacy pipelines. This tool reverses the token and class remappings applied by the patch.

---

## Files

| File | Purpose |
|---|---|
| `revert_legacy_checkpoints.py` | Python script — detects patched tokens, reverts, renames back |
| `run_revert_legacy_checkpoints.sh` | SLURM job — edit `TARGET_PATH` and submit |
| `README_REVERT_TOOLS.md` | This file |

---

## Workflow

**Edit** `TARGET_PATH` in `run_revert_legacy_checkpoints.sh` to the patched folder path, then submit:

```bash
sbatch logs_in_develop/API_UPDATE/Tools/run_revert_legacy_checkpoints.sh
```

The script does **nothing** if the folder name contains no patched tokens (safe to re-run or point at an already-reverted path).

---

## What the Python script does

1. **Always reverts all `.pkl` config files** using a custom `RemapUnpickler` that intercepts new class references and redirects them back to the original names. Re-saves only if changed.
2. **Always reverts `args.json`** files via string replacement of new dotted class-path strings back to the original.
3. **Renames the folder** only if the basename contains patched tokens.
   - If the folder was already renamed manually (basename is clean) → skips rename, still reverts contents.

`losses.pkl` and `state_*.pt` weight files are never touched.

---

## Token remap table (Reverted)

| Patched token (in folder name / pkl / json) | Original token |
|---|---|
| `...VisualFlowMatching` | `...VisualGaussianDiffusion` |
| `...FlowMatchingODE` | `...GaussianDiffusion` |
| `...FlowMatchingDrifting` | `...GaussianDiffusion` |
| `...FlowMatchingIMF` | `...GaussianDiffusion` |
| `...iMeanFlowODE` | `...iMFDiffusion` |

*Note: For detailed fully-qualified module reversals, see `TOKEN_REMAP` and `CLASS_REMAP` inside `revert_legacy_checkpoints.py`.*

---

## SLURM script variables

Open `run_revert_legacy_checkpoints.sh` and edit these two lines before submitting:

```bash
TARGET_PATH="FMPCC/FM-PCC/logs/aligning-d3il-visual/fm_visual_aligning/H8_D...PatchedName..."
EXTRA_FLAGS=""   # optional: --dry-run   --backup   or both
```

`TARGET_PATH` can be relative to `$HOME` or absolute.

---

## Flags

| Flag | Effect |
|---|---|
| `--dry-run` | Print every change without writing or renaming. |
| `--backup` | Save `<file>.pkl.bak` before overwriting each pkl. |
