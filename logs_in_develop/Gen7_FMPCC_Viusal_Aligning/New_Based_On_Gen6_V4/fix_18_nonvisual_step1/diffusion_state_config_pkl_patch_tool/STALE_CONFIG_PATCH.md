# Fix-18 follow-up — Stale `model_config.pkl` overwrite + regen script

**Date**: 2026-05-31
**Trigger**: DPCC non-visual eval crashed with
```
RuntimeError: Error(s) in loading state_dict for VisualGaussianDiffusion:
    size mismatch for model.backbone.downs.0.0.blocks.0.block.0.weight:
        copying a param with shape [32, 23, 5] from checkpoint,
        the shape in current model is [32, 9, 5].
```
even though Fix-18 was applied and the training run was successful (state
dict on disk is correct 23-D).

---

## 1. Why this happened (not a Fix-18 regression)

`utils.Config.__init__` had this line in **both** DPCC and FM stacks
(`diffuser_visual_aligning/utils/config.py:35`, `fm_visual_aligning/utils/config.py:35`):

```python
if not os.path.exists(savepath):
    pickle.dump(self, open(savepath, 'wb'))
```

This silently **skipped** the save when `model_config.pkl` already
existed on disk. Result:

1. **Pre-Fix-18 crash run** wrote a 9-D `model_config.pkl` to disk before
   crashing at the first conv.
2. **Post-Fix-18 successful run** trained correctly (23-D weights), but
   `utils.Config(...)` saw the stale 9-D `model_config.pkl` and refused
   to overwrite it.
3. **Today's eval** loaded the stale 9-D config → instantiated 9-D model
   → tried to load 23-D state_*.pt → shape-mismatch crash.

The state dict was always overwritten on every checkpoint save. Only the
*config artifact* was skipped. This created the silent drift between
weights and config.

---

## 2. Code fix (applied)

| File | Change |
|---|---|
| `diffuser_visual_aligning/utils/config.py:33-38` | Removed the `if not os.path.exists(savepath):` guard. `pickle.dump` now runs every time, with an inline comment pointing here. |
| `fm_visual_aligning/utils/config.py:33-38` | Same change (FM stack had an identical copy of the bug). |

After this fix, any future training run automatically refreshes
`model_config.pkl` to reflect the args namespace it was actually called
with — so the next time Fix-18 (or any other args mutation) lands, the
on-disk config catches up automatically.

The JSON resume-numbering behaviour for `model_config.json` was
**intentionally left intact** — JSON is human-readable and preserving
prior runs as `model_config_resume_<N>.json` is useful for audit. The
pkl is the machine-loadable one and must always be current; the JSON is
the history.

---

## 3. One-shot regen script (for your existing broken checkpoint)

`regen_stale_model_config.py` (alongside this MD): rebuilds
`model_config.pkl` + `model_config.json` from a hand-coded post-Fix-18
args namespace, without re-running the training pipeline.

### Usage

```bash
cd /path/to/FM-PCC

# Auto-detect: regenerates configs for both DPCC and FM non-visual
# canonical checkpoint paths if found
python logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/\
fix_18_nonvisual_step1/regen_stale_model_config.py

# Dry run first if you want to see what would be written:
python .../regen_stale_model_config.py --dry-run

# Or target a specific checkpoint dir explicitly:
python .../regen_stale_model_config.py --checkpoint-dir \
    logs/aligning-d3il-visual/visual_aligning_dpcc/\
H8_K1_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.\
VisualGaussianDiffusion_aw10_VFalse_steps900_bs64/6
```

### What it does

1. Reads the existing `model_config.pkl` (if any) and prints its
   `obs_dim` — confirms whether it's stale 9-D (obs_dim=6) or already
   fixed (obs_dim=20).
2. If stale → constructs a fresh `args` namespace with the correct
   post-Fix-18 values (`obs_dim=20`, `action_dim=3`, `if_vision=False`,
   etc., matching exactly what `train_visual_aligning_dpcc.py` would
   produce after Fix-18 ran).
3. Deletes the stale pkl + json.
4. Constructs a new `utils.Config(VisualUNet, savepath=...,
   config=fresh_args)` — this writes the new pkl + json to disk.
5. Eval can now run.

### Safety

- Pure metadata regeneration. **No state dict, weights, or training
  data is touched.**
- Dry-run mode available (`--dry-run`).
- Idempotent: if the existing pkl is already correct (`obs_dim=20`),
  the script reports "already correct — nothing to do" and exits.
- Reversible: rerun training to overwrite again, or just `rm` the
  pkl/json to clear.

---

## 4. After running the regen

For the DPCC non-visual K=1 run:

```bash
sbatch Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh
```

Expected: eval now loads the correct 23-D `model_config.pkl`, constructs
a 23-D `VisualUNet`, loads the 23-D state dict cleanly, runs the
rollouts. The shape-mismatch crash from §1 disappears.

---

## 5. Files Touched by This Patch

| File | Action | Why |
|---|---|---|
| `diffuser_visual_aligning/utils/config.py` | Modified (one-line + comment) | Always-overwrite for `model_config.pkl` |
| `fm_visual_aligning/utils/config.py` | Modified (one-line + comment) | Same fix for FM stack |
| `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/regen_stale_model_config.py` | Created | One-off cleanup for existing broken checkpoints |
| `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/STALE_CONFIG_PATCH.md` | Created (this file) | Documentation |

**Not touched:** any state dict, dataset, training script, eval script, model
class, or SLURM wrapper. The bug was strictly in the config-save side
effect, and its remediation is strictly there.
