# Gen8 Epoch 1 — Fix_1: UNet Class Mismatch & Missing FiLM Conditioning

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted). Re-submit cluster job to validate.
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md) (Epoch 1 initial implementation)

---

## 1. Symptom

Cluster job `21155` (`train_imf_visual_aligning`, node `i6-gpu-1`) crashed immediately after dataset loading with:

```
ImportError: cannot import name 'UNet1DTemporalCondModel' from
'imf_visual_aligning.models.unet1d_temporal_cond'
(/u/home/llim/FMPCC/FM-PCC/imf_visual_aligning/models/unet1d_temporal_cond.py)
```

Full traceback in [`temp/debug_gen8/gen8_outputs`](../../../temp/debug_gen8/gen8_outputs).

---

## 2. Root cause

**Two-source copy collision.** Gen8's `unet1d_temporal_cond.py` was copied from **Gen3v4** (iMF branch), while `visual_unet.py` and `__init__.py` were scaffolded from **Gen7** (FM visual aligning branch). The two branches used **different class names** for the same architectural role:

| Source | Class name in `unet1d_temporal_cond.py` | Has `h_mlp`? | Has `cond_mlp` / `use_cond_projection`? |
|---|---|---|---|
| Gen3v4 (iMF) | `Flow_matcher_U_Net_v2` | ✅ Yes | ❌ No |
| Gen7 (FM visual) | `UNet1DTemporalCondModel` | ❌ No | ✅ Yes |
| **Gen8 (needed)** | **`UNet1DTemporalCondModel`** | **✅ Yes** | **✅ Yes** |

### Bug 1 — Name mismatch (import crash)

`__init__.py` line 1 and `visual_unet.py` line 61 both import `UNet1DTemporalCondModel`, but the Gen3v4-sourced file only defines `Flow_matcher_U_Net_v2`. Python raises `ImportError` at module import time before any training code executes.

### Bug 2 — Missing FiLM conditioning (latent, would have crashed at model construction)

Even if Bug 1 were bypassed by renaming alone, `visual_unet.py` line 84 passes `use_cond_projection=self.if_vision` to the constructor. Gen3v4's `Flow_matcher_U_Net_v2.__init__` does **not** accept this kwarg — it would raise `TypeError: __init__() got an unexpected keyword argument 'use_cond_projection'`.

Furthermore, without `cond_mlp`, the visual embeddings from the ResNet encoder would silently be ignored (passed as `cond` tensor to `forward()` but never projected into the time-embedding space), producing **no visual conditioning** during training — a silent architectural failure.

---

## 3. Fix applied

**Merged both capabilities** into a single `UNet1DTemporalCondModel` class in `unet1d_temporal_cond.py`:

| Feature | Source | Status in merged class |
|---|---|---|
| Class name `UNet1DTemporalCondModel` | Gen7 | ✅ Primary class name |
| `h_mlp` (iMF h-conditioning via addition to `t`) | Gen3v4 | ✅ Kept |
| `h=None` in `forward()` signature | Gen3v4 | ✅ Kept |
| `use_cond_projection` constructor param | Gen7 | ✅ Added |
| `cond_mlp` (FiLM visual conditioning via concat to `t`) | Gen7 | ✅ Added |
| `cond_dim` stored as `self.cond_dim` | Gen7 | ✅ Added |
| `embed_dim` = `time_dim + cond_embed_dim + returns_dim` | Gen7 | ✅ Correct accounting |
| `Flow_matcher_U_Net_v2` backward-compat alias | — | ✅ Added at module level |

### Conditioning order in `forward()`:

```
t = time_mlp(timesteps)          # time embedding
t = t + h_mlp(h)                 # iMF h-conditioning (additive)
t = cat([t, cond_mlp(cond)])     # FiLM visual conditioning (concat)
t = cat([t, returns_embed])      # returns conditioning (concat, if enabled)
→ ResidualTemporalBlocks receive t as embed
```

### File changed

| File | Change |
|---|---|
| `imf_visual_aligning/models/unet1d_temporal_cond.py` | Rewrote: renamed class, added `use_cond_projection`/`cond_mlp`/`cond_dim` from Gen7, kept `h_mlp`/`h` from Gen3v4, added `Flow_matcher_U_Net_v2` alias |

**No changes needed** in `__init__.py`, `visual_unet.py`, `imf_trajectory_model.py`, or any other file — they already import the correct names.

---

## 4. Verification

| Check | Result |
|---|---|
| AST parse: `unet1d_temporal_cond.py` | ✅ |
| AST parse: `__init__.py` | ✅ |
| AST parse: `visual_unet.py` | ✅ |
| `UNet1DTemporalCondModel` defined as class | ✅ |
| `Flow_matcher_U_Net_v2` defined as alias | ✅ |
| `imf_trajectory_model.py` imports `Flow_matcher_U_Net_v2` → resolves to alias | ✅ |
| `visual_unet.py` imports `UNet1DTemporalCondModel` → resolves to class | ✅ |
| `visual_unet.py` passes `use_cond_projection=True` → accepted by constructor | ✅ |
| `h=None` in `forward()` → iMF h-conditioning threaded end-to-end | ✅ |
| `cond_mlp` path only fires for tensor cond (not dict) → state-based pipeline safe | ✅ |

---

## 5. Why this wasn't caught by Phase-0 smoke checks

The Epoch 1 Phase-0 checks (CHANGELOG §6) verified:
- AST parse on all 7 new/modified `.py` files → ✅
- Cross-package imports → ✅
- Stale dim literals → ✅

But **did not** test whether exported symbols in `__init__.py` actually resolve against the module's defined names. An `import imf_visual_aligning.models` would have caught this instantly, but the Docker environment used for Phase-0 may not have had `diffusers` / `hydra` installed, so full import testing was skipped.

**Lesson**: Future Phase-0 should include an `importlib.util.spec_from_file_location` + AST name-resolution check, or at minimum a `grep` to verify that every name in `__init__.py` import lines exists as a `class` or top-level assignment in the target module.

---

## 6. Next steps

1. **Cluster re-submit**: `pip install -e .` in `imf_visual_aligning/`, then re-run `sbatch Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh`
2. **Monitor**: First 100 steps — watch for tensor broadcast crash (CHANGELOG §8 step 2 still applies)
3. **If loss is NaN at step 0**: likely `embed_dim` mismatch in `ResidualTemporalBlock`; compare `embed_dim` printed at init vs what the block's `time_mlp` expects

---

## 7. Cross-references

| Document | Content |
|---|---|
| [`../CHANGELOG.md`](../CHANGELOG.md) | Epoch 1 initial implementation (pre-fix) |
| [`../PLAN.md`](../PLAN.md) | Full design rationale |
| [`temp/debug_gen8/gen8_outputs`](../../../temp/debug_gen8/gen8_outputs) | Raw cluster crash log |
