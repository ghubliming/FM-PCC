# Gen8 Epoch 1 — Fix_1: Two-Source Copy Collision (Import Name Mismatches)

**Date**: 2026-06-03
**Status**: ✅ Fix_1.1 + Fix_1.2 applied (uncommitted). Re-submit cluster job to validate.
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md) (Epoch 1 initial implementation)

---

## 1. Symptom (Fix_1.1)

Cluster job `21155` (`train_imf_visual_aligning`, node `i6-gpu-1`) crashed immediately after dataset loading with:

```
ImportError: cannot import name 'UNet1DTemporalCondModel' from
'imf_visual_aligning.models.unet1d_temporal_cond'
(/u/home/llim/FMPCC/FM-PCC/imf_visual_aligning/models/unet1d_temporal_cond.py)
```

Full traceback in [`temp/debug_gen8/gen8_outputs`](../../../temp/debug_gen8/gen8_outputs).

## 1b. Symptom (Fix_1.2)

After Fix_1.1 was applied, cluster job `21160` (same node) crashed at the same phase with:

```
ImportError: cannot import name 'FlowMatchingODE' from
'imf_visual_aligning.models.diffusion'
(/u/home/llim/FMPCC/FM-PCC/imf_visual_aligning/models/diffusion.py)
```

Full traceback in [`temp/debug_gen8/outputs_2`](../../../temp/debug_gen8/outputs_2).

---

## 2. Root cause

**Two-source copy collision.** Gen8 was assembled from two branches — Gen3v4 (iMF engine files) and Gen7 (visual aligning scaffold). These branches used **different class names** for the same architectural roles. The `__init__.py` from Gen7 eagerly imports all symbols, so any single name mismatch crashes the entire package at import time.

### Fix_1.1 — `unet1d_temporal_cond.py`: `Flow_matcher_U_Net_v2` vs `UNet1DTemporalCondModel`

| Source | Class name | Has `h_mlp`? | Has `cond_mlp` / `use_cond_projection`? |
|---|---|---|---|
| Gen3v4 (iMF) | `Flow_matcher_U_Net_v2` | ✅ Yes | ❌ No |
| Gen7 (FM visual) | `UNet1DTemporalCondModel` | ❌ No | ✅ Yes |
| **Gen8 (needed)** | **`UNet1DTemporalCondModel`** | **✅ Yes** | **✅ Yes** |

**Bug 1a** — `__init__.py` line 1 and `visual_unet.py` line 61 import `UNet1DTemporalCondModel`, but the Gen3v4-sourced file only defines `Flow_matcher_U_Net_v2` → `ImportError`.

**Bug 1b** — Even with a rename, `visual_unet.py` passes `use_cond_projection=True` which Gen3v4's constructor doesn't accept → `TypeError`. And without `cond_mlp`, visual embeddings would silently be ignored.

### Fix_1.2 — `diffusion.py`: `FlowMatchingIMF` vs `FlowMatchingODE`

| Source | Class name in `diffusion.py` |
|---|---|
| Gen3v4 (iMF) | `FlowMatchingIMF` |
| Gen7 (FM visual) | `FlowMatchingODE` |

**Bug** — `__init__.py` line 2 imports `FlowMatchingODE`, and `visual_gaussian_diffusion.py` line 2+6 imports and inherits from `FlowMatchingODE`. The Gen3v4-sourced `diffusion.py` only defines `FlowMatchingIMF` → `ImportError`.

Note: `VisualFlowMatching(FlowMatchingODE)` is not directly used in the iMF training path (which uses `VisualIMF(iMeanFlowODE)`), but it's still imported eagerly by `__init__.py`, so the mismatch still crashes the package.

---

## 3. Fix_1.1 applied (UNet)

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

---

## 3b. Fix_1.2 applied (diffusion base class)

Added backward-compatible alias at the bottom of `diffusion.py`:

```python
# Fix_1.2 (2026-06-03)
FlowMatchingODE = FlowMatchingIMF
```

### File changed

| File | Change |
|---|---|
| `imf_visual_aligning/models/diffusion.py` | Appended `FlowMatchingODE = FlowMatchingIMF` alias at module level |

### Why an alias is sufficient (no merge needed)

Unlike Fix_1.1's UNet (which required merging `h_mlp` + `cond_mlp` from different sources), the `diffusion.py` class is a straight copy from Gen3v4 — its constructor is a superset of Gen7's `FlowMatchingODE.__init__` (adds ODE solver params that Gen7 didn't have). The `VisualFlowMatching` subclass intercepts those extra params in its own `__init__`, so they safely default. No code merge needed — just the name alias.

---

## 4. Verification

### Fix_1.1 checks

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

### Fix_1.2 checks

| Check | Result |
|---|---|
| `FlowMatchingODE` alias defined in `diffusion.py` | ✅ |
| `__init__.py` line 2 `from .diffusion import FlowMatchingODE` → resolves | ✅ |
| `visual_gaussian_diffusion.py` line 2 import → resolves | ✅ |
| `VisualFlowMatching(FlowMatchingODE)` inheritance → resolves to `FlowMatchingIMF` | ✅ |
| `FlowMatchingIMF.__init__` is superset of Gen7's `FlowMatchingODE.__init__` | ✅ |

### Comprehensive `__init__.py` import audit

| Import | Source module | Actual name | Status |
|---|---|---|---|
| `UNet1DTemporalCondModel` | `unet1d_temporal_cond` | class | ✅ Fix_1.1 |
| `Flow_matcher_U_Net_v2` | `unet1d_temporal_cond` | alias | ✅ Fix_1.1 |
| `FlowMatchingODE` | `diffusion` | alias | ✅ Fix_1.2 |
| `VisualUNet` | `visual_unet` | class | ✅ |
| `VisualFlowMatching` | `visual_gaussian_diffusion` | class | ✅ |
| `iMeanFlowODE` | `imf_diffusion` | class | ✅ |
| `iMeanFlowEngine` | `imf_engine` | class | ✅ |
| `iMFTrajectoryModel` | `imf_trajectory_model` | class | ✅ |
| `VisualIMF` | `visual_imf_diffusion` | class | ✅ |

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
| [`temp/debug_gen8/gen8_outputs`](../../../temp/debug_gen8/gen8_outputs) | Raw cluster crash log (Fix_1.1 — job 21155) |
| [`temp/debug_gen8/outputs_2`](../../../temp/debug_gen8/outputs_2) | Raw cluster crash log (Fix_1.2 — job 21160) |
