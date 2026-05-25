# API Rename Report: Diffusion → FlowMatching

**Date:** 2026-05-22  
**Branch:** `update_into_FM`  
**Status:** Report only — no code changed yet. Review and approve before implementation.

---

## Correction from First Draft

First draft incorrectly used `fm_encdec_vision` as the FMv3ODE module.  
`fm_encdec_vision` is **abandoned** — do not touch it.

**Correct four active modules:**

| User Label | Actual Python Package | Test Folder |
|---|---|---|
| FMv3ODE | `flow_matcher_v3_ode_selectable` | `FM_v3_ode_selectable_test/` |
| FM Drifting | `flow_matcher_v3_drifting` | `FM_v3_drifting_test/` |
| FM IMF | `flow_matcher_v3_imeanflow` | `FM_v3_imeanflow_test/` |
| FM Visual Aligning | `fm_visual_aligning` | `fm_visual_aligning_test/` |

---

## Background

During development, FM modules were patched from DDPM codebases without renaming. Core FM classes carry names like `GaussianDiffusion`, `VisualGaussianDiffusion`, `iMFDiffusion` even though they implement continuous-time Flow Matching ODE — not DDPM. This report covers what to rename.

---

## Proposed Class Renames

### Shared base engine (same rename in 4 packages)

| Package | File | Current Name | Proposed Name |
|---|---|---|---|
| `flow_matcher_v3_ode_selectable` | `models/diffusion.py` | `GaussianDiffusion` | `FlowMatchingODE` |
| `flow_matcher_v3_drifting` | `models/diffusion.py` | `GaussianDiffusion` | `FlowMatchingODE` |
| `flow_matcher_v3_imeanflow` | `models/diffusion.py` | `GaussianDiffusion` | `FlowMatchingODE` |
| `fm_visual_aligning` | `models/diffusion.py` | `GaussianDiffusion` | `FlowMatchingODE` |

### Module-specific renames

| Package | File | Current Name | Proposed Name | Notes |
|---|---|---|---|---|
| `flow_matcher_v3_imeanflow` | `models/imf_diffusion.py` | `iMFDiffusion` | `iMeanFlowODE` | iMF-specific ODE wrapper, distinct from `iMeanFlowEngine` (algorithm class — keep as-is) |
| `fm_visual_aligning` | `models/visual_gaussian_diffusion.py` | `VisualGaussianDiffusion` | `VisualFlowMatching` | Inherits from `GaussianDiffusion` → will inherit from `FlowMatchingODE` after base rename |

> **`GaussianNormalizer`** in all `datasets/normalization.py` — **do NOT rename**. Statistical utility, not a generative model.

---

## `__init__.py` Export Updates

| File | Remove | Add |
|---|---|---|
| `flow_matcher_v3_ode_selectable/models/__init__.py` | `GaussianDiffusion` | `FlowMatchingODE` |
| `flow_matcher_v3_drifting/models/__init__.py` | `GaussianDiffusion` | `FlowMatchingODE` |
| `flow_matcher_v3_imeanflow/models/__init__.py` | `GaussianDiffusion`, `iMFDiffusion` | `FlowMatchingODE`, `iMeanFlowODE` |
| `fm_visual_aligning/models/__init__.py` | `GaussianDiffusion`, `VisualGaussianDiffusion` | `FlowMatchingODE`, `VisualFlowMatching` |

---

## Eval/Train Script Updates (Class-Name String Checks)

These gating checks must match the new class names exactly or the ODE solver branch never fires.

| File | Line | Current Check | Updated Check |
|---|---|---|---|
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | 155 | `== 'GaussianDiffusion'` | `== 'FlowMatchingODE'` |
| `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | 155 | `== 'GaussianDiffusion'` | `== 'FlowMatchingODE'` |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | 156 | `in ['GaussianDiffusion', 'iMFDiffusion']` | `in ['FlowMatchingODE', 'iMeanFlowODE']` |
| `fm_visual_aligning_test/train_fm_visual_aligning.py` | ~199 | `import VisualGaussianDiffusion` + usage | `import VisualFlowMatching` + usage |

---

## Config File Updates — ⚠️ DANGEROUS, READ CAREFULLY

Two config files are affected. **Each has blocks for DDPM/diffuser baselines alongside FM blocks — those must not be touched.**

### `config/avoiding-d3il.py`

This file controls the **avoiding task**. Only update class path values in the four FM blocks. The config key `'diffusion'` is **not renamed** (see note below).

| Block Name | Current Class Path | Updated Class Path |
|---|---|---|
| `flow_matching_v3_ode_selectable` (training, ~line 330) | `'models.diffusion.GaussianDiffusion'` | `'models.diffusion.FlowMatchingODE'` |
| `plan_fm_v3_ode_selectable` (inference, ~line 690) | `'models.diffusion.GaussianDiffusion'` | `'models.diffusion.FlowMatchingODE'` |
| `flow_matching_v3_drifting` (training, ~line 386) | `'models.diffusion.GaussianDiffusion'` | `'models.diffusion.FlowMatchingODE'` |
| `plan_fm_v3_drifting` (inference, ~line 740) | `'models.diffusion.GaussianDiffusion'` | `'models.diffusion.FlowMatchingODE'` |
| `flow_matching_v3_imeanflow` (training, ~line 444) | `'flow_matcher_v3_imeanflow.models.iMFDiffusion'` | `'flow_matcher_v3_imeanflow.models.iMeanFlowODE'` |
| `plan_fm_v3_imeanflow` (inference, ~line 786) | `'flow_matcher_v3_imeanflow.models.iMFDiffusion'` | `'flow_matcher_v3_imeanflow.models.iMeanFlowODE'` |

**DO NOT TOUCH** any other blocks in `avoiding-d3il.py` (DDPM, diffuser, etc.).

---

### `config/aligning-d3il-visual.py`

This file controls the **aligning task**. It contains blocks for DDPM, diffuser, `fm_encdec_vision` (abandoned), AND `fm_visual_aligning` (active). **Only touch the `fm_visual_aligning` blocks.**

| Block Name | Current Class Path | Updated Class Path |
|---|---|---|
| `fm_visual_aligning` training (~line 391) | `'fm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion'` | `'fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching'` |
| `plan_fm_visual_aligning` inference (~line 666) | `'fm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion'` | `'fm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching'` |

**DO NOT TOUCH** lines 94, 143, 209, 262, 328, 483, 533, 586 — those are DDPM/diffuser/`fm_encdec_vision` blocks.

---

## Config Key `'diffusion'` — ⚠️ NOT RENAMED IN THIS PR

The dict key `'diffusion'` appears throughout both config files and is read by `scripts/train.py`. Renaming it to `'flow_matching'` requires updating `scripts/train.py` simultaneously. This is a separate, higher-risk change.

**Scope of this rename: class path values only. The key name `'diffusion'` stays for now.**

---

## Ghost Reference: `GaussianInvDynDiffusion`

This class does not exist anywhere. Referenced only as a dead string check in `sampling/policies.py:27` across all active modules.

| File | Action |
|---|---|
| `flow_matcher_v3_ode_selectable/sampling/policies.py:27` | Remove dead check |
| `flow_matcher_v3_drifting/sampling/policies.py:27` | Remove dead check |
| `flow_matcher_v3_imeanflow/sampling/policies.py:27` | Remove dead check |
| `fm_visual_aligning/sampling/policies.py` | Check if same dead reference exists; remove if so |

Archived module files — **do not touch**.

---

## What Does NOT Change

| Item | Reason |
|---|---|
| `fm_encdec_vision/*` | Abandoned — do not touch |
| `ddpm_encdec_vision/*` | Actual DDPM baseline, name is correct |
| `diffuser_visual_aligning/*` | Actual diffuser baseline, name is correct |
| `GaussianNormalizer` | Statistical utility, not a generative model |
| File names (`diffusion.py`, `visual_gaussian_diffusion.py`) | Only class names inside the files change |
| Config key `'diffusion'` | Deferred — needs `scripts/train.py` update too |
| All `Archived_Codes/` | Do not touch |

---

## File-by-File Change Summary

```
flow_matcher_v3_ode_selectable/
├── models/diffusion.py                  GaussianDiffusion → FlowMatchingODE
├── models/__init__.py                   export updated
└── sampling/policies.py                 remove dead GaussianInvDynDiffusion check

flow_matcher_v3_drifting/
├── models/diffusion.py                  GaussianDiffusion → FlowMatchingODE
├── models/__init__.py                   export updated
└── sampling/policies.py                 remove dead GaussianInvDynDiffusion check

flow_matcher_v3_imeanflow/
├── models/diffusion.py                  GaussianDiffusion → FlowMatchingODE
├── models/imf_diffusion.py              iMFDiffusion → iMeanFlowODE
├── models/__init__.py                   exports updated
└── sampling/policies.py                 remove dead GaussianInvDynDiffusion check

fm_visual_aligning/
├── models/diffusion.py                  GaussianDiffusion → FlowMatchingODE
├── models/visual_gaussian_diffusion.py  VisualGaussianDiffusion → VisualFlowMatching
├── models/__init__.py                   exports updated
└── sampling/policies.py                 check + remove dead reference if present

FM_v3_ode_selectable_test/
└── eval_flow_matching_v3_ode_selectable.py   string check line 155 updated

FM_v3_drifting_test/
└── eval_flow_matching_v3_drifting.py         string check line 155 updated

FM_v3_imeanflow_test/
└── eval_flow_matching_v3_imeanflow.py        string check line 156 updated

fm_visual_aligning_test/
└── train_fm_visual_aligning.py               import + usage updated

config/avoiding-d3il.py
    6 class-path strings in FM blocks only (keys unchanged)

config/aligning-d3il-visual.py
    2 class-path strings in fm_visual_aligning blocks only
```

**Total files to edit: ~17**  
**`fm_encdec_vision`, DDPM, diffuser, archived — untouched**

---

## Auditor Sign-Off

**Auditor:** Antigravity (Claude Opus 4)  
**Date:** 2026-05-25  
**Verdict:** ✅ Report is accurate — all files, line numbers, and class paths verified against live codebase. **5 additions required before implementation.**

### Verified ✅

- All 6 class definitions found at expected lines (L23/L14/L11/L6).
- All 4 `__init__.py` exports match exactly.
- All 8 config class-path strings located at correct lines in `avoiding-d3il.py` and `aligning-d3il-visual.py`.
- All 3 eval script `__class__.__name__` checks at correct lines (155/155/156).
- `train_fm_visual_aligning.py` import at L199, Config usage at L205 — both confirmed.
- `GaussianInvDynDiffusion` ghost reference confirmed dead at L27 in all 3 policies.py files.
- `fm_visual_aligning/sampling/policies.py` — **does not exist** (only `projection.py` + `__init__.py`). Report's "check if exists" item is resolved: no action needed.
- "DO NOT TOUCH" boundaries (DDPM, diffuser, `fm_encdec_vision`, Archived, `GaussianNormalizer`) — all verified safe.

### Missing from Report ⚠️ — Must Add

#### 1. `diffuser/flow_matcher_v3_imeanflow/` shim re-exports (WILL CRASH)

```
diffuser/flow_matcher_v3_imeanflow/models/__init__.py:2   → from .imf_diffusion import iMFDiffusion
diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py:1 → from flow_matcher_v3_imeanflow.models.imf_diffusion import iMFDiffusion
```

These re-export the old name. After renaming `iMFDiffusion` → `iMeanFlowODE`, any import through this path will `ImportError`. **Update both files or confirm they are dead code and delete.**

#### 2. `visual_gaussian_diffusion.py` internal references (WILL CRASH)

The report says rename the class but does not explicitly list the internal references that must change simultaneously:

```python
# fm_visual_aligning/models/visual_gaussian_diffusion.py
Line 2:  from fm_visual_aligning.models.diffusion import GaussianDiffusion   → FlowMatchingODE
Line 6:  class VisualGaussianDiffusion(GaussianDiffusion):                   → class VisualFlowMatching(FlowMatchingODE):
Line 10: "Extends GaussianDiffusion with:"                                   → docstring update
Line 30: "base GaussianDiffusion.__init__ (which has no **kwargs)."          → docstring update
```

#### 3. `imf_diffusion.py:306` error message string (cosmetic)

```python
'Error(s) in loading state_dict for iMFDiffusion:\n'
```

Update to `'iMeanFlowODE'` for consistency.

#### 4. `imf_engine.py:104` docstring (cosmetic)

```python
"objective in iMFDiffusion.p_losses"
```

Update to `'iMeanFlowODE'`.

#### 5. Additional non-active config blocks referencing `GaussianDiffusion`

These are **older FM variants** (not the 4 active modules) in `avoiding-d3il.py` that also use `'models.diffusion.GaussianDiffusion'`:

| Block | Line | Decision |
|---|---|---|
| `flow_matching` (legacy FM v1) | 120 | Leave — separate package |
| `flow_matching_unet_v2` | 169 | Leave — separate package |
| `flow_matching_v2` | 217 | Leave — separate package |
| `flow_matching_v3` (pre-selectable) | 273 | Leave — separate package |
| `flow_matching_hp_tune` | 815 | Leave — separate package |
| `plan_fm` / `plan_fm_unet_v2` / `plan_fm_v2` / `plan_fm_v3` / `plan_fm_hp_tune` | various | Leave — separate packages |

These all point to **different packages** (`flow_matcher/`, `flow_matcher_v2/`, etc.) that are NOT being renamed. Confirmed safe to leave as-is.

### Checkpoint Compatibility — Safe ✅

All 3 eval scripts (`eval_flow_matching_v3_*.py`) use `load_diffusion_with_override()` which:
1. Resolves the config's `target_class` against the current codebase
2. Detects pickle/config class mismatch
3. Overrides the pickled class with the config class
4. Filters kwargs via `inspect.signature()` to avoid `TypeError`

**Existing checkpoints will load correctly** — the eval scripts will print a `[WARNING] Pickled diffusion class does not match` and proceed. No retraining required for inference.

### Filesystem Path Impact — Informational

The class name is embedded in checkpoint directory paths via `{diffusion}` template (e.g. `Dmodels.diffusion.GaussianDiffusion`). After the rename, **new** training runs produce directories with `FlowMatchingODE` in the path. Old checkpoints under the `GaussianDiffusion` path remain valid — just ensure `diffusion_loadpath` in the plan config still resolves to the old directory when evaluating old models.

### Updated File Count

Original report: ~17 files. With additions above: **~20 files** (adding 2 shim files + `imf_engine.py` docstring).

---

## Claude Review of Audit

**Date:** 2026-05-22

### Agree ✅ — All 5 additions confirmed correct

**Item 1 — `diffuser/` shim (critical):** Verified. `diffuser/` is root-level and installed (`diffuser.egg-info` present). Confirmed two files that will `ImportError`:
- `diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py:1` — `from flow_matcher_v3_imeanflow.models.imf_diffusion import iMFDiffusion`
- `diffuser/flow_matcher_v3_imeanflow/models/__init__.py:2` — `from .imf_diffusion import iMFDiffusion`

Both must be updated to `iMeanFlowODE`. **Adding to implementation scope.**

**Additional safe boundary confirmed:** `diffuser/models/diffusion.py` has its own independent `GaussianDiffusion` class (DDPM baseline — line 15). This is NOT a shim and must NOT be renamed. `diffuser/sampling/policies.py` also belongs to the DDPM package — leave it untouched.

**Items 2–4 — Internal refs + docstring strings:** Correct. The `visual_gaussian_diffusion.py` parent-class import and inheritance line are implicit requirements of the rename that were missing from the report. Confirmed required.

**Item 5 — Legacy FM variant config blocks:** Confirmed safe. Each `flow_matching` / `flow_matching_v2` / `flow_matching_v3` block in `avoiding-d3il.py` points to a separate package directory (`flow_matcher/`, `flow_matcher_v2/`, `flow_matcher_v3/`). Those packages are not being renamed.

**Checkpoint safety + path note:** Agreed. `load_diffusion_with_override()` absorbs the class-name mismatch on inference. New training runs will produce new-style directory names. No action needed for existing checkpoints.

### Final Confirmed File List

```
ACTIVE MODULE CLASSES (8 files)
flow_matcher_v3_ode_selectable/models/diffusion.py        GaussianDiffusion → FlowMatchingODE
flow_matcher_v3_drifting/models/diffusion.py              GaussianDiffusion → FlowMatchingODE
flow_matcher_v3_imeanflow/models/diffusion.py             GaussianDiffusion → FlowMatchingODE
flow_matcher_v3_imeanflow/models/imf_diffusion.py         iMFDiffusion → iMeanFlowODE  (+error msg line 306)
fm_visual_aligning/models/diffusion.py                    GaussianDiffusion → FlowMatchingODE
fm_visual_aligning/models/visual_gaussian_diffusion.py    VisualGaussianDiffusion → VisualFlowMatching
                                                          + internal import (L2) + inheritance (L6) + docstrings

EXPORTS (4 files)
flow_matcher_v3_ode_selectable/models/__init__.py
flow_matcher_v3_drifting/models/__init__.py
flow_matcher_v3_imeanflow/models/__init__.py
fm_visual_aligning/models/__init__.py

GHOST REFERENCE CLEANUP (3 files)
flow_matcher_v3_ode_selectable/sampling/policies.py:27    remove dead GaussianInvDynDiffusion check
flow_matcher_v3_drifting/sampling/policies.py:27          remove dead GaussianInvDynDiffusion check
flow_matcher_v3_imeanflow/sampling/policies.py:27         remove dead GaussianInvDynDiffusion check
(fm_visual_aligning has no policies.py — resolved)

SHIM (2 files — critical)
diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py    re-export → iMeanFlowODE
diffuser/flow_matcher_v3_imeanflow/models/__init__.py          export → iMeanFlowODE

EVAL/TRAIN SCRIPTS (4 files)
FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py   L155 string check
FM_v3_drifting_test/eval_flow_matching_v3_drifting.py               L155 string check
FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py             L156 string check
fm_visual_aligning_test/train_fm_visual_aligning.py                 L199 import + usage

DOCSTRING (1 file)
flow_matcher_v3_imeanflow/models/imf_engine.py:104                   docstring update

CONFIG (2 files — surgical)
config/avoiding-d3il.py         6 class-path strings in FM blocks only
config/aligning-d3il-visual.py  2 class-path strings in fm_visual_aligning blocks only

DO NOT TOUCH
diffuser/models/diffusion.py    own DDPM GaussianDiffusion — not a shim
diffuser/sampling/policies.py   DDPM package — leave
fm_encdec_vision/*              abandoned
All Archived_Codes/             archived
GaussianNormalizer everywhere   utility, correct name
Config key 'diffusion'          deferred
```

**Total: 24 files. Ready for implementation.**
