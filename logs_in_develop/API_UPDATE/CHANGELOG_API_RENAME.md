# API Rename Changelog — FM Class Names

**Date:** 2026-05-25 (updated 2026-05-26)  
**Branch:** update_into_FM  
**Scope:** 4 active FM modules + eval scripts + configs  
**Reference:** `API_RENAME_REPORT.md` (same directory) for rationale and audit trail

---

## Summary of Renames

| Old Name | New Name | Module |
|---|---|---|
| `GaussianDiffusion` | `FlowMatchingODE` | FMv3ODE only |
| `GaussianDiffusion` | `FlowMatchingDrifting` | FMDrifting only |
| `GaussianDiffusion` | `FlowMatchingIMF` | FMiMeanFlow (base class) only |
| `iMFDiffusion` | `iMeanFlowODE` | FMiMeanFlow (engine class) only |
| `VisualGaussianDiffusion` | `VisualFlowMatching` | FMVisual only |

> **Note (2026-05-26):** Initial rename used `FlowMatchingODE` for all four modules. Drifting and IMF were subsequently refined to `FlowMatchingDrifting` and `FlowMatchingIMF` respectively so each module's class name reflects its algorithm variant, not just the ODE solver shared mechanism.

---

## Changed Files

### 1. Core model class definitions

**`flow_matcher_v3_ode_selectable/models/diffusion.py`**
- `class GaussianDiffusion(nn.Module)` → `class FlowMatchingODE(nn.Module)`

**`flow_matcher_v3_drifting/models/diffusion.py`**
- `class GaussianDiffusion(nn.Module)` → `class FlowMatchingDrifting(nn.Module)` *(initially FlowMatchingODE, refined 2026-05-26)*

**`flow_matcher_v3_imeanflow/models/diffusion.py`**
- `class GaussianDiffusion(nn.Module)` → `class FlowMatchingIMF(nn.Module)` *(initially FlowMatchingODE, refined 2026-05-26)*

**`fm_visual_aligning/models/diffusion.py`**
- `class GaussianDiffusion(nn.Module)` → `class FlowMatchingODE(nn.Module)`

**`flow_matcher_v3_imeanflow/models/imf_diffusion.py`**
- `class iMFDiffusion(nn.Module)` → `class iMeanFlowODE(nn.Module)`
- Error string in `load_state_dict`: `'iMFDiffusion'` → `'iMeanFlowODE'`

**`fm_visual_aligning/models/visual_gaussian_diffusion.py`**
- Import: `from fm_visual_aligning.models.diffusion import GaussianDiffusion` → `FlowMatchingODE`
- `class VisualGaussianDiffusion(GaussianDiffusion)` → `class VisualFlowMatching(FlowMatchingODE)`
- Docstring updated: `"DDPM engine for Visual-DPCC (Gen6V4). Extends GaussianDiffusion with:"` → `"FM engine for Visual-DPCC (Gen6V4). Extends FlowMatchingODE with:"`
- Internal comment: `# base GaussianDiffusion.__init__` → `# base FlowMatchingODE.__init__`

### 2. Module `__init__.py` exports

**`flow_matcher_v3_ode_selectable/models/__init__.py`**
- `from .diffusion import GaussianDiffusion` → `FlowMatchingODE`

**`flow_matcher_v3_drifting/models/__init__.py`**
- `from .diffusion import GaussianDiffusion` → `FlowMatchingDrifting`

**`flow_matcher_v3_imeanflow/models/__init__.py`**
- `GaussianDiffusion` → `FlowMatchingIMF`
- `iMFDiffusion` → `iMeanFlowODE`

**`fm_visual_aligning/models/__init__.py`**
- `GaussianDiffusion` → `FlowMatchingODE`
- `VisualGaussianDiffusion` → `VisualFlowMatching`

### 3. Sampling / policy files — dead-code removal

The `GaussianInvDynDiffusion` class never existed in any FM module. Three `policies.py` files contained dead `if/else` blocks guarded by `__class__.__name__ == 'GaussianInvDynDiffusion'`. These were removed entirely; `self.inverse_dynamics = False` is now set unconditionally.

**`flow_matcher_v3_ode_selectable/sampling/policies.py`**
- Removed dead `GaussianInvDynDiffusion` if/else block
- Retained comment updated: `# Use FlowMatchingODE model`

**`flow_matcher_v3_drifting/sampling/policies.py`**
- Same removal; retained comment: `# Use FlowMatchingDrifting model`

**`flow_matcher_v3_imeanflow/sampling/policies.py`**
- Same removal; retained comment: `# Use FlowMatchingIMF model`

### 4. IMF engine internal reference

**`flow_matcher_v3_imeanflow/models/imf_engine.py`** (line 104)
- `objective in iMFDiffusion.p_losses` → `iMeanFlowODE.p_losses`

### 5. Eval / test scripts

**`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`** (line 155)
- `fm_model.__class__.__name__ == 'GaussianDiffusion'` → `'FlowMatchingODE'`

**`FM_v3_drifting_test/eval_flow_matching_v3_drifting.py`** (line 155)
- `fm_model.__class__.__name__ == 'GaussianDiffusion'` → `'FlowMatchingDrifting'`

**`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`** (line 156)
- `fm_model.__class__.__name__ in ['GaussianDiffusion', 'iMFDiffusion']` → `['FlowMatchingIMF', 'iMeanFlowODE']`

**`FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py`** (line 155) *(missed in first sweep, fixed separately)*
- `fm_model.__class__.__name__ == 'GaussianDiffusion'` → `'FlowMatchingODE'`

### 6. Training script

**`fm_visual_aligning_test/train_fm_visual_aligning.py`**
- Import: `from fm_visual_aligning.models.visual_gaussian_diffusion import VisualGaussianDiffusion` → `VisualFlowMatching`
- `utils.Config(VisualGaussianDiffusion, ...)` → `utils.Config(VisualFlowMatching, ...)`
- Section comment: `# ── 3. Diffusion engine — VisualGaussianDiffusion` → `# ── 3. FM engine — VisualFlowMatching`

### 7. diffuser/ shim package

**`diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py`**
- `from flow_matcher_v3_imeanflow.models.imf_diffusion import iMFDiffusion` → `iMeanFlowODE`

**`diffuser/flow_matcher_v3_imeanflow/models/__init__.py`**
- `from .imf_diffusion import iMFDiffusion` → `iMeanFlowODE`

### 8. Config files

**`config/avoiding-d3il.py`** — 6 surgical changes in FM blocks only (DDPM blocks untouched):
- `flow_matching_v3_ode_selectable` training: `'models.diffusion.GaussianDiffusion'` → `'models.diffusion.FlowMatchingODE'`
- `flow_matching_v3_drifting` training: `'models.diffusion.GaussianDiffusion'` → `'models.diffusion.FlowMatchingDrifting'`
- `flow_matching_v3_imeanflow` training: `'flow_matcher_v3_imeanflow.models.iMFDiffusion'` → `'flow_matcher_v3_imeanflow.models.iMeanFlowODE'`
- `plan_fm_v3_ode_selectable` inference: `GaussianDiffusion` → `FlowMatchingODE`
- `plan_fm_v3_drifting` inference: `GaussianDiffusion` → `FlowMatchingDrifting`
- `plan_fm_v3_imeanflow` inference: `iMFDiffusion` → `iMeanFlowODE`

**`config/aligning-d3il-visual.py`** — 2 surgical changes in `fm_visual_aligning` blocks only:
- Training block: `VisualGaussianDiffusion` → `VisualFlowMatching`
- Plan/inference block: `VisualGaussianDiffusion` → `VisualFlowMatching`

---

## Intentionally Not Changed

| Location | Reason |
|---|---|
| `diffuser/models/diffusion.py` — `class GaussianDiffusion` | Root `diffuser` package is a real DDPM implementation; renaming breaks it |
| `fm_encdec_vision/` — any class | Module is abandoned; do not touch |
| Config key `'diffusion'` in all config files | Deferred; requires simultaneous update of `scripts/train.py` path template logic |
| `README.md`, `examples/` — class name mentions | Documentation, not imported code |
| `Benchmark_ode_solver_Tests/v2-v4/` | References legacy `flow_matching_v3/` package (different from active modules) |
| `*.py.with_calling_log` files | Debug log snapshots, not active code |

---

## Checkpoint Compatibility

Old checkpoints saved with `GaussianDiffusion`/`iMFDiffusion`/`VisualGaussianDiffusion` weights remain loadable. All eval scripts use `load_diffusion_with_override()` which handles the class name mismatch on `torch.load`. No re-training required.

---

## New Training Runs

The config `{diffusion}` path template means new training runs will produce output directories containing the new class name in the path:
- FMv3ODE → `FlowMatchingODE`
- FMDrifting → `FlowMatchingDrifting`
- FMiMeanFlow → `FlowMatchingIMF` or `iMeanFlowODE`
- FMVisual → `VisualFlowMatching`

Old directories with `GaussianDiffusion` in the name remain valid for eval.
