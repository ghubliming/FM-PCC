# Gen7 FM-PCC Visual Aligning — Implementation Plan

**Date:** 2026-05-20  
**Base:** Gen6V4 (`diffuser_visual_aligning` + `diffuser_visual_aligning_test`) — Proofed  
**Goal:** Replace the DDPM diffusion engine with a Flow Matching (FM) ODE engine, preserving the entire Visual-DPCC framework (dataset, projector, eval sim, normalizers).

> **IMPORTANT**
> This plan follows the Expert Idea: **Copy-Modify** from the Gen6V4 proven base.  
> Reference materials:
> - **Proofed:** `flow_matcher_v3/models/diffusion.py` (FMv3ODE — correct FM math)
> - **Bad code / good principle:** `fm_encdec_vision/` (abandoned Gen7 attempt — running but architecturally wrong)

---

## Table of Contents

1. [Overview: What Changes and What Stays](#1-overview)
2. [Phase 0: Pre-Requisites](#2-phase-0)
3. [Phase 1: Core Module](#3-phase-1)
4. [Phase 2: Entry Scripts](#4-phase-2)
5. [Phase 3: Config Blocks](#5-phase-3)
6. [Phase 4: Eval YAML](#6-phase-4)
7. [Phase 5: Integration Testing](#7-phase-5)
8. [Risk Register](#8-risk-register)
9. [File Manifest](#9-file-manifest)

---

## 1. Overview

### What Changes (DDPM → FM)

| Component | Gen6V4 (DDPM) | Gen7 (FM) | Why |
|-----------|--------------|-----------|-----|
| **Noise schedule** | Cosine β schedule → `sqrt_alphas_cumprod` buffers | Linear interpolation `x_t = (1-t)*noise + t*data` | FM is schedule-free; simpler, better gradients |
| **Training target** | Predict ε (noise) | Predict velocity `v = x_data - x_noise` | FM learns the vector field directly |
| **Time sampling** | `randint(0, T)` → discrete integer t | `Beta(α,β)` → continuous t ∈ [0,1] | Continuous time matches ODE theory |
| **Inference loop** | Reverse diffusion T→0, stochastic `p_sample` with noise injection | Forward ODE integration 0→1, deterministic Euler steps | Deterministic = reproducible; no posterior variance needed |
| **`p_mean_variance`** | `predict_start_from_noise` → `q_posterior` → mean+variance | `velocity * dt` → model_mean, zeros for variance | No posterior math; single Euler step |
| **`p_sample`** | mean + σ·noise (stochastic) | mean only (deterministic ODE) | FM ODE is deterministic |
| **Initial noise** | `0.5 * randn(shape)` | `0.5 * randn(shape)` | Same (inherited convention) |
| **Config keys** | `n_diffusion_steps`, `predict_epsilon` | `flow_steps_v3`, `time_beta_alpha_v3`, `time_beta_beta_v3`, `ode_solver_backend_v3` | New FM-specific hyperparameters |

### What Stays Unchanged (Copy Verbatim)

| Component | Reason |
|-----------|--------|
| `datasets/sequence.py` — `ParityAligningDataset` | 9D trajectory loading is engine-agnostic |
| `datasets/normalization.py` — `LimitsNormalizer` | Normalizer math is engine-agnostic |
| `sampling/projection.py` — DPCC `Projector` | SLSQP math operates on denormalized trajectories; engine-agnostic |
| `models/visual_unet.py` — `VisualUNet` | The backbone network (ResNet encoder + UNet) is engine-agnostic; it takes `(x, cond, t)` and returns a prediction. DDPM calls it to predict ε; FM calls it to predict v. Same architecture, different training target. |
| `models/unet1d_temporal_cond.py` — `UNet1DTemporalCondModel` | Internal backbone; no engine coupling |
| `models/helpers.py` — `apply_conditioning`, `Losses` | Conditioning and loss utilities are engine-agnostic |
| `utils/` — `Trainer`, `Config`, `Parser`, serialization | Training loop, config system, checkpoint I/O are engine-agnostic |
| `d3il/simulation/aligning_sim.py` — `Aligning_Sim` | Sim wrapper is engine-agnostic |
| Eval logging pattern (Tee, diagnostics, expert reference) | Logging infra has no engine dependency |

### Key Architectural Principle

The **only file that fundamentally changes** is `models/diffusion.py` — the `GaussianDiffusion` base class. Everything else is either copied verbatim or receives trivial import-path updates. The `VisualGaussianDiffusion` subclass needs minor adaptation (override `loss()` to use Beta time sampling + velocity target instead of discrete ε target), but its `forward()` and `p_mean_variance()` override structure stays the same.

---

## 2. Phase 0: Pre-Requisites

These are the advisory fixes from the triple-audited `GEN7_UPGRADE_EVALUATION.md`. They must be applied to the Gen6V4 base **before** copying to Gen7, so Gen7 starts clean.

### 2.0.1 BGR/RGB Channel Mismatch (HIGH-PRIORITY)

**Problem:** Training pipeline produces RGB (`cv2.imread` → `cvtColor(BGR2RGB)`). Inference env produces BGR (`get_image()` → `cvtColor(RGB2BGR)`).

**Action:** In `d3il/environments/d3il/simulation/aligning.py`, lines 211-215 — **remove** the `cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)` calls from `get_observation()`. MuJoCo renderer returns RGB natively; training dataset loads RGB. Both pipelines will then agree on RGB.

**Scope:** This is a D3IL env-package edit. If the "copy-modify only" constraint applies to d3il, use **Option C** instead: add `img = img[::-1]` channel flip in `aligning_sim.py` at inference time (before passing to agent). Either way, training and inference must use the same channel order.

**Verification:** Print `bp_image.mean(axis=(1,2))` at both training load time and inference time — the per-channel means should be in the same ballpark.

> ⚠️ **AUDIT [Claude Sonnet 4.6]:**  
> `/note — side effect on Gen6V4: this fix also immediately benefits any pending Gen6V4 evals. The Gen6V4 model trains on RGB; with the fix, inference also receives RGB (matching). Without the fix, Gen6V4 inference received BGR (mismatch, but empirically tolerated per fix7). Apply this before the next Gen6V4 eval run for a cleaner baseline, since the only remaining unknown is the checkpoint sweet-spot./`

### 2.0.2 Fix Seeding (ADVISORY)

**Problem:** `aligning_sim.py:58-64` uses `pid` (always 0 with `n_cores=1`) instead of the configured `seed` for RNG initialization. All eval seeds produce identical `torch.randn` noise.

**Action:** Replace `random.seed(pid)` / `torch.manual_seed(pid)` / `np.random.seed(pid)` with the actual `self.seed` value. This gives each eval seed a distinct stochastic rollout.

### 2.0.3 Fix Comments (ADVISORY)

- `aligning_sim.py:86-89` — Fix 11 comment has inverted reasoning chain. Replace with accurate description (training=RGB, inference=BGR, model robust per fix7 evidence).
- `normalization.py:160` — Comment says "constant dims map to 0"; correct is "map to -1".
- `aligning_sim.py:93` — Add `.copy()` to `des_robot_pos` for symmetry with line 94.

> ⚠️ **AUDIT [Claude Sonnet 4.6]:**  
> `/not do — aligning_sim.py:86-89 comment fix: already applied in this session (2026-05-20). Skip this item; the other two (normalization.py:160 and aligning_sim.py:93 .copy()) still need to be done./`

---

## 3. Phase 1: Core Module — `fm_visual_aligning/`

### 3.1 Directory Scaffold

**Action:** Create `fm_visual_aligning/` as a sibling to `diffuser_visual_aligning/`. Mirror its structure exactly:

```
fm_visual_aligning/
├── __init__.py
├── setup.py
├── models/
│   ├── __init__.py
│   ├── diffusion.py          ← NEW (FM ODE engine from flow_matcher_v3)
│   ├── visual_gaussian_diffusion.py  ← MODIFIED (FM-adapted subclass)
│   ├── visual_unet.py        ← COPY (import path updated)
│   ├── helpers.py             ← MODIFIED (remove cosine_beta_schedule, keep apply_conditioning)
│   └── unet1d_temporal_cond.py ← COPY (verbatim)
├── datasets/
│   ├── __init__.py            ← COPY (update import paths)
│   ├── sequence.py            ← COPY (verbatim — ParityAligningDataset)
│   └── normalization.py       ← COPY (verbatim — LimitsNormalizer)
├── sampling/
│   ├── __init__.py            ← COPY
│   └── projection.py          ← COPY (verbatim — DPCC Projector)
└── utils/
    ├── __init__.py            ← COPY (update import paths)
    ├── training.py            ← COPY (verbatim — Trainer)
    ├── config.py              ← COPY (verbatim)
    ├── serialization.py       ← COPY (verbatim)
    ├── arrays.py              ← COPY (verbatim)
    ├── logger.py              ← COPY (verbatim)
    ├── progress.py            ← COPY (verbatim)
    ├── plot.py                ← COPY (verbatim)
    ├── setup.py               ← COPY (verbatim)
    └── constraints_helpers.py ← COPY (verbatim)
```

**Import path rule:** Every `import diffuser_visual_aligning.X` becomes `import fm_visual_aligning.X`. Use find-and-replace across the copied files. No other logic changes in copied files.

### 3.2 `models/diffusion.py` — FM ODE Engine

**Source:** Copy from `flow_matcher_v3/models/diffusion.py` (the proofed FM engine).

**Modifications required:**

1. **Change the import line:**
   - FROM: `import diffuser.utils as utils`
   - TO: `import fm_visual_aligning.utils as utils`

2. **Change helpers import:**
   - FROM: `from .helpers import (apply_conditioning, Losses,)`
   - TO: Same (it resolves to `fm_visual_aligning.models.helpers`)

3. **Keep `clip_denoised` parameter** in constructor signature for interface compatibility, but it is functionally unused in FM (no clamp path). The VisualGaussianDiffusion subclass may still reference it.

4. **Key FM methods to verify are present from the FMv3ODE source:**

   | Method | Purpose | Key Math |
   |--------|---------|----------|
   | `_time_from_timestep(t)` | Convert integer t to float ∈ [0,1] | `t.float() / (n_timesteps - 1)` |
   | `_predict_velocity(x, cond, t)` | Network forward with optional CFG | `v_uncond + w*(v_cond - v_uncond)` |
   | `q_sample(x_start, t, noise)` | FM interpolation (training) | `(1-t)*noise + t*x_start` |
   | `p_losses(x_start, cond, t)` | Velocity loss | `loss(v_pred, x_start - x_base)` |
   | `loss(x, cond)` | Beta time sampling + p_losses | `Beta(α,β).sample(); t = 1-t` |
   | `p_mean_variance(x, cond, t)` | Single Euler step | `x + velocity * dt` (dt=1/flow_steps) |
   | `p_sample(x, cond, t)` | Deterministic step (no noise) | Returns model_mean only |
   | `p_sample_loop(shape, cond, ...)` | Forward ODE loop 0→1 | `for i in range(flow_steps): ...` |

5. **Projector integration:** The FMv3ODE `p_mean_variance` and `p_sample_loop` already have projector hooks (`projector.gradient`, `projector.project()`, `diffusion_timestep_threshold`). Verify these match the Gen6V4 projector interface. The threshold logic is inverted vs DDPM:
   - DDPM: projector active when `t <= threshold * T` (near t=0, end of reverse chain)
   - FM: projector active when `loop_idx >= (1 - threshold) * flow_steps` (near t=1, end of forward chain)
   - FMv3ODE already implements the FM-correct direction. Verify this is preserved.

6. **DO NOT bring in `torchdiffeq` support** from `fm_encdec_vision/models/diffusion.py`. That code adds complexity with minimal proven benefit. Keep the simple legacy Euler loop. The `ode_solver_backend_v3` parameter can stay in the constructor for future extension but should default to `'legacy_euler'` and the torchdiffeq code path should not be included initially.

> ⚠️ **AUDIT [Claude Sonnet 4.6]:**  
> `/other way to do — ode_solver params in constructor: add BOTH ode_solver_backend_v3='legacy_euler' AND ode_solver_method_v3='euler' as no-op constructor params, not just ode_solver_backend_v3 alone. Reason: Config.__call__() in utils/config.py passes ALL config dict keys directly to __init__() (no filtering — verified at config.py:92). The plan's Phase 3 config blocks (5.2) include both ode_solver_backend_v3 and ode_solver_method_v3. The existing args_to_watch_fmv3_ode_plan also uses ode_solver_method_v3. If only ode_solver_backend_v3 is added to the constructor, ode_solver_method_v3 will cause TypeError at construction time./`

### 3.3 `models/visual_gaussian_diffusion.py`

**Source:** Copy from `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` (Gen6V4).

**Modifications required:**

1. **Change base class import:**
   - FROM: `from diffuser_visual_aligning.models.diffusion import GaussianDiffusion`
   - TO: `from fm_visual_aligning.models.diffusion import GaussianDiffusion`

2. **Change helpers import:**
   - FROM: `from diffuser_visual_aligning.models.helpers import apply_conditioning`
   - TO: `from fm_visual_aligning.models.helpers import apply_conditioning`

3. **Override `loss()` — THE CRITICAL CHANGE:**
   The Gen6V4 `loss()` uses `torch.randint(0, T)` (discrete time) and calls `self.p_losses()` which computes ε-prediction loss. The FM version must:
   - Sample continuous time from `Beta(α,β)` distribution
   - Call the FM `p_losses()` which computes velocity-prediction loss
   
   **Method:** Replace the body of `loss()` to use Beta time sampling, following the pattern from `flow_matcher_v3/models/diffusion.py:loss()`:
   ```
   alpha = torch.tensor(self.time_beta_alpha_v3)
   beta = torch.tensor(self.time_beta_beta_v3)
   t = Beta(alpha, beta).sample((batch_size,))
   t = 1.0 - t
   return self.p_losses(x, cond, t)
   ```
   
   The cond dict construction (extracting `primary_img`, `wrist_img`, `obs_0`, `obs_seq` and building `{'visual': ..., 0: ...}`) stays **exactly the same** as Gen6V4.

4. **Override `p_mean_variance()` — SIMPLIFY:**
   The Gen6V4 version does: predict ε → `predict_start_from_noise` → optional clamp → `q_posterior` → mean+variance. The FM version is much simpler: predict velocity → Euler step → done. 
   
   **Decision:** The FM base class `p_mean_variance` already does the right thing (velocity * dt). The subclass override in Gen6V4 only existed to change the clamp behavior (±5 on actions instead of ±1). Since FM has no clamp (`clip_denoised` is irrelevant for FM), the subclass **does not need to override `p_mean_variance` at all**. Delete the override; inherit the base class version.

5. **`forward()` — COPY VERBATIM** from Gen6V4. The tuple-unpacking logic for `cond[0]` (converting VisualAgentWrapper format to internal format) is engine-agnostic.

### 3.4 `models/visual_unet.py`

**Source:** Copy from `diffuser_visual_aligning/models/visual_unet.py` (Gen6V4).

**Modifications required:**

1. **Change UNet import:**
   - FROM: `from diffuser_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel`
   - TO: `from fm_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel`

2. **No other changes.** The VisualUNet takes `(x, cond, t)` and returns a tensor of the same shape. For DDPM this tensor is interpreted as ε (noise prediction). For FM this tensor is interpreted as v (velocity prediction). The network architecture is identical — only the training target changes, which is handled by `p_losses()` in the diffusion engine.

3. **TRANSITION_DIM stays 9.** The 9D trajectory `[act(3) | des_c_pos(3) | c_pos(3)]` is task-defined, not engine-defined.

### 3.5 `models/helpers.py` and `unet1d_temporal_cond.py`

**`helpers.py`:**
- **Source:** Copy from `diffuser_visual_aligning/models/helpers.py`.
- **Modification:** The FM engine does NOT use `cosine_beta_schedule()` or `extract()` (those are DDPM-specific). They can be left in place (harmless dead code) or removed for cleanliness. `apply_conditioning()` and `Losses` dict MUST be preserved — they are used by the FM engine.

**`unet1d_temporal_cond.py`:**
- **Source:** Copy verbatim from `diffuser_visual_aligning/models/unet1d_temporal_cond.py`.
- **No modifications.** This is the raw UNet backbone with FiLM conditioning. It has no diffusion/FM coupling.

### 3.6 `datasets/` and `sampling/`

**`datasets/sequence.py` (ParityAligningDataset):**
- Copy verbatim. Update import: `diffuser_visual_aligning` → `fm_visual_aligning`.
- The dataset returns `Batch(trajectories, conditions)` with 9D trajectories and image conditions. This format is engine-agnostic.

**`datasets/normalization.py` (LimitsNormalizer):**
- Copy verbatim. No import path changes needed (it's self-contained with numpy).

**`sampling/projection.py` (DPCC Projector):**
- Copy verbatim. Update import paths only. The SLSQP projector operates on denormalized trajectory tensors and has no awareness of whether the engine is DDPM or FM.

### 3.7 `utils/`

- Copy the entire `diffuser_visual_aligning/utils/` directory.
- Global find-and-replace: `diffuser_visual_aligning` → `fm_visual_aligning` in all `.py` files.
- **Key file: `training.py` (Trainer):** The Trainer calls `self.model.loss(*batch)`. For DDPM, batch unpacks to `(trajectories, conditions)` and calls `VisualGaussianDiffusion.loss(trajectories, conditions)`. For FM, **the same interface works** because the FM `VisualGaussianDiffusion.loss()` has the same signature — it just uses Beta time sampling internally instead of `randint`.
- **No Scaler:** The Gen6V4 Trainer does NOT take a `scaler` argument (unlike `fm_encdec_vision`). The normalizer is applied inside the dataset. Keep this pattern.

---

## 4. Phase 2: Entry Scripts — `fm_visual_aligning_test/`

### 4.1 `train_fm_visual_aligning.py`

**Source:** Copy from `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` (Gen6V4).

**Modifications:**

1. **Import path changes:**
   - `import diffuser_visual_aligning.utils as utils` → `import fm_visual_aligning.utils as utils`
   - `from diffuser_visual_aligning.datasets.sequence import ParityAligningDataset` → `from fm_visual_aligning.datasets.sequence import ParityAligningDataset`
   - `from diffuser_visual_aligning.models.visual_unet import VisualUNet` → `from fm_visual_aligning.models.visual_unet import VisualUNet`
   - `from diffuser_visual_aligning.models.visual_gaussian_diffusion import VisualGaussianDiffusion` → `from fm_visual_aligning.models.visual_gaussian_diffusion import VisualGaussianDiffusion`

2. **Experiment name:** Change `experiment='visual_aligning_dpcc'` → `experiment='fm_visual_aligning'` (matches new config block name in Phase 3).

3. **Diffusion config construction — ADD FM PARAMETERS:**
   The Gen6V4 script constructs `VisualGaussianDiffusion` with DDPM-specific params. Add FM params:
   ```python
   diffusion_config = utils.Config(
       VisualGaussianDiffusion,
       ...
       # --- NEW FM parameters ---
       time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.5),
       time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
       flow_steps_v3=getattr(args, 'flow_steps_v3', 16),
       ode_inference_steps_v3=getattr(args, 'ode_inference_steps_v3', 16),
       # --- Keep from Gen6V4 ---
       horizon=args.horizon,
       observation_dim=6,
       action_dim=args.action_dim,
       goal_dim=0,
       n_timesteps=_n_diff_steps,
       loss_type=args.loss_type,
       clip_denoised=False,
       predict_epsilon=True,  # kept for interface compat; FM ignores it
       action_weight=getattr(args, 'action_weight', 10.0),
   )
   ```

4. **W&B project name:** Update `--wandb-project` default to `'FMPCC-fm-visual-aligning'`.

5. **Everything else stays identical:** Dataset loading, normalizer saving, Trainer construction, resume logic, wandb logging — all verbatim.

### 4.2 `eval_fm_visual_aligning.py`

**Source:** Copy from `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4).

**Modifications:**

1. **Import path changes:**
   - `import diffuser_visual_aligning.utils as utils` → `import fm_visual_aligning.utils as utils`
   - `from diffuser_visual_aligning.sampling.projection import Projector` → `from fm_visual_aligning.sampling.projection import Projector`

2. **Experiment name:** `experiment='plan_fm_visual_aligning'` (matches new config block name).

3. **`clip_denoised` override at line 696:** The Gen6V4 script forces `diffusion_model.clip_denoised = False`. For FM, `clip_denoised` is a no-op but keep the line for safety — it won't hurt.

4. **Flow steps diagnostic:** Add a print after model load:
   ```python
   _flow_steps = getattr(diffusion_model, 'flow_steps_v3', '?')
   print(f'[ eval ] FM flow_steps_v3 = {_flow_steps}')
   ```

5. **VisualAgentWrapper — NO CHANGES.** The wrapper calls `self.model(cond, projector=...)` which dispatches to `VisualGaussianDiffusion.forward()` → `conditional_sample()` → `p_sample_loop()`. The FM engine handles the ODE loop internally. The wrapper is engine-agnostic.

6. **Projector setup — NO CHANGES.** `setup_dpcc_projector()` builds the same SLSQP projector with the same 9D constraint layout. The FM projector hooks (in `p_sample_loop`) are already FM-compatible (from FMv3ODE).

7. **All logging, diagnostics, video capture, report generation — COPY VERBATIM.**

---

## 5. Phase 3: Config — `config/aligning-d3il-visual.py`

### 5.1 Training Config Block — `fm_visual_aligning`

Add a new config block `'fm_visual_aligning'` to the `base` dict. Use the existing `'visual_aligning_dpcc'` as template, with these changes:

| Parameter | Gen6V4 Value | Gen7 FM Value | Rationale |
|-----------|-------------|---------------|-----------|
| `model` | `diffuser_visual_aligning.models.visual_unet.VisualUNet` | `fm_visual_aligning.models.visual_unet.VisualUNet` | New package path |
| `diffusion` | `diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion` | `fm_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion` | New package path |
| `n_diffusion_steps` | `100` | `16` | FM needs far fewer steps; FMv3ODE default is 16 |
| `time_beta_alpha_v3` | (absent) | `1.5` | Beta distribution alpha for time sampling |
| `time_beta_beta_v3` | (absent) | `1.0` | Beta distribution beta for time sampling |
| `prefix` | `visual_aligning_dpcc/` | `fm_visual_aligning/` | New log directory |
| `exp_name` | `watch(args_to_watch_dpcc_train)` | `watch(args_to_watch_fmv3_ode_train)` | Use FM-specific watch keys (includes `a`, `b` for Beta params) |

All other params (`action_dim=3`, `obs_dim=6`, `horizon=8`, `dim=32`, `batch_size=32`, `learning_rate=2e-4`, etc.) stay identical.

### 5.2 Planning/Eval Config Block — `plan_fm_visual_aligning`

Add `'plan_fm_visual_aligning'` using `'plan_visual_aligning_dpcc'` as template:

| Parameter | Gen6V4 Value | Gen7 FM Value | Rationale |
|-----------|-------------|---------------|-----------|
| `diffusion` | `diffuser_visual_aligning...` | `fm_visual_aligning...` | New package path |
| `n_diffusion_steps` | `100` | `16` | Must match training |
| `flow_steps_v3` | (absent) | `16` | ODE integration steps at inference |
| `time_beta_alpha_v3` | (absent) | `1.5` | For completeness (used by constructor) |
| `time_beta_beta_v3` | (absent) | `1.0` | For completeness |
| `ode_solver_backend_v3` | (absent) | `'legacy_euler'` | Simple Euler ODE solver |
| `ode_solver_method_v3` | (absent) | `'euler'` | Solver method identifier |
| `prefix` | `f:plans/visual_aligning_dpcc/...` | `f:plans/fm_visual_aligning/...` | New log directory |
| `diffusion_loadpath` | `f:visual_aligning_dpcc/...` | `f:fm_visual_aligning/...` | Points to new training output |
| `exp_name` | `watch(args_to_watch_dpcc_plan)` | `watch(args_to_watch_fmv3_ode_plan)` | FM watch keys |

**CRITICAL:** `max_path_length` in training and planning MUST match (both embed in the checkpoint directory name via `args_to_watch`). Keep both at `1000`.

### 5.3 `args_to_watch` Keys

The existing `args_to_watch_fmv3_ode_train` and `args_to_watch_fmv3_ode_plan` already include FM-specific keys (`time_beta_alpha_v3`, `flow_steps_v3`, `ode_solver_method_v3`). Reuse them directly. They are already defined in the config file.

**Important:** These watch keys do NOT include `if_vision` or `max_path_length`. If you need those in the loadpath, either add them to the watch keys or use the dpcc watch keys. Recommend: create new FM-visual watch keys that combine FM params with visual params:
```python
args_to_watch_fm_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),
    ('time_beta_beta_v3', 'b'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
]
```

> ⚠️ **AUDIT [Claude Sonnet 4.6]:**  
> `/other way to do — also need a plan-side watch key list: the plan config uses args_to_watch_fmv3_ode_plan which has flow_steps_v3 (K), ode_solver_method_v3 (M), diffusion_timestep_threshold (T), diffusion (D) — but is missing if_vision. Create a corresponding args_to_watch_fm_visual_plan by extending args_to_watch_fmv3_ode_plan with ('if_vision', 'V'):`
> ```python
> args_to_watch_fm_visual_plan = [
>     ('prefix', ''),
>     ('horizon', 'H'),
>     ('flow_steps_v3', 'K'),
>     ('ode_solver_method_v3', 'M'),
>     ('diffusion_timestep_threshold', 'T'),
>     ('diffusion', 'D'),
>     ('if_vision', 'V'),
> ]
> ```
> Use args_to_watch_fm_visual_plan (not args_to_watch_fmv3_ode_plan) for the plan config exp_name. The diffusion_loadpath f-string must independently spell out the train directory name components to match what args_to_watch_fm_visual_train generates./`

---

## 6. Phase 4: Eval YAML — `config/visual_aligning_eval.yaml`

**Action:** Add `'fm_visual_aligning'` to the `exps` list:

```yaml
exps: [
  'aligning-d3il-visual',
  'visual_aligning_dpcc',
  'fm_visual_aligning',       # ← ADD
]
```

**No other YAML changes.** All shared parameters (seeds, projection_variants, workspace_bounds, constraint_types, n_contexts, diffusion_timestep_threshold) apply identically to the FM engine. The projector's `diffusion_timestep_threshold: 0.5` is interpreted correctly by the FM `p_sample_loop` (it applies near the END of the forward ODE, which is the data end — same semantic intent as the DDPM version).

---

## 7. Phase 5: Integration Testing

### 7.1 Smoke Test: Module Import

```bash
python -c "from fm_visual_aligning.models.visual_gaussian_diffusion import VisualGaussianDiffusion; print('OK')"
```

Must succeed without import errors. Verifies the entire dependency chain (fm_visual_aligning → models → helpers → utils).

### 7.2 Smoke Test: Training Forward Pass

Run training for 10 steps with `--num-seeds 1`:
```bash
python fm_visual_aligning_test/train_fm_visual_aligning.py \
    --seed 0 --num-seeds 1 \
    n_train_steps=10 n_steps_per_epoch=5
```

**Check:**
- Loss decreases (or at least doesn't NaN)
- `obs_normalizer.pkl` and `act_normalizer.pkl` are saved
- Checkpoint `state_5.pt` is written
- Print confirms `n_timesteps` and `flow_steps_v3` values

### 7.3 Smoke Test: Eval Forward Pass

With the 10-step checkpoint, run 1 context eval:
```bash
python fm_visual_aligning_test/eval_fm_visual_aligning.py \
    --seed 0 --record none
```

**Check:**
- Model loads without error
- `flow_steps_v3` prints correctly
- Inference produces actions in physical range (not ±94!)
- Rollout completes without crash
- The `diag_first_replan.txt` shows normalized actions in [-1, 1]

### 7.4 Full Training Run

After smoke tests pass:
```bash
python fm_visual_aligning_test/train_fm_visual_aligning.py \
    --seeds 5 6 7 8 9 --use-wandb
```

Monitor loss curves on W&B. Compare convergence rate vs Gen6V4 DDPM (FM should converge faster with fewer steps).

### 7.5 Checkpoint Sweet Spot Validation

Based on Gen6V4 audit findings: the DDPM model had a ~50k step sweet spot where actions were well-behaved. For FM:
- Save checkpoints every 10k steps
- Eval at 10k, 20k, 30k, 50k, 100k
- Look for the action range window where `diag_first_replan.txt` shows actions in [-1, +1] normalized

---

## 8. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | FM velocity target produces different gradient landscape → VisualUNet may need different `dim`/`dim_mults` | Low | Medium | Start with identical architecture (dim=32). If loss plateaus, try dim=64 or dim=128. |
| R2 | Beta time distribution (α=1.5, β=1.0) may not be optimal for 9D visual trajectories | Medium | Low | These are the FMv3ODE defaults, proven on non-visual tasks. Tunable — try α=1.0,β=1.0 (uniform) if results are poor. |
| R3 | `flow_steps_v3=16` may be too few for the 9D trajectory space | Medium | Medium | Start with 16 (FMv3 default). If action quality is poor, try 32 or 64. More steps = better ODE approximation but slower inference. |
| R4 | FM deterministic sampling may produce less diverse trajectories than stochastic DDPM | Low | Low | FM ODE is deterministic given initial noise. Diversity comes from different noise draws (batch_size>1). Same mechanism as DDPM. |
| R5 | `clip_denoised=False` in FM means NO safety clamp at any stage | Low | Medium | Actions are denormalized through LimitsNormalizer which bounds them. The projector (when enabled) also constrains. No additional clamp needed. |
| R6 | Import path find-and-replace misses a file → runtime ImportError | High | Low | Run the Phase 5.1 smoke test immediately after copying. `grep -r "diffuser_visual_aligning" fm_visual_aligning/` to catch stragglers. |
| R7 | Config `args_to_watch` mismatch between train and plan blocks → FileNotFoundError on eval | High | High | Compare the generated checkpoint directory name from training against the `diffusion_loadpath` in the plan config. They must match character-for-character. |

---

## 9. File Manifest

### New Files to Create

| File | Source | Action |
|------|--------|--------|
| `fm_visual_aligning/__init__.py` | `diffuser_visual_aligning/__init__.py` | Copy, update paths |
| `fm_visual_aligning/setup.py` | `diffuser_visual_aligning/setup.py` | Copy, update package name |
| `fm_visual_aligning/models/__init__.py` | `diffuser_visual_aligning/models/__init__.py` | Copy, update paths |
| `fm_visual_aligning/models/diffusion.py` | `flow_matcher_v3/models/diffusion.py` | **NEW** — FM ODE engine, update imports |
| `fm_visual_aligning/models/visual_gaussian_diffusion.py` | `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` | **MODIFY** — FM loss, remove p_mean_variance override |
| `fm_visual_aligning/models/visual_unet.py` | `diffuser_visual_aligning/models/visual_unet.py` | Copy, update imports |
| `fm_visual_aligning/models/helpers.py` | `diffuser_visual_aligning/models/helpers.py` | Copy (optionally remove DDPM-only functions) |
| `fm_visual_aligning/models/unet1d_temporal_cond.py` | `diffuser_visual_aligning/models/unet1d_temporal_cond.py` | Copy verbatim |
| `fm_visual_aligning/datasets/__init__.py` | `diffuser_visual_aligning/datasets/__init__.py` | Copy, update paths |
| `fm_visual_aligning/datasets/sequence.py` | `diffuser_visual_aligning/datasets/sequence.py` | Copy, update paths |
| `fm_visual_aligning/datasets/normalization.py` | `diffuser_visual_aligning/datasets/normalization.py` | Copy verbatim |
| `fm_visual_aligning/sampling/__init__.py` | `diffuser_visual_aligning/sampling/__init__.py` | Copy |
| `fm_visual_aligning/sampling/projection.py` | `diffuser_visual_aligning/sampling/projection.py` | Copy, update paths |
| `fm_visual_aligning/utils/*.py` (all files) | `diffuser_visual_aligning/utils/*.py` | Copy, update paths |
| `fm_visual_aligning_test/train_fm_visual_aligning.py` | `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | **MODIFY** — add FM params, update imports |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | **MODIFY** — update imports, add flow_steps diagnostic |

### Existing Files to Modify

| File | Change |
|------|--------|
| `config/aligning-d3il-visual.py` | Add `fm_visual_aligning` and `plan_fm_visual_aligning` config blocks + new `args_to_watch` |
| `config/visual_aligning_eval.yaml` | Add `'fm_visual_aligning'` to `exps` list |

### Files NOT Modified

| File | Reason |
|------|--------|
| `diffuser_visual_aligning/**` | Gen6V4 base stays untouched — proven, don't regress |
| `diffuser_visual_aligning_test/**` | Gen6V4 entry scripts stay untouched |
| `d3il/simulation/aligning_sim.py` | Sim wrapper is engine-agnostic (Phase 0 fixes are separate) |
| `d3il/simulation/aligning.py` | Only modified in Phase 0 for BGR fix |
| `flow_matcher_v3/**` | Reference only — not modified |
| `fm_encdec_vision/**` | Abandoned — reference only, not modified |
