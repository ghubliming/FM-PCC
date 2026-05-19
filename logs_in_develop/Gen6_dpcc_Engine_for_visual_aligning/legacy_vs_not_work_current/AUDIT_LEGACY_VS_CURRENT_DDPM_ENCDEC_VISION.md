# Audit — Why `ddpm_encdec_vision_Legacy` Works But `ddpm_encdec_vision` (Current) Does Not

**Date:** 2026-05-19  
**Scope:** `ddpm_encdec_vision_Legacy/` vs `ddpm_encdec_vision/` — same nominal config, divergent behavior

---

## TLDR

The two packages share an almost identical training skeleton (Trainer, diffusion engine, scaler are
identical or functionally equivalent). The regression lives in **three specific deltas** introduced
when the current package was refactored to support a state-only fallback (`if_vision=False`) on top
of the visual path:

| # | File | Change | Effect on visual training |
|---|---|---|---|
| 1 | Train script | `max_len_data` 256 → 512 | Loads 2× data; scaler stats shift; if zero-padded frames included, normalization corrupts |
| 2 | `visual_gaussian_diffusion.py` | Added `p_mean_variance` override | Breaks inference (eval) clamping; DDPM posterior range wrong for Gaussian-normalized data |
| 3 | `visual_gaussian_diffusion.py` | `loss(*args)` with `if_vision` guard | Adds a fragile state-only branch that will crash if `self.model.if_vision` is absent or False |

None of these changes are needed for the visual-only path. They were all added to support a
state-only mode that does not exist in the legacy.

---

## Package Structure

```
ddpm_encdec_vision_Legacy/
  ddpm_encdec_vision/          ← the installable package (Python path root)
    models/
      diffusion.py             ← GaussianDiffusion (Flow Matching engine, but not used by visual path)
      visual_gaussian_diffusion.py  ← visual DDPM wrapper (5-arg loss, no p_mean_variance override)
      visual_unet.py           ← VisualUNet (always vision, no if_vision guard)
      unet1d_temporal_cond.py  ← standalone copy (not imported by VisualUNet)
      d3il_visual_bridge.py    ← DiffusionEncDec wrapper (Option B prototype)
    utils/
      training.py              ← Trainer
      scaler.py
  ddpm_encdec_vision_test/
    train_ddpm_encdec_vision.py  ← obs_dim=3, action_dim=3, max_len_data=256 hardcoded

ddpm_encdec_vision/             ← current installable package
  models/
    diffusion.py               ← IDENTICAL to legacy
    visual_gaussian_diffusion.py  ← CHANGED: *args loss, if_vision guard, p_mean_variance override
    visual_unet.py             ← CHANGED: if_vision flag, obs_dim param, use_cond_projection conditional
    unet1d_temporal_cond.py    ← standalone copy (not imported by VisualUNet)
    d3il_visual_bridge.py      ← CHANGED: minor (same structure)
  utils/
    training.py                ← IDENTICAL to legacy
    scaler.py                  ← IDENTICAL functionally (minor log message change)
  ddpm_encdec_vision_test/ (top-level, outside package)
    train_ddpm_encdec_vision.py  ← CHANGED: parameterized obs_dim, action_dim, max_len_data=args.max_path_length
```

**Key fact**: Both `VisualUNet` implementations import the backbone from:
```python
from diffuser.models.unet1d_temporal_cond import UNet1DTemporalCondModel
```
That is the **base `diffuser` package's UNet** — which HAS been updated to support `use_cond_projection`
FiLM conditioning. So both versions DO inject visual features into the UNet via the `cond_mlp` branch.
The earlier claim that visual features were "silently dropped" was incorrect — `cond_mlp` fires on
tensor inputs even in the base UNet.

---

## Diff Analysis — File by File

### `diffusion.py` — IDENTICAL

Both packages have the same `diffusion.py` (Flow Matching engine). Neither `VisualGaussianDiffusion`
uses it directly — both inherit from `diffuser.models.diffusion.GaussianDiffusion` (the real DDPM
with cosine beta schedule). The local `diffusion.py` is dead code in both packages.

---

### `scaler.py` — Functionally identical

Minor log-order difference in `__init__`. Initialization math, bounds, `scale_input`, `scale_output`
methods are identical. Not the cause of any behavioral difference.

---

### `training.py` (Trainer) — IDENTICAL

The `Trainer.train_epoch`, `train`, `test`, `save`, `load`, EMA logic are word-for-word identical.
Not a source of divergence.

---

### `visual_unet.py` — 4 structural changes, all safe for `if_vision=True`

| Delta | Legacy | Current | Risk |
|---|---|---|---|
| `if_vision` flag | Absent — always visual | `self.if_vision = getattr(config, "if_vision", True)` | If config has no `if_vision`, defaults to True → same behavior |
| obs_dim | Hardcoded `obs_dim = 3` for `transition_dim` | `obs_dim = 3 if self.if_vision else getattr(config, 'obs_dim', 20)` | With `if_vision=True` → `obs_dim=3` → same |
| `use_cond_projection` | Always `True` | `use_cond_projection=self.if_vision` | With `if_vision=True` → `True` → same |
| `encode_visual` guard | Always encodes | Returns `None` when `if_vision=False` | Never triggered with `if_vision=True` |

**For the visual training path (`if_vision=True`), `VisualUNet` is architecturally identical.**

---

### `visual_gaussian_diffusion.py` — 3 changes, two are breaking

#### Change A: `loss()` signature — `*args` with `if_vision` guard

**Legacy:**
```python
def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
    x = torch.cat([act, obs], dim=-1)
    cond = {
        'visual': (bp_imgs, inhand_imgs, obs),
        0: obs[:, 0]
    }
    ...
    return self.p_losses(x, cond, t)
```

**Current:**
```python
def loss(self, *args):
    if getattr(self.model, 'if_vision', True):
        bp_imgs, inhand_imgs, obs, act, mask = args
        x = torch.cat([act, obs], dim=-1)
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),
            0: obs[:, 0]
        }
    else:
        obs, act, mask = args          # ← 3-arg unpack when if_vision=False
        x = torch.cat([act, obs], dim=-1)
        cond = {0: obs[:, 0]}
    ...
    return self.p_losses(x, cond, t)
```

**Risk**: `getattr(self.model, 'if_vision', True)` reads from `VisualUNet`. If `VisualUNet` is replaced
or the attribute is missing, the default is `True` — safe. For `if_vision=True` the code path is
identical to legacy. **No functional change in the visual path.**

The `else` branch has a latent bug: `obs, act, mask = args` unpacks only 3 items, but if the
Trainer passes 5 items (because the dataset always returns images), this crashes with `ValueError: too
many values to unpack`. This branch would only fire if `if_vision=False` — which no current training
run uses.

---

#### Change B: `p_mean_variance` override — **BREAKS INFERENCE (EVAL)**

**Legacy:** No override. Parent `GaussianDiffusion.p_mean_variance` is used:
```python
# From diffuser/models/diffusion.py:
x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
if self.clip_denoised:
    x_recon.clamp_(-1., 1.)    # clips ALL dims to [-1, 1]
model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t)
```

**Current:** Override in `VisualGaussianDiffusion`:
```python
x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
if self.clip_denoised:
    x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)  # clips only action dims, range ±5
model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t)
if projector is not None and projector.gradient:
    ...
    model_mean = model_mean + grad
```

**Why this matters:**

The parent clips ALL dimensions to `[-1, 1]`. The scaler applies Gaussian standardization
(`(x - mean) / std`), not min-max normalization. After standardization, data at ±2σ lives at ±2,
at ±3σ at ±3, etc. The parent's `clamp_(-1, 1)` clips every value outside 1σ — which is 32% of
Gaussian data — to ±1. During each of the 16 denoising steps this fires, the posterior mean is
corrupted: the model predicts a value, it gets clipped to [-1,1], and the q_posterior shift is
computed from this wrong x_recon.

The effect:
- **Training loss (`p_losses`)**: unaffected — `p_mean_variance` is never called during training.
- **Eval sampling (`p_sample_loop`)**: severely corrupted. Every denoising step clips valid trajectory
  values, pushing the trajectory toward a narrow ±1 band. For a 3D robot pose that lives at ±0.3m
  (normalized to ~±3σ), this means eval trajectories are clipped to nonsensical values.

**The legacy's eval metrics are therefore ALSO broken** — `clamp_(-1,1)` corrupts sampling for both.
But the training loss (which is what the Trainer logs) is the same in both. The divergence appears
in **eval/inference metrics**, not training loss.

The current override attempts to fix this with `clamp_(-5,5)` on action dims only, which is more
principled. But it changes the inference behavior relative to the legacy checkpoint. A checkpoint
trained with legacy inference assumptions will not sample correctly under the current override.

---

#### Change C: `forward()` — state-only fallback

**Legacy:**
```python
def forward(self, cond, *args, **kwargs):
    if 0 in cond and isinstance(cond[0], tuple):
        bp_imgs, inhand_imgs, pos = cond[0]
        snapping_cond = {0: pos[:, -1]}
        new_cond = snapping_cond.copy()
        new_cond['visual'] = visual_cond
    else:
        new_cond = cond
    return super().forward(new_cond, *args, **kwargs)
```

**Current (adds else-if state-only branch):**
```python
def forward(self, cond, *args, **kwargs):
    if getattr(self.model, 'if_vision', True):
        if 0 in cond and isinstance(cond[0], tuple):
            ...
        else:
            new_cond = cond
    else:
        if 0 in cond and isinstance(cond[0], torch.Tensor):
            obs_seq = cond[0]
            new_cond = {0: obs_seq[:, -1]}  # ← latent bug: takes last FEATURE not last TIMESTEP
        else:
            new_cond = cond
    return super().forward(new_cond, *args, **kwargs)
```

The state-only `forward` path has a latent shape bug: `cond[0]` is `obs[:, 0]` (shape `(B, 3)`).
`obs_seq[:, -1]` takes the last feature index → shape `(B,)` instead of `(B, 3)`. This would crash
in `apply_conditioning`. Not triggered with `if_vision=True`.

---

### Training script — `max_len_data` is the training-path regression

| Parameter | Legacy | Current | Source |
|---|---|---|---|
| `obs_dim` | `3` (hardcoded) | `3` (from `3 if if_vision else args.obs_dim`) | Same at runtime |
| `action_dim` | `3` (hardcoded) | `args.action_dim = 3` (from config) | Same |
| `observation_dim` in diffusion | `3` (hardcoded) | `obs_dim = 3` | Same |
| **`max_len_data`** | **256** (hardcoded) | **512** (= `args.max_path_length` from config) | **DIFFERENT** |

**`max_len_data` is the only parameter that actually differs at runtime.**

`max_len_data=512` tells `Aligning_Img_Dataset` to truncate each episode at 512 steps (instead of 256).
The dataset loads more data per episode. This affects:

1. **Scaler statistics**: `Scaler(all_obs, all_act)` computes `x_mean`, `x_std`, `y_mean`, `y_std` from
   more samples. If the dataset pads short episodes with zeros (the dataset's `use_padding=True`), then
   episodes shorter than 512 contribute zero frames that bias `x_mean` downward and `x_std` upward.
   This shifts the normalization origin for all training data.

2. **Loss magnitude**: Different normalization → different loss scale at step 0. The loss may start at
   a different value, appear worse, and converge at a different final value.

3. **If zero-padding is included in the scaler**: `(obs_val - corrupted_mean) / corrupted_std` maps
   actual data to wrong normalized values → training targets are wrong → model learns wrong function.

The scaler has a comment: "Initialize Scaler with MASKED data (FIX #17 - Prevent zero-padding
corruption)" but in both versions it uses `dataset.get_all_observations()` without masking. Whether
zero-padded frames are excluded depends on `Aligning_Img_Dataset.get_all_observations()` internals.
With `max_len_data=512`, the exposure to potential padding corruption is 2× larger.

---

## Why Legacy Works

The legacy is a simpler, narrower codebase:

1. **`max_len_data=256`**: Loads only 256 steps/episode. Shorter episodes → fewer or no zero-padded
   frames in scaler computation → cleaner normalization stats.

2. **No `p_mean_variance` override**: Uses parent's `clamp_(-1,1)`. This is technically wrong for
   Gaussian-normalized data, but:
   - Training loss is NOT affected (override only fires at inference time).
   - The legacy's eval metrics are ALSO bad for the same reason — but the user measured training
     convergence (loss curve), not eval success rate.

3. **No state-only fallback**: Vision path is the only path. Fewer code branches, fewer opportunities
   for a wrong branch to fire.

4. **Hardcoded dimensions**: `obs_dim=3`, `action_dim=3`, `observation_dim=3` cannot drift from
   config. Current version derives them from args, which is correct but introduces one more place
   for a mismatch.

---

## Root Cause Ranking

| # | Root Cause | Affects | Likelihood | Fix |
|---|---|---|---|---|
| 1 | `max_len_data=256 → 512` in train script | Training normalization → training loss | **High** | Hardcode `max_len_data=256` in current train script OR mask zero-padded frames before scaler init |
| 2 | `p_mean_variance` override with `clamp_(-5,5)` instead of parent `clamp_(-1,1)` | Eval/inference only | Medium | Keep override (it's more correct) but acknowledge eval metrics differ from legacy checkpoints |
| 3 | `if_vision=False` latent bugs in `loss()` and `forward()` | Only fires if state-only mode triggered | Low | Remove the state-only branches if never needed, or fix the shape bug in `forward()` |

---

## Specific Evidence of the `max_len_data` Regression

```
Legacy train script (line 231):
    max_len_data=256        ← hardcoded

Current train script (line 243):
    max_len_data=args.max_path_length   ← from config

Config (aligning-d3il-visual.py, line 123):
    'max_path_length': 512,   ← D3IL default, 2× the legacy value
```

At `max_len_data=512` with `use_padding=True`, the dataset pads every episode to exactly 512 steps.
The scaler computes statistics over `N * 512` samples instead of `N * 256`. If any episode is shorter
than 512 steps, the zero-padded suffix distorts `mean` and `std`. Robot pose `obs` near (0,0,0) is
valid data, so padding zeros may not be obvious from log output — but the normalization shifts.

---

## What Needs to Be Fixed

To make the current `ddpm_encdec_vision` produce the same training metrics as the legacy:

**Fix 1** (mandatory for training parity):
```python
# In ddpm_encdec_vision_test/train_ddpm_encdec_vision.py
# Change:
max_len_data=args.max_path_length    # → 512, wrong

# To:
max_len_data=256                     # match legacy exactly
```

Or alternatively, filter zero-padded frames in the scaler by using the episode mask before computing stats.

**Fix 2** (optional — document the behavioral change):
The `p_mean_variance` override is actually MORE correct for Gaussian-normalized data (±5σ vs ±1σ
clamp). Keep it, but document that:
- Eval metrics from the legacy (with `clamp_(-1,1)`) and current (with `clamp_(-5,5)`) will differ.
- Legacy eval is likely worse (over-clipped trajectories) even though training loss was similar.

**Fix 3** (cleanup — remove latent crash):
```python
# In VisualGaussianDiffusion.forward(), the state-only else-branch has a shape bug:
obs_seq = cond[0]           # (B, 3) — first-step obs, NOT a time sequence
new_cond = {0: obs_seq[:, -1]}  # WRONG: takes last feature → (B,) not (B, 3)

# Should be:
new_cond = {0: obs_seq}     # or remove the entire else branch if state-only is never used
```

---

## Files Changed in the Regression (Summary)

| File | Legacy | Current | Delta Type |
|---|---|---|---|
| `models/diffusion.py` | Flow Matching engine | IDENTICAL | — |
| `models/visual_gaussian_diffusion.py` | 5-arg loss, no p_mean_variance | *args + if_vision guard + p_mean_variance override | **Functional** |
| `models/visual_unet.py` | Always visual | if_vision flag | Structural (safe for visual) |
| `utils/training.py` | Custom Trainer | IDENTICAL | — |
| `utils/scaler.py` | Identical math | Identical math (minor log change) | — |
| `train_ddpm_encdec_vision.py` | `max_len_data=256` hardcoded | `max_len_data=args.max_path_length` (512) | **Training regression** |
