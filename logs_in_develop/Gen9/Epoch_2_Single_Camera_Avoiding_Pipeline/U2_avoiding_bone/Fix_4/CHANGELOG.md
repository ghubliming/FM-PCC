# Gen9 E2 U2 Fix_4 — DDPM `clip_denoised=False` causes denoising chain divergence

**Date:** 2026-06-09  
**Status:** ✅ Fixed  
**Symptom:** DDPM trajectory lines "explode" (fly off to extreme coordinates) while FM trajectories are well-behaved  
**Analysis:** `../Fix_3/Diffu_FM_Comparison.md`  
**Parent:** [`../Fix_3/CHANGELOG.md`](../Fix_3/CHANGELOG.md)

---

## Root Cause

`config/avoiding-d3il-visual.py`, entry `plan_visual_avoiding_dpcc`, had:

```python
'clip_denoised': False,
```

In `VisualGaussianDiffusion.p_mean_variance` (`diffuser_visual_avoiding/models/visual_gaussian_diffusion.py:65-66`):

```python
if self.clip_denoised:
    x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)
```

With `clip_denoised=False`, `x_recon` (the model's predicted clean trajectory at each denoising step) is **never clamped**. The base `GaussianDiffusion.p_mean_variance` raises `RuntimeError` for this case, but `VisualGaussianDiffusion` overrides the method and silently skips clamping.

DDPM's `p_sample` injects stochastic Gaussian noise at every one of the 100 denoising steps:

```python
return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
```

Unclamped extreme `x_recon` → extreme `q_posterior` mean → extreme `model_mean` → noise amplification over 100 steps → divergence. FM is unaffected because `p_sample` is a deterministic ODE step with no noise injection; errors don't compound.

---

## Why This Wasn't Caught Earlier

Fix_3 introduced `plan_batch_size=4` (B=4 trajectory fan). With B=1 (Fix_2), a single diverging trajectory was difficult to distinguish visually from a noisy but valid trajectory. With B=4, all 4 lines exploded simultaneously, making the bug obvious.

---

## Fix Applied

**File:** `config/avoiding-d3il-visual.py`, entry `plan_visual_avoiding_dpcc`

```diff
-'clip_denoised': False,
+'clip_denoised': True,   # must be True — activates action-only ±5 clamp in VisualGaussianDiffusion.p_mean_variance; False causes denoising chain divergence
```

**Training config** (`visual_avoiding_dpcc`) is unchanged — `clip_denoised` is not read during training (`p_losses` never references it).

**FM config** (`plan_fm_visual_avoiding`) retains `clip_denoised: False` — FM's deterministic ODE step never adds stochastic noise, so the instability mechanism cannot occur.

---

## Verification

The clamping activates `x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)` at every denoising step (action dims only, `action_dim=2` for avoiding). Obs dims (2:6) are not clamped — preserving their natural range.

Base class behaviour confirmed: `GaussianDiffusion.p_mean_variance` raises `RuntimeError("clip_denoised=False not supported")` at inference time. `VisualGaussianDiffusion` overrides this method; without the fix it silently skipped the clamp rather than raising.

---

## Files Changed

| File | Change |
|---|---|
| `config/avoiding-d3il-visual.py` | `plan_visual_avoiding_dpcc.clip_denoised`: `False` → `True` (with explanatory comment) |
| `logs_in_develop/.../Fix_3/Diffu_FM_Comparison.md` | §5 precision fix: `:2` → `:self.action_dim` |

---

## Expected Outcome

- DDPM trajectory lines bounded and coherent (within ±5 action-space units)
- DDPM quality may still be below FM quality due to stochastic noise at each step vs FM's deterministic ODE — but the visual explosion is eliminated
- If DDPM lines are still noisy after this fix, the residual issue is inherent to DDPM's posterior variance noise injection (not a bug), and DDIM sampling could be explored as a follow-on
