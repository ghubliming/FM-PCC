# Gen9 E2 U2 — FM vs Diffusion "Exploded Lines" Analysis

**Date:** 2026-06-08  
**Question:** Are DDPM's exploded trajectory lines caused by a code bug or is Diffusion just less powerful than FM?  
**Verdict: Code bug (config error) — `clip_denoised=False` in DDPM eval config.**

---

## 1. Direction Inversion — NOT the cause

The user hypothesis was that the DDPM denoising direction might be inverted (0→1 instead of 1→0).

**Checked — this is correct in both models:**

| Model | Loop direction | Code |
|-------|---------------|------|
| DDPM | T→0 (denoising, correct) | `for i in reversed(range(0, n_timesteps))` |
| FM | 0→1 (flow forward, correct) | `for i in range(total_steps)` with `t_cont = i / steps` |

No direction inversion exists in either codebase.

---

## 2. Root Cause: `clip_denoised=False` disables the only stability guard

**File:** `config/avoiding-d3il-visual.py`, entry `plan_visual_avoiding_dpcc`  
**Bug:** `'clip_denoised': False`  

In `VisualGaussianDiffusion.p_mean_variance` (`diffuser_visual_avoiding/models/visual_gaussian_diffusion.py:65-66`):
```python
if self.clip_denoised:
    x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)
```

With `clip_denoised=False`, `x_recon` (the model's predicted clean trajectory at each denoising step) is **never clamped**. The base `GaussianDiffusion.p_mean_variance` would raise `RuntimeError` for this case, but `VisualGaussianDiffusion` overrides that method and silently skips clamping.

---

## 3. Why DDPM diverges but FM does not

DDPM inference (`p_sample`):
```python
return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
```
- Adds stochastic Gaussian noise at every one of the 100 denoising steps
- Unclamped extreme `x_recon` → extreme `q_posterior` mean → extreme `model_mean`
- Noise injection amplifies these extreme values over 100 steps → **divergence**

FM inference (`p_sample`):
```python
return model_mean   # = x + velocity * dt
```
- Deterministic ODE step — NO noise added at any step
- Even without clamping, velocity predictions are bounded functions of the current state
- Errors don't compound → **stable trajectories**

The clamping is necessary for DDPM to stay within the training data manifold; it is a no-op for FM.

---

## 4. Code/Config differences between FM and DDPM

| Property | DDPM | FM | Comment |
|---|---|---|---|
| `clip_denoised` | `False` (BUG → `True`) | `False` | FM doesn't need it |
| `p_sample` noise | `N(0, posterior_var)` injected each step | deterministic | core difference |
| Time representation | int 0-99 | float 0.0-1.0 | each model trained with its own scale |
| `action_weight` | 10 | 1 | training weight, not eval |
| `plan_batch_size` | 4 (hardcoded VisualAgent default) | 4 (hardcoded) | same |
| `mpc_batch_size` in config | 1 | 4 | irrelevant — not passed to VisualAgent |

---

## 5. Fix Applied

**File:** `config/avoiding-d3il-visual.py`, `plan_visual_avoiding_dpcc`  
**Change:** `'clip_denoised': False` → `'clip_denoised': True`

This activates `x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)` (action dims only, `action_dim=2` for avoiding) during every DDPM denoising step, preventing the chain from diverging.

Note: the training config `visual_avoiding_dpcc` does NOT need changing — `clip_denoised` is unused during training (`p_losses` never reads it).

---

## 6. Expected Outcome After Fix

- DDPM trajectory lines should be bounded and coherent
- May still not match FM quality if the DDPM model has a harder optimization landscape, but the visual explosion should be eliminated
- If lines are still noisy, secondary factor is the stochastic noise injection (inherent to DDPM) — consider reducing posterior variance scale or switching to DDIM sampling

---

## 7. Architecture Code Diff (no bugs)

Both FM and DDPM share:
- Identical `VisualUNet` and `UNet1D` architecture
- Identical `VisualAgent.predict()` inference wrapper
- Identical `forward()` → `conditional_sample()` → `p_sample_loop()` call chain
- Identical `apply_conditioning()` logic (clears `cond['visual']` key in iteration, re-applies obs anchor)
- Identical `SinusoidalPosEmb` time embedding (each trained with its own scale: DDPM int 0-99, FM float 0-1)

The only architectural difference is `p_sample` — DDPM adds noise, FM doesn't.
