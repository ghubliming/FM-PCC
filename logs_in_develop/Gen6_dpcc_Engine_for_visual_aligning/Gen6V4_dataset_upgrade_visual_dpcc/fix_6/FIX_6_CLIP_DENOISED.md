# Fix 6 — clip_denoised=True → False: Root Cause of Total Eval Failure (2026-05-19)

## Context

After Fix 5 (wandb crash + DIAG file), the first real visual eval was diagnosed via
DIAG output from the K16/steps256 checkpoint (well-converged: a0 loss 0.107→0.007,
total 0.73→0.05 over 4 epochs × ~11k-step dataset). All rollouts failed with the
robot oscillating at max velocity.

---

## Root Cause

**`clip_denoised=True` in `train_visual_aligning_dpcc.py` (line 213)**

Original DPCC (`/workspaces/dpcc/config/avoiding-d3il.py`) uses:
```python
'clip_denoised': False
```

Our visual train script hardcoded:
```python
clip_denoised=True,   # ← WRONG
```

`VisualGaussianDiffusion.p_mean_variance` contains:
```python
if self.clip_denoised:
    x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)
```

### Why this catastrophically corrupts inference

With cosine schedule at K=16, the noise amplification factor at the first denoising
step (t=T-1=15) is:

```
sqrt_recip_alphas_cumprod[15] ≈ 9.4
```

Inference starts from `x = 0.5 * torch.randn(shape)` (original DPCC behaviour).
At t=15:

```
x_recon = sqrt_recip_alphas_cumprod[15] * x_t
        − sqrt_recipm1_alphas_cumprod[15] * ε_θ(x_t, t)
```

Both terms have std ≈ 9.4 × 0.5 = 4.7, combined std ≈ **10.5**.

The ±5 clamp fires on virtually every element of x_recon at this step.
This corrupts `model_mean` → x_{t-1} is out-of-distribution → the model
receives inputs it was NEVER trained on → the denoising chain never recovers.
The final trajectory is pinned entirely at ±5 (normalizer boundary).

DIAG evidence:
```
[ DIAG first-replan ] horizon act (normalized) range: [-5.0000, 5.0000]
```
Every step of the H=8 horizon hit the clamp simultaneously — characteristic
of this failure mode.

### Why training was unaffected

`clip_denoised` is only read inside `p_mean_variance` (inference path).
Training uses `p_losses`, which computes MSE on the noise prediction directly
without ever calling `p_mean_variance`. The checkpoint is valid.

---

## Fix

### 1. Train script — prevent future checkpoints from inheriting the bug

`diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` line 213:
```python
# Before
clip_denoised=True,

# After
clip_denoised=False,
```

### 2. Eval script — fix existing checkpoints without retraining

The saved `diffusion_config.pkl` stores `clip_denoised=True` for the K16/steps256
checkpoint. Overriding the attribute at eval time bypasses the serialized value:

`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (after `diffusion_model = exp.diffusion`):
```python
diffusion_model.clip_denoised = False
print('[ eval ] clip_denoised forced → False (matches original DPCC)')
```

This is safe because `clip_denoised` is a plain Python attribute on
`VisualGaussianDiffusion`; overriding it at runtime does not affect any
tensors or model weights.

---

## Verification Plan

1. Run eval on existing K16/steps256 checkpoint → DIAG should show
   `range: [<val> < 5.0, ...]` instead of `[-5.0000, 5.0000]`
2. Denormalized a0 magnitude should be << 0.014434 m (not clamped)
3. At least some rollouts should reach the goal (non-zero success rate)

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | line 213: `clip_denoised=True` → `clip_denoised=False` |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | force `diffusion_model.clip_denoised = False` after model load + print confirmation |
