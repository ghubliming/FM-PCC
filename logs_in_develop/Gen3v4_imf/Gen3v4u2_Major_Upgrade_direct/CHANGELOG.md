# Gen3v4u1 — iMeanFlow Real Implementation Upgrade

**Date**: 2026-05-21
**Branch**: update_into_FM
**Source audit**: `logs_in_develop/Gen3v4_imf/Audit_Fix6/AUDIT_REPORT.md`
**Scope**: Implement real iMF dual-velocity decomposition with h-conditioning. Fix all confirmed code bugs except BUG-01 (torchdiffeq) and BUG-04 (projection costs).

All 13 confirmed and actionable findings are addressed below.

| File | Fixes applied |
|------|--------------|
| `flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py` | MATH-05 (h-MLP) |
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | MATH-02, MATH-03/04 (fix samplers), MATH-05 |
| `flow_matcher_v3_imeanflow/models/imf_engine.py` | MATH-03/04 (fix standalone sampler), MATH-05 |
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | MATH-01, MATH-05, MATH-06, MATH-07, BUG-05, BUG-08 |
| `config/avoiding-d3il.py` | BUG-02, BUG-03 |

---

## MATH-05 — h-Conditioning Added to UNet (`unet1d_temporal_cond.py`)

**Problem**: The model never received the step-size `h = t - r` that distinguishes iMF from standard FM. Without `h`-conditioning, one-step generation is impossible — the model has no way to adapt its prediction to the integration interval.

**Fix**:
- Added `self.h_mlp = nn.Sequential(SinusoidalPosEmb(dim), Linear, Mish, Linear)` in `Flow_matcher_U_Net_v2.__init__`, mirroring the `time_mlp` architecture.
- Added `h=None` parameter to `forward()`. When `h` is provided, it is broadcast to `[batch]`, embedded by `h_mlp`, and **fused additively** into the time embedding: `t = t + h_mlp(h)`. This allows the model to jointly condition on both the current time `t` and the step size `h`.
- Handles scalar, 0-dim tensor, and `[batch]` tensor inputs for `h` (same pattern as the existing `timesteps` handling).

---

## MATH-05 (propagation) — h Threaded Through Engine and Diffusion

**Files**: `imf_trajectory_model.py`, `imf_engine.py`, `imf_diffusion.py`

**Problem**: Even after adding h-conditioning to the UNet, `h` was accepted by `iMeanFlowEngine.u_fn` but immediately dropped — never forwarded to the model.

**Fix**:
- `iMFTrajectoryModel.forward(x, t, h=None, cond=None, force_dropout=False)`: passes `h=h` to `velocity_net`
- `iMFTrajectoryModel.forward_train`: same, forwards `h` and `force_dropout`
- `iMeanFlowEngine.u_fn`, `.forward`, `.forward_train`: all now forward `h`
- `iMFDiffusion._predict_uv(x, cond, t, h=None, ...)`: passes `h` to `model.forward_train`
- `iMFDiffusion._predict_velocity`: passes `h` to `_predict_uv`
- `iMFDiffusion.p_sample_loop`: computes `h_batch = full(batch, dt)` and passes to `_predict_velocity` at each step

---

## MATH-01 — Aux Branch Trained Against Zero Target Fixed (`imf_diffusion.py`)

**Problem**: `aux_loss = F.mse_loss(aux_pred, torch.zeros_like(aux_pred))` trained the `v`-branch to always output zero, making it dead code. Combined with zero-initialized weights, the `v`-head was permanently suppressed.

**Fix**: Changed to `aux_loss = F.mse_loss(aux_pred, v_target)` where `v_target = x_start - x_base` — the standard FM instantaneous velocity target. The aux branch now learns a meaningful signal (the full FM velocity), consistent with the iMF paper's `v = dx_t/dt` decomposition.

---

## MATH-02 — Aux Head Made Independent (`imf_trajectory_model.py`)

**Problem**: `aux = self.aux_head(velocity)` made the `v`-branch a deterministic function of `u`, creating a serial dependency that violates iMF's design (parallel `u` and `v` heads on shared features).

**Fix**: Changed to `aux = self.aux_head(x)` where `x` is the input noisy trajectory `x_t`. The aux MLP now independently processes the input, allowing it to predict the instantaneous velocity from the same observation the main head sees — architecturally consistent with the parallel-head iMF design.

---

## MATH-03/04 — Standalone Sampler Direction and Sigma Fixed

**Files**: `imf_engine.py`, `imf_trajectory_model.py`

**Problem**: `iMeanFlowEngine.sample()` used `t_steps = linspace(1.0, 0.0, ...)` with `z -= h * velocity` (1→0 direction), while training uses DATA-AT-1 convention (t=0 is noise, t=1 is data, sampling integrates 0→1). The standalone sampler would produce garbage if called. Both files also used `sigma=1.0` while `p_sample_loop` used `sigma=0.5`.

**Fix**: Both standalone samplers now use 0→1 direction:
- `t_steps = linspace(0.0, 1.0, num_steps+1)` (forward)
- `h = t_next - t_cur > 0` (positive step size)
- `z = z + h * velocity` (forward Euler)
- `z = torch.randn(...)` sigma=1.0 (matching q_sample at t=0)
- Step size `h` is passed to the model at each step

---

## MATH-04 — p_sample_loop Noise Sigma Corrected (`imf_diffusion.py`)

**Problem**: `x = 0.5 * torch.randn(shape)` in `p_sample_loop`. Training samples noise with `torch.randn_like(x_start)` (sigma=1.0). The initial noise distribution at inference (sigma=0.5) mismatched what the model trained on.

**Fix**: Changed to `x = torch.randn(shape, device=device)` (sigma=1.0).

---

## MATH-07 — `u_mix` Weight Applied to Main Loss (`imf_diffusion.py`)

**Problem**: `total_loss = main_loss + self.aux_loss_weight * aux_loss`. The `u_mix` ≈ 0.909 normalization coefficient was computed but never applied to `main_loss`, making it cosmetic.

**Fix**: Changed to `total_loss = self.u_mix * main_loss + self.aux_loss_weight * aux_loss`. The normalized `u_mix` and `v_mix` now actually weight the loss components.

---

## Real iMF Training Objective Implemented (`imf_diffusion.py:p_losses`)

**Problem**: The training objective was identical to standard FMv3ODE (FM velocity target `x_1 - x_0`). No mean-flow computation.

**Fix**: `p_losses` now implements the iMF mean-flow objective:

1. Sample noise `x_base ~ N(0, I)` (sigma=1.0)
2. For each sample, draw `r ~ Uniform(0, t)` → `h = t - r > 0`
3. Compute interpolants: `x_t = (1-t)*x_base + t*x_start`, `x_r = (1-r)*x_base + r*x_start`
4. **Mean flow target**: `u_target = (x_start - x_r) / (h + 1e-8)` — the expected velocity that carries `x_r` toward `x_start` averaged over interval `h`
5. **FM velocity target**: `v_target = x_start - x_base` — standard FM target for the aux branch
6. Query model with h: `(u_pred, v_pred) = model(x_t, t, h=h, cond=cond)`
7. Loss: `u_mix * MSE(u_pred, u_target) + aux_weight * MSE(v_pred, v_target)`

---

## MATH-06 — CFG Infrastructure Added (`imf_diffusion.py`)

**Problem**: Classifier-free guidance was completely stripped — `_predict_uv` dropped `returns` and `force_dropout`.

**Fix**: `_predict_velocity` now performs the CFG double-pass when `self.returns_condition=True` and `condition_guidance_w > 0`:
```python
uncond_vel, _ = self._predict_uv(x, cond, t, h=h, returns=returns, force_dropout=True)
velocity = (1 + w) * velocity - w * uncond_vel
```

`force_dropout` is threaded through the full call chain (diffusion → engine → trajectory model → UNet). The UNet backbone currently has `returns_condition=False`, so CFG has no effect in this upgrade — but the infrastructure is in place for a future upgrade that enables returns conditioning.

---

## BUG-05 — `returns_condition` Returns Actually Forwarded (`imf_diffusion.py`)

**Problem**: `_predict_uv` accepted `returns` but immediately discarded it, never forwarding to the model. `self.returns_condition=True` was stored but unused.

**Fix**: `_predict_uv` now passes `returns` and `force_dropout` to `model.forward_train`. The UNet receives the returns value (and ignores it since `returns_condition=False`), but the path is complete. BUG-05 is resolved as a precondition for the MATH-06 CFG infrastructure above.

---

## BUG-08 — `sample()` No Longer Mutates `flow_steps_v3` (`imf_diffusion.py`)

**Problem**: `iMFDiffusion.sample(num_steps=N)` permanently changed `self.flow_steps_v3`, affecting all subsequent calls.

**Fix**: Removed the assignment `self.flow_steps_v3 = int(num_steps)`. Added `num_steps` parameter to `p_sample_loop` and `conditional_sample` so the step count flows through the call chain without modifying object state. The `sample()` method now computes a local `flow_steps` variable from `num_steps if num_steps is not None else self.flow_steps_v3`.

---

## BUG-02 — `loss_discount=1.0` Added to Config (`config/avoiding-d3il.py`)

**Problem**: The iMF config block was missing `loss_discount`. The train script defaulted to `args.discount = 0.99`, causing exponential decay of trajectory loss weights (0.99^7 ≈ 0.93 at the last step) with no justification.

**Fix**: Added `'loss_discount': 1.0` to the `flow_matching_v3_imeanflow` config block.

---

## BUG-03 — `gradient_accumulate_every=2` Added to Config (`config/avoiding-d3il.py`)

**Problem**: The iMF config block was missing `gradient_accumulate_every`. The train script defaulted to `1` while the original FMv3ODE config uses `2`, making the effective batch size 2x larger in FMv3ODE.

**Fix**: Added `'gradient_accumulate_every': 2` to the `flow_matching_v3_imeanflow` config block.

---

## Not Fixed in This Upgrade

### BUG-01 — torchdiffeq Backend Still Silently Ignored

The `ode_solver_backend_v3='torchdiffeq'` config option is still silently ignored; `p_sample_loop` always runs legacy Euler. Porting the torchdiffeq block from `flow_matcher_v3_ode_selectable/models/diffusion.py` requires careful adaptation of the velocity function signature (which now accepts `h`) to torchdiffeq's `ode_rhs(t_scalar, state)` interface. Deferred to Gen3v4u2.

### BUG-04 — Projection Costs Still Not Fully Tracked

`p_sample_loop` now accumulates a `costs` dict and returns `infos['projection_costs'] = costs` (non-empty when a projector is active). However, cost accumulation depends on the projector implementing a `compute_cost()` method for gradient-mode projection. Non-gradient projectors' costs are captured via the second return value of `projector.project()`. The partial implementation is better than always returning `{}`, but projectors that don't follow this interface will still miss cost entries.

### DEV-03 — `iMFTrainingLoss` Dead Code Not Removed

`flow_matcher_v3_imeanflow/models/imf_losses.py` remains in place and is still exported from `__init__.py`. Removing it risks breaking any external import that references `iMFTrainingLoss`. Left for a dedicated cleanup pass.
