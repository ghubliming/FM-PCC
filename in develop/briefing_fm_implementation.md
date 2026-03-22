# Flow Matcher Implementation Briefing

## Goal
Implement a working Flow Matching engine in the duplicated package while preserving the original public interface expected by the rest of the project.

## Files Updated
- flow_matcher/models/diffusion.py
- flow_matcher/models/helpers.py

## Core Changes in flow_matcher/models/diffusion.py

### 1. Flow Matching Training Path Implemented
- Replaced diffusion noising objective with linear interpolation path training.
- Uses:
  - Base sample: `x_base ~ N(0, I)`
  - Interpolation: `x_t = (1 - t) * x_base + t * x_start`
  - Target velocity: `v_target = x_start - x_base`
- Model now predicts velocity field and is trained with existing weighted loss objects.

### 2. Deterministic Sampling Implemented
- Replaced stochastic DDPM update with deterministic reverse-time Euler step:
  - `x <- x - v_theta(x, t, cond) * dt`
- `p_sample_loop` now performs reverse-time ODE integration while preserving conditioning and projector hooks.

### 3. Interface Compatibility Preserved
- Class name and API remain unchanged (`GaussianDiffusion`, `loss`, `conditional_sample`, `forward`, etc.).
- Legacy-compatible buffers (`betas`, `alphas_cumprod`, posterior buffers) are retained to avoid breakage in callers that reference these attributes.
- `infos` still exposes keys expected by policy code (`diffusion`, `projection_costs`).

### 4. Returns Conditioning Preserved
- Classifier-free guidance style branch retained via `returns_condition` and `condition_guidance_w`, now applied to velocity prediction.

## Change in flow_matcher/models/helpers.py
- `cosine_beta_schedule` is restored as a compatibility utility.
- FM code paths do not depend on it, but keeping it avoids import/runtime breakage in legacy references.

## What This Enables Now
- FM training and FM sampling are implemented in the duplicated package.
- Existing training/policy scaffolding can call the same model APIs without signature changes.

## Notes
- This implementation uses a linear interpolation path and first-order Euler integration.
- Further improvements (higher-order ODE solver, time-dependent path schedules, or specialized FM losses) can be layered on top without changing external interfaces.
