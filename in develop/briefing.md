# Flow Matcher Refactor Briefing

## Objective
Duplicate the existing diffusion package and prepare a Flow Matching skeleton without breaking the surrounding codebase structure.

## What Was Done
- Created a duplicate package at `flow_matcher` from the original `diffuser` folder.
- Replaced diffusion-specific internals in `flow_matcher/models/diffusion.py` with explicit Flow Matching placeholders.
- Replaced diffusion cosine beta schedule in `flow_matcher/models/helpers.py` with an explicit Flow Matching placeholder.
- Preserved class names, method signatures, and module layout to keep integration points stable.

## Placeholder Pattern Used
Each removed diffusion block was replaced with:

```python
# TODO: Implement Flow Matching logic here
# (Replaces: <brief description of removed diffusion code>)
raise NotImplementedError("Flow Matching not yet implemented")
```

## Scope of Diffusion Logic Removed
In `flow_matcher/models/diffusion.py`, placeholders now stand in for:
- Beta/alpha schedule computations and derived diffusion buffers
- Diffusion posterior computations (`q_posterior`, mean/variance logic)
- Reverse diffusion samplers (`p_sample`, `p_sample_loop`, gradient variants)
- Forward noising path (`q_sample`)
- Diffusion training objective path (`p_losses`, `loss`)

In `flow_matcher/models/helpers.py`, placeholder now stands in for:
- `cosine_beta_schedule` diffusion noise schedule

## Compatibility Notes
- Public class/API shape is intentionally preserved (including `GaussianDiffusion` and existing method signatures).
- Non-diffusion utility/training scaffolding in the duplicated package remains untouched.
- Placeholder paths intentionally fail fast with `NotImplementedError` until Flow Matching logic is implemented.

## Current Status
- Static diagnostics report no syntax/errors in the modified duplicate package.
- Runtime behavior for placeholder paths is intentionally unimplemented by design.

## Next Implementation Step
Implement Flow Matching equivalents for:
1. Time interpolation / path sampling between data and base distribution
2. Vector field prediction objective
3. ODE-based sampling loop (replacing reverse diffusion chain)
4. Conditioning/projection integration within FM sampling and training
