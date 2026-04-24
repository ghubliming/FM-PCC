# CRITICAL WARNING: FMv2 ODE Setup Integrity

**Date:** 2026-04-24
**Topic:** FMv2 ODE Solver Steps vs. Training Defaults

> [!CAUTION]
> **ODE=20 evaluation for FMv2 is INCORRECT.** 
> It has been confirmed by code audit that `flow_matcher_v2` is subject to a **"Pickle Lock"**.

### The Problem
In the FMv2 implementation, the ODE integration step count (`ode_inference_steps_v2`) is loaded directly from the `diffusion_config.pkl` created at training time. 

The evaluation runner (`eval_FM_v2.py`) **ignores** any overrides in `config/avoiding-d3il.py` (e.g., in the `plan_fm_v2` block). 

### Consequences
1.  **Fixed Step Count**: If a model was trained with the default `ode_inference_steps_v2 = 10`, it will **ALWAYS** run with 10 steps during evaluation, even if the user sets it to 20 in the config.
2.  **False Metadata**: Any results previously recorded as "FMv2 with ODE=20" (without a fresh training run explicitly set to 20) are **actually results for ODE=10**.
3.  **Invalid Comparison**: Previous comparisons between ODE=10 and ODE=20 in FMv2 are likely identical because the code executed the exact same integration path for both.

### Summary of "Wrong" Evaluations
Before today (24 April 2026), all evaluations of FMv2 that attempted to change the ODE step count without re-training the model are **WRONG**. They were silently falling back to the training default (usually 10).

### Future Action
- **Do not use FMv2** for solver-step sensitivity tests.
- Transition to **FMv3-Selectable**, which has the "Dynamic Override" fix and correctly honors runtime configuration.
- If FMv2 must be used, the `eval_FM_v2.py` script must be manually patched to synchronize the model's step count with the current `args`.
