# ODE/VF Steps Upgrade (v2): Codebase Findings

## Goal of this note
This document records what the two codebases currently do about:
1. VF training-time time handling
2. ODE inference-time step handling
3. whether those are coupled or decoupled

It is intended as the factual baseline before implementing the v2 ODE-step upgrade.

---

## Repo A: FM-PCC (current v2 path)

### Key files checked
1. `flow_matcher_v2/models/diffusion.py`
2. `FM_v2_test/train_FM_v2.py`
3. `config/avoiding-d3il.py`

### Observed behavior

#### 1) Training side (VF time handling)
In `flow_matcher_v2/models/diffusion.py`, training loss currently samples continuous `t` with Beta:
- `t ~ Beta(time_beta_alpha_v2, time_beta_beta_v2)`
- then `t = 1 - t`

Then `p_losses(...)` uses that continuous `t` in:
- interpolation `x_t = (1 - t) * x_base + t * x_start`
- model time input via `_predict_velocity(x_t, cond, t, ...)`

So for v2, VF training now uses continuous-time sampling (not discrete randint).

#### 2) Inference side (ODE step handling)
In `flow_matcher_v2/models/diffusion.py`:
- `p_sample_loop(...)` runs `for i in range(total_steps)` where `total_steps = self.n_timesteps + repeat_last`
- each iteration builds integer timestep: `timesteps = full(..., t, dtype=torch.long)`
- integration step in `p_mean_variance(...)` uses `dt = 1.0 / self.n_timesteps`

So ODE integration count and step size are controlled by `self.n_timesteps`.

#### 3) Config/training wiring
From `FM_v2_test/train_FM_v2.py` and `config/avoiding-d3il.py`:
- diffusion constructor gets `n_timesteps=args.n_diffusion_steps`
- current v2 config (`flow_matching_v2` and `plan_fm_v2`) sets `n_diffusion_steps: 20`

Therefore in current v2 implementation, one knob (`n_diffusion_steps`) still controls ODE iteration count and `dt`.

---

## Repo B: SafeFlowMPC

### Key files checked
1. `safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py`
2. `train_imitation_learning_safe.py`

### Observed behavior

#### 1) Inference ODE steps
In `SafeFlowMPC.py` planning loop:
- `dt = 1.0 / self.config.flow_steps`
- loop: `for flow_step in range(self.config.flow_steps)`
- continuous time query: `t = flow_step / self.config.flow_steps`

Inference integration is controlled by `flow_steps`.

#### 2) Training VF time sampling
In `train_imitation_learning_safe.py`:
- `t ~ Beta(1.5, 1.0)` then `t = 1 - t`
- in one branch, sampled continuous `t` is mapped to precomputed index with `idx_t = (t * 50).int()`

This indicates training-time sampling logic is conceptually independent from runtime planner `flow_steps`.

---

## Coupling Diagnosis (for FM-PCC v2)

### What is coupled today
In FM-PCC v2, `self.n_timesteps` currently determines all of the following in one place:
1. inference loop count (`p_sample_loop`)
2. inference step size (`dt = 1 / self.n_timesteps`)
3. integer time range used during inference calls to the UNet

### Why this matters
If we want:
1. one setting for VF training time-resolution behavior
2. another setting for ODE inference speed/accuracy

then we need explicit decoupling in v2 diffusion sampler logic.

---

## Main conclusion
1. SafeFlowMPC already uses a planner-side step knob (`flow_steps`) for inference integration.
2. FM-PCC v2 currently keeps ODE loop count and `dt` under `n_diffusion_steps`.
3. Next upgrade should add explicit v2-only decoupling knobs so ODE inference steps can change without forcing the same training-time setting.
