# FM-PCC/DPCC vs SafeFlowMPC: Time-Sampling Strategy (Detailed Notes)

## Purpose
This note explains the corresponding approach in FM-PCC/DPCC relative to SafeFlowMPC's Beta-based time sampling.

The key question is:
- SafeFlowMPC biases training toward harder regions of time `t` with a Beta distribution.
- What is the equivalent behavior in FM-PCC/DPCC?

## Short Answer
FM-PCC does **not** currently use a Beta distribution for training-time `t` sampling.

FM-PCC Flow Matching training samples a discrete timestep index uniformly with `torch.randint(...)`, then maps it to continuous time in `[0, 1]` by normalization. There is only a basic safety clamp to `[0, 1]`, not a hard truncation like `[0.2, 0.8]`.

DPCC (`dpcc-r`, `dpcc-t`, `dpcc-c`) is an **inference-time trajectory selection policy**, not a training-time `t`-distribution strategy.

---

## 1) SafeFlowMPC Behavior (Reference)
In SafeFlowMPC training (`train_imitation_learning_safe.py`), time is sampled as:

```python
alpha = torch.tensor(1.5).to(device)
beta = torch.tensor(1.0).to(device)
beta_dist = torch.distributions.Beta(alpha, beta)

# inside loop
t = beta_dist.sample((trajs.shape[0],))
t = 1 - t
```

### Interpretation
1. Raw sampling from `Beta(1.5, 1.0)` tends to produce larger values.
2. Flipping with `1 - t` yields a distribution equivalent to `Beta(1.0, 1.5)`.
3. `Beta(1.0, 1.5)` is skewed toward smaller `t`.

So SafeFlowMPC intentionally focuses training samples on low-`t` regions (harder/noisier states), where gradients are more informative.

---

## 2) FM-PCC Flow Matching Behavior (Current)
In FM-PCC Flow Matching (`flow_matcher/models/diffusion.py`), training loss uses:

```python
t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
```

Then timestep index is converted to continuous time:

```python
t_float = t.float()
denom = max(self.n_timesteps - 1, 1)
t_cont = (t_float / denom).clamp_(0.0, 1.0)
```

And path interpolation is:

```python
x_t = (1.0 - t_cont) * x_base + t_cont * x_start
```

### Interpretation
1. Discrete index is sampled uniformly from `{0, 1, ..., K-1}`.
2. This produces an approximately uniform grid over continuous `t` in `[0, 1]`.
3. No Beta skewing is applied.
4. No restricted training window like `[0.2, 0.8]` is used.

The same structure is mirrored in `flow_matcher_unet_v2/models/diffusion.py`.

---

## 3) What DPCC Means in FM-PCC
It is easy to confuse DPCC variants with time sampling, but they affect a different stage.

In evaluation scripts (`eval_FM.py`, similar variants), behavior is set as:
- `dpcc-r` -> `trajectory_selection = 'random'`
- `dpcc-t` -> `trajectory_selection = 'temporal_consistency'`
- `dpcc-c` -> `trajectory_selection = 'minimum_projection_cost'`

The policy implementation in `flow_matcher/sampling/policies.py` applies this selection among generated candidate trajectories.

### Important distinction
- Training-time `t` sampling: determines which interpolation times contribute to loss.
- DPCC selection: chooses which sampled trajectory to execute/use at inference.

Therefore, DPCC is not the counterpart to SafeFlowMPC's Beta time-sampler. It is an orthogonal inference policy.

---

## 4) Conceptual Comparison

### SafeFlowMPC
- Time sampling: biased (`Beta`, then flip).
- Goal: emphasize hard training regions with stronger learning signal.
- Effect: higher sample density near small `t`.

### FM-PCC (current)
- Time sampling: uniform discrete indices.
- Goal: equal coverage of timesteps by default.
- Effect: each timestep gets similar sampling probability.

### DPCC in FM-PCC
- Not a loss-time weighting method.
- It is a post-sampling trajectory selection strategy.

---

## 5) Why This Matters for Optimization
If your FM-PCC training currently plateaus because high-`t` samples are too easy (small residual velocity target), then uniform sampling may spend too much compute in low-information regions.

A Beta-biased time sampler in FM-PCC could:
1. increase gradient signal on harder portions of the path,
2. speed convergence for difficult constraints,
3. potentially improve robustness under projection-heavy settings.

But it can also:
1. reduce coverage of later/easier time regions,
2. hurt calibration near endpoint if over-biased,
3. require retuning of loss weights and learning rate schedule.

---

## 6) If You Want FM-PCC to Match SafeFlowMPC's Philosophy
A direct analog would be:
1. replace uniform `torch.randint(...)` in FM-PCC loss with Beta-based continuous sampling,
2. map sampled continuous `t` to model's time input consistently,
3. optionally keep a small floor/ceiling epsilon only for numerical stability (not as hard truncation),
4. compare against baseline with identical seeds and evaluation protocol.

For example, conceptually:
- sample `u ~ Beta(a, b)`,
- set `t = 1 - u` if you want small-`t` emphasis,
- compute `x_t = (1 - t) x_base + t x_start` as usual.

---

## 7) Practical Summary for This Project
- FM-PCC Flow Matching currently uses **uniform timestep index sampling**.
- FM-PCC does **not** currently implement SafeFlowMPC-style Beta time bias.
- DPCC variants in FM-PCC are **inference trajectory selectors**, not training `t` samplers.

So the closest "corresponded approach" in FM-PCC today is:
- uniform time coverage during training,
- plus DPCC policy logic at inference for trajectory choice.

These two mechanisms serve different roles and should not be treated as direct substitutes.

---

## 8) Suggested Experiment Matrix (Optional)
If you want evidence before changing defaults, run:

1. Baseline FM-PCC uniform `t`.
2. Mild Beta bias (for small-`t` emphasis), same all else.
3. Stronger Beta bias.
4. Each of the above with `dpcc-r`, `dpcc-t`, `dpcc-c` at inference.

Track:
- success rate,
- constraint violations,
- projection cost,
- mean planning time,
- seed variance.

This cleanly separates training-time effects (time sampler) from inference-time effects (DPCC selection).
