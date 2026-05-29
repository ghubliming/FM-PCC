# iMeanFlow Architecture, Math, and Comparison to Standard FM

**Scope**: Gen3v4 iMF — `flow_matcher_v3_imeanflow/` module  
**Date**: 2026-05-28  
**Branch**: `update_into_FM`  
**Changelog**: [`CHANGELOG.md`](CHANGELOG.md)  
**Projection doc**: [`PCC_PROJECTION_IN_IMF.md`](PCC_PROJECTION_IN_IMF.md)

---

## 1. What iMeanFlow Is and Why It Exists

**iMeanFlow (iMF)** is an improved variant of Flow Matching that achieves high-quality
generation with significantly fewer inference steps. The core idea: instead of learning
the **instantaneous** velocity at a single time `t`, the model learns the **mean flow**
— the average velocity needed to carry a noisy sample at time `r` all the way to the
data `x_0` over an interval `h = t - r`.

In the limit `h → 0`, the mean flow reduces to the standard FM instantaneous velocity.
For large `h` (e.g. a single step from `r=0` to `t=1`), the mean flow enables
**one-shot generation** — sample a trajectory in a single model call instead of 10–100
Euler steps.

This is directly relevant to FM-PCC: MPC replanning happens every 4 environment steps.
Reducing the number of ODE solver calls per replan from 100 to 1–10 without quality
degradation is a direct inference speedup for the planning loop.

---

## 2. Task Context

Gen3v4 iMF operates on the **D3IL avoiding task** — the robot must avoid obstacles
while reaching a goal region. The config lives in `config/avoiding-d3il.py`.

This is distinct from the Gen7 visual aligning pipeline (`config/aligning-d3il-visual.py`).
The avoiding task uses:
- **State-only observations** (no camera input in the base Gen3v4 config)
- **Trajectory dimension**: `action_dim + obs_dim` (from `avoiding-d3il.py` base config)
- **Horizon `H=8`**, same as all other Gen3 variants

---

## 3. Class Hierarchy

```
iMeanFlowODE                    ← FM-PCC / FMv3ODE-compatible outer wrapper
  └─ iMeanFlowEngine             ← iMF inference/training engine
       └─ iMFTrajectoryModel     ← velocity backbone
            ├─ Flow_matcher_U_Net_v2   ← UNet1D with h-conditioning (u head)
            └─ aux_head (MLP)          ← independent v head
```

Each layer preserves the FM-PCC diffusion API (`.loss()`, `.p_sample_loop()`,
`.conditional_sample()`, `.forward()`) so the iMF model is a drop-in replacement
for any standard FM or DDPM diffusion object in the training and eval scripts.

---

## 4. The Two Velocity Branches

iMF decomposes the velocity field into two parallel predictions:

| Branch | Symbol | Target | Head | Role |
|---|---|---|---|---|
| **Main (u)** | `u_pred` | Mean flow `(x_data - x_r) / h` | `velocity_net` (UNet1D) | Primary generation signal |
| **Aux (v)** | `v_pred` | FM velocity `x_data - x_base` | `aux_head` (MLP, zero-init) | Instantaneous anchor, prevents drift |

The aux head is a 2-layer MLP (`Linear → SiLU → Linear`) initialized to output zero
(zero-initialized last layer). This ensures at the start of training the model behaves
as standard FM and gradually learns the auxiliary residual.

**At inference**, the combined velocity used for each Euler step is:

```python
velocity = _predict_velocity(x, cond, t, h=h_batch)
         = u_pred + sample_aux_weight * v_pred    # sample_aux_weight = 0.1 * v_mix
```

The aux contribution is intentionally kept small (`sample_aux_weight ≈ 0.009`) so it
provides a soft correction without dominating. The primary integration signal is `u`.

---

## 5. h-Conditioning

The step-size `h` is the critical iMF-specific input that does not exist in standard FM.
It is threaded through the entire call chain and fused into the time embedding.

### In the UNet backbone (`Flow_matcher_U_Net_v2`)

```python
self.h_mlp = nn.Sequential(
    SinusoidalPosEmb(dim),   # same architecture as time_mlp
    nn.Linear(...),
    nn.Mish(),
    nn.Linear(...),
)

def forward(self, x, cond, timesteps, h=None, ...):
    t = self.time_mlp(timesteps)    # sinusoidal embedding of current time t
    if h is not None:
        h = h * torch.ones(x.shape[0], ...)   # broadcast to batch
        t = t + self.h_mlp(h)                 # additive fusion: t_embed += h_embed
```

The `h_mlp` mirrors the `time_mlp` architecture exactly. Additive fusion means the
model jointly attends to both the current position `t` on the ODE path and the size `h`
of the interval it needs to "jump over". Without h-conditioning, the model cannot
distinguish a 1-step full integration from a single step in a 10-step chain — one-step
generation would be impossible.

### Training: `h = t - r` (random sub-interval)

```python
t = 1.0 - Beta(α=1.5, β=1.0).sample()   # biased toward t ≈ 1 (near data)
r = t * Uniform(0, 1).sample()            # r ∈ [0, t]
h = t - r                                  # step size: positive, ∈ (0, t]
```

Each training sample sees a **random interval** `[r, t]` within `[0, 1]`. The model
must learn to predict the mean flow for all possible sub-intervals, which implicitly
trains it to generalize to any number of inference steps.

### Inference: `h = 1 / flow_steps_v3` (fixed step size)

```python
dt = 1.0 / flow_steps_v3    # e.g. dt = 0.1 for flow_steps_v3 = 10
h_batch = full((batch_size,), dt)   # same h for all samples in a step

for loop_idx in range(flow_steps_v3):
    t_cont = loop_idx / flow_steps_v3   # t ∈ {0, 0.1, 0.2, ..., 0.9}
    velocity = _predict_velocity(x, cond, t_cont, h=h_batch)
    x = x + velocity * dt               # forward Euler
```

---

## 6. Training Objective

### 6.1 DATA-AT-1 Convention

iMF uses DATA-AT-1 (t=0 is noise, t=1 is data) — the same convention as the visual
aligning FM (FMv3ODE):

```
t=0 → x_t = x_base (pure noise)
t=1 → x_t = x_start (demonstration trajectory)
x_t = (1-t) · x_base + t · x_start     (linear OT interpolant)
```

**Integration direction**: 0 → 1 (noise → data). Each Euler step: `x += velocity * dt`.

### 6.2 Full `p_losses` Computation

```python
def p_losses(x_start, cond, t):
    # 1. Sample noise (sigma=1.0 — matches q_sample convention)
    x_base = torch.randn_like(x_start)

    # 2. Draw random sub-interval endpoint r ∈ [0, t]
    r = t * torch.rand_like(t)
    h = t - r                                              # h > 0

    # 3. Compute interpolants at t and r
    x_t = (1 - t) * x_base + t * x_start                  # noisy sample at t
    x_r = (1 - r) * x_base + r * x_start                  # noisy sample at r

    # 4. Mean flow target: average velocity from x_r to x_start over interval h
    u_target = (x_start - x_r) / (h + 1e-8)

    # 5. FM instantaneous velocity target for aux branch
    v_target = x_start - x_base                            # standard FM target

    # 6. Dual prediction
    u_pred, v_pred = model(x_t, t, h=h, cond=cond)

    # 7. Weighted combined loss
    main_loss = loss_fn(u_pred, u_target)                  # weighted MSE per step
    aux_loss  = F.mse_loss(v_pred, v_target)
    total_loss = u_mix * main_loss + aux_loss_weight * aux_loss
```

### 6.3 Why the Mean Flow Target Works

The key insight: `u_target = (x_start - x_r) / h` is the **chord** (secant) velocity
that travels from `x_r` directly to `x_start` in one integration step of size `h`.

Standard FM target `v_target = x_start - x_base` is the **full chord** from noise to
data — the same for all `t`. The model learns one velocity for the entire path.

iMF target `u_target` is the **partial chord** from `x_r` to `x_start`, parameterized
by the current position `r` and interval `h`. By conditioning on `h`, the model learns
to predict the correct partial chord at any point on the path, enabling fewer ODE steps:

```
Standard FM (10 steps):   x_0 →[FM]→ x_0.1 →[FM]→ ... →[FM]→ x_1 (data)
iMF (1 step):             x_0 →[iMF, h=1.0]→ x_1               (direct)
iMF (2 steps):            x_0 →[iMF, h=0.5]→ x_0.5 →[iMF, h=0.5]→ x_1
```

The mean-flow prediction is essentially a **learned integration shortcut** — the model
estimates where the ODE will end up without needing many small steps.

### 6.4 Loss Weights

```
u_loss_weight: 1.0    → u_mix = 1.0 / 1.1 ≈ 0.909
v_loss_weight: 0.1    → v_mix = 0.1 / 1.1 ≈ 0.091
aux_loss_weight = max(0.01, 0.1 * v_loss_weight) = 0.01

total_loss = 0.909 * main_loss + 0.01 * aux_loss
```

The aux head weight is intentionally very small — it must not destabilize the main
mean-flow learning signal.

---

## 7. Inference Sampling

### 7.1 `p_sample_loop` (primary inference path)

```python
@torch.no_grad()
def p_sample_loop(shape, cond, projector=None, num_steps=None):
    x = torch.randn(shape)           # sigma=1.0, matches q_sample training noise
    x = apply_conditioning(x, cond, action_dim)   # anchor obs dims to known values

    dt = 1.0 / flow_steps
    h_batch = full((batch_size,), dt)   # constant step size per step

    for loop_idx in range(flow_steps):
        t_cont = loop_idx / flow_steps   # t ∈ [0, 1)
        velocity = _predict_velocity(x, cond, t_cont, h=h_batch)
        x = x + velocity * dt           # forward Euler 0→1
        x = apply_conditioning(x, cond, action_dim)

        # PCC projection (see Section 8)
        if projector and near_end(loop_idx):
            x = project(x)
            x = apply_conditioning(x, cond, action_dim)
    
    return x, infos
```

### 7.2 Combined Velocity at Inference

```python
def _predict_velocity(x, cond, t, h=None):
    velocity, aux = _predict_uv(x, cond, t, h=h)    # dual prediction

    # Classifier-free guidance (when returns_condition=True)
    if returns_condition and condition_guidance_w > 0:
        uncond_vel, _ = _predict_uv(..., force_dropout=True)
        velocity = (1 + w) * velocity - w * uncond_vel

    return velocity + sample_aux_weight * aux         # small aux correction
```

### 7.3 Default Inference Parameters

| Parameter | Value | Notes |
|---|---|---|
| `flow_steps_v3` | 10 | Much fewer than standard FM's 100 (visual aligning) |
| `ode_solver_backend_v3` | `legacy_euler` | torchdiffeq backend deferred (BUG-01, not yet ported) |
| `time_beta_alpha_v3` | 1.5 | Same Beta prior as FMv3ODE — must match training |
| `time_beta_beta_v3` | 1.0 | Same Beta prior as FMv3ODE |
| `h_batch` | `dt = 1/10 = 0.1` | Fixed step size, uniform across all ODE steps |

---

## 8. PCC Projection Integration

Projection follows the threshold convention from `PCC_PROJECTION_IN_IMF.md`.

### Projection window

```python
snapping_start_idx = int((1.0 - threshold) * flow_steps)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)
```

With `threshold=0.5, flow_steps=10`: projects at steps 5, 6, 7, 8, 9 (last half).

**Rationale**: early ODE steps are "noisy" — trajectories are still far from the data
manifold; projecting there would pull against the velocity field. Late steps are near
data, so projection pushes minimally and the SLSQP solution is stable.

### Gradient vs SLSQP

| Mode | Trigger | How |
|---|---|---|
| Gradient | `projector.gradient=True` | `x += projector.compute_gradient(x, constraints)` |
| SLSQP | `projector.gradient=False` | `x, cost = projector.project(x, constraints)` |

After projection, `apply_conditioning` is called again to restore the obs anchor —
projection may disturb the conditioned dims.

### No `diffusion_timestep_threshold` in training path

The `T` threshold is an **inference-only** parameter. It lives in the `Projector`
object (loaded from `projection_eval.yaml`) and is not baked into the model checkpoint.
This means different threshold values use the **same trained model**, unlike DPCC where
threshold differences are built into the denoising schedule.

---

## 9. Comparison to Other Gen3 Variants

| | DPCC (Gen3) | FMv3ODE (Gen3v5) | iMF (Gen3v4) |
|---|---|---|---|
| **Generative model** | DDPM (denoising) | Standard FM (ODE) | Mean-flow FM (ODE) |
| **Training target** | Noise `ε` | Velocity `x_1 - x_0` | Mean flow `(x_data - x_r) / h` |
| **Aux head** | None | None | `v_pred = x_1 - x_0` (MLP, zero-init) |
| **h-conditioning** | No (no interval) | No (no interval) | Yes — step size fused into time embed |
| **Forward process** | Markov Gaussian chain | Linear OT interpolation | Same as FMv3ODE |
| **Time convention** | Discrete `t ∈ {1,…,T}` | DATA-AT-1, continuous | DATA-AT-1, continuous |
| **Integration dir.** | `T → 0` (denoising) | `0 → 1` (noise→data) | `0 → 1` (noise→data) |
| **Inference steps** | 20–100 | 10 (Gen3v4 context) | **1–10** (goal: 1) |
| **Beta prior** | — | `Beta(1.5, 1.0)` | `Beta(1.5, 1.0)` (same) |
| **PCC projection** | Threshold `t ≤ T·thresh` | Threshold `step ≥ (1-t)·K` | Same as FMv3ODE |
| **action_weight** | 10 | 10 | 10 |
| **UNet backbone** | `UNet1DTemporalCondModel` | `Flow_matcher_U_Net_v2` | `Flow_matcher_U_Net_v2` + `h_mlp` |

The iMF backbone (`Flow_matcher_U_Net_v2` + `h_mlp`) is the FMv3ODE backbone plus the
h-conditioning MLP. All other UNet internals (dim, dim_mults, sinusoidal embedding,
down/up blocks, FiLM conditioning) are shared.

---

## 10. Key Fixes from Gen3v4u1 (CHANGELOG Summary)

The Gen3v4u1 upgrade (`CHANGELOG.md`) corrected 13 bugs that made the model effectively
identical to standard FM with dead code. The most structurally important fixes:

| Fix | What was wrong | What was fixed |
|---|---|---|
| **MATH-05** | `h` never reached the UNet — dropped in `u_fn` | `h_mlp` added to UNet; `h` threaded through all layers |
| **MATH-01** | `aux_loss` trained `v` to output **zero** | `v_target = x_data - x_base` (real FM velocity target) |
| **MATH-02** | `aux = aux_head(velocity)` — serial dependency | `aux = aux_head(x)` — independent parallel head |
| **MATH-03/04** | Standalone sampler direction **1→0** (wrong) | Both samplers use forward `0→1` Euler |
| **MATH-04** | Inference noise `sigma=0.5`, training `sigma=1.0` | Both now `sigma=1.0` |
| **MATH-07** | `u_mix` weight computed but **never applied** | `total_loss = u_mix * main_loss + aux_weight * aux_loss` |
| **Core p_losses** | Training target = standard FM `x_1 - x_0` (no mean flow) | Real iMF target `(x_data - x_r) / h` with sub-interval sampling |
| **BUG-08** | `sample(num_steps=N)` mutated `self.flow_steps_v3` | `num_steps` flows via local variable, object state never changed |

Before these fixes, the "iMF" model was functionally a standard FM model with h=0
(h dropped before UNet) and a dead aux branch (always outputting zero). After the fixes
it is the real iMF dual-velocity mean-flow model.

---

## 11. Known Remaining Issues

| Issue | Status | Notes |
|---|---|---|
| **BUG-01** — torchdiffeq backend | Deferred (Gen3v4u2+) | `ode_solver_backend_v3='torchdiffeq'` silently falls back to legacy Euler. The velocity function now accepts `h`; requires adapting `ode_rhs(t_scalar, state)` signature |
| **BUG-04** — projection cost tracking | Partial | `costs` dict populated for SLSQP; projectors without `.compute_cost()` still miss entries |
| **DEV-03** — `iMFTrainingLoss` dead code | Not removed | `imf_losses.py` still exported; cleanup deferred |

---

## 12. Config Reference

### Training block (`flow_matching_v3_imeanflow`)

```python
'model':     'flow_matcher_v3_imeanflow.models.iMeanFlowEngine',
'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
'horizon':   8,

# iMF architecture (official repo defaults)
'freq_dim':       256,    # UNet channel dim (= 'dim' in FMv3ODE UNet)
'depth':          8,      # UNet depth
'num_heads':      4,
'mlp_dim':        256,    # MLP hidden dim
'time_dim':       256,    # sinusoidal embedding dim for both t and h

# Loss weights
'u_loss_weight':  1.0,    # main mean-flow branch weight
'v_loss_weight':  0.1,    # aux FM-velocity branch weight
'action_weight':  10,     # upweights action dims in loss_fn

# Training
'n_train_steps':           100000,
'batch_size':              32,
'gradient_accumulate_every': 2,   # effective batch = 64
'learning_rate':           5e-4,
'ema_decay':               0.995,
'loss_discount':           1.0,   # uniform step weights (no horizon decay)

# Time prior (same as FMv3ODE — must match at eval)
'time_beta_alpha_v3': 1.5,
'time_beta_beta_v3':  1.0,

# Inference steps (training config — inference uses plan config)
'ode_inference_steps_v3': 10,
```

### Plan block (`plan_fm_v3_imeanflow`)

```python
'diffusion':    'flow_matcher_v3_imeanflow.models.iMeanFlowODE',
'flow_steps_v3': 10,            # Euler steps for inference (goal: reduce to 1–5)
'u_loss_weight': 1.0,
'v_loss_weight': 0.1,
'time_beta_alpha_v3': 1.5,     # MUST match training
'time_beta_beta_v3':  1.0,
'ode_solver_backend_v3': 'legacy_euler',
'diffusion_timestep_threshold': _yaml_threshold,  # for PCC projection window + path naming
'batch_size': 4,               # MPC candidates per replan
```

### Checkpoint path

```
logs/avoiding-d3il/flow_matching_v3_imeanflow/
  H{H}_D{diffusion}_a{alpha}_b{beta}_aw{aw}/
    state_{epoch}.pt
```

Plan path:
```
logs/avoiding-d3il/plans/flow_matching_v3_imeanflow/
  H{H}_D{diffusion}_a{alpha}_b{beta}_aw{aw}/
    H{H}_K{K}_Meuler_T{T}_D{diffusion}/
```

---

## 13. Summary: iMF vs Standard FM

The structural differences from standard FM (FMv3ODE):

```
Standard FM:
  Train: target = x_data - x_noise  (constant chord, same for all t)
  Infer: 100 Euler steps (visual aligning), 10 steps (avoiding-d3il)
  Model: UNet(x_t, t, cond) → velocity

iMeanFlow:
  Train: target = (x_data - x_r) / h  (partial chord, function of r and h)
         + aux: x_data - x_noise       (FM anchor for aux head)
  Infer: 1–10 Euler steps (h-conditioned, mean-flow shortcuts)
  Model: UNet(x_t, t, h, cond) → (u_velocity, _)
         MLP(x_t)               → (_, v_aux)
  Combined: u_velocity + 0.01 * v_aux
```

The mean-flow head learns to "skip ahead" on the ODE path by predicting the right
velocity to reach the data in exactly `h` integration time — enabling one-step generation
in the limit without explicit distillation.

---

## 14. ODE Steps Analysis: One-Shot vs Multi-Step in iMF

### 14.1 Core Question

If we set `flow_steps_v3 = 1` (ODE = 1), are we running the "real" iMF one-shot
generation that the paper advertises? And if we set ODE = 2 or ODE = 10, are we doing
something useful — or wasting compute?

### 14.2 Answer: ODE = 1 Is the "Real" iMF One-Shot

**Yes.** Setting `flow_steps_v3 = 1` (equivalently, `num_sampling_steps = 1` in the
official repo) is exactly the canonical iMF one-shot generation. Here is why:

**Official reference implementation** (`imeanflow/imf.py:117-136`):

```python
t_steps = torch.linspace(1.0, 0.0, num_steps + 1)   # e.g. [1.0, 0.0] for num_steps=1

for i in range(num_steps):
    t = t_steps[i]       # t = 1.0
    r = t_steps[i + 1]   # r = 0.0
    h = t - r             # h = 1.0 (full interval)
    u = self.u_fn(z_t, t, h, ω, t_min, t_max, y)[0]
    z_t = z_t - h * u     # one jump from noise to data
```

With `num_steps = 1`: the model sees `h = 1.0` (the entire [0, 1] interval) and predicts
the mean velocity that carries the sample directly from pure noise to data in a single
evaluation. This is what the paper calls **1-NFE** (one Network Function Evaluation).

The paper explicitly benchmarks at 1-NFE as its primary result: **iMF-XL/2 achieves
FID 1.72 with a single function evaluation** on ImageNet 256×256.

**In our Gen3v4 port** (`imf_diffusion.py`), the equivalent path is:

```python
dt = 1.0 / flow_steps  # dt = 1.0 when flow_steps = 1
h_batch = torch.full((batch_size,), dt)  # h = 1.0

for i in range(flow_steps):   # exactly one iteration
    t_cont = 0.0 / 1 = 0.0
    velocity = _predict_velocity(x, cond, t=0.0, h=1.0)
    x = x + velocity * 1.0    # one-shot jump
```

After this single step, if a projector is configured, the PCC QP projection fires
(because `loop_idx == flow_steps - 1` always triggers). So the full pipeline is:

```
noise → [iMF one-shot] → raw trajectory → [QP projection] → constrained trajectory
```

This is the cleanest form of the iMF + PCC combination: one model call for generation,
one optimization call for constraint enforcement.

### 14.3 What About ODE = 2?

With `flow_steps_v3 = 2`, the sampler does:

```
Step 0: t=0.0, h=0.5  →  x = x + velocity(t=0.0, h=0.5) * 0.5
Step 1: t=0.5, h=0.5  →  x = x + velocity(t=0.5, h=0.5) * 0.5
```

The model sees `h = 0.5` at each step. Because iMF is **trained with random
sub-intervals** (`h = t - r` where `r ~ Uniform(0, t)`), it has learned to predict
the correct mean flow for *any* step size `h ∈ (0, 1]`. So `h = 0.5` is a valid
query — the model predicts the mean velocity to travel half the ODE path per step.

**Does ODE = 2 improve over ODE = 1?** The paper demonstrates **yes**:

| Model | NFE (ODE steps) | FID | Notes |
|---|---|---|---|
| iMF-XL/2 | 1 | 1.72 | One-shot (primary benchmark) |
| iMF-XL/2 | 2 | **1.54** | Two-step (paper Section 5.3) |

The 2-step result is strictly better. This makes mathematical sense: two smaller steps
with `h = 0.5` give the ODE integrator a mid-point correction opportunity. Even though
the model was primarily optimized for one-shot, the mean-flow training with random
sub-intervals also teaches it shorter-interval predictions.

**Note on CFG tuning**: The official 2-NFE evaluation uses different CFG parameters
(`ω = 4.0` instead of `8.0`, wider interval `[0.36, 0.64]` instead of `[0.42, 0.62]`).
This indicates that multi-step sampling shifts the optimal guidance scale — stronger
models (more NFE) favor weaker guidance. Our Gen3v4 port does not use CFG (no
`returns_condition` in the avoiding task), so this nuance does not apply.

### 14.4 What About ODE = 10?

With `flow_steps_v3 = 10` (the current default in `avoiding-d3il.py`), each step uses
`h = 0.1`. The model predicts mean velocities for very short intervals.

**Is this harmful?** No — but there are diminishing returns.

**Why it is not harmful:**
- The model is trained on all possible `(r, t)` pairs, including tiny `h` values.
  In the limit `h → 0`, the mean flow `u(z_t, r, t)` converges to the instantaneous
  velocity `v(z_t, t)` (this is the mathematical definition — the average velocity
  over an infinitesimal interval is the instantaneous velocity). So with many steps,
  iMF gracefully degrades to behaving like a standard FM ODE solver.
- The aux branch `v_pred` is explicitly trained on the FM velocity target `x_data - x_base`,
  so the combined velocity `u + 0.01·v` has a standard-FM component as a safety net.

**Why diminishing returns apply:**
- iMF's entire architectural innovation (h-conditioning, mean-flow target) is designed
  to make **few steps work well**. Using 10+ steps means you are solving a well-behaved
  ODE with many small Euler steps — exactly what standard FM already does. The h-MLP
  inside the UNet has learned the mean-flow shortcuts, but when `h = 0.1`, the "shortcut"
  is only 10% of the path, and the mean flow is nearly identical to the instantaneous
  velocity. You are paying for the extra h-conditioning compute without getting the
  benefit.
- The paper's FID progression (extrapolated from the 1→2 NFE improvement of
  1.72→1.54, roughly 0.18 FID per doubling) strongly suggests convergence: 2→4 might
  gain another ~0.1, 4→8 another ~0.05. By 10 steps, you are deep into diminishing
  returns territory.

**Conclusion for ODE = 10:** It works correctly and will produce valid samples, but
the quality gain over ODE = 2 (or even ODE = 1) is marginal at best. The 10-step
default in `avoiding-d3il.py` is a conservative legacy setting inherited from the
standard FM config. For MPC replanning (where inference latency matters), reducing to
ODE = 1 or 2 is strongly recommended.

### 14.5 Interaction with PCC Projection

The number of ODE steps also affects *when* QP projection fires:

| ODE Steps | Projection threshold = 0.5 | Projection steps | Total model+QP calls |
|---|---|---|---|
| 1 | `snapping_start_idx = 0` | Step 0 (always, via `loop_idx == flow_steps - 1`) | 1 model + 1 QP |
| 2 | `snapping_start_idx = 1` | Step 1 (last step) | 2 model + 1 QP |
| 10 | `snapping_start_idx = 5` | Steps 5, 6, 7, 8, 9 | 10 model + 5 QP |

With ODE = 1:
- The single model call produces a raw trajectory.
- One QP projection enforces constraints on that trajectory.
- This is the **minimal-cost** PCC pipeline: 1 NFE + 1 QP solve.

With ODE = 2:
- Two model calls produce a refined trajectory.
- One QP projection at the end.
- This is the "sweet spot" for quality/speed: 2 NFE + 1 QP solve.

With ODE = 10:
- Ten model calls, five QP projections.
- Much higher compute cost, minimal quality gain.

### 14.6 Summary Table

| Setting | What it is | Quality | Speed | PCC QP calls (T=0.5) |
|---|---|---|---|---|
| **ODE = 1** | True iMF one-shot (1-NFE) | Excellent — the primary iMF design point | Fastest | 1 |
| **ODE = 2** | Two-step iMF | Better than 1-shot (proven by paper: 1.54 vs 1.72 FID) | Fast | 1 |
| **ODE = 5** | Multi-step, approaching standard FM territory | Marginal gain over ODE=2 | Moderate | 2–3 |
| **ODE = 10** | Conservative, standard-FM-like | Near-plateau, not harmful but not helpful | Slow | 5 |
| **ODE = 100** | Pure ODE solving (iMF ≈ standard FM at small h) | No gain — doing useless work | Very slow | 50 |

### 14.7 Recommendation for Gen3v4 FM-PCC

1. **Primary config: `flow_steps_v3 = 1`** — True one-shot iMF. Fastest inference,
   minimal PCC overhead, and the generation quality that iMF was designed to deliver.
   Post-process with a single QP projection for constraint enforcement.

2. **Quality config: `flow_steps_v3 = 2`** — Two-step iMF. Slightly better quality
   at 2× model cost, still only 1 QP call (with `threshold = 0.5`). Good default if
   compute is not the bottleneck.

3. **Avoid `flow_steps_v3 ≥ 10`** — You are paying the cost of h-conditioning compute
   without the benefit of mean-flow shortcuts. If you need 10+ steps, you might as well
   use standard FM (FMv3ODE) which does not carry the aux-head overhead.

### 14.8 Why the Paper Says "One-Shot" but Allows Multi-Step

The iMF paper is clear about this: **one-shot (1-NFE) is the primary design goal**,
but the formulation naturally supports multi-step. Quoting the abstract:

> "iMF achieves 1.72 FID with a single function evaluation (1-NFE)"

And from Section 5.3:

> "When relaxing NFE to 2, iMF achieves an FID of 1.54."

The paper presents multi-step as a "relaxation" — you are spending extra compute for
diminishing-returns quality improvement. The model is not *worse* at multi-step
(because training covers all `h` values), but the marginal return per additional step
is small precisely because the model is already excellent at one-shot.

This is architecturally encoded: the official DiT model (`imfDiT.forward()`) only
conditions on `h = t - r`, **not** on `t` and `r` separately (see line 370 in
`imfDiT.py`: _"We don't explicitly condition on time t, only on h = t - r"_). This
means the model treats all intervals of the same width identically regardless of where
on the ODE path they fall. For ODE = 1, `h = 1.0` is a unique, well-defined query.
For ODE = 10, `h = 0.1` is essentially asking the model to predict a tiny local
velocity — something it can do, but it is not exploiting its mean-flow architecture.

**Key note for our Gen3v4 port:** Our UNet backbone conditions on both `t` **and** `h`
(the `h_mlp` is additive to the `time_mlp`), so `t` information is available. This
means our port can potentially distinguish between `(t=0.0, h=0.5)` and `(t=0.5, h=0.5)`
in a 2-step setting, which the official DiT cannot. Whether this provides any advantage
over the paper's h-only conditioning is an open empirical question.
