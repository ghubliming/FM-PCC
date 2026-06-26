# iMF-PCC — Full Math Reference: Train to Eval on Avoiding

**Scope:** Gen3v4 `flow_matcher_v3_imeanflow/`, U5 Phase 1 state, `imf_objective='meanflow_jvp'` path.  
**Code anchor:** `config/avoiding-d3il.py`, `imf_diffusion.py`, `unet1d_temporal_cond.py`, `imf_trajectory_model.py`.  
**Convention throughout:** τ=0 is pure noise, τ=1 is data (DATA-AT-1). Sampler integrates forward, 0→1.

---

## 0. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `B, H, D` | Batch size, Horizon (trajectory steps), Transition dim (obs+action) | — |
| `x₁` | Ground-truth trajectory (data) | `[B, H, D]` |
| `ε` | Gaussian noise sample | `[B, H, D]` |
| `z_τ` | Noisy interpolant at time τ | `[B, H, D]` |
| `r` | Interval start (noise-side anchor) | `[B]` |
| `t` | Interval end (data-side) | `[B]` |
| `h = t − r` | Interval width | `[B]` |
| `v` | Instantaneous velocity field | `[B, H, D]` |
| `u` | Mean-flow velocity (average over `[r,t]`) | `[B, H, D]` |
| `u_θ` | Network prediction (the u-head) | `[B, H, D]` |
| `v_θ` | Shared v-head prediction (aux) | `[B, H, D]` |
| `H` | Horizon (trajectory steps) | scalar |
| `D` | Transition dim = obs_dim + action_dim | scalar |
| `ω` | Interval-CFG guidance scale | scalar |
| `[τ_min, τ_max]` | Guidance interval | scalars |

---

## 1. Data Representation and Conditioning

The dataset is the `avoiding-d3il` task. Each sample is a **trajectory segment** of length `H` steps:

```
x₁  ∈  ℝ^{H × D}      D = obs_dim + action_dim
```

The **condition** `cond` pins specific timesteps of `z_τ` to observed values. In avoiding, `cond[0]` pins the first timestep's observation to the current robot state (x, y TCP position from `robot.current_c_pos[:2]`). `apply_conditioning` enforces this by overwriting the relevant slice at every trajectory operation (noise generation, interpolant, prediction):

```
apply_conditioning(z, cond, action_dim):
    for timestep k in cond:
        z[:, k, action_dim:] = cond[k]   # observation dims only; action dims free
```

This means the network always sees and produces trajectories whose observation start equals the current robot state — the boundary condition for the planning problem.

---

## 2. Time Sampling (Loss Entrypoint)

`imf_diffusion.py:loss()` is called by the trainer each step. It samples a **batch of times** `t` from a Beta distribution:

```
t ~ 1 − Beta(α, β)       α = time_beta_alpha_v3,  β = time_beta_beta_v3

Default U5 (all-power):   α = 1.0, β = 1.0  →  Beta(1,1) ≡ Uniform(0,1)
Legacy FM schedule:        α = 1.5, β = 1.0  →  mean(t) ≈ 0.4  (noise-leaning)
```

**Why this matters:** `t` is the data-side endpoint of the interval `[r, t]`. The `fm_equivalent` objective is immune to the distribution of `t` (its target is always `x₁−ε`). The `meanflow_jvp` objective is **not** — the JVP target `v + h·du/dr` contains the network's curvature, which can only be learned where `(r,t)` is well-covered. The uniform schedule gives `P(t≥0.9)=10%` vs `3.1%` for the Beta(1.5,1) schedule — three-fold more coverage in the near-data regime that few-step sampling relies on.

---

## 3. The Interpolant (Forward Process)

Given `x₁` (data), `ε ~ N(0,I)` (noise), and time `τ ∈ [0,1]`, the **linear OT interpolant** is:

```
z_τ = (1 − τ)·ε + τ·x₁                   (q_sample, imf_diffusion.py:169-175)
```

This is a straight line in ℝ^{H×D} space between noise and data. The instantaneous velocity along this line is:

```
v(z_τ, τ) = dz_τ/dτ = x₁ − ε             (constant along any single path)
```

Because both endpoints are fixed for a given `(ε, x₁)` pair, the **conditional velocity is constant** — it does not depend on `τ`. This is the crucial property: for the linear interpolant, any finite-difference approximation of the average velocity over a sub-interval collapses to the same constant `x₁ − ε`. The MeanFlow signal only exists in the **marginal** field (averaged over many data points sharing a given `z_τ`), not in the sample-level path.

---

## 4. Interval Sampling (Training)

For both objectives, a sub-interval `[r, t]` is drawn per sample:

```
r = t · U(0,1)           # r ~ Uniform(0, t)       (imf_diffusion.py:408)

Anchor override (meanflow_jvp only):
    anchor = Bernoulli(meanflow_r_equals_t_frac = 0.25)
    r = anchor ? t : r                              (forces h=0 for 25% of batch)

h = t − r    ∈ [0, t]
```

The anchor samples (`r = t`, `h = 0`) reduce the MeanFlow Identity to `u_target = v_inst` — pure FM velocity. They **ground** the average-velocity field to the known instantaneous velocity, preventing drift. The remaining 75% carry the genuine iMF signal.

The anchor point the network sees:
```
z_r = (1 − r)·ε + r·x₁       (q_sample at time r — the noise-side anchor)
z_r[cond_dims] ← cond         (apply_conditioning)
```

The network predicts from `z_r` at time `r`, not from `z_t`. This matches the inference convention: the sampler is always at the noise-side endpoint and asks "what average velocity should I use to step forward by `h`?"

---

## 5. Network Architecture (IMFBackbone)

The backbone is **`Flow_matcher_U_Net_v2`** (`unet1d_temporal_cond.py`), a 1D temporal U-Net with the following conditioning pathway.

### 5.1 Time/Interval Embedding

All scalar conditioning inputs are embedded via a shared sinusoidal MLP pattern:

```
SinEmbed(s) → Linear(dim_emb) → GELU → Linear(time_dim)
```

The conditioning embedding `e` is built additively:

```
e  = time_mlp(t)             # position within the interval (data-side endpoint)
e += h_mlp(h)                # interval width h = t − r  (always active)

if interval_cfg:              # U5: only when the net was built with this flag
    e += omega_mlp(ω)        # CFG scale conditioning
    e += tmin_mlp(τ_min)     # guidance lower bound
    e += tmax_mlp(τ_max)     # guidance upper bound
```

This embedding `e` is injected at every ResBlock in both the encoder and decoder arms of the U-Net via `ResBlock(x, e)`.

### 5.2 U-Net Structure

```
Input: z_r  ∈  ℝ^{B × H × D}

Encoder:  [Down₁, Down₂, Down₃, Down₄]      (dim_mults = 1, 2, 4, 8)
Middle:   [Mid₁, Mid₂]
Decoder:  [Up₄, Up₃, Up₂, Up₁]   with skip connections from Encoder

trunk = Up₁ output  ∈  ℝ^{B × H × freq_dim}    (shared post-up feature)
```

### 5.3 Dual-Head Output (U5, `dual_head=True`)

```
u = final_conv(trunk)          →  [B, H, D]    mean-flow velocity head (deployed)
v = v_final_conv(trunk)        →  [B, H, D]    instantaneous velocity head (aux, training only)
```

Both `final_conv` and `v_final_conv` read the **same `trunk` tensor** — this is the shared-backbone split that mirrors the official `u_heads / v_heads` in `imfDiT.py:374-388`. The v-head gradient therefore flows back through the full U-Net, regularizing the same representation that produces `u`.

**Legacy path** (`dual_head=False`): the v output comes from an orphan `aux_head` MLP acting on the raw input `z_r`, sharing no parameters with the backbone. In this case `meanflow_aux_weight` has no regularizing effect on `u`.

### 5.4 Conditioning Dropout (for CFG)

The backbone uses **condition dropout** (`condition_dropout` rate, default 0.1). When `force_dropout=True` (called during interval-CFG to get the unconditional prediction), the conditioning embedding is zeroed. This gives the CFG pair `(u_cond, u_uncond)`.

---

## 6. Training Objective A — `fm_equivalent` (Legacy Baseline)

This is the finite-difference path. The network is trained to predict the average velocity as:

```
u_target = (z_t − z_r) / h       where z_t = (1−t)ε + tx₁,  z_r = (1−r)ε + rx₁
```

**Algebraic collapse** (linear interpolant):
```
z_t − z_r = [(1−t)ε + tx₁] − [(1−r)ε + rx₁]
           = (t − r)(x₁ − ε)
           = h · (x₁ − ε)

⟹  u_target = (x₁ − ε)      for ALL (r, t)
```

The target is the **constant FM velocity**, independent of the interval. Training with this objective is exactly equivalent to flow matching. The `h`-conditioning is inert: the Bayes-optimal output of the network ignores `h` since the target does not depend on it.

Prediction and loss:
```
u_θ, v_θ = backbone(z_r, r, h)               # query at noise-side anchor
main_loss = L(u_θ, x₁ − ε)                   # u_mix · weighted MSE
aux_loss  = MSE(v_θ, x₁ − ε)                 # aux_loss_weight · aux MSE
total     = u_mix · main_loss + aux_loss_weight · aux_loss
```

---

## 7. Training Objective B — `meanflow_jvp` (Real iMF, U4+U5)

This is the real MeanFlow-Identity objective, implemented via forward-mode automatic differentiation.

### 7.1 The MeanFlow Identity (Math Derivation)

Define the average velocity over `[r, t]`:
```
u(z_r, r, t) · (t − r) = z_t − z_r           [definition: integral = chord]
```

Differentiate w.r.t. `r` at fixed `t`, following the trajectory (so `z_r` changes as the path follows `v`):

```
d/dr [u · (t−r)] = d/dr [z_t − z_r]

Left side:  (du/dr)·(t−r) + u·(−1) = (du/dr)·h − u

Right side: d(z_t)/dr − d(z_r)/dr = 0 − v(z_r, r)     [z_t fixed; dz_r/dr = v]

Equating:   (du/dr)·h − u = −v

⟹  u = v + h · (du/dr)                      [START-anchored MeanFlow Identity]
```

Where `du/dr` is the **total derivative** following the path:
```
du/dr = ∂u/∂z · v + ∂u/∂r − ∂u/∂h          (chain rule: z moves as v, r increases, h=t−r decreases)
```

This is a **Jacobian-vector product (JVP)** with tangent vector `(v, +1, −1)` on inputs `(z, r, h)`:
- `∂z/∂r = v_inst`  (trajectory velocity)
- `∂r/∂r = +1`      (explicit r-dependence)
- `∂h/∂r = −1`      (since h = t−r, dh/dr = −1)

### 7.2 JVP Implementation

```python
def _u_of(z_in, t_in, h_in):
    u, _ = self._predict_uv(z_in, cond, t_in, h=h_in,
                             omega=omega_c, t_min=t_min_c, t_max=t_max_c)
    return u

ones = torch.ones_like(r)
u_pred, du_dr = jvp(_u_of,
    primals  = (x_r,    r,     h   ),
    tangents = (v_inst, ones, -ones))
```

- `u_pred` = network prediction `u_θ(z_r, r, h)` — one full forward pass
- `du_dr`  = directional derivative `∂u_θ/∂z · v_inst + ∂u_θ/∂r − ∂u_θ/∂h` — forward-mode AD, no second-order grads needed

**CFG knobs `(ω, τ_min, τ_max)` are held constant** through the JVP (they are guidance hyperparameters, not trajectory inputs). Only `(z, r, h)` carry tangents.

### 7.3 Identity Target and Stop-Gradient

```python
h_expand = h.view(B, 1, 1)
u_target = (v_inst + h_expand * du_dr).detach()    # STOP-GRADIENT — critical
```

- At the **FM anchor** (`r = t`, `h = 0`): `u_target = v_inst = x₁ − ε` → pure FM signal
- At **general** `(r, t)` (`h > 0`): `u_target = v_inst + h · du_dr` → average-velocity target with curvature correction

The `.detach()` is **not optional**. The identity contains `u_θ` on both sides; without stop-gradient, backprop through `du_dr` would require second-order gradients and the training objective becomes ill-posed (the gradient of a self-referential loss blows up). Detaching makes the JVP-side a **bootstrap target**, structurally similar to TD-learning or consistency models.

### 7.4 Adaptive Loss Weight

The JVP targets have wildly varying magnitudes across the batch (large for wide intervals, small for anchors). A fixed MSE would let large-interval samples dominate:

```python
delta     = u_pred − u_target                     # [B, H, D]
sq        = (delta² * loss_fn.weights)             # apply trajectory/action loss_weights
per_sample = sq.mean(dim=[H, D])                  # [B] — per-sample squared error

w = 1 / (per_sample.detach() + c)^p               # per-sample weight (no gradient)
    c = meanflow_adaptive_c = 1e-3
    p = meanflow_adaptive_p = 0.5

main_loss = mean(w · per_sample)                   # weighted mean over batch
```

The weight `w ∝ 1/‖Δ‖^{2p}` down-weights samples with large prediction error (likely from undertrained regions early in training, or very wide intervals). This stabilizes the self-referential training loop.

### 7.5 Auxiliary v-Head Loss (when `dual_head=True`)

```python
_u2, v_pred = self._predict_uv(x_r, cond, r, h=h, ...)     # second forward pass
aux_loss = MSE(v_pred, v_inst)                               # v_inst = x₁ − ε
total_loss = main_loss + meanflow_aux_weight · aux_loss
```

Because `v_pred` reads from the **same backbone trunk** as `u_pred` (shared `v_final_conv`), the gradient of `aux_loss` flows through the entire U-Net. The v-head is trained to predict `v_inst = x₁ − ε` — the FM instantaneous velocity — which is a clean, constant, stable target. This acts as a **stabilizing regularizer on the shared trunk**: even when the JVP target is noisy (early training, wide intervals), the v-loss keeps the backbone anchored near the FM solution.

The v-head is **dropped at inference** — only `u_pred` is used in the sampler.

### 7.6 Full Training Loss (U5, `meanflow_jvp`, `dual_head=True`)

```
L = w · ‖u_θ(z_r, r, h) − [v_inst + h·JVP]‖²   (adaptive, weighted)
  + λ_aux · ‖v_θ(z_r, r, h) − v_inst‖²           (shared trunk stabilizer)

where:
  λ_aux = meanflow_aux_weight = 0.05
  JVP = du_θ/dr|(z_r,r,h) evaluated with tangents (v_inst, +1, −1)
  [v_inst + h·JVP].detach()                        (bootstrap target)
```

---

## 8. Interval-CFG (Training Conditioning, `interval_cfg=True`)

When `meanflow_cfg_omega > 0`, the network is conditioned on the guidance triple `(ω, τ_min, τ_max)` **during training** as well as inference. This conditions the backbone to be aware of its CFG context, so it can produce coherent conditional/unconditional predictions at inference.

Training conditioning:
```
ω_batch     = full(B, meanflow_cfg_omega)
τ_min_batch = full(B, meanflow_cfg_t_min)
τ_max_batch = full(B, meanflow_cfg_t_max)
```

These are embedded via `omega_mlp / tmin_mlp / tmax_mlp` (sinusoidal → Linear → GELU → Linear, same as `h_mlp`) and **summed into the shared time embedding `e`**. They are held constant when passing through the JVP — they are not differentiated inputs.

---

## 9. Inference: The Sampling Loop (`p_sample_loop`)

At inference, the trained `u_θ` is used to integrate a trajectory from noise to data.

### 9.1 Initialization

```python
x  = N(0, I)  ∈  ℝ^{B × H × D}            # pure noise sample
x  = apply_conditioning(x, cond, ...)       # pin observation boundary
dt = 1 / flow_steps_v3                     # uniform step size
h  = full(B, dt)                           # constant h throughout (uniform schedule)
```

### 9.2 Euler Integration Loop

For step `i = 0, 1, ..., flow_steps − 1`:

```
τ_i = i / flow_steps          # current position in [0, 1]
t_i = full(B, τ_i)

# Interval-CFG: apply guidance only inside [τ_min, τ_max]
ω_step = ω   if  τ_min ≤ τ_i ≤ τ_max   else  0

if ω_step > 0:
    u_cond   = u_θ(x, t_i, h, cond,    ω, τ_min, τ_max)
    u_uncond = u_θ(x, t_i, h, cond=∅,  ω, τ_min, τ_max)   [force_dropout=True]
    u_step   = u_uncond + ω · (u_cond − u_uncond)
else:
    u_step = u_θ(x, t_i, h, cond, ω, τ_min, τ_max)

x = x + u_step · dt                        # forward Euler 0→1
x = apply_conditioning(x, cond, ...)       # re-pin boundary each step
```

**Key design choices:**
- `t` is the **current position** `τ_i`, not frozen. The UNet embeds both `t` and `h`; freezing `t` would put every step out-of-distribution (causing chaotic rollouts — this was Deviation B in fix_3, reverted).
- `h` is constant (`= dt`). At inference the step size equals the interval the model was conditioned to predict over. At 2 NFE, `h = 0.5`; at 1 NFE, `h = 1.0` — the model must have seen such wide intervals during training (enforced by the uniform schedule).
- The u-head is used alone; the v-head is never called at inference (`_predict_velocity` only calls `_predict_uv` to extract `u`).

### 9.3 NFE Accounting

| Setting | `flow_steps_v3` | NFE (no CFG) | NFE (with CFG) |
|---|---|---|---|
| Legacy FM-equiv | 10 | 10 | 10 (no CFG) |
| Real iMF | 2 | 2 | **4** (cond + uncond each step) |
| Real iMF | 1 | 1 | **2** (one guided step) |

With interval-CFG active, each step inside `[τ_min, τ_max]` costs 2 network evaluations. At `flow_steps=2` with the default `[0.4, 0.6]` guidance interval, only the step that falls inside the interval triggers CFG — in practice both steps of a 2-step schedule may or may not be inside the interval depending on `dt`. Net cost is at most 2× per guided step.

---

## 10. DPCC Projector — Constraint Enforcement

After each Euler step, the **DPCC (Diffuser Predictive Constraint Controller) projector** optionally snaps the trajectory to the constraint manifold. In `avoiding-d3il`, the constraint is the **obstacle avoidance region** (the trajectory must not pass through the obstacle bounding box).

### 10.1 When the Projector Fires

```python
snapping_start_idx = int((1 − diffusion_timestep_threshold) · flow_steps)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps − 1)
```

- The projector fires only in the **tail of the rollout** — when the trajectory is close to the data manifold (τ close to 1). At early steps, the iterate is mostly noise; projecting it is geometrically uninformative.
- **At `flow_steps=10`** and a typical `threshold=0.1`: `snapping_start_idx = 9` → projector fires at step 9 only (the very last step).
- **At `flow_steps=2`**: `snapping_start_idx = 1` → projector fires at step 1 and step 2 — the entire rollout is "near end." This changes the projection geometry significantly versus the 10-step case (see §10.3).

### 10.2 Projection Mechanics

**SLSQP projection** (`scipy.optimize.minimize`, method='SLSQP'): given a raw trajectory `x ∈ ℝ^{H×D}`:

```
x* = argmin  ‖x − x_raw‖²          (min deviation from model proposal)
     subject to: g(x) ≤ 0           (obstacle avoidance constraints)
                 x[0, obs_dims] = cond[0]   (boundary condition)
```

The projector operates on the **unnormalized** trajectory space (the normalizer is passed to the projector); the model works in normalized space. The boundary between them is handled by the normalizer's `unnormalize / normalize` calls inside `projector.project`.

The cost `‖x − x*‖²` is recorded per step in `costs[loop_idx]` and returned in `infos['projection_costs']`.

### 10.3 Low-NFE Projection Caveat

The projector was designed and tuned for **10-step rollouts**. At 1–2 NFE:

- The iterates at step 0 (for 1-NFE) or step 0–1 (for 2-NFE) are **single giant jumps from noise** — the trajectory geometry is very different from step 9 of a 10-step rollout.
- The SLSQP starting point (`x_raw`) is far from the constraint manifold; convergence is not guaranteed, and projection cost is high.
- **Re-deriving `snapping_start_idx` for low NFE is required** before shipping a real-iMF DPCC evaluation. The correct approach is to project **only the final step** (set `threshold` so `snapping_start_idx = flow_steps − 1`) and accept that constraint-satisfying quality may differ from the 10-step case.

---

## 11. Policy Execution on Avoiding

The full closed-loop cycle on `avoiding-d3il`:

```
1. ENV OBSERVE:
   state = robot.current_c_pos[:2]         → [x_pos, y_pos] of TCP

2. CONDITION:
   cond = {0: obs_tensor}                  → pin trajectory start to current state

3. SAMPLE (iMF):
   x*, infos = imf_model.p_sample_loop(
       shape=(1, H, D), cond=cond,
       projector=dpcc_projector,
       constraints=obstacle_constraints
   )                                        → x* ∈ ℝ^{1 × H × D}  (normalized)

4. EXTRACT ACTION:
   x* = dataset.normalizer.unnormalize(x*)
   action = x*[0, 0, :action_dim]          → first-step joint velocity command

5. EXECUTE:
   robot.gotoCartPosQuatController(action)  → send to Franka arm
   sim.step()

6. GOTO 1
```

The normalizer inverts the dataset statistics (mean/std) applied during training so the output is in real robot-space units. The **action at timestep 0** is extracted and sent; the rest of the trajectory is discarded (receding-horizon MPC pattern).

---

## 12. End-to-End Math Summary

Putting it all together: **what the network actually computes at each training step and inference step.**

### Training Step (U5, `meanflow_jvp`)

```
Given:     x₁ ~ dataset,  cond,  t ~ 1 − Beta(1,1) = Uniform(0,1)

1.  ε ~ N(0,I)
2.  r = t · U(0,1);  25% of batch: r ← t   (FM anchor)
3.  h = t − r
4.  z_r = (1−r)·ε + r·x₁   [apply_conditioning]
5.  v_inst = x₁ − ε         [apply_conditioning]  ← FM velocity; also JVP tangent on z

6.  JVP through u_θ:
      u_pred, du_dr = jvp(u_θ,  (z_r, r, h),  (v_inst, +1, −1))
      ─── costs exactly 1 extra forward pass in forward-mode AD ───

7.  u_target = (v_inst + h·du_dr).detach()   ← bootstrap target, STOP-GRAD

8.  δ = u_pred − u_target
    w = 1 / (‖δ‖² + 1e-3)^0.5               ← adaptive weight (per sample)
    L_main = mean(w · ‖δ‖²)

9.  [optional, dual_head=True]:
      _, v_pred = u_θ(z_r, r, h)             ← second forward for v-head
      L_aux = ‖v_pred − v_inst‖²
      L = L_main + 0.05 · L_aux

10. Backprop through L → update θ            ← standard Adam step
```

### Inference Step (per Euler step i, with CFG)

```
Given:  x at step i,  τ_i = i/N,  h = 1/N,  cond,  ω, [τ_min, τ_max]

1.  u_cond   = u_θ(x, τ_i, h, cond,    ω, τ_min, τ_max)
2.  if τ_min ≤ τ_i ≤ τ_max:
        u_uncond = u_θ(x, τ_i, h, ∅,   ω, τ_min, τ_max)
        u_step   = u_uncond + ω·(u_cond − u_uncond)     ← interval-CFG
    else:
        u_step = u_cond
3.  x ← x + u_step / N               ← Euler step
4.  [apply_conditioning]
5.  [DPCC project, if near-end]
```

---

## 13. Hyperparameter → Math Mapping

| Config key | Math role | Default (U5 all-power) |
|---|---|---|
| `imf_objective` | Selects §6 vs §7 objective | `'meanflow_jvp'` |
| `time_beta_alpha_v3` | α of Beta for t | 1.0 (Uniform) |
| `time_beta_beta_v3` | β of Beta for t | 1.0 |
| `meanflow_r_equals_t_frac` | Fraction of batch with `r=t` (FM anchor) | 0.25 |
| `meanflow_adaptive_p` | Exponent p in w=(‖Δ‖²+c)^{−p} | 0.5 |
| `meanflow_adaptive_c` | Epsilon c in adaptive weight | 1e-3 |
| `meanflow_aux_weight` | λ_aux (v-head stabilizer) | 0.05 |
| `dual_head` | Shared backbone v-head (vs orphan MLP) | True |
| `interval_cfg` | Build CFG embedding layers in UNet | True |
| `meanflow_cfg_omega` | ω (guidance scale) | 4.0 |
| `meanflow_cfg_t_min` | τ_min (guidance interval lower bound) | 0.4 |
| `meanflow_cfg_t_max` | τ_max (guidance interval upper bound) | 0.6 |
| `flow_steps_v3` | N (inference Euler steps, NFE) | 2 |

---

## 14. What Is Still FM vs What Is Real iMF

| Component | Status | Notes |
|---|---|---|
| Interpolant `z_τ = (1−τ)ε + τx₁` | FM — unchanged | Linear OT; causes sample-level velocity collapse |
| `fm_equivalent` objective | **FM** | Algebraically = x₁−ε; baseline only |
| `meanflow_jvp` objective | **Real iMF** | MeanFlow Identity via JVP; stop-grad target |
| r=t anchor (25%) | Real iMF | FM signal that grounds the field |
| Adaptive loss weight | Real iMF | Required for stable bootstrapped training |
| Shared backbone u/v heads (`dual_head=True`) | **Real iMF** | Official u_heads/v_heads split |
| v-head dropped at inference | Real iMF | Matches official eval_mode |
| Interval-CFG conditioning | **Real iMF** | Official ω/(τ_min,τ_max) conditioning |
| h-conditioning in the UNet | Real iMF | Already present since U1 |
| Network architecture (UNet) | **Not official** | Official uses DiT; UNet is placeholder |
| `flow_steps_v3 = 2` (low NFE) | Real iMF intent | 2-NFE is the iMF selling point |
| DPCC projection schedule | **Needs re-tuning** | Designed for 10-step; see §10.3 |
