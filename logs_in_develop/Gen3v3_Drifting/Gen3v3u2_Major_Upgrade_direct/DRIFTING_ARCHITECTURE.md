# FM-D (Drifting) Architecture, Math, and Comparison to Standard FM

**Scope**: Gen3v3 FM-D — `flow_matcher_v3_drifting/` module  
**Date**: 2026-05-28  
**Branch**: `update_into_FM`  
**Changelog**: [`CHANGELOG.md`](CHANGELOG.md)  
**Projection doc**: [`PCC_PROJECTION_IN_DRIFTING.md`](PCC_PROJECTION_IN_DRIFTING.md)

---

## 1. What FM-D Is and Why It Exists

**FM-D (Flow Matching + Drifting)** extends standard Flow Matching with a **drift
regularization** mechanism: a learned loss that measures how far a trajectory is from
the expert distribution, then adds its gradient to the velocity field during integration
(or to the training loss) to steer trajectories back toward expert-like behaviour.

The motivation is distribution shift. A standard FM model trained offline on expert
demonstrations may generate trajectories that are nominally near-data but drift into
low-density regions — especially under MPC replanning with imperfect conditioning. The
drift field acts as a learned attractor toward the expert distribution, supplementing
the FM flow without changing its architecture.

The drift mechanism has two independent application points:

| Where | How | Implemented |
|---|---|---|
| **Training** | Combined FM loss + drift regularizer; biases model weights toward expert-manifold | `DriftTrainingWrapper` + `DriftLoss` |
| **Inference** | Subtract `λ · ∇_x(drift_loss)` from the velocity field at each ODE step | `DriftAugmentedVelocityField` + `DriftODESolver` |

Both share the same `DriftLoss` module (circular memory bank of expert trajectories +
MLP encoder). They can be used independently or together.

---

## 2. Task Context

Gen3v3 FM-D operates on the **D3IL avoiding task** — identical task context to Gen3v4
iMF. Config: `config/avoiding-d3il.py`, key blocks `flow_matching_v3_drifting` (train)
and `plan_fm_v3_drifting` (eval). Same state-only observations, H=8 horizon.

---

## 3. Class Hierarchy

```
FlowMatchingDrifting                ← FM-PCC-compatible outer wrapper (diffusion)
  └─ Flow_matcher_U_Net_v2          ← Standard FM UNet backbone (same as FMv3ODE)

DriftLoss                           ← Reference distribution module (standalone)
  ├─ encoder (MLP, 3-layer)         ← Embeds trajectories to 128D for comparison
  ├─ discriminator (MLP, optional)  ← Adversarial variant only
  └─ memory_bank (circular buffer)  ← Stores expert trajectories from training batches

DriftConditioner                    ← Trajectory + drift metric embedding (optional)
DriftAugmentedUNet1D               ← Augmented model variant (not default config)

DriftTrainingWrapper                ← Wires DriftLoss into Trainer.train_epoch()
  ├─ DriftLossScheduler             ← Warmup / constant / decay weight schedule
  └─ [owns] DriftLoss               ← Single authoritative reference bank

DriftAugmentedVelocityField        ← Wraps velocity_fn for inference-time drift
DriftODESolver                      ← ODE solver with optional drift guidance
```

**Important**: The default training config uses `'model': 'models.Flow_matcher_U_Net_v2'`
— the same UNet as FMv3ODE, not `DriftAugmentedUNet1D`. Drift enters the model
indirectly (training loss) and optionally at inference via `DriftODESolver`, not by
modifying the UNet's conditioning stream.

---

## 4. Training: FM Loss + Drift Regularization

### 4.1 Standard FM Objective (unchanged from FMv3ODE)

The `p_losses` computation in `FlowMatchingDrifting` is **identical** to FMv3ODE:

```python
def p_losses(x_start, cond, t):
    x_base   = torch.randn_like(x_start)          # sigma=1.0 noise sample
    x_t      = (1-t) * x_base + t * x_start       # linear OT interpolant
    v_target = x_start - x_base                    # standard FM velocity target

    v_pred   = model(x_t, cond, t)                # UNet forward
    loss, _  = loss_fn(v_pred, v_target)           # weighted MSE
    return loss

# Time sampling (DATA-AT-1 convention):
t = 1.0 - Beta(α=1.5, β=1.0).sample()            # biased toward t ≈ 1 (near data)
```

The FM component of training is identical to FMv3ODE. The drift does NOT change the
training target, interpolant, or time sampling.

### 4.2 Drift Regularization Loss

Added on top of the FM loss via `DriftTrainingWrapper.compute_training_loss`:

```python
total_loss = fm_loss + drift_weight * drift_loss

# drift_weight from DriftLossScheduler (warmup → target_weight=0.1)
```

`drift_loss` is computed by `DriftLoss.forward(sampled_trajectory)` against the
reference memory bank populated by expert batches.

**Training workflow per gradient step** (after C-2 fix in `Trainer.train_epoch`):
1. `update_memory_bank_from_batch(batch[0].detach())` — push current expert batch into memory bank
2. `compute_training_loss(batch[0].detach(), fm_loss)` — compute drift loss against bank
3. `drift_wrapper.step()` — advance weight scheduler

> Note: `sampled_trajectory` is the **expert batch** (not a model-generated trajectory),
> which is an approximation. Ideally it would be a mid-training model sample, but that
> is too expensive for the inner loop (C-2 note).

---

## 5. The Drift Loss Module

`DriftLoss` measures distance between a trajectory and the expert distribution.
Three variants are implemented:

### 5.1 `embedding_nn` (default)

Nearest-neighbour proxy in learned embedding space:

```
sampled_traj → encoder (MLP-3) → q_z (B, 128)
memory_bank  → encoder (MLP-3) → p_z (N_ref, 128)

dist = cdist(q_z, p_z)                    # (B, N_ref) pairwise L2
probs = softmax(-dist / temperature)       # temperature=0.1 → near-argmax
loss = -log(max_prob + 1e-8)               # high loss when no close expert neighbour
```

Both `q_z` and `p_z` flow through the encoder with gradients (both branches train the
encoder after M-2 fix). The encoder's embedding space adapts to the data.

> Despite the original name `compute_kl_divergence`, this is NOT KL divergence. It is
> a nearest-neighbour loss using L2 distances in embedding space (M-1 fix renamed it to
> `compute_embedding_nn_loss`).

### 5.2 `mmd` — Maximum Mean Discrepancy

RBF-kernel MMD between sampled and expert embeddings:

```
K_qq = rbf(q_z, q_z),  K_pp = rbf(p_z, p_z),  K_qp = rbf(q_z, p_z)
MMD = sqrt( E[K_qq] - 2·E[K_qp] + E[K_pp] )
```

Both branches train the encoder. Measures distributional distance rather than
nearest-neighbour proximity.

### 5.3 `adversarial`

Discriminator-based: encoder maps trajectories to 128D, discriminator classifies
sampled vs expert. Generator loss trains encoder toward fooling discriminator.
Reference branch uses `detach()` in the discriminator step only.

### 5.4 Memory Bank

`DriftLoss.memory_bank` is a circular buffer of `max_size=5000` flattened trajectories
(`T × state_dim`). Written by `update_memory_bank()` during training. At inference the
bank holds the last `min(n_training_batches × batch_size, 5000)` expert trajectories
seen during training.

### 5.5 Gradient for Inference Guidance

```python
def get_gradient(trajectory):
    trajectory = trajectory.clone().detach().requires_grad_(True)
    loss = self.forward(trajectory)['loss']
    loss.backward()
    return trajectory.grad
```

This gradient points in the direction of increasing drift loss (away from experts).
Subtracting it from the velocity moves the trajectory toward lower loss = toward experts.

---

## 6. Inference: Drift-Augmented Velocity Field

At inference the drift gradient can be subtracted from the FM velocity at each ODE step.
This is implemented in two places:

### 6.1 `DriftAugmentedVelocityField` (in `DriftODESolver`)

```python
class DriftAugmentedVelocityField:
    def __call__(self, t, x):
        velocity = self.velocity_fn(t, x, **self.kwargs)    # FM model velocity

        if self.drift_weight > 0 and self.drift_loss_fn is not None:
            drift_grad = self.drift_loss_fn(x)              # ∇_x drift_loss
            drift_grad = clip_by_norm(drift_grad, self.drift_clip)

            velocity = velocity - self.drift_weight * drift_grad  # gradient descent
        return velocity
```

Sign convention: **subtract** gradient (gradient descent on drift loss = move toward
experts). Before M-3 fix the sign was wrong (`velocity += drift_grad` = gradient ascent
= move *away* from experts).

`DriftODESolver.solve()` wraps `velocity_fn` with `DriftAugmentedVelocityField` when
`drift_weight > 0`. Supports both legacy Euler and torchdiffeq backends.

### 6.2 `p_sample_loop` in `FlowMatchingDrifting`

The standard `p_sample_loop` calls `self.p_sample() → self._predict_velocity() → self.model(x, cond, t)`. If `self.model` is the plain `Flow_matcher_U_Net_v2` (default), there is **no runtime drift subtraction** in this path — drift conditioning only enters through the training weights.

To get explicit runtime drift augmentation via `p_sample_loop`, the model would need to
be a `DriftAugmentedUNet1D` (alternative architecture, not default config) or `DriftODESolver`
would need to be called directly from the eval script.

The `use_drift_augmentation: True` config flag is stored as `self.use_drift_augmentation`
(D-3 fix) but not yet acted upon inside `p_sample_loop` itself — it signals the intent
and is available for future wiring.

---

## 7. `DriftAugmentedUNet1D` — Alternative Model Architecture

Available but not used in default config. Would be selected via:
```python
'model': 'flow_matcher_v3_drifting.models.DriftAugmentedUNet1D'
```

It wraps `Flow_matcher_U_Net_v2` with a `DriftConditioner` that appends a drift-aware
embedding to the conditioning vector before the UNet forward pass:

```
trajectory history (B, T, state_dim)  ──► DriftConditioner encoder ──► (B, cond_dim=64)
drift_metrics (B, metric_dim)          ──► drift_proj Linear           ──► (B, cond_dim)
                                                          ↓
                                        drift_cond = encoder_out + 0.1 * drift_proj_out
                                        augmented_cond = cat([original_cond, drift_cond])
                                                          ↓
                                        base_unet(x, augmented_cond, t)
```

`DriftConditioner.drift_proj` is registered as a proper `nn.Linear` (C-3 fix) so
weights persist across calls. When used, the model can see the current drift quality
and adapt the velocity field accordingly at the architecture level, rather than only via
the gradient subtraction mechanism.

---

## 8. `DriftLossScheduler` — Training Weight Schedule

Three modes for controlling `λ` (drift loss weight) during training:

| Mode | Behaviour |
|---|---|
| `warmup` | Linear ramp `start_weight → target_weight` over `warmup_steps` steps |
| `constant` | Fixed `target_weight` throughout |
| `exponential_decay` | `target_weight × decay_rate^step_count` |

Default: `warmup`, `target_weight=0.1`, `warmup_steps=1000`.

The counter is `self._step_count` (C-1 fix — previously `self.step`, shadowing the
`step()` method, causing `TypeError: 'int' object is not callable`).

---

## 9. ODE Sampling

`FlowMatchingDrifting.p_sample_loop` follows the same forward-Euler 0→1 pattern as
FMv3ODE:

```python
x = 0.5 * torch.randn(shape)       # NOTE: sigma=0.5 (see note below)
x = apply_conditioning(x, cond, action_dim)

for loop_idx in range(flow_steps):
    t_cont = loop_idx / flow_steps    # t ∈ [0, 1)
    dt     = 1.0 / flow_steps
    x      = p_sample(x, cond, t_cont)   # one Euler step: x += v * dt
    x      = apply_conditioning(x, ...)

    # PCC projection near end (see Section 10)
    if projector and near_end:
        x = project_or_gradient(x)

return x, infos
```

**Sigma discrepancy**: `p_sample_loop` uses `0.5 * torch.randn(...)` (sigma=0.5) while
training `p_losses` uses `torch.randn_like(x_start)` (sigma=1.0). This mismatch means
the initial noise at inference is half as large as training expects. The iMF module
fixed this (MATH-04) but the fix was not ported here. The effect is that early ODE
steps start from a tighter Gaussian than the model trained on — trajectories may be
less diverse.

**torchdiffeq**: supported (unlike iMF which still has BUG-01). The backend is
selectable via `ode_solver_backend_v3`: both `legacy_euler` and `torchdiffeq` are
implemented with correct kwargs forwarding (C-6 fix).

---

## 10. PCC Projection

Identical snapping formula and semantics as iMF:

```python
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * flow_steps)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)
```

Gradient and SLSQP modes both supported. The `T` threshold was already baked into the
plan path via `diffusion_timestep_threshold: _yaml_threshold` (no fix needed unlike iMF
which required this addition in the post-release fix).

The drift field (if active) provides a soft pre-push toward feasible regions **before**
the SLSQP snap. In principle this leaves less correction work for the projector —
trajectories that drift naturally toward valid regions need smaller SLSQP adjustments.
This is the core synergy between FM-D and PCC.

---

## 11. Key Fixes from Gen3v3u2 (CHANGELOG Summary)

| Fix | What was wrong | What was fixed |
|---|---|---|
| **C-1** | `DriftLossScheduler.step = 0` shadowed `step()` method → `TypeError` on first call | Counter renamed to `_step_count` |
| **C-2** | `DriftTrainingWrapper` was dead code — `Trainer` never called it | `Trainer.train_epoch()` wired: memory bank update → combined loss → scheduler step |
| **C-3** | `DriftConditioner` created `nn.Linear` fresh every forward call (random weights, memory leak) | `self.drift_proj = nn.Linear(...)` moved to `__init__` |
| **C-4** | `wrap_unet` missing `@staticmethod` — instance passed as `base_unet` | `@staticmethod` added |
| **C-5** | `DriftTrainingWrapper` maintained a duplicate `memory_bank` (Bank A never read) | Duplicate removed; single bank in `DriftLoss` |
| **C-6** | `DriftAugmentedVelocityField` dropped `**kwargs` (cond, returns) → model received no conditioning | `self.kwargs` stored at init, merged at each call |
| **M-1** | `compute_kl_divergence` was nearest-neighbour loss (not KL) | Renamed to `compute_embedding_nn_loss`; `'kl_divergence'` → `'embedding_nn'` everywhere |
| **M-2/M-4** | Reference encoder branch wrapped in `no_grad()` → gradients only through sampled branch | `no_grad()` removed from both `embedding_nn` and `mmd` reference branches |
| **M-3** | Drift guidance sign was `velocity += drift_grad` (gradient ascent = away from experts) | Corrected to `velocity -= drift_grad` (gradient descent = toward experts) |
| **M-5** | `compute_ode_efficiency = steps / max_steps` (more steps = higher efficiency) | Inverted: `efficiency = 1 - steps/max_steps` |
| **D-3** | Config keys `use_drift_augmentation`, `drift_loss_weight`, `drift_loss_type` silently dropped | Added to `GaussianDiffusion.__init__` as stored attributes |

The most consequential: **C-2** made drift training functional, **M-3** corrected the
sign so drift actually steers toward experts, **C-6** ensured conditioning reached the
model during drift-guided inference, **C-1** allowed the scheduler to advance at all.

### Known Remaining Issue

**D-1, D-2 (algorithm mismatch)**: The original `/workspaces/drifting/drift_loss.py`
implements a JAX-based non-parametric **force-field kernel matching** (`[B, C, S]` tensors,
multi-scale `R_list`, symmetric softmax affinity, stop-gradient). The port is a different
algorithm (MLP encoder + NN proxy). Faithfully porting the original requires redesigning
`DriftLoss` from scratch. Tracked for Gen3v3u3.

---

## 12. Comparison to FMv3ODE and iMF

| | **FMv3ODE (Gen3v5)** | **FM-D (Gen3v3)** | **iMF (Gen3v4)** |
|---|---|---|---|
| **Training target** | `x_data - x_noise` | Same FM target + drift regularizer | Mean flow `(x_data - x_r) / h` |
| **Extra training component** | None | `drift_weight * DriftLoss` | `aux_weight * MSE(v_pred, FM_target)` |
| **Model backbone** | `Flow_matcher_U_Net_v2` | Same (default) | `Flow_matcher_U_Net_v2` + `h_mlp` |
| **Extra model component** | None | `DriftLoss` + `DriftTrainingWrapper` | `aux_head` (zero-init MLP) |
| **h-conditioning** | No | No | Yes — step size fused into time embed |
| **Dual output** | Single velocity | Single velocity | `(u_pred, v_pred)` |
| **Inference velocity** | `v_θ(x, t)` | `v_θ(x, t) − λ·∇drift` (if DriftODESolver) | `u_pred + 0.01·v_pred` |
| **Inference steps** | 10 | 10 | 10 (goal: 1) |
| **Expert memory needed** | No | Yes (circular bank, 5000 slots) | No |
| **PCC projection** | Threshold snap, last `(1-T)·K` steps | Same (identical formula) | Same |
| **torchdiffeq** | Yes | Yes | Deferred (BUG-01) |
| **Sigma at inference** | 1.0 | 0.5 (discrepancy vs training) | 1.0 (MATH-04 fix) |
| **Key strength** | Stable FM baseline | Learned distribution pull toward experts | Fewer inference steps |
| **Key dependency** | None | Requires populated memory bank | Requires h-conditioning correct |

### When FM-D adds value over FMv3ODE

The drift regularizer adds value when:
1. The FM model generates trajectories that are geometrically plausible but behaviorally
   out-of-distribution (subtle distribution mismatch the FM loss alone doesn't penalise)
2. The expert distribution is compact and well-represented in the 5000-slot memory bank
3. The drift field synergizes with PCC: trajectories are already leaning toward
   constraint-feasible regions before SLSQP acts

FM-D and FMv3ODE are **trained on the same FM objective** — the only difference at train
time is the additional regularizer. If the drift loss weight is zero, FM-D reduces
exactly to FMv3ODE.

---

## 13. Config Reference

### Training block (`flow_matching_v3_drifting`)

```python
'model':     'models.Flow_matcher_U_Net_v2',         # same backbone as FMv3ODE
'diffusion': 'models.diffusion.FlowMatchingDrifting',
'horizon':   8,
'dim':       32,                 # UNet base channel dim
'dim_mults': (1, 2, 4, 8),
'action_weight':   1,
'loss_discount':   1.0,          # uniform step weights (no decay)
'time_beta_alpha_v3': 1.5,       # Beta prior α — must match plan block
'time_beta_beta_v3':  1.0,

# Drift augmentation (3 locked params)
'use_drift_augmentation': True,
'drift_loss_weight':      0.1,   # λ in combined loss
'drift_loss_type':        'embedding_nn',   # 'embedding_nn' | 'adversarial' | 'mmd'

# Training
'n_train_steps':            100000,
'batch_size':               8,
'gradient_accumulate_every': 2,   # effective batch = 16
'learning_rate':            1e-4,
'ema_decay':                0.995,
```

### Plan block (`plan_fm_v3_drifting`)

```python
'diffusion':       'models.diffusion.FlowMatchingDrifting',
'flow_steps_v3':   10,
'time_beta_alpha_v3': 1.5,        # MUST match training
'time_beta_beta_v3':  1.0,
'ode_solver_backend_v3': 'legacy_euler',    # or 'torchdiffeq'
'ode_solver_method_v3':  'euler',
'use_drift_augmentation': True,
'drift_loss_weight':      0.1,
'drift_loss_type':        'embedding_nn',
'diffusion_timestep_threshold': _yaml_threshold,   # PCC snap + T in path
'batch_size': 4,                  # MPC candidates
```

### Checkpoint path

```
logs/avoiding-d3il/flow_matching_v3_drifting/
  H{H}_D{diffusion}_a{alpha}_b{beta}_aw{aw}/

Plans:
logs/avoiding-d3il/plans/flow_matching_v3_drifting/
  H{H}_D{diffusion}_a{alpha}_b{beta}_aw{aw}/
    H{H}_K{K}_Meuler_T{T}_D{diffusion}/
```

---

## 14. Summary: FM-D in One Paragraph

FM-D is standard Flow Matching (`v_target = x_data - x_noise`, linear OT interpolant,
forward Euler 0→1) plus a **drift regularizer** that pushes training and inference
toward the expert distribution. A circular memory bank accumulates expert trajectory
embeddings during training; a learned MLP encoder maps trajectories to 128D and computes
a nearest-neighbour loss against the bank. At training time this loss is added (weighted
by a warmup schedule) to the FM regression objective. At inference time the gradient of
the loss can be subtracted from the velocity field step-by-step
(`velocity -= λ · ∇_x drift_loss`), steering the ODE trajectory toward expert-like
regions before the SLSQP projector delivers the final constraint snap.

---

## 15. ODE Steps, Drift-at-Inference, and PCC in FM-D

*Companion analysis to iMF §14. The drifting case is more complex and the answers
differ substantially from iMF.*

---

### 15.1 Core Questions

1. Is FM-D natively one-shot like the original drifting paper?
2. Since drift is "all in training", can we drop projection in eval?
3. At ODE = 1, is PCC useless (trajectory already near-feasible from drift)?
4. What is the right ODE step count for FM-D?

---

### 15.2 Architecture Divergence: Original Drifting vs Our FM-D Port

The original drifting paper (`DitGen`, JAX) is **strictly one-shot**:

```
noise → [DiT single forward pass] → trajectory
```

No ODE. No timestep `t`. No loop. The drift loss is a force-field kernel matching
algorithm applied **only during training** to regularize the DiT weights. At inference
there is no drift gradient — just a single transformer forward. PCC projection could be
added as a post-processing step, but the model has no ODE structure to project into.

Our FM-D port (`FlowMatchingDrifting`) is **architecturally completely different**:

```
noise → [FM ODE with flow_steps steps] → trajectory
```

It is standard FM (forward Euler, Beta prior, same as FMv3ODE) with drift as a
**training-time regularizer** only. The `DriftAugmentedVelocityField` / `DriftODESolver`
path (for inference-time drift gradient subtraction) is implemented but **not wired into
`p_sample_loop`** — `use_drift_augmentation=True` in config is stored as an attribute
but never acted upon inside the main sampling loop. In practice:

| Component | Original Drifting | FM-D Port (Gen3v3) |
|---|---|---|
| Generator | Single-pass DiT (no ODE) | Standard FM ODE |
| Drift at training | Force-field kernel matching | Embedding NN loss (approximation) |
| Drift at inference | None (training-only) | **Also none** (unwired in `p_sample_loop`) |
| Model weights | Biased toward expert distribution | Same |
| PCC hook | Not applicable (no ODE) | End-of-ODE SLSQP snap |

**The critical implication**: FM-D at inference is mechanistically **identical to standard
FMv3ODE** plus biased model weights from drift training. There is no runtime drift
gradient subtraction happening. The drift only shapes the learned velocity field during
training; at inference you are running a plain FM ODE.

---

### 15.3 Can Projection Be Dropped in Eval?

**Short answer: No — for hard constraints. Partial credit: drift reduces how much PCC
needs to correct.**

The argument "drift is all in training, so projection isn't needed in eval" conflates two
different things:

| | Drift (training) | PCC Projection (eval) |
|---|---|---|
| **What it enforces** | Soft proximity to expert distribution | Hard geometric constraints (bounds, halfspace, dynamics) |
| **Guarantee type** | Probabilistic — pushes toward expert regions | Deterministic — SLSQP guarantee at snap |
| **Does expert data satisfy constraints?** | Not guaranteed | N/A — PCC enforces constraints regardless |
| **What happens without it** | Distribution may drift out-of-expert | Constraints may be violated |

Drift pushes the FM model to generate trajectories that *look like* expert demonstrations.
If experts were constrained-feasible, drift biases toward feasible trajectories. But it is
a *soft prior*, not a *hard projection*. The SLSQP projector gives a deterministic
guarantee that the workspace bounds, halfspace, and dynamics constraints are satisfied —
something a learned signal cannot provide.

**The synergy**: Drift reduces the magnitude of PCC correction (trajectories are already
near-feasible before snapping), improving SLSQP convergence quality. PCC provides the hard
guarantee drift cannot. They are complementary, not redundant.

---

### 15.4 ODE = 1 Analysis for FM-D

With `flow_steps_v3 = 1`:

```
dt = 1.0,  t = 0.0

x_noise → [FM_velocity(x=noise, t=0.0)] → x_data_raw
         → [PCC snap — fires because loop_idx == flow_steps-1] → x_constrained
```

**Key differences from iMF ODE = 1**:

| | iMF ODE = 1 | FM-D ODE = 1 |
|---|---|---|
| Architecture designed for 1-shot? | **Yes** — h-conditioning, mean-flow training | **No** — standard FM, no h-conditioning |
| Quality at ODE = 1 | Excellent (primary design point, 1.72 FID) | Poor (FM requires multiple Euler steps) |
| Why it can skip steps | Learns chord velocity `u = (x_data - x_r)/h` | Learns instantaneous velocity only |
| Drift help at t = 0? | N/A (iMF has no drift) | Marginal — weights biased, but x=noise gives weak signal |
| PCC correction work | Small — trajectory already near-feasible | Heavy — FM ODE=1 output is far from data manifold |

At ODE=1, the FM velocity is evaluated at `t=0` on **pure Gaussian noise**. The drift
bias in the model weights provides some benefit (the velocity steers toward the expert
distribution), but the fundamental FM accuracy issue remains: a single Euler step from
pure noise cannot traverse the full data manifold accurately without the iterative
correction that multi-step ODE provides.

**Is PCC useless at ODE=1?** The opposite — PCC is *more necessary* at ODE=1 because
the raw trajectory quality is worse. PCC does heavy SLSQP correction on a poorly-formed
trajectory. This is the reverse of iMF, where ODE=1 produces a high-quality trajectory
needing only minor PCC adjustment.

---

### 15.5 ODE = 2: Marginal Improvement

```
Step 0: t=0.0 → x = x + FM_vel(x_noise, t=0.0) * 0.5    [trajectory at ~midpoint]
Step 1: t=0.5 → x = x + FM_vel(x_mid,   t=0.5) * 0.5    [trajectory reaches data region]
               → PCC snap (snapping_start_idx=1, threshold=0.5)
```

Better than ODE=1: the mid-point correction at step 1 significantly improves trajectory
quality. Drift-biased weights help more at step 1 (x is now near the data manifold, so
the drift gradient would be more informative if active). Unlike iMF where the paper
proves ODE=2 improves FID 1.72→1.54, FM-D has no equivalent guarantee — it is just
better-behaved ODE integration.

---

### 15.6 ODE = 10 (Default): Drift-PCC Synergy in Action

With `flow_steps_v3 = 10`, by step 5 the trajectory is already near-feasible (drift-biased
FM has been integrating toward expert-like trajectories). The PCC projector snaps a
trajectory that is *almost* constraint-satisfying → smaller SLSQP displacements, better
convergence.

**Interaction table** (threshold = 0.5):

| ODE Steps | PCC QP calls | Drift benefit | FM quality | Recommendation |
|---|---|---|---|---|
| 1 | 1 (always) | Marginal (pure noise) | Poor | Avoid — PCC overloaded |
| 2 | 1 (step 1) | Modest (mid-trajectory) | Acceptable | Compute-constrained fallback |
| 5 | 2–3 | Good (near data) | Good | Reasonable |
| **10** | 5 | **Best (drift accumulates)** | **Best** | **Default — use this** |

---

### 15.7 Why FM-D Cannot Match iMF One-Shot Quality

iMF's one-shot capability comes from **architectural design**:
- `h_mlp` in the UNet: model conditions on step size → learns mean-flow shortcut
- Mean-flow training target `(x_data - x_r)/h`: explicitly teaches the chord velocity

FM-D has **neither**:
- No `h_mlp` → model cannot adapt to step size → ODE=1 means a bad single Euler step
- Training target is standard FM `v = x_data - x_noise` → no mean-flow shortcuts

The drift bias makes the FM velocity field **point more toward the expert distribution**,
which helps quality. But it does not make FM capable of one-shot generation — the FM ODE
integration needs multiple steps for accurate trajectory reconstruction regardless.

**Key asymmetry vs iMF**:
- iMF: ODE steps ↓ → still good quality (by design)
- FM-D: ODE steps ↓ → quality degrades (standard FM limitation)

---

### 15.8 The "Fake FM-D" Summary

Our Gen3v3 port is "fake" FM-D in two senses:

1. **Drift algorithm mismatch**: Original drifting uses force-field kernel matching
   (JAX, multi-scale `R_list`, symmetric softmax affinity). Our port uses embedding NN
   loss (MLP encoder + L2 nearest-neighbour). Different algorithm, different math
   (D-1/D-2 in CHANGELOG are unfixed).

2. **Inference structure mismatch**: Original drifting is one-shot DiT, no ODE. Our port
   is standard FM ODE, with drift gradient at inference unwired from `p_sample_loop`.
   What we actually run at inference is **FMv3ODE with drift-biased weights** — not
   drifting-augmented ODE integration.

Given this, the practical recommendation for Gen3v3:

| Config | What you get | When to use |
|---|---|---|
| `flow_steps=10, projection=on` | Drift-biased FM ODE + PCC snap (5 QP calls) | **Default — best quality** |
| `flow_steps=2, projection=on` | Minimal ODE + PCC (1 QP call) | Compute-constrained fallback |
| `flow_steps=1, projection=on` | Poor FM + heavy PCC correction | Not recommended |
| `flow_steps=10, projection=off` | Drift-biased FM ODE, no hard constraints | Ablation only |
| `flow_steps=1, projection=off` | Poor quality, no constraints | Avoid |

The only scenario where projection could be considered redundant is if the expert
demonstrations used to fill the drift memory bank were all constraint-satisfying AND the
drift embedding accurately captures the constraint-feasible submanifold — a strong
assumption that cannot be guaranteed without verification.

---

## 16. Is Strict Mathematical SLSQP Projection Possible for One-Shot / Original Drifting?

*Direct answer to: "is there strict math possible for drifting projection?"*

---

### 16.1 Short Answer

**Yes — post-processing SLSQP projection is always mathematically valid regardless of
how the trajectory was generated.** The projector is trajectory-agnostic: it takes any
`x ∈ ℝ^{H×d}` and solves a constrained optimisation to find the nearest feasible point.
It does not care whether `x` came from a DiT one-shot, an FM ODE, or a random sample.

The constraint is on the **trajectory space** (not the generation mechanism), so
projection is always applicable. The question is not *can* you project, but *how much
work* projection has to do and whether it can converge given where the trajectory landed.

---

### 16.2 Three Distinct Projection Modes for Drifting

#### Mode A — Post-Processing (original DiT one-shot)

```
z ~ N(0,I)  →  [DiT one-shot forward]  →  x_raw  →  [SLSQP]  →  x_constrained
```

- **Mathematically identical** to FM-D at ODE=1 with `threshold=1.0`
- One model call, one SLSQP solve
- The projector receives the raw one-shot output and finds the nearest feasible point
- No ODE to inject into — projection can only happen here
- **Valid. Fully supported by existing Projector code** (it's just called once at the end)

#### Mode B — In-Loop Projection (FM ODE, `flow_steps > 1`)

```
x_t → [FM step] → x_{t+dt} → [SLSQP snap] → x_{t+dt}^* → [FM step] → ...
```

- Projection fires inside the ODE loop at steps near `t=1` (threshold-controlled)
- Each projected intermediate trajectory **anchors the subsequent ODE step** to the
  feasible region — the next velocity evaluation starts from a constraint-satisfying point
- This is strictly stronger than post-processing because the ODE *continues* from the
  projected point, not from the raw point
- **Only possible with ODE-based generation (flow_steps ≥ 2)**
- The original drifting DiT cannot do this — there is no ODE loop

#### Mode C — Noise-Space Projection (gradient search through DiT)

```
z ~ N(0,I)  →  [DiT]  →  x  →  compute constraint violation
                                 →  ∇_z constraint_loss(DiT(z))
                                 →  z' = z - α∇_z  →  [DiT]  →  x'  →  repeat
```

- Backpropagates constraint gradients through the entire DiT to find noise `z*` such
  that `DiT(z*)` is constraint-satisfying
- **Theoretically possible** but practically very expensive: requires full DiT backward
  pass per iteration, and convergence is not guaranteed (DiT is highly non-linear)
- Not implemented in FM-PCC and not recommended — SLSQP on the trajectory space (Mode A)
  is far cheaper and more reliable
- **Not used anywhere in our codebase**

---

### 16.3 Why Mode B (In-Loop) Is Strictly Better Than Mode A (Post-Processing)

This is the mathematical key difference between ODE-based generation and one-shot:

**Mode A (one-shot + post-processing)**:

```
x_raw  →  SLSQP  →  x*
```

The SLSQP finds the closest feasible `x*` to `x_raw`. If `x_raw` is far from the
feasible manifold (bad trajectory quality), the displacement `‖x* - x_raw‖` is large.
Large displacements:
- Make SLSQP harder to converge (more iterations, possible local minima)
- Produce trajectories that satisfy constraints but may be unnatural (far from data)
- The trajectory is *feasible* but not necessarily *good*

**Mode B (in-loop at step k)**:

```
x_raw[k]  →  SLSQP  →  x*[k]  →  [FM velocity at x*[k]]  →  x_raw[k+1]
```

Each SLSQP snap at step `k` serves as the **starting point for the next ODE step**.
The FM velocity field at the projected (feasible) point `x*[k]` steers the trajectory
toward data *while respecting that it is starting from a feasible configuration*. This
means:
- SLSQP displacements are smaller at each step (incremental correction)
- The ODE integrator actively works *from within* the feasible region
- The final trajectory is both feasible AND closer to the data manifold

**Mathematical analogy**:
- Mode A = project a completed trajectory (like correcting a path after it's drawn)
- Mode B = draw the path step by step, correcting at each step (like a GPS rerouting continuously)

Mode B cannot exist without an ODE. This is the fundamental reason why FM/DDPM-based
generation is strictly more compatible with PCC than one-shot generation.

---

### 16.4 Drift + Post-Processing (Mode A): When Does It Work?

For the original drifting DiT (or FM-D at ODE=1), the quality of post-processing
projection depends on how close the raw output is to the feasible manifold:

**Best case** — experts were constraint-satisfying AND drift training worked well:
```
x_raw ≈ x_expert ≈ x_feasible   →   small SLSQP displacement   →   good projection
```

**Worst case** — experts violated constraints OR drift training failed:
```
x_raw far from feasibility   →   large SLSQP displacement   →   poor projection quality
                                                               (may not converge)
```

The drift training mechanism provides a probabilistic guarantee that `x_raw` is
near the expert distribution. If the expert distribution overlaps significantly with
the constraint-feasible set, drift implicitly pre-conditions the trajectory toward
feasibility before SLSQP acts. This is the one concrete scenario where Mode A (one-shot
+ projection) is competitive with Mode B (in-loop): **when drift has already done much
of the feasibility work**.

| Expert data | Drift effectiveness | Mode A result |
|---|---|---|
| All constraint-satisfying | High (experts = feasible) | SLSQP correction ≈ 0; projection trivial |
| Mixed (some violate) | Medium | Moderate SLSQP correction |
| All violating constraints | Low or counterproductive | Large SLSQP displacement; may fail to converge |

---

### 16.5 Practical Comparison: One-Shot + Projection vs ODE=10 + Projection

| Property | One-shot (ODE=1) + SLSQP | ODE=10 + SLSQP (threshold=0.5) |
|---|---|---|
| Model calls | 1 | 10 |
| QP solves | 1 | 5 |
| Total compute | Cheapest | 15× more |
| Trajectory quality before snap | Low (FM ODE=1 poor quality) | High (10-step ODE) |
| SLSQP displacement needed | Large | Small |
| Hard constraint guarantee | Yes | Yes |
| Trajectory naturalness after snap | Lower (large displacement distorts) | Higher (small displacement) |
| Applicable to original DiT drifting? | **Yes** | No (needs ODE) |
| Applicable to FM-D (ODE-based)? | Yes | **Yes — recommended** |

**Verdict**: One-shot + projection is mathematically valid and computationally cheapest,
but the trajectory naturalness after projection depends entirely on how good the
one-shot output is. For original drifting (with strong drift training), it can be
competitive. For FM-D's port (standard FM, weaker one-shot quality), ODE=10 with
in-loop projection remains superior.

---

### 16.6 Summary

| Question | Answer |
|---|---|
| Is SLSQP projection mathematically possible on one-shot output? | **Yes** — trajectory-agnostic, always valid |
| Is in-loop projection (Mode B) possible for original drifting DiT? | **No** — no ODE loop to inject into |
| Is Mode B strictly better than Mode A mathematically? | **Yes** — in-loop anchors ODE to feasible region incrementally |
| Can noise-space projection (Mode C) work? | Theoretically yes, practically impractical |
| When is one-shot + projection competitive? | When drift training ensures output is near-feasible |
| Recommended config for FM-D port? | ODE=10, in-loop projection (Mode B) |

---

## 17. Theoretical Evaluation of FM-D Philosophy

*A pure theoretical assessment — not how it works mechanically, but what it is
philosophically and whether the theory behind it is sound.*

---

### 17.1 What FM-D Is Philosophically Trying to Do

The original drifting paper's core philosophy is elegant:

> **Shape the learned representation so that the generator is naturally attracted to the
> expert distribution — without changing the generator architecture or inference procedure.**

In other words: rather than correcting bad samples at inference (what PCC does), drift
tries to make the model *incapable of generating* non-expert-like samples in the first
place. The drift loss is a regularizer on the training objective that adds geometric
structure to what the model learns. At inference nothing changes — the model simply
generates from its learned (biased) distribution.

This is philosophically distinct from PCC:

| Approach | Philosophy | When it acts | Guarantee |
|---|---|---|---|
| PCC | Correct bad samples after generation | Inference | Hard — SLSQP convergence |
| Drift (original) | Prevent bad samples from being generated | Training | Soft — distributional alignment |
| FM-D port | Both, but drift is training-only in practice | Training (+ optionally inference) | Soft + Hard |

The philosophy is sound. The question is whether the mathematical machinery used to
implement it actually achieves the stated goal.

---

### 17.2 Theoretical Problem 1: Perturbing the FM Velocity Field

Standard FM theory (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022) guarantees
that minimising the flow matching objective

```
L_FM = E_{t, x_0, x_1} [ ‖ v_θ(x_t, t) − (x_1 − x_0) ‖² ]
```

recovers the optimal transport velocity field, and that ODE integration under `v_θ`
transports the source distribution `p_0 = N(0,I)` to the target data distribution
`p_data`. This guarantee rests on the loss having a **unique minimum** at the true
conditional velocity.

Adding the drift regularizer gives:

```
L_total = L_FM + λ · L_drift
```

The new minimum `v_θ*` is **no longer the true conditional FM velocity**. It is a
compromise between:
- Fitting the exact OT velocity (FM term)
- Biasing the velocity toward the expert embedding neighbourhood (drift term)

**There is no formal theorem** characterising what distribution is learned under this
perturbed objective. The FM convergence guarantee is broken the moment `λ > 0`. In
practice the learned distribution is *approximately* `p_data` for small `λ`, but "small"
is undefined without empirical calibration.

**Verdict**: FM-D does not inherit FM's theoretical guarantees. The learned velocity field
is a regularised approximation of the true OT velocity, with unknown bias.

---

### 17.3 Theoretical Problem 2: The Embedding NN Proxy

The drift loss in our port is:

```
L_drift = -log( max_j softmax(-‖ encoder(x) − encoder(expert_j) ‖² / τ) )
```

This is a nearest-neighbour proxy in a **learned 128D embedding space**. Several issues:

**Confounded training signal**: The encoder is co-trained with the FM model. Its
embedding space adapts to make `L_drift` easier to minimise — not necessarily to capture
the true geometric structure of the expert distribution. The encoder may collapse
(mapping everything to a small region) or overfit to the training experts.

**Temperature collapse**: With `τ = 0.1`, the softmax is nearly argmax. The gradient of
`L_drift` is almost entirely determined by the single nearest expert. This means:
- The drift loss does not distribute signal uniformly across the expert distribution
- Small clusters of frequently-seen experts dominate the gradient signal
- Rare but constraint-satisfying experts are effectively ignored

**Not the expert distribution**: `L_drift` measures proximity to specific stored
embeddings, not proximity to the *distribution* `p_data`. Two trajectories at the same
distance from an expert embedding will receive the same gradient even if one is physically
implausible and the other is highly expert-like. The loss has no notion of the data
manifold's intrinsic geometry.

**Original force-field contrast**: The original drifting uses non-parametric kernel
matching (`R_list` multi-scale, symmetric softmax affinity). This does not suffer from
encoder confounding and measures distributional proximity more faithfully. Our port's
embedding NN loss is a significantly weaker proxy.

---

### 17.4 Theoretical Problem 3: "Looking Like Experts" ≠ "Constraint-Satisfying"

The drift regulariser pushes trajectories toward the expert distribution. But:

```
Expert distribution P_experts ⊄ Constraint-feasible set C  (in general)
```

Unless every demonstration in the training set was recorded with the robot operating
inside all constraints (bounds, halfspace, dynamics), the expert distribution will
partially overlap with the infeasible region. Drift training toward this distribution
therefore simultaneously pushes *toward* feasibility (for the constraint-satisfying
fraction of experts) *and toward* infeasibility (for the constraint-violating fraction).

The drift regulariser has no mechanism to distinguish between these two subsets of the
expert distribution. It is, in principle, learning the wrong target if experts are
unconstrained demonstrations.

**Implication**: PCC is not merely a "correction" for FM-D — it is the only component
that provides constraint information at all. The drift loss is blind to constraints by
design.

---

### 17.5 What the FM-D Port Actually Approximates (Honest Theory)

Stripping away the philosophical ambition, the FM-D port as implemented is:

> **A regularised FM model, where the regulariser biases the learned velocity field
> toward a learned nearest-neighbour proxy of the expert distribution in a 128D MLP
> embedding space, with the regulariser strength decaying from 0 to λ=0.1 over 1000
> training steps.**

The theoretical behaviour follows two regimes:

**Regime 1 — Drift training converges** (encoder captures expert geometry):

```
v_θ ≈ v_{OT} + small perturbation toward expert-distribution attractor
→ generated trajectories are near p_data AND near p_experts
→ PCC correction is small
→ combined quality: good
```

**Regime 2 — Drift training fails** (encoder collapses, `L_drift` becomes trivial):

```
v_θ ≈ v_{OT}   (drift term adds noise but no signal)
→ generated trajectories ≈ pure FM trajectories
→ FM-D reduces to FMv3ODE + PCC
→ combined quality: same as FMv3ODE
```

In Regime 2, FM-D is a strictly worse FMv3ODE (same quality, extra training overhead).
There is no regime where FM-D is theoretically *worse* than FMv3ODE — the worst case
is equivalence. The drift regulariser is a free bet: upside if it works, no downside
if it fails.

---

### 17.6 The Right Way to Think About FM-D + PCC Together

Given the theoretical murkiness of FM-D alone, the combined FM-D + PCC pipeline has a
cleaner interpretation:

```
FM (trained) → approximately samples from p_data
Drift (training regulariser) → approximately samples from p_data ∩ neighbourhood(p_experts)
PCC (inference projection) → exactly projects onto constraint manifold C
```

The full pipeline approximates sampling from:

```
p_data ∩ neighbourhood(p_experts) ∩ C
```

Each component handles a different property:
- FM: trajectory realism (looks like a plausible robot motion)
- Drift: expert alignment (looks like demonstrations)
- PCC: constraint satisfaction (geometrically feasible)

This decomposition is **philosophically sound** even if the individual FM-D theoretical
guarantees are weak. PCC is the hard-constraint enforcer that compensates for drift's
inability to guarantee feasibility.

The practical question — does drift training actually add value over pure FMv3ODE + PCC?
— is an empirical one that requires running both and comparing constraint satisfaction
rates and trajectory quality.

---

### 17.7 Final Verdict: FM-D Philosophy vs Practice

| Aspect | Philosophy | Theory | Practice |
|---|---|---|---|
| **Core idea** | Train model to "be" near experts, not just "look" near them | Sound intuition | Depends on drift quality |
| **FM loss perturbation** | Small regularisation | Breaks FM convergence guarantee | Minor in practice (λ=0.1 is small) |
| **Embedding NN proxy** | Distributional alignment | Weak proxy — confounded, temperature-collapsed | Unknown — no ablation |
| **Expert ≠ feasible** | Ignored | Fundamental gap | PCC compensates |
| **Original vs port** | Identical intent | Algorithmically divergent (D-1, D-2) | Our port ≈ FMv3ODE + biased weights |
| **Combined with PCC** | Principled decomposition | Each component covers different property | Theoretically best option available |
| **Worst case** | No harm | Reduces to FMv3ODE | Free bet — take it |
| **Best case** | Better data-aligned trajectories with easier PCC | Unproven | Empirically testable |

**One-sentence verdict**: FM-D is a philosophically motivated but theoretically under-
specified regulariser that, in the best case, reduces the PCC correction burden by
pre-aligning the FM trajectory distribution toward experts, and in the worst case
degrades silently to FMv3ODE — making it a low-risk, potentially high-value addition
to the FM-PCC stack, contingent on empirical validation of drift training convergence.

---

## 18. Can a Powerful Enough One-Shot DGM Replicate In-Loop Projection Behaviour?

*Addressing the core theoretical question: original drifting is a pure one-shot DGM.
The diffuser in-loop projection (Mode B) is structurally impossible for it. But can
a sufficiently powerful model approximate the same **outcome** without the mechanism?*

---

### 18.1 The Distinction: Mechanism vs Outcome

Mode B in-loop projection is **mechanistically impossible** for a one-shot model:

```
One-shot:  z → [Model] → x        (one forward pass, no ODE to inject into)
Mode B:    z → [Step 1] → [Snap] → [Step 2] → [Snap] → ... → x*
```

There is no loop, therefore there is no hook. This is not an engineering limitation —
it is a structural one. A one-shot model literally cannot execute Mode B.

**But Mode B's goal is an outcome, not a mechanism**:

```
Mode B outcome:   x* ∈ p_data ∩ C    (samples lie in data distribution AND feasible set)
```

The question is: can a one-shot model learn to produce samples directly from
`p_data ∩ C` without ever running a projector?

**Yes — in the limit of infinite capacity and perfect training data.**

If the model is trained exclusively on demonstrations that lie in `p_data ∩ C`, and is
powerful enough to learn the exact data distribution, it will generate samples from
`p_data ∩ C` by construction. No projector needed. Mode B outcome, achieved by Mode A
mechanism. This is what the original drifting paper implicitly aspires to.

---

### 18.2 What "Really Really Powerful" Means Formally

For a one-shot DGM to replicate in-loop projection behaviour, it must learn:

**1. The data distribution `p_data`** — standard DGM task. Tractable with enough capacity.

**2. The constraint manifold boundary ∂C** — this is the hard part.

The constraint manifold C is defined by hard geometric inequalities:

```
C = { x ∈ ℝ^{H×d} : lb ≤ c_pos[t] ≤ ub,  c_pos[t+1] = c_pos[t] + act[t],  ... }
```

C has **sharp boundaries** (step functions in probability space — density is zero
outside, non-zero inside). For the model to assign zero probability outside C, it must
learn this sharp boundary from training data alone.

This requires:
- **Dense sampling near ∂C**: training data must explore trajectories that approach but
  respect the constraint boundary. If experts never go near the walls, the model has
  no signal about where the boundary is.
- **Sharp density representation**: the model must represent a distribution with
  discontinuous support boundaries. Standard neural networks with smooth activations
  approximate this only asymptotically — the boundary is always blurred.
- **Generalization of constraint geometry across contexts**: the same constraint (e.g.
  workspace bound `x ≤ 0.70m`) must be respected for all box positions and targets,
  not just the ones seen during training.

Meeting all three simultaneously requires a model that is *not just large* but that
has somehow internalized the algebraic structure of the constraint set — which is
a geometric fact, not a statistical one.

---

### 18.3 The Fundamental Asymmetry

The in-loop projection decomposes the problem into two parts with very different
complexities:

```
FM (statistical):   learn v_θ ≈ v_{OT}   → model the data distribution
SLSQP (geometric):  solve a QP           → enforce constraints analytically
```

The SLSQP solver has **perfect knowledge of C** (it is given the constraint
equations explicitly). It does not need to learn the boundary from data. It solves
a deterministic constrained optimisation problem that always converges (under mild
conditions) to the correct projected point.

For a one-shot model to replicate this, it must learn the geometric structure of C
from data — essentially doing statistical approximation of what the SLSQP does
analytically. This is:

```
Diffuser + projector:   geometric constraints handled by exact QP solver
                        → zero error on constraint satisfaction (up to SLSQP tolerance)

One-shot DGM:           geometric constraints learned from data distribution
                        → nonzero error always, decreasing with model capacity and data
```

**There is no model capacity that makes statistical learning of geometric constraints
as reliable as analytical enforcement.** The SLSQP solves a convex QP to optimality;
no DGM can match this with certainty.

---

### 18.4 The "Really Really Powerful" Requirement — Quantified

How powerful does the one-shot model need to be?

Consider a simple workspace bound: `c_pos_x ∈ [0.30, 0.70]` over `H=8` timesteps.
The constraint boundary in trajectory space is a set of 16 hyperplanes (8 lower + 8
upper bounds). The model must learn to assign zero probability to the `O(H × d)`
half-spaces outside these hyperplanes.

For each hyperplane boundary, the model needs training data within `ε` of the boundary
to learn it with precision `ε`. If trajectories in the training set have typical margins
of `δ >> ε` from the constraint boundary (experts are "safe" and stay away from walls),
the model will place the learned boundary at `δ`, not at the true boundary.

The SLSQP projector knows the boundary is at exactly `x = 0.70`, regardless of `δ`.

To make `δ ≈ 0` (model learns the true boundary):
- Training data must densely explore the boundary region — requiring adversarial or
  constraint-aware data collection
- Model capacity must scale with the geometric complexity of C (number of constraints,
  their interactions, their joint effect over the horizon)
- Constraint geometry must be extrapolated to unseen task configurations

In our setting with bounds + halfspace + dynamics + obstacles over H=8, the constraint
set has `O(H × n_constraints)` boundary surfaces. This is a **much harder statistical
learning problem** than just learning `p_data`, and grows with the constraint complexity.

---

### 18.5 The Irony of Drifting + PCC

The original drifting philosophy was: bake constraint behaviour into the model so you
don't need an external projector. But:

1. Drift learns toward *expert distribution*, not toward *constraint set C*
2. Expert distribution ≠ constraint-feasible set (§17.4)
3. So drift does not even target the right objective for constraint satisfaction

The correct version of the original philosophy would require:
```
Training data: demonstrations that are BOTH expert-like AND constraint-satisfying
Drift loss: proximity to THIS filtered distribution
```

Even then, the statistical-learning-of-geometric-boundaries problem remains. The
one-shot model is still statistically approximating what SLSQP does analytically.

**The irony**: to fully replace SLSQP, the one-shot DGM needs the constraint
specification (lb, ub, halfspace equations) to be embedded in the training data
structure — which is essentially the same information the SLSQP uses directly. The
model is learning analytically-available geometric information from statistical signals,
which is always harder and less reliable.

---

### 18.6 Theoretical Summary

```
One-shot DGM (perfect, infinite capacity, constraint-satisfying training data)
    → outcome matches Mode B in-loop projection
    → requires: all training data ∈ C, model learns ∂C exactly, generalises perfectly
    → probability of achieving this in practice: effectively zero

Mode B in-loop projection (FM + SLSQP)
    → outcome: samples from p_data ∩ C, by construction
    → requires: FM learns p_data (standard), SLSQP enforces C (analytical, exact)
    → probability of achieving this in practice: high (SLSQP is deterministic)
```

| Property | Perfect One-Shot DGM | FM + In-Loop SLSQP |
|---|---|---|
| Constraint satisfaction | Statistical approximation of C | Exact (QP to tolerance) |
| Data distribution | Learned | Learned |
| Constraint boundary knowledge | Must be learned from data | Given explicitly to SLSQP |
| Scales with constraint complexity | Exponentially harder | Linearly harder (more QP variables) |
| Required model power | "Really really powerful" | Standard FM capacity |
| Practical achievability | Near-impossible | Standard practice |
| Boundary blurring | Always present | None |

**The conclusion the user identified is correct**: a one-shot DGM *can* theoretically
approach the behaviour of diffuser + in-loop projection, but requires model power and
data richness that scale with the algebraic complexity of the constraint set —
complexity that the SLSQP solver handles analytically and exactly for free.

The diffuser + projector decomposition is not just pragmatically easier — it is the
**correct architectural decomposition** that matches the problem structure:
statistical learning for the data distribution, analytical optimisation for the
geometric constraints. Trying to unify both in a single DGM conflates two
fundamentally different types of knowledge.
