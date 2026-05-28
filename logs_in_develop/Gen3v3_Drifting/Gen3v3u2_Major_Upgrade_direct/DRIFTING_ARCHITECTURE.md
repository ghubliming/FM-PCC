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
