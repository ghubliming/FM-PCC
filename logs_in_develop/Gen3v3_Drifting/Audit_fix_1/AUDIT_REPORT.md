# FM-D (Drifting) Engine — Deep Technical Audit

**Date**: 2026-05-20  
**Auditor**: Opus (Manual Code Review)  
**Scope**: `flow_matcher_v3_drifting/` vs `flow_matcher_v3_ode_selectable/` + `/workspaces/drifting/drift_loss.py`  
**Status**: SIGNIFICANT FINDINGS

**Re-Audit**: 2026-05-20 — claude-sonnet-4-6, verified every finding against live code.  
**Re-Audit result**: 13/14 confirmed. C-7 is a false finding. Two findings (C-1, C-4) have corrected mechanism descriptions.

---

## Executive Summary

The drifting upgrade adds 5 new files (~1,200 lines) on top of a byte-identical copy of the original FMv3ODE engine. The core FM training/inference pipeline (`diffusion.py`, `policies.py`, `projection.py`, `training.py`) is **untouched and correct**. However, the new drift-specific modules contain **critical bugs, math errors, and fundamental deviations** from the original `/workspaces/drifting` repo that undermine the theoretical validity of the drift augmentation.

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Code Logic / Bugs | 1 | 2 | 3 | 1 |
| Math / ML Errors | 2 | 2 | 1 | 0 |
| Derivation from Original | 1 | 1 | 1 | 0 |

> **Re-Audit note**: C-7 (Low) is a false finding — see detail below. Effective count: 1 Critical, 2 High, 3 Medium, 0 Low in Code Logic.

> **CAUTION**: The drift modules are **dead code** — they are never invoked by the actual training loop (`training.py` Trainer class). The training pipeline runs pure FM loss only. See Finding C-2.

---

## Part 1: Code Logic / Bugs

### C-1 CRITICAL — `DriftLossScheduler.step` name shadows `self.step` (int) with method

**File**: `flow_matcher_v3_drifting/utils/drift_training.py` lines 47-51

```python
class DriftLossScheduler:
    def __init__(self, ...):
        self.step = 0          # <-- int attribute

    def step(self) -> None:    # <-- method with SAME NAME
        """Advance scheduler by one step."""
        self.step += 1         # <-- tries to do int += 1 on a method -> TypeError
```

**Impact**: Calling `scheduler.step()` works exactly once (Python binds the method name over the int), then `self.step` becomes the bound method. The second call to `scheduler.step()` crashes with `TypeError: unsupported operand type(s) for +=: 'method' and 'int'`. `get_weight()` also breaks because it compares a method to an int.

**Severity**: CRITICAL — crashes on first use.

**Fix**: Rename the counter to `self._step_count` or rename the method to `advance()`.

> **Re-Audit**: CONFIRMED. Python mechanism description above is wrong, but the bug is real. In Python 3, instance `__dict__` takes precedence over non-data descriptors (regular methods). `self.step = 0` in `__init__` immediately shadows the class-level `step` method. `scheduler.step()` therefore crashes on the **very first call** with `TypeError: 'int' object is not callable` — it does not work even once. The method body is unreachable. Fix is identical: rename counter to `self._step_count`.

---

### C-2 HIGH — `DriftTrainingWrapper` is never wired into the Trainer

**File**: `flow_matcher_v3_drifting/utils/training.py` lines 118-188

The `Trainer.train_epoch()` method calls `self.model.loss(*batch)` then does `loss.backward()`. It **never** instantiates or calls `DriftTrainingWrapper`, `DriftLoss`, or any drift-related code. The config params `use_drift_augmentation`, `drift_loss_weight`, `drift_loss_type` are defined in `avoiding-d3il.py` but **never read** by any code path.

Despite fix_1.md claiming "Updated the Trainer to check for `use_drift_augmentation`", the actual `training.py` contains **zero references** to drift loss.

**Impact**: The entire drift augmentation is dead code. Training runs pure FM.

**Severity**: HIGH — feature is non-functional.

> **Re-Audit**: CONFIRMED. `grep` for `drift` in `utils/training.py` returns zero hits. `train_epoch()` at line 126 calls only `self.model.loss(*batch)`. `GaussianDiffusion.__init__` likewise has no `use_drift_augmentation`, `drift_loss_weight`, or `drift_loss_type` parameters (verified against `models/diffusion.py` line 24).

---

### C-3 HIGH — `DriftConditioner` creates `nn.Linear` on every forward pass

**File**: `flow_matcher_v3_drifting/models/drift_unet.py` lines 80-83

```python
def forward(self, trajectory, drift_metrics=None):
    ...
    if drift_metrics is not None:
        drift_emb = nn.Linear(drift_metrics.shape[-1], self.cond_dim)(drift_metrics)
        cond = cond + 0.1 * drift_emb
```

**Impact**: 
1. A fresh `nn.Linear` with random weights is allocated per forward pass — drift_metrics conditioning is random noise
2. The linear is never registered as a submodule — not saved/loaded with checkpoints
3. Memory leak: garbage collector must collect a new module each step
4. Device mismatch: the ad-hoc Linear defaults to CPU even if inputs are on CUDA

**Fix**: Create `self.drift_proj = nn.Linear(expected_metric_dim, cond_dim)` in `__init__`.

> **Re-Audit**: CONFIRMED. `drift_unet.py` line 82 matches exactly. All four impact points verified.

---

### C-4 MEDIUM — `wrap_unet` missing `@staticmethod`

**File**: `flow_matcher_v3_drifting/models/drift_unet.py` lines 197-208

```python
def wrap_unet(base_unet: nn.Module, **kwargs) -> "DriftAugmentedUNet1D":
```

This is a factory method but lacks `@staticmethod`. Calling `DriftAugmentedUNet1D.wrap_unet(model)` passes `self` as `base_unet`, which is wrong.

> **Re-Audit**: CONFIRMED — but failure trigger is wrong. In Python 3, calling on the **class** (`DriftAugmentedUNet1D.wrap_unet(model)`) works correctly — `base_unet` receives `model`. The bug fires when called on an **instance** (`some_instance.wrap_unet(model)`), which passes the instance as `base_unet` and `model` is lost. Fix is the same: add `@staticmethod`.

---

### C-5 MEDIUM — Duplicate memory bank between `DriftLoss` and `DriftMemoryBank`

**Files**: `drift_loss.py` lines 47-52 and `drift_training.py` lines 80-156

`DriftLoss` has its own `register_buffer('memory_bank', ...)` AND `DriftTrainingWrapper` has a separate `DriftMemoryBank`. The wrapper's `update_memory_bank_from_batch` pushes to BOTH:

```python
self.memory_bank.push(expert_trajectories)          # bank A
self.drift_loss_fn.update_memory_bank(expert_trajectories)  # bank B
```

Bank A is never read by any loss computation. It's wasted memory.

> **Re-Audit**: CONFIRMED. `DriftTrainingWrapper.compute_training_loss()` calls `self.drift_loss_fn.forward()` which reads only Bank B internally. `DriftMemoryBank.sample()` / `get_all()` are never called from any loss path.

---

### C-6 MEDIUM — `DriftODESolver.solve` kwargs leak

**File**: `flow_matcher_v3_drifting/sampling/drift_ode_solvers.py` lines 130-153

When `drift_weight > 0`, the solver creates `DriftAugmentedVelocityField` but the `**kwargs` (containing `cond=`, `returns=`) are NOT passed to the augmented field's `__call__`. They are swallowed:

```python
augmented_fn = DriftAugmentedVelocityField(velocity_fn, ...)  # kwargs not forwarded
```

But when `drift_weight == 0`, a lambda is created that DOES capture kwargs:

```python
def augmented_fn(t, x):
    return velocity_fn(t, x, **kwargs)  # kwargs captured
```

**Impact**: With drift enabled, the velocity function receives no conditioning — model outputs garbage.

> **Re-Audit**: CONFIRMED. `drift_ode_solvers.py` lines 134-142 match exactly. `DriftAugmentedVelocityField.__call__` accepts `**kwargs` but `_solve_legacy` calls it as `velocity_fn(t_tensor, x)` with no kwargs, so `cond=` and `returns=` are never forwarded. The zero-drift else-branch closure captures them correctly, making the asymmetry explicit.

---

### C-7 LOW — `DriftMemoryBank.push` off-by-one on wrap-around ~~[FALSE FINDING — see re-audit note]~~

**File**: `drift_training.py` lines 115-127

When `ptr + B == max_size` exactly, the code sets `ptr += B` (line 117) then hits the check on line 125: `if self.ptr == self.max_size: self.ptr = 0; self.full = True`. This works but the `self.full` flag is set one push late compared to the equivalent logic in `DriftLoss.update_memory_bank`.

> **Re-Audit**: **FALSE FINDING.** Both implementations set `full = True` after the identical push that first fills the buffer to capacity. `DriftMemoryBank` uses a post-write check (line 125) for the exact-fill case; `DriftLoss.update_memory_bank` integrates it inline (`if ptr + B == self.memory_bank_size`). The observable result — `full = True` is set once, after exactly `max_size` elements — is the same in both. No off-by-one exists.

---

## Part 2: Math / ML Errors

### M-1 CRITICAL — "KL Divergence" is not KL divergence

**File**: `flow_matcher_v3_drifting/models/drift_loss.py` lines 110-157

The `compute_kl_divergence` method computes:

```
KL = -log(max_j softmax(-||q_z_i - p_z_j|| / tau))
```

This is **not KL divergence** in any standard sense. It is:
1. A nearest-neighbor proxy using L2 distance in a learned embedding space
2. Applied with `softmax(-dist/tau)` which produces a distribution over reference samples
3. Then takes `-log(max prob)` which is a negative log-likelihood of the closest match

**Mathematical problems**:
- True KL divergence `D_KL(Q || P)` requires density estimation of both Q and P. This computes neither.
- The `max()` over softmax probabilities is not an expectation and does not satisfy KL properties (non-negativity when Q=P only holds approximately).
- Temperature `tau=0.1` makes the softmax extremely peaked, effectively reducing to nearest-neighbor L2 distance in embedding space, making the entire encoder/softmax machinery redundant.

**Impact**: The loss has a valid gradient signal (it pushes sampled trajectories toward expert embeddings), but calling it "KL divergence" is mathematically incorrect and the gradient dynamics are dominated by the encoder's representation quality, which is never properly trained (see M-2).

> **Re-Audit**: CONFIRMED. `drift_loss.py` lines 143-155 match the described formula exactly. Non-negativity at Q=P does not hold, no density estimation is performed, and `tau=0.1` confirms effective nearest-neighbor collapse.

---

### M-2 CRITICAL — Reference encoder is detached — never learns

**File**: `flow_matcher_v3_drifting/models/drift_loss.py` lines 142-144

```python
with torch.no_grad():
    p_z = self.encoder(ref_trajs)  # Reference trajectories encoded WITHOUT gradients
```

The same encoder is used for both sampled (with grad) and reference (without grad) trajectories. Since gradients only flow through the sampled branch, the encoder learns to map sampled trajectories close to wherever the (fixed, randomly-initialized) reference embeddings happen to be. 

**Problems**:
1. At initialization, the encoder outputs random 128-dim vectors with LayerNorm — all reference embeddings cluster near the unit sphere
2. The loss gradient pushes sampled embeddings toward this random cluster — no meaningful distributional matching
3. The encoder is never updated via the reference branch — no contrastive signal

**Correct approach**: Either use a **separate** pretrained encoder, or use a **non-parametric** kernel method (like the original drifting repo does).

> **Re-Audit**: CONFIRMED. `drift_loss.py` lines 143-144 `with torch.no_grad(): p_z = self.encoder(ref_trajs)` confirmed. Same encoder used for both branches; reference branch permanently frozen.

---

### M-3 HIGH — Gradient sign convention is wrong for drift guidance

**File**: `flow_matcher_v3_drifting/sampling/drift_ode_solvers.py` lines 44-68

The augmented velocity field computes:
```
v(x,t) = model_velocity(x,t) + lambda * grad_drift_loss(x)
```

But `grad_drift_loss` is the gradient of the **loss** (distance to expert distribution). To **minimize** the loss (move toward experts), you need **negative** gradient (gradient descent). But the code **adds** the gradient (gradient ascent), which pushes trajectories **away** from the expert distribution.

The docstring on line 309 says `dx/dt += lambda * grad_loss` and calls it "gradient ascent on trajectory quality", but quality is inversely related to loss. This is backwards.

**Fix**: Change to `velocity = velocity - self.drift_weight * drift_grad_clipped`

> **Re-Audit**: CONFIRMED. `drift_ode_solvers.py` line 67: `velocity = velocity + self.drift_weight * drift_grad_clipped`. The `get_gradient()` docstring in `drift_loss.py` line 309 also says `dx/dt += lambda * grad_loss`, confirming the wrong sign is baked into the design intent, not just an off-by-one typo.

---

### M-4 HIGH — MMD loss uses detached reference encoder (same issue as M-2)

**File**: `drift_loss.py` lines 188-190

Same problem as M-2: `p_z = self.encoder(ref_trajs)` is under `torch.no_grad()`. The MMD kernel operates in a representation space that never adapts to the reference distribution.

Additionally, the RBF kernel bandwidth `sigma=1.0` is hardcoded and not adapted to the data scale. After LayerNorm, embeddings have unit variance, so sigma=1.0 may be reasonable, but there's no bandwidth selection (median heuristic etc.).

> **Re-Audit**: CONFIRMED. `drift_loss.py` lines 188-190 `with torch.no_grad(): p_z = self.encoder(ref_trajs)` confirmed in the MMD path. Same structural problem as M-2.

---

### M-5 MEDIUM — `compute_ode_efficiency` metrics are inverted

**File**: `drift_metrics.py` lines 184-204

```python
efficiency = num_steps_taken / max_steps
```

A solver that takes MORE steps gets a HIGHER "efficiency" score. This is semantically inverted — efficiency should be `max_steps / num_steps_taken` or `1 - (num_steps_taken / max_steps)`.

> **Re-Audit**: CONFIRMED. `drift_metrics.py` line 198 matches exactly. Note also that `wasted_budget = 1.0 - efficiency` is therefore also inverted (a faster solver shows lower wasted_budget when it should show higher).

---

## Part 3: Derivation from Original Repo

### D-1 CRITICAL — Fundamental algorithm replacement (not a port)

**Original** (`/workspaces/drifting/drift_loss.py`): A JAX-based **force-field kernel matching** loss:
- Computes pairwise distances between generated and reference samples at multiple kernel scales R
- Uses symmetric softmax affinity: `sqrt(softmax(logits, axis=-1) * softmax(logits, axis=-2))`
- Separates positive (attract) and negative (repel) forces
- Normalizes force magnitude per temperature scale
- Computes a **goal position** via stop-gradient, then regresses generated samples toward it
- Loss = `mean((gen_scaled - goal_scaled)^2)` — a simple MSE in scaled space

**Port** (`flow_matcher_v3_drifting/models/drift_loss.py`): A PyTorch **learned encoder + distribution matching** loss:
- Encodes trajectories through a learned MLP encoder to 128-dim embeddings
- Computes pairwise L2 distances in embedding space
- Applies softmax, takes max probability, negative log
- No force-field dynamics, no multi-scale kernels, no positive/negative sample separation

**These are completely different algorithms.** The original drifting loss:
1. Is **non-parametric** (no learned encoder)
2. Uses **multi-scale kernel** matching (R_list = [0.02, 0.05, 0.2])
3. Has **force-field dynamics** with attraction/repulsion
4. Uses **stop-gradient goal computation** for stable training
5. Supports **weighted samples** and **negative examples**

None of these properties are preserved in the port. The port is a new, different algorithm that happens to be called "drift loss".

> **Re-Audit**: CONFIRMED. Both files read in full. `/workspaces/drifting/drift_loss.py` is JAX, 135 lines, entirely non-parametric force-field with `cdist`, symmetric `sqrt(softmax * softmax^T)`, multi-scale `R_list`, stop-gradient goal, MSE loss. `flow_matcher_v3_drifting/models/drift_loss.py` is PyTorch, 326 lines, learned MLP encoder, L2 embedding distance, softmax nearest-neighbor. Zero algorithmic overlap beyond the name.

---

### D-2 HIGH — Original operates on `[B, C, S]` (channels x spatial), port operates on `[B, T*state_dim]` (flattened)

The original drifting repo processes image-like data with shape `[B, C, S]` where C is channels and S is spatial resolution. The distance function `cdist` computes per-batch pairwise distances across channels.

The port flattens trajectory `[B, T, state_dim]` to `[B, T*state_dim]` and treats the entire flattened vector as a single point. This loses the temporal structure and makes the distance metric dominated by trajectory length T rather than per-step quality.

> **Re-Audit**: CONFIRMED. Original signature `gen: [B, C_g, S]` verified. Port `DriftLoss.__init__` takes `trajectory_dim: int` (the flattened scalar) and `DriftMemoryBank` stores `(max_size, trajectory_dim)` flat tensors.

---

### D-3 MEDIUM — Config declares drift params, but GaussianDiffusion.__init__ doesn't accept them

**File**: `config/avoiding-d3il.py` lines 383-435

The `flow_matching_v3_drifting` config block defines:
```python
'use_drift_augmentation': True,
'drift_loss_weight': 0.1,
'drift_loss_type': 'kl_divergence',
```

But `GaussianDiffusion.__init__` (identical in both repos) does NOT accept these kwargs. They are silently ignored by the Parser/Config system because `diffusion_config` only passes recognized kwargs.

> **Re-Audit**: CONFIRMED. `GaussianDiffusion.__init__` signature at `models/diffusion.py` line 24 has no `use_drift_augmentation`, `drift_loss_weight`, or `drift_loss_type` parameters. The three config keys at `config/avoiding-d3il.py` lines 405-407 are silently dropped.

---

## Summary of Findings

### Inherited Code (CLEAN)

The following files are **byte-identical** to the proven FMv3ODE original and contain no issues:

| File | Status |
|------|--------|
| `models/diffusion.py` | Identical |
| `sampling/policies.py` | Identical |
| `sampling/projection.py` | Identical |
| `utils/training.py` | Identical |
| `models/helpers.py` | Identical (assumed) |
| `datasets/*` | Identical (assumed) |

### New Drift Code (ISSUES FOUND)

| File | Lines | Issues |
|------|-------|--------|
| `models/drift_loss.py` | 326 | M-1, M-2, C-5 |
| `models/drift_unet.py` | 209 | C-3, C-4 |
| `sampling/drift_ode_solvers.py` | 280 | C-6, M-3 |
| `utils/drift_training.py` | 302 | C-1, C-2, C-5 ~~C-7~~ |
| `utils/drift_metrics.py` | 310 | M-5 |
| Train/eval scripts | ~730 | Clean (import rename only) |
| Config block | ~50 | D-3 |

> **Re-Audit summary**: 13/14 findings confirmed. **C-7 is a false finding** — both `DriftMemoryBank` and `DriftLoss.update_memory_bank` set `full=True` after the same push. **C-1** mechanism corrected: crashes on 1st call, not 2nd. **C-4** trigger corrected: breaks on instance access, not class access. All other findings verified line-by-line against live code.

### Why It "Runs Good and Returns Results"

The system produces valid results because:
1. **The drift code is dead** (C-2) — training runs pure FM loss, identical to FMv3ODE
2. **Inference uses the base `p_sample_loop`** — the drift ODE solvers are never invoked during eval
3. The train/eval scripts differ from the original by only two import lines and two experiment name strings

The results ARE the FMv3ODE results. The drifting augmentation has zero effect.

---

## Recommended Fix Priority

| Priority | Finding | Fix |
|----------|---------|-----|
| 1 | D-1 | Re-implement drift loss as faithful port of original force-field algorithm |
| 2 | C-1 | Rename `self.step` counter to `self._step_count` |
| 3 | C-2 | Wire `DriftTrainingWrapper` into `Trainer.train_epoch()` |
| 4 | M-3 | Fix gradient sign: subtract drift gradient, not add |
| 5 | C-6 | Forward `**kwargs` through `DriftAugmentedVelocityField` |
| 6 | C-3 | Move `nn.Linear` to `__init__` with proper registration |
| 7 | M-2 | Either remove learned encoder (use non-parametric kernel) or train both branches |
| 8 | C-4 | Add `@staticmethod` to `wrap_unet` |
| 9 | M-5 | Invert efficiency metric |
| 10 | D-3 | Add drift config params to `GaussianDiffusion.__init__` or create `DriftGaussianDiffusion` subclass |

---

**Report Generated**: 2026-05-20  
**Verdict**: The drift augmentation is non-functional dead code. The underlying FM engine is sound.
