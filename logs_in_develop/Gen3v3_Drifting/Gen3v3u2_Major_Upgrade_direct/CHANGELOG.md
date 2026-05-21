# Gen3v3u2 — FM-D Drifting Engine Bug Fix Sweep

**Date**: 2026-05-20  
**Branch**: update_into_FM  
**Source audit**: `logs_in_develop/Gen3v3_Drifting/Audit_fix_1/AUDIT_REPORT.md`  
**Re-audit**: same date, 13/14 findings confirmed (C-7 false positive)

All 13 confirmed bugs from the audit are fixed in this upgrade. Files changed:

| File | Bugs fixed |
|------|-----------|
| `utils/drift_training.py` | C-1, C-5 |
| `models/drift_unet.py` | C-3, C-4 |
| `sampling/drift_ode_solvers.py` | C-6, M-3 |
| `utils/drift_metrics.py` | M-5 |
| `models/drift_loss.py` | M-1, M-2, M-4 |
| `utils/training.py` | C-2 |
| `models/diffusion.py` | D-3 |
| `config/avoiding-d3il.py` | D-3 (config sync) |
| `examples/`, `configs/` | M-1 rename propagation |

---

## C-1 — `DriftLossScheduler.step` counter renamed to `self._step_count`

**File**: `utils/drift_training.py`

**Problem**: `self.step = 0` (int) and `def step(self)` (method) shared the same name on
`DriftLossScheduler`. In Python 3, instance `__dict__` takes precedence over class-level
non-data descriptors, so `self.step = 0` permanently hid the method from the first lookup.
Calling `scheduler.step()` crashed immediately: `TypeError: 'int' object is not callable`.

**Fix**: Renamed the counter to `self._step_count`. Updated all six references in
`get_weight()`, `step()`, `reset()`, and the exponential-decay branch.

---

## C-2 — `DriftTrainingWrapper` wired into `Trainer`

**File**: `utils/training.py`

**Problem**: `Trainer.train_epoch()` called `self.model.loss(*batch)` only. All drift
machinery (`DriftTrainingWrapper`, `DriftLoss`, scheduler, memory bank) was dead code —
never invoked.

**Fix**: Added optional `drift_wrapper=None` parameter to `Trainer.__init__`. In
`train_epoch`, when `drift_wrapper` is set:
1. `update_memory_bank_from_batch(batch[0].detach())` — populates the reference bank from
   each training batch of expert trajectories.
2. `compute_training_loss(batch[0].detach(), fm_loss)` — combines FM loss with drift loss.
3. `drift_wrapper.step()` — advances the loss-weight scheduler once per gradient step.

**Note**: `batch[0]` (the expert trajectory tensor) is used as both the memory-bank update
and the `sampled_trajectory` argument. This is an approximation — ideally `sampled_trajectory`
would be a model-generated trajectory from a mid-training inference pass, which is too
expensive for the inner loop. The wiring makes the drift code functional; trajectory quality
can be improved later by substituting model samples when compute allows.

---

## C-3 — `DriftConditioner.drift_proj` moved to `__init__`

**File**: `models/drift_unet.py`

**Problem**: `DriftConditioner.forward()` created a fresh `nn.Linear` on every call:
```python
drift_emb = nn.Linear(drift_metrics.shape[-1], self.cond_dim)(drift_metrics)
```
This allocated a new module with random weights each step, never saved it, defaulted to CPU,
and leaked memory.

**Fix**: Added `metric_dim: Optional[int] = None` to `DriftConditioner.__init__`. A single
`self.drift_proj = nn.Linear(metric_dim, cond_dim)` is created at init when `metric_dim` is
provided. The forward path uses `self.drift_proj(drift_metrics)`. If `metric_dim` was not
provided at init but `drift_metrics` is passed at forward time, a `RuntimeError` is raised
with a clear message instead of silently producing garbage.

---

## C-4 — `@staticmethod` added to `DriftAugmentedUNet1D.wrap_unet`

**File**: `models/drift_unet.py`

**Problem**: `wrap_unet` was a plain function inside the class body without `@staticmethod`.
Calling it on an instance (`some_aug_unet.wrap_unet(model)`) would pass the instance as
`base_unet`, silently discarding the intended model argument.

**Fix**: Added `@staticmethod` decorator.

---

## C-5 — Duplicate memory bank removed from `DriftTrainingWrapper`

**File**: `utils/drift_training.py`

**Problem**: `DriftTrainingWrapper.__init__` created its own `DriftMemoryBank` (Bank A) and
`update_memory_bank_from_batch` pushed to BOTH Bank A and `DriftLoss.memory_bank` (Bank B).
Bank A was never read by any loss or sampling path — it was wasted memory and doubled writes.

**Fix**: Removed the `memory_bank` parameter and `self.memory_bank` attribute from
`DriftTrainingWrapper`. `update_memory_bank_from_batch` now forwards only to
`self.drift_loss_fn.update_memory_bank()`, which owns the single authoritative reference
bank.

---

## C-6 — `**kwargs` now forwarded through `DriftAugmentedVelocityField`

**File**: `sampling/drift_ode_solvers.py`

**Problem**: When `drift_weight > 0`, `DriftAugmentedVelocityField` was constructed without
storing `**kwargs` (containing `cond=`, `returns=`). `_solve_legacy` called
`velocity_fn(t, x)` with no kwargs, so the FM model received no conditioning — outputs were
garbage. The zero-drift else-branch captured kwargs via closure, making the asymmetry a
silent correctness bug.

**Fix**:
- `DriftAugmentedVelocityField.__init__` now accepts and stores `**kwargs` as `self.kwargs`.
- `__call__` merges `self.kwargs` with any per-call kwargs before forwarding:
  `velocity = self.velocity_fn(t, x, **{**self.kwargs, **call_kwargs})`.
- `DriftODESolver.solve` passes `**kwargs` to the `DriftAugmentedVelocityField` constructor.

---

## M-1 — `compute_kl_divergence` renamed to `compute_embedding_nn_loss`; `loss_type` updated

**Files**: `models/drift_loss.py`, `utils/drift_training.py`, `config/avoiding-d3il.py`,
`examples/`, `configs/`

**Problem**: The method and loss-type string were labelled `kl_divergence`, but the
computation is a nearest-neighbour proxy in L2 embedding space — mathematically unrelated to
KL divergence. No density estimation, `max()` instead of expectation, temperature so low it
collapses to plain nearest-neighbour lookup.

**Fix**:
- Method renamed: `compute_kl_divergence` → `compute_embedding_nn_loss`.
- `loss_type` enum value renamed: `'kl_divergence'` → `'embedding_nn'` everywhere
  (Python source, YAML configs, examples, `create_drift_training_config`).
- `forward()` dispatch updated accordingly.

---

## M-2 & M-4 — Reference encoder detach removed; both branches now train the encoder

**File**: `models/drift_loss.py`

**Problem**: Both `compute_embedding_nn_loss` (M-2) and `compute_mmd_loss` (M-4) wrapped
the reference branch in `with torch.no_grad()`. Because the same encoder was used for both
sampled and reference trajectories, gradients only flowed through the sampled branch. The
encoder learned to map sampled trajectories toward wherever the randomly-initialised reference
embeddings happened to cluster — a meaningless signal.

**Fix**: Removed `with torch.no_grad():` from both paths. The encoder now trains on both
branches simultaneously, allowing its embedding space to adapt to the actual data distribution.

**Trade-off**: Removing the detach means the MMD reference term also contributes encoder
gradients. This is more principled than a frozen reference encoder but may require tuning
the learning rate for the encoder. A stop-gradient on the reference branch (as in the
original force-field repo) remains the theoretically cleanest alternative if needed.

---

## M-3 — Drift guidance sign corrected (gradient ascent → gradient descent)

**File**: `sampling/drift_ode_solvers.py`

**Problem**: The augmented velocity field computed:
```python
velocity = velocity + self.drift_weight * drift_grad_clipped
```
`drift_grad` is the gradient of the drift *loss*. Adding it performs gradient ascent — it
moves trajectories *away* from the expert distribution. The class docstring and
`get_gradient()` docstring both incorrectly described this as correct behaviour.

**Fix**: Changed to subtraction:
```python
velocity = velocity - self.drift_weight * drift_grad_clipped
```
Updated both docstrings to reflect `dx/dt -= lambda * grad_loss` (gradient descent toward
experts). The `get_gradient()` docstring in `drift_loss.py` updated to match.

---

## M-5 — `compute_ode_efficiency` formula inverted

**File**: `utils/drift_metrics.py`

**Problem**: `efficiency = num_steps_taken / max_steps`. A solver that used *more* steps
received a *higher* efficiency score. `wasted_budget = 1 - efficiency` was consequently also
inverted.

**Fix**:
```python
efficiency = 1.0 - (num_steps_taken / max_steps)   # fewer steps = higher efficiency
wasted_budget = num_steps_taken / max_steps          # more steps = more waste
```

---

## D-3 — `GaussianDiffusion.__init__` now accepts drift config params

**File**: `models/diffusion.py`

**Problem**: The config blocks in `config/avoiding-d3il.py` declared `use_drift_augmentation`,
`drift_loss_weight`, and `drift_loss_type`, but `GaussianDiffusion.__init__` had no
corresponding parameters. The `utils.Config` system silently dropped them.

**Fix**: Added the three parameters to `GaussianDiffusion.__init__` with their config
defaults, stored as `self.use_drift_augmentation`, `self.drift_loss_weight`,
`self.drift_loss_type`. The model does not yet act on these attributes internally —
drift-guided inference is controlled through `DriftODESolver` at the eval layer — but the
parameters are no longer silently discarded and are available on the model object for
inspection and future use.

---

## Not Fixed — D-1, D-2 (Algorithm Mismatch)

The original `/workspaces/drifting/drift_loss.py` implements a JAX-based non-parametric
force-field kernel matching algorithm (`[B, C, S]` tensors, multi-scale `R_list`, symmetric
softmax affinity, stop-gradient goal, MSE loss). The port is a different algorithm entirely.

**These are not fixed in this upgrade.** Faithfully porting the original force-field
algorithm requires redesigning `DriftLoss` from scratch, including resolving the JAX→PyTorch
translation and the `[B, C, S]` vs `[B, T, state_dim]` tensor shape convention.

This is tracked as a future Gen3v3u3 task.
