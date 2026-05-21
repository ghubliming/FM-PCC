# Forensic Audit Report: iMeanFlow (iMF) Adaptation

**Date**: 2026-05-21
**Auditor**: Antigravity (Claude Opus 4.6 Thinking)
**Scope**: `flow_matcher_v3_imeanflow/` vs `flow_matcher_v3_ode_selectable/` vs `/workspaces/imeanflow/`
**Status**: Code runs and produces results. Audit is for correctness, not crash-fixing.

**Re-Audit**: 2026-05-21 — claude-sonnet-4-6, verified every finding against live code.
**Re-Audit result**: 21/22 confirmed. DEV-06 has one factual correction (`freq_dim` is NOT cosmetic). All other findings confirmed line-by-line.

---

## Executive Summary

The iMF adaptation is **functionally a re-skinned FMv3ODE** with a decorative auxiliary branch. The core iMeanFlow dual-velocity decomposition (u, v) from the official repo is **not implemented** — the `v` branch is trained to predict zero and adds negligible noise during sampling. The code runs because the underlying FMv3ODE math is preserved intact, but calling it "iMeanFlow" is misleading.

| Domain | Critical | Major | Minor |
|--------|----------|-------|-------|
| 1. Code Logic / Bugs | 1 | 4 | 3 |
| 2. ML Math Errors | 2 | 3 | 2 |
| 3. Deviations from Reference | 1 | 4 | 2 |

> **Re-Audit note**: All counts confirmed. DEV-06 (Minor) has one inaccuracy — `freq_dim` is effective, not cosmetic. See inline note.

---

## 1. Code Logic / Bugs

### BUG-01 · CRITICAL — `torchdiffeq` Backend Silently Ignored

**Files**: `imf_diffusion.py:130-186`

The config accepts `ode_solver_backend_v3='torchdiffeq'` and stores it, but `p_sample_loop` **always** uses legacy Euler. The entire torchdiffeq integration from the original (`diffusion.py:190-247`) is missing.

```python
# iMF: ALWAYS does this regardless of backend config
velocity = self._predict_velocity(x, cond, t_cont, returns=returns)
x = x + velocity * dt
```

> **CAUTION**: If a user sets `ode_solver_backend_v3='torchdiffeq'` with `method='dopri5'` in the plan config, the iMF variant will silently use Euler instead. Results will differ from FMv3ODE with no warning.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` line 68 stores `self.ode_solver_backend_v3` but `p_sample_loop` (lines 130–186) has no branch reading it — always executes `x = x + velocity * dt`. No `torchdiffeq` import exists in the file.

### BUG-02 · MAJOR — `loss_discount` Config Regression

**Files**: `train_flow_matching_v3_imeanflow.py:173`, `avoiding-d3il.py:437-494`

The iMF config section is **missing** `loss_discount`. The train script falls back:

```python
loss_discount=getattr(args, 'loss_discount', args.discount),  # falls to 0.99
```

The original FMv3ODE config has `loss_discount: 1.0` (uniform weighting across horizon). The iMF version silently uses `0.99` from `discount`, causing exponential decay on trajectory loss weights:

| Horizon step | FMv3ODE weight | iMF weight |
|:---:|:---:|:---:|
| 0 | 1.000 | 1.000 |
| 4 | 1.000 | 0.961 |
| 7 | 1.000 | 0.932 |

This subtly de-emphasizes later timesteps in the trajectory. Not catastrophic, but **unjustified divergence**.

> **Re-Audit**: CONFIRMED. `config/avoiding-d3il.py` lines 437–494 (the iMF block) contain no `loss_discount` key. `train_flow_matching_v3_imeanflow.py` line 173: `loss_discount=getattr(args, 'loss_discount', args.discount)` — falls to `discount=0.99`. `iMFDiffusion.__init__` default is `1.0` but this is overridden by the train-script fallback.

### BUG-03 · MAJOR — `gradient_accumulate_every` Defaults to 1

**Files**: `train_flow_matching_v3_imeanflow.py:194`

The iMF config omits `gradient_accumulate_every`. The train script defaults to 1:

```python
gradient_accumulate_every=getattr(args, 'gradient_accumulate_every', 1),
```

Original FMv3ODE uses `gradient_accumulate_every: 2`. Combined with different `batch_size` and `learning_rate`:

| Parameter | FMv3ODE | iMF |
|---|---|---|
| `batch_size` | 8 | 32 |
| `gradient_accumulate_every` | 2 | 1 (default) |
| `learning_rate` | 1e-4 | 5e-4 |
| **Effective tokens/step** | 16 | 32 |
| **Effective LR** | 1e-4 | 5e-4 |

The iMF trains with **5x higher learning rate** and **2x more data per step**. This may be intentional for faster convergence on the larger batch, but is undocumented.

> **Re-Audit**: CONFIRMED. iMF config (lines 437–494) has no `gradient_accumulate_every` key. `train_flow_matching_v3_imeanflow.py` line 194: `gradient_accumulate_every=getattr(args, 'gradient_accumulate_every', 1)` → defaults to 1. FMv3ODE config line 430 has `'gradient_accumulate_every': 2` confirmed.

### BUG-04 · MAJOR — Projection Costs Dropped

**File**: `imf_diffusion.py:185`

```python
infos['projection_costs'] = {}  # Always empty!
```

The original tracks per-step projection costs in a dict. The iMF version discards all cost data. Any downstream analysis that reads `projection_costs` (e.g., DPCC-C trajectory selection via `minimum_projection_cost`) will receive empty data.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` line 185: `infos['projection_costs'] = {}` — exact match. No cost accumulation loop exists anywhere in `p_sample_loop`.

### BUG-05 · MAJOR — `returns_condition` Flag Inconsistency

**Files**: `train_flow_matching_v3_imeanflow.py:174`, `imf_diffusion.py:112-115`

The train script passes `returns_condition=args.include_returns` (True), but `_predict_uv` **silently drops** the returns parameter:

```python
def _predict_uv(self, x, cond, t, returns=None):
    # Returns-conditioning is intentionally ignored here
    return self.model.forward_train(x, t, cond)
```

Meanwhile the backbone UNet is constructed with `returns_condition=False` (in `imf_trajectory_model.py:42`). So `self.returns_condition=True` is stored in the wrapper but never acted on. **Classifier-free guidance is dead code**.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` lines 112–115: `_predict_uv` accepts `returns` but calls `self.model.forward_train(x, t, cond)` — `returns` is not forwarded. `imf_trajectory_model.py` line 42: `returns_condition=False` hardcoded in UNet constructor. Train script line 174: `returns_condition=args.include_returns` → True, creating the mismatch.

### BUG-06 · MINOR — Missing API Surface Methods

**File**: `imf_diffusion.py`

The following methods from `GaussianDiffusion` are missing: `predict_start_from_noise`, `q_posterior`, `p_mean_variance`, `p_sample`, `grad_p_sample`, `grad_p_sample_loop`, `grad_conditional_sample`, `_time_from_timestep`. If any external code calls these (e.g., visualization tools), it will crash with `AttributeError`.

> **Re-Audit**: CONFIRMED. All eight methods exist in `flow_matcher_v3_imeanflow/models/diffusion.py` (GaussianDiffusion) but are absent from `imf_diffusion.py`. `class iMFDiffusion(nn.Module)` — inherits only from `nn.Module`, not from `GaussianDiffusion`, so no inheritance path provides them.

### BUG-07 · MINOR — `returns_scale` Divergence (Original Has a Bug)

**Files**: `train_flow_matching_v3_ode_selectable.py:239`, `train_flow_matching_v3_imeanflow.py:139`

```python
# Original (likely a bug):
returns_scale=args.max_path_length,  # 150

# iMF (correct):
returns_scale=args.returns_scale,     # 400
```

The original accidentally uses `max_path_length` (150) instead of `returns_scale` (400). The iMF version "fixes" this. However this means models trained under each are using different return normalizations, making checkpoint cross-loading semantically wrong.

> **Re-Audit**: CONFIRMED. `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py` line 239: `returns_scale=args.max_path_length` (150). `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` line 139: `returns_scale=args.returns_scale` (400). The divergence is real and the original bug is still unfixed in its own script.

### BUG-08 · MINOR — Mutable `flow_steps_v3` Side Effect

**File**: `imf_diffusion.py:205-207`

```python
def sample(self, ..., num_steps=None):
    if num_steps is not None:
        self.flow_steps_v3 = int(num_steps)  # Mutates model state!
```

This permanently changes the model's step count for all subsequent calls. Not a crash bug, but violates the principle of least surprise.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` lines 205–207: `if num_steps is not None: self.flow_steps_v3 = int(num_steps); self.ode_inference_steps_v3 = int(num_steps)` — both attributes are mutated. Both are subsequently read by `p_sample_loop`.

---

## 2. ML / Math Errors

### MATH-01 · CRITICAL — Auxiliary Branch Trained Against Zero (iMF Decomposition Absent)

**Files**: `imf_diffusion.py:252`, `imf_trajectory_model.py:46-53`

The defining mathematical contribution of iMeanFlow is the **dual velocity decomposition**:

```
dx_t/dt = u(x_t, t) + v(x_t, t)
```

where `u` is the *mean velocity field* and `v` is the *instantaneous deviation*. The official repo trains both branches with meaningful targets.

In the adaptation:

```python
# Training (imf_diffusion.py:252)
aux_loss = F.mse_loss(aux_pred, torch.zeros_like(aux_pred))  # Target is ZERO

# Sampling (imf_diffusion.py:119)
return velocity + self.sample_aux_weight * aux  # weight ~ 0.009
```

- The aux head is trained to output **zero**
- Zero-initialized weights (`imf_trajectory_model.py:52-53`) + zero target = the head stays near zero forever
- During sampling, `0.009 * ~0 ~ 0` is added

> **WARNING**: **The model is mathematically identical to FMv3ODE.** The aux branch has no learning signal, no meaningful target, and negligible sampling weight. The "iMF" label is a misnomer — this is standard flow matching with extra dead parameters.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` line 252: `aux_loss = F.mse_loss(aux_pred, torch.zeros_like(aux_pred))` — exact match. `imf_trajectory_model.py` lines 52–53: `nn.init.zeros_(self.aux_head[-1].weight); nn.init.zeros_(self.aux_head[-1].bias)`. `imf_diffusion.py` line 85: `self.sample_aux_weight = 0.1 * self.v_mix` → with default weights = `0.1 * (0.1/1.1) ≈ 0.009`. All three conditions verified.

### MATH-02 · CRITICAL — `aux_head` Architecture Violates iMF Design

**Files**: `imf_trajectory_model.py:46-64` vs `/workspaces/imeanflow/models/imfDiT.py:264-295`

| Property | Official iMF | Adaptation |
|---|---|---|
| Architecture | Shared backbone -> parallel u/v heads (8 transformer blocks each) | Single UNet -> tiny 2-layer MLP |
| Input to v-head | Shared intermediate features | **Output of u-head** (velocity) |
| v-head capacity | 8 transformer blocks (~millions of params) | 2 linear layers (~state_dim^2 params) |
| Independence | u and v are computed in parallel | `aux = aux_head(velocity)` — serial dependency |

The serial dependency means `aux` is a deterministic function of `velocity`, not an independent prediction from the input. This is architecturally incompatible with the iMF paper.

> **Re-Audit**: CONFIRMED. `imf_trajectory_model.py` line 63: `aux = self.aux_head(velocity)` — serial; `aux` is computed from `velocity` output, not from `x` independently. Official `imfDiT.py`: dual heads share the transformer backbone and both receive the input features in parallel. The aux_head here is a 2-layer MLP with `~2 * state_dim^2` parameters vs the official's 8 transformer blocks.

### MATH-03 · MAJOR — Integration Direction Conflict in Standalone Samplers

**Files**: `imf_engine.py:118` vs `imf_diffusion.py:150-158`

```python
# iMFEngine.sample (matches official iMF: t goes 1->0)
t_steps = torch.linspace(1.0, 0.0, num_steps + 1)
z_t = z_t - h * velocity  # h = t - r > 0

# iMFDiffusion.p_sample_loop (matches FMv3ODE: t goes 0->1)
t_cont = loop_idx / max(self.flow_steps_v3, 1)  # 0->1
x = x + velocity * dt
```

The model is trained with the FMv3ODE convention (t=0 is noise, t=1 is data, forward integration). If anyone calls `iMFEngine.sample()` directly, it integrates in the **reverse direction**, producing garbage. The method exists, is exported, but would give wrong results.

> **Re-Audit**: CONFIRMED. `imf_engine.py` line 118: `t_steps = torch.linspace(1.0, 0.0, num_steps + 1, ...)` and line 136: `z_t = z_t - h * velocity` where `h = t - r > 0` — explicit 1→0 reverse integration. `imf_diffusion.py` line 152: `t_cont = loop_idx / max(self.flow_steps_v3, 1)` (0→1) and line 158: `x = x + velocity * dt` — forward integration. Directions confirmed opposite.

### MATH-04 · MAJOR — Initial Noise Scale Mismatch in Standalone Sampler

**File**: `imf_engine.py:125` vs `imf_diffusion.py:142`

```python
# iMFEngine.sample:
z_t = torch.randn(...)          # sigma = 1.0

# iMFDiffusion.p_sample_loop (and original FMv3ODE):
x = 0.5 * torch.randn(...)      # sigma = 0.5
```

Training uses `0.5 * randn` as the base noise. The standalone sampler uses `1.0 * randn`. Distribution mismatch leading to degraded sample quality from the standalone path.

> **Re-Audit**: CONFIRMED. `imf_engine.py` line 125: `z_t = torch.randn(batch_size, self.seq_len, self.state_dim, ...)` — sigma 1.0. `imf_diffusion.py` line 142: `x = 0.5 * torch.randn(shape, device=device)` — sigma 0.5. Also applies to `iMFTrajectoryModel.sample_trajectory()` line 88: `z_t = torch.randn(...)` — sigma 1.0, same mismatch.

### MATH-05 · MAJOR — Step-Size Conditioning (`h`) Missing

**Files**: `/workspaces/imeanflow/imf.py:44-69` vs `imf_trajectory_model.py:55-63`

Official iMF conditions the model on `h = t - r` (step size):

```python
# Official: u_fn receives h as a separate input
self.net(x, t, h, omega, t_min, t_max, y)
```

The adaptation does NOT condition on step size:

```python
# Adaptation: only x, t, cond
velocity = self.velocity_net(x, cond, t)
```

In the official iMF, `h`-conditioning is what enables **one-step generation** — the model learns to adapt its prediction based on how large the integration step will be. Without it, the model is a standard flow matcher that requires multiple steps.

> **Re-Audit**: CONFIRMED with clarification. `imf_engine.py` `u_fn` signature (line 72): `def u_fn(self, x, t, h=None, cond=None)` — `h` parameter exists but line 76: `return self.model(x, t, cond)` — `h` is silently dropped, never forwarded. `iMFTrajectoryModel.forward` (line 56): `def forward(self, x, t, cond=None)` — no `h` parameter at all. Confirmed: `h` is accepted by the engine API surface but never reaches the neural network.

### MATH-06 · MINOR — Classifier-Free Guidance Completely Stripped

The official iMF has `omega`, `t_min`, `t_max` for CFG interval scheduling. All removed. Not an error per se (the FM-PCC task may not need CFG), but eliminates a core iMF capability.

> **Re-Audit**: CONFIRMED. `/workspaces/imeanflow/imf.py` line 44: `u_fn(self, x, t, h, omega, t_min, t_max, y)` — all four CFG params present. Adaptation `imf_engine.py` line 72: `u_fn(self, x, t, h=None, cond=None)` — all removed.

### MATH-07 · MINOR — `v_mix` / `u_mix` Weighting Has No Effect

**File**: `imf_diffusion.py:78-84`

```
total_w = 1.0 + 0.1 = 1.1
u_mix = 1.0 / 1.1 ~ 0.909
v_mix = 0.1 / 1.1 ~ 0.091
sample_aux_weight = 0.1 * 0.091 ~ 0.009
aux_loss_weight = max(0.01, 0.1 * 0.1) = 0.01
```

These weights are computed but since aux is trained against zero, they control the magnitude of nothing. The `u_mix`/`v_mix` are logged in `info` but never used for actual loss weighting — `main_loss` gets weight 1.0, `aux_loss` gets `0.01`. The normalized mix values are cosmetic.

> **Re-Audit**: CONFIRMED. `imf_diffusion.py` line 251: `main_loss, info = self.loss_fn(velocity_pred, v_target)` — `main_loss` is unscaled (weight 1.0). Line 253: `total_loss = main_loss + self.aux_loss_weight * aux_loss` — `u_mix` (≈0.909) is never applied to `main_loss`. Lines 258–259: `info['u_weight']` and `info['v_weight']` are set purely for logging, not for loss scaling.

---

## 3. Deviations from Official iMeanFlow Repo

### DEV-01 · CRITICAL — iMF Dual-Velocity is Not Implemented

As established in MATH-01 and MATH-02, the adaptation does **not** implement iMeanFlow. It implements FMv3ODE with:
- A zero-initialized, zero-targeted, negligibly-weighted MLP appendage
- iMF naming conventions applied to FMv3ODE concepts

| Official iMF Concept | Adaptation Reality |
|---|---|
| u = mean velocity field | velocity = standard FM velocity |
| v = instantaneous deviation | aux = zero-regularized MLP(velocity) |
| Shared backbone -> dual heads | Single UNet -> serial MLP |
| One-step generation via h-conditioning | Multi-step Euler (no h-conditioning) |
| CFG with interval scheduling | No CFG |

> **Re-Audit**: CONFIRMED. All five rows verified against both codebases. Established by MATH-01, MATH-02, MATH-05, MATH-06 findings.

### DEV-02 · MAJOR — DiT Replaced with UNet (No Justification)

Official iMF uses `imfDiT` — a modern transformer with RoPE, RMSNorm, SwiGLU, and vector-gated residuals. The adaptation uses `Flow_matcher_U_Net_v2` — a conv-based 1D U-Net. This is a valid engineering choice for trajectory data, but:

- The UNet is **reused verbatim** from FMv3ODE
- No architectural adaptation was made for iMF
- The "depth", "num_heads", "mlp_dim" config params create the iMFTrajectoryModel wrapper but the UNet inside ignores most of them

> **Re-Audit**: CONFIRMED. `imf_trajectory_model.py` lines 36–44: UNet constructed with `dim=freq_dim, dim_mults=(1,2,4,8)` only. `depth`, `num_heads`, `mlp_dim`, `time_dim` are stored as instance attributes but never forwarded to the UNet constructor. Note: `dropout_rate` IS forwarded as `condition_dropout` (see DEV-07).

### DEV-03 · MAJOR — `iMFTrainingLoss` is Dead Code

**File**: `imf_losses.py`

This entire 95-line module is **never called** by any code path. Loss computation happens entirely inside `iMFDiffusion.p_losses`. The class is exported in `__init__.py` but unused. It contains a `compute_losses` method with a proper dual-loss formula that is more faithful to iMF than what `p_losses` actually does.

> **Re-Audit**: CONFIRMED. `grep -rn "iMFTrainingLoss\|imf_losses"` in both the engine folder and train test folder: only hits are the class definition and `__init__.py` export. Never imported by `imf_diffusion.py` or the train script. The `forward()` compatibility method (line 82) sets `v_target = torch.zeros_like(target_trajectory)` — confirming the zero-target pattern is intentionally carried throughout, but even this is dead.

### DEV-04 · MAJOR — `iMeanFlowEngine.sample()` Uses Wrong Convention

As shown in MATH-03, the standalone `sample()` method copies the official iMF's 1->0 integration direction, but the model is trained with FMv3ODE's 0->1 convention. This method would produce incorrect results if called. It should either be removed or fixed to match the training convention.

> **Re-Audit**: CONFIRMED. `imf_engine.py` lines 117–138 verified. Also note: the method is decorated `@torch.no_grad()` (line 87) and is exported publicly, making it the most likely entry point for external callers — who would silently get garbage results.

### DEV-05 · MAJOR — `iMFTrajectoryModel.sample_trajectory()` is Redundant Dead Code

**File**: `imf_trajectory_model.py:74-106`

This method duplicates `iMFEngine.sample()` with a slightly different interface. Neither is ever called by the actual pipeline. Both use the wrong integration direction. Both are dead code.

> **Re-Audit**: CONFIRMED. `imf_trajectory_model.py` lines 74–106 (`sample_trajectory`) and 108–137 (`sample`) both use `z_t = z_t - h * combined` (1→0 direction). Neither is invoked from `p_sample_loop`, the train script, or any eval script. The `sample()` method on `iMFTrajectoryModel` is shadowed/duplicated by `iMeanFlowEngine.sample()` (which also uses the wrong direction).

### DEV-06 · MINOR — Config Has iMF-Specific Params That Do Nothing

**File**: `avoiding-d3il.py:448-460`

```python
'freq_dim': 256,     # Only affects iMFTrajectoryModel wrapper, not UNet internals
'depth': 8,          # Ignored by UNet
'num_heads': 4,      # Ignored by UNet
'mlp_dim': 256,      # Ignored by UNet
'time_dim': 256,     # Ignored by UNet
```

These parameters are passed to `iMFTrajectoryModel.__init__` and stored, but the actual backbone (`Flow_matcher_U_Net_v2`) uses its own `dim=256`, `dim_mults=(1,2,4,8)` hardcoded in the constructor at `imf_trajectory_model.py:36-44`. Changing `depth` or `num_heads` in config has **zero effect** on the model.

> **Re-Audit**: **PARTIALLY INACCURATE — `freq_dim` correction.** `imf_trajectory_model.py` line 40: `dim=freq_dim` — `freq_dim` IS forwarded to the UNet as its feature width. Changing `freq_dim` in config changes the model size. The label `# Only affects iMFTrajectoryModel wrapper, not UNet internals` is wrong for `freq_dim`. The four genuinely cosmetic/unused params are `depth`, `num_heads`, `mlp_dim`, `time_dim` — none are passed to the UNet constructor. `dropout_rate` is also effective (see DEV-07). Corrected table:
>
> | Config param | Reaches UNet? | Effect |
> |---|---|---|
> | `freq_dim` | YES → `dim` | Changes UNet feature width |
> | `dropout_rate` | YES → `condition_dropout` | Changes condition dropout |
> | `depth` | No | Ignored |
> | `num_heads` | No | Ignored |
> | `mlp_dim` | No | Ignored |
> | `time_dim` | No | Ignored |

### DEV-07 · MINOR — `dropout_rate` Mapped to `condition_dropout`

**File**: `imf_trajectory_model.py:43`

The config's `dropout_rate: 0.1` is passed to the UNet as `condition_dropout`. In the original FMv3ODE config, `condition_dropout: 0.25`. So the iMF version uses lower condition dropout (0.1 vs 0.25), reducing regularization strength.

> **Re-Audit**: CONFIRMED. `imf_trajectory_model.py` line 43: `condition_dropout=dropout_rate` — correctly forwarded. Config line 453: `'dropout_rate': 0.1`. FMv3ODE config has `'condition_dropout': 0.25` (line 474 in the avoiding-d3il block). The discrepancy is real.

---

## Summary: What Actually Happened

The adaptation is a **working, stable flow matching model**. It produces good results because it IS FMv3ODE under the hood — the same UNet, the same training objective, the same ODE sampling. The iMF wrapper adds ~`state_dim^2` dead parameters and negligible compute overhead but no mathematical benefit.

The "weak AI" that wrote this made a pragmatic decision: rather than implementing the full iMF architecture (shared backbone, dual heads, h-conditioning, proper u/v training targets), it wrapped FMv3ODE in iMF-named classes and added a vestigial aux head. This works because FMv3ODE is already a solid flow matching implementation, but the iMF-specific capabilities (one-step generation, dual velocity field) are absent.

---

## Remediation Priority

| Priority | Issue | Fix | Re-Audit |
|---|---|---|---|
| **P0** | BUG-01: torchdiffeq silently ignored | Port the torchdiffeq block from original `p_sample_loop` | ✓ Confirmed |
| **P0** | MATH-01/DEV-01: iMF not implemented | Decision needed: implement real iMF or rename to "FMv3ODE-aux" | ✓ Confirmed |
| **P1** | BUG-02: `loss_discount` missing | Add `'loss_discount': 1.0` to iMF config | ✓ Confirmed |
| **P1** | BUG-03: `gradient_accumulate_every` wrong default | Add `'gradient_accumulate_every': 2` to iMF config | ✓ Confirmed |
| **P1** | BUG-04: projection costs dropped | Port cost tracking from original | ✓ Confirmed |
| **P2** | MATH-03/04: standalone samplers wrong | Delete or fix `iMFEngine.sample()` and `iMFTrajectoryModel.sample_trajectory()` | ✓ Confirmed |
| **P2** | DEV-03: dead `iMFTrainingLoss` | Delete `imf_losses.py` or integrate it | ✓ Confirmed |
| **P3** | BUG-05: `returns_condition` flag | Set `returns_condition=False` explicitly in config | ✓ Confirmed |
| **P3** | DEV-06: fake config params | Document which params are cosmetic — **note `freq_dim` IS effective** | ⚠ Partially inaccurate |

**Re-Audit summary**: 21/22 findings confirmed. Only DEV-06 has a factual error: `freq_dim` maps directly to the UNet `dim` parameter and is NOT cosmetic. The four genuinely inert params are `depth`, `num_heads`, `mlp_dim`, `time_dim`. All other 21 findings are verified line-by-line against live code.

---

## Implementation Plan (Gen3v4u1)

**Decision**: Implement real iMF. Keep current name `flow_matcher_v3_imeanflow`. Use DATA-AT-1 convention throughout (noise at t=0, data at t=1), matching FMv3ODE's `q_sample = (1-t)*noise + t*data`. Sampling integrates forward 0→1 in `p_sample_loop`. All fixes apply the mean flow math on top of the existing backbone without changing the ODE convention.

### Training Objective (MATH-01, MATH-02, MATH-05, MATH-07)

Replace the fake `p_losses` with a real iMF mean-flow objective:

1. Sample noise `x_base ~ N(0, I)` (sigma=1.0, fixing MATH-04 for the training distribution)
2. For each sample, draw `r ~ Uniform(0, t)` so that `h = t - r > 0`
3. Compute interpolants: `x_t = (1-t)*x_base + t*x_start`, `x_r = (1-r)*x_base + r*x_start`
4. Mean flow target: `u_target = (x_start - x_r) / h` — the expected velocity that would carry `x_r` to `x_start` in time `h`
5. FM velocity target for aux: `v_target = x_start - x_base` (real target, not zero!)
6. Query model with `h` conditioning: `(u_pred, v_pred) = model(x_t, t, h=h, cond=cond)`
7. Loss: `u_mix * MSE(u_pred, u_target) + aux_weight * MSE(v_pred, v_target)` (apply u_mix — fixes MATH-07)
8. Fix aux_head architecture: compute `aux = self.aux_head(x_t)` instead of `self.aux_head(velocity)` so aux is an independent head (fixes MATH-02)

### h-Conditioning Architecture (MATH-05)

Add `h`-embedding fused into the time embedding in `Flow_matcher_U_Net_v2`:
- Add `self.h_mlp` with same architecture as `self.time_mlp` (SinusoidalPosEmb → Linear → Mish → Linear)
- In `forward`: if `h` is provided, broadcast to `[batch]`, embed via `h_mlp`, fuse: `t = t + h_mlp(h)`
- Thread `h=None` parameter through: UNet → iMFTrajectoryModel → iMeanFlowEngine → iMFDiffusion

### Sampler Fixes (MATH-03, MATH-04, BUG-08)

- `p_sample_loop`: change initial noise sigma from 0.5 to 1.0 (training noise is sigma=1.0); pass `h = dt` as a batch tensor to `_predict_velocity`
- `iMeanFlowEngine.sample()`: change from 1→0 to 0→1 direction (`t_steps = linspace(0,1,...)`; use `z += h * u` not `z -= h * u`); sigma=1.0; pass `h` to model
- `iMFTrajectoryModel.sample_trajectory()`: same 0→1 fixes
- `iMFDiffusion.sample()`: remove mutation of `self.flow_steps_v3`; pass `num_steps` through `conditional_sample → p_sample_loop` without modifying state (fixes BUG-08)

### Config Fixes (BUG-02, BUG-03)

Add to `config/avoiding-d3il.py` → `flow_matching_v3_imeanflow` block:
```python
'loss_discount': 1.0,
'gradient_accumulate_every': 2,
```

### Returns / CFG (BUG-05, MATH-06)

Thread `force_dropout` and `returns` through the full call chain so the infrastructure is ready. The UNet backbone currently has `returns_condition=False` hardcoded, so CFG has no effect in this upgrade — but the plumbing is in place for a future upgrade that enables returns conditioning.

### Not Fixed in This Upgrade

- **BUG-01** (torchdiffeq silently ignored): Porting the torchdiffeq branch requires careful adaptation of the velocity function signature with h-conditioning into torchdiffeq's ODE interface. Deferred to Gen3v4u2.
- **BUG-04** (projection costs dropped): Requires a well-defined projector cost interface. Deferred.
- **DEV-03** (dead `iMFTrainingLoss`): Left in place to avoid breaking `__init__.py` exports. The class remains exported but unused.
- **DEV-02** (DiT vs UNet): Architecture choice, not a bug. Out of scope.
