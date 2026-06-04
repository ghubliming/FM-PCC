# Research: Does Our iMF Have Classifier-Free Guidance (CFG)?

**Date**: 2026-06-04  
**Scope**: Gen3v4u2 iMeanFlow implementation vs. official iMeanFlow repo  
**Question**: The iMF paper uses CFG. Where is it in code? Do we have it?

---

## TL;DR — Executive Summary

| Aspect | Official iMF Repo | Our Gen3v4u2 Code | Status |
|--------|-------------------|-------------------|--------|
| **CFG in the paper?** | ✅ YES — core innovation | — | — |
| **CFG in the backbone model?** | ✅ YES — `omega`, `t_min`, `t_max` baked into DiT | ❌ NO — UNet1D has no CFG params | ⚠️ **MISSING** |
| **CFG in the inference loop?** | ✅ YES — `omega` passed at every step | ⚠️ PARTIAL — `condition_guidance_w` exists but **gated off** | ⚠️ **INACTIVE** |
| **CFG training (label dropout)?** | ✅ YES — `LabelEmbedder` with `num_classes+1` | ❌ NO — no class labels, no label dropout | ❌ **N/A** |
| **Does it matter for us?** | — | — | **Probably NOT yet** |

**Bottom Line**: We do **not** have the iMF paper's CFG mechanism. We have a *different*, inherited FM-PCC returns-based guidance mechanism that is **currently disabled** (`returns_condition=False`). This is **architecturally correct** for our domain (trajectory prediction) — CFG as designed in the paper is a class-conditional image generation technique that does not directly apply to our unconditional trajectory task.

---

## Part 1: What Does the iMF Paper Say About CFG?

### 1.1 The Paper's CFG Innovation

The iMF paper (arXiv:2512.02012, "Improved Mean Flows: On the Challenges of Fastforward Generative Models") identifies CFG flexibility as a **core contribution**:

> **Problem**: The original MeanFlow (MF) method fixed the CFG scale during training, which limited flexibility during inference.

> **Solution**: The authors formulated guidance as **explicit conditioning variables** — the model is conditioned on the CFG parameters `(omega, t_min, t_max)` as input tokens, allowing dynamic adjustment of the guidance scale at test time.

This means:
- **During training**: The model sees random CFG scales (omega) and intervals (t_min, t_max) as input conditioning
- **During inference**: You can sweep omega/interval to control sample quality vs diversity
- This is fundamentally different from "standard" CFG (train with label dropout, do two forward passes at inference)

### 1.2 How CFG Works in the Official iMF Code

The official repo at `/workspaces/imeanflow/` implements CFG through **in-context conditioning** — the CFG parameters are embedded as tokens prepended to the DiT sequence.

**Architecture** ([imfDiT.py](file:///workspaces/imeanflow/models/imfDiT.py)):

```python
# The DiT model takes CFG parameters as EXPLICIT INPUTS
class imfDiT(nn.Module):
    def __init__(self, ..., num_cfg_tokens=4, ...):
        # Separate embedders for each CFG parameter
        self.omega_embedder = TimestepEmbedder(...)      # CFG scale ω
        self.cfg_t_start_embedder = TimestepEmbedder(...) # interval start
        self.cfg_t_end_embedder = TimestepEmbedder(...)   # interval end
        self.y_embedder = LabelEmbedder(num_classes, ...) # class labels
        
        # Learnable tokens for each conditioning type
        self.omega_tokens = nn.Parameter(...)  # 4 tokens for CFG scale
        self.t_min_tokens = nn.Parameter(...)  # 2 tokens for interval start
        self.t_max_tokens = nn.Parameter(...)  # 2 tokens for interval end
        self.class_tokens = nn.Parameter(...)  # 8 tokens for class

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        # Embed all conditioning
        omega_embed = self.omega_embedder(1 - 1/w)   # ω → embedding
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)                   # class → embedding
        
        # Add embeddings to learnable tokens
        omega_tokens = self.omega_tokens + omega_embed
        t_min_tokens = self.t_min_tokens + t_min_embed
        t_max_tokens = self.t_max_tokens + t_max_embed
        class_tokens = self.class_tokens + y_embed
        
        # Prepend ALL conditioning tokens to patch sequence
        seq = torch.cat([
            class_tokens,     # 8 tokens
            omega_tokens,     # 4 tokens  ← CFG scale
            t_min_tokens,     # 2 tokens  ← CFG interval
            t_max_tokens,     # 2 tokens  ← CFG interval
            time_tokens,      # 4 tokens
            x_embed,          # N² patch tokens
        ], axis=1)
        return seq
```

**Inference** ([imf.py](file:///workspaces/imeanflow/imf.py)):

```python
class iMeanFlow(nn.Module):
    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """omega, t_min, t_max = CFG parameters passed at every step."""
        return self.net(x, t, h, omega, t_min, t_max, y)
    
    def generate(self, n_sample, rng, num_steps, omega, t_min, t_max, labels=None):
        """CFG parameters are constant across all sampling steps."""
        for i in range(num_steps):
            u = self.u_fn(z_t, t, h, omega, t_min, t_max, y=y)[0]
            z_t = z_t - h * u
```

**Evaluation** ([evaluate.py](file:///workspaces/imeanflow/evaluate.py)):

```bash
# CFG parameters are command-line arguments!
python evaluate.py evaluate \
    --cfg-omega 8.0 \           # ← CFG scale (default: 8.0)
    --interval-min 0.42 \       # ← CFG interval start
    --interval-max 0.62 \       # ← CFG interval end
```

### 1.3 Key Insight: iMF's CFG is NOT "Standard" CFG

Standard CFG (Ho & Salimans, 2022):
```
v_guided = v_uncond + ω * (v_cond - v_uncond)  # two forward passes
```

iMF's CFG:
```
v_guided = model(x, t, h, omega, t_min, t_max, y)  # ONE forward pass
# omega is an INPUT to the model, not a post-hoc scaling factor
```

The iMF paper embeds the guidance scale **inside the model** via in-context conditioning tokens. The model learns to produce guided outputs directly, without needing a second unconditional forward pass.

---

## Part 2: What Does Our Gen3v4u2 Code Have?

### 2.1 Our Backbone: UNet1D (No CFG Tokens)

Our backbone is `Flow_matcher_U_Net_v2` ([unet1d_temporal_cond.py](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py)), which has:

```python
class Flow_matcher_U_Net_v2(ModelMixin, ConfigMixin):
    def __init__(self, ..., returns_condition=False, condition_dropout=0.1, ...):
        self.time_mlp = ...      # ✅ time embedding
        self.h_mlp = ...         # ✅ h-conditioning (iMF)
        self.returns_mlp = ...   # ⚠️ returns-based guidance (DPCC heritage)
        self.mask_dist = ...     # ⚠️ Bernoulli mask for returns dropout
        
        # ❌ NO omega_embedder
        # ❌ NO cfg_t_start_embedder  
        # ❌ NO cfg_t_end_embedder
        # ❌ NO class_tokens / omega_tokens
```

**Missing from official iMF**: `omega`, `t_min`, `t_max`, class label embeddings. These are the core CFG mechanism from the paper.

### 2.2 Our Guidance: Returns-Conditional (Inherited from FM-PCC)

We have a **different** guidance mechanism inherited from the DPCC/FM-PCC lineage. It works on **return tokens** (reward signals), not class labels:

**In the UNet** ([unet1d_temporal_cond.py:213-221](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py#L213-L221)):

```python
def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False, h=None):
    t = self.time_mlp(timesteps)
    # ...h_mlp fusion...
    
    if self.returns_condition:
        returns_embed = self.returns_mlp(returns)
        if use_dropout:
            mask = self.mask_dist.sample(...)  # random dropout during training
            returns_embed = mask * returns_embed
        if force_dropout:
            returns_embed = 0 * returns_embed  # ← force unconditional pass
        t = torch.cat([t, returns_embed], dim=-1)
```

**In the diffusion wrapper** ([imf_diffusion.py:115-131](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/imf_diffusion.py#L115-L131)):

```python
def _predict_velocity(self, x, cond, t, h=None, returns=None):
    velocity, _aux = self._predict_uv(x, cond, t, h=h, returns=returns)
    
    # This IS standard CFG — two forward passes!
    if self.returns_condition and returns is not None and self.condition_guidance_w > 0:
        uncond_vel, _ = self._predict_uv(x, cond, t, h=h, returns=returns, force_dropout=True)
        velocity = (1 + self.condition_guidance_w) * velocity - self.condition_guidance_w * uncond_vel
    
    return velocity
```

### 2.3 But It's Disabled!

Every config entry in [avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py) sets:

```python
'returns_condition': False,    # ← DISABLED
'condition_guidance_w': 1.2,   # ← exists but never used
```

Because `returns_condition=False`:
1. The `returns_mlp` is **never created** (line 128-139 of unet1d_temporal_cond.py)
2. The `force_dropout` path is **never taken** in `_predict_velocity`
3. The `condition_guidance_w=1.2` parameter is **dead code**

---

## Part 3: Architectural Comparison

### 3.1 Side-by-Side: Three CFG Flavors

```
┌─────────────────────────────────────────────────────────────────────┐
│           OFFICIAL iMF (Image Generation)                          │
│                                                                     │
│  Input: x, t, h, omega, t_min, t_max, y (class label)             │
│  Model: DiT with in-context CFG tokens                             │
│  CFG:   omega baked INTO the model as input conditioning           │
│  Output: u, v (one forward pass with built-in guidance)            │
│  Notes: No two-pass CFG. Model learns guidance scale directly.     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│           OUR Gen3v4u2 (Trajectory Prediction) — POTENTIAL         │
│                                                                     │
│  Input: x, cond, t, h, returns (reward signal)                     │
│  Model: UNet1D with returns-embedding + dropout                    │
│  CFG:   Standard two-pass: v = v_uncond + w*(v_cond - v_uncond)    │
│  Output: u, v (guidance applied post-prediction)                   │
│  Notes: DISABLED (returns_condition=False). Dead code path.        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│           OUR Gen3v4u2 (Trajectory Prediction) — ACTUAL            │
│                                                                     │
│  Input: x, cond, t, h                                              │
│  Model: UNet1D (no returns, no CFG tokens, no class labels)        │
│  CFG:   NONE                                                       │
│  Output: u, v (raw predictions, no guidance applied)               │
│  Notes: Unconditional trajectory prediction only.                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Call Chain Comparison

**Official iMF — CFG is in every call:**
```
evaluate.py: --cfg-omega 8.0 --interval-min 0.4 --interval-max 0.65
  └─ iMeanFlow.generate(omega=8.0, t_min=0.4, t_max=0.65)
      └─ iMeanFlow.u_fn(x, t, h, omega, t_min, t_max, y)
          └─ imfDiT.forward(x, t, h, w=omega, t_min, t_max, y)
              └─ _build_sequence(x, h, w, t_min, t_max, y)
                  └─ omega_embedder(1 - 1/w) → omega_tokens
                  └─ cfg_t_start_embedder(t_min) → t_min_tokens
                  └─ cfg_t_end_embedder(t_max) → t_max_tokens
                  └─ y_embedder(y) → class_tokens
                  └─ cat([class, omega, t_min, t_max, time, patches])
```

**Our Gen3v4u2 — No CFG at all:**
```
iMeanFlowODE.p_sample_loop(shape, cond)
  └─ _predict_velocity(x, cond, t_const=0.5, h=dt)
      └─ _predict_uv(x, cond, t, h=h)  # returns_condition=False, so no CFG
          └─ iMeanFlowEngine.forward_train(x, t, h=h, cond=cond)
              └─ iMFTrajectoryModel.forward(x, t, h=h, cond=cond)
                  └─ Flow_matcher_U_Net_v2.forward(x, cond, t, h=h)
                      └─ time_mlp(t) + h_mlp(h)  # just time + h, no omega
```

---

## Part 4: Why We Don't Have CFG (And Why That's OK)

### 4.1 Domain Difference: Images vs. Trajectories

| Property | Official iMF (Images) | Our Gen3v4u2 (Trajectories) |
|----------|----------------------|----------------------------|
| **Task** | Class-conditional image generation | Unconditional trajectory prediction |
| **Conditioning** | Class labels (1000 ImageNet classes) | State conditioning (robot observation) |
| **CFG purpose** | Quality/diversity tradeoff | Not applicable — no class labels |
| **Backbone** | DiT (Diffusion Transformer) | UNet1D (convolutional) |
| **Token sequence** | Image patches + conditioning tokens | 1D trajectory sequence |

CFG in the iMF paper exists because:
1. The model is **class-conditional** (ImageNet has 1000 classes)
2. You need to balance **sample quality** (high CFG ω) vs **diversity** (low ω)
3. The paper's key contribution is making ω flexible at inference time

Our task is:
1. **Not class-conditional** — we don't have discrete classes
2. State-conditioned — the robot's current observation determines the trajectory
3. We don't have a quality/diversity tradeoff to tune

### 4.2 The `force_dropout` Parameter

You may notice `force_dropout` threaded through our code:

```python
# imf_trajectory_model.py:61
def forward(self, x, t, h=None, cond=None, force_dropout=False):
    velocity = self.velocity_net(x, cond, t, h=h, force_dropout=force_dropout)

# imf_engine.py:148,158
def forward_train(self, x_noisy, t, h=None, cond=None, force_dropout=False):
    """force_dropout: Force condition dropout for CFG"""
    return self.model(x_noisy, t, h=h, cond=cond, force_dropout=force_dropout)
```

This parameter is **plumbed but inert**. It reaches `Flow_matcher_U_Net_v2.forward()` where it would zero out `returns_embed` — but since `returns_condition=False`, the `returns_embed` doesn't exist, so `force_dropout` does nothing.

The plumbing exists as **forward compatibility** — if you later enable returns-based conditioning, the CFG infrastructure is ready.

### 4.3 The `condition_guidance_w` Parameter

```python
# Config: 'condition_guidance_w': 1.2
# Code: self.condition_guidance_w = condition_guidance_w  # stored

# In _predict_velocity:
if self.returns_condition and returns is not None and self.condition_guidance_w > 0:
    # ← This branch is NEVER entered because returns_condition=False
    uncond_vel, _ = self._predict_uv(..., force_dropout=True)
    velocity = (1 + self.condition_guidance_w) * velocity - self.condition_guidance_w * uncond_vel
```

The weight `1.2` is **dead code** — it's stored but the conditional gate (`self.returns_condition`) ensures it's never used.

---

## Part 5: What Would We Need to Add CFG?

If we ever wanted CFG-like behavior, there are two paths:

### Option A: iMF-Style In-Context CFG (Requires Retraining + Architectural Change)

```diff
  # Would need to add to UNet or switch to DiT:
+ self.omega_embedder = TimestepEmbedder(dim)
+ self.cfg_t_start_embedder = TimestepEmbedder(dim)
+ self.cfg_t_end_embedder = TimestepEmbedder(dim)
  
  # Would need to modify forward():
- t = self.time_mlp(timesteps) + self.h_mlp(h)
+ t = self.time_mlp(timesteps) + self.h_mlp(h) + self.omega_mlp(omega) + ...
```

**Pros**: Single forward pass, more efficient  
**Cons**: Requires retraining from scratch, unclear what "omega" means for trajectories

### Option B: Standard Two-Pass CFG (Already Partially Implemented)

```diff
  # In config:
- 'returns_condition': False,
+ 'returns_condition': True,   # ← enable
  'condition_guidance_w': 1.2, # ← now active

  # Would need return tokens from dataset
  # The existing code path would activate:
  # v = v_uncond + 1.2 * (v_cond - v_uncond)
```

**Pros**: No architectural change needed, just config + data  
**Cons**: Requires reward/return labels in dataset, doubles inference cost

### Option C: No CFG (Current — Probably Best)

For trajectory prediction on the D3IL avoiding task:
- The task is **deterministic** — there's one "best" trajectory given the state
- **Diversity is not desired** — we want the robot to avoid obstacles reliably
- The state conditioning via `cond` already provides sufficient guidance
- CFG's quality/diversity tradeoff is **irrelevant**

---

## Part 6: Evidence Map

### Files Where CFG Is Referenced

| File | What | Status |
|------|------|--------|
| [imf.py](file:///workspaces/imeanflow/imf.py) (official) | `omega` passed to `u_fn` at every step | ✅ Active in official repo |
| [imfDiT.py](file:///workspaces/imeanflow/models/imfDiT.py) (official) | `omega_embedder`, `cfg_t_start/end_embedder` as model inputs | ✅ Active in official repo |
| [evaluate.py](file:///workspaces/imeanflow/evaluate.py) (official) | `--cfg-omega`, `--interval-min`, `--interval-max` CLI args | ✅ Active in official repo |
| [embedder.py](file:///workspaces/imeanflow/models/embedder.py) (official) | `LabelEmbedder` with `num_classes+1` for dropout token | ✅ Active in official repo |
| [imf_diffusion.py](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/imf_diffusion.py) (ours) | `condition_guidance_w=0.1`, two-pass CFG in `_predict_velocity` | ⚠️ **Dead code** (gated by `returns_condition=False`) |
| [imf_engine.py](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/imf_engine.py) (ours) | `force_dropout` param threaded through | ⚠️ **Plumbed but inert** |
| [unet1d_temporal_cond.py](file:///workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py) (ours) | `force_dropout` zeros out `returns_embed` | ⚠️ **Would work IF returns_condition=True** |
| [avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py) (ours) | `'returns_condition': False`, `'condition_guidance_w': 1.2` | ❌ **Disabled** |

---

## Part 7: Conclusion & Recommendation

### Answer to "Do We Have CFG?"

**No.** We do not have the iMF paper's CFG mechanism. Specifically:

1. **The paper's CFG** (in-context omega/interval conditioning) is an **image-generation technique** deeply embedded in the DiT transformer architecture. Our UNet1D backbone has no equivalent.

2. **We have an inherited returns-based two-pass CFG** from the FM-PCC lineage, but it is **disabled** in all configs (`returns_condition=False`).

3. **This is correct for our domain.** Trajectory prediction on D3IL avoiding does not need class-conditional guidance. The state-based conditioning (`cond`) provides all necessary guidance for trajectory generation.

### Is This a Problem?

**No.** CFG is not a fundamental part of the iMF algorithm. It is an **orthogonal technique** for class-conditional generation quality. The core iMF innovations we DO have are:

- ✅ Mean flow training objective (u-target)
- ✅ h-conditioning for one-shot generation
- ✅ Dual velocity heads (u + v)
- ✅ Forward Euler sampling 0→1
- ✅ Zero-initialized aux head

These are the parts that matter for trajectory prediction with reduced inference steps.

### When Would CFG Matter?

If we ever move to:
- **Multi-modal trajectory generation** (multiple valid paths) → CFG could help select preferred modes
- **Reward-conditioned generation** → enable `returns_condition=True` and train with reward labels
- **Visual-conditioned generation with FiLM** → a form of guidance already exists through visual features

For now, **no action needed**.

---

*This document is part of the Gen3v4u2 study series. See [00_STUDY_GUIDE_Gen3v4u2.md](./00_STUDY_GUIDE_Gen3v4u2.md) for the main study guide.*
