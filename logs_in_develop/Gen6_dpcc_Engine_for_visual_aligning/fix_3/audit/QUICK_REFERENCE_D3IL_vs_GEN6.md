# QUICK REFERENCE: D3IL Baseline vs Gen6 - Trajectory Representations

## The One-Sentence Summary

**D3IL**: Diffuses on 3D actions conditioned on 7D states → generates 3D actions
**Gen6**: Diffuses on 6D [3D actions + 3D positions] jointly → generates 6D trajectories

---

## Trajectory Dimensionality at a Glance

### D3IL DDPM-ACT Baseline

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT THE MODEL GENERATES                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  pure_noise [B, 3]  ─┐                                      │
│                      │ Reverse diffusion (16 steps)         │
│  +conditioning ──────┤                                      │
│  (encoded obs)       │  Each step adds denoising            │
│                      │  prediction to get from x_t → x_0    │
│  Final output:       │                                      │
│                  ────▶  action_trajectory [B, 3]            │
│                         = [Δx, Δy, Δz] in m/s              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Shape Evolution in D3IL Training:
    Raw data input
    ├─ action: [B=4, T=8, 3]           ← What's being denoised
    ├─ state: [B=4, T=8, 7]            ← Encoding conditioning only
    │
    After scaling
    ├─ action_scaled: [B=4, T=8, 3]
    ├─ state_encoded: [B=4, T=8, 128]
    │
    In diffusion loss
    ├─ Input to model: action [B=4, T=8, 3]
    ├─ Condition: state_encoded [B=4, T=8, 128]
    └─ Noise prediction: [B=4, T=8, 3]

Z-axis unscaling (flat table):
    action_z_scaled = 0.5 (from diffusion)
    action_z_cmd = 0.5 × 0.0001 = 0.00005 m/s ✓ Correct
```

### Gen6 Visual-Aligning

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT THE MODEL GENERATES                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  pure_noise [B, T, 6]  ─┐                                   │
│                         │ Reverse diffusion (16 steps)      │
│  +constraints ──────────┤                                   │
│  (optional DPCC)        │  Each step:                       │
│                         │  1. Denoise 6D trajectory         │
│                         │  2. Optional: project to bounds   │
│                         │                                   │
│  Final output:      ────▶  trajectory [B, T, 6]            │
│                           = [Δx, Δy, Δz, x, y, z]          │
│                                                             │
│  Extract actions: trajectory[0:3]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Shape Evolution in Gen6 Training:
    Raw data input
    ├─ action: [B=4, T=8, 3]
    ├─ state_3d: [B=4, T=8, 3]  (subsample to EE only!)
    │
    Concatenate
    ├─ x = cat([action, state_3d], dim=-1): [B=4, T=8, 6]
    │
    After scaling
    ├─ x_scaled: [B=4, T=8, 6]
    │     ├─ dims [0:3]: actions (std_safe ≈ 0.01)
    │     └─ dims [3:6]: positions (std_safe ≈ 0.01) 🔴
    │
    After vision encoding
    ├─ obs_encoded: [B=4, T=8, 128]
    │
    In diffusion loss
    ├─ Input to model: x [B=4, T=8, 6] ← BOTH action AND state!
    ├─ Condition: {visual: ..., 0: init_state}
    └─ Noise prediction: [B=4, T=8, 6] ← Predicts noise for both!

Z-axis unscaling (flat table):
    action_z_scaled = 0.5 (from diffusion)
    action_z_cmd = 0.5 × 0.01 = 0.005 m/s 🔴 100x TOO LARGE!
```

---

## Key Differences Table

| **Aspect** | **D3IL Baseline** | **Gen6** | **Implication** |
|:---|:---|:---|:---|
| **What's diffused** | 3D actions only | **6D [actions + positions]** | Different problem space |
| **State role** | Conditioning (read-only) | **Joint variables** | Asymmetric vs symmetric |
| **Model input shape** | [B, T, 3] actions | **[B, T, 6] concatenated** | Different latent geometry |
| **Model output** | 3D noise predictions | **6D noise predictions** | Both action and state errors |
| **Loss function** | MSE on 3D actions | **MSE on 6D trajectory** | Different optimization target |
| **Scaler epsilon** | 1e-12 (minimal) | **1e-2 (large)** | 100x inflation for small stds |
| **Z-axis std handling** | 0.0001 → 0.0001 | **0.0 → 0.01 🔴** | Inflates noise for flat tasks |
| **Action unscaling** | ×0.0001 | **×0.01** | Magnifies diffusion errors 100x |
| **Clamping** | Both dimensions | **Actions only** | Asymmetric in Gen6 |
| **Projection** | None | **Optional DPCC** | Adds safety constraints |

---

## Exact Numbers: Flat-Table Z-Axis Example

### The Problem Scenario
- Robot manipulates objects on a flat table
- Vertical motion should be ~0
- Data: z_action_std = 0.0001, z_pos_std ≈ 0

### D3IL Processing
```
Scaler setup:
├─ y_std[z] = 0.0001 (from data)
├─ y_std_safe[z] = 0.0001 + 1e-12 ≈ 0.0001
└─ y_bounds[z] = [(0-mean)/std, (0-mean)/std] = [0, 0]

Inference:
├─ Diffusion outputs: z_scaled ∈ [-1, 1] range
├─ Clamp to bounds: z_scaled → clamp(z_scaled, 0, 0) = 0
├─ Unscale: z_cmd = 0 * 0.0001 + 0 = 0

Result: ✓ Always outputs ZERO Z-action
```

### Gen6 Processing
```
Scaler setup:
├─ x_std_raw = [0.1, 0.1, 0.0001, 0.05, 0.05, ~0.0]
├─ x_std_safe = max(x_std_raw, 1e-2)
│            = [0.1, 0.1, 0.01, 0.05, 0.05, 0.01] 🔴
│
│ Dims:     [Δx,  Δy,  Δz,     x,    y,    z]
│           [↑    ↑    🔴      ↑     ↑     🔴]
│           [good good inflated good good inflated]
│
└─ x_bounds[z-related] = [(0-mean)/0.01, (0-mean)/0.01] = [0, 0]

Inference:
├─ Diffusion outputs: z_scaled ∈ [-1, 1] range
├─ Clamp only actions [0:3]: z_action_scaled → clamp(z_action_scaled, -5, 5)
│                            (stays as is, no effective clamp)
│
├─ Unscale:
│   z_action_cmd = 0.5 * 0.01 + 0 = 0.005 m/s 🔴
│   z_pos = -0.3 * 0.01 + 0.6 = 0.597 m 🔴
│
└─ Projection (if enabled):
    QP solver sees z_pos = 0.597 (near table surface at 0.6)
    Corrects it back, creating feedback loop

Result: 🔴 Z-actions are inflated 100x, requiring projection to correct
```

---

## Mathematical Root Cause

### D3IL: Correct Formulation
$$\min_{a_{0:T}} \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \text{UNet}(x_t, t, s_c) \|^2 \right]$$

where:
- $x_t$ = noisy action trajectory (3D)
- $s_c$ = conditioning state (7D encoded)
- Noise prediction is 3D-only

**Result**: Model learns P(a | s, image)

### Gen6: Problematic Formulation
$$\min_{\tau_{0:T}} \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \text{VisualUNet}([\text{packed } \tau]_t, t, \text{img}) \|^2 \right]$$

where:
- $\tau = [a_{0:T}, s_{0:T}]$ packed into 6D space
- Noise prediction is 6D
- **Assumes $a$ and $s$ were jointly optimized** (FALSE: they're coupled by dynamics + vision encoding)

**Result**: Model learns P([a, s] | image) but projects as if independent

**The Bug**: Gen6 treats a coupled generative model as if it were independently optimized, then applies bounds and QP corrections that assume decoupling.

---

## File Locations for Reference

### D3IL Baseline
- **Scaler**: [d3il/agents/utils/scaler.py](../d3il/agents/utils/scaler.py) lines 1-150
- **Agent**: [d3il/agents/ddpm_vision_agent.py](../d3il/agents/ddpm_vision_agent.py)
- **Diffusion**: [d3il/agents/models/diffusion/diffusion_policy.py](../d3il/agents/models/diffusion/diffusion_policy.py)
- **Core Model**: [d3il/agents/models/diffusion/diffusion_models.py](../d3il/agents/models/diffusion/diffusion_models.py)

### Gen6 Visual-Aligning
- **Scaler**: [FM-PCC/ddpm_encdec_vision/utils/scaler.py](ddpm_encdec_vision/utils/scaler.py)
- **Diffusion**: [FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py](ddpm_encdec_vision/models/visual_gaussian_diffusion.py)
- **Evaluation**: [FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) lines 50-300
- **Projector Setup**: [FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) lines 63-130

---

## Which Document to Read for What

| **Question** | **Read This** |
|:---|:---|
| How does D3IL work end-to-end? | [D3IL_BASELINE_vs_GEN6_ANALYSIS.md](D3IL_BASELINE_vs_GEN6_ANALYSIS.md) Part 1 |
| How does Gen6 work? | [D3IL_BASELINE_vs_GEN6_ANALYSIS.md](D3IL_BASELINE_vs_GEN6_ANALYSIS.md) Part 2 |
| What are the exact tensor shapes? | [D3IL_vs_GEN6_CODE_TRACE.md](D3IL_vs_GEN6_CODE_TRACE.md) |
| Visual flowcharts and diagrams? | [D3IL_vs_GEN6_VISUAL_COMPARISON.md](D3IL_vs_GEN6_VISUAL_COMPARISON.md) |
| Line-by-line code walkthrough? | [D3IL_vs_GEN6_CODE_TRACE.md](D3IL_vs_GEN6_CODE_TRACE.md) Sections A & B |
| What's the mathematical difference? | [D3IL_BASELINE_vs_GEN6_ANALYSIS.md](D3IL_BASELINE_vs_GEN6_ANALYSIS.md) Part 4 |
| Why does Z-axis fail? | [D3IL_vs_GEN6_CODE_TRACE.md](D3IL_vs_GEN6_CODE_TRACE.md) Section C |
| Z-axis concrete numbers? | This file, "Exact Numbers" section |

---

## The Bottom Line

**D3IL is correct**: Diffusion on action space with state conditioning
- Scaler epsilon protects tiny values
- Z-actions stay close to 0 for flat tables
- No spurious vertical motion

**Gen6 has a fundamental mismatch**: Diffusion on 6D joint space with asymmetric projection
- Large epsilon inflates zero-variance dims
- Z-actions are 100x too large
- Projection must correct them iteratively
- Creates coupling between action/state denoising that wasn't in the training objective

**The fix**: Either:
1. Return to pure diffusion on actions (like D3IL), or
2. Redesign the loss to explicitly model the coupled dynamics: P(a | s, image) with s_{t+1} = s_t + a_t × dt
