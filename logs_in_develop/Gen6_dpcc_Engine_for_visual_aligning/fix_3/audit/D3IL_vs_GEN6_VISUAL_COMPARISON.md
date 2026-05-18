# Visual Architecture Comparison: D3IL vs Gen6

## System Flowchart: D3IL Baseline (Action-Only Diffusion)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE - D3IL                          │
└──────────────────────────────────────────────────────────────────────────┘

                         Dataset
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
    ┌───▼────┐                         ┌───────▼──┐
    │ Actions│ [B,T,3]                │ States   │ [B,T,7]
    │(Δx,Δy,│                        │(ee_pos, │
    │ Δz)    │                        │joints)   │
    └───┬────┘                         └────┬────┘
        │                                   │
        │  scaler.scale_output()            │  scaler.scale_input()
        │  (y_mean, y_std)                  │  (x_mean, x_std)
        │  y_std_safe = y_std + 1e-12       │  
        │                                   │
    ┌───▼────────────────────┐             │
    │ a_scaled [B,T,3]       │             │
    │ Means [0.0001]         │             │     ┌──────────────┐
    │ Stds   [0.01, ...]     │             │────▶│ Vision Enc.  │
    └───┬────────────────────┘             │     │ (U-Net)      │
        │                                   │     └──────┬───────┘
        │                                   │            │
        │                                   │     obs_enc [B,T,128]
        │   +─────────────────────────────────────────────+
        │   │  Conditioning (read-only)
        │   │
    ┌───▼──────────────────────────────────────────┐
    │  DDPM Diffusion Model (ACTION SPACE ONLY)   │
    │                                             │
    │  loss = MSE(ε_pred[B,T,3], ε_true[B,T,3])  │
    │                                             │
    │  ε_pred = DiffusionMLPNetwork(              │
    │      x_t[B,T,3] @ t,                       │
    │      state=obs_enc[B,T,128],               │
    │      goal=goal[B,T,128]                    │
    │  )                                          │
    │                                             │
    │  Forward pass: x_t = √ᾱ*x₀ + √(1-ᾱ)*ε     │
    └─────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE - D3IL                           │
└──────────────────────────────────────────────────────────────────────────┘

    Current Visual Observation
            │
            ├──▶ Vision Encoder
            │    └──▶ obs_enc [B, 128]
            │
    ┌───────▼──────────────────────────────────────────┐
    │  Reverse Diffusion (16 denoising steps)          │
    │                                                  │
    │  x_0 = randn([B, 3])     # Start from noise    │
    │  for t in [T-1...0]:                           │
    │    ε = model(x_t, t, obs_enc)                  │
    │    x_{t-1} = (x_t - β/√(1-ᾱ)*ε) / √α + σ*z   │
    │                                                  │
    │  OUTPUT: a_scaled [B, 3]                        │
    └───────┬──────────────────────────────────────────┘
            │
            ├──▶ Clamp to bounds: [-1, 1]
            │
            ├──▶ Unscale: a_cmd = a_scaled * (y_std + 1e-12) + y_mean
            │              = a_scaled * 0.0001 + 0.0 ≈ 0.0  (for Z-axis)
            │
            └──▶ SEND TO ROBOT CONTROLLER ✓
```

---

## System Flowchart: Gen6 Visual-Aligning (Joint 6D Diffusion)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE - GEN6                          │
└──────────────────────────────────────────────────────────────────────────┘

                         Dataset
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
    ┌───▼────┐                         ┌───────▼──┐
    │ Actions│ [B,T,3]                │ States   │ [B,T,7]
    │(Δx,Δy,│                        │(ee_pos, │
    │ Δz)    │                        │joints)   │
    └───┬────┘                         └────┬────┘
        │                                   │
        │                              state_subset = state[:,:,:3]
        │                              (EE position only)
        │                                   │
        │                                   │
        │   ┌───────────────────────────────┘
        │   │
        │   x = cat([action, state_subset], dim=-1)
        │   x shape: [B, T, 6]
        │
    ┌───▼────────────────────────────────────────┐
    │ x_raw [B, T, 6]                           │
    │ = [Δx, Δy, Δz, x_ee, y_ee, z_ee]          │
    │ Means:  [0.0, 0.0, 0.0, 0.4, 0.0, 0.6]   │
    │ Stds:   [0.1, 0.1, 0.0001, 0.05, 0.05, 0.0] (PROBLEM!)
    │                                           │
    │ scaler.fit(x=[B,T,6])                    │
    │ x_std_safe = max(x_std, 1e-2)            │ 🔴 INFLATION!
    │ x_std_safe = [0.1, 0.1, 0.01, 0.05, 0.05, 0.01]
    └───┬────────────────────────────────────────┘
        │
    ┌───▼────────────────────┐
    │ x_scaled [B,T,6]       │
    │ dims [0:3] = actions   │
    │ dims [3:6] = positions │
    └───┬────────────────────┘
        │
        │  Vision Encoder (same as D3IL)
        │        └──▶ obs_enc [B, T, 128]
        │
    ┌───▼──────────────────────────────────────────────┐
    │  VisualGaussianDiffusion (6D JOINT SPACE) 🔴    │
    │                                                  │
    │  loss = MSE(ε_pred[B,T,6], ε_true[B,T,6])      │
    │                                                 │
    │  ε_pred = VisualUNet(                           │
    │      x_t[B,T,6] @ t,  # ← Full 6D trajectory  │
    │      cond={visual: ..., 0: obs[:, 0]}          │
    │  )                                              │
    │                                                 │
    │  ** ISSUE: Assumes [action, state] were       │
    │     jointly optimized, but they're coupled   │
    │     by dynamics + vision encoder              │
    └────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE - GEN6                           │
└──────────────────────────────────────────────────────────────────────────┘

    Current Visual Observation & Robot State
            │
            ├──▶ Vision Encoder
            │    └──▶ obs_enc [B, T, 128]
            │
    ┌───────▼──────────────────────────────────────────────────┐
    │  Reverse Diffusion (16 denoising steps)   +  PROJECTION  │
    │                                                            │
    │  x_0 = randn([B, T, 6])     # Start from noise          │
    │                                                            │
    │  x = apply_conditioning(x, {0: initial_obs})            │
    │  (Snap first frame to observed state)                   │
    │                                                            │
    │  for t in [T-1...0]:                                    │
    │    ε = model(x_t, t, cond)                              │
    │    x_recon = predict_from_noise(x_t, ε)               │
    │                                                            │
    │    ★ Clamp action dims only: x_recon[..., :3] ∈ [-5,5]  │
    │    ★ Leave state dims [3:6] UNCLAMPED 🔴                │
    │                                                            │
    │    IF projector is active:                              │
    │      grad = projector.compute_gradient(x_recon, c)      │
    │      x_recon ← x_recon + grad  # QP correction         │
    │                                                            │
    │    x_{t-1} = denoise(x_recon, ...)                      │
    │    x = apply_conditioning(x, {0: initial_obs})          │
    │                                                            │
    │  OUTPUT: x_final [B, T, 6]                              │
    └───────┬────────────────────────────────────────────────┘
            │
            ├──▶ Extract action: a_scaled = x[0:3]
            │
            ├──▶ Clamp action: a_scaled ∈ [-5, 5]
            │
            ├──▶ 🔴 PROBLEM UNSCALING:
            │    a_cmd = a_scaled * x_std_safe[0:3] + x_mean[0:3]
            │           = a_scaled * [0.1, 0.1, 0.01] + [0.0, 0.0, 0.0]
            │           = a_scaled * [0.1, 0.1, 0.01]  (100x LARGER Z!)
            │
            └──▶ SEND TO ROBOT CONTROLLER
                 ⚠️  Z-channel is inflated 100x from diffusion
```

---

## Key Dimensional Comparison: Z-Axis Flat-Table Example

```
┌─────────────────────────────────────────┐     ┌──────────────────────────┐
│        D3IL (Action Only)               │     │   Gen6 (Joint 6D)        │
├─────────────────────────────────────────┤     ├──────────────────────────┤
│                                         │     │                          │
│  Data Statistics (Z-axis):              │     │ Data Statistics (Z):     │
│  ├─ action_z_std = 0.0001 (vibrations) │     │ ├─ [action_z] = 0.0001  │
│  └─ (states not in diffusion)           │     │ └─ [pos_z] = 0.0        │
│                                         │     │                          │
│  Scaler Setup:                          │     │ Scaler Setup:            │
│  ├─ y_std[2] = 0.0001                   │     │ ├─ x_std[2] = 0.0001    │
│  ├─ y_std_safe[2] = 0.0001 + 1e-12     │     │ ├─ x_std_safe[2] = 0.01 │
│  └─ = 0.0001 ✓ Faithful to data       │     │ └─ = 0.01 🔴 INFLATED   │
│                                         │     │                          │
│  Bounds (during inference):             │     │ Bounds (during inference):│
│  ├─ y_bounds[2] = [0.0, 0.0]           │     │ ├─ x_bounds[2] = [0, 0] │
│  └─ action_z always clipped to 0.0     │     │ └─ (still zero because   │
│                                         │     │     data variance=0)      │
│  Unscaling:                             │     │                          │
│  ├─ a_z = diffusion_output * 0.0001    │     │ Unscaling:               │
│  ├─ Random noise 0.5 → 0.00005         │     │ ├─ a_z = output * 0.01  │
│  └─ = essentially ZERO ✓               │     │ ├─ Random noise 0.5 → 0.005
│                                         │     │ └─ = 50x LARGER! 🔴     │
│                                         │     │                          │
│  Result:                                │     │ Result:                  │
│  ✓ Z-actions are ~0                    │     │ 🔴 Z-actions inflated   │
│  ✓ Correct for flat-table tasks        │     │ 🔴 Must be projected out│
│  ✓ No spurious vertical motion         │     │ 🔴 Creates coupling loop│
│                                         │     │                          │
└─────────────────────────────────────────┘     └──────────────────────────┘
```

---

## QP Projection in Gen6

```
During Inference (Reverse Diffusion Loop):

┌─────────────────────────────────────────────────────────────┐
│  At each denoising step t:                                 │
└─────────────────────────────────────────────────────────────┘

    1. Denoise step produces x_raw [B, T, 6]
       = [a_raw, s_raw]  (unconstrained)
               │
               │
    2. IF projector enabled:
               │
        ┌──────▼──────────────────────────────────────┐
        │  QP Projection Solver                       │
        │                                              │
        │  minimize   (1/2)||x - x_raw||²             │
        │                                              │
        │  subject to:                                │
        │    -∞ ≤ a[0:3] ≤ +∞          (actions)     │
        │    [0.3, -0.5, 0.0] ≤ s[3:6] ≤ [0.8, 0.5, 0.7]
        │                               (positions)   │
        │    s[3:6]_{t+1} = s[3:6]_t + a[0:3]*dt     │
        │                               (dynamics)    │
        │                                              │
        │  OUTPUT: x_corrected                        │
        └──────┬──────────────────────────────────────┘
               │
    3. Next denoising iteration uses x_corrected as input
       (instead of x_raw)
               │
               └──▶ Creates feedback loop between 
                   action/state denoising and constraints
                   
                   ✓ Good: Enforces hard constraints
                   🔴 Bad: Couples what should be separate
```

---

## Summary Table: Trajectory Dimensionality at Each Stage

| Stage | **D3IL** | **Gen6** | **Impact** |
|-------|----------|---------|-----------|
| **Raw Data** | a:[B,T,3], s:[B,T,7] | a:[B,T,3], s:[B,T,7] | Identical source |
| **State Subset** | All 7D | Only 3D (EE pos) | Discards joint info |
| **Concatenation** | Separate feeds | x=[a,s]:[B,T,6] | **FUNDAMENTAL DIFFERENCE** |
| **Scaler Input** | a,s separate | x concatenated | Different statistics |
| **Epsilon σ(std)** | y_std + 1e-12 | x_std + 1e-2 | **100x difference in Z** |
| **Diffusion Space** | 3D action only | 6D [a,s] joint | Different problem formulation |
| **Loss** | MSE on [B,T,3] | **MSE on [B,T,6]** | Asymmetric optimization |
| **Clamping** | [-1,1] on actions | [-5,5] on actions only | **Asymmetric** in Gen6 |
| **Projection** | None | DPCC on 6D | Adds constraint satisfaction |
| **Output** | a:[B,3] | x:[B,T,6] | Extract a from x |
| **Unscaling** | a_cmd = a * y_std | **a_cmd = a * x_std[0:3]** | **100x Z-inflate in Gen6** |

---

## Recommended Reading Order

1. **[D3IL_BASELINE_vs_GEN6_ANALYSIS.md](../D3IL_BASELINE_vs_GEN6_ANALYSIS.md)** - Full details
2. **This file** - Visual diagrams
3. **[logs_in_develop/Gen6_dpcc_Engine_for_visual_aligning/fix_3/Gen6_Current_Status_Audit.md](../../logs_in_develop/Gen6_dpcc_Engine_for_visual_aligning/fix_3/Gen6_Current_Status_Audit.md)** - Critical issues & proposed fixes
