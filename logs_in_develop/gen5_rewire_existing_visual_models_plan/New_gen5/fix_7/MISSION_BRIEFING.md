# Fix 7: Architectural Blindness & FiLM Conditioning Integration

## 1. Executive Summary
This fix addresses the **root cause** of persistent `0.0%` Success Rate in the Visual Aligning evaluation pipeline. Through a complete end-to-end trace from dataset → training loop → backbone → denoising loop → evaluation wrapper, we discovered that the `UNet1DTemporalCondModel` backbone was **architecturally incapable of using visual conditioning**. The visual embeddings from the ResNet encoder were computed during training but silently discarded by the backbone.

This document covers: the vetting methodology, the confirmed bug, the fix, backward compatibility analysis, and retraining requirements.

---

## 2. Problem Vetting: End-to-End Data Flow Trace

### 2.1 Training Data Flow
```
Aligning_Img_Dataset.__getitem__
    → returns: (bp_imgs, inhand_imgs, obs, act, mask)
    → obs: [B, window=8, 3] (robot_des_pos)
    → act: [B, window=8, 3] (velocity deltas)

Trainer.train_epoch
    → batch = next(dataloader)
    → loss, infos = self.model.loss(*batch)

VisualGaussianDiffusion.loss(bp_imgs, inhand_imgs, obs, act, mask)
    → x = cat([act, obs], dim=-1)  → [B, 8, 6]
    → cond = {'visual': (bp_imgs, inhand_imgs, obs), 0: obs[:, 0]}
    → self.p_losses(x, cond, t)

GaussianDiffusion.p_losses(x_start, cond, t)
    → x_noisy = q_sample(x_start, t, noise)
    → x_noisy = apply_conditioning(x_noisy, cond, ...)  # snaps obs at t=0
    → x_recon = self.model(x_noisy, cond, t)             # ← HERE

VisualUNet.forward(x_noisy, cond, t)
    → bp_imgs, inhand_imgs, state = cond['visual']
    → visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state)  # [B, T, 128]
    → return self.backbone(x_noisy, visual_emb, t)

UNet1DTemporalCondModel.forward(x, cond=visual_emb, time)
    → x = rearrange(x, 'b h t -> b t h')
    → t = self.time_mlp(timesteps)              # [B, 128]
    → ★★★ `cond` (visual_emb) IS NEVER USED ★★★
    → for resnet, resnet2, downsample in self.downs:
          x = resnet(x, t)    # only time embedding, no cond
    → return x
```

### 2.2 The Bug: Confirmed Architectural Blindness
At line 169 of `diffuser/models/unet1d_temporal_cond.py`, the `forward()` method signature accepts `cond` as its second argument:
```python
def forward(self, x, cond, time, returns=None, ...):
```
But the function body **never references `cond`**. It only uses `x` (noisy trajectory) and `time` (diffusion step). The visual embeddings from the ResNet encoder flow through the computation graph but are **discarded before reaching any learnable layer**.

### 2.3 Why Training Loss Was Low Despite Blindness
The model is not truly "unconditional" — it uses **trajectory inpainting** (`apply_conditioning`) which stamps the current robot position at `x[:, 0, action_dim:]` at every denoising step. Since the robot's position is highly correlated with the correct push direction in the aligning task, the model learned a strong proprioceptive policy.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| `loss` | `0.000212` | Excellent convergence |
| `a0_loss` | `1.94e-5` | Near-perfect first-action prediction |
| `loss_test` | `0.000326` | Minimal overfitting |

The model successfully learned: "given my current robot position, what velocity should I apply?" — but it cannot adapt to different block positions because it never "sees" the images.

### 2.4 Comparison: D3IL DDPM-ACT vs. Our Model
| Component | D3IL DDPM-ACT (`ddpm_encdec_vision_agent.py`) | Our Model (Pre-Fix) |
| :--- | :--- | :--- |
| **Backbone** | `DiffusionEncDec` (Transformer Encoder-Decoder) | `UNet1DTemporalCondModel` (1D UNet) |
| **Vision Integration** | Cross-Attention: visual features actively queried at every layer | **None**: visual features silently discarded |
| **State Conditioning** | Transformer tokens embed state sequence | Trajectory inpainting (snapping `x[:, 0]`) |
| **Action Output** | `self.action_pred(decoder_output)` | Extracted from denoised trajectory `x[:, :, :3]` |

---

## 3. The Fix: FiLM-Style Conditioning Projection

### 3.1 Approach
We add a **FiLM (Feature-wise Linear Modulation)** projection to the `UNet1DTemporalCondModel`. The visual embeddings are:
1. Mean-pooled over the temporal axis: `[B, T, 128]` → `[B, 128]`
2. Projected through a 2-layer MLP: `[B, 128]` → `[B, dim]`
3. Concatenated with the time embedding `t`: `[B, dim]` → `[B, dim + dim]` = `[B, 256]`
4. This combined embedding modulates every `ResidualTemporalBlock` in the UNet.

### 3.2 Backward Compatibility
**Critical Design Decision**: The state-based avoiding pipeline also passes `cond_dim=20` to `UNet1DTemporalCondModel`, but its `cond` argument is a **dict** (for inpainting), not a tensor. Without protection, the fix would:
- Create a `cond_mlp` (changing the model architecture)
- Change `embed_dim` from `128` to `256`
- **Break all existing state-based checkpoints**

**Solution**: We added an opt-in flag `use_cond_projection=False` (default). Only `VisualUNet` sets `use_cond_projection=True`. State-based pipelines are completely unaffected.

Additionally, the forward pass checks `isinstance(cond, torch.Tensor)` to prevent crashes when `cond` is a dict.

### 3.3 Files Modified

| File | Change |
| :--- | :--- |
| `diffuser/models/unet1d_temporal_cond.py` | Added `use_cond_projection` flag, `cond_mlp` projection, FiLM integration in `forward()`, `isinstance` type guard |
| `flow_matcher/models/unet1d_temporal_cond.py` | Synced (identical copy) |
| `ddpm_encdec_vision/models/visual_unet.py` | Added `use_cond_projection=True` to backbone constructor |
| `ddpm_encdec_vision/models/visual_gaussian_diffusion.py` | Fixed snapping point `pos[:, 0]` → `pos[:, -1]` |
| `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | Removed Scaler (training uses raw coordinates) |

---

## 4. Retraining Requirement

> [!IMPORTANT]
> This fix changes the model architecture. Existing checkpoints are **incompatible** with the new `VisualUNet` because `embed_dim` changes from `128` to `256` in all `ResidualTemporalBlock` layers. **A full retrain is required.**

The retrain command is unchanged:
```bash
sbatch Slurm_Codes/sbatch/Visual_Aligning/train_visual_aligning.sh
```

### Expected Improvement After Retrain
- The ResNet encoder will now receive **meaningful gradients** (its output actually affects the loss).
- The UNet will learn to condition its trajectory predictions on **what the camera sees** (block position, orientation).
- Expected Success Rate: `0.65 – 0.90` (matching D3IL DDPM-ACT baselines).

---

## 5. Timeline of Evaluation Results

| Run | Mean Distance | Success Rate | Root Cause |
| :--- | :--- | :--- | :--- |
| Pre-Fix 6 (Scaler + old snap) | `0.495` | `0.0%` | Scaler corruption + temporal lag |
| Fix 6 (Scaler removed, old snap) | `0.417` | `0.0%` | Temporal lag (snap at `pos[:, 0]`) |
| Fix 7a (Scaler removed + snap fixed) | `0.390` | `0.0%` | Blind model (visual features discarded) |
| Fix 7b (FiLM conditioning + retrain) | **Pending** | **Pending** | — |

---

## 6. Conclusion
The persistent 0.0% Success Rate was caused by a **three-layer bug**:
1. **Layer 1 (Eval)**: The D3IL Scaler was applied but training didn't use it → fixed in Fix 6.
2. **Layer 2 (Eval)**: The snapping point used `pos[:, 0]` (8 steps ago) instead of `pos[:, -1]` (current) → fixed in Fix 7a.
3. **Layer 3 (Architecture)**: The UNet backbone discarded visual embeddings, making the model "blind" → fixed in Fix 7b (this fix). **Requires retrain.**

After retraining with FiLM conditioning, the model will be a true **vision-conditioned diffusion policy**, capable of adapting to different block configurations.
