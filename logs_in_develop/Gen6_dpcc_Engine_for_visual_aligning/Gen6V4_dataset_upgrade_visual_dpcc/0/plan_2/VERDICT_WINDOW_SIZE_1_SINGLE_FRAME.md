# Verdict — Why `window_size=1` (Single-Frame Visual Conditioning) in Gen6V4

**Date:** 2026-05-19  
**Scope:** `plan_visual_aligning_dpcc` / `VisualUNet` / `ParityAligningDataset`

---

## TLDR — Is It Because of DDPM / the ML Model?

**No. DDPM is not the cause.**  
DDPM is just a noise schedule and training objective — it is architecture-agnostic and runs identically whether you condition on 1 frame or 8. The lock lives entirely in the **conditioning interface**: three lines of code that were written for single-frame inputs and never upgraded.

The key gap is that **our backbone and d3il's DDPM vision agent process temporal context completely differently**, even though both use DDPM. If we upgraded to the same or a similar DDPM architecture, window_size would become adjustable — but only with a retrain. The details are below.

---

## Short Answer

`window_size=1` is **not a free choice** — it is a structural lock imposed by the dataset.  
The config comment "Using window_size>1 at eval would mean-pool multiple frames and shift the FiLM conditioning distribution away from what the model learned" is correct, but it only explains half of the story. The deeper reason is that the **training data never provides multi-frame image windows** at all.

---

## The Full Chain

### 1. Dataset (`ParityAligningDataset.__getitem__`)

```python
conditions = {
    0:             obs_norm[0],                       # (6,) — 6D state anchor
    'primary_img': self.bp_cam_imgs[ep][start],      # (C, H, W) — SINGLE frame
    'wrist_img':   self.inhand_cam_imgs[ep][start],  # (C, H, W) — SINGLE frame
}
```

`[start]` is a scalar index — **one frame per sample, always**.  
There is no temporal window built in the dataset; only the trajectory slice `[start:end]` carries horizon.

### 2. Loss function (`VisualGaussianDiffusion.loss`)

```python
primary_img = conditions['primary_img'].unsqueeze(1)   # (B, C,H,W) → (B, 1, C, H, W)
wrist_img   = conditions['wrist_img'].unsqueeze(1)     # hard T_win = 1
```

The loss **hard-codes T_win = 1** via `.unsqueeze(1)`. The ResNet encoder inside `encode_visual` is always called with exactly one frame in the time dimension.

### 3. Encoder (`VisualUNet.encode_visual`)

```python
features = self.obs_encoder(obs_dict)          # (B*T, 128)
return features.view(B, T, -1).mean(dim=1)     # mean over T_win
```

Mean-pooling over T_win is the only temporal aggregation. With T_win=1 during training, this is **identity** — the model never learned to aggregate multiple frames. The ResNet weights are optimized entirely for single-frame feature extraction.

### 4. Eval (`VisualAgentWrapper`)

```python
self.bp_image_context = deque(maxlen=self.window_size)
# ... fills deque, pads if needed, then:
bp_seq = torch.cat(list(self.bp_image_context), dim=0)  # (W, C, H, W)
```

If `window_size > 1`, this deque returns multiple frames.  
`encode_visual` would average their embeddings.  
But training used T_win=1 only → **distribution shift on the FiLM conditioning vector**.

---

## Why Is This Architecture Markovian By Design?

The 9D trajectory already encodes the relevant dynamics:

```
x[t] = [ dx  dy  dz  |  des_x des_y des_z  |  x  y  z ]
         act(3D)          des_c_pos(3D)        c_pos(3D)
```

- `c_pos` tells the model where the end-effector **is right now**.
- `des_c_pos` tells it where it was **commanded** to be.
- The U-Net plans a full horizon `H=8` of future actions from this single snapshot.

The state obs is already rich enough to reason about whether the robot is on-track.  
A temporal image window would add value mainly for:
1. Estimating **velocity** (not captured in a single frame)
2. Handling **partial occlusion** (one frame may be bad; average of recent frames is more robust)

Neither is critical for the current aligning task, which is slow-motion and well-lit.

---

## Comparative Table

| Property | Gen6V4 (current) | d3il baseline (`ddpm_encdec_vision`) |
|---|---|---|
| `window_size` | **1** | **8** |
| temporal encoder | mean pool (identity at T=1) | VAE Transformer with positional embeddings |
| dataset sample | single frame `(C, H, W)` | frame sequence `(W, C, H, W)` |
| temporal signal | from 9D state (c_pos, des_c_pos) | from image history + state |
| policy type | **reactive / Markovian** | memory-enabled |
| FiLM input dim | 128 (one ResNet embedding) | 128 (mean of 8 ResNet embeddings) |

d3il's temporal benefit comes from the **Transformer architecture**, not from mean pooling.  
Mean pooling 8 frames only helps if the appearance is slowly changing — it does not capture temporal ordering.

---

## Is window_size=1 a Bug?

**No.** It is a consistent, correct Markovian design given the current architecture.  
The comment in `config/aligning-d3il-visual.py` is accurate:

> `window_size=1 / obs_seq_len=1` must match training: `ParityAligningDataset` provides single-frame images per sample, so the model is trained on T_win=1.  
> Using `window_size>1` at eval would mean-pool multiple frames and shift the FiLM conditioning distribution away from what the model learned.

The root cause (dataset always returns single images) is documented here.

---

## If We Upgraded to the Same ML as d3il (`ddpm_vision_agent`)

d3il's DDPM vision agent also uses ResNet + DDPM, so it looks similar. But it handles temporal context fundamentally differently from our architecture.

### How d3il's `ddpm_vision_agent` processes T frames

```
1. Encode T frames independently via ResNet
   (B, T, C, H, W) → flatten to (B*T, C, H, W) → ResNet → (B*T, 128) → reshape → (B, T, 128)

2. Pass the full (B, T, 128) sequence directly into the DDPM MLP at every denoising step:
   x_cat = [action (B,T,3) | t_emb (B,1,t_dim) | obs_features (B,T,128)]
   MLP processes each of the T time-slots independently and outputs (B, T, action_dim)
   → predicts one action per frame in the window, uses the last
```

**Key**: d3il never mean-pools. It keeps the full `(B, T, 128)` feature sequence and lets the MLP
process all T slots. Each slot is a separate (action, time-embedding, frame-feature) concatenation.
The MLP implicitly learns that newer frames are more relevant through training — it is not forced to
average them.

### How our `VisualUNet + VisualGaussianDiffusion` works

```
1. Encode T frames via ResNet, then mean-pool to a single vector
   (B, T, C, H, W) → (B*T, C, H, W) → ResNet → (B*T, 128) → (B, T, 128) → mean(dim=1) → (B, 128)

2. Use the single 128D vector as a FiLM conditioning signal on a 1D temporal U-Net
   UNet denoises the FULL horizon-8 trajectory in one shot, conditioned on this one vector
```

**Key**: our architecture collapses the temporal window to a single vector before the U-Net ever
sees it. The U-Net predicts a trajectory plan (H=8 steps), not one action. This is the diffuser
planning model design, not the d3il step-prediction design.

### Side-by-side

| Property | d3il `ddpm_vision_agent` | Gen6V4 `VisualUNet` |
|---|---|---|
| Temporal aggregation | **None** — keeps `(B, T, 128)` | **Mean pool** → `(B, 128)` |
| DDPM output shape | `(B, T, action_dim)` — one action/frame | `(B, H, 9)` — full trajectory plan |
| Temporal order preserved? | Yes (implicitly, via MLP weights) | No (mean destroys order) |
| Can set `window_size > 1` | **Yes, by design** | Only after dataset + loss + retrain |
| What window adds | True temporal reasoning over recent frames | Robustness to single-frame noise (marginal) |

### Verdict: can we just swap to d3il's DDPM architecture?

Yes, with trade-offs. Adopting d3il's MLP-based approach would:
- Unlock `window_size > 1` correctly (no mean pooling, temporal order preserved)
- **Lose** the horizon-H trajectory plan — output becomes single-step action prediction
- **Lose** the DPCC projector integration (which operates on the full H-step trajectory)
- Require full dataset refactor and retrain

If the goal is **trajectory planning with temporal visual context**, the right path is not d3il's MLP
approach but replacing `encode_visual`'s `mean(dim=1)` with a causal LSTM or small Transformer —
keeping the U-Net trajectory backbone intact. See the Gen7 recommendation table below.

---

## Can We Upgrade to window_size > 1?

Yes, but it requires three coordinated changes:

| # | Change | What breaks if skipped |
|---|---|---|
| 1 | `ParityAligningDataset.__getitem__` returns `T_win` consecutive frames per sample | Training still uses T_win=1 → mean pool is still identity |
| 2 | `VisualGaussianDiffusion.loss` removes the hard `.unsqueeze(1)` and accepts pre-stacked `(B, T_win, C, H, W)` | The encoder receives wrong shape |
| 3 | Retrain from scratch | Existing checkpoints were conditioned on single-frame embeddings |

**Mean pooling limitation**: Even with T_win > 1 and the above changes, mean pooling destroys temporal order. To truly exploit temporal context, `encode_visual` should be replaced by an LSTM or a causal Transformer over frame embeddings.

---

## Recommendation for Gen7

| Scenario | Recommended approach |
|---|---|
| Continue with Gen6V4 / same checkpoints | Keep `window_size=1`. Do not change. |
| Improve robustness to occlusion / noise | Upgrade to `window_size=3`, mean pooling, retrain. Low-effort, marginal gain. |
| Proper temporal policy (velocity estimation, memory) | Replace `encode_visual` mean pool with causal 2-layer LSTM over ResNet embeddings, `window_size=4–8`, retrain. Matches d3il baseline's design intent. |

---

## Files That Enforce the window_size=1 Lock

| File | Line(s) | What locks T_win=1 |
|---|---|---|
| `diffuser_visual_aligning/datasets/sequence.py` | 134–135 | `conditions['primary_img']` is `[start]` (scalar index) |
| `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` | 34–35 | `.unsqueeze(1)` in `loss()` |
| `diffuser_visual_aligning/models/visual_unet.py` | 100–101 | `mean(dim=1)` in `encode_visual` |
| `config/aligning-d3il-visual.py` | `plan_visual_aligning_dpcc` block | `window_size: 1`, `obs_seq_len: 1` |
