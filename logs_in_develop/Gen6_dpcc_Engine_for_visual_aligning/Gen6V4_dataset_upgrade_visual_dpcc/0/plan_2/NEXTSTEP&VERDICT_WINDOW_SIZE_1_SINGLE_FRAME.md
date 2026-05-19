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

## Feasibility: Adopting Transformer Backbone (DDPM-ACT Style) With DPCC

### What is "DDPM-ACT style"?

d3il's `ddpm_encdec_vision_agent` uses `DiffusionEncDec` — a Transformer encoder-decoder conditioned
by DDPM noise. This is the ACT-style Transformer (from ACT: Action Chunking with Transformers)
wrapped with a DDPM training objective. The architecture is:

```
Visual frames (B, T, C, H, W)
  └─ ResNet per frame → (B, T, 128)
  └─ Linear projection → (B, T, embed_dim=64)
  └─ + positional embedding
  └─ Transformer Encoder → context (B, 1+obs_seq_len, 64)
                                        ↑ time token prepended
  └─ Transformer Decoder  ← cross-attends on encoder context
       query: action tokens (B, action_seq_len, 64)
       output: pred_actions (B, action_seq_len, action_dim)
DDPM denoises action_seq_len actions per forward pass
```

Key parameters in d3il's config: `obs_seq_len=5`, `action_seq_len=4`, `n_timesteps=16`.

### The DPCC Compatibility Problem

DPCC's projector is a **trajectory-space operator**. Its contract, set at construction time, is:

```python
Projector(horizon=H, transition_dim=9, ...)
# internally builds:
self.Q = torch.eye(9 * H)               # cost matrix: (H*9) × (H*9)
self.A = ...                            # equality constraints over all H steps
self.C = ...                            # inequality constraints over all H steps
# Euler dynamics constraint links c_pos[6:8] ← act[0:2] across H timesteps
```

`project(trajectory)` expects `(batch_size, H, 9)` — the **full 9D state+action trajectory**.
Without `c_pos` (dims 6–8), the `deriv` constraint (`c_pos_x ← dx`, etc.) has nothing to operate on,
and the physical safety guarantee is broken.

`DiffusionEncDec` outputs `(B, action_seq_len=4, action_dim=3)` — **actions only, no obs**.  
This is dimensionally incompatible with DPCC's `transition_dim=9, horizon=8`.

### Three Options and Their Feasibility

---

#### Option A — Transformer as the Denoising Backbone (DiT-style) ✅ Recommended

**Idea**: replace `UNet1DTemporalCondModel` with a Transformer inside the existing
DDPM trajectory-diffusion loop. The outer structure stays identical to Gen6V4.

```
DDPM loop (unchanged)
  └─ p_sample_loop: denoises (B, H, 9) trajectory over T diffusion steps
       at each step calls VisualTransformerUNet.forward(x, cond, t):
         x:    (B, H, 9) noisy trajectory  ← same as UNet
         cond: visual embedding             ← same as UNet
         t:    diffusion timestep           ← same as UNet
         output: (B, H, 9)                 ← same as UNet
DPCC projector: (B, H, 9) → (B, H, 9)     ← unchanged
```

The Transformer backbone denoises the trajectory sequence attending over all H=8 time slots and
the visual context simultaneously — true temporal attention on both trajectory and image history.
Visual frames can be cross-attended as token sequences `(B, T_win, 64)` without mean pooling.

**DPCC impact**: None. Input/output shape is still `(B, H, 9)`. The projector contract is preserved
exactly. Both bound constraints and Euler dynamics constraints continue to function.

**What changes**: `UNet1DTemporalCondModel` → `TransformerDenoiser1D` (new module).  
Visual conditioning path: `encode_visual → mean → FiLM` → `cross-attention over frame tokens`.  
`window_size` becomes freely adjustable once the dataset returns T_win frames.

**Effort**: Medium. Requires writing `TransformerDenoiser1D` (analogous to DiT or DP-Transformer),
updating `VisualUNet.forward` to call it, updating `encode_visual` to return token sequences,
updating the dataset and loss to pass multi-frame windows. No changes to DPCC, diffusion loop,
or training/eval scripts.

---

#### Option B — Adopt DiffusionEncDec Directly, Rebuild DPCC on Action-Only Space ⚠️ Degrades safety

**Idea**: use d3il's `DiffusionEncDec` exactly, adapt DPCC to project `(B, action_seq_len=4, 3)`.

**DPCC impact**: You lose the `deriv` (Euler dynamics) constraint entirely — it requires `c_pos`
which is no longer in the output. Only box-bound constraints on action dims remain. The physical
contract that `c_pos[t+1] = c_pos[t] + act[t] * dt` is no longer enforced.  
DPCC is reduced from a dynamics-aware projector to a simple clamp.

**What changes**: Full model swap, dataset format (remove obs from trajectory), rewrite DPCC,
retrain.

**Effort**: High. Significant capability regression for DPCC.

---

#### Option C — Augmented DiffusionEncDec output (predict 9D instead of 3D) ⚠️ Non-trivial

**Idea**: modify `DiffusionEncDec` to predict `(B, action_seq_len, 9)` — actions + des_c_pos +
c_pos — so DPCC can project the full 9D output.

```
Modified DiffusionEncDec:
  action_pred: Linear(embed_dim, 9)   ← was Linear(embed_dim, 3)
  output: (B, 4, 9)
DPCC: project((B, 4, 9), horizon=4, transition_dim=9)  ← H changes 8→4
```

DPCC's Q matrix and constraint matrices are rebuilt with `horizon=4` instead of 8. Euler dynamics
and bounds still apply. This preserves the full DPCC contract.

**DPCC impact**: Moderate. Projector must be reconstructed with new H=4. Constraint quality
degrades slightly because the planning horizon halves (4 steps vs 8). Longer plans would require
rolling the 4-step chunk forward with receding horizon, as d3il does.

**Effort**: High. Requires architectural modification to DiffusionEncDec, new dataset that returns
9D targets for the Transformer decoder, and DPCC rebuild with H=4.

---

### Summary Table

| Option | DPCC compatibility | Euler dynamics preserved | window_size > 1 | Effort | Recommended? |
|---|---|---|---|---|---|
| A — Transformer backbone (DiT-style) | **Full** (unchanged) | **Yes** | **Yes** | Medium | **Yes** |
| B — DiffusionEncDec direct | Broken (action-only) | **No** | Yes | High | No |
| C — Augmented DiffusionEncDec | Partial (H=4 rebuild) | Yes | Yes | High | Maybe for Gen8 |
| Current Gen6V4 (UNet + mean pool) | Full | Yes | No (lock) | — | For now |

### Verdict

**Option A is the only approach that preserves the full DPCC contract while unlocking temporal
visual context.** It threads the needle: the Transformer replaces only the denoising network
(the inner backbone), while the DDPM trajectory-diffusion loop, the 9D trajectory representation,
and the DPCC projector all remain unchanged. The visual encoder upgrades from mean-pooled FiLM
to cross-attention over frame tokens — this is where window_size becomes a free parameter.

Option B breaks DPCC's Euler dynamics guarantee and is not recommended unless the safety constraint
is explicitly being dropped. Option C is a viable future path for Gen8 if ACT-style action chunking
is a design goal, but requires the highest effort and partial DPCC rebuild.

---

## Existing Dead Code: How Far Is `ddpm_encdec_vision` From Working?

`ddpm_encdec_vision/` is a prototype package that predates Gen6V4. It contains **two separate
model paths**, both unfinished. Neither maps cleanly onto "Option B as described above" — they are
more like the skeletal scaffold before Option B's hardest piece (DPCC) was ever attempted.

---

### Path 1 — `VisualUNet` + `VisualGaussianDiffusion` (intended: multi-frame DDPM trajectory planner)

This was the precursor to Gen6V4: same DDPM trajectory-diffusion outer loop, but with temporal
visual conditioning (no mean-pool collapse, no unsqueeze lock).

**What it got right:**

| File | Line(s) | Correct behaviour |
|---|---|---|
| `ddpm_encdec_vision/models/visual_gaussian_diffusion.py` | 11–33 | `loss()` accepts `(bp_imgs, inhand_imgs, obs, act, mask)` — **no `.unsqueeze(1)` lock**; images arrive as full `(B, T, C, H, W)` |
| `ddpm_encdec_vision/models/visual_unet.py` | 76–100 | `encode_visual` returns `(B, T, 128)` — **no `mean(dim=1)` collapse**; temporal sequence preserved |

**Where it silently broke:**

`VisualUNet.__init__` (line 51–52) imports:

```python
from diffuser.models.unet1d_temporal_cond import UNet1DTemporalCondModel
backbone_class = UNet1DTemporalCondModel
```

This imports the **original `diffuser` UNet** — the one that was never modified for FiLM
conditioning. Its `forward(x, cond, time)` treats `cond` as a state dict for `apply_conditioning`
(integer-key inpainting). When it receives `visual_emb: (B, T, 128)` as a plain tensor, the forward
pass proceeds without error — but **silently ignores the tensor** because no branch in the original
UNet handles a tensor `cond`. The visual features are computed, then thrown away.

Every training step in this path trained a state-conditioned UNet while computing and discarding the
ResNet forward pass. The model learned no visual signal.

**Distance from working:**

| Gap | Fix required | Effort |
|---|---|---|
| Backbone ignores visual tensor | Swap import to `diffuser_visual_aligning.models.unet1d_temporal_cond` with `use_cond_projection=True` | 1 line, but collapses `(B,T,128)` → mean → FiLM = regresses to Gen6V4 T=1 behaviour |
| True multi-frame injection | Rewrite `forward` to cross-attend over frame tokens instead of FiLM mean-pool | Requires new `TransformerDenoiser1D` backbone — this is Option A |
| Dataset returns T>1 frames | `ParityAligningDataset.__getitem__` uses scalar index; dead-code dataset is D4RL-style, not aligning | Full dataset refactor |

**Bottom line for Path 1**: The dataset and loss interfaces were fixed (no unsqueeze lock, no mean
pool). The backbone injection was never fixed. As written, Path 1 trained blindly on images it
never used, producing a state-only DDPM policy with unused ResNet overhead.

---

### Path 2 — `VisualDiffusionBridge` (intended: d3il DiffusionEncDec = Option B skeleton)

`ddpm_encdec_vision/models/d3il_visual_bridge.py` wraps d3il's `DiffusionEncDec`
(Transformer Encoder-Decoder) and is architecturally identical to Option B's starting point.

**What it got right:**

| Property | Status |
|---|---|
| Visual encoder | `encode_visual` → `(B, T, 128)`, temporal preserved, no mean pool ✓ |
| DiffusionEncDec wiring | `hydra.utils.instantiate` from d3il config, `obs_seq_len=5 / action_seq_len=4 / n_timesteps=16` ✓ |
| Output shape | `diffusion_model(visual_emb, None)` → `(B, action_seq_len=4, action_dim=3)` — actions only ✓ |
| Action bounds | `min_action / max_action` initialised from `Scaler` ✓ |

**What is entirely absent (the hard part of Option B):**

```
VisualDiffusionBridge.loss(bp_imgs, inhand_imgs, obs, act, mask):
    visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state=None)
    loss_val = self.diffusion_model.loss(act, visual_emb, goal=None)
    #          ────────────────────────── ^^^ 3D action only
    return loss_val, {}
```

- `act` is 3D `(B, T, 3)`. There is no 9D trajectory `[act | des_c_pos | c_pos]`.
- The `deriv` (Euler dynamics) constraint requires `c_pos` dims 6–8 to be present in the trajectory.
  Without them, DPCC's QP constraint matrix has nothing to link to, and the projector cannot enforce
  `c_pos[t+1] = c_pos[t] + act[t] * dt`.
- **No DPCC projector is instantiated anywhere** in either the train or eval scripts for this path.
  The eval script (`ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`) wires a DPCC Projector
  but uses it on trajectories from `VisualUNet + VisualGaussianDiffusion` (Path 1), not from
  `VisualDiffusionBridge`.

**Distance from working Option B:**

| Gap | What is needed | Why it is hard |
|---|---|---|
| 9D trajectory | Augment `DiffusionEncDec` decoder head: `Linear(embed_dim, 9)` instead of `Linear(embed_dim, 3)` | Requires modifying d3il's model and redefining what the decoder predicts |
| Dataset | Must produce `(B, H, 9)` 9D targets for the Transformer decoder | Full dataset refactor |
| DPCC projector rebuild | `Projector(horizon=4, transition_dim=9)` with Euler constraints, OR accept bounds-only clamp if `deriv` is dropped | Euler dynamics are broken by default; you choose between simple clamp or full QP rebuild |
| Connection to inference | `VisualDiffusionBridge.predict()` returns `(B, 4, 3)` — never routed to DPCC | Plumbing |

**Bottom line for Path 2**: The Transformer architecture is in place. The action-only output `(B, 4, 3)`
exists. But DPCC's critical piece — the Euler dynamics constraint that makes the projector more than
a simple clamp — is geometrically impossible without `c_pos` in the output. Option B as described
in the MD requires rebuilding the DPCC constraint matrices for action-only space and accepting that
the `deriv` constraint is gone. That rebuild was **never started**.

---

### Dead Code vs. Option B: Side-by-Side Gap Table

| Component | Option B (as described in MD) | `VisualDiffusionBridge` (dead code) | Gap |
|---|---|---|---|
| Transformer EncDec | Required | Present ✓ | None |
| Visual encoder temporal | `(B, T, 128)` | `(B, T, 128)` ✓ | None |
| Output shape | `(B, 4, 3)` action-only | `(B, 4, 3)` ✓ | None |
| DPCC projector | Rebuilt for `(B, 4, 3)`, bounds-only | **Absent** ✗ | Full rebuild needed |
| Euler dynamics | Explicitly dropped | **Never considered** ✗ | Design decision missing |
| 9D trajectory loss | Modified EncDec head | **3D loss only** ✗ | Model + dataset refactor |
| Train script wiring | Full train loop | Partially wired, no DPCC | DPCC plumbing |

### Where the Dead Code Is Relative to Option A

Path 1 (`VisualUNet`) is the closest thing to Option A in the repo — same trajectory-diffusion
outer loop, same DPCC-compatible `(B, H, 9)` output shape. It failed only at the backbone injection
step (wrong import). Fixing that import collapses temporal context back to mean-pool FiLM (Gen6V4
again). Actually achieving Option A requires replacing the UNet backbone with a
`TransformerDenoiser1D`, which does not exist anywhere in the codebase.

**Option A gap from the dead code**: write one new module (`TransformerDenoiser1D`), fix the
`VisualUNet` import, upgrade `encode_visual` to return token sequences instead of mean-pooling,
update the dataset to return T>1 frames, remove the `unsqueeze(1)` in `loss()`. The structural
scaffold is reusable — the missing piece is the Transformer backbone itself.

---

## Files That Enforce the window_size=1 Lock

| File | Line(s) | What locks T_win=1 |
|---|---|---|
| `diffuser_visual_aligning/datasets/sequence.py` | 134–135 | `conditions['primary_img']` is `[start]` (scalar index) |
| `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` | 34–35 | `.unsqueeze(1)` in `loss()` |
| `diffuser_visual_aligning/models/visual_unet.py` | 100–101 | `mean(dim=1)` in `encode_visual` |
| `config/aligning-d3il-visual.py` | `plan_visual_aligning_dpcc` block | `window_size: 1`, `obs_seq_len: 1` |
