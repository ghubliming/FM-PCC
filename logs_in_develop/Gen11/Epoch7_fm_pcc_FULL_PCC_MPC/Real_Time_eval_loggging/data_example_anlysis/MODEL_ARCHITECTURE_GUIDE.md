# FM-PCC Model Architecture Study Guide

**Verified 2026-06-26 against current codebase.**

> **The central question**: the UNet takes `transition_dim` as a constructor argument that
> sets the *channel count of the very first Conv1d*. Change the trajectory dim → different
> weight shapes → cannot mix checkpoints. The UNet is NOT a rigid 9D-only object; it IS
> rigid once trained. How the 23D non-visual model works: `VisualUNet.__init__` reads
> `obs_dim=20` from config and builds `UNet1DTemporalCondModel(transition_dim=23)`. The
> checkpoint then has Conv1d weights with `in_channels=23`. Load it with 9D input → shape
> mismatch crash. They are different models.

---

## 1. The Trajectory Tensor — what enters the UNet

Every model in this repo denoises/flows a trajectory tensor of shape `(B, H, transition_dim)`.

| Task | Trajectory | transition_dim | Notes |
|---|---|---|---|
| D3IL avoiding (DPCC paper) | [act(2)\|des_xy(2)\|c_xy(2)] | 6 | 2D plane only |
| FM-PCC visual avoiding | [act(2)\|des_xy(2)\|c_xy(2)] | 6 | same, + FiLM camera |
| FM-PCC visual aligning | [act(3)\|des_c_pos(3)\|c_pos(3)] | 9 | FiLM carries box+target |
| FM-PCC non-visual aligning (WRONG) | [act(3)\|des_c_pos(3)\|c_pos(3)\|box(3)\|box_q(4)\|tgt(3)\|tgt_q(4)] | 23 | see ALIGNING_EXPANSION.md |
| FM-PCC non-visual aligning (CORRECT) | [act(3)\|des_c_pos(3)\|c_pos(3)] | 9 | same as visual, no FiLM |
| FM-PCC UAV | [act(3)\|p_des(3)\|p(3)\|v(3)] | 12 | added velocity, no FiLM |
| iMF state-only avoiding | [act(2)\|des_xy(2)\|c_xy(2)] | 6 | same as DPCC |

The tensor is always `[action_dims first | obs_dims after]` — Janner convention.
`apply_conditioning` pins the obs slice at `t=0` during sampling: `x[:, 0, action_dim:] = obs`.

---

## 2. UNet1DTemporalCondModel — the core backbone

**File**: `diffuser_visual_aligning/models/unet1d_temporal_cond.py`
(copied verbatim into: `fm_visual_aligning/`, `fm_visual_avoiding/`, `imf_visual_aligning/`,
`ddpm_encdec_vision/`, `diffuser_visual_avoiding/`)

### 2.1 Construction — what sets the channel count

```python
UNet1DTemporalCondModel(
    horizon,          # H — sets temporal conv sizes (RIGID once built)
    transition_dim,   # D — first Conv1d in_channels (RIGID once built)
    cond_dim,         # visual embedding dim (0 = no FiLM)
    dim=128,          # base channel width
    dim_mults=(1,2,4,8),  # → channel widths [9, 128, 256, 512, 1024]
    use_cond_projection=False,  # True = enable FiLM via cond_mlp
)
```

Channel progression:
```
Input channels:  transition_dim  (e.g. 9 or 23 or 12)
down-0:          9  → 128
down-1:         128 → 256   (+ Downsample1d, halve H)
down-2:         256 → 512   (+ Downsample1d, halve H)
mid:            512 → 1024 → 1024
up-2:           1024+512 → 512  (+ Upsample1d, double H)
up-1:           512+256  → 256  (+ Upsample1d, double H)
up-0:           256+128  → 128  (no up — skip connection)
final_conv:     128 → transition_dim
```

**Both `transition_dim` AND `horizon` are baked into weight shapes at build time.**
Changing either requires re-instantiating (and retraining) the model.

**Horizon padding**: `VisualUNet` pads H to the next multiple of 8 before the UNet,
then slices back:
```python
padded_horizon = ((H + 7) // 8) * 8   # e.g. H=8 → 8 (no-op)
```

### 2.2 Forward pass — data flow

```python
def forward(self, x, cond, time, ...):
    # x: (B, H, transition_dim)
    x = rearrange(x, 'b h t -> b t h')   # → (B, transition_dim, H)

    t = self.time_mlp(time)               # (B, dim)

    # FiLM path (only when use_cond_projection=True AND cond is a Tensor):
    if self.cond_mlp is not None and isinstance(cond, torch.Tensor):
        cond_pooled = cond.mean(dim=1) if cond.dim()==3 else cond  # (B, cond_dim)
        cond_emb = self.cond_mlp(cond_pooled)  # (B, dim)
        t = cat([t, cond_emb], dim=-1)    # (B, 2*dim) — time + visual info merged

    for resnet, resnet2, downsample in self.downs:
        x = resnet(x, t)    # ← t modulates EVERY ResidualTemporalBlock
        x = resnet2(x, t)
        h.append(x)
        x = downsample(x)

    x = mid_block1(x, t); x = mid_block2(x, t)

    for resnet, resnet2, upsample in self.ups:
        x = cat([x, h.pop()], dim=1)    # skip connection doubles channels
        x = resnet(x, t); x = resnet2(x, t)
        x = upsample(x)

    x = final_conv(x)                   # → (B, transition_dim, H)
    x = rearrange(x, 'b t h -> b h t') # → (B, H, transition_dim)
```

### 2.3 ResidualTemporalBlock — where FiLM actually happens

```python
class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, ...):
        self.blocks = [Conv1dBlock(inp, out), Conv1dBlock(out, out)]
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),  # embed_dim = dim (or 2*dim with FiLM)
            Rearrange('b t -> b t 1'),            # broadcast over H axis
        )
        self.residual_conv = Conv1d(inp, out, 1) if inp!=out else Identity()

    def forward(self, x, t):
        # x: (B, inp_channels, H)
        # t: (B, embed_dim)
        out = self.blocks[0](x) + self.time_mlp(t)  # ← ADDITIVE conditioning
        out = self.blocks[1](out)
        return out + self.residual_conv(x)           # ← skip
```

**Important**: the FiLM here is ADDITIVE BIAS, not true scale+shift FiLM.
`time_mlp(t)` → (B, out_channels, 1) → broadcasts over H and adds to the first
Conv1d output. The visual embedding modulates every layer via this additive channel bias.
This is simpler than true FiLM (`scale * x + shift`) but is what the code actually does.

### 2.4 Non-visual mode — cond is a dict, not a tensor

```python
# In non-visual mode:
# - cond_mlp = None  (use_cond_projection=False)
# - cond = {0: obs_tensor} passed to apply_conditioning EXTERNALLY
# - UNet.forward receives cond as a dict; the isinstance(cond, torch.Tensor) check fails
# - No FiLM fires; the UNet only receives time embedding
```

The `cond` dict `{0: obs}` is consumed by `GaussianDiffusion.conditional_sample()` or
`FlowMatchingODE.sample()` BEFORE calling the UNet. `apply_conditioning` pins the obs
slice in the noisy trajectory at each denoising step; the UNet itself never sees the dict.

---

## 3. VisualUNet — the wrapper that routes visual vs non-visual

**Files**: `diffuser_visual_aligning/models/visual_unet.py`,
`fm_visual_aligning/models/visual_unet.py`, `imf_visual_aligning/models/visual_unet.py`

```python
class VisualUNet(nn.Module):
    TRANSITION_DIM = 9     # HARDCODED for visual mode
    LATENT_DIM     = 128   # dual ResNet-64 concat

    def __init__(self, config):
        if config.if_vision:
            # Vision encoder: MultiImageObsEncoder (dual ResNet-18, imagenet norm)
            #   bp_cam (B,T,3,96,96) → ResNet-64 → (B,T,64)
            #   inhand_cam           → ResNet-64 → (B,T,64)
            #   concat               → (B,T,128) → mean pool over T → (B,128)
            self.obs_encoder = MultiImageObsEncoder(...)
            latent_dim = 128
            transition_dim = 9          # HARDCODED — config.obs_dim ignored!
        else:
            self.obs_encoder = None
            latent_dim = 0
            transition_dim = config.action_dim + config.obs_dim  # 3+20=23 (WRONG) or 3+6=9 (CORRECT)

        self.backbone = UNet1DTemporalCondModel(
            horizon=padded_horizon,
            transition_dim=transition_dim,    # ← sets first Conv1d channels
            cond_dim=latent_dim,              # 128 (visual) or 0 (non-visual)
            use_cond_projection=config.if_vision,  # FiLM ON for visual only
        )
```

**Why `TRANSITION_DIM=9` is hardcoded for visual**: a past bug (fix_5) showed that
`config.obs_dim` gets stale placeholder values (e.g. 128) from legacy configs and would
produce wrong backbone channel counts. The visual trajectory dimension is always 9 by
design; the const protects against config drift.

**For non-visual, `obs_dim` from config is still used** — this is where the 23D vs 9D
question bites. Currently `config.obs_dim=20` → `transition_dim=23`. Should be `6` → `9`.

---

## 4. Flow_matcher_U_Net_v2 — UAV and state-only avoiding

**File**: `flow_matcher_v3_uav/models/unet1d_temporal_cond.py`

Identical `ResidualTemporalBlock` / down-mid-up structure as `UNet1DTemporalCondModel`.
Key differences:

```python
class Flow_matcher_U_Net_v2(ModelMixin, ConfigMixin):
    def __init__(self, horizon, transition_dim, cond_dim, ...):
        # NO use_cond_projection parameter
        # NO cond_mlp — FiLM is structurally absent
        # cond_dim is accepted but never used in __init__
        ...

    def forward(self, x, cond, time, ...):
        x = rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)    # time embedding only
        # cond is accepted but NEVER READ
        ...
```

Used for:
- UAV: `flow_matcher_v3_uav/` — `transition_dim=12` (12D [act(3)|p_des(3)|p(3)|v(3)])
- State-only avoiding: `flow_matcher_v3_imeanflow/` — `transition_dim=6` (6D [act(2)|des_xy(2)|c_xy(2)])
- FM non-visual avoiding variants: `flow_matcher_v3/`, `flow_matcher_v3_ode_selectable/`

---

## 5. iMFTrajectoryModel — Mean Flow wrapper (Gen8 / state-only)

**Files**: `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` (state-only),
`imf_visual_aligning/models/imf_trajectory_model.py` (visual aligning)

### 5.1 State-only (avoiding)

```python
class iMFTrajectoryModel(nn.Module):
    def __init__(self, state_dim, seq_len, ..., imf_backbone='unet', dit_patch_size=1):

        if imf_backbone == 'dit':
            self.velocity_net = IMFDiTTrajectory(
                horizon=seq_len,
                transition_dim=state_dim,
                patch_size=dit_patch_size,
                ...
            )
        elif imf_backbone == 'unet':
            self.velocity_net = Flow_matcher_U_Net_v2(
                horizon=seq_len,
                transition_dim=state_dim,
                cond_dim=state_dim,    # accepted but ignored by Flow_matcher_U_Net_v2
            )

        # aux v-head for dual_head mode (optional; only used when dual_head=True)
        self.v_head = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.Mish(),
            nn.Linear(state_dim, state_dim),
        )
```

For state-only avoiding: `state_dim=6`, `seq_len=8`.

### 5.2 Visual aligning (Gen8)

```python
class iMFTrajectoryModel(nn.Module):
    def __init__(self, ...):
        if vis_config:
            from .visual_unet import VisualUNet
            self.velocity_net = VisualUNet(vis_config)   # 9D visual, FiLM ON
        else:
            self.velocity_net = Flow_matcher_U_Net_v2(
                horizon=seq_len, transition_dim=state_dim, ...
            )
```

The iMF objective wraps `velocity_net` with a JVP computation:
```python
u, jvp_u = torch.func.jvp(
    lambda x: self.velocity_net(x, cond, t, ...),
    primals=(x_t,), tangents=(v_t,)
)
loss = MSE(u - target) + lambda * MSE(jvp_u - d_target/dt)
```

---

## 6. IMFDiTTrajectory — Transformer backbone (U6, state-only)

**File**: `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py`

Faithful port of the official `imfDiT` from 2D images to 1D trajectories.

### 6.1 Architecture differences from UNet

```
UNet1D:
  - Conv1d with stride-2 downsampling (spatial hierarchy)
  - Skip connections at each resolution
  - Time/FiLM conditioning via additive bias in every ResBlock
  - Fixed receptive field via kernel_size=5
  - O(H) compute

IMFDiT:
  - TrajPatchEmbedder: Linear(patch_size*D → hidden_size) over each patch of H steps
  - Self-attention blocks: RoPE + QK-RMSNorm + SwiGLU
  - Full attention over all H/patch_size tokens → O(H²) compute, global receptive field
  - Conditioning tokens for (h, omega, t_min, t_max) prepended to sequence
  - Zero-init residual vector-gates (per-head gating)
  - Dual heads: shared backbone → u-head + v-head (native iMF)
  - No skip connections
```

### 6.2 How trajectory dim works in DiT

```python
class TrajPatchEmbedder(nn.Module):
    def __init__(self, horizon, transition_dim, patch_size, hidden_size):
        self.proj = nn.Linear(patch_size * transition_dim, hidden_size)

    def forward(self, x):
        # x: (B, H, transition_dim)
        # reshape to (B, H/patch_size, patch_size*transition_dim)
        # linear → (B, H/patch_size, hidden_size)
```

`transition_dim` enters as the feature size per timestep. With `patch_size=1`:
- Each of the H trajectory steps becomes one token of size `transition_dim`
- Linear: `transition_dim → hidden_size` → same global hidden_size regardless of `transition_dim`
- So the DiT is also NOT dim-agnostic: changing `transition_dim` changes the input Linear weight shapes

The DiT is used only for state-only avoiding (transition_dim=6) in the current codebase.

---

## 7. FiLM: exactly how visual conditioning enters the UNet

### 7.0 Lineage — who designed what

**Short answer: the conditioning MECHANISM is inherited from Janner's returns conditioning.
The visual signal (camera → ResNet → 128D) is FM-PCC's own design.
D3IL and DPCC have zero FiLM, zero cameras.**

#### What D3IL / DPCC / Janner each contributed

> **Correction**: D3IL DOES have visual aligning (both cameras, confirmed in
> `d3il/simulation/aligning_sim.py:88-121`). The key distinction is HOW the visual
> signal enters the model — D3IL uses early fusion (visual → obs input), not FiLM
> (visual → denoiser modulation).

| Component | D3IL (visual aligning) | DPCC | Janner diffuser | FM-PCC (visual aligning) |
|---|---|---|---|---|
| Camera input | YES — bp + inhand cameras | NONE | NONE | YES — same 2 cameras |
| Visual encoder | `MultiImageObsEncoder` (ResNet-18) | NONE | NONE | SAME `MultiImageObsEncoder` (inherited from D3IL) |
| Where visual goes | into **obs input** (early fusion) | N/A | N/A | into **denoiser ResBlocks** (additive bias) |
| DDPM architecture | action-level MLP DDPM, takes `(x, t, s, g)` — s includes visual features | state trajectory diffuser (no visual) | trajectory diffuser + returns cond | trajectory diffuser + visual cond (FiLM-style) |
| Conditioning mechanism | cat visual+state → concatenated obs input to DDPM | `apply_conditioning` inpainting only | `returns_condition`: scalar → MLP → cat with t → ResBlock bias | reused Janner's mechanism, pointed at visual embedding |
| FiLM into denoiser | **NO** — visual is pre-encoded into obs, not a denoiser modulator | NONE | scalar return → MLP → cat t (not visual) | YES — visual embedding → cond_mlp → cat t → every ResBlock |

**D3IL visual aligning flow** (`ddpm_vision_agent.py:DiffusionPolicy.forward()`):
```python
# D3IL: cameras + state → obs_encoder → combined obs → DDPM action input
obs_dict = {"agentview_image": agentview_image,
            "in_hand_image": in_hand_image,
            "robot_ee_pos": state}
obs = self.obs_encoder(obs_dict)   # → (B*T, combined_dim) : visual+state merged into one obs
pred = self.model(obs, goal)       # action-level DDPM: model(x, t, s=obs, g=goal)
```

Visual features are concatenated INTO the state before the DDPM sees it. The DDPM model
(`diffusion_policy.py:Diffusion.p_mean_variance`) denoises `x` (action) conditioned on `s`
(which already contains the visual encoding as part of obs). The denoiser backbone never
separately receives visual info; it receives one fused obs vector.

**FM-PCC visual aligning flow** (`visual_unet.py:VisualUNet.forward()`):
```python
# FM-PCC: cameras → visual latent → FiLM → EVERY ResBlock of UNet
visual_cond = self.encode_visual(bp_imgs, inhand_imgs)  # → (B, 128)  — separate from obs
out = self.backbone(x, visual_cond, t)                  # UNet receives visual SEPARATELY
# Inside backbone: t = cat[time_emb, cond_mlp(visual_cond)] → modulates every ResBlock
```

Visual features are kept SEPARATE from the trajectory and injected into the UNet's
intermediate feature maps via the time embedding concatenation. The trajectory tensor
itself never contains visual information (9D, robot-only).

#### The Janner `returns_condition` — the exact ancestor in the code

Now the key question: **was FiLM already in Janner and we just activated it,
or did we add new code that happened to fit perfectly?**

**Answer: we added new code (`cond_mlp` + `use_cond_projection`) that follows
Janner's `returns_condition` pattern exactly. It was NOT already there waiting to
be activated. But the architecture made it trivially natural to add.**

Janner's diffuser had this pattern for scalar reward conditioning
(`diffuser_visual_aligning/models/unet1d_temporal_cond.py:140-149`):

```python
# Janner's EXISTING code (returns conditioning for RL):
embed_dim = dim           # 128 (base)
if self.returns_condition:
    self.returns_mlp = nn.Sequential(
        nn.Linear(1, dim),           # scalar 1D → 128D
        nn.Mish(),
        nn.Linear(dim, dim * 4),
        nn.Mish(),
        nn.Linear(dim * 4, dim),     # → 128D
    )
    self.mask_dist = Bernoulli(...)  # CFG dropout
    embed_dim += dim                 # → 256 total
# forward: t = cat[t, returns_mlp(returns)]
```

FM-PCC added a NEW parallel block for visual conditioning:

```python
# FM-PCC ADDED code (not in original Janner):
if use_cond_projection and cond_dim > 0:         # ← NEW flag, not in Janner
    self.cond_mlp = nn.Sequential(               # ← NEW module
        nn.Linear(cond_dim, dim),   # 128D visual → 128D
        nn.Mish(),
        nn.Linear(dim, dim),        # → 128D
    )
    cond_embed_dim = dim
else:
    self.cond_mlp = None
    cond_embed_dim = 0

embed_dim = dim + cond_embed_dim                 # 128 (no FiLM) or 256 (FiLM)
# forward: if cond_mlp and isinstance(cond, Tensor): t = cat[t, cond_mlp(cond)]
```

Side by side:

| | Janner `returns_condition` (existing) | FM-PCC `use_cond_projection` (added) |
|---|---|---|
| In original Janner? | YES | NO — new code added by us |
| Input signal | scalar return value (1D) | visual embedding (128D from ResNet) |
| MLP | `Linear(1→128→512→128)` | `Linear(128→128→128)` |
| Output | (B, 128) → cat with t | (B, 128) → cat with t |
| embed_dim impact | `+dim` → 256 total | `+dim` → 256 total |
| Where it fires | always (if returns given) | only if `isinstance(cond, Tensor)` |
| CFG dropout | yes (Bernoulli mask) | no |

Why it fit so perfectly: Janner's returns conditioning was the **existence proof** for the
additive-bias pattern. The architecture is explicitly parametrised: `embed_dim` is built by
summing contributions from time, optional returns, and optional cond. Adding a third slot
(`cond_embed_dim`) required only a new `__init__` flag + `cond_mlp` block + one line in
`forward`. No residual block code changed. The architecture was designed to accept this kind
of extension — it wasn't coincidence, it was good parametrisation.

**What FM-PCC genuinely added** (not in D3IL, DPCC, or Janner):
1. `use_cond_projection` flag + `cond_mlp` block in the UNet constructor (new code)
2. `MultiImageObsEncoder` → 128D latent fed as a TENSOR (not obs dict) to the UNet (new routing)
3. `VisualUNet` wrapper gluing cameras → `encode_visual()` → `backbone(x, visual_cond, t)` (new)
4. Keeping trajectory tensor 9D (robot-only) instead of concatenating visual into trajectory

**What FM-PCC inherited from D3IL** (not new):
1. `MultiImageObsEncoder` class itself — literally the same file (`agents/models/vision/multi_image_obs_encoder.py`)
2. Dual camera setup: bp_cam (bird's eye) + inhand_cam (wrist)
3. BGR image format convention
4. delta integration pattern in eval (`des_robot_pos += pred_action`)

**What FM-PCC inherited from Janner** (not new):
1. The embed→cat→ResBlock-additive-bias mechanism — the "FiLM" pattern

#### True FiLM vs what we actually implement — are we FiLM or not?

**Short answer: No, we are not real FiLM. We played a trick that gets visual info into every layer cheaply, but it is strictly weaker than FiLM.**

True FiLM (Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer"):
```
True FiLM:
    γ = proj_scale(c)    # (B, out_ch) — learned SCALE from condition c
    β = proj_shift(c)    # (B, out_ch) — learned SHIFT from condition c
    y = γ ⊙ Conv(x) + β  # element-wise SCALE *then* SHIFT per channel
```
Two separate projections from c, applied multiplicatively AND additively.

What we actually do in every ResBlock:
```
Our implementation:
    t   = cat[time_emb(128D), cond_mlp(visual)(128D)]  # → (B, 256)
    out = Conv(x) + Linear(256 → out_ch)(t)             # additive bias only
          └── this single Linear mixes time AND visual together
              into one bias vector. No scale term at all.
```

The "trick": by concatenating visual into `t` BEFORE the ResBlock, the ResBlock's
existing `time_mlp` Linear transparently receives visual info — no ResBlock code
changed. The Linear `(256→out_ch)` learns to mix time and visual jointly into a
single channel-wise bias. From the ResBlock's perspective nothing changed; from
outside we widened `t` and let the Linear sort it out.

**What this means in practice:**

| Property | True FiLM | Our approach |
|---|---|---|
| Visual affects activations | YES | YES |
| Mechanism | `γ(c) ⊙ x + β(c)` — scale + shift | `bias(cat[t, v])` — shift only |
| Scale term (γ) | YES — suppresses/amplifies channels | **NO** |
| Shift term (β) | YES — offsets channels | YES (implicit in bias) |
| Time/visual entangled | NO — separate projections | **YES** — one Linear mixes both |
| Extra parameters | 2 × Linear(cond_dim → out_ch) per layer | 1 × Linear(cond_dim→dim) total (shared via cat) |
| Can silence a channel conditionally | YES (γ=0 → channel ignored) | NO (bias only shifts) |
| Name in our codebase | "FiLM-style" | also called "FiLM-style" (inaccurate) |

**Implication**: our visual conditioning can only ADD a learned offset to each channel
at each ResBlock. It cannot SUPPRESS a channel that becomes irrelevant given the visual
context. True FiLM can do that via the scale gate γ(c)→0. Ours cannot.

**Why it's called FiLM anyway**: the term is used loosely in the community for any
"inject conditioning into intermediate network layers" approach. Strictly it should be
called **additive bias conditioning via time-embedding concatenation**. We inherited
the loose naming from related work (e.g. Janner's returns_condition is also called
"conditioning" even though it is the same additive bias pattern).

**Is it effective despite not being real FiLM?** Empirically yes — the visual info
does reach every layer and the model learns to use it. The lack of scale gate means
the conditioning is softer, but the trajectory model still benefits from having the
camera signal present at every denoising step rather than only at the input.
The "trick" works, it's just not what FiLM strictly means.

---

### 7.4 Is our way the proper way? — Full comparison of visual injection approaches

**No — our approach is the weakest form of denoiser-injection.** Here is the full
ranking of approaches from most principled to cheapest:

---

#### Approach A: Cross-Attention (Diffusion Policy / Chi et al. 2023)

```
cameras → ViT or ResNet → (B, T_vis, feat_dim)   ← sequence of visual tokens

In each UNet ResBlock (or Transformer block):
    Q = Linear(x_features)                        ← trajectory features as queries
    K = Linear(visual_tokens)                     ← visual tokens as keys
    V = Linear(visual_tokens)                     ← visual tokens as values
    out = softmax(QK^T / √d) · V                  ← attend to relevant visual regions
    x = x + out                                   ← add attended visual info
```

- Visual info reaches every layer with FULL spatial attention
- The model can attend to SPECIFIC image regions (e.g. look at the box location)
- Scale + spatial selectivity — the most expressive form
- **Used by**: Diffusion Policy (Chi et al. 2023), many modern robot learning papers
- **Cost**: large additional parameter budget (Q/K/V projections per layer); slower

---

#### Approach B: True FiLM per ResBlock (Perez et al. 2018)

```
cameras → encoder → c: (B, cond_dim)

In each ResBlock independently:
    γ = Linear_scale(c)   → (B, out_ch)   ← learned SCALE per channel from visual
    β = Linear_shift(c)   → (B, out_ch)   ← learned SHIFT per channel from visual
    out = γ.unsqueeze(-1) * Conv(x) + β.unsqueeze(-1)   ← scale THEN shift
```

- **Scale** γ(c): can suppress irrelevant channels to near-zero, amplify relevant ones
- **Shift** β(c): can offset channel means
- Visual info is cleanly separated from time — each ResBlock gets its own `Linear_scale`
  and `Linear_shift` from the visual latent, independent of the time embedding
- **Used by**: FiLM paper, many visual reasoning / VQA systems, some robot policies
- **Cost**: 2 × `Linear(cond_dim → out_ch)` per ResBlock — e.g. 8 ResBlocks = 16 extra Linears
- **What we would need to change**: replace `time_mlp(t)` bias with `γ*Conv(x) + β`
  where γ,β come from separate per-ResBlock projections of visual latent

---

#### Approach C: AdaGN / AdaLayerNorm (DiT / DDPM-v2 style)

```
cameras → encoder → c: (B, cond_dim)

In each ResBlock's GroupNorm or LayerNorm:
    γ, β = Linear(c).chunk(2)   → each (B, out_ch)
    out = γ * GroupNorm(x) + β  ← scale+shift the NORMALIZED activations
```

- Scale+shift applied at normalization, not conv output — slightly different expressiveness
- Used by: DiT, DDPM-v2, many modern diffusion architectures
- Cleaner than our approach (separate scale+shift), cheaper than cross-attention
- **Cost**: 1 × `Linear(cond_dim → 2*out_ch)` per ResBlock

---

#### Approach D: Our approach — additive bias via time-embedding concatenation

```
cameras → encoder → v: (B, 128)
cond_emb = cond_mlp(v)         → (B, 128)
t = cat[time_emb, cond_emb]    → (B, 256)   ← ONE concatenation, happens once

In each ResBlock (unchanged code):
    bias = Linear(256 → out_ch)(t)            ← mixes time AND visual jointly
    out  = Conv(x) + bias                     ← additive shift only
```

- **No scale** — cannot suppress channels
- **Time and visual entangled** — one Linear mixes both; the ResBlock cannot
  distinguish "time contribution" from "visual contribution"
- **Only ONE extra Linear total** (`cond_mlp`, 2 layers, shared globally) vs
  per-ResBlock Linears in B and C
- **Zero code changes to ResBlocks** — purely an embed_dim widening trick
- **Cost**: cheapest possible — adds `cond_mlp` (128→128→128, ~33K params) plus
  wider time_mlp Linears in each ResBlock (embed 128→256, ~33K extra per ResBlock)

---

#### Approach E: D3IL early fusion (input concatenation before DDPM)

```
cameras → MultiImageObsEncoder → (B*T, feat_dim)
visual_feat + robot_state → obs_encoder → fused obs: (B, obs_dim)

DDPM denoiser takes: model(x_noisy, t, s=fused_obs, goal)
  → x_noisy is action-level (1D), s is the visual-enriched state
```

- Visual info is encoded ONCE before denoising starts, stays fixed
- The DDPM denoiser sees visual as part of its state input `s`, not as a separate modulator
- Works fine for action-level DDPM (D3IL) where `s` is just a side input
- **Does NOT work for trajectory UNet** the same way — the visual would need to be in
  the trajectory tensor (adding dims to transition_dim) or passed separately as `s`
- Visual is static: the same visual encoding is used for all denoising steps, not
  re-injected into intermediate features

---

#### Comparison table

| Approach | Scale? | Shift? | Time/visual entangled? | Visual reaches all layers? | Extra params | Spatial attention? |
|---|---|---|---|---|---|---|
| A: Cross-attention | implicit | implicit | NO (separate) | YES | Large (QKV per layer) | YES |
| B: True FiLM | YES | YES | NO (separate) | YES | Medium (2 Lin per layer) | NO |
| C: AdaGN/AdaLN | YES | YES | NO (separate) | YES | Medium (1 Lin per layer) | NO |
| **D: Ours** | **NO** | YES (weak) | **YES** | YES | Small (1 global cond_mlp) | NO |
| E: D3IL early fusion | NO | implicit | NO | **NO** (input only) | Minimal | NO |

**Our approach (D) is ranked 4th out of 5.**

It is better than early fusion (E) because the visual signal reaches every denoising
layer rather than being a fixed input. It is weaker than A/B/C because:
1. No scale term → cannot gate/suppress channels
2. Visual entangled with time in a single Linear → the network cannot cleanly
   separate "what time step am I at" from "what does the camera show"

#### VERDICT: did we invent something, copy something, or build a shithole?

**Invented? No. Novel application? Somewhat. Shithole? No — but not ideal for visual tasks.**

---

**The mechanism is standard DDPM conditioning from 2020–2022.**

The exact pattern — embed a conditioning signal, concatenate with time embedding, use as
additive bias in ResBlocks — appears in:

| Paper | Year | What they condition on | Our equivalent |
|---|---|---|---|
| Ho et al. "DDPM" | 2020 | time step only (not cond) — but establishes the ResBlock-additive-bias structure | the ResBlock structure we inherited |
| Ho & Salimans "CFG" / Nichol "GLIDE" | 2022 | class label → embed → cat with time → ResBlock bias | us: visual latent → cond_mlp → cat with time → ResBlock bias |
| Janner "Diffuser" | 2022 | scalar return → embed → cat with time → ResBlock bias (`returns_condition`) | our DIRECT ancestor in this repo |

Our `use_cond_projection` is a copy-paste of Janner's `returns_condition` mechanism,
pointed at a 128D visual latent instead of a 1D scalar return.

**What IS the FM-PCC contribution (if any):**
- Applying this mechanism to a **trajectory** UNet (not action-level DDPM)
- Using a **visual** signal (not scalar returns) in a **robot manipulation** context
- Keeping the trajectory tensor pure (9D) and injecting visual SEPARATELY
  (D3IL concatenates visual into obs instead)

The specific combination of [Janner trajectory diffuser + visual conditioning via
time-embedding concatenation + PCC projector for robot manipulation] had not been
published when this work was done. The mechanism is not novel; the application is.

**Is it a shithole?**

No, for three reasons:

1. **It works** — Classifier-Free Guidance (one of the most cited diffusion papers)
   uses exactly this pattern for class conditioning. If it were fundamentally broken,
   half of image diffusion would be broken too.

2. **It's appropriate for our task scale** — the aligning task has one box and one
   target. Their positions are captured in a 128D pooled ResNet latent. You don't need
   spatial attention to say "box is upper-left → shift trajectory leftward"; additive
   bias can encode that association through training.

3. **The lack of scale gate is not catastrophic** — the model compensates by learning
   larger bias magnitudes for high-salience channels. It's weaker than true FiLM but
   not disabled.

**Where it would genuinely hurt:**

- Tasks with many objects or fine spatial detail (e.g. "pick the red cube from a pile
  of 10 cubes") — pooled 128D latent loses spatial structure; need cross-attention
- Tasks where some channels MUST be suppressed conditionally — additive bias can't zero out
- Tasks where the scene changes MID-trajectory (the pooled visual is encoded ONCE at
  the start of each predict() call, re-encoded only when called again)

**Bottom line:** standard technique, applied reasonably. Not cutting-edge, not broken.
The label "FiLM" is the main lie — it's additive-bias conditioning, full stop.

---

### 7.5 Head-to-head: D3IL visual aligning vs FM-PCC "Fake FiLM"

#### Full data flow — D3IL visual aligning

```
CAMERAS (at eval step t):
  bp_image:     (H=96, W=96, 3)  BGR, /255
  inhand_image: (H=96, W=96, 3)  BGR, /255

  │  transpose (2,0,1) → (3,96,96) each
  ▼
MultiImageObsEncoder  [d3il/agents/models/vision/multi_image_obs_encoder.py]
  agentview_image → ResNet-18 stem → Linear(512→64) → (B*T, 64)
  in_hand_image   → ResNet-18 stem → Linear(512→64) → (B*T, 64)
  cat                                                → (B*T, 128)
  view+pool                                          → (B, T_win, 128)
  ↓ stored in a deque window
  input_state = (bp_seq, inhand_seq, des_robot_pos_seq)   # kept as tuple

  │
  ▼ DiffusionPolicy.forward()  [ddpm_vision_agent.py:63-80]
  agentview_image  (B, T, 3, 96, 96) → view → (B*T, 3, 96, 96)
  in_hand_image    (B, T, 3, 96, 96) → view → (B*T, 3, 96, 96)
  state (des_robot_pos) (B, T, 3)    → view → (B*T, 3)
  obs_dict = {agentview_image, in_hand_image, robot_ee_pos}
  obs = obs_encoder(obs_dict)          → (B*T, obs_encoded_dim)
  obs = obs.view(B, T, -1)             → (B, T, obs_encoded_dim)   ← FUSED vector
  │
  │  obs = SINGLE fused vector: [visual_features | robot_state]
  │  Visual info is NOW INSIDE the obs tensor.
  │  The DDPM denoiser receives this as its state conditioning s.
  ▼
Diffusion.p_mean_variance()  [diffusion_policy.py:117-142]
  noise = self.model(x_noisy, t_diff, s=obs, g=goal)
  ↑
  model is an MLP or small UNet that denoises action x_noisy
  s = fused obs (visual already in here)
  t_diff = diffusion timestep

  The MLP/model receives visual as PART OF s — one flat vector.
  No separate visual pathway inside the denoiser.
  Visual is STATIC: obs was encoded once, same vector used for ALL denoising steps.

OUTPUT: x_0 = 3D action delta  (B, 3)
  integration: pred_action = x_0 + des_robot_pos  ← single step, action-level
```

Key facts about D3IL:
- Action-level DDPM: denoises a **single 3D action**, not a trajectory
- Visual info enters as part of obs `s`, concatenated with robot state before DDPM
- The denoiser backbone (MLP or small net) has NO separate visual pathway
- Visual encoding is computed ONCE per `predict()` call, static across all DDPM steps
- No trajectory horizon (H), no apply_conditioning, no PCC projector

---

#### Full data flow — FM-PCC "Fake FiLM" (visual aligning)

```
CAMERAS (at eval step t):
  bp_image:     (T_win, 3, 96, 96)  BGR, /255  (window of T_win frames)
  inhand_image: (T_win, 3, 96, 96)  BGR, /255

  │
  ▼ VisualUNet.encode_visual()  [visual_unet.py:92-103]
  B=1, T=T_win, C=3, H=96, W=96
  obs_dict = {agentview_image: (B*T,3,96,96), in_hand_image: (B*T,3,96,96)}
  obs_encoder(obs_dict) → (B*T, 128)        ← same MultiImageObsEncoder as D3IL
  .view(B, T, 128).mean(dim=1)  → (B, 128)  ← POOL over T_win → single latent
  visual_cond: (B, 128)                      ← SEPARATE from trajectory

  trajectory x: (B, H=8, 9)  ← [act(3)|des_c_pos(3)|c_pos(3)], robot-only
  obs_6d = [des_c_pos(3)|c_pos(3)]
  apply_conditioning: x[:, 0, 3:] = obs_6d  ← pin obs at t=0 of trajectory

  │
  ▼ VisualUNet.forward() → backbone(x, visual_cond, t_diff)
  x rearranged: (B, 9, H=8)    ← channel first for Conv1d

  ── CONDITIONING COMPUTED ONCE, USED EVERYWHERE ──────────────────────────
  time_emb  = time_mlp(t_diff)            → (B, 128)   sinusoidal → MLP
  cond_emb  = cond_mlp(visual_cond)       → (B, 128)   2-layer MLP  ← VISUAL
  t = cat[time_emb, cond_emb]             → (B, 256)   ENTANGLED
  ─────────────────────────────────────────────────────────────────────────

  ── ENCODER ──────────────────────────────────────────────────────────────
  down[0]: ResBlock(9→128):
    conv_out = Conv1dBlock(x)              → (B, 128, 8)
    bias     = Linear(256→128)(t)         → (B, 128, 1)  ← visual inside t
    out      = conv_out + bias             → (B, 128, 8)  ← additive only
    ResBlock(128→128) same pattern        → (B, 128, 8)  push skip_0
    Downsample                            → (B, 128, 4)

  down[1]: ResBlock(128→256):
    bias = Linear(256→256)(t)             ← SAME t, visual still here
    out  = Conv(x) + bias                 → (B, 256, 4)  push skip_1
    Downsample                            → (B, 256, 2)

  down[2], down[3], mid, up[0..2]: same pattern, SAME t at every block

  ── DECODER ──────────────────────────────────────────────────────────────
  ... skip connections + ResBlocks with same t ...

  final_conv: Conv1d(128→9)               → (B, 9, H=8)
  rearrange                               → (B, 8, 9)   ← TRAJECTORY

OUTPUT: x_0 = trajectory (B, 8, 9) = H steps of [act(3)|des_c_pos(3)|c_pos(3)]
  execute: act_0 = x_0[:, 0, :3]
  integration: des_robot_pos += act_0
  PCC projector constrains x_0 before execution
```

Key facts about FM-PCC:
- Trajectory-level: denoises **H=8 steps at once**, plans ahead
- Visual info kept SEPARATE from trajectory tensor (9D, robot-only)
- Visual enters as additive bias at EVERY ResBlock via the entangled `t` vector
- Same `t` (with visual mixed in) is used at every denoising step AND every ResBlock
- apply_conditioning pins the current obs at trajectory position t=0 at every step

---

#### Side-by-side comparison

```
                        D3IL visual aligning         FM-PCC "Fake FiLM"
─────────────────────────────────────────────────────────────────────────────
Cameras                 bp + inhand (same hardware)  bp + inhand (same hardware)
Visual encoder          MultiImageObsEncoder          SAME class, same weights init
Visual latent           (B, obs_encoded_dim)          (B, 128)  ← mean-pooled
─────────────────────────────────────────────────────────────────────────────
WHERE visual enters     into obs s                   into time embedding t
                        → BEFORE the denoiser        → INSIDE every ResBlock
─────────────────────────────────────────────────────────────────────────────
Denoiser sees visual    as one flat input vector     as a bias offset per channel
                        (fused with robot state)     (entangled with time step)
─────────────────────────────────────────────────────────────────────────────
Trajectory              NONE — single action (3D)    H=8 steps × 9D
─────────────────────────────────────────────────────────────────────────────
Visual re-encoded per   denoising step? NO — encoded ONCE, static throughout
 denoising step
─────────────────────────────────────────────────────────────────────────────
Visual re-encoded per   NEW predict() call? YES (both) ← same
 env step
─────────────────────────────────────────────────────────────────────────────
Scale gate (γ)          NO                           NO  (neither is true FiLM)
─────────────────────────────────────────────────────────────────────────────
Visual + state mixed    YES (inside obs encoder)     NO (kept separate — visual
                                                      goes to t, state goes to
                                                      trajectory tensor)
─────────────────────────────────────────────────────────────────────────────
Can visual suppress a   NO                           NO (only additive shift)
  channel?
─────────────────────────────────────────────────────────────────────────────
Planning horizon        0 (reactive)                 H=8 (MPC-style)
─────────────────────────────────────────────────────────────────────────────
PCC safety projector    NO                           YES
─────────────────────────────────────────────────────────────────────────────
```

#### What each approach CAN and CANNOT express

**D3IL (early fusion):**
- CAN: "given these camera features AND this robot state combined, predict action"
- CAN: the fused obs vector can encode complex joint patterns across visual + state
- CANNOT: selectively re-condition the denoiser at different noise levels
  (the same fused obs is passed at every DDPM step, unchanging)
- CANNOT: plan ahead (single action, not a trajectory)
- CANNOT: enforce safety constraints (no projector)

**FM-PCC "Fake FiLM":**
- CAN: "shift trajectory activations at every scale of the UNet based on what camera sees"
- CAN: plan H=8 steps ahead and apply safety projection
- CAN: keep trajectory tensor clean (no visual dims) — cleaner projector constraints
- CANNOT: scale/suppress channels based on visual content (additive only)
- CANNOT: attend to specific image regions (pooled 128D latent loses spatial structure)
- CANNOT: disentangle "time conditioning" from "visual conditioning" (entangled in t)
- CANNOT: update visual mid-trajectory (visual pooled once per predict() call)

#### The real question: which is better for our task?

For the aligning task (push a box to a target position):
- The visual info needed: roughly "where is box? where is target?" — positional, not spatial-detailed
- Both approaches can encode this from the ResNet-pooled 128D latent
- FM-PCC gains from trajectory planning (H=8) and PCC safety — these matter more than FiLM quality
- D3IL is simpler and works for action-level; FM-PCC adds value via the trajectory+projector, not via better visual conditioning

**The visual conditioning method (Fake FiLM) is not FM-PCC's competitive advantage.**
The advantage comes from: trajectory diffusion + PCC projector + flow matching.
The visual conditioning is "good enough for this task" — not a strength, not a critical weakness.

---

#### Why we chose Approach D anyway

1. **Zero ResBlock changes** — the Janner `returns_condition` pattern was already there;
   we just widened `embed_dim` from 128 to 256. Approach B/C would require rewriting
   ResidualTemporalBlock.
2. **Empirically sufficient** — our tasks (aligning 9D trajectory) don't need spatial
   attention over visual features; the scene is summarised by a 128D pooled latent.
3. **Parameter efficient** — ~33K extra params in `cond_mlp` vs hundreds of thousands
   for per-ResBlock FiLM projections.
4. **Not our insight** — Janner used this for returns conditioning; we inherited the
   pattern and pointed it at cameras.

#### What proper FiLM for our UNet would look like (if we ever upgrade)

Change `ResidualTemporalBlock` from:
```python
# current (additive bias via entangled t):
out = self.blocks[0](x) + self.time_mlp(t)   # time_mlp: Linear(embed_dim→out_ch)
```
to:
```python
# true FiLM:
gamma = self.film_scale(visual_cond)          # Linear(cond_dim→out_ch), per ResBlock
beta  = self.film_shift(visual_cond)          # Linear(cond_dim→out_ch), per ResBlock
time_bias = self.time_mlp(time_emb)           # Linear(128→out_ch), time only
out   = gamma * self.blocks[0](x) + beta + time_bias
```
This requires: separate `time_emb` (not concatenated with visual), two new Linears per
ResBlock, and `visual_cond` passed explicitly alongside `time` to every `ResidualTemporalBlock.forward()`.

FiLM = "Feature-wise Linear Modulation" — but our implementation is additive only.

### 7.1 Visual embedding pipeline

```
bp_images  (B, T_win, 3, 96, 96)   bird's-eye camera
inhand_imgs(B, T_win, 3, 96, 96)   wrist camera

MultiImageObsEncoder:
  agentview → ResNet-18 stem (ImageNet pretrained) → Linear → (B*T, 64)
  in_hand   → ResNet-18 stem (ImageNet pretrained) → Linear → (B*T, 64)
  concat    → (B*T, 128)
  view+mean → (B, T, 128) → mean over T_win → (B, 128)   ← visual_cond

VisualUNet.encode_visual() → (B, 128)
```

### 7.2 FiLM injection path

```
UNet1DTemporalCondModel.forward():

  t = time_mlp(timestep)          # (B, 128) — sinusoidal pos emb → 2xMLP

  # FiLM (only when use_cond_projection=True):
  cond_emb = cond_mlp(visual_cond)  # (B, 128) → 2xLinear+Mish → (B, 128)
  t = cat([t, cond_emb], dim=-1)    # (B, 256)  ← time + visual merged

  # Every ResidualTemporalBlock receives this 256D vector:
  ResidualTemporalBlock.forward(x, t):
    out = Conv1dBlock(x) + time_mlp(t)   # time_mlp: Linear(256→out_ch) → (B,out_ch,1)
    out = Conv1dBlock(out)               # broadcast over H → additive channel bias
    return out + residual(x)
```

**cond_mlp architecture** (`use_cond_projection=True`):
```python
self.cond_mlp = nn.Sequential(
    nn.Linear(cond_dim, dim),   # 128 → 128
    nn.Mish(),
    nn.Linear(dim, dim),        # 128 → 128
)
```

**What FiLM achieves**: the visual embedding of the current scene (box position, target
position, orientations — all implicit in pixel space) modulates the channel activations
of every ResidualTemporalBlock. The model learns to use visual context to steer which
trajectory features it emphasises at each denoising step.

**What FiLM does NOT achieve**: explicit goal position encoding. The visual embedding is
a flat 128D latent. The model must learn to associate pixel patterns with trajectory
corrections entirely through gradient descent on the demonstration data.

### 7.3 Non-visual mode — FiLM is structurally absent

```python
# Construction:
use_cond_projection=False → self.cond_mlp = None → cond_embed_dim = 0
embed_dim = dim + 0 = dim = 128       # NOT 256

# Forward:
if self.cond_mlp is not None and ...:  # fails: cond_mlp is None
    ...  # FiLM path never reached

# ResidualTemporalBlock receives t of shape (B, 128), NOT (B, 256)
# → time_mlp: Linear(128 → out_ch)  (different from visual's Linear(256 → out_ch))
```

A visual checkpoint (embed_dim=256) CANNOT be loaded into a non-visual model (embed_dim=128).
The weight shapes of `time_mlp` inside every ResidualTemporalBlock differ.

---

## 8. How transition_dim changes the model — the rigidity answer

The question: "how the fuck can the UNet handle 9D vs 23D?"

Answer: **it can't, unless it was built for that dim from the start**.

```python
# Building for 9D:
UNet1DTemporalCondModel(transition_dim=9, ...)
# → first Conv1d: in_channels=9, out_channels=128
# → final Conv1d: in_channels=128, out_channels=9
# Weight shapes: [128, 9, kernel_size] and [9, 128, 1]

# Building for 23D:
UNet1DTemporalCondModel(transition_dim=23, ...)
# → first Conv1d: in_channels=23, out_channels=128
# → final Conv1d: in_channels=128, out_channels=23
# Weight shapes: [128, 23, kernel_size] and [23, 128, 1]
```

The 23D model's first Conv1d has 23 input channels. Pass it 9D input → crashes with:
`RuntimeError: Expected input channels=23 but got 9`.

The old non-visual 23D model works because `VisualUNet.__init__` builds
`UNet1DTemporalCondModel(transition_dim=23)` when `obs_dim=20`. It handles 23D because
it was BUILT for 23D. It's a DIFFERENT model from the 9D visual model. The only
architectural shared structure is the same class definition.

**Summary of rigidity**:

| Parameter | Rigid? | Effect of mismatch |
|---|---|---|
| `transition_dim` | YES | Shape mismatch on first/last Conv1d → crash |
| `horizon` (padded) | YES | Downsample/Upsample layer sizes wrong → crash |
| `embed_dim` (=dim + cond_embed) | YES | time_mlp Linear weight shape mismatch → crash |
| `dim` (base channel) | YES | All channel sizes wrong → crash |
| `dim_mults` | YES | Channel widths throughout UNet wrong → crash |

You cannot hot-swap trajectory dimensions in a trained model. Different `transition_dim`
= different model.

---

## 9. Current model taxonomy — FM-PCC 2026-06-26

| Namespace | Engine | Backbone | Task | traj_dim | FiLM | Status |
|---|---|---|---|---|---|---|
| `diffuser_visual_aligning` | GaussianDiffusion (DDPM) | VisualUNet→UNet1DTemporalCond | visual aligning | 9 | YES | Gen6V4 (older) |
| `fm_visual_aligning` | FlowMatchingODE | VisualUNet→UNet1DTemporalCond | visual aligning | 9 | YES | Gen7 (current) |
| `imf_visual_aligning` | iMFEngine | VisualUNet(→UNet1DTemporalCond) or iMFTrajectoryModel(VisualUNet) | visual aligning | 9 | YES | Gen8 (in-dev) |
| `diffuser_visual_avoiding` | GaussianDiffusion (DDPM) | VisualUNet→UNet1DTemporalCond | visual avoiding | 6 | YES | FM-PCC own |
| `fm_visual_avoiding` | FlowMatchingODE (VisualFlowMatching) | VisualUNet→UNet1DTemporalCond | visual avoiding | 6 | YES | FM-PCC own |
| `flow_matcher_v3_uav` | FlowMatchingODE | Flow_matcher_U_Net_v2 | UAV | 12 | NO | Gen11 active |
| `flow_matcher_v3_imeanflow` | iMFEngine | iMFTrajectoryModel(UNet or DiT) | state avoiding | 6 | NO | Gen8 iMF |
| `flow_matcher_v3` | FlowMatchingODE | Flow_matcher_U_Net_v2 | state avoiding | 6 | NO | Gen3 base |

Non-visual aligning (23D, WRONG) is only in `diffuser_visual_aligning` and
`fm_visual_aligning` via the `if_vision=False` + `obs_dim=20` path. No standalone
namespace for it. Any trained 23D checkpoint should NOT be used (see ALIGNING_EXPANSION.md).

---

## 10. apply_conditioning — how obs pins the trajectory

This is the DPCC-inherited inpainting mechanism. Separate from FiLM.

```python
def apply_conditioning(x, conditions, action_dim):
    # conditions: {0: obs_tensor}  — dict mapping timestep → obs slice
    # x: (B, H, transition_dim)
    for t, val in conditions.items():
        x[:, t, action_dim:] = val   # overwrite obs dims at timestep t
    return x
```

Called inside the sampling loop AT EVERY DENOISING STEP:
```python
for i in reversed(range(n_timesteps)):
    x = denoise_step(x, i)
    x = apply_conditioning(x, conditions, action_dim)  # ← re-pin every step
```

**What it does**: forces the obs slice at `t=0` to equal the actual current observation
at every denoising step. This anchors the start of the H-step trajectory to the real
robot state and lets the model plan the remaining H-1 steps freely.

**What it does NOT do**: condition the UNet directly. The UNet sees the pinned trajectory
as part of its noisy input `x`. The obs information enters the UNet through the trajectory
itself (in the input channels), not through a separate conditioning pathway.

---

## 11. The "non-visual UNet ignores cond" detail

The cond dict `{0: obs}` is passed around through `model(cond)` calls but:

```python
# In GaussianDiffusion / FlowMatchingODE:
trajectory = self.conditional_sample(cond, ...)  # cond = {0: obs_tensor}

def conditional_sample(self, cond, ...):
    x = torch.randn(B, H, transition_dim)      # start from noise
    x = apply_conditioning(x, cond, action_dim) # ← cond USED HERE to pin obs at t=0
    for step in reversed(steps):
        x = model.backbone(x, cond, step)       # ← cond PASSED BUT...
        x = apply_conditioning(x, cond, ...)    # ← cond USED HERE again
    return x

# Inside Flow_matcher_U_Net_v2.forward(x, cond, time):
    t = self.time_mlp(time)   # cond is received as a parameter...
    # but cond is never read
    # There is no 'if cond:' block
```

`cond` (the dict) is carried as a positional arg through the call chain because the
function signature requires it, but `Flow_matcher_U_Net_v2` and the non-visual
`UNet1DTemporalCondModel` (cond_mlp=None) never look at it.

---

## 12. Dimension how-to: changing trajectory dim

If you need to change trajectory dimensions (e.g. fix non-visual 23D→9D):

1. **Dataset**: change `OBS_DIM` / `TRAJ_DIM` in the dataset class, rebuild normalizers
2. **Config**: `obs_dim = 6` (not 20), `action_dim = 3`
3. **VisualUNet**: `transition_dim = action_dim + obs_dim = 9`
4. **Training**: from scratch — weights incompatible with old checkpoint
5. **Eval**: load new checkpoint, confirm `obs_normalizer.mins.shape[0] == 6` (not 20)
6. **Projector**: `trajectory_dim=9` (not 23) in `setup_dpcc_projector`

The projector also uses `trajectory_dim` to size constraint vectors. A 9D projector built
for a 23D trajectory will crash (or silently produce wrong results).

---

---

## 13. WHY changing transition_dim does NOT require changing the backbone code

This is the most important conceptual point. The answer is that **the backbone operates in a
fixed hidden-feature space**. The trajectory dimension D only appears at two boundary layers:
the input projection and the output projection. Everything between them is D-agnostic.

### The two-boundary principle

```
trajectory (B, H, D)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ INPUT PROJECTION  Conv1d(D → 128, kernel=5)                      │
│   → this layer KNOWS D; its weight shape is (128, D, 5)         │
│   → maps from D-dim trajectory space into 128-dim feature space  │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  HIDDEN FEATURE SPACE  128 → 256 → 512 → 1024 → 512 → 256 → 128 │
│   All ResBlocks, Downsample, Upsample, skip-connections,         │
│   time_mlp — NONE of them know D. They see 128 channels.         │
│   D=9 or D=23 or D=12: doesn't matter. Still 128 channels.      │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUT PROJECTION  Conv1d(128 → D, kernel=1)                     │
│   → this layer KNOWS D; its weight shape is (D, 128, 1)         │
│   → maps from 128-dim feature space back to D-dim trajectory     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
trajectory (B, H, D)
```

The hidden feature space is FIXED at 128→256→512→1024 regardless of D.
The backbone code never hardcodes D anywhere — it reads `transition_dim` once at
`__init__` time, uses it to build two Conv1d layers, and ignores it in the rest.

This is the same principle in every deep learning encoder-decoder:
- Transformer: `Linear(vocab_size → hidden)` + transformer blocks + `Linear(hidden → vocab_size)`
- VAE: `Conv2d(3→latent)` + encoder + decoder + `Conv2d(latent→3)`
- iMF DiT: `Linear(D → hidden_size)` + transformer blocks + `Linear(hidden_size → D)`

**The backbone does not care what D is** because it never sees D after the first Conv1d.
It is operating on abstract "feature channels", not trajectory dimensions.

### Why you still can't hot-swap D at runtime

Even though the backbone code is D-agnostic, a TRAINED CHECKPOINT is not.
The trained first Conv1d has weights shaped `(128, D_train, 5)`. If you try to load it
with a model built for `D_other`, PyTorch will crash:

```
RuntimeError: size mismatch for backbone.downs.0.0.blocks.0.block.0.weight:
  copying a param with shape (128, 9, 5) from checkpoint,
  the shape in current model is (128, 23, 5).
```

The CODE is D-agnostic. The WEIGHTS are not. Training bakes D into the boundary layers.

### What this means practically

To support a new trajectory dimension D_new:
1. Instantiate `UNet1DTemporalCondModel(transition_dim=D_new, ...)` — code is unchanged
2. The constructor builds two new boundary Conv1d layers with the right shapes
3. The entire hidden feature space (128→...→128) is shared in ARCHITECTURE but NOT in weights
4. Train from scratch — the new input projection needs to learn to embed D_new dims
5. Only the first/last Conv1d differ from an existing D_old model

You do NOT need to edit a single line of `unet1d_temporal_cond.py` to support a new D.
That is why the backbone "bone" never changes — the code parametrises D via the constructor.

---

## 14. Did we modify the backbone code for different tensor shapes?

**Short answer: NO.** The backbone class (`UNet1DTemporalCondModel` / `Flow_matcher_U_Net_v2`)
is untouched — in several namespaces it is a literal file copy. We only pass different
constructor arguments. The constructor then builds different Conv1d weight shapes from those
args. That is all.

What changes per model:

| Parameter | Who sets it | Effect on weights |
|---|---|---|
| `transition_dim` (D) | `VisualUNet.__init__` or config | first Conv1d `(D→128)` + last Conv1d `(128→D)` |
| `embed_dim` (=dim [+ dim if FiLM]) | `use_cond_projection` flag | `time_mlp` Linear `(embed_dim→out_ch)` inside **every** ResBlock |
| `dim` (base width) | config `dim=128` or `freq_dim=256` | ALL channel widths: 128/256→256/512→... |

So two models can have **identical backbone class** but completely incompatible weight shapes
if they differ in `transition_dim`, `embed_dim`, or `dim`. The class never changes;
only the instantiation args change.

---

## 14. Per-layer shape traces — every model variant

Notation: `(B, channels, H)` for Conv1d layers — note x is stored channel-first inside the UNet
(rearranged at entry, rearranged back at exit). `embed_dim` = the dimension of `t` that every
ResBlock receives.

### Conv1dBlock internals
```
Conv1dBlock(inp, out, kernel_size=5):
    Conv1d(inp, out, kernel=5, pad=2)   → (B, out, H)
    GroupNorm(8 groups, out)            → (B, out, H)
    Mish()
```
The padding `kernel//2 = 2` keeps spatial size H unchanged through every Conv1d in ResBlocks.

### ResidualTemporalBlock internals
```
ResidualTemporalBlock(inp_ch, out_ch, embed_dim):
    block[0]  = Conv1dBlock(inp_ch → out_ch)   # (B, out_ch, H)
    time_mlp  = Mish + Linear(embed_dim → out_ch) + Rearrange→(B, out_ch, 1)
    out       = block[0](x) + time_mlp(t)      # broadcast over H: (B, out_ch, H)
    block[1]  = Conv1dBlock(out_ch → out_ch)   # (B, out_ch, H)
    residual_conv = Conv1d(inp_ch→out_ch, k=1) if inp_ch≠out_ch else Identity
    return block[1](out) + residual_conv(x)
```

---

### 14A. Visual aligning / visual avoiding — UNet (9D, FiLM ON)

**Config**: `transition_dim=9, dim=128, dim_mults=(1,2,4,8), H=8, embed_dim=256`
`(dim=128 + cond_embed=128 via cond_mlp → cat → 256)`

```
INPUT x: (B, 8, 9)  ← (batch, horizon, trajectory_dim)
rearrange → (B, 9, 8)

───── conditioning ──────────────────────────────────────────────────────────
time_mlp: SinPos(128) → Linear(128→512) → Mish → Linear(512→128) → (B, 128)
cond_mlp: Linear(128→128) → Mish → Linear(128→128)                → (B, 128)
t = cat[time_emb, cond_emb]                                        → (B, 256)  ← embed_dim

───── encoder (downs) ───────────────────────────────────────────────────────
down[0]  ResBlock(  9 → 128, embed=256)  → (B, 128, 8)
         ResBlock(128 → 128, embed=256)  → (B, 128, 8)  ← push skip_0
         Downsample1d(128): Conv1d(128,128,3,s=2,p=1)   → (B, 128, 4)

down[1]  ResBlock(128 → 256, embed=256)  → (B, 256, 4)
         ResBlock(256 → 256, embed=256)  → (B, 256, 4)  ← push skip_1
         Downsample1d(256)               → (B, 256, 2)

down[2]  ResBlock(256 → 512, embed=256)  → (B, 512, 2)
         ResBlock(512 → 512, embed=256)  → (B, 512, 2)  ← push skip_2
         Downsample1d(512)               → (B, 512, 1)

down[3]  ResBlock(512 → 1024, embed=256) → (B,1024, 1)
         ResBlock(1024→1024, embed=256)  → (B,1024, 1)  ← push skip_3
         Identity (is_last)              → (B,1024, 1)

───── bottleneck ────────────────────────────────────────────────────────────
mid_block1 ResBlock(1024→1024, embed=256) → (B,1024, 1)
mid_block2 ResBlock(1024→1024, embed=256) → (B,1024, 1)

───── decoder (ups) ─────────────────────────────────────────────────────────
up[0]  cat(x, skip_3): (B, 1024+1024=2048, 1)
       ResBlock(2048 →  512, embed=256)  → (B,  512, 1)
       ResBlock( 512 →  512, embed=256)  → (B,  512, 1)
       Upsample1d(512): ConvTranspose1d(512,512,4,s=2,p=1) → (B, 512, 2)

up[1]  cat(x, skip_2): (B, 512+512=1024, 2)
       ResBlock(1024 →  256, embed=256)  → (B,  256, 2)
       ResBlock( 256 →  256, embed=256)  → (B,  256, 2)
       Upsample1d(256)                   → (B,  256, 4)

up[2]  cat(x, skip_1): (B, 256+256=512, 4)
       ResBlock( 512 →  128, embed=256)  → (B,  128, 4)
       ResBlock( 128 →  128, embed=256)  → (B,  128, 4)
       Upsample1d(128)                   → (B,  128, 8)

  ⚠ skip_0 (B, 128, 8) is NEVER popped — 3 ups but 4 downs (unused first skip)

───── output ────────────────────────────────────────────────────────────────
final_conv[0] Conv1dBlock(128 → 128)      → (B, 128, 8)
final_conv[1] Conv1d(128 → 9, k=1)        → (B,   9, 8)
rearrange     (B, 9, 8) → (B, 8, 9)       = OUTPUT ✓
```

**Total weight differences vs other models**: ONLY `down[0]` first Conv1d `(9→128)` and
`final_conv[1]` `(128→9)` depend on `transition_dim=9`. Everything else (128→256→512→1024
bottleneck) is shape-identical across all dim=128 models.

Also: every ResBlock's `time_mlp` has `Linear(256→out_ch)` (embed=256). Switch to
non-visual (no FiLM, embed=128) → `Linear(128→out_ch)` in every block → incompatible.

---

### 14B. Non-visual aligning WRONG — UNet (23D, no FiLM)

**Config**: `transition_dim=23, dim=128, dim_mults=(1,2,4,8), H=8, embed_dim=128`
`(dim=128, no cond_mlp → embed_dim = 128 only)`

```
INPUT x: (B, 8, 23)
rearrange → (B, 23, 8)

time_mlp: SinPos(128) → Linear(128→512) → Mish → Linear(512→128) → (B, 128)
[cond_mlp: ABSENT — cond_mlp=None, t = time_emb only]
t = time_emb                                                        → (B, 128)  ← embed_dim

down[0]  ResBlock( 23 → 128, embed=128)  → (B, 128, 8)   ← DIFFERS: first conv is 23-ch
         ResBlock(128 → 128, embed=128)  → (B, 128, 8)
         Downsample                      → (B, 128, 4)

down[1..3], mid, up[0..2]:  IDENTICAL shape to 14A EXCEPT embed_dim=128 in every time_mlp
         ...
         Upsample                        → (B, 128, 8)

final_conv[1] Conv1d(128 → 23, k=1)     → (B,  23, 8)    ← DIFFERS: last conv is 23-ch
rearrange → (B, 8, 23) = OUTPUT

Checkpoint incompatibility vs 9D visual:
  down[0].blocks[0].conv: (128, 9, 5) ≠ (128, 23, 5)   ← MISMATCH
  final_conv[1]: (9, 128, 1) ≠ (23, 128, 1)             ← MISMATCH
  every ResBlock time_mlp: Linear(256, out) ≠ Linear(128, out)  ← MISMATCH (FiLM vs no FiLM)
```

---

### 14C. UAV — Flow_matcher_U_Net_v2 (12D, no FiLM)

**Config**: `transition_dim=12, dim=128, dim_mults=(1,2,4,8), H=8, embed_dim=128`

```
INPUT x: (B, 8, 12)
rearrange → (B, 12, 8)

t = time_emb                                                        → (B, 128)

down[0]  ResBlock( 12 → 128, embed=128)  → (B, 128, 8)   ← DIFFERS: first conv is 12-ch
         ResBlock(128 → 128, embed=128)  → (B, 128, 8)
         Downsample                      → (B, 128, 4)

down[1]  ResBlock(128 → 256, embed=128)  → (B, 256, 4)   ← identical to 14B from here
         ResBlock(256 → 256, embed=128)  → (B, 256, 4)
         Downsample                      → (B, 256, 2)
    ... (same as 14B)

final_conv[1] Conv1d(128 → 12, k=1)      → (B,  12, 8)  ← DIFFERS: last conv is 12-ch
rearrange → (B, 8, 12) = OUTPUT

Class used: Flow_matcher_U_Net_v2 (NOT UNet1DTemporalCondModel)
  — same ResBlock structure, no cond_mlp field at all (not even None — it was never added)
  — accepts `cond` arg in forward() but never reads it
```

---

### 14D. iMF state avoiding — iMFTrajectoryModel(UNet) (6D, dim=256)

**Config**: `transition_dim=6, dim=256 (freq_dim), dim_mults=(1,2,4,8), H=8, embed_dim=256`

Note: `dim=256` here (not 128). Channel widths DOUBLE vs all other models.

```
dims = [6, 256, 512, 1024, 2048]    ← bottleneck is 2048, not 1024

INPUT x: (B, 8, 6)
rearrange → (B, 6, 8)

t = time_emb: SinPos(256)→Linear(256→1024)→Mish→Linear(1024→256) → (B, 256)

down[0]  ResBlock(   6 →  256, embed=256) → (B,  256, 8)  ← first conv 6-ch
         ResBlock( 256 →  256, embed=256) → (B,  256, 8)  ← push skip_0
         Downsample1d(256)                → (B,  256, 4)

down[1]  ResBlock( 256 →  512, embed=256) → (B,  512, 4)
         ResBlock( 512 →  512, embed=256) → (B,  512, 4)  ← push skip_1
         Downsample1d(512)                → (B,  512, 2)

down[2]  ResBlock( 512 → 1024, embed=256) → (B, 1024, 2)
         ResBlock(1024 → 1024, embed=256) → (B, 1024, 2)  ← push skip_2
         Downsample1d(1024)               → (B, 1024, 1)

down[3]  ResBlock(1024 → 2048, embed=256) → (B, 2048, 1)
         ResBlock(2048 → 2048, embed=256) → (B, 2048, 1)  ← push skip_3
         Identity                         → (B, 2048, 1)

mid_block1/2: ResBlock(2048→2048, embed=256) × 2 → (B, 2048, 1)

up[0]  cat(2048+2048=4096) → ResBlock(4096→1024) → ResBlock(1024→1024) → Up → (B,1024, 2)
up[1]  cat(1024+1024=2048) → ResBlock(2048→ 512) → ResBlock( 512→ 512) → Up → (B, 512, 4)
up[2]  cat( 512+ 512=1024) → ResBlock(1024→ 256) → ResBlock( 256→ 256) → Up → (B, 256, 8)

final_conv: Conv1dBlock(256→256), Conv1d(256→6)  → (B, 6, 8)
rearrange → (B, 8, 6) = OUTPUT

Checkpoint incompatibility vs visual aligning (dim=128):
  EVERY layer differs — all channel widths are 2× larger (256/512/1024/2048 vs 128/256/512/1024)
  This is NOT just a first/last conv mismatch — the ENTIRE network has different shapes
```

---

### 14E. IMFDiTTrajectory (6D state avoiding, default config)

**Config**: `transition_dim=6, hidden_size=256, depth=8, num_heads=4, patch_size=1, H=8`
`head_dim = 256/4 = 64`

```
INPUT x: (B, 8, 6)

───── patch embedding ───────────────────────────────────────────────────────
TrajPatchEmbedder:
  reshape (B, 8, 6) → (B, 8, 1×6=6)   [patch_size=1]
  Linear(6 → 256)                      → (B, 8, 256)   ← 8 trajectory tokens

───── conditioning tokens (prepended) ──────────────────────────────────────
  h_embedder(h):            SinPos(256)→MLP    → (B, 256) → token
  omega_embedder(omega):    SinPos(256)→MLP    → (B, 256) → token
  cfg_t_start_embedder:     SinPos(256)→MLP    → (B, 256) → token
  cfg_t_end_embedder:       SinPos(256)→MLP    → (B, 256) → token
  time_tokens:              learned param      → (2, 256)  → 2 tokens
  class_tokens (y-embed):   nn.Embedding       → (1, 256)  → 1 token (CFG)
  omega_tokens:             learned param      → (2, 256)  → 2 tokens

  prefix = 1(class) + 2(omega) + 1(t_min) + 1(t_max) + 2(time) = 7 tokens
  total sequence: prefix(7) + traj_patches(8) = 15 tokens
  → x_full: (B, 15, 256)

  RoPE tables: precomputed for seq_len=15, head_dim=64
    cos, sin: (15, 32)  [head_dim//2 = 32 freq pairs]

───── shared backbone ───────────────────────────────────────────────────────
  depth=8, aux_head_depth=2 → shared_depth=6, each head_depth=2

  shared_blocks[0..5]: 6× TransformerBlock(hidden=256, heads=4):
    RMSNorm(256) → RoPEAttention(256,4): Q/K/V Linear(256→256), out Linear(256→256)
                                         QK-RMSNorm per head (head_dim=64)
                                         attn weights: (B,4,15,15)
    zero-init gate (scalar per hidden): x = x + attn_out * attn_scale(256)
    RMSNorm(256) → SwiGLUMlp(256 → int(256*8/3)=682):
                     w1: Linear(256→682), w3: Linear(256→682), w2: Linear(682→256)
    zero-init gate: x = x + mlp_out * mlp_scale(256)
    → (B, 15, 256)

───── dual heads (split at shared backbone output) ─────────────────────────
  u_heads[0..1]: 2× TransformerBlock(256, 4) → (B, 15, 256)
  v_heads[0..1]: 2× TransformerBlock(256, 4) → (B, 15, 256)

───── output projection ─────────────────────────────────────────────────────
  u_final_layer: RMSNorm(256) + zero-init Linear(256 → 1×6=6) → (B, 15, 6)
  v_final_layer: RMSNorm(256) + zero-init Linear(256 → 6)     → (B, 15, 6)

  slice [prefix:] to recover trajectory tokens: → (B, 8, 6)
  unpatchify: (B, 8//1=8, 6) = OUTPUT (B, 8, 6) ✓
```

The DiT has **NO skip connections, NO downsampling, NO channel-based conv**.
Full attention over all 15 tokens at every layer → global receptive field vs UNet's
local (kernel=5) hierarchical view.

---

### 14F. Summary: which weight shapes differ between models

```
                        first Conv1d    last Conv1d    ResBlock time_mlp    bottleneck
                        (D→128)         (128→D)        Linear(E→out_ch)     max channels

Visual aligning 9D      (128, 9, 5)    (9, 128, 1)    Linear(256→*)        1024  dim=128
Non-vis aligning 23D    (128,23, 5)    (23,128, 1)    Linear(128→*)        1024  dim=128  ← FiLM also differs
UAV 12D                 (128,12, 5)    (12,128, 1)    Linear(128→*)        1024  dim=128
iMF UNet 6D (dim=256)   (256, 6, 5)    (6, 256, 1)    Linear(256→*)        2048  dim=256  ← ENTIRE net differs
iMF DiT 6D              Linear(6→256)  Linear(256→6)  — no ResBlocks —     hidden=256
```

The 9D visual vs 23D non-visual models: differ at BOTH first/last conv AND every ResBlock's
time_mlp (256 vs 128). You cannot share a single checkpoint between them.

The 9D visual vs 12D UAV: differ at first/last conv AND every time_mlp (256 FiLM vs 128).

The iMF UNet (dim=256) vs visual UNet (dim=128): EVERY SINGLE LAYER differs in size
(all channel counts are 2×). Not just first/last.

*Generated 2026-06-26. Cross-references: DPCC_TENSOR_ORIGIN.md, ALIGNING_EXPANSION.md.*
