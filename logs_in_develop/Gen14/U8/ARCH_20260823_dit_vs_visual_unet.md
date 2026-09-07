# Gen14 U8 — Architecture comparison: the DiT bone vs. the Visual U-Net

**Date:** 2026-08-23 · **Generation:** Gen14 (`mix_visual_aligning` ↔ `mix_visual_aligning_test`)
**Unit:** U8 (visual DiT bone) · **Task:** D3IL aligning, visual (dual-camera)
**Evidence:** `temp/2208/2026-08-22/13_49_30_{gates,train}_*_248{73,74}.log` (GIT REV `eb82d0b`, all 14 gates PASS),
`temp/1208/Gen14_Mix_FilmV2/2026-08-09/23_53_20_train_*_24454.log` (U-Net reference run),
plus a static read of the model sources.

This is an **architecture** document. For what the two bones actually *scored*, see the sibling
[`DA_20260823_Gen14_U8_mf_dit_visual_aligning.md`](DA_20260823_Gen14_U8_mf_dit_visual_aligning.md);
§9 here carries only the numbers needed to interpret the architecture.

---

## 0. Naming correction, read this first

The bone that trained on 2026-08-22 is **`ml_bone='dit'`**, which `visual_dit_twotime.py:_BONES`
resolves to `dit_mf` → **`MFDiTTrajectory`** — the **iMF RoPE DiT** ported from
`aux_repo/imeanflow/models/imfDiT.py`.

It is **not** `ml_bone='mf_dit'` (`MFDiTOfficialTrajectory`, the official MeanFlow adaLN DiT). Those
are two architecturally distinct networks that both live in this tree, and the run log is explicit:

```
train log:24874:96   [ VisualDiTTwoTime ] bone=dit_mf (MFDiTTrajectory)  hidden=160 depth=8 heads=4 patch=1  cond_dim=128
train log:24874:84   ml_bone='dit', dit_hidden_size=160, dit_depth=8, dit_num_heads=4, dit_patch_size=1
savepath             ..._Bdit_Emf_tslogit_normal_TB80pct/6
```

⚠️ The existing DA file is named `DA_..._Gen14_U8_mf_dit_visual_aligning.md`, which reads as the
*other* bone. Its **contents** are correct (they analyse candidate 12, the `_Bdit_` run); only the
filename is misleading. Suggested rename, not applied:

```bash
git mv logs_in_develop/Gen14/U8/DA_20260823_Gen14_U8_mf_dit_visual_aligning.md \
       logs_in_develop/Gen14/U8/DA_20260823_Gen14_U8_dit_visual_aligning.md
```

Throughout this document **"the DiT"** means the iMF RoPE DiT that trained. The other two bones
(`mf_dit`, `sit`) are built and gated but were not trained on aligning; they appear in the tables
because the parameter budget is the whole point of U8 and they define its bracket.

---

## 1. What is shared, and why that matters

Both wrappers — `VisualUNetTwoTime` (`visual_unet_twotime.py`) and `VisualDiTTwoTime`
(`visual_dit_twotime.py`) — hold a vision encoder and a trajectory bone. **The encoder is
byte-identical between them**, deliberately:

> `visual_dit_twotime.py:88` — *"🔴 BYTE-IDENTICAL to visual_unet_twotime.py:76-96. Any drift here
> silently breaks the U-Net-vs-DiT comparison this whole unit exists to make."*

| Shared component | Value |
|---|---|
| Encoder | `MultiImageObsEncoder` (D3IL / `diffusion_policy` upstream) |
| Cameras | `agentview_image` + `in_hand_image`, both `3×96×96` |
| Backbone per camera | ResNet-18, `output_size=64`, **`share_rgb_model=False`** (two separate towers) |
| Norm | GroupNorm (`use_group_norm=True`), ImageNet input norm |
| Latent | 64 + 64 concatenated → **`LATENT_DIM = 128`**, mean-pooled over the obs window |
| Trained? | **Yes** — end-to-end, in the optimizer's parameter list (not frozen) |
| Size | **≈ 22.36 M** parameters (≈ 11.2 M per ResNet-18 tower) |

Also shared:

| | |
|---|---|
| `TRANSITION_DIM` | **9** = action 3 + [`des_c_pos` 3 + `c_pos` 3]. Hardcoded; `config.obs_dim` deliberately ignored (fix_5 lesson) |
| Horizon | `H = 8` |
| Engine | `VisualMeanFlow` (`mf`), `dual_head=True`, `t_schedule='logit_normal'`, `aw=1`, `bs=64`, `lr=2e-4`, `ema=0.995` |
| Contract | `forward(x, cond, t, returns, use_dropout, force_dropout, h, omega, t_min, t_max, return_v)` → `u` or `(u, v)` |
| JVP short-circuit | `cond['visual_latent']` pre-encoded once by the engine, so inside `torch.func.jvp` it is a captured constant with a zero tangent and the ResNets never enter the differentiated function |

**Consequence for the comparison.** 22.36 M of ~26 M parameters — **~86%** — are the *same network*
in both arms. Everything U8 varies lives in the remaining ~4 M. That is what makes a bone A/B
meaningful here, and also what caps how much any bone swap can possibly move.

---

## 2. Side-by-side architecture

| | **VisualUNetTwoTime** (`unet`) | **VisualDiTTwoTime** (`dit`) |
|---|---|---|
| Bone class | `Flow_matcher_U_Net_v2` (v1) / `Flow_matcher_U_Net_v2_FiLM` (v2) | `MFDiTTrajectory` |
| Provenance | Gen3v6 two-time U-Net + Gen7 visual graft | `aux_repo/imeanflow/models/imfDiT.py`, ported |
| Family | 1-D temporal conv U-Net | Pre-norm transformer |
| Trunk | 4 down levels + 2 mid blocks + 3 up levels, skip connections | 6 shared blocks → 2 u-head blocks + 2 v-head blocks |
| Width | `dim = 32`, `dim_mults = (1,2,4,8)` → 32/64/128/256 ch | `hidden_size = 160`, constant |
| Depth | 8 down/up residual blocks + 2 mid = **10 residual blocks** | **10 transformer blocks** (6 shared + 2 + 2) |
| Temporal mixing | Conv1d, `kernel_size=5`, padding 2 | Full self-attention, 4 heads, `head_dim=40` |
| Positions | Implicit (convolution + stride) | **RoPE**, real-valued interleaved rotation |
| Norm | GroupNorm(8) inside `Conv1dBlock` | RMSNorm (pre-norm) + **QK-RMSNorm** |
| Activation / MLP | Mish, conv-only (no MLP) | SwiGLU, `mlp_ratio = 8/3` → hidden 426 |
| Length handling | Pads `H=8 → 8` (must be ÷8 for 3 stride-2 levels), crops back | **No padding** at `patch_size=1`: 8 steps = 8 tokens |
| Visual conditioning | v1: project → **concat into the time embedding**; v2: per-block **γ/β FiLM** | **One prefix token**, appended last in the prefix |
| Two-time `h` | `h_mlp(h)` **added** to `time_mlp(τ)` | Its own **`h` token** (`time_tokens + h_embed`) |
| CFG plumbing | `interval_cfg` flag (off) | Native tokens: `omega`, `t_min`, `t_max`, null-class |
| u/v heads | Two parallel `final_conv` stacks on the shared trunk | Two 2-block transformer heads + zero-init `FinalLayer`s |
| Init | diffusers default | scaled-variance (`std = c/√fan_in`), **zero-init residual gates**, zero-init final linears |
| Bone params | **4,035,666** (v1) / 4,100,946 (v2) | **3,370,578** |
| Ratio to U-Net v1 | 1.00× | **0.84×** |
| Total `velocity_net` | 26.4 M (v1) / 26.5 M (v2) | **25.8 M** |

---

## 3. Parameter budget, component by component

Counted analytically from the sources (no torch in this container). All three totals reproduce the
cluster gate log **exactly**, which validates the arithmetic:

```
gates log:24873:58   reference: VisualUNetTwoTime bone = 4.04 M
gates log:24873:62   ok  mf@mf_dit: bone 4.04 M (1.00x U-Net)
gates log:24873:66   ok  mf@dit:    bone 3.37 M (0.84x U-Net)
gates log:24873:70   ok  af@sit:    bone 3.97 M (0.98x U-Net)
```

### 3.1 Visual U-Net v1 — 4,035,666

| Component | Params | Share |
|---|---:|---:|
| `time_mlp` (τ) | 8,352 | 0.2% |
| `h_mlp` (two-time interval) | 8,352 | 0.2% |
| `cond_mlp` (128 → 32 visual projection) | 5,184 | 0.1% |
| Down path (4 levels + 3 downsamples) | 1,698,752 | 42.1% |
| Mid (2 blocks @ 256 ch) | 1,347,072 | 33.4% |
| Up path (3 levels + 2 upsamples) | 956,928 | 23.7% |
| `final_conv` (u head) | 5,513 | 0.1% |
| `v_final_conv` (v head, `dual_head=True`) | 5,513 | 0.1% |

`embed_dim = dim + cond_embed_dim = 32 + 32 = 64` — the visual latent **widens the time embedding**,
which is what v1's "FiLM" actually is (an additive per-channel bias via concat, not a scale).

### 3.2 Visual U-Net v2 (`filmv2`) — 4,100,946 (+65,280, +1.6%)

v2 changes *where* the visual latent enters, at almost no parameter cost:

```
v1 per block:  out = Conv(x) + time_mlp([ t(τ,h) ‖ cond_emb ])
v2 per block:  out = (1 + γ(cond)) · ( Conv(x) + time_mlp(t(τ,h)) ) + β(cond)
```

Per block the time MLP **shrinks** (`embed_dim` drops 64 → 32, time-only) and a `film_proj`
`Linear(32, 2·out_ch)` is **added**, zero-initialised so step 0 is numerically identical to v1.
Net **+34 params per output channel**, summed over 1,920 output channels = +65,280. The FiLM block
itself is *imported* from `unet1d_temporal_film.py` (a G0-verbatim Gen7 copy), so mf's v2 and fm's v2
are literally the same class object.

### 3.3 iMF RoPE DiT (`dit`, the bone that trained) — 3,370,578

| Component | Params | Share |
|---|---:|---:|
| Transformer blocks (10 × 307,600) | 3,076,000 | **91.3%** |
| Timestep embedders ×4 (`h`, `omega`, `cfg_t_start`, `cfg_t_end`) | 267,520 | 7.9% |
| `vis_projector` (128 → 160) | 20,640 | 0.6% |
| `x_embedder` (9 → 160) | 1,600 | 0.05% |
| Final layers ×2 (u, v) | 3,218 | 0.10% |
| Learned prefix tokens (8 × 160) | 1,280 | 0.04% |
| `y_embedder` (null-class, CFG) | 320 | 0.01% |

Per block (307,600): attention 102,480 (4 × 160² projections, bias-free, + two head-dim QK-RMSNorms)
· SwiGLU MLP 204,480 (3 × 160×426, bias-free) · 2 RMSNorms 320 · 2 residual gate vectors 320.

`t_embedder` is **absent**: `condition_on_t=False` (the official iMF recipe conditions on the
interval `h`, not the anchor `t`). Turning `dit_condition_on_t=True` would add one more 66,880-param
embedder — worth knowing before anyone flips it.

### 3.4 The two built-but-untrained bones

**Official MeanFlow DiT (`mf_dit`) — 4,036,658 = 1.00× U-Net, the exactly-matched bone.**

| Component | Params |
|---|---:|
| adaLN-zero blocks (8 × 463,280) | 3,706,240 |
| Timestep embedders ×3 (`t`, `r`, `w`) | 200,640 |
| Final layers ×2 (adaLN + linear) | 105,938 |
| `vis_projector` + `vis_token` | 20,800 |
| `x_embedder` | 1,600 |
| Learned absolute pos-embed (9 × 160) | 1,440 |

Note how differently the budget distributes: the adaLN modulation (`Linear(160, 6·160)` per block,
154,560 = 33% of each block) is pure conditioning machinery. That is why `mf_dit` hits 4.04 M at
depth 8 while the iMF DiT needs 10 blocks to reach 3.37 M.

**alpha-Flow SiT (`sit`) — 3.97 M = 0.98× U-Net** (gate log; not recomputed here). Same adaLN-zero
family as `mf_dit` but with two timestep embedders (`noise_labels`, `noise_labels_next`) and a
**frozen** sin-cos `pos_embed` (`requires_grad=False`).

### 3.5 The parameter-match guard

`visual_dit_twotime.py:169-186` hard-codes 160 as the visual width and prints a loud warning at 256:

> *"dit_hidden_size=256 is the STATE-ONLY default, NOT the parameter-matched visual width (160).
> This bone will be ~2.5x the U-Net and any U-Net-vs-DiT comparison from it is CONFOUNDED. …
> Gate G-B2 fails on it."*

This exists because of a real prior failure: the Fix_8 defect (`bb_unet_ablation`, 2026-07-25)
compared an unmatched backbone and had to be **retracted** by the 2026-08-19 study. G-B2 now enforces
the bracket on every run.

### 3.6 External reference points — the D3IL baseline and Diffusion Policy

Everything above is internal to this repo, so it says nothing about whether **4 M is a sane size for
this task at all**. Two outside anchors settle that: the benchmark's own reference implementation,
and the field's default visuomotor policy.

#### 3.6.1 The original D3IL aligning-vision model

`aux_repo/d3il/configs/aligning_vision_config.yaml` selects `agents: ddpm_encdec_vision` — so the
model D3IL itself reports for *this exact task* is **`DiffusionEncDec`**
(`agents/models/diffusion/diffusion_models.py:687`), a transformer encoder–decoder denoiser inside a
16-step DDPM, with blocks from `agents/models/act/act_vae.py`.

**Its vision encoder is our vision encoder.** Verified, not assumed:

```bash
diff /workspaces/aux_repo/d3il/agents/models/vision/multi_image_obs_encoder.py \
     /workspaces/FM-PCC/d3il/agents/models/vision/multi_image_obs_encoder.py   # → identical
```

Same `shape_meta` (two `3x96x96` cameras), same `get_resnet(output_size=64)` (robomimic `VisualCore`,
ResNet18Conv + SpatialSoftmax(32 kp) + Linear→64), same `share_rgb_model=False`, `use_group_norm=True`,
`imagenet_norm=True`. Same **128-D latent**. That is a large piece of luck for this comparison: the
D3IL baseline and every Gen14 arm differ *only* in the trajectory net.

Denoiser budget, counted analytically at `embed_dim=64`, `state_dim=128`, `action_dim=3`,
`obs_seq_len=5`, `action_seq_len=4`, `linear_output=True`:

| Component | Params |
|---|---:|
| `TransformerEncoder` — 2 × `EncoderBlock` (49,856) + final LN | 99,776 |
| `TransformerDecoder` — 4 × `DecoderBlock` (62,336) + final LN | 249,408 |
| `tok_emb` (128 → 64) | 8,256 |
| `pos_emb` (seq 8 × 64) | 512 |
| `time_emb` (Sinusoidal + 64→128→64) | 16,576 |
| `action_emb` (3 → 64) | 256 |
| `action_pred` (64 → 3, linear head) | 195 |
| **Total `DiffusionEncDec`** | **374,979** |

`DecoderBlock` is the heavier one because `CausalSelfCrossAttention` carries **seven** `Linear(64,64)`
(q/k/v + cross-q/k/v + proj) against `SelfAttention`'s four.

The rest of the D3IL vision family is the same order: `beso_vision` is a 4-layer GPT at `n_embd=72`
(≈ 0.25 M of blocks), `act_vision` a 2+2 encoder / 4 decoder VAE at `embed_dim=64` (≈ 0.45 M of
blocks). **No D3IL vision agent has a trajectory net above ~0.5 M.**

#### 3.6.2 Diffusion Policy (Chi et al., RSS 2023) — the consensus flagship

The field default for visuomotor manipulation, and the *actual upstream of this repo's vision encoder*
(§4, point 1). Counted from `aux_repo/visual_transformer_refs_(Claude_pulled)/diffusion_policy` at its
shipped config values:

| Variant | Config | Geometry | Denoiser params |
|---|---|---|---:|
| **DP-C** (CNN) | `train_diffusion_unet_image_workspace.yaml` | `ConditionalUnet1D`, `down_dims=[512,1024,2048]`, `dsed=128`, `k=5`, FiLM `cond_predict_scale=True` | **255.1 M** at our geometry (act 3, global_cond 128×2); 306.5 M at DP's robomimic-image geometry (act 10, cond 2048) |
| **DP-T** (Transformer) | `train_diffusion_transformer_hybrid_workspace.yaml` | 8 decoder layers, `n_emb=256`, 4 heads, MLP cond encoder | **9.0 M** at our geometry; 9.2 M at DP's |

DP's encoder is the same ResNet-18 pair but with `fc = Identity`, so it emits **512 per camera → 1024**,
and feeds `n_obs_steps=2` of it — an **8× wider** conditioning signal than the 128-D latent D3IL (and
we) compress to. That gap, not the block counts, is where DP's extra capacity actually goes.

#### 3.6.3 The full bracket

Encoder column is the shared 22.36 M wherever the D3IL encoder is used (measured: gate log
`26.4 M total − 4.04 M bone`); DP's own encoder is the same backbone with a wider head (~22.4 M).

| Model | Trajectory net | Bone params | × U-Net v1 | Total policy | Inference NFE |
|---|---|---:|---:|---:|---:|
| **D3IL `ddpm_encdec_vision`** — the benchmark's own aligning-vision model | Transformer enc-dec, E=64, 2+4 blocks | **0.375 M** | **0.09×** | **22.74 M** | 16 (DDPM) |
| D3IL `beso_vision` | GPT, E=72, 4 layers | ~0.3 M | ~0.08× | ~22.7 M | 3 (`euler_ancestral`) |
| D3IL `act_vision` | CVAE transformer, E=64 | ~0.5 M | ~0.12× | ~22.9 M | 1 |
| Gen14 `unet` v1 — **our reference** | 1-D conv U-Net, `dim=32` | **4.036 M** | 1.00× | 26.4 M | K (1–20) |
| Gen14 `unet` v2 (`filmv2`) | + per-block FiLM | 4.101 M | 1.02× | 26.5 M | K |
| Gen14 `mf_dit` (adaLN, untrained here) | DiT 160 × 8 | 4.04 M | 1.00× | 26.4 M | K |
| Gen14 `sit` (untrained here) | SiT 160 × 8 | 3.97 M | 0.98× | 26.4 M | K |
| **Gen14 `dit` — the bone that trained** | iMF RoPE DiT 160 × 10 | **3.371 M** | **0.84×** | 25.8 M | K |
| **Diffusion Policy — Transformer** | 8 × 256 transformer | **8.99 M** | **2.23×** | ~31.4 M | 100 DDPM / 10 DDIM |
| **Diffusion Policy — CNN** | `ConditionalUnet1D` [512,1024,2048] | **255.1 M** | **63.2×** | ~277.4 M | 100 DDPM / 10 DDIM |

Three things fall out of this table.

1. **The U8 bracket is ~10× the head we inherited this encoder from.** Every Gen14 bone
   (3.37–4.10 M) is roughly **9–11×** the D3IL aligning-vision denoiser. This is a sizing fact, not
   an endorsement — D3IL's own agents do not solve this task, so their number is not a target. What
   it does establish is that the trajectory net is not the small part of this design, which matters
   for §13.

2. **Where each design spends its parameters is completely different.** Share of the total policy
   living in the trajectory net: D3IL **1.6%**, Gen14 **~15%**, DP-C **92%**. Two consequences. First,
   any claim of the form "architecture X is better for visuomotor control" that crosses these regimes
   is comparing perception budgets, not architectures. Second — and this is the one that matters —
   **85% of our trainable parameters are a from-scratch ResNet pair**, which on a 900-episode dataset
   is the dominant design risk in the whole stack. See §13.1.

3. **DP-C's default width is almost exactly the retracted Fix_8 build.** 255.1 M here vs the
   253 M `freq_dim=256` U-Net that invalidated `bb_unet_ablation` (§3.5). These are *different
   networks* and the collision is arithmetic coincidence, not lineage — but it is a useful reframing:
   the Fix_8 defect was accidentally training a **DP-C-scale** trunk on a D3IL-scale dataset. That is
   why it failed, and it is also why DP-C is not a target to chase here.

**Caveats on these numbers.** All external counts are analytic (no torch in this container) from the
shipped source at the shipped config values, using the same method that reproduced the Gen14 gate log
exactly (§3). They are *denoiser/trajectory-net* counts on the aligning geometry; DP has never been
run on D3IL aligning in this repo, so its row is a **capacity reference, not a scored baseline**. The
D3IL row is a capacity reference too — no D3IL agent has been re-run under our eval harness, so do
not read it as a performance comparison.

---

## 4. Where the image enters — the one real design decision

Three mechanisms, one per design point:

| Bone | Mechanism | Sees the image at |
|---|---|---|
| U-Net **v1** | `cond_mlp(latent)` concatenated onto `t`, widening `embed_dim` 32 → 64 | every residual block, as an **additive bias** |
| U-Net **v2** | `film_proj(cond) → (γ, β)`, `(1+γ)·f + β` | every residual block, as a **multiplicative gate + shift** |
| **DiT** | `vis_tokens + vis_projector(latent)` prepended as **one token** | every block, through **attention** |

The token choice is argued in `DECISION_Gen14_U8_injection_choice.md` and restated at
`visual_dit_twotime.py:14-32`. The reasoning, briefly:

1. **`diffusion_policy` is the actual upstream of this repo's vision encoder** — D3IL's
   `multi_image_obs_encoder.py` is their file, verbatim but for import paths. Their
   `TransformerForDiffusion` ingests the obs latent as **tokens**; adaLN appears nowhere in it. They
   reserve per-channel modulation for their `ConditionalUnet1D`.
2. **The U-Net already occupies the modulation design point** (`film_mode` v1/v2). Putting adaLN on a
   DiT would re-ask the same conditioning question with a different trunk; a token asks a new one.
3. **The token is the only mechanism spanning all four bones** — `MFDiTTrajectory` /
   `AFDiTTrajectory` blocks are `forward(x, cos, sin)` and have no adaLN pathway at all.
4. At `window_size=1` (a dataset-level lock), `diffusion_policy`'s `T_cond = 1 + n_obs_steps`
   collapses to exactly one visual token — so this is their stack at our settings, not an
   approximation of it.

In the iMF DiT the visual token is appended **last in the prefix** so every pre-existing token keeps
its RoPE position and state-only checkpoints stay positionally comparable. The two constants that
must move together are flagged in red:

> `mf_dit_trajectory.py:307-313` — *"🔴 these two MUST move together with num_visual_tokens …
> Bumping one and not the other yields a model that trains fine and reads the WRONG positions.
> Gate G-B6 asserts they agree."*

Gate G-B3 confirms the token is not decorative: `|grad vis_projector| = 1.305e-01`, `encoder
trains=True`, `d(out)/d(latent) max = 1.096e-02`.

---

## 5. Sequence handling — the sharpest structural difference

**This is the finding I would carry forward from this document.**

`H = 8`. The U-Net has three stride-2 levels, so the temporal axis collapses `8 → 4 → 2 → 1`:

| U-Net stage | Channels | **Temporal length seen** | Params |
|---|---:|---:|---:|
| L0 | 9 → 32 | **8** | 21,664 (+3,104 downsample) |
| L1 | 32 → 64 | **4** | 82,880 (+12,352) |
| L2 | 64 → 128 | **2** | 313,216 (+49,280) |
| L3 | 128 → 256 | **1** | 1,216,256 |
| mid ×2 | 256 → 256 | **1** | 1,347,072 |
| U0 | 512 → 128 | **1** | 657,280 (+65,664 upsample) |
| U1 | 256 → 64 | **2** | 168,896 (+16,448) |
| U2 | 128 → 32 | **4** | 44,512 (+4,128) |

**≈ 79.8% of the U-Net bone (3.22 M of 4.04 M) operates on a sequence of length 1.** A `kernel_size=5`
convolution over one timestep is a 1×1 channel mix with four fifths of its taps hitting zero padding.
At `H = 8`, the bulk of this "temporal U-Net" is not doing temporal modelling at all — it is an MLP
on a pooled trajectory vector.

The DiT has no such collapse. Its sequence is 16 tokens throughout:

```
[ class(1) | omega(2) | t_min(1) | t_max(1) | time/h(2) | visual(1) ] + [ 8 trajectory patches ]
   \_______________________ prefix_tokens = 8 _______________________/
```

Every one of the 10 blocks attends over all 16 positions, so **100% of the DiT's parameters see the
full horizon**. Note also that **half the sequence is conditioning** — a structural difference from a
vision DiT, where conditioning is a rounding error against hundreds of patches. Here the visual token
competes with 7 other prefix tokens for 1/16 of the attention mass.

This is the honest mechanistic hypothesis for U8, and §9 shows it has some empirical support: the two
bones' *unguided* behaviour differs in the direction this predicts, and their *variance* differs a
lot.

---

## 6. Two-time (`h`) conditioning

MeanFlow needs both an anchor time and an interval. The two bones thread it differently:

**U-Net** — `t = time_mlp(τ) + h_mlp(h)`, then (v1) concat the visual embedding. The order is
load-bearing and flagged in the source:

> `unet1d_twotime_cond.py:27-30` — *"🔴 ORDER MATTERS. `h_mlp` and the interval-CFG embeddings are
> ADDED to `t` (both emit [B, dim]), so they must run while `t` is still [B, dim] — i.e. BEFORE the
> cond CONCAT widens it to [B, dim + cond_embed_dim]."*

**DiT** — `time_tok = time_tokens + h_embedder(h)`, a pair of dedicated sequence positions. `t` enters
only if `condition_on_t=True` (it is False, per the official recipe). `omega`, `t_min`, `t_max` get
their own tokens and their own embedders — which is why the DiT spends 267 K params (7.9%) on
timestep embedding against the U-Net's 16.7 K (0.4%).

**Both are JVP-safe by construction**, and both were verified on the cluster (G-B4/5 PASS, finite
losses on all four bones). The mechanisms differ:
- U-Net: `h` is additive on a `[B, dim]` vector; the visual latent is a captured constant.
- DiT: RoPE is implemented as a **real-valued interleaved rotation** rather than the official complex
  bitcast, specifically so forward-mode AD flows through it. That is the port's single deliberate
  deviation; all learned components are unchanged.

---

## 7. Initialisation and stability

| | U-Net | iMF DiT |
|---|---|---|
| Trunk | diffusers default | scaled-variance, `std = 0.32/√fan_in` (blocks), `1.0/√fan_in` (embedders) |
| Residual path | plain residual add | **zero-init vector gates** (`attn_scale`, `mlp_scale`) — every block starts as identity |
| Output | default | zero-init `FinalLayer.linear` — network outputs exactly 0 at step 0 |
| Norm placement | GroupNorm(8) after each conv | pre-norm RMSNorm + QK-RMSNorm inside attention |

The zero-init gates are called out as deliberate: *"block starts as identity (stable under the stiff
JVP target)"*. This is a materially different optimisation problem from the U-Net's, and is a
plausible contributor to the variance difference in §9 — a network that begins as the identity and
grows its blocks in tends to produce a more conservative, lower-spread policy early on.

---

## 8. Compute

**Training throughput** (both on `NVIDIA RTX A5000`, 1000-step epochs, `bs=64`, `grad_accum=2`):

| Bone | s / 1000 steps | it/s | 80 k steps |
|---|---|---|---|
| U-Net `filmv2` (job 24454) | 10:40 – 14:23 | 1.16 – 1.56 | — |
| **iMF DiT** (job 24874) | **7:28 – 8:14** | **2.02 – 2.23** | ~10.1 h |

The DiT trains roughly **1.5× faster per step** despite the JVP. Two reasons: it is 0.84× the
parameters, and — more importantly — a 16-token attention is trivial work for an A5000, whereas the
U-Net's deep 256-channel stack at T=1 is a stack of small, launch-bound kernels. ⚠️ Same GPU *model*,
but different jobs on a shared cluster: treat this as indicative, not a controlled benchmark.

**Eval wall-clock** (`candidates_ranking.csv`, K2 rows, per-step control time):

| Candidate | Bone | Time (ms) |
|---|---|---|
| 12 | **DiT** | **48.3** |
| 15 | U-Net `filmv2` | 50.0 |
| 14 | U-Net `filmv1` | 62.3 |

The DiT is the fastest of the three at eval as well, though the gap to `filmv2` (3%) is inside the
noise of a shared cluster.

**Memory / model size:** total `velocity_net` 25.8 M (DiT) vs 26.4 M (U-Net v1) — a 2.3% difference,
because the shared ResNet pair dominates both.

---

## 9. What the trained comparison showed (summary only)

Full analysis in the sibling DA. The architecturally interesting parts:

| | U-Net (pooled) | DiT |
|---|---|---|
| `mean_dist_per_rollout` (task metric, lower better) | **0.3425** | 0.3959 |
| Unguided (arm A) distance | 0.4656 | **0.4187** |
| Gain extracted from projection | **0.179** | 0.061 |
| Distance sd across contexts | 0.310 | **0.175** |
| Fraction of rollouts < 0.05 m | **13.8%** | 6.7% |
| Success & constraints | 46/2204 = 2.09% | 6/1856 = 0.32% |

Read architecturally: **the DiT's raw generative trajectory is better, and its policy is much less
variable, but it responds far less to the MPC projector and produces fewer excellent rollouts.**
Lower variance with a worse tail is the signature of a more conservative, mode-averaging policy —
consistent with both the zero-init-gate optimisation (§7) and with attention over 16 tokens where
half are conditioning (§5).

⚠️ **The comparison is confounded and cannot be reported as a clean bone A/B.** At 0.84× the U-Net's
parameters the DiT is *under* the matched bracket. The parameter-matched bone is `mf_dit` at 4.04 M
(1.00×), and **it has not been trained on aligning**. Per the architecture-matched-beat rule, any U8
claim must lead with a matched row, and right now there isn't one.

---

## 10. Verdict

1. **The bone swap is clean and gated.** Vision encoder byte-identical, contract identical, all 14
   gates PASS at `eb82d0b`, G-B2 enforces the parameter bracket, G-B3 proves the visual token
   receives gradient, G-B4/5 proves the JVP and bootstrap survive it.
2. **86% of the network is shared.** U8 varies ~4 M of ~26 M parameters. This bounds how large any
   bone effect can be, and argues for reading U8's results as modest by construction.
3. **The bones are genuinely different where it matters, not cosmetically.** Conv-with-collapse vs
   attention-over-16-tokens; concat/FiLM vs prefix token; additive `h` vs `h` token; default init vs
   zero-init identity blocks.
4. **§5 is the structural headline.** At `H = 8` roughly 80% of the U-Net bone runs on a length-1
   sequence. Whatever the U-Net's advantage is on this task, it is unlikely to be *temporal
   modelling* — and that reframes the question U8 is asking.
5. **The trained comparison is not parameter-matched (0.84×) and rests on one seed.** It is
   suggestive, not a result. `mf_dit` at 1.00× is the run that would make it one.

---

## 11. Next steps (eval-only unless marked)

> ⚠️ **Read §13 first (added 2026-08-24).** Items 1–3 below assume the bone is the interesting
> variable. §13 argues from the U7+U8 evidence that it is not — the 85%-of-parameters from-scratch
> encoder on 900 episodes is — and puts two hours-scale probes ahead of every training item here.

1. **[train]** `mf @ mf_dit` at full budget — the 4.04 M exactly-matched bone. Without it U8 has no
   clean claim, and every DiT-vs-U-Net sentence has to carry the 0.84× caveat.
2. **[train, cheap]** A U-Net at `H = 32` or a DiT at `H = 8` with `dim_mults = (1,2)` — directly
   tests §5 by giving the U-Net's deep blocks an actual sequence to convolve over.
3. **[train]** `af @ sit` on aligning (3.97 M, 0.98×) — the second matched bone, and the only one
   testing the adaLN design point against our token choice.
4. **[eval]** Run arm C (HardFlow) on the DiT — currently zero cells; see the HF DA §4 for the
   `HFFM_VARIANTS` command.
5. **[eval]** Seeds 7–10 on both bones. Every §9 gap is single-seed.
6. **[code, small]** Port the Gen16 avoiding tree's `nfe_total` / `nlp_solves` instrumentation into
   the aligning eval, so §8's cost table stops being wall-clock-only.

---

## 12. One line

The DiT that trained is the **iMF RoPE bone at 3.37 M (0.84×)**, not the parameter-matched 4.04 M
`mf_dit`; it shares 86% of its weights with the U-Net, trains ~1.5× faster, replaces conv-with-
temporal-collapse by attention over 16 tokens of which half are conditioning — and the most useful
thing this comparison surfaced is that **at `H = 8` about 80% of the "temporal" U-Net never sees more
than one timestep.**

---

## 13. Direction — what is actually the bottleneck? (added 2026-08-24, rewritten same day)

Asked directly: *is the model too weak? pull a pretrained visual backbone and fine-tune? enlarge the
ML bone?*

**Framing correction, and it changes the answer.** The first draft of this section argued partly from
"D3IL and Diffusion Policy do X, we do Y". That is an appeal to authority and it is **withdrawn**.
D3IL built this setup and its own agents do not solve it; there is no normative value in matching
their choices. From here they appear in exactly two roles, both factual rather than prescriptive:

* **as an audit of our own code** — we inherited their encoder *verbatim* (§1, §3.6.1), so their
  design decisions are sitting in our repo whether or not we endorse them, and
* **as parameter reference points** (§3.6), which is arithmetic, not advice.

Every argument below stands on our own numbers and on first principles.

### 13.1 The real mismatch is data, not capacity — and it is in the encoder, not the bone

The dataset is **900 episodes** (`config/aligning-d3il-visual.py:422`), `max_path_length = 512`.

The trainable model is **26.4 M parameters, of which 22.36 M — 85 % — is a pair of ResNet-18 towers
initialised at random and learned end-to-end**, because
`config/aligning-d3il-visual.py:1309,1349` ship `'mf_freeze_vision_encoder': False`.

> **We are training two ImageNet-scale convnets from scratch on 900 demonstrations, and spending 15 %
> of the model on the part that actually does the task.**

That is the data-efficiency defect, and it is *five times larger* than the entire bone question this
document was written to study. Nothing about the U-Net-vs-DiT comparison touches it: both arms carry
the identical untrained 22.36 M.

**This flips part of the earlier verdict.** Pretrained or frozen visual features are now the *leading*
candidate — not as a fix for "too little capacity", but as a fix for **too little data for the
capacity we are already spending**. With 900 episodes, the correct move under a limited-data regime is
to *shrink the trainable footprint*, not grow it.

**And the hook already exists, unwired.** `mf_freeze_vision_encoder` is plumbed end-to-end
(`visual_mf_diffusion.py:29,35,51`, `visual_af_diffusion.py:29,34,45`,
`train_mix_visual_aligning.py:500`) and defaults to `False`. Flipping it is a **config-line ablation,
no code**. `d3il/agents/models/vision/model_getter.py` additionally carries an unwired
`_get_resnet(name, weights=…)` (ImageNet) and `get_r3m` (R3M robot-pretrained) — so "pretrained
encoder" is a wiring job, not a port.

### 13.2 Correction to the earlier argument

The first draft claimed: *"it fails on the train split, so a pretrained encoder — which buys
generalisation — is the wrong tool."* **That was too strong and is withdrawn.**

Failing on seen data rules out a *classical* generalisation gap. It does **not** rule out a perception
problem, because a from-scratch encoder on 900 episodes can drive training loss down while the 128-D
latent carries little task-relevant signal — the trajectory head then fits the *state* channel and
treats the image as noise. **Train-split failure and an image-blind latent are fully compatible.**
G-B3 proves gradient reaches `vis_projector`; it does not prove the latent is *informative*.

That distinction is measurable and cheap — see §13.4, probe P2.

### 13.3 What still stands, independent of any outside comparison

**(a) The failure is invariant across every axis we have varied.** Pooling U7 + U8: three structurally
different trunks (U-Net v1 concat-cond, U-Net v2 FiLM, iMF RoPE DiT), two engines (`mf`, `af`), two
projector arms (DPCC B, HardFlow C), two geometries, eleven variants. Every cell lands in
**0.29–0.47 m** against a do-nothing baseline of **0.4547 m**, at `n_steps = 400.0` (the cap), at
**< 2.1 % S&C**. Changing the bone moves ~0.05 m; nothing leaves the band. **A shared bottleneck
upstream of the bone** — and §13.1 names the largest thing sitting upstream of it.

**(b) Enlarging the bone is the wrong lever, and §5 is the reason.** At `H = 8`, **~80 % of the U-Net
bone already operates on a length-1 sequence**. Adding width to a trunk that is mostly an MLP on a
single timestep buys parameters, not capability — and under a 900-episode budget, added parameters are
a cost, not a hedge. If the trunk changes at all, the variable is the **horizon** (`H = 8 → 32`,
§11.2), which gives the deep blocks a sequence to convolve over. `mf@mf_dit` at 4.04 M (§11.1) stays
on the list because it removes the 0.84 × / 80 %-budget confounds — not because it will escape the band.

**(c) The conditioning is a single frame.** `config/aligning-d3il-visual.py:729-730` set
`window_size: 1, obs_seq_len: 1`, so the mean-pool at `visual_dit_twotime.py:216` is a no-op over
`T = 1`. On first principles: from one frame the box's **velocity is unobservable**, so a pushing task
is not Markov in the conditioning we supply — the policy cannot tell "box moving toward target" from
"box stalled against the gripper". It is a config lock, not an architectural limit, and on the DiT the
fix is the mechanism §4 already chose (emit `n_obs` visual tokens instead of one).

**(d) The latent is 128-D behind a 32-keypoint spatial-softmax.** Whatever its provenance, the
question is only whether 128 numbers can carry box pose + target pose + gripper relation well enough
to push accurately. That is measurable (§13.4 P2), and widening it costs ~20 K parameters in
`vis_projector` — trivial next to the 22.36 M above it.

**(e) The harness may not be winnable as configured.** `max_episode_length: 400`
(`config/aligning-d3il-visual.py:722`) and `n_steps = 400.0` in essentially every projected cell:
**nothing ever terminates**. Independently, DA §3.6 found `bounds_free` is the best or near-best cell
for **every bone in both geometries** at comparable violations — the bounds cage is over-restrictive
and is costing 0.03–0.05 m for safety the task was not losing.

### 13.4 Four probes, none of which need a new architecture

Ordered by cost. The first three are hours, not GPU-weeks.

| # | Probe | Cost | What it decides |
|---|---|---|---|
| **P1** | **Replay a ground-truth demonstration through our eval harness** — open loop, no model | no GPU | If a real demo does not score `success` under our criterion + 400-step cap + bounds + `combined_5`, the ceiling is in the harness and every number in this generation is measuring the harness. **Run this first regardless of everything else.** |
| **P2** | **Latent-informativeness probe** — freeze a trained checkpoint's encoder, fit a linear head from the 128-D latent to the ground-truth box pose / target pose | minutes | Directly answers §13.2. High error ⇒ the from-scratch encoder never learned the task variables ⇒ §13.1 is confirmed and pretrained/frozen features are the fix. Low error ⇒ perception is fine and the bottleneck is downstream. |
| **P3** | **State-only aligning through the same stack** (`base['ddpm_encdec_vision_nonvisual']`, `config/aligning-d3il-visual.py:781`; Gen7 `fix_18_nonvisual_step1`) | 1 train run | Works ⇒ the FM + MPC + harness stack is sound and the gap is purely perception. Fails ⇒ perception is not the story at all and §13.3(e)/planner is. |
| **P4** | **`mf_freeze_vision_encoder: True` + pretrained init** — one config line, plus wiring `weights='IMAGENET1K_V1'` or `get_r3m` | 1 train run | The direct test of §13.1. Trainable footprint drops 26.4 M → ~4 M, which is the right size for 900 episodes. |

**P1 → P2 → P3 → P4** is the order. P1 and P2 together cost less than one training job and can
invalidate or confirm the whole §13.1 thesis before any GPU-week is committed.

### 13.5 The direction: win on data efficiency, not on model size

Taking the limited-data constraint seriously as the *design premise* rather than an obstacle, the
build target is:

1. **Small trainable footprint.** ~4 M trainable (the bone), with the 22.36 M perception stack frozen
   or pretrained — the inverse of today's 85/15 split. This is the single change most aligned with a
   900-episode budget.
2. **Richer conditioning, not a richer trunk.** Obs window 1 → 2+ (velocity becomes observable), latent
   128 → wider if P2 says it is starved. Both are cheap in parameters; both add *information*, which is
   what data-limited regimes are short of — unlike width, which adds *capacity*, which is what they
   already have too much of.
3. **Keep the horizon honest.** `H = 8` wastes ~80 % of the U-Net bone (§5). Either raise `H` or shrink
   `dim_mults`; do not pay for blocks that see one timestep.
4. **Fix or replace the measurement.** Strict success at 0.3–2.1 % is 0–2 episodes per 30-cell; every DA
   on this task has abandoned success for distance and flagged the noise floor (U8 DA §9.6, U7 DA §227).
   A comparative claim cannot be made on an instrument that cannot resolve the comparison. P1 decides
   whether that floor is ours to fix; if it is, fixing it is worth more than any model change here.
   Until then, `avoiding-d3il` (S&C 1.000) is where the FM-vs-DPCC claim is testable, and visual
   aligning is a development target rather than a headline one.

**Pretrained encoder: yes — but as frozen features under a limited-data budget (P4), not as a
capacity upgrade, and after P1/P2 have said which failure it is fixing.
Bigger bone: no — §13.3(b), and it is the wrong direction for 900 episodes.**

### 13.6 What would change this verdict

- **P2 shows the 128-D latent linearly decodes box and target pose accurately** → perception is not the
  bottleneck, §13.1 is wrong, and attention moves to the planner/horizon/projector.
- **P1 shows a ground-truth demo fails our success criterion** → nothing above is measurable yet; fix
  the harness before running P2–P4.
- **P4 (frozen + pretrained, ~4 M trainable) leaves the 0.29–0.47 m band unmoved** → the data-efficiency
  thesis is falsified too, and the remaining suspects are the action space, the replan cadence and the
  400-step cap.

### 13.7 The pretrained-encoder idea: verdict, and what it actually costs

**Verdict: it is now the leading candidate, and parking it in the first draft was wrong.** But the
reason it is right is *not* "our model needs more capacity" — it is §13.1: **85 % of our trainable
parameters are a from-scratch ResNet-18 pair being fitted to 900 episodes.** That reframing decides
*which kind* of pretraining to reach for, so it matters.

Three different things get called "use a pretrained model". They have wildly different costs.

| | What it means | Change | Trainable params |
|---|---|---|---:|
| **A. Pretrained init** | `ResNet18Conv(pretrained=True)` — same architecture, ImageNet weights | ~1 line | 26.4 M |
| **B. Freeze** | `mf_freeze_vision_encoder: True` — encoder stops learning | 1 config line, already plumbed | **~4.0 M** |
| **C. Foundation encoder** | swap in R3M / DINOv2 / CLIP | real port | varies |

**A is one bool and nothing else moves.** `d3il/agents/models/vision/model_getter.py::get_resnet`
hard-codes `backbone_kwargs=dict(input_coord_conv=False, pretrained=False)`, and
`base_nets.py:510` passes it straight to `vision_models.resnet18(pretrained=...)`. Adding a
`pretrained: bool = False` kwarg to `get_resnet` is **backwards-compatible by default** — no other
generation changes behaviour — and then the flag is set from the `rgb_model` block of the encoder
config. ⚠️ That block is **byte-identical by design** between `visual_unet_twotime.py:83-96` and
`visual_dit_twotime.py:101-113` (red comment on both); the change must land in **both** or the
U-Net-vs-DiT comparison silently breaks and G0 will say so.

**B is already wired**, end to end and unused: `mf_freeze_vision_encoder`
(`visual_mf_diffusion.py:29,35,51`, `visual_af_diffusion.py:29,34,45`,
`train_mix_visual_aligning.py:500`), default `False` at `config/aligning-d3il-visual.py:1309,1349`.

**C needs care.** `get_r3m` does an unguarded `import r3m` (`model_getter.py:59`) — the package must
exist in the cluster env or the build dies at instantiation. Do not schedule C before A/B report.

#### The experiment is a 2×2, and the fourth cell is not optional

| init | encoder | trainable | what it isolates |
|---|---|---:|---|
| random | trained | 26.4 M | today's baseline |
| **ImageNet** | trained | 26.4 M | does the initialisation alone help? |
| **ImageNet** | **frozen** | **4.0 M** | the data-efficiency bet — the cell the idea points at |
| random | **frozen** | 4.0 M | **control** |

Without the fourth cell, a win in the third is confounded between *"pretrained features are good"*
and *"training 4 M instead of 26 M parameters on 900 episodes is good"*. Those are different claims
and only one of them generalises to the next task. Four runs, and P2 (§13.4) should be measured on
each checkpoint — a latent that linearly decodes box pose is the mechanism; the distance metric is
only the consequence.

#### Three things that will bite, found by reading the code

1. **🔴 The GroupNorm surgery discards part of the pretraining.** `use_group_norm=True` runs
   `replace_submodules(…, isinstance(x, nn.BatchNorm2d), → nn.GroupNorm(C//16, C))`
   (`multi_image_obs_encoder.py:62-69`). **Every BatchNorm2d in the pretrained ResNet is replaced by
   a freshly initialised GroupNorm** — its affine parameters and running statistics are thrown away.
   The conv filters survive, which is where most of the transferable structure lives, so this is not
   a blocker (robomimic and `diffusion_policy` both do exactly this and it works). But the network
   arrives *decalibrated*, which argues **against hard-freezing everything**: prefer
   *pretrained init + low-LR fine-tune*, or freeze the convs and let the GroupNorms train. A hard
   freeze of a BN-stripped ImageNet ResNet is the weakest version of this idea, and it is the one
   the existing flag implements — worth knowing before reading its result.

2. **🔴 The spatial bottleneck is 3×3.** `ResNet18Conv.output_shape` is
   `ceil(96/32) = 3` (`base_nets.py:535-537`), so the trunk emits `512 × 3 × 3` and SpatialSoftmax
   places **32 keypoints over 9 spatial positions**. Whatever the initialisation, sub-cell
   localisation of the box comes only from the softmax over a 3×3 grid. If P2 says the latent cannot
   decode box pose, **this** is a stronger suspect than the weights — and the fix is input
   resolution or a shallower stride, not pretraining. Test resolution and pretraining separately or
   the result is uninterpretable.

3. **No internet on compute nodes.** `pretrained=True` triggers a torchvision download into
   `~/.cache/torch/hub/checkpoints/`. Pre-fetch it on the login node once, or the first Slurm job
   fails at model construction — after the gates have passed, which makes it look like a code bug.
   One good thing: `imagenet_norm: True` is already on, so input normalisation already matches the
   pretrained statistics; nothing to change there.

#### Where it sits in the order

P1 (demo replay) and P2 (latent probe) still come first — they cost hours, and P2 in particular tells
you *whether the latent is the problem at all*, which is exactly the question the pretrained encoder
is being proposed to answer. If P2 says the 128-D latent already decodes box and target pose
accurately, the encoder is fine and this whole branch is dead; if it says the latent is
uninformative, the 2×2 above is the right next spend and **the frozen+pretrained cell is the one to
beat**. Either way P2 costs minutes and converts the idea from a bet into a measurement.
