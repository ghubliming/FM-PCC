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
