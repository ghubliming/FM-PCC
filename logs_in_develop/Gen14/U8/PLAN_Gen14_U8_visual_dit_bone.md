# PLAN — Gen14 U8: a Visual **DiT/SiT ML-bone** for the mf / af arms

> **Date**: 2026-08-20
> **Generation**: Gen14 (Visual-Mix-ML) — `mix_visual_aligning/` ↔ `mix_visual_aligning_test/`
> **Status**: PLAN — nothing implemented. No code touched by this document.
> **Prior art** (four documents, see §1.2 — read §1.2 before §5):
> - `logs_in_develop/Gen6_dpcc_Engine_for_visual_aligning/Gen6V2_Pending/backbone_switchability_analysis.md` (2026-05-17)
> - `logs_in_develop/Add_Visual_to_ML_bones/INVESTIGATION_visual_dit_sit_backbones.md` (2026-08-11)
> - `logs_in_develop/Gen3v7_AlphaFlow/bb_unet_ablation/RESULTS_Gen3v7_backbone_ablation_unet_vs_dit.md` (2026-07-25) — 🔴 **retracted**
> - `logs_in_develop/Gen3v7_AlphaFlow/Study/STUDY_why_af_sit_works_unet_not_and_mf_unet_works.md` (2026-08-19) — the retraction
> **Scope decided with user**: mf arm → DiT, af arm → DiT, **plus SiT for af if possible**.
> Placement: **additive inside `mix_visual_aligning/`**, as a NEW ML-bone alongside `VisualUNet` —
> not a new sibling generation folder.

---

## 0. TL;DR

| Question | Answer |
|:--|:--|
| Can the visual aligning task run on a DiT? | **Yes.** All four transformer backbones already live in `mix_visual_aligning/models/`; they are blocked from vision by one `raise` per arm, not by any architectural gap. |
| Is "ResNet → tokens → DiT" the right mental model? | **Half.** The encoder emits **one 128-D vector**, not tokens. Its 3×3 spatial map is destroyed by `SpatialSoftmax` *before* the vector exists. Getting real tokens is possible but is a separate, larger change (Option 3 below). |
| Which injection per backbone? | **Not a free choice** — see §3. `mf_dit`/`sit` are adaLN-native; `dit`/`af_dit` are prefix-token-native and have **no adaLN path whatsoever**. |
| Has a DiT ever beaten a U-Net here? | **No — and the one result that said so was retracted.** `bb_unet_ablation` (2026-07-25) reported DiT 3.5–7× better; the 2026-08-19 STUDY showed its "U-Net" was the 253 M Fix_8 build. At matched width the U-Net **Pareto-dominates** the SiT on the tightened DPCC arms. See §1.2(c)/§1.3 — read before budgeting GPU time. |
| Do we need to pull new upstream repos? | **Not for Options 1–2.** One is worth pulling anyway: `diffusion_policy` — already the upstream of our vision encoder, and the canonical trajectory-transformer-conditioned-on-images reference. See §12. |
| Biggest non-obvious risks | (a) checkpoint-path collision — `film_mode` is a path key, backbone is not; (b) the DiT at its default size is **2.5× the UNet's parameter count**, which silently confounds any A/B; (c) Gen14's "architecture-controlled" premise breaks the moment mf/af move and fm/diffusion cannot. |

---

## 1. Why this document exists

Gen14 pins all four arms (`diffusion`, `fm`, `mf`, `af`) to one backbone stack so the four-way
comparison isolates *objective + sampler*. The bone is `VisualUNet` / `VisualUNetTwoTime` —
a 1-D temporal conv U-Net with FiLM conditioning, in two flavours (`film_mode` v1 = additive
bias, v2 = per-block γ/β). U5 added v2; nothing since has touched the bone.

The 2026-08-11 investigation established that **no visual model in this repo has ever used a
Transformer**, and estimated adding one at ~1–2 days per backbone. This plan converts that
estimate into a concrete, verified design and evaluates the injection options against the
code as it stands on 2026-08-20.

### 1.1 Verified: nothing has been implemented since the investigation

- `mix_visual_aligning/models/mf_trajectory_model.py:82-87` — still raises
  `if_vision=True requires imf_backbone='unet'`.
- `mix_visual_aligning/models/af_trajectory_model.py:82-88` — same raise.
- No `VisualDiTTwoTime` (or equivalent) exists anywhere in the repo.
- `git log --oneline -- mix_visual_aligning/` shows 8 commits, none DiT-related.

---

### 1.2 Prior art — four earlier explorations, one already retracted

Searched the full repo and the git history for anything covering "visual backbone → Transformer".
Four documents matter, and they do not agree with each other.

#### (a) 2026-05-17 — Gen6V2, the ORIGINAL exploration

`Gen6_dpcc_Engine_for_visual_aligning/Gen6V2_Pending/backbone_switchability_analysis.md` (223 lines)

The first time this question was asked. It targets **D3IL's ACT VAE Transformer**, not a DiT, and
its verdict is *"Architectural Mismatch Detected (Requires Structural Adapters)"* — three hard
blockers in `d3il/agents/models/act/act_vae.py`:

| # | Blocker | Line | Why it breaks Gen6 |
|:--:|:--|:--|:--|
| 1 | `state = state[:, :1, :]` — history truncated to frame 1 | `act_vae.py:391` | Gen6 denoises proprioception across the whole window |
| 2 | `nn.Embedding(act_seq_size=4, hidden_dim)` fixed query bank | `act_vae.py:362` | cannot emit 8-step trajectories |
| 3 | `Linear(hidden_dim, action_dim)` — 3-D action head only | `act_vae.py:359` | cannot denoise the 6-D/9-D `[act, obs]` joint state |

🟢 **None of these apply to U8.** Every blocker is a property of *ACT's encoder-decoder VAE*, not
of transformers. The four DiT/SiT backbones already in `mix_visual_aligning/models/` take
`[B, H, D]` in and emit `[B, H, D]` out at arbitrary `H` and `D` — they were built for exactly
this trajectory shape. **The 2026-05 "no" was a correct answer to a different question**, and
this is why U8 goes via DiT rather than resurrecting the ACT path (which Gen5 abandoned into
`Archived_Codes/ddpm_encdec_vision_Legacy/`).

The same day's `MASTER_TEST_HISTORY.md:892` also recorded a *"20× capacity difference between
Gen5 U-Net (18 M+) and native ACT Transformer (~0.9 M)"* — the parameter-fairness concern of §8,
raised fifteen months before this plan and never acted on.

#### (b) 2026-08-11 — the DiT/SiT investigation

`Add_Visual_to_ML_bones/INVESTIGATION_visual_dit_sit_backbones.md`. Summarised throughout this
plan. Two corrections established here: the encoder output is keypoint coordinates rather than a
pooled map (§2), and Option 3 does **not** require touching `MultiImageObsEncoder` (§4).

#### (c) 🔴 2026-07-25 → 2026-08-19 — the DiT-beats-UNet result, and its retraction

This is the pair that changes what U8 should expect, and it is the reason this section exists.

`Gen3v7_AlphaFlow/bb_unet_ablation/RESULTS_Gen3v7_backbone_ablation_unet_vs_dit.md` (2026-07-25)
reported a spectacular state-only DiT win:

| goal+constr success (seed 6) | DiT | UNet | ratio |
|:--|--:|--:|--:|
| MeanFlow | **0.49** | 0.14 | 3.5× |
| α-Flow | **0.50** | 0.07 | 7× |

and concluded *"The backbone dominates the objective… the two-time field needs the transformer's
in-context h/attention tokens; the UNet's scalar-embedding h-conditioning is too weak."*

**That conclusion was wrong, and has been formally retracted.** The "UNet" in that ablation was
the 253 M `freq_dim=256` build — the Fix_8 defect. `Gen3v7_AlphaFlow/Study/STUDY_why_af_sit_works_unet_not_and_mf_unet_works.md`
(2026-08-19) re-ran it at the correct width and concluded the opposite:

> *"The puzzle was a mirage caused by a channel-width bug. With the bug fixed, BOTH objectives
> work on BOTH backbones, and AF + UNet@32 is the Pareto-best arm on the deployable tightened
> DPCC pipeline."*

At **parameter-matched** width, on the arms that actually deploy:

| K=2, tightened DPCC arms | AF UNet@32 (4.0 M) | AF SiT (~10 M) |
|:--|--:|--:|
| mean S&C, 3 tightened arms | **0.958** | 0.722 |
| `dpcc-c-tightened` | **0.96** / 91 steps | 0.25 / 177 steps |
| `dpcc-t-tightened` | **1.00** / 58 steps | 0.92 / 68 steps |

The SiT retains real advantages — untightened arms (0.79/0.83 vs 0.42/0.46: its raw field is
closer to feasible *before* projection), ~37% cheaper per step (0.019 s vs 0.030 s), and a
genuinely better `(t, h)` representation — but it also inherits MF-DiT's *"crushed to a point"*
collapse on `dpcc-c`, which the UNet does not have. Training cost runs the other way: ~11 h/seed
for the DiT vs ~4 h/seed for the UNet.

### 1.3 🔴 What (c) means for U8

Three things, and they should be read before anyone budgets GPU time for this:

1. **There is no evidence a DiT beats a U-Net at matched capacity on this repo's tasks.** The one
   result that said so was a capacity artefact. U8 must not be planned or written up as
   "upgrading to the better backbone" — it is *adding a second backbone so the question can be
   asked honestly for the first time in the visual setting*.
2. **§8's parameter matching is not pedantry — it is the entire lesson of (c).** An unmatched
   visual DiT run would reproduce a retraction that already cost this project a month.
3. **The state-only evidence predicts the DiT loses on the tightened arms**, which are the
   deployable ones. That is a real possibility the plan should be able to publish: "we built it,
   parameter-matched it, and the U-Net still wins" is a legitimate and citable outcome, and it
   would corroborate (c) in the visual setting. Plan the gates and the DA around being able to
   state that cleanly, not around a win.

---

## 2. 🔴 What the vision encoder ACTUALLY produces

This is the correction that reshapes the whole design. The investigation MD describes the
encoder output as a "pooled 128D vector". The truth is stronger, and it matters.

```
 per camera:
   96×96×3
      │
      ├─ ResNet18Conv                  (torchvision resnet18 minus avgpool+fc)
      │      → [512, 3, 3]             ← ceil(96/32) = 3. THE REAL SPATIAL MAP.
      │
      ├─ SpatialSoftmax(num_kp=32)     ← 🔴 SPACE IS DESTROYED HERE
      │      → [32, 2]                   32 keypoints, each an (x, y) COORDINATE
      │
      ├─ Flatten → 64
      └─ Linear(64, 64) → 64-D

 two cameras (share_rgb_model=False → independent ResNets):
      concat → 128-D

 then, in FM-PCC:
   VisualUNetTwoTime.encode_visual() mean-pools over the T_win window → (B, 128)
```

**Sources** (all vendored, all read for this plan):
- `d3il/agents/models/vision/model_getter.py:7-31` — it is a robomimic `VisualCore`
  (`ResNet18Conv` + `SpatialSoftmax` + flatten + `Linear`), **not** a plain torchvision ResNet.
- `d3il/agents/models/robomimic/models/base_nets.py:522-537` — `output_shape` = `[512, ceil(H/32), ceil(W/32)]` ⇒ `[512, 3, 3]`.
- `d3il/agents/models/robomimic/models/obs_core.py:108-139` — the backbone → pool → flatten → linear chain.
- `mix_visual_aligning/models/visual_unet_twotime.py:160-168` — `encode_visual`, the window mean-pool.

### 2.1 The consequence for "input as token"

There is nothing token-shaped in the current pipeline. Feeding the encoder output to a DiT
"as a token" means feeding **exactly one** token — a global summary vector. That is a valid
design (it is how DiT conditions on a class label), but understand what it is *not*: a DiT
conditioned on one global vector is a FiLM-UNet with attention instead of convolutions. The
spatial cross-referencing that motivates a transformer is not present, because the spatial
map was collapsed two layers upstream.

### 2.2 Parameter reality check — the bone is a minority of the model

| Component | Params | Note |
|:--|--:|:--|
| Vision encoder (2× ResNet18Conv + pool + linear) | **≈ 22.4 M** | identical for every bone; never changes |
| `VisualUNetTwoTime` bone (`dim=32`, mults 1/2/4/8) | **≈ 4.0 M** | the Fix_8 baseline width |
| **Total visual mf/af model today** | **≈ 26.4 M** | the generative bone is **~15%** of it |

`dim=32` comes from `config/aligning-d3il-visual.py:486` (the `fm_visual_aligning` parent that
the mf/af arms inherit), *not* from `freq_dim` — in the `if_vision` branch `freq_dim` is unused
and the bone is built from `vis_config.dim`. Do not repeat the Fix_8 mistake in reverse.

---

## 3. Backbone inventory — what each one can actually accept

Four transformer backbones exist in `mix_visual_aligning/models/`. **Their conditioning
mechanisms are not interchangeable**, so the injection strategy is dictated per backbone, not
chosen freely.

| Key | Class / file | Conditioning mechanism | adaLN available? | Prefix tokens available? |
|:--|:--|:--|:--:|:--:|
| `mf_dit` | `MFDiTOfficialTrajectory` — `mf_dit_official_trajectory.py:261` | adaLN-zero, `c = t_emb + r_emb + w_emb`, learned **absolute** sin-cos pos-embed | ✅ native | ⚠️ needs pos-embed resize |
| `sit` | `AFSiTTrajectory` — `af_sit_trajectory.py:252` | adaLN-zero, `c = t_emb + r_emb` (α-Flow's SiT, y-embedder off) | ✅ native | ⚠️ needs pos-embed resize |
| `dit` (mf) | `MFDiTTrajectory` — `mf_dit_trajectory.py:241` | **in-context prefix tokens only** (class/ω/t_min/t_max/time), RoPE | ❌ **none** | ✅ native |
| `dit` (af) | `AFDiTTrajectory` — `af_dit_trajectory.py:241` | same iMF port as above | ❌ **none** | ✅ native |

🔴 **The finding that constrains the plan**: `MFDiTTrajectory`'s block signature is
`forward(self, x, cos, sin)` (`mf_dit_trajectory.py:201`) — it never receives a conditioning
vector. There is no adaLN modulation anywhere in the iMF DiT. Everything it conditions on is
already a prepended token. So:

> For `mf_dit` / `sit` → adaLN is the natural route.
> For `dit` / `af_dit` → a prefix token is the **only** route.

The investigation MD's Strategy A/B split turns out to be forced by architecture rather than a
preference, which is a good sign: each backbone gets vision the same way it gets everything else.

---

## 4. The injection options — introduced and evaluated

### Option 1 — adaLN sum ("vision as a modulation signal")

```python
# mf_dit_official_trajectory.py, in forward():
c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
if self.use_visual and visual_latent is not None:
    c = c + self.vis_embedder(visual_latent)      # nn.Linear(128, d) → SiLU → nn.Linear(d, d)
for block in self.blocks:
    x = block(x, c)                               # every block modulates on c already
```

- **What the network sees**: a per-sample scale/shift/gate applied uniformly to every
  trajectory step, in every block, including the two `FinalLayer`s.
- **Sequence length**: unchanged (8 tokens at H=8, patch=1). Pos-embed untouched.
- **Applies to**: `mf_dit`, `sit`. **Not possible** for `dit`/`af_dit`.

**Evaluation**

| Verdict | Detail |
|:--:|:--|
| ✅ | Smallest diff in the plan: one MLP + ~4 lines per backbone. |
| ✅ | Architecturally faithful — this is literally DiT's class-label conditioning. |
| ✅ | JVP-safe by construction: `visual_latent` is a captured constant, adaLN is `x·(1+s)+b`, forward-AD trivial. |
| ✅ | Zero-init preserved: adaLN output layers are zeroed in `initialize_weights`, so step 0 is identity — same "grows into the gates" property that made FiLM v2 stable. |
| ⚠️ | Conditioning is **global and uniform**: every trajectory step gets the same modulation. No step-dependent visual reasoning. |
| ⚠️ | Functionally very close to FiLM v2 on the UNet. An A/B against `film_mode=v2` measures *conv vs attention*, which is a real question — but do not expect it to measure "better use of vision". |

### Option 2 — one visual prefix token ("vision as a token")

```python
# mf_dit_trajectory.py, in _build_sequence():
vis_tok = self.vis_tokens[None] + self.vis_projector(visual_latent)[:, None]   # [B, 1, d]
return torch.cat([class_tok, omega_tok, tmin_tok, tmax_tok, time_tok, vis_tok, x_embed], dim=1)
```

- **What the network sees**: one extra sequence position that every trajectory token can attend
  to, with a learned per-head, per-layer attention weight.
- **Sequence length**: 8 → 9 (mf_dit/sit) or prefix 7 → 8, total 15 → 16 (iMF dit).
- **Applies to**: all four — natively to `dit`/`af_dit`, with a pos-embed resize on `mf_dit`/`sit`.

**Evaluation**

| Verdict | Detail |
|:--:|:--|
| ✅ | The only route for `dit`/`af_dit`, and native there: `prefix_tokens` and the RoPE table are already computed from a token count. |
| ✅ | Strictly more expressive than Option 1 for the same 128-D input: attention weights are per-head and per-layer, so different depths can weight vision differently. |
| ✅ | Still JVP-safe — a constant token, then softmax/RoPE, all forward-AD friendly. RoPE here is the **real-valued interleaved** form (`mf_dit_trajectory.py:135-149`), deliberately not the complex bitcast, precisely so JVP works. |
| ⚠️ | **The bookkeeping trap.** `self.prefix_tokens` (`mf_dit_trajectory.py:297`) is used to strip the prefix at the output (`:381` `u_seq[:, self.prefix_tokens:]`), and `total_tokens` sizes the RoPE table at `:299-303`. Adding a token without bumping **both** yields a model that trains fine and reads the wrong positions. Silent, plausible-looking failure. |
| ✅ | Mitigating detail: the RoPE tables are registered `persistent=False`, so they are rebuilt at construction and are **not** in the checkpoint — resizing them does not corrupt loading. |
| ⚠️ | For `mf_dit`/`sit`, `pos_embed` **is** a `nn.Parameter` of shape `[1, num_patches, d]` and **is** in the state_dict. A prefix token there means deciding whether the visual position gets a pos-embed entry, and it changes the checkpoint shape. |

### Option 3 — real spatial visual tokens ("what a transformer is actually for")

Tap the ResNet trunk **before** `SpatialSoftmax`:

```python
feat = self.obs_encoder.key_model_map[cam].backbone(img)   # [B*T, 512, 3, 3]
tokens = feat.flatten(2).transpose(1, 2)                   # [B*T, 9, 512]  → 9 tokens/cam
# 2 cams → 18 visual tokens → Linear(512, d) → prepend (or cross-attend)
```

- **Sequence length**: 18 visual + 8 trajectory = 26 tokens. Cheap — 3×3 is tiny.
- **Applies to**: all four, as prepended tokens.

**Evaluation**

| Verdict | Detail |
|:--:|:--|
| ✅ | The **only** option where the DiT can do something the FiLM-UNet structurally cannot: attend to a specific image region per trajectory step. |
| ✅ | The investigation MD overstates the cost: it claims this needs "changes to `MultiImageObsEncoder`". **It does not.** `self.obs_encoder.key_model_map[key]` is the per-camera `VisualCore` and `.backbone` is the ResNet trunk (`obs_core.py:104`) — an additive read. The 128-D path stays intact and both can coexist behind one knob. The transforms in `key_transform_map[key]` must be applied first. |
| ✅ | Still JVP-safe: pre-encode outside the JVP exactly as today, just a larger captured constant. |
| ⚠️ | **It discards D3IL's chosen inductive bias.** `SpatialSoftmax(num_kp=32)` is not a lazy pool — keypoint coordinates are a strong, well-motivated prior for manipulation. Bypassing it is a research bet, not a free upgrade. |
| ⚠️ | 18 of 26 tokens are visual at H=8. The trajectory is a **minority of its own sequence**. Unknown whether that helps or drowns the signal at 96 demos. |
| ⚠️ | Data budget. 96 demonstration trajectories. `Gen13/fix_4/ARCHITECTURE_Gen13_iMF_in_HardFlow.md:31` already argued a DiT is data-starved at this scale; adding 18 tokens of visual attention raises that risk further. |
| ⚠️ | Largest diff, new encoder tap, new failure surface. Deserves its own unit, evaluated against Option 1/2 rather than bundled with them. |

### 4.1 Comparison at a glance

| Property | Opt 1 adaLN | Opt 2 one token | Opt 3 spatial tokens |
|:--|:--:|:--:|:--:|
| Uses existing 128-D encoder output unchanged | ✅ | ✅ | ❌ (new tap) |
| Works on `mf_dit` / `sit` | ✅ | ⚠️ pos-embed resize | ⚠️ pos-embed resize |
| Works on `dit` / `af_dit` | ❌ impossible | ✅ native | ✅ |
| Sequence length at H=8 | 8 | 9 (or 16) | 26 |
| Step-dependent visual reasoning | ❌ | ⚠️ weak | ✅ |
| Spatial grounding | ❌ | ❌ | ✅ |
| JVP-safe | ✅ | ✅ | ✅ |
| New failure modes | ~none | prefix/RoPE off-by-one | encoder tap, token balance |
| Est. effort | 0.5 d | 1 d | 3–4 d |

---

## 5. Recommendation

**Phase A (this unit, U8)** — build the bone and ship Options 1 + 2 behind one knob:

| Arm | Backbone key | Injection |
|:--|:--|:--|
| `mf` | `mf_dit` (official MeanFlow DiT, adaLN-zero) | **Option 1** (adaLN), default |
| `mf` | `dit` (iMF RoPE DiT) | **Option 2** (prefix token) — the only route |
| `af` | `sit` (α-Flow SiT, adaLN-zero) | **Option 1** (adaLN), default |
| `af` | `dit` (iMF RoPE DiT) | **Option 2** (prefix token) — the only route |

This satisfies the scope as stated — *mf and af arms both get a DiT, and SiT is not merely
possible for af, it is the α-Flow-native choice and needs the same one-line adaLN change as
`mf_dit`.* Options 1 and 2 share the wrapper, the `encode_visual`/`resolve_visual_cond` pair
and the `cond_dim` constructor argument; only the last few lines of each `forward` differ, so
building both costs little more than building either.

### 5.1 Expected payoff — state it honestly up front

Per §1.3, the only parameter-matched backbone evidence this repo owns says the **U-Net wins on the
deployable tightened arms** and the transformer wins on untightened arms and per-step latency.
There is no reason to expect the visual setting to flip that on its own — if anything the
opposite, since a global 128-D conditioning vector (Options 1–2) gives the transformer none of the
spatial leverage that would justify it.

So the honest framing of Phase A is: **make the visual bone swappable and measure it**, with three
publishable outcomes, all of them useful:

| Outcome | Reading |
|:--|:--|
| DiT ≈ UNet at matched params | corroborates §1.3 in the visual setting; the bone is not the lever |
| DiT < UNet on tightened arms | corroborates the state-only Pareto result; strengthens the U-Net headline |
| DiT > UNet | the first matched-capacity transformer win in this repo — and the case for Phase B |

Do not budget this as an upgrade. Budget it as the experiment that lets §1.3 be tested rather than
assumed.

**Phase B (a later unit)** — Option 3 (spatial tokens), evaluated against Phase A's numbers.
It is a research bet with its own failure modes and should not ride along on a plumbing unit.

---

## 6. Implementation plan (Phase A)

### 6.1 New file — `mix_visual_aligning/models/visual_dit_twotime.py`

`VisualDiTTwoTime`, the sibling of `VisualUNetTwoTime`. Roughly 140 lines, most of it copied.

1. Owns the `MultiImageObsEncoder` — **byte-identical `shape_meta` / `obs_encoder_cfg`** to
   `visual_unet_twotime.py:76-96`. Any drift here silently breaks bone-vs-bone comparability.
2. `encode_visual()` and `resolve_visual_cond()` copied verbatim — this is the contract
   `VisualMeanFlow._visual_backbone()` (`visual_mf_diffusion.py:41`) depends on.
3. Selects the DiT class from a `dit_variant` key and passes `cond_dim=128`.
4. `forward(...)` accepts the full two-time surface (`h`, `omega`, `t_min`, `t_max`, `return_v`)
   and forwards `visual_cond` as the backbone's `cond` argument.
5. **No horizon padding.** The U-Net pads to a multiple of 8 for its stride-2 levels; a DiT at
   `patch_size=1` takes H=8 as 8 tokens directly. Dropping the pad also drops the crop-back.

### 6.2 Backbone edits (additive, `cond_dim=0` ⇒ byte-identical to today)

- `mf_dit_official_trajectory.py` — `cond_dim` arg; `vis_embedder`; `c = c + vis_embedder(cond)`.
- `af_sit_trajectory.py` — identical change.
- `mf_dit_trajectory.py` / `af_dit_trajectory.py` — `cond_dim` arg; `vis_projector` +
  `vis_tokens` parameter; one extra entry in the `_build_sequence` concat; **`prefix_tokens += 1`
  and `total_tokens += 1` before the RoPE precompute.**

🔴 Every one of these guards on `cond_dim > 0`. At `cond_dim=0` the modules must construct an
identical state_dict to today, so the three state-only generations (Gen3v4/v6/v7) that import
the same architecture are provably unaffected.

### 6.3 Trajectory-model routing

`mf_trajectory_model.py:77-95` and `af_trajectory_model.py:77-95` — replace the blanket raise
with: `unet` → `VisualUNetTwoTime`; `mf_dit`/`dit`/`sit` → `VisualDiTTwoTime`; anything else
still raises. Keep the `state_dim` pin to the visual transition dim (9) for the aux head.

The `(u, v)` return path already works: `mf_trajectory_model.py:176` routes
`imf_backbone in ('dit', 'mf_dit')` down the `return_v=True` branch, and both DiTs carry native
twin heads. `af_trajectory_model.py:175` does the same for `('dit', 'sit')`. **No change needed
to the objective, the JVP, or the sampler.**

### 6.4 Plumbing — the hop that does not exist yet

`mix_visual_aligning_test/train_mix_visual_aligning.py:359-374` builds the mf/af `model_config`
**without `imf_backbone` or any `dit_*` key**, so the engine default `'unet'` is currently the
only reachable value. Add them there, validated and printed alongside the existing `film_mode`
check at `:346-352`.

Eval needs no code change for reconstruction: `eval_mix_visual_aligning.py:2291` loads
`model_config.pkl` and `:2355` calls `model_config()`, so any constructor kwarg written at train
time reaches eval for free. It *does* need the `film_mode` print at `:2309-2316` taught not to
claim a FiLM mode for a DiT checkpoint.

### 6.5 Config

New per-arm knob in `config/aligning-d3il-visual.py`, resolved exactly like `_film_mode()`
(`:1081-1106`) — env override `MIX_BONE_MF` / `MIX_BONE_AF`, then `MIX_BONE`, then `'unet'`;
unknown values **raise**, never fall back. Plus DiT sizing keys (§7).

---

## 7. 🔴 Path identity — the trap flagged a year in advance

`CHANGELOG_Gen14_U5_engine_rename_and_twotime_filmv2.md:208-209` says it plainly:

> *"if a vision-capable DiT ever lands, **add the key to the watch list first**, or U-Net and
> DiT checkpoints will collide in the same directory."*

Two things must happen together, before any DiT job is submitted:

1. **Add the bone key to `args_to_watch_mix_visual_train`** (`config/aligning-d3il-visual.py:881`)
   and to `args_to_watch_mix_visual_plan`. `_mix_plan_block`'s mirror loop then propagates it to
   the plan block and `_mix_loadpath` rebuilds `diffusion_loadpath` from the same list — the
   scheme is already collision-proof *once the key is in the list*.
2. **Suppress `film_mode` on DiT bones.** FiLM is a U-Net concept; a `_filmv1_` fragment on a
   DiT path is an actively lying directory name. `watch()` skips keys a block does not define,
   so the fix is for the DiT config path not to define `film_mode` at all.

**Sizing keys are also identity keys.** `dit_hidden_size` and `dit_depth` change the state_dict;
if they are tunable they belong in the watch list too, otherwise two differently-sized DiTs
overwrite each other. Simplest safe alternative: pin them in the config block and treat a change
as requiring a new bone name.

---

## 8. 🔴 Fairness — the DiT is 2.5× the U-Net at its defaults

| Bone | Config | Bone params | vs U-Net |
|:--|:--|--:|--:|
| `VisualUNetTwoTime` | `dim=32`, mults (1,2,4,8) | **≈ 4.0 M** | 1.0× |
| `MFDiTOfficialTrajectory` | **defaults** `dit_hidden_size=256`, `dit_depth=8` | **≈ 9.9 M** | **2.5×** |
| `MFDiTOfficialTrajectory` | `dit_hidden_size=160`, `dit_depth=8` | **≈ 3.9 M** | **1.0×** ✅ |

Arithmetic: an adaLN DiT block is ≈ `18·d²` params (qkv `3d²`, proj `d²`, MLP `8d²` at
ratio 4, adaLN `6d²`). At `d=256, L=8`: `18·8·65536 ≈ 9.44 M`, plus three `TimestepEmbedder`s
and two `FinalLayer`s ≈ 9.9 M. At `d=160`: `18·8·25600 ≈ 3.69 M` ≈ 3.9 M total. `head_dim = 40`
at 4 heads — even, so RoPE is fine.

This is the **Fix_8 lesson repeating in the opposite direction**.
`Gen14/Fix_8/NOFIX_ANALYSIS_Gen14_unet_architecture_comparison.md:172` calls the Gen3v6 `bbunet`
A/B — *"253 M vs 10 M DiT"* — a mistake precisely because the two arms differed in size. Running
a 9.9 M DiT against a 4.0 M U-Net and reporting "DiT wins" would reproduce that error exactly.

**Therefore**: `dit_hidden_size=160, dit_depth=8` is the *default* for the visual bone, and
`params=...M` must be printed at build time (`mf_trajectory_model.py:147` already does this —
it just needs to be read).

Note the honest caveat: at 26.4 M total, the ResNet encoder is ~85% of the model either way, so
bone parameter matching is a smaller lever than it looks. Report both numbers.

---

## 9. JVP safety

Unchanged from today, and the reason is structural rather than incidental.
`VisualMeanFlow.loss()` (`visual_mf_diffusion.py:74-84`, the `_encode_once` call at `:82`) encodes **once**, up front, and passes
the result down as `cond['visual_latent']`. Inside `_p_losses_meanflow`'s JVP closure that
tensor is a captured **constant**, so its forward-mode tangent is zero *by construction* — the
ResNets never enter the differentiated function. That mechanism lives in the engine, above the
bone, so it protects the DiT identically. `VisualDiTTwoTime.resolve_visual_cond()` only has to
honour the same `visual_latent` → `visual` preference order.

Per-option: adaLN is `x·(1+s)+b` — trivially forward-AD friendly. Prefix tokens add a constant
row, then softmax attention and real-valued interleaved RoPE, all forward-AD friendly (the
complex bitcast was deliberately avoided — `mf_dit_trajectory.py:27`, `:135-149`).

---

## 10. Validation (all **run on cluster** — nothing executes in this container)

Extend `mix_visual_aligning_test/gates_mix_visual.py` with a bone gate:

| # | Gate | Passes when |
|:--|:--|:--|
| G-B1 | **Zero-diff regression** — build all four DiTs at `cond_dim=0` | state_dict keys and shapes byte-identical to pre-U8 |
| G-B2 | Build each visual bone (`mf_dit`, `dit`, `sit`) at `if_vision=True` | constructs; `vis_embedder`/`vis_projector` present; printed `params` within 10% of 4.0 M |
| G-B3 | **Vision is actually live** | grad w.r.t. `vis_embedder.weight` is non-zero after one loss step (the FiLM-v2 lesson: a zero-init path can look wired and be inert) |
| G-B4 | One `mf` loss step under the JVP | finite; no forward-AD `NotImplementedError` |
| G-B5 | One `af` loss step on `sit` and on `dit` | finite (α-Flow's bootstrap re-enters the backbone — exercise it) |
| G-B6 | Prefix bookkeeping | `prefix_tokens` + `num_patches` == RoPE table length; output after stripping is `[B, 8, 9]` |
| G-B7 | Path identity | two configs differing only in bone resolve to **different** `savepath`, and each `diffusion_loadpath` round-trips |
| G-B8 | Train → eval round-trip | a 50-step checkpoint loads in eval and produces a rollout |

---

## 11. 🔴 What this does to the Gen14 comparison

Gen14's premise is *"the backbone is locked to the VisualUNet stack, so the four-way comparison
is architecture-controlled: only objective + sampler vary"* (`config/aligning-d3il-visual.py:863-865`).

**A DiT on mf/af alone breaks that premise**, and it cannot be fixed within this unit: the
`diffusion` and `fm` arms are single-time (`wraps_unet=False`), and **no single-time visual
transformer exists anywhere in this repo**. There is nothing to switch them to.

So the results discipline is:

- ✅ **Valid**: `mf@unet` vs `mf@mf_dit` — same objective, same data, same encoder, bone varies.
  A clean bone A/B, and the actual question this unit answers.
- ✅ **Valid**: `mf@mf_dit` vs `af@sit` — *only* if both are parameter-matched and both use the
  same injection option. Otherwise it confounds objective with bone.
- ❌ **Invalid**: `mf@mf_dit` vs `fm@unet`. This is the `Gen15/init/PLAN_Gen15_uav_mix_ml.md:450`
  trap verbatim — *"Running MF-on-DiT against FM-on-UNet answers nothing."*
- 🔴 **Against the DPCC baseline**: per the standing rule that the architecture-matched result is
  the strong claim, the **U-Net** rows stay the headline. Any DiT win is a secondary,
  bone-confounded result and must be reported carrying its backbone and parameter count.

---

## 12. Upstream repos worth pulling into `aux_repo/`

`ls /workspaces/aux_repo/` today: `alphaflow`, `d3il`, `diffuser`, `dpcc`, `drifting`,
`drifting_policy`, `flow_guidance`, `HardFlow`, `HardFlow_Paper_Files`, `imeanflow`, `l4casadi`,
`MeanFlow`, `mujoco_mpc`, `SafeFlowMPC`, `Self-Flow`, `UAV-Flow`.

Every DiT/SiT *architecture* reference is already covered: `MeanFlow/models/dit.py` and
`alphaflow`'s SiT are the sources the four ports were made from, and the ports are documented as
verbatim. **Nothing new is needed for Options 1 and 2.**

What is genuinely missing is an upstream that answers *"how do you condition a **trajectory**
transformer on **image** features?"* — which is a robot-learning question, not a generative-model
question, and none of the sixteen repos above address it (they are all state-only).

| Priority | Repo | Why it is the right reference |
|:--:|:--|:--|
| 🥇 **1** | **`diffusion_policy`** (Chi et al., real-stanford) | 🔴 **This repo is already the upstream of our vision encoder.** `d3il/agents/models/vision/multi_image_obs_encoder.py` imports `CropRandomizer`, `ModuleAttrMixin`, `dict_apply`, `replace_submodules` — all verbatim diffusion_policy module names. D3IL vendored it. Its `TransformerForDiffusion` is a 1-D **trajectory** transformer conditioned on that exact encoder's output, i.e. our precise problem, solved by the people who wrote the encoder. It settles Option 1-vs-2 empirically rather than by argument, and shows their conditioning-dropout / time-embedding conventions. |
| 🥈 **2** | **`act`** (tonyzhaozh/act) | Only needed for **Phase B**. D3IL vendors just `act_vae.py` — a partial copy. The full DETR-style repo is the canonical example of trajectory queries cross-attending to **ResNet spatial feature maps**, which is Option 3's design. It is also the model the Gen6V2 doc (§1.2a) failed to adapt, so having the full source makes that verdict re-checkable. |
| 🥉 **3** | `octo` (rail-berkeley) | Optional. A transformer policy where observations *are* tokens — useful prior art for the token-budget question in §4 Option 3 (18 visual vs 8 trajectory tokens). JAX, heavy; read for design, not for code. |
| — | `DiT` (facebookresearch) | **Not needed.** `aux_repo/MeanFlow/models/dit.py` is already the adaLN-zero reference our port was made from. |

**Recommendation: pull `diffusion_policy` now** — it is the one that changes how Phase A is
written, and it is the direct ancestor of code already running in this repo. `act` only when
Phase B starts. Neither is a blocker: Options 1 and 2 can be implemented today from the ports
already in `mix_visual_aligning/models/`.

---

## 13. Non-goals

- The `diffusion` and `fm` arms. They import only verbatim Gen6V4/Gen7 files
  (`engine_registry.py`, "THE STRUCTURAL RULE") and nothing in this plan may reach them.
- Option 3 (spatial tokens) — deferred to Phase B.
- Cross-attention blocks (the MD's Strategy C proper) — not planned; if Option 3 lands,
  prepended tokens come first, cross-attention only if tokens demonstrably help.
- Any change to `MultiImageObsEncoder`, `get_resnet`, or the image preprocessing.
- The UAV mix generation (Gen15) — `PLAN_Gen15_uav_mix_ml.md:545` already defers DiT/SiT there.
  If U8 works, that deferral is worth revisiting; it is not in scope now.
- `MASTER_TEST_HISTORY.md` — not edited by this plan.

---

## 14. Open decisions before coding

1. **Bone key naming.** `imf_backbone` is the existing internal name but it is an iMF-era word
   for a Gen14 config key. Proposal: config key `ml_bone` ∈ `{unet, mf_dit, dit, sit}`, mapped
   to `imf_backbone` at the train-script boundary, with the path fragment `B{ml_bone}`.
2. **Injection knob exposure.** ✅ **DECIDED** — see [`DECISION_Gen14_U8_injection_choice.md`](./DECISION_Gen14_U8_injection_choice.md): Option 2 (visual token) only, pinned per backbone, no knob. Original wording kept below for the record.
   ~~ `vis_inject` ∈ `{adaln, prefix1}` as a real config key (an
   identity key, therefore a path key), or pinned per backbone to its native route with no knob?
   Pinning is simpler and less likely to be mis-set; a knob buys an adaLN-vs-token A/B on
   `mf_dit`, which is a genuinely interesting question.~~
3. **DiT size.** Confirm `dit_hidden_size=160, dit_depth=8` as the parameter-matched default —
   or accept the 2.5× default and always report the gap.
4. **`p_mean`/`p_std` and the rest of the objective constants** stay untouched, correct?
   Nothing in this plan changes them, but a bone swap sometimes tempts a re-tune. It should not.
