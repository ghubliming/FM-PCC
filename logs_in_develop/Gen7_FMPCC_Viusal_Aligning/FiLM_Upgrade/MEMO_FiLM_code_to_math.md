# MEMO — FiLM From Code to Math (v1 "Fake FiLM" vs v2 "True FiLM")

**Date**: 2026-06-27
**Purpose**: A self-contained note tracing the *exact code lines* to the *math* for both the
current additive-bias conditioning (`film_mode: 'v1'`) and the new True FiLM (`film_mode: 'v2'`).
Read alongside [PLAN_FiLM_V2.md](./PLAN_FiLM_V2.md) and [CHANGELOG_FiLM_V2.md](./CHANGELOG_FiLM_V2.md).

---

## TL;DR (read this if the math gives you a headache)

The robot looks at two cameras and turns the picture into a list of numbers — call it the
**"what I see" note** (`v`). That note has to influence the trajectory the model draws.
The only question is: **how strongly is the model allowed to use that note?**

- **v1 (old, "Fake FiLM")** — the model can only **ADD** the note. It's like a volume *offset*:
  "given what I see, nudge every channel up or down by a fixed amount." It can push, but it
  can never turn a channel **off** or **double** it.

- **v2 (new, "True FiLM")** — the model can **MULTIPLY then ADD** the note. Now it has a real
  **volume knob (×)** *plus* the offset (+): "given what I see, turn this feature off, crank
  that one up, then nudge." Strictly more powerful.

**One-line picture:**

```
v1:  result = feature            + note      ← can only nudge   (push)
v2:  result = feature × (knob)   + note      ← can gate & nudge  (push + dial)
```

That's the whole idea. `×` is the new power. Everything below is just the bookkeeping of where
that `×` lives in the code and the math.

> **Nothing is broken or slower for current work.** v1 stays the default; v2 only switches on
> when you set `film_mode: 'v2'` and retrain.

---

## First-Principles Version (no notation, plain English)

**The setup.** A "denoiser" repeatedly cleans up a rough trajectory until it's a smooth plan.
At each cleaning step it processes the trajectory through ~16 small blocks (a U-Net). We want
the camera image to steer that cleaning.

**The image becomes a note.** Two cameras → a neural net → one fixed-length list of numbers,
`v` (128 numbers). Think of `v` as a compressed summary: *"box is here, target is there."*
This step is **identical** in v1 and v2 — both start from the same note.

**The real question: how does the note touch each block?** Inside every block the trajectory
is represented as a stack of "feature channels" (think audio tracks). The note must influence
those tracks. Two ways to do it:

1. **Add (shift).** For each track, add a number derived from the note.
   *Analogy:* a graphic equalizer where the note can only raise/lower each band by a fixed
   amount. You can emphasize, but you can't mute a band or invert it. **← this is v1.**

2. **Multiply, then add (scale + shift).** For each track, first multiply by a factor derived
   from the note (`×0` = mute, `×2` = double, `×−1` = flip), *then* add the offset.
   *Analogy:* the same equalizer but now each band has a real gain knob, not just an offset.
   **← this is v2 ("True FiLM").**

**Why v2 doesn't blow up on day one.** We start the multiply-factor at exactly "×1 and +0", so
on the very first training step v2 behaves *identically* to having no image steering at all.
It only *learns* to use the knobs gradually. So switching to v2 can't destabilize early training.

**Why bother.** For the current easy task (one box, one target) just *adding* is usually enough —
the note says "go up-left," adding that is fine. But when scenes get harder (clutter, multiple
objects), the model needs to **ignore** irrelevant tracks (mute) and **focus** on relevant ones
(boost). Only `×` can do that. v2 buys that headroom; v1 can't.

**The cost.** v2 is a different network shape, so old saved models (`.pth`) can't load into it —
you must train v2 from scratch. That's why it's opt-in and gets its own checkpoint folder.

---

## Is v1 Actually Worse? — Honest Answer: NOT Strictly (and sometimes v1 wins)

Short version: **v2 is more *powerful*, but "more powerful" ≠ "always better."** In theory v2
can never be worse; in the real world (finite data, real optimizers, an easy task) v1 can tie
or beat it. Here's the careful split.

### The one place v2 is guaranteed not-worse: pure representation

v2 **contains** v1 as a special case. Set every scale `γ = 0` and v2's formula
`(1+γ)·h + β` collapses to `1·h + β = h + β` — exactly v1. So *if you had infinite data and a
perfect optimizer*, v2 could always at least match v1 by learning `γ=0` where the knob isn't
useful. In that idealized sense v2 ≥ v1.

> But we never have infinite data or a perfect optimizer. So this guarantee is theoretical, not
> practical. The practical question is **generalization**, and there v1 can win.

### Five real reasons v1 can match or BEAT v2

| # | Reason | Why it helps v1 |
|---|---|---|
| 1 | **Small dataset (~900 demos)** | v2's extra ~1.2M knobs are extra capacity to **overfit**. When the task doesn't need gating, those knobs fit noise instead of signal → worse test/rollout. v1's smaller hypothesis class generalizes better (Occam / bias-variance). |
| 2 | **The task is easy** | One box, one target. The image basically says "go this direction." An additive *shift* already encodes a direction perfectly. Gating (mute/boost channels) provides **no useful signal** here, so it's wasted (and overfit-prone). |
| 3 | **Multiplicative terms are harder to optimize** | `γ·h` couples the conditioning and the feature multiplicatively → a bumpier, less convex loss surface, larger/again amplified gradients, more sensitivity to LR. Additive conditioning is gentler and more stable. (Zero-init helps v2 *start* safe, but the learned regime can still be twitchier.) |
| 4 | **Strong empirical precedent for additive-only** | DDPM, Classifier-Free Guidance, GLIDE, Janner's Diffuser — all SOTA, all **additive-only** conditioning, none use FiLM's scale. Trajectory/diffusion conditioning being additive is a well-trodden, well-behaved choice, not a defect. |
| 5 | **Fewer params = cheaper + faster to train** | v1 has no `film_proj` heads. Slightly less memory, slightly faster steps, fewer things to tune. On a task where v2's extra power is unused, that's pure win for v1. |

### Where v2 genuinely pulls ahead

v2's scale `γ` earns its keep exactly when the model needs to **select** features based on the
image — i.e. *"this channel matters now, that one is irrelevant, mute it."* That shows up in:

- **Cluttered / multi-object scenes** — must suppress distractor channels.
- **Tasks with conditional feature relevance** — a channel matters only in some visual contexts.
- **Larger datasets** — enough data to actually *fit* the extra knobs as signal, not noise.

(Note: FiLM was *introduced* for visual question answering — a task that is all about
selecting features conditioned on a question. That's its home turf, and it's far harder than
push-box-to-target.)

### The honest verdict

```
Representational power:   v2  ⊇  v1            (v2 can mimic v1 exactly; can't be worse in theory)
Current aligning task:    v1  ≈  v2  (maybe v1 ≥ v2, due to data size + task simplicity)
Harder/cluttered tasks:   v2  >  v1            (gating starts to matter)
```

So: **v1 is not a broken or strictly-inferior approach** — it's the *right-sized* tool for the
current task, and it's exactly what the most successful diffusion models use. v2 is an
**upgrade in headroom**, most valuable when the task gets harder. Treat v2 as "insurance for
scaling up," not as "fixing a v1 defect."

> **Practical takeaway:** don't expect v2 to automatically beat v1 on the single-box aligning
> task. If it ties or slightly loses there, that is *expected*, not a bug — the win, if any,
> comes on harder visual scenes or with more data. Always compare them head-to-head on the
> **same** seed/data before concluding.

---

## 0. Notation (shared by both versions)

| Symbol | Meaning | Shape |
|---|---|---|
| `x` | noisy trajectory (after rearrange to channels-first) | `[B, C, H]` |
| `B` | batch | — |
| `C` | channel count of the current block (`out_ch`) | — |
| `H` | temporal horizon at this block (8 → 4 → 2 → 1 down the U-Net) | — |
| `t` | scalar diffusion/flow time → embedding | `[B, embed_dim]` |
| `v` | visual latent from `MultiImageObsEncoder` (dual-ResNet, pooled) | `[B, 128]` |
| `γ` (gamma) | per-channel **scale** from conditioning (v2 only) | `[B, C]` |
| `β` (beta) | per-channel **shift** from conditioning | `[B, C]` |

The denoiser is a 1D temporal U-Net; the conditioning question is **how `v` enters each
`ResidualTemporalBlock`**. Everything else (convs, downsample/upsample, skip connections) is
identical between v1 and v2.

---

## 1. The vision encoder (identical in BOTH versions)

`visual_unet.py :: encode_visual()`

```python
features = self.obs_encoder(obs_dict)          # (B*T_win, 128)
return features.view(B, T, -1).mean(dim=1)     # (B, 128)
```

→ math: two camera streams → ResNet → concat → mean-pool over the time window:

$$ v \;=\; \frac{1}{T_\text{win}} \sum_{\tau} \text{ResNet}_\text{dual}(\text{img}_\tau) \;\in\; \mathbb{R}^{128} $$

**This is the same `v` in v1 and v2.** Only what happens to `v` *next* differs.

---

## 2. v1 — "Fake FiLM" (additive bias via time-embedding concat)

### 2.1 Code path

**Projection + concat** — `unet1d_temporal_cond.py :: UNet1DTemporalCondModel.forward()`:

```python
cond_emb = self.cond_mlp(cond_pooled)   # (B, 128) → (B, dim)
t = torch.cat([t, cond_emb], dim=-1)    # (B, dim) ‖ (B, dim) → (B, 2·dim)
```

so the embedding fed to every block is the **concatenation** of time and visual:
`embed_dim = dim + dim = 2·dim`.

**Per-block use** — `ResidualTemporalBlock.forward()`:

```python
out = self.blocks[0](x) + self.time_mlp(t)   # time_mlp: Linear(2·dim → C), broadcast over H
out = self.blocks[1](out)
return out + self.residual_conv(x)
```

`self.time_mlp` is `Mish → Linear(embed_dim, C) → reshape to [B, C, 1]`.

### 2.2 Math

Let `h = Conv₁(x) ∈ ℝ^{B×C×H}`. The block computes:

$$ \text{out} \;=\; \text{Conv}_2\!\Big(\, h \;+\; \underbrace{W\,[\,e_t \,\Vert\, \phi(v)\,] + b}_{\text{broadcast over } H}\,\Big) \;+\; \text{Res}(x) $$

where `[e_t ‖ φ(v)]` is the concatenated time+visual embedding, `φ = cond_mlp`, and `W,b` are
`time_mlp`'s linear layer. Pull the linear apart over the concat:

$$ W\,[e_t \Vert \phi(v)] + b \;=\; \underbrace{W_t\, e_t}_{\text{time bias}} \;+\; \underbrace{W_v\, \phi(v)}_{\text{visual bias}} \;+\; b $$

So per channel `c`, the visual signal contributes **only an additive term** that is constant
across the horizon:

$$ \boxed{\;\text{out}_c \;=\; \text{Conv}_2\big(h_c + \tau_c + \beta_c(v)\big) + \text{Res}_c\;,\qquad \beta_c(v) = (W_v\,\phi(v))_c\;} $$

- `τ_c` = time bias, `β_c(v)` = visual **shift** only.
- **There is no multiplicative term.** The implicit scale is fixed at 1.
- Time and visual are **entangled** — they pass through the *same* `time_mlp` weight `W`.

> [!NOTE]
> **Why "Fake FiLM":** FiLM (Perez 2018) is `γ·h + β`. v1 has only `β` (shift), no `γ` (scale).
> It is the standard DDPM/CFG/Janner "embed → concat with time → additive bias" conditioning,
> *not* FiLM. The name in old code comments was misleading.

### 2.3 What v1 can / cannot do
- ✅ **Can** translate (shift) feature channels based on the image.
- ❌ **Cannot** suppress a channel (`γ→0`), amplify it (`γ>1`), or flip it (`γ<0`).
- ❌ Cannot route visual independently of time (shared `W`).

---

## 3. v2 — "True FiLM" (per-block scale + shift, routed separately)

### 3.1 Code path

**Projection (kept), but NOT concatenated** — `unet1d_temporal_film.py :: UNet1DTemporalFiLMModel`:

```python
embed_dim = dim                      # TIME-ONLY (no widening by cond)
cond_emb  = self._project_cond(cond) # cond_mlp(v): (B,128) → (B, dim), delivered separately
...
x = resnet(x, t, cond=cond_emb)      # cond passed as its OWN argument to every block
```

**Per-block FiLM head** — `FiLMResidualTemporalBlock`:

```python
self.film_proj = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels*2))  # → γ‖β
nn.init.zeros_(self.film_proj[-1].weight)   # ZERO-INIT
nn.init.zeros_(self.film_proj[-1].bias)

def forward(self, x, t, cond=None):
    out = self.blocks[0](x) + self.time_mlp(t)            # time bias (Linear(dim → C))
    if self.use_film and cond is not None:
        gamma, beta = self.film_proj(cond).chunk(2, dim=-1)   # each (B, C)
        out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)   # ← True FiLM
    out = self.blocks[1](out)
    return out + self.residual_conv(x)
```

### 3.2 Math

Each block has its **own** projection `ψ_ℓ` (one per block ℓ) producing both γ and β:

$$ [\,\gamma(v)\,\Vert\,\beta(v)\,] \;=\; \psi_\ell(v) \;=\; W_\ell\,\text{Mish}(v) + b_\ell \;\in\; \mathbb{R}^{2C} $$

With `h = Conv₁(x)` and time bias `τ = W_t e_t`:

$$ \boxed{\;\text{out}_c \;=\; \text{Conv}_2\Big(\big(1 + \gamma_c(v)\big)\cdot\big(h_c + \tau_c\big) \;+\; \beta_c(v)\Big) + \text{Res}_c\;} $$

This is a per-channel **affine** modulation: scale `(1+γ)` **and** shift `β`, both functions of
the image, **independent** of the time path.

### 3.3 The zero-init trick (why training stays stable)

`film_proj`'s last Linear is zero-initialized → at step 0, `γ = β = 0` for all channels:

$$ \text{out}_c \big|_{t=0} \;=\; \text{Conv}_2\big((1+0)(h_c+\tau_c) + 0\big) + \text{Res}_c \;=\; \text{Conv}_2(h_c+\tau_c) + \text{Res}_c $$

i.e. **exactly a no-FiLM block** at initialization. The network starts identical to an
unconditioned U-Net and *grows into* the gates as gradients flow (same idea as DiT/AdaLN-Zero).

### 3.4 What v2 unlocks (vs v1)
- ✅ **Suppress** a channel: `γ_c → -1` ⇒ `(1+γ_c) → 0` ⇒ channel silenced.
- ✅ **Amplify**: `γ_c > 0` boosts a channel (e.g. "target direction" when target appears).
- ✅ **Sign-flip**: `γ_c < -1` reverses polarity.
- ✅ Visual routed **separately** from time (own `ψ_ℓ` per block), not entangled in `W_t`.

---

## 4. Side-by-side

| Aspect | v1 (Fake FiLM, default) | v2 (True FiLM, opt-in) |
|---|---|---|
| Formula (per channel) | `out_c = Conv₂(h_c + τ_c + β_c(v)) + Res_c` | `out_c = Conv₂((1+γ_c(v))(h_c+τ_c) + β_c(v)) + Res_c` |
| Scale γ | ❌ implicit 1 | ✅ learned `(1+γ)` |
| Shift β | ✅ via shared `time_mlp` | ✅ via dedicated `film_proj` |
| Visual delivery | `cat([t, φ(v)])` into `time_mlp` | separate `cond=` arg per block |
| `embed_dim` into block | `2·dim` | `dim` (time only) |
| Degrees of freedom / channel | 1 (shift) | 2 (scale + shift) |
| Identity at init | n/a | ✅ (zero-init γ,β) |
| Extra params | 0 | ~1.2M (16 `film_proj` heads) |
| Checkpoint compat | loads current | fresh train only |

**One-liner:** v1 can only *push* features (`+β`). v2 can *push **and** gate* features (`(1+γ)·h + β`).

---

## 5. Where each lives (quick code index)

| Thing | v1 | v2 |
|---|---|---|
| Block class | `ResidualTemporalBlock` (`unet1d_temporal_cond.py`) | `FiLMResidualTemporalBlock` (`unet1d_temporal_film.py`) |
| Model class | `UNet1DTemporalCondModel` | `UNet1DTemporalFiLMModel` |
| Concat line | `t = torch.cat([t, cond_emb], dim=-1)` | *(removed)* — `cond_emb` passed to blocks |
| Modulation line | `out = blocks[0](x) + time_mlp(t)` | `out = out * (1+γ) + β` |
| Selector | `visual_unet.py`: `getattr(config,'film_mode','v1')` → `'v1'` branch | → `'v2'` branch |

> Reminder: `v1` is the default and untouched. `v2` only constructs when `film_mode: 'v2'`.

---

## 6. How the code GOES from v1 → v2 (the actual diff, step by step)

This section is the "I still don't see how math becomes code" part. We make **three** concrete
edits to turn v1 into v2. Nothing else changes.

### Edit ① — STOP concatenating the visual note into time

**v1** (`UNet1DTemporalCondModel.forward`): the note `φ(v)` is glued onto the time vector, so the
block receives a `2·dim`-wide vector that mixes time + visual:

```python
cond_emb = self.cond_mlp(cond_pooled)     # φ(v):  (B,128) → (B,dim)
t = torch.cat([t, cond_emb], dim=-1)      # [e_t ‖ φ(v)]: (B, 2·dim)   ← glue
```

**v2** (`UNet1DTemporalFiLMModel.forward`): the note is computed the same way, but **not glued**.
It is carried as its own variable and handed to each block separately:

```python
cond_emb = self._project_cond(cond)       # φ(v):  (B,128) → (B,dim)   (same projection)
# NO torch.cat — t stays time-only (B, dim)
...
x = resnet(x, t, cond=cond_emb)           # note delivered as its OWN argument
```

**Math meaning:** in v1, `e_t` and `φ(v)` go through *one shared* weight `W` (entangled). In v2,
time uses `W_t` and the note uses a separate `film_proj` per block (disentangled). That single
"remove the cat, pass it separately" is what lets v2 give the note its own scale knob.

### Edit ② — Each block GROWS a second head that outputs γ and β

**v1** block has one head, `time_mlp`, that maps the (wide) embedding to a per-channel bias:

```python
self.time_mlp = nn.Sequential(
    nn.Mish(),
    nn.Linear(embed_dim, out_channels),   # embed_dim = 2·dim  (time+visual mixed)
    Rearrange('batch t -> batch t 1'),    # (B, C) → (B, C, 1)  so it broadcasts over H
)
```

**v2** block keeps `time_mlp` (now `embed_dim = dim`, time-only) **and adds** a second head
`film_proj` that maps the note to `2·C` numbers — the first `C` are γ, the second `C` are β:

```python
self.time_mlp = nn.Sequential(            # unchanged role: time → per-channel bias τ
    nn.Mish(), nn.Linear(embed_dim, out_channels), Rearrange('batch t -> batch t 1'),
)
self.film_proj = nn.Sequential(           # NEW: note → (γ ‖ β)
    nn.Mish(),
    nn.Linear(cond_dim, out_channels * 2),    # ψ_ℓ : (B,dim) → (B, 2C)
)
nn.init.zeros_(self.film_proj[-1].weight)     # zero-init  → γ=β=0 at start (identity)
nn.init.zeros_(self.film_proj[-1].bias)
```

**Math meaning:** `film_proj` *is* the symbol `ψ_ℓ` from §3.2. `Linear(cond_dim, 2C)` is the
matrix `W_ℓ` (+ bias `b_ℓ`); the output `[γ ‖ β] ∈ ℝ^{2C}` is split into the two halves.

### Edit ③ — Replace "add a bias" with "scale then add"

This is the heart of it — the literal line where `+` becomes `× then +`.

**v1** block `forward`:

```python
out = self.blocks[0](x) + self.time_mlp(t)   # h + τ      ← ONLY a shift
out = self.blocks[1](out)
return out + self.residual_conv(x)
```

**v2** block `forward`:

```python
out = self.blocks[0](x) + self.time_mlp(t)               # h + τ   (same so far)
if self.use_film and cond is not None:
    gamma, beta = self.film_proj(cond).chunk(2, dim=-1)  # split (B,2C) → γ:(B,C), β:(B,C)
    out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)   # (1+γ)·(h+τ) + β
out = self.blocks[1](out)
return out + self.residual_conv(x)
```

**Line-by-line ↔ math:**

| Code | Math | Why |
|---|---|---|
| `self.blocks[0](x)` | `h = Conv₁(x)` | first conv → features, shape `(B,C,H)` |
| `self.time_mlp(t)` | `τ`, shape `(B,C,1)` | time bias, **broadcasts** over the H axis |
| `out = h + τ` | `h + τ` | the v1 part, untouched |
| `self.film_proj(cond)` | `ψ_ℓ(v) ∈ ℝ^{2C}` | note → 2C numbers |
| `.chunk(2, dim=-1)` | split into `γ`,`β` ∈ ℝ^C | first half scale, second half shift |
| `gamma.unsqueeze(-1)` | `γ`: `(B,C)`→`(B,C,1)` | add the H axis so it broadcasts like τ |
| `out * (1 + gamma…)` | `(1+γ)·(h+τ)` | **the new ×**: per-channel scale, `1+` keeps identity at γ=0 |
| `+ beta.unsqueeze(-1)` | `+ β` | per-channel shift |

That's it. **Edit ① + ② + ③ = the whole v1→v2 transformation.**

---

## 7. Reading the tensor shapes (so the broadcasting is obvious)

The thing that confuses people is *why* a `(B, C)` vector can scale a `(B, C, H)` feature map.
Answer: we add a length-1 axis and PyTorch **broadcasts** it across all `H` timesteps — i.e. the
*same* γ and β are applied to every point along the trajectory (per channel).

```
feature map  h+τ        :  (B, C, H)        e.g. (64, 128, 8)
γ  after unsqueeze(-1)  :  (B, C, 1)        e.g. (64, 128, 1)
(1 + γ) * (h+τ)         :  (B, C, H)   ← γ copied across all H columns
β  after unsqueeze(-1)  :  (B, C, 1)
... + β                 :  (B, C, H)   ← β copied across all H columns
result                  :  (B, C, H)        same shape in, same shape out
```

So a block "decides per channel" (128 decisions), and applies that decision uniformly along the
8 trajectory steps. The U-Net repeats this at every block, at every resolution
(`C = 128 → 256 → 512 → 1024` going down, then back up), which is why there are 16 `film_proj`
heads (one per `ResidualTemporalBlock`), each sized to its own `C`.

### Worked micro-example (one channel, one block)
Say after the first conv a channel has activations along the horizon
`h+τ = [0.2, 0.9, -0.1, 0.4, ...]` (length H=8). The note for this channel produces `γ = -0.8`,
`β = 0.05`. Then:

```
(1 + (-0.8)) * [0.2, 0.9, -0.1, 0.4, ...] + 0.05
= 0.2        * [0.2, 0.9, -0.1, 0.4, ...] + 0.05
= [0.09, 0.23, -0.07, 0.13, ...]     ← channel strongly DAMPENED (×0.2) then nudged (+0.05)
```

v1 could only have done `[0.2, 0.9, -0.1, 0.4, ...] + 0.05` — it can shift the whole track up by
0.05 but it **cannot** shrink it toward zero. That shrink (`×0.2`) is precisely the new power.

---

## 8. Mental checklist (map any line of code to the picture)

When you read either backbone, locate these four anchors and you understand the whole thing:

1. **Where the note is born** → `encode_visual` / `_project_cond` (`cond_mlp(v)`). Same in both.
2. **Where the note meets time** → v1: `torch.cat([t, cond_emb])`. v2: *nowhere* (kept separate).
3. **Where the note becomes γ/β** → v1: it doesn't (only β, hidden inside `time_mlp`). v2: `film_proj(cond).chunk(2)`.
4. **The modulation line** → v1: `out = blocks[0](x) + time_mlp(t)`. v2: `out = out*(1+γ) + β`.

If you can point at those four lines, you can re-derive both the math and the difference from
memory. Everything else (convs, skips, up/down sampling) is shared scaffolding.

---

## 9. "So v2 is just a bigger U-Net with more parameters?" — Yes-but, and the trap

You spotted it: v2 bolts an **extra little neural net (`film_proj`) onto every block**, so v2 has
**more parameters** than v1. Correct. But there are three things to get right, and the third is
the one that matters.

### 9.1 It's not a pure addition — v2 also SHRINKS the time head

v2 doesn't only *add* params. Because the visual note is no longer concatenated into time, the
per-block `time_mlp` input narrows from `2·dim` to `dim`:

| Per block | v1 | v2 |
|---|---|---|
| `time_mlp` | `Linear(2·dim → C)` | `Linear(dim → C)`  ← **smaller** |
| `film_proj` | — | `Linear(dim → 2C)`  ← **new** |

So the net change = (new `film_proj`) − (shrunk `time_mlp`). It's still a net increase, just not
the full size of the FiLM heads.

### 9.2 The actual number (with THIS config, `dim=32`)

The visual-aligning config uses `dim: 32`, `dim_mults: (1,2,4,8)` → block channels
`C ∈ {32,64,128,256}`. Summed over all 16 `ResidualTemporalBlocks`, `ΣC ≈ 1920`. Then:

```
film_proj added   ≈ (dim·2C + 2C) summed   ≈ 66 · 1920  ≈ 127 K
time_mlp saved    ≈ (dim·C)       summed   ≈ 32 · 1920  ≈  61 K
NET v2 − v1       ≈ 127K − 61K              ≈  +66 K parameters
```

> [!NOTE]
> The **~1.2M** figure quoted in the PLAN/CHANGELOG assumed `dim=128` (the Diffuser default).
> Params scale ≈ `dim²`, so at the real `dim=32` it's ~**16× smaller → only ~tens of thousands
> of extra params**. Either way it's tiny next to the conv backbone — *not* a GPU concern.

### 9.3 The trap: "then why not just make v1 wider and match v2?"

This is the key insight. **You cannot turn v1 into v2 just by adding parameters/width**, because
the difference is **structural, not size**:

- v1's conditioning is a **sum**: `… + β(v)`. No matter how wide you make v1, the note can only
  ever be *added*. The note never multiplies a feature. Widening v1 buys a *richer shift*, never
  a *gain knob*.
- v2's conditioning has an explicit **product**: `(1+γ(v))·h`. The feature's own gain depends on
  the image. That multiplicative coupling is a **different operation**, not a bigger version of
  the same one.

**Capacity (how many params) and inductive bias (what operations are easy) are different axes.**
v2 spends a few extra params, but what you're really buying is the *multiply* operation, not the
parameters themselves.

> [!TIP]
> **Honest caveat (don't overstate it):** because there are nonlinearities (`Mish`) *between*
> blocks, a deep enough v1 can **indirectly approximate** some gating — stacking "shift then
> nonlinearity" can fake a product in the limit. So it's not a hard mathematical impossibility
> for the whole network. The point is v2 makes gating **direct and easy to learn**; v1 would
> have to discover it the hard way through many layers and much more data. FiLM is an
> *inductive bias*, not a new theorem.

### 9.4 How this loops back to "is v1 worse?"

The extra capacity is a **double-edged sword**, which is exactly why §"Is v1 Actually Worse?"
concluded v1 can win on the current task:

- More params + the easy multiply ⇒ v2 can express feature-selection ⇒ **good on hard scenes**.
- More params + small dataset (~900 demos) + a task that doesn't need gating ⇒ v2 has spare
  capacity to **overfit** ⇒ can **lose** to the leaner v1.

So "v2 has more capacity" is **not automatically a win** — it's headroom. Headroom helps when the
task demands it (clutter, multi-object, more data) and can hurt (overfit) when it doesn't.

**One-line summary:** v2 = v1's shared scaffolding + a small per-block side-net that grants the
*multiply* operation. The cost is a handful of extra params; the real product you're buying is a
new capability (gating), which is worth it only when the task actually needs to gate.

---

## 10. v1 already adds an MLP too — the comparison vs the NON-visual model

Worth being precise: **v1 is not "the plain U-Net + nothing."** Relative to the non-visual
(state-only) backbone, **v1 already grows extra parameters** — so the jump "non-visual → v1" is
itself an upgrade, and "v1 → v2" is a second, smaller one. Two pieces are added in v1:

### 10.1 The `cond_mlp` (a real extra MLP)

Built only on the visual path (`use_cond_projection=True and cond_dim>0`):

```python
self.cond_mlp = nn.Sequential(
    nn.Linear(cond_dim, dim),   # 128 → dim
    nn.Mish(),
    nn.Linear(dim, dim),        # dim → dim
)
```

On the **non-visual** path this is `None` — the state is passed in as a **dict** (for
inpainting/conditioning), *not* a tensor, so the `cond_mlp` branch never fires. So `cond_mlp` is
purely a visual-path addition. **Note: `cond_mlp` is SHARED by v1 and v2** — both project the
128-D note to `dim` the same way; only what happens *after* differs (concat vs FiLM heads).

### 10.2 Every block's `time_mlp` gets wider

Because v1 concatenates the note into the time vector, `embed_dim = dim + dim = 2·dim`, so each
of the 16 blocks' `time_mlp` is `Linear(2·dim → C)` instead of the non-visual `Linear(dim → C)`.

### 10.3 The time-head width ladder (all three side by side)

| Model | `cond_mlp`? | per-block `time_mlp` input width | how the note acts |
|---|---|---|---|
| non-visual (state-only) | ❌ `None` (state is a dict, not a tensor) | `dim` | n/a (no visual note) |
| **v1** visual (fake FiLM) | ✅ `Linear(128→dim→dim)` | `2·dim`  (note concatenated into time) | **add** `+β(v)` |
| **v2** visual (true FiLM) | ✅ same `cond_mlp` | `dim`  (note routed elsewhere) | **multiply + add** `(1+γ)·h + β` |

> [!NOTE]
> **The neat part:** v2 actually returns each block's time head to the **non-visual width** (`dim`),
> and moves the visual capacity out of the (concatenated) time path and into the dedicated
> `film_proj` heads. So "v1 → v2" is less "add a huge net" and more "**relocate** where the note
> enters": out of the time embedding, into a per-block scale+shift head. `cond_mlp` is unchanged
> across v1/v2; only the delivery mechanism moves.
