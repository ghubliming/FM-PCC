# The DiT Backbone Explained (for newcomers) — with the Math, and How It Works in FM-PCC

**Who this is for:** you are new to transformers / DiTs and want to understand, from the ground up,
**(a)** what the DiT we added actually computes, with the real math, and **(b)** how it plugs into the
iMF / FM-PCC trajectory pipeline. Focus is on the **DiT itself**.
**Code:** `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py`.
**Companion math (the objective side):** [iMFPCC_Full_Math_Reference](../U5/iMFPCC_Full_Math_Reference.md).
**Why DiT vs UNet:** [UNet_vs_DiT_for_iMF_Principle](../U5/UNet_vs_DiT_for_iMF_Principle.md).

---

## 0. The one-paragraph picture

**DiT = "Diffusion Transformer".** Instead of a convolutional U-Net, the network that predicts the
velocity field is a **Transformer**. A Transformer turns its input into a list of **tokens** (vectors),
lets every token **look at every other token** via **attention**, and outputs a refined list of tokens.
For us, the input is a trajectory of `H` timesteps, each a `D`-dim vector — so we make **one token per
timestep**. The conditioning information (interval size `h`, the CFG knobs `ω, τ_min, τ_max`) becomes
**extra tokens glued to the front**. After the Transformer processes everything, we read out the
trajectory tokens and project them back to `[H, D]` to get the predicted velocity. That's the whole idea.

```
trajectory [B,H,D] ─► tokens ─┐
conditioning (h,ω,…) ─► tokens ─┼─► [ Transformer blocks ] ─► read trajectory tokens ─► velocity [B,H,D]
                                ┘     (attention + MLP)
```

---

## 1. Why tokens and attention at all?

A **token** is just a vector of length `hidden_size` (we use 256). A Transformer maintains a *set* of
these vectors and repeatedly updates each one using information from all the others.

The core operation, **attention**, answers: *"for this token, which other tokens are relevant, and what
should I copy from them?"* Mathematically, each token produces a **query** `q`, a **key** `k`, and a
**value** `v`. Token `i` attends to token `j` with a weight that grows when `q_i` and `k_j` point in
similar directions; then token `i` collects a weighted average of the `v_j`. This lets timestep 2 of the
trajectory directly read timestep 7, or read the "interval size" conditioning token — **in a single
step**, with no notion of distance. (A conv U-Net, by contrast, only mixes *neighbours* and needs many
layers to connect far-apart steps.)

---

## 2. From a trajectory to tokens (patch embedding)

Our input is `x ∈ ℝ^{B×H×D}` (batch `B`, horizon `H=8`, transition dim `D=`obs+action). We turn each
timestep into one token of width `hidden=256` with a single linear map (`TrajPatchEmbedder`):

```
token_p  =  W_patch · x[:, p, :]  +  b_patch        for p = 0 … H−1
W_patch ∈ ℝ^{hidden × (patch·D)},   patch_size = 1
```

With `patch_size=1`, that's literally "lift each `D`-vector to a 256-vector." (If `patch_size=2`, we'd
group 2 consecutive steps into one token of `2·D` features — fewer, fatter tokens. It must divide `H`.)
Result: **`H` content tokens**, each in ℝ²⁵⁶.

---

## 3. Conditioning as tokens (the heart of the DiT's advantage)

The network must also *know* the interval size `h`, and (for guidance) `ω, τ_min, τ_max`. The DiT injects
these as **their own tokens**, prepended to the content tokens. This is called **in-context
conditioning**.

**Step 3a — embed each scalar.** A scalar `s` (say `h`) becomes a vector via a **sinusoidal embedding**
then a small MLP (`TimestepEmbedder`):

```
sinemb(s)_k = [ cos(s·f_0), …, cos(s·f_{m−1}), sin(s·f_0), …, sin(s·f_{m−1}) ]
   with geometric frequencies  f_k = 1 / 10000^{k/m}
e_s = MLP(sinemb(s)) ∈ ℝ^{hidden}        (MLP = Linear → SiLU → Linear)
```

The sinusoids give the network an easy, smooth code for "how big is this number" across many scales (the
same trick used for time in diffusion models).

**Step 3b — add to learnable tokens.** We keep a few trainable "slot" tokens per signal and add the
embedding into them:

```
time_tokens  ←  Time  + e_h        (interval size h ; +e_t too if dit_condition_on_t)
omega_tokens ←  Omega + e_ω         (ω passed through 1 − 1/ω, the official recipe)
t_min_tokens ←  Tmin  + e_{τmin}
t_max_tokens ←  Tmax  + e_{τmax}
class_token  ←  Class + e_y         (see §3c — used for CFG on/off)
```

**Step 3c — the class token = the CFG switch.** In the original image DiT, `y` is the ImageNet class.
We have no class label (our real conditioning — the start observation — is pinned into `x` outside the
network). So we keep **one** class slot and use it purely as the **classifier-free-guidance (CFG)
switch**: normally feed label `0`; when we want the **unconditional** prediction (`force_dropout=True`),
feed the **null label** `num_classes`. Two different learned vectors ⇒ the network can produce a
"conditioned" vs "unconditioned" output, which is exactly what guidance needs (§7.3).

**Step 3d — concatenate.** The full token sequence fed to the Transformer is:

```
seq = [ class | omega | t_min | t_max | time | x_0 … x_{H−1} ]
       └──────────── prefix (7 tokens) ───────────┘ └─ content (H=8) ─┘     total S = 15
```

> **Why this matters (vs the UNet):** the UNet **sums** all conditioning into one bias vector added
> identically to every position — a weak, fixed nudge. The DiT gives each signal its **own token** that
> **every trajectory step can attend to differently**. The iMF objective needs the output to genuinely
> *depend* on `h` and the interval; tokens provide that capacity, an additive bias does not.

---

## 4. The Transformer block, component by component (with math)

Each block updates the sequence `seq ∈ ℝ^{B×S×hidden}` (here `S=15`). It has two sub-layers —
**attention** and an **MLP** — each wrapped in a normalize → transform → gated-residual pattern.

### 4.1 RMSNorm (normalization)

Before each sub-layer we normalize every token vector to unit scale (per token, independently):

```
RMS(z) = sqrt( (1/d) Σ_i z_i²  +  ε )
RMSNorm(z) = (z / RMS(z)) ⊙ γ            γ ∈ ℝ^{hidden} learned
```

It rescales each token to a stable magnitude (no mean-subtraction, cheaper than LayerNorm). Because it
acts **within one token** (no mixing across the batch), it is also safe for the derivative trick we need
later (§7.2).

### 4.2 RoPE self-attention (the mixing step)

**Project** each (normalized) token into queries, keys, values, then split into `num_heads=4` heads of
size `head_dim = 256/4 = 64`:

```
q = W_q·z,  k = W_k·z,  v = W_v·z          each ℝ^{S×hidden} → reshaped to ℝ^{S×heads×64}
```

**QK-RMSNorm:** normalize `q` and `k` per head (stabilizes the dot products).

**RoPE = Rotary Position Embedding (how the model knows token order).** Attention by itself is
order-blind. RoPE injects position by **rotating** each query/key by an angle proportional to its
position `p`. Pairing up coordinates `(q_{2i}, q_{2i+1})` and rotating by angle `p·θ_i` (with
`θ_i = 1/10000^{2i/64}`):

```
q'_{2i}   = q_{2i}·cos(pθ_i) − q_{2i+1}·sin(pθ_i)
q'_{2i+1} = q_{2i}·sin(pθ_i) + q_{2i+1}·cos(pθ_i)        (same for k)
```

The magic: the dot product of a rotated query at position `p` and a rotated key at position `p'` depends
only on the **relative** offset `p − p'`. So attention naturally learns "how far apart" two timesteps
are. *(Our code computes this with real `cos/sin` tables — mathematically identical to the official
complex-number version, but written so the §7 derivative works.)*

**Attention scores and output** (per head):

```
score_{ij} = (q'_i · k'_j) / sqrt(64)                 similarity of token i's query to token j's key
α_{ij}     = softmax_j(score_{ij}) = e^{score_{ij}} / Σ_l e^{score_{il}}    (weights sum to 1)
out_i      = Σ_j α_{ij} · v_j                          token i collects a weighted blend of values
```

Concatenate heads, apply an output projection `W_o`. **This is the only step where tokens exchange
information** — including content↔conditioning. Everything else is per-token.

### 4.3 SwiGLU MLP (per-token nonlinearity)

After attention, each token passes through a gated MLP (no mixing across tokens):

```
SwiGLU(z) = ( SiLU(z·W1) ⊙ (z·W3) ) · W2          SiLU(a) = a·σ(a),  ⊙ = elementwise
```

The `SiLU(zW1)` branch acts as a **gate** modulating the `zW3` branch — a more expressive nonlinearity
than a plain ReLU-MLP. This is where each token "thinks" about what it gathered.

### 4.4 Zero-initialized gated residuals (why training is stable)

Both sub-layers are added back as **residuals**, each scaled by a learnable per-channel gate that
**starts at zero**:

```
z ← z + g_attn ⊙ Attention(RMSNorm(z))          g_attn, g_mlp ∈ ℝ^{hidden}, initialized to 0
z ← z + g_mlp  ⊙ SwiGLU(RMSNorm(z))
```

At initialization `g=0`, so each block is the **identity** (`z ← z`) and the whole deep network starts as
a clean pass-through. Training then *gently* opens the gates. This is what lets a deep Transformer train
stably under the demanding iMF target (the bootstrapped JVP loss of §7).

---

## 5. Shared backbone → two heads (u and v)

iMF predicts **two** velocity fields (see the objective doc): the **average** velocity `u` (the one we
deploy) and the **instantaneous** velocity `v` (a training-only helper). The DiT computes both from a
**shared trunk**, then splits into two short stacks of blocks:

```
seq ──► shared_blocks (depth − aux_head_depth = 6 blocks)  ──► common features
              ├──► u_heads (2 blocks) ──► u_final_layer ──► u
              └──► v_heads (2 blocks) ──► v_final_layer ──► v   (only when return_v=True)
```

`FinalLayer` = `RMSNorm → Linear(hidden → patch·D)`, with the linear **zero-initialized** (so the model
outputs ≈0 velocity at step 0 of training — a safe start). We then take only the **content** tokens
(drop the 7 prefix tokens) and **un-patchify** back to `[B, H, D]`:

```
u_tokens = u_seq[:, prefix:]              # [B, H, patch·D]
u        = reshape(u_tokens) → [B, H, D]  # un-patchify
```

At **inference** we set `return_v=False`, so the `v_heads` are skipped entirely — only `u` is computed.

---

## 6. End-to-end forward pass (concrete shapes, H=8)

```
x:[B,8,D] ─patch─► content:[B,8,256]
h,ω,τmin,τmax,y ─embed+slot tokens─► prefix:[B,7,256]
concat ─► seq:[B,15,256]
 │
 ├─ shared_blocks ×6:   seq ← seq + g·Attn(RMSNorm(seq));  seq ← seq + g·SwiGLU(RMSNorm(seq))
 │
 ├─ u_heads ×2 ─► u_final ─► u_tokens[B,15,patch·D] ─► drop prefix ─► [B,8,patch·D] ─► u:[B,8,D]
 └─ v_heads ×2 ─► v_final ─► … ─► v:[B,8,D]     (training only)
returns:  u            (inference)
          (u, v)       (training)
```

Total parameters scale with `depth · hidden²`; we deliberately keep `hidden=256, depth=8` **small**,
because the trajectory is only 8 steps long (a huge image-scale DiT would be wasteful — see the principle
doc).

---

## 7. How the DiT works **inside FM-PCC / iMF**

The DiT is just the **`velocity_net`** — a swappable predictor. Everything around it (the training
objective, the sampler, the constraint projector) is unchanged from the UNet. Here is the chain.

### 7.1 The contract

`iMFTrajectoryModel` calls the backbone with a fixed signature and gets back `(u, v)`:

```
velocity_net(x, cond, t, h=h, force_dropout=…, omega=…, t_min=…, t_max=…, return_v=True) → (u, v)
```

The DiT matches this exactly, so the UNet and DiT are interchangeable (`imf_backbone: 'unet'|'dit'`).

### 7.2 The MeanFlow training target — and why it differentiates the DiT

iMF's target is the **MeanFlow Identity** (full derivation in the U5 math doc):

```
u_target = v_inst + h · du/dr           du/dr = ∂u/∂z·v_inst + ∂u/∂r − ∂u/∂h
```

The term `du/dr` is a **directional derivative of the network's own output** — i.e. *how does the DiT's
predicted `u` change as we move along the trajectory?* We compute it with **forward-mode autodiff**
(`torch.func.jvp`): feed the DiT the point `(z_r, r, h)` together with a *direction* `(v_inst, +1, −1)`,
and it returns both `u` and the derivative `du/dr` in one pass:

```
u_pred, du_dr = jvp( u_of, primals=(z_r, r, h), tangents=(v_inst, +1, −1) )
u_target      = (v_inst + h · du_dr).detach()        # stop-gradient: it's a target, not a path to grad
loss          = weighted ‖u_pred − u_target‖²
```

**This is the reason the DiT had to be built derivative-safe.** Forward-mode AD must flow through every
op: that's why we use **RMSNorm** (per-token, no batch coupling), a **real-valued RoPE** (no
non-differentiable complex bitcast), and hold the CFG knobs **constant** through the JVP (they carry no
tangent). If any of those were AD-hostile, the iMF loss would crash. (The UNet is safe for the same
reasons — InstanceNorm is per-instance.)

### 7.3 Interval-CFG through the DiT (using the class-token switch)

At inference we sharpen the prediction with **classifier-free guidance**, applied only inside the
guidance interval `τ ∈ [τ_min, τ_max]`:

```
u_cond   = DiT(x, …, force_dropout=False)     # class token = label 0   (conditioned)
u_uncond = DiT(x, …, force_dropout=True)       # class token = null label (unconditioned)
u_guided = u_uncond + ω · (u_cond − u_uncond)
```

The only difference between the two calls is **which class token** is used (§3c) — that's how the DiT
realizes the cond/uncond pair. Outside `[τ_min, τ_max]`, we skip guidance and just use `u_cond`.

### 7.4 Sampling: turning the DiT into a trajectory

Generation is a short forward-Euler integration from noise (`τ=0`) to data (`τ=1`), using `u` as the
step direction (full loop in the U5 math doc §9):

```
x ← noise;   dt = 1/N                          # N = flow_steps_v3 (e.g. 2)
for i in 0…N−1:
    τ = i/N
    u = (interval-CFG of §7.3 using the DiT at this step)
    x ← x + u · dt                              # Euler step toward data
    x ← apply_conditioning(x)                   # re-pin the start observation
    x ← DPCC.project(x)  if near the end        # snap to obstacle-avoidance constraints
```

Because `u` is an **average** velocity over the step, iMF can take **very few steps** (`N=1–2`) where
plain FM would need ~10 — that is the entire point of using iMF, and the DiT is just a (hopefully better)
engine for predicting that `u`.

### 7.5 DPCC (the constraint layer) is backbone-agnostic

The DPCC projector snaps the trajectory onto the avoidance constraints near the end of sampling. It acts
on `x` **after** the velocity step and never inspects the network — so it works identically whether `u`
came from the UNet or the DiT. (Caveat from the plan: at very low NFE the snap *schedule* needs
re-tuning; that's orthogonal to the backbone.)

---

## 8. Why bother with the DiT here? (one paragraph)

For **images**, a DiT beats a conv U-Net mainly because attention sees the whole image at once. For our
**8-step trajectory** that particular advantage is small (the U-Net already mixes all 8 steps at its
bottleneck). The DiT's *real* edge for iMF is **conditioning capacity**: the iMF target genuinely depends
on the interval `(h, ω, τ_min, τ_max)`, and the DiT exposes each of those as a token every step can attend
to — versus the U-Net's single summed bias. Whether that edge shows up at 1–2 NFE on avoiding is exactly
the A/B the U6 work sets up. See [UNet_vs_DiT_for_iMF_Principle](../U5/UNet_vs_DiT_for_iMF_Principle.md).

---

## 9. Mini-glossary

| Term | Plain meaning |
|---|---|
| **Token** | one vector (length `hidden=256`) the Transformer carries around |
| **Patch embedding** | linear map turning each trajectory step into a token |
| **Attention** | step where each token reads a weighted blend of all others |
| **Query/Key/Value** | the three projections that define "who attends to whom and copies what" |
| **RoPE** | rotates q/k by position so attention sees *relative* distance |
| **RMSNorm** | per-token rescaling to a stable magnitude |
| **SwiGLU** | gated MLP nonlinearity applied per token |
| **Zero-init gate** | residual multiplier starting at 0 ⇒ block starts as identity ⇒ stable |
| **Shared backbone + u/v heads** | common trunk, then two short stacks predicting `u` and `v` |
| **In-context conditioning** | feeding `h, ω, τ_min, τ_max` as extra tokens (not as a bias) |
| **CFG (classifier-free guidance)** | mixing a conditioned and unconditioned prediction to sharpen output |
| **JVP** | forward-mode autodiff giving `du/dr`, the core of the iMF target |
| **NFE** | number of network evaluations (sampler steps); iMF aims for 1–2 |
| **DPCC** | the projector that snaps trajectories onto avoidance constraints |
