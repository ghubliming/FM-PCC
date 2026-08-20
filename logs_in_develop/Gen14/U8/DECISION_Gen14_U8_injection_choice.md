# DECISION — Gen14 U8: which visual→transformer injection to implement

> **Date**: 2026-08-20 · **Type**: decision record · **Status**: ⏸ awaiting review, **no code written**
> **Decides**: §13 open decision #2 of [`PLAN_Gen14_U8_visual_dit_bone.md`](./PLAN_Gen14_U8_visual_dit_bone.md)
> **New evidence since the plan**: `aux_repo/visual_transformer_refs_(Claude_pulled)/` — `diffusion_policy` and `act`, pulled 2026-08-20.

---

## 1. Verdict

> ## ✅ **Implement Option 2 — the visual token — as the single injection, on all four transformer backbones.**
> ## ❌ Do not implement Option 1 (adaLN) in Phase A. ❌ Do not implement Option 3 (spatial tokens) in Phase A.

One mechanism, one new path key, four bones, one question answered cleanly.

| Aspect | Phase A (U8) |
|:--|:--|
| Injection | **Option 2 — visual token** |
| Backbones | `mf_dit`, `dit` (mf arm) · `sit`, `dit` (af arm) |
| Encoder | untouched — the existing 128-D output |
| New path key | bone only (**not** bone × injection) |
| Not built | adaLN (§4.1), spatial tokens (§4.2), cross-attention decoder (§4.3) |

---

## 2. The decisive evidence: what the flagship actually does

`diffusion_policy` was pulled because it is not merely a good reference — **it is the upstream of
code already running in FM-PCC.** `d3il/agents/models/vision/multi_image_obs_encoder.py` *is*
`diffusion_policy/model/vision/multi_image_obs_encoder.py`; verified by diff, the only differences
are import paths and whitespace, comments verbatim. The vision encoder every Gen6–Gen14 visual
model uses was written by these authors. So their answer to *"what do you connect that encoder to?"*
is not analogy — it is the source.

Their answer, in their own two files:

| Their backbone | How the obs latent enters | Our equivalent |
|:--|:--|:--|
| `TransformerForDiffusion` | **as tokens.** `cond_obs_emb = nn.Linear(cond_dim, n_emb)` (`:53`), one token per obs step (`T_cond += n_obs_steps`, `:40`), encoded to `memory`, trajectory tokens **cross-attend** to it (`nn.TransformerDecoder`, `:332-336`) | Option 2 |
| `ConditionalUnet1D` | **as FiLM.** *"FiLM modulation https://arxiv.org/abs/1709.07871 — predicts per-channel scale and bias"* (`:29-30`), `out = scale * out + bias` (`:61`) | our `film_mode=v2` |

**adaLN appears nowhere in their transformer.** The modulation design point is where they put their
*U-Net*, and it is where our U-Net already is.

🔴 **This is the argument that settles it.** Building adaLN-on-a-DiT would produce a second model at
the **same conditioning design point** as the FiLM-U-Net we already have — a different trunk
answering the same conditioning question. Option 2 is the only choice that puts a genuinely
different mechanism on the board, which is the entire reason to add a bone.

### 2.1 At `window_size=1`, Option 2 **is** their design — not an approximation

Diffusion Policy conditions on `T_cond = 1 (time) + n_obs_steps` tokens. Our visual window is
**structurally locked to 1** — `window_size=1` is not a tuning choice but a dataset-level lock
(`Gen6..._dpcc/0/plan_2/NEXTSTEP&VERDICT_WINDOW_SIZE_1_SINGLE_FRAME.md`: *"the training data never
provides multi-frame image windows at all"*), the eval default is `getattr(args, 'window_size', 1)`
(`eval_mix_visual_aligning.py:2830`), and `encode_visual`'s `mean(dim=1)` over a `T=1` window is a
no-op.

So `n_obs_steps = 1` ⇒ **one visual token.** Option 2 is numerically their conditioning stack at our
settings, not a cheap stand-in for it.

The one remaining difference — prefix token in a self-attention trunk vs a cross-attention decoder —
is close to vacuous at 1 cond token and 8 trajectory tokens: an encoder-only trunk with a prepended
token spans the same function class, and additionally lets the visual token attend back to the
trajectory. See §4.3 for why we do not build the decoder.

---

## 3. Why Option 2, restated as four independent reasons

Each of these would be sufficient on its own; together they are decisive.

| # | Reason | Weight |
|:--:|:--|:--|
| 1 | **The flagship, on our exact encoder and problem shape, uses tokens.** adaLN is absent from their transformer and present (as FiLM) only in their U-Net (§2). | decisive |
| 2 | **It is the only mechanism that works on all four bones.** `dit`/`af_dit` have *no adaLN path* — their blocks are `forward(x, cos, sin)` (`mf_dit_trajectory.py:201`), conditioning is prefix-tokens-only. Picking adaLN means shipping two injections or dropping two bones. | decisive |
| 3 | **One mechanism keeps the experiment answerable.** Per PLAN §1.3, the honest question is *"does attention beat convolution at matched capacity, in the visual setting"* — after a DiT-vs-U-Net result was already retracted once. A bone × injection matrix confounds exactly that question, and doubles the checkpoint tree. | strong |
| 4 | **Strictly more expressive than adaLN for the same 128-D input.** Attention weights are per-head and per-layer, so depth-dependent visual weighting is representable; adaLN's `c` is a single summed vector. Same input, more capacity to use it. | supporting |

**Cost**: one `nn.Linear(128, d)`, one learned token parameter, one entry in a `cat`, and a
`+1` in two bookkeeping constants. The only real risk is the prefix off-by-one, and it is
mechanically testable — gate **G-B6** in PLAN §10 exists for it.

---

## 4. What is NOT being built, and why

### 4.1 ❌ Option 1 — adaLN

| Against | Detail |
|:--|:--|
| Duplicates an occupied design point | We already own a per-block scale/shift conditioner: `film_mode=v2`. adaLN-on-DiT re-asks the same conditioning question with a different trunk, instead of asking a new one. |
| Architecturally impossible on half the bones | `dit`/`af_dit` have no `c` pathway at all. |
| Doubles the path-key matrix | bone × injection ⇒ four checkpoint trees per arm instead of two, for a secondary result. |
| Not what the reference does | §2. |

**Not a permanent no.** If the token result is negative and we need to know whether the *mechanism*
was at fault, adaLN on `mf_dit`/`sit` is ~30 lines **on top of the wrapper Phase A ships** — a
cheap follow-up, correctly sequenced after we have a number rather than before.

### 4.2 ❌ Option 3 — spatial visual tokens

Two independent reasons, one of which is new since the plan:

1. 🔴 **`diffusion_policy` proves spatial tokens are not *required*.** Their transformer conditions
   on the same pooled global latent we have and is competitive with their U-Net across their whole
   benchmark. So "the DiT can only win with spatial grounding" — an argument PLAN §4 entertained —
   is **falsified by the reference**. Global tokens are enough to be viable, which removes the
   urgency and lets Option 3 be judged on its own merits later.
2. **`act` shows doing it properly is a different model, not a knob.** Its backbone keeps `layer4`
   via `IntermediateLayerGetter(return_layers={'layer4': "0"})` (`backbone.py:70-71`), flattens to
   tokens (`transformer.py:54`), and needs a dedicated **2-D positional encoding**
   (`position_encoding.py`) plus query/cross-attention. That is the piece a naive spatial-token
   implementation forgets, and it is why this belongs in its own unit.

Plus the standing concerns from PLAN §4: it discards D3IL's `SpatialSoftmax` keypoint prior, and
18 visual vs 8 trajectory tokens makes the trajectory a minority of its own sequence at 96 demos.

### 4.3 ❌ Full cross-attention decoder (maximum DP fidelity)

Tempting — it is literally what DP does — and rejected for Phase A:

- All four of our backbones are **encoder-only trunks with twin `u`/`v` FinalLayers**. A decoder
  changes the trunk topology, the `(u, v)` contract, and the surface the JVP differentiates.
- At `T_cond = 1`, the decoder buys essentially nothing over a prefix token (§2.1).
- It would make the bone no longer a port of `MeanFlow/models/dit.py` or α-Flow's SiT, forfeiting
  the provenance that makes the mf/af arms defensible as faithful implementations.

Revisit only if Option 3 lands and `T_cond` becomes 18.

---

## 5. Implementation spec (what I will write on approval)

Scope is unchanged from PLAN §6; this pins the injection-specific details.

### 5.1 The token, on `dit` / `af_dit` (native)

```python
# __init__, guarded on cond_dim > 0
self.vis_projector = nn.Linear(cond_dim, hidden_size)      # 128 → d
self.vis_tokens    = nn.Parameter(tok(torch.empty(1, hidden_size)))   # same init as the other tokens

# _build_sequence(), one more entry in the existing cat
vis_tok = self.vis_tokens[None] + self.vis_projector(visual_latent)[:, None]
return torch.cat([class_tok, omega_tok, tmin_tok, tmax_tok, time_tok, vis_tok, x_embed], dim=1)
```

🔴 **Both constants move together**, before the RoPE precompute at `mf_dit_trajectory.py:299-303`:
`self.prefix_tokens` 7 → 8 (it strips the prefix at `:381`) and `total_tokens` 15 → 16. The RoPE
buffers are `persistent=False`, so resizing them cannot corrupt checkpoint loading — but a
half-applied bump yields a model that trains fine and reads the wrong positions. **G-B6.**

The learned-token-plus-projection shape (`self.vis_tokens[None] + proj(latent)[:, None]`) is
deliberately identical to how this backbone already builds `class_tok`, `omega_tok`, `time_tok`.
Vision enters the way everything else enters.

### 5.2 The token, on `mf_dit` / `sit` (needs a position)

These have a learned **absolute** `pos_embed` of shape `[1, num_patches, d]`, sin-cos initialised,
and it **is** in the state_dict. So:

- extend to `[1, 1 + num_patches, d]` and initialise from `get_1d_sincos_pos_embed(d, 1 + num_patches)`;
- prepend the visual token, run the trunk, then **strip position 0** before the two FinalLayers so
  the output stays `[B, 8, 9]`.

This mirrors DP giving its cond tokens their own `cond_pos_emb` (`transformer_for_diffusion.py:59`),
in the simplest form that keeps one table.

Everything else — the `VisualDiTTwoTime` wrapper, routing, train/eval plumbing, the bone path key,
`dit_hidden_size=160` parameter matching, the gates — is exactly PLAN §6–§10, unchanged.

### 5.3 Invariant that must hold

`cond_dim=0` ⇒ **byte-identical state_dict to today** on all four backbones, so Gen3v4/v6/v7 (which
import the same architectures) are provably untouched. Gate **G-B1**.

---

## 6. What would change this decision

Stated in advance so the record is falsifiable rather than retrofitted:

| If this turns out to be true | Then |
|:--|:--|
| The prefix bump cannot be made to pass G-B6 on `mf_dit`/`sit` without reshaping `pos_embed` in a way that breaks their port fidelity | fall back to Option 1 (adaLN) **on those two only**, and keep tokens on `dit`/`af_dit` — accepting the mixed matrix as the lesser evil |
| Token-conditioned DiT trains but is image-blind (G-B3 fails: zero gradient into `vis_projector`) | that is a wiring bug, not a design refutation — fix and re-gate, do not switch mechanism |
| Phase A lands and the DiT loses to the U-Net at matched params | **expected** per PLAN §1.3/§5.1. Publish it. Then adaLN (§4.1) becomes the cheap next question, and Option 3 the expensive one |
| `window_size` ever unlocks from 1 | Option 2 generalises for free — emit one token per window step, which is precisely DP's `T_cond += n_obs_steps` |

---

## 7. Provenance

- `aux_repo/visual_transformer_refs_(Claude_pulled)/README.md` — what was pulled, why, per-file pointers.
- Encoder ancestry verified by direct diff (import paths + whitespace only).
- All line numbers read on 2026-08-20 against the working tree.
- **No code has been modified.** Awaiting review of this document and PLAN §13's remaining open decisions (#1 bone key naming, #3 DiT size, #4 objective constants untouched).
