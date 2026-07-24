# NOTE — Backbone fidelity across Gen3v4 (iMF) / Gen3v6 (MeanFlow) / Gen3v7 (α-Flow)

**Date:** 2026-07-23 · **Type:** fidelity note · **Status:** ⚠️ **open items — needs refinement later**
Companion to [`CHANGELOG_Gen3v7_coding1.md`](CHANGELOG_Gen3v7_coding1.md)

---

## Short answer

**All three generations share ONE backbone: the iMF DiT.** Gen3v7's `af_dit_trajectory.py` is a
rename-only copy of Gen3v6's `mf_dit_trajectory.py`, which is a rename-only copy of Gen3v4's
`imf_dit_trajectory.py` — a port of the official **imeanflow `imfDiT.py`**.

```
iMF (Gen3v4) ── DiT = imfDiT port ─┐
MeanFlow (Gen3v6) ── same file ────┼── identical architecture, identical parameter count
α-Flow (Gen3v7) ── same file ──────┘
```

**No problem for the experiment.** That is exactly what the plans ask for — change **one**
thing (the target), hold the architecture fixed. Changing objective *and* backbone at once
would make the three-way comparison uninterpretable.

**But the gap is not symmetric.** Gen3v4 sits on its *own* upstream's network; Gen3v6 and
Gen3v7 do not. Both borrow iMF's.

---

## 1. Where the GAP is, per generation

| gen | objective from | backbone **should** be (its own upstream) | backbone **is** | gap |
|---|---|---|---|---|
| **Gen3v4 iMF** | `imeanflow` (JAX, official) | `imfDiT` | port of `imfDiT` | ✅ **none of substance** — see §3 for the 3 minor deviations |
| **Gen3v6 MeanFlow** | MeanFlow 2505.13447 / `aux_repo/MeanFlow` | `MFDiT` (adaLN SiT-style) | port of **imfDiT** | ⚠️ **wrong-family network** |
| **Gen3v7 α-Flow** | `alphaflow` 2510.20771 | `SiT` (adaLN) | port of **imfDiT** | ⚠️ **wrong-family network** |

So the honest claims are:

- Gen3v4 = **iMF**.
- Gen3v6 = **"the MeanFlow objective on iMF's DiT"** — *not* "MeanFlow".
- Gen3v7 = **"the α-Flow objective on iMF's DiT"** — *not* "α-Flow".

⚠️ Gen3v6's changelog calls itself a *"faithful MeanFlow baseline"*. That is true of the
**objective** (analytic-v JVP tangent, official adaptive loss, two independent logit-normals)
and **not** of the network. The Gen3v6 results MD should say so.

## 2. The four networks side by side

Evidence: `flow_matcher_v3_*/models/*_dit_trajectory.py` ·
`aux_repo/imeanflow/models/imfDiT.py` · `aux_repo/MeanFlow/models/dit.py` ·
`aux_repo/alphaflow/src/training/dit.py`

| | **ours** (all 3 gens) | **imeanflow** `imfDiT` | **MeanFlow** `MFDiT` | **α-Flow** `SiT` |
|---|---|---|---|---|
| positional | **RoPE** (real-valued) | **RoPE** (complex) | sincos `pos_embed` | sincos `pos_embed` (+temporal) |
| block norm | RMSNorm | RMSNorm | `nn.RMSNorm` | LayerNorm |
| QK norm | ✅ | ✅ | ✅ | ✅ (timm `qk_norm`) |
| MLP | **SwiGLU** | **SwiGLU** | timm `Mlp` (GELU) | timm `Mlp` (GELU) |
| conditioning | **in-context prefix tokens** | **in-context prefix tokens** | **adaLN modulation** | **adaLN modulation** |
| time inputs | **`h` only** | **`h` only** (only `h_embedder` exists) | **`t` AND `r`** (`t_embedder`+`r_embedder`) | **`t` AND `t_next`** (two embedders) |
| u/v heads | shared trunk → **separate head blocks** + 2 final layers | same | shared trunk → **2 final layers** only | **single** final layer — no v head |
| residual init | zero-init vector gates | zero-init vector gates | zero-init adaLN | zero-init adaLN |
| I/O | 1-D traj `[B,8,6]` | 2-D images | 2-D images | 5-D latent video |
| published size | depth 8, dim 256, 4 heads | — | — | B/2: 12 blocks, dim 768, 12 heads |

## 3. Gen3v4's three minor deviations from `imfDiT` (for completeness)

1. **Real-valued interleaved RoPE** instead of the complex bitcast — required so
   `torch.func.jvp` can differentiate it. Mathematically identical.
2. **1-D trajectory I/O** — `TrajPatchEmbedder` replaces the 2-D conv patch embed.
3. **`dit_condition_on_t` exists as an optional extra** (default `False` = upstream
   behaviour). `imfDiT` has no `t` embedder at all.

None affects the learned function class. Gen3v4 is a genuine port.

## 4. 🔴 The gap that actually bites: `h`-only conditioning

`_build_sequence` conditions on `h` alone; the `t`/`r` argument is **accepted and discarded**
on the DiT arm (`af_dit_trajectory.py:338`, gated by `condition_on_t`). Faithful to iMF —
but **both** MeanFlow's and α-Flow's reference nets embed *both* time endpoints.

Consequences, all three shared equally by Gen3v6 and Gen3v7 (so they do **not** bias the A/B):

- **Restricted function class.** The true average velocity `u*(z, r, h)` genuinely varies
  with `r`; an `h`-only net can only reach it through `z`, and the same `z` can occur at
  different `r`. Our models are fitting a strictly smaller family than MeanFlow's or
  α-Flow's reference models are.
- **The MeanFlow identity loses a term.** With `∂_r u ≡ 0`,
  `D_tot = ∂_z u·v + ∂_r u − ∂_h u` collapses to `∂_z u·v − ∂_h u`. Gen3v6's JVP still closes
  — one of its three terms is simply zero by architecture. Since the Gen3v6 hypothesis *is*
  about that JVP target, this is worth naming in its results MD.
- **α-Flow's bootstrap is unaffected in kind.** `u(z_shift, r+dt, h−dt)` varies only through
  `z_shift` and `h−dt`; it is an *interval* composition and the interval is what the net
  reads. But the restricted class above still applies.

⚠️ Stale comment, pre-existing: the sampler's *"NEVER freeze t for this architecture … the
learned weights encode u(x, t, h)"* is inherited from the **UNet** arm (U3-B1) and is wrong
on the DiT arm, where `t` is discarded. Not introduced by Gen3v7; flagged so nobody debugs
the wrong thing.

## 5. Other named gaps

**`dual_head=True` is not an α-Flow feature.** α-Flow's `SiT` returns one tensor
(`pred_mean_velocity, ctx = net(...)`); its `trajectory_FM` term is computed from that *same*
u output. Our v head with its own full loss is an inherited iMF/MeanFlow component riding
along, so the Gen3v7 arm carries a loss term upstream does not have. Kept on purpose —
dropping it changes the parameter count and breaks the controlled A/B.
*(MeanFlow's `MFDiT` **does** have two output layers, so this is a Gen3v7-only gap.)*

**The RoPE-for-JVP deviation is inert for α > 0.** Gen3v7's discrete branch takes no network
derivative. It still binds for the α=0 JVP branch.

## 6. ⚠️ Refine later — open items, prioritised

Nothing here blocks the current runs. These are what would have to change before any claim
of *reproducing* MeanFlow or α-Flow, roughly in decreasing value:

| # | item | why it matters | cost |
|---|---|---|---|
| **R1** | **Ablate `dit_condition_on_t=True`** on one generation | directly tests §4 — the only gap with a plausible effect on the few-NFE result, and the cheapest to settle. If large-`h` accuracy improves, `h`-only conditioning was a real handicap and every MeanFlow-family number in this repo inherits it. | one train run; the flag already exists and is watched via `ts`… **⚠️ actually it is NOT in `args_to_watch` — add it before sweeping, or two runs collide** |
| **R2** | **adaLN sibling backbone** (`MFDiT`/`SiT`-style: sincos + GELU Mlp + adaLN, both times embedded) as a 4th arm | the honest "did we reproduce the paper" control; also removes the RoPE/SwiGLU confound | new backbone file + a generation folder |
| **R3** | **Single-head α-Flow arm** (`dual_head=False`) | removes the extra v loss upstream does not have | config flip + one run; changes the state_dict, so a new folder |
| **R4** | Match published sizing (depth/dim) if anyone compares to reported FID-scale numbers | our depth 8 / dim 256 is far below B/2 — irrelevant for D3IL, relevant for any cross-paper claim | large |

**Do R1 first.** It is one run, it uses code that already exists, and it is the only item on
this list that could change the *interpretation* of the Gen13/Gen3v6/Gen3v7 few-NFE verdict
rather than just the wording of it.

## 7. Verdict

| question | answer |
|---|---|
| Is Gen3v7's DiT α-Flow's DiT? | **No** — it is iMF's. |
| Is Gen3v6's DiT MeanFlow's DiT? | **No** — it is iMF's. Same gap. |
| Is Gen3v4's DiT iMF's DiT? | **Yes** — a genuine port (§3). |
| Problem for Gen3v4 vs Gen3v6 vs Gen3v7? | **No.** Identical architecture is what makes the three-way A/B controlled, and it is what the plans specify. |
| Problem for "we reproduced MeanFlow / α-Flow"? | **Yes** — we did not, and must not claim it. We ported the **objectives**, deliberately not the networks and not the launchers (Gen3v7 PLAN §6: *"We take the objective only"*). |
| Does anything need changing now? | **No.** §6 is follow-up work, not a blocker. |
