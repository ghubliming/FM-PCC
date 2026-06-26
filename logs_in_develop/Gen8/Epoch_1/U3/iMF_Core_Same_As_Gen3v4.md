# U3 — Gen8 Visual Aligning iMF: Same iMF Core as Gen3v4 (Cross-Link)

**Date:** 2026-06-13
**Scope:** `imf_visual_aligning/` (Gen8 iMF visual aligning) vs `flow_matcher_v3_imeanflow/` (Gen3v4 iMF).
**Bottom line:** Gen8's iMF **training and inference math is byte-identical to Gen3v4**. The
"iMF body, FM brain" diagnosis and the full-unleash plan written for Gen3v4 apply here
**unchanged**. The only Gen8 difference is the **visual encoder** swapped into the backbone — it
does **not** touch the iMF objective.

---

## 1. Why this doc exists

Gen8 (iMF visual aligning) was built on top of the Gen3v4 iMF stack. The principled analysis of
*whether iMF actually beats FM* and *how to unleash the real iMF power* was written against
Gen3v4. Rather than duplicate it, this note **verifies** that the iMF core is the same and
**cross-links** to the canonical documents.

**Read these (they are the source of truth):**

- **iMF vs FM, math/principle + DPCC-avoiding verdict:**
  [`../../../Gen3v4_imf/U3/iMF_vs_FM_Math_Principle.md`](../../../Gen3v4_imf/U3/iMF_vs_FM_Math_Principle.md)
- **Full plan to unleash real iMF (JVP MeanFlow-Identity training, low NFE):**
  [`../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md`](../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md)

---

## 2. Verification — the iMF core is identical (checked 2026-06-13)

Diff of `imf_visual_aligning/models/` against `flow_matcher_v3_imeanflow/models/`:

| Component | File | Result |
|---|---|---|
| **Training objective** `p_losses` | `imf_diffusion.py:258-313` | **Identical** (only a comment label differs: "U2-B2" vs "U3-B2"). Same finite-difference target `u_target = (x_t − x_r)/h`, same `v_target = x₁ − ε`. |
| **Inference predictor** `_predict_velocity` / `_predict_uv` | `imf_diffusion.py:112-135` | **Byte-identical** — u-only at sampling, aux discarded (the "FIX-3" behaviour). |
| **Loss helper** | `imf_losses.py` | **Byte-identical** (`diff -q` clean). |
| **Engine** | `imf_engine.py` | **Differs — but only visual wiring** (see §3). The iMF logic is unchanged. |

So the exact thing that makes Gen3v4 "iMF body, FM brain" — the finite-difference target that
collapses to `x₁−ε` on the linear interpolant — is **present unchanged in Gen8**. Same
conclusion: **Gen8 iMF visual aligning is FM-equivalent in capability**, and the same upgrade
unleashes the real power.

---

## 3. The only Gen8 difference: the visual backbone (not iMF math)

`imf_engine.py` adds, relative to Gen3v4:

- `if_vision: bool` + `vis_config` constructor args;
- routing `velocity_net → VisualUNet` (FiLM-conditioned **dual-camera** encoder) when `if_vision=True`;
- `state_dim` overridden to `VisualUNet.TRANSITION_DIM = 9`;
- removal of the unused standalone `sample()` helper.

All of this is **conditioning/backbone plumbing**. The velocity field is still produced by the
same dual-head (u + aux) structure, trained by the same `p_losses`, sampled by the same u-only
rule. **No iMF-specific change.**

---

## 4. Consequence for the unleash plan

The Gen3v4 [U4 plan](../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md) applies to Gen8 **as-is**,
with **one extra item** owing to the visual encoder:

> **JVP functional-purity now also covers the visual path.** The real-iMF objective computes a
> JVP through the network (`torch.func.jvp`). For Gen8 that network includes the **VisualUNet
> (dual-cam ResNet + FiLM)**. The image features must be treated as a **constant w.r.t. the JVP
> tangents** (the JVP is in `z`, `t`, `h` — *not* in the pixels), and any dropout/in-place op in
> the visual encoder must be deterministic/functional inside the JVP, exactly as required for the
> state UNet (Gen3v4 U4 §Phase 1 risks). Precompute the image embedding once, then JVP only the
> trajectory branch.

Everything else — the MeanFlow-Identity target, stop-gradient, 25% `r=t` anchor, adaptive
weighting, low-NFE inference, and the DPCC projector re-tune — is identical to the Gen3v4 plan.

---

## 4A. Keep vs drop / imfv2 — the flag covers Gen8 for free

The Gen3v4 [U4 plan §2A](../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md) decides: **keep the old
iMF code, replace only `p_losses`, gate old-vs-new with `imf_objective`** (`'fm_equivalent'`
default | `'meanflow_jvp'` = imfv2). Because Gen8 **shares `imf_diffusion.py`** with Gen3v4
(verified §2), this single in-place flag **propagates to the visual variant automatically**:

- **Do not** fork an `imf_v2/` folder or a Gen8-specific objective — that would split the shared
  core and create drift.
- Gen8's old (FM-equivalent) visual iMF stays as the **A/B baseline arm**; flipping
  `imf_objective: 'meanflow_jvp'` turns the *same* visual stack into real iMF.
- The only Gen8-specific addition remains the **JVP visual-purity** item in §4 (hold the image
  embedding constant through the JVP).

So nothing about the keep/drop decision is different for Gen8 — it inherits it through the shared
file. The old code is **not dropped**; it is the verified foundation plus a baseline arm.

---

## 5. One-line summary

> Gen8 = Gen3v4 iMF core (FM-equivalent today) + a visual encoder. To make Gen8 a *real* iMF,
> follow the [Gen3v4 U4 plan](../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md) verbatim, plus
> hold the image embedding constant through the training-time JVP.

---

## Caveats

- Diff performed 2026-06-13, branch `update_into_FM`:
  `imf_visual_aligning/models/{imf_diffusion.py, imf_engine.py, imf_losses.py}` vs
  `flow_matcher_v3_imeanflow/models/`.
- If either fork's iMF core is edited later, re-run the diff — this cross-link assumes they stay
  in sync on the iMF math.
