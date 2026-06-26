# U3 — CHANGELOG: imfv2 (flag-gated MeanFlow-JVP objective) for Gen8 visual aligning

**Date:** 2026-06-13
**Branch:** `update_into_FM`
**Implements:** the Gen3v4 [U4 plan](../../../Gen3v4_imf/U4/PLAN_Unleash_Full_iMF.md), applied to
the Gen8 visual fork per [`./iMF_Core_Same_As_Gen3v4.md`](./iMF_Core_Same_As_Gen3v4.md).
**Status:** Code complete, **untested** (no local runtime). Cluster verification required — §6.

---

## 1. What changed

Added the **same flag-gated `meanflow_jvp` objective** to the Gen8 visual iMF, default OFF:

```
imf_objective: 'fm_equivalent'   # DEFAULT — unchanged FM-equivalent baseline arm
             | 'meanflow_jvp'     # NEW — real MeanFlow Identity via JVP (imfv2)
```

Existing visual iMF runs are unaffected (default `fm_equivalent`).

---

## 2. Files touched

| File | Change |
|------|--------|
| `imf_visual_aligning/models/imf_diffusion.py` | `iMeanFlowODE.__init__`: same 5 new params + storage. `p_losses`: dispatch on `imf_objective`. **New** `_p_losses_meanflow_jvp(...)` (visual-annotated). |
| `config/aligning-d3il-visual.py` (`imf_visual_aligning` block) | Added `imf_objective: 'fm_equivalent'` (default) + the 4 meanflow knobs + switch instructions. |

**Why this is enough for the visual path:** `VisualIMF` (the class the visual config actually
loads) **subclasses `iMeanFlowODE`** and its `loss()` calls `self.p_losses(...)`
(`visual_imf_diffusion.py:43,61`). Since `p_losses` now dispatches on `imf_objective`, the new
objective is inherited by the visual model with no edit to `visual_imf_diffusion.py`.

---

## 3. The objective (identical math to Gen3v4)

START-anchored MeanFlow Identity in DATA-AT-1: `u_target = v_inst + h·du/dr`, JVP tangents
`(∂z=v_inst, ∂time_r=+1, ∂h=−1)`, stop-gradient, 25% `r=t` anchor, adaptive + `loss_weights`-scaled
MSE. Full derivation in the [Gen3v4 U4 changelog §3](../../../Gen3v4_imf/U4/CHANGELOG.md) — not
duplicated here (the cores are the same file logic).

### Visual-specific point (the one Gen8 addition)

The JVP differentiates **only** the trajectory inputs `(z, r, h)`. The camera images live in
`cond` and are **held constant** (captured in the JVP closure, not a primal), so the visual
embedding does not move under the tangent — this satisfies the plan's "hold the image embedding
constant through the JVP" requirement automatically. `apply_conditioning` skips the string
`'visual'` key, so only the numeric obs anchor is inpainted (consistent with the existing path).

> Efficiency note: the encoder is re-run inside the linearized pass (correct but not free).
> A precompute-the-embedding optimization is a later refinement, not needed for correctness.

---

## 4. How to run an imfv2 visual experiment

In `config/aligning-d3il-visual.py` `imf_visual_aligning` block:
```python
'imf_objective': 'meanflow_jvp',
# then drop the inference steps to 1–2 in the eval/inference config (few-step),
# optional: 'meanflow_aux_weight': 0.05
```

---

## 5. Not done (deferred)

- DPCC / projector low-NFE re-tune (shared Phase 4 task).
- Improved-iMF interval CFG (Phase 3, optional).
- No train/eval run.

---

## 6. Verification status

**Done locally (2026-06-14):** the JVP sign/tangent derivation (shared with Gen3v4) was validated
by a numpy finite-difference check on a non-linear field — implemented sign `u=v_inst+h·du/dr` is
exact (`max|err|=1.9e-06`) vs the flipped sign (`0.34`, ~184,000× worse). See
[Gen3v4 U4 CHANGELOG §6.0](../../../Gen3v4_imf/U4/CHANGELOG.md). So the sign is **not** the
suspect for the visual fork — **BatchNorm-in-JVP is** (§6.2 below).

**REQUIRED on the cluster** (cannot be done here — no torch/GPU). Same gate as Gen3v4 **plus** a
visual-encoder check:

1. **1-NFE reconstruction RMS** — if diverging, suspect the encoder/forward-AD (item 2) before the
   JVP sign, which is independently verified.
2. **forward-mode AD through VisualUNet** — confirm `torch.func.jvp` survives the ResNet/FiLM
   encoder. **BatchNorm in train mode is the prime suspect** for forward-AD failure; if it errors,
   switch the encoder norm to eval/GroupNorm for the imfv2 run or precompute the image embedding
   outside the JVP.
3. **A/B** — `meanflow_jvp`@1–2 NFE vs `fm_equivalent`@10 NFE on quality + `fm_ms`.

---

## Caveats

- Default behaviour unchanged; zero risk to existing visual runs.
- The `meanflow_jvp` visual path is **unverified** — §6 is a hard gate, BatchNorm-in-JVP being the
  most likely first failure mode unique to the visual fork.
- Syntax-checked (`py_compile`) only; no execution.
