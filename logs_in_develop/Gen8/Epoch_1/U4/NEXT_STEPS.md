# Gen8 U4 — Forward plan (step in step, visual fork)

**Date:** 2026-06-14
**Premise:** U3 imfv2 code is complete + the JVP sign is verified locally (shared with Gen3v4).
**Run current code first, then take the next step** — one step at a time.

Back-refs: [U3 CHANGELOG](../U3/CHANGELOG.md) · [iMF core = Gen3v4](../U3/iMF_Core_Same_As_Gen3v4.md) ·
Gen3v4 mirror: [Gen3v4 U5 NEXT_STEPS](../../../Gen3v4_imf/U5/NEXT_STEPS.md).

---

## Honest flags carried into this step

1. **Untested at runtime — cannot be done here** (Docker = AI-coding only). The 1-NFE
   reconstruction check is the hard gate. JVP sign is **independently verified locally** (numpy
   finite-diff, exact to 1.9e-06 vs 184,000× worse for the flipped sign) — so divergence points to
   the encoder, not the sign.
2. **Gen8-specific risk:** `torch.func.jvp` through the **VisualUNet** — **BatchNorm in train mode**
   is the likely first failure for forward-mode AD. Workaround: eval/GroupNorm for the imfv2 run,
   or precompute the image embedding outside the JVP.

---

## Step 1 — Run current code (cluster) — correctness gate

1. Train briefly with `imf_objective: 'meanflow_jvp'` (config `imf_visual_aligning` block,
   `config/aligning-d3il-visual.py`).
2. **First failure to expect = forward-AD through the visual encoder.** If `torch.func.jvp` errors
   → apply the BatchNorm workaround above before anything else.
3. If it runs: sample at 1–2 NFE; measure **1-NFE reconstruction RMS**. Pass ⇒ go to Step 2.

## Step 2 — Phase 4(a): NFE-aware snap schedule (if visual low-NFE is wanted) — *code*

- Mirror the Gen3v4 fix in `imf_visual_aligning/models/imf_diffusion.py` (snap window degenerates
  at low NFE). **Gate it** so the validated 10-NFE visual path is byte-for-byte unchanged.

## Step 3 — Phase 4(b): re-validate constraints — *cluster-only gate*

- Re-verify the visual avoiding/aligning constraints hold at the new NFE, not just at 10.

## Step 4 — A/B vs FM (Phase 5)

- Same data/seeds: FM-equivalent @10 NFE vs full-iMF @1–2 NFE. Report **quality** *and* `fm_ms`.
- **Success = iMF @1–2 NFE matches @10 NFE quality at a fraction of latency.**

---

**Discipline:** finish each step and confirm before the next. No commit/push until asked.
Default behaviour is byte-for-byte unchanged; imfv2 is one config key away.
