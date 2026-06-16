# U5 — Forward plan (step in step)

**Date:** 2026-06-14
**Premise:** U4 imfv2 code is complete + sign-verified locally. Now **run current code first, then
take the next step** — one step at a time.

Back-refs: [U4 CHANGELOG](../U4/CHANGELOG.md) · [U4 PLAN](../U4/PLAN_Unleash_Full_iMF.md).
Gen8 mirror: [Gen8 U4 NEXT_STEPS](../../Gen8/Epoch_1/U4/NEXT_STEPS.md).

---

## Step 1 — Run current code (cluster) — the correctness gate

1. Train briefly with `imf_objective: 'meanflow_jvp'` (config `flow_matching_v3_imeanflow` block).
2. Sample at `ode_inference_steps_v3: 1`; measure **1-NFE reconstruction RMS** vs ground truth.
3. **Pass** (low RMS) ⇒ JVP objective is working → go to Step 2.
   **Fail** (diverging RMS) ⇒ check **forward-AD/encoder** first (sign is verified); for Gen8 the
   prime suspect is **BatchNorm-in-JVP** (switch to eval/GroupNorm or precompute embedding).

## Step 2 — Phase 4(a): NFE-aware snap schedule — *code, writable now*

- Make the DPCC snap schedule robust at 1–2 NFE (currently
  `snapping_start_idx = int((1−threshold)·flow_steps)`; at low NFE the "near-end" window is the
  whole rollout — U3 §9.3).
- **Gate it** so the validated 10-NFE default path stays byte-for-byte unchanged.
- File: `flow_matcher_v3_imeanflow/models/imf_diffusion.py:203-204`
  (+ mirror in `imf_visual_aligning` for Gen8 if visual low-NFE is wanted).

## Step 3 — Phase 4(b): re-validate constraint satisfaction — *cluster-only gate*

- Tune `diffusion_timestep_threshold` for the new NFE; **re-verify avoiding constraints hold at
  1–2 NFE**, not just at 10. This is a safety gate, not a nicety — do not ship low-NFE avoiding
  until it passes.

## Step 4 — A/B vs FM (Phase 5)

- Three columns, same data/seeds: (i) FMv3ODE @10 NFE, (ii) Gen3v4-iMF-old @10 NFE, (iii)
  full-iMF @1–2 NFE. Report **quality** *and* **`fm_ms`**.
- **Success = iMF @1–2 NFE matches FM @10 NFE quality at a fraction of the latency.**

## Optional / later
- Phase 3 interval-CFG.
- Aux v-head stabilizer (`meanflow_aux_weight: 0.05`) if main loss is noisy.

---

**Discipline:** complete each step and confirm before starting the next. No commit/push until asked.
