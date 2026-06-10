# Gen9E2U2F4 — Problem & Solution

**Date:** 2026-06-10 · **Scope:** Avoiding Visual-DPCC (DDPM) "exploded chaotic lines" at eval
**Inputs:** `Fix_4/INVESTIGATION.md`, `Fix_4/VERDICT.md`, `temp/Gen9E2_debugging/{losses.pkl, model_config.pkl}`, full code audit this session
**Principle applied:** code bugs first — training-quality explanations only where code is exonerated.

---

## Problem

DDPM avoiding eval produces exploded lines that look like **zero learning** — while training loss converged to 0.0015 (near-perfect memorization). The FM engine on the same data/UNet/eval works. `clip_denoised` was investigated twice (Fix_3, Fix_4) and is a confirmed red herring.

## Root causes found (code/config level — ranked)

### C1 — Config drift: checkpoint trained with K=20, proven recipe is K=100 ⚠️ PRIMARY

- The evaluated checkpoint is `H8_**K20**_D..._aw10_VTrue_steps200_bs64`; its frozen `model_config.pkl` confirms `n_diffusion_steps = 20`.
- The **original** `config/avoiding-d3il-visual.py` (commit `e5d0291`) had `'n_diffusion_steps': 20`. A later hotfix (`d0c2a5c`) corrected it to **100** — matching the working aligning DDPM recipe — **but the model was never retrained**. Eval silently loads K=20 from the pkl (config .py is never read at eval).
- Why K=20 is fatal with `clip_denoised=False` (mandatory per DPCC): cosine schedule over 20 steps gives `sqrt(1/ᾱ−1)` amplification of ε-error ≈ **12.8× at t=18, 1284× at t=19**, betas clip at 0.999 on the final steps (posterior noise std ≈ 1 at the first denoise step), and the chain has only 20 correction opportunities instead of 100. Nothing bounds `x_recon`. The per-step ε-accuracy demand is far beyond what K=100 requires — the aligning engine (byte-identical code) succeeds at K=100.

### C2 — Eval uses the worst checkpoint (no best-checkpoint logic)

`losses.pkl`: test loss minimum **0.0325 @ step 11,000**; eval loads "latest" = step 99,000 with test loss **0.1685 (5.19× worse)**. Trainer never saves a best checkpoint; eval calls `utils.get_latest_epoch`. The model overfit hard after ~11k and the pipeline guarantees the overfit weights are the ones evaluated.

### C3 — Silent pkl precedence (the Fix_4 no-op trap)

Eval reconstructs everything from frozen config pkls. The intended design ("`.py` config overrides pkl with a console warning") was never implemented — `.py` edits at eval time do nothing, which is exactly why Fix_4 changed nothing. This is a recurring foot-gun, not just a one-off.

### Exonerated (audited clean this session)

Datasets, normalizers, UNets, helpers: byte-identical between FM (works) and DDPM (fails) avoiding packages. Eval scripts: functionally identical after name normalization (image preprocessing `/255.`, BGR→RGB, obs normalization all match). Engine: identical to working aligning DDPM. The ±5 action-only clamp is identical in aligning.

## Solution (for the implementing agent)

1. **Retrain avoiding DDPM with full recipe parity to aligning:** `n_diffusion_steps=100` (config already says 100 — just retrain), everything else unchanged. This is the main fix.
2. **Add best-checkpoint saving** to the shared Trainer (save `state_best.pt` when test loss improves) and make eval prefer it over latest. Apply to both packages.
3. **Cheap pre-test (before retraining):** if a ~step-11k checkpoint exists on the cluster, re-eval it as-is. It isolates C2 from C1: partial structure → C1+C2 both matter; still chaos → C1 dominates.
4. **Implement the missing override warning (C3):** at eval load, compare pkl values vs `.py` config for `n_diffusion_steps`, `clip_denoised`, `dim`, `horizon`; print a loud mismatch warning (the n_timesteps warning that already exists in the imf eval is the pattern to copy).
5. Do **NOT** touch `clip_denoised` (must stay False) and do **NOT** train longer at K=20 — both already disproven.

**Expected outcome:** K=100 + best-checkpoint eval produces bounded trajectories comparable to aligning DDPM. If it still trails FM, that residual is the legitimate paradigm result — but only claim it after C1/C2 are fixed.
