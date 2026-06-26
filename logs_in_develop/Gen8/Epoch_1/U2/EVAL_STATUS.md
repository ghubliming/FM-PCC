# Gen8 Epoch_1 U2 — Post-Retrain Eval Status

**Date:** 2026-06-12
**Status:** OPEN — mixed results, partially working, needs digging
**Flag:** continue before closing or advancing Gen8

---

## What was done

U2 applied four fixes to `imf_visual_aligning/` across three files:
- **C1 (B1):** Frozen `t=0.5` → true `t_i = loop_idx / N` (`imf_diffusion.py`)
- **C2 (B3):** Deleted dead inconsistent `iMFEngine.sample()` (`imf_engine.py`)
- **C3 (B3):** Deleted dead `iMFTrajectoryModel.sample_trajectory()` + `.sample()` (`imf_trajectory_model.py`)
- **C4 (B2):** `p_losses` conditioning endpoint `(x_t, t)` → `(x_r, r)` (`imf_diffusion.py`)

Full retrain from scratch + eval run completed on cluster.

---

## Observed result

> "Good/bad mixed — close, but not as good as expected. Might have worked a little. Still needs digging."

- Some trajectories / episodes look qualitatively good — structured, reaching goal, avoiding obstacles.
- Other trajectories are still clearly failing — wrong direction, stalling, or collisions.
- Net result is a bimodal distribution: a portion of the model works, a portion does not.
- The good fraction is higher than before U2, confirming the fixes had a real effect, but
  the bad fraction is too high to call this a working policy.

---

## What this tells us

A good/bad bimodal output is a different failure mode than pure chaos (pre-U2). It suggests:

1. **Partial fix confirmed:** The B1 + B2 fixes were necessary and had real impact. The
   model is no longer completely broken. The "worked a little" fraction is the signal.

2. **Something is still wrong for a subset of inputs:** The bad cases are likely tied to
   specific conditions — certain scenes, homotopies, start positions, or conditioning
   values. If the failures cluster, there is a remaining code or conditioning bug. If
   they are random, it is a capacity/data problem.

3. **Visual conditioning is the new suspect:** Gen8 adds image observation on top of the
   iMF backbone. The image encoder path (CNN + projection into `cond`) was not audited in
   U2. If the image features are corrupted, missing, or misaligned with training, this
   would produce exactly a good/bad split — good cases where the iMF trajectory alone
   carries the task, bad cases where visual avoidance is required.

4. **`clip_denoised` / normalisation drift:** Gen8's visual diffusion config may have
   mismatched normalisation between the image encoder used at training time and the one
   used at eval. The U2 B9 fix (pkl/config mismatch warning) should be checked — if it
   fired during eval, those warnings identify the exact parameters that drifted.

---

## Next steps (to investigate before closing)

- [ ] **Stratify good vs bad cases** — pull the per-episode metric from eval logs. Break
      down by scene and homotopy. If a specific scene or homotopy dominates the failures,
      that is the debugging target.
- [ ] **Check B9 mismatch warnings** — scan the eval slurm log for
      `[WARN] pkl/config mismatch` lines. Any fired warning is a direct lead.
- [ ] **Audit image encoder path at eval** — verify that `bp_img` is being passed to the
      visual encoder with the same preprocessing (resize, channel order, normalisation)
      used during data collection and training. A single channel-swap or normalisation
      scale difference here explains good/bad splitting.
- [ ] **Compare with Gen8 baseline (no visual)** — if a trajectory-only ablation (masking
      image input to zeros) produces similar or better success rate, the visual path is
      actively hurting. If it's much worse, the visual path is essential but noisy.
- [ ] **Run with `clip_denoised=True`** (visual avoiding config default) — confirm this
      was active during the retrain and eval. A mismatch between train/eval on this flag
      shifts the output distribution at every denoising step.
- [ ] **Check EMA vs raw weights at eval** — confirm the eval loaded `ema_model.model`
      (B6 fix in Gen9 U4 audits). If the Gen8 eval script has the same B6 bug, it's
      running on raw weights which converge more slowly and are noisier.

---

## Verdict

**Do not close.** Gen8 is the most promising candidate so far — the good fraction is real
and the fixes made a measurable difference. The bad fraction has identifiable candidate
causes (image pipeline, EMA loading, normalisation). One focused debug pass may be enough
to push this to an acceptable success rate. Dig before concluding architectural failure.
