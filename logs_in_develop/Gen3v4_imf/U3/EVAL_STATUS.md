# Gen3v4 U3 — Post-Retrain Eval Status

**Date:** 2026-06-12
**Status:** OPEN — marginal improvement, not yet acceptable
**Flag:** continue digging before closing Gen3v4

---

## What was done

U3 applied the two iMF fixes to `flow_matcher_v3_imeanflow/models/imf_diffusion.py`:
- **C1 (B1):** Replaced frozen `t=0.5` with true `t_i = loop_idx / N`
- **C2 (B2):** Swapped `p_losses` conditioning endpoint from `(x_t, t)` → `(x_r, r)`

Full retrain from scratch + eval run completed on cluster.

---

## Observed result

> "Looks slightly better — but still feels bad."

- The fixes had a detectable but small positive effect.
- Trajectories are not chaotic in the way the frozen-t model was, but behavior quality
  remains clearly below the acceptable bar.
- The improvement is not enough to call this a working model.

---

## What this tells us

The B1 + B2 fixes were the highest-confidence code bugs in the architecture. That they
helped only slightly means one or more of the following:

1. **Capacity / expressiveness**: The UNet backbone (`unet1d_temporal_cond.py`) may be
   too small or the wrong inductive bias for iMF in this obs space. Gen3v4 was built on
   top of Gen3v3 which already had structural issues.

2. **Training data quality**: Gen3v4 trains on the existing trajectory data. If the data
   is noisy, sparse in important regions, or has systematic biases, the model cannot learn
   well regardless of the iMF formulation.

3. **Remaining conditioning mismatch**: B1+B2 fixed the most obvious mismatches, but there
   may be subtler train/inference gaps not yet audited (e.g., normalisation, `h` field
   initialisation, `returns` tensor shape).

4. **Architectural ceiling**: iMF is more demanding than vanilla FM — it requires the model
   to learn a mean-field velocity, not just a regression target. Gen3v4 may not have the
   depth/width to learn this reliably.

---

## Next steps (to investigate before closing)

- [ ] **Quantify "slightly better"** — pull specific metrics from the eval log (success
      rate, avg deviation, failure mode breakdown) vs the pre-U3 baseline. Confirm the
      improvement is statistically real and not noise.
- [ ] **Inspect failure cases** — are failures clustered by scene, speed, or homotopy?
      Structured failures point to a remaining code bug; random failures point to capacity.
- [ ] **Compare Gen3v4 vs Gen3v3** — if Gen3v3 (vanilla FM) was also poor, the problem
      pre-dates iMF and is in the shared architecture or data. If Gen3v3 was decent, the
      issue is iMF-specific.
- [ ] **Audit `h` initialisation at inference** — verify `h` is initialised identically
      to how it was used during training. A shape or dtype mismatch here could suppress
      the mean-field contribution entirely.
- [ ] **Try reducing `flow_steps`** — if 1-step or 2-step sampling is dramatically worse
      than 10-step, the velocity field is not well-learned (capacity). If 1-step ≈ 10-step,
      the field is clean and the problem is elsewhere.

---

## Verdict

**Do not close.** The fixes worked marginally, which rules out pure frozen-t chaos but
does not establish a working model. Gen3v4 needs at least one more investigation pass
before a go/no-go conclusion.
