# Gen3v4u2 fix_3 — Apply both deviations found by REFERENCE_IMF_AUDIT.md

**Date**: 2026-06-01
**Branch**: `update_into_FM`
**Based on**: [`../fix_2/REFERENCE_IMF_AUDIT.md`](../fix_2/REFERENCE_IMF_AUDIT.md)
**Precedents**: [`../fix_1/INVESTIGATION.md`](../fix_1/INVESTIGATION.md) (target formula correction), [`../fix_2/INVESTIGATION.md`](../fix_2/INVESTIGATION.md) (jitter hypothesis tree).

---

## Summary

Per the second-pass reference iMF audit (`fix_2/REFERENCE_IMF_AUDIT.md`
§7), our iMF implementation has **two architectural deviations** from
the canonical iMF reference at `/workspaces/imeanflow`. Both are
applied here as one-line / few-line code changes; **no retraining is
required.**

| Deviation | Fix | File |
|---|---|---|
| **A** — `_predict_velocity` adds `sample_aux_weight * aux` at inference. Reference iMF uses ONLY the u (mean-velocity) head; the v/aux head's transformer blocks are explicitly NOT instantiated when `eval_mode=True` (cf. `imeanflow/models/imfDiT.py:282-288`). | Removed the aux contribution from the inference output: `return velocity` instead of `return velocity + self.sample_aux_weight * aux`. The aux head's trained weights stay loaded but unused at inference. | `flow_matcher_v3_imeanflow/models/imf_diffusion.py:_predict_velocity` |
| **B** — `p_sample_loop` passes the iterating `t_cont = loop_idx/N` to the model at every sampling step. Reference iMF (`imeanflow/models/imfDiT.py:370-372`) explicitly conditions ONLY on `h`, citing the iMeanFlow paper (Kaiming He et al., arXiv:2502.13129). Conditioning on a varying `t` at inference can excite spurious t-dependence learned during training. | Freeze the `t` input to a constant (`T_CONST_INFERENCE = 0.5`) at every sampling step. The model's `time_mlp(t)` contribution then becomes a fixed bias for the duration of the rollout, while `h_mlp(h)` continues to vary per step (`h = 1/N`). This converts our (t,h)-conditioned model into an effectively h-only-conditioned model at inference, without touching the trained `time_mlp` weights. | `flow_matcher_v3_imeanflow/models/imf_diffusion.py:p_sample_loop` |

Together with fix_1 (target-formula correction) and the aux-disable
that fix_2's Hypothesis A predicted, these two changes bring our iMF
inference into structural alignment with the reference iMF code.

---

## Files Changed

| File | Lines | Change |
|---|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | `_predict_velocity` (~lines 115-130) | Drop `+ sample_aux_weight * aux` from return value. Aux output renamed to `_aux` to signal disuse at inference. Comment block added with reference-code citations and a brief justification. |
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | `p_sample_loop` (~lines 155-200) | Replace per-iteration `t_cont = loop_idx/N` with a single pre-loop `t_const = 0.5` tensor. Velocity call now uses `t_const` instead of `t_cont`. `loop_idx` is still computed inside the loop (it's used by the projector code further down). Comment block added with reference-code citations and the t=0.5 rationale (training-distribution midpoint of `1 − Beta(1.5, 1.0)`). |

**Files NOT touched:**

- Training code (`p_losses`, target formula, loss weighting): unchanged. The aux head is still trained — fix_3 only changes inference behavior.
- The model architecture (`iMFTrajectoryModel`, `Flow_matcher_U_Net_v2`): unchanged. Both `time_mlp` and `h_mlp` modules stay; only what we PASS at inference changes.
- Sampling step direction (`x = x + velocity * dt`) and forward Euler 0→1: unchanged (verified equivalent to reference's backward integration in DATA-AT-0).
- fix_1's target formula `(x_t − x_r) / h`: unchanged (verified correct by reference audit).

---

## Why This Doesn't Require Retraining

### Deviation A

The aux head's trained weights stay in the checkpoint file. They were
trained against a meaningful target (`v_target = x_start - x_base`,
i.e., the instantaneous velocity) and the training loss continues to
use them. We just don't read the aux output at inference. The
`sample_aux_weight = 0.1 * v_mix ≈ 0.009` scalar parameter also stays
in the checkpoint; it's now unused. No checkpoint compatibility issue.

### Deviation B

The `time_mlp` weights stay in the checkpoint and continue to be
applied at inference — just with a CONSTANT input (`t = 0.5`) instead
of a varying input. The model computes `time_mlp(0.5)` once per
batch; this produces some fixed embedding vector that gets added to
the h-dependent embedding at every step. Equivalent (at inference) to
running a model that lacks `time_mlp` and has a fixed bias term in
the conditioning, which is structurally what reference iMF does.

The risk: the trained `time_mlp` might have learned a `t`-dependence
that is meaningful (not flat). In that case, freezing it to t=0.5
loses that signal. Mitigations:

1. We chose `t=0.5` as the midpoint of training's t distribution
   (training samples `t = 1 - Beta(1.5, 1.0)` which has mean ≈ 0.4).
   Close enough to the training mean that the constant should be
   on-distribution for time_mlp.
2. If post-fix_3 trajectories are still jittery, the next escalation
   is to retrain a model with `time_mlp` removed entirely (so the
   model is structurally h-only-conditioned, matching reference iMF
   exactly).

The post-fix_3 expectation: trajectories should be SMOOTHER than
post-fix_1 (Deviation A alone explains step-to-step jitter), and
similar to or better than reference iMF on linear-interpolant
manifolds.

---

## Verification Plan (post-eval)

1. **Sanity:** load the existing checkpoint, run eval at `flow_steps_v3=10`.
   No crash expected — the model architecture and trained weights are
   identical; we only changed inference plumbing.
2. **Visual:** compare the resulting rollout PNG to the pre-fix_3 PNG
   (`logs/avoiding-d3il/plans/.../halfspace_both-hard/diffuser.png`).
   Expected: trajectories now appear smoother and follow the data
   distribution more closely; no chaotic straight lines (those were
   fix_1's symptom) AND no per-step jitter (that was fix_2-A's
   symptom).
3. **Quantitative:** check the per-step velocity magnitudes in the
   first-replan diagnostic. Pre-fix_3, the model output had ~0.9%
   contribution from aux per step. Post-fix_3, that contribution is
   zero.

---

## Reverts (if needed)

Both fixes are localized to ~10 lines in one file. Git revert via:

```bash
# Revert Deviation A:
git checkout HEAD~ -- flow_matcher_v3_imeanflow/models/imf_diffusion.py
# (restores _predict_velocity to the aux-mixed form)

# Or selectively revert just one of A/B by editing the file
# back to the previous return statement / per-iteration t_cont.
```

If post-fix_3 results are WORSE than pre-fix_3 (unexpected), the most
likely culprit is the chosen `T_CONST_INFERENCE = 0.5`. Try other
values (0.0, 0.4, 1.0) to find a working constant — or revert
Deviation B only and keep Deviation A (the safer half of the patch).

---

## Cross-References

- `fix_2/REFERENCE_IMF_AUDIT.md` §5.3 — original finding of Deviation A.
- `fix_2/REFERENCE_IMF_AUDIT.md` §7.1 — STRONGER evidence for Deviation A from `imfDiT.py:282-288` (v_heads not instantiated in eval_mode).
- `fix_2/REFERENCE_IMF_AUDIT.md` §7.2 — Deviation B finding from `imfDiT.py:370-372`.
- `fix_2/disable_aux_at_inference.py` — original monkey-patch test for Deviation A. This file is now obsolete (the fix is permanent in source); keep for historical reference only.
- iMeanFlow paper (Kaiming He et al., arXiv:2502.13129) — cited in the reference repo's source comment.

---

## One-line Summary

Per the REFERENCE_IMF_AUDIT.md findings, two architectural deviations
from canonical iMF were applied as inference-only code changes (no
retrain): (A) drop aux contribution from `_predict_velocity` to match
reference's u-only inference; (B) freeze sampling-time `t` to a
constant (0.5) to mimic reference's h-only model conditioning. Both
fixes are reversible per `imf_diffusion.py` with no checkpoint
incompatibility.
