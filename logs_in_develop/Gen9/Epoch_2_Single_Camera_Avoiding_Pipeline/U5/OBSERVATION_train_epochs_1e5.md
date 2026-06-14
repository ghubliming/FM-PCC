# U5 — Observation: 1e4 → 1e5 Training Steps

**Date:** 2026-06-13

---

Extending training from 1e4 to 1e5 steps shows visibly better loss curves overall.
However, both train loss and val loss **rebound from their lowest point around ~1e4** —
i.e. the model reaches a trough early, then loss slightly rises and plateaus through the
remaining 90k steps.

This is consistent with a few possibilities:

- The LR cosine schedule decays too slowly relative to the longer horizon, keeping the
  model in a noisy regime past the trough.
- The `episode_split` test set is small enough that val loss has high variance; the
  apparent rebound may be noise rather than true overfitting.
- `state_best` (saved on lowest val loss) was likely captured around the ~1e4 trough,
  meaning the extra 90k steps do not affect which checkpoint eval actually uses — but the
  final trajectory shape of the curve looks healthier.

**Practical consequence:** `diffusion_epoch: 'best'` is doing real work here. The model
that actually gets evaluated is from the trough, not the end of training. The better-
looking curves at 1e5 are evidence of better optimisation dynamics, but the operative
checkpoint may have been available much earlier.

Worth checking: compare `state_best` step index from the 1e5 run against the 1e4 run to
confirm whether the extra training actually shifted the winning checkpoint or just extended
the plateau.
