# FROM v3 → v2 · 2026-09-16 · training settings differ across generative models

**Finding (v3.14, `v3` §5.5 `tab:train`, from the resolved configs in the training logs):**

| environment | diffusion, flow matching | MeanFlow, consistency training |
| :-- | :-- | :-- |
| obstacle avoidance | batch 8×2, lr 1e-4, raw weights | **batch 32×2, lr 5e-4, EMA weights** |
| alignment | all 64×2, lr 2e-4; action loss weight **10** (diffusion) vs **1** (flow-based) | |
| quadrotor | uniform: 8×2, lr 1e-4, action weight 1, raw weights | |

Backbone and parameter count are matched; the training budget per update is not.

**For v2:** if Ch 1, 3 or 4 says the models differ *only* in the training target, or that training is held
fixed, qualify it — "same backbone and parameter count" is true, "same training" is not. A search on
2026-09-16 found no explicit "identical training" wording in `thesis_v2.tex`; check contribution and
positioning sentences by reading. v3 states the caveat in a `\guard` below `tab:train`.
