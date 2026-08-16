---
name: benchmark-hierarchy-who-beats-whom
description: FM-PCC benchmark hierarchy — diffusion-DPCC is THE baseline; FM/MF/AF must beat it, MF/AF must also beat naive FM; HardFlow must beat the DPCC projector
metadata:
  type: project
---

Every FM-PCC result is a claim about beating something specific. The hierarchy, and what each
comparison has to show (judged by Pareto dominance — see [[pareto-definition-of-good]]):

| arm | must beat | claim it supports |
|---|---|---|
| **diffusion + DPCC projection** | — | **THE baseline.** Every env (avoiding-d3il, UAV, …) has one; nothing is a result until it clears this |
| FM (`FlowMatchingODE`) | diffusion-DPCC | flow matching is a viable engine swap |
| **MeanFlow / AlphaFlow** | diffusion-DPCC **and naive FM** | few-step generation is a *capability*, not just a cheaper K |
| **HardFlow (arm C)** | **the DPCC projector** (arm B), not the generator | in-loop constrained sampling beats post-hoc projection |

**MF/AF beating the baseline but losing to naive FM is not a result** — it would mean the gain came
from the flow-matching swap, not from the few-step objective, which is the whole point of those
generations.

**HardFlow is a different comparison and needs a different lever.** It solves an NLP inside the
sampling loop, so it is *mathematically* guaranteed to cost more wall-clock per step than arm B's
post-hoc projection (measured ~0.10 s/step vs DPCC's ~0.027 s/step at K=2). It can therefore never
win on raw time at a matched activation threshold. The lever is the **projection threshold**: lower
it (fewer NLP solves — `1.0` = every step, `0.5` = last half, `0.0` = terminal-only) until HF
reaches the Pareto frontier against arm B on `(S&C, n_steps, avg_time)`. **A HardFlow-vs-DPCC table
at equal thresholds is a foregone conclusion, not an experiment.**

**Why:** the point of FM-PCC is to replace DPCC's stochastic diffusion engine with a deterministic
flow-matching one and keep the safety, so diffusion-DPCC is the number every generation is measured
against. Each newer generation exists to beat the previous one for a *stated* reason; comparing it
to the wrong opponent proves nothing.

**Backbone matters as much as the opponent.** The baseline is a temporal UNet, so a win by our
`sit`/`mf_dit` arms changes architecture *and* objective at once. Lead with the architecture-matched
row and mark cross-architecture wins as secondary — see
[[architecture-matched-beat-is-the-strong-claim]].

**How to apply:**
- Before reporting any result, name the opponent it is supposed to beat, and check that one first.
  Concretely, the diffusion-DPCC opponent is the baseline's **best** variant row in that batch, and
  any of our variants may be the one that beats it — see [[da-target-is-best-baseline-variant]].
- For MF/AF, always include **both** diffusion-DPCC and naive FM in the table.
- For HardFlow, compare against **arm B on the same checkpoint** — not against the generator or
  another generation — and sweep the activation threshold rather than reporting a single point.
- Thresholds: arm B = `diffusion_timestep_threshold` (env `DPCC_THRESHOLD`), arm C =
  `hardflow.activation_threshold` (env `HFFM_ACT_THRESHOLD`). Separate knobs by design; both land
  in the results-folder name as the `T` and `A` tokens (Fix_9), so sweep points no longer collide.
- Candidate baselines already measured live in the DA batch CSVs (e.g. `CAND_7`/`CAND_10` = DPCC
  K10/K20, `CAND_105` = FM ODE K20) — see [[pareto-definition-of-good]] for the file and the
  seed-matching caveat.
