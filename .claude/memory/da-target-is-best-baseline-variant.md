---
name: da-target-is-best-baseline-variant
description: "DA first principle — Target = best row of the PAPER's DPCC config (K20 + aw10 + GaussianDiffusion); any of our variants beating it on ANY axis (S&C held) is a win; always run this check in every DA"
metadata:
  type: feedback
---

In every Data_Analysis batch, first pick **one Target row**: the **best-performing variant of the
paper's baseline DPCC configuration**. "Best" is read the [[pareto-definition-of-good]] way: gate on
S&C, then the row that is Pareto-non-dominated on `(n_steps, avg_time)`.

🔴 **The paper's DPCC config is pinned: `K = 20`, `aw = 10`, `models.GaussianDiffusion`.**
Target selection ranges over the **projection variants** of that config (`dpcc-c`, `dpcc-r`,
`dpcc-t`, `-tightened`, `diffuser`, …) — **not** over K and **not** over `aw`. Rows at other K or
other `aw` are ineligible as Target no matter how good they look.
On `avoiding-d3il` the Target resolves to `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`,
variant `dpcc-c-tightened`.

**Other K values are additional diagnostics, and must be reported as such.** They are worth
analysing — e.g. on `avoiding` the baseline's own **K10 strictly Pareto-dominates its K20**
(1.000 / 68.70 steps / 0.3217 s vs 1.000 / 70.13 / 0.5534, 5 seeds), so doubling the diffusion
budget buys nothing — but that does *not* move the Target. Report the K20 margin as the headline
(it is the paper comparison) and the K10 margin as the **conservative check**: "does it also beat
the best DPCC can do on this task?" Both belong in the write-up; only K20 is the Target.

**The baseline is always the DPCC results.** Every batch must contain them; without a DPCC row
there is no Target and no claim.

**The Target is a single number set, not a per-variant pairing.** Once it is fixed, *any* row from
our side that beats it counts as a **beat** — `FMv3ODE dpcc-t` beating a `dpcc-c` Target is a beat;
a new arm like `HF-r tightened` beating the same Target is also a beat. Horizontal
same-variant comparison (`dpcc-c` vs `dpcc-c`) is still worth showing as diagnostics, but it is
**not** the gate and losing one such pairing does not cancel a Target beat.

**Win rule: S&C is the only gate; after that, beating the Target on ANY axis is a win.** At S&C at
least as good as the Target, a row with fewer `n_steps` **or** lower `avg_time` has reached the
goal — report it as a win, not as "merely a trade-off". Strict Pareto dominance (both axes) is the
*stronger* claim; label it as such when it happens, but do not withhold the win when only one axis
lands. This is the intended reading of [[pareto-definition-of-good]] for the Target comparison —
that memory's stricter "both axes or it's a trade-off" phrasing applies to calling a config the
*best overall*, not to whether the goal was reached.

**This check runs in every DA, always** — no batch report is complete without the Target row and
the per-axis beat verdict against it, even when the user only asked for something else.

**Why:** the baseline we are arguing against is DPCC *as a method*, and a method is deployed at its
best setting — nobody ships DPCC's worse variant. So the honest bar is the baseline's best row.
Conversely, our engine is also free to pick its best variant. Matching variant names across the two
families is a convenience of the sweep, not a scientific constraint, and insisting on it hides real
wins.

**How to apply:**
- State the Target explicitly at the top of any DA writeup: variant, seed/trial count, and the
  `(S&C, n_steps, avg_time)` triple it is defined by.
- **Check `aw` and `K` before fixing the Target.** Several `diffusion/` candidates in a batch are
  `aw1` or non-K20 and look competitive; they are not the paper's DPCC. Grep the `Full_Path` for
  `K20` + `aw10` first, and say in the DA which candidates were excluded and why.
- Give a **candidate-index table** near the top mapping every quoted row to its `Candidate` ID and
  `Full_Path`, so the reader can re-derive any number. Candidate IDs are per-batch and unstable —
  never carry them between CSVs.
- Compare within the **same batch / same env / same seeds** — the Target must come from the same
  run set as the challengers, or the comparison is unmatched.
- Report as "**Target reached**: <our row> beats <baseline best row> on <axis>", and always name
  which axes won and which lost — one axis is enough for the win, both axes upgrades it to
  "dominates the Target".
- Still respect the step-count caveat: `n_steps` is averaged over successful trials only, so a row
  with worse S&C posts a flattering step count and cannot claim a step win.
- The arm-level obligations in [[benchmark-hierarchy-who-beats-whom]] still apply on top of this
  (MF/AF must also clear naive FM; HardFlow's opponent is the projector arm, swept over its
  activation threshold).
