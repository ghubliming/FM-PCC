---
name: da-target-is-best-baseline-variant
description: "DA first principle — target = BEST baseline DPCC row in the batch; any of our variants beating it on ANY axis (S&C held) is a win; always run this check in every DA"
metadata:
  type: feedback
---

In every Data_Analysis batch, first pick **one Target row**: the **best-performing baseline
(diffusion-)DPCC variant**, whichever variant name that happens to be — `dpcc-c`, `dpcc-r`,
`dpcc-t`, `diffuser`, … "Best" is read the [[pareto-definition-of-good]] way: gate on S&C, then
the row that is Pareto-non-dominated on `(n_steps, avg_time)` — fewest steps at acceptable time,
*or* fastest time at only marginally worse steps.

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
