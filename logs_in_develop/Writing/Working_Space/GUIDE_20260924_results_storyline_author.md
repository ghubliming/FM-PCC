# GUIDE — the storyline of the results, per environment (author, 2026-09-24)

**Status:** the author's principle for every conclusion of the thesis: the per-section conclusions and §6.4 of Ch 6,
Ch 7, Ch 8, the abstract and the Ch 1 contributions. Every draft (v2, v3, v4) reads its results through it. If a
sentence dilutes one of these lines, the sentence is wrong, not the line.

## The author's words (verbatim, 2026-09-24)

> "you know we spend so much effort prove in the avoiding the mf/af donimte and then proved plus the HF in the alining
> this is already done. the UAV - pillars is just a again reaffirm of the avoiding and it success and we also point the
> UAV controller in the pillars vs avoiding; in the corridor we proved something new that the expertdata issue and the
> whole DPCC concept works on our new env; in hte s_Curve we deep discuss the controller in extreme trajs."

## The five lines

| Environment | Its role in the thesis | What its conclusion must say first |
| :-- | :-- | :-- |
| **D3IL-avoiding** (§6.1) | **The main proof.** Done. | The average-velocity models (MeanFM, CI-MeanFM) **Pareto-dominate the diffusion baseline of DPCC**, with its own 4.0 M temporal U-Net: the same success with constraint satisfaction, fewer control steps, about a thirtieth of the time per action. |
| **D3IL-aligning** (§6.2) | **The proof again, plus endpoint projection (HardFlow).** Done. | Analytic average-velocity matching dominates the baseline on a vision-conditioned, contact-rich task, and **endpoint projection** is the projector that wins at the budget this task needs. |
| **UAV-pillars** (§6.3.1) | **A reaffirmation** of D3IL-avoiding on the quadrotor, plus the **plant and controller** point. | The operating points that win on D3IL-avoiding keep their result when a quadrotor flies them: the declared constraints transfer exactly, and the models keep their order. What the vehicle and its controller change against the manipulator: closer tracking, and collisions that end the flight. |
| **UAV-corridor** (§6.3.2) | **Something new**, in two parts. | (1) **The whole DPCC concept works in our new environment**: projection repairs plans against constraints no demonstration satisfies, in three dimensions. (2) **The expert-data issue**: on straight-line demonstrations the generative model does not matter; the budget does. |
| **UAV-s-curve** (§6.3.3) | **The controller** on extreme trajectories, discussed in depth. | The tracking controller becomes the limit on a route of sharp turns. Model and projector results there are read for the controller. |

## How to apply

- **Lead each conclusion with its line.** Numbers, trade-offs and caveats follow it; they never replace it.
- **Never write a summary that levels the flow models with the baseline on D3IL** ("≈ diffusion", "do not separate",
  "reach its success"). The D3IL result is a Pareto dominance. The v3.68d "≈" sentence was withdrawn for this at v3.89.
- **Scope limits stay, as caveats and not as conclusions.** For example, UAV-pillars did not fly the baseline. That is a
  `\guard`, not the row of a summary table.
- **Honesty still holds.** "Good" = Pareto-dominant (fewer steps AND less time at equal success and constraints).
  Otherwise say "trade-off". MeanFM on D3IL-avoiding trades one episode in thirty (0.967), and that is said.

Recorded by Claude (Opus 5.5, Claude Code, v3) at the author's request, 2026-09-24.
