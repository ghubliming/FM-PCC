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
| **D3IL-avoiding** (§6.1) | **The main proof.** Done. | The average-velocity models (MeanFM, CI-MeanFM) **Pareto-dominate the diffusion baseline of DPCC**, with its own 4.0 M temporal U-Net: the same success with constraint satisfaction, fewer control steps, about a thirtieth of the time per action; and no endpoint-projection (HF) configuration, which needs K ≥ 3, improves on the K1 per-step operating point (v3.98). |
| **D3IL-aligning** (§6.2) | **The proof again, plus endpoint projection (HardFlow).** Done. | Analytic average-velocity matching dominates the baseline on a vision-conditioned, contact-rich task, and **endpoint projection** is the projector that wins at the budget this task needs. |
| **UAV-pillars** (§6.3.1) | **A reaffirmation** of D3IL-avoiding on the quadrotor, plus the **plant and controller** point. | The Pareto-optimal operating points of D3IL-avoiding, flown again by the quadrotor to see what the plant changes (v3.97), keep their result: the declared constraints transfer exactly, and the models keep their order. What the vehicle and its controller change against the manipulator: closer tracking, and collisions that end the flight. |
| **UAV-corridor** (§6.3.2) | **Something new**, in two parts. | (1) **The whole DPCC concept carries over to our new environment**: projection repairs plans against constraints no demonstration satisfies, in three dimensions; endpoint and per-step projection are a trade-off there, with no clear dominance (v3.96). (2) **The expert-data issue**: on straight-line demonstrations the generative model does not matter; the budget does. |
| **UAV-s-curve** (§6.3.3) | **The controller** on extreme trajectories, discussed in depth. | **The controller matters** for the UAV in the FM-PCC framework: the same plans succeed or fail by the controller that flies them. On the configuration of record **our cascaded geometric controller is the better one**. MuJoCo MPC is **not** better: equal success unprojected, 0/10 against 5/10 projected, 24× the cost. FM leads because the data are simple (one route). A caveat: where flights are easy to lose, more K is not always better; the rougher low-K plans vary more, which appears to help the controller. *(Updated v3.93, author, 2026-09-24; the v3.90 wording "the tracking controller becomes the limit" is withdrawn.)* |

## The thesis in three steps (author, 2026-09-24, v3.97)

> "After all the main claim and test is ending at beat DPCC/HF on the avoiding benchmark, then extended on the V_A with
> more complex task and reaffirm it, then the UAV is majorly the extended controller transfer test."

1. **The main claim, made and tested on D3IL-avoiding**, the benchmark of DPCC, on which HardFlow was also evaluated:
   - the average-velocity models Pareto-dominate DPCC's diffusion model with its own U-Net;
   - no endpoint-projection configuration improves on their one-evaluation per-step operating point, because it needs
     K ≥ 3 to guide.
   - This is **"beat HF" in the thesis's own evaluation**, not against HardFlow's published numbers, which use another
     protocol.
2. **D3IL-aligning extends the claim** to a more complex, vision-conditioned, contact-rich task and reaffirms it.
   Endpoint projection helps there.
3. **The UAV scenes are mainly a transfer test** of the framework to a new plant and its tracking controller: where the
   result holds, and what the plant, the controller and the demonstrations change.

In the thesis: the opening paragraph of §6.4.3 (v3.97).

## The general conclusion (author, 2026-09-24, v3.96), for Ch 6.4 and Ch 8

> "thesis conclusion should say: in all the environments with expert data, like imitation learning, the MeanFlow family
> wins; for hard tasks the HF is helping; the UAV scenes reveal that it needs expert data, and when the data is simple we
> don't need MeanFlow models, FM is enough, all the models are equivalent as expected; and only in hard tasks like
> avoiding/pillars and aligning model + projection matters; and the controller matters, in s-curve and UAV-pillars."

**As written into §6.4.3 (v3.96), checked against the data:**
- **Human demonstrations: the average-velocity models lead.**
  - D3IL-avoiding: MF/AF Pareto-dominate DPCC, with its own U-Net.
  - D3IL-aligning: analytic average-velocity matching dominates.
  - UAV-pillars flies only the two average-velocity models, so it confirms transfer, not a model ranking.
- **Hard tasks: the model and the projection matter together.** Per-step projection at K1 completes D3IL-avoiding.
  Endpoint projection helps on D3IL-aligning (K20) as the better projector **on the constraint**, not a dominance.
- **Simple generated demonstrations: FM is enough.**
  - UAV-corridor: the three objectives cannot be told apart, as expected.
  - UAV-s-curve: FM even leads, under the half-scale-noise caveat.
- **UAV-corridor:** DPCC's projection scheme carries over to 3-D. There the two projectors are a **trade-off with no
  clear dominance** (author, v3.96). Endpoint projection over the hump at K20 has fewer violations at less time but
  flights ~30 steps longer.
- **The plant and its controller matter.**
  - UAV-pillars: a collision the constraints do not cover ends the flight.
  - UAV-s-curve: the same plans succeed or fail by the controller; our cascaded geometric controller is the better one.

## How to apply

- **Lead each conclusion with its line.** Numbers, trade-offs and caveats follow it; they never replace it.
- **Never write a summary that levels the flow models with the baseline on D3IL** ("≈ diffusion", "do not separate",
  "reach its success"). The D3IL result is a Pareto dominance. The v3.68d "≈" sentence was withdrawn for this at v3.89.
- **Scope limits stay, as caveats and not as conclusions.** For example, UAV-pillars did not fly the baseline. That is a
  `\guard`, not the row of a summary table.
- **Honesty still holds.** "Good" = Pareto-dominant (fewer steps AND less time at equal success and constraints).
  Otherwise say "trade-off". MeanFM on D3IL-avoiding trades one episode in thirty (0.967), and that is said.

Recorded by Claude (Opus 5.5, Claude Code, v3) at the author's request, 2026-09-24.
