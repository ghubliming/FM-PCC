# INSIGHTS — the findings inside each result section, and where each conclusion uses them (2026-09-24)

**Asked (author):** "also inside each scene we have special insight/conclusion like the diffu K2 time discuss in the
d3il-avoiding, I remember it, it is good and insightful, plz gather our insights! and really apply into the
conclusions."

This file is the companion of [`GUIDE_20260924_results_storyline_author.md`](GUIDE_20260924_results_storyline_author.md).
The guide says what each environment's conclusion must say first; this file lists the insights the conclusions draw on.
- Every insight names the Ch 6 section it comes from, by label.
- "Applied" says where it now stands (v3.90).
- The numbers are the sections' own; nothing here is new data.

## D3IL-avoiding (§6.1)

| # | Insight | Source | Applied |
| :-- | :-- | :-- | :-- |
| I1 | **The central claim.** MeanFM and CI-MeanFM at K1 Pareto-dominate DPCC's diffusion model at its protocol, with its own 4.0 M U-Net. CI-MeanFM: 1.000 in 59.2 steps at 18.1 ms, against 1.000 in 70.1 at 553.4 ms (1/30 of the time). MeanFM: 0.967, one episode in thirty. | `tab:avoiding-dpcc-protocol`, `tab:avoiding-conclusion` | §6.1.4 ¶1; §6.4.1; §6.4.3 |
| I2 | **Not an artefact of the baseline's 20 steps.** Retrained at K2 the baseline also reaches 1.000, in 61.2 steps at 211.4 ms. Against that configuration the saving is **12×**, not 30×. | `sec:res:avoiding:dpccprotocol` | §6.1.4 ¶2; §6.4.1 table |
| I3 | **A projection costs what the projector is handed.** At K2 the diffusion baseline projects a plan one denoising step from noise: about 45 ms per candidate, against 2–6 ms for a nearly denoised plan. Its K2 time is therefore 6× its K1 time. Flow models at K ≤ 2 project only the terminal step. The same law, from the other side: endpoint projection at K3 **halves** the time of the average-velocity models, because the predicted endpoint is near the feasible set. | `sec:res:avoiding:caveat:noise`, `sec:res:avoiding:projectors` | §6.1.4 ¶2 and the endpoint paragraph; §6.4.3 (cross-cutting) |
| I4 | **The frontier is a region, not a point.** Within one or two evaluations of a flow model, the remaining choices trade a control step against a millisecond. | `sec:res:avoiding:pareto` | §6.1.4 ¶2 |
| I5 | **Temporal consistency is the flow models' rule.** Cumulative projection cost, the baseline's best rule, occasionally selects a stalled plan for them (MeanFM 72.4 against 58.6 steps). | `sec:res:ablations` | §6.1.4 ¶2 |
| I6 | **FM at K1 dominates the baseline too.** So much of the saving is the flow sampler at a small budget. MF/AF add the fewest steps of the benchmark: about 8 fewer than FM for about 1 ms more. | `tab:avoiding-dpcc-protocol` | §6.1.4 ¶2; §6.4.1 table |
| I7 | At K2, endpoint projection gives bit-identical rollouts at 2.45× the cost, because there is no guiding step. | `sec:res:constraints:degenerate` | not in the conclusion; its endpoint paragraph says "no guiding step" |
| I8 | Solving the four candidate programs in parallel would take MeanFM at K1 from 18.1 to about 11.7 ms (a prediction). | `sec:res:ablations` | not applied (a prediction) |

## D3IL-aligning (§6.2)

| # | Insight | Source | Applied |
| :-- | :-- | :-- | :-- |
| A1 | MeanFM under per-step projection dominates the baseline's best projected row: 9 violation-free contexts against 4, 0.179 against 0.462 m, 266 against 809 ms. | `sec:res:aligning:projection:models` | §6.2.4 ¶1; §6.4 |
| A2 | Every non-dominated unprojected configuration is MeanFM. At K2, at 1/7–1/10 of the others' K20 cost, it has already closed two fifths of the distance. | `sec:res:aligning:budget` | §6.2.4 ¶1; §6.4.1 table |
| A3 | **CI-MeanFM's stop-gradient target:** part of its own error enters the field it learns and does not shrink with smaller steps. A task that rewards more evaluations exposes this (aligning: 60 % only at K100). A task saturated at K1 (avoiding) hides it. | `sec:res:aligning:consistency` | §6.2.4 ¶1; §6.4.3 (cross-cutting) |
| A4 | **Endpoint projection is better on the constraint** (10/10; 7 of 9 comparisons). It is the reverse of avoiding, and the reason is the budget (guiding steps at K10/K20), not the task. It does not dominate: the advantage is on the constraint, at a comparable price. | `sec:res:aligning:projection:models` | §6.2.4 ¶2; §6.4.2/§6.4.3 |
| A5 | **The task's trade:** zero violation costs completion. The cheap budget (K2, 43 ms, 10/10 violation-free) is not the budget that solves the task (K20). | `sec:res:aligning:projection` | §6.2.4 ¶2 |
| A6 | The activation threshold is compressed as the budget grows (η 0.2 at K20, 0.4 at K10: three guiding steps). | `sec:res:aligning:projection` | not applied (method detail) |
| A7 | Under projection the baseline's violations change family: action-bound steps rise from 64 to 145–408. | `sec:res:aligning:projection:models` | not applied (detail) |

## UAV-pillars (§6.3.1)

| # | Insight | Source | Applied |
| :-- | :-- | :-- | :-- |
| P1 | **Reaffirmation.** The declared constraints transfer exactly: 0 violating steps after projection, the same steps and time. MeanFM and CI-MeanFM keep their order and gap. | `sec:res:uav:models`, `tab:uav-pillars-raw` | §6.3.1.3 ¶1; §6.4 |
| P2 | The quadrotor tracks more closely than the arm: 0.007 against 0.02–0.04 table units. | `DA_20260923_pillars_v2_live.md` §2 | §6.3.1.3 ¶2 |
| P3 | **Collisions:** undeclared pillars are skimmed at the rod's radius, which the scale makes the rotor reach; 50 of 51 lie on the commanded path. A collision is catastrophic for a quadrotor and ends the flight. | `sec:res:uav:pillars` (after Table 6.10) | §6.3.1.3 ¶2; §6.4 |

## UAV-corridor (§6.3.2)

| # | Insight | Source | Applied |
| :-- | :-- | :-- | :-- |
| C1 | **The DPCC concept works in 3-D.** The flights descend (540/560) and climb (560/560); violating steps go from 29–99 to 0–10. | `sec:res:uav:projected` | §6.3.2.4 ¶1; §6.4 |
| C2 | **Expert data:** on straight lines the model does not matter; the budget does (within 1.6 violating steps at a shared budget). | `tab:uav-corridor-raw`, `tab:uav-corridor` | §6.3.2.4 ¶2; §6.4 |
| C3 | **The hand-over:** the tilt is enforced over its span at the commanded position, 0.5 m ahead. The residue is the vehicle catching up with a plan the constraint has already released. | `sec:res:uav:projected` (paths) | §6.3.2.4 ¶1 |
| C4 | **The apex:** the plans never rise above 1.30 m, the top of the demonstrations' range, while the apex needs 1.48 m. Per-step projection at K ≤ 2 stops short there. | `sec:res:uav:projected` (paths) | §6.3.2.4 ¶2; §6.4.3 (cross-cutting) |
| C5 | The baseline flies wider (0.37 against 0.19 m off-route) and longer (290/321 steps). | `sec:res:uav:projected` | §6.3.2.4 ¶2 |
| C6 | **RQ2: a trade-off, no clear dominance** (author, v3.96). At K20 endpoint projection has fewer violating steps under both constraints. Over the hump with one candidate it is also cheaper (0.2 against 1.8 at 291 against 355 ms) but its flights are ~30 control steps longer. Under the tilt it costs more time (417 against 360 ms). At K3–5 per-step leaves as many violating steps or fewer. | `sec:res:uav:projection` | §6.3.2.3; §6.3.2.4 ¶3; §6.4.2 |

## UAV-s-curve (§6.3.3) — final (R44 complete, v3.92)

| # | Insight | Source | Applied |
| :-- | :-- | :-- | :-- |
| S1 | **The controller matters** (author, v3.93). The same plans succeed or fail by the controller that flies them: 5/10 or 0/10 projected, 23 or 42 violating steps unprojected. On the configuration of record the cascaded geometric controller is the better one. MuJoCo MPC only keeps the unprojected flights upright and closer to the goal, and that gain does not survive projection. *(The v3.90–3.92 reading "stability is the controller's limit, MPC lifts it" is withdrawn.)* | `sec:res:uav:controller` (DA R44bc §3) | §6.3.3.3; §6.3.3.4; §6.4 |
| S2 | **The controller's price, on the selected configuration with ten flights against ten:** 125.4 against 5.2 ms per control step (×24). That is 12× the planner and ~7× avoiding's entire action. The pilot showed the same split (125.8 against 6.5). | `tab:uav-controller-cost` (DA R44bc §8.3) | §6.3.3.3; §6.3.3.4 |
| S3 | On single-route data, MF/AF are not ahead at K1 (FM crosses 9/10, CI-MeanFM 6, MeanFM 3), and crossings fall with K. | `tab:uav-scurve` | §6.3.3.4 ¶3 |
| S4 | **The second limit is the plan's: the corner cut.** Every commanded path cuts the second inside corner of the crossover, 8–19 cm inside its keep-out, where the demonstrations clear it by 0.121 m. The cascaded controller follows the command to within ~1 cm and never comes more than 1.4 cm closer to the corner. The 0.30–0.33 m is the setpoint's lead along the path. The plans cross the gap ~0.2 m late: their commanded paths cross y = 0 at x = 0.21–0.27 m, where the demonstrations' Z-route crosses at x = 0, 0.5 m from both corners (DA §8b; v3.92). | DA R44bc §4, §6 | §6.3.3.1 (after Table 6.14); §6.3.3.4 |
| S5 | **Neither remedy removes the cut as configured.** MuJoCo MPC follows the command less closely (5 cm mean, up to 29 cm) and violates more (41.5 against 23.0 steps). Per-step projection binds the measured position (DPCC's binding), not the setpoint; the corridor binds both. The cut stays 8–9 cm deep, crossings fall from 9 to 5 under every rule, and it costs 126–155 ms per step. | DA R44bc §2–§4, §7; `sec:setup:protocol:uav` | §6.3.3.2, §6.3.3.3, §6.3.3.4 |
| S6 | **The rescue depends on the plan.** Under per-step projection, MuJoCo MPC loses all ten flights: seven invert 15.7–20.7 s in, against the outer wall of the second straight, and three exceed the contact limit. The cascaded controller brings 5/10 across. No flight of the forty in Table 6.16 is violation-free. | DA R44bc §3 (C1, job 26204) | §6.3.3.3; §6.3.3.4; §6.4 |
| S7 | **Lower plan quality helps here** (author's reading, v3.93). The low-K plans are rough (candidates scatter and zig-zag) and the K20 plans are smooth (`fig:uav-scurve-plans`). Yet every flow model crosses most at K1 (FM 9→7→6). A rough plan re-drawn every step varies more, which appears to help the controller through the turns. It does not save MeanFM at the second corner. This is one flight per panel; the counts carry the claim. | `fig:uav-scurve-plans` (extract/scurve_r44_plans.py) | §6.3.3.1; §6.3.3.4 |
| S8 | **FM leads because the data are simple** (author, v3.93): one route, so the few-step objectives have nothing to add. **Caveat:** where flights are easy to lose, more K is not always better (crossings fall with K for every model). The half-scale noise guard stays. | `tab:uav-scurve` | §6.3.3.4; §6.4 |

## Cross-cutting (§6.4.3)

1. **The price of a projection is set by what the projector is handed.** Source: I3.
2. **A task saturated at one evaluation hides what a task that rewards more evaluations exposes.** Source: A3.
3. **The demonstrations bound what a model can show.** Source: C2, C4.

Gathered by Claude (Opus 5.5, Claude Code, v3) at the author's request, 2026-09-24.
