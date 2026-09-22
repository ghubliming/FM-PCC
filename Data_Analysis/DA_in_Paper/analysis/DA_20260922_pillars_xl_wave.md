# DA — 2026-09-22 · UAV-pillars on the enlarged geometry (`pillars_xl`, Gen15 U17)

> # ⛔ Gen15 U17 ABANDONED 2026-09-22. This is the post-mortem, not an analysis of record for any thesis number. The thesis falls back to `pillars_hg`; see [`Gen15/U17/CLOSURE_20260922_U17_abandoned.md`](../../../logs_in_develop/Gen15/U17/CLOSURE_20260922_U17_abandoned.md).

**Corpus of record for this analysis**
`Data_Analysis/analysis_results_checkpoint/22-09-UAV-Pillars/batch_uav_20260922_113112`
— seed 6, ten flights per cell, geometry `pillars_xl_bounds+dynamics+geo_bounds+obstacles`, eval tag
`u7xl`. Recomputed with [`pillars_grid.py`](pillars_grid.py), repointed at this batch (its `pillars_hg`
setting is kept as a comment so the withheld tables stay reproducible).

Runbook: [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](../../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260919_pillars_enlarged.md)
· changelog: [`Gen15/U17`](../../../logs_in_develop/Gen15/U17/CHANGELOG_20260918_pillars_enlarged.md)

> 🔴 **Verdict.** The wave is complete and clean, and **it fills no thesis slot.** Under the thesis
> metric — S&C, the goal line passed on a collision-free flight — **every cell is 0.00**: all four
> models, every budget, per-step and endpoint, and the unprojected rows too. No flight of the 940
> evaluated on this geometry was collision-free. The section `sec:res:uav:pillars` stays withheld;
> its table and figure templates are **not** to be filled from this batch. What the batch does
> establish is *why*, and that is recorded below because it rules out one explanation and points at
> the projector.

---

## 1 · Coverage — what was evaluated

| model | checkpoint | $\nfe$ | per-step (7) | endpoint (4) | flights |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MeanFM (`mf`) | `MeanFlowODE_9D_dp0.5_bbunet` | 1, 2, 5 | ✅ ✅ ✅ | — — ✅ | 10 each |
| CI-MeanFM (`af`) | `AlphaFlowODE_9D_as1_ae0.2_bbunet`, `EPlatest` | 1, 2, 5 | ✅ ✅ ✅ | — — ✅ | 10 each |
| FM (`fm`) | `FlowMatchingODE_9D` (U-Net) | 1, 2, 5 | ✅ ✅ ✅ | — — ✅ | 10 each |
| Diffusion | `GaussianDiffusion_9D_K20` | 20 | `diffuser`, `dpcc-r` only | n/a by construction | 10 each |

94 cells × 10 flights = 940 flights. Endpoint projection (`hardflow_sls{,-r,-c,-t}`) exists at $\nfe=5$
only: at $\nfe\le2$ under $\eta=0.5$ it has no non-terminal step to act on and the eval drops it.
Diffusion's projected block stopped at the 24 h limit after one variant (18.6 h for ten flights at
10.4 s per control step); the remainder was closed by decision, not re-run — runbook §3d. Excluded from
every table: the verify cell (tag `u7xlchk`) and the 2-flight remnant of diffusion `dpcc-r-tightened`.

The unprojected rows reproduce their `pillars_hg` values exactly where the two overlap (mf and fm 0.90
strict success at $\nfe=5$), which pins every difference below on the constraint, not the models.

## 2 · The thesis metric — all zero

**S&C (`n_success_relaxed_and_constraints`) and the collision-free rate are 0.00 in all 94 cells.** The
grids are in the `pillars_grid.py` output; there is nothing to tabulate. The *mean violating steps per
flight* is the number that carries information:

| model | $\nfe$ | unproj. | $r$ | $c$ | $t$ | $r$-tight | $c$-tight | $t$-tight | endpoint single / $r$ / $c$ / $t$ |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| MeanFM | 1 | 213.8 | 243.1 | 219.5 | 251.4 | 234.2 | 229.6 | 253.3 | — |
| MeanFM | 2 | 196.1 | 237.5 | 221.9 | 242.3 | 247.3 | 225.8 | 244.1 | — |
| MeanFM | 5 | **68.7** | 321.8 | 339.8 | 347.1 | 419.4 | 461.8 | 368.1 | 249.9 / 251.0 / 233.6 / 253.1 |
| CI-MeanFM | 1 | 250.2 | 230.0 | 208.2 | 243.5 | 221.9 | 218.9 | 235.6 | — |
| CI-MeanFM | 2 | 216.6 | 214.6 | 213.2 | 220.1 | 224.9 | 211.5 | 226.0 | — |
| CI-MeanFM | 5 | 242.3 | 342.2 | 358.8 | 290.2 | 385.5 | 480.2 | 322.6 | 217.6 / 219.7 / 210.0 / 216.5 |
| FM | 1 | **113.2** | 235.5 | 234.8 | 252.6 | 237.8 | 237.4 | 240.3 | — |
| FM | 2 | 182.8 | 221.0 | 217.3 | 219.9 | 225.3 | 220.1 | 239.0 | — |
| FM | 5 | **102.0** | 226.9 | 266.9 | 266.4 | 310.7 | 258.8 | 264.3 | 237.6 / 237.4 / 227.7 / 239.1 |
| Diffusion | 20 | 230.2 | 268.4 | — | — | — | — | — | — |

Read across any row: **projection adds violating steps, it does not remove them.** At $\nfe=5$ the
per-step projector takes MeanFM from 68.7 to 322–462 and FM from 102.0 to 227–311; endpoint projection
takes both to ~230–250. The only rows that come close to obeying the constraint are the *unprojected*
flows at $\nfe=5$ — because they fly the demonstrated lane at $|y|=1.11$, 0.15 m inside the keep-out,
and clip each of the three pillar columns for ~20 steps.

## 3 · Where the projected flights go — the centre corridor

Each flight is classified from its own row of `per_rollout_detail.csv`: **demo-lane** (goal distance
< 0.40 m and mean penetration < 0.12 m — the demonstrated route at $|y|=1.11$, clipping each column by
~0.07 m); **centre-lane** (goal distance 0.80–1.15 m, contact-free, mean penetration > 0.28 m — flown
*between* the two pillar rows); **contact** (`phys_safe = 0`); other. Mean penetration is
`total_violations / n_violations`, the average depth inside the enforced 0.66 m keep-out per violating step.

| cell | demo | **centre** | **contact** | other | mean depth | goal dist (median) |
| :-- | --: | --: | --: | --: | --: | --: |
| MeanFM 5 unprojected | 9 | 1 | 0 | 0 | 0.14 m | 0.30 m |
| FM 5 unprojected | 10 | 0 | 0 | 0 | 0.06 m | 0.30 m |
| CI-MeanFM 5 unprojected | 0 | **9** | 0 | 1 | 0.31 m | 0.95 m |
| MeanFM 5 endpoint single / $r$ / $c$ / $t$ | 0 | **10 / 10 / 8 / 10** | 0 | 0/0/2/0 | 0.43–0.44 m | 0.99–1.03 m |
| FM 5 endpoint single / $r$ / $c$ / $t$ | 0 | **10 / 9 / 10 / 9** | 0 | 0/1/0/1 | 0.37–0.41 m | 0.93–0.96 m |
| CI-MeanFM 5 endpoint single / $r$ / $c$ / $t$ | 0 | 10 / 7 / 5 / 3 | 0 | 0/3/5/7 | 0.27–0.32 m | 0.79–0.83 m |
| MeanFM 1 per-step, six rules | 0 | 3–8 | 0 | 2–7 | 0.30–0.43 m | 0.78–1.07 m |
| FM 1 per-step, six rules | 0 | 6–9 | 0 | 1–4 | 0.29–0.35 m | 0.83–0.92 m |
| MeanFM 5 per-step $r$ / $r$-t / $c$ / $c$-t | 1/0/0/0 | 0/0/0/2 | **8 / 9 / 10 / 8** | 1/1/0/0 | 0.11–0.16 m | 1.40–5.47 m |
| FM 5 per-step $r$ / $r$-t | 0 | 0 | **10 / 10** | 0 | 0.19–0.21 m | 1.56–2.70 m |
| MeanFM 5 per-step $t$ / $t$-t | 0 | 6 / 6 | 4 / 4 | 0 | 0.24 m | 0.88–1.01 m |
| Diffusion 20 `dpcc-r` | 0 | 8 | 0 | 2 | 0.32 m | 0.85 m |

The full grid is printed by `pillars_grid.py` (`lane classification` block). Three things follow.

**The projected plans are coherent, and they all go to the same place.** The two pillar rows sit at
$y=\pm0.6$; the corridor between them is physically 0.96 m wide against a 0.62 m vehicle, so it is an
open lane in MuJoCo. Under the enlarged constraint it is *closed* — the 0.66 m keep-out discs overlap
across the centre line by 6 cm on each side — and closing it was the point of `pillars_xl`. Every
endpoint-projected flight of MeanFM and FM (38 of 40) and most of CI-MeanFM's fly exactly that lane:
contact-free, across the finish line, 0.9–1.0 m to the side of the goal point at $y=\pm1.11$, with a
mean penetration of 0.37–0.44 m — a drone centre 0.22–0.29 m from a pillar axis at each column,
0.10–0.17 m from the physical surface, inside the 0.31 m rotor reach. MuJoCo reports no contact on any
of them. Per-step projection at $\nfe\le2$ does the same, less cleanly. Per-step at $\nfe=5$ under the
$r$ and $c$ rules weaves between the lanes column by column and hits pillars on 8–10 flights of 10.

**The goal line is passed; the goal point is not.** `success_relaxed` (line crossed and contact-free) is
1.00 on every endpoint cell. A centre-lane flight ends ~1 m from a goal that sits at the end of the
outer lane, so strict success is 0. Under the thesis metric — the relaxed one — **these flights are
successes; what makes S&C zero is solely `collision_free = 0`.**

**CI-MeanFM at $\nfe=5$ takes the centre lane unprojected.** 9 of 10 of its raw flights are
centre-lane (0.31 m penetration, 0.95 m from the goal) — a property of that checkpoint on this scene,
not of any projector, and the reason its unprojected strict success was already 0.00.

## 4 · What this rules out, and what it establishes

**The goal-radius hypothesis of the first read (runbook §3e) is refuted.** It predicted near misses
just above 0.30 m on constraint-satisfying flights. The flights miss by ~1 m and violate on 35–45 % of
their steps. **So is the "non-converged iterates / garbage plans" reading** for the endpoint block: a
plan that puts 38 of 40 flights on the same lane, contact-free, across the line, is not garbage. It is
the projector resolving an infeasible lane by moving *inward* into the one region that is feasible
between the columns and only marginally infeasible at them, rather than 0.15 m *outward* into the
region the constraint actually leaves open. HardFlow's `[NLP-FAILURE] non-converged SLSQP at
tau=0.600` on every variant is consistent with that: at a column the centre lane is inside both discs
at once and the solver does not find its way out, so the last iterate is kept and it is the centre
lane. The per-step DPCC projector, solving column by column with a 7–8 step horizon, is pulled the
same way at $\nfe\le2$ and starts crossing between lanes at $\nfe=5$, which is where the contacts come
from. The top-down `constraint_overview.png` of any cell (which draws the *enforced* discs) laid over
the flown path would show this directly; the per-variant trajectory PNGs do **not** — see §7.

**What the scene has actually shown, then:** with the constraint enlarged so that the demonstrated
lane is 0.15 m infeasible, every projector in this study prefers the forbidden centre corridor to the
legal outer detour, and the local per-step one becomes unsafe at the higher budget. That is a real
statement about local projection against a keep-out that is only marginally closed, and it is the
same class of finding as U11–U13 on the corridor. It is not a ranking of the four generative models,
and it does not fill the section.

## 5 · Consequences

- **`tab:uav-pillars`, `tab:uav-pillars-projection`, `fig_uav_pillars_xl_paths` are not filled.** An
  all-zero grid is not a comparison. The section stays withheld; the draft rests on UAV-corridor.
- **`pillars_hg` remains withdrawn too** (it enforced the clearance its own demonstrations kept). The
  scene is now known to be degenerate in *both* directions: feasible-by-construction at $R=0.12$,
  unrepairable-by-the-projector at $R=0.35$. `pillars_xxl` ($R=0.55$) is off the table — harder is the
  wrong direction.
- **The diffusion baseline on this scene** is reported, if at all, from its raw output only, with the
  time argument (10.4 s per control step at $\nfe=20$, 18.6 h per projected variant, 24 h cap). Runbook
  §3d.
- **The timing column is real and usable** — `ms per control step`: $\nfe=1$ 122–265, $\nfe=2$ 96–162,
  $\nfe=5$ per-step 2,272–3,355, HardFlow at $\nfe=5$ 77–313, diffusion $\nfe=20$ 10,387 — but it
  describes a projector that failed, and the thesis should not quote it as a cost of something that
  worked.
- **If the scene is to be rescued** the lever is now specific: the centre corridor must be closed by
  more than the solver's tolerance, or removed from the feasible set altogether (a virtual wall along
  $y=0$ over the pillar span, in the corridor slide's halfspace form), so that the outer detour is the
  only route a local projector can settle into. That is a geometry change plus a re-run, and until it
  is done the scene ranks nothing. It is new work and not owed to the draft.

## 6 · The visual check, and why the trajectory PNGs look fine

The author inspected `…/Emf_K5_…_u7xl/6/pillars_xl_…/hardflow_sls-r/hardflow_sls-r.png` and read it as
a clear pass with a mild violation. Both impressions are explained by the classification above and by
what that picture draws. The per-variant trajectory PNG draws the **physical** scene —
`SCENE_OBSTACLES` from the demonstration generator, cylinders of radius 0.12 m
(`mix_uav_test/eval_artifacts.py:383`, `:796`) — while the projector and the scorer both enforce
0.35 + 0.31 = 0.66 m around the same centres. A centre-lane flight passes 0.10–0.17 m from the drawn
pillar surface and looks clear; it is 0.37–0.44 m inside the enforced keep-out. The plot that shows
the enforced surface is `constraint_overview.png` in the same folder (drawn at the true planning
margin, `plot_geo_constraints`), and the foresight SVGs draw both (`eval_artifacts.py:660–719`).

**"Score only the trajectory, not the full UAV size."** The scorer already scores only the trajectory —
the drone-centre point — against the obstacle inflated by the rotor reach, which is the standard
configuration-space form and is *identical* to checking the vehicle disc against the raw obstacle. It
is also exactly the set the projector was given (`planning_inflation: r_drone 0.31`), so plan and
score agree, as Chapter 5 states. Dropping the 0.31 m from the scorer would not be a scoring change
but a new, looser constraint set, and on this geometry it is self-defeating: with a keep-out of
0.35 m on the centre point, the demonstrated lane ($|y|=1.11$, 0.51 m from a pillar axis) and the
centre lane (0.45 m) are both legal, every flight in the batch becomes collision-free, and the scene
is trivially feasible again — the `pillars_hg` degeneracy with a different number — while the
projector would still be solving the 0.66 m set it was scored against. The violation counts stand.

## 7 · Regression check



The unprojected `mf` $\nfe=5$ row of this batch (strict success 0.90, 68.7 violating steps,
collision-free 0.00, `track_err` 0.446) reproduces the `u7xlchk` verify cell exactly; the same cell at
`pillars_hg` read 0.90 / 0 / 1.00. The batch loader flagged no pillars cell for circuit-breaker
sentinels or missing seeds (`data_quality.csv`); the only sub-10-flight cell is the diffusion remnant,
excluded above.
