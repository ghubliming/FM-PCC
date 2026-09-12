# CLOSURE — the corridor-obstacle investigation (U11 → U12 → U13)

*Gen15 · 2026-09-12 · closes the `corridor_ball` line. Evidence: batches
`batch_uav_20260911_202307_TEMP`, `batch_uav_20260912_090334`, `batch_uav_20260912_201035`, and the
per-step rollout logs of jobs 25690 / 25702.*

## 0. The finding, in one measurement

Comparing the **executed** trajectories of `diffuser` (unprojected) and `dpcc-t` (projected) on the
**same trial seeds**, `corridor_ball_v3_2`, 10 rollouts:

| axis | max deviation over all 10 rollouts |
|---|---|
| lateral (y) | **0.0000 m** |
| vertical (z) | **0.0010 m** |
| longitudinal (x) | 0.28 – 1.03 m |

**The projector does not change the path. It changes only the speed along it.** Zero sideways, one
millimetre vertically, against a detour requirement of 0.32 m.

## 1. The "violation reduction" was an artefact

Violations are counted per step, and the projected episode is shorter:

| corridor_ball_v3_2 (mf K=2) | violations | steps | **viol / step** |
|---|---|---|---|
| `diffuser` | 19.60 | 271.7 | **0.0721** |
| `dpcc-t` | 18.20 | 254.1 | **0.0716** — **0.99×** |

The per-step rate is unchanged. The 19.60 → 18.20 drop is entirely explained by finishing 18 steps
sooner. **Nothing was avoided.** Earlier notes in U11 §4 and U12 §5 cited this reduction as evidence
the projector was "responding to the geometry" — that reading is **withdrawn**.

## 2. 🔴 But the projector is NOT broken — it demonstrably works elsewhere

The same test on `pillars` K=5, af:

| variant | violations | steps | **viol / step** |
|---|---|---|---|
| `diffuser` | 197.70 | 634.0 | **0.3118** |
| `dpcc-c` | 22.90 | 391.8 | **0.0584** — **0.19×** |
| `dpcc-t` | 41.30 | 377.8 | 0.1093 — 0.35× |

A **5.4× reduction in rate**, and `collision_free` 0.000 → 0.900. Same code, same horizon H=8, same
constraint machinery, a *larger* keep-out. **On pillars the projector genuinely finds an avoidance.**

An earlier claim in this investigation — that the projector "only refines an already-avoidant plan
and never discovers a detour" — is **wrong and withdrawn**. This table refutes it.

## 3. What was ruled out, and at what cost

| hypothesis | test | result |
|---|---|---|
| ball too large | r = **0.35 → 0.12 → 0.05 → 0.01** (U11, U12, U13, U13.2) | ❌ `cfree` 0.000 at every radius |
| no vertical room | ceiling `ub[2]` **1.80 → 2.80**, slot 0.08 → 1.04 m | ❌ unchanged |
| ball off the flown path | centre moved to the **measured** path z = 1.13 (`phys_min_z == phys_final_z` on 100 rollouts) | ❌ unchanged |
| action-magnitude cap | `dpcc-t-bounds_free` (cap removed entirely), job 25683 | ❌ `cfree` 0.000, and **worse than no projection** (27.70 vs 27.10) |
| projection fraction | `diffusion_timestep_threshold` 0.5 → 1.0, job 25704 | ⏸ cancelled; the four above already settle it |

Four ball radii spanning 35×, an 11.6× larger escape slot, a correctly-placed obstacle, and an
uncapped action bound — all produced `cfree` = 0.000 and a 0 mm path change.

## 4. Why corridor is structurally different from pillars

Two independent walls, both specific to this scene:

**(a) No lateral room.** The corridor is 0.90 m wide; `planning_inflation.r_drone` = 0.31 gives the
planner \|y\| ≤ 0.14. **The minimum possible keep-out is 0.31** — larger than the total lateral slack
of 0.28 m — so a point obstacle at r = 0 is already unavoidable sideways. On `pillars` the same drone
has ±2.19 m of lateral freedom.

**(b) No lateral or vertical action authority.** `action_bounds='auto'` derives the projector's
per-step cap from the dataset's own range, and the corridor expert flies a straight, level line:

| scene | Δx | Δy | Δz |
|---|---|---|---|
| **corridor** | 4.4e-02 | **±2.2e-05** | **±2.2e-05** |
| pillars | 4.4e-02 | **±3.97e-02** | ±3.1e-05 |
| s_curve | 2.2e-02 | ±2.29e-02 | ±1.1e-05 |

Δy on pillars is **1800×** corridor's. Δz is degenerate on every scene — no UAV expert ever changes
altitude — so a *vertical* detour is unreachable corpus-wide, which is what made corridor's
walls-then-ceiling design a dead end from the start.

## 5. 🔴 The one thing (a) and (b) do not explain

`dpcc-t-bounds_free` removed the cap and still produced `cfree` 0.000. If (b) were the whole story,
the uncapped run should have moved. **It is not known whether it moved the path**, because only
`diffuser` and `dpcc-t` rollout logs were downloaded for `_hgb2`.

**Open item:** run the §0 path-comparison on
`…/Emf_K2_*_T0.5_u7hg/6/corridor_hgb2_*/dpcc-t-bounds_free/`.
*max \|Δy\| > 0* → the cap was binding and the room (a) is what stopped it.
*max \|Δy\| ≈ 0* → the cap was never the active limit, and the cause is in how the sphere row enters
the NLP — a code question, not a config one.

## 6. Three corrections to the record

1. **`PCC constraint=dynamics`** in the rollout logs reports nothing — it is a hardcoded default
   argument of `behavior_logger.step()` (line 100) that the caller never overrides. It does **not**
   mean only the dynamics constraint was enforced.
2. **The constraint-overview and foresight plots draw obstacles at the workspace mid-height**,
   not their own z — `cz_mid = (lb+ub)/2` at `eval_mix_uav.py:1157` and `eval_artifacts.py:507`.
   For `corridor_ball_v3_2` that is 1.55 against a true centre of 1.13: the ball is drawn **0.42 m
   too high**, which is why it appears to float above the trajectory. A reviewed one-line fix exists
   (`temp/geo_demo/fixed/`) and is **not applied**. This bug misled this investigation repeatedly.
3. The violation-reduction and "projector responds to geometry" readings of §1 and §2 are withdrawn
   as noted above.

## 7. Status

**`corridor_ball` v1/v2/v3/v3.2 are closed.** No further ball geometry should be built: the radius
axis is exhausted (35× range, no effect), and (a) makes lateral avoidance impossible at any radius.

Successor design: [`../U14/`](../U14/) — an obstacle placed in the corridor's **open approach region**
(\|x\| > 2.0, where the wall halfspaces are `x_active`-inactive and y is unbounded), which removes
blocker (a) and leaves (b) as the single isolated variable.
