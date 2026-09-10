# Outlook — a perception front-end that writes the constraint set

**Created:** 2026-09-10 · **Type:** outlook, idea + feasibility check · **Status:** 🟡 **nothing built; the interface is known, the stack is not**
**Thesis home:** `sec:conc:future` (primary) · one sentence in `sec:disc:limitations` (where the geometry comes from today)
**Companion:** [`OUTLOOK_20260910_zero_shot_constraint_retasking.md`](OUTLOOK_20260910_zero_shot_constraint_retasking.md) — the deployment property this idea feeds
**Reads on:** [`Auxiliary/Methodology_Sources/AUX_constraint_geometry.md`](../../Auxiliary/Methodology_Sources/AUX_constraint_geometry.md),
[`AUX_rendering_and_gif_pipeline.md`](../../Auxiliary/Methodology_Sources/AUX_rendering_and_gif_pipeline.md)

---

## 1. The idea (author, 2026-09-10)

> Use an **additional tool** to recognise the real-world obstacle and interpret it into obstacle
> *shapes*, then feed those into the DPCC/FM-PCC framework — an ImageNet-pretrained model, refined
> on obstacle-specific training, so it recognises the obstacle shape and turns it into the **number
> data** the DPCC framework consumes. The input could be an image, but **LiDAR is also a good
> choice, a depth camera too — maybe RGB+D is the best choice.** And it could be combined with
> **SLAM**, to do the mapping at the same time.

The sentence that matters is *"into the number data"*. The projector does not want a feature vector.
It wants **geometry**: planes, spheres, cylinders, a box, with metric centres and radii, in the
planner's frame. That is a much narrower and much more checkable output than "perception".

## 2. Why this fits *this* architecture in particular

FM-PCC already has one place where pixels enter — the visual-conditioning path. This idea proposes a
second, and the two are structurally different:

| | **A — conditioning path** (exists, Gen14) | **B — constraint path** (this idea) |
| :-- | :-- | :-- |
| where pixels enter | encoder → latent $\vect{e}$ → U-Net input (`thesis_v2.tex:1211,1257`) | perception → constraint list → projector |
| what the downstream consumer reads | a learned latent | metric numbers: plane rows, sphere centres/radii |
| needs demonstrations in that modality | **yes** — the policy must be trained on it | **no** |
| needs the generative model retrained | **yes** | **no** — the model never sees a constraint |
| failure mode | degraded plan quality | a wrong *feasible set* — silent, and it can mean collision |

**B is cheap because of the dual-path design, and that is the whole argument for it.** The generative
brain is untouched; only the physical brakes get a new source for their geometry. A thesis sentence
can be built on that asymmetry: adding a perception-driven constraint source is an *integration*
task, not a learning task — which is precisely what a projector-based architecture buys you, and
what an end-to-end constrained policy would not.

## 3. The interface a perception module has to hit

This is fixed by the existing code and is the useful half of this note — anyone building the front
end must emit exactly these objects. Verified in the UAV stack (`mix_uav_test/eval_mix_uav.py:1193`,
`flow_matcher_v3_uav/utils/constraints_helpers.py`, `flow_matcher_v3_uav/sampling/projection.py`):

| family | what the projector consumes | what perception must therefore produce |
| :-- | :-- | :-- |
| `halfspace` | two points + a feasible side, `[[x1,y1],[x2,y2],'above'\|'below']`, optionally `{x_active: [lo,hi]}`; turned into one linear row $C^\top x \le d$ (`constraints_helpers.py:4`) | a **plane**: a supporting point pair (axis-aligned walls included — the degenerate branches exist), which side is free, and optionally the arc-length interval over which it applies |
| `obstacles` | `('sphere_outside', dims, center, radius)` → the quadratic $s^\top P s + q^\top s \le v$ (`projection.py:494` and its `build_matrices` docstring) | a **primitive with metric extent**: centre in world coordinates and a radius; cylinders are the same object with a reduced `dims` tuple |
| `geo_bounds` | a workspace box `lb`/`ub` on the actual-position dims, shrunk inward by the margin (`eval_mix_uav.py:~1253`) | the **navigable volume**, i.e. the free-space envelope, not the obstacles |
| inflation | `r_drone + margin_base`, with `planning_inflation` overriding it for the projector only (`eval_mix_uav.py:1236–1242`) | 🔴 **the place detection uncertainty goes.** See §7 |

Three consequences worth writing down now:

1. **The output is a small, structured list — not a map.** Six spheres and four planes is a typical
   scene. A perception stack that emits a dense occupancy grid has not yet solved the problem; the
   primitive-fitting step *is* the problem.
2. **Everything is in the planner's world frame**, and the quadratic rows are additionally built
   through the dataset normaliser (`ObstacleConstraints.build_matrices` folds `normalizer.mins/maxs`
   into $P$, $q$, $v$). Perceived geometry that falls outside the training data's normalisation range
   is not rejected — it is silently mis-scaled. This is a real integration hazard, not a detail.
3. **The feasible side is part of the output.** A detector that finds a wall but not which side of it
   the drone may occupy has produced nothing usable.

## 4. Sensor choice — what each one actually buys

The author's list is right, and the ordering is defensible; here is the reason for each entry.

| sensor | gives | costs / breaks on |
| :-- | :-- | :-- |
| **monocular RGB** | semantics (what the obstacle *is*), cheap, light | **no metric scale** — needs learned depth or motion parallax; scale error maps straight into a wrong radius, i.e. a wrong feasible set |
| **stereo / depth camera** | metric depth, dense, cheap | short range, poor on textureless or reflective surfaces, active-IR units fail in sunlight |
| **RGB-D** | semantics *and* metric geometry from one registered frame — colour segments the object, depth sizes it | same range/lighting limits as depth; registration and time-sync errors become geometry errors |
| **LiDAR** | metric, long range, light-independent — the natural input for plane/cylinder fitting | sparse (small objects, thin structures), heavier and costlier on a small airframe, no semantics on its own |
| **RGB-D + LiDAR** | LiDAR for the geometry, camera for the labels | integration and calibration burden; only worth it if labels change the constraint (e.g. "person" gets a larger margin) |

**RGB-D as the default first target is the right call**, for a reason specific to this system: the
constraint list needs *both* an extent (depth) and a decision about *which* things become obstacles
(semantics), and RGB-D delivers both in one registered frame.

## 5. The model — what "ImageNet-pretrained, refined on obstacles" has to become

An ImageNet-pretrained backbone is the right starting point, but the deliverable is **not a
classifier**. It is a *shape-and-pose* estimator. Three candidate stacks, cheapest first:

| # | stack | output | why start here / why not |
| :-- | :-- | :-- | :-- |
| **P1** | pretrained 2-D detector or instance segmenter (ImageNet/COCO init, fine-tuned on the obstacle classes) **+** depth → back-project the mask → fit a sphere/cylinder/plane | primitives | Cheapest, reuses off-the-shelf weights, and the fine-tuning set is small because the class list is small. Fitting quality depends entirely on depth quality at the mask edges. |
| **P2** | point cloud (LiDAR or depth) → RANSAC plane/cylinder/sphere fitting, no learning at all | primitives | **The honest baseline, and it should be run first.** If classical fitting is enough in the target scenes, the learned stack has to justify itself against it. |
| **P3** | learned 3-D box / primitive detection end-to-end | primitives | Strongest and most expensive; only worth it if P1/P2 measurably fail. |

Whichever is used, the fine-tuning target must be *geometry error* (centre and radius in metres,
plane normal in degrees), not detection mAP. **mAP is not the metric this system cares about**, and
optimising it would optimise the wrong thing — see §8.

## 6. SLAM — why it is not optional in the end

Two reasons specific to this planner, both worth one sentence in the conclusion:

- **The plan is longer than the field of view.** The planner emits a horizon-$H$ trajectory, and the
  projector constrains *every* state on it. Obstacles that have left the camera frame must still be
  in the constraint list, or the plan is projected onto a feasible set that has forgotten them.
- **A map is a constraint store.** SLAM turns per-frame detections into persistent, de-duplicated
  primitives with a pose — which is exactly the constraint list, maintained over time.

The risk to state alongside it: **map drift is constraint drift.** A pose error of a few centimetres
moves every plane and sphere by that amount, and the projector will faithfully enforce the wrong
geometry. Drift therefore has to be budgeted into the same margin as detection error (§7).

## 7. Where the uncertainty goes — the one design decision this project already made for us

The repo has already separated **the planner's safety pad** from **the body radius used to score
collisions** (`planning_inflation` vs `inflation`, `eval_mix_uav.py:1236–1242`), introduced in Gen15
U7 precisely so that loosening the planner's tube does not loosen the yardstick.

That split is the natural home for perception uncertainty:

```
planning_inflation = r_body + margin_base + margin_perception(detector recall, depth σ, SLAM drift)
inflation          = r_body                       ← unchanged; still scores collisions honestly
```

Keeping `inflation` fixed is what stops a noisy detector from being compensated by quietly grading
its own homework. **Write it this way or the evaluation is meaningless** — it is the same failure the
honest-geometry work already caught once.

## 8. The first experiment — and it needs no perception model at all

Before any detector is trained, the question worth answering is: **how much perception error does
the projector tolerate before success-and-constraints collapses?** That is measurable today, in
simulation, with ground truth on hand.

**Protocol (run on cluster):** take an existing UAV scene, and instead of passing the true geometry
to `setup_dpcc_projector`, pass a **perturbed** copy — then sweep the perturbation:

| perturbation | models | sweep |
| :-- | :-- | :-- |
| centre offset | localisation / SLAM drift | 0 → ~2× tracking error |
| radius error, both signs | depth scale error; under-estimate is the dangerous sign | ±% of true radius |
| **dropout of one primitive** | a missed detection | per-obstacle |
| spurious primitive | a false positive | 1–2 phantoms |
| one-step-stale geometry | perception latency | 1–3 replans |

Report S&C and constraint violation against each axis. The output is a **tolerance budget** — a
specification any future detector must meet, stated in metres — and it is a legitimate result on its
own even if no perception stack is ever built. It also costs nothing new: it is a config-level
perturbation in front of an existing eval path.

**Kill condition:** if S&C is already destroyed by a centre offset well below what any real sensor
achieves, the constraint-from-perception route is not viable for these scenes at this tracking
error, and the conclusion should say so instead of recommending it.

## 9. What the repo already provides, if this is ever built

| existing piece | why it matters here | where |
| :-- | :-- | :-- |
| **per-replan projector rebuild** — `policy.projector = rebuild_projector(float(p[0]))` inside the rollout loop | 🔴 **The hook already exists.** It was built for `s_curve`'s segment-switched halfspaces, but it is exactly the shape a perception front-end needs: swap the whole feasible set between replans. Generalising it from a scalar `current_x` to a perceived geometry is a small, well-defined change. | `mix_uav_test/eval_mix_uav.py:1554–1555`, closure at `:1968` |
| **offscreen rendering + state injection** ("replay without re-flying") | a synthetic RGB/RGB-D training and evaluation set for the detector is nearly free from scenes that already exist, with ground-truth geometry attached | `AUX_rendering_and_gif_pipeline.md` |
| **exec-time violation metrics + slack-in-metres feasibility gate** | the scoring side is already honest about margins, so a perception experiment can be graded without new metrics | `AUX_constraint_geometry.md` §3 |
| **the constraint families themselves** | no solver work: `SafetyConstraints` / `ObstacleConstraints` / `DynamicConstraints` already accept every primitive a detector would emit | `flow_matcher_v3_uav/sampling/projection.py` |

## 10. Risks to state plainly

1. **A missed obstacle is not a soft error.** Every other error in this system degrades a metric; a
   false negative removes a constraint and the projector will happily route through it.
2. **The NLP grows with the scene.** Six spheres is not fifty. SLSQP cost and conditioning both scale
   with row count, and this feasible set has already shown a solver-conditioning failure (the
   τ = 0.850 non-convergence, `AUX_constraint_geometry.md` §4). Timing and reliability would have to
   be re-measured, not extrapolated.
3. **Latency compounds.** Perception sits in front of a loop whose timing is already a
   data-rate/cluster artefact — and per the project's standing rule, `budget_ms` / 33 Hz is **not** a
   real-time pass/fail criterion, so no claim about onboard feasibility can be made from these
   numbers either way.
4. **None of this is validated.** Nothing above has been run. It is an interface analysis and an
   experiment proposal.

---

**Status:** idea recorded, interface verified against code, first experiment specified. Nothing
implemented. Do not cite as a contribution; cite only as future work.
