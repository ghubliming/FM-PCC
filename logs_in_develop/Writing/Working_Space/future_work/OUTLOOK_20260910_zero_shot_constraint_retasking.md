# Outlook — train once, re-task at deployment: zero-shot constraint changes

**Created:** 2026-09-10 · **Type:** outlook — a property the architecture already has, and the experiment that would let us claim it
**Status:** 🟢 mechanism **verified in code** · 🟡 the *envelope* is unmeasured · 🔴 the honest-geometry finding already bounds it
**Thesis home:** `sec:conc:future`; the bound belongs in `sec:disc:limitations` whether or not the outlook is written
**Companion:** [`OUTLOOK_20260910_perception_to_constraints.md`](OUTLOOK_20260910_perception_to_constraints.md) — where the new geometry would come from in the field
**Reads on:** [`AUX_constraint_geometry.md`](../../Auxiliary/Methodology_Sources/AUX_constraint_geometry.md) §3 (honest geometry)

---

## 1. The idea (author, 2026-09-10)

> The real-world usage scenario for the whole DPCC/FM-PCC framework is: **train in the lab / a
> general environment**, and then, when something special happens — an obstacle appears, a halfspace
> or some volume is banned — it **still works zero-shot**, no retraining. A UAV trained from the
> building's CAD model and a virtual environment; then in real life a fire closes a room or a
> gallery, and it still flies. Or, in another case, the UAV is told to avoid some part of the
> environment.

## 2. Why the architecture already has this property — verified, not assumed

**Constraints never enter training.** They appear only in the sampling path:

| check | finding | evidence |
| :-- | :-- | :-- |
| does the training script know about constraints? | **no** — zero occurrences | `mix_uav_test/train_mix_uav.py` |
| does the loss? | **no** — `constraints=` appears only on `p_mean_variance` / `p_sample` / `p_sample_loop` | `mix_uav/models/diffusion.py:144,168,180`; `mf_diffusion.py:197` |
| when is the feasible set built? | at **evaluation**, from config, and handed to the policy | `mix_uav_test/eval_mix_uav.py:1193` → `:1952` |
| can it change *during* an episode? | **yes, already** — the projector is swapped between replans | `eval_mix_uav.py:1554–1555` |
| does the generative model see it? | **no** — the projector is a separate object passed alongside the model | `Policy(model=…, projector=projector, …)` |

So the deployment statement is a structural fact, not a hope:

> **A new restriction is a configuration change, not a training run.** No new demonstrations, no
> fine-tuning, no new weights — the same checkpoint is projected onto a different feasible set.

This is the deployment-facing version of the dual-path design, and it is arguably **the most
practically compelling thing the thesis can say**. It also inherits cleanly: the background chapter
already defines $\mathcal{Z}$ as the set of plans satisfying the *test-time* constraints
(`thesis_v2.tex:711`), and DPCC's own release speaks of "novel test-time constraints". FM-PCC keeps
that property while replacing the engine — worth one explicit sentence, because a reader may assume
the switch to flow matching costs it.

*(Arm C — in-loop trajectory optimisation — is not an exception. Its NLP also runs at sampling time;
nothing about the constraints reaches the loss.)*

## 3. The scenarios, written concretely

| scenario | what changes on the day | what has to be edited |
| :-- | :-- | :-- |
| **Building inspection, route learnt from CAD-derived sim** | a fire, a spill or maintenance closes a gallery or a room | one exclusion box / a halfspace pair added to the scene geometry |
| **Temporary obstruction** | a crane, scaffolding, a parked vehicle appears | one sphere/cylinder primitive |
| **Operator-declared keep-out volume** | a zone is declared off-limits — a restricted airspace, a cordon, an exclusion zone | a box or halfspace set, declared at launch |
| **Tightened clearance** | the same route flown with a larger safety margin (payload, wind, a nervous operator) | `enlarge_constraints` / the `-tightened` sibling — already an experimental factor in the corpus |
| **Same skill, new building** | the geometry is entirely new, the flying skill is not | a new scene geometry; **this is the case that needs the experiment in §5** |

The first four are variations of "same environment, different feasible set". The fifth is the
genuinely open one, and it should not be promised without evidence.

## 4. 🔴 The honest bound — this project has already measured how far re-tasking can go

Zero-shot re-tasking is not unlimited, and the limit is not speculative: **it is the honest-geometry
finding** (`AUX_constraint_geometry.md` §3).

The projector can only pick an admissible plan out of what the generator proposes. If the new
feasible set leaves less room than the policy's own tracking error, success-and-constraints is
bounded near zero *before any engine runs*:

| scene | slack left around the expert route | measured `track_err_mean` |
| :-- | :-- | :-- |
| `corridor` L/R | **0.000 m** | 0.30–0.49 m |
| `pillars` outer | 0.060 m | 0.30–0.49 m |
| `s_curve` | 0.120 m | 0.30–0.49 m |

— and the corpus observed exactly that (S&C 0/2876 on `pillars`). For `corridor`/`s_curve` the
deficit is physical, not synthetic: a 0.90 m gap against a 0.62 m drone.

**Stated as the rule that governs this whole outlook:**

> Re-tasking is free in *training* cost and bounded in *feasibility*. A new constraint set works
> zero-shot only while it leaves slack larger than the closed-loop tracking error — and while it
> does not exclude everything the demonstrations actually do. Otherwise the projector is asked to
> pull the plan off the data manifold, and success collapses rather than degrades.

This is the honest and, written properly, the *strong* form of the claim: the work does not just
propose test-time re-tasking, it can say **what breaks it and at what number**. Two quantities
therefore have to travel with any such claim: **slack (m)** and **tracking error (m)**.

## 5. The experiment that would turn this into a thesis claim

Two pieces, both buildable on existing machinery.

**(a) The re-tasking envelope.** Sweep the slack between the new constraint set and the demonstration
manifold; plot S&C against it. The apparatus already exists — `enlarge_constraints`, the
`-tightened` siblings, the `*_hg` honest-geometry entries and the slack-in-metres gate. The output is
one curve and one number: **the minimum slack, in multiples of tracking error, at which zero-shot
re-tasking still holds.** That number is the deployable form of the whole idea.

**(b) Cross-geometry transfer.** Train on pooled data (`uav-all` already pools all four scenes —
`mix_uav/datasets/d4rl.py:10,44–49`), then evaluate under a geometry the training data never
contained. This is the "trained in the lab, flown in the real building" test in miniature.

> ⚠️ **Engineering gap, name it before promising the experiment.** Evaluation currently keys the
> dataset — and therefore the **normaliser** — to the evaluated scene (`p.dataset = f'uav-{scene}'`,
> `eval_mix_uav.py:389`), and the constraint rows are built *through* that normaliser
> (`ObstacleConstraints.build_matrices`). Decoupling "which checkpoint" from "which geometry"
> requires the normaliser to travel with the checkpoint rather than be rebuilt from the eval scene.
> Small, but it is a code change, and a silent mis-scaling if skipped.

**Kill condition:** if the envelope turns out to be narrower than the tracking error achievable on
these scenes, the outlook is written as a *negative* — projector-based re-tasking needs a tracking
controller before it needs a perception stack. That is still a useful conclusion.

## 6. What may be written today, and what may not

| ✅ defensible now | ❌ not without §5 |
| :-- | :-- |
| "Constraints enter only at sampling time; a new restriction requires no retraining." | "The policy transfers zero-shot to unseen environments." |
| "The feasible set can be replaced between replans; the code already does this for segment-switched walls." | "The system re-plans around newly perceived obstacles." (needs the perception note, which is unbuilt) |
| "Re-tasking is bounded by the slack the new constraint set leaves against the tracking error — measured on our own scenes." | "Zero-shot re-tasking is robust." — the envelope is unmeasured |

## 7. The sim2real ladder behind the scenario

Recording the ordering so the conclusion does not present one rung as the whole ladder:

1. **CAD → simulated scenes** — done; the UAV scenes and the Skydio X2 asset are built this way
   (`AUX_uav_model_and_scenes.md`), which is exactly the "trained from the building CAD" premise.
2. **Constraint re-tasking within simulation** — §5(a); the missing measurement.
3. **Cross-geometry transfer within simulation** — §5(b).
4. **Tracking error as a first-class budget** — the honest-geometry finding says this, not the
   planner, is the binding constraint on how tight a re-tasked scene may be.
5. **Perception supplying the geometry** — the companion note.
6. **Onboard compute and latency** — untouched, and note that `budget_ms` / 33 Hz in the current
   results is a data-rate and cluster-latency artefact, never a real-time pass/fail criterion.
7. **Real flight, with a safety fallback** — out of scope for this thesis; say so.

Rungs 1 and 2 are within reach of the existing code. Everything from 5 on is future work in the
ordinary sense.

---

**Status:** mechanism verified in code (2026-09-10); envelope unmeasured; the bound is already
published inside this project's own methodology notes. Cite §2 as an architectural property; cite
§4 as a limitation; cite nothing else here as a result.
