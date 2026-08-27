# IDEA — **render the constraint**: make the forbidden region something the camera can actually see

*Filed under Gen16 U1 · build target: `mix_visual_avoiding/` ↔ `mix_visual_avoiding_test/` (Gen16 @ HEAD)*

**Date:** 2026-08-26 · **Type:** idea assessment / experiment design · **NO CODE WRITTEN**
**Status:** proposal — read §1 and §3 before scoping anything, and §7 S0 before spending any GPU time.

**Trigger:** user note, 2026-08-26 (quoted verbatim in §0).
**Companions:**
- [`SNAPSHOT_20260826_visual_avoiding_env_status.md` §8b](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260826_visual_avoiding_env_status.md) — the three-condition criterion this note is an answer to.
- [`IDEA_20260826_multi_scene_visual_uav.md`](../../Gen11/Epoch10_Visual_UAV/IDEA_20260826_multi_scene_visual_uav.md) — the *other* answer to §8b (pool the scenes). Same defect, different fix; §8 compares them.
- [`PLAN_Gen16_visual_avoiding_mix_ml.md`](../init/PLAN_Gen16_visual_avoiding_mix_ml.md) — the host frame.

---

## 0. The note

> *"So we should add the non-seen constraints INTO the env — in training there is NO such constraint,
> obstacles/half-space, only the perfect env. What will happen if the visual camera REALLY sees it?
> … thinking from the virtual case, the virtual restriction of a UAV: some space is forbidden, but not
> necessarily an obstacle. Like in a FIRE, some space is forbidden — how to fast reach the goal? We should
> let the obstacle be seen in the env. At least for UAV. Half-space no.
> Real-world usage: robot arm too — in an industrial env the half-space is forbidden to use. We sure can
> mark the obstacles, but let the vision do the identification, better?
> No — the obstacle is given as math in the projector. Half-space? is ok. Bounds is ok.
> But the obstacle: how to deal with its real-world meaning? It is still a mapping issue. …
> What will happen if the NN gets the never-seen object? Feels still not good — the NN never learned how to
> deal with the visual signal of a never-seen object, it won't be better…
> But still worth a try. … maybe some good interaction with the Projector? i.e. the raw NN outputs will be
> better? Worth a try — maybe in V_Aligning first? To test if the Projector's job can be done somehow by the
> visual NN, i.e. fewer steps to the goal. Is it a good finding? Maybe no. At least it is a study! …
> Can also DA-focus on how the visual feedback works near the obstacles."*

---

## 0.1 Verdict up front

| | |
|---|---|
| ✅ **Is the diagnosis right?** | **Yes, and it is the more interesting of the two available fixes.** [§8b](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260826_visual_avoiding_env_status.md) said *"make the scene vary"*; this note says *"make the constraint visible"*. Both satisfy the same criterion, but rendering is the one that creates a **new question** (can perception offload the projector?) instead of only repairing a defect. |
| ✅ **Is the premise right?** | **Yes. The constraint is drawn nowhere, in any task.** What the camera sees on avoiding is the **task** — D3IL's six posts, which are the benchmark itself, not the constraint set. The DPCC constraint set is a *separate* object that lives only in a yaml. §1. |
| 🔴 **But "the constraint" is four different things** | Only **two of the four families are spatial regions at all.** `dynamics` (a relation between consecutive waypoints) and `bounds` (an action-magnitude cap) **cannot be drawn** — there is no region to draw. That halves the scope before anything is built. §1.1. |
| ⚖️ **So which family?** | Two targets, and they are **different perceptual jobs**: the **half-space** = draw a forbidden region in empty space (nothing there today); the **obstacle sphere** = draw the **margin** around a post that is already visible. §2. The note's *"half-space no"* is worth revisiting — it is the larger, cleaner, and genuinely-unseen one. |
| 🔴 **A gap that is already in the repo, unlabelled** | On avoiding the enforced spheres are **not** the posts: centres shifted ~25 mm, radii inflated 2–3×, **third row absent entirely**. So "what is enforced" and "what is in the picture" have been decoupled since upstream DPCC — rendering the constraint is what would finally close that gap visually. §1.2. |
| ✅ **"The NN never learned the never-seen object, it won't be better"** | **Correct, and it kills the naive version outright.** Rendering at eval only = OOD input. It must be in training. §3.1. |
| 🔴 **And that is still not enough** | The demos were collected **without** the marker. Paint one onto existing data and you teach one of two wrong lessons — *"the marker means nothing"* or nothing at all. **Consistency between demo and marker is the hard requirement, not visibility.** §3.2. |
| ✅ **The cheap way to get consistency** | **Hindsight relabelling against avoiding's 24-modality.** The env already computes which gap each demo took (`mode_encoding = np.zeros(2+3+4)`). Sample a marker that blocks a route the demo *did not* take → consistent by construction, varies per episode, **zero new expert data**. §4. |
| ✅ **Is the tooling there?** | **Yes, and this was not obvious.** The replay-render collector exists and works (`collect_visual_avoiding_data.py`), a rendered-but-non-physical object is already precedented in the same scene file (`finish_line`, `visual_only=True`), and `MjScene._set_obj_pos_and_quat` explicitly repositions **static** bodies via `model.body_pos` — so per-episode marker placement needs no XML recompile and does not touch physics. §5. |
| ⚖️ **"Can the visual NN do the Projector's job?"** | **This is the real question in the note and it is a good one.** Formalised as three arms in §6 with a measurable answer: **pre-projection feasibility** and **projector intervention magnitude**, not "fewer steps to the goal". |
| ⚠️ **"Is it a good finding? Maybe no."** | **It is a finding either way**, and that is the strongest argument for running it. If intervention drops → visually-conditioned planners need less projection. If it does not → *perception cannot substitute for declared constraints*, which **supports** DPCC's premise. §6.3. |
| 🔴 **Cheapest decisive first move** | **Not training. Render 20 episodes and check the marker is legible at 96×96** — a translucent floor patch in a 30°-tilt cage view may be a handful of pixels. This gate can kill the whole idea for the price of one CPU job. §7 S0. |
| ✅ **"DA-focus near the obstacles"** | **Keep this — it is the right place to look**, and it should be the *primary* readout, not an extra. §6.2. |

---

## 1. What is enforced vs what is drawn

The constraint set and the scene are **two separate things** in this repo, and they always have been. The scene
comes from D3IL; the constraint set comes from DPCC's yaml. Nothing has ever drawn the second one.

🔴 **The six red posts in the avoiding scene are the TASK, not the constraint.** They are D3IL's benchmark —
`avoiding_objects.py` has exactly one commit in this repo's history (`e5d0291b`, *"D3IL folder added into
FMPCC"*) and is byte-identical to the pristine upstream at `/workspaces/aux_repo/d3il/`. Weaving through them
*is* the task, and the 24 routes they create are the multimodality D3IL was built to test. They are in the
pixels because they are the benchmark, not because anyone decided the policy should see a constraint.

### 1.1 Only two of the four constraint families can be drawn at all

`constraint_types: ['halfspace', 'obstacles', 'dynamics', 'bounds']`
(`config/visual_avoiding_mix_eval.yaml`). Before choosing what to render, note that half of them are not
regions:

| family | a spatial region? | in the pixels today | what "make it visible" would mean |
|---|---|---|---|
| **`halfspace`** | ✅ yes | 🔴 **nothing at all** — no physical counterpart anywhere in the scene | draw a forbidden region **in empty space** — pure addition |
| **`obstacles`** (spheres) | ✅ yes | ⚠️ posts are visible, but **the constraint is not the posts** (see below) | draw the **margin** on top of an object that is already there |
| **`dynamics`** | ❌ no — a relation between consecutive waypoints (`x[t+1] = x[t] + dt·a[t]`) | — | ❌ **not renderable** — there is no region |
| **`bounds`** | ❌ no — an action-magnitude cap | — | ❌ **not renderable** |

**So the scope of "render the constraint" is the top two rows.** That is not a limitation of the idea; it is
the idea's actual footprint, and it is worth stating before any sampler is written.

### 1.2 On avoiding, the enforced spheres are already not the posts

This decoupling exists today and is not written down anywhere:

| row | physical cylinder — the **task** (`avoiding_objects.py:8-65`) | radius | the **constraint** (`config/visual_avoiding_mix_eval.yaml`) |
|---|---|---|---|
| l1 | `(0.500, −0.10)` | 0.030 | ✅ `(0.5, −0.1)` r 0.06 **and** `(0.5, −0.09)` r 0.08 |
| l2 top | `(0.425, 0.08)` | 0.025 | ⚠️ approximated as `(0.4, 0.08)` r 0.06 / 0.08 |
| l2 bottom | `(0.575, 0.08)` | 0.025 | ⚠️ approximated as `(0.6, 0.08)` r 0.06 / 0.08 |
| l3 top | `(0.350, 0.26)` | 0.025 | ❌ **absent** |
| l3 mid | `(0.500, 0.26)` | 0.025 | ❌ **absent** |
| l3 bottom | `(0.650, 0.26)` | 0.025 | ❌ **absent** |

Centres shifted ~25 mm, radii inflated 2–3×, and **the entire third row is unconstrained**. This is
**inherited verbatim from upstream DPCC** (`/workspaces/aux_repo/dpcc/config/projection_eval.yaml:71-81` is
byte-identical, comments included), so it is DPCC's deliberate synthetic constraint set, not a Gen16 bug — do
not "fix" it, it would break every cross-generation table.

**Read positively, this is an argument *for* the idea:** the projector already keeps the arm out of bubbles
much larger than the posts, and out of nothing at all at the third row. Today the picture shows neither.
Rendering the constraint is what would make the picture agree with what is actually being enforced.

⚠️ **A real inconsistency to flag separately:** `config/avoiding-d3il-visual-mix.py:157-164`
(`_AVOIDING_OBSTACLES`, used as `constraint_list` at train time) lists **all six posts at their true centres,
r = 0.04** — a different set again. Train-time and eval-time disagree about what the obstacle constraint is.
That one is ours, not upstream's. Audit it before U1 numbers land (§9 R7).

### 1.3 Aligning — the constraints are invisible **and** placeholders

`config/visual_aligning_eval.yaml`, active entry `combined_5`:
- a `'\'`-shaped half-space from `(0.65, 0.45)` to `(0.80, −0.45)`, feasible side `below` — **nothing physical there**;
- a workspace box `lb [0.20,−0.45,0.02] / ub [0.80,0.45,0.50]`, explicitly *"relaxed for test purpose"* — **invisible**;
- one obstacle sphere at `(0.50, 0.00)` r 0.06, whose own comment says **`# tune to measured obstacle position`** — i.e. **`PLACEHOLDER` geometry with no object in the scene at all**.

Here there is no task-object confusion at all: the scene is a box and a target, and **every constraint is
maths**. This is the purest instance of the situation the note describes, which is why *"maybe in V_Aligning
first?"* is a reasonable instinct. §8 argues it is still the wrong task to *start* on, for a reason the note
does not anticipate.

### 1.4 UAV — the walls are real, the visual arm is not

`config/uav_projection.yaml:172-198` documents `pillars` as *"XML: 6 pillars, y=±0.6, x∈{−2,0,2}, radius 0.12,
full-height"* — here the constraint set **is** a faithful transcription of scene geometry. Fix_12 went further
and *deleted* the synthetic `y=±1.2` envelope half-spaces precisely because *"no wall exists in the XML"*. So
UAV is the one task where enforced and drawn already agree, and it is the model for what §1.2 would look like
if closed.

What UAV lacks is the camera: no `visual_*.py`, no `MultiImageObsEncoder`, no camera key anywhere in
`mix_uav/`. And the scene is constant within a run. Both are the subject of the companion UAV idea doc; this
note does not add to it beyond §8.

---

## 2. The two renderable targets

Per §1.1 there are exactly two, and they ask the network for different things.

| | **A — half-space region** | **B — obstacle margin ("bubble")** |
|---|---|---|
| what gets drawn | a translucent floor region on the forbidden side of a line | a translucent disc of the *enforced* radius around each constrained post, including the **missing l3 row** |
| in the pixels today | 🔴 **nothing** | ⚠️ the post is there; **the margin is not** |
| perceptual task | *"is there a forbidden region over there?"* | *"the thing you can see is bigger than it looks"* |
| legible at 96×96? | ✅ spans a large fraction of the workspace | ⚠️ r 0.06–0.08 vs a 0.025–0.03 post — a visible ring, but small |
| per-episode variation | ✅ **free** — sample the line continuously | ⚠️ requires moving posts (physics) **or** varying the margin only |
| demo-consistency (§3.2) | ✅ **free** via the 24-modality (§4.1) | ⚠️ harder — the enforced radius is what it is |
| eval axis already wired | ✅ `avoiding_halfspace_variants: ['top-right-hard','top-left-hard','both-hard']`, `n_trials: 30` | ❌ no |
| ever seen by the model in any form | ❌ **never** — eval-only geometry, absent from `constraint_list` at train time | partially (the post) |
| closest to the note's real-world case | the fire zone / no-fly volume | the marked keep-out ring around machinery |

🔴 **Recommendation: start with A.** It is the only family that is (a) invisible today, (b) large enough to
survive the encoder's input resolution, (c) free to vary per episode with consistent labels, and (d) already
wired into the eval as a variant axis. The note's *"half-space no"* is worth revisiting on exactly these
grounds — as a *rendering* target it is the strongest of the two, whatever its status as a maths constraint.

**B is the better long-term story** — a learned safety margin around a perceived object is closer to the
industrial keep-out case, and it directly attacks the §1.2 decoupling. It is the natural U2 once A has
answered whether the encoder can read a drawn region at all. Running both as an ablation is also legitimate if
the S0 gate (§7) says both are legible.

⚠️ §8b's warning applies to A's three named variants: three fixed layouts lets the network *classify* the
scene from a handful of pixels — technically perception, trivially so. **Sample the line continuously**
(§4.2); it costs nothing extra and it is the difference between "reads a 3-way label" and "reads geometry".
## 3. Why the naive version fails — the note's own objection, extended

### 3.1 Eval-only rendering is out-of-distribution (the note is right)

*"What will happen if the NN gets the never-seen object? … it won't be better."* Correct. A `visual_avoiding`
encoder trained on the clean scene, shown a scene with a novel coloured region at eval, produces an activation
pattern it has no mapping for. Best case: it is ignored. Realistic case: it perturbs the conditioning and
**degrades** the plan. There is no version of this experiment that does not re-render the training data.

### 3.2 Rendering in training is *also* not enough — the demos must agree with the marker

This is the failure mode the note does not reach, and it is the one that decides the design.

The avoiding demos were recorded in a scene with **no** marker. Paint one in afterwards and exactly one of
three things happens:

| where the marker lands | what the data teaches | verdict |
|---|---|---|
| **on the demonstrated path** | *"the expert drives straight through the forbidden zone"* → the marker is explicitly labelled irrelevant | 🔴 **actively harmful** — worse than no marker |
| **far from any demonstrated path** | the marker never correlates with any decision | ⚠️ **inert** — costs encoder capacity, buys nothing, and will read as a null result you cannot interpret |
| **same place every episode** | encoder memorises it, exactly as it can memorise the fixed post field today | ❌ **§8b condition #1 violated** — back to square one |

**So the requirement is not "make it visible". It is: the marker must vary per episode, must be consistent
with the trajectory shown in that episode, must be absent from the obs vector, and — for the headline arm —
must be absent from the projector.** That is §8b's three conditions plus a fourth (consistency) that only
appears once you are relabelling existing data rather than collecting new data.

Meeting all four without re-collecting expert demos is the whole trick, and avoiding happens to hand it to us.

---

## 4. The design that satisfies all four conditions for free

### 4.1 Exploit the 24-modality

D3IL avoiding is deliberately multi-modal: the arm passes l1 on the **left or right** (2), then one of **three**
l2 gaps, then one of **four** l3 gaps — `self.mode_encoding = np.zeros(2 + 3 + 4)`
(`avoiding.py:113`), maintained by `check_mode()` (`:173-201`), and the env already reports it every step
(`:171`, `return observation, reward, done, (self.mode_encoding, self.success)`). 2 × 3 × 4 = **24 routes**,
and the dataset is built to cover them — the env's own `mode_decoding` normalises its entropy by `log(24)`
(`avoiding.py:275-279`).

**Relabelling rule:** for each recorded demo, decode which route it took, then **sample a keep-out region that
blocks one or more routes it did not take**, and re-render that episode's frames with the region drawn.

This gives, by construction:

| condition | how it is met |
|---|---|
| **varies per episode** ✅ | the region is resampled per episode |
| **consistent with the demo** ✅ | it is *defined* to exclude the demonstrated route |
| **absent from obs** ✅ | obs stays 4-D `[des_xy, c_xy]` — untouched |
| **absent from the projector** ✅ | the region is not in `constraint_types` for the headline arm (§6, arm P1) |
| **cost** ✅ | **zero new expert demonstrations** — replay + re-render only |

And the payoff is real: because the demos span 24 routes, conditioning on the marker genuinely changes
`p(trajectory | image)`. The marker is not decoration — it is the variable that collapses a 24-mode mixture
the 4-D state cannot resolve. **That is the same argument the UAV note makes for pooled scenes (§3.0 there),
applied inside a single scene.**

### 4.2 Preferred instantiation — target A, continuously sampled

Concretely, per episode (target **A**, the half-space region — §2):
1. decode the demo's route from `mode_encoding`;
2. sample a line `(p1, p2, side)` in the family the eval already uses (`[[x1,y1],[x2,y2],'below']`), rejecting
   any sample the demo path violates by less than a margin `δ_margin`;
3. bias the sampler toward lines that are **tight** on the demo (small clearance) and that **do** exclude some
   alternative routes — a line the demo clears by 20 cm is as inert as no line at all;
4. draw the forbidden side as a translucent floor patch;
5. re-render the episode's `bp-cam` frames.

Step 3 is the one that determines whether the experiment has any power. Record the realised clearance
distribution — it is also the stratification axis for §6.2.

### 4.3 🔴 The honest limitation: no active detour in the data

Hindsight relabelling produces demos that are **compatible** with the marker, never demos that **swerved
because of** it. The model therefore learns *"do not be where the marker is"* correlationally, not causally.

Consequences to state in any write-up:
- generalisation to markers that would require a genuinely new detour is **not tested** by this design;
- the claim ceiling is *"the visual channel carries the constraint"*, **not** *"the policy plans around a
  perceived constraint it has never had to plan around"*;
- the mitigation is biasing toward tight, mode-excluding placements (§4.2 step 3), which makes the retained
  demos the *rarer* routes and forces the marker to carry real routing information — but it is a mitigation,
  not a fix.

The full fix is re-collecting expert data with the marker live. That is a much larger job and should only be
scoped **after** S0–S2 say the cheap version shows anything at all.

---

## 5. What already exists (this is cheaper than it looks)

| | asset | state |
|---|---|---|
| ✅ | **Replay + re-render collector** — `collect_visual_avoiding_data/collect_visual_avoiding_data.py`; replays each expert episode in MuJoCo (EGL offscreen), captures `bp-cam` + `inhand-cam` at 96×96 into `images/bp-cam/env_N/*.png` | **exists** — and Gen16's `sequence.py` reads exactly its output layout (`images/bp-cam/<ep>/*`) |
| ✅ | **Rendered-but-non-physical object precedent, in the same file** — `Box(name='finish_line', …, rgba=[0.,1.,0.,0.3], visual_only=True, static=True)` (`avoiding_objects.py:56-62`) | exists — the marker is this, moved |
| ✅ | **Runtime repositioning of `static` bodies** — `MjScene._set_obj_pos_and_quat` (`:255-277`) branches on `body_jnt_addr == -1` and writes `model.body_pos[body_id]` directly | **exists** — no XML recompile, no freejoint, no physics change |
| ✅ | **Runtime repositioning is already used in production** — aligning's `set_context` moves `aligning_box`/`target_box` per episode (`aligning.py:109-123`) | exists |
| ✅ | **Route labels** — `mode_encoding`, `check_mode()` | exists |
| ✅ | **Gen16 pipeline, 4 engines × 3 arms** | code complete, unverified on hardware |
| ❌ | Marker object + per-episode sampler + relabelling pass | **to build — this is the U1 work** |
| ❌ | A **separate** eval config carrying the marker geometry | to build — see §9 risk R6 |
| ❌ | Near-boundary stratified DA | to build — §6.2 |

**Read this as: the expensive half (a working replay-render pipeline on this exact task) is done.** The new
work is a marker object, a sampler, and a config fork.

---

## 6. The actual research question, made measurable

*"Can the Projector's job be done somehow by the visual NN?"* — this is the good question in the note. It
needs three arms and a metric that is not "steps to goal".

### 6.1 Arms

| arm | marker rendered? | marker in the projector? | what it isolates |
|---|---|---|---|
| **P2** *(= Gen16 today)* | ❌ | ✅ (the DPCC constraint set) | the incumbent baseline |
| **P0** | ✅ | ✅ | **declared + perceived** — does seeing a constraint help even when you are also told it? |
| **P1** | ✅ | ❌ | 🔴 **the interesting one — perception-only.** Can the generative planner keep itself out of a region nobody told it about? |

⚠️ **P1 is not a safety proposal.** DPCC's guarantee comes from the projection; removing the marker from the
projector removes the guarantee for that constraint by construction. P1 measures **what the network learned**,
and its violation rate is a *model* metric, not a deployment claim. Say so explicitly wherever it is reported.

### 6.2 Metrics — the readout is projector intervention, not task success

The note's *"the raw NN outputs will be better"* is the right target; *"fewer steps to the goal"* is the wrong
proxy (it moves with the task, the engine, K, and the seed). Measure the raw field directly:

| metric | definition | why |
|---|---|---|
| 🔴 **pre-projection feasibility** | fraction of **raw sampled waypoints** already satisfying the constraint set, before any solve | the direct answer to "is the unguided output better?" |
| 🔴 **intervention magnitude** | `‖x_proj − x_raw‖` per solve, summed and per-waypoint | how much work the projector still had to do |
| **NLP solve count / NFE** | already instrumented (arm-C step-budget work, `resolve_hf_batch_size` parity) | cost side of the trade |
| **success + constraint-satisfaction** | unchanged, nominal (δ=0) | the gate — these must be **held**, per the repo's Pareto rule |
| ⏱ `avg_time`, `proj_ms` | `RTRecorder` | the other Pareto axis |

**And stratify every one of them by distance to the nearest active constraint boundary** — the note's own
suggestion, and it should be the *primary* view rather than an appendix. Bands: `d < δ` (inside the tightening margin), `δ ≤ d < d_near`,
`d ≥ d_near` — with `d` the signed distance to the boundary, which is well defined for both targets (a
point-to-line distance for **A**, a radial distance for **B**). If a visual conditioning effect exists anywhere it exists in the near band, and a whole-rollout
average will dilute it into noise. Re-use the realised-clearance distribution from §4.2 step 3 as the axis.

### 6.3 What each outcome means

| result | claim |
|---|---|
| P0/P1 raise pre-projection feasibility at held success + constraint rate, and lower intervention | ✅ **"a visually-conditioned planner needs less projection"** — a Pareto win in the repo's own language, and it complements DPCC rather than competing with it |
| feasibility unchanged, cost unchanged | ⚖️ **"perception cannot substitute for declared constraints"** — a clean negative that *supports* DPCC's premise. Pair it with the snapshot's §4b.2 encoder cost (`+35 % per network call, +3.2 ms/NFE`) for the full price/benefit statement |
| P1 violates while P0 does not | the marker is read only as a *hint*, not as a constraint — bounds the claim, still publishable |
| everything degrades | suspect the marker is illegible (S0 gate) or the relabelling is inconsistent (§3.2) before believing the science |

**Every branch is reportable.** That answers *"Is it a good finding? Maybe no."* — the design is falsifiable in
a direction that is still worth writing down, which is the strongest reason to run it.

---

## 7. Staged scope for Gen16 U1

| stage | what | gate to pass before the next stage |
|---|---|---|
| 🔴 **S0** *(hours, CPU, no training)* | Add the marker object; render ~20 replayed episodes with sampled markers at **96×96**. Look at them. | **Is the marker legible?** Quantify: marker pixels ≥ some floor of the frame, **and** a linear probe on frozen encoder features recovers marker side/position above chance. **If this fails, stop — the idea is dead at this resolution** and the only continuations are a bigger input or a second camera. |
| **S1** *(CPU, hours)* | Route-decode the full demo set, implement the §4.2 sampler with the tightness bias, re-render the whole dataset into a **new** data root. Log the realised clearance distribution. | Distribution is not degenerate: markers are actually tight on a decent fraction of episodes, and every episode's demo is feasible w.r.t. its own marker. |
| **S2** *(GPU)* | Train `mf` first (cheapest — K = 2), then `fm`. Arms **P2 / P0 / P1**. Same seeds, same `n_trials = 30`. | Training curves sane; encoder gradient actually flowing (the Gen14 U8/U9 `G-B*` gates apply). |
| **S3** *(analysis)* | Near-boundary stratified DA per §6.2, against the pinned DPCC target. | — |

**Do not skip S0.** It is the cheapest kill-switch in the plan and it tests the assumption everything else
rests on.

---

## 8. Where to start — avoiding first, not aligning

The note asks *"maybe in V_Aligning first?"*. Aligning is the only env that satisfies §8b's three conditions
today, so the instinct is sound — but for *this* experiment it is the **harder** task, not the easier one:

| | avoiding | aligning |
|---|---|---|
| replay-render collector for this task | ✅ exists | needs checking/building |
| per-episode marker consistency without new demos | ✅ **free** via 24-modality (§4.1) | ❌ **no mode structure to exploit** — the demos are single-mode pushes. A keep-out zone either sits in free space (inert) or contradicts the push path |
| existing constraint geometry | inherited from upstream DPCC, load-bearing for cross-gen tables | 🔴 explicitly `PLACEHOLDER` / *"relaxed for test purpose"* — you would be validating a placeholder |
| state baseline fairness | ✅ 4-D obs has no geometry either way | ⚠️ needs care |
| host frame status | Gen16 code complete | Gen14 @ HEAD, actively churning (U9) |

🔴 **Recommendation: Gen16 U1 on avoiding.** Aligning becomes the follow-up if the avoiding result is
ambiguous — it is the right place to ask *"can this encoder read geometry at all?"* precisely because its
scene already varies per episode, but that is a different question from the one this note poses.

**Relation to the UAV idea doc:** the two are complementary answers to the same §8b criterion — *pool the
scenes* (vary the geometry across episodes) vs *render the constraint* (make an invisible constraint
visible). They are not competing, and neither blocks the other. If both land, UAV gets the stronger version:
pooled scenes **and** rendered no-fly volumes, which is the closest thing in this repo to the fire-zone
scenario the note opens with.

---

## 9. What will silently break it

| # | risk | mitigation |
|---|---|---|
| **R1** 🔴 | **96×96 is small.** A translucent floor patch under a 30°-tilt cage view may not survive downsampling. | **S0 gate.** Target **A** (a workspace-spanning region) is far more likely to survive than **B** (a ring a few pixels thick) — §2. |
| **R2** | **Colour collision.** The scene already uses saturated red (posts) and translucent green (`finish_line`). A third marker in a nearby hue is unreadable at 96×96. | Pick a distinct hue **and** a distinct alpha; verify in S0. |
| **R3** | **Frozen / pretrained encoder.** Gen14 U9 added `vis_pretrained` and `vis_cond_mode`. An ImageNet-pretrained frozen trunk has no reason to encode a translucent floor patch well. | Decide the knob deliberately, record it, and include it in the S0 linear probe. |
| **R4** | **Dataset path drift.** The Gen16 hotfix consolidated dataset paths into `mix_visual_avoiding/datasets/sequence.py` specifically to stop config drift. Re-rendering into the wrong root will silently train on the **old, unmarked** images and produce a perfect null result. | New data root, explicit path, and a gate that asserts the marker is present in a sampled frame at train start. |
| **R5** | **Sample size.** The existing Gen16 `fm`/`dpcc` visual cells are **n = 2 episodes** (§2 of the snapshot); only the `mf` cell is n = 30. | Do not compare U1 against those cells. Run `n_trials = 30` throughout, and if a P2 baseline is needed, **re-run it**. |
| **R6** 🔴 | **Do not touch `config/visual_avoiding_mix_eval.yaml`'s geometry.** Its header declares the geometry blocks byte-identical to Gen9 and Gen3v6 and says so explicitly: *"If you change geometry here, you have forked the task and every cross-generation table becomes invalid."* | **Fork the config.** U1 gets its own eval yaml; P2 keeps reading the existing one unchanged. |
| **R7** | **Train/eval constraint-set disagreement already exists** (§1.2: `constraint_list` has six posts at r = 0.04; the eval yaml has three enlarged spheres and no l3 row). | Audit and *document* it before U1 numbers land, so it is not mistaken for a U1 effect. Do not silently "fix" it. |
| **R8** | **"The camera sees through, so it is safe."** | Not supportable from a 96×96 third-person view — no depth, no occlusion reasoning. Do not motivate the work this way; motivate it with §6. |

---

## 10. On the real-world framing — this part is genuinely good

The note's motivation is the strongest part of it, and it should survive into any write-up:

> *"the virtual restriction of a UAV — some space is forbidden, but not necessarily an obstacle. Like in a
> FIRE… in an industrial env the half-space is forbidden to use."*

This is a real and under-served category. Most vision-based avoidance work assumes **physical** obstacles, and
most constrained-planning work (DPCC included) assumes the constraint is **declared** in closed form. The gap
between them — **regions that are semantically forbidden, physically penetrable, and legible only from
markings** — is exactly what a `visual_only=True` MuJoCo body models, and it is exactly the pairing this repo
is built to test: a generative brain that *perceives* the soft constraint, and physical brakes that *enforce*
the hard one.

⚠️ Scope the claim honestly. In simulation a painted zone is a texture. *"The network learned the semantics of
a no-fly marking"* requires the zone to vary in **position and appearance**; with fixed appearance the
supportable claim is *"the network reads a geometric region from pixels"* — which is enough for U1, and less
than it sounds like.

---

## 11. Open questions for you to decide

1. **Rendering target** — **A** (half-space region, recommended, §2), **B** (obstacle margin bubble, incl. the
   missing l3 row), or both as an ablation? Note only these two are renderable at all — `dynamics` and
   `bounds` are not regions (§1.1).
2. **Sampling** — continuous line (recommended, §8b's "small finite set is a weak version") or the three
   existing named variants?
3. **Appearance** — fixed colour/alpha (cheaper, weaker claim) or randomised (stronger claim, more data)?
4. **Arms** — all three of P2/P0/P1, or P2/P1 only to halve the GPU cost?
5. **Engines** — `mf` alone for the pilot (K = 2, cheapest) then widen, or `mf` + `fm` from the start?
6. **R7** — audit the train/eval constraint mismatch now, or after U1?

---

## 12. One-line summary

**The premise holds: the constraint is drawn nowhere — what the camera sees on avoiding is D3IL's post field,
which is the *task*, while DPCC's constraint set lives only in a yaml and has never been in a pixel. Of its
four families only two are regions at all (`dynamics` and `bounds` cannot be drawn), leaving two targets: the
half-space, which is invisible today and large enough to survive 96×96, and the obstacle margin, which is the
enforced radius around a post that is already visible and is 2-3x bigger than it looks with the third row not
constrained at all. Render the half-space first, per episode, relabelled against the task's own 24-modality so
every demo is consistent with the marker it is shown with — which costs zero new expert data, because the
replay-render collector, the `visual_only` object precedent and static-body repositioning all already exist.
Then measure pre-projection feasibility and projector intervention stratified by distance to the boundary —
not steps-to-goal — and accept that the negative result ("perception cannot substitute for declared
constraints") is as publishable as the positive one. Gate the whole thing on one cheap CPU job first: at
96×96, can you even see the marker?**
