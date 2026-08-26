# IDEA — **train one visual policy across all UAV scenes, test per-scene**

*Filed under Gen11 E10 · **build target: Gen15** (`mix_uav/` ↔ `mix_uav_test/`) · linked from `logs_in_develop/Gen15/`*

**Date:** 2026-08-26 · **Type:** idea assessment / experiment design · **NO CODE WRITTEN**
**Status:** proposal — read §3 before scoping anything.

> 🔴 **Filed under Gen11 E10 (that is where the UAV-visual thinking lives), but the BUILD TARGET IS
> GEN15.** The host frame is **`mix_uav/` ↔ `mix_uav_test/`** (the multi-engine UAV frame, and the
> only one with deployable results), **not** Gen11's `flow_matcher_v3_uav/` ↔ `FM_v3_uav_test/`
> that the 2026-08-11 plan was written against. Every code path cited below is already the Gen15 one
> (`mix_uav/datasets/d4rl.py`, `config/uav_mix.py`). Linked from `logs_in_develop/Gen15/`.
> See §9.1 for what changes when the old plan's Gen11 references are read against the Gen15 frame.
**Companion (older, partly stale):** [`PLAN_E10_uav_visual_mode.md`](PLAN_E10_uav_visual_mode.md)
(2026-08-11) — the *how* of grafting the vision stack onto the UAV frame (asset audit §1, collector
defects §1.1, encoder/FiLM plumbing). **Not superseded — it is the implementation half — but it
targets the Gen11 frame and must be read through the translation table in §9.1.**
This file is the *why* and the *experiment design*, which that plan does not cover: it assumes
one-scene-at-a-time training throughout.
**Trigger:** [`SNAPSHOT_20260826_visual_avoiding_env_status.md` §8b](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260826_visual_avoiding_env_status.md)
— visual-avoiding turned out not to be a perception experiment, and §8b generalised why.

---

## 0. The idea, and the verdict up front

> *"If the ENV is fixed, not so much sense. Could we train our 4 ENVs together, with the goal line
> added, plus the visual component — train one NN on the 4 ENVs and test on an individual ENV?
> …but will it even work? pillars / s_curve / corridor are very different and empty is even not a
> scene."*

**Verdict: this is the right instinct, and it is the correct fix for the defect §8b identified.
It is also cheaper than it looks — the multi-scene data path already exists. But three things will
silently break it, and one of them is the goal-line addition itself.**

| | |
|---|---|
| ✅ **Is the diagnosis right?** | Yes. Scene variation is condition #1 of the §8b criterion, and it is the one every FM-PCC visual run has failed so far. Pooling scenes is exactly how you satisfy it. |
| ✅ **Is the data plumbing there?** | **Largely yes, and this was not obvious.** `scene='all'` is already implemented — `mix_uav/datasets/d4rl.py:44-47` returns all four scenes, and `config/uav.py:47,146` already sizes `max_path_length=750` for the pooled case. **Nobody has trained on it.** |
| 🔴 **Will the goal-line addition help?** | **It may destroy the experiment.** The route endpoints differ per scene (`x_end=3.2` pillars vs `2.8` corridor), so a goal in obs is a **scene ID in disguise**. §3.1. |
| ⚠️ **Is `empty` a problem?** | Your instinct is right. It has no constraint group at all and an ill-defined goal. §3.2 gives the disposition — it is not simply "drop it". |
| ✅ **Is "state can't handle multi-scene" right?** | **Yes — twice over.** `goal_dim = 0` today (`config/uav_mix.py:19`), *and* pooling makes `p(τ | state)` **multi-modal over an unobserved scene latent**, which no amount of capacity fixes. **The image is the conditioning variable that collapses the mixture.** §3.0, §3.0.1. |
| ⚠️ **What will arm B's failure look like?** | **Not incompetence — confident wrong-scene flight.** A generative planner samples a *mode*, so it will fly a smooth trajectory that is correct for a different scene. Predicted signature + the fan-divergence diagnostic in §3.0.1. |
| 🔴 **So will the image "work"? Yes — but watch *why*.** | Neither camera contains a goal marker, so the only way vision fixes *"where do I go"* is **scene classification + memorised route (T1)** — which a one-hot label does for free. Vision must earn its place on *"what do I avoid"* (T2). **§3.0, §3.0.2 — this is the main addition to this file.** |
| ✅ **Do we need a dual-camera collector?** | **It already exists and has never been run** — `uav_expert_data_collect/collect_camera_images.py` + its SLURM wrapper. Two defects to clear first, one of which is a real design choice. §2.1. |
| ⚖️ **Will it work?** | Honest answer in §6: pooled-train/per-scene-test is **likely to work and is a modest claim**; leave-one-scene-out is the "ultimate test" you're excited about and is **genuinely uncertain** at n=3 scenes. |
| 🔴 **Cheapest decisive first move** | **Do not build the encoder first.** A state-only `scene='all'` run answers most of this with code that already exists. §7. |

---

## 1. Why the instinct is right — the §8b criterion applied to UAV

A vision branch earns its cost only if the task-relevant scene state **(1)** varies per episode,
**(2)** is absent from obs, and **(3)** is not handed to the controller by another route.

| env | 1. varies? | 2. absent from obs? | 3. no other route? | perception load-bearing? |
|---|---|---|---|---|
| visual **aligning** | ✅ box/target re-sampled | ✅ 6-D robot only | ✅ | ✅ **yes** — the only one today |
| visual **avoiding** | ❌ obstacles are literal constants | ✅ | ❌ hardcoded in the projector | ❌ no |
| **UAV as run today** | ❌ **one scene per run** | ✅ | ❌ walls/pillars are projector constraints | ❌ no |
| **UAV, this proposal** | ✅ **pooled scenes** | ⚠️ **only if §3.1 is handled** | ⚠️ §3.3 | ⚖️ **that is the experiment** |

UAV fails #1 today not because its scenes are fixed by design — it has four — but because the run
protocol is single-scene. Per
[`SNAPSHOT_20260825_uav_mix_env_status_PILOT.md`](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260825_uav_mix_env_status_PILOT.md),
the entire deployable core is **13 candidates × 2 943 rollouts, all `corridor`, one seed**;
`pillars` and `s_curve` ran **n = 3** and are quarantined. **This proposal is a protocol change
before it is a modelling change**, which is why it is cheap.

---

## 2. What exists vs what must be built

| | asset | state |
|---|---|---|
| ✅ | **Pooled scene loader** — `_scenes_for('uav-all')` → all four (`mix_uav/datasets/d4rl.py:44-47`) | **exists, never trained** |
| ✅ | Pooled horizon sizing — `MAX_PATH_LENGTH_PER_SCENE`, `'all'` bounded by `s_curve` = 750 (`config/uav.py:44-52,146`) | exists |
| ✅ | `SafeLimitsNormalizer` for constant-valued dims — already needed *because* of pillars (`config/uav.py:138`) | exists, and see §3.4 |
| ✅ | Four scene generators with distinct geometry (`uav_expert_data_collect/trajectories.py:66,112,123,174`) | exists |
| ✅ | Vision encoder + FiLM U-Net, visual engine wrappers, image dataset pattern | exists — old plan §1, assets A3–A5 |
| ⚠️ | Dual-camera expert-image collector | **written, never run** — old plan §1.1 lists defects D-1/D-2 to fix first |
| ❌ | **UAV visual arm** — no `visual_*.py`, no `MultiImageObsEncoder`, no camera key anywhere in `mix_uav/` | **does not exist** |
| ❌ | Per-scene eval breakdown of a pooled-trained model | does not exist |
| ❌ | Scene-balanced sampling | does not exist — §3.4 |

**Read this table as: the expensive half (pooled data) is done; the missing half is the graft the
old plan already specifies.** The new work in *this* file is the experiment design, not the code.

### 2.1 The dual-camera collector — it exists, it has never run, and it needs a decision first

**You do not need to write the tool.** It is `uav_expert_data_collect/collect_camera_images.py`
(11.9 kB) with a SLURM wrapper at `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh`
(EGL + GPU, 8 h). Both dated 2026-07-07, **both never executed** (old plan §1, assets A1/A2). It
replays the E4 expert pickles, injects `qpos/qvel/q`, and renders two 96×96 views per step.

| camera | what it is | what it can supply |
|---|---|---|
| **`fpv`** | nose-mounted, looks along the +x body axis; a real camera in each scene XML (`collect_camera_images.py:57-59`) | **obstacles ahead** → failure (b). The load-bearing view. |
| **`bp_overhead`** | **not** a fixed world camera — a free camera built at render time with `cam.lookat[:] = drone_pos` (`:124-130`), i.e. **it follows the drone**, 5 m out | **local free space around the vehicle** → also (b), from a second aspect |
| — | *goal marker* | ❌ **neither camera has one.** See §3.0. |

⚠️ **Two blockers before it can be run** (old plan §1.1):
- **D-1** — writes to `images/track-cam/…` while the E5 spec and every downstream doc say
  `fpv-cam`; the docstring still describes an old `track` chase cam. Cosmetic, but it will break
  the loader path.
- **D-2** — the docstring at `:6` claims *"bp-cam: fixed overhead bird's-eye camera"* while the code
  at `:124-130` builds a **drone-following** camera. **This is now a design decision, not a doc fix**,
  and multi-scene inverts which choice is right:

  | overhead camera | effect on this experiment |
  |---|---|
  | **fixed world camera** | The drone's absolute position within the frame is visible ⇒ **scene identity and progress are trivially readable** ⇒ pushes the model toward **T1**. |
  | **drone-following** (current code) | Translation-invariant ⇒ shows *local* geometry only, no global position ⇒ **suppresses T1 and forces T2-style local reasoning.** |

  🔴 **The current, "undeclared" behaviour is probably the better one for this experiment** — the
  opposite of what the old plan implies. Decide deliberately, and record which was used, because
  **it determines which tier of claim (§3.0.2) the run can support.**

⚠️ **Collection cost scales with scenes.** The 8 h wrapper was sized for one scene's episode set.
Four scenes with `s_curve` at 750 steps is a different budget — re-estimate before submitting, and
per the repo convention set `--time` to ~2× the expected wall (24 h cap).

---

## 3. 🔴 The traps

### 3.0 "State can't know where to go" — correct, and it is the wrong reason for vision to win

**Confirmed, and more strongly than assumed: the UAV observation carries no goal at all today.**
`cond_mode='pos_only'` → `obs = [p_des | p]`, 6-D (`config/uav_mix.py:19,131`), and
`get_goal_dim()` counts obs dims that are *constant across the episode*
(`mix_uav/datasets/sequence.py:98-105`) — neither `p_des` nor `p` is, so **`goal_dim = 0`.** A pooled
state-only policy has literally nothing to aim at. It is not merely handicapped; the pooled task is
**ill-posed** for it.

So yes — arm B will fail. But **why** it fails decides whether the visual result means anything,
because there are two independent failures and only one of them is a perception problem:

| | failure | what fixes it |
|---|---|---|
| **(a)** | **"where do I go?"** — goal ambiguity. Three scenes, three different routes, no goal in obs. | **a goal in the conditioning** — this is what your goal-line addition is for. **Not vision.** |
| **(b)** | **"what do I avoid?"** — geometry ambiguity. Same position, same goal, different obstacles. | **vision** — this is the only thing a camera is uniquely good for here. |

🔴 **If vision "works" by fixing (a), it is not a perception result.** Look at what the cameras
actually see (§2.1): `fpv` is nose-mounted along +x, `bp_overhead` is a top-down view centred on the
drone. **Neither contains a goal marker** — the routes are analytic paths
(`uav_expert_data_collect/trajectories.py`), not objects rendered in the scene. So the *only* way a
camera can answer (a) is: **classify which scene this is → recall that scene's memorised route.**
That is scene classification plus memorisation, and a 2-bit integer would do the same job.

**Therefore: add the goal (it makes the task well-posed), make it scene-agnostic (§3.1), and then
vision has to earn its place on (b) alone.** That is the clean experiment, and it is the one worth
running.

#### 3.0.1 Why this is a **conditioning** problem, not a capacity problem

*(This is the theoretical core, and it is worth stating precisely because it predicts what arm B's
failure will look like — which is what makes B worth running at all.)*

The planner learns `p(τ | c)`. Today `c = [p_des | p]` — **"where I am and what I am doing"**, and
nothing else. Pool the scenes and the *same* `c` now occurs in several scenes with **different
correct `τ`**. The conditional becomes a mixture over an unobserved latent:

```
p(τ | c)  =  Σ_s  P(scene = s | c) · p(τ | c, scene = s)
```

**The scene is a latent variable that is not in `c`.** No amount of network capacity fixes that —
the target distribution is genuinely multi-modal given the input. This is why "train a bigger net on
pooled data" is not an alternative: the information is absent, not under-extracted.

🔴 **And a *generative* planner fails differently from a regressor — which matters for how the logs
will read.** FM-PCC is generative (FM / MeanFlow / AlphaFlow / DDPM):

| model class | behaviour under an unobserved latent |
|---|---|
| deterministic regressor | collapses to `E[τ|c]` — the **mean of the mixture**, a trajectory valid in *no* scene. Flies between the pillars' L and R channels, i.e. into the pillar. Obvious in the logs. |
| **generative planner (ours)** | **samples a mode.** Produces a smooth, entirely plausible trajectory — **that is correct for a different scene.** Right with probability ≈ `P(correct scene | c)`. |

So **do not expect arm B to look incompetent.** Expect **confident, well-formed, wrong-scene
flight** — which is the more dangerous failure and the easier one to misread as "the model is fine,
the projector is broken."

**Predicted failure signature of arm B — write these down before running it:**

1. **Trajectories look good and go to the wrong place.** High action plausibility, low
   scene-appropriateness. `safe` collapses while the path stays smooth.
2. **Front-loaded mode commitment.** The scenes overlap most in `(p_des, p)` at episode start (all
   traverses begin near `x ≈ −3` heading `+x`) and separate later. So the model commits to a mode
   early, and a wrong commit persists for the whole episode. **Failures should cluster at t ≈ 0**,
   not accumulate.
3. 🔴 **MPC-fan divergence spikes — and this is the best diagnostic available.** `mpc_batch_size: 4`
   (`config/uav_mix.py:187`) means every replan draws **4 samples from the mixture**. Under B those 4
   candidates should *disagree with each other* (different scene modes); under C, conditioned on the
   image, they should **agree**. **Inter-candidate spread within one plan is a direct read on whether
   the image collapsed the mixture** — far more informative than `S&C`, available at every control
   step rather than once per episode, and it needs no new eval machinery, only logging the fan.
   **Instrument this before arm B runs**; it is the measurement that turns "vision helped" into
   "vision resolved the ambiguity, and here is the ambiguity being resolved."
4. **The projector fights the generator.** Plans feasible for *another* scene are infeasible here, so
   projection intervention and violation counts rise. Watch `proj_ms` and the constraint columns —
   but do not diagnose from them alone (see 1).

**This also makes arm B2 rigorous rather than heuristic.** A one-hot scene label supplies exactly
`log₂ K` bits — **precisely enough to collapse the mixture and nothing more.** So:

- **`C ≈ B2`** ⇒ the image supplied **only scene identity** (T1).
- **`C > B2`** ⇒ the image supplied **within-scene geometry** the label cannot carry (T2). *That is
  the result worth having.*

⚠️ **One latent the image cannot fix — do not score it.** `corridor` already carries a *second*
unobserved latent: the seeded `homotopy` (L/C/R), which the eval tracks as `match` — *"the
state-only policy is **never told** the homotopy"*
([UAV snapshot §0.1](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260825_uav_mix_env_status_PILOT.md)).
Homotopy is an **intended mode**, not a scene property — no camera can read "which side did you mean"
off an arena. **Pooled training therefore has two latents, and vision collapses only one.** `match`
will stay near chance even for a perfect visual model; **scoring the visual arm on `match` would
penalise it for something vision cannot supply.** (It is a fine diagnostic for arm B's *scene*
confusion, which is why it is worth logging — just never a success metric here.)

#### 3.0.2 Three tiers of "the image works" — they are worth very different amounts

| tier | what the encoder is actually doing | detected by | worth |
|---|---|---|---|
| **T1** | **Scene classification** — recognise the layout, recall the trained route | ❌ **collapses on a perturbed variant** of a trained scene (pillars shifted 0.3 m) | **weak** — a one-hot scene label is equivalent and free |
| **T2** | **Local geometry perception** — read free space / obstacle bearing ahead, plan around it | ✅ survives perturbation · ❌ fails on a novel layout | **real** — the honest target for a first result |
| **T3** | **Generalisation to unseen geometry** | ✅ survives LOSO (§5b) | **strong** — the "ultimate test" |

**Two cheap probes separate these, and both should be scoped in from the start:**

1. 🔴 **The one-hot scene-ID baseline (arm B2, §4).** Train state + goal + a 3-way one-hot scene
   label — **no camera**. If the visual model only matches it, **vision bought exactly one scene ID's
   worth of information: T1.** This is a far sharper control than plain state-only and costs one
   extra training run with zero new data collection. **Run it alongside C.**
2. **The perturbation probe.** Test the trained visual model on a shifted/rescaled version of a scene
   it trained on. T1 collapses; T2 and T3 survive. This is the §6 parametric-scene recommendation,
   now with a specific job: it is **the only way to tell T1 from T2**, and no amount of in-distribution
   scoring can do it.

### 3.1 The goal-line addition may leak scene identity — this is the one that kills it silently

You proposed adding *"where the Goal Line is"* to the conditioning. **Check the route endpoints
before doing this:**

| scene | generator | traverse |
|---|---|---|
| `pillars` | `pillar_path(...)` (`trajectories.py:66-67`) | `x_start=-3.2 → x_end=+3.2` |
| `corridor` | `corridor_path(...)` (`trajectories.py:112-113`) | `x_start=-2.8 → x_end=+2.8` |
| `s_curve` | `s_curve_scene_path(...)` (`trajectories.py:123`) | Z-route, different again |
| `empty` | `empty_path(p_start, p_end, ...)` (`trajectories.py:174`) | arbitrary endpoints |

**The endpoints are scene-specific constants.** A model handed the goal can read `|x_end| = 3.2` and
know it is in `pillars` without ever looking at the image — and once it knows the scene, it can
recall that scene's fixed obstacle layout from training, exactly as the avoiding model does today.
**You would have rebuilt the §8b defect inside the fix.**

This is not an argument against goal-conditioning — a goal-conditioned policy is the right design,
and pooled training *needs* one. It is an argument for making the goal **uninformative about the
scene**:

- **Harmonise the traverse** so every scene runs the same `x_start`/`x_end` (e.g. ±2.8 everywhere).
  Cheapest, and it makes the scenes differ *only* in obstacle geometry — which is what you want to
  test.
- **Or jitter the endpoints** per episode with overlapping ranges across scenes, so the goal carries
  no scene information. Stronger, and it also fixes the "small finite scene set" weakness of §8b.2.
- **Or run the ablation**: goal-conditioned **without** vision. If that already solves all scenes,
  the goal was the scene ID and the whole visual experiment is void. This is the §7 control, and it
  is the reason to run §7 first.

🔴 **Whatever is chosen, the goal must be verified non-diagnostic of the scene before any visual run
is scored.** A 3-line check — fit a classifier from goal → scene label on the training set; if it
beats chance, the leak is live.

### 3.2 `empty` is not a scene — you are right, and "drop it" is not quite the disposition

Evidence, from the UAV snapshot: `empty` is quarantined as **"(C9, ill-defined goal)"**, and the
batch ranking's top row is `empty` at *"S&C 1.00 / **`CF` = `nan`** — `empty` records no constraint
group at all"*. So it is degenerate on **two** independent axes: no obstacles to constrain, and no
well-defined goal to score.

| use | verdict |
|---|---|
| In the **scored** test set | ❌ **No.** `CF = nan` and an ill-defined goal — it cannot be ranked against the other three, and including it would inflate any pooled average. |
| In the **training** set | ⚖️ **Optionally yes, and it is arguably useful.** It is the degenerate "no obstacles" case; a model that must decide *from the image* whether there is anything to avoid is doing more perception, not less. It also supplies the negative class the encoder otherwise never sees. |
| As a **held-out probe** | ✅ **Yes, and this is the interesting use.** "Does the policy fly straight when the image shows an empty room?" is a clean, cheap sanity check on whether the encoder is read at all — and it does not need a goal metric to answer. |

**Recommendation: train on 4, score on 3, report `empty` separately as a probe, never in a pooled
number.** Fix its goal definition first if it is ever to be scored (that is a data-collection fix,
out of scope here).

### 3.3 The projector still knows the walls — scope the claim accordingly

The corridor walls and pillar positions are **projector constraints**, supplied per scene. Under
pooled training they must be supplied per *episode* — which is correct and is **not a cheat to
remove**: DPCC *assumes* known constraints; that assumption is the method.

**The consequence is a scoping rule, not a redesign.** Perception is load-bearing for the
**generative policy** (which must produce a plan that fits this scene's geometry), never for the
projector. So the defensible claim is:

> *"A single visual policy produces scene-appropriate plans across N geometries"* —
> **not** *"the system perceives its constraints."*

The second claim needs the projector to consume perceived rather than declared geometry, which is a
different (and much larger) research programme. Do not let a reviewer read the first as the second.

### 3.4 Two smaller ones worth pinning now

- **Scene imbalance.** Episode counts per scene are not equal, and the horizons differ by 2×
  (`corridor` 360, `s_curve` 750). Pooled training bounded by `s_curve` means corridor episodes are
  heavily padded. **Check the per-scene episode counts and pad fractions on the cluster before
  training** — if `corridor` dominates, the model learns corridor and the pooled result is a
  corridor result wearing a hat. Scene-balanced sampling does not exist yet (§2).
- **Normalisation across scenes.** `SafeLimitsNormalizer` is already in place *because* pillars has
  constant-valued dims (`config/uav.py:138`). Pooling four scenes with different spatial extents
  will change every normalisation statistic. Re-derive them on the pooled set; do not reuse a
  per-scene checkpoint's stats.

---

## 4. The experiment matrix — and which single comparison decides it

Four arms. **The decisive one is C vs B**, and it is the only comparison that isolates vision.

| arm | train | obs | scored on | answers |
|---|---|---|---|---|
| **A** | per-scene specialist | state | its own scene | the status quo — Gen15 `corridor` already is this |
| **B** | **pooled 3 scenes** | **state + goal** | each scene | 🔴 **the control.** Can one *state* model already serve all scenes? Expected to fail (§3.0) — the point is *how* and *by how much*. |
| **B2** | **pooled 3 scenes** | **state + goal + one-hot scene ID** | each scene | 🔴 **the T1 ceiling (§3.0.2).** What is scene *identity* alone worth, with no camera? One extra run, no new data. |
| **C** | **pooled 3 scenes** | **state + goal + image** | each scene | 🔴 **the proposal.** Does vision add anything once the scene varies? |
| **D** | **leave-one-scene-out** | state vs state+image | the held-out scene | the "ultimate test" — §5 |

- 🔴 **C vs B2 is the comparison that matters most, not C vs B.** `C > B` only shows the model used
  *something* about the scene; `C > B2` shows it used **more than the scene's identity** — i.e. T2,
  not T1 (§3.0.2). **`C ≈ B2` means the camera is an expensive one-hot vector.** Report both.
- **C > B ⇒ vision is load-bearing** — necessary, not sufficient. Pair it with C-vs-B2 before calling
  it a perception result.
- **C ≈ B ⇒ the scene was inferable from state/goal** (almost certainly via §3.1) — fix the leak and
  re-run, or conclude vision does not pay here either.
- **B ≈ A ⇒ pooling is free** (no negative transfer). Good news, and a publishable minor result.
- **B ≪ A ⇒ negative transfer.** The scenes fight each other; a single model cannot hold them. This
  is a real possible outcome (§6) and is worth knowing before any encoder is written.

⚠️ **A must be re-measured, not quoted from the snapshot.** Gen15's `corridor` numbers are one seed
at n = 10, and `pillars`/`s_curve` are n = 3 and quarantined. A specialist baseline for the two
missing scenes does not currently exist at usable n.

---

## 5. "Test on an individual ENV" is two different experiments — say which one

This phrase hides the entire difference between a modest result and a strong one.

| | protocol | claim | difficulty |
|---|---|---|---|
| **5a. Pooled-train, per-scene test** | scenes seen in training; report per scene instead of pooled | *"one model serves N scenes"* | **modest** — this is multi-task learning, and it is the necessary first step |
| **5b. Leave-one-scene-out (LOSO)** | train on 2, test on the **unseen** 3rd | *"the policy generalises to geometry it has never seen"* | **hard, and this is the one you are excited about** |

**Both are worth running, in that order.** 5a validates the plumbing and gives the C-vs-B answer;
5b is the ultimate test. Note the arithmetic: three scorable scenes ⇒ **three LOSO folds, each
training on only two scenes**. That is thin, and a negative LOSO result at n = 2 training scenes is
**not** evidence that the approach fails — it is evidence that two scenes are not enough diversity.

🔴 **Per the reporting rules used across this project, LOSO folds are never averaged.** Three folds
are three separate results with their own n; a mean over "held-out `pillars`, held-out `s_curve`,
held-out `corridor`" hides the only thing that separates them.

---

## 6. Will it work? An honest prior, per question

**Will the image work at all? Almost certainly yes — and that is the problem.** The scenes are
visually unmistakable, so an encoder will separate them from a handful of pixels. **A "vision works"
result is therefore close to guaranteed and close to uninformative unless it clears B2 and the
perturbation probe (§3.0.2).** Design the run so it can distinguish T1 from T2 *before* it is
launched; no amount of post-hoc analysis recovers that distinction.

**5a (pooled train, per-scene test): likely yes, and expect it to be unexciting.** Goal-conditioned
multi-task imitation over 3 layouts is well within a FiLM-conditioned U-Net's capacity, and the
scenes are visually *very* distinguishable — which is your worry ("very different") but is actually
the easy direction: distinguishable scenes make the encoder's job trivial. The risk here is not
capacity, it is §3.1 — that the model solves it without looking.

**Your "they are very different" concern, inverted:** for *classification* (which scene am I in?)
difference is helpful. For *generalisation* (5b) difference is the problem — three wildly dissimilar
layouts give a model almost nothing to interpolate between. A corridor and an s-curve are not two
samples from a smooth family; they are two points. **This is why I would not bet on 5b succeeding
with the current four scenes, and why it is not a reason to skip it** — a negative LOSO with a clean
5a is still a real, reportable finding about what this data supports.

**The thing most likely to actually bite: nothing perceptual at all.** §3.4. Pooled horizons differ by 2×,
per-scene episode counts are unequal, and normalisation statistics change under pooling. Multi-task
runs usually fail on data plumbing long before they fail on representation.

**What would make 5b genuinely promising rather than a coin flip:** parametric scene generation —
pillar spacing, corridor width, s-curve amplitude as continuous knobs rather than three named XMLs.
That converts "three points" into a family with interpolation and extrapolation, and it is the same
recommendation §8b.2 makes for avoiding (continuous randomisation over a small finite set). The
generators are already parameterised (`pillar_path(homotopy_seq, altitude, duration, x_start, x_end)`,
`corridor_path(homotopy, altitude, duration, …)`, `s_curve_scene_path(altitude, duration, y_jitter, …)`)
— **the knobs exist; only the scene XMLs are fixed.** That is the highest-value follow-up and it is
much less work than it sounds.

---

## 7. 🔴 The cheapest decisive first step — and it needs no camera

**Do not build the visual arm first.** Run **arm B**: state-only, `scene='all'`, per-scene eval.

⚠️ **This is not a bet that B might succeed.** §3.0 says it should not — with `goal_dim = 0` the
pooled task is ill-posed for a state policy. Run it anyway, because **a control you expect to fail is
still the control**, and here it does four jobs a visual run cannot:

1. **The code already exists.** `env='uav-all'` is a supported selector (`d4rl.py:44-47`); the
   pooled horizon is already configured (`config/uav.py:47,146`). This is a training run and an eval
   sweep, not a graft.
2. **It surfaces every §3.4 plumbing problem** (imbalance, padding, normalisation) at a fraction of
   the cost of a visual run, before the encoder can be blamed for them.
3. 🔴 **It is a falsification test for the §3.1 leak — and the more confident you are that state
   cannot work, the more informative a success would be.** If a pooled state model *does* fly all
   three scenes, then position or goal is silently identifying the scene, and the visual experiment
   is void as designed. **That is exactly the outcome you cannot afford to discover after collecting
   images for four scenes.**
4. **It measures the size of the gap vision must close**, turning "vision helps" into a number.
5. **It is arm B.** Not a detour — you need it for the matrix regardless.

**Then run B2 (§4) before or alongside C.** One extra state-side run, no camera, no new data — and
it is what separates a real perception claim from an expensive one-hot vector (§3.0.2).

**Order of operations:** **instrument the MPC-fan spread (§3.0.1)** → B (state pooled) → fix §3.1 if the leak shows → **B2 (one-hot scene ID)** →
decide the overhead-camera question (§2.1 D-2) → fix D-1 and run the collector (§2.1) → C (visual
pooled) → **perturbation probe (T1 vs T2, §3.0.2)** → D (LOSO) → parametric scenes (§6) if D is to
mean anything.

---

## 8. Open questions to settle on the cluster before scoping

- [ ] **Per-scene episode counts and pad fractions** in `data/uav_fm/v1/{empty,corridor,s_curve,pillars}/`.
      Decides whether scene-balanced sampling is needed (§3.4). *(Data is gitignored — cluster-side.)*
- [ ] **Goal → scene leak check** (§3.1): can a trivial classifier recover the scene from the goal
      alone? If yes, harmonise or jitter endpoints before anything else.
- [ ] **Has `uav-all` ever been trained?** No evidence found in the logs; confirm against
      `Slurm_Codes/logs/important_runs/important_runs.md` before assuming it is new.
- [ ] **Do the four scenes share a camera rig?** The old plan's D-2 finding — the "bp-cam" is a
      *drone-following* free camera (`collect_camera_images.py:124-137`), not a fixed world camera.
      🔴 **An ego-centric following camera may show a nearly scene-invariant view** (drone centred,
      5 m out), which would blunt exactly the signal this experiment depends on. **Re-open D-2 as a
      design decision, not a documentation gap** — a fixed world camera may be the right choice here
      even though it was not for single-scene work.
- [ ] **`empty` goal definition** — fix or formally exclude from scoring (§3.2).
- [ ] **Specialist baselines for `pillars` and `s_curve` at usable n** — arm A does not exist for two
      of three scenes (§4).
- [ ] **Decide the goal representation.** `goal_dim = 0` today (§3.0); pooled training needs one. Pick
      the encoding (absolute endpoint vs bearing-to-goal vs remaining-displacement) **and check it is
      not scene-diagnostic** (§3.1). *Bearing/remaining-displacement leak far less than an absolute
      endpoint — prefer them.*
- [ ] **Does position alone identify the scene?** Separate from the goal leak: the scenes have
      different spatial extents, so `p` itself may be diagnostic. Same classifier check as §3.1, on
      `p` instead of the goal.
- [ ] **Re-estimate the collector's wall-clock for 4 scenes** (§2.1) — the 8 h wrapper was sized for
      one.
- [ ] 🔴 **Log the MPC fan's inter-candidate spread** (`mpc_batch_size: 4`, `config/uav_mix.py:187`)
      **before arm B runs.** It is the direct measurement of scene ambiguity being resolved (§3.0.1),
      it costs one array per replan, and it cannot be recovered after the fact.
- [ ] **Confirm `match` is logged but excluded from scoring** on every pooled arm — vision cannot
      resolve the homotopy latent (§3.0.1).

---

## 9. Relationship to the existing E10 plan

[`PLAN_E10_uav_visual_mode.md`](PLAN_E10_uav_visual_mode.md) (2026-08-11) remains the implementation
reference: asset audit (§1), collector defects D-1/D-2 (§1.1), encoder/FiLM plumbing, the
30.3 ms-budget discipline. **Two amendments this idea forces:**

1. **It assumes single-scene training throughout.** Everything in §3 and §4 here is additive to it.
2. **Its D-2 finding is upgraded from a documentation gap to a live design decision** — see §8.
   A drone-following camera is defensible for one scene and possibly disqualifying for four.

### 9.1 🔴 The build target moved: Gen11 → Gen15

The old plan names Gen11's UAV frame as the host (`flow_matcher_v3_uav/` ↔ `FM_v3_uav_test/`).
**That is no longer where UAV work lives.** Translate before using it:

| old plan says (Gen11) | build against (Gen15) |
|---|---|
| `flow_matcher_v3_uav/` (model) | **`mix_uav/`** |
| `FM_v3_uav_test/` (eval) | **`mix_uav_test/`** |
| `config/uav.py` | **`config/uav_mix.py`** (`uav.py` remains the Gen11 entry; the scene tables are duplicated in both — `uav_mix.py:48-51`) |
| single-engine FM | the **engine registry** — `mix_uav/models/engine_registry.py`, arms `fm`/`mf`/`af`/`ddpm` |
| asset A7 `FM_v3_uav_test/eval_fm_uav.py:780-840` (per-scene renderer, EGL discipline) | the equivalent in `mix_uav_test/` — **verify it carried over before assuming the renderer exists there** |

⚠️ **The engine axis is the substantive difference.** Gen15 runs four engines; a visual graft must
compose with `engine_registry` rather than hardcode one engine, exactly as Gen16 did for avoiding
(`mix_visual_avoiding/models/visual_{fm,mf,af,gaussian}_diffusion.py` — one thin subclass per arm
over a shared `visual_spec`). **Copy that shape.** Building a single `VisualFlowMatching` for UAV
would repeat the Gen7→Gen14 mistake the `visual_spec.py` header documents.

⚠️ Also carry over the old plan's own warning: `VisualUNet` **hardcodes `TRANSITION_DIM = 9`**
(old plan §0.1 item 4). Gen16 solved exactly this class of problem by hoisting the spec into
`visual_spec.py` — **a UAV visual arm should copy Gen16's `visual_spec` pattern, not Gen7's
hardcoded one** (`mix_visual_avoiding/models/visual_spec.py` header explains why, at length).
And per [`SNAPSHOT_20260826…` §4b.1](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260826_visual_avoiding_env_status.md),
**encode the image once per plan (Gen16 `visual_latent` short-circuit), not once per network call
(Gen9)** — at K = 20 that is ~19 redundant ResNet passes per plan, and the UAV frame is the one
place in this repo where per-step latency is actually tracked.

---

## 10. One-line summary

**Pooling the UAV scenes is the correct fix for the defect that made visual-avoiding meaningless, the
pooled data path and the dual-camera collector both already exist, and you are right that a state
policy cannot do this — `goal_dim = 0`, so the pooled task is ill-posed for it. But the cameras carry
no goal marker, so vision will "work" the cheap way — classify the scene, recall the route — unless
the run is designed to rule that out: give the model a scene-agnostic goal, benchmark it against a
free one-hot scene label (B2), and probe it on a perturbed scene. Do the state-only `uav-all` run
first; it needs no camera and it is the one result that can tell you the experiment is broken before
you collect four scenes of images.**
