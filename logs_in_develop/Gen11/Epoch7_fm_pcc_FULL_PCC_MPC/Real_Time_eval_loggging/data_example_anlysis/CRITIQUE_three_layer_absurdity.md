# Critique — Are the "Three-Layer Absurdity" Claims Actually Real Problems?

**Date**: 2026-06-27
**Scope**: A skeptical re-reading of [WHY_FM_KEEPS_PLANNING.md](./WHY_FM_KEEPS_PLANNING.md) §"The three-layer
absurdity, stated plainly" (lines ~1866–1883). Question posed: Layer 1 is resolved (no catastrophe).
Are Layer 2 and Layer 3 *de facto* problems — for DPCC, and for our UAV?

> [!IMPORTANT]
> **Verdict up front.** The three "layers" are NOT three independent defects of equal weight — and
> after confirming the diffuser's anchor includes **real measured `p`** (§0.1), **all three are
> wrong or benign as stated.** Layer 1 is a resolved naming issue; Layer 2's "hallucinated state" is
> wrong (the obs is `[real command｜real measurement]`); Layer 3's "self-reference" is normal
> *grounded* autoregression. The one genuine defect they grope toward lives elsewhere — in the
> **constraint** (binds command `p_des`, not real `p`; §9). Ranked by how "real" the problem is:
>
> | Layer | Claim | Is it a real *problem*? | Verdict |
> |---|---|---|---|
> | **1** | "Units lie" (velocity vs Δpos) | ❌ No — resolved | Real *discrepancy*, benign. Naming only; `dt=1` reconciles it. |
> | **2** | "State is half-hallucinated" | ❌ Largely wrong **as framed** | The obs is **`[real command｜real measurement]`** — the anchor pins **real measured `p`** too (§0.1). Nothing is "hallucinated": `des`/`p_des` is a real *command*, `p`/`c_pos` is a real *measurement*. Surviving true part = a nitpick ("paper doesn't flag `des` is a command"). |
> | **3** | "Self-reference never named" | ❌ Mostly wrong | The loop is **grounded** — real `p` is conditioned on every step *alongside* `p_des`. Normal grounded autoregression, not an echo chamber. "Reads its own write" omits that it also reads reality. |
>
> The original doc's word choices ("absurdity", "hallucinated", "reads its own write") are
> **rhetorical**, and — now that we've confirmed the anchor includes **real measured `p`** (§0.1) —
> L2/L3 are not merely *overstated*, they are **pointed at the wrong component.** The genuine defect
> is **not** the state or the conditioning (the model *does* see reality); it is the **constraint**,
> which binds the command `p_des` and never real `p` (§9), letting the UAV drift. The three-layer
> framing blamed "fake state"; the real issue is "the projector acts on the command, not reality."

---

## 0. The one fact everything hinges on: tracking error

Both pipelines build state as `[ desired_pos | actual_pos | … ]`, where `desired_pos` is the
FM's own integrated output and `actual_pos` is a real sensor read. Whether the "echo" half is a
problem depends entirely on **how far `desired_pos` drifts from `actual_pos`** — i.e. the
low-level controller's **tracking error** `e_t = ‖p_des,t − p_t‖`.

| System | Low-level controller | Tracking error `e_t` | Consequence |
|---|---|---|---|
| **DPCC arm** (D3IL aligning) | IK / PD, position-controlled rigid arm | **≈ 0** (arm reaches commanded pos within a step) | `des_c_pos ≈ c_pos` → the two halves are **redundant**, not contradictory |
| **Our UAV** (quadrotor) | underactuated, momentum, geometric controller | **> 0, time-varying** (`p` *lags* `p_des`) | `p_des` and `p` genuinely diverge → the "echo" carries information that disagrees with reality |

This is stated in our own code:

> `eval_fm_uav.py:130–132` — *"the dynamics `deriv` binds **p_des** to the action … because
> p_des is the exact integrator of the action (`p_des[t+1]=p_des[t]+act`), while the drone's p
> **lags**. (Visual-aligning binds c_pos because its arm **tracks perfectly**.)"*

And we already **log it**: `eval_fm_uav.py:301` declares `track_err = []`. So the entire
Layer-2/Layer-3 question is **empirically answerable from data we already collect** — it does not
need to be argued in prose. **Go read `track_err`.**

---

## 0.1 Primer — what the FM actually denoises, and what gets executed (read this if the channels blur together)

Easy to mix up `transition_dim`, conditioning, and execution. Concretely, for the **UAV (12D per
step)** each horizon row is:

```
x[h] = [ Δp_des(3) | p_des(3) | p(3) | v(3) ]
indices:  action 0–2   cmd 3–5   REAL 6–8  vel 9–11
```

The FM denoises the **whole `[H=8, 12]` block** — 8 future rows, *all* channels. So it does output
"next action **and** next `p_des` **and** next real `p` **and** next `v`," for all 8 steps. It is a
**trajectory generator** (Janner-style), not an action predictor. One call:

1. Build obs from reality **now**: `obs = [p_des_now | p_now | v_now]` (`p,v` = **real measured** sensor; `p_des` = the **current command**).
2. `apply_conditioning` **pins row 0's obs part** to that obs → `x[0] = [free action | current p_des | REAL p | REAL v]`. **← the FM IS told the real measured position `p`.**
3. FM denoises → an 8-step plan of `(action, p_des, p, v)`.
4. Take **only** `action = x[0]`'s action (indices 0–2).
5. Execute: `p_des += action` (integrate the command) → PID tracks `p_des` → drone flies → measure **new real `p, v`**.
6. Re-plan (receding horizon — only the *first* action ever executes).

> [!IMPORTANT]
> **Terminology (do not say "real `p_des`").** `p_des` is **always a command/setpoint** — there is
> no "real `p_des`," because it is never measured; it is what we *told* the controller. "Real /
> measured" applies **only to `p` and `v`** (read from `data.qpos`/`data.qvel`). So at the anchor:
> `p_des` = the current command, `p` = the **real measured position**. Earlier shorthand "real
> p_des" was wrong wording for "the current `p_des`."
>
> **And yes — we DO tell the diffuser the real location.** The anchor (`apply_conditioning`) pins
> the *full* obs sub-vector at row 0, which **includes real measured `p`** (indices 6–8). The
> diffuser is conditioned on reality, at least as an **input**. The whole `p_des` critique is **not**
> "the model can't see real `p`" — it can — it is "real `p` is *only* an input; the integration and
> the constraint act on the command `p_des`, never on `p`."

**What the anchor pins (and doesn't):**
```
x[0] = [ Δp_des(0:3)    |  p_des(3:6)   p(6:9)   v(9:12) ]
          FREE (denoised)   └─── PINNED to current obs (9D) ───┘
         (the action you      includes REAL measured p & v
          execute)            + current command p_des
rows 1..7 : fully FREE (the FM's predicted rollout); goal_dim=0 → no goal channel
```
Janner's exact anchor (`x[:, t, action_dim:] = val`), re-applied every denoising/flow step
(`diffusion.py:185,267,279,312`). It pins **all of the obs** at H0, **not** the action of H0, and
**not** H1…H7.

**Three clarifications that dissolve the "wtf":**
- **Conditioning *does* include real measured `p`** — pinned at t=0 (indices 6–8). Real `p` is an **input**. (We are not hiding reality from the model.)
- **The dynamics constraint does NOT use real `p`** — it binds `p_des` (3–5). So real `p` is
  *observed, never controlled/constrained*. (Input vs constraint = the §9.1 "three roles.")
- **The FM's predicted `p` for t>0 is imagination** — the *actual* next `p` comes from physics,
  measured fresh next step. Predicted-`p` is never executed; only the action is.

**One-sentence model:** the FM is a *"given where I am (current `p_des`, real `p,v`) + goal, what
action next?"* oracle packaged as a full trajectory; the loop integrates/constrains the **command
`p_des`**, while the real-`p` loop is closed by the **PID the FM never models**. (For the arm it's 9D
`[Δdes(3)｜des(3)｜c_pos(3)]`; same idea, and there the constraint *does* bind real `c_pos` — §9.7.)

---

## 0.2 Plain-language reset — what's *actually* wrong (read this if the chapters blur together)

Forget the layer numbers for a second. The whole thing in five lines:

- The drone's obs is `[p_des | p | v]`: **`p_des`** = where we *told* it to go (command), **`p`** =
  where it *actually is* (real sensor), **`v`** = real velocity. The FM is anchored to all of this at
  t=0 — **so it sees reality.**
- It outputs an action `Δp_des`; we do `p_des += action`; a **PID** flies the drone toward `p_des`;
  the drone **lags**.
- **L2 ("state is half-fake")** is wrong: the state is `[real command | real measurement]` — nothing
  is fake, and the real position `p` is right there.
- **L3 ("reads its own output")** is mostly wrong: yes `p_des` is its own past, but it *also* reads
  real `p` every step → grounded, normal autoregression.
- **The real problem (§9):** the model *sees* real `p`, but everything that *acts* — the integration
  `p_des += action` and the dynamics **constraint** — operates on the **command `p_des`, never on
  real `p`.** Tight-tracking arm: command ≈ real → fine. Lagging drone: command ≠ real → **drift.**

> **One sentence:** `p_des` isn't "wrong" and the model isn't blind to reality — the system just
> **controls and constrains the command, not the measured position.** L2/L3 blamed the *state*; the
> culprit is the *constraint*.

## 0.3 Does real `v` belong in the tensor — and should we add `v_des`?

**Real `v` (velocity) in the obs: yes, it makes sense — and it's a genuine upgrade over the arm.**
A robot arm (D3IL) is quasi-static and position-controlled, so its obs has **no velocity** at all
(`[des｜c_pos]`). A quadrotor is a **second-order** plant — momentum matters; a fast drone can't
instantly reverse — so the planner *needs* to know the current velocity to propose feasible position
changes. Real `v` (`data.qvel[:3]`) is **real, independent information** the command can't supply, so
including it is correct. (Its *role* is the same as real `p`: conditioning-only — the FM sees it at
t=0 and imagines it for t>0, but the constraint doesn't act on it. That's fine for an input.)

**Should we also add `v_des` (commanded velocity)? No — and your "illegal" instinct is right, but
for a sharper reason than "we can't get it."**

| | source | independent info? | verdict |
|---|---|---|---|
| real `v` | sensor / state-estimate | **yes** — the plant's actual velocity | **keep** |
| `v_des` | `= Δp_des / dt` (a derivative of the action) | **no** — redundant with the action we already output | **don't add to the tensor** |

- **It's not "unavailable."** During data generation `v_des` *does* exist — `traj_fn(t)` returns it
  (`generator.py:231`) and the PID uses it. So the issue isn't that IL can't produce it.
- **It's redundant.** `v_des` is just `Δp_des / dt` — the time-derivative of the position command the
  FM *already* emits. Putting it in the tensor as a separate channel would duplicate the action's
  information and create a consistency burden (the FM's `v_des` channel would have to equal its own
  `Δp_des/dt`, or they'd contradict — exactly the kind of self-consistency the dynamics projection
  already babysits, §9.8).
- **It breaks the command-centric convention** (command = *position*-delta only; §6.2d). Adding a
  commanded-velocity channel turns the action space into "position + velocity command," which neither
  D3IL nor our pipeline is built for.
- **What we actually do is the right call:** at eval, `v_des = action / dt_fm` is **derived** from
  the FM's position action (`eval_fm_uav.py:339`) and handed to the PID as feed-forward — *computed,
  not stored, not a separate FM output.*

> [!NOTE]
> **Where the numbers come from (sim vs hardware), and the `_des` vs real symmetry.**
> - **real `v`**: in sim it's ground truth (`data.qvel`); on **hardware** it is **state-estimated**
>   (an EKF fusing **IMU/accelerometer + position** — GPS/mocap/VIO). You do **not** read body
>   velocity "from the motor" (motors give RPM/thrust, not velocity).
> - **`v_des`** (like `p_des`) is **always a computed command** — there is no "real `v_des`" to
>   measure, exactly as there is no "real `p_des`" (§0.1). So the only *measured* kinematics are `p`
>   and `v`; everything `_des` is authored. Including real `v` and **omitting `v_des`** keeps the
>   tensor = "**all the measured state we have** + **only the position command the action defines**,"
>   which is the clean, non-redundant choice.

---

## 1. Layer 1 — "Units lie" → RESOLVED, not a problem

**Claim:** paper says action = "desired Cartesian velocity (m/s)"; code computes
`des_c_pos[t+1] − des_c_pos[t]` (a position delta, m). Equal only because `dt=1`.

**Critique:** this is a **paper-vs-code labeling** mismatch, not a runtime defect. The code is
self-consistent: actions are position deltas, the integrator is `p_des += action`, and
`config/visual_aligning_eval.yaml:24` explicitly records *"dt=1.0 IS CORRECT: actions are
position deltas, not velocities."* Nothing miscomputes. For the UAV we even keep a separate
`v_des = action / dt_fm` when an actual velocity is needed (`eval_fm_uav.py:339`).

➡️ **Status: closed.** It affects how you *write the paper*, not whether the system *works*. No
catastrophic, no de-facto problem. Leave it; just never call the action "velocity" in our writeup.

---

## 2. Layer 2 — "State is half-hallucinated" → real mechanism, impact depends on tracking error

**Claim:** `s_t = [des_c_pos | c_pos]`; `c_pos` is a sensor read, `des_c_pos` is the FM's own
accumulated output. The paper treats both as "current state".

### 2.1 The "hallucinated" framing is overstated

"Hallucinated" implies *fabricated / disconnected from reality*. But `des_c_pos` is not random —
it is **the exact integral of the FM's own committed actions**, and (crucially) it is the same
quantity the **training data** contained (teleoperator's commanded `des_c_pos`). So at both train
and eval, `des_c_pos` means the same thing: "the position target stream." It is a *legitimate*
state component, not a hallucination. The real question is only whether its *relationship to
reality* (the `des−actual` gap) is the **same at eval as in training**.

### 2.2 For DPCC (the arm): NOT a de-facto problem

You observed: *"the DPCC codebase results return all no tracking error."* Exactly. If `e_t ≈ 0`:

```
des_c_pos_t ≈ c_pos_t   for all t
```

Then the two halves of `s_t` are **redundant copies of the same physical truth**. Conditioning on
a redundant-but-accurate signal is not hallucination — it is harmless duplication. There is **no
distribution shift** (train: des≈actual; eval: des≈actual). The model's input manifold is the
same. So for the arm, Layer 2 is **de facto a non-problem**, and the original doc's alarm is
unwarranted *for that system*. The "half-hallucinated" label only has teeth when `e_t` is large.

### 2.3 For UAV: POSSIBLY real — but conditional and measurable, not automatic

The drone lags, so `e_t > 0`. Does that make Layer 2 a real problem? **Only if eval-time drift
leaves the training distribution.** Two sub-cases:

- **Benign case (likely if data is good):** the expert demos were also flown on a drone with the
  *same* lag dynamics. Then training already contains the `(p_des, p)` lag relationship, and at
  eval the same relationship holds → the conditioning is **in-distribution** → fine. The lag is a
  *feature the model learned to expect*, not a corruption.
- **Pathological case:** the FM at eval commands `p_des` trajectories more aggressive than any
  expert demo (so the drone lags *more* than training ever showed), OR the lag compounds over a
  long rollout. Then `(p_des, p)` enters a region the model never saw → OOD conditioning → the
  classic "crash-then-drift" the original doc attributes to self-reference.

➡️ **Status: open, but empirical.** It is **not** automatically a problem. It is a problem *iff*
the eval `track_err` distribution exceeds the training `track_err` distribution. We have both:
log eval `track_err`, compare to `NOISE_SIGMA`-era training stats. **Decision by measurement, not
by adjective.** Mitigations in §4.

### 2.4 Mitigating circumstance the original doc undersells (UAV)

Our UAV state is **9-D `[p_des | p | v]`** — only **1/3** is the echo; **2/3** (`p`, `v`) is real
sensor data, and `v` (velocity) is something the arm pipeline didn't even have. So the UAV state
is *better grounded* than the arm's 4-D `[des|actual]`, not worse. The phrase "half-hallucinated"
is literally wrong for the 9-D UAV vector — it's "one-third echo, well-grounded."

---

## 3. Layer 3 — "Self-reference never named" → mostly NOT a real problem

**Claim:** the FM is conditioned on `des_c_pos_t` (its own accumulated output), then outputs
`action_t` that updates `des_c_pos_{t+1}`, which feeds the next step. "A closed loop where the
model reads its own write."

### 3.1 Why you (correctly) don't see why it's a problem — because mostly it isn't

Reading your own previous output is the **normal, correct structure** of essentially every
sequential controller and generative model:

- **Receding-horizon MPC**: execute your own planned action, re-plan from the resulting state. Self-referential by design. Not a bug — it's the definition.
- **Autoregressive models (LLMs)**: each token conditions on previously generated tokens. The model "reads its own write" every step. That's how generation works.
- **Any integrator / dead-reckoning**: `x ← x + Δ` literally feeds its own output back.

So "the model reads its own write" is **not, by itself, a defect**. The original doc presents a
*description of normal autoregression* as if it were an *indictment*. That's the "maybe unreal"
part you sensed.

### 3.2 When self-reference IS dangerous — and why this loop is (mostly) safe

A self-referential loop is pathological only when it is **ungrounded** — i.e. no external signal
re-injects reality, so errors compound with nothing to correct them. **This loop is grounded:**

```
        ┌──────────────────────── grounded feedback ───────────────────────┐
        │                                                                   │
   FM ──► action ──► p_des += action ──► env.step(p_des) ──► REAL p, v ──► s_t = [p_des | p | v] ──► FM
        (echo path)                       (physics)        (sensor truth)     │           │
                                                                              echo        reality
```

Every step, **real `p` and `v` re-enter the state from physics**. The loop is not a closed echo
chamber; reality is injected each tick. As long as the FM actually *uses* the real `p`/`v`
channels (it was trained to), accumulated `p_des` error is observable to the model and
correctable. That is the opposite of an ungrounded loop.

➡️ **Status: largely unreal as stated.** Layer 3 mostly re-labels Layer 2's mechanism and dresses
normal grounded autoregression as a sin. The honest one-liner: *self-reference is fine; only
**ungrounded** self-reference is bad, and this loop is grounded by `p`/`v`.*

### 3.3 The ONE real kernel buried inside Layer 3: the binding asymmetry

There *is* a legitimate concern hiding under the rhetoric, and it's worth isolating:

> The dynamics constraint binds **`p_des`** (the echo) to the action, **not `p`** (reality).
> `eval_fm_uav.py:157` → `('deriv',[3,0]),('deriv',[4,1]),('deriv',[5,2])` binds p_des(3,4,5).

This means the **projector enforces self-consistency of the *echo* trajectory**
(`p_des[t+1]=p_des[t]+act`, which is a *tautology* the FM satisfies by construction), and does
**not** enforce that the *physical* trajectory `p` is dynamically feasible for the drone. So the
projector can certify a plan as "feasible" when it is only *internally* consistent, while the real
drone cannot track it. For the arm this is fine (it tracks perfectly, so echo≈physical). **For the
UAV, this is the actual technical risk** — and notice it has nothing to do with "self-reference
being absurd"; it's a **constraint-design** choice. That's the part worth engineering on.

---

## 4. So what (if anything) do we fix? — UAV action items

Do these in order; stop as soon as the data says "non-problem."

### Step 1 — MEASURE before fixing (cheap, decisive)
1. Plot `track_err = ‖p_des − p‖` over each UAV rollout (we already collect `obs_traj` = `[p_des|p|v]`).
2. Compare eval `track_err` distribution against the **training** `(p_des − p)` distribution
   (from `uav_expert_data_collect`). 
   - If eval ⊆ train support → **Layer 2/3 are non-problems for UAV. Stop.** Write that down.
   - If eval drifts beyond train support → proceed.

### Step 2 — If drift is real, the cheapest fix: re-anchor the integrator
The unbounded-echo failure is classic integrator windup. Standard fix — periodically pull `p_des`
back toward reality so the gap can't grow without bound:
```python
# instead of: p_des = p_des + action
p_des = (1 - α) * (p_des + action) + α * p      # α small, e.g. 0.05–0.2; α=0 = current behavior
```
This keeps `p_des` honest without changing the model. (Validate: does it move the model OOD? It
shouldn't if α is small and training `e_t` was small.)

### Step 3 — Fix the binding asymmetry (the Layer-3 real kernel)
Add a *tracking-feasibility* awareness so the projector doesn't bless physically-untrackable
plans. Options, increasing effort:
- (a) Add a soft constraint/penalty coupling `p` to `p_des` (bound `‖p − p_des‖`) so projected
  plans respect that the drone lags.
- (b) Bind the dynamics on `p` (reality) with a *learned/identified* lag model instead of binding
  the tautological `p_des`. Heavier — needs a drone dynamics model.

### Step 4 — If OOD conditioning is the culprit: close the train/eval gap
- Retrain with **DAgger-style** feedback: include the FM's *own* eval-time `(p_des, p)` rollouts
  in the dataset so the model sees its own drift distribution.
- Or condition on **`p`/`v` more strongly** (they're real); de-emphasize `p_des` as the anchor.

> [!TIP]
> **Most likely outcome:** Step 1 shows UAV `track_err` is bounded and in-distribution for short
> rollouts, making Layers 2/3 non-problems *in practice* — exactly as they are for the arm — and
> the real lever for "FM keeps planning / drift" is elsewhere (no termination check, horizon, or
> the binding asymmetry of Step 3), **not** the philosophical "self-reference."

---

## 5. "The three things maybe unreal" — explicit reality rating

Per your instinct, here is how real each claim is, plainly:

| # | Original framing | Reality rating | Plain restatement |
|---|---|---|---|
| **L1** | "Units lie" — *absurd* | 🟢 Real but **resolved & benign** | A label mismatch reconciled by `dt=1`. Paper wording only. |
| **L2** | "Half-hallucinated" — *absurd* | 🟡 Real mechanism, **impact conditional & measurable** | State has an echo component. Harmless when tracking error ≈ 0 (arm); *possibly* relevant for UAV — check `track_err`. Not "hallucination." |
| **L3** | "Reads its own write" — *absurd* | 🔴 **Mostly rhetorical / unreal as stated** | Normal grounded autoregression (like MPC/LLMs). Safe because real `p,v` re-enter each step. The only real sub-issue is the **constraint binds the echo, not reality** — a design choice, fixable, unrelated to "self-reference being bad." |

**Bottom line for the writeup:** keep Layer 1 as a footnote correction; present Layer 2 as a
*measurable distribution-shift question* (with the `track_err` evidence), and **drop or heavily
de-escalate Layer 3** — replace the "self-reference is absurd" narrative with the precise,
defensible point about **constraint-binding asymmetry**. The dramatized version weakens the
analysis by attacking a structure (autoregression) that is standard and correct.

---

## 6. The Deep Reason — One Category Error at the Seam (Philosophical Synthesis)

> §§1–5 dissect the three claims *mechanically*. This section asks the question above all of them:
> **why does grafting H8 (Janner) onto 9D (D3IL) produce exactly these three problems, and not
> others?** The answer is that all three are not three bugs — they are **three faces of a single
> category error** created the instant the two philosophies are stitched together. Once you see
> the category error you can predict the three problems from first principles, predict that the
> arm survives them, predict that the UAV won't, and derive "why `p_des` not real `p`" without
> looking at any code. (Mechanical graft detail: [WHY_FM_KEEPS_PLANNING.md](./WHY_FM_KEEPS_PLANNING.md) §§10, 12.)

### 6.1 The two parents are not two architectures — they are two *worldviews*

`H8` and `9D` are not neutral hyperparameters we happened to pick. Each is the fingerprint of a
complete, internally-consistent answer to *"what is a state, and what does the model control?"* —
and the two answers are incompatible.

| | **Janner / Diffuser** (gives us **H8**) | **D3IL** (gives us **9D**) |
|---|---|---|
| Worldview | **Planning in reality** (trajectory optimization) | **Cloning a command stream** (behavior cloning) |
| What the model produces | a *trajectory through real states* | a *reactive map* obs → next command |
| What is a "state"? | a point in the **plant's** real phase space | whatever the **logger** recorded |
| How many states? | **one**, and it is real | **two**: `des` (commanded) + `c_pos` (actual) |
| What is the "action"? | whatever the **environment integrates** | the **increment of the human's command** (`Δdes`) |
| Dynamics model `f`? | the env's true physics — never modeled, just re-observed | **none** — BC needs no `f` |
| How is it grounded? | **structurally**: every replan re-reads `env.step()` | not grounded — it has no loop to ground |
| Time horizon | **H steps of foresight** (the whole point) | **window = 1**, single action (no horizon) |

`H8` imports the *planning* worldview (D3IL's own diffusion has no horizon at all). `9D` imports
the *cloning* worldview's signature artifact: the **two-state split** `[des | c_pos]`, which exists
only because a position-controlled robot's logger records command and result separately. Neither
artifact is wrong **in its own house**. The trouble is they come from different houses.

### 6.2 The trajectory tensor IS the chimera — its two axes are the two parents

```
                 FEATURE axis  D = 9  ← D3IL  (act | des | c_pos : a logged command/actual pair)
                 ┌───────────────────────────►
        TIME     │ x[0,:]
        axis     │ x[1,:]
        H = 8    │  ...                         every cell x[h,d] simultaneously means
        ↑Janner  │ x[7,:]                       "planned future step h" (Janner) AND
        (a plan) ▼                              "command/actual feature d" (D3IL)
```

The **rows** are Janner (a plan unrolling into the future). The **columns** are D3IL (a logged
command-stream snapshot). The tensor is Frankenstein's body; `H8` and `9D` are the two grafted
limbs. The model must *plan* (rows) using a *cloned command stream* (columns) as if that stream
were a real dynamical state.

### 6.2a Reverse-engineering the parents — exact tensors & models (read from `/workspaces/diffuser` and `/workspaces/d3il`)

Before judging the graft, pin down *exactly* what each parent denoises and with which network.
Both were read directly from source.

**Janner Diffuser** (`/workspaces/diffuser`):

| Item | Value (verbatim) | File |
|---|---|---|
| Denoised tensor | `trajectories = concat([actions, observations])` → `[H, action_dim + observation_dim]` | `datasets/sequence.py:88` |
| `transition_dim` | `observation_dim + action_dim` — **both real env quantities** (D4RL MuJoCo state + torque) | `models/diffusion.py:53` |
| Network | `TemporalUnet` — 1-D conv U-Net **over the time axis H** | `models/temporal.py:49` |
| Horizon `H` | **32** (locomotion planning), 4 (value) | `config/locomotion.py:25` |
| Conditioning | `{0: observations[0]}` — pin the **real** obs at t=0 (and goal at t=H−1) via inpainting | `datasets/sequence.py:77`, `models/diffusion.py:164` |
| Dynamics model | **the environment's true physics** — never written down, re-observed via `env.step()` | `plan_guided.py` |

**D3IL** (`/workspaces/d3il`, avoiding) — the relevant model is the **action-level DDPM agent**:

> [!NOTE]
> D3IL is a **method-agnostic benchmark with ~11 IL methods** (BC, BeT, IBC, ACT, CVAE, BeSo, and
> several diffusion variants). DPCC's `[des｜actual]` + `Δdes` format was authored for D3IL's
> **DDPM** path: `ddpm_agent.py` → `Diffusion` (`diffusion_policy.py`) → `DiffusionMLPNetwork`
> (`diffusion_models.py`). D3IL also ships other diffusion denoisers — `DiffusionTransformerNetwork`
> (GPT-style) and `DiffusionEncDec` (the encdec/VAE-transformer, the parent of our
> `ddpm_encdec_vision`) — but **every one of them is action-level**: they denoise *actions*
> conditioned on obs (possibly an action *chunk*), and **none denoise a Janner state-trajectory, none
> inpaint, none project.** So the conclusion below holds across all D3IL diffusion variants, not just
> the MLP one.

| Item | Value (verbatim) | File |
|---|---|---|
| Denoised tensor | **the ACTION only** — `output_dim = action_dim` (2-D avoiding / 3-D aligning) | `models/diffusion/diffusion_models.py:50` |
| Network (DDPM agent) | `DiffusionMLPNetwork` — a plain **MLP**, **no time axis** (siblings: `DiffusionTransformerNetwork`, `DiffusionEncDec` — also action-level) | `models/diffusion/diffusion_models.py:20` |
| How obs enters | as **conditioning only**: `x = cat([x, t, state, goal])` at the MLP input | `diffusion_models.py:102/104` |
| Sample shape | `(B, action_dim)` (or a short action chunk) — **conditioned on `state`** | `diffusion_policy.py:212–215` |
| `window_size` | **1** (single-timestep context for the MLP agent) | `dataset/avoiding_dataset.py:24` |
| The `[des｜actual]` vector | `concat(robot_des_pos[:2], robot_c_pos[:2])` = 4-D — **the conditioning `state`** | `dataset/avoiding_dataset.py:57` |
| Action | `vel_state = robot_des_pos[1:] − robot_des_pos[:-1]` = **Δdes** | `dataset/avoiding_dataset.py:59` |

> [!IMPORTANT]
> **No D3IL method had a 6-D / 9-D *trajectory* tensor.** The DDPM agent denoises the **action**
> (2-D/3-D) with an **MLP**, and the `[des｜actual]` vector is **conditioning** — numbers the model
> reads, never a denoised state, never inpainted, never given a dynamics model (and the transformer/
> encdec variants are action-level too). The "9-D-over-H8 trajectory" is **100 % Janner's container**
> (`[act｜obs]` over horizon). D3IL contributed only the **channel definitions** — the `[des｜actual]`
> split (from its IK logger) and the `action = Δdes` convention — which in D3IL's own house are
> *conditioning-only*.

### 6.2b The verdict — a dimensionally-perfect API bolted over a semantic category error

So did DPCC find a clean connector or commit a control/math bug? **Both — that is precisely the
trap.**

**The API connection is real and dimensionally perfect.** Janner's `transition_dim =
observation_dim + action_dim` with layout `[act｜obs]`. DPCC sets `obs_dim = 6` (`[des_c_pos｜c_pos]`)
and `action_dim = 3` → `transition_dim = 9`, layout `[act｜des｜c_pos]`. It drops into `TemporalUnet`
without a single line of model change, because the first/last `Conv1d` map `transition_dim ↔ dim`
and **the convolution is agnostic to what each channel means** (§ the "two-boundary principle").
The connector *fits*. This is not random fabrication — it is a deliberate, type-checking graft.

**But the semantics it connects are incompatible**, and that is where the bug lives:

| Slot | Janner fills it with… | DPCC fills it with… (from D3IL) | Consequence |
|---|---|---|---|
| `obs` channels of the denoised trajectory | a **real Markovian state** with true env dynamics | a **conditioning-only command stream** `[des｜c_pos]` | inpainting now pins a value that is **not ground truth** (L2) |
| dynamics model for the projection | the env's real physics (never needed inside the model) | **none existed** → DPCC must **invent** `f` | only the **tautology on `des`** closes (L1 units, L3 binding) |
| the conditioning loop | re-reads **real** `env.step()` obs | re-reads the model's **own** `des` | **self-reference** (L3) |

The three "absurdities" are exactly the byproducts of promoting a *conditioning-only* vector
(D3IL) into a *denoised-state* slot (Janner). **The connector is sound; the thing pushed through it
is mis-typed.** "Perfect API, illegitimate semantics."

**On `H8` specifically:** Janner plans at `H=32`; DPCC uses `H=8`. That is **the minimum legal
horizon** for a 4-level U-Net (`dim_mults=(1,2,4,8)` ⇒ three stride-2 downsamples `8→4→2→1`). So
`H8` = "keep Janner's 4-level `TemporalUnet`, take the **shortest horizon the architecture
allows**" — a reactive compromise sitting between Janner's planning `H=32` and D3IL's action-level
`H=1`. The **time axis is purely Janner's** (D3IL has no time axis to contribute); D3IL only shapes
the **feature axis**. The chimera of §6.2 is now exact: *rows = Janner's horizon, columns = D3IL's
conditioning vector promoted to state.*

**Bottom line of the reverse-engineering:** the graft is a competent, dimensionally-valid
engineering move that found a genuine API seam — **not** a careless hack. The defect is not a
loose wire; it is a **type error one level up**: a controller reference (legitimate as D3IL
conditioning) was promoted to a dynamical state (Janner's denoised+inpainted+projected variable).
That promotion is silently correct **iff the controller tracks perfectly** (`des ≈ actual`), which
is why it passed on the arm and fails on the UAV (§6.5–6.6).

### 6.2c Origin audit — where `[des｜c_pos]` (4D/6D) is born, and why DPCC tiled it to H8

You asked the sharpest version of the question: **which D3IL *model* defines the 4D/6D
`[des｜c_pos]` format, and why did DPCC naively stack it to H8?** The honest answer from the source:
**no model defines it — the *dataset* does — and the H8 stacking is mechanically valid but
semantically uncritical.**

**(1) The format is a benchmark-wide *dataset* convention, not a model's design.**
`[des｜c_pos]` + `action = Δdes` is built identically in **every D3IL task dataset**
(`avoiding`, `aligning`, `pushing`, `sorting`, `stacking`) on top of the shared
`TrajectoryDataset` base (`base_dataset.py:7`). It is consumed unchanged by **all ~11 IL methods**
(DDPM, BeT, IBC, ACT, BeSo, …). So it is **method-agnostic**: it predates and is independent of the
diffusion model. It is born one layer lower — in the **robot hardware logger**:

```
IKControllers.py:75,309   robot.des_c_pos = desired_c_pos     # the position-controlled IK setpoint
logger.py:71,74           record BOTH c_pos (actual) AND des_c_pos (commanded)
avoiding_dataset.py:54-59 input_state = concat(des, c_pos);  action = des[1:] − des[:-1]  (= Δdes)
```

**Why both `des` and `c_pos`?** Because the action is a **delta on the command** (`Δdes`), and the
robot is **position-controlled**. For delta-action IL on a position-controlled arm, the natural
observation is **"command + feedback"**: `des` is the baseline the next `Δdes` is added to; `c_pos`
is the plant feedback. That is a sound *conditioning* choice for an action policy — and in D3IL it
is *only ever conditioning* (§6.2a). The split is a **control-data artifact**, not a planning state.

**(2) The visual twist: D3IL visual *drops* `c_pos`; DPCC *re-adds* it for the projector.**
D3IL **visual** aligning shrinks the state to **`des` only (3D)** (camera carries box/target;
`obs_dim:3`). DPCC's visual dataset then **re-adds `c_pos`**, by its own admission, *to give the
projector a real-position channel*:

```python
# diffuser_visual_aligning/datasets/sequence.py (ParityAligningDataset)
#   obs_6d  = concat(robot_des_pos, robot_c_pos)        # 6D = [des | c_pos]
#   actions = robot_des_pos[1:] - robot_des_pos[:-1]    # Δdes
#   TRAJ_DIM = 9   # [act | des | c_pos]
#   docstring: "des_c_pos alone (6D) would project on command targets instead of
#               real end-effector positions, violating the DPCC physical contract."
```

So the 9D is **assembled, not inherited**: `des` (from D3IL's logger convention) **+** `c_pos`
(re-added for the projection) **+** `act=Δdes`, arranged in **Janner's `[act｜obs]` layout**. The
format is a *deliberate composite* aimed at the projector — which is also exactly where it goes
wrong on the UAV (the arm projects on real `c_pos`; the UAV projects on the `p_des` echo, §6.7).

**(3) Why "expand to H8" — the windowing mechanism, audited.**
DPCC's projection is defined over a **trajectory**, so it needs Janner's `[H, transition_dim]`
tensor. D3IL data is stored as **full per-timestep episodes**, so DPCC simply **slides a length-H
window** over consecutive `(act, des, c_pos)` steps (`ParityAligningDataset._make_indices`,
`horizon=8`). The "expansion" is just *windowing a sequence that was already a sequence* — that part
is **mathematically sound** (episodes genuinely are time series; this is the standard Janner move).

Why **8** specifically (audited reasons, in order of force):
1. **Architecture floor.** `dim_mults=(1,2,4,8)` ⇒ three stride-2 downsamples (`8→4→2→1`) ⇒ `H`
   must be divisible by 8. **8 is the smallest legal horizon** for the inherited 4-level U-Net.
2. **Stay reactive.** D3IL is action-level (`H=1`); Janner plans long (`H=32`). H8 is the shortest
   planning window — closest to D3IL's reactivity while still giving the projector ≥1 step pair to
   constrain.
3. **Cheap inheritance.** Keeping `dim_mults=(1,2,4,8)` means they didn't redesign the U-Net depth;
   H8 falls out of that default.

**Audit verdict.** The windowing is **not** the naive part — turning episodes into H-windows is
correct and standard. The naive part is **what got windowed**: D3IL's *conditioning-only*
`[des｜c_pos]` was tiled into Janner's *denoised + inpainted + projected* state channels **without
re-asking whether `des` (a command) belongs there.** H8 itself is a forced/architectural number;
the latent defect rode in on the *channel semantics*, not on the horizon length. In one line:
**the format comes from a hardware logger, the H8 from a U-Net constraint, and the bug from gluing
the two without re-typing `des` from "conditioning" to "state."**

### 6.2d The ultimate "why `p_des` not `p`" — it's turtles down to the gamepad (data-generation audit)

This is the deepest layer, and it finally answers the question the whole document circles: **why is
the policy conditioned on (and integrating) `p_des`, never real `p`?** §6.7 answered it from the
*planning* side ("Janner needs an `f`, D3IL has no plant model"). Here is the *data-generation*
side — and the two meet.

**(1) The demonstrations are position-command teleoperation.** D3IL data is recorded by a human
driving a **gamepad** (`README:111`, `gamepad_control/record_data.py`). The human's input *is* a
Cartesian **setpoint**: `IKControllers.py:127` `self.desired_c_pos = action`; the IK controller then
drives the **actual** toward it (`:58/184` `xd_d = desired_c_pos − current_c_pos`) and writes
`robot.des_c_pos = self.desired_c_pos` (`:75`). The logger stores **both** the command `des_c_pos`
and the result `c_pos` (`logger.py:71–74`). D3IL even names the gap: `tracking_error =
Σ|desired_c_pos − current_c_pos| > 0.01` (`IKControllers.py:278`).

**(2) Therefore the only expert "action" that exists is the command increment `Δdes`.** The human
never produced torques, velocities, or real-position deltas — only a stream of Cartesian
**commands**. So the imitable action is `Δdes = des[t+1] − des[t]` (`avoiding_dataset.py:59`). This
is not a modeling preference; it is **the only thing in the data**.

**(3) A delta-on-command policy is structurally forced to carry `des`.** If your action is `Δdes`,
you integrate it in **command space**: `des[t+1] = des[t] + Δdes`. To output the next increment you
must know the current command baseline — so **`des` must be in the state**. `c_pos` is added as
*feedback* (observation-only), never as the integration base. That is exactly the unified format:

| Task | DOF | Robot state | Why |
|---|---|---|---|
| avoiding | planar (xy) | `[des_xy｜c_xy]` (4D), `Δdes_xy` (2D) | command baseline + feedback for a 2-D delta policy |
| aligning | spatial (xyz) | `[des_xyz｜c_xyz]` (6D) **+ box/target**, `Δdes_xyz` (3D) | same, plus objects (no camera) |

So `[des｜c_pos]` is **not invented arbitrarily** — it is the *minimal* observation for cloning a
**position-command teleoperator on a position-controlled robot**: "what did I command (`des`), what
actually happened (`c_pos`)."

**(4) Why visual aligning drops `c_pos` (and box/target) but KEEPS `des` — and why that is NOT
backwards.** It *looks* backwards ("keep the command, throw away the real position??"), but it is
minimal-correct. The state's only job is to supply what the policy **can't get elsewhere**:

| Quantity | In the pixels? | Needed by the `Δdes` action? | Verdict |
|---|---|---|---|
| `c_pos` (actual EE pos) | **Yes** (camera sees the arm) | **No** (action is `Δdes`, not `Δc_pos`) | **drop** (doubly redundant) |
| box / target | **Yes** | No | drop |
| `des` (internal setpoint) | **No** (a camera can't see an internal command) | **Yes** (`des += Δdes` needs the baseline) | **keep** (doubly required) |

So `des` is kept for *two* reasons (invisible to camera **and** it is the action's accumulator) and
`c_pos` is dropped for *two* (camera provides it **and** the action never needs it). `aligning_
vision_config: obs_dim:3`. The tell: **`des` is retained not because it is a good plant state, but
because it is the policy's own command channel that nothing else can provide.** That "feels
backwards" precisely because the paradigm is **command-centric** (gamepad + `Δdes`), so the command
is the privileged channel and the camera fills in everything *real* (`c_pos`, boxes). Harmless on a
tight-tracking arm (`des≈c_pos`); the same DNA that becomes the category error on a lagging drone.

**(5) The conclusion — `p_des` is the accumulator of a command-delta policy, not a perception of
the world.** Putting `p_des` in the state is **forced** by "action = command increment." You
literally cannot run a `Δdes` policy without carrying `des`. To condition/integrate on real `p`
instead, you would have to **redefine the action as `Δp` (a real-position increment)** — and that is
blocked three ways:

- **No `Δp` expert exists.** The gamepad only ever produced `Δdes`. There are no real-position-delta
  demonstrations to clone. (On a tight-tracking arm `Δp ≈ Δdes`, so it never mattered; on a lagging
  drone they differ and you have nothing to learn from.)
- **Integrating commands onto feedback is unstable.** `des[t+1] = p[t] + Δdes` feeds the tracking
  error back into the command every step → the controller chases its own lag → wind-up/drift. D3IL
  deliberately integrates in **command space** (`des += Δdes`) and keeps `c_pos` observation-only to
  keep the command loop clean.
- **Mapping a desired real-position change to a command needs a plant model** (command→motion) that
  BC/D3IL never had — the same missing `f` as §6.7 (the streetlight).

> [!IMPORTANT]
> **Turtles down to the gamepad.** `p_des`-not-`p` ⟸ action is `Δdes` ⟸ the only expert signal is a
> human's **position command** ⟸ demos are **gamepad teleoperation on a position-controlled arm**.
> The "category error" (§6.3) and "bind the tautology" (§6.7) are the *planning-side* and
> *control-side* shadows of this one *data-side* fact. DPCC inherited `p_des` because it trained on
> this data and accepted its action definition; we inherited it because we copied DPCC. None of the
> three layers were chosen — they were **downstream of how a human held a joystick.**
>
> **And this is exactly why the arm is fine and the UAV is not:** on the arm `Δdes ≈ Δp` and
> `des ≈ c_pos` (tight IK tracking, the `>0.01` threshold), so command-space integration is
> indistinguishable from real-state integration. A quadrotor breaks `Δdes ≈ Δp` (lag,
> underactuation), so integrating in command space with no plant model lets `p_des` walk away from
> `p` — the drift this whole file is about. **The fix is therefore not "condition on `p`" alone; it
> is "stop having a command-delta-without-a-plant-model action," i.e. give the projector a real
> model (§6.9 Re-ground / SafeFlowMPC §7) so the command is tied back to physics.**

### 6.2e Our UAV's turtles bottom out at a SCRIPT, not a gamepad (Gen11 dataset audit)

§6.2d ended "turtles down to the gamepad" for D3IL. **For our UAV the bottom turtle is different —
and it makes the inherited format *less* justified, not more.** Audited from `uav_expert_data_collect`:

```python
# generator.py :: run_trial
traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)  # analytic path generator
p_des, v_des, a_des, yaw_des = traj_fn(t)        # "expert" = a CLOSED-FORM curve f(t)
u = pid.compute(p, q, v, om, p_des, v_des, ...)  # CascadedPID tracks it; mujoco evolves p
steps.append({'p': p, 'v': v, 'p_des': p_des})
# dataset_writer.py: obs = [p_des | p | v] (9D); actions = Δp_des
#   comment: "Matches D3IL: vel_state = des_c_pos[1:] - des_c_pos[:-1]"
```

We built the **same `[command | real]` tensor** (`[p_des｜p｜v]`, action `Δp_des`) and the code says
so explicitly ("Matches D3IL"). But what the command *is* differs at the root:

| | D3IL | Our UAV (Gen11) |
|---|---|---|
| Source of `des`/`p_des` | **human gamepad** → IK setpoints | **scripted analytic path** `traj_fn(t)` (`empty/corridor/s_curve/pillar_path`) |
| Low-level controller | IK (arm) | `CascadedPID` (drone) |
| What is imitated | a **human's** command stream (intent, multimodality) | a **deterministic curve we already wrote** + PID tracking |
| Is `des` a real *demonstration*? | **Yes** — human intent lives in it | **No** — it is a closed-form function's output |
| Does the command-centric format earn its keep? | **Yes** — the human command *is* the demo | **No** — inherited from D3IL; `p_des` holds no human intent |

> [!IMPORTANT]
> **D3IL's `[des｜c_pos]` is *justified*; ours is *inherited baggage*.** D3IL's whole point is cloning
> irreplaceable **human gamepad** behavior, so `des` is privileged because it **is** the demonstration.
> We copied that format, but our "command" is **synthetic** — a parametric path we still hold in
> closed form (`traj_fn`). So we took on **every downside** of the command-centric design
> (self-reference, the category error §6.3, `p_des≠p` drift on a lagging drone) for **none of the
> upside** (there is no human intent to preserve). The tell is `dataset_writer.py: NOISE_SIGMA=0.02`,
> which adds a per-episode offset *"to thicken the data manifold"* — an explicit admission that the
> scripted demos are too clean/deterministic to be a natural demonstration distribution; a real
> gamepad dataset never needs that.
>
> **The one legitimate reason an FM still fits us:** *multimodality across homotopy classes* (L/R/C
> routes around obstacles) is a genuine distribution worth learning. That justifies a generative
> planner — but it does **not** justify the `[p_des｜p]` *state format*, which we carried over from a
> human-imitation benchmark into a distill-a-scripted-planner setting where its premise (human
> intent in the command) does not hold.

### 6.2f In our dataset the `p_des − p` gap is *real physics*, not a demonstration artifact — and that flips the meaning of the split

You put your finger on it: **our UAV dataset genuinely contains `p_des ≠ p`**, and the reason is
*physical*, not human and not numerical. Each step records the **reference** `p_des = traj_fn(t)`
and the **actual** `p = data.qpos[:3]` after the PID + physics step (`generator.py:230–250`). The
gap between them is the quadrotor's **closed-loop tracking error**, caused by:

1. **Cascaded-PID transient lag** — `CascadedPID` turns position error into a *desired tilt*, then
   rotor commands (`pid.compute(p, q, v, om, p_des, …)` → `data.ctrl[:4]`). The cascade takes time
   to settle, so `p` lags `p_des`.
2. **Quadrotor underactuation** — a drone must **tilt (change attitude `q`) to translate**, so
   lateral position physically *cannot* track a command instantly; position trails attitude.
3. **Momentum / inertia** and **thrust saturation** — when the reference is aggressive the rotors
   saturate (`n_clip` counts it), and the drone falls further behind.

> [!IMPORTANT]
> **Same tensor, opposite meaning.** For D3IL **gamepad IL**, the `[command｜real]` split is a
> **demonstration split** — `des` is the human's intent, `c_pos` is feedback (and the arm tracks so
> tightly that `des ≈ c_pos`, so the gap is ~0). For **our UAV** the `[p_des｜p]` split is a
> **physics split** — `p_des` is a *scripted* reference and `p` is the drone's *lagged dynamical
> response*, so the gap is **real and structured**. We do **not** need imitation learning to justify
> the split (there is no human to imitate); what we need is to read it as **"reference vs the plant's
> actual tracking,"** which is a genuinely meaningful, physically-grounded signal.

**Why this matters — two consequences:**

- **(good) The category error is *not masked* in our training data.** On the arm `des ≈ c_pos`, so
  the FM trains on a manifold where command and state coincide and the defect is invisible
  (§6.5). Our FM trains on data where `p_des` and `p` genuinely diverge — so the real position `p`
  and the gap are **in-distribution information the model could actually use** (this is the benign
  side of §2.3: the lag the FM sees at eval was present in training too).
- **(the catch) We collected the meaningful channel and then *discard it in the constraint*.** The
  dataset hands the FM real `p` **and** a real `p_des−p` gap — but the PCC projector binds the
  **tautology** on `p_des` (`deriv` on indices 3–5) and never constrains `p` (§6.7). So we paid to
  generate genuine tracking physics and then told the projector to ignore the one channel that
  carries it. The fix is not more data — it is to **let the constraint/condition use the real `p`**
  the dataset already contains (§6.9 Re-ground; SafeFlowMPC §7).

**The deciding, measurable question** (ties to §2.3): is the `p_des−p` gap the FM produces at eval
**within the training gap distribution?** We have both — the dataset's per-step `p_des−p` histogram
vs eval `track_err` (`eval_fm_uav.py:355`). If eval ⊆ train support, the split is benign in-
distribution physics; if eval drifts beyond it, the scripted-reference→self-generated-reference
shift (diffusion jitter on a lagging plant) is the culprit. **Decide by that histogram, not by
adjectives.**

### 6.3 The single root: a **category error** — the controller's *reference* placed in the *plant-state* slot

Control theory keeps two things ontologically distinct:

- **Plant state** `p` — *where the drone IS*. An **output of physics**. **Measured.**
- **Controller reference** `p_des` — *where we told it to go*. An **input we choose**. **Commanded.**

A reference is a *cause we author*; a state is an *effect the world returns*. They live on opposite
sides of the causal arrow.

> [!IMPORTANT]
> **The graft puts the controller's reference (`des`/`p_des`, from D3IL) into the plant-state slot
> of Janner's planner.** It asks a *chosen input* to play the role of a *measured output*. That is
> the category error from which all three problems fall out. `des_c_pos` is an **output of the
> model masquerading as an input from the world.** The error exists in **neither parent** — D3IL
> never *planned* with `des`; Janner's pinned state was always real. It is *manufactured at the
> seam*.

### 6.4 The three "layers" are three faces of that one error

| Layer | What it really is | Same root |
|---|---|---|
| **L1 — "units lie"** | `Δdes` is a **reference increment**, not a **plant velocity**. Janner's action *is* a plant input; D3IL's is a command delta. | reference ≠ state ⇒ its rate ≠ a physical velocity |
| **L2 — "half-hallucinated"** | the state-slot holds `des` (a reference), so half the "state" is **not a measurement**. Janner's inpainting assumes the pinned value is real; here half isn't. | reference in the state-slot ⇒ "state" is half-authored |
| **L3 — "self-reference" + binding** | the receding-horizon loop (Janner) re-reads a slot holding the model's own reference (D3IL) → **reads its own write**; and the projector enforces `f` on `des` (controllable, tautological) not `p` (unmodeled). | a reference is *model-generated* ⇒ feeding it back is self-reference; only the reference admits a tautological `f` |

Not three independent mistakes — **one** ontological confusion with three shadows. That is why
fixing them piecemeal feels like whack-a-mole: they share a root.

### 6.5 Why the ARM survives: a *load-bearing coincidence*

The category error is **invisible whenever `reference ≈ state`** — i.e. whenever the controller
tracks perfectly. A rigid, position-controlled arm is ~kinematic: **command now = result now**, so
`des_c_pos ≈ c_pos`. Then L1: `Δdes ≈ Δp` ("velocity" ≈ true); L2: reference half ≈ real half
(state ≈ fully real); L3: fed-back reference ≈ reality (loop effectively grounded; binding `des` ≈
binding `p`). The arm doesn't *solve* the error — it **anesthetizes** it. DPCC arm results are
clean not because the design is sound but because the test body hides the unsoundness.

> [!NOTE]
> **The arm actually has TWO grounding mechanisms, not one** (see §9.7). Besides tight tracking
> (`des≈c_pos`), the arm **binds real `c_pos`** in the dynamics constraint, and the projector's
> `skip_initial_state` **hard-resets `c_pos[0]` to the *measured* position every step**. So even a
> drift would be yanked back to reality each planning step. The UAV runs the identical reset but
> **binds `p_des`**, so the reset lands on its own command and grounds nothing — the arm's second
> safety net is silently absent on the UAV. This is *why* the same code is robust on one body and
> not the other, beyond the tracking-error story.

### 6.6 Why the UAV breaks: transplant *removes the anesthetic*, it doesn't add a bug

A quadrotor is a *dynamic* plant (momentum, underactuation, attitude lag) ⇒ command ≠ result ⇒
`p_des` and `p` diverge (`track_err > 0`, which we log). The instant `reference ≠ state`, all three
shadows snap into focus at once:

```
   tracking error e_t = ‖p_des − p‖
   e_t ≈ 0  (arm)   →  category error MASKED   →  3 problems invisible  →  "DPCC works"
   e_t  > 0  (UAV)  →  category error EXPOSED   →  3 problems visible    →  "FM keeps planning / drifts"
```

> [!IMPORTANT]
> The UAV did not introduce three new bugs. It **withdrew the one coincidence (`e_t≈0`) that hid
> three old ones.** The graft was only ever valid on the condition "the plant tracks its reference
> perfectly." So the "simple mix" does cause severe problems on transplant — but the sharp
> statement is: **the mix is conditionally correct, and the condition is a property of the robot,
> not of the code.**

### 6.7 "Why `p_des` and not real `p`?" — the philosophical answer (deeper than the mechanical one)

The mechanical answer (WHY_FM §6): binding `p_des` is a tautology (`p_des[t+1]=p_des[t]+act`),
binding real `p` makes the constraint unsatisfiable. The **deeper** reason:

- From **Janner**, the graft inherited a *need for a dynamics model* `f` (the projection constraint
  `s_{t+1}=f(s_t,a_t)` is meaningless without one).
- From **D3IL**, it inherited the *absence of any plant model* (BC never had one).
- The only `f` writable **without modeling the plant** is the identity integrator on the thing the
  model *already controls* — its **own reference** `p_des`. Real `p` obeys drone physics the graft
  never imported, so no closed-form `f` for `p` exists here.

> Binding `p_des` is the **streetlight effect**: search under the lamppost (the tautology you can
> write), not where the keys are (the plant you care about). It constrains the variable it can
> *describe* (the reference), not the one it wants to *constrain* (the state). Forced by parentage:
> **Janner's demand for `f` − D3IL's plant model = bind the reference.** Never a control-quality
> choice.

### 6.8 Why this literally makes "**the FM keep planning**"

A receding-horizon planner should *converge*: as the goal nears, successive plans agree and the
loop quiets. But the slot it re-plans from holds, in part, the model's **own reference**, not the
plant. When `p_des ≠ p`, every replan starts from a target the body hasn't reached, so the planner
keeps issuing corrections toward a point the drone keeps lagging behind. **The loop re-plans
against its own intentions instead of the world, so it never settles** — a fiction can never be
reached. The title symptom is the category error in motion.

### 6.9 The three exits — each a decision about which parent to obey

| Exit | Move | Which worldview it commits to |
|---|---|---|
| **Re-ground** | put the *plant* in the state-slot (condition on real `p`/`v`; model the drone) + retrain | go **full Janner** — plan in real state space |
| **Re-anchor** | periodically collapse reference→plant: `p_des ← (1−α)(p_des+act)+α·p` | **force `e_t≈0`** to stay true by fiat (keep the arm's anesthetic alive) — see §4 Step 2 |
| **Terminate** | detect when the coincidence breaks (`track_err` large) and stop | **admit** the system is valid only in the `e_t≈0` regime — the cheap, honest patch |

### 6.10 Synthesis (one paragraph)

`H8` is Janner's *planning* worldview; `9D` is D3IL's *command-cloning* worldview; the DPCC graft
welds them at the dataset boundary. The weld forces a single **category error** — the controller's
**reference** (`des_c_pos`/`p_des`) is made to sit in the planner's **plant-state** slot, an
authored input wearing the mask of a measured output. The three layers (mislabeled units, half-real
state, self-referential loop + reference-binding) are three projections of that one error. A
perfectly-tracking arm makes `reference ≈ state`, so the error is numerically invisible and "DPCC
works." A quadrotor makes `reference ≠ state`, withdrawing that anesthetic and exposing all three at
once — which is why the FM keeps planning and the drone drifts. And binding `p_des` over real `p` is
not a choice but a forced consequence: the graft inherited Janner's demand for a dynamics model and
D3IL's lack of one, leaving only the tautology on its own reference. **The graft is conditionally
correct, and the condition (`tracking error ≈ 0`) is a property of the robot. Transplanting to the
UAV didn't add bugs; it removed the coincidence that hid them.**

---

## 7. How a Published Sibling (SafeFlowMPC, ICRA 2026) Avoids All Three Accusations

> [SafeFlowMPC](https://arxiv.org/abs/2602.12794) (Oelerich et al., ICRA 2026; repo
> `/workspaces/SafeFlowMPC`) is the **closest living relative** of DPCC / our FM-PCC: it is
> *flow-matching trajectory generation + a projection at every flow step*. If DPCC = "diffusion +
> SLSQP projection" and we = "flow + PCC projection," then SafeFlowMPC = "flow + **NMPC safety
> filter**." Same family, same skeleton — but it makes the **opposite choice at exactly the seam
> where §6's category error lives.** That makes it a near-perfect natural experiment for our
> thesis: if the category error were unavoidable, SafeFlowMPC would have it too. It does not — and
> seeing *how* it dodges all three layers tells us precisely what to change for the UAV.

### 7.1 What SafeFlowMPC actually is (read from code, not the abstract)

The paper's abstract is vague ("a combination of flow matching and online optimization … a
suboptimal model-predictive control formulation … guarantees safety at all times"). The code is
explicit:

| Component | SafeFlowMPC (code) |
|---|---|
| Planning space | **joint space** `q ∈ ℝ⁷` of a KUKA 7-DoF arm |
| Trajectory tensor | `[n_horizon=16, n_out=7]` — **pure `q` plan; NO des/actual doubling** (`PlannerConfig.n_out=7`) |
| Backbone | `TemporalUnet` — **literally Janner's** (`FlowMatchingField.py`) |
| Conditioning vector | `q_prev` history (10 past **actual** configs) + **FK of the actual `q`** (EE pose `p0,r0`, 7 collision-sphere positions) + goal pose (`create_condition_vector`, `SafeFlowMPC.py`) |
| Per-flow-step projection | **Acados NMPC** "safety filter": projects the flow's proposed `q_des` onto {real-dynamics-feasible ∧ joint/vel/accel/jerk-limited ∧ collision-free ∧ workspace-bounded}, **anchored at the real current state `q0`** (`SafetyFilterAcados.step(q0, q_des)`) |
| Filter dynamics model | real kinematic chain: state `x=[q,dq,ddq,u_prev]`, control `u=jerk`, discrete triple-integrator (`SafetyFilterAcados.py:34–45`) with `robot_model` limits |
| Execution | `n_actions=1` → execute one step, **re-plan** (receding horizon, Janner-style) |
| State feedback | `state.q = safety_filter.q` — the **real-dynamics-integrated** next state; `q_prev` ← actual past (`_update_state_and_metrics`) |

The only `p_des` token in the whole repo is an **ephemeral goal-guidance vector** inside
`_compute_guidance` (`p_des = p_goal[:3]`, recomputed every step from the goal) — it is **never
stored, never fed back as state.** There is no `des_c_pos`/`p_des` state channel anywhere.

### 7.2 Accusation by accusation — SafeFlowMPC dodges each by construction

| Layer (DPCC sin) | What SafeFlowMPC does instead | Result |
|---|---|---|
| **L1 — units lie** (Δdes called "velocity", `dt=1` Euler) | The action is **jerk**, integrated by a true 3rd/4th-order integrator with a real `dt` (`q_new = ddq·dt²/2 + dq·dt + q + jerk terms`). The filter's state literally carries `q,dq,ddq`. | **No unit lie.** Every quantity is the physical thing it's named. |
| **L2 — half-hallucinated state** (`[des｜actual]`, half is a reference) | **No split.** `n_out=7` = `q` only; conditioning is the **actual** configuration (and its FK). | **No hallucinated half.** The state *is* the plant. |
| **L3 — self-reference + binding asymmetry** (loop reads its own `p_des`; projector binds the echo) | Still receding-horizon, so still autoregressive — **but the loop is grounded by a real dynamics model**: the next state is a *dynamically-feasible evolution of the real arm*, not a tautological `p_des += act`. The NMPC binds the **real state `q0`** to the **real robot model**. | **Grounded autoregression**, exactly the safe case from §3.2. **No binding asymmetry** — there is only one (real) state, bound to a real `f`. |

### 7.3 The crux you asked about: condition on `p_des` or real `p`?

> **SafeFlowMPC conditions on the REAL state.** The network sees `q` (actual joint config) and its
> forward-kinematics (actual EE pose, actual collision-sphere positions) plus the actual recent
> history `q_prev`. **There is no `p_des` in the conditioning at all.** The flow *proposes* a
> desired trajectory; the safety filter *immediately* projects it through the real robot dynamics;
> and only the **dynamically-realized** state becomes the next conditioning input. The reference
> never feeds back — only the model-grounded realization does.

| | DPCC / our UAV | SafeFlowMPC |
|---|---|---|
| Conditioned on | `[des_c_pos｜c_pos]` / `[p_des｜p｜v]` — **reference + real mixed** | `q` (real) + FK(`q`) + `q_prev` (real) — **real only** |
| Reference in the state slot? | **Yes** (`des`/`p_des`) | **No** |
| Projection binds | `p_des` (the echo — a **tautology**, no plant model) | `q` (the real state) to a **real robot dynamics model** |
| Dynamics model `f` | invented Euler `s+aΔt` (`dt=1`) | actual kinematic triple-integrator + joint/vel/accel/jerk limits |
| Next state comes from | `p_des += action` (model's own write) | NMPC-integrated real dynamics (grounded) |

This is the single most important contrast in this whole document: **SafeFlowMPC made the choice
DPCC could not — bind and condition on the real state — because it paid for a real dynamics model.**

### 7.4 In the language of §6: SafeFlowMPC is the "Re-ground" exit, fully committed

§6 named three exits. SafeFlowMPC is **Exit 1 (Re-ground) executed completely** — it commits to
Janner's worldview *all the way down*:

- It **borrows Janner's horizon** (`n_horizon=16`, receding) — just like the H8 borrow. **But that
  is fine here**, because it *also* keeps Janner's grounding (real state, re-observed/re-integrated
  each step). The H8 borrow was never the problem; borrowing the horizon (Janner) while keeping the
  *state format* of D3IL was. SafeFlowMPC takes the horizon **and** the real-state grounding
  together, so the seam never forms.
- It **never inherited D3IL's data format.** Its data are VP-STO joint-space plans (`q`
  trajectories), so a reference-stream was never available to accidentally drop into the state slot.
- It **imported a real plant model** (the Acados robot model), so its projection binds the real
  state to a real `f` — no streetlight-effect tautology.

In other words: **the category error is not intrinsic to "flow + projection."** It is intrinsic to
"flow + projection **on top of a position-controlled command-stream dataset with no plant model**."
Remove either of those two D3IL-inherited conditions and the three layers vanish. SafeFlowMPC
removes both.

### 7.5 The honest asymmetry — why SafeFlowMPC *could* and DPCC "couldn't"

This is not magic; SafeFlowMPC paid a price DPCC dodged:

- **It needs an analytic plant model.** A rigid manipulator has known kinematics, so writing the
  real `f` (and joint/vel/accel/jerk limits, FK collision spheres) is cheap. DPCC inherited D3IL's
  *modelless* BC setup; it had no `f` to bind, so it bound the tautology.
- **It pays a constrained NMPC solve at every flow step.** That is expensive — which is exactly why
  the paper uses a *suboptimal* MPC, a per-step **time limit**, and a **fallback to the last safe
  trajectory** when the solver runs long (`SafeFlowMPC.step`: `if … > self.time_limit: x_current =
  last_safe_trajectory`). Grounding has a real-time cost; the tautology is free but fictional.

So the trade is explicit: **SafeFlowMPC buys correctness with a real model + per-step optimization;
DPCC bought speed/simplicity with a fiction that only holds when tracking error ≈ 0.**

### 7.6 What this means concretely for our UAV (the payoff)

SafeFlowMPC is a *published, ICRA-2026, real-hardware* proof that §6.9's "Re-ground" exit works.
Porting its three moves to the UAV:

1. **Condition on the real state.** We already carry real `p`, `v` in the 9D obs `[p_des｜p｜v]` —
   stop treating `p_des` as the anchor; condition on `p`/`v` (and retrain). (SafeFlowMPC conditions
   on real `q`/FK only.)
2. **Bind a real model, not the tautology.** Replace the `deriv`-on-`p_des` constraint with a
   (even coarse) **drone dynamics model** in the projector — or at minimum a *tracking-feasibility*
   constraint bounding `‖p − p_des‖` — anchored at the **measured `p`** each step. This is the
   direct analog of binding `q0` to the real robot model. (See also §4 Step 3.)
3. **Adopt the robustness pattern.** A per-step time budget + **fallback to last safe trajectory**
   (SafeFlowMPC has exactly this) is a clean, drop-in safeguard — and a far better answer to "FM
   keeps planning / drifts" than the current no-termination loop.

The result is the conversion §6 prescribed: from a **conditionally-correct** self-referential graft
(valid only when `e_t≈0`) into an **unconditionally-grounded** controller — at the cost of a real
per-step projection, which SafeFlowMPC shows is affordable in real time.

### 7.7 Caveats (so we don't overclaim)

- SafeFlowMPC is a **manipulator** (known analytic model, bounded workspace, tracks well). A
  quadrotor adds underactuation and aerodynamic lag, so a *perfect* real-dynamics filter is harder
  to build than the KUKA's. But the argument is monotone: **even a coarse drone model in the
  projector + conditioning on real `p` strictly dominates binding a tautology on `p_des`.**
- In the released sim code, `state.q` is the *filter's* dynamically-integrated state, not a hardware
  encoder read — so the grounding is "by a real **model**," and on hardware (the paper's real
  experiments) the loop closes through the actual robot. The decisive difference from DPCC is the
  presence of a **real `f`**, regardless of sensor-vs-model closure.
- We did not run SafeFlowMPC; this section is from a close read of `/workspaces/SafeFlowMPC` plus
  the arXiv abstract. Claims are traceable to the files cited in §7.8.

### 7.8 References (SafeFlowMPC)

| Source | What it shows |
|---|---|
| arXiv:2602.12794 (abstract) | flow matching + *suboptimal MPC* safety filter; KUKA 7-DoF; "guarantees safety at all times" |
| `safe_flow_mpc/SafeFlowMPC/PlannerConfig.py` | `n_out=7` (q-only trajectory), `n_horizon=16`, `n_actions=1` (receding) |
| `safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` `create_condition_vector` | conditioning = `q_prev` + **FK(actual q)** + goal — no `p_des` |
| `safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` `step` / `_update_state_and_metrics` | per-step flow → safety filter → `state.q = safety_filter.q`; time-limit fallback to `last_safe_trajectory` |
| `safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py:34–45, 101–138` | real kinematic model `x=[q,dq,ddq,u_prev]`, jerk control, joint/vel/accel limits, FK collision-set constraints |
| `safe_flow_mpc/SafeFlowMPC/FlowMatchingField.py` | `TemporalUnet` (Janner backbone), conditioned velocity field |
| grep `p_des` over repo | only an ephemeral goal-guidance local in `_compute_guidance` — never a stored/fed-back state |

---

## 8. Deeper Q&A — Real Model vs Outer PID, Where the Model Lives, the Fallback, and the Threshold

Five precise questions came out of §7. Each is answered from code, with the upgrade implication.

### 8.1 "Our 'dynamics model' is just the outer PID, not a real model — right?"

**Right — and worse: the planner doesn't even model the PID.** Two separate things must not be
confused:

- **What moves the drone (real, but OUTSIDE the planner):** `eval_fm_uav.py:347` —
  `u = pid.compute(p, q, v, om, p_des, v_des); data.ctrl[:4] = u; mujoco.mj_step(...)`. A PID tracks
  `p_des`/`v_des` → motor commands → real rigid-body physics. This is the **true closed-loop
  dynamics** (drone + PID).
- **What the PCC projection uses (a fiction, INSIDE the planner):** the tautology
  `p_des[t+1] = p_des[t] + act` (`deriv` binds `p_des`; `p_des += action`).

So the planner's "dynamics model" is a **kinematic placeholder on its own reference** — it knows
*nothing* about the PID or the rigid-body physics that actually realize the motion. The real plant
(drone + PID) is an **unmodeled outer loop**. The gap between the two is logged every step as
`track_err = ‖qpos[:3] − p_des‖` (`eval_fm_uav.py:355`). DPCC's arm had the same structure but an
IK controller that tracks so tightly the gap is ~0; the UAV's PID cannot, so the gap is real.

### 8.2 "Does SafeFlowMPC feed the real dynamics model INTO the flow network?"

**No — and this is the clarifying point.** The dynamics model is **not** inside the flow network
(`TemporalUnet`); it lives in the **safety filter** (Acados NMPC), a separate projection layer that
wraps the FM and runs at every flow step. There are two channels, neither of which puts dynamics
equations into the FM weights:

| Channel | Where | Hard or soft |
|---|---|---|
| Real dynamics model (`x=[q,dq,ddq,u_prev]`, jerk integrator, joint/vel/accel/jerk limits, collision sets) | **safety filter** (`SafetyFilterAcados`) — projection layer | **hard** (enforced every flow step) |
| "Safety" bias | FM finetuned on **safe demonstration data** (`train_imitation_learning_safe.py`, `use_safe_model`) | **soft** (a prior, not a guarantee) |

> The FM is **model-agnostic in both SafeFlowMPC and DPCC** — it just predicts a velocity field.
> The dynamics model sits in the **same architectural slot** in both: the projection that wraps the
> FM. The *only* difference is what's in that slot — **SafeFlowMPC: a real robot model; DPCC/us: a
> tautology.** That is the whole ballgame. We do not need to change the FM; we need to put a real
> model where the tautology is.

### 8.3 "Is our 'trajectory / trajectory+velocity → IK/PID' a problem by SafeFlowMPC's logic?"

**Yes — it is the same gap, from two angles:**

1. **We plan in the wrong space.** We emit Cartesian setpoints `p_des` (and `v_des`) and hand them
   to a controller the planner doesn't model. SafeFlowMPC plans **directly in the controlled
   variable** (`q`, joint space, `n_out=7`) — **no IK at all** at plan time (FK is used only for
   conditioning/collision). Planning in the consumed variable removes the IK/tracking translation
   where error is injected.
2. **We don't model the controller's feasibility.** SafeFlowMPC's filter bakes the joint
   vel/accel/jerk limits into the projection, so the committed trajectory is **trackable by
   construction**. We commit a `p_des` stream that may be physically untrackable and *hope* the PID
   keeps up.

The point is **not** "controllers are bad" — SafeFlowMPC's hardware has a low-level controller too.
The point is **the planner must MODEL the controlled plant's feasibility.** We model nothing; we
bind a tautology and trust an outer PID. By SafeFlowMPC's logic that is exactly the defect.

### 8.4 "The fallback feels weird — does it fall back to an IK solver?"

**No. It falls back to the last *certified-safe trajectory*, not an IK solver.**

- After every successful projection, `SafeFlowMPC.step` stores the result:
  `self.last_safe_trajectory = self.x_current.detach()` (line 392) — i.e. the most recent
  **dynamically-feasible + collision-free** plan the filter has blessed.
- If the per-step **time budget** is exceeded, it reuses it: `if … > time_limit: self.x_current =
  self.last_safe_trajectory` (lines 351–352).
- If the **SQP fails to converge**, it shifts the last solution forward
  (`update_from_last_solution(n_actions)`, line 406) — a warm-start/feasibility-recovery, still no
  IK.
- (Separately, a CasADi-QP `safety_filter_init` only *seeds* the initial trajectory feasibly — it is
  not the runtime fallback.)

This is the standard MPC safety net: **"if you can't compute a fresh safe plan in time, keep
executing the last one you proved safe."** It is robust precisely because every stored candidate
already passed the real-dynamics + collision constraints. It's a pattern we should copy verbatim
(see §8.6).

### 8.5 "Quick check: is it a diffuser-style H-horizon planner, and does it gate projection on a ~0.5 threshold?"

| Question | DPCC / our UAV | SafeFlowMPC |
|---|---|---|
| Diffuser-style H-horizon planner? | yes (H=8, receding) | **yes** — `TemporalUnet` (Janner), `n_horizon=16`, `n_actions=1` receding. (Flow-matching = continuous cousin of diffusion.) |
| Project only after a denoising **threshold** (DPCC's `diffusion_timestep_threshold`, ~0.5/late steps)? | **yes** — gated to late steps | **NO** — the safety filter runs at **every** flow step (no threshold gate). |

Two takeaways: (1) SafeFlowMPC **confirms the H-horizon/receding "planner" worldview is fine** — it
uses H=16 happily; the horizon was never the problem (§6.4). (2) It **projects every step, not just
late ones** — it never lets an unsafe intermediate trajectory exist, whereas DPCC only enforces
feasibility near the end. For hard safety, "every step" is the stronger choice; our threshold
gating is a speed optimization that trades away mid-chain feasibility.

### 8.6 Upgrade path (for the future work the user flagged)

Concretely, to move our UAV from "conditionally-correct graft" toward SafeFlowMPC-grade grounding,
in increasing order of effort:

1. **Copy the fallback now (cheap, big robustness win).** Keep a `last_safe_trajectory`; on
   time-budget overrun or projector failure, re-execute it instead of pushing a fresh unvetted plan.
   Add the missing termination (`track_err` large → stop) the loop currently lacks.
2. **Put a real model in the projection slot.** Replace the `deriv`-on-`p_des` tautology with a
   (even coarse) drone+PID feasibility model — or at minimum a tracking-feasibility constraint
   bounding `‖p − p_des‖` — anchored at the **measured `p`** each step. This is the direct analog of
   binding `q0` to the real robot model.
3. **Condition on the real state.** We already carry real `p`, `v` in the 9-D obs — stop anchoring
   on `p_des`; condition on `p`/`v` and retrain. (SafeFlowMPC conditions on real `q`/FK only.)
4. **(Largest) plan in the controlled variable with its limits**, à la SafeFlowMPC's joint-space
   `q` plan, so the committed trajectory is trackable by construction and the IK/tracking
   translation disappears.

Steps 1–3 are incremental and keep the existing FM; step 4 is the full "Re-ground" commitment (§6.9
Exit 1) that SafeFlowMPC validates end-to-end on real hardware.

---

## 9. The Ultimate Question — Can DPCC *prove* `p_des` makes sense for the diffuser, and why not just use real `p`?

Everything above converges here. The honest, complete answer needs one move first: **stop saying
"condition on `p_des` vs `p`" as if it were one choice.** `p_des`/`p` play **three distinct roles**,
and the answer is different for each.

### 9.1 The three roles (don't conflate them)

| Role | What it is | What DPCC/our-UAV does | Is real `p` usable here? |
|---|---|---|---|
| **(A) Trajectory channel** | which dims the diffuser *denoises* | denoises `[act｜p_des｜p]` — **both** present | both already denoised |
| **(B) Conditioning** | what is *inpainted at t=0* (`apply_conditioning: x[:,0,action_dim:] = obs`) | pins the **full obs `[p_des｜p｜v]`** — **both** | **we ALREADY condition on real `p`** |
| **(C) Constraint binding** | which dim the projector's `deriv` ties to the action | **arm: real `c_pos`** (`deriv [6,7,8]`); **UAV: `p_des`** (`deriv [3,4,5]`) | **this is the only place real `p` is *not* used (UAV)** |

> [!IMPORTANT]
> **The popular framing "DPCC conditions on `p_des` instead of `p`" is imprecise.** Via inpainting
> the FM is conditioned on **both** `p_des` and real `p` at t=0 (role B). The genuine asymmetry is
> only in role **C** — the dynamics *constraint*. And there, **the arm already binds real `c_pos`**;
> only our **UAV** binds `p_des`. So "use real `p`, it's more sensible" is (i) already true for
> conditioning, and (ii) already what the arm does for the constraint. The question reduces to one
> narrow thing: **why does the UAV's *constraint* bind `p_des` and not real `p`?**

### 9.2 Can DPCC *prove* binding `p_des` "makes sense"? — It proves **feasible**, never **meaningful**

DPCC (and we) can prove exactly one property of `p_des` binding: it is **always satisfiable**,
because `p_des[t+1] = p_des[t] + act` is **true by construction** (it is how `p_des` is integrated).
That is a *feasibility* guarantee — the projector never fights an impossible constraint, `proj_cost`
stays clean. **But feasibility is not meaning.** A tautology constrains the model's *own command*,
not the *world*; it certifies "the plan is self-consistent," not "the drone can fly it." So:

> **DPCC can prove `p_des` binding is feasible. It cannot prove it is physically meaningful — and it
> never claims to** (the paper states the design, never justifies it; WHY_FM §13). The "sense" it
> has is the *minimum* sense: it keeps the optimizer well-posed. That is the whole of the proof.

### 9.3 Why not just bind real `p`? — Because that needs a plant model we don't have

Binding real `p` means asserting `p[t+1] = p[t] + act` inside the projector. **That equation is
false every step** for a quadrotor (the drone lags — §6.2f). So:

- bind `p_des`: `p_des[t+1] = p_des[t] + act` ✅ exact (tautology) — but constrains the command.
- bind real `p`: `p[t+1] = p[t] + act` ❌ false by `e_t = ‖p_des−p‖` each step — the projector would
  fight physics, `proj_cost` permanently high, projection meaningless (WHY_FM §6).

To bind real `p` *correctly* you must replace the tautology with the **true map command→motion**,
i.e. a **drone dynamics model**. DPCC inherited D3IL's **model-free** BC setting (§6.2a–d): there is
**no plant model in the box**. So binding `p_des` is not chosen over `p` on merit — it is the **only
constraint you can write without a model**. (Streetlight effect, §6.7.)

### 9.4 The trap that kills every clever relabeling

"Then redefine the action as real-position delta `Δp` — we even record real `p`!" It doesn't escape:
**anything the FM emits is a *command*** (the FM is the high-level planner; a PID/IK tracks it), and
**the plant lags any command.** Rename `Δp_des → Δp_desired` all you like — the achieved motion
still trails it by `e_t`, so the "bind real `p`" equation is still false. **No relabeling removes
the lag; only a model of the lag does.** This is why the problem is fundamentally "missing plant
model," not "wrong variable name."

### 9.5 The arm is the existence proof that real-`p` binding is the *intended* design

Tellingly, **the original DPCC arm binds real `c_pos`** (`deriv [6,7,8]`), not `des`. It can,
because the arm tracks tightly (`c_pos ≈ des`), so `c_pos[t+1] ≈ c_pos[t] + act` is *approximately*
true — the tautology holds for the **real** variable. So DPCC's *own* preferred design **is** "bind
real position"; the UAV only falls back to `p_des` because, on a lagging plant, the approximation
that made real-binding feasible **breaks** (§6.6). Real-`p` binding isn't a heresy — it's the arm's
actual behavior, lost on the UAV for lack of a model.

### 9.6 Did SafeFlowMPC "say" this? — No; it **sidestepped** the question by never having `p_des`

SafeFlowMPC never debates `p_des` vs `p` because **it has no `p_des`** (§7). It plans directly in
**real joint state `q`**, conditions on real `q`/FK, and its safety filter binds the **real state to
a real robot dynamics model** (Acados NMPC). It is the **existence proof** that the "sensible" thing
— *condition and constrain on the real plant* — **works on real hardware**, at the stated price of
**carrying a real model + a per-step optimization**. It didn't write "use `p` not `p_des`"; it did
something stronger — it **declined the command-centric format entirely**, which is the format that
manufactured the `p_des`-vs-`p` dilemma in the first place (§6.2). So SafeFlowMPC's answer to the
ultimate question is: *the dilemma is optional — don't adopt the data format that creates it.*

### 9.7 The mechanism that changes the picture: `skip_initial_state` hard-resets the bound dim to the *measured* state

The analysis so far treated the dynamics constraint as "just the Euler chain." It is actually **two
operations**, and the second one is the game-changer you flagged. From `projection.py` (the
`dynamic` constraint, **on for the projected variants, off for `diffuser`** where `projector=None`):

```python
# build: skip_initial_state prepends a "fix the initial state" row to the equality block
mat_fix_initial[0, x_idx] = 1                    # projection.py:394–398
# project(): each step, set that row's target to the CURRENT MEASURED value
if self.skip_initial_state:                      # projection.py:99–108
    s_0 = trajectory_reshaped[0, :transition_dim]      # t=0 slice = the inpainted obs
    b[counter*horizon] = s_0[x_idx]              # "Must be changed to current state in each iteration!"
```

So the constraint does **(1) hard-pin the bound dim's value at t=0 to the current measured state**,
then **(2) Euler-chain it forward.** Operation (1) is a literal **per-step reset to reality** — and
it is the difference between `dynamic` and `diffuser`:

| | `diffuser` (no projector) | `dynamic` (projected) |
|---|---|---|
| t=0 reset of bound dim to measured | **none** | **yes** (`skip_initial_state`) |
| Euler chain | none | yes |
| Re-grounding to reality each step | **only soft inpainting** | **hard equality pin** |

> [!IMPORTANT]
> **But the reset only grounds the dimension it BINDS — and that is exactly where arm and UAV
> diverge.**
> - **Arm binds `c_pos` (`deriv [6,7,8]`)** ⇒ `skip_initial_state` pins **`c_pos[0] = real measured
>   position`** every step, then Euler-chains. The plan is **hard-reset to where the arm actually
>   is**, each step. *This is the arm's real anti-drift mechanism* — stronger than "tracking is
>   tight": even if it drifted, the projector yanks t=0 back to the measured `c_pos`.
> - **UAV binds `p_des` (`deriv [3,4,5]`)** ⇒ `skip_initial_state` pins **`p_des[0] = the
>   accumulated command`** (because `s_0[3,4,5]` is the inpainted *command* channel), then chains.
>   The reset re-grounds to the **command, not reality**. The UAV gets the *machinery* of the reset
>   aimed at the **wrong channel**, so it provides **zero** correction toward real `p`.
>
> So the `dynamic` constraint **is** a "hard reset to real" — *for the arm.* The UAV runs the same
> code path and gets a "hard reset to its own command," which does nothing for drift. You were
> right that this is a big deal; the subtlety is that the binding index decides whether the reset
> lands on reality or on the echo.

**This also refines "why not bind real `p`" (§9.3) — there are now two blockers, not one.** Suppose
we switched the UAV's `deriv` to bind `p` (`[6,7,8]`). Then `skip_initial_state` *would* pin
`p[0] = real measured` (good — we'd get the arm's reset). **But the forward Euler chain
`p[t+1]=p[t]+act` would now fight the FM's own output:** the FM's `p` channel was *trained to be the
lagging real position*, which does **not** satisfy `p=∫act`, so the projector would have to distort
the FM's `p` heavily (high `proj_cost`, §9.3). The `p_des` channel, by contrast, satisfies
`p_des=∫act` *by training construction*, so binding it is free. Hence:

> **Binding `p_des` is "constrain the channel that already obeys the constraint."** The reset-to-real
> we want lives in operation (1); the infeasibility we fear lives in operation (2). A correct UAV
> fix must therefore **keep the t=0 reset on real `p`** *and* **replace the rigid `p=∫act` chain with
> a real (or relaxed/banded) tracking model** — exactly SafeFlowMPC's "bind real state to a real
> dynamics model" (§7). Switching the binding index alone trades a useless reset for an infeasible
> chain.

### 9.8 Does the "dynamics model" even *do* anything? — with vs without (DPCC-avoiding vs our UAV)

Your suspicion is right to a degree that matters: **the "dynamics constraint" is not a dynamics
model.** It is an **Euler self-consistency relation** — "make the action channel equal the position
channel's delta," `act[t] = (pos[t+1]−pos[t])/dt` — plus the `skip_initial_state` pin (§9.7). It
encodes **no plant physics** (cf. §6.7, Layer 1). Whether it does anything *useful* depends on two
independent "jobs," and you have to ask which are active:

| Job | What it needs to be useful | Arm avoiding (DPCC) | **Our UAV (this epoch)** |
|---|---|---|---|
| **A. Grounding** (reset t=0 to reality) | the bound dim must be the **real** position | ✅ binds real `c_pos` → resets to measured | ❌ binds `p_des` → resets command **to itself** (`p_des[0]` already = inpainted `p_des`) → no-op |
| **B. Glue for safety** (let position-space obstacle/bound constraints reshape the action) | **other** constraints (obstacles/bounds) must be active | ✅ obstacle constraints present → `deriv` propagates them into `act` | ❌ `constraint_types=['dynamics']` only; bounds/halfspace/obstacles are **PLACEHOLDERS, not run** (free-space) |

**So for original DPCC avoiding the constraint earns its keep — both jobs fire.** It resets the plan
to the real `c_pos` each step *and* is the mechanism that turns "this position hits an obstacle" into
"change the action." Remove it → you lose collision avoidance **and** grounding → unsafe. It is
load-bearing there.

**For our UAV this epoch, the constraint is near-vacuous.** Job A is a no-op (it pins `p_des[0]` to
the command, which inpainting already set to that same value). Job B has nothing to glue (no
obstacle/bound constraints active). What's left is only a mild **FM-internal regularizer**: it
nudges the FM's `act` channel to agree with its own `p_des` channel before the first action is read.
And even that barely reaches execution, because the eval loop **re-integrates `p_des = p_des +
action`** itself (`eval_fm_uav.py:319`, `:338`) — it does not consume the FM's downstream `p_des`
channel at all. So the SLSQP solve is mostly paying `proj_ms` to enforce a tautology on a channel
that is then recomputed by hand.

**What if we drop it and use only the FM-generated action?**

- **DPCC avoiding:** breaks — no obstacle projection, no real-state grounding. Don't.
- **Our UAV (free-space, this epoch):** **behaviour would be ≈ unchanged.** The first action comes
  from the FM either way; `p_des += action` and the PID are untouched. Crucially, **the drift /
  self-reference is NOT caused by this constraint** — it comes from `p_des` accumulation +
  conditioning (§3, §6), which exist with or without the projector. So removing a near-vacuous
  constraint **won't fix the drift and won't meaningfully hurt** free-space flight; it would just
  save the projection compute. (The `diffuser` variant — `projector=None` — is essentially this
  experiment already; comparing `diffuser` vs `dynamic` on the *same* seed is the clean A/B.)

> [!IMPORTANT]
> **Bottom line on "does it make sense":** the dynamics constraint makes sense **exactly when it
> binds the real state and/or carries real safety constraints** — that is the arm-avoiding case. In
> our **current UAV** it does **neither**, so it is a mislabeled, near-idle self-consistency step,
> not "dynamics." It is **neither the cause of the drift nor a cure for it.** It only starts earning
> its name when we (a) bind **real `p`** (needs a model / the §9.7 fix) or (b) switch on real
> obstacle/bound constraints (then Job B's act↔position glue becomes essential). Until one of those,
> "we have a dynamics constraint" is, for the UAV, closer to "we run an extra SLSQP that re-derives
> `act = Δp_des`."

### 9.9 Verdict & the only honest resolutions

> **Ultimate verdict.** DPCC **cannot prove `p_des` binding is *right*; it can only prove it is
> *feasible*.** Conditioning on real `p` is **already done** (inpainting). The real machinery that
> grounds the arm is the `dynamic` constraint's **`skip_initial_state` hard-reset of the bound dim's
> t=0 to the measured state** (§9.7) — but it only grounds the dim it binds: the **arm binds real
> `c_pos`** so it is reset to reality each step, while the **UAV binds `p_des`** so the same reset
> lands on its own command and corrects nothing. The lone real asymmetry — the UAV's *constraint*
> (reset + Euler chain) operating on `p_des` instead of real `p` — is **forced by two things**: a
> missing drone model **and** the fact that the FM's `p` channel was trained as the *lagging* real
> position (so `p=∫act` would fight it), while `p_des` obeys `p_des=∫act` by construction. The arm
> shows the intended design is real-`p` binding; the UAV degrades to `p_des` only because the lag
> breaks both the feasibility *and* the usefulness of the reset. **"Bind real `p`, it's more
> sensible" is correct — the blockers are exactly: (i) keep the t=0 reset on real `p`, and (ii)
> supply a plant model so the forward chain is feasible.** SafeFlowMPC supplies both and confirms the
> sensible design works.

Three honest ways forward (in rising effort; cf. §6.9, §7, §8.6):
1. **Re-anchor** `p_des ← (1−α)(p_des+act)+α·p` — force the tautology to stay near reality (cheap; keeps the arm's `e_t≈0` regime alive by fiat). *Note: this is the software analog of giving the UAV the `skip_initial_state` reset the arm gets for free.*
2. **Give the projector a model** — even a coarse drone+PID model (or a `‖p−p_des‖` feasibility band) so the constraint can **reset t=0 to real `p` AND chain it feasibly** (§9.7). Switching the `deriv` index to `p` *without* this only swaps a useless reset for an infeasible chain.
3. **Re-ground fully** — plan in real state with a real model, à la SafeFlowMPC; drop `p_des` from the state slot. Removes the dilemma at the root.

---

## 10. Can we fix it for the UAV? And is DPCC "cheating"?

### 10.1 Your understanding — confirmed, with one wording fix

> "The dynamics constraint is weak: it really does correct the *action tensor* (`act ↔ p_des`
> self-consistency), but it **cannot** correct the **real-position drift**, which comes from the
> low-level controller lag."

**Correct.** The constraint binds `p_des`, so it can only make the FM's action channel agree with its
command channel — a real but **narrow** job. Real-`p` drift is invisible to it.

**One fix to the framing:** it isn't "PID vs IK." Both are low-level controllers; the real axis is
**tracking error**. The arm's IK happens to track *tightly* (`c_pos ≈ des`), so the same Euler
constraint lands on a quantity that ≈ reality. The drone's PID *lags* (`p ≠ p_des`) because a
quadrotor is underactuated/second-order. So say **"tight-tracking plant vs lagging plant,"** not
"IK vs PID" — swapping the drone's PID for an "IK-equivalent" would not help; the lag is the
**plant's** physics, not the controller's name.

### 10.2 "Re-anchor the constraint to real `p`" — yes, but it's *half* the fix

Switching the `deriv` to bind `p` (indices 6–8) instead of `p_des` (3–5) does buy you the arm's
**`skip_initial_state` reset to real `p`** (§9.7) — good. **But the forward Euler `p[t+1]=p[t]+act`
would then fight the FM's `p` channel**, which was trained as the *lagging* real position and does
**not** obey `p=∫act` → infeasible, `proj_cost` explodes. So index-switch *alone* trades a useless
reset for an infeasible chain. A real fix needs **both** (1) reset t=0 to real `p` **and** (2) a
*feasible* forward relation. Ranked, cheapest first:

| # | Fix | What it does | Cost |
|---|---|---|---|
| 1 | **Re-anchor the integrator**: `p_des = (1−α)(p_des+act) + α·p` | bleeds the command back toward reality each step → bounds drift. *This is literally giving the UAV the reset the arm gets for free.* | software-only, no retrain, no projector change |
| 2 | **Tracking-feasibility band**: add inequality `‖p − p_des‖ ≤ ε` to the projector | lets the projector see real `p` and forbid the command running away from it — a **relaxed** bind (feasible, unlike rigid `p=∫act`) | small projector change |
| 3 | **Coarse drone model in the projector** | replace the tautology with a real command→motion relation → bind real `p` correctly | medium; needs a (even rough) dynamics model — SafeFlowMPC §7 |
| 4 | **Re-ground fully** | condition + plan in real state with a real model; drop `p_des` from the state slot | large; SafeFlowMPC-style rewrite |
| 5 | **Terminate** (orthogonal) | `track_err` large → stop the rollout (the UAV loop currently never terminates) | trivial; honest safety net |

**Recommendation:** start with **#1 (+#5)** — they bound the drift and stop runaways with no
retraining; escalate to **#2/#3** if you need the projector itself to respect reality.

### 10.3 Is the dynamics constraint "always on," and is DPCC cheating? — fair verdict: **No, not cheating**

**"Always on?"** Precisely: the dynamics constraint lives in the **projector**, which fires only on
the **projected variants** (it is *off* for `diffuser`, where `projector=None`), and only in the
**late part of the flow chain** — `near_end`, gated by `diffusion_timestep_threshold` (e.g. last
~50%), not every step (`diffusion.py:206–275`). So for "dpcc-with-constraint" runs it is on (late
chain); for the `diffuser` baseline it is entirely off. Comparing those two on the same seed is the
clean ablation.

**Does the paper hide this? Is it cheating?** **No — and it's worth being fair here:**

- The paper is **upfront that it does not model the controller.** It states *"we do not assume
  knowledge of the dynamics of the low-level controller"* and writes the dynamics as Euler
  `s_{t+1} = s_t + [aᵀ,aᵀ]ᵀ·ts + w_t`, where **`w_t` is an explicit "model-mismatch" slack** for the
  low-level controller + numerical error (WHY_FM §13). That is a **declared limitation**, not a
  hidden trick.
- On the **arm**, the paper's design binds the **real** position (`c_pos`) and the Euler holds
  *approximately* because the arm tracks; the small gap is exactly what `w_t` absorbs. That is a
  legitimate, grounded constraint — it genuinely does work on the demonstrated task.
- What the paper **omits** (not fraud, but a real gap): any analysis of the **large-tracking-error
  regime** (where `w_t` is no longer small and the Euler approximation breaks), and any
  justification of the self-referential `des` at eval (WHY_FM §13: "states the design, never
  justifies it"). That is an **unexamined assumption**, the honest weakness.
- **The UAV's `p_des` binding is OUR deviation, not the paper's.** The paper/arm binds real
  position; **we** bind the command `p_des` (indices 3–5) because the drone lags and we have **no
  drone model** to bind real `p` feasibly (§9.7). So the specific weakness you're feeling is mostly
  *our forced adaptation* of an honest-but-under-examined method onto a plant that violates its
  hidden assumption (`tracking error ≈ 0`) — and even that isn't cheating, it's the only feasible
  option absent a model.

> **Verdict:** DPCC is **not cheating** — it openly declares the model-free Euler + `w_t` slack and,
> on the arm, binds the real position so the constraint is grounded and works. Its honest flaw is an
> **unexamined regime** (what happens when `w_t`/tracking-error is large). Our UAV walked straight
> into that regime *and* further weakened the design by binding the command instead of the real
> position (forced by the missing drone model). The fix is §10.2 — restore the reset-to-real and give
> the constraint a feasible way to respect `p`.

---

## 11. References (this critique)

| File | Line | What it shows |
|---|---|---|
| `WHY_FM_KEEPS_PLANNING.md` | ~1866–1883 | The three-layer claim being critiqued |
| `eval_fm_uav.py` | 130–132 | Comment: p_des bound *because drone p lags*; arm tracks perfectly |
| `eval_fm_uav.py` | 157 | `('deriv',[3,0/1/2])` — constraint binds **p_des (echo)**, not p (reality) |
| `flow_matcher_v3_uav/sampling/projection.py` | 394–398 | `skip_initial_state` prepends a "fix initial state" equality row to the bound dim |
| `flow_matcher_v3_uav/sampling/projection.py` | 99–108 | `project()`: each step sets that row's target `b = s_0[x_idx]` = **current measured** value ("must be changed to current state in each iteration") — the per-step hard reset (§9.7) |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 119–121 | **arm** binds real `c_pos` (`deriv [6,7,8]`) → reset lands on reality |
| `eval_fm_uav.py` | 301 | `track_err = []` — we already log the deciding quantity |
| `eval_fm_uav.py` | 312, 338–339 | obs `[p_des|p|v]`; `p_des += action`; `v_des = action/dt_fm` |
| `config/visual_aligning_eval.yaml` | 24 | `dt=1.0 IS CORRECT: actions are position deltas` (Layer 1 resolution) |
| `diffuser_visual_aligning/.../sequence.py` | 76–83 | training trajectory `[act|des_c_pos|c_pos]` — des is teleop command stream |
