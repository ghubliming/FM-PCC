# Why is the UAV-FM trajectory terrible? — diagnosis + next-step direction

Companion to `ANALYSIS.md` / `pillars_seed6_results.json`. **UPDATED** with hard
evidence from the U3 eval npz (`temp/eval/fm_only/fm_only.npz`, pillars seed 6,
4 trials). The earlier guesswork is superseded by the CONFIRMED findings below.

---

## READ THIS FIRST — plain-language walkthrough (no jargon)

You are right about the pieces: we have a trajectory with `p` (position), `v`
(velocity), and `p_des` (the waypoint), we train a NN on it, and MuJoCo flies it.
The confusion is about **two things that are easy to miss**: (a) there are **two
loops at different rates**, and (b) the NN does **not** output a position — it
outputs a *change*, and we *add it up* at eval in a loop training never saw.

### There are TWO control loops (this is the "different dimensions" point)
```
 OUTER loop  (33 Hz, the NN / policy level)
   obs = [ p_des(3) , p(3) , v(3) ]   ← 9 numbers
   Δp_des(3) = NN(obs)                ← NN outputs a 3-number CHANGE, not a position
   p_des  = p_des + Δp_des            ← we ADD it to the waypoint  (accumulate!)
        │
        ▼
 INNER loop  (100 Hz, MuJoCo + PID, the physics level)
   PID reads p_des → drives 4 motors → MuJoCo steps physics → new p, v
```
The NN never touches motors. It only nudges the **waypoint** `p_des`. A PID inside
MuJoCo then flies the drone toward whatever `p_des` says. So the NN is a *waypoint
generator*, and the inner loop is a faithful *waypoint follower*.

### What the NN actually outputs
NOT the next position. It outputs `Δp_des` = "how much to move the waypoint this
step" (`action = p_des[t+1] − p_des[t]`, 3 numbers). At eval we **integrate** it:
`p_des += Δp_des`, every step. So `p_des` is a running sum of the NN's own outputs.

### Training vs eval — the one thing that's different (the bug)
- **Training:** `p_des` in the obs is the **expert's** waypoint — always smooth,
  sensible, bounded. The NN only ever practiced on clean inputs.
- **Eval:** `p_des` in the obs is built by **summing the NN's own outputs**. There is
  no expert here. This summing loop **did not exist during training.**

So the NN is now eating its own output. One tiny error → `p_des` drifts a little →
that drifted `p_des` is an input the NN **never saw in training** → it errs more →
`p_des` drifts more → … This snowball is the whole failure. We measured it: the
waypoint altitude `p_des_z` was summed all the way to **−227 m** (227 m underground).

### Then MuJoCo did its job — too well
Nothing is wrong with MuJoCo. The PID was simply *told* the waypoint is 227 m
underground, so it dutifully drove the drone straight into the floor. The drone
"never takes off" because **we commanded it downward**, not because physics failed.

### Where "world frame" fits in (plain version)
`p_des` is an **absolute point in the room** (meters from a fixed origin). When the
snowball above happens, that absolute number can become anything — like −227. If
instead the NN worked in **relative** terms ("move 2 cm forward from where I am right
now") and we re-anchored to the drone's actual position each step, there would be **no
absolute number that can run away**. That is the only thing "use a local/body frame"
means here. It is not exotic — it just removes the runaway absolute waypoint.

### So what did you miss? — nothing about the variables; just this:
1. The NN outputs a **change**, and we **add changes up** in a loop at eval.
2. That add-up loop **was never present in training**, so the NN is unprepared for
   the drifted inputs it now feeds itself (this is "compounding error").
3. Because the waypoint is an **absolute** coordinate, the drift has nothing to stop
   it → −227 m → drone flown into the floor.

### "So the NN output is wrong — is the model badly trained?" — NO.
This is the key distinction, so be precise:

- **YES, the NN's outputs become wrong** — but only *late* in the rollout, and only
  *after its input has gone bad*. The outputs are **correct** while the input is
  normal.
- **NO, the model is not badly trained / too weak / under-trained.** It learned the
  expert mapping correctly. More epochs or a bigger network would **not** help.

The proof is in our own data. Look at the same rollout at the start vs later:

| step | input `p_des_z` | NN output `Δp_des_z` | verdict |
|-----:|----------------:|---------------------:|---------|
| 0–11 | ~1.11 (normal)  | ±0.006 (tiny, correct) | **NN is RIGHT** |
| 207  | −56.8 (insane)  | −0.93 (runaway)        | NN is wrong |
| 413  | −227.7 (insane) | −0.81 (runaway)        | NN is wrong |

When the input is sane (steps 0–11, `p_des_z ≈ 1.11`), the NN outputs the right
thing (hold altitude). It only outputs garbage **once the input is already garbage**
(`p_des_z = −56, −227`). And the training loss was healthy (`empty` → 0.0019), so the
learning itself worked.

**So this is "garbage in → garbage out," not "bad model."** The right mental model:

> A student who aced every practice problem (training) is handed a question from a
> topic that was never taught (an input far outside training). They guess wrong — not
> because they are a bad student, but because the question is off-syllabus. Worse,
> here their wrong guess **rewrites the next question**, dragging it further
> off-syllabus each step.

Why this matters for the fix: if the model were merely "weak," the cure would be
*train more / make it bigger*. It is **not** weak, so that cure does nothing — the
off-syllabus inputs (out-of-distribution `p_des`) would still appear and still get
wrong answers. The cure must instead **stop the inputs from going off-syllabus** —
i.e. keep `p_des` bounded and in-distribution (the local-frame / re-grounding fix
below). That is the difference between fixing the *model* and fixing the *setup the
model runs in*; here it is the **setup**.

The rest of this doc is the evidence and the fix direction for exactly this.

---

## "But this follows the old working code — why does it fail NOW?" (fair question)

You are right that the convention was copied from the working D3IL-avoiding code.
Two things reconcile that with the failure — and **this was a known, written-down,
deferred risk, not a sudden new claim**.

### 1. Yes — the pattern IS the old code, copied faithfully
The old avoiding eval does the *exact same* accumulate-and-feed-back loop
(`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`):
```python
obs = np.concatenate((action[:2], obs))        # prepend desired pos to obs
action, samples = policy({0: obs}, ...)         # predict Δdesired
next_pos_des = action + obs[:2]                 # ACCUMULATE desired pos (open-loop)
obs, ... = env.step(next_pos_des, ...)          # follower tracks it
obs = np.concatenate((next_pos_des[:2], obs))   # feed accumulated desired back into obs
```
So the open-loop accumulation of a desired position that is fed back as observation
is **identical**. The UAV code did not invent anything — it mirrored this.

### 2. The UAV obs was a DOCUMENTED deviation, flagged as an OPEN, UNRESOLVED risk
`logs_in_develop/Gen11/Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md` (dated 2026-06-06,
**"Status: Open concern — not resolved before Epoch 4 data collection"**) already:
- Noted the UAV first **omitted** `p_des` from obs, then **"U2: p_des prepended"** added
  it back to mirror avoiding — i.e. the obs schema was explicitly in flux.
- Rated **🟠 High** the risk that the UAV FM has **no real goal signal**, so it
  "samples a **mixture** of homotopies … coincidental geometry, not principled goal
  conditioning."
- **Predicted the exact failure pattern we now see**: *"Likely to pass the mini-FM
  sanity gate on the empty scene … The problem surfaces in multi-homotopy obstacle
  scenes."* → `empty` trained clean; **`pillars` (4 homotopies) is precisely where it
  blew up.**
- Recorded that "the schema was locked before this concern was raised" and "nothing in
  the logs explicitly discussed the goal-conditioning consequence."

So this isn't me suddenly declaring the old code wrong — the eval **made concrete a
risk that was written down months ago and carried forward unresolved.**

### 3. It is NOT about dimensionality — 3D visual aligning works too. CORRECTION.
An earlier draft of this doc blamed "2-D arm vs 3-D drone." **That is wrong.** The
**3-D visual aligning** task uses the *same* scheme and works:
`obs = [des_c_pos(3), c_pos(3)]`, `action = [dx, dy, dz]`, dynamics
`c_pos[t+1] = c_pos[t] + act[t]` (`config/aligning-d3il-visual.py`,
`fm_visual_aligning_test/eval_fm_visual_aligning.py`). Same 3-D accumulate-and-feed-
back loop, no blow-up. So dimensionality / gravity-as-such is NOT the cause.

The two things that actually differ — both **absent** in our UAV setup:

**(a) The plant TRACKS — so `commanded ≈ actual`, and the obs stays self-consistent.**
The aligning obs pairs *commanded* (`des_c_pos`) with *actual* (`c_pos`). On a
position-controlled arm the actual position follows the command tightly — modelled
literally as `c_pos[t+1] = c_pos[t] + act[t]` — so `des_c_pos ≈ c_pos` **always**, and
the obs lives on the narrow "command = actual" diagonal it was trained on.
Our UAV obs pairs *commanded* `p_des` with *actual* `p`, but the drone does **not**
track tightly: it is 2nd-order under gravity, so `p_des` and `p` **diverge**. The eval
hit `p_des_z = −227` while `p_z = 0.087` — an obs combination (`huge command,
grounded actual`) that is **wildly off the training diagonal** and was never seen.
That off-diagonal obs is the OOD input; the arm can never reach it because it tracks.

**(b) Visual aligning is GOAL-CONDITIONED (images); our UAV is state-only.**
Aligning is `if_vision: True` — it sees camera images, so the policy is anchored to
the actual goal/scene and can self-correct toward it. Our UAV FM sees only
`[p_des, p, v]` with **no goal/image signal** — exactly the 🟠 High risk in
`DPCC_OBS_DEVIATION.md`. With nothing pulling it back toward a goal, once it drifts it
has no reference to recover, so it wanders / commits to a runaway.

> Corrected bottom line: the failure is **not** 2-D-vs-3-D and **not** model capacity.
> The working tasks keep the policy IN-distribution two ways our UAV does neither:
> the plant **tracks** (so commanded≈actual, obs stays on the trained diagonal) **and**
> it is **goal-conditioned** (vision anchors it). Our drone neither tracks tightly
> (commanded/actual diverge → off-diagonal OOD obs) nor has any goal signal (nothing
> to recover toward). Same code, but the two stabilising conditions it relied on are
> both missing here.

### Which of (a)/(b) dominates — a clean discriminating test
`empty` has a **single** homotopy (`N/A`) → no goal ambiguity. So:
- If `empty` **flies** but `pillars` fails → the **goal/conditioning** gap (b) /
  homotopy-mixture is the dominant cause.
- If `empty` **also** runs away → the **plant-tracking / off-diagonal OOD** path (a) is
  dominant, independent of goal ambiguity.
Run the `empty` eval next; it cleanly separates the two.

> Bottom line: nothing "suddenly broke." The UAV faithfully reused a working pattern
> whose known weakness (open-loop absolute desired-position + no goal signal) was
> flagged as an unresolved Epoch-2 concern, and that weakness is benign for a damped
> 2-D arm but catastrophic for a 3-D 2nd-order drone. The eval is the first time we
> actually ran the closed loop long enough on the hard scene to see it.

---

## Symptoms (pillars, seed 6)
- `success = 0`, `contact = 0`, `min_z ≈ 0.086 m` (never airborne), identical
  resting `final_z`, `track_err ≈ 87 m`.

## CONFIRMED root cause (from npz `act_all` / `obs_all`)

**The drone takes off fine, then the z-action runs away and drives it into the floor.**

Rollout 0 altitude trace (`p_des_z` = commanded, `p_z` = actual, `dpdes_z` = action):

| step | p_des_z | p_z | dpdes_z |
|-----:|--------:|----:|--------:|
| 0    | 1.107   | 1.107 | +0.0001 |
| 5    | 1.114   | 1.109 | −0.0024 |
| 11   | 1.113   | 1.111 | −0.0005 |
| 207  | −56.78  | 0.087 | **−0.926** |
| 413  | −227.66 | 0.087 | **−0.814** |

- For the **first ~15 steps the FM is correct**: `dpdes_z ≈ ±0.006`, the drone holds
  z ≈ 1.11 m. So it is **NOT** a "can't command altitude from step 0" problem.
- Then `dpdes_z` **saturates strongly negative** (down to the −1.0 normalizer clamp);
  `sum(dpdes_z) = −229.6` → `p_des_z` integrates to **−227 m** (commanded 227 m
  underground). The PID drags the drone to the floor and it stays pinned at 0.086 m.

Per-axis action stats across all trials confirm z is the culprit:
```
dpdes_x  min=+0.000 max=+0.032 mean=+0.003 std=0.006     (small, fine)
dpdes_y  min=-0.040 max=+0.007 mean=-0.026 std=0.020     (small, fine)
dpdes_z  min=-1.000 max=+0.506 mean=-0.581 std=0.442     (RUNAWAY — hits the clamp)
```
`p_des_z` over the run: min **−262.9**, mean −86. The other axes stay sane.

## The REAL bug: TWO stabilisers the working tasks have and our UAV lacks

The mechanism is **distribution shift in the open-loop feedback** (`p_des` is the
integral of the model's own output AND part of its observation). That alone is not
fatal — the working 3-D aligning task has the same loop. It blows up here because the
**two things that keep the working tasks in-distribution are both missing**:

1. **The plant does not TRACK → commanded and actual diverge → off-diagonal OOD obs.**
   The obs pairs commanded (`p_des`) with actual (`p`). In training they move together
   (the expert PID keeps `p` near `p_des`), so the obs lives on a narrow "command ≈
   actual" diagonal. On a position-controlled **arm** that diagonal is preserved at
   eval (it tracks tightly, `c_pos[t+1]=c_pos[t]+act`). Our **drone is 2nd-order under
   gravity** — once the command drifts, `p` cannot follow: we measured `p_des_z=−227`
   while `p_z=0.087`. That `(huge command, grounded actual)` pair is far off the
   training diagonal → an input the model never saw → garbage out.
2. **No goal conditioning → nothing anchors or recovers the command.** Visual aligning
   is `if_vision:True`; the image anchors the policy to the actual goal so it self-
   corrects. Our UAV FM sees only `[p_des, p, v]`, **no goal/image** — the 🟠 High risk
   in `DPCC_OBS_DEVIATION.md`. With no goal reference, a drift has nothing pulling it
   back, so it commits to the runaway instead of correcting.

Why a bigger network cannot save it: capacity does not make a 2nd-order drone track
like a damped arm, and it does not invent a goal signal that isn't in the obs. The
model is not weak — it is **missing the two conditions that keep the working tasks
in-distribution**.

### Sub-symptoms (consequences, not root causes)
- **S1 dead Δz channel** — constant-altitude expert → `Δp_des_z ≡ 0`
  (`Constant data in dimension 2`); makes z the least-anchored axis, so it diverges first.
- **S2 `eps=1` over-scale** — gives the dead z-action a ±1 m range → decides z fails
  *first*, not *whether* it fails.
- **S3 OOD positive feedback** — the self-conditioned integrator is the path along
  which the off-diagonal drift compounds.

### About the "world-frame / local-frame" note (earlier framing — partially right)
Local/body-frame coordinates would **bound** the state (no −227 m) and add translation-
invariance, so they are a legitimate *mitigation*. But aligning is also absolute-frame
and works, so absolute coordinates are **not** the root cause by themselves — the
**tracking** and **goal-conditioning** gaps above are. Treat local-frame as one tool,
not the diagnosis.

> Refuted earlier guesses: "undertrained" / "can't lift from start" (it flies ~15
> steps first) and "2-D vs 3-D" (3-D aligning works).

## Next-step direction

### The real fix — restore the two stabilisers the working tasks rely on
1. **Add a goal/conditioning signal** (the bigger lever; matches visual aligning's
   image and `DPCC_OBS_DEVIATION.md`'s recommendation). Condition the FM on the actual
   target/waypoint so the command is anchored and recoverable, not free-drifting. This
   is the principled fix for multi-homotopy scenes and the natural E7 direction.
2. **Keep commanded near actual so the obs stays on the trained diagonal.** Close the
   loop on `p_des`: re-ground it to the actual `p` (don't let an open-loop integral run
   away from the drone), and/or condition on actual state rather than the accumulated
   command. This is what the damped arm gets "for free" by tracking; the drone must get
   it by construction.
   - A **local/body-frame** reformulation is one concrete way to achieve #2 (bounded,
     translation-invariant state) — useful, but a *means* to keep commanded≈actual, not
     the diagnosis itself.

### First: run the discriminating `empty` eval (decides #1 vs #2 priority)
`empty` = single homotopy, no goal ambiguity (see test above). Flies → prioritise #1
(goal conditioning). Also runs away → prioritise #2 (tracking / re-grounding).

### Band-aids (triage only, NOT the fix)
A fast eval-only clamp of `Δp_des` / `p_des` would *confirm* the runaway mechanism (the
drone should fly if clamped); dropping Δz or shrinking `eps` masks z specifically.
None restore the missing stabilisers, so treat them as diagnostics, not solutions.

### Confirm upstream
Inspect `data/uav_fm/v1/pillars`: confirm `Δp_des_z ≡ 0` (constant-altitude expert) and
that `(p_des, p)` sit on a tight "command ≈ actual" diagonal in training — that diagonal
is exactly what the drone leaves at eval.

## What U3 bought us
This diagnosis was only possible because U3 added npz `act_all`/`obs_all` + the
altitude panel — the per-step `dpdes_z` trace and the `p_des_z → −227 m` integral are
the smoking gun. Keep running evals with the U3 artifacts.
