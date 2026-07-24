# Why the FM Keeps Planning After a Crash

---

## TLDR — Plain Language (start here)

### Three concepts you need first

**What is `p_des`?**
The "commanded position" — where we are TELLING the drone to fly to next. It is NOT where the
drone actually is. The PID controller then tries to make the real drone (`p`) chase `p_des`.
Every FM step: `p_des = p_des + FM_output`. We move the target; the drone tries to follow.

**What is the "dynamics constraint" / projection?**
Imagine the FM outputs a 8-step flight plan: step 0 → step 1 → step 2 … → step 7. But the
plan can be physically nonsense — e.g. "jump 2 m in one step" when the drone can only move
0.03 m per step. The **dynamics constraint** says: "each step's commanded move (`act`) must
equal the change in `p_des` between that step and the last." So `act[t] = p_des[t] - p_des[t-1]`.
It keeps the plan internally self-consistent like a speed limit on the trajectory.

**What is "projection"?**
The FM outputs a trajectory that might violate the dynamics constraint. The **projector** is
a solver (SLSQP) that takes that illegal plan and finds the *nearest legal plan* that satisfies
the constraint. Think of it as "snap the FM's messy sketch to the nearest physically-valid
blueprint." The cost of that snap is `proj_cost` — high cost means the FM's raw plan was far
from legal, so the projector had to work hard.

---

### Why the drone crashes and the FM doesn't care

Step by step in plain language:

1. **The drone hits the floor and freezes.** Real position `p` stops changing. Velocity `v ≈ 0`.

2. **The FM still runs every 30 ms.** It looks at `obs = [p_des, p, v]` and outputs a new
   `Δp_des`. Nobody tells it to stop.

3. **`p_des` is the FM's own homework.** The FM writes `Δp_des`, we add it to `p_des`, and
   then `p_des` is fed BACK into the FM next step as part of `obs`. The FM is effectively
   reading its own previous answer as the input. `p` and `v` are real, but `p_des` is the
   FM talking to itself.

4. **`p_des` keeps marching forward.** The frozen drone never caught up to `p_des`, so
   `p_des` drifts further ahead each step. After 190 steps it is 2.5 m ahead of the corpse.

5. **The FM sees "huge gap between `p_des` and `p`" and thinks the drone is just far behind.**
   In training, that pattern meant "drone fell behind, issue a big forward command to catch
   up." So it issues a big forward command. Which pushes `p_des` even further ahead. Repeat.

6. **The dynamics constraint cannot help.** It only checks whether the commanded
   PLAN is internally consistent (speed-limit rule above). A plan that says "advance `p_des`
   forward" is perfectly legal by that rule — even if the real drone is dead on the floor 2 m
   behind. The constraint has no knowledge of `p` at all.

7. **The FM never saw a dead drone during training.** All training episodes are successful
   expert flights. `v = (0,0,0)` for 190 steps is a situation the FM has never encountered.
   It makes its best guess based on what it DID see — "large tracking error" — and that
   guess makes things worse.

### Why the original DPCC arm handles crashes better (NOT because obs is different)

**Both DPCC arm and our UAV have self-referential `des_c_pos`/`p_des` in obs** — the FM
reads its own accumulated output as part of the conditioning. The user was right to question
the earlier claim. The actual differences are:

1. **Constraint anchor:** DPCC arm binds `c_pos` (dim 6, REAL position, line 119) to action.
   Our UAV binds `p_des` (dim 3, FM's own setpoint, `eval_fm_uav.py:157`) to action.
   The projector's `skip_initial_state` re-anchors the plan to real `c_pos[0]` every step
   for the arm — so even with self-referential obs, the PLAN is grounded in reality.
   Our UAV anchors the plan to `p_des[0]` (FM's own state) — no grounding.

2. **Arm tracking is near-perfect:** `des_c_pos ≈ c_pos` so the self-referential loop
   never diverges — the arm physically catches up every step.

3. **Environment termination:** DPCC sim fires `terminated=True` on failure. Ours does not.

Our UAV binds `p_des` rather than real `p` because the UAV does NOT track perfectly —
`p[t+1] ≠ p[t] + act[t]` due to PID lag. Binding real `p` makes the constraint
unsatisfiable at every step. Binding `p_des` makes it exactly satisfiable (by definition),
at the cost of losing any grounding to where the drone actually is.

### What would actually fix it

- **Kill switch in the eval loop:** if `‖p - p_des‖ > 0.5 m` for 5 consecutive steps → stop
  the episode. One line of code. This is Priority 3 in `ANALYSIS.md`.
- **Tracking constraint on the projector:** add `‖p_des - p‖ ≤ δ` so the projector
  prevents `p_des` from drifting more than δ metres from the real drone. This is a spatial
  constraint — it would live in `config/uav_eval.yaml` under `obstacle_constraints`.
- **The dynamics constraint alone is not enough** because it only polices the plan's
  internal consistency, not where `p_des` is relative to `p`.

---

## Codebase Taxonomy — who is who

This document references six distinct codebases. Every claim should be anchored to one of
these. **D3IL defines the tasks; DPCC is visual aligning ONLY; everything else in this repo
is FM-PCC's own work.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  D3IL  (KIT, ICLR 2024)  ·  d3il/                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Defines task environments (avoiding + aligning) and datasets.              │
│  Has 11 own IL methods (BC, BET, BESO, DDPM-MLP, IBC, ACT…) — none use    │
│  Janner's TemporalUnet. avoiding/aligning environments vendored into repo.  │
└─────────────────────────────────────────────────────────────────────────────┘
                     ↓ defines envs + dataset format
┌─────────────────────────────────────────────────────────────────────────────┐
│  Janner diffuser  (2022)  ·  /workspaces/diffuser  (read-only reference)   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Original trajectory-level diffusion. d4rl locomotion only (hopper,        │
│  maze2d, walker2d). NOT avoiding/aligning. No FiLM, no p_des, no PID.     │
│  Conditioning = apply_conditioning (inpainting) only.                       │
└─────────────────────────────────────────────────────────────────────────────┘
                     ↓ architecture grafted by DPCC
┌─────────────────────────────────────────────────────────────────────────────┐
│  DPCC  (Carvalho et al.)  ·  diffuser_visual_aligning/ + _test/            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Task: D3IL VISUAL ALIGNING ONLY (the paper's sole contribution).           │
│  Model: GaussianDiffusion (DDPM) + UNet1DTemporalCondModel + FiLM.         │
│  Janner's trajectory architecture grafted onto D3IL's aligning env.        │
│  Added: FiLM (visual ResNet embedding), PCC projector (SLSQP + deriv).    │
│  Trajectory: 9D = act(3) + des_c_pos(3) + c_pos(3).                       │
│  Constraint binds c_pos (real) ← act.                                      │
│  Eval: eval_visual_aligning_dpcc.py — runs to max_episode_length, no       │
│  terminated break.                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                     ↓ FM-PCC adaptations (this repo, our work)
┌─────────────────────────────────────────────────────────────────────────────┐
│  FM-PCC: Diffuser-style avoiding  ·  diffuser_visual_avoiding/ + _test/    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Task: D3IL AVOIDING (FM-PCC applied DPCC approach to avoiding task).      │
│  Model: GaussianDiffusion (DDPM) — same class as DPCC, different task.     │
│  Train script: diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py │
│  Eval "U2 rebuild": eval_visual_avoiding_dpcc.py imports fm_visual_avoiding │
│  (FlowMatchingODE) for eval — mixed naming, beware.                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  FM-PCC: FM visual avoiding  ·  fm_visual_avoiding/ + fm_visual_avoiding_test/│
│  ─────────────────────────────────────────────────────────────────────────  │
│  Task: D3IL AVOIDING with camera ("Gen7" visual avoiding).                  │
│  Model: FlowMatchingODE (FM replaces DDPM).                                 │
│  Eval: eval_fm_visual_avoiding.py — has if terminated: break.              │
├─────────────────────────────────────────────────────────────────────────────┤
│  FM-PCC: FM visual aligning  ·  fm_visual_aligning/ + fm_visual_aligning_test/│
│  ─────────────────────────────────────────────────────────────────────────  │
│  Task: D3IL ALIGNING with camera ("Gen7" visual aligning).                  │
│  Model: FlowMatchingODE (FM replaces DPCC's DDPM). Copy of Gen6V4 → Gen7. │
│  FiLM: inherited from DPCC's UNet1DTemporalCondModel.                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  FM-PCC: UAV-FM (this epoch)  ·  flow_matcher_v3_uav/ + FM_v3_uav_test/   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Task: UAV corridor navigation. NOT D3IL avoiding/aligning.                 │
│  Model: FlowMatchingODE (flow_matcher_v3_uav). No visual/camera. No FiLM. │
│  Trajectory: 12D = act(3) + p_des(3) + p(3) + v(3).                       │
│  Constraint binds p_des (FM's own) ← act. No goal conditioning.            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What comes from D3IL vs what was invented by DPCC vs what is ours:**

| Thing | Invented by | Lives in |
|---|---|---|
| Avoiding + aligning task environments | D3IL (KIT) | `d3il/environments/` |
| `[des_c_pos \| c_pos]` data format | D3IL hardware logger | `d3il/…/logger.py`, `sequence.py` |
| Trajectory-level diffusion (H-step, apply_conditioning) | Janner | `/workspaces/diffuser` |
| FiLM visual conditioning on trajectory FM | DPCC (Carvalho) | `diffuser_visual_aligning/models/` |
| PCC projector (SLSQP + deriv constraint) | DPCC (Carvalho) | `diffuser_visual_aligning/sampling/projection.py` |
| FM (FlowMatchingODE) replacing DPCC's DDPM | FM-PCC ours | `fm_visual_avoiding/`, `fm_visual_aligning/` |
| UAV application + 12D obs + p_des binding | FM-PCC ours | `flow_matcher_v3_uav/`, `FM_v3_uav_test/` |

---

**Question (full version):** Even in `dpcc-*` variants, when the real drone is on the floor
and frozen, the commanded trajectory (`p_des`) keeps advancing as if nothing happened. We
condition on `obs = [p_des, p, v]`, so the FM CAN see the real state. Why doesn't it stop?

**Short answer:** Two compounding mechanisms.
1. `p_des` is not ground truth — it is the FM's own integrated output. The FM is conditioning on itself, not on an independent external signal.
2. The FM was never trained on crashed-drone states. The crash state is out-of-distribution; the model maps it to the nearest in-distribution pattern ("large tracking error — command forward") and diverges.

The dynamics constraint (`dpcc-*`) enforces internal trajectory consistency but has no power to stop this — it does not constrain the relationship between `p_des` and `p`.

---

## 1. What conditioning actually means here

### Observation vector (code)

At the start of each FM step (`eval_fm_uav.py:312`):

```python
obs = np.concatenate([p_des, p, v]).astype(np.float32)   # [p_des | p | v] (9,) raw
```

So `obs_t ∈ ℝ⁹` is:

```
obs_t = [ p_des_t(3) | p_t(3) | v_t(3) ]
         └──FM output──┘ └─MuJoCo state─┘
```

The FM receives this as `conditions = {0: obs}` (`eval_fm_uav.py:315`):

```python
action, traj = policy({0: obs}, batch_size=batch_size, horizon=horizon)
```

### What `apply_conditioning` does

Inside `p_sample_loop`, before and after every denoising/ODE step, `apply_conditioning` is called
(`flow_matcher_v3_uav/models/diffusion.py:185` and `:267`):

```python
x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
```

The function (`flow_matcher_v3_uav/models/helpers.py:157–164`):

```python
for t, val in conditions.items():
    if isinstance(t, str):
        continue
    else:
        x[:, t, action_dim:] = val.clone()   # pin obs slice at timestep t
```

With `cond = {0: obs}`, this **hard-pins the obs dimensions of the trajectory at horizon step
t=0** to the current observed state. Horizon steps t=1, 2, …, H-1 are NOT pinned — the model
fills those in freely.

**The conditioning guarantees exactly one thing:** the trajectory's first obs slot matches what
we observe NOW. The rest of the H-step plan is the model's free prediction.

---

## 2. The self-referential p_des loop (the core bug)

### The update rule

After the FM outputs the first action (`eval_fm_uav.py:323, 338`):

```python
action = np.asarray(action, dtype=float).reshape(-1)[:3]   # first Δp_des from FM
...
p_des = p_des + action                                      # integrate into new setpoint
```

This is the **receding-horizon MPC update**:

```
p_des_{t+1} = p_des_t + FM(obs_t)[0]
```

And the next step's obs is assembled as:

```python
obs_{t+1} = [p_des_{t+1} | p_{t+1} | v_{t+1}]
           = [p_des_t + FM(obs_t)[0]  |  p_{t+1}  |  v_{t+1}]
```

**`p_des` in `obs_{t+1}` is the FM's own previous output accumulated over all past steps.**
It is not an external ground-truth signal. The only external signals are `p` and `v` from
MuJoCo.

### After a crash: what the FM actually sees

Let `τ_crash` be the step where the drone hits the floor and freezes:

```
p_{t}   = p_crash  (constant ∀ t ≥ τ_crash)
v_{t}   ≈ 0        (constant ∀ t ≥ τ_crash)
```

But `p_des_t` is NOT frozen — every step the FM still outputs some `Δp_des` and accumulates
it. So:

```
p_des_{τ_crash + k} = p_des_{τ_crash} + Σ_{i=0}^{k-1} FM(obs_{τ_crash+i})[0]
```

The track error grows unboundedly:

```
track_err_{t} = ‖p_t - p_des_t‖  →  grows as Σ FM outputs ≠ 0
```

From the `rollout_corridor_C_10001.log`:

```
step 52 (crash):  p = (-2.316, -0.477, 0.087),  p_des_x = -2.800,  track_err = 0.304 m
step 242 (end):   p = (-2.316, -0.477, 0.087),  p_des_x = -0.286,  track_err = 2.072 m
                                                  Δp_des accumulated = +2.514 m
```

The drone is frozen. `p_des` marched 2.5 m ahead by itself.

---

## 3. Why the FM does not "notice" (out-of-distribution failure)

### What the training distribution looks like

The FM was trained on expert trajectories from `uav_expert_data_collect/generator.py`. Every
episode in the dataset satisfies:

```
‖p_t - p_des_t‖ ≤ ε_track      (drone tracking the setpoint)
‖v_t‖ > 0                       (drone actually moving)
z_t > 0.2 m                     (airborne)
```

The noise injected at data collection time (`dataset_writer.py:31`, `NOISE_SIGMA = 0.02 m`)
is small — it thickens the manifold but keeps it in the "healthy flight" region.

### The crash state is out-of-distribution

After the crash, the FM receives:

```
obs = [p_des | p_crash | v≈0]   with ‖p_crash - p_des‖ → 2 m
```

`v = (0, 0, 0)` for 190 consecutive steps never appears in training data. Large track error
(`> 0.5 m`) appears briefly in training (drone catching up), not persistently.

The FM has no explicit OOD detector. It maps the crash observation to the nearest in-distribution
latent pattern, which is **"drone far behind the setpoint — issue large forward Δp_des."**
That is the correct behaviour for a recovering drone; it is catastrophic for a dead drone
because it makes `p_des` diverge further, which makes `obs_{t+1}` look even more like
"drone even further behind — issue even more forward Δp_des."

This is a **positive feedback loop in obs space**:

```
crash → large track_err → FM outputs large Δp_des → p_des advances → larger track_err → ...
```

### Why proj_cost returned to normal (the "healthy" signal is wrong here too)

After step 52 (`rollout_corridor_C_10001.log`), proj_cost dropped back to 0.8–1.2.
This is because `v ≈ 0` and the FM correctly predicts small `Δp_des` for a nearly-stationary
input — the dynamics constraint `act[t] = Δp_des[t]` is trivially satisfied by small actions.
Low proj_cost does NOT mean healthy flight. It means the FM and the dynamics model have
converged on the "nothing is moving" basin.

---

## 4. Why the dynamics constraint (DPCC) cannot fix this

### What the constraint enforces

The DPCC projector enforces (`eval_fm_uav.py:157`, `setup_dpcc_projector`):

```python
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
# binds p_des (dims 3,4,5) to act (dims 0,1,2)
```

In math, for each horizon step `h`:

```
x[h, 0] = x[h, 3] - x[h-1, 3]     (act_x  = Δp_des_x)
x[h, 1] = x[h, 4] - x[h-1, 4]     (act_y  = Δp_des_y)
x[h, 2] = x[h, 5] - x[h-1, 5]     (act_z  = Δp_des_z)
```

This is a constraint on the **internal structure of the planned trajectory** — that the
action dims are consistent Euler derivatives of the p_des dims. It says nothing about:

- where `p_des` is relative to `p`
- whether `p` is moving
- whether the drone has crashed

A trajectory that has `p_des` advancing 2.5 m ahead of a frozen drone is **perfectly
dynamics-consistent** — the actions are valid Euler steps of `p_des`. The projector has no
violation to fix.

The projector would need a spatial constraint `‖p_des - p‖ ≤ δ` (a tracking constraint)
to catch this. That constraint is not wired this epoch.

---

## 5. Summary: three independent failures stacked

| Layer | What fails | Why DPCC doesn't fix it |
|---|---|---|
| **obs construction** (`eval_fm_uav.py:312`) | `p_des` in obs is FM's own integrated output, not an external signal | Constraint is over `(act, p_des)` consistency — not `(p_des, p)` proximity |
| **FM generalisation** (`flow_matcher_v3_uav/models/diffusion.py:185`) | crash state is OOD → FM maps to "catch up" pattern → positive feedback | Projector does not alter the FM's conditioning or its "which direction to go" prediction |
| **episode control** (`eval_fm_uav.py:309`, `for k in range(n_fm)`) | no termination check — eval runs for all `n_fm` steps regardless of drone state | No signal from environment; MuJoCo never calls `done=True` |

**Fix required at all three layers:**

1. **Tracking constraint:** add `‖p_des - p‖ ≤ δ` as a spatial constraint in the projector
   — this limits how far `p_des` can drift from `p` within one planning horizon.
2. **Episode termination:** `if track_err > 0.5 and consecutive >= 5: break` in
   `eval_fm_uav.py` after the physics loop (Priority 3 in `ANALYSIS.md`).
3. **(Future) recovery-aware training:** expose crashed/recovery states in training data so
   the FM learns a "hover and wait" or "re-center p_des" policy for OOD inputs.

---

## 6. Q: Why is DPCC's obs setup OK, but ours is not? And why not use real `p` instead of `p_des`?

> **User-raised correction:** "I really think DPCC uses the same logic — check the workspace."
>
> Verified. The user is right. DPCC visual-aligning has the SAME self-referential obs
> structure. The difference is not in the obs — it is in which dimension the constraint
> pins, and in what happens when the system fails. Details below.

### What DPCC visual-aligning actually puts in obs

`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:1441`:

```python
obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,) [des_c_pos | c_pos]
```

So the 6D obs is `[des_c_pos(3) | c_pos(3)]` — BOTH desired AND real position. The
trajectory format at `line 7–9`:

```python
# trajectory is 9D  [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
```

And `des_robot_pos_np` is self-referential — it comes from the sim, which updated it as
`des_c_pos[t] + action[t-1]` (FM's own previous output). Inside `predict()` the agent also
tracks this locally (`line 1657`):

```python
self.mental_robot_pos += next_action_np   # FM's own accumulated output, same as des_c_pos
```

**So DPCC has exactly the same self-referential `des_c_pos` pattern as our `p_des`.**
The user is correct.

### What IS different: which dimension the constraint is pinned to

Our UAV (`eval_fm_uav.py:157`):
```python
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
#                              ↑ dim 3 = p_des_x  (FM's own accumulated setpoint)
```

DPCC visual-aligning (`eval_visual_aligning_dpcc.py:119–121`):
```python
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))
#                                ↑ dim 6 = c_pos_x  (REAL end-effector position from sim)
```

The `skip_initial_state` logic in `projection.py:394–398` pins the initial state of the
constrained dimension to the current obs value (`b[counter*horizon] = s_0[x_idx]`).

For DPCC: `s_0[6]` = current real `c_pos` from the sim → constraint chain starts at
**measured ground truth**. Every replanning step re-anchors the plan to where the arm
actually is.

For our UAV: `s_0[3]` = current `p_des` = FM's own accumulated output → constraint chain
starts at **the FM's own history**, not where the drone actually is.

| | DPCC visual-aligning | Our UAV FM-PCC |
|---|---|---|
| Obs structure | `[act \| des_c_pos \| c_pos]` — both desired and real | `[act \| p_des \| p \| v]` — both desired and real |
| Self-referential in obs? | **Yes** — `des_c_pos` comes from FM's accumulated output | **Yes** — `p_des` comes from FM's accumulated output |
| Constraint binds dim | `c_pos` (dim 6–8, **real measured**) ← `act` | `p_des` (dim 3–5, **FM's own**) ← `act` |
| Plan initial anchor | `c_pos[0]` = real position from sim — **ground truth** | `p_des[0]` = FM's accumulated output — **self-referential** |
| Constraint satisfiable? | Approximately yes (arm nearly tracks) | Exactly yes (by definition — it's a tautology) |

### Why DPCC is still "more OK" than us despite both having self-referential obs

**Reason 1 — constraint anchor is real state.**
Even though `des_c_pos` is self-referential in obs, the CONSTRAINT forces
`c_pos[t+1] = c_pos[t] + act[t]` starting from REAL `c_pos[0]`. So after projection, the
planned `c_pos` sequence is realistic — it starts where the arm actually is. For our UAV,
the planned `p_des` sequence starts where the FM's own history says it is, which after a
crash is 2 m ahead of the drone.

**Reason 2 — arm tracking is nearly perfect (`des_c_pos ≈ c_pos`).**
`eval_visual_aligning_dpcc.py:738`:
```python
# (c_pos not exposed by D3IL; des_c_pos ≈ c_pos under PD control)
```
Because the arm's PD controller is stiff, `c_pos ≈ des_c_pos` at every step. The gap
between the self-referential `des_c_pos` and real `c_pos` is small — a few mm, not the
0.1–2 m gap our UAV accumulates after a crash. The self-referential feedback loop exists
but never diverges because the arm always catches up.

**Reason 3 — environment termination signal.**
The original DPCC visual aligning eval (`eval_visual_aligning_dpcc.py`) does NOT have a
`if terminated: break` — it runs for `max_episode_length` steps (line 1963), same as us.
What it does have: the D3IL aligning environment is sim-native and constrains the arm
physically (joint limits, collision detection), so catastrophic OOD divergence is bounded
by the sim itself. Our MuJoCo UAV has no such physical bounding — the drone freezes and
the FM runs free for all `n_fm` steps (`eval_fm_uav.py:309`).

Note: the file `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` in this repo
DOES have `if terminated: break` at line 447 — but that is **FM-PCC's own visual avoiding
adaptation** (imports `fm_visual_avoiding.models`), not the original DPCC paper.

### Why we bind `p_des` and not real `p`

The arm can use real `c_pos` in the constraint because `c_pos ≈ des_c_pos` — the Euler
equation `c_pos[t+1] ≈ c_pos[t] + act[t]` is approximately true.

For our UAV, binding real `p` would require:
```
p[t+1] = p[t] + act[t]
```
But this is FALSE every step — `p[t+1]` depends on PID dynamics, rotor inertia, attitude,
aerodynamic lag. The constraint would be violated by 0.01–0.2 m at EVERY step. The SLSQP
projector would permanently fight an unsatisfiable constraint; `proj_cost` would be high
even during perfect flight and the projection would be meaningless.

So we bind `p_des` because the constraint `p_des[t+1] = p_des[t] + act[t]` is EXACTLY
satisfiable — because that is literally how we compute `p_des` in the eval loop
(`eval_fm_uav.py:338`: `p_des = p_des + action`). It is a tautology, not a physics model.
That exactness is useful (constraint is always feasible, proj_cost reflects only how much
the FM's plan needed to be reshaped). The cost is that the constraint has no relationship
to where the drone (`p`) actually is.

`eval_fm_uav.py:130–132` captures this design decision:
```python
# The dynamics `deriv` binds **p_des (3,4,5)** to the action (0,1,2) — NOT the actual p —
# because p_des is the exact integrator of the action (`p_des[t+1]=p_des[t]+act`), while
# the drone's p lags.
```

### Why Janner's original diffuser has no `p_des` at all

Checked `/workspaces/diffuser` directly. Janner's rollout (`/workspaces/diffuser/scripts/plan_guided.py`):

```python
observation = env.reset()                              # real state from env

for t in range(args.max_episode_length):
    conditions = {0: observation}                      # conditions = REAL state, nothing accumulated
    action, samples = policy(conditions, ...)
    next_observation, reward, terminal, _ = env.step(action)   # action goes directly to env
    ...
    if terminal:                                       # env signals done
        break
    observation = next_observation                     # obs = REAL next state from env
```

And `apply_conditioning` (`/workspaces/diffuser/diffuser/models/helpers.py:142`):

```python
def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()    # pins REAL state into plan at step t
    return x
```

Three structural differences from both DPCC and our UAV:

| | Janner diffuser | DPCC arm / our UAV |
|---|---|---|
| `conditions` source | `env.reset()` / `env.step()` — real env state | FM's own accumulated `des_c_pos` / `p_des` mixed with real |
| What action does | `env.step(action)` directly → env evolves | `p_des += action` → accumulates setpoint; env tracks `p_des` via PID |
| Self-referential? | **No** — obs always comes from the environment | **Yes** — `des_c_pos`/`p_des` in obs is the FM reading its own history |
| Termination | `if terminal: break` — env signals done | DPCC: `if terminated: break`; our UAV: **never fires** |

Janner's FM outputs `action` that goes directly to `env.step()`. The environment evolves by its own physics and returns the next REAL state. Nothing is accumulated outside the env. There is no concept of a desired position separate from the actual position — the FM plans in actual state space and the env enforces physics.

DPCC and our UAV introduced `p_des` when layering a PID controller under the FM. That is where the self-referential pattern was born — the FM's output goes to a setpoint integrator rather than directly to the environment, and that accumulated setpoint feeds back into the FM's obs.

### Summary: what causes the divergence

Both DPCC and our UAV have self-referential `des_c_pos`/`p_des` in obs. The divergence
after a crash happens in BOTH architectures in principle. DPCC avoids it in practice
because:

1. The constraint re-anchors the plan to real `c_pos` every step — limiting how far the
   planned state can drift from reality
2. The arm tracks closely enough that the self-referential gap never grows large
3. The environment signals termination when things go wrong

Our UAV has none of those three safeguards: constraint anchored to FM's own `p_des`,
imperfect tracking, no termination signal. The fix is at point 3 — a termination condition
(`track_err > 0.5 m for N steps → break`) — because adding points 1 and 2 requires
redesigning the obs space and retraining.

### The honest trade-off table

| Choice | Constraint anchor | Self-reference in obs | Constraint satisfiable | What breaks |
|---|---|---|---|---|
| Bind `p_des` (ours) | FM's own output — no grounding | Yes (both codebases) | Always exactly — it's a definition | Crash: plan drifts from frozen `p` with no bound |
| Bind real `p` | Real measured state — grounded | Yes (but plan is re-anchored) | Only approximately (PID lag) | UAV: proj_cost permanently high, SLSQP fights physics every step |
| DPCC arm (bind `c_pos`) | Real measured state — grounded | Yes (des_c_pos still self-ref) | Approximately (arm tracks well) | Arm: works because tracking is tight; fails if arm gets stuck but env terminates |

---

## 7. Q: Why not always use the dynamics constraint — and is it even "real dynamics"?

### Part A: what the constraint actually is (not what the name implies)

When you hear "dynamics constraint" you might think: UAV mass, inertia, motor thrust model,
aerodynamic drag, quaternion kinematics. **None of that is in this constraint.**

The constraint is a single Euler integration equation wired in `projection.py:344–401`:

```
x_0[t+1] = x_0[t] + dt * x_2[t]   # from build_matrices docstring, line 349
```

For our UAV with `('deriv', [3, 0])` = (p_des_x, act_x) and `dt=1.0`:

```
p_des_x[t+1] = p_des_x[t] + 1.0 * act_x[t]
```

That is the constraint. For all three axes x, y, z (`eval_fm_uav.py:157`):

```python
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
```

In plain terms: **"the commanded position must move by exactly the commanded action, step by
step, across the H-step plan."** It is a setpoint-consistency rule, not a UAV physics model.

The real UAV dynamics — thrust, attitude, rotor speed, drag — live entirely inside MuJoCo,
handled by the PID at 100 Hz. The FM and DPCC projector never see any of that. They operate
on the outer position-setpoint loop only.

### Why `dt=1.0` and not `1/33 s`?

Because `act` IS `Δp_des` by definition. The "derivative" is not velocity (m/s) — it IS the
position increment (m/step). So `dt=1.0` means one FM step. If `dt` were `1/33`, the
constraint would be `p_des[t+1] = p_des[t] + (1/33)*act[t]`, requiring `act` in units of
m/s. Our actions are in m/step, so `dt=1.0` (`eval_fm_uav.py:183`):

```python
dt=config.get('dt', 1.0),  # action IS Δp_des → Euler dt=1.0 (NOT 1/33)
```

### Part B: why not always use the dynamics constraint?

**For `dpcc-*` variants, the dynamics constraint IS always used.** The only variants without
it are:

| Variant | Why no dynamics |
|---|---|
| `diffuser` | No projector at all — pure FM baseline, no modification |
| `model_free` | Spatial constraints only (`eval_fm_uav.py:156`: `'model_free' not in variant`) |
| `model_free-tightened` | Same — spatial only with tightening margin |

The `diffuser` variant exists purely as a **measurement baseline** — "what does the FM alone
do, with zero projection overhead?" Without it, you can't measure how much the projector
helps or costs. The `model_free` variants exist to isolate the effect of spatial constraints
independently of dynamics. These are research comparisons, not production variants.

The gate in code (`eval_fm_uav.py:156`):

```python
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
```

And in `config/uav_eval.yaml:43`:

```yaml
constraint_types: ['dynamics']   # 'dynamics' is the only active type this epoch
```

So `dynamics` IS in `constraint_types` for all runs. The only thing that turns it off is the
`model_free` variant name check.

### Part C: what the constraint actually buys us (and doesn't)

**What it buys:** the FM's raw H-step plan might be incoherent — e.g. the actions don't
add up to the `p_des` deltas across the horizon. The projector snaps the plan so every step
satisfies `p_des[t+1] = p_des[t] + act[t]`. This means the first executed action `act[0]`
is the ACTUAL change you apply to `p_des`, not a hallucinated action that happens to be in
a trajectory where the other steps are inconsistent. This is the whole point of PCC — make
the plan internally physically consistent before extracting the executed action.

**What it doesn't buy:** any relationship between `p_des` and where the drone (`p`) actually
is. The constraint enforces plan-internal consistency, not plan-world consistency. The drone
could be 2 m away from `p_des` and the constraint would be perfectly satisfied.

### What DPCC code says

The `DynamicConstraints.build_matrices` in `flow_matcher_v3_uav/sampling/projection.py:382–401`
builds the equality constraint matrix `A` and vector `b` such that `Ax = b` enforces Euler
integration across the horizon:

```python
# projection.py:389–391 (no normalizer branch — clearest form)
mat_append[i, i * self.transition_dim + x_idx]       = 1        # p_des[t]
mat_append[i, i * self.transition_dim + dx_idx]      = self.dt  # + dt * act[t]
mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1       # - p_des[t+1] = 0
```

Then SLSQP minimises `‖trajectory - FM_raw‖²` subject to `Ax = b` (`projection.py:127,135–138`):

```python
constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)
res = minimize(fun=cost_fun, ..., constraints=constraints, method='SLSQP', ...)
```

The cost function is "stay as close as possible to the FM's raw trajectory." The constraint
is "enforce Euler integration." The result is the nearest Euler-consistent trajectory to what
the FM wanted — which is the "projection" in PCC.

---

## 8. Where did the `[des_c_pos | c_pos | act]` format come from? (Origin investigation)

The user's question: DPCC's 6D obs with both desired and actual positions is not in
Janner's diffuser — so where does it come from? Is it from the DPCC paper, or somewhere
else? Investigated by reading the workspace repos.

### Answer: it comes from the D3IL robot hardware logger, not from any paper design choice

The chain is fully traceable through four files.

**Step 1 — D3IL IK controller writes `des_c_pos` at runtime**

`d3il/environments/d3il/d3il_sim/controllers/IKControllers.py:75, 309`:

```python
self.desired_c_pos = desired_pos          # receives commanded Cartesian target
xd_d = self.desired_c_pos - robot.current_c_pos   # IK tracks toward it
robot.des_c_pos = self.desired_c_pos      # stores command on robot object
```

The D3IL robot arm uses an IK (inverse kinematics) controller. At every timestep it receives
a Cartesian position TARGET (`desired_c_pos`) and tries to reach it by computing joint
angles. It stores BOTH the target it received (`des_c_pos`) and the actual position it
achieved (`current_c_pos` / `c_pos`). This is standard robot controller bookkeeping — not
specific to DPCC.

**Step 2 — Data logger records both**

`d3il/environments/d3il/gamepad_control/logger/logger.py:71, 74`:

```python
env_state["robot"]["c_pos"]     = robot_logger.cart_pos    # actual EE position
env_state["robot"]["des_c_pos"] = robot_logger.des_c_pos   # commanded EE target
```

The logger saves both into the episode pickle. Every episode file in the D3IL aligning/
avoiding datasets contains `des_c_pos` and `c_pos` as separate arrays.

**Step 3 — DPCC dataset loads both into the trajectory**

`diffuser_visual_aligning/datasets/sequence.py:76–83`:

```python
robot_des_pos = env_state['robot']['des_c_pos']   # (T+1, 3) commanded positions
robot_c_pos   = env_state['robot']['c_pos']       # (T+1, 3) actual positions

obs_6d  = np.concatenate([robot_des_pos[:T], robot_c_pos[:T]], axis=-1)  # (T, 6)
actions = (robot_des_pos[1:] - robot_des_pos[:-1]).astype(np.float32)    # (T, 3) Δdes_c_pos
```

And the trajectory returned to the FM is:

```python
trajectories = np.concatenate([act_norm, obs_norm], axis=-1)   # (H, 9) [act|des_c_pos|c_pos]
```

**This is where the 9D format `[act(3) | des_c_pos(3) | c_pos(3)]` is born** — in the
dataset loader, from data that the D3IL logger recorded from the hardware controller.
The same file says explicitly (`sequence.py:26–28`):

```
Why 9D: DPCC projector enforces Euler dynamics on the *actual* robot position
(c_pos, indices 6-8).  des_c_pos alone (6D) would project on command targets
instead of real end-effector positions, violating the DPCC physical contract.
```

**Step 4 — FM trains on this format → must receive it at eval**

The FM is trained on trajectories of shape `(H, 9) = [act | des_c_pos | c_pos]`. At eval
time it therefore expects `conditions = {0: obs_6d}` where `obs_6d = [des_c_pos | c_pos]`.
The `des_c_pos` at eval time is the FM's accumulated output (self-referential) — not because
anyone designed it that way, but because the FM IS the thing generating new `des_c_pos`
values, so what it sends to the controller gets recorded as `des_c_pos` the next step.

### Why Janner's d4rl does NOT have this

`/workspaces/diffuser/diffuser/datasets/d4rl.py` — Janner's d4rl loader:

```python
# d4rl returns: observations, actions, rewards, terminals
# observations = real joint angles + velocities (qpos + qvel)
# NO des/actual split — torque-controlled robots have no "commanded position" concept
```

d4rl uses **torque-controlled** locomotion robots (HalfCheetah, Walker2d, Hopper). The
action is a torque vector — there is no separate "desired Cartesian position" that a
lower-level IK controller tracks. The entire state is the real joint configuration. So:

- No `des_c_pos` in the dataset → no `des_c_pos` in obs → no self-referential feedback
- Action goes directly to the physics simulation (`env.step(action)`)
- The FM plans in real state space throughout

The self-referential pattern only emerges when adapting the FM to a **position-controlled**
robot whose dataset happens to record both commanded and actual positions.

### The DPCC paper itself

The DPCC paper (Carvalho et al.) describes the trajectory as containing both state and
action dimensions but is ambiguous about the exact split in the 6D/9D observation. It does
not discuss the `des_c_pos` vs `c_pos` distinction as a design choice — because it was not
one. It inherited the format from the D3IL robot's existing logging infrastructure.

The paper's key contribution is the PROJECTION mechanism (the SLSQP solver) and the
constraint formulation (`deriv` = Euler link). The observation format was determined by
what was available in the D3IL dataset.

### The chain of origin

```
D3IL hardware IK controller (IKControllers.py:75, 309)
  ↓  records both commanded target and actual position
D3IL data logger (logger.py:71, 74)
  ↓  saves [des_c_pos | c_pos] per timestep in episode pkl
DPCC dataset loader (sequence.py:76–83)
  ↓  builds obs = [des_c_pos | c_pos], action = Δdes_c_pos, traj = [act | obs] (9D)
FM training
  ↓  learns P(Δdes_c_pos | des_c_pos, c_pos)
FM eval
  ↓  must receive [des_c_pos | c_pos] as input
  ↓  des_c_pos at eval = FM's own accumulated output (self-referential — unavoidable given training format)
CRASH: des_c_pos drifts from frozen c_pos → divergence
```

**The self-referential problem is not a bug or design choice. It is the unavoidable
consequence of training on position-controlled robot data where the controller records
commanded positions, and then deploying the FM as the thing that generates those commands.**

### Our UAV inherits the same pattern for the same reason

Our dataset records:
- `targets` = `des_c_pos` equivalent = `p_des` (commanded setpoint)
- `qpos[:3]` = `c_pos` equivalent = `p` (actual drone position)
- `actions = targets[t+1] - targets[t]` = `Δp_des` (same as `Δdes_c_pos`)

`dataset_writer.py` builds the obs exactly as D3IL does: commanded + actual. The FM trains
on `[p_des | p | v]` and at eval must receive `p_des` — which is its own past output.

---

## 9. Three-way comparison: Janner diffuser vs DPCC arm vs our UAV (verified from repos)

### Janner's original diffuser — verified from `/workspaces/diffuser`

`/workspaces/diffuser/scripts/plan_guided.py` — the actual rollout loop:

```python
observation = env.reset()                           # real state, nothing accumulated

for t in range(args.max_episode_length):
    conditions = {0: observation}                   # REAL state pinned at t=0
    action, samples = policy(conditions, ...)
    next_observation, reward, terminal, _ = env.step(action)   # action → env directly
    ...
    if terminal:                                    # env signals done → break
        break
    observation = next_observation                  # obs = REAL next state from env
```

`/workspaces/diffuser/diffuser/models/helpers.py:142` — `apply_conditioning`:

```python
def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()   # pins real env state into plan
    return x
```

Three things to note:
1. `conditions = {0: observation}` — observation is `env.reset()` / `env.step()` return — always real environment state, never accumulated
2. `action` goes directly to `env.step(action)` — no setpoint integrator, no `p_des`
3. `observation = next_observation` — updated with real env return every step, not FM output
4. `if terminal: break` — env signals termination

**There is no `p_des` at all.** The FM plans in real state space. The environment enforces
physics and signals done. Janner's architecture has none of the problems described in this
document.

### DPCC visual aligning (original paper task) — self-referential pattern

The original DPCC paper covers the **visual aligning** task only — not avoiding. The
self-referential `des_c_pos` accumulation in `eval_visual_aligning_dpcc.py` happens
via the agent's internal state tracker (`line 1657`):

```python
self.mental_robot_pos += next_action_np   # FM's own accumulated output
```

And the obs built for the next step (`line 1441`):

```python
obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])   # [des | real]
```

Same self-referential loop as our UAV: `mental_robot_pos` accumulates FM outputs, feeds
back as the "desired" half of the 6D obs. The aligning eval runs to `max_episode_length`
(line 1963) — no `terminated` signal breaks it early.

Note: `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` in this repo IS a
real file, but it is **FM-PCC's own visual avoiding adaptation** (imports
`fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching`) — NOT the
original DPCC paper. It has `if terminated: break` (line 447) that DPCC aligning lacks.

### Full three-way comparison table

| Mechanism | Janner diffuser (`/workspaces/diffuser`) | DPCC visual aligning (original paper) | Our UAV FM-PCC |
|---|---|---|---|
| Self-referential setpoint in obs | **No** — `obs = env.step()` always | **Yes** — `mental_robot_pos += action` (`aligning_eval:1657`) | **Yes** — `p_des += action` (`eval_fm_uav.py:338`) |
| Action destination | `env.step(action)` directly | `des_c_pos += action` → PD arm tracks | `p_des += action` → PID UAV tracks |
| Obs after step | `next_observation` from env | `[mental_robot_pos \| real_c_pos]` | `[p_des \| p \| v]` from MuJoCo |
| OOD crash → FM hallucinates | N/A — real env always consistent | Bounded by arm joint limits | **Yes** — drone frozen, FM runs free |
| Dynamics constraint | None — no projector in base | Present, binds `c_pos` (real) ← `act` | Present, binds `p_des` (FM's own) ← `act` |
| Constraint anchor | N/A | `c_pos[0]` = real measured position | `p_des[0]` = FM's accumulated output |
| Episode termination | `if terminal: break` — env signals | Runs to `max_episode_length` (line 1963) — no `terminated` break | **Never** — `for k in range(n_fm)` |
| Where `p_des` concept was born | **Absent** | Introduced here — FM as setpoint generator | Inherited from DPCC |

### Conclusion

The `p_des` self-reference problem is a **DPCC extension artefact** — it does not exist in
Janner's original diffuser (`/workspaces/diffuser`). Janner's FM outputs actions that go
directly to `env.step()` and receives real state back. There is nothing to accumulate.

DPCC introduced `p_des` when adding a PD controller under the FM for the arm. From that
point, the FM's output goes to a setpoint integrator whose state feeds back into obs. Our
UAV inherits this.

Our UAV is worse than DPCC's arm at handling the resulting problem for two compounding
reasons (not three — the termination claim was wrong): constraint anchored to FM's own
`p_des` (not real `c_pos`), and imperfect UAV tracking (PID lag, unlike near-perfect arm
PD). The practical fix remains: a kill switch on `track_err` in the eval loop.

---

## 10. H2H: D3IL's own DDPM vs Janner's diffuser — the math that shows what DPCC grafted

> **User question:** "What is D3IL's original 6D used in the DDPM model? What does Janner's
> original diffuser use? Compare the math H2H."

These are two fundamentally different diffusion formulations. Understanding both makes it
immediately clear what DPCC grafted and why the self-referential problem emerged.

### D3IL's own DDPM — action-level diffusion, per-timestep

D3IL's DDPM (`d3il/agents/ddpm_agent.py` + `DiffusionMLPNetwork`) does NOT plan a horizon-length
trajectory. It denoises a **single action** conditioned on a **single observation vector**.

**Avoiding task (`d3il/environments/dataset/avoiding_dataset.py:55–64`):**

```python
robot_des_pos = env_state['robot']['des_c_pos'][:, :2]  # (T, 2)  x,y only
robot_c_pos   = env_state['robot']['c_pos'][:, :2]       # (T, 2)  x,y only

input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)  # (T, 4)
vel_state   = robot_des_pos[1:] - robot_des_pos[:-1]                  # (T-1, 2) Δdes_xy

observations = input_state[:-1]   # shape (T-1, 4)
actions      = vel_state           # shape (T-1, 2)
```

**Aligning task (`d3il/environments/dataset/aligning_dataset.py:62–84`):**

```python
input_state = np.concatenate((
    robot_des_pos,   # (T, 3)
    robot_c_pos,     # (T, 3)
    push_box_pos,    # (T, 3)
    push_box_quat,   # (T, 4)
    target_box_pos,  # (T, 3)
    target_box_quat, # (T, 4)
), axis=-1)                        # (T, 20)   ← full env state, NOT just 6D robot

vel_state = robot_des_pos[1:] - robot_des_pos[:-1]  # (T-1, 3)

observations = input_state[:-1]   # shape (T-1, 20)
actions      = vel_state           # shape (T-1, 3)
```

Important: D3IL's own aligning uses **20D obs** (full robot + both boxes), not 6D. The DDPM
is conditioned on everything the agent can see, including the target. Obs dimensions 0–5 contain
`[des_c_pos | c_pos]` but they are embedded in 20D alongside box poses.

**The DDPM model call (`ddpm_agent.py:161`):**

```python
loss = self.model.loss(action, state, goal)
#                      ─────   ─────
#                      (2D or 3D)  (4D or 20D)
#                      ← diffusion variable    ← conditioning
```

The diffusion **variable** `x` is the action. The obs `state` is the condition. This is
`P(action | obs)` — a conditional distribution over actions given a state snapshot.

**Math of D3IL DDPM inference:**

```
Given:  s_t ∈ ℝ^{4 or 20}   (current obs)
Denoise: x ∈ ℝ^{2 or 3}     (action)  from pure noise
Model:   ε_θ(x_noisy, s_t, k) → ε     (MLP denoiser, no temporal blocks)
Output:  a_t ∈ ℝ^{2 or 3}   (single action for this step)
Execute: des_c_pos_{t+1} = des_c_pos_t + a_t
Next obs: s_{t+1} from env (new des_c_pos + c_pos from environment)
```

**Key property: the diffusion variable `x` is only the action.** The obs `s_t` is injected as a
conditioning signal to the MLP — it never gets pinned, snapped, or modified by the denoising
loop. No `apply_conditioning`. No horizon. No trajectory structure. The obs always comes from
the environment between steps.

---

### Janner's original diffuser — trajectory-level diffusion

**Math of Janner inference (`/workspaces/diffuser/scripts/plan_guided.py`):**

```
Given:  s_0 ∈ ℝ^{obs_dim}          (current env state)
Build:  τ ∈ ℝ^{H × (action_dim + obs_dim)}   (full H-step trajectory)
              └─ diffusion variable x ─┘

Denoise: x = pure noise (H, D) → denoised trajectory
At each denoising step k:
    x = TemporalUnet(x, k)           # 1D UNet over time axis
    x = apply_conditioning(x, {0: s_0})  # hard-pin s_0 into x[:,0,action_dim:]
Output:  τ[:, 0, :action_dim] = a_0  (execute only first action of H-step plan)
Execute: env.step(a_0) → s_1  (real env, real physics)
Next:    s_0 = s_1 from env  (always real state, never accumulated)
```

The diffusion **variable** `x` is the ENTIRE H-step trajectory `[act | obs]` for all H steps.
The `TemporalUnet` is a 1D convolutional UNet that denoises across the time axis. `apply_conditioning`
pins the first obs slot to current real state **inside the denoising loop** at every step.

**Trajectory tensor format (Janner d4rl, locomotion tasks):**

```
x[h] = [ a_h(action_dim) | s_h(obs_dim) ]   for h = 0 … H-1
x.shape = (H, action_dim + obs_dim)

s_h = real joint angles + velocities (qpos + qvel)
a_h = joint torques
→ NO des/actual split. One ground-truth state per step.
```

---

### Head-to-head math comparison

| Dimension | D3IL DDPM | Janner diffuser | DPCC (the graft) | Our UAV |
|---|---|---|---|---|
| **Diffusion variable `x`** | `a_t ∈ ℝ^2` or `ℝ^3` (single action) | `τ ∈ ℝ^{H×(act+obs)}` (full trajectory) | `τ ∈ ℝ^{H×9}` (trajectory) | `τ ∈ ℝ^{H×9}` (trajectory) |
| **Denoising network** | `DiffusionMLPNetwork` (MLP, state-conditioned) | `TemporalUnet` (1D UNet over time) | `UNet1DTemporalCondModel` (≈ TemporalUnet) | same `flow_matcher_v3_uav` TemporalUnet |
| **Horizon H** | none (window_size=1) | 32–128 steps | 32 steps | 8 steps |
| **`apply_conditioning`** | **absent** — obs is just MLP input | present — pins obs at `t=0` inside denoising | present — pins 6D obs at `t=0` | present — pins 9D obs at `t=0` |
| **Obs in trajectory** | N/A — obs is conditioning, NOT part of `x` | `s_h = env.step()` — real state every step | `[des_c_pos \| c_pos]` — includes self-referential `des_c_pos` | `[p_des \| p \| v]` — includes self-referential `p_des` |
| **What `des_c_pos`/`p_des` is in `x`** | N/A | N/A (no desired/actual split) | dim 3–5 of `x[h]` — FM's accumulated setpoint | dim 3–5 of `x[h]` — FM's accumulated setpoint |
| **`apply_conditioning` source** | N/A | `env.step()` — always fresh ground truth | `[des_c_pos | c_pos]` — `des_c_pos` is FM output | `[p_des | p | v]` — `p_des` is FM output |
| **Action destination** | `env.step(a_t)` → real physics | `env.step(a_0)` → real physics | `des_c_pos += a_0` → PID tracks | `p_des += a_0` → PID tracks |
| **Self-referential obs?** | **No** — obs always from env between steps | **No** — obs always from `env.step()` | **Yes** — `des_c_pos` is FM's own accumulated output | **Yes** — `p_des` is FM's own accumulated output |
| **Root cause of self-reference** | N/A | N/A | Trajectory diffusion requires obs in `x`; `des_c_pos` is the FM's own output | Same |

---

### Where the self-reference was born

D3IL's DDPM never had a self-referential problem because **the diffusion variable is the action
only**. The obs is injected as a conditioning vector to the MLP and discarded after the forward
pass. It has no slot in `x` to accumulate. Between steps, the obs is freshly read from the
environment. Nothing from the FM's output flows back into the obs.

Janner's diffuser never had it either because **the trajectory includes only real env states**.
`apply_conditioning` pins `env.step()` output into `x[:,0,:]` — always fresh ground truth.

DPCC introduced self-reference by doing two things simultaneously:
1. Adopting Janner's trajectory-level formulation (so `x` must contain obs at every step `h`)
2. Using D3IL's `[des_c_pos | c_pos]` obs format (so the obs in `x` includes `des_c_pos`)

When the FM generates `des_c_pos` as output (because that is what the trajectory predicts),
the executed `des_c_pos` gets fed back in as the next obs via `apply_conditioning`. No one
chose this — it is the unavoidable consequence of training a trajectory FM on a dataset where
obs includes a field generated by the controller, and then using the FM as the controller.

---

### Why D3IL kept obs and action separate (the correct design in hindsight)

D3IL's DDPM design is:
```
Train:  P(a_t | s_t)  — learn the distribution of actions given state
Infer:  sample a_t ~ P(a_t | s_t)  — get one action
Step:   env.step(a_t) → s_{t+1}    — environment provides next state
```

The agent never writes anything into the obs. The env is the sole source of truth for obs.
The action is the only thing the FM produces. This is a clean factorisation.

Janner's design does the same at the trajectory level — the env provides all obs values; the
FM fills in what actions to take at each step of the H-step plan.

DPCC's problem is that it adopted Janner's trajectory structure, where **the trajectory tensor
`x` must carry obs values at every horizon step** (for `apply_conditioning` to pin `t=0`, and
for the UNet to attend to obs across the horizon). But D3IL's obs includes `des_c_pos` — and
when the FM is the thing setting `des_c_pos`, the FM is writing into its own future obs.

D3IL's action-only DDPM sidesteps this cleanly: obs never enters `x`, so the FM can never
corrupt it. The price is that D3IL's DDPM has no lookahead — it acts greedily one step at a
time. DPCC and Janner get multi-step lookahead but must carry obs in `x`. For Janner (real
states only) this is fine. For DPCC/UAV (`des_c_pos` in obs) this creates the self-reference.

---

## 11. Trajectory tensor and FiLM: visual-aligning (DPCC) vs D3IL native vs Janner vs our UAV

> **User question:** "Check the visual aligning — our model vs D3IL — is the 9D tensor and
> FiLM the same as the D3IL setup? And is FiLM working well based on diffuser, since diffuser
> is the ground truth?"

### Part A — Trajectory tensor: what is 9D, and who has it?

**D3IL native aligning (`d3il/environments/dataset/aligning_dataset.py:77`):**

```python
input_state = np.concatenate((
    robot_des_pos,   # (T, 3)
    robot_c_pos,     # (T, 3)
    push_box_pos,    # (T, 3)
    push_box_quat,   # (T, 4)
    target_box_pos,  # (T, 3)
    target_box_quat, # (T, 4)
), axis=-1)   # → (T, 20)   obs = 20D

actions = robot_des_pos[1:] - robot_des_pos[:-1]   # (T-1, 3)
# Total trajectory dim for D3IL DDPM: 20 + 3 = 23D per step
```

**D3IL native avoiding (`d3il/environments/dataset/avoiding_dataset.py:58, 60`):**

```python
input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)  # (T, 4)  x,y only
vel_state   = robot_des_pos[1:] - robot_des_pos[:-1]                  # (T-1, 2)
# Total trajectory dim for D3IL DDPM: 4 + 2 = 6D per step
```

**D3IL does NOT use a 9D tensor** for either task. Its 9D claim would require robot-only obs
(6D) + 3D action — that is NOT how D3IL loads its data. D3IL always includes box/target
object state in the observation (aligning: +14D). The 9D format was introduced by DPCC.

**DPCC visual-aligning (`diffuser_visual_aligning/datasets/sequence.py:76–83`):**

```python
obs_6d  = np.concatenate([robot_des_pos[:T], robot_c_pos[:T]], axis=-1)   # (T, 6)
actions = (robot_des_pos[1:] - robot_des_pos[:-1]).astype(np.float32)      # (T, 3)
traj    = np.concatenate([act_norm, obs_norm], axis=-1)                     # (H, 9)
# 9D = act(3) + des_c_pos(3) + c_pos(3)
# Object/box positions moved to VISUAL branch (ResNet image embedding)
```

DPCC gets to drop to 6D robot obs by providing box/goal state via the CAMERA — the FiLM
visual embedding carries goal context that D3IL's 20D obs carries explicitly. Without FiLM,
DPCC would have no goal information at all.

**Our UAV (`eval_fm_uav.py:446`, `eval_fm_uav.py:312`):**

```python
obs = np.concatenate([p_des, p, v])   # (9,): p_des(3) + p(3) + v(3)
action = Δp_des                        # (3,)
# trajectory_dim = 12 = act(3) + obs(9)
# 12D = Δp_des(3) + p_des(3) + p(3) + v(3)
```

We added velocity `v(3)` relative to DPCC because UAV inertia makes velocity essential
for dynamics prediction. The result is 12D per step, NOT 9D.

**Comparison table:**

| Codebase | Obs vector | Action | Traj dim/step | Goal info source |
|---|---|---|---|---|
| D3IL DDPM avoiding | `[des_xy\|c_xy]` = 4D | `Δdes_xy` = 2D | 6D | Both object positions in obs |
| D3IL DDPM aligning | `[des\|c\|box\|quat\|tgt\|tgt_q]` = 20D | `Δdes_c` = 3D | 23D | Full env state in obs |
| DPCC visual-aligning | `[des_c_pos\|c_pos]` = 6D | `Δdes_c_pos` = 3D | **9D** | FiLM visual embedding |
| Janner diffuser (d4rl) | `qpos+qvel` = env-specific | torques = env-specific | env-dim | None (reward-guided) |
| Our UAV | `[p_des\|p\|v]` = 9D | `Δp_des` = 3D | **12D** | None (scene-implicit in training) |

**The 9D is DPCC's own invention.** It is not the same as D3IL's native format (which is 6D
or 23D). DPCC reduced D3IL's 20D obs to 6D robot-only obs precisely BECAUSE FiLM provides
the missing goal information from the camera. Without FiLM, the DPCC model would be blind
to object positions and goals.

---

### Part B — FiLM: where it exists, where it doesn't

FiLM (Feature-wise Linear Modulation) means: the conditioning signal modulates the network's
intermediate feature maps via scale-and-shift. In the ResidualTemporalBlock context:

```
out = Conv1d(x) + time_mlp(t)      # t here = time embedding ± cond embedding
```

If a conditioning vector is concatenated into `t`, it modulates every channel of every
ResidualTemporalBlock in the UNet — the conditioning is "baked into the weights" of every
layer. This is global FiLM conditioning.

**Janner's TemporalUnet (`/workspaces/diffuser/diffuser/models/temporal.py:49–146`):**

```python
class TemporalUnet(nn.Module):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, ...):
        time_dim = dim
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), ...)
        # cond_dim is accepted but NEVER used anywhere in __init__
        # No cond_mlp, no cond embedding, no concatenation

    def forward(self, x, cond, time):
        t = self.time_mlp(time)   # only time; cond is received but IGNORED
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)      # t = time only
            ...
```

`cond` is in the signature but **unused**. The `cond_dim` parameter in `__init__` does
nothing — it was likely reserved for future use. Janner's conditioning = `apply_conditioning`
only (hard inpainting of t=0 obs slot). **No FiLM in Janner.**

**Our UAV `Flow_matcher_U_Net_v2` (`flow_matcher_v3_uav/models/unet1d_temporal_cond.py:87–237`):**

```python
class Flow_matcher_U_Net_v2(ModelMixin, ConfigMixin):
    def __init__(self, horizon, transition_dim, cond_dim, dim=128, ...):
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), ...)
        # No cond_mlp defined — cond_dim accepted but unused

    def forward(self, x, cond, time, returns=None, ...):
        t = self.time_mlp(timesteps)   # only time
        # cond is received, NEVER referenced again
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)   # t = time only
            ...
```

Identical pattern to Janner. `cond` received, ignored. **No FiLM in our UAV model.**
Conditioning = `apply_conditioning` only (inpainting of t=0 obs slot, `helpers.py:157–164`).

**DPCC's `UNet1DTemporalCondModel` (`diffuser_visual_aligning/models/unet1d_temporal_cond.py:84–267`):**

```python
class UNet1DTemporalCondModel(ModelMixin, ConfigMixin):
    def __init__(self, ..., cond_dim, use_cond_projection=False):
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), ...)

        if use_cond_projection and cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
            cond_embed_dim = dim   # will be concatenated with time embedding
        else:
            self.cond_mlp = None
            cond_embed_dim = 0

        embed_dim = dim + cond_embed_dim   # ← grows when FiLM enabled

    def forward(self, x, cond, time, ...):
        t = self.time_mlp(timesteps)

        if self.cond_mlp is not None and cond is not None and isinstance(cond, torch.Tensor):
            cond_pooled = cond.mean(dim=1)          # pool visual embeddings over time axis
            cond_emb = self.cond_mlp(cond_pooled)   # project to dim
            t = torch.cat([t, cond_emb], dim=-1)    # concat: t = [time_emb | cond_emb]

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)   # t now carries both time AND visual context
```

`cond` = visual embeddings from ResNet/encoder (shape `[B, T_img, cond_dim]`). Pooled over
the image-temporal axis → projected to `dim` → concatenated with the time embedding. The
combined `t` modulates ALL `ResidualTemporalBlock`s. **FiLM IS PRESENT in DPCC.**

---

### Part C — FiLM "working well" relative to Janner's ground truth

The user's question: does Janner validate FiLM as a good mechanism?

**The answer: Janner does not use FiLM, so it cannot validate or invalidate it.**

| | FiLM in network? | How goal info is provided | Conditioning mechanism |
|---|---|---|---|
| Janner diffuser | **No** — `cond` ignored in forward | No goal (reward-guided via energy) | `apply_conditioning` only (inpainting) |
| D3IL DDPM | **No** — obs is MLP input, not FiLM | Goal/target positions in obs vector explicitly | State-conditioned MLP, no trajectory planning |
| DPCC visual-aligning | **Yes** — `cond_mlp` when `use_cond_projection=True` | ResNet image embedding (FiLM) + inpainting at t=0 | Both FiLM + `apply_conditioning` |
| Our UAV | **No** — same as Janner | No goal conditioning at all | `apply_conditioning` only (inpainting) |

**Key finding: our UAV FM has NO goal conditioning.**

Janner's FM is guided during rollout by a value/energy function (reward-conditioned sampling
from `plan_guided.py`). D3IL's DDPM has the target position in the 20D obs vector. DPCC has
FiLM from the camera image. **Our UAV FM relies entirely on the training distribution to
encode goal information** — the goal is implicit in the trajectory dataset structure, not
explicitly provided at inference time.

This is architecturally coherent (the FM was trained on a specific set of scenes and
homotopies, so routes are baked in), but it means the FM has NO way to generalise to
unseen goals or scenes. If the training scene and goal were fixed, this works. If they vary,
the FM cannot condition on a new goal unless the scene variant appeared in training.

**Whether DPCC's FiLM is working correctly** is a separate question from Janner. FiLM is
DPCC's own contribution — Janner's code cannot answer it. From the code structure:

```
FiLM correct path: visual_encoder → [B, T, cond_dim] → pool → cond_mlp → [B, dim]
                   → torch.cat([time_emb, cond_emb]) → ResidualTemporalBlocks
```

The mechanism is standard and well-implemented. Whether it HELPS depends on whether the
visual encoder is trained (or pretrained) to extract goal-relevant features — that is a
training question, not an architecture question. Architecturally, the FiLM path in
`UNet1DTemporalCondModel` is correct and mirrors common practice in diffusion-based policies.

---

### Part D — architecture family tree (updated)

```
Janner TemporalUnet (2022)
  │  trajectory diffusion, inpainting only, NO FiLM
  │  action_dim + obs_dim = transition_dim
  │
  ├─→ DPCC UNet1DTemporalCondModel (grafted)
  │     same trajectory structure (H, transition_dim)
  │     transition_dim = 9 (act3 + des_c3 + c3)         ← D3IL robot obs
  │     ADDED: FiLM from visual encoder (cond_mlp)       ← DPCC contribution
  │     ADDED: PCC constraint projector                   ← DPCC contribution
  │     obs reduced 20D→6D because FiLM carries goal     ← key coupling
  │
  └─→ Our UAV Flow_matcher_U_Net_v2 (FM-PCC graft)
        same trajectory structure (H, transition_dim)
        transition_dim = 12 (act3 + p_des3 + p3 + v3)   ← added velocity
        NO FiLM (no visual encoder, no goal image)        ← our simplification
        PCC constraint projector (from DPCC)              ← inherited
        goal info = NONE (scene implicit in training data)← gap vs DPCC
```

The key coupling to notice: DPCC reduced D3IL's 20D obs to 6D BECAUSE FiLM provides goal
information via the camera. Our UAV copied the 6D→9D (robot-only) obs idea but did NOT
add FiLM. That means our FM has no mechanism to distinguish between different goal
locations — all goal information must be compressed into the training distribution.

---

## 12. Is DPCC a "brutal mix" of Janner diffuser into D3IL? (Confirmed — architecture graft investigation)

> **User question:** "Is the DPCC brutal mix of the diffuser into the D3IL outputs? He actually
> uses the Janner diffuser to train a D3IL avoiding job, and the avoiding job is actually
> designed for another ML model. Check."
>
> **Short answer: Yes.** D3IL is a standalone benchmark with its own IL implementations —
> none of which use Janner's architecture. DPCC authors took Janner's diffuser (GaussianDiffusion +
> TemporalUnet-style backbone + `apply_conditioning`) and grafted it onto D3IL's avoiding/aligning
> task environments and dataset format. These are two separate codebases stitched together at the
> environment and dataset interface layer.

### What D3IL actually is

`d3il/README.md` (KIT ICLR 2024 benchmark):

```
D3IL — Benchmarking Imitation Learning with Diverse Tasks and Environments
ICLR 2024, Karlsruhe Institute of Technology

11 imitation learning methods: BC, BET, BESO, DDPM, IBC, ACT, GPT-BC, CVAE, ...
7 tasks: avoiding, aligning, stacking, pushing, sorting, inserting, unfolding
```

D3IL is a **comparison benchmark** — its job is to evaluate multiple IL methods on a shared set of
robotic tasks. The task environments (avoiding, aligning, etc.) and the data collection framework
(IK controller + logger + pickle format) were designed to be method-agnostic. Any of the 11 methods
could be swapped in.

### D3IL's own diffusion implementations (not Janner)

D3IL ships with its **own** diffusion-based agents:

**`d3il/agents/models/diffusion/diffusion_models.py`:**
```python
class DiffusionMLPNetwork(nn.Module):
    """MLP-based denoising network — NOT a temporal UNet."""
    # adapted from twitter/diffusion-rl (offline RL diffusion)
    # action-only, not trajectory-level planning
```

**`d3il/agents/models/diffusion/diffusion_policy.py:18`:**
```python
# code adapted from https://github.com/twitter/diffusion-rl/blob/master/agents/diffusion.py
class Diffusion(nn.Module):
    ...
    # state + goal conditioning; samples single actions, not H-step trajectories
    # p_sample_loop produces shape (batch, action_dim), NOT (batch, horizon, action_dim)
```

D3IL's `ddpm_agent` uses `DiffusionMLPNetwork` — a state-conditioned MLP denoiser that samples
**one action at a time**, not an H-step trajectory plan. No horizon, no `apply_conditioning`, no
PCC projector. The `beso_agent` uses k-diffusion (completely separate framework). Neither has
Janner's TemporalUnet anywhere in their stack.

**Grep result across all D3IL agents:**
```
grep "TemporalUnet" d3il/   → zero hits
grep "GaussianDiffusion" d3il/  → zero hits
grep "apply_conditioning" d3il/ → zero hits
```

D3IL does not import from or reference Janner's diffuser at any point.

### DPCC's architecture — what it actually uses

DPCC (`diffuser_visual_aligning/`) uses a **completely different** architecture that was imported
from Janner's diffuser, not extended from D3IL's agents:

**`diffuser_visual_aligning/models/diffusion.py:15`:**
```python
class GaussianDiffusion(nn.Module):
    # GaussianDiffusion with cosine schedule, predict_epsilon, apply_conditioning
    # This is Janner's DDPM-trajectory architecture, re-implemented in FM-PCC codebase
```

**`diffuser_visual_aligning/models/visual_gaussian_diffusion.py:6`:**
```python
from diffuser_visual_aligning.models.diffusion import GaussianDiffusion
from diffuser_visual_aligning.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    # Extends Janner's GaussianDiffusion with visual goal conditioning
```

**`diffuser_visual_aligning/models/unet1d_temporal_cond.py:84`:**
```python
class UNet1DTemporalCondModel(ModelMixin, ConfigMixin):
    # Temporal 1D UNet with ResidualTemporalBlocks
    # Architecture mirrors Janner's TemporalUnet (same building blocks, adapted for visual cond)
```

**`diffuser_visual_aligning/models/helpers.py`:**
```python
def apply_conditioning(x, conditions, action_dim, goal_dim=0):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()   # identical to Janner's helpers.py:142
```

The DPCC model stack is: `UNet1DTemporalCondModel` (≈ Janner's `TemporalUnet`) inside
`VisualGaussianDiffusion` (extends Janner's `GaussianDiffusion`) with `apply_conditioning`
(identical to Janner's). This is Janner's trajectory-diffusion architecture with a visual
conditioning layer on top.

### What was grafted where

The graft boundary is the dataset/environment interface:

```
┌─────────────────────────────────────────────────────────┐
│   D3IL layer (environment + data collection)            │
│   ─────────────────────────────────────────────────     │
│   IKControllers.py: position-controlled robot arm       │
│   logger.py:        records [des_c_pos | c_pos]         │
│   sequence.py:      builds obs=(6D), act=(3D), traj=9D  │
│   Task environment: avoiding/aligning (D3IL designed)   │
│   Evaluation infra: MuJoCo sim, episode loop            │
├─────────────────────────────────────────────────────────┤
│   DPCC graft layer (Janner architecture applied)        │
│   ─────────────────────────────────────────────────     │
│   GaussianDiffusion → VisualGaussianDiffusion           │
│   UNet1DTemporalCondModel (≈ TemporalUnet)              │
│   apply_conditioning (pins obs at t=0)                  │
│   PCC projector (SLSQP + deriv constraints) ← DPCC NEW │
│   Train on D3IL traj format → eval in D3IL environment  │
└─────────────────────────────────────────────────────────┘
```

The D3IL layer was designed for method-agnostic IL (ddpm-MLP, BeSo, BeT, IBC, ACT…). DPCC plugged
in Janner's trajectory-level diffuser at the boundary where D3IL's dataset hands off training data.
The D3IL environment continues to run as before — the DPCC authors replaced only the model and
added the projector. They did NOT extend D3IL's `ddpm_agent` or any of D3IL's own diffusion code.

### Why this matters for the self-referential problem

The tension in §6–§8 has a clear root: **Janner's diffuser assumes real-state observations**
(no `p_des`, action → env directly, obs from `env.step()`). **D3IL's robot records commanded
positions separately from actual positions** (`des_c_pos` + `c_pos`, because the position-
controlled IK arm can't avoid this split). When you graft Janner's architecture onto D3IL's data:

1. The FM must accept `[des_c_pos | c_pos]` because that is what D3IL's dataset provides
2. At eval, the FM generates new `des_c_pos` — which becomes the input to itself next step
3. The self-referential `p_des` loop emerges **as the unavoidable consequence of the graft**,
   not as a design choice by either Janner or D3IL

D3IL designed the avoiding/aligning environments for action-output IL methods. Janner designed
GaussianDiffusion for real-state Markovian planning. DPCC bridged them by training Janner's FM on
D3IL's `[des_c_pos | c_pos]` format — inheriting the split, and with it the self-referential obs.

Our UAV inherits this exactly: same `[p_des | p | v]` split, same self-referential FM-as-
setpoint-generator pattern, same dynamics constraint (from DPCC), for the same reason.

### Summary

| Layer | Origin | In DPCC | In D3IL's own agents |
|---|---|---|---|
| Task environment (avoiding, aligning) | D3IL (KIT) | Used as-is | Used as-is |
| Dataset format `[des_c_pos\|c_pos\|act]` | D3IL logger → sequence.py | Used as-is | Used as-is |
| Diffusion model class | **Janner** (GaussianDiffusion, TemporalUnet) | **Grafted in** | Never used |
| `apply_conditioning` | **Janner** | **Grafted in** | Never used |
| Action-only diffuser | twitter/diffusion-rl | Not used | ddpm_agent |
| k-diffusion (BeSo) | karras et al. | Not used | beso_agent |
| PCC constraint projector | DPCC authors | New contribution | Does not exist |

D3IL's avoiding task was designed for any IL method — DPCC used it as a host for Janner's
trajectory FM. The "brutal mix" description is accurate: two entirely separate codebases (Janner's
trajectory diffuser, D3IL's position-controlled robot benchmark) merged at the dataset format
boundary. The self-referential `p_des` problem documented throughout this file is one of the
architectural tensions that resulted.

---

## 13. The Ultimate Question: Does the DPCC Paper (arXiv:2412.09342) Actually Justify Using `[des_c_pos | c_pos]` and Conditioning on `p_des` Rather Than Real `p`?

> **Paper:** "Diffusion Predictive Control with Constraints" — Römer, von Rohr, Schoellig (2024)
> **arXiv:** https://arxiv.org/abs/2412.09342

### Short answer: No. The paper states the design but never justifies it.

The paper uses `[des_c_pos | c_pos]` in state and conditions on both without explaining WHY.
The self-referential eval-time problem is never mentioned. However, the implicit reasoning
CAN be reconstructed from one key passage about the inner controller.

---

### What the paper actually says (exact quotes)

**State and action (Section 6.1):**

> "The state **s_t** ∈ ℝ⁴ consists of the current and desired end-effector positions
> in the 2D plane. The action **a_t** ∈ ℝ² contains the **desired Cartesian velocities**,
> which are sent to a low-level controller."

State = 4D = `[des_xy(2) | c_xy(2)]`. The paper says action = "velocities."
**This is misleading.** See the code reality below.

**The black-box inner controller (Section 6.1) — the KEY passage:**

> "We **do not assume knowledge of the dynamics of the low-level controller**. Instead,
> we approximate the system dynamics by a simple Euler integration:
>
> **s_{t+1} = s_t + [a_t^T, a_t^T]^T · ts + w_t**
>
> where ts is the sampling time, and the model mismatch **w_t** accounts for the
> low-level controller and the numerical error of the Euler integration."

**"Velocity" vs position delta — the paper is wrong about units.**

The paper calls the action "desired Cartesian velocities" (m/s). The code computes:

```python
# diffuser_visual_aligning/datasets/sequence.py:83
actions = (robot_des_pos[1:] - robot_des_pos[:-1]).astype(np.float32)
# = des_c_pos[t+1] - des_c_pos[t]  =  POSITION DELTA  (metres, not m/s)
```

And the projector is configured with `dt=1.0`:

```yaml
# config/visual_aligning_eval.yaml:24
# dt=1.0 IS CORRECT for this task: actions are position deltas [dx,dy,dz], not
```

With `ts = dt = 1.0`, the Euler equation `s_{t+1} = s_t + [a,a]^T · 1.0` reduces to:

```
des_c_pos[t+1] = des_c_pos[t] + action
c_pos[t+1]     = c_pos[t]     + action + w_t
```

So `action = Δdes_c_pos` — a **position increment in metres**, not a velocity in m/s.
Setting `dt=1.0` (one step, dimensionless) makes velocity and position delta numerically
equal, so the paper's math works. But physically:

- The paper says the action goes to a "low-level velocity controller"
- D3IL's actual IK controller (`IKControllers.py:75`) receives `desired_c_pos` —
  a **Cartesian position target**, not a velocity command
- The FM outputs `Δdes_c_pos` → the eval loop computes `new_des_c_pos = old + action` →
  sends the NEW POSITION to the IK controller

**The paper's "velocity" label is a semantic error.** The action is a position increment.
The IK controller is position-controlled, not velocity-controlled. The paper describes the
architecture as if the FM issues velocity setpoints, but the code integration is:

```
paper says:   action → velocity controller → arm moves at v m/s
reality:      action = Δpos → new_des_pos = old_des_pos + action → IK receives position target
```

The Euler dynamics in the paper `s_{t+1} = s_t + [a,a]^T · ts` are CORRECT as written,
because with `ts=1` the math is identical regardless of whether you call `a` a velocity
(m/s with ts=1s) or a position delta (m with ts=1 step). The label "velocity" in §6.1 is
simply wrong — it is a position increment. This is why `dt=1.0` is hardcoded.

**Conditioning (Section 5 / Equation 6):**

> "**c = (s_t, g)** — Get current state s_t and set c = (s_t, g)"

The paper treats `s_t = [des_c_pos | c_pos]` as a single observable state to be read
at each replanning step. No distinction is made between the two components in conditioning.

---

### What the Euler dynamics model `s_{t+1} = s_t + [a, a]^T · ts + w_t` actually means

Unpacking the concatenated update `[a_t^T, a_t^T]^T`:

```
des_c_pos[t+1] = des_c_pos[t] + a_t · ts        ← EXACT (pure integrator, no mismatch)
c_pos[t+1]     = c_pos[t]     + a_t · ts + w_t  ← APPROXIMATE (inner controller error in w_t)
```

The action `a_t` (commanded velocity) drives BOTH components of state by the same amount.
For `des_c_pos` this is a definition — it IS the integrator. For `c_pos` it is an
approximation — the real arm follows approximately but with lag captured in `w_t`.

This is the implicit (unstated) reason why `[des_c_pos | c_pos]` must both appear in state:
- `des_c_pos[t+1]` is EXACT given `a_t` → the FM can plan future `des_c_pos` exactly
- `c_pos[t]` is needed to compute **tracking error** `= des_c_pos[t] - c_pos[t]`
- The FM needs tracking error to plan adaptively (if arm is lagging, future commands should compensate)
- The constraint on `c_pos` (DPCC aligning: dims 6–8 ← act) enforces `c_pos[t+1] ≈ c_pos[t] + a_t · ts`
  under the Euler approximation — meaningful safety enforcement even with `w_t`

The paper never says any of this. It just writes the dynamics and moves on.

---

### What the paper DOES NOT address: the self-referential problem

The paper's conditioning line is: "Get current state **s_t**."

It does not say HOW `des_c_pos` is obtained as part of `s_t` at eval time. In the
paper's mental model, `s_t` is a SENSOR READING — a number you read off the robot. But:

```
In training:  des_c_pos at time t = the commanded position logged from the D3IL IK controller
              (ground truth — what the human teleoperator actually commanded)

In eval:      des_c_pos at time t = what the FM set as the commanded position at t-1
              (model-accumulated — NOT a ground-truth sensor reading)
```

**These are not the same thing.** In training, `des_c_pos` is an independent signal (human
command). In eval, `des_c_pos` is the FM's own previous output. The paper treats `s_t` as
a sensor reading in both cases — this discrepancy is never mentioned.

The paper's formulation implicitly assumes: `des_c_pos` at eval = real robot's stored
desired position = what was last commanded = FM's own output. For the D3IL arm this is
approximately true (the IK controller stores `robot.des_c_pos` and you can read it back
at the next step). But "reading back what you last wrote" IS the self-referential loop.
The paper does not call it out as a design tension.

---

### Does the paper justify conditioning on `des_c_pos` rather than just `c_pos`?

Not explicitly. There is exactly ONE piece of indirect evidence:

The Euler dynamics model uses the same action to update BOTH components:
`des_c_pos[t+1] = des_c_pos[t] + a·ts` (exact), `c_pos[t+1] = c_pos[t] + a·ts` (approx).

If you dropped `des_c_pos` from state and used only `c_pos`, the FM would lose:
1. Knowledge of the current command target (where the arm was sent last step)
2. The exact integrator dimension — the dimension where the dynamics constraint is
   EXACTLY satisfiable without any `w_t` error
3. Tracking error signal (des_c_pos - c_pos) needed for adaptive planning

But the paper never frames it this way. It is left as an unexplained design choice derived
from the D3IL dataset format (which records both, as shown in §8).

---

### What this means for our UAV

The paper's design `s_t = [des_c_pos | c_pos]` with action = velocity came from fitting
the trajectory diffuser to a POSITION-CONTROLLED ARM with a black-box inner controller.

Our UAV has:
- `s_t = [p_des(3) | p(3) | v(3)]` — added velocity `v` for inertia (no equivalent in arm)
- Action = `Δp_des` (position delta, not velocity — same thing at fixed dt)
- Inner controller = PID (also black box to the FM)
- Constraint binds `p_des` ← `act` (not `p` ← `act`, unlike DPCC aligning binds `c_pos`)

The constraint difference is consequential:
- DPCC binds `c_pos` (current pos, real): constraint chain starts at real arm position
- Our UAV binds `p_des` (setpoint, FM's own): constraint chain starts at FM's own history

Both come from the paper's Euler model `s_{t+1} = s_t + [a,a]^T·ts`, but we chose
to enforce the EXACT half (`des_c_pos`) while DPCC chose the APPROXIMATE half (`c_pos`).
DPCC's choice gives better safety grounding (constraint anchored to real state). Our choice
gives exact constraint satisfaction (no `w_t` error, `proj_cost` reflects FM quality only).

---

### Verdict

| Question | Answer |
|---|---|
| Does the paper justify `[des_c_pos \| c_pos]` in state? | **Implicit only** — Euler model requires both; stated without explanation |
| Does the paper justify conditioning on `des_c_pos` not `c_pos`? | **No** — both included in `s_t`, treated equally as "state" |
| Does the paper discuss the self-referential eval problem? | **No** — `s_t` treated as a sensor reading; `des_c_pos` being model-accumulated at eval is never acknowledged |
| Does the paper justify the `deriv` constraint on `c_pos`? | **Implicitly** — Euler model `c_pos[t+1] = c_pos[t] + a·ts + w_t` is what the constraint enforces |
| Is the inner controller dynamics justified? | **Partially** — "We do not assume knowledge" + Euler approximation + `w_t` mismatch term |
| Is the H=8 horizon justified? | **Empirically only** — "B=4 trajectories with H+1=8" stated as hyperparameter |
| Is the action "desired Cartesian velocity" as the paper claims? | **No — paper is wrong about units.** Code computes `Δdes_c_pos` (position delta, metres). `dt=1.0` makes the numbers identical but the action goes to an IK POSITION controller, not a velocity controller. The "velocity" label is a semantic error. |

The paper's contribution is the PROJECTOR mechanism (SLSQP + constraint tightening), not
the observation design. The `[des_c_pos | c_pos]` format was inherited from D3IL's data
collection infrastructure (§8) and the paper accepted it as-is. The self-referential
consequence of using the FM itself as the `des_c_pos` generator at eval time is an
unexamined gap in the paper that carries through to our UAV implementation.

---

## 14. Does the Paper Say Whether Conditioning Uses `des_c_pos` or Real `c_pos`? (It's Absurd)

> **User question:** "Did the paper say it uses p_des or real p as diffuser conditioning?
> Did it legalize or at least hint which way it wants to do? This is absurd and weird!"

### The algorithm — every word the paper says

The complete DPCC algorithm pseudocode (quoted verbatim from arXiv:2412.09342):

```
DPCC Algorithm:

Input: Diffusion model ε_θ, goal g, dynamics f, state/action constraints

Set t = 0
Compute tightened constraints via (17)

While goal g not reached:
    ① Get current state s_t and set c = (s_t, g)
    ② Sample trajectory batch from noise: τ^{K,1:B} ~ N(0, I)
    For k = K,...,1:
        ③ Denoising step: τ̃^{k-1} ~ N(μ_θ(τ^k, k, c), σ²_k I)
        ④ Model-based projection: τ^{k-1} = Π(τ̃^{k-1})
    Select trajectory τ* from batch
    ⑤ Apply first action a_{t|t} from τ*
    Set t ← t+1
```

And from Section 6.1: **"condition on the current state using inpainting"**

That is ALL the paper says. Step ① is one line. The paper never says:
- Where `des_c_pos` in `s_t` comes from at t=1, t=2, …
- Whether `des_c_pos` is a sensor reading or the FM's own accumulated output
- Whether to condition on `des_c_pos` ONLY, `c_pos` ONLY, or BOTH

---

### What "Get current state s_t" actually means in the code

The paper defines: **`s_t ∈ ℝ⁴ = [des_xy(2) | c_xy(2)]`** — a 4D vector of BOTH components.

Conditioning via inpainting (`apply_conditioning`) pins the FULL `s_t` at t=0 of the trajectory:

```python
x[:, 0, action_dim:] = s_t   # pins [des_c_pos | c_pos] at horizon step 0
```

So the FM is conditioned on **BOTH** `des_c_pos` AND `c_pos` simultaneously. The paper
treats `s_t` as a single observable "current state" — a number you read off the robot.

But the two components come from FUNDAMENTALLY DIFFERENT sources at eval time:

```
s_t = [ des_c_pos_t  |  c_pos_t ]
        ───────────     ─────────
        FM's own        Real arm
        accumulated     measured
        output          from D3IL
        (self-ref)      (ground truth)
```

**`c_pos_t`** = read from `env.robot.current_c_pos` — the arm's actual Cartesian position,
measured by the sim. This is ground truth.

**`des_c_pos_t`** = read from `robot.des_c_pos` — the IK controller's stored desired
position, which was SET by the previous FM step: `des_c_pos_t = des_c_pos_{t-1} + action_{t-1}`.
This is the FM's own previous output. Not a measurement. Not independent.

The paper bundles both into "Get current state s_t" and never distinguishes them.

---

### Why this is absurd: the conditioning is half real, half hallucinated

At every planning step, the FM receives `s_t = [des_c_pos_t | c_pos_t]` and conditions
on it via inpainting. But the two halves are epistemically different:

| Component | What it represents | Source at eval | Self-referential? |
|---|---|---|---|
| `c_pos_t` (dim 6–8) | Where the arm ACTUALLY IS | `env.step()` return | **No** — ground truth |
| `des_c_pos_t` (dim 3–5) | Where the FM told the arm to go | FM's own output at t-1 | **Yes** — FM reading its own past |

The FM is conditioning on a mix of **one real signal** and **one echo of itself**. The paper
calls this "Get current state" as if both are sensor readings.

In training, `des_c_pos` was recorded from the human teleoperator's commands — an
independent signal. At eval, `des_c_pos` is generated by the FM. The paper uses the
same conditioning formula for both without acknowledging the distributional shift.

---

### Does the paper say which it WANTS to use?

No. Not even a hint. The paper never mentions:
- Any concern about `des_c_pos` being self-referential at eval
- Any alternative formulation using only `c_pos` as conditioning
- Any comparison of "condition on des" vs "condition on real"
- Any analysis of what happens when `des_c_pos` drifts from `c_pos`

The only mention of tracking error is implicit in the Euler dynamics `w_t` term. The paper
defines `w_t` as "model mismatch accounting for the low-level controller and numerical
error" — it doesn't name the self-referential problem as a component of `w_t`.

---

### Does Equation (9) help? (The projection constraint)

The projection constraint (Eq. 9, verbatim):

> "Π_{Z_f}(τ) = argmin s.t. **s_{t'+1|t} = f(s_{t'|t}, a_{t'|t})**, ∀t'"

Where the dynamics `f` = Euler: `s_{t+1} = s_t + [a, a]^T · ts`.

This enforces the PLANNED trajectory's internal consistency. In the DPCC code the `deriv`
constraint is applied to `c_pos` (dim 6–8, the real position component). The `des_c_pos`
component (dim 3–5) is NOT explicitly constrained via `deriv` in DPCC aligning
(`eval_visual_aligning_dpcc.py:119–121`).

**Why `des_c_pos` doesn't need constraining** (another thing the paper doesn't explain):
The FM generates `des_c_pos` trajectories. The Euler relation `des_c_pos[t+1] =
des_c_pos[t] + action[t]` is automatically satisfied in whatever trajectory the FM
produces — because the FM defines `des_c_pos` via its own action predictions. Constraining
`des_c_pos` would be enforcing a tautology. The `deriv` constraint on `c_pos` enforces the
approximation that `c_pos` approximately tracks `des_c_pos` under the Euler model.

So:
- `des_c_pos` in `s_t`: used as conditioning INPUT (pinned at t=0 via inpainting)
- `c_pos` in `s_t`: used as conditioning INPUT (pinned at t=0) AND as constraint target
- The paper never explains this asymmetry

---

### The three-layer absurdity, stated plainly

**Layer 1 — Units lie:** The paper calls the action "desired Cartesian velocity" (m/s).
The code computes it as `des_c_pos[t+1] - des_c_pos[t]` (position delta, m). These are
numerically equal only because `dt=1`. The IK controller is position-controlled, not
velocity-controlled. The paper's physical description is wrong.

**Layer 2 — State observation is half-hallucinated:** `s_t = [des_c_pos | c_pos]`. The
paper says "Get current state s_t" (Algorithm, line ①). `c_pos` IS a sensor reading.
`des_c_pos` IS NOT — it is the FM's own integrated past output, stored in `robot.des_c_pos`
because that's the last value the FM commanded. The paper treats both as measurements.

**Layer 3 — The self-reference is never named:** At each step t, the FM is conditioned on
`des_c_pos_t` = the FM's accumulated output through t-1. The FM then outputs `action_t`,
which updates `des_c_pos_{t+1} = des_c_pos_t + action_t`. At t+1, the FM is conditioned
on `des_c_pos_{t+1}` which includes its own previous output. This is a closed loop where
the model reads its own write. The paper's Euler model, dynamics constraint, and algorithm
pseudocode do not acknowledge this loop anywhere.

---

### What the paper should have said (but didn't)

A careful paper would have noted:

> "At eval time, the `des_c_pos` component of `s_t` is not a sensor reading — it is
> the IK controller's stored target, which equals the FM's own previous output. Therefore,
> conditioning on `s_t = [des_c_pos | c_pos]` at each step means the model is partially
> conditioning on its own accumulated history. This self-referential structure is inherent
> to the position-controlled robot setup: the FM generates `des_c_pos` at each step, and
> `des_c_pos` feeds back as part of the next step's conditioning. The `c_pos` component
> remains a true sensor reading at all times."

No such text exists in the paper.

---

### Summary

| Question | Paper's answer | Reality |
|---|---|---|
| Which component is used as diffusion conditioning? | `c = (s_t, g)` where `s_t = [des_xy \| c_xy]` — **both** | Both, but `des_c_pos` is FM's own output, `c_pos` is real |
| Does the paper distinguish des vs real in conditioning? | **No** — "Get current state s_t" (one line) | They are epistemically different; paper doesn't say so |
| Does the paper prefer `des_c_pos` or `c_pos` as conditioning? | **Not stated** — no preference expressed | Code uses both; `c_pos` is also the constraint target |
| Does the paper acknowledge self-referential conditioning? | **No** — never mentioned | It IS the mechanism; causes crash-then-drift in our UAV |
| Does the paper's "velocity" action match the code? | **No** — paper says velocity, code uses Δdes_c_pos, `dt=1.0` | Position delta with dt=1 = velocity numerically only |

The paper's contribution is the projection mechanism. The observation design — including
the self-referential `des_c_pos`, the "velocity" label, and the absence of any discussion
of what `des_c_pos` is at eval time — is accepted from D3IL's data format without
examination. This gap propagates directly into our UAV: we inherited the design, the
self-reference, and the unexamined assumption that `p_des` at eval can be treated as
a state observation rather than as the FM's own accumulated output.

---

## References

**Our UAV codebase (`/workspaces/FM-PCC`)**

| File | Line | Role |
|---|---|---|
| `FM_v3_uav_test/eval_fm_uav.py` | 312 | obs assembly `[p_des \| p \| v]` |
| `FM_v3_uav_test/eval_fm_uav.py` | 315 | FM called with `{0: obs}` |
| `FM_v3_uav_test/eval_fm_uav.py` | 323 | first action extracted |
| `FM_v3_uav_test/eval_fm_uav.py` | 338 | `p_des += action` — self-referential integration |
| `FM_v3_uav_test/eval_fm_uav.py` | 309 | `for k in range(n_fm)` — no termination check |
| `FM_v3_uav_test/eval_fm_uav.py` | 157 | dynamics constraint binds `act` ↔ `p_des` (not `p`) |
| `FM_v3_uav_test/eval_fm_uav.py` | 130–132 | comment explaining why `p_des` not `p` is bound |
| `flow_matcher_v3_uav/models/helpers.py` | 157–164 | `apply_conditioning` — pins obs only at `t=0` |
| `flow_matcher_v3_uav/models/diffusion.py` | 185, 267 | `apply_conditioning` called each denoising step |
| `flow_matcher_v3_uav/models/diffusion.py` | 190–288 | `p_sample_loop` — no termination awareness |
| `flow_matcher_v3_uav/sampling/projection.py` | 337–401 | `DynamicConstraints.build_matrices` — Euler equation |
| `flow_matcher_v3_uav/sampling/projection.py` | 394–398 | `skip_initial_state` — pins `p_des[0]` from obs |
| `uav_expert_data_collect/dataset_writer.py` | 31 | `DATASET_HZ=33`, `NOISE_SIGMA=0.02` — training distribution |

**Original DPCC (`/workspaces/FM-PCC`)**

| File | Line | Role |
|---|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 7–9 | trajectory layout `[act \| des_c_pos \| c_pos]` |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 119–121 | constraint binds `c_pos` (dim 6, real) ← `act` |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 1441 | obs built as `[des_c_pos \| c_pos]` — both present |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 1657 | `mental_robot_pos += next_action_np` — self-referential accumulation |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 738 | comment: `des_c_pos ≈ c_pos under PD control` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | 1 | **FM-PCC's own** visual avoiding adaptation — imports `fm_visual_avoiding.*`, NOT original DPCC paper |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | 433–437 | FM-PCC avoiding: `next_pos_des = action + obs[:2]` → `p_des` pattern (our adaptation, not paper) |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | 447 | FM-PCC avoiding: `if terminated: break` — this file has it; original DPCC aligning does not |

**Janner original diffuser (`/workspaces/diffuser`) — verified this session**

| File | Line | Role |
|---|---|---|
| `scripts/plan_guided.py` | (rollout loop) | `conditions={0:observation}` — real env state, never accumulated |
| `scripts/plan_guided.py` | (rollout loop) | `env.step(action)` — action goes directly to env, no `p_des` |
| `scripts/plan_guided.py` | (rollout loop) | `observation = next_observation` — obs always from env |
| `scripts/plan_guided.py` | (rollout loop) | `if terminal: break` — env signals termination |
| `diffuser/models/helpers.py` | 142–145 | `apply_conditioning` — identical pattern, but `val` is always real env state |
| `diffuser/datasets/d4rl.py` | 65–66 | d4rl dataset: only `observations` + `actions` — no des/actual split |

**Origin of `[des_c_pos | c_pos]` format — D3IL hardware layer**

| File | Line | Role |
|---|---|---|
| `d3il/environments/d3il/d3il_sim/controllers/IKControllers.py` | 75, 309 | IK controller writes `robot.des_c_pos = self.desired_c_pos` |
| `d3il/environments/d3il/gamepad_control/logger/logger.py` | 71, 74 | logger records both `c_pos` (actual) and `des_c_pos` (commanded) |
| `diffuser_visual_aligning/datasets/sequence.py` | 76–83 | loads both → `obs=[des_c_pos\|c_pos]`, `act=Δdes_c_pos`, `traj=[act\|obs]` (9D) |
| `diffuser_visual_aligning/datasets/sequence.py` | 26–28 | comment explains why both needed: constraint on `c_pos` not `des_c_pos` |
| `diffuser_visual_avoiding/datasets/sequence.py` | 81–88 | same pattern, 2D x,y only for avoiding |

**D3IL benchmark and DPCC architecture graft (§10 / §11)**

| File | Line | Role |
|---|---|---|
| `d3il/README.md` | — | D3IL = ICLR 2024 KIT benchmark, 11 IL methods, 7 tasks — designed for method-agnostic IL |
| `d3il/agents/models/diffusion/diffusion_models.py` | — | `DiffusionMLPNetwork` — D3IL's own MLP denoiser; action-only, no horizon, no Janner architecture |
| `d3il/agents/models/diffusion/diffusion_policy.py` | 18 | D3IL diffusion adapted from `twitter/diffusion-rl`, NOT Janner; samples single actions |
| `d3il/VENDORED_FROM.md` | — | Vendored from `/workspaces/d3il` commit `1d9c718` on 2026-04-08 |
| `diffuser_visual_aligning/models/diffusion.py` | 15 | `GaussianDiffusion` — Janner's DDPM-trajectory architecture, re-implemented for DPCC |
| `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` | 6 | `VisualGaussianDiffusion(GaussianDiffusion)` — Janner base + visual goal conditioning |
| `diffuser_visual_aligning/models/unet1d_temporal_cond.py` | 84 | `UNet1DTemporalCondModel` — temporal 1D UNet (≈ Janner's TemporalUnet) |
| `diffuser_visual_aligning/models/helpers.py` | — | `apply_conditioning` — identical to Janner's `helpers.py:142`, pins obs at t=0 |

**D3IL native dataset loaders — H2H math source (§10)**

| File | Line | Role |
|---|---|---|
| `d3il/environments/dataset/avoiding_dataset.py` | 55–64 | D3IL DDPM avoiding: obs=4D `[des_xy\|c_xy]`, action=2D `Δdes_xy`, window=1 — NO trajectory |
| `d3il/environments/dataset/aligning_dataset.py` | 62–84 | D3IL DDPM aligning: obs=20D (robot+boxes), action=3D `Δdes_c_pos`, window=1 — NO trajectory |
| `d3il/environments/dataset/aligning_dataset.py` | 245–246 | Visual aligning img variant: obs=`des_c_pos` only (camera carries box info), action=3D |
| `d3il/agents/ddpm_agent.py` | 161 | `loss(action, state)` — action is diffusion variable; state is just MLP conditioning |
| `d3il/agents/ddpm_agent.py` | 214–274 | `predict(state)` → one action per call; no horizon, no `apply_conditioning` |

**Trajectory tensor and FiLM comparison (§11)**

| File | Line | Role |
|---|---|---|
| `diffuser/diffuser/models/temporal.py` | 49–146 | `TemporalUnet.__init__`: `cond_dim` accepted but unused; `forward(x,cond,time)`: `cond` ignored — NO FiLM |
| `flow_matcher_v3_uav/models/unet1d_temporal_cond.py` | 87–237 | `Flow_matcher_U_Net_v2`: same as Janner — `cond` received in `forward` but never used — NO FiLM |
| `diffuser_visual_aligning/models/unet1d_temporal_cond.py` | 116–132 | `UNet1DTemporalCondModel`: `cond_mlp` when `use_cond_projection=True` — FiLM IS PRESENT |
| `diffuser_visual_aligning/models/unet1d_temporal_cond.py` | 220–228 | FiLM forward: pool visual emb → `cond_mlp` → `t = cat([time_emb, cond_emb])` modulates all blocks |
| `flow_matcher_v3_uav/models/unet1d_temporal_cond.py` | 84, 174 | `Flow_matcher_U_Net_v2` class and forward — no `cond_mlp`, no FiLM path |
| `eval_fm_uav.py` | 446 | `trajectory_dim = 12` = `act(3) + obs(9)` (12D, not 9D; we added velocity) |
| `flow_matcher_v3_uav/models/helpers.py` | 157–164 | `apply_conditioning` — only conditioning mechanism for our UAV and Janner (NO FiLM path) |

**DPCC paper (§13)**

| Source | Content |
|---|---|
| arXiv:2412.09342 §6.1 | State definition: `s_t ∈ ℝ^4 = [desired_pos(2D) \| current_pos(2D)]` |
| arXiv:2412.09342 §6.1 | Action: "desired Cartesian velocities, sent to a low-level controller" |
| arXiv:2412.09342 §6.1 | Dynamics: `s_{t+1} = s_t + [a_t^T, a_t^T]^T · ts + w_t` (Euler, same action for both components) |
| arXiv:2412.09342 §6.1 | "We do not assume knowledge of the dynamics of the low-level controller" |
| arXiv:2412.09342 §5 | Conditioning: `c = (s_t, g)` — no distinction between des/actual in conditioning |
| arXiv:2412.09342 (absent) | **No discussion of self-referential des_c_pos at eval time** — treated as sensor read |
| arXiv:2412.09342 (absent) | **No explicit justification for including des_c_pos in state** — stated without motivation |
| arXiv:2412.09342 §6.1 (WRONG) | Claims action = "desired Cartesian velocities" — code uses position delta; `dt=1.0` makes numbers equal but physical claim is wrong. IK controller is position-controlled. |
| arXiv:2412.09342 Algorithm | `"Get current state s_t and set c = (s_t, g)"` — one line; never says whether des_c_pos or c_pos, or that des_c_pos is FM's own output |
| arXiv:2412.09342 §6.1 | `"condition on the current state using inpainting"` — inpainting pins BOTH des_c_pos AND c_pos; self-referential nature of des_c_pos never mentioned |
| `diffuser_visual_aligning/datasets/sequence.py` | 83 | `actions = des_c_pos[t+1] - des_c_pos[t]` — position delta, NOT velocity |
| `config/visual_aligning_eval.yaml` | 24 | `dt=1.0 IS CORRECT: actions are position deltas, not [velocities]` |

**Internal analysis**

| File | Section | Role |
|---|---|---|
| `data_example_anlysis/ANALYSIS.md` | §3, §4 | proj_cost spike sequence and crash phase timeline |
