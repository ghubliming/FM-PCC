# What HardFlow's "fit a linear dynamics model" actually means

**Date:** 2026-07-22
**Question answered:** the HardFlow README says *"A linear dynamics model fitted from the training data is
used as a physical-fidelity constraint at inference time. To fit it, run `python run/fit_dynamics.py`."*
— what is this, and how does it relate to the pretrained flow checkpoint?
**Follow-up questions (added 2026-07-22):** is there an equivalent in DPCC? (§5a — yes, it's the *same
equation* with asserted coefficients). Is it cheating? (§6 — no, and it's less hand-fed than ours; but see
§6b for a real protocol asymmetry). Is it eval-only, nothing to do with training? (§7 — correct for the flow
model, with one caveat). Does eval auto-run it? (§9 — **no, manual, always**; plus why it's a separate
script and a misleading log message to watch for). **Does OUR sbatch / HF replica / Gen13 HF_iMF run it?
(§10 — yes, automatically, all six HF eval jobs; but the guard is a cache check, not a validity check.)**
**→ If §0–§7 didn't land, read [§8](#8-the-concrete-version--follow-the-actual-numbers-from-fm-output-to-mujoco)
first.** It traces the literal 96 numbers from the flow model to `env.step()` and shows the exact failure the
constraint prevents, plus the one line where DPCC and HardFlow actually differ on this task.
**Code source:** `/workspaces/aux_repo/HardFlow` (the `d3il` / `avoiding-v0` branch)
**Companion:** [`MAP_Algorithm1_to_AvoidingCode.md`](MAP_Algorithm1_to_AvoidingCode.md)

---

## 0. The one-paragraph answer

**It has nothing to do with the flow checkpoint.** It is not a second policy, not a fine-tune, not an
alternative to `train.sh`. It is a **4×4 matrix, a 4×2 matrix and a 4-vector** — 26 numbers — obtained by
ordinary least squares on the *same* demonstration dataset the flow model was trained on. It answers one
question: *"given state $s_t$ and action $a_t$, where does the robot end up?"*

$$s_{t+1} = A s_t + B a_t + c$$

That fit takes seconds (scikit-learn `LinearRegression`), is completely separate from the generative model,
and is saved to `logs/avoiding-v0/dynamics/linear_model.npz`. At **inference** time it is fed to the CasADi
optimizer as a **hard equality constraint** so that the trajectory HardFlow produces is not just
obstacle-free but *physically executable* — the state path and the action sequence agree with each other.

So the three artifacts are orthogonal:

| artifact | produced by | what it is | can you skip it? |
|---|---|---|---|
| `model_ema_20.pth` | `run_scripts/train.sh` **or** their Google Drive link | the flow-matching U-Net (the generative brain) | no — you need one or the other |
| `linear_model.npz` | `run/fit_dynamics.py` | 26 numbers: $A, B, c$ + the normalizer | only if you never run a `--dynamics_constraint` eval (see §4) |
| `trajectories.csv` | `run_scripts/eval_*.sh` | the actual results | — |

The pretrained checkpoint they give you is a substitute for **row 1 only**. It does **not** bundle row 2 —
you must run `fit_dynamics.py` yourself even when using their downloaded checkpoint.

---

## 1. Why this is needed at all (the "physical fidelity" problem)

The flow model generates a whole trajectory in **one shot**, as a single flat tensor per timestep:

```
[ a_0 | a_1, s_1 | a_2, s_2 | ... | a_H, s_H ]     transition_dim = action_dim + state_dim = 2 + 4
```

Actions and states are generated **jointly**, as correlated samples from a learned distribution. Nothing in
the sampler *enforces* that applying $a_t$ from $s_t$ actually lands you at the $s_{t+1}$ it also generated.
In-distribution the model has learned the correlation well enough, so it roughly holds. But HardFlow's whole
point is to **push the sample off-distribution** — the constraint projection shoves the state path sideways
to clear an obstacle. Once you do that, you can easily get a trajectory whose *states* dodge the obstacle
beautifully while its *actions* would never produce those states if you executed them on the real robot.
The plan looks safe on paper and collides in simulation.

The dynamics constraint closes that loop: it forces the optimizer to move actions and states **together**
along the manifold of physically realizable trajectories, instead of editing the state path in isolation.

This is exactly the FM-PCC "physical brakes" idea, with the dynamics half made explicit.

---

## 2. What `fit_dynamics.py` actually does

Anchors are `/workspaces/aux_repo/HardFlow/run/fit_dynamics.py`.

1. **Collect transitions** (`_collect_data`, `fit_dynamics.py:46`) — loads the same `SequenceDataset`
   (`LimitsNormalizer`, up to 500 episodes) and flattens every episode into single-step pairs:
   `fit_dynamics.py:68` walks `t = 0 .. path_length-2` and emits `(obs[t], act[t]) → obs[t+1]`.
   Note this is a **raw transition dump**, not sequence windows — `horizon` never enters the regression.
2. **Fit** (`fit_model`, `fit_dynamics.py:86`) — one independent `LinearRegression` per output dimension
   (4 of them), on the concatenated feature vector `[s_t ; a_t]` (dim 6). The coefficient block for the
   state part becomes row $i$ of $A$, the action part row $i$ of $B$, and the intercept element $i$ of $c$.
   No gradient descent, no GPU — it's a closed-form pseudo-inverse.
3. **Analyze** (`analyze_model`) — reports per-dimension $R^2$ / MSE and, usefully, the **spectral radius**
   of $A$; it warns if $\rho(A) \ge 1$ (the fitted linear system would be unstable).
4. **Verify against the real simulator** (`verify_with_env`, `fit_dynamics.py:155`) — the honest part:
   for 1000 sampled transitions it un-normalizes the state, does `env.set_state(...)`, takes the *real*
   MuJoCo step with the real action, and compares against $A s + B a + c$. This is the number that tells
   you whether the linear approximation is actually good enough to constrain on.
5. **Save** (`save_model`, `fit_dynamics.py:336`) → `logs/<env>/dynamics/linear_model.npz`, containing
   `A`, `B`, `c`, `metadata`, `metrics`, the **`normalizer` object**, and `verification_metrics`.
   Plus three PNGs (all trajectories, prediction scatter, verification scatter).

### Why a *linear* model is defensible here

The `avoiding-v0` observation is 4-dimensional and, per the env's own comment
(`d3il/.../envs/avoiding.py:379`), is literally

```
obs = [desired_x, desired_y, actual_x, actual_y]
```

i.e. a Cartesian setpoint plus the end-effector's Cartesian position, with the underlying arm running a
position controller and the z-axis pinned. That is close to a first-order tracking system, which a linear
map fits well. **This is a property of this task, not a general result** — do not assume the same trick
transfers to a task with real rigid-body coupling (a UAV, contact dynamics).

Everything above happens in **normalized space** (`LimitsNormalizer`), which is also why the normalizer is
pickled into the npz: the constraint at inference time operates on normalized decision variables.

---

## 3. Where the fitted model is consumed at inference

Two entirely different mechanisms exist in the code:

### 3a. Hard equality constraint — CasADi (the one that ships enabled)

`hardflow/models_flow/flow_policy.py:350` `_apply_dynamics_constraints`, plus the parallel
constraint-assembly path at `flow_policy.py:619`. For each $i$ in `range(horizon - 2)` it slices $s_i$,
$a_i$, $s_{i+1}$ out of the flat decision vector and adds

```python
opti.subject_to(A @ s_i + B @ a_i + c == s_{i+1})
```

plus a separate term tying the **measured current state** $s_0$ to $s_1$ through $a_0$ — so the plan is
anchored to where the robot really is, not just internally consistent. `eval_projection_relaxed.sh` uses
the `relaxation` branch, which replaces the equality with a symmetric $\pm\varepsilon$ box.

`flow_policy.py:354` **asserts** the model is loaded, so this path fails loudly if the npz is missing.

Two extras ride along with the same flag (`flow_policy.py:651`): when `dynamics_constraint` is on, the
optimizer also gets **action box bounds** $-1 \le a_i \le 1$ in normalized space. Toggling the flag
therefore changes *two* things, which matters if you are A/B-ing it.

### 3b. Soft penalty — PyTorch (present but disabled in every shipped script)

`run/eval.py:170` `_compute_dynamics_constraints` computes the same residual in torch and adds its squared
norm to `constraint_penalty` (`eval.py:118`) for the guidance-style baselines. Note it loops
`range(self.horizon - 3)` (`eval.py:188`) — **one fewer** than the CasADi path's `horizon - 2`, and with no
$s_0$ anchor term. The two implementations do not constrain the same set of transitions.

### Which scripts actually enable it

| script | flag |
|---|---|
| `eval_projection.sh` | `--dynamics_constraint` ✅ |
| `eval_hardflow.sh` | `--dynamics_constraint` ✅ |
| `eval_hardflow_new.sh` | `--dynamics_constraint` ✅ |
| `eval_projection_relaxed.sh` | `--no-dynamics_constraint` ❌ |
| `eval_oc_flow.sh` | `--no-dynamics_constraint` ❌ |
| `eval_gradient_guidance.sh` | `--no-dynamics_constraint` ❌ |
| `eval_original.sh` | not passed → default `False` (`hardflow/config/flow_matching.py:77`) ❌ |

So as shipped, **only the three CasADi-projection variants use the dynamics model at all**, and the soft
penalty path in `eval.py` is dead code in every provided configuration. Despite the `_relaxed` script's
name, its relaxation applies to the *obstacle* constraints — it has the dynamics constraint fully off.

---

## 4. Practical consequences for us

- **You must run `fit_dynamics.py` before `eval_projection.sh` / `eval_hardflow.sh` / `eval_hardflow_new.sh`,**
  even with their downloaded checkpoint. The other four scripts don't need it.
- ⚠️ **Silent-degradation trap.** `run/eval.py:517` loads the npz only `if os.path.exists(...) and
  cfg.dynamics_constraint`, and on a miss just prints *"No fitted dynamics found, proceeding without
  dynamics model"* (`eval.py:529`). For the CasADi path the downstream assert catches it — but any
  torch-penalty configuration would run happily with the dynamics term **silently contributing zero**, and
  the results would look like a valid constrained run. If we port this path into FM-PCC, make the missing
  file fatal. (Same class of bug as the silent-overwrite issue fixed in Gen13 U9.)
- **Not a local job.** Fitting itself is trivial CPU work, but `verify_with_env` starts MuJoCo (`env.start()`),
  so `fit_dynamics.py` is a **cluster job** in this project — it can't run in the AI-coding container.
- **Check the printed $R^2$ / RMSE before trusting any constrained result.** If the environment-verification
  $R^2$ is poor, the "hard" equality constraint is enforcing a *wrong* model exactly, which is worse than
  not constraining: the optimizer will faithfully satisfy a physics that isn't the simulator's.
- **`np.load(..., allow_pickle=True)`** is required (`eval.py:520`) because the normalizer is a pickled
  Python object — so the npz is **not portable across refactors** of the normalizer class. Regenerate it
  rather than copying it between branches.

---

## 5. Contrast with DPCC / FM-PCC

DPCC solves the same problem the **opposite** way. In
`/workspaces/aux_repo/dpcc/diffuser/utils/constraints_helpers.py:34` `formulate_dynamics_constraints`,
the dynamics for `avoiding` are **hand-specified analytic finite-difference relations**:

```python
('deriv', [x, vx]), ('deriv', [y, vy]), ('deriv', [x_des, vx]), ('deriv', [y_des, vy])
```

i.e. "position difference equals velocity", written by hand per environment, with **nothing fitted from data**.

### 5a. Yes — and it's the *same equation*, with the coefficients hard-coded instead of fitted

This is the key realisation, and it's stronger than "DPCC has something analogous."

`DynamicConstraints.build_matrices` (`dpcc/diffuser/sampling/projection.py:348-400`) expands each
`('deriv', [x_idx, dx_idx])` pair into a **linear equality row** on the flat trajectory vector:

$$x[t+1] = x[t] + \Delta t \cdot \dot{x}[t] \qquad \text{(explicit Euler)}$$

and for `avoiding`, the `dx_idx` entries (`vx`, `vy`) are **action** dimensions. So DPCC's constraint is
literally $s_{t+1} = s_t + \Delta t \cdot a_t$ — which is exactly HardFlow's

$$s_{t+1} = A s_t + B a_t + c \quad\text{with}\quad A = I,\; B = \Delta t \cdot I,\; c = 0.$$

**DPCC's dynamics constraint is the special case of HardFlow's where you assert the coefficients instead of
measuring them.** Our own `config/visual_aligning_eval.yaml:24-25` says this out loud for the aligning task:
*"The Euler dynamics constraint is `c_pos[t+1] = c_pos[t] + 1.0 * act[t]`"* (with `dt=1.0` because the actions
are position deltas, not velocities).

Both stacks even solve the surrounding engineering the same way: both fold the normalizer's min/max into the
constraint coefficients so the equality holds in **normalized** space (`projection.py:365-380` vs HardFlow's
pickled normalizer), and both **pin the first state to the measured $s_0$** rather than letting the plan
float (`skip_initial_state` at `projection.py:104-112` and `:175-182`, vs HardFlow's extra $s_0 \to s_1$ term).

Two real differences remain:

- **Density.** DPCC constrains *hand-picked coordinate pairs* — 4 sparse rows per timestep for `avoiding`,
  each touching 3 entries. HardFlow constrains **all 4 state dims with dense rows** over the full $[s_t; a_t]$.
  HardFlow's is therefore strictly **more binding**: it also pins down cross-coupling between dimensions that
  DPCC leaves completely free.
- **Where the numbers come from** — asserted vs. regressed, which is the whole of §6.

| | DPCC / FM-PCC | HardFlow |
|---|---|---|
| equation | $s_{t+1} = s_t + \Delta t\, a_t$ | $s_{t+1} = A s_t + B a_t + c$ |
| coefficients | asserted (`1, dt, -1`) | fitted by least squares |
| structure | sparse, hand-picked coordinate pairs | dense, all state dims |
| per-env work | write new `deriv` pairs by hand | rerun `fit_dynamics.py` |
| tunable knob | `dt` (we sweep it: `dt0p25…dt4p0`) | nothing — the data decides |
| failure mode | wrong if you mis-derive the relation | wrong if the system isn't linear |
| verified against sim? | never checked in code | yes — `verify_with_env` |

**Takeaway for FM-PCC:** we already run this constraint family — `'dynamics'` appears in
`config/projection_eval.yaml`, `config/visual_avoiding_eval.yaml` and `config/uav_projection.yaml`, and we
already ablate it (`model_free` = dynamics OFF) and sweep its one knob (`dt`). What HardFlow adds is not the
constraint; it's (a) dense instead of sparse coupling and (b) **a measured number for how wrong the
constraint model is**. Note that our `dt` sweep is a crude one-parameter search for what `fit_dynamics.py`
solves in closed form — if a swept `dt` beats `dt=1.0` on a task, that is evidence the asserted coefficients
are wrong and a fit would do better. If we adopt any of this, adopt `verify_with_env` first; the regression
is the easy half, and we currently have **no** measurement of our own constraint's fidelity.

---

## 6. "Is it cheating?"

Short answer: **no, not in the information sense — and it is *less* hand-fed than what we already do.**
But there is one real fairness problem, and it isn't the one you'd expect.

### 6a. Why it is not cheating

1. **No extra data.** The regression reads the *same demonstration dataset* the flow model trains on
   (`SequenceDataset`, same normalizer, `fit_dynamics.py:46`). No test set, no held-out episodes, no
   simulator rollouts enter the fit. There is no information in $A, B, c$ that the policy wasn't already
   trained on.
2. **It is strictly less domain knowledge than DPCC uses.** Per §5a, DPCC/FM-PCC *tells* the solver the exact
   integrator. HardFlow has to **discover** it from data and can only get it approximately right. If
   hand-coding the true dynamics is legitimate — and it is, it's standard MPC practice — then regressing an
   approximation of it is a strictly weaker assumption. If anyone is "cheating" by injecting privileged
   physics, it's the hand-coded side.
3. **It's a constraint, not an objective.** It carries no information about the goal, the obstacles, or the
   reward. It only enforces internal self-consistency between the states and actions the model itself
   generated. It cannot steer the trajectory toward success on its own.
4. **It cannot manufacture real feasibility.** Satisfying a *fitted linear approximation* exactly is not the
   same as being executable on MuJoCo. The constraint is weaker than its "hard" framing suggests — this is
   the opposite of an unfair advantage.
5. **It's ablatable, and it gets ablated.** We already run `model_free` (dynamics OFF) as a standard variant.

### 6b. The one thing that *is* a fairness problem

Not information leakage — **an asymmetric baseline protocol**. From §3's table:

- `--dynamics_constraint` is **ON** for `projection`, `hardflow`, `hardflow_new`.
- It is **OFF** for `oc_flow`, `gradient_guidance`, `projection_relaxed`, `original`.

Meanwhile `run/eval.py:170` contains a **working soft-penalty implementation of the same constraint** aimed
squarely at the guidance-style baselines — it is simply switched off in every shipped script. So if the
dynamics constraint improves results, part of the reported gap between HardFlow and its baselines is
attributable to a constraint the baselines were run without, not to the HardFlow algorithm.

This is a configuration asymmetry visible in the run scripts; it is **not** proof of anything about the
paper, whose protocol may well justify it (the guidance baselines may be argued to be structurally unable to
enforce equalities). **Action item before we cite or reproduce their numbers: check the paper's experimental
protocol section against this table.** Since `eval.py` already implements the penalty, the cheap decisive
experiment is to re-run `gradient_guidance` with the flag flipped on and see how much of the gap survives.

Two smaller hazards worth recording:

- **The flag is not a clean single-factor toggle.** Turning on `dynamics_constraint` *also* adds action box
  bounds $-1 \le a \le 1$ (`flow_policy.py:651`). Any A/B on that flag is measuring two changes at once.
- **`verify_with_env` does query the real simulator** — 1000 × `set_state` + `step` (`fit_dynamics.py:155`).
  In the shipped code those numbers are only *printed and stored*, never fed back into the fit, so there is
  no leak. But they are exactly the kind of number one is tempted to tune against (episode count, relaxation
  $\varepsilon$, which constraints to enable). If we port this, keep the verification **report-only** and
  never let it influence a hyperparameter, or it quietly becomes tuning on the eval environment.

---

## 7. Train vs. eval — where this sits in the pipeline

**Your reading is correct for the generative model: the flow model never sees this, at any point.**
Verified — `run/train.py` and `run_scripts/train.sh` contain **zero** occurrences of the string `dynamics`.
`linear_model.npz` is read only by `run/eval.py:517`. You can train the flow model, or download their
checkpoint, in complete ignorance of the dynamics model, and nothing changes.

One correction to the phrasing though: it is **not** "nothing about training" — it is a *third pipeline
stage* that is itself a (tiny) fitting procedure run on the **training split**:

```
   stage 1: train.sh          → model_ema_20.pth      (GPU, hours)      ← trains the flow model
   stage 2: fit_dynamics.py   → linear_model.npz      (CPU, seconds)    ← fits A,B,c on TRAIN data
   stage 3: eval_*.sh         → trajectories.csv                        ← consumes both
```

The distinctions that matter:

| claim | verdict |
|---|---|
| "it affects how the flow model is trained" | ❌ false — zero coupling, verified |
| "it is used only at inference" | ✅ true — consumed only by `run/eval.py` |
| "it involves no fitting at all" | ❌ false — it *is* a fit, just a 26-parameter one |
| "it is fit on evaluation data" | ❌ false — training demonstrations only |
| "you can skip it if you use the pretrained checkpoint" | ❌ false — it's a separate artifact (§0) |

Practical consequence: stage 2 is a **one-time, per-environment** artifact. Rerun it when the dataset, the
normalizer, or the env changes — not per seed and not per eval. And because it depends on the *normalizer*,
it is coupled to the dataset config, not to the checkpoint: swapping in a different flow checkpoint trained
on the same data does **not** require refitting, but changing `LimitsNormalizer` does.

---

## 8. THE CONCRETE VERSION — follow the actual numbers from FM output to MuJoCo

Ignore everything above for a moment. Here is the literal data path in `avoiding-v0`.

### 8a. What the flow model emits

Every shipped script uses **`horizon=16`, `replan_steps=8`** (all seven of `run_scripts/eval_*.sh`;
the `horizon: int = 8` in `hardflow/config/flow_matching.py:47` is a default that is always overridden).

One `policy(...)` call produces **one tensor of 96 numbers**: 16 timesteps × `transition_dim` 6.

```
 t=0 :  [ a_x  a_y | des_x des_y  act_x act_y ]     ← 2 action dims + 4 state dims
 t=1 :  [ a_x  a_y | des_x des_y  act_x act_y ]
  ...                                                   16 rows
 t=15:  [ a_x  a_y | des_x des_y  act_x act_y ]
        └── 32 numbers ──┘└──────── 64 numbers ────────┘
```

The 4 state dims are, per the env's own comment (`avoiding.py:379`):
`obs = [desired_x, desired_y, actual_x, actual_y]` — the **commanded setpoint** and the **real end-effector
position**. They are *not* the same thing, and that gap is the entire story (§8d).

### 8b. What MuJoCo actually receives — and what gets thrown away

`run/eval.py:396-408`:

```python
planned_actions = samples.actions[0]     # ← the 32 action numbers ONLY
...
action = planned_actions[action_index]   # one 2-vector
observation, reward, terminated, info = env.step(action)
```

> ### 🔑 **MuJoCo never sees the 64 state numbers. They are discarded.**
> Only `samples.actions` is indexed. `samples.observations` is used for plotting and logging and nothing
> else. And of the 16 planned actions, only the **first 8** are ever executed — then `action_index >=
> replan_steps` triggers a fresh plan from the new real observation.

So per plan: **96 numbers generated → 16 executed** (8 steps × 2 dims). Everything else is scaffolding.

And `env.step` (`avoiding.py:400-407`) is:

```python
action = np.clip(action, low, high)
next_desired_pos = self.desired_pos + action    # the action is a POSITION DELTA on the setpoint
self.desired_pos = next_desired_pos
full_action = concat(next_desired_pos, fixed_z, [0,1,0,0])   # → Cartesian position controller
# ... then 35 MuJoCo substeps (n_substeps=35) of the arm chasing that setpoint
```

### 8c. The failure this constraint exists to prevent

Now put those two facts together. The obstacle constraint is imposed on **state dims 2–3 only** —
`run/eval.py:221-222` (`x_pos = observations[:,:,2]`, `y_pos = observations[:,:,3]`) and
`flow_policy.py:589`. It never references an action.

So consider the projection step with **`dynamics_constraint = False`**:

```
   generated plan:   a_0..a_15  (executed)      s_0..s_15  (discarded)
   obstacle says:    "s_5 must be 0.03 m from the pillar"
   optimizer's job:  minimise ‖edit‖ subject to that
   cheapest edit:    move s_5. Actions are not in the constraint, so leave them alone.
   result:           s_5 now clears the pillar beautifully.  a_0..a_7 are BIT-IDENTICAL to before.
   what ships:       a_0..a_7  →  MuJoCo  →  robot drives straight into the pillar.
```

**The optimizer satisfied the constraint by editing the numbers that get thrown away.** The plan is
provably obstacle-free and the robot still crashes. The logged trajectory shows a violation while the
solver reports success.

With **`dynamics_constraint = True`**, the equality rows `A·s_i + B·a_i + c = s_{i+1}` glue every state to
the action before it. Now `s_5` cannot move unless `a_4` moves too, which drags `s_4`, which drags `a_3`…
The constraint propagates the obstacle avoidance **backwards into the action sequence** — the only part
MuJoCo will ever see.

> **That is the whole job of the fitted dynamics model: it is the wire connecting the numbers the
> optimizer edits to the numbers the robot executes.** Without it, constrained sampling on a
> jointly-generated (action, state) trajectory is editing a picture of the plan rather than the plan.

*(Nuance: with the flag off, actions and states are still* correlated *— the flow model learned them
jointly. The projection is what breaks that correlation, and nothing puts it back.)*

### 8d. DPCC vs HardFlow on this exact task — the one line that differs

Both impose the same *kind* of equality. Here is what each asserts about `avoiding`, in unnormalized terms
(`dpcc/diffuser/utils/constraints_helpers.py:47-53`, expanded via `projection.py:365-380`):

| constraint row | DPCC / FM-PCC (asserted) | HardFlow (fitted) | true dynamics |
|---|---|---|---|
| setpoint `des` | `des[t+1] = des[t] + dt·a[t]` | row of `A,B,c` | **exact** — it's literally `desired_pos += action` (`avoiding.py:403`) |
| end-effector `act` | `act[t+1] = act[t] + dt·a[t]` | row of `A,B,c` | **approximate** — the arm chases the setpoint through 35 MuJoCo substeps and lags behind it |

The setpoint row is trivially exact for both — it's a line of Python, not physics. **The end-effector row is
where they diverge, and it is the row that matters,** because the obstacle constraint is on `act_x/act_y`
(dims 2–3), not on the setpoint.

- **DPCC asserts `act[t+1] = act[t] + dt·a[t]`** — i.e. *the end-effector moves exactly by the commanded
  delta*. That is an assumption of **perfect, zero-lag tracking**. The real arm is a position-controlled
  manipulator settling over 35 substeps; it does not do this. DPCC's only recourse when the assumption is
  bad is the **`dt` fudge factor** — which is precisely why our own configs carry a `dt0p25/dt0p5/dt2p0/
  dt4p0` sweep. That sweep is a one-parameter manual search for the tracking gain.
- **HardFlow measures it.** The regression is free to return `act[t+1] = 0.85·act[t] + 0.15·des[t] + 0.6·a[t]`
  or whatever the data says — a first-order lag with cross-coupling to the setpoint. DPCC's sparse `deriv`
  pairs **cannot express that** at any `dt`: they only relate one position dim to one action dim, with a
  coefficient of exactly 1, and no `act↔des` coupling term exists in the formulation.

That is the concrete, checkable difference:

> **DPCC assumes the robot goes exactly where you tell it. HardFlow regresses how far behind it actually
> lags, including the setpoint-to-position coupling that DPCC's formulation has no slot for.**

**Testable prediction, if we ever run `fit_dynamics.py` on the cluster:** rows 0–1 of `A,B,c` (the setpoint)
should come out at $R^2 \approx 1.000$ with coefficients indistinguishable from `A=I, B=I, c=0` — because
that row *is* an assignment statement. Rows 2–3 (end-effector) should show $R^2 < 1$ and a diagonal `A`
term meaningfully below 1. **If rows 2–3 also come back at ~1.0 with `B≈I`, then DPCC's assumption was right
all along and this whole mechanism buys nothing on this task** — and our `dt=1.0` default is already
optimal. That single printout decides whether porting any of this to FM-PCC is worth the effort. It is a
few CPU-seconds of work, gated only on MuJoCo being available for the `verify_with_env` stage.

### 8e. ⚠️ Disambiguation — there are THREE different "8"s here, and only one of them is a horizon

*(Added after "why is 8 step??? i thought is H16 vs DPCC H8" — **your memory is correct**, HardFlow is H16
and DPCC is H8. Nothing in §8a–8d contradicts that. But the number `8` appears three times in this task
meaning three unrelated things, so pin them down:)*

| the "8" | what it is | value |
|---|---|---|
| `horizon` in `hardflow/config/flow_matching.py:47` | a **dead default**, overridden by every script | 8 → **never used** |
| `replan_steps` in `run_scripts/eval_*.sh:20` | how many planned actions get **executed** before replanning | 8 |
| DPCC `'horizon'` in `dpcc/config/avoiding-d3il.py:22,83` | DPCC's **actual planning horizon** | 8 ✅ |

**So: HardFlow plans H=16. DPCC plans H=8. Confirmed, both verified in the configs.** The `8` in §8b is
`replan_steps` — how much of the 16-step plan gets used — which is a *different knob* that happens to share
the number. HardFlow's effective planning horizon is 16 everywhere; the config default of 8 is a leftover
that no shipped script ever reaches.

### 8f. The bigger difference I found while checking this: **DPCC replans every single step**

`dpcc/scripts/eval.py:231` calls `policy(...)` inside the per-timestep loop with **no `replan_steps` gate at
all** — one plan, one action, throw the rest away, remeasure, replan:

```python
for _ in range(max_steps):
    action, samples = policy(conditions={0: obs}, ...)   # ← fires EVERY timestep
    next_pos_des = action + obs[:2]
    obs, rew, terminated, info = env.step(...)
```

Versus HardFlow (`run/eval.py:390-402`), which plans once and then runs **8 steps open-loop** before it
looks at the world again.

| | DPCC | HardFlow |
|---|---|---|
| planning horizon $H$ | **8** | **16** |
| actions executed per plan $n$ | **1** | **8** |
| `max_episode_length` | **200** (`dpcc/config/avoiding-d3il.py:68`) | **100** (`flow_matching.py:54`) |
| plans per episode | up to **200** | up to **13** |
| open-loop stretch | none — remeasures every step | **8 steps blind** |

**This changes the §8d verdict, and it's the real reason HardFlow needs a fitted dynamics model where DPCC
can get away with asserting one.** DPCC's "perfect tracking" assumption (`act[t+1] = act[t] + dt·a[t]`) is
wrong, but it is wrong for exactly **one step** before the true observation comes back in and resets the
error to zero. Replanning every step is a brutally effective error-correction mechanism — it papers over a
crude dynamics model almost completely.

HardFlow commits to 8 steps of its own predictions with no feedback. A per-step tracking error that DPCC
erases immediately gets **compounded eight times** before HardFlow re-measures. At that point the accuracy
of `A, B, c` stops being cosmetic: it is the only thing standing between the plan and drift.

Two consequences worth carrying forward:

- **Don't read "HardFlow uses a fitted model, DPCC doesn't" as HardFlow being more rigorous for its own
  sake.** It is a *forced* move. The fitted model buys back the fidelity that `replan_steps=8` gives away.
  The two designs are trading the same currency in opposite directions — compute (DPCC: ~100 solver calls
  per episode) against model accuracy (HardFlow: ~13 calls, but the model must hold for 8 steps).
- **This is a confound in any DPCC-vs-HardFlow number we might quote.** The two differ in planning horizon
  (8 vs 16), replan frequency (1 vs 8), *and* dynamics-model source simultaneously. Three factors, not one.
  For FM-PCC the honest comparison would fix `replan_steps` before attributing anything to the dynamics
  model — and note that our stack inherits DPCC's every-step replanning, which means **the fitted model
  would buy us proportionally less than it buys HardFlow.** That weakens the case for porting it, and it
  should be weighed against the §8d test before spending effort.

### 8g. "But they're BOTH receding horizon — how can they differ?"

**They are both receding horizon. You are right about that, and it is not in conflict with §8f.**
HardFlow's own code literally calls it that (`cfg.controller == "rh"  # receding horizon`,
`run/eval.py:389`). The confusion is that "receding horizon" is a **family**, not a single behaviour.

Receding horizon has *two* independent knobs:

$$\text{plan } H \text{ steps} \;\to\; \text{execute the first } n \le H \;\to\; \text{remeasure} \;\to\; \text{replan}$$

- $H$ = **planning horizon** — how far ahead you think.
- $n$ = **execution horizon** (a.k.a. replan interval / control stride) — how much of that thinking you
  actually use before throwing the rest away and starting over.

Every RH/MPC controller has both. Textbook MPC sets $n=1$, which is so standard that $n$ is often not even
mentioned — which is exactly why "receding horizon" sounds like it pins down the behaviour. It doesn't.
$n = 8$ is still receding horizon; the horizon still recedes, just in strides of 8 instead of 1.

**DPCC: $H=8, n=1$. HardFlow: $H=16, n=8$.** Both RH. Different points in the same family.

Verified line-by-line, both directions:

| | code | what it proves |
|---|---|---|
| DPCC | `dpcc/scripts/eval.py:203` `for _ in range(max_episode_length):` … `:231` `action, samples = policy(...)` — **unconditional**, no gate, no cached plan, no index | replans every timestep |
| DPCC | `dpcc/diffuser/sampling/policies.py:91` `action = actions[which_trajectory, 0]` — comment above it reads *"extract first action"* | uses **1** of its 8 planned actions; the other 7 are discarded |
| HardFlow | `run/eval.py:390` `if planned_actions is None or action_index >= cfg.replan_steps:` | replans **only** when the index reaches 8 |
| HardFlow | `run/eval.py:401-402` `action = planned_actions[action_index]; action_index += 1` | walks the *stored* plan on the other 7 timesteps — no policy call, no new observation |
| HardFlow | `replan_steps=8` in **all seven** `run_scripts/eval_*.sh` | not a one-off; it's the shipped protocol |

Trace HardFlow's index to see it plainly: `t=0` plan → exec `[0]`, `t=1` exec `[1]`, … `t=7` exec `[7]`,
`t=8` index hits 8 → **replan** → exec `[0]`. One fresh observation every 8 environment steps.

**Why would anyone set $n=8$?** Cost. HardFlow solves a constrained nonlinear program (IPOPT, via CasADi)
*inside every ODE integration step* — 20 `ode_t_steps` per plan. Replanning every timestep would multiply
that by 8. Setting $n=8$ is the standard MPC bargain: buy compute by spending model accuracy. That is
consistent with `computation_time` being recorded per *plan* (`run/eval.py:399`) rather than per step —
and it means any wall-clock comparison between the two stacks is comparing ~13 solves against up to 200.

**And this is precisely why §8f matters.** $n=1$ makes a crude dynamics model survivable, because the true
observation wipes the error every step. $n=8$ makes an accurate dynamics model necessary, because nothing
corrects the error for 8 steps. **HardFlow didn't fit a dynamics model because fitting is better in
principle — it fit one because $n=8$ left it no choice.** The two design decisions are the same decision.

---

## 9. Auto or manual? And why is it a separate script at all?

### 9a. **In UPSTREAM HardFlow: manual, always. Nothing there runs it for you.**

> ⚠️ **Scope.** This section describes `/workspaces/aux_repo/HardFlow` only. **Our FM-PCC sbatch stack
> DOES auto-run it** — every HF eval job, including the Gen13 HF_iMF ones, fits it behind a guard. See
> **§10**, which is the answer if you're asking "does *our* pipeline run this?"

Verified by grepping the entire upstream HardFlow repo for the string `fit_dynamics`:

```
README.md:86:python run/fit_dynamics.py
```

**That is the only hit outside the file itself.** No `run_scripts/*.sh` calls it. No Makefile. No import.
No hook in `run/eval.py`. It is a documentation line and nothing else.

`run/eval.py:516-529` **only ever loads, never fits**:

```python
dynamics_path = os.path.join("logs", cfg.env, "dynamics", "linear_model.npz")
if os.path.exists(dynamics_path) and cfg.dynamics_constraint:
    ...load the npz...
else:
    print("No fitted dynamics found, proceeding without dynamics model")
```

There is no `else: fit it now` branch. If the file isn't on disk, it is simply not used.

### 9b. What actually happens when you forget to run it

Three distinct outcomes, and only one of them is loud:

| situation | what happens | how bad |
|---|---|---|
| `--dynamics_constraint` **ON**, npz **missing** | `dynamics_model = None` → `flow_policy.py:354` `assert self.dynamics_model is not None` fires | ✅ **loud crash** — you cannot get a wrong result this way |
| `--dynamics_constraint` **OFF**, npz **present** | prints *"No fitted dynamics found"* **anyway**, runs unconstrained | ⚠️ **misleading log** — see 9c |
| torch soft-penalty path, npz missing | `_compute_dynamics_constraints` returns `zeros(batch, 0)` (`eval.py:175-176`) → penalty contributes **exactly zero** | 🔴 **silent** — dead in shipped scripts, but a live trap if we enable it (§6b) |

### 9c. 🪤 The log message lies

Read the condition again: `if os.path.exists(...) and cfg.dynamics_constraint`. The `else` branch prints
**"No fitted dynamics found"** — but that branch is *also* taken when the file exists perfectly and you
simply didn't pass the flag.

So `eval_original.sh`, `eval_oc_flow.sh`, `eval_gradient_guidance.sh` and `eval_projection_relaxed.sh`
(all `--no-dynamics_constraint`, §3) will print **"No fitted dynamics found"** on every single run, with a
valid `linear_model.npz` sitting right there on disk. Anyone debugging will conclude their fit failed and
go re-run `fit_dynamics.py` for nothing. **The message conflates "file missing" with "feature disabled".**

### 9d. Why it's a separate script — four reasons, and the fourth is the real one

1. **It's a per-*environment* artifact, not a per-*run* one.** One `linear_model.npz` serves all 7 eval
   scripts × every seed × every checkpoint. Refitting inside each eval would redo identical work dozens of
   times per experiment sweep.
2. **Different dependency footprint.** Fitting needs `sklearn` + a live MuJoCo env (for `verify_with_env`).
   Eval needs the flow model + CasADi/IPOPT. Keeping them apart means the eval process never has to stand
   up the fitting stack, and vice versa.
3. **Different lifecycle — it is coupled to the *dataset*, not the checkpoint.** Swap in a different flow
   checkpoint trained on the same data → **no refit needed**. Change the dataset or the normalizer → **must
   refit**. That dependency edge doesn't match the eval script's, so bolting it on would create a stale-cache
   bug the first time someone changed normalizers.
4. **🔑 It produces numbers a human is supposed to LOOK AT.** `fit_dynamics.py` prints per-dimension $R^2$,
   RMSE, the spectral radius of $A$ with a stability warning, and the real-simulator verification $R^2$ —
   plus three PNGs. **That output is the only evidence that the constraint you're about to enforce as a hard
   equality is actually true.** Auto-running it inside eval would bury the one number that tells you whether
   the whole mechanism is valid, and you'd be enforcing a possibly-garbage model exactly, in silence.
   Making it a deliberate manual step is a *feature*: it forces you to read the fit quality once per
   environment before trusting it.

### 9e. What this means for us on the cluster

- Treat it as an explicit **stage 2** in the sbatch pipeline (§7), run **once per environment**, not per
  job and not per seed. It's CPU-seconds — but `verify_with_env` starts MuJoCo, so it still needs a real
  cluster node with EGL, not the AI-coding container.
- If we port any of this into FM-PCC: **make the missing file fatal**, and **split the log message** into
  "dynamics constraint disabled" vs "dynamics constraint requested but `linear_model.npz` not found". The
  current conflation (9c) plus the silent-zero penalty (9b, row 3) is exactly the class of failure that
  produced the Gen13 U9 silent-overwrite bug — a run that looks successful and is quietly measuring
  nothing.
- Note we have **no equivalent artifact today** and don't need one: our coefficients are asserted in code
  (`formulate_dynamics_constraints`, §5a), so there is nothing to fit, nothing to cache, and nothing to
  forget to run. That's the upside of the hand-coded approach — and the reason we also have no measurement
  of whether our constraint is true.

---

## 10. Does *our* FM-PCC stack run it? — **YES, automatically, in every HF eval job**

Checked directly in this repo. This supersedes §9a for anything we run.

### 10a. We vendor HardFlow, and `fit_dynamics.py` is untouched

`Slurm_Codes/sbatch/hardflow/_hardflow_common.sh:32` sets `HARDFLOW_REPO="$REPO/HardFlow"` — we run the
**vendored copy at `FM-PCC/HardFlow/`**, not `/workspaces/aux_repo/HardFlow`, and it ships to the cluster by
`git pull`. `diff -rq` against upstream confirms **`run/fit_dynamics.py` and `run/eval.py` are byte-identical
to upstream**. Our additions are the iMF layer alongside them: `run/eval_imf.py`, `run/train_imf.py`,
`run/train_fm.py`, `hardflow/models_flow/imf/`, `run_scripts/*_imf.sh`, `run/make_fig11_*.py`.

So everything in §0–§9 about the mechanism applies verbatim to our runs.

### 10b. All six HF eval sbatch scripts fit it behind an `[ -f ]` guard

```bash
DYN="logs/avoiding-v0/dynamics/linear_model.npz"
if [ ! -f "$DYN" ]; then
    echo "[ HF-EVAL ] dynamics model missing -> python run/fit_dynamics.py"
    python run/fit_dynamics.py
else
    echo "[ HF-EVAL ] dynamics model present: $DYN"
fi
```

| sbatch script | line | style |
|---|---|---|
| `eval_hardflow.sh` | 30-35 | if/else, logs "present" |
| **`eval_imf_hardflow.sh`** | **25-31** | if/else, logs "present" — **this is the Gen13 HF_iMF path** |
| `eval_paired_n200_hardflow.sh` | 45-48 | if/else |
| `fig11_compare_hardflow.sh` | 35-36 | one-liner `\|\|` |
| `eval_matched_nfe_hardflow.sh` | 39-40 | one-liner `\|\|` |
| `eval_smoothness_diag_hardflow.sh` | 36-37 | one-liner `\|\|` |

**Gen13 HF_iMF specifically:** `imf_pipeline_hardflow.sh:95,111` chains
`train_imf_hardflow.sh → eval_imf_hardflow.sh` via `--dependency=afterok`, and the guard sits inside the
eval job (`eval_imf_hardflow.sh:25`) — so **yes, the Gen13 iMF runs fit it**, on first use, automatically.

The plumbing is already correct: `_hardflow_common.sh:129` does `cd "$HARDFLOW_REPO"` (so the relative
`logs/...` path resolves to `FM-PCC/HardFlow/logs/`), and `:101-105` export `MUJOCO_GL=egl` +
`MUJOCO_EGL_DEVICE_ID`, which is exactly what `verify_with_env` needs. The conda env is the
`hardflow_clone` (gym 0.20), not the live FMPCC env.

### 10c. 🪤 The guard is a **cache check, not a validity check**

This is the part to be careful about, and it's a live risk in our stack, not a hypothetical:

1. **`[ -f "$DYN" ]` tests existence only.** A stale, corrupt, or wrong-normalizer `linear_model.npz`
   passes the guard forever. Per §7, the npz is coupled to the **dataset + normalizer** — so if we ever
   change either, **every subsequent eval silently reuses a dynamics model fitted to the old data**, and
   nothing anywhere will say so.
2. **The fit's quality report is seen once and then never again.** `fit_dynamics.py` prints the per-dim
   $R^2$, the spectral radius, and the real-sim verification $R^2$ (§9d.4) — but only in the log of
   whichever eval job happened to run *first*, buried mid-job. Every run after that prints
   `dynamics model present` and moves on. **We are almost certainly enforcing a hard equality constraint
   whose fit quality nobody has ever read.** Worth grepping the oldest HF job log for
   `Overall R²` / `spectral radius` to see what it actually said — that number is the §8d test, and we may
   already have the answer sitting in a log.
3. **§9c's misleading message applies to our iMF eval too.** `HardFlow/run/eval_imf.py:368-381` copies the
   load-or-warn logic verbatim, including `print("No fitted dynamics found, proceeding without dynamics
   model")` on the `else`. So any `--no-dynamics_constraint` iMF run prints *"No fitted dynamics found"*
   right after the sbatch printed *"dynamics model present"* — two contradictory lines in the same log,
   both correct, about different things.
4. **Minor: the guard's stated rationale is stronger than the real failure mode.** The comment at
   `eval_hardflow.sh:28-29` says eval *"SILENTLY proceeds without it (degrades)"*. For the CasADi methods we
   actually run (`hardflow_new`, `hardflow`, `projection`) a missing model **crashes** instead —
   `flow_policy.py:354` asserts, or `:620` raises `TypeError` on `None["A"]`. The silent-degradation path is
   the torch penalty (§9b row 3), which those methods don't use. **The guard is still right and should
   stay** — it just prevents a crash rather than a silent wrong number.

### 10d. Suggested hardening (not applied — flagging only)

- Replace `[ -f "$DYN" ]` with a check that also compares the npz's stored `metadata`/normalizer against the
  current dataset config, or simply **delete the npz whenever the dataset changes** and note it in the run
  log.
- **Echo the fit quality on every run**, not just the first: load the npz in the guard's `else` branch and
  print the stored `metrics['overall_r2']` and `verification_metrics['overall_r2']` (both are already saved
  into the npz by `fit_dynamics.py:336-352`). Two lines of Python, and it puts the constraint's validity in
  every job log instead of one.
- Consider fitting it in an **explicit pipeline stage** (like `extract_dataset_job.sh`) rather than
  lazily inside whichever eval fires first — that makes the artifact's provenance visible and stops the
  quality report from hiding in an unrelated job's log.

---

## Footnotes

- `avoiding.py` defines `get_observation` **twice** in the same class (line 122 returning 2 dims, line 372
  returning 4). Python keeps the **later** one, so the effective observation is the 4-dim
  `[desired_x, desired_y, actual_x, actual_y]` and `verify_with_env`'s shape check works. The line-122
  definition is shadowed dead code — don't be misled when reading that file.
- `LinearDynamicsAnalyzer.__init__` defaults to `horizon=16` (`fit_dynamics.py:21`) while the eval config
  uses `horizon=8` (`hardflow/config/flow_matching.py:12`). This does **not** affect $A, B, c$ — the
  regression reads raw per-episode transitions and never touches the sequence windows — but it can change
  which short episodes survive dataset construction.
- `fit_dynamics.py` takes `--env` and nothing else; horizon and episode cap are hard-coded constructor
  defaults, not CLI flags.
