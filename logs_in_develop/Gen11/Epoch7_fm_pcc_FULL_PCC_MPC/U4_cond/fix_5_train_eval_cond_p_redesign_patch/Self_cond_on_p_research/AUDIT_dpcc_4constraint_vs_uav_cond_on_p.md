# AUDIT — DPCC 4-Constraint Logic vs UAV "cond-on-p" Redesign

**Audited file:** `Self_cond_on_p_research/research.md`
**Auditor stance:** verify every claim against the *actual* source, not the prose.
**Date:** 2026-06-30

> ⚠️ The research.md is written in an over-confident chatbot register
> ("I AM NOT LYING", "1000% YES", "EXACTLY!"). Tone is not evidence. This audit
> re-derives each claim from code + config line quotes.

---

## TL;DR Verdict

| research.md claim | Verdict | Note |
|---|---|---|
| Avoiding builds **4** `deriv` tuples `[4,0],[5,1],[2,0],[3,1]` | ✅ **TRUE** | exact match to helper + config |
| Index `0` = "Action X" | ✅ **TRUE (this codebase)** | col 0 is the action dim (keyed `vx`, but action = Δpos) |
| The 4 rows **couple p and p_des** (share the same action column) | ✅ **TRUE (structural)** | precise effect = *frozen offset*, not literal teleport |
| "Forces the drone to **teleport**" | ⚠️ **OVERSTATED** | rigorous statement: identical per-step increments → offset frozen at t=0; equals teleport *only if* p₀=p_des₀ |
| The 4-constraint design is a **flaw / bug** | ❌ **MISLEADING** | it is *correct* for the tight-tracking manipulator; only *wrong* for an inertial UAV |
| UAV eval **bypasses** `constraints_helpers.py`, hardcodes 3 constraints | ✅ **TRUE** | verified: `formulate_dynamics_constraints` never called in `eval_fm_uav.py` |
| UAV "**deleted / destroyed**" the old code | ❌ **FALSE** | nothing deleted — helper logic is fully intact, just not invoked |
| Matrix is `[1, dt, -1]`, `b=0` | ⚠️ **INCOMPLETE** | true only for the *un-normalized* branch; UAV runs the *normalized* branch (scaled entries, `b≠0`) |

**Bottom line:** The research's *mechanics* are mostly right; its *framing* ("bug",
"teleport", "deleted") is wrong and dangerous. **Dropping/"fixing" the original
4-constraint code would break the avoiding/pointmaze/antmaze pipelines and is NOT
warranted.** The UAV already does the safe thing — a separate, additive bypass.

---

## 1. The avoiding column layout (ground truth)

`config/projection_eval.yaml:16-25`
```yaml
dt: { 'avoiding': 1, }                                  # action is Δpos, so Euler dt=1
observation_indices: { 'avoiding': {'x_des':0,'y_des':1,'x':2,'y':3} }
action_indices:      { 'avoiding': {'vx':0,'vy':1} }    # the action dims
```
For `GaussianDiffusion` (states_actions), obs indices are shifted by `action_dim`
(`scripts/eval.py:110-111`):
```python
obs_indices_updated = {key: val + action_dim for key, val in obs_indices.items()}
act_obs_indices = {**act_indices, **obs_indices_updated}
```
With `action_dim = 2` → resolved layout, `transition_dim = 6`:

| col | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| name | `vx`=aₓ | `vy`=a_y | `x_des` | `y_des` | `x`(actual) | `y`(actual) |

**Note on "action vs velocity":** the config *keys* the action `vx/vy`, but the
inline comment (`:17`) states `a = [delta_x, delta_y] and not [vx, vy]`. So col 0
is the **action**, semantically a position increment. research.md calling it
"Action X" is therefore correct **for this codebase** (with the caveat that the
key name is misleading, not the research).

---

## 2. The 4 tuples — verified exactly

`flow_matcher_v3_uav/utils/constraints_helpers.py:47-53`
```python
if 'avoiding' in exp and action_dim > 0:
    dynamic_constraints = [
        ('deriv', np.array([act_obs_indices['x'],     act_obs_indices['vx']])),   # [4,0]
        ('deriv', np.array([act_obs_indices['y'],     act_obs_indices['vy']])),   # [5,1]
        ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),   # [2,0]
        ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),   # [3,1]
    ]
```
Resolves to `[4,0],[5,1],[2,0],[3,1]` — **identical to research.md Step 1.** ✅

These are passed to the projector via `scripts/eval.py:145`:
```python
if 'dynamics' in constraint_types:
    dynamics_constraints = utils.formulate_dynamics_constraints(exp, act_obs_indices, action_dim)
```

---

## 3. What the 4 constraints REALLY do (corrected math)

Each `deriv [x_idx, dx_idx]` row enforces (un-normalized branch,
`projection.py:389-392`):
$$ x^{(t+1)} = x^{(t)} + \Delta t \cdot a^{(t)} $$

For avoiding, **both** the actual position and the desired position read the
**same action column**:
$$ x^{(t+1)} = x^{(t)} + \Delta t\, a_x^{(t)}, \qquad x\_des^{(t+1)} = x\_des^{(t)} + \Delta t\, a_x^{(t)} $$

Subtract the two:
$$ \big(x - x\_des\big)^{(t+1)} = \big(x - x\_des\big)^{(t)} \quad\Rightarrow\quad x^{(t)} - x\_des^{(t)} = \text{const} $$

**This is the precise effect: identical per-step increments → the offset between
actual and desired is *frozen* at its initial value.** It is NOT "teleport to the
same point" in general. It becomes effective perfect-tracking (p ≡ p_des) **only
when the anchored initial states coincide** (p₀ = p_des₀), which `skip_initial_state`
pins via `projection.py:99-108`:
```python
for constraint in self.dynamic_constraints.constraint_list:
    if constraint[0] == 'deriv':
        x_idx = int(constraint[1][0])
        b[counter * self.horizon] = s_0[x_idx]     # anchor each dim's t=0 to current state
```
In D3IL `avoiding` the end-effector starts on its commanded point and tracks
tightly, so p₀≈p_des₀ and the coupling ≈ perfect tracking — **which is the intended
physics of that manipulator task, not a bug.** research.md itself admits this at
its line 283 ("designed to force Perfect Tracking"), contradicting its own "flaw"
framing elsewhere.

### Omission in research.md: the normalized branch
UAV passes a normalizer (`eval_fm_uav.py:237` `normalizer=ProjectorNormalizer(...)`),
so the **normalized** branch runs, NOT the plain `[1, dt, -1] / b=0` matrix the
research draws (`projection.py:383-387`):
```python
mat_append[i, i*T + x_idx]       = 1 * x_diff
mat_append[i, i*T + dx_idx]      = self.dt * dx_diff
mat_append[i, (i+1)*T + x_idx]   = -1 * x_diff
vec_append[i]                    = - dx_sum * self.dt      # b ≠ 0
```
The coupling argument survives (the structure is the same), but the research's
"`b=0`, entries are exactly 1/dt/−1" is only the un-normalized special case.

---

## 3B. Why the avoiding 4-constraint solver is NOT a failure — what it actually solves

### The problem DPCC has to fix
The flow/diffusion model samples a **joint** trajectory over the horizon —
`z = [a, x_des, y_des, x, y]` at every step. The *sampler* guarantees nothing about
mutual consistency: the raw sample can contain an action sequence that does **not**
actually produce its own position sequence (the network learned the data
distribution, not a hard dynamics law). It can output "action = move right" while
"position = stays put." DPCC projects that raw sample onto the feasible set
= {dynamically consistent} ∩ {safe / obstacle-free}.

**The four `deriv` rows are the "dynamically consistent" half.** They solve exactly
one thing: *force the trajectory to be a single coherent rollout in which the action
channel is the finite-difference derivative of the position channels.*

### What "both p and p_des read the same vx/vy" means in dynamics math
For avoiding, `dt=1` and the action **is** the displacement (`a = Δpos`,
`projection_eval.yaml:17`). The four rows are:
$$ x^{(t+1)} = x^{(t)} + a_x^{(t)}, \qquad y^{(t+1)} = y^{(t)} + a_y^{(t)} \quad\text{(actual } p\text{)} $$
$$ x\_des^{(t+1)} = x\_des^{(t)} + a_x^{(t)}, \qquad y\_des^{(t+1)} = y\_des^{(t)} + a_y^{(t)} \quad\text{(commanded } p\_des\text{)} $$

Feeding the **same** $a_x$ into both equations encodes one physical statement:

> **the single control action is the common cause of motion in BOTH the commanded
> channel and the realized channel.**

In dynamics terms: the action is the *shared latent control input*; `p_des` and `p`
are two **views of the same motion**, so they must share a derivative. For a system
that tracks its command tightly (the D3IL manipulator), this is *literally true* —
one commanded displacement advances both the setpoint and the end-effector by the
same amount. The constraint is not inventing physics; it is imposing the correct
physics of a tightly-tracked arm.

### What the coupling buys you (3 concrete wins → why it's a feature)
1. **Executable actions.** After projection `a^{(t)} = x^{(t+1)} − x^{(t)}` holds
   exactly. The action you pop off the plan and send to the robot moves the
   *actual* state to precisely the next planned point. Drop the `x`↔`a` rows and the
   executed action no longer keeps the robot on the planned path.
2. **Safety that is real, not on-paper.** Obstacle / halfspace constraints act on
   *positions* (`projection.py:114-127`). They only mean anything if executing the
   actions keeps the *realized* position on those safe points — which requires
   `x`↔`a` consistency. So the dynamics rows are what make the safety projection
   actually binding when the robot moves.
3. **Coherent re-conditioning.** `x_des` is part of the observation the policy
   re-conditions on each step; tying it to the same action keeps the whole state
   vector self-consistent for the next sample.

### Why "frozen offset" is the CORRECT behavior here (not a teleport bug)
Subtracting the two equations gives $(x - x\_des)^{(t)} = \text{const}$ (§3). That
constant is the **tracking error**, pinned to its value at the anchored initial
state (`projection.py:99-108`). For the avoiding arm the robot starts on its
setpoint ($p_0 \approx p\_des_0$) and tracks tightly, so the frozen error $\approx 0$
→ $p \approx p\_des$ throughout. That is exactly how that hardware behaves. **The
solver is encoding the true tracking physics of the domain — which is precisely why
it is a correct design, not a failure.**

### Why the same assumption (correctly) fails for the UAV — one line
A quadrotor has inertia: it cannot execute $\Delta p$ in one step, so
$x \neq x_{prev} + a$. The premise "the action is the common cause of both channels"
is **false** for a drone — its realized motion is governed by second-order dynamics
plus a downstream controller, not by the commanded displacement. Forcing the
4-constraint coupling would demand the actual position teleport onto the command
every step → an infeasible, unphysical plan. So the UAV keeps the **command
channel** rows only (`p_des` obeys the action integrator) and lets a real controller
(PID / MPC) realize `p` with true dynamics. **Same math machinery; a different,
domain-correct choice of which rows to include.**

---

## 3C. What the SciPy projection REALLY solves (exact math)

### 3C.1 The optimization problem (one batch element)
`Q = I` by default (`projection.py:41`) and `r = −z_raw @ Q` (`projection.py:88`), so
the per-sample cost (`projection.py:133`)
$$ \tfrac12 x^\top Q x + r^\top x \;=\; \tfrac12\lVert x\rVert^2 - z_{\text{raw}}^\top x \;=\; \tfrac12\lVert x - z_{\text{raw}}\rVert^2 - \text{const} $$
So **the objective is literally "find the feasible trajectory closest to the raw
network sample."** That is the entire meaning of "DPCC projection." The full program
solved per batch element $i$ (`projection.py:124-142`):

$$
\boxed{\;
\min_{x\in\mathbb{R}^{H\cdot T}} \tfrac12\lVert x - z_{\text{raw}}\rVert^2
\quad\text{s.t.}\quad
\underbrace{A x = b}_{\text{dynamics + anchor}},\;\;
\underbrace{C x \le d}_{\text{bounds / halfspace}},\;\;
\underbrace{x_t^\top P x_t + q^\top x_t \le v}_{\text{keep-out spheres (Geo)}},\;\;
-5\le x\le 5
\;}
$$

- $x = \mathrm{flatten}(\tau)\in\mathbb{R}^{H\cdot T}$ — the whole trajectory as one vector.
- Solver: **SLSQP**, warm-started at $x_0 = z_{\text{raw}}$ (`projection.py:135-142`).
- With only $A,C$ it is a **QP**; the sphere rows are nonconvex ⇒ it is a genuine **NLP** (§3C.5).

### 3C.2 What the dynamics block `A x = b` encodes (4-constraint avoiding)
Each `('deriv', [x_idx, a_idx])` produces, for $t=0..H{-}2$ (`projection.py:389-392`),
the recursion row plus one initial-anchor row whose RHS is overwritten with the
**current measured state** $s_0$ (`projection.py:99-108`):
$$ x_{\text{idx}}^{(0)} = s_0[x_{\text{idx}}], \qquad x_{\text{idx}}^{(t+1)} = x_{\text{idx}}^{(t)} + \Delta t\, a_{\text{idx}}^{(t)} $$
The 4 avoiding tuples therefore impose **simultaneously** (with $\Delta t=1$):
$$
\begin{aligned}
\text{(1) } & p_x^{(t+1)} = p_x^{(t)} + a_x^{(t)} &\text{(2) } & p_y^{(t+1)} = p_y^{(t)} + a_y^{(t)} \quad(\text{actual } p)\\
\text{(3) } & p\_des_x^{(t+1)} = p\_des_x^{(t)} + a_x^{(t)} \qquad &\text{(4) } & p\_des_y^{(t+1)} = p\_des_y^{(t)} + a_y^{(t)} \quad(\text{desired } p\_des)
\end{aligned}
$$
**What it solves:** of all trajectories whose position channels are the exact Euler
integral of the *same* action channel, pick the one nearest the raw sample. It
removes the sampler's freedom to emit actions that don't match its own positions.

### 3C.3 H2H — 4-constraint vs cond-on-p in TRACKING-ERROR math
Define the tracking error $e_t \equiv p_t - p\_des_t$.

**4-constraint (avoiding).** Subtract (3) from (1):
$$ e_{t+1} = \big(p^{(t)}+\Delta t\,a\big) - \big(p\_des^{(t)}+\Delta t\,a\big) = p^{(t)} - p\_des^{(t)} = e_t \;\Rightarrow\; \boxed{e_t = e_0\ \forall t} $$
The error is a **conserved quantity**: the optimizer bakes in a *rigid* tracking
model ($\dot e = 0$). With $e_0 = p_{\text{now}}-p\_des_{\text{now}}\approx 0$ for the
tightly-tracked arm ⇒ $e_t\approx 0$. Correct **iff the plant tracks rigidly.**

**cond-on-p default** (`[3,0],[4,1],[5,2]` — only $p\_des$, `eval_fm_uav.py:213-214`).
$A$ contains the $p\_des$ Euler rows; the actual $p$ (cols 6,7,8) appears in **no row
of $A$** ⇒ $p$ is a **free variable** in the projection. Because the cost is separable
and $p$ is unconstrained, $p_{\text{proj}} = p_{\text{raw}}$ (the projection doesn't
touch it). Hence
$$ \boxed{e_t\ \text{is UNCONSTRAINED in the optimizer}} $$
The error model is **delegated to the real controller** (PID/MPC) integrating true
second-order physics downstream: $p^{(t+1)} = f(p^{(t)},\dot p^{(t)},u^{(t)}) \neq p^{(t)}+a$.

**cond-on-p `anchor_to_p`** (`[6,0],[7,1],[8,2]` — only $p$, `eval_fm_uav.py:211`).
Symmetric: $p$ obeys the Euler integral anchored to the **measured drone position**;
$p\_des$ floats. Used to ground the plan in where the drone actually is.

| | constrained channel | tracking-error model in optimizer | valid when |
|---|---|---|---|
| 4-constraint | $p$ **and** $p\_des$ | $e_{t+1}=e_t$ (rigid, $e\approx0$) | plant tracks tightly (arm) |
| cond-on-p (default) | $p\_des$ only | $e_t$ free → solved by real controller | plant has lag (UAV) |
| cond-on-p (anchor_to_p) | $p$ only (to measured state) | $p\_des$ free | want plan grounded on real $p$ |

**One-line H2H:** the 4-constraint *embeds* the tracking dynamics ($e=\text{const}$)
inside the projector; cond-on-p *removes* it from the projector and hands $e_t$ to the
physical controller — which is the only correct choice for an inertial drone.

### 3C.4 What literally flows through SciPy (data trace)
`projection.py:81-152`: `τ[B,H,T] → reshape [B,H·T] → numpy float64`; then per
batch element `minimize(fun = ½xᵀx − z_rawᵢ·x, x0 = z_rawᵢ, jac, constraints,
method='SLSQP', bounds=[−5,5], tol=1e-6, maxiter=1000)`; `sol[i]=res.x`;
`projection_costs[i] = ½ solᵢᵀQ solᵢ + rᵢ·solᵢ + ½ z_rawᵢᵀQ z_rawᵢ`
( = the perturbation $\tfrac12\lVert sol-z_{\text{raw}}\rVert^2$, used by `dpcc-c`).
`b`'s anchor rows are rewritten to the current state every call
(`projection.py:99-108`). Output reshaped back to `[B,H,T]`.

### 3C.5 The "Geo" constraints — how SciPy NLP really enforces them (dpcc-r/-c/-t)
The geometric obstacle is a **keep-out ball**. `sphere_outside` flips the sign of
$P,q,v$ (`projection.py:456-459`) so the row becomes
$$ (x_t - c)^\top (x_t - c) \ge r^2 \quad\Longleftrightarrow\quad \text{stay OUTSIDE radius } r $$
This is **nonconvex** — exactly why a QP is insufficient and **SLSQP (sequential
quadratic programming)** is used: each iteration linearizes this nonlinear inequality
at the current iterate (using the analytic Jacobian `−2Px−q`, `projection.py:121-122`)
and solves a QP subproblem, iterating to a KKT point. One such row is added **per
obstacle per timestep** $t=1..H{-}1$ (`projection.py:114-122`). Triangle/wall edges
are **linear** halfspaces $Cx\le d$ (`constraints_helpers.py:4-20`); the workspace is
the **box** $-5\le x\le5$. So a fully-loaded DPCC solve mixes: linear-eq (dynamics) +
linear-ineq (halfspace/bounds) + nonconvex-quadratic-ineq (spheres), all projected
against $\tfrac12\lVert x-z_{\text{raw}}\rVert^2$.

**`dpcc-r / -c / -t` are not different solvers — they are selection rules over the
$B$ projected candidates** (`policies.py:65-75`), applied *after* the solve above:
- **`-r` (random):** take candidate 0 (`policies.py:74-75`).
- **`-c` (minimum_projection_cost):** $\arg\min_i \sum_t \text{cost}_i$ — the
  least-perturbed (closest raw→feasible) trajectory (`policies.py:69-73`).
- **`-t` (temporal_consistency):** $\arg\min_i \lVert \text{obs}_i[:\!-1] - \text{prev\_plan}[1:]\rVert$ —
  the candidate most consistent with last step's chosen plan (`policies.py:65-68`).

(Projection only fires on flow steps past `diffusion_timestep_threshold`=0.5, or all
steps for `post_processing`=0.0, `eval_fm_uav.py:229`. The `gradient` variant replaces
the SLSQP solve with a gradient step on $\tfrac12\lVert Ax-b\rVert^2 + \text{penalties}$,
`projection.py:157-211`.)

---

## 3D. "But the old research said it's just `p_des[t+1]=p_des[t]+a`?!" — both are true (don't cry)

**There is NO contradiction.** The simple equation and the "SciPy 4/2-constraint
problem" are the *same thing* seen at two zoom levels. Here is the reconciliation,
with code support for each level.

### Level 1 — ONE tuple = ONE simple recursion (the old research's view) ✅ code-supported
A single `('deriv', [x_idx, a_idx])` literally means
$$ x^{(t+1)} = x^{(t)} + \Delta t\,a^{(t)} $$
That is exactly the old research's `p_des[t+1] = p_des[t] + action`. It is encoded by
three matrix entries (`projection.py:389-391`):
```python
mat_append[i, i*T + x_idx]       = 1     #  +x^{(t)}
mat_append[i, i*T + dx_idx]      = self.dt   #  +dt·a^{(t)}
mat_append[i, (i+1)*T + x_idx]   = -1    #  -x^{(t+1)}        →  row · z = 0  ⇔  x^{(t+1)} = x^{(t)} + dt·a
```
So the "simple form" is **100% real and code-backed.** The old research was not wrong
about *what one row says*.

### Level 2 — STACK the rows = a matrix `A` (where "4 / 2 / 3" comes from)
The single recursion is repeated **for every timestep** `t = 0..H−2` and **for every
constrained channel**. Each tuple becomes `H−1` recursion rows + 1 anchor row, stacked
into `A` (`projection.py:400-401`, `:99-108`). Counting the *tuples*:

| pipeline | tuples (rows families) | which channels |
|---|---|---|
| avoiding (2D) | **4** = 2(p: x,y) + 2(p_des: x,y) | both `p` and `p_des` |
| cond-on-p 2D | **2** | one channel only |
| cond-on-p UAV (3D) | **3** (x,y,z) | one channel only |

That "**4 vs 2**" is just *how many of these simple recursions you stack into `A`*.
Same equation — you're choosing how many copies to enforce.

### Level 3 — WHY a solver and not just forward-integration (the part that confuses)
If the action were *given* and you only wanted `p_des`, you would just compute
`p_des[t+1] = p_des[t] + a` directly — no solver needed. **But that is not the
situation.** The flow network emits **all channels at once** — `action`, `p_des`,
*and* `p` — as a raw sample, and **they do not agree** (the raw `p_des` is not the
integral of the raw `action`; the net only learned a distribution, not a hard law).

So DPCC does not *forward-integrate*; it **projects**: find the closest trajectory in
which the recursion holds (`projection.py:133` cost = $\tfrac12\lVert x-z_{\text{raw}}\rVert^2$,
`:126-127` the recursion added as an **equality constraint** `A x = b`, `:135` solved by
SLSQP). The optimizer is free to nudge **both the action and the positions** to make
them consistent while staying nearest the sample — that is why it is an *optimization*,
not a one-line update. And once you also demand **safety** (`Cx≤d`, keep-out spheres,
§3C.5), forward-integration is impossible in principle — only a constrained solve can
satisfy dynamics **and** obstacles simultaneously.

### The one-sentence answer
> `p_des[t+1] = p_des[t] + a` is the **content of one constraint row**; the "SciPy
> 4/2-constraint problem" is what you get when you enforce **all such rows at once**
> on a network sample that violates them (and add safety) — so DPCC *projects* the raw
> trajectory onto the set where those simple equations hold. Same physics, two zoom
> levels. Nothing changed; nothing is contradictory.

---

## 3E. Three direct answers (the questions you keep asking)

### Q1. "Is `p_des[t+1] = p_des[t] + a` solved JUST in the `A x = b` matrix?"
**Yes — 100%.** The recursion lives *entirely* in the equality block `A x = b`.
Nothing else in the projector touches it:
- `A x = b` → the **dynamics recursion** (this equation) + the initial-state anchor.
- `C x ≤ d` → safety (bounds, walls). **Not** dynamics.
- `x_tᵀP x_t + q·x_t ≤ v` → keep-out spheres. **Not** dynamics.
- `½‖x − z_raw‖²` → "stay close to the raw sample." **Not** dynamics.

So: the recursion = the `A x = b` rows, full stop (`projection.py:389-391` builds the
row; `:126-127` adds it as the `'eq'` constraint; `:135` solves).

### Q2. "Old DPCC '4 constraints' = 4 copies of `p_des[t+1]=p_des[t]+a`?"
**Almost — one correction.** The 4 are the **same recursion form** applied to **4
different position variables**, and it is **NOT 4× `p_des`** — it is 2× actual `p`
PLUS 2× desired `p_des`, all driven by the **same 2 actions** `(a_x, a_y)`:

$$
\underbrace{p_x^{(t+1)}=p_x^{(t)}+a_x,\quad p_y^{(t+1)}=p_y^{(t)}+a_y}_{\text{2 rows: ACTUAL } p}
\qquad
\underbrace{p\_des_x^{(t+1)}=p\_des_x^{(t)}+a_x,\quad p\_des_y^{(t+1)}=p\_des_y^{(t)}+a_y}_{\text{2 rows: DESIRED } p\_des}
$$

(`constraints_helpers.py:48-53` → tuples `[4,0],[5,1],[2,0],[3,1]`.) So yes, each of
the 4 is structurally `q[t+1]=q[t]+a` — but the set is `{p_x, p_y, p_des_x, p_des_y}`,
covering **both** the actual and the desired channel. That "both, sharing the same
action" is the whole coupling story (§3, §3B).

### Q3. "On `anchor_to_p=True` (cond-on-p), do we swap `p_des` → `p_real + action`?"
**Yes — and it happens in TWO independent places. This is the real answer.**

**(A) Inside the projector** — the equality row switches which channel it pins
(`eval_fm_uav.py:207-214`):
```python
if anchor_to_p:
    constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # pin ACTUAL p (6,7,8)
else:
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # pin DESIRED p_des (3,4,5)
```
So in `A x = b`: `anchor_to_p=True` makes the recursion read
`p_real[t+1] = p_real[t] + a` (and `p_des` is left **free** in the solve);
`anchor_to_p=False` makes it `p_des[t+1] = p_des[t] + a` (and `p_real` is free).

**(B) Inside the rollout** — how the NEXT setpoint is computed each control step
(`eval_fm_uav.py:414-419`):
```python
if anchor_to_p:
    p_des = p + action          # GROUNDED: next setpoint = MEASURED drone pos + action  (closed-loop)
else:
    p_des = p_des + action      # FREE-RUNNING: next setpoint = previous setpoint + action (open-loop)
```

**What really happens when `anchor_to_p=True`:** the setpoint `p_des` is no longer an
open-loop accumulator marching in command space — it is **re-anchored to where the
drone actually is** every step (`p_des = p + action`). The name `p_des` is not deleted;
its *definition* changes from "previous command + action" to "measured position +
action."

**Why this matters (open-loop vs closed-loop setpoint):**
- `anchor_to_p=False` (free-running): if the drone lags (inertia), `p_des` keeps
  marching ahead → the gap `p_des − p` **grows without bound** → the drone chases a
  runaway target it can never reach.
- `anchor_to_p=True` (grounded): the next target is always "current real position +
  one action step" → the setpoint stays **reachable**, the error cannot run away. This
  is exactly closed-loop re-grounding (and why the projector also pins `p`, not `p_des`
  — both halves agree to ground on the real state).

> Terminology note: `anchor_to_p` (this eval/projector/rollout switch) is **different**
> from `cond_mode` (`pos_only` vs `p_des`, `eval_fm_uav.py:382-388`), which only changes
> the **observation layout fed to the model** (`[p_des|p]` vs `[p_des|p|v]`). "cond-on-p"
> in your notes = `anchor_to_p=True`.

---

## 3F. Control-theory reading: what the solver solves FOR, and what 4→2 really means

### 3F.1 What is the decision variable? (action? p_des? p?) → ALL of them
The QP/NLP does **not** "solve for the action" nor "solve for p_des" — its decision
variable is the **entire flattened trajectory**, every channel at every timestep
(`projection.py:81-85`):
$$ x = z = \big[\,\underbrace{a_0}_{\text{cols }0..2},\ \underbrace{p\_des_0}_{3..5},\ \underbrace{p_0}_{6..8},\ (v_0)\,\big|\,a_1, p\_des_1, p_1, \dots\big]^\top \in \mathbb{R}^{H\cdot T} $$
The **action columns are part of `x` and are projected too** — confirmed: the executed
command is sliced from the *projected* trajectory, `action = actions[which,0]`
(`policies.py:88-92`), with `actions = trajectories[:, :, :action_dim]`.

So `A x = b` is **not** "compute `p_des[t+1] = p_des[t] + a`." It is a **least-squares
projection that adjusts the action AND the positions jointly** so the recursion holds
while staying nearest the raw sample (cost `½‖x − z_raw‖²`, `projection.py:133`). The
recursion is the **constraint**; the **solved object** is a self-consistent
`(action, p_des, p)` trajectory. What we then *use* is `action[0]` — a
dynamically-cleaned action to execute.

> Mental model: the network proposes a messy `(a, p_des, p)`; the projector finds the
> nearest `(a*, p_des*, p*)` that actually obeys `Δposition = action` (and is safe);
> we send `a*[0]`.

### 3F.2 The old 4-constraint in strict control terms — a *holonomic perfect-servo* law
Stacking integrator rows on **both** `p` and `p_des` from the **same** input `a`:
$$ \dot p = a,\qquad \dot p\_des = a \;\;\Rightarrow\;\; \dot e = \dot p - \dot p\_des = 0,\quad e \equiv p - p\_des = \text{const}. $$
This is a **holonomic constraint** that *welds the plant output to the reference*: it
asserts the closed-loop transfer from reference to output is the **identity**
($G_{cl}\equiv 1$, infinite-bandwidth perfect servo, zero error dynamics). The
projector is modelling **plant + reference as one rigid body.** Valid for the D3IL arm
(it really does track that tightly); a *fiction* for an inertial drone.

### 3F.3 Dropping 4→2(→3): removing a false plant model → a *reference governor*
| | what the projector models | control-theory object | error `e=p−p_des` |
|---|---|---|---|
| **old 4-constraint** | reference **and** plant, welded | holonomic perfect servo, $G_{cl}=1$ | forced `e=const` (≈0) |
| **new default (p_des only)** | **reference only** ($\dot p\_des=a$) | **feedforward reference governor** (open-loop) | **free** — produced by the *real* plant + PID/MPC |
| **new anchor_to_p (p only)** | **plant**, re-grounded to measured `p` | **state-feedback reference governor** (closed-loop, MPC-style re-init) | regulated by re-anchoring each step |

So the change is **not** "weaken the constraint by deleting rows." It is a **change of
what is being modelled:**
- We **stop asserting a plant model inside the projector** (the false `p = ∫a`
  first-order claim for a second-order drone) and keep only the **reference
  kinematics** (`p_des` is a quantity *we define*, so `ṗ_des = a` is exactly true).
- The genuine plant dynamics + tracking error are handed to the **real controller**
  (PID/MPC) — the layer that is *supposed* to own them. This is the standard
  **planning ⊥ control separation**: a kinematically-feasible reference generator on
  top, a dynamics-aware tracker below.

### 3F.4 Why this is the correct control architecture (not just a hack)
- Forcing `p = ∫a` (1st-order) on a 2nd-order plant demands instantaneous velocity
  changes → infeasible / "teleport" reference (§3). Removing those rows removes an
  **infeasible model**, not a safety guarantee.
- `anchor_to_p`'s rollout `p_des = p + action` (`eval_fm_uav.py:417`) is precisely the
  **receding-horizon re-initialization** of MPC: each step, re-seed the plan at the
  measured state so the reference can never run away from the plant (§3E-Q3).
- Safety still holds: obstacle/halfspace rows act on whichever position channel is
  modelled, and the executed `action` remains the consistent integrator of that
  channel — so the executed command still drives the realized state along the safe
  reference (subject to the real tracker's bandwidth).

### 3F.5 One-paragraph summary
`A x = b` is a **constraint** (`Δposition = action`), not an assignment; the QP/NLP
solves for the **whole `(action, p_des, p)` trajectory** nearest the raw sample subject
to it, and we execute the projected `action[0]`. The old DPCC stacked this integrator
on **both** `p` and `p_des` → a **holonomic perfect-servo** law (`p≡p_des`), right for a
tight manipulator. The UAV drops the plant rows so the projector becomes a **reference
governor** over `p_des` only (open-loop) — or, with `anchor_to_p`, a **state-feedback
reference governor** over `p` re-grounded each step (closed-loop) — delegating the true
second-order error dynamics to the real PID/MPC controller. That is a deliberate,
correct **planning/control separation**, not a weakening of the constraints.

---

## 3G. A concrete number example — watch EVERY channel move (yes, really)

To kill the confusion: here is one timestep, **1 dimension**, `H=2`, `dt=1`, worked by
hand. The projection minimizes `½‖x − z_raw‖²` (identity cost: every coordinate
*wants* to stay at its raw value) subject to the dynamics equality. `p_des[0]` and
`p[0]` are anchored to the **measured** state (= 0 here) by `skip_initial_state`
(`projection.py:99-108`).

**Raw network sample (inconsistent on purpose):**
| channel | raw value |
|---|---|
| `a[0]` (action) | **1.0** |
| `p_des[1]` (desired next) | **1.6** |
| `p[1]` (actual next) | **0.3** |

Note the raw is **not** self-consistent: `0 + a = 1.0`, but the net emitted
`p_des[1]=1.6` and `p[1]=0.3`. The projector must fix that.

### Case A — NEW default (cond-on-p **off**): constrain `p_des` only `[3,0]`
Only one row: `p_des[1] = p_des[0] + a[0] = 0 + a[0]`. Variables coupled: `a[0]`,
`p_des[1]`. (`p[1]` is in **no** row → it is **free**.)

Residual `r = p_des_raw[1] − a_raw[0] = 1.6 − 1.0 = 0.6`. Min-norm projection onto the
line splits `r` equally (constraint normal `(1,−1)`, ‖·‖²=2):
$$ a^*[0] = 1.0 + \tfrac{r}{2} = \mathbf{1.3}, \qquad p\_des^*[1] = 1.6 - \tfrac{r}{2} = \mathbf{1.3}, \qquad p^*[1] = 0.3\ \text{(unchanged)} $$

| channel | raw → projected |
|---|---|
| `a[0]` | 1.0 → **1.3** ← the ACTION changed! |
| `p_des[1]` | 1.6 → **1.3** |
| `p[1]` | 0.3 → **0.3** (free, drone allowed to lag) |

**We send `a*[0]=1.3`, not the raw 1.0 and not 1.6.** The action and `p_des` both moved
to meet in the middle; the actual position was left alone.

### Case B — OLD 4-constraint: constrain `p_des` **AND** `p`, same action `[3,0]+[6,0]`
Two rows now share `a[0]`: `p_des[1]=a[0]` and `p[1]=a[0]`. Substitute and minimize
`(a−1)² + (a−1.6)² + (a−0.3)²` → `3a = 2.9` →
$$ a^*[0] = p\_des^*[1] = p^*[1] = \mathbf{0.967} $$

| channel | raw → projected |
|---|---|
| `a[0]` | 1.0 → **0.967** |
| `p_des[1]` | 1.6 → **0.967** |
| `p[1]` | 0.3 → **0.967** ← actual position **DRAGGED** to equal `p_des` |

The actual position was yanked from 0.3 to 0.967 to satisfy "p tracks the same action
as p_des." **That drag is the "teleport"** (§3): for a real drone with inertia, `p`
cannot jump like that — which is exactly why the UAV drops the `p` row.

### The answer to "really??? all changed???"
**Yes — the projector returns a whole consistent `(a*, p_des*, p*)`, and which ones
move depends on which rows you include:**
- **Default (cond-on-p off):** `a` and `p_des` move; `p` is free → drone may lag. ✅
- **Old 4-constraint:** `a`, `p_des`, **and `p`** all move — `p` is forced to match
  `p_des` (teleport), correct only for a perfect-tracking arm.

It is **never** "just compute `p_des = p_des + a`." It is a least-squares **redistribution
of the inconsistency across all coupled channels**, including the action we ultimately
execute. That is the whole point of *projection* (vs. forward integration).

---

## 3H. The deepest critique — "does predicting/projecting `p_real` make ANY control sense?"

This is the strongest objection raised, and **it is mostly correct.** Answered point
by point, without sugar-coating.

### Claim 1 — "The NN predicting `p_real` over the horizon is a defect."
**Partly right — depends on role (conditioning vs output vs constraint):**
- **As a conditioning INPUT** (feedback) `p_real` is *valuable*: it tells the model
  where the drone actually is so the next plan can correct. `eval_fm_uav.py:386,388`
  feeds `[p_des|p|…]` as the observation. ✅ keep.
- **As a generated OUTPUT over H1..H8** `p_real` is a *byproduct*: the controller only
  ever consumes `action[0]` (`policies.py:92`) and the rollout setpoint
  (`eval_fm_uav.py:417-419`). The predicted `p_real[1..H]` is **never executed**. So
  it is wasted forecast, not a control signal. ⚠️ harmless only if it doesn't distort
  the action.
- **As a CONSTRAINT** (forcing `p_real[t+1]=p_real[t]+a`) it is **a genuine defect for
  a UAV** — see Claim 3. ❌

### Claim 2 — "The projection changes ALL of `p_real`. Changing `p_des` is OK, but `p_real`?!"
**Only when `p_real` is in a `deriv` tuple.** Two regimes:
- **Default (cond-on-p OFF):** constraints are `[3,0],[4,1],[5,2]` = `p_des` only
  (`eval_fm_uav.py:213-214`). `p_real` (6,7,8) is in **no** row → **free** → cost
  leaves it at its raw value. **The projector does NOT touch `p_real` at all.** ✅ This
  is exactly your instinct: don't let the projector rewrite the plant's position.
- **Old 4-constraint / `anchor_to_p`=True:** `p_real` *is* constrained → it gets
  dragged (§3G Case B). For a drone, that drag is unphysical. ❌

So your "WTF, why rewrite `p_real`?" is **the correct objection to the 4-constraint
design**, and the UAV default already answers it by **not constraining `p_real`.**

### Claim 3 — "If H0 `p_real` is changed during optimizing, the whole receding-horizon output is wrong."
**Correct in principle — but H0 is LOCKED, so this failure does not occur.**
`skip_initial_state=True` adds a hard anchor row fixing each constrained dim at `t=0`
to the conditioned initial state (`projection.py:394-398`):
```python
mat_fix_initial[0, x_idx] = 1                 # pin dim x_idx at t=0
...
b[counter * self.horizon] = s_0[x_idx]        # = the t=0 (measured/conditioned) value  (projection.py:107)
```
and the flow sampler pins `observations[:,0,:] = obs_measured` by conditioning. So:
- H0 `p_real` = the **measured** position, **never moved** by the solve. The receding
  re-initialization is intact — the executed `action[0]` is grounded at the true state.
- Your worry "if H0 p_real changed → output total wrong" is a *valid* control
  principle; it simply doesn't trigger here because H0 is hard-pinned.

### Claim 4 — "Even with H0 locked, the H1..H8 difference between `p_real` and `p_des` is meaningless."
**Right for the 4-constraint / on-`p_real` case.** Forcing `p_real[t+1]=p_real[t]+a` is
asserting a **first-order, zero-lag plant** (`ṗ=a`). A quadrotor is **second-order with
inertia and a closed-loop tracker** — its true response is `p^{(t+1)}=f(p,\dot p,u)`,
NOT `p+a`. Therefore the projected `p_real[1..H]`:
- is **not** the true plant trajectory (wrong physics),
- is **not** a usable reference (the reference is `p_des`),
- is **not** executed (only `action[0]`/setpoint are).

⇒ It is a **kinematic fiction with no control meaning** — and worse, when constrained
it **distorts the action** by forcing it to serve a bogus `p_real` integrator. **You are
correct: from a control view it has ~0 meaning, and that is precisely why the UAV
default removes the `p_real` rows.**

### So what is `anchor_to_p`=True actually good for, if `p_real` horizon is fiction?
A narrow, honest use: it is **not** "predicting the plant." It **repurposes the
`p_real` channel as a feasible reference grounded at the measured position** —
`p_real[0]`=measured, then a kinematic path `+a` — and the rollout sets
`p_des = p + action` (`:417`). The *only* value extracted is a **grounded `action[0]` /
setpoint** that starts from where the drone really is (anti-windup / no setpoint
run-away, §3E-Q3). The H1..H8 `p_real` is still a discardable byproduct. So even the
"on-p" mode earns its keep **only** through `action[0]`, never through its horizon.

### Honest bottom line on your critique
| your objection | verdict |
|---|---|
| NN *generating/projecting* `p_real` over the horizon is wasted/defective | ✅ **agree** (byproduct, not executed) |
| projector *rewriting* `p_real` makes no control sense | ✅ **agree** — for a UAV it's wrong physics |
| if H0 `p_real` were altered the MPC output is invalid | ✅ valid principle, **but H0 is hard-locked** so it never happens |
| H1..H8 `p_des` vs `p_real` difference is meaningless under the on-`p_real` constraint | ✅ **agree** |
| therefore constrain `p_des` only (or use `p_real` purely as feedback) | ✅ **this is exactly the UAV default** |

**Conclusion:** your instinct *is* the correct design. The right architecture is:
**use `p_real` only as a conditioning/feedback input, never as a projected constraint;
plan and constrain `p_des` only** — which is `anchor_to_p=False`, the UAV default. The
4-constraint coupling and the on-`p_real` constraint are inherited DPCC artifacts that
are only valid for a perfect-tracking manipulator, and they are correctly *not* used in
the UAV default path.

---

## 3I. Direct critique of the DPCC METHOD itself (not just the UAV port)

Context: `README.md:20` — this repo descends from **Diffusion Predictive Control
(DPCC)**; the avoiding 4-tuple comes straight from DPCC's `formulate_dynamics_constraints`
(`constraints_helpers.py:47-53`). The critique below targets **DPCC's own design choice**
of constraining the *actual* position to the action.

### Step 1 — split the 4 constraints into two DIFFERENT classes
In D3IL `avoiding`: `x_des` = the controller **setpoint** (the action defines its
increment), `x` = the **actual** end-effector. The action is Δposition. So:

| rows | meaning | status |
|---|---|---|
| `[x_des,vx],[y_des,vy]` | `x_des[t+1]=x_des[t]+a` — how the **reference is generated** | **DEFINITIONAL / exact** ✅ |
| `[x,vx],[y,vy]` | `x[t+1]=x[t]+a` — the **plant** moves exactly like the command | **ASSUMED (perfect tracking)** ⚠️ |

DPCC stacks both classes off the **same** action. The reference rows are legitimate
(the action *is* the setpoint increment). **The actual-position rows are a hidden
modelling assumption** — and grep finds **no comment anywhere in the repo justifying
them** (the only "explanation" is the audited research.md prose). An undocumented
physical assumption baked into a solver is exactly the kind of thing that bites later.

### Step 2 — your core question: "is the tracking error actually tracked?" → NO
Define the tracking error $e_t \equiv x_t - x\_des_t$. The two classes together give
$$ e_{t+1} = (x_t + a) - (x\_des_t + a) = e_t \;\Rightarrow\; e_t = e_0\ \forall t. $$
DPCC **does not track, measure, or model the error dynamics** — it **freezes** the
error at its initial value. There is no $\dot e = g(e,\cdot)$ anywhere; the "error
model" is the *assumption* $\dot e = 0$. So the honest statement is:

> **DPCC does not track the tracking error — it assumes it away** (welds plant to
> reference as one rigid body over the whole horizon).

### Step 3 — "does forcing `p_des ≡ p` over H destroy all meaning?"
**Two halves, both true:**
1. **It destroys the *independent* meaning of the two channels.** After projection,
   `x` and `x_des` differ only by the frozen constant `e_0` — they carry the *same*
   information. Predicting both is **redundant**: the model could output one channel and
   add `e_0`. Over H1..H8 there is no genuine plant/reference distinction left.
2. **Whether that is harmful depends entirely on the domain:**
   - **Perfect-tracking manipulator (DPCC's avoiding):** `e_0≈0`, real `ė≈0`, so the
     assumption ≈ true. Nothing meaningful is lost because there *is* no meaningful
     error. The executed action stays consistent → benign. So DPCC is **not wrong on
     its own benchmark.**
   - **Any plant with real lag (UAV, soft/contact robots):** `ė≠0`. The actual-position
     rows are then **physically false**, the frozen-error assumption is a lie, and
     constraining `x` actively **corrupts the executed action** (§3G Case B drags `p`
     and distorts `a`). Here it genuinely **does** destroy meaning.

### Step 4 — verdict on DPCC (fair, but pointed)
- ✅ **DPCC's core idea is sound:** project a generative sample onto {dynamically
  consistent} ∩ {safe}. Not under attack.
- ✅ **The `x_des` (reference) rows are correct and exact.**
- ❌ **The `x` (actual-position) rows are a hidden perfect-tracking assumption**, valid
  *only* in the zero-lag regime, undocumented in the code, and silently wrong for any
  inertial/lagging plant. Calling them "dynamics constraints" oversells them: they are
  a **tracking assumption**, not measured dynamics.
- ❌ **DPCC does not track tracking error** — it freezes it. A method that claims to
  enforce "dynamics" while assuming the plant equals its reference is, for general
  systems, modelling a fiction.

### Step 5 — what the correct method is (and FM-PCC's UAV already does it)
Constrain **only the definitional reference channel** (`p_des`) and treat the **actual
position as feedback only**, never as a projected constraint — letting the real
closed-loop dynamics produce the error. That is precisely the UAV default
(`anchor_to_p=False`, §3H). So the UAV redesign is **not merely a port** — it is a
**correction of a DPCC modelling shortcut** that happened to be invisible because DPCC
was only ever evaluated on a perfect-tracking manipulator.

> One-line DPCC critique: *the "dynamics" projection conflates reference and plant by
> driving both off one action; it assumes (does not track) zero tracking error, which is
> exact only for a perfect servo — so on any lagging plant the actual-position rows are
> physically false and corrupt the executed action.*

---

## 3J. Does cond-on-p ground all 8 steps of H8, or only H0?

**Short answer: it is genuinely grounded to the REAL drone position only at H0. H1..H7
are kinematic fiction. But the receding-horizon loop re-grounds H0 on every executed
control step, so in practice every *executed* action is grounded — never the foresight.**

### Two mechanisms, two different reaches
1. **The projector constraint spans all H steps — but to a FICTION.** With
   `anchor_to_p=True`, the `p_real` deriv rows are built for every `t=0..H-2`
   (`projection.py:382` `for i in range(self.horizon-1)`), so `p_real[t+1]=p_real[t]+a[t]`
   is enforced across the whole horizon. **However**, only the **H0** row is pinned to
   the **measured** state (`mat_fix_initial`, `b[counter*H]=s_0[x_idx]`,
   `projection.py:394-398, 107`). H1..H7 are just the **forward integral of the action**
   — the drone has not flown there, nothing measures them. So the projector makes the
   plan *self-consistent* over H8, but **grounded to reality only at H0**; H1..H7 carry
   the same "kinematic fiction" status as §3H/§3I.

2. **The rollout grounds only H0, then re-solves.** Each control step `k`:
   - `p_des = p + action` uses the **measured** `p` and the **first** action only
     (`eval_fm_uav.py:417`),
   - only `action[0]` is executed (`policies.py:92`),
   - the loop re-measures `p` and re-plans (`eval_fm_uav.py:379-391`).

   So the H1..H7 of any single plan are **discarded**, never executed.

### The honest picture across the episode
| | within ONE H8 plan | across the rollout (many `k`) |
|---|---|---|
| grounded to measured `p` | **H0 only** | **every executed step** (because H0 is re-solved each `k`) |
| H1..H7 | kinematic fiction (action integral) | discarded, never executed |

So "does cond-on-p work for all 8?" — **No: the *real-position* grounding is an H0
property.** What makes the controller effectively grounded *at every step* is the
**receding-horizon re-solve** (fresh measured `p` → fresh H0 each control tick), not the
8-step plan. The horizon H1..H7 exists to give the flow model foresight and to let the
projector produce a consistent `action[0]`; it is **not** a grounded 8-step prediction
of the drone.

### Consequence (consistent with your earlier critique)
This is exactly why making the horizon *longer* does not buy grounded accuracy — beyond
H0 the `p_real` channel is the integrator fiction, so the only physically meaningful,
grounded quantity each tick is `action[0]` (and the H0 setpoint). The value of cond-on-p
is **per-tick grounding via re-solve**, not a trustworthy H8 trajectory.

---

## 3K. Reframing cond-on-p in real control / MPC / ML terms (the innovation)

The previous sections show *what the code does*. This one states *what it IS* in
standard control language — and why that reframing is the contribution.

### 3K.1 The canonical MPC problem, and how FM-PCC maps onto it
Classic receding-horizon MPC at tick $k$ (measure state $x_k$, solve, apply first input,
repeat):
$$
\min_{u_{0:N-1}} \;\; V_f(x_N) + \sum_{t=0}^{N-1}\ell(x_t,u_t)
\quad\text{s.t.}\quad x_{t+1}=f(x_t,u_t),\;\; x_0=x_k,\;\; g(x_t,u_t)\le 0 .
$$
FM-PCC is **exactly** this with three substitutions — two of them *learned* (the ML part):

| MPC ingredient | FM-PCC realization | code |
|---|---|---|
| stage+terminal cost $\ell + V_f$ | **learned proximity** $\tfrac12\lVert z - z_{\text{raw}}\rVert^2$, $z_{\text{raw}}\!\sim\! G_\theta(\cdot\mid o_k)$ | `projection.py:133`, flow sampler |
| internal model $x_{t+1}=f(x_t,u_t)$ | the `deriv` equality $A z = b$ (kinematic integrator) | `projection.py:362-401` |
| initial condition $x_0=x_k$ | the H0 anchor `b[..]=s_0` (measured) | `projection.py:394-398,107` |
| path constraints $g\le0$ | walls / keep-out spheres $Cz\le d,\ z_t^\top P z_t\!+\!\dots\le v$ | `projection.py:114-127` |
| apply $u_0$, discard rest | execute `action[0]`, re-measure, re-solve | `policies.py:92`, `eval_fm_uav.py:379-419` |

So **DPCC/FM-PCC = MPC whose cost is a generative model and whose model is a kinematic
integrator.** The "diffusion/flow" is the ML stage cost; everything else is textbook MPC.

### 3K.2 Where cond-on-p lives: the choice of *internal model* $f$
The only thing cond-on-p changes is **what $f$ models** — i.e. which rows go in $A$.
This is the classic MPC design lever of the **internal model**, and DPCC made the
*wrong* choice for a lagging plant:

- **DPCC (4-constraint):** $f$ models **reference *and* plant** off one input
  ($\dot p\!=\!a,\ \dot p\_des\!=\!a$) ⇒ asserts $p\equiv p\_des$ ⇒ **internal model =
  "perfect servo."** Maximal model–plant mismatch on an inertial drone (the
  *internal-model principle*: your controller is only as good as the model it carries;
  here the carried model is a fiction).

- **cond-on-p default (`anchor_to_p=False`) — HIERARCHICAL / CASCADED MPC.**
  $f$ models **only the reference** $\dot p\_des = a$ (a quantity we *define*, so the
  model is **exact**). The true plant $p$ is removed from $f$ and instead enters as
  **output feedback through the conditioning** $o_k=[p\_des,p,(v)]$ (`eval_fm_uav.py:386-388`).
  Architecture = **outer learned reference governor** (kinematic, re-solved each tick)
  **+ inner dynamics-aware tracker** (PID/MPC). This is the standard *planning ⊥ control*
  / cascade decomposition, and it is offset-free because the inner loop owns the real
  error.

- **cond-on-p anchor (`anchor_to_p=True`) — OUTPUT-FEEDBACK MPC with measurement re-init.**
  $f$ models the **actual** $p$ as a single integrator **re-initialized to the measured
  state every tick** ($p_0 = p_k$, then $p_{t+1}=p_t+a$), and the executed setpoint is
  $p\_des = p_k + a_0$ (`eval_fm_uav.py:417`). This is precisely the textbook cure for
  model–plant mismatch: **re-ground the nominal model on the measurement each step**
  (certainty-equivalence + receding-horizon re-initialization), which yields
  **offset-free / anti-windup** setpoint generation *without retraining the model.*

### 3K.3 The innovation, stated precisely
> **cond-on-p reinterprets DPCC's "dynamics constraint" as the MPC *internal model*,
> and corrects it for non-perfect-tracking plants by (i) reducing $f$ to the part that
> is definitionally exact — the reference kinematics — and (ii) injecting the true plant
> state as *output feedback* (conditioning) and/or *measurement re-initialization*
> (anchor), instead of as a false constrained state.**

Equivalently, in ML terms: the flow model is a **learned terminal/stage cost** (a prior
over expert-like trajectories); cond-on-p ensures the **only hard model** the optimizer
trusts is one we *know* is exact ($\dot p\_des=a$), and lets **measurement feedback +
the inner controller** absorb everything the learned prior and the kinematic model do not
capture (inertia, disturbances). That is a clean separation of *learned intent*,
*exact kinematics*, and *measured reality* — the part that is genuinely novel here.

### 3K.4 Why this is the principled fix (one line each)
- **Internal-model principle:** carry only a model you can trust ⇒ keep $\dot p\_des=a$,
  drop the perfect-servo fiction.
- **Separation / cascade:** plan a kinematically-feasible reference; let a dynamics-aware
  inner loop realize it ⇒ tracking error is *handled*, not *assumed*.
- **Offset-free MPC:** re-initialize the internal model to the measurement each tick
  (`anchor_to_p`) ⇒ no setpoint run-away, robust to model–plant mismatch, no retraining.
- **Certainty equivalence + receding horizon:** trust `action[0]`, re-measure, re-solve
  ⇒ the H1..H7 fiction never reaches the plant (§3J).

---

## 3L. Practical diff — what *actually* changed in the code (the "drop 2" + "extra line")

These are **TWO separate edits** people keep conflating. Here is the literal before/after.

### Change A — the projector: DROP the actual-position rows
DPCC (avoiding, 2D) builds **4** rows = 2 for `p` + 2 for `p_des`, off the same action
(`constraints_helpers.py:48-53`). In 3D that pattern is **6** (3 `p` + 3 `p_des`). The
UAV **does not build the `p` rows at all** — it hardcodes only one channel's 3 rows
(`eval_fm_uav.py:206-214`):

```python
# OLD DPCC pattern (conceptual, all channels) — 2D: 4 rows / 3D: 6 rows
#   p rows:     [x, vx], [y, vy]   (+ [z, vz])
#   p_des rows: [x_des, vx], [y_des, vy]   (+ [z_des, vz])

# UAV PRACTICE — only ONE channel, 3 rows:
if anchor_to_p:
    constraint_list += [('deriv',[6,0]),('deriv',[7,1]),('deriv',[8,2])]   # keep p   (drop p_des rows)
else:                                                                       # DEFAULT
    constraint_list += [('deriv',[3,0]),('deriv',[4,1]),('deriv',[5,2])]   # keep p_des (drop p rows)
```

So the "**drop 2 from 4**" (2D) = "**keep 3 of 6**" (3D) = **keep exactly one position
channel, drop the other.** It is not a tweak to the rows — it is *omitting half the
constraint families*.

**Why drop them in practice (not theory):** the actual-`p` rows force `p[t+1]=p[t]+a`,
which on a real drone (a) is physically false (inertia/lag) and (b) **drags the executed
`action[0]`** toward a teleporting `p` (§3G Case B — `a` went 1.0→0.967 only because of
the `p` row). Removing those rows means the solver no longer corrupts the action to chase
an impossible actual-position integral; the action now serves **only** the reference it
can actually command.

### Change B — the rollout: did it add `p_des = p + action`? YES — but only in anchor mode
This is a **different** edit, in the rollout loop (`eval_fm_uav.py:414-419`):

```python
if anchor_to_p:
    p_des = p + action          # ← the ADDED grounding line (measured p)   [anchor mode only]
else:
    p_des = p_des + action      # ← DEFAULT: free-running open-loop integration
```

- **Default** (`anchor_to_p=False`): the setpoint integrates **itself** (`p_des += action`)
  — the natural open-loop reference matching the projector's `p_des` rows. No grounding
  line.
- **Anchor** (`anchor_to_p=True`): the extra line `p_des = p + action` re-grounds the
  setpoint on the **measured** position each tick — and it is **paired** with Change A's
  anchor branch (projector constrains `p`, rollout grounds on `p`).

### The pairing (so it's unambiguous)
| mode | projector rows (Change A) | rollout line (Change B) | meaning |
|---|---|---|---|
| **default** `anchor_to_p=False` | keep `p_des` only `[3,4,5]` | `p_des = p_des + action` | open-loop reference governor |
| **anchor** `anchor_to_p=True` | keep `p` only `[6,7,8]` | `p_des = p + action` (added) | measurement-grounded reference |

### One-paragraph answer
**Yes to both, but they are two edits.** Practically the UAV (1) *omits half the
`deriv` families* — keeps one position channel's 3 rows instead of all 6 (the "drop 2
from 4" in 2D terms) — so the solver stops corrupting `action[0]` with a false
actual-position integral; and (2) **only in `anchor_to_p` mode** adds the rollout line
`p_des = p + action` to re-ground the setpoint on the measured position, paired with the
projector switching to constrain `p`. The plain default keeps the ordinary
`p_des = p_des + action` integration.

---

## 3M. "Is the UAV default a BUG? It should be 4 when false!" — decisive H2H across ALL pipelines

This is the strongest bug claim, so it gets a hard, code-checked answer. **Verdict: NOT
a bug — but the premise ("everything else uses 4") is false.** Here is what every
pipeline actually builds, verified line-by-line.

### The actual constraint count in each codebase
| pipeline | layout | how built | dynamics rows | channel(s) constrained |
|---|---|---|---|---|
| **legacy avoiding** (`fmv3`, `fmv3ode`, diffuser) | 6D, **2D** task | helper `formulate_dynamics_constraints` (`scripts/eval.py:145`) | **4** | **BOTH** `p`(4,5) **and** `p_des`(2,3) |
| **fm_visual_aligning** | **9D** | hand-built in its eval (`eval_fm_visual_aligning.py:126-128`) | **3** | **actual EE** `(6,7,8)` only |
| **UAV default** (`anchor_to_p=False`) | 9D+ | hand-built (`eval_fm_uav.py:213-214`) | **3** | **desired** `p_des`(3,4,5) only |
| **UAV anchor** (`anchor_to_p=True`) | 9D+ | hand-built (`eval_fm_uav.py:211`) | **3** | **actual** `p`(6,7,8) only — *identical to aligning* |

### What this table proves
1. **"4" exists ONLY in the legacy 6D `avoiding` task.** It is `2 spatial dims × 2
   channels = 4`. It is a property of the **avoiding helper**, not a universal standard.
2. **Every 9D pipeline already uses 3 (one channel).** `fm_visual_aligning` — your own
   cited baseline — constrains **only the actual EE** `(6,7,8)`, three rows, **never
   six**. So the UAV is **not** the odd one out; the "one-channel" design predates it.
3. **The UAV did NOT "destroy 4 into 2."** In 9D the faithful count was never 6 — the
   sibling (`aligning`) was already 3. The UAV keeps 3.
4. **The only thing the UAV adds is the *choice of channel*:** `aligning` (and UAV
   `anchor_to_p=True`) pin the **actual** position; the **UAV default pins the command
   `p_des` instead.** That is the deliberate `fix_5` redesign — correct for a *lagging*
   drone (don't constrain the inertial actual position), where `aligning`'s perfect-PD
   arm could safely pin the actual.

### So your two specific hypotheses, judged
- ❌ **"when false it should be 4"** — No. Even the closest 9D sibling (`aligning`) uses
  **3**, not 6/4. Putting the second channel back would **re-introduce the DPCC
  perfect-servo coupling** (§3, §3I) — i.e. it would re-create the *teleport* defect the
  9D pipelines already abandoned. Default = 3 is the correct cardinality.
- ✅ **"when true, 3 makes sense"** — Yes, and it is *literally the aligning choice*
  (pin actual `(6,7,8)`).
- ⚠️ **count confusion:** you keep saying "2"; in 9D it is **3** (x,y,z of one channel).
  "4→2" is the 2D avoiding mental image; the 9D reality is "never-6, always-3."

### Your redundancy intuition ("after anchor in H0, 2 of them are the same as the other 2")
Mathematically sharp, and the answer pins down *why one channel is enough*:
- The two channels' rows are `p[t+1]=p[t]+a` and `p_des[t+1]=p_des[t]+a`. They are
  **identical recursions driven by the same `a`** — they differ **only by their H0 anchor
  value** (`p[0]` vs `p_des[0]`, set by `skip_initial_state`, `projection.py:99-108`).
- **If `p[0] = p_des[0]`** (drone exactly on its setpoint): the two row-blocks are
  **literally redundant** — solving with 1 channel = solving with 2. Dropping to 3 loses
  **nothing**.
- **If `p[0] ≠ p_des[0]`** (drone has lagged): keeping both **forces the lag to stay
  frozen** for the whole horizon (`e_t=e_0`) = the teleport coupling. Dropping to one
  channel correctly lets the other float.
- ⇒ **One channel is never worse than two, and strictly better when there is lag.** So
  `default = 3` is correct in *both* cases. (Note: the H0 *value*-pin from
  `skip_initial_state` is a **different** "anchor" from the `anchor_to_p` *channel*
  switch — conflating those two is the likely source of the "feels contradictory"
  feeling.)

### Two different things both called "anchor" (disambiguation)
| term | what it is | scope | always on? |
|---|---|---|---|
| `skip_initial_state` H0 pin | fixes each constrained dim's **t=0 value** to the measured state | inside the projector, per constraint | **yes** (default True) |
| `anchor_to_p` | chooses **which channel** the deriv rows constrain (`p` vs `p_des`) + adds rollout `p_des=p+action` | eval-level mode | no (default False) |

### Bottom line
**Not a bug.** The "4" is a legacy-`avoiding`-only artifact; all 9D pipelines use 3, and
`aligning` proves the one-channel design is the established norm. The UAV default
correctly uses 3 and merely **chooses the command channel** (drone-correct) instead of
the actual channel (arm-correct). Restoring the second channel to "make it 4" would
*reintroduce* the perfect-servo/teleport defect, not fix anything.

---

## 3N. Is DPCC right and our ports wrong? — humility, the open question, and a debug plan

### Epistemic stance (you are right to force this)
DPCC is **published / peer-reviewed**; the d3il `avoiding` pipeline mirrors it almost
verbatim. The `aligning` and `UAV` pipelines are **AI-generated derivatives**. So the
correct prior is: **if a port diverges from DPCC, suspect the PORT, not DPCC.** I must also
flag honestly: my earlier sections asserted *why* DPCC constrains both channels by
**inferring from code, not from the DPCC paper.** That inference is **unverified** — treat
it as a hypothesis, not fact.

### What is CERTAIN (verified from code, not inference)
- **d3il avoiding** (`scripts/eval.py:145` + helper): **4** rows = `p`(4,5) **and**
  `p_des`(2,3), both ← action. Both channels REAL; `p ≈ p_des` (tight arm).
- **visual aligning** (`eval_fm_visual_aligning.py:126-128`): **3** rows on the "actual"
  channel `(6,7,8)` — **but that channel is a literal copy of `des`** (`:728-730`: obs =
  `[des_robot_pos, des_robot_pos]`; D3IL never exposes the real `c_pos`).
- **UAV default** (`eval_fm_uav.py:213-214`): **3** rows on `p_des`(3,4,5); the **REAL**
  `p`(6,7,8) (`data.qpos`, `:380`) is **left unconstrained**.
- **UAV anchor** (`:211,417`): **3** rows on `p`(6,7,8); `p_des` reconstructed as `p+action`.
- action `a = Δp_des` (`:240`).

### The OPEN QUESTION that decides whether the ports are wrong (needs the DPCC paper)
*Why does DPCC constrain BOTH channels?* Two readings, only resolvable from the paper:
- **(A) "constrain every predicted state channel."** The diffusion model predicts the full
  observation trajectory; each predicted position channel must be a dynamically-consistent
  (Euler) rollout so the **whole** predicted state-action plan is feasible. Under (A) the
  4 rows are not about coupling — they are 2 independent feasibility constraints.
- **(B) "couple command to actual"** (the perfect-servo reading I used earlier).

**Under (A) — the conservative, more-likely-correct reading — our ports are INCOMPLETE:**
- avoiding ✅ both predicted real channels constrained.
- aligning ⚠️ constrains a **fake** channel; the prediction is not made feasible on a real
  state at all.
- **UAV default ⚠️ leaves the REAL predicted `p` channel dynamically UNconstrained** — a
  genuine deviation from DPCC. (This is exactly your worry, and under (A) it is valid.)

So I am **withdrawing** my earlier flat claim that "UAV default is simply correct." The
honest status is: **it deviates from DPCC, and whether that deviation is acceptable depends
on (A) vs (B), which I have not verified from the paper.**

### Why the deviation is *probably* control-benign — but still a real fidelity gap
- The **executed** command `action[0]` comes from the `p_des` channel, which **is**
  constrained ⇒ the command we actually send is dynamically consistent.
- The unconstrained `p` horizon is a **non-executed byproduct** (§3H, §3J) ⇒ no *control*
  harm in dynamics-only runs **today**.
- BUT: (a) it is a **deviation from DPCC's principle (A)**; (b) it **bites** once obstacles
  act on `p` (latent geo bug); (c) the predicted trajectory is only **partially feasible**.
⇒ Likely not catastrophic, but a **real fidelity gap that needs the debug pass below** —
not something to wave away.

### The crucial physics caveat for any UAV "fix to 6"
If you do constrain the real `p` to honor DPCC, it **must** be `p ← v` (true kinematics),
**NOT** `p ← a`. Because `a = Δp_des` and `p ≠ p_des`, the row `p[t+1]=p[t]+a` asserts the
drone moves by the *command* delta each step — false for an inertial plant. The faithful
"full" is `p_des←a` (command) **+** `p←v` (actual), two **different** integrators (different
`dt`). This needs the velocity channel (only in `p_des`/12D cond_mode, absent in today's
`pos_only`/9D runs) and per-constraint `dt` (unsupported: single `self.dt`,
`projection.py:341,385`). **So "just add 3 more `p←a` rows to match DPCC" is the wrong fix.**

### `cond_on_p = True` — the ONE case where a drop is rigorously justified
You are right to single this out. `anchor_to_p=True` constrains `p←a` (grounded at measured
`p`) and **reconstructs** `p_des = p + action` in the rollout (`:417`). Because `p_des` is
*computed*, not freely planned, its projector rows are genuinely unnecessary → dropping them
is **exact, not a simplification.** This is the only mode where "drop half" is provably
lossless. (In the *default* mode the drop of `p` is weaker — justified only by "p is not
executed / `p←a` would be false," NOT by reconstruction.)

### MASSIVE DEBUG PLAN — re-audit every model against DPCC
1. **Settle (A) vs (B): READ the DPCC paper / reference repo** for the *stated purpose* of
   the dynamics constraint. This is the linchpin; everything below depends on it.
2. **Per model** (`avoiding`, `fm_/imf_/diffuser_visual_aligning`, `fm_/ddpm_encdec_vision`,
   `UAV`): tabulate — (a) which state channels the model *predicts*, (b) real vs fake,
   (c) which are *constrained*, (d) which are *executed*. **Flag every PREDICTED+REAL channel
   left unconstrained** (under (A) that is a bug).
3. **aligning family:** confirm the actual channel is a `des`-copy across all three ports;
   decide whether `dynamics` is a harmless `des`-proxy or should constrain `des`(3,4,5)
   directly. Re-run and compare to the avoiding reference.
4. **UAV:** decide (a) keep command-only default (accept the documented fidelity gap) vs
   (b) add `p←v` for DPCC fidelity (needs `v` + per-constraint `dt`). A/B test on
   success/safe/track-err.
5. **Use `avoiding` as GROUND TRUTH:** any port whose metrics diverge unexpectedly from the
   DPCC-faithful avoiding behavior → prime suspect = a dropped/fake channel.

### Bottom line (honest)
Your instinct is sound: **trust DPCC, suspect the ports.** Confirmed facts: avoiding is
DPCC-faithful (both real channels); **aligning constrains a FAKE channel (a des-copy)**;
**UAV default leaves the REAL `p` channel unconstrained** — a deviation from DPCC. Whether
that deviation is "fine" or "a bug" hinges on the DPCC paper's actual intent (reading A vs
B), which I have **not** verified — so I am downgrading my earlier "UAV is correct" to
"**UAV deviates; benign for today's executed command, but unproven and needs the debug
pass.**" The only rigorously-justified drop is `cond_on_p=True` (because `p_des` is
reconstructed). If the real `p` must be constrained, it is `p←v`, never `p←a`.

---

## 3O. WHICH CODES ARE WRONG (missing dynamics rows) + the FIX

**The principle (applied, no paper):** *every distinct REAL predicted position channel
gets a dynamics row-set.* 2D avoiding = 2 real channels × 2 dims = **4 rows**. A 3D
pipeline with **2 real channels** therefore needs 2 × 3 = **6 rows**. A pipeline with
**1 real channel** needs 3. **Constrain fewer than that ⇒ WRONG (under-constrained).**

### The verdict table — count real channels, compare to rows built
| code | dims | REAL position channels | rows built | rows REQUIRED | status |
|---|---|---|---|---|---|
| **d3il avoiding** (`fmv3ode`,`fmv3`,`diffuser`) | 6D | **2** (`x_des`, `x`, both real) | **4** (`scripts/eval.py:145`) | 4 | ✅ **CORRECT** |
| **fm_encdec_vision** | 6D | **1** (`des` only; no actual exposed) | **3** (`:91-93`) | 3 | ✅ **CORRECT** |
| **ddpm_encdec_vision** | 6D | **1** (`des` only) | **3** (`:91-93`) | 3 | ✅ **CORRECT** |
| **fm_visual_aligning** | 9D | 1 real `des`(3,4,5) + 1 **FAKE** copy(6,7,8) | **3 on the FAKE** (`:126-128`) | constrain the **real** one | ❌ **WRONG (constrains the copy, leaves real `des` free)** |
| **imf_visual_aligning** | 9D | same | **3 on FAKE** | — | ❌ **WRONG (same)** |
| **diffuser_visual_aligning** | 9D | same | **3 on FAKE** | — | ❌ **WRONG (same)** |
| **UAV — default** (`anchor_to_p=False`) | 9D/12D | **2** (`p_des` real + `p` real) | **3** (`p_des` only) | **6** | ❌ **WRONG (missing the real `p` channel)** |
| **UAV — anchor** (`anchor_to_p=True`) | 9D/12D | 2, but `p_des` reconstructed in rollout | **3** (`p` only) | 3 (p_des not free) | ✅ **OK (justified drop, §3N)** |

### So the WRONG list (to fix)
1. **`fm_visual_aligning`** — constrains the duplicated/fake actual `(6,7,8)`, not the
   canonical `des (3,4,5)`.
2. **`imf_visual_aligning`** — same.
3. **`diffuser_visual_aligning`** — same.
4. **`UAV default`** — constrains `p_des (3,4,5)` only; the **real** `p (6,7,8)` channel is
   left unconstrained → **3 rows where the layout has 2 real channels = should be 6.**

### CORRECT (leave alone)
- `d3il avoiding` — full 4 (2 real channels). ✅
- `fm_encdec_vision`, `ddpm_encdec_vision` — 6D, **only one real channel**, full 3. ✅
- `UAV anchor` — drops `p_des` rows but reconstructs `p_des = p + action`, so 3 is exact. ✅

### THE FIX (concrete code)

**A) Visual aligning ×3** — `eval_*_visual_aligning.py:126-128`. The `(6,7,8)` channel is a
copy of `des`; constrain the **canonical des** channel too (and since they're equal, this
also fixes the "constrains the copy" smell). Make it the full pair:
```python
# add the desired-channel rows alongside the existing actual(copy) rows  → 6 rows
constraint_list.append(('deriv', [3, 0]))   # des_x ← a
constraint_list.append(('deriv', [4, 1]))   # des_y ← a
constraint_list.append(('deriv', [5, 2]))   # des_z ← a
# (existing) [6,0],[7,1],[8,2]  — actual(copy) ← a
```
(Because actual≡des here, this is degenerate-but-now-complete; the cleaner alternative is
to constrain only `des (3,4,5)` and stop pretending `(6,7,8)` is a real plant state.)

**B) UAV default** — `eval_fm_uav.py:213-214`. Add the missing **real `p`** channel so the
9D layout's *both* real channels are constrained (6 rows total):
```python
else:  # anchor_to_p == False
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]   # p_des ← a  (existing)
    constraint_list += [('deriv', [6, 9]), ('deriv', [7,10]), ('deriv', [8,11])]   # p ← v      (ADD: real kinematics)
```
**Critical:** the added rows must be **`p ← v`** (cols 9,10,11 = velocity), **NOT `p ← a`**.
`a = Δp_des`; a real drone does **not** move by the command delta (`p ≠ p_des`), so `p←a`
is false physics (§3N). `p←v` is the true kinematic law.
**Prerequisites for B (must do first):**
- run in **`p_des`/12D cond_mode** (has the `v` channel); `pos_only`/9D has **no `v`** → B
  is impossible there, so either switch cond_mode or skip B for pos_only.
- add **per-constraint `dt`** to `DynamicConstraints.build_matrices` (today single
  `self.dt`, `projection.py:341,385`): `p_des←a` uses `dt=1`, `p←v` uses `dt=1/control_hz`.

### Order of work
1. Add per-constraint `dt` support in `projection.py` (blocks B).
2. UAV default → add `p←v` rows (6D-equivalent full); A/B vs current 3-row on
   success/safe/track-err.
3. Visual aligning ×3 → either complete to 6 or switch to constraining real `des(3,4,5)`;
   re-run vs the avoiding reference.
4. Leave `avoiding`, `encdec`, `UAV anchor` untouched (already correct).

### Bottom line
**4 codepaths are under-constrained:** the **3 visual-aligning ports** (constrain a fake
copy, not the real `des`) and the **UAV default** (missing the real `p` channel — 3 rows
where 6 are required). Fix = add the missing channel's 3 rows — **`p←v` for the UAV** (never
`p←a`), and the real `des` rows for aligning. `avoiding`, both `encdec`, and `UAV anchor`
are already correct.

---

## 4. The UAV redesign — verified, and it is SOUND

`FM_v3_uav_test/eval_fm_uav.py:206-214`
```python
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    if anchor_to_p:
        constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]   # bind p_actual
    else:
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]   # bind p_des (default)
```
UAV layout (`eval_fm_uav.py:192,208`): `[act 0-2 | p_des 3-5 | p_actual 6-8 | v…]`,
`dt=1.0` because the action *is* Δp_des (`:240`).

- **Default (`anchor_to_p=False`)**: 3 rows constrain **only `p_des`**; `p_actual`
  (6,7,8) is absent from `A` → floats. ⇒ commanded path obeys Euler integration of
  the action; the drone's actual position is free to **lag due to inertia**. ✅ correct.
- **`anchor_to_p=True`**: 3 rows constrain **only `p_actual`**; `p_des` floats.
- **Never both.** It is an `if/else`, so the avoiding-style p↔p_des coupling
  (6 stacked rows) **cannot occur** in the UAV path. ✅

This matches the design intent documented at `eval_fm_uav.py:180-182`:
> "The dynamics `deriv` binds **p_des (3,4,5)** to the action (0,1,2) — NOT the
> actual p — because p_des is the exact integrator of the action … while the
> drone's p lags."

**Conclusion:** decoupling p from p_des is the *physically correct* choice for an
inertial UAV. research.md's conclusion on this point is right; only its "the old
code was buggy" justification is wrong (it was right for its own domain).

---

## 5. "Did the UAV delete the original code?" — NO (this is the key safety point)

- `flow_matcher_v3_uav/utils/constraints_helpers.py:47-53` **still contains** the
  full avoiding 4-tuple logic, untouched.
- `scripts/eval.py:145` **still calls** `formulate_dynamics_constraints` for
  avoiding / pointmaze / antmaze.
- `eval_fm_uav.py` **never imports/calls** `formulate_dynamics_constraints`
  (verified: `grep` returns nothing) — it hardcodes its own 3 tuples additively.

So the UAV uses the **safe pattern: bypass, not delete.** The original DPCC paper
pipeline is fully preserved and still reproducible.

> research.md's wording — "you completely deleted that line", "destroying the
> 4-constraint problem", "you bypassed `constraints_helpers.py` completely" — is
> **misleading**. Bypassing ≠ deleting. Nothing was removed.

### Why dropping the original WOULD be dangerous (answering the user's worry)
1. The 4-constraint avoiding logic is **correct** for tight-tracking manipulators
   (perfect tracking is the intended behavior, not a defect).
2. It is **still in active use** by `scripts/eval.py` for the published DPCC
   baselines (avoiding/pointmaze/antmaze). Editing the shared helper to "fix" a
   non-bug would silently break those results.
3. The coupling is **avoiding-specific** by construction: only `avoiding` puts both
   `x` and `x_des` in the observation, so only it generates the doubled rows.
   pointmaze (`:37-40`) and antmaze (`:42-46`) have no `x_des` and are unaffected.

**Recommendation: do NOT modify `constraints_helpers.py` or `scripts/eval.py`.**
Keep the UAV's separate, additive bypass exactly as it is.

---

## 6. Files referenced (line-anchored)
- `flow_matcher_v3_uav/utils/constraints_helpers.py:34-54` — `formulate_dynamics_constraints` (4 avoiding tuples)
- `flow_matcher_v3_uav/sampling/projection.py:55-68` — routing `deriv`→`DynamicConstraints`
- `flow_matcher_v3_uav/sampling/projection.py:362-401` — `build_matrices` (normalized vs un-normalized branch, skip_initial_state)
- `flow_matcher_v3_uav/sampling/projection.py:99-108,124-127` — `project()` initial-state anchoring + SLSQP eq constraint
- `FM_v3_uav_test/eval_fm_uav.py:180-214,237,240` — UAV hardcoded 3-constraint bypass + normalizer + dt=1.0
- `scripts/eval.py:110-111,145` — original avoiding index resolution + helper call
- `config/projection_eval.yaml:16-25` — avoiding dt / obs / action index layout
