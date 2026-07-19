# D1 — Does the initial box position collide with the obstacle constraint, and does that make DPCC projection provably infeasible?

**Scope:** pure theory + code check. No execution (cluster-only per CLAUDE.md).
**Trigger:** review of `temp/22_57_36_eval_visual_aligning_dpcc_23514.log` (job 23514, `combined_5`, seed 6),
where the SLSQP circuit breaker (`Fix_15.2`) tripped 4×.
**Question asked:** *"if the start box overlaps the obstacle theoretically, projection will 100 % fail — is that
already impossible, already safeguarded, or does it need a fix?"*

---

## 0. Verdict (read this first)

| Claim | Answer |
|---|---|
| Does the **EE start** inside the obstacle? | **No.** 0.351 m from centre vs r = 0.06 / 0.09 m. Safe by ~4–6×. Not possible for any context. |
| Does the **box start** overlap the obstacle volume? | **Yes, geometrically** — up to 0.031 m (nominal) / 0.061 m (tightened) for rotated boxes near the corridor edge. |
| Does that box overlap make the projection infeasible? | **No.** The obstacle constrains **EE dims (6,7) only**. The projector never sees the box. Box/obstacle overlap is *not* a constraint at all. |
| Is there a real 100 %-infeasible trap? | **Yes — but mid-episode, not at init.** The obstacle sits dead-centre in the *only* push corridor, and ~65–76 % of its interior is a provably inescapable pocket under the action bound. |
| Is it safeguarded? | **Partially and by accident.** `Fix_15.2` catches the *symptom* (wall-clock), never the *cause* (infeasibility). `res.success` is **never checked** — an infeasible solve is silently accepted as a valid projection. |
| Did it fire in run 23514? | **No.** `obs=0` violations in *every* rollout; the box never moved. The 4 breaker trips came from **halfspace + action-bounds** thrash, not the obstacle. |

**Bottom line:** the thing you suspected (init overlap ⇒ guaranteed failure) is **not** the live bug. But a
strictly worse latent version of it exists, and it is armed to detonate **exactly when the policy starts working** —
because the trap only gets visited on a *successful* push. Two small fixes are warranted (§6).

---

## 1. Scene geometry — the actual numbers

All from source, not assumption.

### 1.1 Obstacle constraint (config)
`config/visual_aligning_eval.yaml`, entry `combined_5` (the only active one — `active_geo_variants: [combined_5]`):

```yaml
obstacle_constraints:
  - type: sphere_outside
    dimensions: ['x', 'y']       # → trajectory dims 6, 7 = EE ACTUAL c_pos
    center: [0.50, 0.00]
    radius: 0.06
enlarge_constraints: 0.03        # tightened twin → r = 0.09
```

Mapped in `eval_visual_aligning_dpcc.py:~185` via `_DIM = {'x': 6, 'y': 7, ...}`.
So it is a **2-D cylinder on the end-effector position**, infinite in z.

> **It is explicitly a PLACEHOLDER.** The YAML says so three times
> (`# PLACEHOLDER: tune to actual obstacle position`). And critically —

### 1.2 There is no physical obstacle in the scene
`d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/objects/aligning_objects.py`:

```python
def get_obj_list():
    push_box   = PushObject(... "aligning_box" ...)
    target_box = PushObject(... "target_box" ...)
    obj_list = [push_box, target_box]
    return obj_list                 # ← returns here

    obj_list += [ Sphere(...), ... ]  # ← DEAD CODE, never reached
```

The scene contains **exactly two objects**: the push box and the target box. The obstacle at `(0.50, 0.00)` is a
**purely virtual constraint with no physical counterpart**. Nothing in the simulator stops the EE from going there;
only the projector objects.

### 1.3 Initial conditions
`aligning_objects.py:13` and `aligning.py:56–68` (`BlockContextManager`):

| Quantity | Value |
|---|---|
| EE init (`init_end_eff_pos`) | `[0.525, -0.35, 0.25]` — **fixed**, not sampled |
| Box init `box_space` | x ∈ [0.40, 0.60], y ∈ **[-0.25, -0.10]**, θ ∈ [-90°, +90°] |
| Target `target_space` | x ∈ [0.40, 0.60], y ∈ **[+0.20, +0.35]**, θ ∈ [-90°, +90°] |
| Box geometry (`robot_push_box.xml`) | base geom `size="0.05 0.05 0.01"` → **0.10 × 0.10 m footprint**, half-side a = 0.05 |

Box support radius as a function of yaw θ: `a(|cos θ| + |sin θ|)` ∈ [0.05, **0.0707**] (max at 45°, which is inside
the sampled ±90° range).

### 1.4 Action bound (the escape budget)
`action_bounds: 'auto'` → self-derived from `act_normalizer`. From the run log, line 65:

```
[ eval ] act_normalizer  mins=[-0.0083 -0.0083 -0.0083]  maxs=[0.0083 0.0083 0.0134]
```

With `dt = 1.0` (not set in YAML → `config.get('dt', 1.0)`; the YAML explicitly documents that dt=1 is correct
because actions are position deltas), the dynamics constraint is `c_pos[t+1] = c_pos[t] + 1.0 · act[t]`.

> **Per-step EE displacement is capped at |Δx|, |Δy| ≤ 0.0083 m.**
> Confirmed empirically by the log's own diagnostic: `denorm|a0|=8.33e-03 m`.

Horizon **H = 8**.

---

## 2. The init-overlap question, answered precisely

### 2.1 EE start vs obstacle — never a problem
```
EE init      = (0.525, -0.350)
obstacle ctr = (0.500,  0.000)
distance     = sqrt(0.025² + 0.350²) = 0.3509 m
```
vs r = 0.06 (nominal) / 0.09 (tightened). Clearance **0.29 / 0.26 m**. The EE init is fixed (not context-dependent),
so this is true for *every* context, always. **No init-time infeasibility is possible. No safeguard needed here.**

### 2.2 Box start vs obstacle — overlap IS possible, and IS irrelevant
Closest the box *centre* can be sampled to the obstacle centre: `(0.50, -0.10)` → **0.10 m**.

| | centre clearance | + box support radius (worst θ=45°: 0.0707) | physical overlap? |
|---|---|---|---|
| nominal r=0.06 | 0.10 − 0.06 = 0.040 m | 0.0707 + 0.06 = 0.1307 > 0.10 | **yes, up to 0.031 m** |
| tightened r=0.09 | 0.10 − 0.09 = 0.010 m | 0.0707 + 0.09 = 0.1607 > 0.10 | **yes, up to 0.061 m** |

So your intuition is geometrically correct: **the start box can physically intersect the obstacle cylinder.**

**But it does not cause any projection failure**, because of how the constraint is built
(`diffuser_visual_aligning/sampling/projection.py`, `ObstacleConstraints.build_matrices`):

```python
P[dim, dim] = delta_s**2 / 4      # dim ∈ {6, 7} only
```

`P` and `q` are non-zero **only on trajectory dims 6 and 7 = EE `c_pos`**. The box pose is not in the 9-D
trajectory at all (layout: `[dx dy dz | des_x des_y des_z | x y z]`). The projector has **no knowledge of the box**.
A box sitting inside the virtual cylinder violates nothing and is never evaluated.

> **Conclusion for the literal question asked: "start box overlapping the obstacle" is already harmless —
> not because it is impossible (it isn't), but because the obstacle is an EE-only constraint.**

### 2.3 Also: t=0 is exempt anyway
`projection.py:184`:
```python
for t in range(1, self.horizon):        # ← obstacle applied to t = 1 … H-1, NOT t = 0
```
The initial state is pinned by the dynamics anchor (`b[counter*horizon] = s_0[x_idx]`, `skip_initial_state=True`)
and is deliberately excluded from the obstacle constraint. So even an EE *starting* inside the sphere is not an
immediate contradiction — the contradiction appears at t=1 (§3).

---

## 3. The real trap: one-step escape is impossible

This is the failure mode that actually deserves attention.

### 3.1 The feasibility condition
Let `p` = current EE xy (pinned at t=0), `c` = (0.50, 0.00), `d = ‖p − c‖`.
The t=1 obstacle constraint requires `‖p + Δ − c‖ ≥ r`, where by dynamics + action bounds
`Δ = act[0]` with `|Δx|, |Δy| ≤ 0.0083`.

Best escape is radially outward, `u = (p−c)/d`. Max radial gain over the action box is
`0.0083·(|u_x| + |u_y|)`, which ranges from **0.0083** (axis-aligned) to **0.0117** (45°).

> **Feasible at t=1  ⟺  d + 0.0083·(|u_x|+|u_y|) ≥ r**

| | guaranteed INFEASIBLE if d < | possibly infeasible | always feasible if d ≥ |
|---|---|---|---|
| nominal r = 0.06 | **0.0483 m** | 0.0483–0.0517 | 0.0517 m |
| tightened r = 0.09 | **0.0783 m** | 0.0783–0.0817 | 0.0817 m |

### 3.2 How big is the trap
Fraction of the obstacle disc that is an inescapable pocket:

- nominal:   (0.0483 / 0.06)² = **65 %** of the disc area
- tightened: (0.0783 / 0.09)² = **76 %** of the disc area

If the EE is anywhere in that core, **the SLSQP feasible set is provably empty**. Not slow — *empty*. No amount of
iterating helps; SLSQP grinds to `maxiter=1000` and returns whatever it has.

Escaping from the very centre needs `0.06 / 0.0083 ≈ 7.2` steps, but the constraint demands it in **one**.
Crossing the disc through the centre takes `0.12 / 0.0083 ≈ 15` EE steps, of which ~11.6 are inside the trap core.

### 3.3 Why the EE will eventually be driven into it
This is the part that makes it more than academic:

```
box start    y ∈ [-0.25, -0.10]
OBSTACLE     y ∈ [-0.06, +0.06]  at x ∈ [0.44, 0.56]
target       y ∈ [+0.20, +0.35]
box x-range  [0.40, 0.60]   ← obstacle x-range [0.44,0.56] is dead-centre in it
```

The task **requires** transporting the box from y ≈ −0.175 to y ≈ +0.275 — i.e. straight through y = 0 — and the EE
must stay behind the box to push it. The obstacle is placed **exactly on the centreline of the only corridor the
task uses**. A successful push at x ≈ 0.5 drives the EE through the trap core by construction.

There is a detour: the corridor x ∈ [0.40, 0.60] is wider than the obstacle x ∈ [0.44, 0.56], so passing at
x < 0.44 or x > 0.56 is feasible (tightened: x < 0.41 or x > 0.59 — a 1–2 cm slot). But nothing *teaches* the
policy that; the obstacle is invisible to the model (it's a projector-only constraint, and there is no physical
object in the scene to see — §1.2). The FM/diffusion prior will happily aim straight down the middle.

> **Therefore: the trap is only reachable on a *successful* rollout. The better the policy gets, the more often
> the projection becomes provably infeasible.** That is a nasty failure ordering.

---

## 4. What the safeguards actually cover

### 4.1 `Fix_15.2` catches the symptom, not the cause
`projection.py:8–36, 242–301` gives two layers:
1. per-solve wall-clock **backstop** (`FMPCC_PROJ_SOLVE_BACKSTOP_S = 60 s`) via the SLSQP `callback`;
2. a sliding-window **circuit breaker** (40 calls, ≥90 % slower than 1000 ms → OPEN, skip 40, half-open probe).

Both are **time-based**. An infeasible solve is detected only if it happens to be *slow*. Infeasibility and
slowness correlate but are not the same thing — and an infeasible-but-fast solve slips through completely.

### 4.2 **`res.success` is never checked** ← genuine gap
```python
res = minimize(..., method='SLSQP', options={'maxiter': 1000, 'disp': False})
...
sol_np[i] = res.x                                  # accepted unconditionally
projection_costs[i] = 0.5 * sol_np[i] @ Q @ ... # finite cost
```
`res.success` / `res.status` appear **only inside a commented-out debug block** (line ~277). In live code:

- an infeasible or `maxiter`-terminated solve returns its last iterate `res.x`, which may violate the obstacle,
  the halfspace, the workspace box and the dynamics arbitrarily;
- it is written back as if it were a valid projection;
- it receives a **finite** cost — so `dpcc-c` (`minimum_projection_cost` trajectory selection) can actively
  **prefer** a failed solve if its garbage iterate happens to score low.

Only the 60 s backstop path sets `np.inf` cost. Everything else is trusted blindly. This is the one place where a
cheap, strictly-correct fix exists.

### 4.3 The breaker window is measured in the wrong unit
The `Fix_15.2` header comment states the window counts *"the last WINDOW project() calls (= replan steps)"*.
**That equivalence is false in this pipeline.** `diffuser_visual_aligning/models/diffusion.py:185–191`:

```python
if projector is not None and not projector.gradient and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
    x, projection_costs = projector.project(x, constraints)
```

`project()` is called **at every denoising timestep below the threshold** — with `steps=400` and
`diffusion_timestep_threshold=0.5`, that is **~200 `project()` calls per replan**, not one.

Consequences:
- `WINDOW=40` fills in ~20 % of a *single* replan's denoising loop — the breaker judges "sustained slowness" over a
  fraction of one replan, not over an episode as designed;
- `COOLDOWN=40` skips happen **inside** the denoising loop, meaning ~40 consecutive denoising steps silently lose
  their DPCC guidance mid-denoise, then projection resumes — a *partially guided* sample, which is a different and
  undocumented object from both `diffuser` and `dpcc-r`;
- the "one slow solve among 400 steps costs nothing" reasoning in the comment is calibrated to the wrong timescale.

This is not wrong-in-effect (it still stops runaway wall-clock), but the documented semantics do not match reality
and the constants were tuned against the wrong denominator.

---

## 5. What actually happened in run 23514 (obstacle is exonerated)

Cross-checking the log against the theory:

```
[ constraints ] sat=0.115  violated=354steps  (bounds=273 hs=354 obs=0)
[ constraints ] sat=0.290  violated=284steps  (bounds=241 hs=284 obs=0)
[ constraints ] sat=0.210  violated=316steps  (bounds=253 hs=316 obs=0)
...
```

**`obs=0` in every single rollout.** The obstacle was never violated, because the EE never went near it — and the
EE never went near it because the box never moved:

```
- Box  init  XY=(0.521, -0.145)  angle=-51.6°
- Box  final XY=(0.521, -0.145)  angle=-51.6°   ← identical
- Success status: False
```

The breaker trips instead coincide with the EE running away in **+x** at saturated action magnitude:

```
[ DIAG replan=300 ] norm|a0|=5.651  denorm|a0|=1.44e-02 m  dir=[ 0.885  0.381 -0.268]
[ DIAG replan=400 ] norm|a0|=6.616  denorm|a0|=1.44e-02 m  dir=[ 0.756  0.529 -0.385]
```

Driving +x pushes the EE across the `combined_5` halfspace, whose line through `(0.65,0.45)–(0.80,−0.45)` is
`6x + y ≤ 4.35` (slope m = −6, `'below'` ⇒ `C_row = [−m, 1] = [6, 1]`, `d = 4.35`). Sanity check of the init state:
`6(0.525) + (−0.35) = 2.80 ≤ 4.35` ✓ — feasible with a large margin, as expected. It is the *runaway*, not the
geometry, that breaks it (`hs` violations 112–357 steps, `bounds` 224–278 steps).

> **So: the 4 circuit-breaker trips in job 23514 are a halfspace/action-bound thrash caused by a diverging policy,
> not the obstacle. The obstacle trap analysed in §3 is latent and was never exercised.**

---

## 6. Recommendations

Ordered by value-to-effort. **Nothing here has been changed — proposal only.**

### R1 — Move or delete the placeholder obstacle *(highest value, zero code)*
The obstacle is a documented PLACEHOLDER, has **no physical object in the scene** (§1.2), and is positioned
`(0.50, 0.00)` — the exact centreline of the only corridor the task uses (§3.3). As configured it does not model
anything real; it just makes the task's required path provably un-projectable over 65–76 % of its interior.

Either
- **delete it** from `combined_5` (keeping dynamics + geo_bounds + halfspace + bounds), or
- **move it off-corridor**, e.g. centre `(0.30, 0.35)` or `(0.65, −0.30)` — outside both `box_space` and
  `target_space` and away from the y = 0 crossing — so it constrains without being an unavoidable trap.

If an obstacle *on* the corridor is deliberately wanted as a hard test, at minimum shrink it so the detour slot is
usable and document that the tightened twin (r = 0.09) leaves only a ~1 cm gap.

### R2 — Check `res.success` *(small, strictly correct, independent of R1)*
Treat a failed/`maxiter` solve like the backstop path already does:

```python
if not res.success:
    sol_np[i] = trajectory_np_double[i]      # keep unprojected rather than a garbage iterate
    projection_costs[i] = np.inf             # keep dpcc-c from selecting it
    # count + log once per N, e.g. res.status 4 == "inequality constraints incompatible"
    continue
sol_np[i] = res.x
```

This closes §4.2 and makes infeasibility **observable** (`res.status == 4` is exactly "infeasible") instead of
being laundered into a plausible-looking trajectory. It also gives a real signal to distinguish "obstacle trap"
from "policy runaway" in future logs — which is precisely the ambiguity that made this investigation necessary.

### R3 — If the corridor obstacle is kept: relax the t=1 obstacle constraint when already inside
The mathematically honest fix for §3.1 is to not demand a one-step exit. When `d < r` at t=0, replace the hard
constraint for the first `k = ceil((r − d)/0.0083)` steps with a **monotone-recession** requirement
(`‖p_{t+1} − c‖ ≥ ‖p_t − c‖ + ε`), restoring the hard constraint from step k onward. This keeps the feasible set
non-empty, still drives the EE out, and removes the "provably infeasible" regime entirely.

### R4 — Fix the `Fix_15.2` window semantics / comment
Either re-tune `FMPCC_PROJ_CB_WINDOW`/`COOLDOWN` to the ~200-calls-per-replan reality (§4.3), or move the breaker
bookkeeping up to the replan level, or — cheapest — correct the comment so the next reader is not misled by
"(= replan steps)". The current constants were justified against a denominator that does not exist here.

---

## 7. Summary answer to the original question

> *"if start box is overlapping the box theoretically, if there is already impossible or already safeguarded no
> need to do anything, if not maybe adjust something or safeguard. since if obstacle interferes the start box init
> position, the projection method 100 % fail"*

1. **Start-box / obstacle overlap: possible, but it cannot cause projection failure.** The obstacle constrains the
   **EE only** (dims 6, 7); the box is not in the trajectory vector. → *no action needed for the literal concern.*
2. **EE init / obstacle overlap: impossible.** Fixed init at `(0.525, −0.35)`, 0.351 m away, every context.
   Additionally t=0 is exempt from the obstacle constraint by construction. → *no action needed.*
3. **A genuine 100 %-infeasible regime does exist** — but mid-episode, when the EE is deeper than ~0.048 m
   (nominal) inside the cylinder, where one-step escape under `|Δ| ≤ 0.0083 m` is arithmetically impossible.
   65–76 % of the obstacle interior is such a pocket, and the obstacle is placed on the task's mandatory path.
   → *this is worth fixing (R1, optionally R3).*
4. **It is not currently safeguarded at the right level.** `Fix_15.2` is a wall-clock guard on the symptom, and
   `res.success` is never inspected, so infeasible solves are silently accepted with finite cost.
   → *R2 is worth doing regardless of what happens to the obstacle.*
5. **It did not cause the run-23514 breaker trips** (`obs=0` everywhere, box never moved); those were halfspace +
   action-bound thrash from a diverging policy. → *the obstacle is exonerated for this run, but stays armed.*

---

### Files inspected
- `diffuser_visual_aligning/sampling/projection.py` (`Projector.project`, `ObstacleConstraints.build_matrices`)
- `diffuser_visual_aligning/models/diffusion.py:170–200` (projection call site / frequency)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (`setup_dpcc_projector`)
- `config/visual_aligning_eval.yaml` (`combined_5`, `enlarge_constraints`, `action_bounds`)
- `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py` (`BlockContextManager`)
- `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/objects/aligning_objects.py`
- `d3il/environments/d3il/models/mj/common-objects/robot_push_box/robot_push_box.xml`
- `temp/22_57_36_eval_visual_aligning_dpcc_23514.log`

### Verified side-note: obstacle normalisation math is correct
`ObstacleConstraints.build_matrices` maps `(s−c)ᵀ(s−c) ≥ r²` into normalised coords via
`s = (sₙ+1)(Δs)/2 + s_min`. Expanding gives `P=Δs²/4`, `q=Δs²/2 + Δs(s_min−c)`,
`v −= Δs²/4 + Δs(s_min−c) + (s_min−c)²`, with all terms negated for `sphere_outside` — which is **exactly** what
the code writes. No bug there; the problem in §3 is geometric placement, not the algebra.
