# Does HardFlow degenerate at K=1/2? — yes, and it is provable from the update rule

**Date:** 2026-08-16 · **Type:** code+math study (no run) · **Scope:** `hardflow_new` as implemented in FM-PCC
**Sources read:**
`/workspaces/aux_repo/HardFlow/hardflow/models_flow/flow_policy.py` (upstream, `hardflow_new_forward` :1286-1435, `hardflow_formulate` :683-751) ·
`/workspaces/aux_repo/HardFlow_Paper_Files/arXiv-2511.08425v3/main.tex` (Alg. 1 :534-556, Prop. 1 proof :939-951, Thm. 4 :698-706, Rmk. 9 :708-711, exp. setup :1242) ·
`mix_uav/sampling/hardflow_projection.py` (Gen15 port — the one the K-sweep runs) ·
`mix_uav/models/mf_diffusion.py:190-330`, `mix_uav/models/af_diffusion.py:246-345` (the arms' own samplers) ·
`mix_uav/sampling/projection.py` (arm B's DPCC projector) ·
`config/uav_mix.py:204-215`, `mix_uav_test/eval_mix_uav.py:1361-1402`, `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh:30` ·
**§8 additionally reads** `logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`,
`…/DA_20260811_MF_UNet32_full5seeds_avoiding.md`,
`Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md`

**Companions:** `H2H_iMF_vs_HardFlow_stepwise.md` · `MAP_Algorithm1_to_AvoidingCode.md`

---

## 0. Bottom line

**Yes. At K=1 the HardFlow math is gone entirely, and at K=2 it is gone too under the
threshold we actually ship (`activation_threshold: 0.5`).** What remains is exactly
*sample-then-project* — the very baseline HardFlow is supposed to beat.

Three separate collapses stack at low K, and they are independent:

| # | Collapse | Where it comes from | Affects |
|---|---|---|---|
| **D1** | The per-step update reduces to an exact Euclidean projection of the endpoint, with no feedback | HardFlow's own algorithm (universal, upstream too) | every engine, K=1 always; K=2 at `A≤0.5` |
| **D2** | The τ² prox weight and `reg_scale` are **already** no-ops in our port at *every* K | our port drops the cost term `C(·)` | every engine, every K |
| **D3** | The `mf`/`af` arms lose the MeanFlow interval field — arm C integrates `h=0` while arms A/B integrate `h=dt` | our port's `two_time` velocity query | `mf`/`af` only; harmless at large K, catastrophic at K=1/2 |

D1 answers the question as asked. D2 makes the collapse total rather than partial. D3
means the low-K HardFlow rows on `mf`/`af` are **confounded** — arm C is not sampling the
same base trajectory as arms A/B, so a loss there is not evidence about HardFlow.

The K sweep ships `K=[1 2 5 10 20]` (`eval_k_sweep.sh:30`), so this is live, not
hypothetical. **The `hardflow_new*` rows at K=1 and K=2 should not be read as HardFlow
results.**

**§8 checks this against the `avoiding-d3il` DA corpus we already have,** and the answer is
sharper than expected: HardFlow's *only* win anywhere in that corpus is at K=1 — zero genuine
HardFlow steps — and its gate score falls monotonically as genuine steps are added
(1.000 → 0.933 → 0.933 → 0.833 for K = 1/2/5/10). §8 also resolves the "K=2 HardFlow was
better" recollection (it was not — §8.1), corrects a stated failure mechanism in the
2026-08-02 DA that cannot apply to the runs it explains (§8.3), and shows why the roadmap's
MeanFlow-exact-endpoint fix is a no-op at the operating point it was proposed for (§8.4).

---

## 0.1 The breakpoint — exactly which K, and why

A step does real HardFlow work only if it is **active** *and* **not the last step**. Count them:

```
n_genuine(K, A) = max(0, K − 1 − floor((1 − A)·K))          A = activation_threshold
```

`n_genuine = 0` ⇒ the run is a plain Euler sample followed by one projection.

| `A` | dead at | alive from | why the cutoff sits there |
|---|---|---|---|
| **0.0** | **every K** | never | gate never fires; only the forced terminal solve runs |
| **0.5** *(shipped)* | **K ≤ 2** | **K = 3** | `floor(0.5·2) = 1` deactivates step 0, leaving only the terminal solve |
| **1.0** | **K = 1** | **K = 2** | every step is active, so only the last-step collapse remains |
| any | **K = 1** | — | the only step *is* the last step |

**The why, in three lines** — all of it is the last step, where `τ⁺ = 1` exactly:

1. lookahead `(1 − τ⁺)·f(X_ref, τ⁺)` = **0** → the "predicted endpoint" is just the Euler point;
2. pull-back gain `τ⁺` = **1** → full snap onto the feasible set, not a damped nudge;
3. there is no step `k+1` → the network never sees the correction, so it cannot repair it.

All three HardFlow mechanisms are *defined* by `τ⁺ < 1` plus a successor step. Remove those and
what executes is `Π_S(Euler sample)` — the sample-then-project baseline.

**Alive ≠ working.** The first genuine step carries lookahead `1 − τ⁺`, and the paper's error bound
grows with it (Thm. 4, §7). At `K=2, A=1.0` that single step runs at `τ⁺ = 0.5` — the largest
lookahead any non-terminal step can have. Practical floor for a meaningful HardFlow run:
**K ≥ 5 with A ≥ 0.5** (2 genuine steps at `τ⁺ ∈ {0.6, 0.8}`), matching the paper's own N=10 / A=0.5.

## 0.2 Degenerates to *what* — DPCC, not "no projection"

**It becomes DPCC.** The projection still runs, the constraint set is still hard-enforced, and
the safety guarantee still holds — that terminal solve *is* Proposition 1 (§2). Nothing is
skipped; what is lost is the guidance, not the enforcement.

| | unprojected (`diffuser`) | **degenerate HardFlow** | DPCC (arm B) |
|---|---|---|---|
| projections per plan | 0 | **1, at the end** | 1, at the end (K=1: same gate) |
| constraints enforced | ❌ | ✅ hard | ✅ hard |
| what it computes | Euler sample | **`Π_S(Euler sample)`** | `Π_S(Euler sample)` |
| S&C on `avoiding` | 0.000–0.267 | 0.933–1.000 | 0.933–1.000 |

On the UAV/Gen15 wiring the two are the *same problem*, deliberately:

* **same constraint set** — the HardFlow NLP is built from the identical `constraint_list` the
  DPCC `Projector` consumes (`eval_mix_uav.py:840-844`, "if the two arms enforced different
  constraint sets the comparison would be void");
* **same variables** — `variant='states_actions'` on the DPCC side (`:854`), actions+states on
  the HardFlow side; step 0 pinned either way (`skip_initial_state` vs `s0` as an NLP parameter);
* **same objective** — DPCC's `Q = I` (`projection.py:74`) and HardFlow's `0.5·λ·τ⁺²·‖·‖²` have
  the same argmin (§5, D2);
* **same timing at K=1** — both gates fire only on the final step.

**What still differs, and it is not nothing:**

1. **Solver.** IPOPT/CasADi vs scipy SLSQP (`solver='scipy'`, `:858`).
2. **Failure fallback.** HardFlow keeps the last IPOPT iterate on a failed solve
   (`hardflow_projection.py:339-346`), which is *not* guaranteed feasible; DPCC has the
   sustained-slowness circuit breaker instead (`projection.py:9-31`).
3. **The sample being projected**, on `mf`/`af` only — arm C integrates `h=0`, arms A/B `h=dt`
   (§5, D3). At K=1 that is a completely different trajectory.

Empirically the residual is real, so "degenerate ⇒ identical to arm B" is true of the *math*,
not of the *numbers*: K=1 gives HF-tight 1.000 vs `dpcc-t-tight` 0.967 (§8.2), and the naive-FM
terminal-only ladder gives HF 2/6 vs DPCC 6/6 at K=2 (§8.3). Same problem, different solver —
which makes those gaps a solver finding, not an algorithm finding.

## 0.3 Strictly the same as DPCC? No — and it could not be

Short answer: **the same *kind* of problem, not the same problem, and even if it were, the answer
would not be unique.** Do not expect the two arms to agree at K=1, and do not read their gap as
solver noise around a single right answer.

**What genuinely matches** — both minimise `‖x − x_ref‖²` in **normalized** coordinates
(`Q = I` over `H·T`, `projection.py:74`; `cs.sumsqr(x1 − x1_ref)` on the normalized dof,
`hardflow_projection.py:182`), both build from the one shared `constraint_list`, both apply
obstacle/ineq/eq rows over `t = 1 … H−1`, and both fire only on the last step at K=1.

**Three differences that are mathematical, not numerical:**

| # | DPCC | HardFlow NLP | consequence |
|---|---|---|---|
| 1 | `s_0` is a **decision variable**; only the `deriv` `x_idx` dims of step 0 are pinned, via `b[counter·H] = s_0[x_idx]` (`projection.py:149-157`) | `s_0` is an **NLP parameter** — every dim hard-pinned (`:174`, `_transition_expr` :210) | DPCC may move parts of step 0, HardFlow cannot. Different feasible sets. |
| 2 | box `Bounds(−5, +5)` on **every** normalized variable (`:196`) | no such box | a constraint HardFlow simply does not have |
| 3 | sphere as `sᵀPs + qᵀs ≤ v` with `P, q` **pre-substituted into normalized coords** (`ObstacleConstraints.build_matrices` :529-539) | sphere as `(x_u−c_x)² + (y_u−c_y)² ≥ r²` on **unnormalized** expressions (`:257-262`) | algebraically the same set; *that* it is identical is an assumption no gate checks |

**And the decisive one: the problem is nonconvex.** `sphere_outside` makes the feasible set
non-convex, so "the projection onto `S`" has **no unique solution**. Two solvers started from the
same point can converge to different local minima and both be correct. Strict equality is not just
unmet — it is not a coherent expectation. (Also: SLSQP with analytic Jacobians vs IPOPT with
limited-memory BFGS and `solve_limited`, plus two different failure paths — HardFlow returns the
last IPOPT iterate, which may be **infeasible** (`:339-346`); DPCC's breaker returns the trajectory
**unprojected** (`:127-133`). Both are silent.)

**So how should the K=1 gaps be read?** `1.000 vs 0.967` (§8.2) and `2/6 vs 6/6` (§8.3) are
differences between two non-identical projections of a nonconvex problem. They say nothing about
HardFlow-the-algorithm — but they are not noise either, and the 2/6-vs-6/6 one is large enough that
something is genuinely wrong on one side.

**The gate that would settle it** (offline, no rollout, cheap — does not exist yet): take one fixed
`X_ref`, run both projectors on it, and report

* `‖Π_HF(X_ref) − Π_DPCC(X_ref)‖` — how far apart the two answers are;
* max constraint residual of **each** output — is either one actually infeasible?
* `nlp_failures` / `last_proj_skipped` — did either silently fall back?

If both are feasible and merely land on different local minima, the gap is a legitimate
solver-choice effect. If one is infeasible, that is a bug, and §8.3's 2/6 stops being mysterious.

---

## 1. The update rule, and its four moving parts

Per ODE step `k = 0 … K-1`, with `dt = 1/K`, `τ_k = k·dt`, `τ⁺ = τ_k + dt`
(`mix_uav/sampling/hardflow_projection.py:517-556`):

```
(1) reference Euler step   X_ref  = X_k + f(X_k, τ_k)·dt                       :521-522
(2) terminal prediction    X1_ref = X_ref + (1 − τ⁺)·f(X_ref, τ⁺)              :539-540
(3) prox-NLP               X1*    = argmin ½·λ·τ⁺²·‖X1 − X1_ref‖²  s.t. h(X1) ≤ 0   :181-183, 548
(4) pull-back              X_k+1  = X_ref + τ⁺·(X1* − X1_ref)                  :553
```

This is byte-for-byte the upstream loop (`flow_policy.py:1321-1371`) and, for the linear
scheduler `α_t = t, β_t = 1−t`, algebraically identical to paper Algorithm 1 — I checked
the substitution: the paper's `x_{i+1} = α·x̂_N* + β·(−α̇·x̄ + α·v)/(αβ̇ − α̇β)` expands to
`τ⁺·x̂_N* + (1−τ⁺)·(x̄ − τ⁺·v)`, which equals line (4) exactly.

The four things that make this *HardFlow* rather than *projection*:

| Ingredient | Mechanism | Controlled by |
|---|---|---|
| **I1. endpoint-space** | constraints are imposed on the *predicted terminal* `X1`, not on the noisy iterate | the `(1 − τ⁺)` lookahead in (2) |
| **I2. damped pull-back** | the terminal correction is applied to the iterate scaled by `τ⁺ < 1` — a nudge, not a snap | the `τ⁺` factor in (4) |
| **I3. feedback** | the corrected iterate is re-fed to the network, which pulls it back onto the data manifold | there being a step `k+1` at all |
| **I4. cost/constraint trade-off** | `C(x̂_N)` competes with the τ²-weighted prox term | the `C(·)` term in (3) |

Everything below is about which of I1–I4 survive at a given K.

---

## 2. The terminal step is *always* a plain projection — by design

At `k = K-1`: `τ⁺ = 1.0` **exactly** (verified in float for K ∈ {1,2,5,10,20}: `1.0 − τ⁺ == 0.0`,
no rounding residue). So:

* (2) becomes `X1_ref = X_ref + 0·f(X_ref, 1.0) = X_ref` → **I1 dead**, and the network call
  is a wasted NFE (§6).
* (4) becomes `X_k+1 = X_ref + 1·(X1* − X_ref) = X1*` → **I2 dead**, full snap.
* there is no step `k+1` → **I3 dead**.

This is not a bug. It is precisely the paper's safety proposition: *"for i = N−1 … α₁ = 1 and
β₁ = 0, therefore x_N = x̂_N*, consequently h(x_N) ≤ 0"* (main.tex:939-951). **HardFlow's hard
guarantee lives entirely in the terminal step, and the terminal step is a projection.**

The corollary is the whole answer to the question: *HardFlow-ness is what happens in the
`K−1` steps **before** the last one.* Take those away and only the projection remains.

---

## 3. K=1 — total degeneracy, no conditions attached

`K=1`, `dt=1.0`, one iteration `k=0`, which is simultaneously the first and the terminal step:

```
τ_0 = 0.0,  τ⁺ = 1.0
X_ref  = X_0 + f(X_0, 0)·1.0                      one Euler step across the whole interval
X1_ref = X_ref + (1−1)·f(X_ref, 1.0) = X_ref      I1 dead
X1*    = argmin ‖X1 − X_ref‖²  s.t. h(X1) ≤ 0     Euclidean projection Π_S(X_ref)
X_1    = X_ref + 1.0·(X1* − X_ref) = X1*          I2 dead
                                                   I3 dead (no next step)
```

**`K=1` HardFlow ≡ `X_1 = Π_S( X_0 + f(X_0,0) )`.** One unguided sample, one projection at
the end. Zero HardFlow-specific arithmetic executes.

Consequences worth stating explicitly:

* `activation_threshold` is **inert** at K=1. `int((1−A)·1)` is `0` for every `A > 0`, and
  `k == K-1` forces the solve regardless (`:537`). All of `all` / `late` / `0.0` / `1.0`
  produce bit-identical output.
* `reg_scale` and the τ² weight are inert (see §4 — they are inert at every K in our port,
  but here even upstream's version would be, since τ⁺=1 ⇒ τ⁺²=1).
* Compared with **arm B at K=1**: arm B's DPCC gate also fires at the single step
  (`mf_diffusion.py:288-289`, `loop_idx == flow_steps-1`), and its objective is
  `Q = I` (`projection.py:74`), i.e. also Euclidean. So at K=1 arms B and C run *the same
  algorithm*, differing only in (i) the solver — SLSQP/proxsuite vs IPOPT — (ii) which
  variables are in scope (DPCC's `variant='states'` vs HardFlow's full action+state dof),
  and (iii) for `mf`/`af`, the field (§5). Any K=1 B-vs-C gap measures those three things,
  **not** HardFlow.

---

## 4. K=2 — degenerate at the shipped default, one surviving step at `A=1.0`

The activation gate (`:537`, floor rounding from Gen12 fix_8):

```python
active = (k >= int((1.0 - self.activation_threshold) * K)) or (k == K - 1)
```

Ship default is `activation_threshold: 0.5` (`config/uav_mix.py:214`, and the eval falls
back to `diffusion_timestep_threshold` = 0.5 if absent, `eval_mix_uav.py:1385-1386`).

**K=2, A=0.5:** `int((1−0.5)·2) = int(1.0) = 1`. Step `k=0` is **not** active. Step `k=1` is
the forced terminal step. Result:

```
k=0:  X_1 = X_0 + f(X_0, 0.0)·0.5          plain Euler, no NLP
k=1:  X_2 = Π_S( X_1 + f(X_1, 0.5)·0.5 )   terminal projection (§2)
```

→ **two-step Euler, then project.** Identical in structure to K=1. Still zero HardFlow math.

**K=2, A=1.0 (`'all'`):** `int(0)=0`, so `k=0` activates and one genuine step survives:

```
k=0:  τ⁺ = 0.5
      X1_ref = X_ref + 0.5·f(X_ref, 0.5)       I1 alive, but lookahead is a full 0.5
      X_1    = X_ref + 0.5·(X1* − X1_ref)      I2 alive at gain 0.5
k=1:  terminal projection                       (§2)
```

One HardFlow step exists — and it sits at the *worst* point of the paper's own error bound.
Theorem 4 (main.tex:698-706) bounds the surrogate gap by a factor
`r(2−r)/(1−r)²` with `r = |β_{τ⁺}|·L_W = (1−τ⁺)·L_W`. At τ⁺=0.5 this is the largest `r` any
non-terminal step can have for that K, and Remark 9 (:708-711) is an explicit instruction to
*avoid* exactly this regime ("skip control in the early stages … activate the subproblems in
later steps"). Setting `A=1.0` at K=2 to "get HardFlow back" buys one step of the least
trustworthy kind.

### 4.1 Genuine-HardFlow-step count

Define `n_active = K − floor((1−A)·K)` (≥1), and a **genuine** step as one that is active
*and* not the terminal step, i.e. `n_genuine = n_active − 1`.

| K | A=0.5 (shipped) `n_active` / `n_genuine` | genuine τ⁺ values | A=1.0 `n_active` / `n_genuine` |
|---|---|---|---|
| **1** | 1 / **0** | — | 1 / **0** |
| **2** | 1 / **0** | — | 2 / 1 (τ⁺=0.5) |
| 5 | 3 / 2 | 0.6, 0.8 | 5 / 4 |
| 10 | 5 / 4 | 0.6, 0.7, 0.8, 0.9 | 10 / 9 |
| 20 | 10 / 9 | 0.55 … 0.95 | 20 / 19 |

Fraction of the ODE that is genuinely HardFlow-guided is `n_genuine/K`: **0 at K=1 and K=2**,
0.4 at K=5, 0.4 at K=10, 0.45 at K=20. K=1 and K=2 are not "weakly HardFlow"; they are a
different algorithm.

---

## 5. Two degeneracies that are *ours*, not the paper's

### D2 — we dropped `C(·)`, so the τ² schedule was already a no-op

Upstream supports two objectives (`hardflow_formulate` :732-745): `""` (prox only) and
`"distance"` (prox **+** `−distance_objective·hardflow_cost_scale`). The paper always runs
with a cost (`C(x)` = squared distance of `s_{H−1}` to the target, main.tex:1242). **Our port
implements only the prox term** — `HardFlowNLP` builds `self.cost` at :181-183 and calls
`self.opti.minimize(self.cost)` at :187, with no `C` anywhere.

For a pure quadratic prox with no competing cost, `argmin c·‖X1 − X1_ref‖² s.t. h ≤ 0` is
independent of `c > 0`. Therefore:

> **In FM-PCC, `reg_scale` and the `τ⁺²` weight do not change the solution at any K.** The
> NLP is exactly `Π_S(X1_ref)`, the Euclidean projection of the predicted endpoint, at every
> active step. **I4 does not exist in our port.**

(Analytically, that is. IPOPT's internal scaling and `solve_limited`'s iteration cap can see
the scale, so it is not *bit*-identical — but it is not the mechanism the docstring at
:179-181, "early steps are nudged and late steps are pulled hard", advertises. The τ-schedule
that actually damps early steps in our port is the **pull-back factor `τ⁺` in (4)**, not the
cost weight.)

**Known and already pinned, not a new discovery.** `FM_v3_hardflow_test/gates_hardflow.py::gate_g2`
(:228-240) asserts exactly this — *"With a PURE proximal objective, multiplying the cost by tau^2
does not move the argmin at all … The actual tau-gating of the trajectory is the LINEAR factor in
the pull-back … it is kept in the code but must not be described as the schedule"* — and
`config/meanflow_projection_eval.yaml:109-112` says the same of `reg_scale`. What is new here is
only its **consequence at low K**: with I4 absent, the terminal-only regime has nothing left to be
except a plain projection. (The module docstring at `hardflow_projection.py:21-22` still describes
the τ² factor as the schedule, contradicting the gate; worth correcting.)

This matters for the low-K question because it removes the one ingredient that could have
survived: with a cost term, even a terminal-only solve would still be doing HardFlow's
cost/constraint trade-off. Without it, terminal-only is *just projection*.

### D3 — `mf`/`af` arm C samples a different base trajectory than arms A/B

The engines' own samplers integrate the **interval-average** field:

```python
# mf_diffusion.py:216, 282-283   (af_diffusion.py:272, 340-341 identical)
h_batch = torch.full((batch_size,), dt, ...)          # h = dt = 1/K
velocity = self._predict_velocity(x, cond, t_i, h=h_batch, returns=returns)
x = x + velocity * dt
```

The HardFlow sampler deliberately queries `h = 0`:

```python
# hardflow_projection.py:452-454
if self.two_time:
    v = self.model._predict_velocity(traj, cond, t, h=torch.zeros_like(t), returns=returns)
```

The comment at :437-443 justifies this correctly — `u(x,t,0) = v(x,t)` exactly, so the field
handed to the NLP is the instantaneous one, matching the projection math. **At large K that
is right and the two integrators converge to the same ODE solution.** At small K they do not:

| K | arm A/B (`mf`) base sample | arm C (`mf`) base sample |
|---|---|---|
| 1 | `x₀ + 1.0·u(x₀, 0, 1)` — the trained one-step MeanFlow jump, exact by the MeanFlow identity | `x₀ + 1.0·v(x₀, 0)` — instantaneous velocity extrapolated across the whole unit interval |
| 2 | two exact half-interval jumps | two Euler steps of the instantaneous field |
| 10 | 10 jumps, `h=0.1` | 10 Euler steps, `O(dt)` truncation error |

At K=1 arm C throws away the entire reason MeanFlow exists. Its base trajectory is a
first-order Euler extrapolation of a curved flow over `Δt = 1`; the projection then has to
repair a sample that was already far off-manifold — and, per §3, that repair is the *only*
thing arm C does. So the low-K `mf`/`af` HardFlow rows lose for two stacked reasons, neither
of which is "HardFlow is worse than DPCC".

The `fm` arm has no instantaneous/average distinction, so **`fm` is the only clean engine for
studying D1 at low K**, and there arm C ≡ arm B modulo solver and variable scope.

---

## 6. NFE accounting: the budget-parity claim breaks at low K

`eval_k_sweep.sh:15-17` is emphatic — *"MATCHED BUDGET OR NOTHING … at equal K, which arm
holds success"*. But arm C spends `K + n_active` network evaluations, not `K`, because every
active step calls the network a second time at `X_ref` (`:539`).

| K | A | arm A/B NFE | arm C NFE | of which multiplied by `(1−τ⁺)=0` | arm C overhead |
|---|---|---|---|---|---|
| 1 | any | 1 | **2** | 1 | **+100 %** |
| 2 | 0.5 | 2 | 3 | 1 | +50 % |
| 5 | 0.5 | 5 | 8 | 1 | +60 % |
| 10 | 0.5 | 10 | 15 | 1 | +50 % |
| 20 | 0.5 | 20 | 30 | 1 | +50 % |

The terminal-step evaluation is **always** pure waste: `(1 − τ⁺)` is exactly `0.0`, so
`V_next` is computed, multiplied by zero, and discarded. At K=1 that is half of arm C's
entire network budget spent on a term that is mathematically zero — for an arm that, at K=1,
does no HardFlow at all.

Latent hazard, not observed: `0.0 * NaN = NaN` under IEEE. If the backbone ever returns a
non-finite value at `t = 1.0` (which is the closed edge of the training support — CFM samples
`t ~ U[0,1)`), the multiply-by-zero will not neutralise it; it will poison `X1_ref` and hence
the NLP warm start. Skipping the call when `τ⁺ ≥ 1` removes both the waste and the hazard.

---

## 7. What the upstream authors actually ran

* `N = 10` for **all** methods in manipulation, maze and PDE control; `N = 100` for image
  editing (main.tex:1242, :1254).
* HardFlow is run with the solve active **only in the second half** of steps (`:1242`), i.e.
  `A = 0.5`, which at N=10 leaves 4 genuine steps — the row in §4.1 that matches our K=10.
* Remark 9 (:708-711) recommends *skipping* early control because the terminal estimator and
  the one-step fixed-point truncation are unreliable near `t=0`.

Nothing in the paper is evaluated below N=10, and the algorithm's own error analysis argues
against the small-N regime rather than for it. There is no upstream claim that low-K HardFlow
should work, so a low-K failure in our sweep refutes nothing of theirs.

---

## 8. What our own DA already shows — and what the "K=2 HardFlow was better" memory actually was

**Sources for this section:**
`logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md` (§5.6, §11.3-11.7) ·
`…/DA_20260811_MF_UNet32_full5seeds_avoiding.md` (§4.3, §8, §9.2-9.4) ·
`Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md` (§2, §3, §4).
All rows are `avoiding-d3il`; every HardFlow run in the corpus used **`A = 0.5`, `hf_batch = 1`**
(5-seed DA §8 "No activation-threshold sweep — HardFlow ran `A0.5_B1` only").

### 8.1 Short answer: no, K=2 HardFlow was never better

Three things could be the memory. None of them is "K=2 HardFlow beat DPCC".

**(a) K=2 HardFlow reached the same gate at 3× the cost — and was recommended for deletion.**
2026-08-02 DA §5.6: *"the `hardflow_new-*-tightened` variants reach the same 1.000 as
`dpcc-*-tightened` but cost 0.066–0.080 s (3.3× the dpcc variants) … Recommend dropping hardflow
from the headline table."* §11.3 measures it per-episode, 30 episodes:

| generator | rule | DPCC | HardFlow | ratio |
|---|---|---|---|---|
| AlphaFlow K=2 | `r-tight` | 30/30, **1.36 s** | 30/30, 4.51 s | 3.31× |
| AlphaFlow K=2 | `t-tight` | 28/30, **1.61 s** | 30/30, 4.58 s | 2.84× |
| MeanFlow K=2 | `r-tight` | **29/30**, **1.77 s** | 27/30, 5.46 s | 3.08× |
| MeanFlow K=2 | `t-tight` | 29/30, **1.73 s** | 29/30, 5.60 s | 3.24× |

Equal on the gate (within ±2/30), 2.8–3.3× more expensive. **Equal, not better.**

**(b) The one row that *looks* like "K=2 HardFlow wins" is retracted.**
`AF bbsit K2 hardflow_new-tightened → 1.000 / 4.58 s/ep` (5-seed DA §4.4.1). That DA's §8 kills it:
the AF and `mf_dit` HardFlow folders predate Fix_9 (`808cb1a4`, 2026-08-07) and carry **no `A`/`B`
token in the path**, so seed 6 ran `hf_batch=4, A=0.5` while seeds 7–10 ran `hf_batch=1, A=1.0`
into the same directory — visible as a 1.6× per-step timing split. Verdict quoted verbatim:
*"What this invalidates: … AF's `hardflow_new-*` rows including the 4.58 s/ep figure. Treat all of
them as unusable."* Only our own `bbunet` ladder (C114–119) is post-Fix_9 and clean.

**(c) There *is* one real K=2 HardFlow win, and it is neither safety nor speed.**
2026-08-02 DA §11.4: HardFlow **rescues the `-c` selection rule** — AF 6/30 → 24/30, MF 3/30 →
23/30. Mechanism, and it is a good one: `minimum_projection_cost` ranks candidates by
`Σ‖x1_proj − x1_ref‖²`. Under DPCC at K=2 that cost is measured on a *half-integrated iterate*,
where a stalled trajectory is trivially cheap to project, so the rule selects the stalled candidate
(signature: 199 steps = cap, 0 successes, 0 violations). Under HardFlow the same cost is measured on
the *predicted endpoint*, where a stalled trajectory is expensive because it fails to reach the
goal. So the win is: **HardFlow makes `-c` a valid ranking criterion at low K.**

Worth noting what that costs to keep — the DA says so itself: the cheaper fix is to compute DPCC's
ranking cost on an extrapolated endpoint, which needs no NLP at all. And note this win survives the
degeneracy analysis intact: it comes from ingredient **I1** (endpoint-space) applied to the
*ranking*, which is alive at K=2 for the terminal solve even though the *guidance* is dead.

### 8.2 What actually won: K=1 — the fully degenerate configuration

5-seed DA §4.3, same checkpoint, `hardflow_new-tightened` vs `dpcc-t-tightened`, joined here with
the genuine-step count from §4.1 of this study (`A = 0.5`):

| K | **genuine HF steps** | NLP solves / env step (measured) | HF-tight S&C / steps | DPCC-t-tight S&C / steps | verdict |
|---|---|---|---|---|---|
| **1** | **0** | 1.02 | **1.000** / 63.77 | 0.967 / 58.57 | ✅ HF +0.033 gate, −5.2 steps lost |
| **2** | **0** | 1.02 | 0.933 / 64.80 | **0.967** / 59.43 | ❌ HF loses both |
| 5 | 2 | 3.05 | 0.933 / 66.57 | 0.933 / **60.60** | ❌ tie on gate, HF loses steps |
| 10 | 4 | 5.08 | 0.833 / 66.37 | **0.933** / 63.63 | ❌ HF loses both |

**HardFlow's only win in the entire corpus sits at zero genuine HardFlow steps, and its gate score
falls monotonically as genuine steps are added: 1.000 → 0.933 → 0.933 → 0.833.** That is the
opposite ordering from the one the method predicts.

The 5-seed DA had already seen half of this and said so plainly — *"At K = 1 the HardFlow arm is
terminal-only projection … The §2.1 Target beat is therefore not evidence that in-loop constrained
sampling works"* (§4.3). Two additions from §2–§4 of this study:

1. **K=2 is terminal-only too.** The DA's own table reads the K=2 row as *"1 of 2 — 50 %"*, which
   counts the solve but not *which* solve. At `A = 0.5`, `int((1−0.5)·2) = 1`, so step `k=0` is
   inactive and the single solve is the forced `k=1` terminal one, where `τ⁺ = 1` and all three
   ingredients are dead (§2). The honest cell is **"0 of 2 genuinely guided"**, not 50 %.
2. **The `-r`/`-c`/`-t` equivalence at K=1 has a second cause.** The DA notes they are identical at
   `hardflow.batch_size: 1` (§2.1) — true. Independently, `activation_threshold` is also inert at
   K=1 and K=2 (§3, §4), so the entire HardFlow knob surface is frozen at the operating point.

**Do not over-read the K=1 win either.** K=1 and K=2 are *both* fully degenerate, yet K=1 wins and
K=2 loses — so degeneracy does not explain the win. What it is: a 1/30 episode difference, on a
checkpoint whose gate score decays with K on *every* arm (the `dpcc-*` ladder does the same thing:
0.967 → 0.967 → 0.933 → 0.933), in a configuration that runs no HardFlow-specific arithmetic. The
defensible statement is the one the 5-seed DA reached: **the cheapest possible configuration of arm
C works, and its single NLP is a slightly better one-shot projection than DPCC's** — not that
in-loop constrained sampling helps.

### 8.3 A correction: the stated mechanism in the 2026-08-02 DA §11.5 cannot be the cause

That section reports the standalone naive-FM ladder (`flow_matching_v3_hardflow`, FlowMatchingODE,
seed 6, HF arm vs its own DPCC control on the identical model and seed):

| cand | K | HF threshold | HF s+c | DPCC s+c |
|---|---|---|---|---|
| CAND_39 | **2** | **0.0** | **2/6** | **6/6** |
| CAND_40 | **5** | **0.0** | **3/6** | **6/6** |
| CAND_35 | 10 | 0.0 | 6/6 | 6/6 |
| CAND_37 | 20 | 0.0 | 6/6 | 6/6 |

and attributes the collapse to: *"HF projects an endpoint estimate that is only as good as a
first-order extrapolation, and that estimate degrades exactly as K falls."*

🔴 **That explanation does not apply to these rows.** Every one of them runs `threshold = 0.0`, i.e.
terminal-only: the gate `k >= int((1−0.0)·K)` never fires, only the forced `k == K−1` solve does,
and there `τ⁺ = 1` so `(1 − τ⁺)·V_next = 0` **exactly** (§2, verified in float). *The extrapolation
error the explanation invokes is identically zero in the configuration it is explaining.* The
collapse at K=2/K=5 there is real; its stated cause is not.

What is left as candidate causes, given the input sample and the constraint set are both shared:
`dynamics_mode: deriv` is the default in every config (`config/hardflow_projection_eval.yaml:97`,
`meanflow_projection_eval.yaml:105`, `alphaflow_projection_eval.yaml:117`), and that mode exists
precisely so *"arms B and C enforce an IDENTICAL feasible set and differ only in WHEN the constraint
is applied"* (yaml comment). At `threshold = 0.0` the *when* is also identical. So a 2/6-vs-6/6 gap
on the same sample, same constraints, same timing points at the **solver path**, not the algorithm —
most likely `solve_limited` failing and the fallback keeping the last IPOPT iterate, which is not
guaranteed feasible (`hardflow_projection.py:339-346`). `nlp_failures` is already counted and
returned in `infos`; reading it on those runs is the cheapest possible test and it has not been done.

### 8.4 Consequence for the §11.7 roadmap item (MeanFlow-exact endpoint)

The 2026-08-02 DA's headline proposal — *"the single highest-value idea in this section"* — is to
replace the Euler endpoint extrapolation with MeanFlow's exact interval jump,
`x₁ = x_τ + (1−τ)·u(x_τ, τ, h = 1−τ)`, removing the error term for one network pass.

**The idea is sound and worth keeping.** But it cannot rescue the regime §11.5 says it would:

* At `A ≤ 0.5` with K ∈ {1, 2}, the only solve is terminal, `1 − τ⁺ = 0`, and the endpoint estimate
  is **already exact**. The fix is a strict no-op there — it will reproduce the current numbers bit
  for bit.
* It becomes live only where a genuine step exists: `A = 1.0` at K ≥ 2, or `A = 0.5` at K ≥ 5 — i.e.
  exactly the settings where HardFlow currently loses the gate anyway (§8.2). That is the right
  place to test it, and it is the same place §9.4 of the 5-seed DA already identified as the only
  one with threshold headroom.

So the correct experiment is **MeanFlow-exact endpoint *and* raised `A`, measured at K = 5 and
K = 10** — not at the project's K=1–2 operating point.

There is a second, separate `h` to fix, and the two must not be conflated (§5, D3):

| where | current | faithful-port rationale | "make HF work on MF" alternative |
|---|---|---|---|
| Euler integration step (1) | `h = 0` | instantaneous field, identical to the Gen12 FM port | `h = dt` — matches arms A/B, makes the step exact instead of first-order |
| endpoint prediction (2) | `h = 0` then `×(1−τ⁺)` | first-order extrapolation, as upstream | `h = 1 − τ⁺` — exact endpoint, the §11.7 idea |

Both currently sit at `h = 0` for the deliberate reason documented at
`mix_uav/sampling/hardflow_projection.py:437-443`: keeping arm C a verbatim Gen12 port so the
projection math is unchanged. Changing either turns arm C into a *new method* rather than a ported
baseline. That is a legitimate thing to build — but it is a different experiment from "does
HardFlow transfer", and it should be run and reported as one.

### 8.5 What this changes about the curated snapshot

The architecture-matched headline is **MeanFlow-UNet K1 `hardflow-tightened`, S&C 1.000, 63.77
steps, 2.64 s/ep, 14.6× the Target** (SNAPSHOT §2). The *number* stands — 5 seeds, post-Fix_9,
clean provenance, and it is our only architecture-matched row clearing the gate.

What does not stand is the *label*. Per §3, at K=1 that arm executes one Euler step and one IPOPT
projection; nothing in it is HardFlow. The snapshot's own §4 line already reports the right shape —
*"In-loop (HardFlow) vs post-hoc (DPCC), same checkpoint: K1 1.000 vs 0.967 … K ≥ 2: post-hoc equal
or better"* — but it reads naturally as "in-loop helps at K=1", when §8.2 shows the correct reading
is **"the in-loop mechanism never ran, and wherever it did run it lost"**.

Suggested wording, if the snapshot is regenerated (not applied — see §10):

> `K1 hardflow-tight` → `K1 hardflow-tight` **(terminal-only NLP — degenerate, see
> `HF_Study/DEGENERACY_HardFlow_at_low_K.md` §3)`**, with a footnote that at K=1 the arm is one
> Euler step plus a single IPOPT projection and carries no in-loop guidance.

This does not weaken the snapshot's headline claim, which is about *cost at S&C 1.000* versus the
K=20 `aw10` Target. It relabels the mechanism that delivers it, and it removes a standing invitation
to write "our in-loop constrained sampler beats DPCC" in a paper draft — a claim the data does not
support at any K.

---

## 9. Falsifiable predictions (cheap, from data we already log)

All of these are computable from existing eval output — no new code path.

1. **`nlp_solves` per plan** should equal `n_active` from §4.1: 1 at K=1 and K=2 (A=0.5),
   3 at K=5, 5 at K=10, 10 at K=20. If it does, D1 is confirmed operationally.
2. **`candidate_costs`** (the prox cost `Σ‖X1* − X1_ref‖²`, `:550-551`) should be
   *dramatically* higher at K=1/2 than K=10 for the same scene/seed: at low K the terminal
   projection absorbs the entire constraint violation in one snap, whereas at K=10 the four
   genuine steps have already walked the endpoint most of the way into `S`.
3. **`hardflow_new` vs `hardflow_new` at `A=0.5` and `A=1.0`, K=1** must be *bit-identical*
   (§3). If they differ, something outside the sampler is reading the threshold.
4. **K=1, `fm` engine, arm B vs arm C** should agree to solver tolerance on the executed
   trajectory. A large gap there is a solver/variable-scope finding, not an algorithm finding.
5. **K=1, `mf` engine**: arm C's unprojected base sample should be visibly worse than arm A's
   (D3). Compare `disable_projection=True` on both — the HardFlow policy's
   `disable_projection` branch (`:711-713`) routes to `self.model(...)`, i.e. the engine's own
   `h=dt` sampler, so that comparison isolates D3 exactly.

---

## 10. Recommendations (proposals — nothing changed)

**Reporting.** Mark `hardflow_new*` at K∈{1,2} as *degenerate — equivalent to
sample-then-project* wherever those rows appear, and do not use them to support or refute any
HardFlow claim. If a low-K anchor is wanted, the honest label for those rows is
"IPOPT terminal projection", not "HardFlow". This applies retroactively to the curated
snapshot's architecture-matched headline row (§8.5) — the number stands, the label does not.

**Two cheap reads on data that already exists**, before any new cluster time (§8.3, §8.2):
`nlp_failures` on CAND_39/CAND_40 (the 2/6 and 3/6 terminal-only collapses — if IPOPT is bailing
and the fallback keeps an infeasible iterate, that is the whole story and it is not algorithmic);
and the per-K `nlp_solves` on our own `bbunet` ladder, to confirm the 1.02 / 1.02 / 3.05 / 5.08
figures line up with §4.1's `n_active` as predicted.

**If low-K HardFlow is genuinely wanted**, the minimum non-degenerate configuration is
`K ≥ 3` with `A = 1.0` (2 genuine steps), and even then §4's Theorem-4 caveat applies to the
`τ⁺ = 1/3` step. `K = 5, A = 1.0` (4 genuine steps at τ⁺ ∈ {0.2,…,0.8}) is the first setting
that is structurally comparable to the paper's N=10 / A=0.5 configuration.

**Two small code changes worth considering** (both no-ops at K≥5, both fix real waste):

* Skip the second network call when `τ⁺ ≥ 1`, and set `X1_ref = X_ref` directly. Saves 1 NFE
  per plan, removes the `0·NaN` hazard, and makes the NFE table in §6 read `K + n_active − 1`.
* Add a startup warning when `n_genuine == 0` (i.e. `K − floor((1−A)·K) ≤ 1`), printed next to
  the existing `[hardflow] engine=… K=… A=…` line (`eval_mix_uav.py:1399-1402`), so a
  degenerate configuration announces itself in the log instead of being discovered in the DA.

**Not recommended:** adding `C(·)` to the NLP to restore I4. It would make arm C solve a
different problem than arm B (which has no cost term either), breaking the constraint-set
parity that `setup_dpcc_projector(..., return_constraint_list=True)` exists to guarantee
(`:1370-1374`). D2 is a *deliberate* divergence for comparability; it should be documented as
one, not undone.

---

## 11. One-paragraph answer

HardFlow's per-step update is `X_ref → project the predicted endpoint → pull back by τ⁺`. At
the last ODE step `τ⁺ = 1` exactly, which kills the endpoint lookahead `(1−τ⁺)`, makes the
pull-back a full snap, and leaves no subsequent network call to react to the correction — the
paper proves precisely this, because that collapse *is* its safety guarantee. So HardFlow's
distinctive behaviour lives only in the `K−1` non-terminal active steps. At **K=1** there are
none, unconditionally. At **K=2** with our shipped `activation_threshold=0.5` there are none
either, because the floor gate deactivates step 0; only `A=1.0` leaves a single genuine step,
and it sits at the τ=0.5 point the paper's own error bound warns against. In both cases the
executed algorithm is a plain Euclidean projection of an unguided Euler sample. Our port
makes this stricter still: with no cost term `C(·)`, `reg_scale` and the τ² weight cannot
change the argmin at any K, so nothing else is left to survive. And on the `mf`/`af` arms the
low-K rows are additionally confounded, because arm C integrates the instantaneous field
(`h=0`) while arms A/B use the trained interval field (`h=dt`) — a difference that vanishes at
K=20 and dominates at K=1.

And the `avoiding-d3il` data agrees, uncomfortably. Every HardFlow run in that corpus used
`A = 0.5`, so K=1 and K=2 both have **zero** genuine HardFlow steps, K=5 has two and K=10 has
four — and the gate score goes 1.000 → 0.933 → 0.933 → 0.833 in that order. The single
configuration that clears the gate, and that carries our architecture-matched headline, is the
one running no HardFlow math at all. K=2 HardFlow was never better than DPCC: it tied the gate
at 3× the cost, the one row that looked like a win was retracted for provenance contamination,
and the only genuine K=2 HardFlow advantage is that it makes the `-c` candidate-selection rule
work — a ranking effect that a much cheaper change to DPCC would also buy.
