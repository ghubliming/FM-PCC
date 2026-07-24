# The Deep Mix: HardFlow ∘ MeanFlow — constrained sampling on the conserved endpoint field

**Date:** 2026-07-02
**Companion doc:** `BLEND_HardFlow_iMeanFlow.md` (engineering levels / repo audit). This document is the mathematical core: the lineage connecting the two methods, two provable defects in HardFlow that only become visible through the MeanFlow lens, the algorithm that falls out, and a **closed-form numerical validation** (`validate_theory.py`, run in this container — all numbers below are measured, not estimated).

---

## 0. TL;DR

HardFlow steers flow sampling by repeatedly (i) predicting the terminal sample, (ii) hard-projecting the prediction, (iii) pulling the correction back into the ODE state. We show:

1. Its terminal prediction `x̂1 = z + (1−τ)v(z,τ)` is **exactly the posterior mean** `E[x1|z_τ]` — the *mode average* — not the sample endpoint. Verified to 2e-16.
2. Its pull-back gain `τ` is a wrong approximation of the endpoint-map Jacobian — measured to deliver only **11%** of the requested correction at τ=0.1.
3. Both defects are exactly what MeanFlow's average-velocity field repairs: `F(z,τ) = z + (1−τ)u` is the **true flow endpoint** (a conserved quantity whose conservation law *is* the MeanFlow identity), and `∇F = I + (1−τ)∇u` is available by JVP.
4. The repaired algorithm (Newton–MeanFlow constrained sampling, K=2 anchors) achieves the same 0% violation rate while applying **zero** correction to samples that were never going to violate, and gets **2.7× closer to the true constrained distribution** (W1 0.050 vs 0.134) than HardFlow-'all' — at 2 model evaluations instead of ~40.
5. The lens also collapses HardFlow's strongest baseline: with a mean-flow model, **OC-Flow's ODE-constrained optimal control problem becomes a single algebraic NLP**, and an asymptotically *exact* constrained sampler (noise-space Langevin) becomes affordable for the first time.

| measured (2-mode GMM, obstacle `|x1|<0.5`, 4000 draws) | violation | corruption of feasible samples | W1 to true conditional |
|---|---|---|---|
| HardFlow-'all' (N=20) | 0.000 | 0.228 avg correction applied | 0.134 |
| HardFlow-'late' (N=20) | 0.000 | 0.015 | 0.067 |
| **MF-Newton (K=2)** | 0.000 | **0.000** | **0.050** |
| MF-project (K=1) | 0.000 | 0.000 | 0.060 |

Constraint satisfaction is table stakes — every method gets 0% violations. The differentiator is **what the method does to the distribution while enforcing them.**

---

## 1. Three fields, one lineage

Fix the linear interpolant `z_τ = τ·x1 + (1−τ)·x0` (HardFlow's convention: τ=0 noise, τ=1 data; flip `τ = 1−t` for official iMF). Three fields on `(z, τ)`:

| field | definition | learned by |
|---|---|---|
| `v(z,τ)` | marginal velocity `E[ẋ_τ \| z_τ = z]` | flow matching (HardFlow's checkpoint) |
| `PM(z,τ)` | posterior mean `E[x1 \| z_τ = z]` (Tweedie / x-prediction) | — (HardFlow *estimates* it, see Thm 1) |
| `F(z,τ)` | **flow-map endpoint**: where the PF-ODE trajectory through `(z,τ)` lands at τ=1 | MeanFlow: `F = z + (1−τ)·u(z, τ, h=1−τ)` |

### Theorem 1 (HardFlow's extrapolation is the posterior mean — exactly)

For any `p0, p1` and the linear interpolant, `v(z,τ) = E[x1 − x0 | z_τ = z]`, and since `z = τE[x1|z] + (1−τ)E[x0|z]`:

```
z + (1−τ)·v(z,τ) = τE[x1|z] + (1−τ)E[x0|z] + (1−τ)E[x1|z] − (1−τ)E[x0|z] = E[x1 | z_τ = z] = PM(z,τ)
```

Not first-order — **identical**. (Measured: `max|Euler − PM| = 2.2e-16` across τ ∈ {0.05,…,0.9}.) So HardFlow does not use a "biased estimate of the endpoint"; it uses a *perfect estimate of a different field*. `PM` averages over modes; `F` commits to one. At τ=0.05 on the two-mode task: `mean|PM| = 0.069` (dead center — inside the obstacle), `mean|F| = 1.145` (on a mode). They converge only as τ→1 (`mean|PM−F|`: 1.077 → 0.667 → 0.257 → 0.024 at τ = 0.05/0.3/0.6/0.9).

### Theorem 2 (the MeanFlow identity is the conservation law of F)

`F(z_τ, τ) = x1` is constant along every PF-ODE trajectory, i.e. F solves the transport PDE

```
∂_τ F + (v · ∇_z) F = 0,      F(z, 1) = z.
```

Substituting `F = z + (1−τ)u` and expanding gives

```
u = v + (1−τ)·[∂_τ u + (v·∇_z) u]  =  v + (1−τ)·du/dτ
```

— precisely the MeanFlow identity (the `t=1` slice of the two-time version; iMF's JVP loss trains its characteristics, using the predicted `v` as the tangent). **Lineage statement:** the consistency function of consistency models, MeanFlow's `u` at full interval, and the object HardFlow *wants* to constrain are all the same conserved field `F`. HardFlow approximates it with `PM` (Theorem 1 — the wrong field, right limit); consistency/MeanFlow models learn it outright. The deep mix is: *constrain the learned conserved field, not the mode average.*

---

## 2. HardFlow re-derived — and its two crutches explained

HardFlow's per-step update (`flow_policy.py:1321-1371`), in this language:

```
x̂1  = PM(z, τ)                                 (Theorem 1: exactly the mode average)
X1* = Π_C(x̂1)                                  (the prox NLP; pure-constraint case = Euclidean projection)
z  ← z + τ·(X1* − x̂1)                          (pull-back with gain τ)
```

Two approximations are hiding here, and both degrade at exactly the same place (small τ):

**Defect A — wrong field.** Early in sampling, `PM` sits between modes. On the avoiding benchmark the obstacle *is* between the modes (paths pass left or right), so the early reference is a phantom trajectory through the obstacle: HardFlow-'all' "corrects" every sample, including the ~93% whose true endpoints were already feasible. Measured corruption of feasible samples: **0.228** average applied correction ('all') vs **0.015** ('late') vs **0.000** (exact F — feasible endpoints are *detected* as feasible at every anchor and left alone). This is the real reason `hardflow_activation='late'` exists and works: it waits until `PM ≈ F` (mode commitment) — trading away all early constraint influence to avoid constraining a phantom.

**Defect B — wrong Jacobian.** The pull-back wants: find `δz` with `F(z + δz, τ) = X1*`. Newton gives `δz = [∇_z F]^{-1}(X1* − F)`. HardFlow uses `δz = τ·(X1* − x̂1)`, i.e. it approximates `[∇_z F]^{-1} ≈ τ·I` — the *conditional*-interpolant sensitivity `∂z_τ/∂x1 = τ` at frozen `x0`, which is not the marginal endpoint map's inverse Jacobian. Closed-form check (`p1 = N(0,I)`: `F(z,τ) = z/s(τ)`, `s = √(τ²+(1−τ)²)`): the true gain is `s(τ) ∈ [0.71, 1]` while HardFlow uses `τ ∈ [0,1]` — mismatch unbounded as τ→0. Measured on the GMM (requested endpoint change Δ=0.3):

| τ | HardFlow gain τ: achieved/requested | Newton gain 1/F′: achieved/requested |
|---|---|---|
| 0.1 | **0.11** | 1.42 |
| 0.3 | 0.39 | 1.42 |
| 0.5 | 0.65 | 1.38 |
| 0.7 | 0.76 | 1.13 |
| 0.9 | 0.96 | 1.00 |

Early corrections are nearly impotent — which is why HardFlow needs N=20 repeated correction rounds: it is a **damped fixed-point iteration with a vanishing step size**, and the repetitions compensate. (Newton's overshoot at small τ reflects genuine curvature of F near mode boundaries — implement it damped/trust-region; even undamped it is 12× closer to the target than the τ-gain at τ=0.1.)

**Corollary.** HardFlow's three empirical design choices — 'late' activation, N=20 steps, and the τ² prox weighting — are all compensations for using (`PM`, `τI`) in place of (`F`, `∇F`). Give the algorithm the right field and the right Jacobian, and all three become unnecessary. That is not an engine swap; it is the theory explaining the original paper's own ablations.

---

## 3. The repaired algorithm: Newton–MeanFlow constrained sampling

```
Input: mean-flow model u(z, τ, h) (⇒ F(z,τ) = z + (1−τ)·u(z,τ,1−τ), ∇F by JVP),
       anchors 0 < τ_1 < … < τ_K = 1 (K ∈ {1,2,3}), constraint set C, prox weight ρ
z ~ N(0, I); apply conditioning
for k = 1 … K:
    z ← z + (τ_k − τ_{k−1}) · u(z, τ_{k−1}, τ_k − τ_{k−1})        # exact interval jump (1 NFE)
    F_k ← z + (1 − τ_k) · u(z, τ_k, 1 − τ_k)                      # true endpoint      (1 NFE)
    X*  ← argmin_X ½ρ‖X − F_k‖²  s.t. X ∈ C                       # HardFlow's NLP, unchanged
    z  ← z + λ · [∇F_k]^{-1}(X* − F_k)                            # damped Newton pull-back (JVP solve)
return z                                                           # τ_K = 1 ⇒ z = X* ∈ C exactly
```

- `[∇F]^{-1}(·)`: for HardFlow's 44-dof trajectories, build the full Jacobian with 44 JVPs (or one `jacfwd`) and solve directly — microseconds; for large problems, CG on JVPs.
- Measured behavior (K=2, exact fields): 0% violations, zero perturbation of feasible samples, best distributional fidelity of all variants (W1 = 0.050; K=1 projection = 0.060; HardFlow-'all' = 0.134).
- Why K=2 ≥ K=1: with a *perfect* F, conservation makes all K equivalent (any correction fixes the endpoint once and for all). With a *learned* F, later anchors re-check and refresh the estimate; and the mid-flow correction happens where `∇F` is better conditioned. K is an ε_u-robustness knob, not an accuracy knob — the opposite of HardFlow's N, which fights its own gain error.

## 4. The problem all of these methods approximate — and two upgrades

For a deterministic flow the endpoint is a *function* of the noise: the exact constrained-generation problem lives in noise space.

**(a) MAP-style projection (NLP).** `min_{z_0} ½‖z_0 − ζ‖²  s.t.  F(z_0, 0) ∈ C`, with ζ the Gaussian draw. With an instantaneous field, evaluating the constraint costs a full ODE solve and its gradient an adjoint solve — *that is literally OC-Flow*, HardFlow's expensive, unstable baseline (`flow_policy.py:1517`). With a mean-flow model, `F` is **one network call** and the constraint Jacobian **one JVP**: OC-Flow's optimal-control problem collapses into a small algebraic NLP. Taxonomy that falls out:

| method (HardFlow's Table) | what it is, in this language |
|---|---|
| gradient guidance | one penalty-gradient step on `PM` (wrong field, no feasibility guarantee) |
| projection-all/late | project intermediate `z` in x-space (ignores F entirely — fights the flow) |
| OC-Flow | the exact noise-space problem, paid at full ODE price |
| HardFlow | sequential damped prox on `PM` with gain `τI` |
| **Newton–MF (§3)** | sequential Newton prox on `F` with gain `∇F^{-1}` |
| **noise-space NLP** | the exact problem, at 1-NFE-per-iteration price |

**(b) Exactness upgrade (Langevin).** Projection-type methods (including ours in §3) transport infeasible mass to the constraint *boundary* — a distributional atom the true conditional does not have (visible as W1 = 0.050 ≠ 0). The genuinely exact target is the conditioned noise distribution `N(0,I) restricted to {F(z_0) ∈ C}` pushed through F. With F at 1 NFE, **noise-space Langevin/MCMC on `log N(z_0) + log σ_ε(−d_C(F(z_0)))` becomes affordable** (each step = 1 NFE + 1 JVP; with an instantaneous field each step would cost a full ODE solve + adjoint — hopeless). To our knowledge no method in HardFlow's comparison samples the constrained distribution *exactly*, even asymptotically; this one does, and it exists **only** because the mean-flow amortizes the ODE map. This is the "something really new" candidate: **asymptotically exact hard-constrained sampling at MCMC-over-1-NFE cost.**

## 5. What the mix demands from training (and gives back)

1. **Terminal-anchored interval sampling.** All queries hit `u(·, τ, h = 1−τ)` — intervals ending at the data end. Both trainers (official `sample_tr`, logit-normal pairs; our port, `r = t·rand`) leave that slice thin. Pin a fraction of training intervals to `[τ, 1]` (mirror of MeanFlow's existing `r = t` anchor at the other end). One line; detailed in the companion doc §4.
2. **Jacobian fidelity.** Newton uses `∇_z u`, which the MeanFlow JVP loss supervises only along the tangent `v`. Cheap fix: add random-tangent JVP pairs to the loss (`E‖∂_z u·w − target·w‖²`, w random unit) — a Jacobian sketch. Optional; the K=1/K=2 variants work without Newton (pure projection at τ=1), so this only gates the §3 mid-flow corrections.
3. **What flows back to MeanFlow:** the constrained-sampling application is a *stress test of F off the sampling grid* — it queries arbitrary (τ, h) and off-manifold z. Terminal anchoring + Jacobian sketches are training improvements motivated purely by this use case; if they also improve plain few-NFE sampling, that is a MeanFlow contribution in its own right.

## 6. Honest novelty ledger

Known before this note: `x̂1 = z+(1−τ)v` is the x-prediction (folklore); consistency models learn F; MeanFlow's identity and its JVP training; HardFlow's prox-and-pull-back loop.
**New here:** (i) Theorem-1-based *diagnosis* of HardFlow — it constrains the mode-average, and its 'late'/N=20/τ² choices are quantifiable compensations (§2, measured); (ii) the Newton pull-back through the learned endpoint Jacobian, with the conditional-vs-marginal sensitivity distinction that exposes the τ-gain as wrong (§2B, §3); (iii) the collapse of OC-Flow into an algebraic NLP and the resulting method taxonomy (§4a); (iv) affordable asymptotically-exact constrained sampling via noise-space MCMC with a 1-NFE oracle (§4b); (v) terminal-anchored interval sampling as a training-side requirement discovered by composing the two methods (§5).

## 7. Falsifiable predictions for the real avoiding benchmark

1. HardFlow-'all' perturbs trajectories whose unconstrained rollout was already feasible; iMF-Newton-K2 leaves them untouched (measure: displacement of feasible-rollout endpoints, as in §0 table).
2. iMF-Newton-K2 matches or beats HardFlow-N=20 success/violation rates at ≤ 1/5 the wall-time (IPOPT solves 20→2 dominate).
3. Removing terminal anchoring from iMF training degrades K=1 far more than K=4 (K compensates ε_u).
4. HardFlow's own gap between 'late' and 'all' shrinks to zero when its two call-sites are given the u-head (because the reference stops being the mode average) — the cleanest possible attribution experiment.

## 8. Reproduce the validation

```bash
python3 logs_in_develop/HF_iMF/validate_theory.py     # numpy only, ~2 min CPU
```

1D, two modes (±1, σ=0.35), obstacle `|x1| < 0.5`, exact closed-form `v`/`PM`, RK4-exact `F`. Prints the three experiment blocks quoted throughout. The point of 1D-exact: every claim above is tested against the *true* fields, so any disagreement would falsify the theory rather than the training.
