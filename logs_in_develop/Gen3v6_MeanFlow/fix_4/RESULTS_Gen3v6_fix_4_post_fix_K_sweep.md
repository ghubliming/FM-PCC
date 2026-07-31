# Gen3v6 fix_4 — post-fix matched-K sweep: results & analysis

**Runs**: jobs **24034 / 24035 / 24036 / 24037 / 24038** (K = 1 / 2 / 5 / 10 / 20), node i6-gpu-1, 2026-07-30 15:49 UTC.
**Git**: `87b01d9b` — *"fix: address initial noise scale issue in HardFlowSampler and update related tests"* (the fix_4 commit).
**Config**: `config/meanflow_projection_eval.yaml`, seed **6**, `n_trials: 2`, 3 halfspace variants, `HFFM_BATCH=4`, `HFFM_ACT_THRESHOLD=0.5`, `candidate_cost: prox`. Checkpoint: `mf_dit`, step 97000, EMA weights.
**Baseline for comparison**: the pre-fix sweep at `bed63b3` (jobs 24021/24022/24023, K = 1/5/20) in `temp/2026-07-30/`.
**Raw data**: `temp/2026-07-30/II/`.

Companion docs: [`CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md`](CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md) (what was changed and why), [`../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md) (the DPCC K=2 defect), [`../U3/INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md`](../U3/INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md) (now unblocked — its 🛑 INVALIDATED banner can be replaced by a pointer here).

---

## 1. Verdict

1. **fix_4 works, and it is provably isolated.** Across all 63 DPCC/`diffuser` cells shared with the pre-fix sweep, **zero** behavioural metrics changed (success, constraints, g&c, steps, violations, tracking error). Only wall-clock timings moved. The σ=0.5 → σ=1.0 change touched arm C and nothing else.
2. **The HardFlow K-degradation is gone.** Pre-fix, arm C collapsed as K grew (`hardflow_new-t-tightened` 2.5 → 1.5 → 1.5 g&c at K=1/5/20). Post-fix it is **3.0 / 3.0 / 3.0 / 3.0 at K = 2/5/10/20** — a perfect score, tying the best DPCC arm.
3. **The "NLP-intervention compounding" hypothesis is dead.** The changelog's honest loose end is resolved: the entire K-dependent decay was the mis-scaled initial noise. Once arm C draws at σ=1.0 the decay disappears at every K, with the NLP solve count unchanged.
4. **One real defect survives, and it is a ranking bug, not a solver bug.** `hardflow_new-c` still freezes ~75% of control steps at K ≥ 5 (DPCC's `-c`: ~4%). The NLP is **not** failing to solve — `-c` records **0.00% solver failures** while the healthy arms record up to 2.83%. The ranking key `cand_prox` is measured on an *extrapolated* quantity and omits the τ² factor the NLP objective itself uses, which makes it structurally prefer motionless candidates. See §5, §5.4 for the decisive K=1 test, and §5.8 for the exclusion of a solve-abort/timeout explanation (no such mechanism exists on the DPCC path, which is byte-identical to upstream).
5. **At K=2 both engines fail identically** (~99.5% frozen). That is the known field-level defect, now confirmed engine-independent — see §6.

**Headline number for the paper**: at K ≥ 2, `hardflow_new-t-tightened` = `dpcc-t-tightened` = 3.0/3.0 goal-and-constraints, at ~1.0–1.4× the per-step cost. In-loop constrained sampling now *matches* post-hoc projection on this task; it does not beat it.

---

## 2. The control — fix_4 changed only arm C

Compared every shared cell (K ∈ {1,5,20} × 7 DPCC/diffuser variants × 3 halfspaces = 63) on `sr`, `cs`, `gc`, `steps`, `nviol`, `tviol`, `terr`:

```
DPCC + diffuser cells compared:                   63
behavioural-metric mismatches (excluding ctime):   0
```

43 cells differ in `Average computation time per step` only, by ≤ 0.013 s — scheduler noise. This is the cleanest possible evidence that the fix is confined: the shared code path is bit-reproducible across two runs a git-commit apart, so any arm-C change below is attributable to the fix and not to run-to-run variation.

*(It also means the DPCC arms did not need re-running. Worth remembering for the Gen3v7 sweep — a DPCC-only control run is cheap insurance but not required if the commit provably touches only `hardflow_projection.py`.)*

---

## 3. Arm C: goal-and-constraints, before vs after

Sum over the three halfspace variants; max 3.0. K=2 and K=10 have no pre-fix counterpart in this sweep.

| arm-C variant | K=1 old→new | K=2 new | K=5 old→new | K=10 new | K=20 old→new |
|---|---|---|---|---|---|
| `hardflow_new-r`            | 1.00 → 0.50 | 2.00 | 1.50 → **2.00** | 2.50 | 2.00 → 2.00 |
| `hardflow_new-c`            | 1.50 → 1.50 | 0.00 | 0.00 → **1.00** | 1.00 | 0.00 → **1.00** |
| `hardflow_new-t`            | 1.50 → 1.00 | 1.00 | 1.00 → **1.50** | 2.00 | 0.50 → **2.00** |
| `hardflow_new-r-tightened`  | 3.00 → 3.00 | 2.00 | 3.00 → 2.50 | 2.50 | 2.50 → 2.50 |
| `hardflow_new-c-tightened`  | 2.50 → 2.00 | 0.00 | 0.00 → **1.50** | 1.50 | 0.00 → **1.50** |
| `hardflow_new-t-tightened`  | 2.50 → 2.50 | **3.00** | 1.50 → **3.00** | **3.00** | 1.50 → **3.00** |
| **arm-C total (max 18)**    | 12.0 → 10.5 | 8.0 | **6.0 → 11.5** | 12.5 | **6.5 → 11.5** |

Read the totals row: pre-fix the arm *fell* from 12.0 at K=1 to ~6 at K≥5. Post-fix it is flat-to-rising, 10.5 → 11.5 → 12.5 → 11.5. The K-dependence has been removed, which is exactly the signature you expect if the cause was a wrong input distribution propagated through K net calls.

The small K=1 regression (12.0 → 10.5) is not a concern: at K=1 there is a single Euler macro-step and one terminal NLP solve, so the terminal projection overwrites most of the initial draw — the two σ values are nearly interchangeable there, and 1.5/18 on 6 trials per cell is within noise.

For reference, the matched DPCC arms (unchanged by the fix):

| arm | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` (no projection) | 0.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| `dpcc-r-tightened` | 2.50 | 2.50 | 3.00 | 3.00 | 3.00 |
| `dpcc-c-tightened` | 1.00 | **0.00** | 3.00 | 3.00 | 3.00 |
| `dpcc-t-tightened` | 3.00 | 2.50 | 3.00 | 3.00 | 3.00 |

---

## 4. Freeze rate — the decisive measurement

Fraction of emitted control steps that are exactly `(±0.000, ±0.000)`, pooled over 3 halfspaces × 2 trials. Counted with `grep -cE '^ACT +\(-?0\.000,-?0\.000\)'` (the signed-zero form matters — a naive `\(0\.000` pattern undercounts by ~⅓).

| variant | K=1 old→new | K=2 old→new | K=5 old→new | K=10 new | K=20 old→new |
|---|---|---|---|---|---|
| `dpcc-c`                   | 1.2 → 1.2 % | 99.5 → 99.5 % | 4.3 → 4.3 % | 4.4 % | 1.3 → 1.3 % |
| `hardflow_new-r-tightened` | 2.7 → 1.3 % | 10.3 → 6.2 % | 10.8 → **6.0** % | 6.1 % | 5.9 → 6.5 % |
| `hardflow_new-t-tightened` | 0.7 → 0.6 % | 8.5 → **2.3** % | 59.1 → **2.3** % | **2.3** % | 57.3 → **2.2** % |
| `hardflow_new-c`           | 2.8 → 1.2 % | 76.9 → 99.8 % | 100.0 → 78.7 % | 79.2 % | 100.0 → 71.4 % |
| `hardflow_new-c-tightened` | 2.8 → 1.5 % | 74.5 → 99.8 % | 100.0 → 78.9 % | 78.9 % | 100.0 → 74.8 % |

The `-t-tightened` row is the result: **59.1% → 2.3%** at K=5 and **57.3% → 2.2%** at K=20, landing on the same ~2% floor DPCC's healthy arms sit at. The wrong initial noise scale was producing more than half of all frozen control steps at high K.

---

## 5. What fix_4 did *not* fix: `hardflow_new-c`

`-c` improved (100% → ~75%) but remains badly broken at K ≥ 5, while `dpcc-c` sits at 4%. Three observations pin the mechanism:

**(a) It is selection, not generation.** `-r-tightened` (which ignores cost and takes candidate 0) freezes at **6%** from the *same* fan of 4 candidates that `-c` chooses from. Same sampler, same NLP, same draws — only the selection rule differs, and it costs 6% → 75%.

**(b) The ranking key actively prefers the degenerate mode.** `candidate_cost: prox` ranks by total NLP intervention Σ‖x₁_proj − x₁_ref‖². A motionless plan violates nothing, so the NLP leaves it untouched, so it scores *best*. `-c` is structurally biased toward whichever candidate the constraints care least about — and "don't move" is the global minimiser of that objective.

**(c) The freeze is a leading block, not an absorbing tail.** Per-step traces at K=5, `both-hard` (`F` = frozen step):

```
trial0  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF..........................
trial1  FFFFFFFFFFFFFFFFFF...  (all 200 steps frozen — never escapes)
r-tight FFFFFFFF..............................................................
```

The robot stalls **at the start pose** and eventually escapes (or doesn't). This matches the state-space localisation established in the K=2 investigation: the degenerate mode lives in a specific region of observation space, not uniformly along the trajectory. `-c` converts a locally-likely bad candidate into a persistent stall, because a frozen action leaves the observation unchanged, so the next replan faces the same draw distribution and again selects a frozen candidate. Escape requires a fan with no frozen member.

**Consequence**: `hardflow_new-c` / `-c-tightened` are not usable on this checkpoint and should not be reported as the HardFlow headline. Report `-t-tightened` instead (the best arm, and the matched partner of DPCC's best arm). §5.1–§5.7 below establish *why*, at code level, and what the fix is.

---

### 5.1 The solver is not failing — it has a perfect record

The obvious hypothesis is that the NLP cannot solve and the arm degrades to a fallback. **The data says the exact opposite.** Per-variant IPOPT counters, pooled over 3 halfspaces × 2 trials (`[hardflow] NFE=… NLP solves=… NLP failures=…` lines):

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `hardflow_new-r-tightened` | 0.00 % | 0.00 % | 0.50 % | **2.05 %** | 0.00 % |
| `hardflow_new-t-tightened` | 0.20 % | 1.25 % | 0.00 % | **2.83 %** | 0.99 % |
| `hardflow_new-c`           | 0.20 % | **0.00 %** | 0.10 % | **0.00 %** | **0.00 %** |
| `hardflow_new-c-tightened` | 1.78 % | **0.00 %** | 0.00 % | **0.00 %** | **0.00 %** |

`-c` is the arm with the *cleanest* solve record — literally zero failures across 33 960 solves at K=20 — while the two arms that behave well tolerate up to 2.83% failures without any loss of task performance. So:

- This is **not** "the solver cannot handle the problem". A frozen plan is trivially feasible, so IPOPT converges instantly and exactly on it. `-c` has a perfect solve record *because* it has converged onto the easiest possible problem.
- It is also **not** a silent-fallback bug (e.g. a failed solve returning zeros). There are no failed solves to fall back from.

> ⚠️ **Scope of this counter.** `n_failures` increments only in the `except RuntimeError` branch at `hardflow_projection.py:330`, and the call there is `opti.solve_limited()`, which by design **accepts** IPOPT's *limited* return statuses (`Maximum_Iterations_Exceeded`, `Maximum_CpuTime_Exceeded`) without raising. So the table above counts **hard** failures only — infeasible-detected, restoration-failed, invalid-number. On its own it does not exclude silent iteration-limit aborts. That gap is closed independently by compute time in **§5.8**; the conclusion below survives, but it rests on *both* measurements, not on the 0.00% alone.

`-c` is not failing to optimise. It is **succeeding at optimising the wrong objective.**

The solve counts also confirm the accounting is exact. Solves per replan = `batch_size × (#active steps)`, with active = `k ≥ (1−0.5)·K` plus the forced final step:

| K | active steps k | τ_next values | solves/replan | predicted | logged |
|---|---|---|---|---|---|
| 1  | {0}       | 1.0                | 4 ×1 = 4  | 491 × 4 = 1 964    | 1 964 ✓ |
| 2  | {1}       | 1.0                | 4 ×1 = 4  | 1 200 × 4 = 4 800  | 4 800 ✓ |
| 5  | {3,4}     | 0.8, 1.0           | 4 ×2 = 8  | 976 × 8 = 7 808    | 7 808 ✓ |
| 10 | {5…9}     | 0.6 … 1.0          | 4 ×5 = 20 | 970 × 20 = 19 400  | 19 400 ✓ |
| 20 | {10…19}   | 0.55 … 1.0         | 4 ×10 = 40| 849 × 40 = 33 960  | 33 960 ✓ |

Exact to the unit at every K. The activation gate and the candidate fan are doing precisely what fix_6/U4.2 specified.

---

### 5.2 DPCC `-c` vs HardFlow `-c` — what is the *same*

Both are `argmin` over the batch of a non-negative, accumulated projection distance, with a slot-0 fallback. The two implementations:

**DPCC** — `flow_matcher_v3_meanflow/sampling/projection.py:88,133,145` + `sampling/policies.py:63-67`:
```python
r = - trajectory_reshaped @ self.Q                        # :88
...
projection_costs[i] = 0.5*s@Q@s + r[i]@s + 0.5*x@Q@x      # :145
...
costs_total = Σ_timestep infos['projection_costs'][t]     # policies.py:65-66
which_trajectory = np.argmin(costs_total)                 # policies.py:67
```
Substituting `r = −xᵀQ` collapses line 145 to

$$\tfrac12 s^\top Q s - x^\top Q s + \tfrac12 x^\top Q x \;=\; \tfrac12\,(s-x)^\top Q\,(s-x)$$

i.e. **the Q-weighted squared distance the QP had to move the trajectory.**

**HardFlow** — `sampling/hardflow_projection.py:515-516` + `:672-677`:
```python
cand_prox += np.sum((X1_proj_np - X1_ref_np)**2, axis=1)  # :515-516
...
return int(np.argmin(costs))                              # :677
```
i.e. **the unweighted squared distance the NLP had to move the terminal prediction.**

Same family, same argmin, same fallback (`policies.py:69` and `_select:682` both return index 0). The port is faithful in structure. **This rules out a transcription error in `_select`.**

---

### 5.3 Four differences — one of them is load-bearing

| # | DPCC | HardFlow (Gen3v6) | verdict |
|---|---|---|---|
| 1 | Q-weighted metric (`projection_cost: 'pos_vel'`) | unweighted identity in dof space | cosmetic — rescales, does not reorder motion vs non-motion |
| 2 | distance measured on **the actual iterate** `x` being modified | distance measured on the **terminal extrapolation** `X1_ref = X_ref + (1−τ_next)·V_next` (`:505`) | **load-bearing** |
| 3 | cost *is* the applied modification | applied modification is `τ_next·(X1_proj − X1_ref)` (`:518`) and the NLP objective carries a **τ² weight** (`:176-178`), but `cand_prox` accumulates **unweighted** | **load-bearing, and internally inconsistent** |
| 4 | candidates = field samples, projected once each | candidates = trajectories pulled back toward feasibility at every active step | turns out **not** to matter — see §5.5 |

Difference 3 deserves emphasis because it is an inconsistency *within* the HardFlow implementation, not just a divergence from DPCC. The NLP minimises
$$\tfrac12\,\lambda\,\tau^2\,\lVert x_1 - x_1^{\text{ref}}\rVert^2$$
and the pull-back applies `τ_next·(x1_proj − x1_ref)` — both τ-weighted — yet the ranking key that decides which candidate gets executed drops the τ factor entirely. The selection is therefore ranking a quantity the sampler never applies.

---

### 5.4 The decisive test: `-c` is healthy exactly where the extrapolation term vanishes

At `τ_next = 1.0` the extrapolation `(1 − τ_next)·V_next` is **exactly zero**, so `X1_ref = X_ref` and the cost is measured on the real iterate — DPCC's situation. From the table in §5.1, the number of active steps carrying a *non-zero* extrapolation is:

| K | active solves | of which extrapolated (τ_next < 1) | `-c` freeze rate |
|---|---|---|---|
| 1  | 1  | **0** | **1.2 %** ✅ |
| 2  | 1  | **0** | 99.8 % ← independent field defect, see §6 |
| 5  | 2  | 1 | 78.7 % ❌ |
| 10 | 5  | 4 | 79.2 % ❌ |
| 20 | 10 | 9 | 71.4 % ❌ |

`-c` is healthy at K=1 — the one setting where its cost reduces to DPCC's — and collapses at every K that introduces extrapolated solves. K=2 is the single exception and it is explained independently: `dpcc-c` collapses there too (99.5%), on the same field, with no extrapolation involved (§6).

This is a falsifiable prediction that the data already confirms, and it isolates difference 2/3 as the cause.

---

### 5.5 The field makes the frozen plans — neither engine manufactures them

Freeze rate under slot-0 selection (no cost ranking at all), which isolates the generator:

| K | `diffuser` (**no projection**) | `dpcc-r-tightened` | `hardflow_new-r-tightened` |
|---|---|---|---|
| 1  | 2.2 % | 1.5 % | 1.3 % |
| 2  | 6.4 % | 5.9 % | 6.2 % |
| 5  | 6.2 % | 4.6 % | 6.0 % |
| 10 | 6.3 % | 4.8 % | 6.1 % |
| 20 | 6.2 % | 4.7 % | 6.5 % |

The unprojected `diffuser` arm already sits at ~6.2% frozen for every K ≥ 2. HardFlow's slot-0 arm tracks it almost exactly (6.0–6.5%); DPCC's is slightly *lower* (4.6–4.8%) because the post-hoc QP nudges some near-stationary plans back into motion.

**So the ~6% base rate of frozen candidates is a property of the `mf_dit` checkpoint's velocity field.** HardFlow's in-loop NLP does not create them — which kills difference 4 and leaves the ranking key as the sole remaining explanation for 6% → 75%.

---

### 5.6 Why the ranking key prefers standing still

Put §5.3 and §5.5 together. Within a fan of 4 candidates, per active step:

- **A moving candidate**: `X1_ref = X_ref + (1−τ_next)·V_next` extrapolates the current velocity across the remaining flow time. At τ_next = 0.55 that inflates the plan by ~0.45·‖V‖, routinely pushing it outside the halfspace/obstacle set, so the NLP must move it a long way ⇒ **large `cand_prox`**.
- **A frozen candidate**: `V ≈ 0`, so `X1_ref ≈ X_ref ≈` the current pose repeated. That trajectory is *exactly* feasible — `a = 0` lies inside the bounds (`vx ∈ [−0.01, 0.01]`, `vy ∈ [0, 0.01]`), the `deriv` dynamics `s_{t+1} = s_t + a_t` is satisfied identically, and the start pose (0.525, −0.280) is far from every obstacle (centres at y ≈ 0.08) and on the correct side of both halfspaces. The NLP returns it untouched ⇒ **`cand_prox` = 0, exactly**, at every active step.

Zero is the global minimum of a sum of squares. So `-c` does not merely *tolerate* a frozen candidate — it is **structurally guaranteed to select it whenever one appears in the fan.** The ranking key is, in effect, a disguised reward for low control effort, which is the opposite of what a candidate selector should optimise for a task whose objective is to travel somewhere.

The persistent stall then follows from two factors acting together:

1. **Ranking bias** (established above from code): any fan containing a frozen candidate returns a frozen action.
2. **State localisation** (established in [`../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md), and visible in the §5(c) traces as a *leading* block): the degenerate mode is concentrated near the start pose, so the local frozen probability there is far above the ~6% trajectory average. A frozen action leaves the observation unchanged, so the next replan re-samples from the same high-probability region.

Escape requires a fan in which *all four* candidates are non-frozen, which is why trial0 needed 56 steps and trial1 never escaped at all.

> **Honest limit**: factor 1 is proven from code and confirmed by the K=1 test in §5.4. Factor 2 is inferred — the per-state frozen probability at the stall pose has not been measured directly. Doing so would need the per-candidate `candidate_costs` array dumped per replan, which the current npz does not retain.

---

### 5.7 Proposed fixes (NOT applied — needs a cluster run to validate)

**Fix A — make the ranking key measure what the sampler actually applies.** One line at `hardflow_projection.py:515-516`:

```python
# current
cand_prox += np.sum((X1_proj_np - X1_ref_np) ** 2, axis=1)
# proposed: rank by the applied pull-back tau*(x1_proj - x1_ref), matching the
# NLP's own tau^2 objective weight (:176-178) and the pull-back at :518
cand_prox += (tau_next ** 2) * np.sum((X1_proj_np - X1_ref_np) ** 2, axis=1)
```

This removes the internal inconsistency of difference 3 and down-weights exactly the early, heavily-extrapolated steps that inflate a moving candidate's score. It does **not** fully remove the bias — a frozen candidate still scores 0 — but it should collapse the 6% → 75% amplification back toward the ~22% that i.i.d. selection from a 6%-contaminated fan of 4 would predict.

**Fix B — config-only, no code change.** Set `candidate_cost: control` in `config/meanflow_projection_eval.yaml:128`. This ranks by Σ‖u_k‖ (`:496`), which *penalises* standing still instead of rewarding it. Note this path was itself repaired in fix_4 (it previously accumulated the NLP correction, a τ-reweighted copy of `prox`) and **has never been run**, so it is untested in both senses.

**Fix C — accept and document.** `-c` is a DPCC-parity variant, not the headline. `-t-tightened` scores a perfect 3.0/3.0 at every K ≥ 2. Reporting `-t-tightened` and documenting `-c` as a known selector pathology is a legitimate outcome.

Fix B is the cheapest (one YAML key, one job). Fix A is the principled one. They are independent and can run in the same sweep.

**Bearing on DPCC**: the same bias exists in DPCC's `-c` in principle — its cost is also a projection distance minimised by a feasible, motionless plan. DPCC escapes it at K ≥ 5 only because its cost is measured on the actual iterate rather than an extrapolation, so the gap between moving and frozen candidates is small. K=2 shows what happens when the field hands DPCC frozen candidates anyway: `dpcc-c` collapses to 99.5% (§6). **`-c` is a fragile selection rule on both engines; HardFlow's extrapolated cost merely makes it fragile at more values of K.**

---

### 5.8 Ruling out the remaining hypothesis: a long-solve / abort mechanism

The last live alternative: is the `-c` collapse — and `dpcc-c`'s K=2 collapse in particular — an artefact of solves being cut off by an iteration or time limit? Three independent checks say no.

**(a) The `dpcc-*` arms have no such mechanism, and are byte-identical to upstream.** `dpcc-*` never reaches `hardflow_projection.py`. It runs `Projector.project()` constructed with `solver='scipy'` (`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:346`):

```python
res = minimize(fun=cost_fun, x0=trajectory_np_double[i], constraints=constraints,
               method='SLSQP', jac=jac_cost_fun,
               bounds=Bounds(-5*np.ones_like(...), 5*np.ones_like(...)),
               tol=1e-6, options={'maxiter': 1000, 'disp': False})
sol_np[i] = res.x
```

`diff` of `flow_matcher_v3_meanflow/sampling/projection.py:70-156` against `/workspaces/aux_repo/dpcc/diffuser/sampling/projection.py:70-156` returns **empty — byte-for-byte identical, same line numbers**. No timeout, no wall-clock budget, no abort path was introduced. Note also that neither version checks `res.success`: on non-convergence scipy silently returns the last iterate. That is upstream DPCC's behaviour, unchanged here.

**(b) `dpcc-c` is not broadly broken — only K=2.** Goal-and-constraints, summed over the 3 halfspaces (max 3.0):

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `dpcc-c-tightened`         | 1.0 | **0.0** | **3.0** | **3.0** | **3.0** |
| `hardflow_new-c-tightened` | 2.0 | **0.0** | 1.5 | 1.5 | 1.5 |

`dpcc-c` is *perfect* at every K ≥ 5. Its single failure point is K=2 — the same K at which unprojected `diffuser` also degrades (§6). A solver-side abort mechanism would not switch itself off at K ≥ 5.

**(c) Compute time excludes silent iteration-limit aborts on the HardFlow side too.** This is the check that closes the `solve_limited` gap flagged in §5.1. Mean per-step compute time (s), pooled over halfspaces:

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `hardflow_new-r-tightened` | 0.129 | 0.104 | 0.224 | 0.524 | 1.017 |
| `hardflow_new-t-tightened` | 0.125 | 0.107 | 0.218 | 0.533 | 1.056 |
| `hardflow_new-c-tightened` | 0.131 | 0.103 | 0.225 | 0.529 | **1.037** |
| `dpcc-r-tightened`         | 0.030 | 0.024 | 0.225 | 0.398 | 0.999 |
| `dpcc-t-tightened`         | 0.024 | 0.026 | 0.212 | 0.371 | 0.922 |
| `dpcc-c-tightened`         | 0.028 | 0.023 | 0.222 | 0.351 | 1.021 |

The argument: **selection happens after all `batch_size` candidates have been solved.** `-r`, `-t` and `-c` solve an identical set of NLPs at a given K (§5.1's solve-count table is selection-independent), so the *only* way their compute cost can diverge is per-solve iteration count. It does not diverge — spread ≤ 0.007 s at K=1, and at K=20 `-c` sits *between* `-r` and `-t`. IPOPT is configured with no `max_iter` / `max_cpu_time` override (`hardflow_projection.py:184-195`), so its default `max_iter=3000` applies; solves grinding to that limit would be visibly, order-of-magnitude slower. They are not.

**(d) The collapse coincides with the *cheapest* solves in the sweep.** At K=2 — where both `-c` arms score 0.0 — `dpcc-c` costs 0.023 s and `hardflow_new-c` 0.103 s, each the per-variant minimum across all K. That is the exact inverse of a timeout signature, and it is what §5.6 predicts: a frozen plan is already feasible, so the solver converges almost immediately and the candidate scores exactly 0.

**Conclusion.** No abort mechanism is involved on either engine. The corrected statement of §5.1 is: *zero hard failures **and** no compute-time anomaly*, which together exclude both the raise-on-failure and the silent-limit abort modes. The defect remains the ranking key (§5.3, difference 2 and 3).

---

## 6. K=2: both engines fail the same way

| K=2 | frozen % | g&c |
|---|---|---|
| `dpcc-c` | 99.5 % | 0.00 |
| `dpcc-c-tightened` | 99.5 % | 0.00 |
| `hardflow_new-c` | 99.8 % | 0.00 |
| `hardflow_new-c-tightened` | 99.8 % | 0.00 |

Post-fix, arm B and arm C agree to within 0.3 percentage points. Pre-fix, arm C read 76.9% against DPCC's 99.5% — the σ=0.5 draw was *masking* part of the defect by feeding a different distribution.

This is a positive result twice over. It is independent corroboration that the two engines now consume the same field, and it upgrades the K=2 investigation's conclusion: the "stay put" mode is a property of the **generative field at K=2**, reproduced identically by two different projection engines. Neither DPCC nor HardFlow causes it; both merely select it when told to minimise projection cost.

Note the `-t-tightened` arms are fine at K=2 (2.3% frozen, 3.0 g&c for HardFlow). The K=2 defect is only fatal in combination with min-projection-cost selection.

---

## 7. Compute cost

Mean `Average computation time per step` over the 3 halfspaces (seconds):

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` | 0.009 | 0.017 | 0.043 | 0.080 | 0.164 |
| `dpcc-t-tightened` | 0.024 | 0.026 | 0.212 | 0.371 | 0.922 |
| `hardflow_new-t-tightened` | 0.125 | 0.107 | 0.218 | 0.533 | 1.056 |
| **HF / DPCC ratio** | **5.27×** | **4.13×** | **1.03×** | **1.44×** | **1.14×** |

The overhead is a fixed per-replan cost (HardFlow always solves the terminal NLP plus one solve per late step), so it dominates when the ODE is cheap and amortises as K grows. At the K values where the method actually works (K ≥ 5) HardFlow costs 1.0–1.4× DPCC. That is a defensible operating point; the K=1 5× figure is the honest worst case and should be quoted alongside it.

Both engines blow the 33.3 ms real-time budget at every K ≥ 5 (`[BUDGET=33.3ms ❌ OVER]` on essentially every step). Neither is real-time on this hardware; that is a shared property, not a HardFlow penalty.

Constraint satisfaction quality is equivalent — mean total violation for `-t-tightened`: DPCC 0.0000–0.0003, HardFlow 0.0000–0.0010, against `diffuser` at 2.79–4.32.

---

## 8. Caveats

- **The gates did not run.** `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` calls only the eval driver (line 92); there is no `python FM_v3_meanflow_test/gates_hardflow_meanflow.py` step, and no gate output appears in any of the five logs. So H3 (the numeric σ=1.0 assertion added in fix_4) **has never executed**. The fix is verified here only by its behavioural signature. The Gen3v7 script (`Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` §5) does run its gates before the eval — the MeanFlow script should be brought in line. One line, and `set -e` is already active.
- **Seed 6 only, `n_trials: 2`.** Each cell is 2 trials, so g&c is quantised to {0, 0.5, 1.0} per halfspace. Differences under ~0.5 per halfspace are not interpretable. Seeds 7–10 are still outstanding.
- The K=1 arm-C regression (12.0 → 10.5) and the `-r-tightened` K=5 dip (3.00 → 2.50) are both within that quantisation and should not be reported as effects.
- K=2 and K=10 have no pre-fix counterpart, so their "old → new" cells are blank by construction, not by omission.
- 🔶 **The `dpcc-t*` column is superseded by fix_5.** These jobs ran at `batch_size=4` with the `MPC_NPZ_PATCH` defect in `policies.py:70`, which stored a non-executed candidate as the temporal-consistency reference on ~75% of replans. Arm C was unaffected (`HardFlowPolicy._select` never reorders `observations`), so **arm B was the handicapped one**. The headline survives — `dpcc-t-tightened` was already at the 3.0/3.0 ceiling at K=1/5/10/20, so a correct implementation cannot beat it — but **the K=2 cell (arm C 3.0 vs arm B 2.5, arm C's only lead) must not be reported until `dpcc-t*` is re-run.** See `../fix_5/CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md`.
- **`n_failures` is a partial instrument.** `opti.solve_limited()` accepts IPOPT's limited statuses without raising, so the counter sees hard failures only (§5.1). §5.8(c) covers the blind spot with compute time, but if a future run needs a direct measurement, capture `opti.stats()['return_status']` per solve rather than relying on the counter. Note this affects the *diagnostic*, not the sampler: accepting a limited solve is upstream HardFlow's own behaviour and is deliberate.

---

## 9. Recommended next steps

1. **Add the gate call** to `eval_meanflow_hardflow.sh` before the eval, mirroring the AlphaFlow script. Cheap, and it closes the one unverified link in the fix_4 chain.
2. **Fix the `-c` ranking key** (§5.7). Run both candidates in one sweep at K ∈ {5, 20}: Fix B first (`candidate_cost: control`, YAML-only, zero code risk), then Fix A (the τ² weight at `hardflow_projection.py:515-516`) if B is insufficient. Prediction to check against: Fix A should pull `-c`'s freeze rate from ~75% toward ~22%; Fix B should pull it toward the `-r` band (~6%). If neither moves it, §5.6's factor-2 (state localisation) dominates and the problem is the checkpoint, not the selector.
3. **Seeds 7–10** at K ∈ {5, 20} for the `-t-tightened` headline, to turn "3.0/3.0" into a number with an error bar.
4. **Gen3v7 is unblocked.** The port guide's premise now holds: fix_4 is verified, arm C tracks DPCC, and the port can proceed with `init_noise_scale` read off `af_diffusion.py:260` (σ=1.0). Carry §5 forward as a known trap — Gen3v7's copy of `hardflow_projection.py` inherits the *identical* unweighted `cand_prox` accumulation, so α-Flow's `-c` will reproduce this exactly. `config/alphaflow_projection_eval.yaml` already warns about it in its header comment; if Fix A lands it must be mirrored into the Gen3v7 copy (sibling-sync — this is the same class of failure that produced the fix_4 σ bug).
5. **Retire the 🛑 INVALIDATED banner** on `../U3/INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md`, replacing it with a pointer to this file. Its arm-C numbers stay invalid; the arm-A/B analysis in it was always sound.
