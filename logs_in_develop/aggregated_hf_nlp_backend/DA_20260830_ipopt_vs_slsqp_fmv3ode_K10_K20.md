# DA — HardFlow NLP backend: IPOPT vs SLSQP — **verdict: adopt SLSQP**

Job 25222, i6-gpu-1, rev `0c258d0`. 2026-08-30 11:42:48 → 14:31:36 UTC (2 h 49 m), clean exit.
Log `temp/3008/2026-08-30/13_42_47_eval_fmv3_hardflow_job_25222.log`,
data `temp/3008/batch_avoiding_combined_20260830_143724/`.
avoiding-d3il, `K{10,20}_thres1_mpc4_n2`, seed 6, n_trials 2, act_thr 1.0, both fans = 4.
Four passes: K ∈ {10,20} × backend ∈ {ipopt, slsqp}. Shared arms (`diffuser`,
`dpcc-c-tightened`) ran once in the IPOPT pass, so both hardflow arms share one identical DPCC row.

Prior: bench job 25121 — `Gen12/Solver_Bench/RESULTS_20260827_solver_bench_ipopt_vs_slsqp.md`.

---

## VERDICT

**Use SLSQP. The trade is 3.88× speed for 0.28 % of the projection's constraint-repair value, in
1 of 12 cells, and 0 % in the other 11.**

| | measured |
|---|---|
| Speed gain | **3.88×** end-to-end (range 3.78–4.00× over 12 cells) |
| Success / S&C / collision-free change | **0 in all 12 cells** |
| Cells with any constraint degradation | **1 of 12** (K10 `-t`) |
| Size of that degradation | total violation 0.00046 → 0.01037 per rollout |
| …as a share of what projection repairs | **0.28 %** (unprojected baseline is 3.52) |
| Cost vs DPCC projector | 4.07–4.75× → **1.05–1.21×** |

This is not a close call. The quality loss is three orders of magnitude smaller than the quantity
being protected, and it is confined to one selection variant.

**The bench's own decision rule is now satisfied.** `RESULTS_20260827` §3 set it out: *"cost
collapses to parity → cost stops being the story; S&C and steps decide."* Cost did collapse to
parity (below), S&C is identical in all 12 cells, and steps are identical in 11 of 12. The rule
resolves to adopt.

---

## 1 — The bench's extrapolation is confirmed

`RESULTS_20260827` §3 projected HF/DPCC would fall from 2.32× to **1.22×** after the swap, and
listed that as its top *not settled* item: *"only an actual arm-C-with-SLSQP eval run confirms it."*
This is that run.

| | bench projection | measured here |
|---|---|---|
| HardFlow / DPCC after swap | 1.22× | **1.21× (K10), 1.05× (K20)** |
| IPOPT / SLSQP per solve | 4.33× | 5.4× (lower bound, §3) |

The projection was accurate to 1 % at K10 and conservative at K20. That item is closed.

## 2 — How the two solvers work

**Same problem.** HardFlow minimises `0.5 · reg_scale · τ² · ‖x − x_ref‖²` over the constraint set.
The prefactor is a positive scalar, which does not move an argmin, so the NLP is exactly the
Euclidean projection `Π_S(x_ref)` — what DPCC's `Projector.project` computes with `Q = I`,
`r = −x_ref`. `_solve_slsqp` therefore accepts `tau` and ignores it: exact, not an approximation.
Both backends get the same `constraint_list` and, via `_StubNormalizer`, the same `mins`/`maxs`.

| | IPOPT (`hardflow_new-*`) | SLSQP (`hardflow_sls-*`) |
|---|---|---|
| via | CasADi `Opti`, NLP built once symbolically | DPCC `Projector` → `scipy.optimize.minimize` |
| method | interior-point (log-barrier) | active-set SQP |
| inner loop | barrier subproblems, each a Newton/KKT solve | QP from quadratic Lagrangian model + linearised constraints, line search |
| Hessian | limited-memory (L-BFGS) | scipy BFGS update |
| designed for | large sparse NLPs | small dense problems |
| on non-convergence | keeps last iterate | keeps last iterate |

Our problem is **44 dense DOF** — the regime where interior-point per-call setup dominates. The
bench measured the signature directly: triple the problem difficulty and IPOPT slows **1.14×**
while SLSQP slows **3.09×**. SLSQP's time is nearly all work; IPOPT's is nearly all overhead.

Consequence the bench drew, still standing: HardFlow's central optimisation — projecting the clean
endpoint instead of a noisy iterate — is worth **3.09× to SLSQP and 1.14× to IPOPT**. Shipping
HardFlow on IPOPT means shipping the one solver that cannot cash in HardFlow's own design claim.

## 3 — Speed

| K | arm | IPOPT s/step | SLSQP s/step | speedup | S&C IPOPT | S&C SLSQP |
|---|---|---|---|---|---|---|
| 10 | `-r` | 0.963 | 0.246 | 3.91× | 100 % | 100 % |
| 10 | `-c` ⚠️ | 0.930 | 0.246 | 3.78× | 83 % | 83 % |
| 10 | `-t` | 0.956 | 0.248 | 3.86× | 50 % | 50 % |
| 10 | `-r-tightened` | 0.990 | 0.249 | 3.97× | 100 % | 100 % |
| 10 | `-c-tightened` | 0.985 | 0.252 | 3.91× | 100 % | 100 % |
| 10 | `-t-tightened` | 1.008 | 0.252 | 4.00× | 100 % | 100 % |
| 20 | `-r` | 1.941 | 0.503 | 3.86× | 100 % | 100 % |
| 20 | `-c` ⚠️ | 1.898 | 0.502 | 3.78× | 100 % | 100 % |
| 20 | `-t` | 1.912 | 0.505 | 3.78× | 67 % | 67 % |
| 20 | `-r-tightened` | 1.987 | 0.508 | 3.91× | 100 % | 100 % |
| 20 | `-c-tightened` | 1.978 | 0.511 | 3.87× | 100 % | 100 % |
| 20 | `-t-tightened` | 2.030 | 0.516 | 3.93× | 100 % | 100 % |

⚠️ `-c` at B=4 is the known-bad arm (49 % timeouts, `HF_Batch_Parity/`).

**Against DPCC**, same job, same DPCC row:

| K | `diffuser` (unprojected) | `dpcc-c-tightened` | HF IPOPT `-r` | HF SLSQP `-r` |
|---|---|---|---|---|
| 10 | 0.089 s, 16.7 % S&C | 0.203 s, 100 % | 0.963 s = 4.75× DPCC | 0.246 s = **1.21× DPCC** |
| 20 | 0.174 s, 0 % S&C | 0.478 s, 100 % | 1.941 s = 4.07× DPCC | 0.503 s = **1.05× DPCC** |

**Where the time goes.** HardFlow runs `K × B` solves per env step (B=4); measured 40.6 at K=10 and
81.2 at K=20 (predicted 40, 80). Δt ÷ solves-per-step gives the per-solve saving, generation
cancelling out:

| K | `-r` | `-c` | `-t` | `-r-t'nd` | `-c-t'nd` | `-t-t'nd` |
|---|---|---|---|---|---|---|
| 10 | 17.7 ms | 16.9 | 17.4 | 18.3 | 18.1 | 18.6 |
| 20 | 17.7 ms | 17.2 | 17.3 | 18.2 | 18.1 | 18.6 |

Mean 17.8 ms, sd 0.54 ms — **constant per solve, unchanged when K doubles**. Fixed per-call
overhead, not extra optimisation work; the same conclusion the bench reached on the difficulty
axis, now on a third axis. Absolute, using `diffuser` as generation reference: IPOPT ≈ 21.7 ms,
SLSQP ≈ 4.0 ms, ratio ≥ 5.4× (lower bound — `diffuser` generates at batch 1, so true generation is
larger and the true projection ratio higher).

## 4 — Quality: what exactly is lost

Pairwise IPOPT → SLSQP, averaged over three geometries. `≡` = identical.

| K | arm | Succ | S&C | CF | violations | total_viol | steps | solves | non-conv |
|---|---|---|---|---|---|---|---|---|---|
| 10 | `-r` | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ |
| 10 | `-c` | ≡ | ≡ | ≡ | ≡ | ≡ | 75.17→75.33 | +13 | ≡ |
| 10 | **`-t`** | ≡ | ≡ | ≡ | **0.67→1.00** | **0.00046→0.01037** | **58.67→61.50** | **+227** | **+13.3** |
| 10 | `-r-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ |
| 10 | `-c-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ |
| 10 | `-t-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | +2.0 |
| 20 | `-r` | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ | ≡ |
| 20 | `-c` | ≡ | ≡ | ≡ | ≡ | ≡ | 72.17→71.67 | −80 | ≡ |
| 20 | `-t` | ≡ | ≡ | ≡ | ≡ | 0.00121→0.00125 | 63.67→64.33 | +107 | +0.3 |
| 20 | `-r-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | 69.83→69.67 | −27 | ≡ |
| 20 | `-c-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | 72.50→72.33 | −27 | ≡ |
| 20 | `-t-tightened` | ≡ | ≡ | ≡ | ≡ | ≡ | 62.17→62.33 | +27 | +1.3 |

### The magnitude, in context

`total_violations` is the summed violation magnitude per rollout. The scale is set by the
unprojected arm:

| K | variant | n_viol | total_viol |
|---|---|---|---|
| 10 | `diffuser` (no projection) | 16.17 | **3.52316** |
| 10 | `dpcc-c-tightened` | 0.00 | 0.00000 |
| 10 | `hardflow_new-t` (IPOPT) | 0.67 | 0.00046 |
| 10 | `hardflow_sls-t` (SLSQP) | 1.00 | 0.01037 |
| 20 | `diffuser` | 18.50 | 4.85470 |
| 20 | `hardflow_new-t` / `hardflow_sls-t` | 0.67 / 0.67 | 0.00121 / 0.00125 |

Projection removes 3.52 units of violation. IPOPT leaves 0.00046 behind (**99.99 % repaired**);
SLSQP leaves 0.01037 (**99.71 % repaired**). **SLSQP gives back 0.0099 units — 0.28 % of what
projection is there to remove.** Every other arm is at exactly 0.00000 under both backends.

That is the whole quality cost. It is not "small enough to accept with reservations"; it is
0.28 % of the protected quantity, in the worst of twelve cells, in an arm that scores 50 % S&C
under *both* solvers and so is failing for reasons that have nothing to do with the backend.

### Non-convergence

| backend | NLP solves | non-converged | rate |
|---|---|---|---|
| IPOPT | 299,720 | 0 | 0.000 % |
| SLSQP | 300,440 | 51 | 0.017 % |

All 51 in `-t` arms (K10 `-t` 40, K10 `-t-tightened` 6, K20 `-t-tightened` 4, K20 `-t` 1); zero in
`-r` or `-c`. Two things stop this from being a reason to keep IPOPT:

1. **Failures mostly do not propagate.** K10 `-t-tightened` has 6 failures and zero movement on
   every metric; K20 `-t` and `-t-tightened` have 1 and 4 and move only at the ±0.2 % noise level.
   One of the four cells with failures degraded.
2. **IPOPT is not failure-free in general.** Its 0/299,720 here is specific to this configuration.
   The bench measured IPOPT at **26 % non-convergence** on `iterate` references with a 3× larger
   violation than SLSQP on the same inputs, and Q&A 2b already recorded IPOPT at **12.5–13.5 %** on
   visual-avoiding TL untightened. The choice is not reliable-IPOPT vs flaky-SLSQP; both keep a
   possibly-infeasible last iterate on failure, and IPOPT's failure mode has been the worse one
   wherever it has been measured.

### Sub-1 % drift where nothing failed

`-c` at both K and the K20 tightened arms move `steps`/`solves` by ≤0.8 %, bidirectionally, with
zero failures. That is the ~1e-3 per-solve disagreement the bench measured (mean 3.4e-4, max
1.0e-3 over 100 solves) compounding through a closed loop. No outcome changes.

## 5 — Three-way: DPCC projector vs HardFlow-IPOPT vs HardFlow-SLSQP

All three arms in the same job, same seed, same geometries. DPCC (`dpcc-c-tightened`) projects
**once, post-hoc**, on the finished trajectory; HardFlow projects **in-loop**, at every ODE step,
running `K × B` solves per environment step.

### 5.1 Outcome — all three are equivalent

| K | arm | S&C | Succ | violations | total_viol | steps | s/step |
|---|---|---|---|---|---|---|---|
| 10 | `diffuser` (no projection) | 17 % | 100 % | 16.17 | 3.52316 | 64.5 | 0.089 |
| 10 | **DPCC** `dpcc-c-tightened` | 100 % | 100 % | 0.00 | 0.00000 | **63.2** | **0.203** |
| 10 | HF-SLSQP `-t-tightened` | 100 % | 100 % | 0.00 | 0.00000 | 65.0 | 0.252 |
| 10 | HF-SLSQP `-r` | 100 % | 100 % | 0.00 | 0.00000 | 68.8 | 0.246 |
| 10 | HF-IPOPT `-t-tightened` | 100 % | 100 % | 0.00 | 0.00000 | 65.0 | 1.008 |
| 10 | HF-IPOPT `-r` | 100 % | 100 % | 0.00 | 0.00000 | 68.8 | 0.963 |
| 20 | `diffuser` | 0 % | 100 % | 18.50 | 4.85470 | 65.0 | 0.174 |
| 20 | **DPCC** `dpcc-c-tightened` | 100 % | 100 % | 0.00 | 0.00000 | **62.2** | **0.478** |
| 20 | HF-SLSQP `-t-tightened` | 100 % | 100 % | 0.00 | 0.00000 | **62.3** | 0.516 |
| 20 | HF-SLSQP `-r` | 100 % | 100 % | 0.00 | 0.00000 | 68.8 | 0.503 |
| 20 | HF-IPOPT `-t-tightened` | 100 % | 100 % | 0.00 | 0.00000 | 62.2 | 2.030 |
| 20 | HF-IPOPT `-r` | 100 % | 100 % | 0.00 | 0.00000 | 68.8 | 1.941 |

On constraint quality the three are indistinguishable: **100 % S&C and exactly 0.00000 residual
violation for DPCC and for every tightened HardFlow arm, under both backends.** Neither the
projection strategy nor the solver separates them. The unprojected row shows what is at stake —
16–18 violations and 3.5–4.9 units of violation mass per rollout.

**Pareto (S&C held equal → fewer steps and lower time both required):**

- K=10 — DPCC wins both axes (63.2 steps / 0.203 s vs HF-SLSQP's best 65.0 / 0.252). **DPCC
  Pareto-dominates.**
- K=20 — DPCC 62.2 steps / 0.478 s, HF-SLSQP `-t-tightened` 62.3 steps / 0.516 s. Steps are a tie
  (0.2 %); DPCC is 8 % faster. **DPCC still dominates, but the margin is now wall-clock only, and
  it is 8 %.** Under IPOPT the same comparison was 4.2×.

No HardFlow arm Pareto-dominates DPCC at either K. The swap did not make HardFlow win; it made the
gap small enough to argue about.

### 5.2 Cost decomposition — the surprise

Splitting each arm into generation and projection. Generation cost is calibrated from the
projection-free `diffuser` arm: 2.20 ms/NFE at K=10 and 2.14 ms/NFE at K=20 (mean **2.17**), linear
and consistent across the two budgets. Self-check: applying that rate back to `diffuser` leaves a
residual projection cost of +0.001 s (K10) and −0.002 s (K20) — i.e. zero, as it must be for an arm
that does no projection.

| K | arm | NFE/step | solves/step | total s | generation s | **projection s** | proj vs DPCC |
|---|---|---|---|---|---|---|---|
| 10 | DPCC | 40.6 | — | 0.203 | 0.088 | **0.115** | 1.00× |
| 10 | HF-SLSQP `-r` | 77.1 | 40.6 | 0.246 | 0.167 | **0.079** | **0.69×** |
| 10 | HF-SLSQP `-t-tightened` | 77.2 | 40.6 | 0.252 | 0.167 | 0.085 | 0.74× |
| 10 | HF-IPOPT `-r` | 77.1 | 40.6 | 0.963 | 0.167 | 0.796 | 6.94× |
| 10 | HF-IPOPT `-t-tightened` | 77.2 | 40.6 | 1.008 | 0.167 | 0.841 | 7.33× |
| 20 | DPCC | 81.3 | — | 0.478 | 0.176 | **0.301** | 1.00× |
| 20 | HF-SLSQP `-r` | 158.3 | 81.2 | 0.503 | 0.343 | **0.161** | **0.53×** |
| 20 | HF-SLSQP `-t-tightened` | 158.5 | 81.3 | 0.516 | 0.343 | 0.172 | 0.57× |
| 20 | HF-IPOPT `-r` | 158.3 | 81.2 | 1.941 | 0.343 | 1.598 | 5.30× |
| 20 | HF-IPOPT `-t-tightened` | 158.5 | 81.3 | 2.030 | 0.343 | 1.686 | 5.60× |

**HardFlow-SLSQP's projection is cheaper than DPCC's projection — 0.69× at K=10, 0.53× at K=20 —
while doing roughly 20× as many solves.** DPCC runs one projection per candidate on the finished
trajectory; HardFlow runs 40–80 per environment step. HardFlow still comes out ahead on projection
cost because each of its solves is on a *near-feasible predicted endpoint* while DPCC's is on a
*noisy iterate*. That is HardFlow's central design claim, and this is the first time it has been
measured end-to-end on our constraint set. The bench priced the same effect at 3.09× for SLSQP.

**So HardFlow's remaining cost disadvantage is not the projector at all — it is generation.**
HardFlow needs 158.3 NFE/step against DPCC's 81.3, almost exactly 2×, because each in-loop ODE step
evaluates the network for both the velocity field and the endpoint estimate the projection needs.
At K=20: HF-SLSQP spends 0.343 s generating and 0.161 s projecting; DPCC spends 0.176 s generating
and 0.301 s projecting. HardFlow saves 0.140 s on projection and gives back 0.167 s on generation —
which is the entire 8 % gap in §5.1.

### 5.3 What the swap actually changed

| | under IPOPT | under SLSQP |
|---|---|---|
| HF projection vs DPCC projection | 5.30–6.94× | **0.53–0.69×** |
| HF total vs DPCC total | 4.07–4.75× | 1.05–1.21× |
| Where HF's cost sits | the solver | generation (2× NFE) |
| HardFlow's endpoint trick | swallowed by per-call overhead (worth 1.14×) | collectable (worth 3.09×) |

Under IPOPT the projector was 5–7× DPCC's and the endpoint argument was unobservable. Under SLSQP
the projector is *better* than DPCC's and the argument shows up in the measurement. The remaining
gap is a generation-side cost, which is a different problem with different levers (NFE reduction,
caching the endpoint estimate) and is not a solver question at all.

⚠️ The generation/projection split is **derived**, not instrumented: total time and NFE are measured,
the split uses the calibrated 2.17 ms/NFE. The calibration is stable across K (2.20 vs 2.14) and
reproduces zero projection cost for `diffuser`, but a direct `proj_ms` counter on the D3IL arms — as
the UAV path already has — would settle it. Worth adding before this decomposition is cited.

### 5.4 🔴 This run is NOT threshold-matched — HardFlow vs DPCC cannot be read from it

`config/hardflow_projection_eval.yaml` defines the two knobs as the same quantity:

```
diffusion_timestep_threshold: 0.5          # DPCC (arms A/B)

# ── U4 late-activation threshold (fix_6: DPCC polarity) ──
# SAME meaning as DPCC's diffusion_timestep_threshold: the fraction of the (late)
# trajectory over which the NLP is active — HIGHER = MORE projection.
#   1.0 -> every step        0.5 -> last half (== DPCC diffusion_timestep_threshold 0.5)
activation_threshold: 1.0
```

Every row in this run confirms it: `dpcc_threshold = 0.50`, `activation_threshold = 1.00`.
**DPCC projected the last half of the trajectory; HardFlow projected all of it.**

This is the shipped default, not a mistake in the submission — and the default has never been DPCC
parity. `activation_threshold` was introduced in U4 (`18fa5c28`) with the opposite polarity and a
default of `0.0` = "every ODE step"; Fix 6 (`3e90c136`) inverted the polarity to match DPCC's
convention and set the default to `1.0`, which is the *same behaviour* under the new sign. The
config names `0.5` explicitly as the DPCC-parity value, and matching it has always required setting
`HFFM_ACT_THRESHOLD=0.5` by hand.

**Consequence for §5.1–5.3:** HardFlow was handed roughly twice DPCC's projection workload —
81.2 solves/step and 158.3 NFE/step against DPCC's 0 and 81.3 — and then compared on wall clock.
The IPOPT-vs-SLSQP result is unaffected (both backends ran the identical setting, so the A/B is
internally matched and every number in §1–4 stands). But the **HardFlow-vs-DPCC** rows in §5.1–5.3
are not a fair comparison and must not be quoted as one.

What can still be said from this run:

- At 2× the projection workload, HardFlow-SLSQP lands within **8 % of DPCC's wall clock** at K=20
  (0.516 vs 0.478 s) at equal 100 % S&C and equal 0.00000 violations.
- Its **projection** is already cheaper than DPCC's — 0.53× at K=20 (§5.2) — while doing ~20× the
  solves.
- Its remaining deficit is generation, and that deficit is exactly the active-step count (§5.2),
  which is what `activation_threshold` controls.

What cannot be said: whether HardFlow beats, ties, or loses to DPCC. This run does not test it.

## 6 — Decision

**Keep `slsqp` as `DEFAULT_NLP_BACKEND`.** Rationale, in order of weight:

1. 3.88× speed at zero cost on success, S&C and collision-free across all 12 cells.
2. The only quality cost measured is 0.28 % of the projection's repair value, in 1 of 12 cells.
3. HardFlow moves from 4.07–4.75× DPCC's cost to 1.05–1.21× — it stops being priced out, and the
   bench's endpoint-trick advantage (3.09× under SLSQP, 1.14× under IPOPT) becomes collectable.
4. IPOPT's zero failures here do not generalise; its measured failure rates elsewhere are worse.

**On replacing the DPCC projector with HardFlow-SLSQP: undecided — this run cannot answer it, by
construction.** HardFlow ran at `activation_threshold = 1.0` against DPCC at `0.5`, i.e. twice the
projection workload (§5.4). The comparison has to be re-run matched before any verdict.

The indicators from this run are favourable: carrying 2× the workload, HardFlow-SLSQP still lands
within 8 % of DPCC on wall clock at equal S&C and equal violations, and its projection term is
already 0.53× DPCC's. Its whole remaining deficit is generation, which scales exactly with the
active-step count — the knob that was left at maximum.

**The deciding run** — one job, cheap, fully specified:
`HFFM_ACT_THRESHOLD=0.5` (DPCC parity) at K20 with `FMPCC_HF_NLP_BACKEND=slsqp`, tightened variants,
DPCC at its existing 0.5, ≥3 seeds and `n_trials` ≥ 10, reporting **steps and violations alongside
S&C** and confirming the degeneracy gate on every row. Worth sweeping
`HFFM_ACT_THRESHOLD ∈ {0.1, 0.25, 0.5}` in the same job, since lower values cut solves and NFE
together.

**Conditions attached:**
- Report `nlp_failures` on every SLSQP run — it is the only signal that a plan lost the
  terminal-solve guarantee.
- Treat `-t` as the watch arm. All 51 failures and the single degraded cell are there.
- Report `violations`/`total_violations` alongside S&C. At n_trials = 2 the S&C axis is quantised
  to 0/50/100 % and did not register the K10 `-t` change; the violation columns did.

## 7 — Caveats and open items

- n_trials = 2, seed 6, single scene (avoiding-d3il), single threshold (`thres1`). Timing is
  well-sampled (~390 step measurements per variant, speedup sd small, per-solve saving sd 0.54 ms);
  success rates are not. The verdict rests on the violation-magnitude argument (§4), which does not
  depend on the coarse S&C axis.
- `-c` rows carry the known-bad B=4 caveat.
- Degeneracy gate ✅ on every row (`hf_degenerate = 0`, `hf_n_active = K`, `hf_n_genuine = K−1`) —
  all rows are genuine HardFlow math.
- **Open:** why are all 51 non-convergences in `-t` and none in `-r`/`-c`?
- **Not addressed here:** whether HardFlow beats DPCC on S&C and steps. The bench predicted the
  swap would "remove HardFlow's cost excuse without rescuing it". Cost parity is now confirmed; the
  S&C/steps question is unchanged by this run and stands where chapters 1–3 left it.

---

Other jobs in this download: **25215** `eval_mix_visual_aligning` (rev `73adff1`) crashed —
`MIX_PROJ_T='' is not a float`, both T passes; **25216** (rev `81e9ea7`, after hotfix `81e9ea73`)
cleared it and was still running at download time. `MIX_PROJ_T=''` and the
`HFFM_FLOW_STEPS='10 20'` crash are the same bug class — an env var hitting a bare `float()`/`int()`
at config-import time. `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` still has the
un-fixed list variant.
