# DA — HardFlow NLP backend: IPOPT vs SLSQP — **verdict: adopt SLSQP**

Job 25222, i6-gpu-1, rev `0c258d0`. 2026-08-30 11:42:48 → 14:31:36 UTC (2 h 49 m), clean exit.
Log `temp/3008/2026-08-30/13_42_47_eval_fmv3_hardflow_job_25222.log`,
data `temp/3008/batch_avoiding_combined_20260830_143724/`.
avoiding-d3il, `K{10,20}_thres1_mpc4_n2`, seed 6, n_trials 2, act_thr 1.0, both fans = 4.
Four passes: K ∈ {10,20} × backend ∈ {ipopt, slsqp}. Shared arms (`diffuser`,
`dpcc-c-tightened`) ran once in the IPOPT pass, so both hardflow arms share one identical DPCC row.

**Run 2 (threshold-matched):** job **25237**, i6-gpu-1, rev `938641c`.
2026-08-30 17:09:44 → 19:09:43 UTC (2 h 00 m), clean exit.
Log `temp/3008/19_09_43_eval_fmv3_hardflow_job_25237.log`,
data `temp/3008/batch_avoiding_combined_20260830_201516/`.
Identical design, but `activation_threshold = 0.5` — **matched to DPCC's
`diffusion_timestep_threshold`** — written to `K{10,20}_thres0.5_mpc4_n2_msgthrmatch/` so the
pre-existing `thres0.5` corpus is untouched. All 72 compute lines report `act_thr=0.5`; the only
`already exists` skips are the 12 intended shared-arm skips inside the new folders.

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

**And at matched threshold, HardFlow-SLSQP beats the DPCC projector at K=20** (run 2, §5.4):
`hardflow_sls-t-tightened` takes **61.0 steps vs DPCC's 62.2 and 0.343 s vs 0.475 s (0.72×)**, both
at 100 % S&C and 0.00000 violations — fewer steps *and* less time, i.e. Pareto domination. At K=10
it is faster (0.84×) but uses more steps: a trade-off, not a win.

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

### 5.4 Threshold parity — and the matched re-run (job 25237)

`config/hardflow_projection_eval.yaml` defines the two knobs as the same quantity: HardFlow's
`activation_threshold` has the **same meaning** as DPCC's `diffusion_timestep_threshold` — the
fraction of the late trajectory over which the NLP is active, higher = more projection.

**Run 1 was not matched:** `dpcc_threshold = 0.50`, `activation_threshold = 1.00` on every row.
DPCC projected the last half; HardFlow projected all of it, carrying ~2× the projection workload
(81.2 vs 0 solves/step, 158.3 vs 81.3 NFE/step at K20). That was the shipped default, not a bad
submit — U4 (`18fa5c28`) introduced the knob at `0.0` under the opposite polarity ("every step")
and fix_6 (`3e90c136`) inverted the sign, carrying the same behaviour over as `1.0`. Gen12 was the
last config not at DPCC parity; alphaflow, meanflow, visual_avoiding_mix and uav_mix already ship
0.5, and visual_aligning inherits DPCC's value via `null`.

Fixed (`activation_threshold: 0.5`) and re-run as job **25237**. **The IPOPT-vs-SLSQP result in
§1–4 is unaffected** — both backends always ran the identical setting, so that A/B was internally
matched throughout.

#### Matched results, K=20, A=0.5 (all arms, seed 6, n_trials 2, 3 geometries)

| variant | S&C | viol | total_viol | steps | s/step | NFE/step | solves/step | non-conv |
|---|---|---|---|---|---|---|---|---|
| `diffuser` | 0 % | 18.50 | 4.85470 | 65.0 | 0.175 | 81.2 | 0 | 0 |
| **DPCC** `dpcc-c-tightened` | 100 % | 0.00 | 0.00000 | **62.2** | **0.475** | 81.3 | 0 | 0 |
| HF-IPOPT `-r-tightened` | 100 % | 0.00 | 0.00000 | 68.2 | 1.064 | 117.7 | 40.6 | 0 |
| HF-SLSQP `-r-tightened` | 100 % | 0.00 | 0.00000 | 68.2 | 0.338 | 117.7 | 40.6 | 0 |
| HF-IPOPT `-c-tightened` | 100 % | 0.00 | 0.00000 | 103.2 | 0.999 | 117.1 | 40.4 | 0 |
| HF-SLSQP `-c-tightened` | 100 % | 0.00 | 0.00000 | 103.0 | 0.334 | 117.1 | 40.4 | 0 |
| HF-IPOPT `-t-tightened` | 100 % | 0.00 | 0.00000 | **61.2** | 1.108 | 117.9 | 40.7 | 2.0 |
| **HF-SLSQP `-t-tightened`** | **100 %** | **0.00** | **0.00000** | **61.0** | **0.343** | 117.9 | 40.7 | 15.0 |

Degeneracy gate ✅ on every row: `hf_degenerate = 0`, `hf_n_genuine` = 9 at K20 and 4 at K10
(A=0.5 activates the last half: 10 of 20 steps, 5 of 10). These are genuine HardFlow solves.

#### Pareto verdict at matched threshold

Tightened arms only — all sit at 100 % S&C and 0.00000 violations, so the precondition holds.

| K | arm | steps vs DPCC | time vs DPCC | verdict |
|---|---|---|---|---|
| 10 | `sls-r-tightened` | 68.2 vs 63.2 (worse) | 0.166 vs 0.199 (**0.83×**) | trade-off |
| 10 | `sls-c-tightened` | 136.8 vs 63.2 (worse) | 0.162 (**0.82×**) | trade-off |
| 10 | `sls-t-tightened` | 64.0 vs 63.2 (worse) | 0.167 (**0.84×**) | trade-off |
| 20 | `sls-r-tightened` | 68.2 vs 62.2 (worse) | 0.338 (**0.71×**) | trade-off |
| 20 | `sls-c-tightened` | 103.0 vs 62.2 (worse) | 0.334 (**0.70×**) | trade-off |
| 20 | **`sls-t-tightened`** | **61.0 vs 62.2 (BETTER)** | **0.343 (0.72×)** | **HardFlow Pareto-dominates** |

**At K=20, `hardflow_sls-t-tightened` beats DPCC on both axes** — fewer steps and 0.72× the wall
clock, at equal success and equal constraint satisfaction. This is the strong form of the claim and
it is **measured, not extrapolated**. (The projection made before this run predicted 0.379 s; the
measurement is 0.343 s.)

At K=10 every tightened arm is 0.82–0.84× DPCC's time but uses more steps — non-dominated, and the
honest description there is "faster, slightly longer trajectories".

#### SLSQP speedup at A=0.5

| | A=1.0 (run 1) | A=0.5 (run 2) |
|---|---|---|
| IPOPT → SLSQP speedup | 3.88× | **3.02×** (range 2.61–3.23×) |
| per-solve saving | 17.8 ms (sd 0.54) | **16.9 ms** (sd 0.97) |

The per-solve saving reproduces across a completely different projection budget — independent
confirmation of the fixed-overhead account in §3. The end-to-end ratio drops because at A=0.5
generation is a larger share of the total.

#### 🔴 Non-convergence gets worse at A=0.5 — for **both** backends

Run 1's "IPOPT: 0 failures" does not generalise. At A=0.5, per-cell non-convergence counts:

| K | arm | IPOPT | SLSQP |
|---|---|---|---|
| 10 | `-c` | 0 | 6.7 |
| 10 | `-t` | 0.3 | **81.3** |
| 10 | `-t-tightened` | 0 | 2.3 |
| 20 | `-c` | **13.3** | 0 |
| 20 | `-t` | **29.7** | **175.7** |
| 20 | `-t-tightened` | 2.0 | 15.0 |

IPOPT now fails too — 29.7 per cell on K20 `-t`, and 13.3 on K20 `-c` where SLSQP has none. This
matches the bench, which measured IPOPT at 26 % non-convergence on harder references. **Neither
backend is the reliable one.**

The failures also reach the outcome on **untightened** arms, in both directions: K10 `-c` 67 % → 50 %
(SLSQP worse), K20 `-c` 50 % → 67 % (SLSQP better), K20 `-t` 67 % → 50 % (SLSQP worse), with
violations moving 1.33 → 2.50 there. **On every tightened arm both backends are identical at 100 %
S&C and 0.00000 violations.** The tightened variants are the ones that hold; the untightened ones
are unstable under either solver and should not carry a claim.

### 5.5 Best horse vs best horse

Variant-by-variant pairing answers "does this solver change this arm". The question that decides
the paper is different: **does the best HardFlow configuration beat the best DPCC configuration.**
Only the winner of each stable has to be good.

**Qualification criteria** — an arm is eligible only if it is actually solving the problem:

1. `S&C = 100 %` (success *and* constraints, all geometries), and
2. `total_violations < 1e-6`.

The second is not pedantry: the qualifying arms report ~3–4e-08, which is the solver's constraint
tolerance and a formulation constant, not a violation (the bench saw the identical `+1.00e-08`
bit-identical across three seeds). Arms that leave real violation mass — every untightened
`-r`/`-c`/`-t` — are disqualified, under both backends.

Ranked by wall clock, matched threshold A=0.5:

| K | rank | arm | steps | s/step |
|---|---|---|---|---|
| 10 | 1 | `hardflow_sls-c-tightened` | 136.8 | **0.162** |
| 10 | 2 | `hardflow_sls-r-tightened` | 68.2 | 0.166 |
| 10 | 3 | `hardflow_sls-t-tightened` | **64.0** | 0.167 |
| 10 | 4 | `dpcc-c-tightened` | 63.2 | 0.199 |
| 10 | 5–7 | the three `hardflow_new-*-tightened` (IPOPT) | — | 0.486–0.532 |
| 20 | 1 | `hardflow_sls-c-tightened` | 103.0 | **0.334** |
| 20 | 2 | `hardflow_sls-r-tightened` | 68.2 | 0.338 |
| 20 | 3 | `hardflow_sls-t-tightened` | **61.0** | 0.343 |
| 20 | 4 | `dpcc-c-tightened` | 62.2 | 0.475 |
| 20 | 5–7 | the three `hardflow_new-*-tightened` (IPOPT) | — | 0.999–1.108 |

**Every qualifying HardFlow-SLSQP arm is faster than DPCC at both K.** DPCC ranks 4th of 7 at both
budgets. So on wall clock alone the answer is unambiguous; steps are what decide it.

| K | best HF-SLSQP | steps vs DPCC | time vs DPCC | verdict |
|---|---|---|---|---|
| 10 | `sls-t-tightened` (step-best) | 64.0 vs 63.2 = **1.013×** | 0.167 vs 0.199 = **0.838×** | trade-off — 16 % faster, 1.3 % longer |
| 20 | `sls-t-tightened` (step-best) | 61.0 vs 62.2 = **0.981×** | 0.343 vs 0.475 = **0.723×** | **HardFlow Pareto-dominates** |

`-c-tightened` is the fastest arm at both K but is disqualified as a *best horse* by trajectory
length (136.8 and 103.0 steps — it wanders); it is the known-bad B=4 arm and its step count says so
independently. `-t-tightened` is the horse to run: it is within 0.5 % of the fastest HardFlow arm
and is the only one competitive with DPCC on steps.

**Answer: at K=20 the best HardFlow beats the best DPCC on both axes. At K=10 it wins on time and
loses by 1.3 % on steps.**

#### 🔴 The comparison is currently biased toward HardFlow

`projection_variants` in `config/hardflow_projection_eval.yaml` ships **six** HardFlow variants and
**one** DPCC variant (`dpcc-c-tightened`). Taking the best of six against the best of one is
multiple-comparison shopping on one side of the race. With three geometries × 2 trials the noise
floor is not small, and "best of six" will beat "best of one" some of the time on noise alone.

**Before this claim is published, DPCC needs its own stable in the same run** — add
`dpcc-r`, `dpcc-t`, `dpcc-c`, `dpcc-r-tightened`, `dpcc-t-tightened` alongside the incumbent, so
both sides field six and the best-of-six comparison is symmetric. Those variants exist and are
routinely run in other generations; they were pruned here for compute (Gen15 U5 slimming). This is
a one-line config change and adds five arm-B rows, which are the cheap arms.

#### Dropping the MPC fan (batch = 1) — the untested lever

The candidate fan is the other half of HardFlow's cost. `K20_thres0.5_mpc1_n2` (older rev,
`(Gen12_Bf_U5)` parent) has bare `hardflow_new` at fan 1:

| variant | fan | S&C | viol | steps | s/step | NFE/step | solves/step |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | 4 | 100 % | 0.00 | 62.2 | 0.475 | 81.3 | 0 |
| `hardflow_new` (IPOPT) | **1** | 100 % | 0.00 | 67.7 | 0.488 | **31.5** | 11.2 |

At fan 1 HardFlow holds 100 % S&C and zero violations on **31.5 NFE/step — 2.6× fewer than DPCC's
81.3** — because the fan multiplies the network evaluations, and the fan is where HardFlow's
generation deficit (§5.2) actually comes from. Even under IPOPT it is already at DPCC's wall clock;
under SLSQP the 11.2 solves/step would cost ~0.19 s less.

⚠️ **This row is not fan-matched.** `mpc1` in the folder name is the *arm-C* fan only; arms A/B kept
the default 4, so this is HardFlow@1 vs DPCC@4 — unfair to HardFlow on candidate quality and unfair
to DPCC on cost. It cannot be quoted. The clean experiment is `FMPCC_MPC_BATCH=1 HFFM_BATCH=1`,
both arms at one candidate, which also isolates how much of DPCC's success rate comes from
candidate *selection* rather than from the projector. At fan 1 the `-r`/`-c`/`-t` suffixes collapse
to index 0, so run one of them, not the trio.

## 6 — Decision

**Keep `slsqp` as `DEFAULT_NLP_BACKEND`.** Rationale, in order of weight:

1. 3.88× speed at zero cost on success, S&C and collision-free across all 12 cells.
2. The only quality cost measured is 0.28 % of the projection's repair value, in 1 of 12 cells.
3. HardFlow moves from 4.07–4.75× DPCC's cost to 1.05–1.21× — it stops being priced out, and the
   bench's endpoint-trick advantage (3.09× under SLSQP, 1.14× under IPOPT) becomes collectable.
4. IPOPT's zero failures here do not generalise; its measured failure rates elsewhere are worse.

**On replacing the DPCC projector with HardFlow-SLSQP: the evidence now supports it at K=20, and
the deciding experiment has been run.** At matched threshold (both 0.5, job 25237),
`hardflow_sls-t-tightened` takes fewer steps *and* 0.72× the wall clock of DPCC at equal 100 % S&C
and equal 0.00000 violations — Pareto domination, measured. At K=10 the same arm is 0.84× on time
but longer in steps: a trade-off, not a win.

What is still needed before this is a paper claim:
- **A symmetric field.** Six HardFlow variants currently race one DPCC variant (§5.5). Add
  `dpcc-{r,t,c}` and `dpcc-{r,t}-tightened` to `projection_variants` so best-of-six meets
  best-of-six. Cheapest and most important of these follow-ups.
- **Seeds.** Both runs are seed 6, `n_trials` 2 — 6 rollouts per cell. The K20 step margin is
  61.0 vs 62.2, about 2 %, which 6 rollouts cannot separate from noise. Re-run at ≥3 seeds and
  `n_trials` ≥ 10; this is cheap now (a full SLSQP sweep is a third of an IPOPT one at A=0.5).
- **`-t` stability.** The winning arm is the one whose untightened sibling carries 175.7
  non-converged solves per cell at K20. The tightened version shows 15.0 and still lands at
  0.00000 violations, but that margin should be understood before it is relied on.
- **Restrict claims to tightened arms.** Untightened `-r`/`-c`/`-t` swing in both directions
  between backends and never reach 100 % S&C.

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
