# Gen12 — K-sweep {2,5,10}: the low-K regime refutes the contribution

**Date:** 2026-07-25 · **Type:** results / insight · **Status:** preliminary (n=2, single seed, one confound)
**Run:** job 23815, git `dbd21b1`, node i6-gpu-1 · sweep K ∈ {2,5,10}, seed 6, n_trials=2, 3 halfspace variants
**Log:** `temp/Gen12/2507/00_21_46_eval_fmv3_hardflow_job_23815.log`
**Results:** `…/plans/flow_matching_v3_hardflow/H8_K10_…FlowMatchingODE/6/results/halfspace_*/K{2,5,10}_n2/`
**Model:** FMv3ODE checkpoint `…FlowMatchingODE_a1.5_b1.0_aw10`, seed 6, step 98000

> This is the run the earlier K=20 smoke (fix_2 RESULTS) said was needed: the **low-K regime**, where
> the field is coarse and in-loop guidance was *hypothesised* to beat post-hoc projection. It does the
> **opposite**. Still n=6 episodes/arm/K and one batch confound (§5) — read the caveats before quoting.

---

## 0. First: the K-override is now confirmed on hardware

fix_2's debug left the K-override verified only by local simulation (the K=20 run had a cluster
config hand-edit). This sweep applied **K=2, then 5, then 10** correctly (`matched K … = 2/5/10`,
separate `K2/K5/K10` result dirs). The fix_3 sweep sbatch works and the plan-block K path is live.

## 1. Headline — goal+constraints success vs K (mean over 3 halfspace variants, n=2 each)

| K | A `diffuser` | B `dpcc-c-tightened` | C `hardflow_new` |
|---|---|---|---|
| 2  | 0.17 | **1.00** | 0.33 |
| 5  | 0.00 | **1.00** | 0.50 |
| 10 | 0.17 | **1.00** | 1.00 |
| 20¹ | 0.00 | **1.00** | 1.00 |

¹ K=20 from the earlier fix_2 run, same setup.

**Two facts jump out:**

- **B (post-hoc DPCC projection) is rock-solid at EVERY K** — 100% goal + 100% constraints from
  K=2 up. It needs almost no ODE steps to work.
- **C (in-loop constrained sampling) is a monotone ramp: 0.33 → 0.50 → 1.00 as K goes 2→10.** It
  only catches B at K≥10, and it **fails badly at low K.**

## 2. The pre-registered hypothesis is refuted

fix_2 RESULTS §5 pre-registered the interesting test: *"if at low K arm C's success stays at/above
arm B while arm B degrades, that is the first real evidence for the contribution."*

**The data is the exact opposite:** at low K, **B does not degrade at all** and **C collapses.** So
in-loop constrained sampling does **not** rescue coarse-field trajectories — post-hoc projection is
the robust low-K method. This is the strongest form of the Gen13 "projection dominates" finding:
the projection delivers safety+success regardless of field quality; the in-loop sampler inherits the
field's coarseness and needs a fine field (high K) to work at all.

## 3. Where C fails, and how (per-variant, the asymmetry)

| K | variant | C: goal | C: constraints | note |
|---|---|---|---|---|
| 2 | top-right `/` | **0.0** | 0.0 | never reaches goal (steps=0), 3.5 violations |
| 2 | top-left `\` | 1.0 | 1.0 | fine |
| 2 | both `/\` | 1.0 | **0.0** | reaches goal but drives through the obstacle (1 viol) |
| 5 | top-right `/` | **0.0** | 0.0 | never reaches goal (no viol — just stuck) |
| 5 | both `/\` | 1.0 | **0.5** | half the runs violate |
| 10 | all | 1.0 | 1.0 | fine (matches B) |

Two distinct low-K failure modes for C: **(a) fails to reach the goal** (`top-right`, K=2/5 →
steps=0), and **(b) reaches the goal but is unsafe** (`both`, K=2/5). The `top-right /` geometry is
its consistent worst case; `top-left \` works even at K=2. So the breakdown is
**geometry-dependent**, not uniform.

## 4. The mechanistic insight — NLP feasibility ≠ closed-loop success

**Arm C's NLP never failed — 0 failures at every K** (148–1460 solves per variant, all converged).
Yet the *executed* trajectory still fails at low K (`both-hard` K=2: 1 real violation with 0 NLP
failures). So:

> The NLP guarantees the **predicted terminal** `x̂1` is feasible in normalised plan-space. At low K
> that terminal prediction `x̂1 = x_ref + (1−τ)v` is built from a 2–5-step Euler integration of a
> coarse field — it is a poor estimate of where the trajectory actually goes. Projecting a bad
> estimate onto the feasible set yields a feasible-but-wrong plan, and the closed-loop MPC rollout
> then drifts off it. Feasibility of the plan ≠ safety/success of the execution when the field is
> coarse.

Post-hoc projection (B) avoids this because it snaps the **fully-sampled** trajectory onto the
feasible set at the end (and, see §5, fans 4 candidates) — it does not rely on an early terminal
prediction.

## 5. Compute — C is always the most expensive, and one confound

| K | s/step: A | B | C |
|---|---|---|---|
| 2  | 0.019 | 0.026 | 0.078 |
| 5  | 0.044 | 0.110 | 0.189 |
| 10 | 0.088 | 0.199 | 0.374 |

C runs one NLP per ODE step in the loop, so it is ~2–3× slower than B at every K, and the gap grows
with K. So even at K=10 where C finally matches B on success, it does so at ~1.9× the wall time.

🔴 **The confound that must be closed before any conclusion:** **B runs batch_size=4** (candidate
fan + selection) while **C runs batch_size=1** (faithful, PLAN §3.4). B's low-K robustness may be
partly its 4-candidate fan, not the post-hoc-vs-in-loop distinction. This is now the **single most
important next experiment** — see §7.

## 6. Caveats (unchanged from fix_2, still binding)

- **n = 6** episodes per arm per K (2 trials × 3 variants × 1 seed). A "0.33" or "0.50" is 2–3
  episodes; CIs are huge. PLAN §5 wants **n ≥ 100**.
- **One seed.** The seed-6 checkpoint is the only one that exists.
- **Batch asymmetry** (§5) — the primary threat to the headline.
- **One checkpoint, FMv3ODE only** (by design).

## 7. Verdict and next steps

**Verdict (preliminary):** the low-K sweep is a **clean, strong negative** for Gen12's contribution.
Post-hoc projection (B) is 100%/100% at every K from 2 up; in-loop sampling (C) fails at low K,
matches B only at K≥10, and always costs ~2–3×. On this data, `hardflow_new` has **no operating
point where it beats DPCC projection** — it is dominated. Consistent with, and a sharper version of,
the Gen13 "projection dominates" result.

**Before writing it up as final:**
1. **Close the batch confound (decisive):** re-run arm C at `hardflow.batch_size: 4` (+ candidate
   selection). If C-at-batch-4 becomes robust at low K, then B's edge was the fan, not the method,
   and the story changes. If C-at-batch-4 still fails at low K, the negative is real.
2. **n ≥ 100 + multiple seeds** (needs seeds 7–10 trained) to make the ramp statistically real.
3. Keep the **per-variant** breakdown — the `top-right /` asymmetry is a genuine signal about where
   in-loop projection breaks, worth understanding (likely the terminal-prediction error interacts
   with that obstacle layout).

**If the batch-4 arm-C run still loses:** Gen12 is a validated *negative* — the port is correct
(0 NLP failures, feasible plans, matches B at high K) but the method does not improve on DPCC. That
is a publishable/loggable result: *in-loop constrained sampling buys nothing over post-hoc
projection on FMPCC's own model, and costs more.*
