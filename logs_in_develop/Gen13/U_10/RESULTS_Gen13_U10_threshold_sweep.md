# Gen13 U10 — threshold sweep on ORIGINAL HardFlow (FM, H16): the free lunch is real

**Date:** 2026-07-26 · **Type:** results / insight · **Status:** solid single-seed result (n=50/threshold)
**Run:** job 23832 (`eval_threshold_sweep_hardflow.sh`), node i6-gpu-1
**Data:** `temp/Gen13U10/H16_1e6steps_hardflow_new_10steps_thres{0.0,0.5,1.0}/trajectories.csv`
**Setup:** ORIGINAL HardFlow FM backbone (`hardflow_new`), avoiding-v0, **H16, K=ode_t_steps=10**,
50 trials/threshold, HardFlow's own metrics. This is the paper's algorithm on the paper's benchmark.

> Unlike the Gen12 K=20 run (saturated, n=6), this is **n=50 at K=10** with HardFlow's own
> Safety-Rate / Steps / Computation-Time metrics — enough to read both safety *and* quality, not just
> compute.

---

## 1. Results (n=50 per threshold)

| threshold | safety rate | success | violations (mean/max) | steps (safe) | s/plan | vs baseline |
|---|---|---|---|---|---|---|
| **0.0** (full-step) | **1.00** | 1.00 | 0.000 / 0 | 50.7 ± 0.9 | 0.840 ± 0.037 | baseline |
| **0.5** (late) | **1.00** | 1.00 | 0.000 / 0 | 51.0 ± 1.4 | **0.545 ± 0.027** | **−35%** |
| **1.0** (terminal-only) | 0.98 | 0.98 | 0.020 / 1 | 61.1 ± 7.7 | **0.235 ± 0.013** | **−72%** |

NLP solves/plan follow the gate: `10 / 6 / 1` (full / last-half / terminal-only).

## 2. Findings

### 2.1 Threshold 0.5 is a genuine free lunch ✅⭐
Identical **safety (1.00), success (1.00), zero violations, same path length** (50.7 → 51.0 steps) as
the full-step baseline — at **35% less wall time** (0.840 → 0.545 s/plan, NLP solves 10 → 6). On
HardFlow's own H16 avoiding-v0 benchmark, at n=50, the early-step NLP solves contribute **nothing** to
safety or quality; dropping them is pure speedup. This is the HardFlow paper's efficiency claim
("skip early steps … good balance") **confirmed on the paper's own algorithm and environment**, not
just on the FMPCC port.

### 2.2 Terminal-only (1.0) is NOT free — it costs safety *and* quality ⚠️
Going all the way to terminal-only buys another big speedup (−72%, 1 NLP solve/plan) but:
- **safety drops 1.00 → 0.98** (1 of 50 trials collides; mean violations 0.020),
- **paths get longer and far more variable**: steps 50.7 ± 0.9 → **61.1 ± 7.7** — even on the 49 safe
  trials the trajectories are worse (≈ +20% steps, ~8× the spread).

So the intermediate NLP solves are unnecessary for the *terminal feasibility guarantee* but **do
shape the closed-loop path** to stay safe and efficient.

### 2.3 The mechanistic reading — terminal guarantee ≠ closed-loop safety
The paper's Prop. guarantees the *plan's terminal state* is feasible, and indeed the NLP never failed.
Yet terminal-only still produced one real collision. Why: this is a **receding-horizon MPC** loop —
at each replan only the first action is executed. With NLP active only at the very last ODE step, the
early trajectory is unguided, so the executed path can drift toward an obstacle before the terminal
projection "catches up," and a step can land inside the obstacle. Some in-loop steering (as in
threshold 0.5) keeps intermediate states clear; terminal-only removes it. Same lesson as Gen12 fix_3:
**plan-feasibility in constraint-model space is not the same as closed-loop safety** — but here the
degradation is mild (2% safety, longer paths) rather than a collapse.

## 3. The takeaway

**Threshold ≈ 0.5 is the operating point.** It is the "good balance" the paper describes, quantified:
full HardFlow safety + quality at ~⅓ less compute. Do **not** push to terminal-only — the last steps'
NLP is free to drop, but the middle steps earn their cost in closed-loop path safety. On this
benchmark the efficiency knob has a clear knee between 0.5 (free) and 1.0 (lossy).

## 4. Cross-check with Gen12 U4

Consistent and complementary:
- Gen12 U4 (FMv3ODE, K=20, n=6): threshold 0.5 gave **−35% time, −46% NLP solves, zero safety cost** —
  but at K=20 everything was saturated, so it couldn't test quality.
- Gen13 U10 (original HF, K=10, n=50): threshold 0.5 gives the **same −35%** *and* confirms **safety +
  path quality are preserved** at real sample size, on the paper's own code.

Two independent codebases, same conclusion: **late-activation at 0.5 is a safe, ~⅓ compute saving.**
This materially strengthens the case for adopting threshold 0.5 as the Gen12 default too.

## 5. Caveats

1. **Single checkpoint / seed** (H16_1e6steps, seed baked into the run). n=50 trials gives tight
   per-threshold estimates, but one model. The `0.98` at thres1.0 is 1/50 — indicative, not a rate.
2. **K=10 only.** The threshold's effect at very low K (K=2) — where the ODE is too coarse for the
   early predictions to matter at all — is untested; 0.5 might behave differently there.
3. **avoiding-v0 / novel constraint** (HardFlow's geometry), not avoiding-d3il — do not compare the
   absolute step counts to Gen12's numbers.
4. Only `hardflow_new` (the paper's canonical black-box algorithm); the l4casadi `hardflow` path was
   not swept.

## 6. Next
1. **Adopt threshold 0.5** as the practical default for HardFlow-style sampling (both Gen13 and Gen12).
2. Sweep a finer grid around the knee (0.5, 0.7, 0.9) to locate exactly where safety/quality starts
   to degrade.
3. Repeat at **low K** (K=2,5) to confirm 0.5 still holds when the ODE is coarse.
4. Multiple seeds/checkpoints before any headline claim.

---

## ⚠️ POST-HOC NOTE (fix, 2026-07-26): threshold labels were pre-flip

This sweep used the OLD (inverted) threshold polarity. It has since been flipped to DPCC polarity
(higher = more projection). Re-map §1's table:

| this MD's label | means | now written as (DPCC) |
|---|---|---|
| `0.0` | full-step (10/10 NLP solves) | **`1.0`** |
| `0.5` | last half (6/10) | `0.5` (unchanged) |
| `1.0` | terminal-only (1/10) | **`0.0`** |

**Findings unchanged:** the free lunch is at 0.5 (fixed point); the *terminal-only* row (labelled
`1.0` here) is DPCC-`0.0`, and the *full-step* baseline (labelled `0.0` here) is DPCC-`1.0`. See
[`../fix_11/CHANGELOG_fix11_dpcc_threshold_polarity.md`](../fix_11/CHANGELOG_fix11_dpcc_threshold_polarity.md).
