# Results — HardFlow IPOPT vs DPCC SLSQP on the identical NLP

Job **25121**, node i6-gpu-1, git `1897f4f`, 2026-08-27 17:24 UTC.
Log: `temp/2808/2026-08-27/12_38_52_bench_solver_hf_vs_dpcc_25121.log`
Config: `horizon=8`, `dof=44`, `vars(DPCC)=48`, `halfspace=both-hard`, `tau=1.0`, `reps=50`, `runs=3` (seeds 1/2/3), `ref=both`.
Script/changelog: `CHANGELOG_20260827_solver_bench_ipopt_vs_slsqp.md`.

---

## 1 · Headline

**On the identical NLP, IPOPT costs 4.3× what SLSQP costs — and IPOPT's cost barely moves when the problem gets harder.**

| reference regime | IPOPT (HF) median ms | SLSQP (DPCC) median ms | IPOPT / SLSQP |
|---|---:|---:|---:|
| `endpoint` (near-feasible — what HardFlow solves) | 47.6 | 11.0 | **4.33×** |
| `iterate` (heavy noise — what post-hoc projection solves) | 54.2 | 34.0 | 1.63× |

Per seed (median ms):

| seed | IPOPT ep | SLSQP ep | ratio ep | IPOPT it | SLSQP it | ratio it |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.638 | 10.837 | 4.40× | 55.900 | 42.325 | 1.32× |
| 2 | 47.327 | 10.871 | 4.35× | 56.027 | 29.466 | 1.90× |
| 3 | 47.700 | 11.228 | 4.25× | 50.538 | 30.302 | 1.67× |

IPOPT's `endpoint` median is **47.3–47.7 ms across three independent seeds** — a spread of 0.4 %. That is not a solve time, that is a floor.

---

## 2 · Overhead domination — now measured directly, not fitted

Audit §0.1 argued from a two-point *horizon* fit (2.09× variables → 1.66× time) that ~81 % of IPOPT's cost is size-independent per-call setup. This bench tests the same claim on a second, cleaner axis: **same problem size, different problem difficulty.**

| solver | `endpoint` → `iterate` |
|---|---:|
| IPOPT | 47.6 → 54.2 ms = **1.14×** |
| SLSQP | 11.0 → 34.0 ms = **3.09×** |

Per seed: IPOPT 1.17× / 1.18× / 1.06×; SLSQP 3.91× / 2.71× / 2.70×.

**Read:** SLSQP's time is nearly all work — triple the difficulty, triple the time. IPOPT's time is nearly all overhead — triple the difficulty, +14 %. Two different axes (size, difficulty) now give the same answer, so §0.1's conclusion no longer rests on the two-point fit.

⚠️ **Absolute ms here are not eval ms.** The bench forces `halfspace=both-hard`, uses synthetic references, runs `tau=1.0` and a warm process with no generator. IPOPT measures 47.6 ms here against 30 ms in the fan-matched parity run. **Only the ratios transfer**; do not paste these absolutes into a cost table.

---

## 3 · 🔴 The consequence: HardFlow's own solver cancels HardFlow's own speed argument

HardFlow's design claim is that projecting the *predicted clean endpoint* is cheaper than projecting a *noisy iterate*. Section 2 prices that claim on both solvers:

| solver | what the endpoint trick is worth |
|---|---:|
| SLSQP | **3.09×** |
| IPOPT | **1.14×** |

The mechanism is real — it is worth 3× — but IPOPT's fixed per-call cost swallows it whole. **HardFlow ships the one solver that cannot cash in HardFlow's central optimisation.** That is the direct answer to "why is their HF the cheap arm and ours the expensive one", and it is now measured on our own constraint set rather than reconstructed from their table.

### Projected effect of the swap (extrapolation — flagged as such)

Applying the measured **4.33×** `endpoint` ratio to the fan-matched parity run (audit §0.1: generator 18.5 ms, DPCC 2.4 ms/step, HF 30 ms/step):

| arm | per step now | per step with SLSQP |
|---|---:|---:|
| DPCC (arm B) | 18.5 + 2.4 = **20.9 ms** | unchanged |
| HardFlow (arm C) | 18.5 + 30 = **48.5 ms** | 18.5 + 6.9 = **25.4 ms** |
| HF / DPCC | 2.32× | **1.22×** |

⚠️ Extrapolation, on the assumption that the 4.33× ratio transfers from the bench's constraint config to the eval's. It changes **only the solve term** — solve *count* (`K − floor((1−A)·K)`) and generator cost are untouched, and nothing here touches success or constraint metrics.

**Decision-rule outcome (audit §0.2):** this is the first row — *"cost collapses to parity → cost stops being the story; S&C and steps decide."* Chapters 1–3 already answer S&C and steps, and HardFlow loses them. So the swap is now expected to **remove HardFlow's cost excuse without rescuing it**, unless the failure finding in §5 turns out to be doing real damage.

---

## 4 · Chapter 4's question, answered: yes, the degenerate rows agree

The Q&A chapter 4 asked whether swapping the solver should give ~0 difference in the degenerate regime. Measured `‖Π_IPOPT − Π_SLSQP‖` on `endpoint` references:

| seed | mean | max |
|---:|---:|---:|
| 1 | 0.05821 | 2.89165 |
| 2 | 0.00040 | 0.00103 |
| 3 | 0.00034 | 0.00102 |

**Seeds 2 and 3: the two projectors return the same point to ~1e-3 over 100 solves.** Theory said ~0; measurement says ~1e-3. Confirmed — in the degenerate regime the solver is not the story for *what* is produced, only for what it costs.

**Seed 1 is one outlier, not a trend.** `2.89165 / 50 = 0.0578`, which plus the ~0.0004 baseline reconstructs the 0.05821 mean exactly — so **exactly one solve in 50 diverged**. That same seed is the only `endpoint` run where SLSQP returned a mildly infeasible point (`+7.19e-04`). So the divergence is an SLSQP failure, not a systematic disagreement: **1 bad solve in 150.**

On `iterate` references the two disagree wholesale (mean 2.7–2.9, max 5.6–6.6) — expected, since both are frequently failing there (§5).

---

## 5 · 🔴 The feasibility gate fired — and it is mostly the harness, not the eval

The script exits non-zero if either projector returns output violating a constraint by > 1e-4. It fired on all three runs.

| regime | IPOPT max residual | SLSQP max residual | IPOPT non-convergence |
|---|---:|---:|---:|
| `endpoint` | +1.00e-08 (×3) | +7.19e-04 / +4.77e-10 / +4.77e-10 | **0 / 150** |
| `iterate` | +6.69e-02 / +6.14e-02 / +4.09e-02 | +1.76e-02 / +1.87e-02 / +1.99e-02 | **39 / 150 (26 %)** |

Residuals are in each constraint's own units (m² for the `sphere_*` rows, m for `lb`/`ub`).

**Three separate readings, do not merge them:**

1. **`endpoint` IPOPT `+1.00e-08` is not a violation.** It is bit-identical across all three seeds, so it is a formulation constant at the solver's constraint tolerance, not a solve outcome. Clean.
2. **`iterate` failures are probably the reference generator, not the solvers.** The `iterate` regime is synthetic — σ = 0.60 in normalized coords, clipped to ±1 — and **both** solvers return infeasible output on it. When both independent codes fail on the same inputs, the likeliest cause is that those inputs are near-infeasible for the constraint set. **This is not evidence that DPCC is broken in real eval**, where SLSQP is not observed failing at this rate. If the gate is to be used as a real regression check, the `iterate` σ needs calibrating against actual mid-ODE iterates first.
3. **IPOPT is nonetheless the worse failure mode.** 26 % non-convergence and a 3× larger violation than SLSQP on the same references, each failure silently returning a possibly-infeasible last iterate. That is the same defect already recorded at 12.5–13.5 % on visual-avoiding TL untightened (Q&A 2b), reproduced here offline with no GPU and no checkpoint.

---

## 6 · What this settles, and what it does not

**Settled:**
- IPOPT vs SLSQP on our NLP is **4.33×** on the reference HardFlow actually solves. "It is only a solver" is answered: yes, and it is worth 4.3×.
- IPOPT is overhead-dominated — confirmed on a second axis, no fitting.
- HardFlow's endpoint trick is worth 3.1× to SLSQP and 1.14× to IPOPT.
- In the degenerate regime the two projectors agree to ~1e-3.

**Not settled:**
- The projected 1.22× is an extrapolation. Only an actual arm-C-with-SLSQP eval run confirms it.
- No S&C / steps evidence here — this bench times solvers, it does not run policies. It cannot rescue HardFlow on quality and was not built to.
- The horizon sweep (`HFFM_BENCH_HORIZON=8` then `16`) was **not** run. It is now optional confirmation rather than the deciding test, since §2 already establishes overhead domination on the difficulty axis.
- The `iterate` regime is uncalibrated (§5.2).

**Nothing in chapters 1–3 is retracted.** This changes the *explanation* of HardFlow's cost, not any measured cost, success, or step count.
