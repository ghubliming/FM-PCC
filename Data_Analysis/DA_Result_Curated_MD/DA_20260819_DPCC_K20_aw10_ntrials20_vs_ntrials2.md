# DA — DPCC baseline K20 / aw10 at `n_trials=20` vs `n_trials=2`

**Date:** 2026-08-19 · **Task:** avoiding-d3il · **Job:** 24639 (`eval_dpcc_job`, log `12_43_45_eval_dpcc_job_24639.log`)
**Data:** `temp/1808/H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials/`
**Batch analysis:** `batch_avoiding_combined_20260818_152911`

| | n=2 reference | n=20 new |
|---|---|---|
| candidate folder | `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` | `H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials` |
| checkpoint parent | `.../plans/diffusion/` (flat) | `H8_K20_Dmodels.GaussianDiffusion_aw10/` |
| K / aw / threshold | 20 / 10 / 0.5 | 20 / 10 / 0.5 |
| seeds | 6–10 | 6–10 |
| episodes per cell | 10 | **100** |
| S&C resolution | 0.10 | **0.01** |

---

## 0. Run status — INCOMPLETE

Job 24639 was killed by the 24 h wall limit:

```
slurmstepd-i6-gpu-1: error: *** JOB 24639 ON i6-gpu-1 CANCELLED AT 2026-08-18T10:44:04 DUE TO TIME LIMIT ***
```

| halfspace | seed 6 | 7 | 8 | 9 | 10 | usable |
|---|---|---|---|---|---|---|
| `top-right-hard` | 13/13 | 13/13 | 13/13 | 13/13 | 13/13 | ✅ full |
| `top-left-hard` | 13/13 | 13/13 | 13/13 | 13/13 | 13/13 | ✅ full |
| `both-hard` | 5/13 | 0 | 0 | 0 | 0 | ❌ 5 cells, seed 6 only |

The loop order is halfspace-outer / seed-inner, so two of three halfspaces finished cleanly and the third died 5 variants into seed 6. Config snapshot verified correct: `n_trials: 20`, `seeds: [6,7,8,9,10]`, 13 variants, Table-2 `dt` sweep commented out.

**Everything below uses only `top-left-hard` and `top-right-hard`.** `both-hard` is reported separately as indicative-only.

---

## 1. Answer: no, n=20 is *not* the same as n=2

Three distinct effects, in increasing order of importance.

**(a) The 1.00 ceiling was an artefact of 10 episodes.** Several cells that read exactly `1.00` at n=2 are not perfect at n=20:

| cell | n=2 | n=20 |
|---|---|---|
| `dpcc-c-tightened` @ top-right | 1.00 | 0.95 |
| `dpcc-t-tightened` @ top-right | 1.00 | 0.92 |
| `post_processing-tightened` @ top-left | 1.00 | 0.97 |

With 10 episodes a variant only needs to survive 10 rollouts to score 1.00. Any "perfect" n=2 number should be read as "≥0.90 with ~±0.10 resolution", not as 1.00.

**(b) Mid-range cells move by up to 0.35 — in both directions.** These are the largest shifts among the untainted variants:

| cell | n=2 | n=20 | Δ |
|---|---|---|---|
| `dpcc-t` @ top-left | 0.80 | 0.54 | **−0.26** |
| `dpcc-c` @ top-right | 0.60 | 0.41 | −0.19 |
| `dpcc-t` @ top-right | 0.50 | 0.34 | −0.16 |
| `dpcc-r` @ top-left | 0.20 | 0.37 | +0.17 |
| `dpcc-r` @ top-right | 0.20 | 0.31 | +0.11 |

Systematically, the **untightened** DPCC variants that looked mid-good at n=2 got *worse* (they were sampling-lucky), and the weak ones got slightly better. The **tightened** variants are stable to within ~0.08.

**(c) The bottom of the table did not move.** `diffuser`, `gradient*`, `model_free*` are 0.00–0.15 at both n=2 and n=20, with ~25 violations/episode on top-right. Their ordering is meaningless noise at either trial count — do not rank within that group.

---

## 2. ⚠️ The n=2 `post_processing` rows are corrupt — do not use them

In the n=2 baseline, `post_processing` is **numerically identical to `dpcc-r`**, and `post_processing-tightened` identical to `dpcc-r-tightened`, on *every* metric except wall-clock:

```
n=2  top-left   post_processing           vs dpcc-r          : 12/14 metrics EXACTLY equal
n=2  top-left   post_processing-tightened vs dpcc-r-tightened: 12/14 metrics EXACTLY equal
n=2  top-right  post_processing           vs dpcc-r          : 16/18 metrics EXACTLY equal
n=2  top-right  post_processing-tightened vs dpcc-r-tightened: 12/14 metrics EXACTLY equal
        (the only two that differ are avg_time / avg_time_std, in the 3rd decimal)
```

Identical `n_steps`, `n_success`, `n_success_and_constraints`, `n_violations`, `total_violations` — to full precision — cannot happen by chance. Timings differing only in the 3rd decimal is the signature of **the same projection being executed twice**: the n=2 run's `post_processing` branch was resolving to the `dpcc-r` projector.

At n=20 the two separate completely (**0/14** metrics equal), and the timings become physically sensible: `post_processing` costs 0.196 s/step ≈ `diffuser` (0.184) + ε, exactly what a one-shot post-hoc projection should cost, versus 0.557 s/step for `dpcc-r`'s per-step NLP.

**Consequence:** every historical table carrying n=2 `post_processing*` numbers for this baseline is reporting `dpcc-r` under the wrong name. The n=20 numbers are the first valid ones.

---

## 3. Per-variant results

#### `top-left-hard` — 5 seeds x 20 trials = 100 episodes

| variant | S&C n=20 | ±SEM | S&C n=2 | Δ | succ n=20 | viol n=20 | steps n=20 | s/step n=20 | **s/ep n=20** | s/ep n=2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | **1.00** | ±0.000 | 1.00 | +0.00 | 1.00 | 0.00 | 70.0 | 0.558 | **39.1** | 38.7 |
| `dpcc-t-tightened` | **1.00** | ±0.000 | 1.00 | +0.00 | 1.00 | 0.00 | 75.5 | 0.581 | **43.8** | 43.5 |
| `dpcc-r-tightened` | **1.00** | ±0.000 | 1.00 | +0.00 | 1.00 | 0.00 | 73.7 | 0.594 | **43.8** | 48.7 |
| `post_processing-tightened` ⚠️ | **0.97** | ±0.058 | 1.00 | -0.03 | 0.99 | 1.30 | 78.0 | 0.194 | **15.1** | 48.5 |
| `dpcc-c` | **0.70** | ±0.196 | 0.80 | -0.10 | 1.00 | 1.47 | 73.0 | 0.486 | **35.5** | 34.3 |
| `dpcc-t` | **0.54** | ±0.211 | 0.80 | -0.26 | 1.00 | 2.20 | 77.2 | 0.525 | **40.5** | 32.4 |
| `post_processing` ⚠️ | **0.51** | ±0.196 | 0.20 | +0.31 | 0.95 | 13.68 | 85.0 | 0.196 | **16.7** | 46.8 |
| `dpcc-r` | **0.37** | ±0.213 | 0.20 | +0.17 | 1.00 | 3.85 | 77.1 | 0.557 | **43.0** | 47.0 |
| `model_free-tightened` | **0.15** | ±0.109 | 0.10 | +0.05 | 0.99 | 13.61 | 74.7 | 0.297 | **22.2** | 18.8 |
| `model_free` | **0.13** | ±0.099 | 0.10 | +0.03 | 0.99 | 13.97 | 74.9 | 0.274 | **20.5** | 18.2 |
| `diffuser` | **0.12** | ±0.104 | 0.10 | +0.02 | 1.00 | 14.28 | 72.0 | 0.184 | **13.2** | 12.3 |
| `gradient` | **0.12** | ±0.099 | 0.10 | +0.02 | 1.00 | 14.50 | 69.5 | 0.198 | **13.8** | 12.5 |
| `gradient-tightened` | **0.11** | ±0.094 | 0.10 | +0.01 | 1.00 | 14.12 | 69.7 | 0.199 | **13.8** | 12.4 |

#### `top-right-hard` — 5 seeds x 20 trials = 100 episodes

| variant | S&C n=20 | ±SEM | S&C n=2 | Δ | succ n=20 | viol n=20 | steps n=20 | s/step n=20 | **s/ep n=20** | s/ep n=2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `post_processing-tightened` ⚠️ | **0.99** | ±0.019 | 0.90 | +0.09 | 1.00 | 0.04 | 93.9 | 0.198 | **18.6** | 42.9 |
| `dpcc-c-tightened` | **0.95** | ±0.073 | 1.00 | -0.05 | 0.99 | 0.10 | 77.6 | 0.517 | **40.2** | 38.9 |
| `dpcc-r-tightened` | **0.95** | ±0.073 | 0.90 | +0.05 | 0.99 | 0.08 | 84.0 | 0.553 | **46.5** | 42.9 |
| `dpcc-t-tightened` | **0.92** | ±0.087 | 1.00 | -0.08 | 0.98 | 0.21 | 90.1 | 0.548 | **49.4** | 52.1 |
| `dpcc-c` | **0.41** | ±0.131 | 0.60 | -0.19 | 0.88 | 2.71 | 72.0 | 0.406 | **29.3** | 27.4 |
| `dpcc-t` | **0.34** | ±0.121 | 0.50 | -0.16 | 0.89 | 4.45 | 80.7 | 0.415 | **33.5** | 36.7 |
| `dpcc-r` | **0.31** | ±0.124 | 0.20 | +0.11 | 0.91 | 3.93 | 78.1 | 0.442 | **34.5** | 29.0 |
| `post_processing` ⚠️ | **0.10** | ±0.100 | 0.20 | -0.10 | 0.87 | 8.19 | 83.3 | 0.201 | **16.8** | 28.9 |
| `gradient-tightened` | **0.02** | ±0.039 | 0.00 | +0.02 | 0.94 | 25.01 | 69.3 | 0.204 | **14.2** | 12.0 |
| `model_free` | **0.01** | ±0.019 | 0.00 | +0.01 | 0.91 | 25.85 | 75.7 | 0.270 | **20.4** | 17.3 |
| `model_free-tightened` | **0.01** | ±0.019 | 0.00 | +0.01 | 0.94 | 25.98 | 74.7 | 0.297 | **22.2** | 19.8 |
| `diffuser` | **0.00** | ±0.000 | 0.00 | +0.00 | 0.92 | 25.62 | 70.6 | 0.190 | **13.4** | 11.7 |
| `gradient` | **0.00** | ±0.000 | 0.00 | +0.00 | 0.95 | 25.54 | 69.8 | 0.205 | **14.3** | 11.9 |

`s/ep` = `n_steps × avg_time` = wall-clock seconds per episode. `±SEM` = across-seed std / √5.
⚠️ marks the variants whose n=2 column is invalid (§2) — the Δ for those rows is meaningless.

#### `both-hard` — seed 6 only, 5 of 13 variants (INDICATIVE ONLY, do not cite)

| variant | S&C n=20 (1 seed) | S&C n=2 (5 seeds) |
|---|---|---|
| `dpcc-c-tightened` | 1.00 | 1.00 |
| `dpcc-r-tightened` | 0.95 | 1.00 |
| `dpcc-r` | 0.55 | 0.40 |
| `dpcc-c` | 0.50 | 0.40 |
| `dpcc-t` | 0.35 | 0.70 |

---

## 4. Does the DA Target change?

Target convention: the best projection variant of DPCC K20 / aw10 (`da-target-is-best-baseline-variant`).

**Among DPCC variants: no change — `dpcc-c-tightened` is still the target.**

| | top-left | top-right | s/ep (TL / TR) |
|---|---|---|---|
| `dpcc-c-tightened` | **1.00** | **0.95** | 39.1 / 40.2 |
| `dpcc-r-tightened` | 1.00 | 0.95 | 43.8 / 46.5 |
| `dpcc-t-tightened` | 1.00 | 0.92 | 43.8 / 49.4 |

It ties the other two tightened variants on S&C and is cheapest on both halfspaces, at n=2 and at n=20 alike. Pin the target at:

> **DPCC K20 / aw10 / T0.5 / `dpcc-c-tightened` — S&C 1.00 (TL), 0.95 (TR); 39.1 / 40.2 s/ep.**

Note the top-right value drops from 1.00 to 0.95, so the bar our methods must clear on top-right is *slightly lower* than previously assumed, while the wall-clock bar is unchanged.

## 5. ⚠️ `post_processing-tightened` now threatens the baseline

With its n=20 numbers valid for the first time:

| | S&C top-left | S&C top-right | s/ep TL | s/ep TR |
|---|---|---|---|---|
| `dpcc-c-tightened` (target) | 1.00 ±0.000 | 0.95 ±0.073 | 39.1 | 40.2 |
| `post_processing-tightened` | 0.97 ±0.059 | **0.99** ±0.020 | **15.1** | **18.6** |

The S&C differences are inside seed noise in both directions (−0.03 ±0.06 on top-left, +0.04 ±0.076 on top-right), so on the success gate these are **statistically indistinguishable**. The wall-clock difference is not: post-processing is **2.2–2.6× cheaper per episode**.

It is not a clean Pareto dominance — `post_processing-tightened` uses *more* control steps (78.0 / 93.9 vs 70.0 / 77.6), and its violation count on top-left is 1.30 vs 0.00, so it is not constraint-clean the way DPCC is. Call it **non-dominated / a trade-off**: same success, far less compute, slightly dirtier trajectories.

This matters because `post_processing` is a *baseline* in the DPCC paper, not a DPCC method. Before anything is written up:

1. Confirm the n=20 `post_processing` path is the real post-hoc projector and not another aliasing bug in the other direction. Its cost profile (0.194–0.201 s/step, flat across halfspaces, tiny std 0.003–0.011) is consistent with a one-shot projection, which is evidence for, not proof.
2. Check whether the fix that separated `post_processing` from `dpcc-r` between the two runs was intentional (config/code change) or incidental.
3. Re-run `both-hard` before drawing any conclusion — it is the hardest setting and is exactly where a cheap post-hoc projector is most likely to fall over.

---

## 6. Practical implications

- **All n=2 comparisons against this baseline need re-checking.** The two effects that bite hardest: any 1.00 was inflated, and untightened DPCC variants were flattered by ~0.2.
- **`SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md` is stale** — it uses n=2 baseline numbers, including the corrupt `post_processing` rows.
- **Trial-count parity is now mandatory** in any table: an n=2 method row against an n=20 baseline row is not a comparison.
- Seed noise at n=20 is still substantial (SEM up to 0.073 for a 5-seed mean). Differences under ~0.10 in S&C are not real.

## 7. Outstanding

- [ ] `both-hard` at n=20, 5 seeds × 13 variants — resubmit with the halfspace list narrowed to `['both-hard']` and the same `FMPCC_RUN_MSG=20trials`; ~12 h from the observed rate (24 h bought ~2.1 halfspaces), so it fits the wall.
- [ ] Verify the `post_processing` projector path (§5).
- [ ] Supersede the 20260813 snapshot once `both-hard` lands.
