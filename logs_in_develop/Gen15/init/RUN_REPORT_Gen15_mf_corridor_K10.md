# RUN REPORT — Gen15 first arm: `mf` (MeanFlow) on UAV corridor, K=10

**Date:** 2026-08-15 · **Type:** run report / first-results read
**Status:** ✅ train complete · ✅ eval complete (20/20 variants) · ⚠️ two timing columns invalid (§5)
**Data:** `temp/1508/UAV_MIX/` · logs `temp/1508/2026-08-14/`

| | job | wall clock | outcome |
|---|---|---|---|
| gates | **24578** | 12 s | 6 PASS, G1 SKIP |
| train | **24579** | 7 h 14 m | 100 k steps, completed |
| eval | **24583** | ~6 h | 20 variants × 10 trials, completed |

**Provenance:** git `5eaea24` · node i6-gpu-1 (RTX A5000) · engine `mf`, backbone `unet`,
`freq_dim=32`, **3,969,222 params (3.97 M)** · scene `corridor`, seed 6, `cond_mode=pos_only`
(obs 6-D, action 3-D, transition 9-D), H=8 · `dp0.5` · K=10 · `mpc4`, `pid_stopgo`, T=0.5 ·
geo_tag `corridor_bounds+dynamics+geo_bounds+halfspace+obstacles`.

Checkpoint evaluated: **step 80000**, not 100000. `save_freq = n_train_steps // 5` with the save
check running before the step increment, so the last periodic save is 80 k. Gen11's trainer is
byte-identical here, so an fm-vs-mf comparison stays fair — but it must be stated.

---

## 1. Did the wiring work? Yes — all four things Gen15 added are confirmed live

1. **Engine dispatch** — `[ train ] Gen15 UAV Mix-ML — engine: mf (MeanFlow …)`, savepath
   `logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_D…MeanFlowODE_9D_dp0.5_bbunet/6`. The `dp`/`bb`
   tokens are present, so a second `mf` run at a different data-proportion cannot overwrite this.
2. **K plumbing (the Gen11 bug closed)** — `[ eval ] engine=mf NFE budget K=10 pinned on the
   loaded model`, output leaf `Emf_K10_mpc4_pid_stopgo_T0.5`. K reached both the sampler and the
   path. In Gen11 it reached neither.
3. **Fix_12 geometry is active** — `corridor feasibility check: homotopy=L/C/R expert route OK
   under planning margin 0.33 m`. The near-infeasible-constraint era is behind us.
4. **Architecture control** — gates G3: `fm` 3,955,177 vs `mf`/`af` 3,969,222 params (+0.4 %,
   accounted for by `h_mlp` + the dual v-head). The three arms are the same network.

---

## 2. Training

`raw_mse_u` is the readable signal; the adaptive loss is pinned near its ceiling by construction
(0.9999 → 0.890 over 100 k steps) and says nothing, exactly as the plan predicted.

| step | train `raw_mse_u` | test `raw_mse_u` |
|---|---|---|
| 0 | 93.26 | 93.65 |
| 20 k | 1.895 | 1.862 |
| 50 k | 0.793 | 0.963 |
| 99 k | **0.154** | **0.717** |

The v-head learns too (`raw_mse_v` 87.3 → 0.172 train / 0.671 test). Convergence is real and
monotone; no loss spikes.

### 2.1 🔴 The gradient clip is binding on **100 %** of steps

`grad_norm_history`: min 2.93, **median 265.7**, p95 444.4, max 621.1 — against
`gradient_clip = 1.0`.

Every single step is clipped, by a factor of ~265 at the median. Training is therefore effectively
**normalised-gradient descent**: the update direction is the model's, the magnitude is a constant.

This is not a code defect — the clip is wired and doing what it is told (that wiring was itself a
Gen3v6 fix). It is a **value inherited without rescaling**: `gradient_clip: 1.0` comes from
Gen3v6/v7's `avoiding-d3il` blocks, where the gradient scale is a different order of magnitude.
On UAV it is ~2 orders too tight.

It did not prevent convergence, so nothing here is invalid. But the effective learning rate is now
decoupled from the loss, and the setting was never chosen for this task. **Candidate for the next
fix/U-step: sweep `gradient_clip` ∈ {1.0 (current), 100, 1000, 0 (off)} on one seed.** Note the
`af` arm inherits the same value.

### 2.2 The h-stratified residual predicts trouble at low K

| bucket | last value | samples |
|---|---|---|
| b0 (smallest h) | 0.267 | 100 |
| b1 | 0.095 | 97 |
| b2 | 0.074 | 79 |
| **b3 (largest h)** | **0.555** | **11** |

`h` is the mean-flow interval, and **large `h` is exactly the one-/two-step sampling regime**. It
is both the worst-fitted bucket (7.5× b2) and the least-sampled one (11 recorded points vs 100).

This is the single most useful number in the training run, because Gen15's thesis lives at K=1/K=2
— the only budgets that fit the 30.3 ms control deadline (§4). **Expect the K sweep to degrade
sharply at K=1/2**, and if it does, this is the mechanism, not a wiring problem. `af_ratio_fm` /
`meanflow_data_proportion` and the h-sampling distribution are the knobs that address it.

### 2.3 Train/test gap

`raw_mse_u` 0.154 train vs 0.717 test — 4.7×. `raw_mse_v` 3.9×. On a 500-episode single-scene
dataset that is unsurprising, but worth watching when more scenes come in.

---

## 3. Eval — all 20 variants (corridor, seed 6, 10 trials, K=10)

`succ` = strict success · `S&C` = success **and** constraints · `cfree` = collision-free rate ·
`gdist` = final distance to goal (m) · `s2g` = steps to goal · `terr` = tracking error ·
`tot_ms` = measured per-replan wall clock. **`CB` = circuit-breaker trips: 0 everywhere.**

| variant | succ | S&C | safe | cfree | goal | gdist | steps | s2g | terr | tot_ms | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` (unprojected) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 39.26 | 396 | — | 97.00 | 88.5 | 92.9 |
| `gradient` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.91 | 396 | — | 5.57 | 97.2 | 98.4 |
| `gradient-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.70 | 396 | — | 5.82 | 97.1 | 98.3 |
| `post_processing` | 0.10 | 0.00 | 0.10 | 0.00 | 0.10 | 13.94 | 382.6 | 262.0 | 55.87 | 196.1 | 372.2 |
| `post_processing-tightened` | 0.20 | 0.00 | 0.20 | 0.00 | 0.20 | 21.14 | 366.2 | 247.0 | 65.12 | 196.1 | 350.1 |
| `model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.31 | 396 | — | 108.49 | 249.5 | 298.0 |
| `model_free-tightened` | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 36.87 | 396 | — | 96.02 | 257.6 | 408.3 |
| **`bounds_free`** | **1.00** | 0.70 | 1.00 | 0.70 | 1.00 | 0.29 | 257.8 | 257.8 | 0.56 | 242.0 | 308.9 |
| `bounds_free-tightened` | 0.40 | 0.30 | 0.40 | 0.30 | 0.40 | 11.60 | 336.7 | 247.8 | 30.09 | 431.2 | 872.1 |
| `geo_free` | 0.70 | 0.50 | 0.70 | 0.50 | 0.70 | 1.08 | 299.3 | 257.9 | 9.65 | 184.8 | 573.1 |
| `geo_free-bounds_free` | 0.70 | 0.50 | 0.70 | 0.50 | 0.70 | 1.01 | 302.1 | 261.9 | 11.66 | 178.7 | 477.6 |
| `geo_free-model_free` | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 61.25 | 396 | — | 84.67 | 143.7 | 192.4 |
| `model_free-bounds_free` | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 26.56 | 396 | — | 97.78 | 220.7 | 327.0 |
| `model_free-bounds_free-tightened` | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 18.04 | 396 | — | 101.93 | 215.0 | 254.8 |
| **`dpcc-r`** | **1.00** | **0.80** | 1.00 | 0.80 | 1.00 | 0.29 | **255.6** | 255.6 | 0.57 | 271.0 | 372.2 |
| `dpcc-r-tightened` | 0.40 | 0.10 | 0.40 | 0.10 | 0.40 | 7.36 | 339.3 | 254.2 | 24.48 | 498.7 | 991.2 |
| **`dpcc-c`** | **1.00** | 0.70 | 1.00 | 0.70 | 1.00 | 0.29 | 259.0 | 259.0 | **0.55** | 269.7 | 342.6 |
| `dpcc-c-tightened` | 0.60 | 0.10 | 0.60 | 0.10 | 0.60 | 5.67 | 308.9 | 250.8 | 29.02 | 430.3 | 982.7 |
| **`dpcc-t`** | **1.00** | 0.70 | 1.00 | 0.70 | 1.00 | 0.29 | 259.1 | 259.1 | 0.56 | 273.0 | 371.6 |
| `dpcc-t-tightened` | 0.60 | 0.00 | 0.60 | 0.00 | 0.60 | 2.41 | 312.4 | 256.7 | 34.40 | 501.9 | 1037.5 |

### 3.1 The headline

**MeanFlow + DPCC solves corridor at K=10.** `dpcc-r`, `dpcc-c` and `dpcc-t` all reach
**success 1.00 / goal 1.00 / safe 1.00**, 256–259 steps, tracking error 0.55–0.57, `gdist` 0.29 m
(inside the 0.30 m goal radius). `dpcc-r` is the best row: S&C **0.80** and the fewest steps.

### 3.2 The S&C gap is a threshold artifact, not a safety failure

`dpcc-c` scores success 1.00 but S&C 0.70. The raw counts explain it:

| variant | violating steps (mean) | total violation magnitude | contact fraction |
|---|---|---|---|
| `dpcc-c` | **2.1** of ~259 | **0.01** | **0.0000** |
| `dpcc-r` | 4.2 | 0.08 | 0.0012 |
| `bounds_free` | 3.9 | 0.03 | 0.0006 |
| `geo_free` | 94.0 | 33.24 | 0.1828 |
| `diffuser` | 368.8 | **11024.77** | 0.0000 |

`dpcc-c`'s failures are ~2 steps of numerically-negligible residual (magnitude 0.01) with **zero
physical contact**. That is a strict-inequality artifact, not a crash. `geo_free`'s 0.50 is real
(94 violating steps, 18 % contact). `diffuser`'s is catastrophic — it flies straight through the
walls.

⚠️ **Do not report S&C 0.70 as "30 % unsafe"** without this table. Conversely, do not quietly
relabel it a success — the metric is what it is; the magnitude column is the context.

### 3.3 ⚠️ The unprojected policy fails completely — and that caps what this run can claim

`diffuser` (no projector) = **0.00 success, gdist 39.26 m, 368.8 violating steps**. At K=10 the raw
MeanFlow policy does not fly the corridor at all; **the DPCC projector supplies 100 % of the
performance**.

This is the most important caveat in the report. It means this run demonstrates that *the Gen15
frame works end to end*, **not** that the MeanFlow objective is good. Until the Gen11 FM
`diffuser` row is in hand, we cannot tell whether ~0 unguided success is a property of the task or
a deficiency of this arm. **That is the single most decision-relevant number still missing.**

### 3.4 Tightening inverts

Every `-tightened` variant is worse than its base — `dpcc-c` 1.00 → 0.60, `dpcc-r` 1.00 → 0.40,
and S&C collapses (0.70 → 0.10). Cost rises too: 269.7 → 430.3 ms, p95 342.6 → 982.7 ms. The
enlarge margin (0.025 m) on top of the already-tight 0.33 m planning margin appears to over-close
corridor's ~0.12 m feasible band. Check whether Gen11 FM shows the same inversion: if yes it is
task geometry, if no it is ours.

### 3.5 Projector health: clean

`n_tripped_trials = 0` and `total_skipped_steps = 0` on **every** variant. The E9 Fix-15 circuit
breaker and deadline guard never fired — no solver thrashing anywhere in this run.

---

## 4. Real-time: nothing at K=10 is deployable

`total_over_budget` is a **count of steps**, not a ratio. `OVER×3960` = 396 steps × 10 trials, i.e.
**every step exceeded the 30.3 ms budget**, on every variant.

| | measured | vs 30.3 ms budget |
|---|---|---|
| generation only (`diffuser`, K=10) | 88.5 ms | 2.9× over |
| `dpcc-c` total | 269.7 ms (p95 342.6) | 8.9× over |
| `dpcc-t-tightened` total | 501.9 ms (p95 1037.5) | 16.6× over |

Gate G6 measured the sampler in isolation: **K=1 → 8.8 ms, K=2 → 17.7 ms, K=5 → 43.7 ms,
K=10 → 87.2 ms** (which independently confirms the 88.5 ms above). Only **K=1 and K=2 fit the
control deadline at all**, and that is *before* the projector's ~181 ms.

So K=10 is a **quality reference point, not a deployable operating point**, and Gen15's actual
thesis — few-step generation buys real-time feasibility — can only be tested at K=1/K=2. Combined
with §2.2's b3 residual, that is where the interesting failure or success lives.

---

## 5. ⚠️ Two columns from this run are invalid (Gen15 Fix_1)

`proj_ms` reads **0.0 on every variant** because `MeanFlowODE`/`AlphaFlowODE` never emitted
`infos['projection_ms']`. Since the eval computes `fm_ms = total_ms − proj_ms`, the consequence is:

| field | status |
|---|---|
| `total_ms`, `p95`, `total_over_budget` | ✅ **valid** — measured directly by the eval |
| `proj_ms` | ❌ hard 0.0 |
| `fm_ms` | ❌ **invalid** — silently absorbed the projector cost |

All success / safety / steps / tracking numbers are unaffected — the fix cannot change a
trajectory. Reconstruction for this run: `proj_ms ≈ total_ms − 88.5` (e.g. `dpcc-c` ≈ 181 ms of
projector). Fixed in
[`../fix_1/CHANGELOG_fix1_projection_ms_contract.md`](../fix_1/CHANGELOG_fix1_projection_ms_contract.md);
gate **G7** now asserts the contract. **No retraining needed** — re-running the eval is optional
and only buys the exact split.

---

## 6. Where this leaves the campaign

**Confirmed:** the Gen15 frame is correct end to end — dispatch, K plumbing, path isolation,
Fix_12 geometry, architecture-matched backbone, clean projector health. The first arm trains and
solves the task under DPCC.

**Not yet shown:** anything about the MeanFlow *objective*. Three things gate that claim:

1. **The Gen11 FM `diffuser` row** (§3.3) — is unguided ~0 the task, or this arm?
2. **The K=1/K=2 rungs** (§4) — the only real-time-feasible budgets, and the ones §2.2 predicts
   are weakest.
3. **`gradient_clip = 1.0`** (§2.1) — 100 % clipping means this checkpoint was trained under a
   setting no one chose for this task.

Suggested order: (1) is free — read it out of the existing Gen11 logs. (2) is the K sweep already
scripted (`eval_k_sweep.sh`, grid `1 2 5 10 20`). (3) is one training job on one seed.
