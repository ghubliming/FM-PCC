# RESULTS — Fix_8 verification re-run: UNet at `freq_dim=32` trains, and it overturns `fix_1`

**Date:** 2026-08-06 · **Type:** results / fix verification · **Status:** fix confirmed on cluster; `fix_1`'s verdict is falsified
**Runs:** train **24317** (`mf_train`, i6-gpu-1, 10:32:36 → 18:39:30 UTC, 8 h 07 m, git `2d85f03`) ·
eval **24334** (`eval_meanflow`, 21:02:55 → 21:14:03 UTC, 11 m, git `b237c2b`)
**Source logs:** `temp/0608/19_25_29_mf_train_24317.log`, `temp/0608/23_02_54_eval_meanflow_24334.log`,
results tree `temp/0608/H8_Dflow…_bbunet_…/H8_K2_Meuler_T1_…/6/`
**Fix under test:** [`CHANGELOG_Fix_8_unet_width.md`](./CHANGELOG_Fix_8_unet_width.md) §7 follow-up —
*"Gen3v6 `bbunet` re-run at width 32 — the missing architecture control."*
**Falsifies:** [`fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md`](../fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md)
**Note:** a failed eval attempt (**24318**) preceded this one — it targeted seed 7, which was never trained
(`FileNotFoundError: …/7/dataset_config.pkl`). 24334 is the real eval, seed 6.

---

## 0. TL;DR

**Yes — the UNet is fixed, and the fix matters.** Both logs print the new build-time guard:

```
[ MFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=32  params=4.0M
```

exactly the value Fix_8 §3(d) predicted (4.0 M, not 253.0 M).

**And the model trains.** The `freq_dim=256` UNet was pinned at the loss ceiling for all 100 k
steps with its best checkpoint at step 3000. At width 32 the same objective, same backbone class,
same seed converges: `test/loss` **1.000 → 0.912** (still falling at step 100 k), `raw_mse_u`
**19.3 → 1.9**, `per_dim_rms_u` **0.635 → 0.199** — within ~15 % of the DiT arm's final numbers.

**So `fix_1`'s headline conclusion is wrong.** *"The analytic-v MeanFlow JVP objective requires the
DiT backbone; the UNet does not learn it"* was **a capacity artifact, not an architecture result**.
The UNet learns the MeanFlow objective fine; it was a 63.8×-oversized UNet on 96 demonstrations
that did not.

**Eval (seed 6, n_trials = 2, K = 2):** all six tightened/DPCC projection arms reach `1.0 / 1.0 / 0
violations` on both-hard and top-left-hard, and 4 of 6 do so on top-right-hard as well. The single
residual failure is **`dpcc-c` on `top-right-hard` (SR = 0)** — and it is *not* the U3 "crushed to a
point" mode (§5).

**And it is Pareto-good (§4.2).** Against all 106 candidates of the 2026-08-02 DA batch, filtered to
seed 6 so the trial counts match: on `dpcc-t-tightened` and `dpcc-r-tightened` UNet@32 **dominates 8
configs each and is dominated by none** — at S&C = 1.00 it takes fewer steps *and* less time than
`mf_dit`, `bbdit`, FM ODE K20, DPCC K10/K20 and Diffusion/FMv3 K5/K10/K20. Only AlphaFlow K2 and
Diffusion/FMv3 K1 are non-dominated alternatives (faster per step, more steps to goal) — a
**trade-off, not a loss**. On `dpcc-c-tightened` it is genuinely worse (dominated by
Diffusion/FMv3 K1, 87 steps) — but `mf_dit` and AlphaFlow both score **0.00** there, so it is still
the best MeanFlow-family option at K = 2.

⚠️ **This is a smoke test, not a benchmark: one seed, two rollouts per cell.** Every eval number is
a multiple of 0.5. The *training* curves are the load-bearing evidence; the eval is directional.

---

## 1. Is the fix actually in the run? — yes, all three §5.1 checks pass

| Fix_8 §5.1 check | evidence | verdict |
|---|---|---|
| 1. The new `[ …TrajectoryModel ]` line appears with the expected size | `24317:140` and `24334` both print `backbone=unet  unet_width(freq_dim)=32  params=4.0M` | ✅ |
| 2. Config carries the fixed value | train log config dump: `freq_dim: 32`, `imf_backbone: unet` | ✅ |
| 3. Existing checkpoints still load | eval restored `step 99000` + EMA + optimizer/LR state without error | ✅ |

And the §5.2 hazard — a stale `model_config.pkl` claiming 256 next to a 32-wide checkpoint — was
**avoided**: the old `_bbunet_` tree was cleared before the run (`[ train ] Seed 6: no checkpoint
found, starting from step 0`), so `utils/config.py` wrote a fresh pkl with `freq_dim: 32`. Eval
rebuilt the backbone from that pkl and loaded `state_dict` cleanly.

Everything else in the run matches the `fix_1` A/B configuration: seed 6, `dp = 0.5`,
`t_schedule = logit_normal`, `p_mean = -0.4`, `dual_head = True`, `gradient_clip = 1.0`,
`lr = 5e-4`, 100 k steps, EMA eval at K = 2.

**Side benefit:** 8 h 07 m for 100 k steps (≈4.85 s/epoch-step-1000) versus ~11 h for the DiT run —
the correct-width UNet is the cheapest Gen3v6 arm to train.

---

## 2. Training — the three-way comparison

`fix_1`'s table, extended with this run. DiT = 23745, UNet@256 = 23813, UNet@32 = **24317**.

| signal (seed 6) | **DiT (23745)** | **UNet @256 (23813)** | **UNet @32 (24317)** |
|---|---|---|---|
| params | ~10 M† | **253.0 M** | **4.0 M** |
| `train/loss` final | 0.965 | **0.9998** (ceiling, min 0.9945) | **0.907** (min **0.848** @ep74) |
| `test/loss` final | 0.967 | **0.9998** | **0.912** (min = final epoch) |
| `diffusion_loss` final | ~1.9 | **2.0** exactly (saturated) | **1.81** (min 1.70) |
| `raw_mse_u` first → last | 56 → **1.67** (min ~1.0–1.4) | 64 → **69.6** (min 8.0, **rose**) | 19.3 → **1.90** (min **1.54** @ep74) |
| `per_dim_rms_u` first → last | 1.08 → **0.187** | 1.16 → **1.20** (never moved) | 0.635 → **0.199** (min 0.179) |
| `aux_loss` first → last | — | — | 17.4 → **2.16** (8×↓) |
| `a0_loss_test` final | — | 1.48-ish | **0.365** (min 0.298) |
| **best checkpoint** | late | **step 3000** (never beaten) | **late** — `test/loss` min is the *final* epoch |

† DiT param count is from Fix_8 §3(d)'s expectation (`params=10.xM`), not measured in 23745 — that
run predates the guard.

`test/loss` trajectory, UNet@32: **1.000 → 0.980 (ep25) → 0.958 (ep50) → 0.928 (ep75) → 0.912
(ep99)**, monotone from ep50 onward with max 0.960 over the second half. Contrast the 256-wide run,
which never left 0.9998.

**Residual instability is real but survivable.** 14 of 100 epochs end on a `raw_mse_u` spike > 20,
worst at **ep60 (920, `per_dim_rms_u` 4.38, `a0_loss` 27.9)**, with later spikes at ep85/86/96. This
is the same JVP-target outlier behaviour `fix_1` §2 and the first-run insight §2 documented on the
DiT (bursts to 32–68 there) — it is a property of the analytic-v tangent, not of the backbone. The
difference from the 256-wide run is that **the UNet@32 recovers from every spike** (ep60's 920 is
back to 2.9 by ep61 and 2.02 by ep70) instead of ratcheting upward.

⚠️ **Metric coverage caveat.** This train log only carries the tqdm-line metrics. The h-stratified
buckets (`h_mse_b0…b3`), `grad_norm`, `fm_frac` and `h_mean` that `fix_1` §2 leaned on exist only in
W&B (`FMPCC-MeanFlow/runs/ycyj8oq9`) and were not exported here. The rows above are the intersection
of what both logs contain; the h-bucket comparison is still owed.

---

## 3. Eval — full result, seed 6, n_trials = 2, K = 2, EMA

Job 24334, `config/meanflow_projection_eval.yaml`, 13 arms × 3 halfspaces.
`SR` = goal, `CS` = constraints satisfied, `S&C` = goal ∧ constraints, `v` = avg # violations.

### 3.1 `top-right-hard`

| arm | SR | CS | S&C | v | total viol | s/step |
|---|---|---|---|---|---|---|
| diffuser (unprojected) | 1.0 | 0.0 | 0.0 | 21.0 | 5.222 | 0.021 |
| dpcc-r | 0.5 | 0.0 | 0.0 | 2.5 | 0.054 | 0.026 |
| **dpcc-r-tightened** | **1.0** | **1.0** | **1.0** | **0** | 0.000 | 0.026 |
| **dpcc-c** | **0.0** | 0.0 | 0.0 | 3.5 | 0.074 | 0.026 |
| dpcc-c-tightened | 0.5 | 0.5 | 0.5 | 0 | 0.000 | 0.028 |
| dpcc-t | 0.5 | 0.5 | 0.5 | 1.5 | 0.020 | 0.026 |
| **dpcc-t-tightened** | **1.0** | **1.0** | **1.0** | **0** | 0.000 | 0.026 |
| hardflow_new-r | 0.5 | 0.0 | 0.0 | 3.0 | 0.073 | 0.115 |
| **hardflow_new-c** | **1.0** | **1.0** | **1.0** | **0** | 0.000 | 0.086 |
| hardflow_new-t | 0.5 | 0.0 | 0.0 | 1.5 | 0.007 | 0.110 |
| hardflow_new-r-tightened | 0.5 | 0.5 | 0.5 | 2.0 | 0.016 | 0.119 |
| **hardflow_new-c-tightened** | **1.0** | **1.0** | **1.0** | **0** | 0.000 | 0.091 |
| **hardflow_new-t-tightened** | **1.0** | **1.0** | **1.0** | **0** | 0.000 | 0.115 |

### 3.2 `top-left-hard`

| arm | SR | CS | S&C | v | total viol |
|---|---|---|---|---|---|
| diffuser | 1.0 | 0.0 | 0.0 | 13.5 | 0.832 |
| dpcc-r | 1.0 | 0.5 | 0.5 | 2.0 | 0.024 |
| **dpcc-r-tightened** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| **dpcc-c** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| **dpcc-c-tightened** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| dpcc-t | 1.0 | 0.5 | 0.5 | 2.0 | 0.012 |
| **dpcc-t-tightened** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| **hardflow_new-r** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| hardflow_new-c | 1.0 | 0.0 | 0.0 | 5.0 | 0.042 |
| **hardflow_new-t** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |
| **hardflow_new-{r,c,t}-tightened** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |

### 3.3 `both-hard`

| arm | SR | CS | S&C | v | total viol |
|---|---|---|---|---|---|
| diffuser | 1.0 | 0.0 | 0.0 | 12.0 | 1.460 |
| dpcc-r | 1.0 | 0.5 | 0.5 | 0.5 | 0.001 |
| dpcc-t | 1.0 | 0.5 | 0.5 | 1.5 | 0.006 |
| hardflow_new-c | 1.0 | 0.5 | 0.5 | 2.5 | 0.075 |
| **all six DPCC/hardflow `-tightened` + dpcc-c + hardflow_new-{r,t}** | 1.0 | 1.0 | 1.0 | 0 | 0.000 |

### 3.4 Pooled over the three halfspaces (6 episodes/arm)

| arm | SR | CS | S&C |
|---|---|---|---|
| **dpcc-r-tightened**, **dpcc-t-tightened**, **hardflow_new-c-tightened**, **hardflow_new-t-tightened** | 1.000 | 1.000 | **1.000** |
| dpcc-c-tightened, hardflow_new-r-tightened | 0.833 | 0.833 | 0.833 |
| hardflow_new-r, hardflow_new-t | 0.833 | 0.667 | 0.667 |
| dpcc-c | 0.667 | 0.667 | 0.667 |
| dpcc-t, hardflow_new-c | 0.833–1.000 | 0.500 | 0.500 |
| dpcc-r | 0.833 | 0.333 | 0.333 |
| **diffuser (unprojected)** | **1.000** | **0.000** | **0.000** |

The DPCC design intent holds on this backbone: **the generative brain reaches the goal everywhere
(`diffuser` SR = 1.0 in all three envs), and projection supplies the safety it lacks (`CS` 0.0 →
1.0).**

---

## 4. The comparison that matters — width 32 vs width 256, same backbone

Shared arms only (the old eval used `model_free`/`gradient`/`post_processing`; this one uses the U3
`hardflow_new-*` set, so only `diffuser` and the `dpcc-*` family are common). Format `g / b / v`.

| arm | env | **UNet @256 (23814)** | **UNet @32 (24334)** |
|---|---|---|---|
| **dpcc-c-tightened** | both-hard | g1 / b1 / v0 | g1 / b1 / v0 |
| | top-left | g1 / **b0** / v3.5 | **g1 / b1 / v0** |
| | top-right | g0.5 / b0.5 / v0 | g0.5 / b0.5 / v0 |
| **dpcc-r-tightened** | both-hard | g1 / **b0.5** / **v2.5** | **g1 / b1 / v0** |
| | top-left | g1 / **b0.5** / **v2** | **g1 / b1 / v0** |
| | top-right | **g0** / b0 / v0 | **g1 / b1 / v0** |
| **dpcc-t** | both-hard | g1 / b0 / **v14** | g1 / b0.5 / **v1.5** |
| | top-left | g1 / b0 / **v41** | g1 / b0.5 / **v2** |
| | top-right | **g0** / b0 / v0 | **g0.5** / b0.5 / v1.5 |
| **diffuser** (unprojected) | both-hard | g1 / b0 / v21.5 | g1 / b0 / **v12** |
| | top-left | g1 / b0 / v27 | g1 / b0 / **v13.5** |
| | top-right | **g0.5** / b0 / v17.5 | **g1** / b0 / v21 |

**Reading:**

1. **Goal-reaching is restored where it had collapsed.** `top-right-hard` was the 256-wide run's
   catastrophic cell — `g = 0` for `dpcc-r-tightened` and `dpcc-t`, `g = 0.5` unprojected. At width
   32 the unprojected field reaches the goal in **every** env, and `dpcc-r-tightened` /
   `dpcc-t-tightened` are clean 1/1/0 there.
2. **Violation counts collapse by an order of magnitude** on the un-tightened arms: `dpcc-t`
   top-left **41 → 2**, both-hard **14 → 1.5**. The projector no longer has to drag a bad
   trajectory a long way back.
3. **The unprojected field improved on 2 of 3 envs** (21.5 → 12, 27 → 13.5) and is nominally worse
   on `top-right` (17.5 → 21) — but the 256-wide run only reached the goal on half those episodes,
   so it had fewer steps in which to violate anything. At n = 2 this cell means nothing either way.

### 4.1 vs the DiT reference (23777)

| `dpcc-c-tightened` | both-hard | top-left | top-right |
|---|---|---|---|
| **DiT (23777)** | g1 / b1 / v0 | g1 / b1 / v0 | **g1 / b1 / v0** |
| **UNet @32 (24334)** | g1 / b1 / v0 | g1 / b1 / v0 | **g0.5 / b0.5 / v0** |
| **UNet @256 (23814)** | g1 / b1 / v0 | g1 / **b0** / v3.5 | g0.5 / b0.5 / v0 |

On the DiT's own headline cell the UNet@32 is one episode short on `top-right-hard` — but the DiT
run had its own 0.5 cell there (`dpcc-t-tightened`: g0.5/b0.5), where the UNet@32 scores 1/1/0.
**At n = 2 the two backbones are indistinguishable.** Claiming either direction from these tables
would be over-reading; the honest statement is *"UNet@32 is in the same regime as the DiT, and the
`fix_1` gap is gone."*

### 4.2 Pareto position against the 2026-08-02 candidate field

**Source:** `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_raw.csv`
(106 candidates). **Filtered to `seed == 6`**, so every row below is the same 2 trials × 3
halfspaces = 6 episodes as the new run — no seed-count confound.

**"Good" here means Pareto-dominant on the triple `(S&C ↑, n_steps ↓, avg_time ↓)`**: at equal or
better success-and-constraints, *both* fewer steps *and* lower time. Winning one and losing the
other is a **trade-off, not a win**. Dominance is computed with a 10 % relative tolerance on
`avg_time` (cross-job wall-clock noise) and 0.5 steps on `n_steps`.

Note `CAND_98` in that batch **is** the old 253 M UNet — its per-env S&C on `dpcc-c-tightened`
(`[0.5, 0.0, 1.0]`) reproduces `fix_1` §3 exactly, so old-vs-new here is same-pipeline,
same-metric.

#### `dpcc-t-tightened` — the arm where MeanFlow's low-K claim lives

| config | S&C | steps | s/step | viol |
|---|---|---|---|---|
| **MeanFlow UNet@32 K2 (NEW)** | **1.00** | **58.67** | 0.0270 | 0 |
| Diffusion/FMv3 K10 | 1.00 | 59.67 | 0.1912 | 0 |
| DPCC K10 | 1.00 | 61.50 | 0.3167 | 0 |
| DPCC K20 | 1.00 | 62.00 | 0.5961 | 0 |
| AlphaFlow (sit) K2 | 1.00 | 63.17 | **0.0201** | 0 |
| FM ODE K20 | 1.00 | 63.50 | 0.4679 | 0 |
| MeanFlow mf_dit K2 | 1.00 | 65.50 | 0.0269 | 0 |
| Diffusion/FMv3 K5 | 1.00 | 65.50 | 0.1522 | 0 |
| Diffusion/FMv3 K20 | 1.00 | 66.33 | 0.4517 | 0 |
| Diffusion/FMv3 K1 | 1.00 | 66.50 | **0.0172** | 0 |
| MeanFlow DiT (bbdit) K2 | 0.83 | 65.67 | 0.0368 | 0 |
| DPCC K1 | 0.50 | 55.33 | 0.0232 | 2.33 |
| MeanFlow UNet@256 K2 | **0.00** | 51.50 | 0.0571 | 4.67 |

> **UNet@32 dominates 8 configs** (mf_dit K2, bbdit K2, FM ODE K20, DPCC K10, DPCC K20,
> Diffusion/FMv3 K5/K10/K20) and **is dominated by none — Pareto-non-dominated.**
> It posts the **lowest step count of the entire perfect-safety club**.

The two configs it does *not* dominate are **AlphaFlow K2** (0.0201 s) and **Diffusion/FMv3 K1**
(0.0172 s): they are ~25–35 % faster per step but need 4.5 and 7.8 more steps. That is a genuine
trade-off, not a loss — and the honest reading is that the three of them form the frontier.

#### `dpcc-r-tightened`

| config | S&C | steps | s/step |
|---|---|---|---|
| **MeanFlow UNet@32 K2 (NEW)** | **1.00** | **63.17** | 0.0267 |
| AlphaFlow (sit) K2 | 1.00 | 64.33 | 0.0199 |
| Diffusion/FMv3 K10 | 1.00 | 69.67 | 0.1984 |
| Diffusion/FMv3 K1 | 1.00 | 71.00 | 0.0172 |
| MeanFlow mf_dit K2 | 0.83 | 66.33 | 0.0259 |
| DPCC K10 / K20 | 0.83 | 66.50 / 65.33 | 0.3399 / 0.5633 |
| MeanFlow DiT (bbdit) K2 | 0.83 | 67.50 | 0.0418 |
| FM ODE K20 | 0.83 | 71.50 | 0.6755 |
| MeanFlow UNet@256 K2 | 0.33 | 54.67 | 0.0369 |

Same verdict: **dominates 8, dominated by 0.** Here it is one of only four configs that clear
S&C = 1.00 at all.

#### `dpcc-c-tightened` — where it is *not* good

| config | S&C | steps | s/step |
|---|---|---|---|
| DPCC K10 | 1.00 | 59.83 | 0.3209 |
| Diffusion/FMv3 K1 | 1.00 | 70.50 | 0.0173 |
| MeanFlow DiT (bbdit) K2 | 1.00 | 71.17 | 0.0380 |
| **MeanFlow UNet@32 K2 (NEW)** | **0.83** | **87.33** | 0.0270 |
| MeanFlow UNet@256 K2 | 0.50 | 55.83 | 0.2180 |
| **MeanFlow mf_dit K2** | **0.00** | **199.00** | 0.0245 |
| **AlphaFlow (sit) K2** | **0.00** | **199.00** | 0.0187 |

**UNet@32 is dominated here** — by Diffusion/FMv3 K1 (higher S&C, fewer steps, faster). 87.33 steps
is the worst of any config that reaches the goal at all: `-c` picks the minimal-correction plan and
the UNet's field makes it dawdle.

**But the 199.00 rows are the point.** `199` = `max_episode_length - 1`, i.e. **timeout** — that is
the U3 "crushed to a point" collapse, and **both `mf_dit` and AlphaFlow score 0.00 on it at K=2**.
The UNet@32 does **not** have that failure mode: it scores 0.83 and its plots show full-length
motion (§5). So on the one arm where UNet@32 looks weak, it is still the best MeanFlow-family
option at K=2 by a wide margin.

### 4.3 Per-environment consistency, and the unprojected field

The pooled step ranking above is only meaningful if it survives per-env inspection (n = 2 per cell).

`dpcc-t-tightened`, steps per env (S&C in parens):

| config | top-right | top-left | both-hard | sweep vs UNet@32 |
|---|---|---|---|---|
| **UNet@32 (NEW)** | 57.0 ± 4.0 (1.0) | 60.5 ± 0.5 (1.0) | 58.5 ± 4.5 (1.0) | — |
| MeanFlow mf_dit K2 | 65.5 (1.0) | 66.0 (1.0) | 65.0 (1.0) | **UNet wins 3/3** |
| DPCC K10 | 60.0 (1.0) | 60.0 (1.0) | 64.5 (1.0) | 2/3 |
| Diffusion/FMv3 K10 | 61.5 (1.0) | 57.0 (1.0) | 60.5 (1.0) | 2/3 |
| AlphaFlow (sit) K2 | 70.5 (1.0) | 56.0 (1.0) | 63.0 (1.0) | 2/3 |

**Only the comparison against its own generation's DiT sibling is a clean 3/3 sweep.** Against the
rest the win is 2/3 — so the #1 pooled ranking is partly env-level noise, and the defensible claim
is *"UNet@32 reaches the goal in fewer steps than mf_dit in every halfspace, and is competitive
with the rest"*, not *"it is the fastest-to-goal config."*

Unprojected `diffuser` arm — raw field quality, avg # violations (lower = better field):

| config | top-right | top-left | both-hard | mean |
|---|---|---|---|---|
| AlphaFlow (sit) K2 | 20.0 | 10.5 | 1.0 | **10.5** |
| MeanFlow DiT (bbdit) K2 | 20.0 | 14.5 | 5.0 | 13.2 |
| Diffusion/FMv3 K1 | 26.0 | 7.5 | 12.0 | 15.2 |
| **MeanFlow UNet@32 (NEW)** | 21.0 | 13.5 | 12.0 | **15.5** |
| MeanFlow mf_dit K2 | 12.5 | 24.5 | 11.5 | 16.2 |
| MeanFlow UNet@256 K2 | 17.5 | 27.0 | 21.5 | **22.0** |
| DPCC K10 | 16.5 | 29.0 | 21.0 | 22.2 |

**Mid-pack: better than its own 256-wide ancestor (22.0) and than `mf_dit` (16.2), worse than
`bbdit` (13.2) and AlphaFlow (10.5).** So UNet@32's Pareto strength on the projected arms is *not*
explained by a better raw field — it comes from the field being **easier for the projector to
correct** (fewer steps at zero violations), which is the DPCC design intent working.

---

## 5. The one residual failure: `dpcc-c` on `top-right-hard` (SR = 0)

Worth isolating, because Gen3v6 already has a documented `dpcc-c` pathology —
[`U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md),
where the `mf_dit` checkpoint had a K=2-only "stay put" generation mode: plans collapsed to 8
waypoints within 1e-4 of the current position, `-c` locked onto them because a motionless plan is
trivially the cheapest to leave alone, and the agent never left its start point.

**This is not that.** From `all_seeds/top-right-hard/dpcc-c.png`, both trials travel the full
corridor from `(0.52, -0.28)` up to `y ≈ 0.23` and stall against the halfspace boundary near the
`(0.5, 0.26)` obstacle — full-length motion, 3.5 violations, just no goal crossing at `y = 0.35`.
The `diffuser` arm on the same checkpoint crosses the line, so the **generator is fine and the `-c`
selection rule is stalling at the constraint boundary**, a different (and milder) failure than the
U3 collapse.

Supporting detail: `dpcc-c-tightened` on the same env is **exactly one trial each way** — one run
threads the right-hand corridor to the goal, the other loops at `(0.55, 0.19)`. That is the whole
of the 0.5.

Also note `Avg number of steps: 0.00` on the `dpcc-c` row is **not** a degenerate rollout — the
script averages `n_steps` over *successful* trials only and prints 0 when SR = 0
(`eval_flow_matching_v3_meanflow.py:518`).

---

## 6. Caveats — read before quoting any of this

- 🔴 **One seed (6), two trials per cell.** Every eval number is 0.0/0.5/1.0. Directional only. The
  training curves (§2) are the trial-count-independent part of this result.
- 🔴 **The width-32-vs-width-256 comparison is NOT a one-variable A/B.** 23813 ran 2026-07-25;
  24317 ran 2026-08-06. Four commits touched Gen3v6's train-side files in between
  (`b82e290` U2 mf_dit backbone, `d97eb92` + `15b82d6` fix_6/6.2 auto-resume, `2d85f03` Fix_8
  itself). None of them alters the UNet branch or the MeanFlow objective, but that is an inspection,
  not an audited diff. The eval side changed more (U3 HardFlow port, fix_4/5/7).
- ⚠️ **The arm set changed.** 23814 evaluated `model_free`/`gradient`/`post_processing`; 24334
  evaluates the `hardflow_new-*` family instead. Only `diffuser` and `dpcc-*` are comparable.
- ⚠️ **`T1` in the results path is a naming artifact, not a threshold change.** The exp_name's `T`
  token is read from `config/projection_eval.yaml` (`config/avoiding-d3il.py:9-12`), whose cluster
  copy had `1` at run time — the snapshot in the results dir proves it. The value the eval script
  *actually* gates on comes from `config/meanflow_projection_eval.yaml`, which is `0.5` in git at
  `b237c2b` — the same threshold as the DiT baseline's `T0.5`. The snapshot mechanism did not save
  `meanflow_projection_eval.yaml`, so the runtime value is inferred from git, not verified from the
  run. **Worth fixing: snapshot the yaml the eval actually loaded.**
- ⚠️ `HFFM_BATCH=4` was set in the sbatch (log: `batch(mpc)=4`) although the yaml default is 1, so
  the `-r`/`-c`/`-t` hardflow arms are genuinely distinct here.
- 🔴 **The §4.2 Pareto tables cross two different batches.** The candidate rows come from jobs run
  2026-08-02, the UNet@32 row from 2026-08-06. `avg_time` is wall-clock per step and moves with GPU
  contention, so **differences under ~10–20 % are not resolvable** — the 0.0270 vs 0.0269 gap
  against `mf_dit` is a tie, and only the order-of-magnitude gaps (K=2 at ~0.027 s vs DPCC K10/K20
  at 0.32/0.60 s) carry weight. `n_steps` is hardware-independent and is the sounder axis.
- ⚠️ **`n_steps` is averaged over *successful* trials only** (`eval_…meanflow.py:518`), so it may
  only be compared between rows at equal S&C. Two artifacts to recognise: `0.00` means SR = 0
  (§5), and `199.00` means the episode hit `max_episode_length` — a timeout, not a path length.
  UNet@256's flattering `51.50` steps on `dpcc-t-tightened` is exactly this bias (S&C = 0.00).
- ⚠️ **Window-level train/test split leak** (inherited, POST_U10_III §4.2): at H = 8 adjacent
  windows share 7/8 frames, so `test/loss` is optimistic. It affects the DiT and UNet arms equally.
- ⚠️ tqdm bars leaked into the batch log again (`Epoch N: 100%|…`), against the repo convention.

---

## 7. What this changes

1. **`fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md` needs a correction header.** Its §0/§6
   verdict ("MeanFlow needs the DiT"), its §4.1 claim that the config default `imf_backbone='dit'`
   is "now empirically justified", and its §4.2 mechanism story (additive `time_mlp(t)+h_mlp(h)`
   cannot represent the two-time field) all rest on a run whose only real defect was 63.8× excess
   width. The §1 "it is a true A/B (one variable)" claim is the specific sentence that was wrong —
   the two runs differed in backbone *and*, unknowingly, in 249 M parameters.
2. **The architecture control Fix_8 §7 asked for now exists**, and it comes back positive: the UNet
   is a viable Gen3v6 backbone, and the cheapest one to train (8.1 h vs ~11 h).
3. **The Gen3v4/Gen3v7 UNet arms are now suspect in the same way** — any `bbunet` run in those
   generations before `2d85f03` carries the same 253 M backbone. `fix_1`'s conclusion was used to
   justify `dit`/`sit` defaults there too.
4. **`MASTER_TEST_HISTORY.md`** — the Gen3v6 row (and the Gen3v4 row Fix_8 §4 already flagged) wants
   a note. **Not edited here** (standing convention: never self-edit the master index).

## 8. Next, in priority order

1. **Multi-seed.** Seeds 7–10 at width 32, one seed per job, then re-eval with `n_trials ≥ 10`. Until
   then neither the "UNet works" nor the "UNet ≈ DiT" claim has an error bar. (Note 24318 already
   failed by evaluating an untrained seed 7 — the driver should check for the checkpoint first.)
2. **Matched UNet@32-vs-mf_dit A/B** at the same K and the same eval yaml, both under the current
   HardFlow arm set. The §4.1 comparison above is against a 2026-07-24 DiT run with a different arm
   list; it is a sanity check, not the A/B.
3. **Export the h-stratified buckets** (`h_mse_b0…b3`, `grad_norm`, `h_mean`) from W&B run
   `ycyj8oq9` and finish the §2 table. The `fix_1` claim that the UNet's large-h buckets "blow up"
   (b1 2.5e4, b3 9.7e4) has no width-32 counterpart yet.
4. **Probe the `dpcc-c` / `top-right-hard` stall** (§5) — is it the same `(r, t)` interior-weakness
   the U3 investigation localized on `mf_dit`, or a plain projector-feasibility stall? A K-sweep
   {1, 2, 5, 20} on this checkpoint answers it the same way it did there.
5. **Snapshot the eval yaml that was actually loaded**, not just `config/projection_eval.yaml`
   (§6 caveat 4).
6. **Re-run UNet@32 inside a DA batch** so §4.2 is a within-batch Pareto comparison instead of a
   cross-batch one, and the `avg_time` axis becomes claimable at fine resolution.
7. **Add UNet@32 to the low-K ablation** (`DA/DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`). Its L3
   leg — "low K is a capability, not a discount" — currently rests on AlphaFlow alone; a second
   K=2 config that dominates DPCC K10/K20 on both steps and time strengthens it, and a
   **UNet@32 K-ladder {1, 2, 5, 10}** would give MeanFlow the ladder §1 of that doc lists as missing.

## 9. One-line verdict

**The UNet was never broken by the MeanFlow objective — it was broken by being 63.8× too wide.** At
`freq_dim=32` it trains to DiT-comparable field quality in 8 h, gives 100 %-safe K=2 control on
4 of 6 tightened arms across all three halfspaces, and on `dpcc-t/r-tightened` is
**Pareto-non-dominated against the entire 2026-08-02 candidate field** — dominating DPCC K10/K20,
FM ODE K20 and its own `mf_dit` sibling on steps *and* time. `fix_1`'s "MeanFlow needs the DiT" is
retracted.
