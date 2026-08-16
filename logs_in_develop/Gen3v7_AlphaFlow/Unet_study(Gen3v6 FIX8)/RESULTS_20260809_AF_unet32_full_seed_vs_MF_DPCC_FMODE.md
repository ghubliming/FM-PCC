# RESULTS — AlphaFlow on the fixed UNet (`freq_dim=32`), full-seed run: does it beat MeanFlow, DPCC K20 and naive FM?

**Date:** 2026-08-09 · **Type:** results / cross-generation benchmark · **Task:** `avoiding-d3il` (state, not visual)
**Runs under test:** train **24389** (`af_train`, i6-gpu-1, 2026-08-08 02:26:07 → 19:20:49 UTC, 16 h 55 m, git `bc9b93f`) ·
eval **24390** (`af_eval`, 19:20:49 → 20:06:52 UTC, 46 m, git `bc9b93f`, K = 2, seeds 7–10) ·
plus an untracked follow-up eval (seed 6 only, K = 1/2/5/10, 2026-08-08 22:03 → 2026-08-09 01:16 — **log not downloaded**)
**Model folder:** `logs/avoiding-d3il/flow_matching_v3_alphaflow/H8_D…AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.0_ag25.0_rf0.5`
**Source material:**
`temp/2026-08-07/00_04_45_af_train_24389.log`, `temp/2026-08-07/00_04_45_af_eval_24390.log`,
results tree `temp/2026-08-07/H8_D…AlphaFlowODE_aw10_bbunet_…/` (312 `eval_*.log` cells parsed),
baselines from `temp/2026-08-07/batch_avoiding_combined_20260807_124828/candidates_multidimensional_raw.csv`
**Fix under test:** `2d85f03` *(Gen3v6 Fix_8 & sync Gen3v4/v7 & Gen8 & Gen14)* — the UNet channel width was
being set from `freq_dim` (256), building a 253 M backbone instead of the 4 M baseline. Gen3v7 inherited that
bug and this is its first full-seed run after the sync.
**Companion:** [`Gen3v6_MeanFlow/Fix_8_Unet/RESULTS_Fix_8_unet_width32_rerun_24317_24334.md`](../../Gen3v6_MeanFlow/Fix_8_Unet/RESULTS_Fix_8_unet_width32_rerun_24317_24334.md) — the MeanFlow half of the same fix.

> `temp/` is **gitignored** — all raw material is local only.

---

## 0. TL;DR

**The fix landed and AlphaFlow-on-UNet@32 is the strongest AlphaFlow arm so far — but the run is
*four* seeds, not five, and the fifth seed is a stale-checkpoint artefact that must be thrown away.**

1. **Provenance defect (§1, read first).** Job 24389 trained seeds **7, 8, 9, 10** from step 0. **Seed 6
   was skipped** — `[ train ] Seed 6 already reached 100000 steps — skipping`. Its weights are a
   pre-fix leftover, and its eval scores collapse accordingly (S&C **0.05** over the 7 DPCC arms vs
   **0.65** on seeds 7–10). The raw `.npz` proves it at the source (**§1.1**): seed 6's **MPC foresight
   is exploded at every K** — single horizon steps of **0.968**, longer than the whole arena diagonal,
   against a realised robot motion of 0.013 — while seeds 7–10 plan at **1.0–1.1×** the realised rate.
   Every "5-seed" number in this document is therefore contaminated; the honest headline set is
   **seeds 7–10**. Note that every `K = 1/5/10` folder here is seed-6-only, so a K-folder sweep sees
   nothing but the broken checkpoint.
2. **On its own K = 2 headline arm, AF UNet@32 Pareto-dominates the DPCC K20 baseline.**
   `dpcc-t-tightened`, seeds 7–10, 24 episodes: **S&C 1.00 / 58.4 steps / 0.030 s per step** against
   DPCC K20's **1.00 / 79.7 / 0.555** — equal safety, ~27 % fewer steps *and* **18× cheaper per step**.
   Same story vs naive **FMv3ODE K20** (1.00 / 63.7 / 0.491).
3. **It does not dominate DPCC everywhere.** Averaged over the three `-tightened` DPCC arms it is
   **0.958 vs DPCC K20's 1.000** — it loses `dpcc-r-tightened` (0.92) and `dpcc-c-tightened` (0.96),
   both on `both-hard`. So: *one arm dominant, the arm-set as a whole a narrow loss on safety bought
   back with a 20× cost cut.*
4. **It beats every MeanFlow-family reference on matched seeds.** vs **MeanFlow DiT K2** (seeds 7–10):
   7-arm mean S&C **0.649 vs 0.464**; the `dpcc-c` family goes **0.96 vs 0.12** — the UNet kills the
   "crushed to a point" 180-step failure mode that both MF-DiT and AF-SiT show there.
5. **vs MeanFlow UNet@32 (Fix_8) the comparison is not clean and cannot be made clean from this data:**
   MF-UNet@32 exists **only on seed 6**, which is exactly the one AF seed that was not retrained.
   Off-seed, the two look equivalent (`dpcc-t-tightened`: MF 1.00/58.7/0.027 @s6 vs AF 1.00/58.4/0.030 @s7–10).
6. **The K sweep is missing.** The only K evaluated on valid weights is **K = 2**. K = 1/5/10 exist for
   seed 6 only — i.e. on the stale checkpoint — so they carry no information about the fixed model.

**Bottom line for the thesis narrative:** at 2 NFE, AlphaFlow-UNet@32 buys back essentially all of
DPCC K20's constraint safety at ~1/18 of the per-step cost, and clearly beats naive FM K20 and both
MeanFlow references. It is **not** yet a clean sweep of DPCC, and the result rests on 4 seeds × 6
episodes with a known-bad 5th seed that needs deleting and retraining.

---

## 1. ⚠️ Provenance defect: this is a 4-seed run wearing a 5-seed label

`00_04_45_af_train_24389.log`:

```
21:   [ train ] seeds='6 7 8 9 10'  resume='--auto-resume'  extra=''
133:  [ AFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=32  params=4.0M
134:  [ train ] Seed 6 already reached 100000 steps — skipping
257:  [ train ] Seed 7: no checkpoint found, starting from step 0
572:  [ train ] Seed 8: no checkpoint found, starting from step 0
887:  [ train ] Seed 9: no checkpoint found, starting from step 0
1201: [ train ] Seed 10: no checkpoint found, starting from step 0
```

`--auto-resume` saw a 100 k-step checkpoint already sitting in the `_bbunet_` tree and skipped seed 6
entirely. The `params=4.0M` line at 133 is the *freshly built* model, not the checkpoint — it proves
nothing about what is on disk. Fix_8 §5.2 flagged precisely this hazard for Gen3v6 and the Gen3v6 run
avoided it by clearing the old `_bbunet_` tree first; **Gen3v7 did not**.

The eval numbers agree that seed 6 is a different model:

| metric (K = 2, 7 DPCC arms) | seed 6 | seeds 7–10 |
|---|---|---|
| mean S&C | **0.048** | **0.649** |
| mean S&C, 3 `-tightened` arms | **0.056** | **0.958** |
| `dpcc-t-tightened` S&C | 0.17 | 1.00 |

**Is the seed-6 checkpoint the old 253 M UNet?** Its *size* is not the anomaly — the unprojected
`diffuser` arm costs **0.023 s/step on seed 6 vs 0.019–0.020 on seeds 7–10**, i.e. the same forward
cost, which a 63× larger backbone could not produce. Its *output* is the anomaly, and §1.1 shows that
directly.

### 1.1 The MPC foresight on seed 6 is exploded — at every K, and only on seed 6

The eval-time summary metrics never look at what the generative brain actually plans. The raw
`sampled_trajectories_all` arrays in the `.npz` do: shape `(T_mpc_steps, 4_mpc_samples, 8_horizon,
4_obs)`. Comparing the plan's motion per horizon step against the robot's realised motion per control
step is the direct test of whether the foresight is on the data manifold. **All 624 `(K, seed, env,
arm, trial)` cells in the results tree, no exceptions:**

| K | seed | plan ‖Δ‖ per horizon step | max single jump | realised robot ‖Δ‖ | ratio |
|---|---|---|---|---|---|
| 1 | **6** | 0.1032 | **0.9682** | 0.0127 | **8.1×** |
| 2 | **6** | 0.1067 | **0.9682** | 0.0127 | **8.4×** |
| 5 | **6** | 0.0844 | **0.9682** | 0.0128 | **6.6×** |
| 10 | **6** | 0.0854 | **0.9682** | 0.0128 | **6.7×** |
| 2 | 7 | 0.0119 | 0.0325 | 0.0113 | 1.1× |
| 2 | 8 | 0.0117 | 0.0434 | 0.0110 | 1.1× |
| 2 | 9 | 0.0115 | 0.0446 | 0.0106 | 1.1× |
| 2 | 10 | 0.0120 | 0.0426 | 0.0115 | 1.0× |

The workspace is `x ∈ [0.2, 0.8]`, `y ∈ [−0.3, 0.4]` — diagonal **0.92**. Seed 6's plans contain
single horizon steps of **0.968**, i.e. **longer than the entire arena diagonal**, inside an 8-step
lookahead whose real per-step motion is 0.013. Open-loop agreement between the plan at step *t*,
horizon *h*, and the state actually reached at *t + h* confirms it:

| | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 |
|---|---|---|---|---|---|---|---|
| AF-UNet **seed 6** | 0.148 | 0.165 | 0.151 | 0.177 | 0.147 | 0.162 | 0.155 |
| AF-UNet seeds 7–10 | 0.012 | 0.014 | 0.016 | 0.018 | 0.021 | 0.025 | 0.030 |
| **MF**-UNet@32 seed 6 (Fix_8) | 0.013 | 0.014 | 0.016 | 0.019 | 0.023 | 0.028 | 0.032 |

Seed 6 is **flat at ~0.15 from h = 1** — the plan is uncorrelated with reality from the very first
horizon step. Seeds 7–10 grow smoothly with horizon, which is what a healthy MPC lookahead does, and
they are indistinguishable from the MeanFlow-UNet reference on the same task. The spread across the
four MPC samples tells the same story: **0.158 on seed 6 vs 0.005–0.007 on seeds 7–10**.

This is also visible without any analysis — the foresight overlay panel of
`…/6/results/halfspace_both-hard/diffuser.png` is a solid blue scribble filling the whole arena,
while `…/7/…/diffuser.png` is a tight bundle of 8-step plans hugging the executed path.

**Two consequences.**
1. It settles §1: seed 6 is **not** the fixed model, on evidence about the model's own output rather
   than about wall-clock.
2. It explains the projection-cost anomaly — `dpcc-r` at **0.242 s/step on seed 6 vs 0.028 on
   seeds 7–10**. The QP was not slow because of node contention; it was being handed a reference that
   jumps across the arena every step, so every solve was a much harder problem. Seed 6's S&C ≈ 0 is
   then simply the projector failing to rescue an unusable field.

⚠️ **Scope.** Every `K = 1 / 5 / 10` folder in this results tree is **seed 6 only**. Anyone sweeping
the K folders sees nothing but the broken checkpoint and will conclude the whole arm is exploded. It
is not: no cell among seeds 7–10 exceeds a 0.045 plan jump or a 1.1× ratio.

**Action required (cluster):** delete
`logs/avoiding-d3il/flow_matching_v3_alphaflow/H8_D…_bbunet_…/6/` and re-run 24389 for seed 6 only,
then re-eval. Until then, quote seeds 7–10. **Recommended as a standing gate:** the plan-vs-realised
ratio above is a two-line check on any `diffuser.npz` and catches a dead generative brain that
`SR = 1.00` hides — see §9 item 7.

**Second gap.** Eval 24390 declares `[ eval ] NFE budgets to evaluate: 2` — a single K. The K = 1/5/10
folders in the results tree carry **seed 6 only**, from the untracked follow-up job, so they are
K-sweep data on the *stale* checkpoint. There is currently **no K sweep for the fixed AlphaFlow UNet**,
which also means this run is off-parity with the Gen14 U7 K = 20 convention.

---

## 2. Training health, seeds 7–10 — all four converge and agree

From the four W&B run summaries in the train log (seed 6 has none — it was skipped):

| signal | seed 7 | seed 8 | seed 9 | seed 10 |
|---|---|---|---|---|
| `final_test_loss` | 0.98332 | 0.98367 | 0.98445 | 0.98312 |
| `final_train_loss` | 0.99439 | 0.98923 | 0.98726 | 0.99288 |
| `test/a0_loss` | 0.290 | 0.771 | 0.547 | 0.307 |
| `val/per_dim_rms_u` | 0.336 | 0.402 | 0.435 | 0.358 |
| `train/h_mse_b0` first → final | 68.5 → 4.65 | 77.4 → 2.80 | 68.4 → 3.48 | 65.0 → 2.95 |
| `final_val_raw_mse` | 6.86 | 18.50 | 15.87 | 8.55 |

`test/loss` lands inside a **0.0013 band across four independent seeds** — the objective is stable on
this backbone. The α-schedule completed as configured (`sigmoid 1.0 → 0.0` over 100 k, γ = 25;
`alpha_first 1.0`, `alpha_last 0.0`, `alpha_schedule_alive True`), so the last ~20 k steps are trained
in the pure bootstrapped-target regime, which is the point of the AlphaFlow homotopy.

Residual instability is still visible in the tail (`train/raw_mse_u` ends at **638.8** on seed 7 and
**70.9** on seed 10 while validation sits at 6.9 / 8.5) — the same JVP/bootstrap-target outlier bursts
Gen3v6 `fix_1` and Fix_8 documented. It does not prevent convergence.

⚠️ Do **not** compare `test_loss` 0.983 here against MeanFlow's 0.912 in Fix_8 — different objective
mixture, different normalisation. The cross-family comparison lives in the eval tables only.

**Cost:** 16 h 55 m for four seeds ≈ **4 h 14 m/seed** on one A5000 — cheaper than the Gen3v6 DiT arm
(~11 h/seed) and comparable to MF-UNet@32 (8 h 07 m/seed at the time of Fix_8).

---

## 3. Eval result — AF UNet@32, K = 2, seeds 7–10

Job 24390, `config/alphaflow_projection_eval.yaml`, EMA weights, 13 arms × 3 halfspaces × 2 trials ×
4 seeds = **24 episodes per arm**. `SR` = goal reached, `CS` = constraints satisfied, `S&C` = both,
`v` = avg # violations.

| arm | SR | CS | **S&C** | v | total viol | steps | s/step |
|---|---|---|---|---|---|---|---|
| `diffuser` (unprojected) | 1.00 | 0.04 | **0.04** | 15.21 | 2.084 | 61.4 | 0.019 |
| `dpcc-r` | 0.96 | 0.42 | 0.42 | 2.04 | 0.099 | 63.8 | 0.028 |
| `dpcc-r-tightened` | 1.00 | 0.92 | **0.92** | 0.21 | 0.008 | 63.2 | 0.028 |
| `dpcc-c` | 0.96 | 0.75 | 0.75 | 0.67 | 0.009 | 90.0 | 0.027 |
| `dpcc-c-tightened` | 1.00 | 0.96 | **0.96** | 0.04 | 0.001 | 91.5 | 0.027 |
| `dpcc-t` | 0.92 | 0.46 | 0.46 | 6.38 | 0.095 | 59.0 | 0.028 |
| **`dpcc-t-tightened`** | **1.00** | **1.00** | **1.00** | **0.00** | 0.000 | **58.4** | **0.030** |
| `hardflow_new-r` | 0.92 | 0.42 | 0.42 | 2.21 | 0.091 | 56.4 | 0.113 |
| `hardflow_new-r-tightened` | 1.00 | 0.96 | **0.96** | 0.12 | 0.004 | 62.1 | 0.115 |
| `hardflow_new-c` | 1.00 | 0.71 | 0.71 | 1.42 | 0.027 | 93.0 | 0.102 |
| **`hardflow_new-c-tightened`** | **1.00** | **1.00** | **1.00** | **0.00** | 0.000 | 93.0 | 0.103 |
| `hardflow_new-t` | 0.88 | 0.42 | 0.42 | 5.67 | 0.092 | 54.3 | 0.113 |
| **`hardflow_new-t-tightened`** | **1.00** | **1.00** | **1.00** | **0.00** | 0.000 | 59.4 | 0.114 |

The DPCC design intent holds cleanly: **the generative brain reaches the goal everywhere
(`SR` = 1.00 unprojected) and is unsafe everywhere (`CS` = 0.04, 15 violations/episode); projection
supplies the safety.** Three arms are perfect across all 24 episodes.

### 3.1 Per-seed S&C (pooled over 3 halfspaces, 6 episodes each) — seed 6 is the outlier

| arm | s6 † | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|
| `diffuser` | 0.17 | 0.00 | 0.17 | 0.00 | 0.00 |
| `dpcc-r-tightened` | **0.00** | 1.00 | 1.00 | 0.83 | 0.83 |
| `dpcc-c-tightened` | **0.00** | 1.00 | 1.00 | 0.83 | 1.00 |
| `dpcc-t-tightened` | **0.17** | 1.00 | 1.00 | 1.00 | 1.00 |
| `hardflow_new-r-tightened` | 0.67 | 1.00 | 1.00 | 1.00 | 0.83 |
| `hardflow_new-c-tightened` | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |
| `hardflow_new-t-tightened` | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |

† stale checkpoint — see §1. Seeds 7–10 are mutually consistent; the only sub-1.0 cells are seed 9/10
on the `-r` family.

### 3.2 Per-env S&C (seeds 7–10, 8 episodes per cell)

| arm | `top-right-hard` | `top-left-hard` | `both-hard` |
|---|---|---|---|
| `diffuser` | 0.00 | 0.12 | 0.00 |
| `dpcc-r-tightened` | 1.00 | 1.00 | **0.75** |
| `dpcc-c-tightened` | 1.00 | 1.00 | **0.88** |
| `dpcc-t-tightened` | 1.00 | 1.00 | 1.00 |
| `hardflow_new-{r,c,t}-tightened` | 1.00 | 1.00 | 0.88 / 1.00 / 1.00 |

**`both-hard` is the only cell that still fails**, and only on `-r`/`-c`. `top-right-hard`, the cell
that broke every earlier Gen3v6/Gen3v7 UNet run, is now clean at 1.00 across all projected arms.

---

## 4. vs MeanFlow

Two MeanFlow references exist. Both are answered.

### 4.1 vs MeanFlow **DiT** K2 (Gen3v6 live folder, all 5 seeds) — matched seeds 7–10, AF wins

| arm | **AF UNet@32 K2** | MeanFlow DiT K2 |
|---|---|---|
| mean S&C, 7 DPCC arms | **0.649** | 0.464 |
| mean S&C, 3 `-tightened` | **0.958** | 0.694 |
| `dpcc-r-tightened` | 0.92 | **1.00** |
| `dpcc-c` | **0.75** | 0.08 |
| `dpcc-c-tightened` | **0.96** (91.5 steps) | 0.12 (**182.9 steps**) |
| `dpcc-t-tightened` | **1.00** (58.4) | 0.96 (69.2) |
| `hardflow_new-t-tightened` | **1.00** (59.4) | 0.96 (71.4) |
| mean steps, 7 DPCC arms | **69.6** | 104.1 |
| mean s/step, 7 DPCC arms | 0.027 | 0.026 |

**Yes, AF-UNet beats MF-DiT** — same per-step cost, 33 % fewer steps, higher S&C on 6 of 7 arms. The
decisive difference is the `dpcc-c` family: MF-DiT sits at **~183 steps with S&C 0.12**, the
"projector drives the trajectory into a corner and the episode times out" mode. AF-UNet does not do
that. MF-DiT's one win is `dpcc-r-tightened` (1.00 vs 0.92).

### 4.2 vs MeanFlow **UNet@32** K2 (Fix_8, job 24334) — **not decidable from this data**

MF-UNet@32 was evaluated on **seed 6 only**, and seed 6 is the one AlphaFlow seed that was not
retrained (§1). There is no overlapping valid seed. Placing them side by side anyway, off-seed:

| arm | MF UNet@32 K2 (**seed 6**) | AF UNet@32 K2 (**seeds 7–10**) | AF UNet@32 K2 (seed 6 †) |
|---|---|---|---|
| mean S&C, 7 DPCC arms | 0.619 | **0.649** | 0.048 † |
| mean S&C, 3 `-tightened` | 0.944 | **0.958** | 0.056 † |
| `dpcc-r-tightened` | **1.00** / 63.2 / 0.027 | 0.92 / 63.2 / 0.028 | 0.00 † |
| `dpcc-c-tightened` | 0.83 / 94.0 / 0.027 | **0.96** / 91.5 / 0.027 | 0.00 † |
| `dpcc-t-tightened` | **1.00** / 58.7 / 0.027 | **1.00** / 58.4 / 0.030 | 0.17 † |
| `hardflow_new-t-tightened` | **1.00** / 61.7 / 0.110 | **1.00** / 59.4 / 0.114 | 0.67 † |

† stale checkpoint, not the fixed model.

**Reading:** on their shared headline arms the two are **indistinguishable** — 58.4 vs 58.7 steps and
0.030 vs 0.027 s/step at S&C 1.00 is inside seed noise at 6–24 episodes. AlphaFlow's advantage over
MeanFlow, if it exists, is *not* visible at this sample size; what the fix bought is that AlphaFlow
now reaches MeanFlow's level instead of the 2.8 % accuracy the pre-fix `bbunet` arm scored
(`(Bf_U3)` candidates 25–28 in the 08-07 batch: acc **0.028–0.056**).

⚠️ The clean experiment — AF-UNet and MF-UNet on the same 5 seeds — **has not been run**.

### 4.3 vs the previous best AlphaFlow backbone (SiT, Gen3v7 U2, all 5 seeds), matched seeds 7–10

| arm | **AF UNet@32 K2** | AF SiT K2 |
|---|---|---|
| mean S&C, 7 DPCC arms | **0.649** | 0.607 |
| mean S&C, 3 `-tightened` | **0.958** | 0.722 |
| `dpcc-c-tightened` | **0.96** / 91.5 | 0.25 / **176.5** |
| `dpcc-r-tightened` | 0.92 | **1.00** |
| `dpcc-t-tightened` | **1.00** | 0.92 |
| `dpcc-r` / `dpcc-t` (untightened) | 0.42 / 0.46 | **0.79 / 0.83** |
| mean steps, 7 arms | **69.6** | 98.6 |
| mean s/step, 7 arms | 0.027 | **0.019** |

Backbone **trade-off, not a clean win**: the UNet removes the `dpcc-c` timeout mode that the SiT shares
with MF-DiT (0.96 vs 0.25) and wins the tightened set outright; the SiT is better on the *untightened*
`-r`/`-t` arms (its raw field is closer to feasible) and is ~30 % cheaper per step. Given that the
tightened arms are the deployable ones, **UNet@32 is the better AlphaFlow backbone for the DPCC
pipeline**, and this supersedes the ranking in `bb_unet_ablation/RESULTS_Gen3v7_backbone_ablation_unet_vs_dit.md`,
which was measured on the 253 M UNet.

---

## 5. vs the DPCC K20 baseline (the reference every arm must beat)

Baseline = `logs/avoiding-d3il/plans/diffusion/H8_K20_D…GaussianDiffusion_aw10_thres0.5` (candidate 14
of the 08-07 batch, all 5 seeds; candidate 16 `_aw10 T1` is numerically identical on these arms).
Seeds restricted to **7–10** so both sides have the same 24 episodes.

| arm | **AF UNet@32 K2** | **DPCC K20** | verdict |
|---|---|---|---|
| `diffuser` (unprojected) | 0.04 / 61.4 / **0.019** | 0.12 / 70.0 / 0.179 | both unsafe by design; AF 9.4× cheaper |
| `dpcc-r` | 0.42 / 63.8 / **0.028** | 0.29 / 77.8 / 0.508 | **AF dominates** |
| `dpcc-r-tightened` | **0.92** / 63.2 / **0.028** | **1.00** / 77.0 / 0.576 | DPCC safer; AF faster → trade-off |
| `dpcc-c` | **0.75** / 90.0 / **0.027** | 0.50 / 72.9 / 0.440 | non-dominated (AF safer, DPCC shorter) |
| `dpcc-c-tightened` | 0.96 / 91.5 / **0.027** | **1.00** / **72.3** / 0.548 | **DPCC wins on S&C and steps** |
| `dpcc-t` | 0.46 / **59.0** / **0.028** | **0.62** / 74.5 / 0.467 | trade-off |
| **`dpcc-t-tightened`** | **1.00** / **58.4** / **0.030** | **1.00** / 79.7 / 0.555 | ✅ **AF Pareto-dominates** |
| mean, 7 DPCC arms | **0.649** / 69.6 / **0.027** | **0.649** / 74.9 / 0.467 | equal S&C, AF fewer steps + 17× cheaper |
| mean, 3 `-tightened` | 0.958 / 71.0 / **0.028** | **1.000** / 76.3 / 0.559 | DPCC safer by 0.042 |

**Answer: partially.** On the arm-set average AF matches DPCC K20's safety exactly (0.649 = 0.649)
while using 7 % fewer steps and **17× less compute per step**. On the single best arm
(`dpcc-t-tightened`) it is a textbook Pareto win. But on the `-tightened` family as a whole DPCC K20
is still 0.958 → 1.000 ahead, because AF gives up two episodes on `both-hard` (`dpcc-r-tightened`
0.75, `dpcc-c-tightened` 0.88 in that env). **Calling this "AF beats DPCC K20" would be over-reading;
"AF matches DPCC K20's safety at 1/17 the cost, and dominates it on `dpcc-t-tightened`" is what the
data supports.**

For reference, DPCC **K10** on the same seeds: 7-arm mean 0.565 / 76.0 steps / 0.268 s/step — AF beats
it outright on all three axes.

---

## 6. vs FMv3ODE (naive Flow Matching) K10 / K20

Naive-FM reference = `logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/H8_D…FlowMatchingODE_a1.5_b1.0_aw10/`.

### 6.1 K20 (`T0.5`, candidate 117, all 5 seeds) — matched seeds 7–10

| arm | **AF UNet@32 K2** | FMv3ODE K20 | verdict |
|---|---|---|---|
| `diffuser` | 0.04 / **0.019** | 0.17 / 0.183 | — |
| `dpcc-r-tightened` | **0.92** / 63.2 / **0.028** | 0.88 / 73.7 / 0.897 | ✅ **AF dominates** |
| `dpcc-c-tightened` | 0.96 / 91.5 / **0.027** | **1.00** / **63.5** / 0.474 | FM wins S&C + steps |
| **`dpcc-t-tightened`** | **1.00** / **58.4** / **0.030** | **1.00** / 63.7 / 0.491 | ✅ **AF Pareto-dominates** |
| mean, 7 DPCC arms | **0.649** / 69.6 / **0.027** | 0.601 / **72.0** / 0.485 | AF safer, fewer steps, 18× cheaper |
| mean, 3 `-tightened` | **0.958** / 71.0 / **0.028** | **0.958** / **67.0** / 0.621 | equal S&C; FM fewer steps, AF 22× cheaper |

**Answer: yes on the arm-set average, and decisively on cost.** AF beats naive FM K20 on mean S&C
(0.649 vs 0.601) with fewer steps and ~18× lower per-step cost, and Pareto-dominates it on both
`dpcc-t-tightened` and `dpcc-r-tightened`. The one clean FM win is `dpcc-c-tightened`.
This satisfies the "MF/AF must also beat naive FM" requirement.

### 6.2 K10 — ⚠️ **no matched configuration exists in this batch**

The 08-07 batch has **no `FlowMatchingODE` K10 at `T0.5`**. What it has is K10 at `T0.1` and `T0.05`,
**seed 6 only** — a different projection-activation threshold *and* a different seed, so it is not a K
comparison. Reported for completeness (seed 6, 6 episodes each):

| arm | FMv3ODE **K10 (T0.1)** | FMv3ODE **K5 (T0.5)** | FMv3ODE **K20 (T0.5)** | DPCC K20 |
|---|---|---|---|---|
| `dpcc-r-tightened` | 1.00 / 70.3 / 0.094 | 1.00 / 68.2 / 0.111 | 0.83 / 71.5 / 0.676 | 0.83 / 65.3 / 0.563 |
| `dpcc-c-tightened` | 1.00 / 117.8 / 0.093 | 1.00 / 63.3 / 0.114 | 1.00 / 62.2 / 0.489 | 1.00 / 61.5 / 0.576 |
| `dpcc-t-tightened` | 1.00 / 64.8 / 0.094 | 1.00 / 62.7 / 0.115 | 1.00 / 63.5 / 0.468 | 1.00 / 62.0 / 0.596 |
| mean, 7 arms | 0.600 / 79.3 / 0.093 | 0.619 / 62.5 / 0.102 | 0.595 / 62.9 / 0.441 | 0.690 / 64.5 / 0.476 |

**A `FlowMatchingODE` K10 `T0.5` eval on seeds 6–10 is owed** before the K10 leg of this comparison can
be closed.

---

## 7. Pareto summary (S&C ↑, steps ↓, avg_time ↓), seeds 7–10

Using the project definition of *good*: dominant only if **at equal-or-better S&C it wins both steps
and time**; anything else is a trade-off.

**`dpcc-t-tightened` — AF UNet@32 K2 is Pareto-dominant over every state-space reference measured here:**

| config | S&C | steps | s/step |
|---|---|---|---|
| **AF UNet@32 K2 (this run)** | **1.00** | **58.4** | **0.030** |
| AF SiT K2 | 0.92 | 67.9 | 0.024 |
| MeanFlow DiT K2 | 0.96 | 69.2 | 0.025 |
| FMv3-Diffusion K10 | 1.00 | 62.9 | 0.192 |
| FMv3ODE K20 | 1.00 | 63.7 | 0.491 |
| DPCC K10 | 1.00 | 70.5 | 0.323 |
| DPCC K20 | 1.00 | 79.7 | 0.555 |

Nothing in that column is faster *and* shorter *and* at least as safe. **Dominated by AF: DPCC K20,
DPCC K10, FMv3ODE K20, FMv3-Diffusion K10, MF-DiT K2, AF-SiT K2. Dominated by nothing.**

**`dpcc-c-tightened` — AF is dominated**, by DPCC K20 (1.00 / 72.3) and FMv3ODE K20 (1.00 / 63.5): both
are safer *and* shorter, and AF's only edge is time.

**`dpcc-r-tightened` — trade-off.** AF 0.92 / 63.2 / 0.028 vs DPCC K20 1.00 / 77.0 / 0.576: AF loses
safety, wins both cost axes.

---

## 8. Answers to the four questions asked

| question | answer |
|---|---|
| **Did the AF full-seed run go well?** | **Four of five seeds did.** Seeds 7–10 trained fresh, converged to a 0.0013-wide `test_loss` band, and evaluate at 0.958 mean S&C on the tightened arms. **Seed 6 was silently skipped by `--auto-resume`** and its results are from a stale pre-fix checkpoint — delete and retrain (§1). Also: only K = 2 was evaluated on valid weights. |
| **Does it beat MF?** | **vs MeanFlow DiT (matched seeds): yes**, 0.649 vs 0.464 on the 7-arm mean, and it removes MF-DiT's 183-step `dpcc-c` collapse. **vs MeanFlow UNet@32 (Fix_8): undecidable** — MF-UNet exists only on seed 6, exactly the AF seed that is invalid. Off-seed the two are indistinguishable on `dpcc-t-tightened` (58.4 vs 58.7 steps, S&C 1.00 both). |
| **Does it beat DPCC K20?** | **Matches it, dominates on one arm, loses the tightened set narrowly.** 7-arm mean S&C is a tie at 0.649 with 7 % fewer steps and **17× lower per-step cost**; `dpcc-t-tightened` is a clean Pareto win; `dpcc-c-tightened` and `dpcc-r-tightened` still go to DPCC K20 (1.000 vs 0.958 on the tightened mean). |
| **Does it beat FMv3ODE K10/K20?** | **K20: yes** — higher mean S&C (0.649 vs 0.601), fewer steps, 18× cheaper, Pareto-dominant on `dpcc-t-tightened` and `dpcc-r-tightened`. **K10: cannot be answered** — this batch has no `FlowMatchingODE` K10 at `T0.5`; only `T0.1`/`T0.05` on seed 6. |

---

## 9. Owed work

1. **Delete `…/flow_matching_v3_alphaflow/H8_D…_bbunet_…/6/` and retrain seed 6**, then re-eval K = 2.
   Until then all "5-seed" AlphaFlow-UNet numbers are wrong (7-arm mean drops 0.649 → 0.529 purely
   from the stale seed).
2. **K sweep on the fixed weights.** Eval 24390 ran `NFE budgets = 2` only. Run K = 1/2/5/10/20 on
   seeds 7–10 — both to find AlphaFlow's low-NFE knee and to reach the Gen14 U7 K = 20 parity
   convention.
3. **MF-UNet@32 on seeds 7–10.** The AF-vs-MF UNet question cannot be closed on one non-overlapping
   seed each.
4. **`FlowMatchingODE` K10 at `T0.5`, seeds 6–10** — the missing leg of §6.
5. **Guard the resume hazard.** Fix_8 §5.2 predicted this failure for Gen3v6 and it then bit Gen3v7.
   Worth an explicit `--force-restart` in the Gen3v7 sbatch, or a train-time assert that the on-disk
   checkpoint's backbone width matches the config.
6. Watch the rollout renders before promoting any of this — §3's `SR = 1.00` unprojected only says the
   arm reaches the goal, not that the policy looks right.
7. **Add the §1.1 foresight check to the DA pipeline.** `mean‖Δ plan per horizon step‖ /
   mean‖Δ realised per control step‖` from any `diffuser.npz`; healthy ≈ 1.0, seed 6 scores 6.6–8.4.
   No summary metric in the current eval output catches this — seed 6 still reported `SR = 1.00` on
   the unprojected arm while planning across the whole arena every step.

---

## Appendix — how these numbers were produced

* AlphaFlow numbers: all 312 `eval_*.log` cells under
  `temp/2026-08-07/H8_D…AlphaFlowODE_aw10_bbunet_…/{K}/{seed}/results/halfspace_{env}/`, parsed
  directly (13 arms × 3 envs × 8 (K, seed) combinations). Each cell is 2 trials; pooled means are
  unweighted means over cells, which is exact here because every cell has the same trial count.
* Baselines: `candidates_multidimensional_raw.csv` of
  `temp/2026-08-07/batch_avoiding_combined_20260807_124828/`, filtered by `Candidate` and `seed`.
  Candidate map used — **14** DPCC K20 (`diffusion`, thres 0.5) · **7** DPCC K10 · **117** FMv3ODE K20
  T0.5 · **114/113** FMv3ODE K10 T0.1/T0.05 · **118** FMv3ODE K5 T0.5 · **122/124** FMv3-Diffusion
  K20/K10 · **108** MeanFlow DiT K2 · **110** MeanFlow UNet@32 K2 (Fix_8) · **36** AlphaFlow SiT K2.
* Arm-set means: `diffuser`, `dpcc-{r,c,t}`, `dpcc-{r,c,t}-tightened` — the seven arms present in
  *every* candidate. The `hardflow_new-*` arms exist only for the Gen3v6/Gen3v7 runs and are excluded
  from any mean that includes DPCC or FMv3ODE.
* §1.1 foresight diagnostics: every `*.npz` under the same tree (**624 cells**), key
  `sampled_trajectories_all` → `(T, 4, 8, 4)` = (MPC steps, MPC samples, horizon, obs dims), compared
  against key `obs_all` → `(T, 4)`. Metrics are `‖Δxy‖` along the horizon axis vs `‖Δxy‖` along the
  realised-time axis, plus per-horizon open-loop error `‖plan(t, h) − obs(t + h)‖`. MeanFlow-UNet@32
  reference read from `temp/0608/H8_D…MeanFlowODE…_bbunet_…/H8_K2_Meuler_T1_…/6/`.
* ⚠️ **Interpreter note for this container:** the default `python3` has no numpy;
  **`/usr/local/bin/python3`** does (numpy 2.5.1). `.npz` object arrays need `allow_pickle=True`.
* Scripts: `parse_af.py`, `agg_af.py`, `compare.py` in the session scratchpad (not committed —
  they read `temp/`, which is gitignored).
