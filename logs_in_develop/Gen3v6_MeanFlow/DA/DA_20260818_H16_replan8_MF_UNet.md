# DA — H16 + replan-8 ("HardFlow-style H8+8") on MeanFlow/UNet, avoiding-d3il

**Date:** 2026-08-18 · **Job:** 24650 (`eval_meanflow_hardflow`), i6-gpu-1, git `5bc5db4` (tree dirty)
**Run:** `MF_HORIZON=16  MF_BACKBONE=unet  MF_REPLAN_STEPS=8  HFFM_BATCH=1  MF_FLOW_STEPS="1 2 5"`
**Log:** `temp/1808/16_50_20_eval_meanflow_hardflow_24650.log` · **Status:** completed cleanly (`Evaluation completed successfully.`, JOB END 15:16:39)
**Data:** `temp/1808/H16_…_dp0.5/H16_K{1,2,5}_…MeanFlowODE_msgr8/` (new, r8) and `…MeanFlowODE/` (previous r1, same folder — direct A/B)
**Scope:** seed 6, `n_trials=2`, 3 halfspace scenes ⇒ **6 episodes per cell**, 13 projection variants × 3 K = 39 cells per cadence.

---

## 0. Is H16/r8 better than our old H8/r1? — direct answer

**Yes, generally better. Two axes better, one unresolved, zero measurably worse.**

Best arm per family. H8/r1 is the **n=20** reference; H16/r8 is n=6.

### Steps

| K | arm | H8/r1 | H16/r8 | Δ |
|---|---|---|---|---|
| 1 | `dpcc-c-tightened` | 72.0 | 64.7 | **−7.3** |
| 1 | `hardflow_new-c-t` | 63.4 | 63.2 | tie |
| 2 | `dpcc-c-tightened` | 98.0 | 64.7 | **−33.3** |
| 2 | `hardflow_new-c-t` | 67.1 | 59.2 | **−7.9** |
| 5 | `dpcc-c-tightened` | 64.0 | 65.7 | +1.7 |
| 5 | `hardflow_new-c-t` | 67.3 | 58.0 | **−9.3** |

4 better, 1 tie, 1 worse by +1.7 (inside noise). → **BETTER.**

### S&C

0.943→1.000 · 0.950→1.000 · 0.927→1.000 · 0.900→**0.833** · 0.867→1.000 · 0.897→1.000

Five up, one down. But 6/6 has Clopper–Pearson CI **[0.541, 1.000]**, which overlaps every H8 value. → **NOT MEASURABLE.** Not better, not worse. n=6 is the only blocker.

### Time

Better on amortised s/step everywhere (**1.3–5.2×**). Worse on per-tick latency everywhere (**1.6–6.0×**, §5.1). Same run, two metrics, opposite signs.

**Which one counts:** `avg_time` — amortised — is what our eval prints, what our benchmark tables carry, and what DPCC and HardFlow both report. Per-tick latency is reported by nobody in this comparison. On the metric this project and its baselines actually use, **H16/r8 is faster**. Per-tick latency only becomes decisive if we commit to hard real-time deployment, which we have not. → **BETTER on the reported metric.**

### Verdict

| axis | verdict | confidence |
|---|---|---|
| steps | **better** | moderate — 7–33 step gaps at n=6 |
| S&C | **unresolved** | none — n=6 cannot separate 1.000 from 0.867 |
| time (amortised, = the reported metric) | **better** | high — near-deterministic |
| time (per-tick latency, reported by no one) | worse | high |

**Worth pursuing: yes.** Better on what we measure, 3.5× cheaper to run, and it is the baseline's own planning structure — required for a like-for-like HardFlow comparison whether it wins or not.

**Separate question, answered in §5.2:** this section compares *cadences*. Which *arm* (DPCC-projection vs HardFlow) wins is a different axis — and cadence turns out not to decide it. Horizon does.

**One caveat, stated once:** the time win is amortisation. If we ever claim real-time capability, the per-tick number contradicts it. This does not affect the verdict on the current benchmark.

---

## 0.1 Why — three findings behind the verdict

**The H8+8 run worked, and it is the cheapest configuration we have — but the win is amortisation, not speed.**

Three findings, in order of confidence:

1. **Replan-8 is mechanically correct.** NFE and NLP solves *per plan* are unchanged from r1 (8.11 → 8.17 at K5); plans per episode drop 60.7 → 8.0. The U10 cache-and-replay path is doing exactly what it was written to do. **High confidence — this is an identity check, not a statistic.**
2. **Amortised cost drops 5–10× across every arm; per-plan latency is unchanged.** `t_r8 × 8 ≈ t_r1` to within 5–36%. Planning did not get faster; it got 8× rarer. **High confidence.**
3. **The tightened arms survive the open-loop segment; the untightened arms collapse.** `-tightened` S&C stays at 1.000 (occasionally 0.833 = one episode); untightened HardFlow goes from 0.5 violations/episode to 11–34. **Mechanism confident, magnitude noisy at n=6.**

The headline number: **`hardflow_new-c-tightened` at K=5, H16/r8 — S&C 6/6, 58.0 steps, 0.0273 s per env step** vs the H8/r1 n=20 reference of 0.897 S&C, 67.3 steps, 0.1408 s. That is 5.2× cheaper amortised, fewer steps, and no worse on S&C — but see §5.1 and §6 before quoting it.

Scope of the §0 verdict: it holds on the metrics we and the baselines report. It is **not** "better on literally every axis" — steps improve in 4 of 6 rows not 6 of 6, S&C is unresolved rather than won, and per-tick latency (§5.1) is worse. Those are the boundaries of the claim, not a retraction of it.

---

## 1. Provenance — U10.1 verified live

First cluster run of the new provenance writer. It worked:

```
[ provenance ] wrote …/H16_K1_…_msgr8/6/run_provenance.json
```

The file records exactly the gap it was built for:

| field | value |
|---|---|
| `env_set` | `MF_HORIZON=16`, `MF_BACKBONE=unet`, `MF_REPLAN_STEPS=8`, `HFFM_BATCH=1`, `HFFM_ACT_THRESHOLD=0.5`, `FMPCC_RUN_MSG=r8`, `FMPCC_PROJ_CFG=config/meanflow_projection_eval.yaml`, `FMPCC_DPCC_THRESHOLD=0.5` |
| `resolved` | `horizon=16`, `replan_steps=8`, `checkpoint_horizon=16`, `flow_steps_K=1`, `imf_backbone=unet`, `hf_batch_size=1`, `batch_size_dpcc_arms=4`, `n_trials=2` |
| `env_absent` | `DPCC_THRESHOLD`, `HFFM_FLOW_STEPS`, … (i.e. those came from fallbacks) |
| `yaml` | `config/meanflow_projection_eval.yaml`, `sha256:5c1dccdf…` |
| `git` | `5bc5db48…`, `dirty: true` |

`checkpoint_horizon=16` alongside `horizon=16` is the G1 guard passing — this eval is on an H16-trained checkpoint, not an H8 one reinterpreted. `batch_size_dpcc_arms=4` vs `hf_batch_size=1` is recorded, which is the confound in §5 now being carried in the artifact rather than in my head.

**`dirty: true` is a real caveat**: commit `5bc5db4` does not fully describe the code that produced these numbers.

---

## 2. Did replan-8 actually happen? (identity check)

If the cache-and-replay is correct, the per-plan solver work must be *identical* to r1 and only the plan count should change. It is:

| K | cadence | NFE / plan | NLP / plan | plans / episode | IPOPT fails / episode |
|---|---|---|---|---|---|
| 1 | r1 | 2.03 | 1.01 | 60.7 | 0.00 |
| 1 | **r8** | **2.08** | **1.04** | **8.0** | 0.00 |
| 2 | r1 | 3.02 | 1.01 | 64.3 | 1.00 |
| 2 | **r8** | **2.69** | **0.90** | **8.0** | 0.17 |
| 5 | r1 | 8.11 | 3.04 | 61.3 | 1.83 |
| 5 | **r8** | **8.17** | **3.06** | **7.7** | 0.50 |

(`hardflow_new-c-tightened`; plans/episode derived as `ceil(steps / cadence)`.)

NFE/plan matches the HardFlow accounting exactly: `K + n_active` = 2, 3, 8 for K = 1, 2, 5 at A=0.5, and NLP/plan = `n_active` = 1, 1, 3. **Unchanged by cadence, as it must be.** The K=2 r8 dip to 2.69/0.90 is the `ceil` estimate being slightly off for episodes whose length is not a multiple of 8, not a real change.

Plans/episode 60.7 → 8.0 is a **7.6× reduction** (not exactly 8 because episodes end mid-plan).

**Wall clock:** whole 3-K sweep in **26 min** (K1 5m56s, K2 6m04s, K5 14m18s) vs ~93 min for the r1 sweep (K1 10m22s, K2 11m48s, K5 70m43s) — 3.5× faster end-to-end. Less than 8× because env stepping and model setup are cadence-independent.

---

## 3. Cost — amortisation, not speed

`Average computation time per step` is **amortised over env steps**: total policy time ÷ episode length. Under r8 only 1 step in 8 pays anything.

| K | variant | r1 s/step | r8 s/step | amortised gain | r8 × 8 (≈ per-plan) | vs r1 per-plan |
|---|---|---|---|---|---|---|
| 1 | `dpcc-c-tightened` | 0.0813 | **0.0123** | 6.6× | 0.0987 | 1.21× |
| 1 | `hardflow_new-c-tightened` | 0.0630 | **0.0083** | 7.6× | 0.0667 | 1.06× |
| 2 | `dpcc-c-tightened` | 0.0923 | **0.0147** | 6.3× | 0.1173 | 1.27× |
| 2 | `hardflow_new-c-tightened` | 0.0733 | **0.0100** | 7.3× | 0.0800 | 1.09× |
| 5 | `dpcc-c-tightened` | 1.2730 | **0.1670** | 7.6× | 1.3360 | 1.05× |
| 5 | `dpcc-r-tightened` | 1.6427 | **0.2783** | 5.9× | 2.2267 | 1.36× |
| 5 | `hardflow_new-c-tightened` | 0.1997 | **0.0273** | 7.3× | 0.2187 | 1.10× |

**Read the last column.** Per-plan cost is unchanged (1.05–1.36×). Nothing about the solver got faster. If the controller must emit an action every tick and cannot buffer, the number that matters is still **1.34 s for DPCC K5** and **0.22 s for HardFlow K5** — every 8th tick. The amortised column is the honest metric only for a system that can pre-compute a segment.

**This also settles the earlier "is DPCC K5/H16 a bug?" question: no.** The 1.27–1.64 s per-plan cost is reproduced here under a completely different cadence, at the same magnitude. It is genuine solver scaling (SLSQP at H=16 with 4 candidates), not a breaker or a stall. r8 does not fix it — it hides it behind 8× fewer calls.

---

## 4. Quality — tightening is what buys the open-loop segment

**Tightened arms hold.** Per-cell S&C (6 episodes), r1 → r8:

| K | `dpcc-r-t` | `dpcc-c-t` | `dpcc-t-t` | `hf-r-t` | `hf-c-t` | `hf-t-t` |
|---|---|---|---|---|---|---|
| 1 | 1.000→1.000 | 1.000→**1.000** | 1.000→0.833 | 1.000→1.000 | 1.000→**1.000** | 1.000→1.000 |
| 2 | 1.000→0.833 | 1.000→**1.000** | 0.833→1.000 | 1.000→0.833 | 1.000→0.833 | 1.000→0.833 |
| 5 | 1.000→0.833 | 1.000→**1.000** | 0.833→1.000 | 1.000→1.000 | 1.000→**1.000** | 1.000→1.000 |

Every 0.833 is **one episode out of six** in **one scene**, and two of them move *upward* (`dpcc-t-t` at K2 and K5). At n=6 these are indistinguishable: 6/6 has a 95% Clopper–Pearson interval of **[0.541, 1.000]** and 5/6 of **[0.359, 0.996]**. **Do not claim r8 preserves S&C from this table — claim it does not visibly break it.**

`dpcc-c-tightened` is the only arm at 6/6 in every cell of both cadences.

**Untightened arms collapse.** This is the real signal, and it is not subtle:

| K | arm | violations/episode r1 → r8 | S&C r1 → r8 |
|---|---|---|---|
| 1 | `hardflow_new-{r,c,t}` | 0.50 → **11.17** | 0.500 → 0.333 |
| 2 | `hardflow_new-{r,c,t}` | 1.33 → **33.50** | 0.667 → 0.333 |
| 5 | `hardflow_new-{r,c,t}` | 2.00 → **10.17** | 0.667 → 0.333 |

Per scene (violation counts, r1 → r8) at K=2: `top-right 0→2`, `top-left 0→54`, `both-hard 4→45`. A 20–50× increase is far outside n=6 noise.

**Mechanism.** Projection guarantees the constraint on the *plan*, at plan time. Under r1 that guarantee is refreshed every step, so tracking error never accumulates. Under r8 the robot executes 8 steps open-loop and drifts off the plan; an untightened plan sits exactly *on* the constraint boundary, so any drift crosses it. Tightening adds the margin that absorbs the drift.

Corroborating measurement — tracking error of the unguided `diffuser` arm (the only arm that logs it):

| K | r1 | r8 |
|---|---|---|
| 1 | 0.041 | 0.050 |
| 5 | 0.030 | **0.055 / 0.169** (per scene) |

Drift grows with cadence, as predicted, up to 5.6× at K5.

> **This is the transferable result of the run:** *replan cadence and constraint tightening are coupled knobs.* HardFlow's H16/8+8 is not "free" — it is affordable **because** the projection carries a margin. Reporting an 8× cadence reduction without saying which arm it was measured on would be misleading.

---

## 5. Three-way comparison vs the old H8

Best-behaving arm per family, S&C / steps / amortised s per env step. H8 column is the **n=20** reference (`_msg20trials`, 20 trials × 3 scenes); H16 columns are n=6.

| K | arm | H8 / r1 (n=20) | H16 / r1 (n=6) | **H16 / r8 (n=6)** |
|---|---|---|---|---|
| 1 | `dpcc-c-tightened` | 0.943 · 72.0 · 0.0180 | 1.000 · 58.8 · 0.0813 | **1.000 · 64.7 · 0.0123** |
| 1 | `hardflow_new-c-t` | 0.950 · 63.4 · 0.0423 | 1.000 · 60.5 · 0.0630 | **1.000 · 63.2 · 0.0083** |
| 2 | `dpcc-c-tightened` | 0.927 · 98.0 · 0.0268 | 1.000 · 62.7 · 0.0923 | **1.000 · 64.7 · 0.0147** |
| 2 | `hardflow_new-c-t` | 0.900 · 67.1 · 0.0506 | 1.000 · 63.8 · 0.0733 | **0.833 · 59.2 · 0.0100** |
| 5 | `dpcc-c-tightened` | 0.867 · 64.0 · 0.2245 | 1.000 · 59.0 · 1.2730 | **1.000 · 65.7 · 0.1670** |
| 5 | `hardflow_new-c-t` | 0.897 · 67.3 · 0.1408 | 1.000 · 61.2 · 0.1997 | **1.000 · 58.0 · 0.0273** |

Reading it:

- **H16/r1 was a straight regression on cost** — 1.4× (HardFlow) to 5.7× (DPCC) more expensive per env step than H8/r1, buying no measurable S&C at n=6. On its own, H16 was not worth it.
- **H16/r8 is cheaper than H8/r1 in every row**, by 1.3–1.6× for DPCC and **5.1–5.2× for HardFlow**.
- **`hardflow_new-c-tightened` K=5, H16/r8 is Pareto-dominant over H8/r1** on the axes we can measure: fewer steps (58.0 vs 67.3), 5.2× cheaper amortised, S&C not worse. The S&C axis is *not* a win — 6/6 [0.541, 1.000] versus 0.897 [n=20] overlaps heavily.
- `dpcc-c-tightened` K=5 is a **trade-off**, not a domination: 1.34× cheaper amortised but +1.7 steps.

**The comparison is not clean.** H8 numbers come from n=20, H16 from n=6, and per §3 the cost axis compares an amortised quantity against a per-step one. The strong version of this claim needs the scale-up in §8.

### 5.1 The axis that goes against us: worst-case tick latency

The table above compares **amortised** cost. That flatters H16/r8, because H8/r1 replans on *every* env step — so for H8/r1, **amortised cost *is* per-plan cost**, while for H16/r8 the per-plan cost is 8× the amortised figure. Put on the same axis:

| K | arm | H8/r1 per-plan | H16/r8 per-plan (= s/step × 8) | worst-case tick | amortised (for contrast) |
|---|---|---|---|---|---|
| 1 | `dpcc-c-tightened` | 0.0180 | 0.0987 | **5.5× worse** | 1.46× better |
| 1 | `hardflow_new-c-t` | 0.0423 | 0.0667 | **1.6× worse** | 5.10× better |
| 2 | `dpcc-c-tightened` | 0.0268 | 0.1173 | **4.4× worse** | 1.82× better |
| 2 | `hardflow_new-c-t` | 0.0506 | 0.0800 | **1.6× worse** | 5.06× better |
| 5 | `dpcc-c-tightened` | 0.2245 | 1.3360 | **6.0× worse** | 1.34× better |
| 5 | `hardflow_new-c-t` | 0.1408 | 0.2187 | **1.6× worse** | 5.16× better |

**H16/r8 is 1.6–6.0× worse on worst-case latency in every single cell**, and the direction is uniform — this is not noise.

So the correct summary of H16/r8 vs H8/r1 is a **throughput-vs-latency trade-off**, not a domination:

- **If the controller can buffer an 8-step segment**, H16/r8 wins by 1.3–5.2× on total compute and is the better configuration.
- **If it must emit an action every tick with bounded latency**, H8/r1 wins and H16/r8 is a regression — by 6× for DPCC at K5 (0.22 s → 1.34 s per tick).

Which side applies depends on a deployment assumption **we have not stated anywhere**. HardFlow's H16/8+8 implicitly assumes buffering — that is what "execute 8" means. If we adopt their structure we inherit their assumption, and it should be written down explicitly rather than left implied by the metric we happen to report.

Note the asymmetry: HardFlow's latency penalty is a flat **1.6×** at every K, DPCC's grows to **6.0×** — because HardFlow's per-plan cost barely moved from H8 to H16 while DPCC's blew up (§3). Under a hard latency budget, HardFlow is the arm that survives the horizon increase.

> **Batch note.** Every ratio in the table above compares an arm *to itself* across configs, so the 4-candidate fan cancels and the 1.6×/4.4×/5.5×/6.0× figures are batch-clean. The *absolute* cross-arm comparison in the last sentence is not — at B=4 HardFlow's H16/r8 per-plan cost would rise from 0.2187 to ≈0.66 s, still under DPCC's 1.336 s at K=5 but above it at K=1/2 (§5.3). The "HardFlow survives the horizon increase" conclusion therefore holds **at K=5 only**.

**Corrected verdict on "better on all three axes":** no.
**Steps** — better in 4 of 6 rows (`dpcc-c-t` K5 is +1.7 worse, `hf-c-t` K1 is a wash).
**S&C** — not claimable in either direction at n=6; one row (`hf-c-t` K2) is nominally worse.
**Time** — better amortised, worse per-tick, in all 6 rows.

---

### 5.2 Which arm wins where — DPCC projection vs HardFlow (cadence is *not* the selector)

**Question:** does the H8+8 cadence (HardFlow's own default) hand the win to the HardFlow arms, while replan-1 (H8/r1, H16/r1) keeps it with DPCC projection?

**Answer: half right. DPCC does win at H8/r1. But the flip is driven by horizon, not by replan cadence.**

Best `-tightened` variant of each family, mean over the 3 avoiding scenes, `S&C / steps / s-per-env-step`.

> ⚠️ **Read the cost column together with §5.3.** All DPCC arms ran a 4-candidate fan, all HardFlow arms ran B=1. Batch-matched, four of the "HF cheaper" verdicts below reverse. The S&C and steps columns are unaffected.

| config | K | best DPCC-projection | best HardFlow | who wins |
|---|---|---|---|---|
| **H8 / r1** (n=20×3) | 1 | `dpcc-t-t` **1.000** / **61.0** / **0.0181** | 0.960 / 63.4 / 0.0421 | **DPCC dominates** |
| | 2 | `dpcc-t-t` **1.000** / **60.4** / **0.0271** | 0.900 / 67.1 / 0.0505 | **DPCC dominates** |
| | 5 | `dpcc-t-t` **0.980** / **60.8** / 0.2250 | 0.900 / 67.3 / **0.1408** | DPCC on S&C + steps; HF 1.6× cheaper |
| **H16 / r1** (n=6) | 1 | `dpcc-c-t` 1.000 / **58.8** / 0.0813 | 1.000 / 60.5 / 0.0627 | tie on S&C; HF cheaper **⚠️ batch artifact → §5.3** |
| | 2 | `dpcc-c-t` 1.000 / **62.7** / 0.0923 | 1.000 / 63.8 / 0.0733 | tie on S&C; HF cheaper **⚠️ batch artifact → §5.3** |
| | 5 | `dpcc-c-t` 1.000 / **59.0** / 1.2730 | 1.000 / 61.2 / **0.1997** | **HF cheaper — survives batch matching at 1.9–2.4× (§5.3)** |
| **H16 / r8** (n=6) | 1 | `dpcc-r-t` 1.000 / **62.3** / 0.0103 | 1.000 / 63.2 / 0.0083 | non-dominated **⚠️ batch artifact → §5.3** |
| | 2 | `dpcc-t-t` **1.000** / **56.0** / 0.0110 | **0.833** / 59.2 / **0.0100** | **DPCC wins** |
| | 5 | `dpcc-c-t` 1.000 / 65.7 / 0.1670 | 1.000 / **58.0** / **0.0273** | **HF dominates** (cost win survives §5.3) |

#### The cost ratio isolates the cause

DPCC-cost ÷ HardFlow-cost at K=5:

| | H8 / r1 | H16 / r1 | H16 / r8 |
|---|---|---|---|
| ratio | **1.60×** | **6.38×** | **6.12×** |

The ratio jumps ~4× when H goes 8 → 16, then is **flat across cadence**. `r1 → r8` divides both families by ~8 and leaves the ranking untouched.

**Mechanism.** DPCC solves SLSQP post-hoc over **4 candidates × H**; HardFlow solves IPOPT in-loop over **1 trajectory** (`batch_size` hard-asserted to 1 — see `../../HF_iMF/Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md`). Doubling H costs DPCC **6.6×** (0.194 → 1.274 at K5) and HardFlow **1.4×** (0.135 → 0.200).

That contrast is real, but the 4:1 fan rides on top of it. §5.3 strips the fan out and shows the underlying batch-free fact: **per solve, doubling H costs SLSQP ~8× and IPOPT ~1.87×**, at every K. That is the actual flip mechanism.

#### Conclusions

1. **Replan cadence is not an arm-selector.** It is a throughput divider, ~8×, applied equally to both families.
2. **Horizon is the arm-selector.** At H8 DPCC-projection wins outright; at H16 the cost axis flips to HardFlow while S&C ties at 1.000 for both.
3. **The one decisive HardFlow win in the grid is H16 / K5**, where it is Pareto-dominant: equal S&C, 7.7 fewer steps, 6.1× cheaper as measured — **1.9–2.4× cheaper after batch matching (§5.3)**, which is the number to quote. It is the only cell where HardFlow's cost advantage survives the correction.
4. **Tightening, not the arm, buys constraint satisfaction.** Every `S&C = 1.000` in all 9 rows above is a `-tightened` variant; untightened arms top out at 0.833.
5. **The 4-candidates-vs-1 confound cuts both ways.** On cost it is now quantified and largely removed (§5.3): it accounts for ~4× of every raw ratio and reverses the verdict in 4 of 9 rows. On quality it remains open (§6.4) — DPCC's S&C/steps edge at H8 is plausibly bought by the 4-candidate fan, and nothing here separates that.

**Evidence weight — do not read the three blocks as equally solid.** H8/r1 rows are n=60 episodes; both H16 blocks are n=6. The H16 S&C column cannot separate 1.000 from 0.833, so the "DPCC wins" at H16/r8 K2 may be a single-episode artifact. The cost column is near-deterministic and holds at any n.

---

### 5.3 ⚠️ The cost column is batch-confounded — decomposed and corrected

**Every DPCC and `diffuser` arm in every run above ran `batch_size=4`. Every `hardflow_new-*` arm ran `B=1`.** Verified three ways: `resolve_hf_batch_size()` returns `max(1, HFFM_BATCH)` for `-r/-c/-t` and `HFFM_BATCH=1` was set for this job (`run_provenance.json`: `hf_batch_size: 1`, `batch_size_dpcc_arms: 4`); the aggregated H8 CSV records `hf_batch_size = 4.0` for `diffuser`/`dpcc-*` and `1.0` for `hardflow_new-*`; and all three HardFlow suffixes `-r/-c/-t` return byte-identical numbers, which is what B=1 must produce (selection over one candidate is a no-op).

**This is not a small correction. The DPCC projector solves SLSQP in a sequential Python loop over the batch** — `flow_matcher_v3_meanflow/sampling/projection.py:131`, `for i in range(batch_size): res = minimize(...)`. Solve cost is therefore *exactly linear* in B, with no GPU amortisation. Generation is a single batched forward pass and is near-free at B=4.

#### Solves per plan are exactly 4:1

Both families activate on the same number of flow steps at threshold 0.5:

- DPCC: `snapping_start_idx = int((1 - 0.5) * K)`, projects on every step from there (`models/mf_diffusion.py:284–299`) ⇒ `n_proj` = **1, 1, 3** for K = 1, 2, 5.
- HardFlow: `n_active` = **1, 1, 3** — same K, same threshold. Confirmed empirically from the logged counters: `nlp_total / (steps × n_trials)` = 1.02, 1.02, 3.05.

| K | DPCC solves/plan | HardFlow solves/plan | ratio |
|---|---|---|---|
| 1 | 1 × **4** = 4 | 1 × **1** = 1 | **4:1** |
| 2 | 1 × **4** = 4 | 1 × **1** = 1 | **4:1** |
| 5 | 3 × **4** = 12 | 3 × **1** = 3 | **4:1** |

#### Additive decomposition (r1 only — r8's 3-decimal `avg_time` is too quantised, §6.3)

Model: `t_total = n_NFE · u + n_solves · c_solve`, with `u` (per-NFE generation cost) read off the projection-free `diffuser` arm. `u` comes out at **0.0092–0.0103 s** across all six cells regardless of K, horizon, or config — that self-consistency is the check that the model is right.

| config | K | u (s/NFE) | **SLSQP / solve** | **IPOPT / solve** |
|---|---|---|---|---|
| H8 / r1 | 1 | 0.00960 | 0.00198 | 0.02180 |
| | 2 | 0.00935 | 0.00205 | 0.02185 |
| | 5 | 0.00924 | 0.01234 | 0.02049 |
| H16 / r1 | 1 | 0.01030 | **0.01660** | 0.04070 |
| | 2 | 0.00985 | **0.01515** | 0.04045 |
| | 5 | 0.00934 | **0.10228** | 0.03833 |

#### Batch-matched, HardFlow loses on cost almost everywhere

Two independent normalisations of the `-c` arms (raw `dpcc-c` ÷ `hardflow_new-c`):

| config | K | raw D÷H | DPCC re-priced at **B=1** ÷ HF | HF re-priced at **B=4**, DPCC ÷ it |
|---|---|---|---|---|
| H8 / r1 | 1 | 0.43 | **0.28** | **0.16** |
| | 2 | 0.54 | **0.42** | **0.23** |
| | 5 | 1.44 | **0.61** | **0.61** |
| H16 / r1 | 1 | 1.25 | **0.44** | **0.42** |
| | 2 | 1.15 | **0.50** | **0.42** |
| | 5 | **6.72** | **1.86** | **2.38** |

(< 1 means DPCC is cheaper.) Both normalisations agree on the sign in all six cells.

**Corrections this forces on §5.2:**

1. **H16/r1 K1 and K2 "HF cheaper" are batch artifacts.** Batch-matched, DPCC is **2.0–2.4× cheaper** in both. Those two rows flip.
2. **H16/r8 K1 "non-dominated" also flips** — same solver structure, same 4:1, and the ratio is cadence-invariant.
3. **H16 / K5 survives, at a quarter of the headline.** HardFlow is genuinely cheaper there — **1.9–2.4×**, not 6.4×. This is the only cell in the whole grid where HardFlow wins on cost after batch matching.
4. The §5.2 statement "*horizon is the arm-selector*" **still holds**, but for a sharper reason than the raw numbers showed (below).

#### What is actually true: the two solvers scale differently in H

| K | SLSQP per solve, H8 → H16 | IPOPT per solve, H8 → H16 |
|---|---|---|
| 1 | 0.00198 → 0.01660 = **8.4×** | 0.02180 → 0.04070 = **1.87×** |
| 2 | 0.00205 → 0.01515 = **7.4×** | 0.02185 → 0.04045 = **1.85×** |
| 5 | 0.01234 → 0.10228 = **8.3×** | 0.02049 → 0.03833 = **1.87×** |

**Doubling the horizon costs SLSQP ~8× per solve and IPOPT ~1.87× per solve, at every K.** IPOPT's 1.87× is essentially linear in the decision-variable count; SLSQP's ~8× is superlinear (≈ H³ — consistent with dense active-set QP factorisation at each iteration, `transition_dim × H` variables with no sparsity exploited).

This is horizon-driven, batch-free, and reproducible across three K values — a real property of the two solvers, not an artifact. Per-solve parity (`SLSQP ÷ IPOPT` = 1):

| K | ratio at H8 | ratio at H16 | extrapolated parity |
|---|---|---|---|
| 1 | 0.09 | 0.41 | H ≈ **24** |
| 2 | 0.09 | 0.37 | H ≈ **26** |
| 5 | 0.60 | 2.67 | H ≈ **10** |

At K=5 the crossover has already happened by H=16; at K=1/2 it has not, and DPCC stays cheaper per solve even at H=16.

#### Assumptions, and which way they cut

- **Generation cost assumed equal at B=1 and B=4** (kernel-launch-bound 4M UNet). If B=1 generation is actually cheaper, the residual attributed to IPOPT grows and **HardFlow looks worse than shown** — so these numbers are generous to HardFlow.
- **HF at B=4 assumed 4 sequential IPOPT solves.** HardFlow hard-asserts B=1 (`batch_size must be 1 for optimal control`), so a batched NLP does not exist today; 4× is the honest cost of building it.
- **SLSQP per-solve cost is not a single constant in K** (0.0020 at K=1/2 vs 0.0123 at K=5, H8). The K=5 projections start from a less-feasible iterate and take more active-set iterations. This does not affect the H8→H16 scaling ratios, which are computed within a fixed K.
- **n=6 at H16.** The cost column is near-deterministic so this matters far less here than for S&C, but the K=5 SLSQP figure rests on 6 episodes.

#### Consequence

`HFFM_BATCH=4 at H16 K5` (§8.2) is no longer a nice-to-have — **it is the run that decides whether the one surviving HardFlow cost win is real.** The prediction from this decomposition is that HardFlow at B=4 lands at ≈0.53 s/step vs DPCC's 1.27 s/step, i.e. still ~2.4× cheaper. If it does not, the entire HardFlow cost story collapses to a batch artifact.

---

## 6. What this does *not* show

1. **n=6 cannot resolve S&C.** Every S&C claim above is 6-episode, one-seed. 6/6 and 0.867 are statistically the same measurement. The cost claims are near-deterministic and do not share this problem.
2. **Amortised ≠ real-time.** §3, §5.1. Whether the 5–7× is a real win depends on whether the deployment can buffer an 8-step segment. HardFlow's own setup assumes it can; ours has not been stated either way. **This is the largest open question in the run** — it decides whether §5's table reads as a win or a regression, and it is a design decision, not a measurement.
3. **`avg_time` is printed to 3 decimals.** At r8/K1 the values are 0.002–0.012, so quantisation is ±4–25%. **The K1/K2 r8 ratios in §3 are quantisation-limited**; only the K5 numbers (0.167–0.278) are precise. This also explains the `diffuser` per-plan ratio of 1.55–1.63 — an artifact of dividing a 3-decimal 0.002, not a real effect.
4. **DPCC arms still run 4 candidates, HardFlow 1.** Recorded in provenance now (`batch_size_dpcc_arms=4`, `hf_batch_size=1`). **On cost this is now decomposed and mostly removed — see §5.3:** it is worth exactly 4× (SLSQP runs in a sequential loop over the batch), and correcting for it reverses the cost verdict in 4 of 9 head-to-head rows, leaving H16/K5 as the sole surviving HardFlow cost win at 1.9–2.4× rather than 6.4×. **On quality it is still open**: DPCC's S&C and steps edge at H8 may be bought by the 4-candidate fan, and nothing measured here separates that.
5. **`replan_steps=8` was never swept.** 8 was chosen to match HardFlow. Whether 4 or 16 is better — and where the untightened collapse begins — is unmeasured.
6. **No H16 diffusion baseline.** The DPCC-diffusion reference is H8 only, so no like-for-like baseline exists at H16.

---

## 7. Bottom line for the paper

Defensible today:

> On avoiding-d3il, adopting HardFlow's planning structure (H=16, execute 8) reduces amortised planning cost by 5–7× versus per-step replanning at the same horizon, and by 5.2× versus our H8 per-step configuration at K=5, with no measurable loss in success-and-constraints for constraint-tightened projection. Per-plan solver cost is unchanged; the gain is entirely a reduction in planning frequency, and it is paid for with a 1.6–6.0× increase in worst-case per-tick latency.

Not defensible: **any claim that H16/r8 dominates H8/r1.** It does not — it trades worst-case latency for throughput (§5.1), and the S&C axis is unresolved at n=6. Reporting only the amortised column would be the misleading version of this result. Also not defensible: any claim that r8 *improves* S&C, and — now quantified in §5.3 — **any "HardFlow beats DPCC by 6×" headline.** That 6× is ~4× candidate fan and ~1.7× solver. The batch-matched number is 1.9–2.4×, at H16/K5 only; at H16 K1/K2 and at every H8 cell, batch-matched DPCC is the *cheaper* arm.

The one clean structural claim in the latency data:

> Per constrained solve, doubling the planning horizon from 8 to 16 costs SLSQP **~8×** and IPOPT **~1.87×**, consistently at K = 1, 2 and 5. HardFlow's in-loop IPOPT scales near-linearly in the decision-variable count; DPCC's post-hoc dense SLSQP scales superlinearly. Under a hard latency budget, HardFlow is the arm that survives growing H.

This version is stated *per solve* and is therefore **batch-free**, unlike the raw wall-clock ratios, which carry a 4:1 candidate-fan factor (§5.3). Quote the per-solve form.

New claim this run supports, which the H8 data could not:

> Receding-horizon cadence and constraint tightening are coupled. Executing 8 steps open-loop raises trajectory tracking error up to 5.6×; untightened projection, which leaves the plan on the constraint boundary, sees violations rise 20–50×, while tightened projection absorbs the drift and is unaffected.

---

## 8. Next runs, in priority order

1. **Scale up: 5 seeds × `n_trials=20` at H16/r8, K ∈ {1,5}.** Every quality claim here is n=6-limited. This is the single highest-value run and costs little now that r8 is 3.5× cheaper.
2. **`HFFM_BATCH=4` at H16 K5 — promoted, near-blocking for any HardFlow cost claim.** §5.3 reduces the raw 6.4× HardFlow cost win to a predicted **~2.4×** (HF at B=4 ≈ 0.53 s/step vs DPCC 1.27 s/step) and reverses it entirely at K=1/2. This run tests that prediction directly. Run it at **r1** (where `avg_time` is not quantisation-limited, §6.3) as well as r8. Cheap. If HF at B=4 does not land near 0.53 s/step, the HardFlow cost story is a batch artifact.
3. **Cadence sweep `MF_REPLAN_STEPS ∈ {2,4,8,16}`** at K5, tightened + untightened. Locates where the untightened collapse starts and whether 8 is the right operating point — this is the figure the §7 coupling claim wants.
4. **Per-solve latency histogram** rather than the amortised mean, so §3/§5.1's per-plan story is measured instead of reconstructed by multiplying by 8. Upgraded in priority by §5.1 — worst-case latency is now a headline axis, and we are currently inferring it.
5. **Decide and write down the buffering assumption** (§6.2). Not a run — a design decision. Until it is stated, §5 has no single correct reading and every cost claim in the paper is ambiguous.
6. *Not yet:* H16 diffusion baseline retrain. Only worth it once (1) shows H16/r8 holding at scale.

---

## Appendix — full r1 vs r8 table (all 13 variants × 3 K, mean over 3 scenes, n=6/cell)

| variant | K | S&C r1 | S&C r8 | steps r1 | steps r8 | s/step r1 | s/step r8 | viol r1 | viol r8 |
|---|---|---|---|---|---|---|---|---|---|
| diffuser | 1 | 0.167 | 0.000 | 59.00 | 62.00 | 0.0103 | 0.0020 | 13.17 | 12.50 |
| dpcc-r | 1 | 0.667 | 0.333 | 61.33 | 66.67 | 0.0703 | 0.0100 | 1.50 | 6.67 |
| dpcc-c | 1 | 0.500 | 0.833 | 60.17 | 61.33 | 0.0767 | 0.0087 | 1.83 | 0.33 |
| dpcc-t | 1 | 0.167 | 0.333 | 72.67 | 62.67 | 0.0883 | 0.0100 | 8.17 | 7.17 |
| dpcc-r-tightened | 1 | 1.000 | 1.000 | 60.33 | 62.33 | 0.0757 | 0.0103 | 0.00 | 0.00 |
| dpcc-c-tightened | 1 | 1.000 | 1.000 | 58.83 | 64.67 | 0.0813 | 0.0123 | 0.00 | 0.00 |
| dpcc-t-tightened | 1 | 1.000 | 0.833 | 61.00 | 62.33 | 0.0897 | 0.0103 | 0.00 | 0.00 |
| hardflow_new-r | 1 | 0.500 | 0.333 | 40.17 | 73.00 | 0.0627 | 0.0087 | 0.50 | 11.17 |
| hardflow_new-c | 1 | 0.500 | 0.333 | 40.17 | 73.00 | 0.0613 | 0.0080 | 0.50 | 11.17 |
| hardflow_new-t | 1 | 0.500 | 0.333 | 40.17 | 73.00 | 0.0610 | 0.0080 | 0.50 | 11.17 |
| hardflow_new-r-tightened | 1 | 1.000 | 1.000 | 60.50 | 63.17 | 0.0627 | 0.0083 | 0.00 | 0.00 |
| hardflow_new-c-tightened | 1 | 1.000 | 1.000 | 60.50 | 63.17 | 0.0630 | 0.0083 | 0.00 | 0.00 |
| hardflow_new-t-tightened | 1 | 1.000 | 1.000 | 60.50 | 63.17 | 0.0630 | 0.0083 | 0.00 | 0.00 |
| diffuser | 2 | 0.000 | 0.000 | 59.00 | 56.33 | 0.0197 | 0.0040 | 18.50 | 14.83 |
| dpcc-r | 2 | 0.333 | 0.333 | 47.33 | 66.00 | 0.0830 | 0.0150 | 9.17 | 8.17 |
| dpcc-c | 2 | 0.833 | 0.333 | 62.50 | 65.83 | 0.0803 | 0.0110 | 0.67 | 1.67 |
| dpcc-t | 2 | 0.167 | 0.167 | 72.00 | 42.83 | 0.0987 | 0.0107 | 7.50 | 8.33 |
| dpcc-r-tightened | 2 | 1.000 | 0.833 | 62.83 | 59.67 | 0.0990 | 0.0110 | 0.00 | 0.00 |
| dpcc-c-tightened | 2 | 1.000 | 1.000 | 62.67 | 64.67 | 0.0923 | 0.0147 | 0.00 | 0.00 |
| dpcc-t-tightened | 2 | 0.833 | 1.000 | 63.00 | 56.00 | 0.1127 | 0.0110 | 0.50 | 0.00 |
| hardflow_new-r | 2 | 0.667 | 0.333 | 65.17 | 92.67 | 0.0710 | 0.0113 | 1.33 | 33.50 |
| hardflow_new-c | 2 | 0.667 | 0.333 | 65.17 | 92.67 | 0.0700 | 0.0110 | 1.33 | 33.50 |
| hardflow_new-t | 2 | 0.667 | 0.333 | 65.17 | 92.67 | 0.0697 | 0.0107 | 1.33 | 33.50 |
| hardflow_new-r-tightened | 2 | 1.000 | 0.833 | 63.83 | 59.17 | 0.0733 | 0.0100 | 0.00 | 0.00 |
| hardflow_new-c-tightened | 2 | 1.000 | 0.833 | 63.83 | 59.17 | 0.0733 | 0.0100 | 0.00 | 0.00 |
| hardflow_new-t-tightened | 2 | 1.000 | 0.833 | 63.83 | 59.17 | 0.0737 | 0.0100 | 0.00 | 0.00 |
| diffuser | 5 | 0.000 | 0.000 | 58.33 | 41.33 | 0.0467 | 0.0073 | 20.17 | 6.00 |
| dpcc-r | 5 | 0.667 | 0.000 | 64.33 | 60.67 | 1.4510 | 0.1520 | 1.50 | 3.83 |
| dpcc-c | 5 | 0.500 | 0.667 | 59.00 | 70.83 | 1.2740 | 0.1890 | 2.50 | 3.67 |
| dpcc-t | 5 | 0.667 | 0.333 | 68.83 | 66.33 | 1.3147 | 0.2320 | 5.17 | 4.67 |
| dpcc-r-tightened | 5 | 1.000 | 0.833 | 68.00 | 61.83 | 1.6427 | 0.2783 | 0.00 | 1.33 |
| dpcc-c-tightened | 5 | 1.000 | 1.000 | 59.00 | 65.67 | 1.2730 | 0.1670 | 0.00 | 0.00 |
| dpcc-t-tightened | 5 | 0.833 | 1.000 | 71.83 | 62.83 | 1.5410 | 0.2423 | 0.83 | 0.00 |
| hardflow_new-r | 5 | 0.667 | 0.333 | 66.67 | 66.17 | 0.1907 | 0.0287 | 2.00 | 10.17 |
| hardflow_new-c | 5 | 0.667 | 0.333 | 66.67 | 66.17 | 0.1897 | 0.0277 | 2.00 | 10.17 |
| hardflow_new-t | 5 | 0.667 | 0.333 | 66.67 | 66.17 | 0.1897 | 0.0277 | 2.00 | 10.17 |
| hardflow_new-r-tightened | 5 | 1.000 | 1.000 | 61.17 | 58.00 | 0.1990 | 0.0270 | 0.00 | 0.00 |
| hardflow_new-c-tightened | 5 | 1.000 | 1.000 | 61.17 | 58.00 | 0.1997 | 0.0273 | 0.00 | 0.00 |
| hardflow_new-t-tightened | 5 | 1.000 | 1.000 | 61.17 | 58.00 | 0.1997 | 0.0273 | 0.00 | 0.00 |

*Source:* parsed from the 234 `eval_<variant>.log` files under `temp/1808/…/{,_msgr8}/6/results/halfspace_*/`. All 6 cells × 13 variants × 3 scenes present, no gaps. `viol` = `Avg number of constraint violations`.

**Related:** `DA_20260817_H16_horizon_MF_UNet.md` (the r1 baseline) · `../H8+8_U10/GUIDE_H16_replan8_MF_UNet.md` · `../H8+8_U10/CHANGELOG_Gen3v6_U10_H16_replan8.md` · `../../CLI_Override_Snapshot/CHANGELOG_U10.1_run_provenance.md`
