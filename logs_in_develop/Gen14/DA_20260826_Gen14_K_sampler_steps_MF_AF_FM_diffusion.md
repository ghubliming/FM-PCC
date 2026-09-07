# DA (Gen14, cross-epoch) — visual-aligning: does `K` (sampler steps) get the box closer? — with the d3il baseline

> **Scope: internal dev log, not a paper DA.** Single seed on the K arms, `diffuser` variant, n=30 paired
> rollouts per cell. Knob-triage note for Gen14 spanning all four engine arms plus the d3il vision baseline.
> Not publishable — see §8 for why.

**Date:** 2026-08-26 · **Task:** aligning-d3il-visual · **Batch:** `temp/2608/batch_va2_20260826_142750`
**Lead metric:** `context_final_xy_dist` — **box→target distance in metres, raw.** Lower is better.

---

## TL;DR

**1. The d3il vision baseline does not move the box.** Median final distance is **1.000× the starting
distance** over 1080 rollouts, and 0.999× over 2804 rollouts across 6 seeds. **70 % / 56 % of its rollouts end
with the box within 5 mm of where it started** — never meaningfully touched. This is the reference policy
this whole line of work is measured against, and on distance it is a no-op.

**2. Our better arms beat it by a wide margin, and one of them proves it on the same split.** On the *test*
split, `cand4` (FM filmv1, K=20) leaves 0.352× of the starting gap and lands **20 % of rollouts within 5 cm**,
against the baseline's 0.1–0.8 %. On train, MeanFlow K=100 reaches 0.277× and 33 %.

**3. But half our arms are also no-ops.** FlowMatching (both K), Diffusion K=20, `cand3` and `cand17` all sit
at 0.95–1.00× — statistically indistinguishable from the baseline, and `cand17` is *worse* (80 % untouched).
The batch is close to bimodal: an arm either engages the box or it does not.

**4. `K` flips arms between those two states rather than tuning them.** Diffusion goes 0.957× → 0.409× from
K=20 → K=100; MeanFlow 0.602× → 0.277× from K=2 → K=100. AlphaFlow goes the *other* way (0.289× at K=2 →
0.689× at K=100). FlowMatching never engages at either setting.

| engine | median fraction of start left | ≤5 cm | verdict |
|---|---|---|---|
| **MeanFlow** | **0.28×** (K100) vs 0.60× (K2) | **33 %** vs 20 % | K=100 closer (trend, p=0.069) |
| **AlphaFlow** | 0.69× (K100) vs **0.29×** (K2) | 3 % vs **17 %** | K=2 closer (solid, p=0.008) |
| **FlowMatching** | 0.95× vs 0.98× | 3 % vs 3 % | **both ≈ baseline** |
| **Diffusion** ⚠️ | **0.41×** (K100) vs 0.96× (K20) | **23 %** vs 3 % | K=100 closer, confounded |
| *d3il baseline* | *1.000× / 0.999×* | *0.1 % / 0.8 %* | *reference* |

---

## 0. ⚠️ Read this before the numbers

**`K` is NOT the MPC candidate fan.** From `config/aligning-d3il-visual.py:933-937`:

```python
('n_diffusion_steps', 'K'),      # diffusion only
('flow_steps_v3',     'K'),      # fm/mf/af only  (mutually exclusive per arm)
('mpc_batch_size',  'mpc'),
```

`K` = **sampler / integrator steps (NFE per plan)**. The MPC fan is `mpc4` — **4 everywhere, never varied**.
Nothing here evaluates the fan. Confirmed by cost: wall-clock ratios track NFE ratios (Diffusion 5.12× vs 5.0
expected; FM 4.85× vs 5.0; MF 32× vs 50, the rest absorbed by 28 ms fixed per-step overhead).

**Which distance this is.** `context_final_xy_dist` = **XY distance box→target at rollout end**, in metres.
Straight from the env context record — no normalisation, no blending.

**🪤 `mean_dist_per_rollout` is a trap and this DA does not use it.** Despite the name it is neither a mean
nor a distance: `eval_visual_aligning_dpcc.py:1379-1382` captures it at rollout end and `aligning.py:316`
defines it as `0.5*(pos_dist_3D + rot_err/π)` — a **blend of position and rotation**. Any distance conclusion
drawn from it is contaminated by the angle term.

**⚠️ True minimum-over-episode distance is NOT available.** The eval logs a per-step curve (`dist_to_target`,
`eval_visual_aligning_dpcc.py:1449`) into each rollout's JSON on the cluster, but DA_VA_v2 never ingests it,
so it is in no local CSV. Consequences: (a) everything below is the **final** distance, so a rollout that
reached the target and drifted off scores as a miss; (b) recovering the real minimum needs a cluster-side pass
over the rollout JSONs — and note that logged curve is the *blended* `mean_distance`, so pure minimum XY
distance is recorded nowhere and needs an eval change. See §9.

---

## 1. Why success rate is useless here (30 seconds, then we move on)

`aligning.py:198-199,344-345` gates success on `pos ≤ 0.018 m` **AND** `rot ≤ 8.64°`. Every cell scores
0–2/30 (baseline: 0/1080 and 8/2804), so it separates nothing.

It stays at the floor because of the **rotation** half: median final rotation error runs 32–62° across all
cells, with a median final/initial ratio of ~1.00× — orientation ends about as misaligned as it started, at
every engine and every `K`. That is a separate problem, untouched by any sampler-step setting.

**That is the last word on angle here.** Everything from this point is distance in metres.

---

## 2. 🔑 The d3il vision baseline — the reference does not move the box

Two baseline cells are present, both `d3il_baseline_ddpm_encdec_vision` (DDPM encoder-decoder, vision),
**test split**, `geo=none` (no constraint projection):

| cell | n | seeds | init median | final median | **fraction of start left** | ≤5 cm | untouched (≤5 mm moved) | success |
|---|---|---|---|---|---|---|---|---|
| baseline (seed 42) | 1080 | 42 | 0.444 m | 0.434 m | **1.000×** | 0.1 % | **70.0 %** | 0/1080 |
| baseline `__Bf_U3` | 2804 | 0–4, 42 | 0.458 m | 0.394 m | **0.999×** | 0.8 % | **55.6 %** | 8/2804 (0.29 %) |

**Read the ratio column.** A median of 1.000× means the typical rollout ends *exactly* as far from the target
as it began. This is not a small-sample artefact — it holds over 1080 rollouts on one seed and 2804 rollouts
across six. The per-seed medians of the `__Bf_U3` cell are 0.424, 0.424, 0.430, 0.302, 0.372, 0.428 against a
~0.46 start; only seeds 3 and 4 show any movement at all.

**The "untouched" column is the sharpest diagnostic in this document.** It counts rollouts whose final
box→target distance differs from the initial by less than 5 mm — i.e. the policy never meaningfully
contacted the box. At **70 % and 56 %**, the baseline mostly does not engage the task.

### 2a. Same-split comparison — test split, no confound

Only three of our arms have test-split cells, but they settle the question:

| arm | split | n | final median | fraction of start left | ≤5 cm | ≤10 cm | untouched |
|---|---|---|---|---|---|---|---|
| **`cand4` FM filmv1 K=20** | test | 30 | **0.163 m** | **0.352×** | **20.0 %** | **40.0 %** | 20.0 % |
| `cand3` FM filmv2 K=20 | test | 30 | 0.444 m | 1.000× | 0.0 % | 3.3 % | 23.3 % |
| `cand17` Diffusion K=20 steps400 | test | 30 | 0.430 m | 1.000× | 0.0 % | 0.0 % | **80.0 %** |
| *baseline (seed 42)* | test | 1080 | *0.434 m* | *1.000×* | *0.1 %* | *0.2 %* | *70.0 %* |
| *baseline `__Bf_U3`* | test | 2804 | *0.394 m* | *0.999×* | *0.8 %* | *2.4 %* | *55.6 %* |

**`cand4` beats the baseline decisively on the same split** — 20 % of rollouts within 5 cm versus 0.1–0.8 %,
and it removes ~65 % of the starting gap where the baseline removes none. That is the clean, confound-free
version of the claim.

**`cand3` and `cand17` are at or below baseline.** `cand17` is the worst cell in the batch on engagement
(80 % untouched). Note `cand3`/`cand4` differ in **two** ways — `filmv2`/`steps1000` vs `filmv1`/`steps900` —
so this is suggestive of a FiLM-mode effect but is **not** a clean film ablation; do not cite it as one.

### 2b. Cross-split reference — the K arms against the baseline

⚠️ **The four K-pair engines have no test-split cells** (only `diffuser`/train/`combined_5`), so the rows
below are **train-split numbers against a test-split baseline**. Train numbers are optimistic and this is
**not** a fair head-to-head. It is included because the *magnitude* of the gap (0.28× vs 1.00×, 33 % vs 0.1 %)
is far larger than any plausible train/test gap, and because `cand4` above confirms the direction on matched
splits.

| arm | split | n | final median | fraction of start left | ≤5 cm | ≤10 cm | ≤15 cm | untouched |
|---|---|---|---|---|---|---|---|---|
| **MeanFlow K=100** | train | 30 | **0.139 m** | **0.277×** | **33.3 %** | 43.3 % | 53.3 % | 10.0 % |
| **AlphaFlow K=2** | train | 30 | **0.140 m** | **0.289×** | 16.7 % | 33.3 % | **56.7 %** | 10.0 % |
| **Diffusion K=100** ⚠️ | train | 30 | **0.185 m** | **0.409×** | 23.3 % | 43.3 % | 46.7 % | 16.7 % |
| MeanFlow K=2 | train | 30 | 0.235 m | 0.602× | 20.0 % | 26.7 % | 43.3 % | 3.3 % |
| AlphaFlow K=100 | train | 30 | 0.340 m | 0.689× | 3.3 % | 26.7 % | 36.7 % | 13.3 % |
| FlowMatching K=100 | train | 30 | 0.400 m | 0.951× | 3.3 % | 13.3 % | 20.0 % | 40.0 % |
| Diffusion K=20 ⚠️ | train | 30 | 0.414 m | 0.957× | 3.3 % | 13.3 % | 23.3 % | 36.7 % |
| FlowMatching K=20 | train | 30 | 0.373 m | 0.979× | 3.3 % | 13.3 % | 20.0 % | 43.3 % |
| *baseline (seed 42)* | *test* | *1080* | *0.434 m* | *1.000×* | *0.1 %* | *0.2 %* | *0.4 %* | *70.0 %* |
| *baseline `__Bf_U3`* | *test* | *2804* | *0.394 m* | *0.999×* | *0.8 %* | *2.4 %* | *5.0 %* | *55.6 %* |

### 2c. The structural finding — engaged or not, with little in between

Sort every cell by "fraction of start left" and two clusters appear:

* **Engaged (0.28× – 0.41×):** MeanFlow K=100, AlphaFlow K=2, `cand4`, Diffusion K=100. Untouched 10–20 %.
* **Partially engaged (0.60× – 0.69×):** MeanFlow K=2, AlphaFlow K=100.
* **Not engaged (0.95× – 1.00×):** FlowMatching (both K), Diffusion K=20, `cand3`, `cand17`, **and both
  baselines**. Untouched 23–80 %.

**This reframes the whole `K` question.** `K` is not gradually tuning distance — it is flipping arms between
"engages the box" and "behaves like the baseline". Diffusion at K=20 is a baseline-grade no-op (0.957×,
36.7 % untouched); at K=100 it is a working policy (0.409×, 16.7 %). MeanFlow moves from partial to full
engagement. AlphaFlow moves the wrong way. FlowMatching never leaves the no-op cluster at any `K` — and with
40–43 % untouched it is *worse engaged* than the 6-seed baseline.

⚠️ **Caveats on this section.** (a) The baseline runs `geo=none` — no constraint projection — which if
anything should make it *easier* to move the box, so the comparison is conservative against us. (b) Baseline
n is 1080/2804 versus 30 for our cells; the baseline numbers are far more stable and our 30-rollout
percentages carry wide intervals (±~9 pp at 33 %). (c) Seeds differ throughout. (d) The train/test mismatch
in §2b is real and unresolved for MF/AF.

---

## 3. MeanFlow — K=100 vs K=2  🔑

✅ **Clean inference-only contrast** — identical checkpoint, only `flow_steps_v3` differs.

**K=100 gets the box closer, and it is not a mean artefact — the whole distribution moves.**

Median final distance **0.139 m vs 0.235 m**. At the median, K=100 leaves 28 % of the starting gap; K=2
leaves 60 %. 26/30 rollouts end closer than they started, against 19/30.

Look at the counts: **10/30 rollouts finish within 5 cm at K=100, versus 6/30 at K=2**; 13 vs 8 within 10 cm;
28 vs 20 within 50 cm. K=100 has more rollouts under every threshold from 5 cm upward.

The tail is where the mean gap comes from: K=2's worst rollout ends **2.75 m** away (rollout 16 — the box is
flung right off the workspace), K=100's worst is 0.72 m. K=2 has 6 rollouts that end *further* from the target
than they started, three of them badly (0.90 m, 0.76 m, 0.75 m on rollouts 0, 7, 11 — all from a ~0.42 m
start). K=100 has 4, and its worst overshoot is only 1.06× the start.

**The caveat, stated plainly:** K=100 is closer on 19 of 30 contexts and further on 10 — decent, but the sign
test still reads p=0.14. K=100 wins big and loses small, so Wilcoxon (magnitude-aware) gives p=0.069 while the
sign test (direction-only) does not clear 0.05. On a single seed, call this a **strong trend, not a proven
effect**.

### 3a. Final box→target distance, metres (n=30)

| arm | min | p10 | p25 | **median** | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| *starting distance* | *0.3598* | *0.3689* | *0.4104* | *0.4671* | *0.4886* | *0.5209* | *0.5903* | *0.4547* |
| **K=100** | 0.0049 | 0.0291 | 0.0348 | **0.1388** | 0.3995 | 0.4717 | 0.7249 | 0.2200 |
| K=2 | 0.0128 | 0.0223 | 0.0689 | **0.2354** | 0.6929 | 0.8613 | 2.7474 | 0.4192 |

### 3b. How many rollouts finish within X metres

| within | 0.018 m | 0.03 m | 0.05 m | 0.075 m | 0.1 m | 0.15 m | 0.2 m | 0.3 m | 0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| **K=100** | **2** | **4** | **10** | **12** | **13** | **16** | **16** | **20** | **28** |
| K=2 | 2 | 4 | 6 | 8 | 8 | 13 | 13 | 17 | 20 |
| *(at start)* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *24* |

*First column is the 1.8 cm success gate. Bottom row = how many rollouts already satisfied the threshold before the episode began.*

### 3c. Paired comparison on distance

| arm | mean | **median** | closer than start | median fraction of start left |
|---|---|---|---|---|
| **K=100** | 0.2200 m | **0.1388 m** | 26/30 | **0.277×** |
| K=2 | 0.4192 m | 0.2354 m | 19/30 | 0.602× |

**K=100 is closer on 19/30 contexts, further on 10, tied on 1.** Δ mean = -0.1992 m · Wilcoxon p = **0.0693** · sign p = 0.1360

### 3d. Every rollout — direct distances in metres

| rid | start | **K=100** | **K=2** | Δ (K=2−K=100) | K=100 left | K=2 left |
|---|---|---|---|---|---|---|
| 0 | 0.474 | **0.0564** | **0.9030** | +0.8466 🟢 | 0.12× | 1.90× |
| 1 | 0.471 | **0.1339** | **0.1380** | +0.0041 | 0.28× | 0.29× |
| 2 | 0.407 | **0.2384** | **0.0689** | -0.1694 🔴 | 0.59× | 0.17× |
| 3 | 0.434 | **0.4618** | **0.5533** | +0.0915 | 1.06× | 1.27× |
| 4 | 0.522 | **0.0419** | **0.4779** | +0.4360 🟢 | 0.08× | 0.91× |
| 5 | 0.484 | **0.2828** | **0.1097** | -0.1731 🔴 | 0.58× | 0.23× |
| 6 | 0.459 | **0.0781** | **0.1205** | +0.0424 | 0.17× | 0.26× |
| 7 | 0.415 | **0.0352** | **0.7591** | +0.7239 🟢 | 0.08× | 1.83× |
| 8 | 0.452 | **0.0276** | **0.1280** | +0.1004 | 0.06× | 0.28× |
| 9 | 0.410 | **0.0049** | **0.4177** | +0.4129 🟢 | 0.01× | 1.02× |
| 10 | 0.521 | **0.2673** | **0.0223** | -0.2450 🔴 | 0.51× | 0.04× |
| 11 | 0.361 | **0.0331** | **0.7508** | +0.7177 🟢 | 0.09× | 2.08× |
| 12 | 0.590 | **0.0618** | **0.0687** | +0.0069 | 0.10× | 0.12× |
| 13 | 0.406 | **0.2420** | **0.2318** | -0.0101 | 0.60× | 0.57× |
| 14 | 0.369 | **0.0146** | **0.2339** | +0.2193 🟢 | 0.04× | 0.63× |
| 15 | 0.489 | **0.7249** | **0.8613** | +0.1364 | 1.48× | 1.76× |
| 16 | 0.386 | **0.4717** | **2.7474** | +2.2756 🟢 | 1.22× | 7.11× |
| 17 | 0.489 | **0.4504** | **0.6929** | +0.2425 🟢 | 0.92× | 1.42× |
| 18 | 0.481 | **0.4453** | **0.0387** | -0.4066 🔴 | 0.93× | 0.08× |
| 19 | 0.500 | **0.0291** | **0.0345** | +0.0054 | 0.06× | 0.07× |
| 20 | 0.430 | **0.0316** | **0.0178** | -0.0138 | 0.07× | 0.04× |
| 21 | 0.463 | **0.0319** | **0.0128** | -0.0191 | 0.07× | 0.03× |
| 22 | 0.534 | **0.1436** | **0.0203** | -0.1234 | 0.27× | 0.04× |
| 23 | 0.516 | **0.5612** | **0.6468** | +0.0856 | 1.09× | 1.25× |
| 24 | 0.360 | **0.3598** | **0.3598** | +0.0000 | 1.00× | 1.00× |
| 25 | 0.484 | **0.0348** | **0.8630** | +0.8281 🟢 | 0.07× | 1.78× |
| 26 | 0.425 | **0.3995** | **0.7070** | +0.3075 🟢 | 0.94× | 1.66× |
| 27 | 0.362 | **0.3619** | **0.2368** | -0.1251 | 1.00× | 0.65× |
| 28 | 0.473 | **0.1014** | **0.1125** | +0.0111 | 0.21× | 0.24× |
| 29 | 0.472 | **0.4723** | **0.2403** | -0.2321 🔴 | 1.00× | 0.51× |

*🟢 = K=100 closer by >15 cm · 🔴 = K=2 closer by >15 cm · "left" = final ÷ start (1.00× = no progress, >1 = ended further away than it started)*

---

## 4. AlphaFlow — K=100 vs K=2  🔴 reversed

✅ **Clean inference-only contrast** — identical checkpoint, only `flow_steps_v3` differs.

**AlphaFlow runs the opposite way: fewer steps get the box closer, and here the statistics are solid.**

Median final distance **0.136 m at K=2 vs 0.340 m at K=100** — the reverse of MeanFlow and a bigger gap.
K=2 leaves 29 % of the starting gap at the median; K=100 leaves 69 %.

K=100 is closer on only 7 of 30 contexts and further on 22 — **sign test p=0.008**, so this one clears both
tests, unlike MeanFlow's. Counts favour K=2 at every threshold: 5 vs 1 within 5 cm, 10 vs 8 within 10 cm,
17 vs 11 within 15 cm, 21 vs 14 within 30 cm.

**Mechanically this is a red flag.** More Euler steps should converge toward the exact ODE solution. AlphaFlow
lands *further away* with 50× more steps, on more than two thirds of rollouts — meaning the coarse 2-step
approximation beats the converged one. The 2-step result is benefiting from truncation error, not from the
model. Check `afsch=sigmoid` and the `ts=logit_normal` two-time parameterisation.

### 4a. Final box→target distance, metres (n=30)

| arm | min | p10 | p25 | **median** | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| *starting distance* | *0.3598* | *0.3689* | *0.4104* | *0.4671* | *0.4886* | *0.5209* | *0.5903* | *0.4547* |
| **K=100** | 0.0152 | 0.0880 | 0.0946 | **0.3403** | 0.4477 | 0.8573 | 1.1264 | 0.3598 |
| K=2 | 0.0071 | 0.0437 | 0.0794 | **0.1398** | 0.4320 | 0.5361 | 1.2994 | 0.2589 |

### 4b. How many rollouts finish within X metres

| within | 0.018 m | 0.03 m | 0.05 m | 0.075 m | 0.1 m | 0.15 m | 0.2 m | 0.3 m | 0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| **K=100** | **1** | **1** | **1** | **2** | **8** | **11** | **14** | **14** | **24** |
| K=2 | 3 | 3 | 5 | 7 | 10 | 17 | 17 | 21 | 25 |
| *(at start)* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *24* |

*First column is the 1.8 cm success gate. Bottom row = how many rollouts already satisfied the threshold before the episode began.*

### 4c. Paired comparison on distance

| arm | mean | **median** | closer than start | median fraction of start left |
|---|---|---|---|---|
| **K=100** | 0.3598 m | **0.3403 m** | 22/30 | **0.689×** |
| K=2 | 0.2589 m | 0.1398 m | 24/30 | 0.289× |

**K=100 is closer on 7/30 contexts, further on 22, tied on 1.** Δ mean = +0.1010 m · Wilcoxon p = **0.0323** · sign p = 0.0081

### 4d. Every rollout — direct distances in metres

| rid | start | **K=100** | **K=2** | Δ (K=2−K=100) | K=100 left | K=2 left |
|---|---|---|---|---|---|---|
| 0 | 0.474 | **0.1892** | **0.1073** | -0.0818 | 0.40× | 0.23× |
| 1 | 0.471 | **0.0946** | **0.2034** | +0.1088 | 0.20× | 0.43× |
| 2 | 0.407 | **0.0742** | **0.1488** | +0.0746 | 0.18× | 0.37× |
| 3 | 0.434 | **0.4345** | **0.4345** | +0.0000 | 1.00× | 1.00× |
| 4 | 0.522 | **0.1690** | **0.1440** | -0.0250 | 0.32× | 0.28× |
| 5 | 0.484 | **0.1666** | **0.1289** | -0.0377 | 0.34× | 0.27× |
| 6 | 0.459 | **0.5143** | **0.5112** | -0.0032 | 1.12× | 1.11× |
| 7 | 0.415 | **0.4157** | **0.0147** | -0.4010 🔴 | 1.00× | 0.04× |
| 8 | 0.452 | **0.0904** | **0.0888** | -0.0016 | 0.20× | 0.20× |
| 9 | 0.410 | **1.1264** | **0.6125** | -0.5139 🔴 | 2.74× | 1.49× |
| 10 | 0.521 | **0.1190** | **0.6789** | +0.5599 🟢 | 0.23× | 1.30× |
| 11 | 0.361 | **0.3577** | **0.3573** | -0.0004 | 0.99× | 0.99× |
| 12 | 0.590 | **0.4094** | **0.0437** | -0.3657 🔴 | 0.69× | 0.07× |
| 13 | 0.406 | **0.1078** | **0.0794** | -0.0283 | 0.27× | 0.20× |
| 14 | 0.369 | **1.0167** | **0.0594** | -0.9573 🔴 | 2.76× | 0.16× |
| 15 | 0.489 | **0.8573** | **0.4874** | -0.3699 🔴 | 1.75× | 1.00× |
| 16 | 0.386 | **0.3796** | **0.2377** | -0.1419 | 0.98× | 0.62× |
| 17 | 0.489 | **0.4887** | **0.2974** | -0.1913 🔴 | 1.00× | 0.61× |
| 18 | 0.481 | **0.0784** | **0.4320** | +0.3536 🟢 | 0.16× | 0.90× |
| 19 | 0.500 | **0.0880** | **0.0149** | -0.0731 | 0.18× | 0.03× |
| 20 | 0.430 | **0.0945** | **0.1280** | +0.0335 | 0.22× | 0.30× |
| 21 | 0.463 | **0.4477** | **0.2651** | -0.1826 🔴 | 0.97× | 0.57× |
| 22 | 0.534 | **0.0152** | **0.1088** | +0.0935 | 0.03× | 0.20× |
| 23 | 0.516 | **0.7824** | **0.5361** | -0.2463 🔴 | 1.52× | 1.04× |
| 24 | 0.360 | **0.0927** | **0.0465** | -0.0462 | 0.26× | 0.13× |
| 25 | 0.484 | **0.9344** | **0.1356** | -0.7988 🔴 | 1.93× | 0.28× |
| 26 | 0.425 | **0.3751** | **0.0071** | -0.3680 🔴 | 0.88× | 0.02× |
| 27 | 0.362 | **0.1402** | **0.0826** | -0.0576 | 0.39× | 0.23× |
| 28 | 0.473 | **0.4125** | **1.2994** | +0.8869 🟢 | 0.87× | 2.75× |
| 29 | 0.472 | **0.3229** | **0.0747** | -0.2482 🔴 | 0.68× | 0.16× |

*🟢 = K=100 closer by >15 cm · 🔴 = K=2 closer by >15 cm · "left" = final ÷ start (1.00× = no progress, >1 = ended further away than it started)*

---

## 5. FlowMatching — K=100 vs K=20  ⬜ barely moves the box

✅ **Clean inference-only contrast** — identical checkpoint, only `flow_steps_v3` differs.

**K changes nothing, because this arm is not doing the task at either setting.**

Median final distance 0.390 m (K=100) vs 0.360 m (K=20) — from a 0.463 m median start. **The box ends
essentially where it began**: median 0.95× and 0.98× of the starting distance. Compare MeanFlow K=100 at
0.28×.

Only **1/30 rollouts finishes within 5 cm** at either setting, and 4/30 within 10 cm. K=100 is closer on 9
contexts, further on 13, p=0.56 — noise.

Every other diagnostic agrees this arm is broken: `sat_rate` ≈ 0.41 (worst of the four), ~232 violations per
rollout, `max_phys_error` ≈ 2.3 (5× MeanFlow's 0.44), 0/30 success strict *and* relaxed. This needs
debugging, not tuning — exclude it from comparisons until it moves the box.

### 5a. Final box→target distance, metres (n=30)

| arm | min | p10 | p25 | **median** | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| *starting distance* | *0.3598* | *0.3689* | *0.4104* | *0.4671* | *0.4886* | *0.5209* | *0.5903* | *0.4547* |
| **K=100** | 0.0387 | 0.0926 | 0.2498 | **0.4004** | 0.4811 | 0.5157 | 0.5903 | 0.3471 |
| K=20 | 0.0369 | 0.0742 | 0.2271 | **0.3725** | 0.4842 | 0.5209 | 0.5903 | 0.3373 |

### 5b. How many rollouts finish within X metres

| within | 0.018 m | 0.03 m | 0.05 m | 0.075 m | 0.1 m | 0.15 m | 0.2 m | 0.3 m | 0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| **K=100** | **0** | **0** | **1** | **2** | **4** | **6** | **7** | **10** | **26** |
| K=20 | 0 | 0 | 1 | 4 | 4 | 6 | 7 | 11 | 25 |
| *(at start)* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *24* |

*First column is the 1.8 cm success gate. Bottom row = how many rollouts already satisfied the threshold before the episode began.*

### 5c. Paired comparison on distance

| arm | mean | **median** | closer than start | median fraction of start left |
|---|---|---|---|---|
| **K=100** | 0.3471 m | **0.4004 m** | 26/30 | **0.951×** |
| K=20 | 0.3373 m | 0.3725 m | 26/30 | 0.979× |

**K=100 is closer on 9/30 contexts, further on 13, tied on 8.** Δ mean = +0.0098 m · Wilcoxon p = **0.5590** · sign p = 0.5235

### 5d. Every rollout — direct distances in metres

| rid | start | **K=100** | **K=20** | Δ (K=20−K=100) | K=100 left | K=20 left |
|---|---|---|---|---|---|---|
| 0 | 0.474 | **0.0387** | **0.2271** | +0.1884 🟢 | 0.08× | 0.48× |
| 1 | 0.471 | **0.4652** | **0.3205** | -0.1447 | 0.99× | 0.68× |
| 2 | 0.407 | **0.1182** | **0.1089** | -0.0093 | 0.29× | 0.27× |
| 3 | 0.434 | **0.4345** | **0.4345** | +0.0000 | 1.00× | 1.00× |
| 4 | 0.522 | **0.5342** | **0.5443** | +0.0101 | 1.02× | 1.04× |
| 5 | 0.484 | **0.4842** | **0.4842** | +0.0000 | 1.00× | 1.00× |
| 6 | 0.459 | **0.4833** | **0.4868** | +0.0035 | 1.05× | 1.06× |
| 7 | 0.415 | **0.4149** | **0.4149** | +0.0000 | 1.00× | 1.00× |
| 8 | 0.452 | **0.1404** | **0.1380** | -0.0024 | 0.31× | 0.30× |
| 9 | 0.410 | **0.4104** | **0.4104** | +0.0000 | 1.00× | 1.00× |
| 10 | 0.521 | **0.5209** | **0.5209** | +0.0000 | 1.00× | 1.00× |
| 11 | 0.361 | **0.0588** | **0.0369** | -0.0219 | 0.16× | 0.10× |
| 12 | 0.590 | **0.5903** | **0.5903** | +0.0000 | 1.00× | 1.00× |
| 13 | 0.406 | **0.0781** | **0.2604** | +0.1823 🟢 | 0.19× | 0.64× |
| 14 | 0.369 | **0.3296** | **0.0514** | -0.2782 🔴 | 0.89× | 0.14× |
| 15 | 0.489 | **0.2983** | **0.0742** | -0.2240 🔴 | 0.61× | 0.15× |
| 16 | 0.386 | **0.3864** | **0.3852** | -0.0012 | 1.00× | 1.00× |
| 17 | 0.489 | **0.3904** | **0.4016** | +0.0112 | 0.80× | 0.82× |
| 18 | 0.481 | **0.4811** | **0.4811** | -0.0000 | 1.00× | 1.00× |
| 19 | 0.500 | **0.1711** | **0.1748** | +0.0037 | 0.34× | 0.35× |
| 20 | 0.430 | **0.2520** | **0.2512** | -0.0008 | 0.59× | 0.58× |
| 21 | 0.463 | **0.4629** | **0.4598** | -0.0031 | 1.00× | 0.99× |
| 22 | 0.534 | **0.2498** | **0.5323** | +0.2824 🟢 | 0.47× | 1.00× |
| 23 | 0.516 | **0.5157** | **0.5157** | +0.0000 | 1.00× | 1.00× |
| 24 | 0.360 | **0.3598** | **0.3598** | +0.0000 | 1.00× | 1.00× |
| 25 | 0.484 | **0.4842** | **0.4842** | +0.0000 | 1.00× | 1.00× |
| 26 | 0.425 | **0.0926** | **0.0647** | -0.0278 | 0.22× | 0.15× |
| 27 | 0.362 | **0.3245** | **0.3491** | +0.0247 | 0.90× | 0.96× |
| 28 | 0.473 | **0.4116** | **0.3079** | -0.1037 | 0.87× | 0.65× |
| 29 | 0.472 | **0.4323** | **0.2493** | -0.1830 🔴 | 0.92× | 0.53× |

*🟢 = K=100 closer by >15 cm · 🔴 = K=20 closer by >15 cm · "left" = final ÷ start (1.00× = no progress, >1 = ended further away than it started)*

---

## 6. Diffusion — K=100 vs K=20  ⚠️ confounded

🔴 **NOT a clean K contrast.** `n_diffusion_steps` is a **training** key too (`config/aligning-d3il-visual.py:901`), so the two arms load **different checkpoints** (`…/H8_K100_D…` vs `…/H8_K20_D…`). Read as a checkpoint+schedule comparison, never as a pure NFE ablation.

**The largest distance gap in the batch — and the least attributable.**

Median final distance **0.185 m vs 0.414 m**. Median 0.41× vs 0.96× of the starting gap: the K=20 arm leaves
the box almost exactly where it found it, the K=100 arm removes ~59 % of the gap. 27/30 rollouts end closer
than they started vs 23/30.

Counts: **7/30 within 5 cm vs 1/30**; 13 vs 4 within 10 cm; 21 vs 11 within 30 cm. K=100 is closer on 18 of
30 contexts (Wilcoxon p=0.014 — the strongest distance result here after AlphaFlow), though the sign test
reads 0.26: small losses, large wins again.

**But the checkpoints differ**, so none of this is attributable to sampler steps. It may simply be that the
K=100-trained diffusion model is better. Settling it needs a retrain holding `n_diffusion_steps` fixed at
train time and varying only the sampler.

### 6a. Final box→target distance, metres (n=30)

| arm | min | p10 | p25 | **median** | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|
| *starting distance* | *0.3598* | *0.3689* | *0.4104* | *0.4671* | *0.4886* | *0.5209* | *0.5903* | *0.4547* |
| **K=100** | 0.0255 | 0.0338 | 0.0513 | **0.1849** | 0.3324 | 0.4846 | 0.5074 | 0.2167 |
| K=20 | 0.0280 | 0.0947 | 0.1586 | **0.4140** | 0.4811 | 0.5223 | 1.5126 | 0.3901 |

### 6b. How many rollouts finish within X metres

| within | 0.018 m | 0.03 m | 0.05 m | 0.075 m | 0.1 m | 0.15 m | 0.2 m | 0.3 m | 0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| **K=100** | **0** | **2** | **7** | **11** | **13** | **14** | **16** | **21** | **28** |
| K=20 | 0 | 1 | 1 | 3 | 4 | 7 | 9 | 11 | 26 |
| *(at start)* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *0* | *24* |

*First column is the 1.8 cm success gate. Bottom row = how many rollouts already satisfied the threshold before the episode began.*

### 6c. Paired comparison on distance

| arm | mean | **median** | closer than start | median fraction of start left |
|---|---|---|---|---|
| **K=100** | 0.2167 m | **0.1849 m** | 27/30 | **0.409×** |
| K=20 | 0.3901 m | 0.4140 m | 23/30 | 0.957× |

**K=100 is closer on 18/30 contexts, further on 11, tied on 1.** Δ mean = -0.1734 m · Wilcoxon p = **0.0137** · sign p = 0.2649

### 6d. Every rollout — direct distances in metres

| rid | start | **K=100** | **K=20** | Δ (K=20−K=100) | K=100 left | K=20 left |
|---|---|---|---|---|---|---|
| 0 | 0.474 | **0.1718** | **0.4741** | +0.3023 🟢 | 0.36× | 1.00× |
| 1 | 0.471 | **0.4708** | **0.4713** | +0.0005 | 1.00× | 1.00× |
| 2 | 0.407 | **0.0463** | **1.5126** | +1.4663 🟢 | 0.11× | 3.72× |
| 3 | 0.434 | **0.1979** | **0.1754** | -0.0225 | 0.46× | 0.40× |
| 4 | 0.522 | **0.0776** | **0.5223** | +0.4447 🟢 | 0.15× | 1.00× |
| 5 | 0.484 | **0.4846** | **0.4842** | -0.0004 | 1.00× | 1.00× |
| 6 | 0.459 | **0.5074** | **0.3302** | -0.1772 🔴 | 1.11× | 0.72× |
| 7 | 0.415 | **0.2772** | **0.1367** | -0.1405 | 0.67× | 0.33× |
| 8 | 0.452 | **0.0638** | **0.4524** | +0.3887 🟢 | 0.14× | 1.00× |
| 9 | 0.410 | **0.0614** | **0.1174** | +0.0560 | 0.15× | 0.29× |
| 10 | 0.521 | **0.0292** | **0.0947** | +0.0655 | 0.06× | 0.18× |
| 11 | 0.361 | **0.0322** | **0.1223** | +0.0901 | 0.09× | 0.34× |
| 12 | 0.590 | **0.0255** | **0.5721** | +0.5466 🟢 | 0.04× | 0.97× |
| 13 | 0.406 | **0.0393** | **0.4063** | +0.3670 🟢 | 0.10× | 1.00× |
| 14 | 0.369 | **0.2300** | **0.2734** | +0.0434 | 0.62× | 0.74× |
| 15 | 0.489 | **0.0802** | **0.4886** | +0.4084 🟢 | 0.16× | 1.00× |
| 16 | 0.386 | **0.0389** | **0.0280** | -0.0108 | 0.10× | 0.07× |
| 17 | 0.489 | **0.4887** | **0.4887** | +0.0000 | 1.00× | 1.00× |
| 18 | 0.481 | **0.4837** | **0.4811** | -0.0026 | 1.01× | 1.00× |
| 19 | 0.500 | **0.5001** | **0.4725** | -0.0276 | 1.00× | 0.94× |
| 20 | 0.430 | **0.2205** | **0.4305** | +0.2100 🟢 | 0.51× | 1.00× |
| 21 | 0.463 | **0.0338** | **0.1586** | +0.1248 | 0.07× | 0.34× |
| 22 | 0.534 | **0.0620** | **0.9481** | +0.8860 🟢 | 0.12× | 1.78× |
| 23 | 0.516 | **0.1254** | **0.4749** | +0.3495 🟢 | 0.24× | 0.92× |
| 24 | 0.360 | **0.0513** | **0.3598** | +0.3085 🟢 | 0.14× | 1.00× |
| 25 | 0.484 | **0.3244** | **0.0665** | -0.2579 🔴 | 0.67× | 0.14× |
| 26 | 0.425 | **0.2845** | **0.0689** | -0.2156 🔴 | 0.67× | 0.16× |
| 27 | 0.362 | **0.3324** | **0.3811** | +0.0487 | 0.92× | 1.05× |
| 28 | 0.473 | **0.2958** | **0.2888** | -0.0070 | 0.63× | 0.61× |
| 29 | 0.472 | **0.4657** | **0.4217** | -0.0440 | 0.99× | 0.89× |

*🟢 = K=100 closer by >15 cm · 🔴 = K=20 closer by >15 cm · "left" = final ÷ start (1.00× = no progress, >1 = ended further away than it started)*

---

## 7. Cross-engine summary — distance only

Identical 30 contexts per K cell, median start 0.463 m, seed 6, `filmv1` + `mpc4`, `diffuser`, `combined_5`,
train split. Baseline rows are test split, `geo=none` — see §2b.

| arm | K | median final | fraction of start left | ≤5 cm | ≤10 cm | ≤30 cm | untouched | worst rollout | cost/step |
|---|---|---|---|---|---|---|---|---|---|
| **MeanFlow** | **100** | **0.139 m** | **0.28×** | **33 %** | **43 %** | **67 %** | 10 % | 0.72 m | 893 ms |
| MeanFlow | 2 | 0.235 m | 0.60× | 20 % | 27 % | 57 % | **3 %** | **2.75 m** | **28 ms** |
| AlphaFlow | 100 | 0.340 m | 0.69× | 3 % | 27 % | 47 % | 13 % | 1.13 m | 902 ms |
| **AlphaFlow** | **2** | **0.140 m** | **0.29×** | 17 % | 33 % | **70 %** | 10 % | 1.30 m | **27 ms** |
| FlowMatching | 100 | 0.400 m | 0.95× | 3 % | 13 % | 33 % | 40 % | 0.59 m | 1426 ms |
| FlowMatching | 20 | 0.373 m | 0.98× | 3 % | 13 % | 37 % | 43 % | 0.59 m | **294 ms** |
| **Diffusion** ⚠️ | **100** | **0.185 m** | **0.41×** | 23 % | **43 %** | **70 %** | 17 % | 0.51 m | 1527 ms |
| Diffusion ⚠️ | 20 | 0.414 m | 0.96× | 3 % | 13 % | 37 % | 37 % | 1.51 m | **298 ms** |
| *d3il baseline (s42)* | *—* | *0.434 m* | *1.000×* | *0.1 %* | *0.2 %* | *5.8 %* | *70 %* | *—* | *—* |
| *d3il baseline `__Bf_U3`* | *—* | *0.394 m* | *0.999×* | *0.8 %* | *2.4 %* | *19.6 %* | *56 %* | *—* | *—* |

### Best arms by distance

| rank | arm | median final | ≤5 cm | fraction of start left |
|---|---|---|---|---|
| 1 | **MeanFlow K=100** | 0.139 m | **33 %** | 0.28× |
| 2 | **AlphaFlow K=2** | 0.140 m | 17 % | 0.29× |
| 3 | Diffusion K=100 ⚠️ | 0.185 m | 23 % | 0.41× |
| 4 | **`cand4` FM filmv1 K=20** (test) | 0.163 m | 20 % | 0.352× |
| 5 | MeanFlow K=2 | 0.235 m | 20 % | 0.60× |
| 6 | AlphaFlow K=100 | 0.340 m | 3 % | 0.69× |
| 7 | FlowMatching / Diffusion K=20 / `cand3` / `cand17` | 0.37–0.44 m | 0–3 % | 0.95–1.00× |
| — | *d3il baseline* | *0.39–0.43 m* | *0.1–0.8 %* | *0.999–1.000×* |

MeanFlow K=100 and AlphaFlow K=2 tie on median (0.139 vs 0.140 m), but MeanFlow places roughly **twice as
many rollouts inside 5 cm** (33 % vs 17 %) — at 33× the per-step cost. AlphaFlow K=2 is the throughput winner
by a wide margin. `cand4` is the only arm in the top group verified on the **test** split.

---

## 8. Caveats

1. **Single seed (6) on all K arms.** MeanFlow's K=100 advantage is a **trend** (Wilcoxon 0.069, sign 0.14),
   not a proven effect. AlphaFlow's K=2 advantage clears both tests (0.032 / 0.008).
2. **Train vs test.** The four K-pair engines have **no test-split cells**; §2b compares them against a
   test-split baseline and is therefore not a fair head-to-head. `cand4` (§2a) is the only same-split
   confirmation. Nothing here demonstrates generalisation for MF or AF.
3. **Final distance, not closest approach** (§0). A rollout that arrived and drifted off scores as a miss.
4. **Baseline runs `geo=none`** — no constraint projection, which should make moving the box *easier*, so the
   comparison is conservative in our favour.
5. **Sample sizes are lopsided:** 1080/2804 baseline rollouts against 30 per K cell. Our percentages carry
   roughly ±9 pp at the 33 % level; the baseline's are tight.
6. **Diffusion pair is checkpoint-confounded** (§6). MF / AF / FM pairs are clean.
7. **`cand3` vs `cand4` differ in two knobs** (filmv2/steps1000 vs filmv1/steps900) — suggestive, not a clean
   FiLM ablation.
8. **`#14` is a partial run** — only `diffuser` plus a truncated 11-rollout `dpcc-r`, so nothing is verified
   through the projector. On that partial cell MF K=100 costs **≈15 s per control step** (≈297× K=2).
9. **XY distance ignores height;** the env's gate uses a 3-D norm. Small on a flat table, but not identical.
10. **`K` ≠ MPC fan** (§0), which is pinned at 4 and never ablated.

---

## 9. Next steps

1. 🔑 **Run the K arms on the test split.** This is now the biggest hole: MeanFlow K=100 and AlphaFlow K=2 are
   the two best arms in the batch and neither has a single test-split rollout. Without it, "we beat the d3il
   baseline" rests on `cand4` alone.
2. 🔑 **Get the real minimum distance.** Cluster-side pass over the per-rollout JSONs to extract
   `min(dist_to_target)` and its timestep — tells us whether these policies *reach* the target and drift off,
   or never arrive. Completely different diagnosis, completely different fix. Log raw XY distance per step at
   the same time (the existing curve is the blended `mean_distance`).
3. **Adopt "untouched %" as a standard reported metric.** Fraction of rollouts whose box→target distance
   changes by <5 mm. It separated engaged from no-op arms more cleanly than anything else here, and it would
   have flagged `cand17` and the FlowMatching arms immediately.
4. **Set `K` per engine, not globally** — §7: MeanFlow and Diffusion want high K; AlphaFlow and FlowMatching
   want low K unconditionally.
5. **Confirm MeanFlow K=100 across ≥3 seeds** before relying on it. One seed, sign p=0.14.
6. 🔴 **Chase AlphaFlow's inverted step response** (§4) — more Euler steps landing further from the target on
   22/30 rollouts means the converged ODE solution is worse than its coarse approximation. Check
   `afsch=sigmoid` and `ts=logit_normal`.
7. 🔴 **Debug FlowMatching and `cand17`** (§5, §2a) — at 0.95–1.00× with 40–80 % untouched these are
   baseline-grade no-ops, not tuning candidates. Exclude them from comparisons until they move the box.
8. **Log `pos_dist` and `rot_dist` separately** rather than only their 0.5/0.5 blend.
9. **Still open:** the MPC candidate fan (`mpc_batch_size`) has never been ablated.

---

*Paired per-rollout analysis from `per_rollout_detail.csv` of batch `batch_va2_20260826_142750`. Wilcoxon
signed-rank (normal approx., tie-corrected) and exact two-sided sign test, pure Python (no SciPy in this
container). `K` semantics from `config/aligning-d3il-visual.py:898-946`; success gate from
`aligning.py:198-199,344-345`; `mean_dist_per_rollout` semantics from `eval_visual_aligning_dpcc.py:1379-1382`
and `aligning.py:316`; `dist_to_target` export at `eval_visual_aligning_dpcc.py:1449`. Baseline cells are
candidates 18/19 (`d3il_baseline_ddpm_encdec_vision`), test split, `geo=none`.*
