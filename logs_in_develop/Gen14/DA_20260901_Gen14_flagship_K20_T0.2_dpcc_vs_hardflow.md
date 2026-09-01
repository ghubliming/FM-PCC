# DA 2026-09-01 — Gen14 flagship: MeanFlow K=20 @ T=0.2, DPCC vs HardFlow-SLSQP

**Runs analysed:** jobs **25247** (mf) and **25248** (fm), both launched 2026-08-31 14:32 UTC on i6-gpu-1, git rev `938641c`.
**Aggregation:** `temp/3008/batch_va2_20260901_093100/per_rollout_detail.csv` (11 842 rollout rows; 540 of them from these two jobs).
**Raw logs:** `temp/3008/2026-08-31/16_32_53_eval_mix_visual_aligning_25247.log`, `…/16_33_05_eval_mix_visual_aligning_25248.log`.

---

## Metric conventions (read this first)

| column | meaning | trap |
|---|---|---|
| `context_final_xy_dist` | raw box→target XY distance at rollout end, **metres** | this is the correct distance metric |
| `context_init_xy_dist` | same at rollout start (≈ 0.453 m here) | |
| **untouched** | `final == init` to 1e-6 → **the box was never moved at all** | the single most interpretable number on this task |
| progress | `1 − final/init`, mean over rollouts | negative = pushed the box further away |
| `constraint_exec_zero_violation` | 1 if the rollout had zero executed violations | ≡ `collision_free_completed` |
| `avg_time_ms` | **per replan step**, ~400 steps/rollout | not per rollout |
| `mean_dist_per_rollout` | `0.5·(pos_dist_3D + rot_err/π)` | **not a distance — never quote it** |

**`n_success` is 0.000 for every arm in this corpus**, ours and the baseline alike. The strict aligning success criterion (box inside the target pose tolerance) is met by nothing at these checkpoints. Every claim below is therefore about **distance / progress / constraints**, never about success rate. Do not read any of this as a success-rate result.

Rollouts are paired across arms by geometry fingerprint `(box_init_xy_x, box_init_xy_y, target_xy_x, target_xy_y)` rounded to 1e-4, because `context_index` is blank in `per_rollout_detail.csv`. All sign tests are exact two-sided; McNemar is exact.

---

# Part 1 — Are the results ready?

**Half of them.**

| job | engine | K | T | items | status |
|---|---|---|---|---|---|
| **25247** | `mf` (MeanFlow, Gen3v6 wrapped in Gen14 VisualUNetTwoTime, 26.4 M) | 20 | 0.2 | **38 / 38** | ✅ `Job completed successfully.` |
| **25248** | `fm` (FlowMatchingODE, Gen7, same VisualUNet) | 20 | 0.2 | **17 / 38**, died inside item 17 | ❌ `PASS T=0.2 FAILED (exit 1)` |

Row counts confirm it: 380 mf rows = 38 items × 10 rollouts; 160 fm rows = 16 completed items × 10.

**What this means for the arm-B vs arm-C question you asked about:** the comparison **exists and is complete for MeanFlow**, on both `combined_5` and `combined_5-tightened`. It **does not exist for FM** — job 25248 crashed on the very first arm-C item. Root cause in Part 4.

### The flagship setup, stated once

| knob | value | where it came from |
|---|---|---|
| engine | `mf` / `fm` | CLI arg 1 |
| seed | 6 | CLI arg 2 |
| **K** (`flow_steps_v3`) | **20** | CLI arg 4 — overrides the checkpoint's train-time K=100 |
| **T** (`diffusion_timestep_threshold`) | **0.2** | `MIX_PROJ_T=0.2` → `--proj-threshold` |
| projector calls / replan | **4** (`int((1−0.2)·20)=16` → steps 16,17,18,19, i.e. τ ∈ [0.80, 1.00]) | log line `projection budget: 10 -> 4` |
| arm C variants | `hardflow_new-{r,c,t}` | `HFFM_VARIANTS` |
| arm C threshold | **inherited = 0.2** | `hardflow.activation_threshold: null`; log confirms `act_thr=0.2` |
| NLP backend | **slsqp**, both arms | `[hardflow][NLP-BACKEND] slsqp … dof=66` |
| candidate fan | **4** both arms (B4_PARITY) | `-r/-c/-t` suffix → `configured_batch` = `mpc_batch_size` = 4 |
| contexts × rollouts | 10 × 1 | cluster-side yaml |
| film / backbone | `v1` / `unet` | must match checkpoint |

### The matching held

Everything I said would make this a clean DPCC-vs-HardFlow comparison actually held in the run:

- **Degeneracy guard passed silently.** `hardflow_step_budget(20, 0.2)` → `n_active = max(20 − int(0.8·20), 1) = 4`, `n_genuine = 3` → tier `HF_OK`. No `[hardflow][BLOCKED]` line anywhere in 25247. (For contrast, the same guard *did* fire in the unrelated α-Flow job 25254 at K=2, A=0.5 → `n_genuine=0 [DEGENERATE]`, disabling all six arms. The guard works.)
- **Both arms fire at the same 4 ODE steps**, τ ∈ [0.80, 1.00], same solver, same `dof=66`, same fan of 4.
- **Same feasible set** — `dynamics_mode: deriv`, so the arms differ only in *when* the constraint is applied.

### SLSQP convergence

6 `[hardflow][NLP-FAILURE]` first-failure notices in 25247 — one per arm-C cell (3 variants × 2 geos), all at **τ = 0.850**, i.e. the *second* of the four solves. Subsequent failures are silent by design; the per-run totals live in `nlp_failures` in each run summary. **τ=0.850 is exactly where T=0.1 failed in job 25216 too** — the first genuine lookahead solve after the sampler leaves the free-flow region is consistently the hardest one. Worth a look, but note the terminal τ=1 solve is separate and is what carries the safety guarantee.

---

# Part 2 — DPCC vs HardFlow-SLSQP (MeanFlow, the flagship comparison)

## 2.1 `combined_5-tightened` — where the projector actually does its job

| variant | dist (m) | 0-viol | violations | ms/step |
|---|---|---|---|---|
| `diffuser` (unguided) | 0.1431 | 0.200 | 31.60 | 172.5 |
| `post_processing` | 0.2469 | 0.800 | 11.90 | 193.6 |
| **`dpcc-r`** | 0.2001 | 0.900 | 2.70 | 265.9 |
| **`hardflow_sls-r`** | 0.2108 | **1.000** | **0.00** | 275.3 |
| **`dpcc-c`** | 0.3059 | 0.900 | 4.30 | 325.3 |
| **`hardflow_sls-c`** | 0.2863 | **1.000** | **0.00** | **282.3** |
| **`dpcc-t`** | 0.2398 | 0.800 | 1.50 | 323.2 |
| **`hardflow_sls-t`** | 0.2555 | **0.900** | **0.60** | **301.5** |
| `gradient` | 0.2370 | 0.300 | 68.10 | 178.9 |

**HardFlow is ≥ DPCC on constraints in 3/3 matched pairs**, and strictly better in 2/3 (perfect 1.000 / zero violations). It is also cheaper per step in 2/3 pairs. Distance is a wash.

Pooled over the three selection rules (n = 30 paired rollouts):

| metric | HardFlow | DPCC | test | p |
|---|---|---|---|---|
| zero-violation rate | **0.967** | 0.867 | McNemar 3 / 0 | **0.250** |
| violations per rollout | **0.200** | 2.833 | sign 4 / 0 / 26 | **0.125** |
| `avg_time_ms` | **286.4** | 304.8 | sign 10 / 17 / 3 | 0.248 |
| `context_final_xy_dist` | 0.2509 | 0.2486 | sign 12 / 10 / 8 | 0.832 |

**Nothing is significant.** Read that plainly: the direction is consistently in HardFlow's favour on constraints — every discordant pair goes HardFlow's way, 4/0 and 3/0 — but with 26 of 30 pairs tied at zero violations the sign test cannot resolve it. **p = 0.125 is the floor** a 4/0 split can reach; even a perfect result at this n would not clear 0.05. This is a power problem, not a null result. **More seeds is the fix, not more analysis.**

## 2.2 `combined_5` — the untightened geo is not informative here

| variant | dist (m) | 0-viol | violations | ms/step |
|---|---|---|---|---|
| `diffuser` | **0.0933** | 0.100 | 61.40 | 190.5 |
| `post_processing` | 0.2167 | 0.300 | 42.30 | 215.7 |
| `dpcc-r` | 0.1553 | 0.300 | 52.20 | 346.7 |
| `hardflow_sls-r` | 0.1576 | 0.400 | 46.30 | 297.7 |
| `dpcc-c` | 0.2985 | 0.000 | 50.60 | 349.1 |
| `hardflow_sls-c` | 0.3393 | 0.200 | 49.40 | 340.9 |
| `dpcc-t` | 0.2317 | 0.200 | 56.80 | 323.6 |
| `hardflow_sls-t` | 0.2358 | 0.300 | 52.70 | 341.9 |

On the untightened geo **no projector materially reduces violations** (61.4 unguided → 42–57 projected), and zero-violation stays at 0.0–0.4 for everything. Pooled HF-vs-DPCC: 49.5 vs 53.2 violations (17/10, p = 0.248), 0.300 vs 0.167 zero-viol (McNemar 6/2, p = 0.289). Directionally the same as tightened, equally unresolved.

**Do not build a projector claim on `combined_5`.** The executed-violation check is stricter than the constraint set the projector is handed, so the plan is feasible and the execution still violates. That gap is exactly what the tightened twin exists to close, and on the tightened twin it does close (31.6 → 0.0–2.7). Quote §2.1, not §2.2, for anything about constraint satisfaction.

## 2.3 What to conclude

**HardFlow-SLSQP is non-dominated against the DPCC projector at K=20, T=0.2, and probably better on constraints — but this run cannot prove it.** At equal distance, it achieves ≥ the zero-violation rate in 3/3 pairs and is cheaper in 2/3, which is a favourable trade-off, not a demonstrated win. The honest statement for now is "matched on distance, directionally ahead on constraints and cost, n too small."

---

# Part 3 — What the flagship settles that earlier runs could not

## 3.1 The K sweep under the pure `diffuser` (unguided) metric

⚠ **Correction to an earlier version of this DA.** My first pass at this table keyed rollouts by `(engine, K/T, geo, variant)` only — but the corpus holds **four different MeanFlow checkpoints** at K=2 (`Bdit`, `Bmf_dit`, `filmv1`, `filmv2`), so that key silently mixed models and produced a contaminated "K=2 → K=20: 1/9, p=0.021". Everything below is restricted to the **`filmv1` UNet checkpoint** — the same weights used at K=20 and K=100 — on the **same 10 contexts**. The conclusion survives; the statistic does not, and the mechanism turns out to be different from what a sign test suggested.

### Why the unguided arm is the right place to read K

`diffuser` runs **no projector at all**. `T` is therefore inert on this arm, and the only thing that changes across these cells is the number of Euler steps. It is the one clean measurement of what K does to the generative field, uncontaminated by the NLP.

### The distribution, not the mean

Mean initial distance = **0.4530 m**. "worse than nothing" = the box ended *further* from the target than it started.

| cell | NFE | mean | median | sd | min | max | worse than nothing | > 0.3 m | ms/replan |
|---|---|---|---|---|---|---|---|---|---|
| **K=2** | 2 | 0.3676 | 0.2779 | **0.3011** | 0.0689 | **0.9030** | **4 / 10** | **5 / 10** | **28.4** |
| **K=20** | 20 | **0.0933** | **0.0741** | **0.0728** | 0.0278 | **0.2617** | **0 / 10** | **0 / 10** | 190.5 |
| **K=100** (T0.1) | 100 | 0.1440 | 0.0952 | 0.1462 | 0.0123 | 0.4618 | 1 / 10 | 1 / 10 | 924.6 |
| **K=100** (T0.5) | 100 | 0.1361 | 0.0673 | **0.1475** | **0.0049** | 0.4618 | 1 / 10 | 1 / 10 | 892.4 |

Per-context, same checkpoint, same contexts, unguided:

| ctx | init | K=2 | K=20 | K=100 (T0.1) | K=100 (T0.5) |
|---|---|---|---|---|---|
| 0 | 0.4345 | 0.5533 | **0.0665** | 0.4618 | 0.4618 |
| 1 | 0.4741 | **0.9030** | 0.0546 | 0.0564 | 0.0564 |
| 2 | 0.4104 | 0.4177 | 0.0309 | 0.0143 | **0.0049** |
| 3 | 0.4066 | **0.0689** | 0.2617 | 0.2384 | 0.2384 |
| 4 | 0.5223 | 0.4779 | **0.0278** | 0.0419 | 0.0419 |
| 5 | 0.4591 | 0.1205 | 0.0817 | 0.1582 | **0.0781** |
| 6 | 0.4524 | 0.1280 | 0.1391 | 0.0403 | **0.0276** |
| 7 | 0.4842 | 0.1097 | **0.0986** | 0.2828 | 0.2828 |
| 8 | 0.4149 | 0.7591 | 0.0288 | **0.0123** | 0.0352 |
| 9 | 0.4713 | 0.1380 | 0.1433 | 0.1339 | **0.1339** |
| **mean** | 0.4530 | 0.3676 | **0.0933** | 0.1440 | 0.1361 |

### Tests

Sign tests are the wrong instrument here — they discard magnitude, and K=2's failure mode is entirely in the tail. Both are given; the **exact paired sign-flip permutation test on the mean difference** is the one to quote.

| comparison | mean Δ (m) | sign W/L/T | sign p | **permutation p** |
|---|---|---|---|---|
| K=2 vs **K=20** | +0.2743 | 3/7/0 | 0.344 | **0.043** |
| K=2 vs K=100 (T0.1) | +0.2236 | 3/7/0 | 0.344 | 0.092 |
| K=2 vs K=100 (T0.5) | +0.2315 | 2/8/0 | 0.109 | 0.080 |
| **K=20** vs K=100 (T0.1) | −0.0507 | 5/5/0 | 1.000 | 0.365 |
| **K=20** vs K=100 (T0.5) | −0.0428 | 5/5/0 | 1.000 | 0.459 |
| K=100 T0.1 vs K=100 T0.5 | +0.0079 | 1/3/6 | 0.625 | 0.625 |

### The noise floor — measured, not assumed

The last row is the important control. **K=100/T0.1 and K=100/T0.5 on the unguided arm are the identical configuration** — same checkpoint, same K, and T is inert without a projector. They are two independent samples of one thing. 6 of 10 contexts came back **bit-identical**; the 4 that differ shift the mean by **0.0079 m**.

**So ~0.008 m is the run-to-run noise on this metric.** That calibrates everything above: the K=2 gap (0.23–0.27 m) is ~30× the noise; the K=20-vs-K=100 gap (0.043–0.051 m) is ~6× the noise but does not survive a permutation test at n=10, because it lives in 2 contexts (ctx 0 and 7) out of 10.

### Reading it

**K=2 is broken, and not in the way the mean suggests.** Its median (0.278 m) is bad but its *tail* is the story: on **4 of 10 contexts it pushes the box further from the target than where it started**, worst case 0.903 m from a 0.474 m start — it nearly doubled the error. sd = 0.301, over 4× K=20's. Two Euler steps do not integrate this field; the sampler produces plans that actively work against the task on 40 % of contexts. K=20 does this on **0 of 10**, max 0.262 m.

**K=20 → K=100 buys nothing measurable, and K=20 is nominally ahead.** p = 0.37 / 0.46, and the sign split is a dead 5/5. K=100's own worst case (0.4618 m on ctx 0) is *worse* than K=20's worst (0.2617 m), and it has one worse-than-nothing rollout where K=20 has none. There is no reading of this table on which 80 extra network evaluations pay for themselves.

**Cost is exactly linear in K.** Fitting `ms = a + b·K` to K=2 and K=100 gives **b = 8.98 ms/NFE, a = 10.4 ms**, which predicts K=20 at 190.0 ms against 190.5 measured — a 0.5 ms error. The sampler is ~99 % network forwards; there is no per-step overhead to optimise away.

| K | ms/replan | vs K=20 | quality |
|---|---|---|---|
| 2 | 28.4 | **0.15×** | broken — 4/10 worse than doing nothing |
| **20** | **190.5** | **1.0×** | **best measured; 0/10 failures** |
| 100 | ~908 | **4.8×** | indistinguishable from K=20 (p ≈ 0.4) |

**K=20 is the operating point.** It is the smallest K at which the tail catastrophes disappear, and everything above it is paid-for noise. This is consistent with the h-coverage mechanism argued in the K=100 DA — the sampler queries `u(x, t, h=1/K)`, distance from the trained `h→0` atom scales as `1/K`, so ~91 % of the closure is already bought by K=20 — but note the mechanism predicts a *smooth* saturation, whereas what the data actually shows is a **threshold**: below some K the Euler integration fails outright on a subset of contexts.

**Caveats.** n = 10 contexts, single seed (6), one task. No K=5 or K=10 cell exists, so the threshold is bracketed only as 2 < K* ≤ 20 — if the 4.8× saving from K=100 matters, a K=5/K=10 sweep is cheap (~15 min/cell unguided) and would locate it. **There is no K=1 cell for visual aligning at all**; the K=1 arms in this corpus (`Emf_K1_mpc4_pid_stopgo_T0.5`, and the α-Flow avoiding runs) are different tasks and different checkpoints, and cannot be compared here.

## 3.2 T=0.2 is free

FM, K=20, same checkpoint, T=0.2 vs T=0.5 paired (n = 10):

| variant | dist @T0.2 | dist @T0.5 | W/L/T | p | ms @T0.2 | ms @T0.5 |
|---|---|---|---|---|---|---|
| `dpcc-r` | 0.4373 | 0.4371 | 3/3/4 | 1.000 | **439.6** | 1065.2 |
| `dpcc-c` | 0.4081 | 0.4589 | 3/2/5 | 1.000 | **481.4** | 1066.6 |
| `dpcc-t` | 0.4042 | 0.3913 | 3/4/3 | 1.000 | **471.7** | 1085.4 |

**Identical quality, 2.3× cheaper.** Cutting the projection budget from 10 calls to 4 costs nothing measurable and removes 60 % of the replan latency. Consistent with the K=100 T=0.1 finding — the late-τ solves are the ones that matter, the early ones are expensive and near-useless.

## 3.3 MeanFlow beats Flow Matching at matched everything — the strongest result here

Same VisualUNet backbone, same 26.4 M/params class, same K=20, same T=0.2, same seed, same 10 contexts, same Euler sampler. The **only** difference is the trained velocity field.

| variant | mf dist | fm dist | W/L/T | p |
|---|---|---|---|---|
| `diffuser` | **0.0933** | 0.3266 | 9/1/0 | **0.022** |
| `post_processing` | **0.2167** | 0.4529 | 8/1/1 | **0.039** |
| `dpcc-r` | **0.1553** | 0.4373 | 9/0/1 | **0.0039** |
| `dpcc-c` | **0.2985** | 0.4081 | 7/3/0 | 0.344 |
| `dpcc-t` | **0.2317** | 0.4042 | 7/3/0 | 0.344 |
| **pooled (6 variants)** | **0.1875** | 0.4085 | **49 / 9 / 2** | **9.0 × 10⁻⁸** |

**54 % lower final distance, p ≈ 10⁻⁷.** This is an architecture-matched result — the backbone is identical, so the win is attributable to the MeanFlow objective and nothing else. It is the cleanest claim in the corpus.

## 3.4 The "untouched box" count — the number to lead with

`untouched` = the box was never moved at all (`final == init`). It cuts through the mean-distance noise, because a mean over untouched rollouts just reports the initial distance back to you.

| model | arm | n | **untouched** | progress |
|---|---|---|---|---|
| **mf K=20 T=0.2** | unguided | 10 | **0 / 10** | **0.790** |
| **mf K=20 T=0.2** | `dpcc-r` | 10 | **1 / 10** | **0.647** |
| **mf K=20 T=0.2** | `hardflow_sls-r` | 10 | **1 / 10** | **0.636** |
| fm K=20 T=0.2 | unguided | 10 | 4 / 10 | 0.260 |
| fm K=20 T=0.2 | `dpcc-r` | 10 | 6 / 10 | 0.035 |
| **DPCC-diffusion (`visual_aligning_dpcc`, K=20 T=0.5)** | unguided | 33 | **23 / 33** | −0.111 |
| **DPCC-diffusion** | `dpcc-r` | 33 | **31 / 33** | **0.017** |
| DPCC-diffusion | `dpcc-t` | 33 | 32 / 33 | −0.078 |
| DPCC-diffusion | `post_processing` | 33 | 31 / 33 | 0.000 |

MeanFlow moves the box on essentially every rollout. FM moves it on about half. **The visual DPCC-diffusion baseline moves it on 2 of 33.**

⚠ **Do not put the baseline row in a paper without checking the checkpoint first.** 31/33 untouched and progress ≈ 0.017 is not "a weaker baseline", it is "a policy that does nothing". That is a plausible genuine result for visual aligning, but it is equally consistent with an undertrained or mismatched checkpoint, and the run also has only **3 contexts overlapping** the flagship's 10, so the paired test is powerless (3/0, p = 0.25 on every metric). Treat §3.4's baseline rows as **unverified** until the visual DPCC checkpoint is re-confirmed. The mf-vs-fm result in §3.3 does not depend on it.

## 3.5 Cost breakdown at K=20, T=0.2

Using the unguided arm as a direct sampler measurement and subtracting:

| engine / arm | ms/replan | projector share | per NLP call (4 calls) |
|---|---|---|---|
| mf unguided | 190.5 | — | — |
| mf `post_processing` | 215.7 | 25.3 | 6.3 |
| mf `dpcc-r` | 346.7 | 156.3 | 39.1 |
| mf `hardflow_sls-r` | 297.7 | **107.3** | **26.8** |
| mf `dpcc-c` | 349.1 | 158.7 | 39.7 |
| mf `hardflow_sls-c` | 340.9 | 150.4 | 37.6 |
| mf `dpcc-t` | 323.6 | 133.2 | 33.3 |
| mf `hardflow_sls-t` | 341.9 | 151.4 | 37.9 |
| fm unguided | 295.0 | — | — |
| fm `dpcc-r` | 439.6 | 144.7 | 36.2 |

Two things worth noting. **The sampler is now the dominant cost again** — 190 ms of a 347 ms replan is network forwards, versus K=100/T=0.5 where 94 % was SLSQP. And **HardFlow's per-solve cost is now comparable to DPCC's, not 1.8–2.2× it** (26.8–37.9 ms vs 33.3–39.7 ms) — the B4_PARITY fan match is holding, and at 4 calls the in-loop arm is not paying a penalty.

Also note **fm's unguided sampler is 1.55× slower than mf's** (295.0 vs 190.5 ms) at identical K and backbone. That is unexpected and unexplained — same NFE, same UNet width, and the two-time engine should if anything be heavier. Flag for a follow-up; it does not affect any quality claim.

---

# Part 4 — BUG: arm C is unreachable for the FM engine

Job 25248 died at item 17/38, the first `hardflow_new-r` cell:

```
File "mix_visual_aligning_test/eval_mix_visual_aligning.py", line 2437, in predict
    _hf_cond = encode_visual_cond(self.model, cond)
File "mix_visual_aligning/sampling/hardflow_projection.py", line 938, in encode_visual_cond
    'visual_latent': model._encode_once(bp_imgs, inhand_imgs)}
AttributeError: 'VisualFlowMatching' object has no attribute '_encode_once'
```

**Root cause.** `_encode_once` is a **two-time-engine method**. It is defined in `mix_visual_aligning/models/visual_mf_diffusion.py:43` and `visual_af_diffusion.py:42`, and its docstring says why it exists: *"Downstream this is a captured CONSTANT inside `_p_losses_meanflow`'s JVP closure, so its forward-mode tangent is zero by construction."* It is a MeanFlow/α-Flow JVP requirement. **`VisualFlowMatching` never needed it and never had it** — it passes the raw tuple through as `'visual': (bp_imgs, inhand_imgs, obs_seq)` and lets the backbone encode (`visual_fm_diffusion.py:93-101`).

`encode_visual_cond` was written against the two-time convention only. Its own docstring claims it *"Mirrors `VisualMeanFlow.forward` / `VisualAlphaFlow.forward` / `VisualFlowMatching.forward` exactly"* — **that third claim is false**, and the code has been carrying it since arm C was added.

**Why it never showed up before:** every prior arm-C run on this task was `mf` or `af`. This flagship is the first time arm C was pointed at `fm`.

**Scope.** `mf` and `af` are unaffected — 25247 proves it. Only `engine=fm` + `HFFM_VARIANTS` hits this. Nothing already in the corpus is invalidated.

**Fix — APPLIED 2026-09-01** (one branch, no math touched), at `mix_visual_aligning/sampling/hardflow_projection.py:936-938`. Changelog: `logs_in_develop/Gen14/Fix_11/CHANGELOG_20260901_encode_visual_cond_single_time_engine.md`.

```python
    if 0 in cond and isinstance(cond[0], tuple):
        bp_imgs, inhand_imgs, obs_seq = cond[0]
        if not hasattr(model, '_encode_once'):
            # Single-time engines (VisualFlowMatching) have no `_encode_once` —
            # that method exists only to zero the JVP tangent in the two-time
            # `_p_losses_meanflow` closure. They carry raw images under 'visual'
            # and encode inside the backbone; mirror their own forward() exactly.
            return {0: obs_seq[:, -1], 'visual': (bp_imgs, inhand_imgs, obs_seq)}
        return {0: obs_seq[:, -1],
                'visual_latent': model._encode_once(bp_imgs, inhand_imgs)}
```

This is structurally safe: `_VISUAL_COND_KEYS = frozenset({'visual_latent', 'visual'})` (line 884) already accepts `'visual'`, so the sampler's own guard at line 1112 passes, and the returned dict is byte-for-byte what `VisualFlowMatching.forward` builds at lines 98-101. The docstring's false third clause was corrected in the same edit. `hardflow_projection.py` is in none of G0's ledgers (`GRAFTED_DIFF` / `GRAFTED` / COPIED), so the edit carries no gate risk. Syntax checked; **not executed — no Python env here, needs the cluster run to confirm.**

**Cost to re-run:** 25248 needs a full resubmit, ~4 h based on 25247's wall time for 38 items.

---

# Part 5 — Also in this drop (secondary, different generation)

Not part of the flagship; noted so nothing is lost.

- **25251** — Gen3v7 α-Flow training, `avoiding-d3il`, seed 7, `AF_ALPHA_CLAMP=0.05` (ten times the shipped 0.005). 4 h 17 m, completed. ⚠ `[ DISK ] /dev/md2p1 7.0T 7.0T 27G 100% /data` — **the cluster data volume is full at 27 GB free.** This will start dropping checkpoints. Highest-priority operational item in this drop.
- **25253** — α-Flow eval, seeds 7-10, **failed**: `torch.load` on a missing checkpoint for seed 7 under the *un*-clamped path (`…_rf0.5/7`, no `_ac0.05` suffix). Wrong `AF_ALPHA_CLAMP` for the checkpoint that exists.
- **25254** — same eval with `AF_ALPHA_CLAMP=0.05`, seed 7 only. Completed, 8 min. All six arm-C variants correctly **`[hardflow][BLOCKED] … K=2 A=0.5 -> n_genuine=0 [DEGENERATE]`**. The guard from `a86f1f30` is working as designed in the field.

---

# Provenance

- **Jobs:** 25247 (mf, ✅ 38/38), 25248 (fm, ❌ 17/38). Both `git rev 938641c`, node i6-gpu-1, started 2026-08-31 14:32/14:33 UTC.
- **Aggregation:** `temp/3008/batch_va2_20260901_093100/` (built 2026-09-01 09:31 UTC).
- **Rows used:** 380 mf + 160 fm at `H8_K20_Meuler_T0.2`; cross-K comparisons draw on 3 380 rows at `K2_Meuler_T0.5`, 184 at `K100_Meuler_T0.1`, 41 at `K100_Meuler_T0.5`; baseline 277 rows at `visual_aligning_dpcc / H8_K20_T0.5`.
- **Stats:** exact two-sided sign test and exact McNemar, pure stdlib. No SciPy/numpy in this container.
- ⚠ **Provenance gap (carried over):** `config/visual_aligning_eval.yaml` at HEAD says `n_contexts: 3`, but these runs delivered 10 contexts. The cluster's copy of that yaml is edited outside git, so the committed file is **not** an accurate record of what ran. The item counts and `[ eval ] >>> item i/38` lines in the raw logs are the authoritative record.
- ⚠ Single seed (6). Every p-value above is within-seed. Multi-seed replication is the outstanding requirement for Part 2.
- ⚠ The DPCC-diffusion baseline rows in §3.4 are **unverified** — see the warning there.

## Recommended next actions

1. **Fix `encode_visual_cond`** (Part 4) and resubmit 25248 — that recovers the fm arm-C ablation. ~4 h.
2. **More seeds on the mf flagship.** Part 2 is direction-without-power; at 3–5 seeds the 4/0 and 3/0 constraint splits would become resolvable. This is the single highest-value run to queue.
3. **Free disk on `/data`** — 27 GB left (Part 5).
4. **Re-verify the visual DPCC-diffusion checkpoint** before it is quoted as a baseline anywhere.
5. **Retire K=100 as an operating point.** §3.1 shows K=20 matches it at 4.8× lower sampler cost (p ≈ 0.4, dead 5/5 split); keep the K=100 arms only as evidence for the h-coverage mechanism.
6. **Run a K=5 / K=10 unguided sweep** (~15 min/cell) to locate the threshold, currently bracketed only as 2 < K* ≤ 20. If it sits at 5, that is another 4× off the sampler.
7. **Always split K comparisons by checkpoint.** The corpus holds four MeanFlow checkpoints at K=2 (`Bdit`, `Bmf_dit`, `filmv1`, `filmv2`); a key of `(engine, K/T, geo, variant)` silently mixes them — that is what corrupted the first version of §3.1.
