# PLAN 2026-09-04 — Make α-Flow beat MeanFlow on the Visual-Aligning U-Net

*Rebuilt 2026-09-04 on the Gen3v7 template
([`DA_20260901_AF_UNet_alpha_clamp_T1_negative.md`](../Gen3v7_AlphaFlow/DA/DA_20260901_AF_UNet_alpha_clamp_T1_negative.md) §0)
and the Gen14 funnel
([`Report_20260829_VA_funnel`](../../Data_Analysis/DA_Result_Curated_MD/Report_20260829_VA_funnel/README.md)).
Supersedes the K-sweep draft of the same name — that plan is withdrawn.*

---

## 🎯 The objective

> ## **Make α-Flow beat MeanFlow on the Visual-Aligning U-Net. Nothing else is in scope.**
>
> **Success criterion — `diffuser` arm, seed 6, the MF flagship setup (K = 20, T = 0.2), paired
> on the same contexts:**
>
> ### **ratio ≤ 0.267 mean / 0.178 median × start  AND  0-viol ≥ 0.150**
>
> Those are **MeanFlow K20 T0.2's own numbers** (§1.2). `fm` and `diffusion` are context, not targets.

**Why `diffuser` first, and only `diffuser`.** It is the unprojected rollout — the raw network output
with no MPC, no DPCC, no HardFlow. It is the only thing that speaks to *"is AF's U-Net better than
MF's U-Net"*, and it is Stage 1 of the funnel: **an arm that fails Stage 1 is never ranked on Stage 2.**
The projected comparison (HardFlow-SLSQP at T = 0.2) happens **only if Stage 1 is won.**

**Hard constraints, carried from Gen3v7 §5:**

> 🔒 **The U-Net does not change. The parameter count does not change.**
> 🔒 **Everything that is not an α-Flow knob must match MeanFlow's flagship exactly.**
> 🔒 **No K sweep.** K = 20 is the flagship and the only K used.

---

## 1. The setup that must match

### 1.1 MeanFlow's flagship — the thing to beat

From [`DA_20260901_Gen14_flagship_K20_T0.2_dpcc_vs_hardflow.md`](DA_20260901_Gen14_flagship_K20_T0.2_dpcc_vs_hardflow.md),
job **25247**, `38/38` items, `Job completed successfully`:

| | value |
|---|---|
| engine | `mf` — MeanFlow (Gen3v6) in Gen14's `VisualUNetTwoTime` |
| bone | `unet`, FiLM **v1**, `freq_dim = 32`, **26.4 M** params |
| **K** | **20** |
| **T** | **0.2** |
| projector | DPCC **and** HardFlow-SLSQP (arm C) |
| seed | 6 |
| horizon / vision / bs | H8 / `VTrue` / 64 |
| `t_schedule` | `logit_normal`, `p_mean −0.4`, `p_std 1.0` |
| U9 vision knobs | **all default** — `vis_pretrained=False`, `vis_lr_scale=1.0`, `vis_cond=token` |

**α-Flow must be run at exactly this**, differing only in the α knobs. It already does: the
`AFAFend0p*` trees are `unet` + `filmv1` + `logit_normal` + all-U9-default, same seed, same bone,
same 26.4 M. **The only thing never matched is K — af has only ever been evaluated at K = 2 and
K = 100.**

### 1.2 The Stage-1 board today

`diffuser` arm, `split=train`, **ratio = final/init × start** (the funnel's lead metric — lower is
better; the d3il baseline scores **1.000×**, a no-op).

| arm | K | n | mean × | **median ×** | 0-viol | sat | ms |
|---|---|---|---|---|---|---|---|
| **`mf` K20 T0.2 — 🎯 THE TARGET** | 20 | 20 | **0.267** | **0.178** | 0.150 | 0.884 | 181.5 |
| `mf` K100 T0.1 | 100 | 10 | 0.321 | 0.202 | 0.100 | 0.907 | 924.6 |
| `mf` K10 T0.4 | 10 | 20 | 0.405 | 0.271 | 0.100 | 0.864 | 94.4 |
| `mf` K100 T0.5 | 100 | 30 | 0.491 | 0.277 | 0.200 | 0.804 | 892.9 |
| `af` α_end=0 @best *(α dead)* | 2 | 60 | 0.584 | 0.370 | 0.200 | 0.795 | 25.0 |
| `diffusion` K100 | 100 | 30 | 0.476 | 0.409 | 0.267 | 0.717 | 1526.6 |
| **`mf` K2** | 2 | 60 | 0.786 | 0.517 | 0.550¹ | 0.803 | 25.8 |
| `af` α=0.05 const @best | 2 | 20 | 0.711 | 0.595 | 0.050 | 0.707 | 26.4 |
| `af` α_end=0 @best | 100 | 30 | 0.809 | 0.689 | 0.200 | 0.718 | 902.0 |
| **`af` α_end=0.05 @latest** | 2 | 20 | **0.707** | 0.919 | **0.750** | 0.905 | 26.0 |
| **`af` α_end=0.2 @latest** | 2 | 20 | 0.772 | 0.983 | 0.100 | 0.627 | 26.6 |
| *d3il baseline (funnel)* | — | — | — | *1.000* | — | — | — |

¹ 0.550 on the 20 contexts paired against the af arms; 0.350 over all 60 of its own rows.

---

## 2. Gate 0 — is there buffer space at all? **Zero compute. Already answered.**

The medians look like a massacre: the two α-ON arms sit at **0.919× and 0.983×**, i.e. d3il-baseline
no-op territory, against the flagship's 0.178×. **But the medians are the wrong read**, and the
paired test says so.

### 2.1 The paired test, `diffuser` only, matched K = 2, same contexts

Exact two-sided sign test on the ratio; exact McNemar on 0-viol. n = 20 pairs (10 contexts × 2 geos).

| arm vs `mf` K2 | mean × A/mf | median × A/mf | A<mf / A>mf | **p (dist)** | 0-viol A/mf | **p (0-viol)** |
|---|---|---|---|---|---|---|
| **`af` α_end=0.05 @latest** | **0.707** / 0.761 | 0.919 / 0.700 | 7 / 12 | **0.359 — tie** | **0.750 / 0.550** | 0.289 — tie |
| `af` α_end=0.2 @latest | 0.772 / 0.761 | 0.983 / 0.700 | 9 / 9 | **1.000 — tie** | 0.100 / 0.550 | **0.004 — WORSE** |
| `af` α_end=0 @best *(α dead)* | 0.584 / 0.786 | 0.370 / 0.517 | 31 / 25 | 0.504 — tie | 0.200 / 0.350 | **0.035 — worse** |
| `af` α=0.05 const @best | 0.711 / 0.761 | 0.595 / 0.700 | 8 / 11 | 0.648 — tie | 0.050 / 0.550 | **0.002 — WORSE** |

### 2.2 Verdict: **not zero. Thin, and it is `α_end=0.05` — not 0.2.**

- **On distance, no α-Flow arm is significantly different from MeanFlow at matched K.** Every row is
  a tie. The median gap is a **bimodal field**, not a uniform failure: for `α_end=0.05`,
  mean 0.707 < median 0.919 means a minority of rollouts engage the box *well* while most do not
  move it. MeanFlow is the reverse (mean 0.761 > median 0.700 — it engages usually, fails badly
  sometimes). Two different failure shapes at the same average.
- **On 0-viol, exactly one af arm is not significantly worse than `mf` — and it is numerically
  ahead: `α_end=0.05` at 0.750 vs 0.550.** Same direction as the 320-rollout aggregate
  (`p = 1.6e-23` in the DA), just under-powered here at n = 20.
- **`α_end=0.2` is dead on this metric.** Tied on distance, `p = 0.004` worse on 0-viol. The
  avoiding winner is *not* the Gen14 candidate. **This overturns the α=0.2 recommendation in the
  previous DA, which was made on progress aggregated across projected variants, not on the raw field.**

**So the live hypothesis is exactly one arm, at one untried setting:**

> **`α_end=0.05 @latest` at the flagship K = 20 Pareto-dominates `mf` K20 T0.2 on the raw field —
> equal or better distance, better 0-viol.**

### 2.3 The counter-signal, stated up front

**`af`'s raw field gets *worse* with more NFE; `mf`'s gets better.**

| | K2 | K100 | direction |
|---|---|---|---|
| `af` α_end=0 @best | 0.370× | 0.689× | **1.9× worse** |
| `mf` | 0.517× | 0.277× | 1.9× better |

That is the single strongest argument that Gate 1 will fail. It is measured on the α-dead arm at
`best`, so it is not a clean read on `α_end=0.05 @latest` — but it points the wrong way and it must
be in the record before the run, not after. **If `af` follows its own K-trend, K = 20 lands worse
than K = 2 and the line dies at Gate 1.**

---

## 3. Gate 1 — the one submit. 2 eval jobs, **zero training.**

Both checkpoints are on disk (`state_100000.pt`, α verified live). K is inference-only. Nothing is
retrained, nothing existing is overwritten.

```bash
# ── THE SHOT: alpha-Flow at MeanFlow's flagship setup, K=20, T=0.2 ──────────────────
# arm A — alpha_end 0.05: the ONLY arm that survived Gate 0
MIX_AF_ALPHA_END=0.05 MIX_EPOCH=latest MIX_PROJ_T=0.2 FMPCC_RUN_MSG=afon005_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh \
  af 6 all 20

# arm B — alpha_end 0.2: the avoiding winner. Dead on Gate 0, but it is the published
# recipe and one job settles whether K=20 revives it.
MIX_AF_ALPHA_END=0.2 MIX_EPOCH=latest MIX_PROJ_T=0.2 FMPCC_RUN_MSG=afon02_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh \
  af 6 all 20
```

**`diffuser` is variant 1 of 19** (`config/visual_aligning_eval.yaml:projection_variants[0]`, and
confirmed first in job 25373's log). **Stage 1 lands in the first few minutes of each job even if it
later hits the 24 h wall** — which is the real risk at K = 20 × T = 0.2 (job 25248, the `fm` twin,
died inside item 17). Read Stage 1 the moment `variant=diffuser_train_set` prints its summary; do
not wait for the job.

### Read exactly this, and nothing else

```
--- aligning-d3il-visual [seen training set] diffuser_train_set seed=6 ---
```
then from `results_train_set/combined_5/diffuser_train_set/`:
`context_final_xy_dist / context_init_xy_dist` (mean **and** median) and `collision_free_completed`,
paired per context against `mf` K20 T0.2.

### The decision, fixed now

| outcome on `diffuser`, paired vs `mf` K20 T0.2 | verdict | next |
|---|---|---|
| ratio ≤ 0.267 mean **and** 0-viol ≥ 0.150 | 🏆 **STAGE 1 WON** | go to Stage 2 — the projected comparison at T = 0.2, DPCC **and** HardFlow-SLSQP, same job's later items |
| ratio tied (`p > 0.05`) **and** 0-viol > 0.150 | 🟡 **BUFFER** | one power buy: seeds 7–8 on `α_end=0.05` only (§4) |
| ratio significantly worse, **or** > 0.517× (`mf`'s *weakest* K) | ⛔ **KILL** | stop. Write the negative (§5). No further AF spend on this scene |

---

## 4. Gate 2 — only if Gate 1 says BUFFER. One thing, then stop.

Per Gen3v7 §5 the sanctioned search space with the U-Net frozen is **the α schedule and
`af_clamp_utgt`** — and it is nearly exhausted:

| knob | status |
|---|---|
| `af_alpha_end` | ✅ **done** — 0.05 and 0.2 both trained and evaluated. This is the knob that won on avoiding. |
| `af_alpha_clamp` | ⚠️ **already tested and negative** on avoiding (`DA_20260901_..._T1_negative`). Wired here as `MIX_AF_ALPHA_CLAMP`, but the prior is bad. |
| `af_alpha_gamma` | 🟢 **untried.** `MIX_AF_ALPHA_GAMMA=5` — γ=25 compresses the whole FM→MeanFlow homotopy into steps 40 k–60 k, 20 % of the budget. Wired, no code, path key `_AFend0p05-g5`. |
| `af_ratio_fm` | ⛔ Gen3v7 §9.4: *"theory says it makes things worse until the probe is fixed."* Do not spend. |
| `af_clamp_utgt` | ⛔ no env knob — needs a config edit. Not without a go-ahead. |
| U-Net rewrite (§10.4 / §10.5) | ⛔ **withdrawn** by the frozen-parameter constraint. |

**If BUFFER, buy power before you buy knobs.** The Gate-0 signal is a real effect at n = 20 that just
misses significance; two more seeds is a better use of 8 GPU-h than another α value.

```bash
# power the ONE surviving arm — seeds 7 and 8, flagship setup
for S in 7 8; do
  MIX_AF_ALPHA_END=0.05 MIX_EPOCH=latest MIX_PROJ_T=0.2 FMPCC_RUN_MSG=afon005_s${S} \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
    af $S 20
done
# and the matching mf seeds, or the comparison is unpaired
for S in 7 8; do
  MIX_PROJ_T=0.2 ./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf $S 20
done
```

*(pipeline signature verified: `<engine> <seeds> <flow_steps>`, `mix_visual_aligning_pipeline.sh:79-86`,
forwarded to the eval at line 300 — so the trailing `20` is K = 20.)*

---

## 5. The kill — what we write if Gate 1 fails

No hedging, no third attempt. The negative is publishable and it is already well-supported:

> **α-Flow's few-step advantage is real on a 4.0 M state-space U-Net (`avoiding-d3il`: Pareto-dominant
> over MeanFlow at K = 1, 33× cheaper than DPCC K20) and does not survive transfer to a 26.4 M visual
> U-Net with a randomly-initialised encoder. At matched backbone, seed, contexts and K, the raw
> generative field of α-Flow is statistically indistinguishable from MeanFlow's on distance, and its
> constraint behaviour is either equal (α→0.05) or significantly worse (α→0.2). The α floor that wins
> on avoiding forfeits the full-weight JVP repair phase that Gen3v7 §7.4 predicts the U-Net needs
> most — and on the visual bone that cost is not repaid.**

Supporting evidence already banked: the α-mechanism gates all green
([`DA_20260904`](DA_20260904_Gen14_U12_alpha_floor_and_latest_checkpoint.md) Part 1), the paired
Gate-0 ties above, the inverted K-trend (§2.3), and `S&C = 0.000` for every arm on this scene
including the pinned `diffusion` K20 aw10 target.

**Where the AF budget goes instead:** `s_curve` (Gen15), which has a live `diffusion` target arm and
can rank a result — per
[`RUNSTATUS_20260904`](../Gen15/U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md).

---

## 6. Free pre-checks — do these while Gate 1 queues

Both cost zero compute and either can pre-empt the result.

1. **`h_mse_b3` overlay (Gen3v7 §9.2).** At low NFE the sampler evaluates `u` at large `h`, so
   `h_mse_b3` (h ≥ 0.6) is the training-time proxy for few-step quality — **already logged**. We have
   the α-Flow endpoints: `val/h_mse_b3` = **0.0033** (α_end=0.05, job 25376) and **0.0747**
   (α_end=0.2, job 25372). **Pull `mf`'s from W&B project `FM-PCC-visual-aligning-gen14` and compare.**
   If `mf` is an order of magnitude below 0.0033, Gate 1 is already lost on the training side.
2. **The 28.8 k coincidence check (Gen3v7 §10.1).** Steps 0 → ~28.8 k hold α = 1.0, where α-Flow is
   *bitwise* plain flow matching. Overlay `raw_mse_u` / `per_dim_rms_u` for af vs mf on the step axis.
   **If af is already behind at 28.8 k, the problem is upstream of α-Flow entirely** — data,
   normalisation, LR, EMA — and none of this plan applies.

---

## 7. One-line summary

**Gate 0 is not a zero.** Every α-Flow arm ties MeanFlow on raw distance at matched K, and
**`α_end=0.05` leads on zero-violation (0.750 vs 0.550) without being significantly worse anywhere.**
The one thing never tried is that arm at MeanFlow's own flagship budget, K = 20 — **two eval jobs, no
training.** The K-trend argues it will fail. Run it, read `diffuser` in the first five minutes, and
take the answer either way.
