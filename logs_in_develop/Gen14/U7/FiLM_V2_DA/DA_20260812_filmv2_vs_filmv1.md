# DA — FiLM V2 (true FiLM) vs FiLM V1 (fake FiLM): the visual conditioning route

**Date:** 2026-08-12
**Question:** V2 replaces the "fake FiLM" conditioning route — visual latent concatenated
into the time embedding, additive bias only — with **true FiLM**: a per-block, per-channel
`γ` scale and `β` shift driven by the visual latent. Architecturally it is the right thing.
**Does it actually help?**
**Answer, in one line:** **No — V2 is decisively *worse*, and one of the two V2 runs is not
even a valid test.** On the fair arm (MeanFlow, 896 paired rollouts) V2 loses on **every
single metric measured**: constraint satisfaction **−11.9 pp**, violated steps **+47.6**,
collision-free **385 → 265**, goal successes **25 → 12**, and it is **7.6 ms slower**
(all *p* < 0.05 after Holm, most *p* < 0.001). The AlphaFlow V2 run **collapsed during
training at ~72k steps** and its eval checkpoint is post-collapse, so it says nothing about
FiLM — it must be re-run before the AF arm counts.

**Data:** `temp/1208/Gen14_Mix_FilmV2/batch_va2_20260812_114647/` — DA_VA_v2 batch,
15 candidates, 282 units, **5634 rollouts**, `logs/aligning-d3il-visual/plans` on i6-gpu-1,
**seed 6**, train split. Plus `mf_losses.pkl` / `af_losses.pkl` (the two V2 training runs,
slurm 24454 / 24457) and `2026-08-09/` job logs from the same folder.
**Script:** `da_20260812_filmv2.py` (stdlib only — this container has no project env).
Full printed output: `da_20260812_filmv2_output.txt`. Every number below is from that run.
**Predecessor:** `DA_20260809_gen14_horizontal_4engines.md` — same harness, same seed, the
*engine* and *projector* cuts. This is the **conditioning-route** cut, and it is the first
one where the answer is a clean negative.

---

## The design — why this is a strong test despite one seed

For each mix engine, the V1 and V2 checkpoints were evaluated **on the same contexts, under
the same projector grid, by the same harness**. The pairing was verified rollout-by-rollout
on the context fields (`box_init_xy`, `box_angle`, `target_xy`):

> **960 matched rollouts per engine, 0 context mismatches.** 16 projector variants × 2 geos
> × 30 contexts. After dropping the 64 frozen rollouts: **n = 896 paired rollouts per
> engine**, **n = 1792** pooled.

So the FiLM axis is a **within-context paired design** — the same initial states, the same
projector, the only thing changing is the conditioning route. That is a lot of power, and it
is why the effects below survive Holm correction so comfortably.

The 180 unpaired V1 rollouts are the three `hardflow_new-{c,r,t}` variants, which V2 did not
run. They are excluded from every paired test rather than pooled.

**Method.** Paired sign-flip permutation test (20 000 resamples) on the mean difference for
continuous metrics; exact McNemar (binomial on discordant pairs) for binary outcomes;
Holm-Bonferroni within each family (continuous and binary treated as separate families, per
contrast).

**The one confound, stated once.** There is **one checkpoint per (engine, FiLM version)**.
Formally "V2 is worse" means "*this* V2 checkpoint is worse than *this* V1 checkpoint". What
lifts it above a single-run anecdote is that **two independently trained engines were tested
and both moved the same way on 8 of 9 continuous metrics** — and the one that did not is
explained below (§5). It is still not a seed sweep, and §9 says what would settle it.

---

## 0. TL;DR

**Decisive:**

1. **FiLM V2 loses to V1 on every axis in the fair arm.** MeanFlow K=2, n = 896 paired:
   sat **0.866 → 0.747**, violated steps **53.5 → 101.1**, collision-free **385 → 265**
   (discordant 186/66), relaxed successes **68 → 19**, goal successes **25 → 12**, max
   physical tracking error **0.274 → 0.457 m**, and time **46.0 → 53.6 ms**. Nine of nine
   continuous metrics and four of four binary outcomes point at V1; all survive Holm. §3
2. **The damage is upstream of the projector.** In the least-projected variants
   (`diffuser`, `model_free`) — which read the generative model almost directly — MF sat
   drops **0.789 → 0.543** and **0.804 → 0.596**, and tracking error nearly doubles. **The
   plans themselves are worse.** The projector is not the problem, and cannot fix it. §4
3. **The AlphaFlow V2 run is invalid as a FiLM test.** `test_raw_mse` **3.21 @ 71k →
   9.71 @ 72k → 24.08 @ 75k**, never recovers (9–23 through 99k) even as LR anneals to
   5e-8; the α₀-anchor loss spikes to **2.31 @ 75k** against a 0.265 median. Normalized test
   loss goes **0.746 @ 70k → 0.994 @ 80k** — back to its value at initialization. The
   evaluated checkpoint is **post-collapse**, and the eval shows exactly that: a policy whose
   endpoint barely depends on the context (final-distance std **0.062 m** vs 0.23–0.25 m for
   every other arm) and whose plans are physically untrackable (**2.20 m** max tracking
   error). **Re-run before drawing any AF conclusion.** §5
4. **AF V2 posts the best constraint numbers in the whole batch, and they are worthless.**
   At `dpcc-c-dt4p0` it reaches sat **0.986** and **52/56** collision-free — better than any
   V1 cell — with **0 goal successes** and a raw plan 2.2 m off. More projector authority
   simply overwrites a broken plan. **Constraint metrics are only meaningful next to a
   non-zero success rate.** §6

**Supported, weaker:**

5. **The legacy `fm_visual` arm points the same way** (V2 worse on distance-to-goal,
   0.32 → 0.55 m, *p* < 0.001) but only **46 usable pairs** survive and the two runs differ in
   training length (900 vs 1000 steps/epoch) — **directional only**. §7
6. **No V2 arm gets near the DPCC target on success**, though the DPCC baseline is itself so
   weak (0 successes in 277 rollouts) that MF V2 still beats it on the constraint/time
   axes. §8

**What is NOT shown here:** that true FiLM is a bad idea. Two checkpoints, one seed, one task,
and one of them broken by an unrelated training instability. §9.

---

## 1. What V1 and V2 actually are

From `mix_visual_aligning/models/unet1d_twotime_film.py` and
`mix_visual_aligning/models/unet1d_temporal_film.py`:

| | route | per-block update |
|---|---|---|
| **V1** ("fake FiLM") | visual latent **concatenated into the time embedding**; `embed_dim` widened by `cond_dim` | `out = Conv(x) + time_mlp([ t(τ,h) ‖ cond ])` |
| **V2** ("true FiLM") | visual latent leaves the time path entirely, becomes a **per-channel scale/shift in every residual block** | `out = (1 + γ(cond)) · ( Conv(x) + time_mlp(t(τ,h)) ) + β(cond)` |

Two things worth recording, because they rule out the obvious suspects:

- **γ/β are zero-initialised** (`nn.init.zeros_` on both weight and bias of `film_proj[-1]`),
  so at step 0 the V2 block is numerically **identical** to no-FiLM. This is not a bad-init
  story.
- **V2 is JVP-safe by construction.** `cond` is a captured constant in
  `MeanFlowODE._p_losses_meanflow`'s forward-mode differentiation, so `γ`/`β` carry an
  identically-zero tangent and `d/dr[(1+γ)f + β] = (1+γ)·df/dr` — a per-channel rescaling of
  the same derivative V1 computes. The MeanFlow identity is unchanged. This is not a
  broken-gradient story either.

The `FiLMResidualTemporalBlock` is **the same class object** for the mf/af arms and the Gen7
fm arm, so "is mf's V2 the same FiLM as fm's V2?" is yes, by import.

---

## 2. What is comparable

All mix candidates: `seed 6`, train split, `if_vision=True`, `horizon=8`, `mpc_batch_size=4`,
`diffusion_timestep_threshold=0.5`, `steps1000_bs64`, `aw1`, `tslogit_normal`. The V1 and V2
checkpoints of an engine differ **only** in `film_mode`.

| contrast | V1 | V2 | K | paired n | verdict |
|---|---|---|---|---|---|
| **mix MeanFlow** | C13 | C14 | 2 | **896** | **the fair test** — §3 |
| **mix AlphaFlow** | C6 | C7 | 2 | **896** | **invalid** — V2 training collapsed, §5 |
| legacy `fm_visual` | C4 | C3 | 20 | 46 | confounded, directional only — §7 |
| mix `fm`, mix `diffusion` | C9–C11 | — | 20/100 | — | **no V2 arm exists** |
| DPCC baseline | — | C15 | 20 | — | V2 only; the Target row, §8 |

Batch-level inventory (unprojected pooling over all variants, for orientation only):

| cand | arm | FiLM | K | n | succ | coll-free | sat | t/replan |
|---|---|---|---|---|---|---|---|---|
| C13 | mix mf | **v1** | 2 | 1140 | **29** | **563** | **0.885** | 62.3 ms |
| C14 | mix mf | v2 | 2 | 960 | 12 | 329 | 0.764 | 50.0 ms |
| C6 | mix af | **v1** | 2 | 1140 | **13** | **511** | **0.866** | 62.8 ms |
| C7 | mix af | v2 | 2 | 960 | **0** | 249 | 0.738 | 58.5 ms |
| C4 | fm_visual | v1 | 20 | 348 | 0 | 73 | 0.829 | 920.9 ms |
| C3 | fm_visual | v2 | 20 | 82 | 0 | 7 | 0.573 | 2580.0 ms |
| C15 | **DPCC baseline** | v2 | 20 | 277 | 0 | 39 | 0.593 | 1044.7 ms |

---

## 3. The fair test — MeanFlow K=2, V1 vs V2 (n = 896 paired)

Every metric, pooled over 16 variants and both geos. Δ is V2 − V1; `p_holm` is Holm-adjusted
within the family.

| metric | V1 | V2 | Δ (V2−V1) | p_holm | |
|---|---|---|---|---|---|
| constraint sat rate | **0.8660** | 0.7472 | **−0.1188** | 0.0004 | *** V1 |
| violated steps | **53.51** | 101.10 | **+47.59** | 0.0004 | *** V1 |
| constraint margin (m) | **0.0875** | 0.0809 | −0.0066 | 0.0004 | *** V1 |
| max phys tracking err (m) | **0.2743** | 0.4565 | **+0.1822** | 0.0004 | *** V1 |
| final xy dist to target (m) | **0.2907** | 0.3540 | +0.0633 | 0.0004 | *** V1 |
| time per replan (ms) | **45.99** | 53.59 | +7.60 | 0.0004 | *** V1 |
| episode steps | **391.99** | 396.37 | +4.38 | 0.0154 | * V1 |
| max obstacle penetration (m) | **0.0066** | 0.0081 | +0.0015 | 0.0250 | * V1 |
| mean distance to goal (m) | **0.3473** | 0.3659 | +0.0186 | 0.0444 | * V1 |

| binary outcome | V1 | V2 | discordant (V1-only/V2-only) | p_holm | |
|---|---|---|---|---|---|
| collision-free completed | **385** | 265 | 186 / 66 | <0.0001 | *** V1 |
| zero-violation rollout | **385** | 265 | 186 / 66 | <0.0001 | *** V1 |
| goal success (relaxed) | **68** | 19 | 60 / 11 | <0.0001 | *** V1 |
| goal success | **25** | 12 | 20 / 7 | 0.0192 | * V1 |

**Nine of nine continuous metrics and four of four binary outcomes favour V1.** There is no
axis on which V2 wins, not even the cheap ones — V2 is also **slower** (+7.6 ms) and takes
**more** episode steps.

The result holds in both geometries separately: nominal `combined_5` (n = 480) sat
**0.834 → 0.704**, collision-free **141 → 52**; `combined_5-tightened` (n = 416) sat
**0.903 → 0.797**, collision-free **244 → 213**, goal successes **10 → 0**.

**Per-variant** (n = 56 each), the effect is not carried by a few cells — sat is lower under
V2 in **all 16 of 16** variants, significantly so in 11:

| variant | V1 sat | V2 sat | Δ | | variant | V1 sat | V2 sat | Δ |
|---|---|---|---|---|---|---|---|---|
| `bounds_free` | 0.945 | 0.880 | −0.066 ** | | `dpcc-r` | 0.916 | 0.890 | −0.026 ns |
| `diffuser` | 0.789 | 0.543 | **−0.246** *** | | `dpcc-t` | 0.911 | 0.877 | −0.034 ns |
| `dpcc-c` | 0.890 | 0.845 | −0.045 ns | | `geo_free` | 0.791 | 0.580 | **−0.211** *** |
| `dpcc-c-dt0p25` | 0.933 | 0.874 | −0.059 * | | `geo_free-bounds_free` | 0.763 | 0.544 | **−0.219** *** |
| `dpcc-c-dt0p5` | 0.910 | 0.860 | −0.050 ns | | `geo_free-model_free` | 0.802 | 0.600 | **−0.203** *** |
| `dpcc-c-dt2p0` | 0.916 | 0.865 | −0.050 ** | | `gradient` | 0.770 | 0.615 | −0.155 ** |
| `dpcc-c-dt4p0` | 0.970 | 0.898 | −0.072 *** | | `model_free` | 0.804 | 0.596 | **−0.208** *** |
| `post_processing` | 0.917 | 0.890 | −0.027 ns | | `model_free-bounds_free` | 0.830 | 0.600 | **−0.230** *** |

Note the pattern in that table: **the gap is smallest where the projector is strongest**
(`dpcc-*`, `post_processing`: −0.03 to −0.07) and **largest where it is weakest**
(`diffuser`, `model_free`, `geo_free*`: −0.20 to −0.25). That is the signature of a
model-side regression being partially masked by projection — which §4 confirms directly.

---

## 4. Where the damage is — upstream of the projector

The `diffuser` and `model_free` variants apply the least correction, so they read the
generative model almost directly:

| cand | variant | sat | max phys err | mean dist | final dist | coll-free /56 |
|---|---|---|---|---|---|---|
| C13 mf **v1** | `diffuser` | **0.789** | **0.495** | 0.408 | 0.339 | **17** |
| C14 mf v2 | `diffuser` | 0.543 | 0.798 | 0.390 | 0.377 | 4 |
| C13 mf **v1** | `model_free` | **0.804** | **0.422** | 0.361 | **0.251** | **14** |
| C14 mf v2 | `model_free` | 0.596 | 0.799 | 0.335 | 0.326 | 5 |
| C6 af **v1** | `model_free` | **0.749** | **0.498** | 0.354 | **0.264** | **9** |
| C7 af v2 | `model_free` | 0.493 | **2.200** | 0.386 | 0.451 | **0** |

**The V2 plans are worse before any projector touches them.** MF V2's unprojected
satisfaction is 21–25 pp below V1's and its tracking error is ~1.9× higher. This rules out
"V2 interacts badly with the projector" and places the regression squarely in the conditioned
U-Net.

---

## 5. The AlphaFlow V2 run collapsed — it is not a FiLM result

The AF paired contrast looks even more lopsided than MF (sat 0.843 → 0.719, tracking error
**0.257 → 1.196 m**, goal successes **12 → 0**, collision-free 334 → 185, all *p* ≤ 0.0005).
**Discard it.** The training curve says why:

| step | 60k | 65k | 70k | **71k** | **72k** | 74k | **75k** | 80k | 90k | 99k |
|---|---|---|---|---|---|---|---|---|---|---|
| AF V2 `test_raw_mse` | 3.3 | — | 3.2 | **3.21** | **9.71** | 14.69 | **24.08** | 11.77 | 9.4 | 11.40 |
| AF V2 norm test loss | 0.759 | 0.872 | **0.746** | — | — | — | — | **0.994** | 0.993 | 0.993 |
| AF V2 `train/a0_loss` | 0.237 | 0.097 | 0.186 | 0.102 | 0.278 | 0.99 | **2.31** | 0.342 | 0.416 | 0.343 |

- `test_raw_mse` **triples between 71k and 72k**, peaks at **24.08 @ 75k** (7.6× its 3.174
  minimum at 65k), and never returns — final 11.398, **3.59× the minimum**.
- Normalized test loss returns to **0.99**, i.e. **its value at initialization**.
- The α₀-anchor loss spikes to **2.31 @ 75k** against a **0.265** median. Note the order:
  the test error breaks at **72k** and the α₀ spike follows **2–3k steps later**, so the
  spike is better read as part of the same event than as its trigger. What it does pin down
  is *which* term destabilised — the AlphaFlow bootstrap target, a component FiLM V1 shares.
  MF V2, on the same FiLM head, shows no comparable a0 excursion (median 0.322, max 1.354,
  and that maximum is at step 0).
- LR was already annealing (7e-5 @ 60k → 5e-8 @ 99k) and did not rescue it.
- **The evaluated checkpoint is the 99k one — 28k steps past the collapse.**

The eval reproduces this exactly. Behavioural spread across the 56 `model_free` contexts:

| arm | final-distance mean | **std** | min | max |
|---|---|---|---|---|
| C13 mf v1 | 0.2505 | 0.2486 | 0.0115 | 0.8905 |
| C14 mf v2 | 0.3264 | 0.2434 | 0.0066 | 1.3620 |
| C6 af v1 | 0.2642 | 0.2305 | 0.0098 | 1.1389 |
| **C7 af v2** | 0.4511 | **0.0617** | **0.3064** | 0.5872 |

AF V2's endpoint is **nearly independent of the context** (std 0.062 vs 0.23–0.25 everywhere
else) and **never once comes within 0.31 m of the target** — hence 0 successes in 896
rollouts. That is a collapsed policy, not a conditioning ablation.

**Note that MF V2 did not collapse**: its curve is stable (raw MSE ~9.6, final 9.965 = 1.21×
its minimum; normalized loss descending monotonically to 0.947) and its behavioural spread
matches V1's. **The MF contrast in §3 is a fair test. The AF contrast is not.**

> **Gap:** no FiLM **V1** loss curves were exported with this batch, so "did V1 train more
> stably than V2?" cannot be answered here. See §9.

---

## 6. The trap — AF V2's constraint numbers are the best in the batch and mean nothing

| variant | V1 sat | V2 sat | V1 phys err | V2 phys err | V1 coll-free | V2 coll-free | V1 succ | V2 succ |
|---|---|---|---|---|---|---|---|---|
| `dpcc-c-dt0p25` | 0.891 | 0.589 | 0.105 | 1.689 | 25 | 3 | 1 | 0 |
| `dpcc-c-dt0p5` | 0.880 | 0.523 | 0.135 | 1.866 | 27 | 0 | 0 | 0 |
| `dpcc-c` | 0.860 | 0.799 | 0.123 | 0.636 | 23 | 20 | 0 | 0 |
| `dpcc-c-dt2p0` | 0.863 | **0.971** | 0.071 | 0.117 | 16 | **46** | 0 | 0 |
| `dpcc-c-dt4p0` | 0.882 | **0.986** | 0.053 | 0.050 | 18 | **52** | 0 | 0 |

At high `dt`, the collapsed AF V2 model posts **sat 0.986 and 52/56 collision-free — the best
constraint cell anywhere in this batch** — while producing **zero** goal successes. Raising
the threshold hands the projector enough authority to overwrite the broken plan entirely; what
is being measured at `dt4p0` is the projector, not the model.

**Rule this batch re-confirms:** a constraint metric without a non-zero success rate beside it
ranks projector authority, not policy quality. Any ranking that sorts on satisfaction alone
would have crowned the single most broken checkpoint in the batch.

---

## 7. Secondary — the legacy `fm_visual` arm (directional only)

C4 (v1, `steps900`) vs C3 (v2, `steps1000`), K=20. Only **46 usable pairs** survive matching
(269 V1 rollouts have no V2 partner; 18 context mismatches), and the two runs differ in
training length — **confounded, do not quote as evidence.**

| metric | V1 | V2 | Δ | p_holm |
|---|---|---|---|---|
| mean distance to goal (m) | **0.3225** | 0.5520 | +0.2294 | 0.0004 *** |
| final xy dist to target (m) | **0.2912** | 0.6164 | +0.3252 | 0.0004 *** |
| constraint sat rate | 0.7949 | 0.5624 | −0.2325 | 0.0672 ns (n=27) |
| violated steps | 82.0 | 175.0 | +93.0 | 0.0672 ns (n=27) |
| goal success | 0 | 0 | — | ns |

Same direction as the mix arms, on an independent codebase and an independent pair of training
runs. Consistent, not conclusive.

---

## 8. Target check — vs the DPCC baseline

Per the standing convention, the Target is the best row of the baseline DPCC candidate
(**C15**, `diffuser_visual_aligning` `VisualGaussianDiffusion` K=20, FiLM V2, 277 rollouts).
The baseline scores **0 goal successes in every one of its 15 variants**, so there is no
success axis to contest; the comparison reduces to constraints and cost.

| row | succ | coll-free | sat | dist | t/replan | n |
|---|---|---|---|---|---|---|
| **Target** — C15 `dpcc-c-dt4p0` | 0 | 2/6 | **0.950** | 0.382 | 752.1 ms | 6 |
| **Target** — C15 `bounds_free` | 0 | 1/6 | 0.790 | 0.382 | 1014.4 ms | 6 |
| C13 mf **v1** `dpcc-c-dt4p0` | 0 | 39/56 | **0.970** | 0.387 | **44.0 ms** | 56 |
| C13 mf **v1** `bounds_free` | **2** | 37/56 | 0.945 | **0.283** | **46.1 ms** | 56 |
| C14 mf v2 `dpcc-c-dt4p0` | 0 | 30/56 | 0.898 | 0.386 | **47.7 ms** | 56 |
| C14 mf v2 `bounds_free` | **2** | 23/56 | 0.879 | 0.354 | **50.8 ms** | 56 |

- **MF V1 `dpcc-c-dt4p0` Pareto-dominates the Target**: higher satisfaction (0.970 vs 0.950)
  at **17× less time**. This restates the predecessor DA's finding, on V1.
- **MF V2 still beats the Target** on `bounds_free` (2 successes vs 0, sat 0.879 vs 0.790,
  20× faster) — but by a visibly smaller margin than V1, and the baseline cells rest on
  **n = 6** rollouts.
- **No V2 arm improves on any Target axis relative to its own V1 counterpart.** V2 moves
  every arm *away* from the target, never toward it.

---

## 9. Limits, and what is genuinely noise

**Noise — do not read into these:**

- Absolute goal-success rates. 37 successes in 1792 V1 rollouts and 12 in 1792 V2 rollouts;
  the paired McNemar (20/7 discordant, *p* = 0.019) is real, but "2.5% vs 1.3%" is not a
  number to plan against.
- The **one** metric where V2 "wins": AF max obstacle penetration, 0.0080 → 0.0007
  (*p* < 0.001). This is an artifact of the collapse — a policy flailing 2.2 m off-plan
  rarely gets close enough to an obstacle to penetrate it. It is also the single metric where
  the two engines **disagree** in sign (MF: +0.0015, i.e. V2 worse). Discard.
- Every C15 baseline cell with n = 6.

**Real limits:**

1. **One seed (6), one checkpoint per cell.** The MF result is one training run vs one
   training run. It replicates directionally in AF and in the legacy fm arm, but neither is
   clean.
2. **One task** (`aligning-d3il-visual`), train split only.
3. **No V1 training curves** were exported, so the stability comparison that would settle §5
   cannot be made.
4. **Two of four mix engines have no V2 arm at all** (`fm`, `diffusion`).

---

## 10. What to run next

Ordered by how much each buys:

1. **Re-train AF FiLM V2** (or evaluate the **65–70k** checkpoint, where `test_raw_mse` was at
   its 3.174 minimum). Until then the AF arm is unusable, and the α₀ blow-up at 75k is worth
   chasing on its own — it may not be FiLM-specific.
2. **Export the V1 loss curves** alongside the V2 ones for every arm. Without them, "did V2
   train worse, or just end worse?" is unanswerable.
3. **Seeds 7–10 on the MF V1/V2 pair.** This is the cheapest path from "this checkpoint" to
   "this conditioning route" — K=2 MeanFlow runs at ~50 ms/replan, so the eval side is nearly
   free.
4. **Add the missing V2 arms** (`fm`, `diffusion`) so the ablation is 4×2 instead of 2×2.
5. **Run the V2 arms with `hardflow_new-{c,r,t}`**, the three variants missing from every V2
   candidate — they are the ones where V1's constraint numbers are strongest.
6. **Before any of that**, decide whether true FiLM is worth the retrain budget at all. The
   current evidence is one clean negative and one broken run; the honest summary is
   **"V2 does not help on aligning, and costs ~15% more time per replan."**

---

## Appendix — reproducing

```bash
# on this container (stdlib only, no project env needed):
python3 logs_in_develop/Gen14/U7/FiLM_V2_DA/da_20260812_filmv2.py
```

Reads `temp/1208/Gen14_Mix_FilmV2/batch_va2_20260812_114647/per_rollout_detail.csv` and the
two `*_losses.pkl` files. `temp/` is gitignored — the CSVs are local only; re-sync from
i6-gpu-1 (`Data_Analysis/analysis_results/batch_va2_20260812_114647/`) if the folder is gone.
Permutation tests are seeded (`random.seed(20260812)`), so the p-values reproduce exactly.
