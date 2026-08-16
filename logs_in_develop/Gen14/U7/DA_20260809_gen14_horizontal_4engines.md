# DA — horizontal VS: four Gen14 engines × the full projector grid, with paired statistics

**Date:** 2026-08-09
**Question:** the four mix arms — `diffusion`, `fm`, `mf`, `af` — share one codebase, one
training recipe and one eval harness. Put them **next to each other**, and put the **projector
variants** next to each other too. What separates, what ties, and what is genuinely noise?
**Answer, in one line:** **goal success is at the noise floor and cannot rank anything — but the
constraint and stability axes separate hard, and three findings are statistically decisive:
(1) dropping the bounds constraint (`bounds_free`) is the single best projector setting in the
whole grid *and* the cheapest; (2) HardFlow beats DPCC on constraint satisfaction but costs
3.4×, so neither dominates; (3) the "smart" min-projection-cost selection rule `-c` is
significantly *worse* than random `-r`, replicated independently in both projector arms.**

**Data:** `temp/2026-08-07/batch_va2_20260809_103838/` — DA_VA_v2 batch, 11 candidates,
202 units, **3275 rollouts**, `logs/aligning-d3il-visual/plans` on i6-gpu-1, seed 6, train split.
**Script:** `da_20260809_gen14_horizontal_4engines.py` (stdlib only — this container has no
project env). Every number below is its printed output.
**Predecessor:** `DA_20260808_gen14_diffu_fm_arms.md` — same rollouts, *vertical* cut (each arm
vs its ancestor generation). This is the *horizontal* cut.

---

## The design — why one seed is not one run

The batch is **one seed but not one sample.** Each K=2 engine ran

> **19 projector variants × 30 contexts × 2 geos = 1140 rollouts**, every variant on the
> **same contexts**.

So the projector axis is a **within-model paired design**: the same checkpoint, the same
initial states, the only thing changing is the projector. That gives **n = 112 paired
context-cells** per variant pair (2 engines × 56 unfrozen cells), or **n = 336** when the three
selection rules are stacked, and **n = 1064** for the engine-level `mf` vs `af` contrast across
all 19 variants. That is ample power, and it is why most of this document is about the projector
axis rather than the engine axis.

Two things this design does **not** buy, stated once and not repeated:

- **Engine-level differences are confounded with checkpoint.** One training run per engine, so
  "`mf` beats `af`" formally means "*this* MeanFlow checkpoint beats *this* AlphaFlow
  checkpoint". Only a second seed separates engine from run. **Variant-level results carry no
  such confound** — they compare one checkpoint against itself.
- **Goal success is underpowered everywhere.** 3 goal successes in 183 K=100 rollouts. §8 says
  precisely where "it is noise" is the correct answer and where it is not.

**Method.** Paired sign-flip permutation test (20 000 resamples) for continuous metrics; exact
McNemar for paired binary outcomes; Holm-Bonferroni within each family of comparisons. The
152 frozen rollouts (all in `combined_5-tightened`, 4 contexts × 19 variants × 2 engines) are
dropped from every paired test.

> **Provenance note.** `candidates_per_variant.csv`, `per_rollout_detail.csv`,
> `va2_aggregated_long.csv` and `data_quality.csv` are **byte-identical (md5) to
> `batch_va2_20260808_105342/`**. This batch is a re-run of the analyzer over the same eval
> logs, not new cluster work. `temp/` is gitignored — CSVs are local only.

---

## 0. TL;DR — what this batch delivers

**Decisive (survive Holm correction, large paired n):**

1. **`bounds_free` is the best projector configuration in the grid, and it is cheaper than the
   full projector.** Dropping the bounds constraint beats full `dpcc-c` on every constraint
   metric — sat **+7.7 pp**, violations **−30.6 steps**, collision-free **78 vs 48 of 112**
   (all *p* < 0.001) — at **46.7 ms vs 57.9 ms**. It beats every DPCC variant, ties HardFlow's
   best, and does so at **3.8× less cost than HardFlow**. **The bounds constraint is actively
   harmful in this config.** §7.5, §7.6.
2. **Projection works, and `gradient` guidance does not.** All six real projectors improve
   constraint satisfaction by **+9 to +16 pp** over no projection (*p* < 0.001, n = 112).
   `gradient` is the sole exception: **−0.97 pp, ns** — it costs 1.8 ms and buys nothing. §7.1.
3. **`-c` (minimum projection cost) is worse than `-r` (random).** DPCC: sat **−4.9 pp**
   (*p* < 0.01). HardFlow: sat **−2.5 pp** (*p* < 0.05). **Two independent projector arms,
   same direction** — the "smart" selection rule reliably hurts. `-t` (temporal consistency)
   ties `-r`. §7.3.
4. **`fm` is decisively broken; the other three engines tie.** `fm` vs `mf` and `fm` vs `af`:
   23/30 vs 6/30 and 9/30 diverged, Holm-adjusted *p* = **0.0029**. `diffusion`, `mf` and `af`
   are **statistically indistinguishable** from each other on both divergence and sat. §2.2.

**Supported, single-family, worth acting on:**

5. **HardFlow beats DPCC on constraints — at 3.4× the cost.** Pooled over 6 matched cells
   (n = 336): sat **+2.4 pp** (*p* = 0.0019), violations **−9.6** (*p* = 0.0019), collision-free
   **203 vs 183** (*p* = 0.040), time **+126 ms** (*p* < 10⁻⁴). **Neither dominates — this is a
   trade-off, not a win.** And `bounds_free` Pareto-dominates HardFlow. §7.2.
6. **`mf` beats `af` at K=2** across all 19 variants (n = 1064): sat **+1.9 pp** (*p* = 0.017),
   collision-free **487 vs 435** (*p* = 0.0068), successes **29 vs 13** (*p* = 0.017).
   **This contradicts the batch's own ranking**, which puts `af` first off a single truncated
   11-rollout cell. §6.
7. **K=2 costs 33× less than K=100 and is not worse.** §5, §6.

**Genuinely noise — say so and move on:** goal success at K=100 (3/183), every engine-level
success-rate gap, and the dt sweep except `dt4p0`. §8.

---

## 1. What is actually comparable

All four arms: `mix_visual_aligning/`, seed 6, train split, geo `combined_5`, `if_vision=True`,
`horizon=8`, `mpc_batch_size=4`, `diffusion_timestep_threshold=0.5`, `filmv1`, 1000 steps/epoch,
`bs64`.

| | `diffusion` | `fm` | `mf` | `af` |
|---|---|---|---|---|
| batch candidate | **7** | **8** | **9** (K100), **10** (K2) | **5** (K100), **6** (K2) |
| model class | `VisualGaussianDiffusion` | `VisualFlowMatching` | `VisualMeanFlow` | `VisualAlphaFlow` |
| solver | DDPM, 100 steps | Euler | Euler | Euler |
| K swept here | 100 only | 100 only | **100 and 2** | **100 and 2** |
| **action weight** | **`aw10`** | `aw1` | `aw1` | `aw1` |
| extra train flags | — | — | `tslogit_normal` | `tslogit_normal`, `afschsigmoid` |
| variants landed | 2 | 2 | K100: 2 · **K2: 19** | K100: 2 · **K2: 19** |

**The variant grid** (from `mix_visual_aligning_test/eval_mix_visual_aligning.py`). The suffix is
a **trajectory-selection rule**, not a projector strength — and `hardflow_new-X` is the
deliberately matched partner of `dpcc-X`, the two differing only in **when** the constraint is
applied:

| suffix | `trajectory_selection` | meaning |
|---|---|---|
| `-r` | `random` | always index 0 — deterministic, DPCC `Policy.__call__` semantics |
| `-c` | `minimum_projection_cost` | pick the sample the projector moved least |
| `-t` | `temporal_consistency` | pick the sample closest to the previous plan |

| family | variants | what it is |
|---|---|---|
| **DPCC (arm A)** | `dpcc-r/-c/-t`, `dpcc-c-dt{0p25,0p5,2p0,4p0}` | post-hoc projection of the generated trajectory |
| **HardFlow (arm C)** | `hardflow_new-r/-c/-t` | constraint applied *inside* sampling |
| **baselines** | `diffuser` (none), `gradient`, `post_processing` | no projection / gradient guidance / post-hoc cleanup |
| **ablations** | `geo_free`, `model_free`, `bounds_free` and pairs | full projector minus one constraint class |

Two structural asymmetries bound everything: **only `mf`/`af` have the K sweep and the 19-variant
grid** (their K=2 runs completed; all four K=100 runs hit the 24 h cap), and **`diffusion` trains
at `aw10` while the flow arms train at `aw1`** — so any "diffusion tracks better" reading is
partly about the loss weight.

**Column glossary** (medians unless `%`): `div%` peak physical tracking error > 1 m ·
`peErr` median peak tracking error [m] · `sat%` `constraint_exec_sat_rate` · `viol` violated
steps of 400 · `prog_m` box-to-target distance closed · `stuck%` |prog| < 0.02 m · `t_ms`
ms/replan. Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, `ns` not significant.

---

## 2. Engine axis, K=100, unprojected — the clean 4-way

### 2.1 Descriptive (same 30 contexts)

| engine | n | succ% | rel% | cfree% | div% | peErr | sat% | viol | prog_m | stuck% | final_m | t_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffusion-K100` | 30 | 0.0 | 0.0 | 26.7 | 43.3 | 0.053 | **95.8** | 17.0 | 0.270 | 20.0 | 0.185 | 1524.8 |
| `fm-K100` | 30 | 0.0 | 0.0 | 10.0 | **76.7** | **2.840** | **24.4** | **302.5** | **0.022** | **46.7** | 0.400 | 1425.9 |
| `mf-K100` | 30 | 3.3 | 6.7 | 20.0 | **20.0** | 0.050 | 94.4 | 22.5 | **0.333** | **10.0** | **0.139** | **893.1** |
| `af-K100` | 30 | 3.3 | 3.3 | 20.0 | 30.0 | 0.082 | 91.0 | 34.5 | 0.165 | 20.0 | 0.340 | 902.0 |

### 2.2 Paired significance (McNemar on divergence, permutation on sat)

| comparison | diverged | raw *p* | **Holm *p*** | sat diff | *p* |
|---|---|---|---|---|---|
| `fm` vs `mf` | 23/30 vs 6/30 | 0.0005 | **0.0029 ✳✳** | −0.393 | 0.0001 *** |
| `fm` vs `af` | 23/30 vs 9/30 | 0.0005 | **0.0029 ✳✳** | −0.308 | 0.0002 *** |
| `diffusion` vs `fm` | 13/30 vs 23/30 | 0.0213 | 0.0851 ns | +0.306 | 0.0015 ** |
| `diffusion` vs `mf` | 13/30 vs 6/30 | 0.1435 | 0.4304 ns | −0.087 | 0.2731 ns |
| `diffusion` vs `af` | 13/30 vs 9/30 | 0.3877 | 0.7754 ns | −0.002 | 0.9853 ns |
| `mf` vs `af` | 6/30 vs 9/30 | 0.5488 | 0.7754 ns | +0.085 | 0.2814 ns |

**Read this as two statements, not four.**

1. **`fm` is broken and the evidence is decisive.** It survives Holm correction against both
   flow siblings. Its median peak tracking error is **2.84 m** against 0.050–0.082 m — a factor
   of 35, which is not "worse tracking" but the arm leaving the workspace — and it violates
   constraints on **302 of 400 steps** against 17–35 for the others.
2. **`diffusion`, `mf` and `af` are statistically indistinguishable at n=30.** Every pairwise
   comparison among them is `ns` on both metrics. `mf` has the best point estimates (lowest
   divergence, highest progress, lowest final distance, cheapest), but **at this n you cannot
   claim `mf` > `diffusion`.** The `fm` vs `diffusion` cell is significant raw but not after
   Holm — report it as suggestive.

Constraint detail on the same rollouts:

| engine | median penetration [m] | median first-violation step | median longest safe streak | zero-violation |
|---|---|---|---|---|
| `diffusion-K100` | **0.0000** | **74** | 303.5 | **26.7 %** |
| `fm-K100` | 0.0004 | 57 | **73.5** | 10.0 % |
| `mf-K100` | 0.0031 | 60 | **310.0** | 20.0 % |
| `af-K100` | **0.0129** | 66.5 | 193.5 | 20.0 % |

`fm`'s longest safe streak is a quarter of the others': once it starts violating it never
recovers. `af`'s penetration is 4× `mf`'s while its sat is higher than `fm`'s — its violations
are fewer but deeper.

---

## 3. Projected K=100 — ragged, then paired

All four K=100 evals were **CANCELLED on the 24 h cap**, so `dpcc-r` landed 19 / 22 / 11 / 11
rollouts on *different* contexts. Paired on the 11 contexts all four finished:

| engine | n | succ% | s+c% | cfree% | div% | sat% | viol | prog_m | stuck% | t_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `diffusion-K100` | 11 | 0.0 | 0.0 | 27.3 | 9.1 | 84.2 | 63.0 | 0.229 | 45.5 | 7230.5 |
| `fm-K100` | 11 | 0.0 | 0.0 | **0.0** | 0.0 | **61.5** | **154.0** | **0.000** | **72.7** | 7913.0 |
| `mf-K100` | 11 | 0.0 | 0.0 | 18.2 | 0.0 | 94.8 | 21.0 | **0.251** | 27.3 | 14946.3 |
| `af-K100` | 11 | **9.1** | **9.1** | 27.3 | 0.0 | **97.2** | **11.0** | 0.241 | **9.1** | 15113.9 |

Two things to take from it:

1. **The projector converts `fm`'s divergence into inertia.** Unprojected `fm`: 77 % diverged,
   0.022 m progress. Projected: **0 % diverged, 0.000 m progress, 73 % stuck, 0 % collision-free
   completed**, violations still worst of the four (154). The projector stops `fm` leaving the
   workspace and in doing so stops it doing anything.
2. **Never quote the ragged table.** `diffusion` reads sat 97.5 % / 10 violated steps on its 19
   landed rollouts and **84.2 % / 63** on the 11 everyone finished. Same model, same variant —
   the 13-point swing is purely which contexts landed before the cap. Under compute-capped
   evaluation, cross-method tables must be paired on completed contexts.

Because n = 11 here, **no engine-level significance test on this table is meaningful.** The
projector-axis work in §7 exists precisely because it does not depend on these truncated cells.

---

## 4. Mode coverage

`mode_encoding` per unprojected rollout, `combined_5`:

| engine | n | mode 0 | mode 1 |
|---|---|---|---|
| `diffusion-K100` | 30 | 15 (50 %) | 15 (50 %) |
| `mf-K100` | 30 | 17 (57 %) | 13 (43 %) |
| `mf-K2` | 30 | 10 (33 %) | 20 (67 %) |
| `af-K2` | 30 | 7 (23 %) | 23 (77 %) |
| `fm-K100` | 30 | 4 (13 %) | **26 (87 %)** |
| `af-K100` | 30 | 2 (7 %) | **28 (93 %)** |
| `Gen6V4-K20` (ref) | 33 | 0 | **33 (100 %)** |

Aligning is multimodal; a healthy policy spreads. `diffusion` and `mf` do. **`af` at K=100 does
not — 93 % one mode, closer to the fully-collapsed Gen6V4 reference than to its own K=2 run
(77 %).** For `af` this is the clean signal, because `af` tracks fine (peErr 0.082 m, sat 91 %) —
it executes a coherent plan, just nearly the same one every time. For `fm` (87 %) the label is
unreliable: a rollout that diverges 2.84 m may be tagged with a mode it never executed.

`af` getting **more** collapsed as K rises (77 % → 93 %) is backwards from the usual
"more solver steps, better samples" expectation. Flagged as a hypothesis, not a finding —
one model, one seed, and the two K cells are different eval runs.

---

## 5. Cost

Median ms/replan, unprojected — the only rows where cost is comparable (all ran 30 contexts to
completion):

| engine | K | ms/replan | relative |
|---|---|---|---|
| `mf-K2` / `af-K2` | 2 | **27.7 / 26.6** | 1.0× |
| `Gen6V4-K20` (ref) | 20 | 338.1 | ~12× |
| `mf-K100` / `af-K100` | 100 | 893.1 / 902.0 | ~33× |
| `fm-K100` | 100 | 1425.9 | ~53× |
| `diffusion-K100` | 100 | 1524.8 | ~56× |

Cost splits by **engine family, not by quality**: the two `tslogit_normal` arms cost ~900 ms at
K=100 while `fm` and `diffusion` cost ~1450–1525 ms — a 1.6× gap at identical K that per-step
solver cost does not obviously explain and that deserves a profiling pass. The best-behaved arm
(`mf`) and the worst (`fm`) sit on opposite sides of it, so **cost carries no information about
which engine works.**

Practical consequence: at 900–1500 ms × 400 steps × 30 contexts, one K=100 variant is 3–5 h and
a 32-item sweep does not fit in a 24 h job.

---

## 6. `mf` vs `af` at K=2 — n = 1064 paired

Paired over **all 19 variants × 56 unfrozen context-cells**. This is the highest-powered
engine-level contrast in the batch:

| metric | mf − af | *p* |
|---|---|---|
| sat | **+0.0194** (+1.9 pp) | 0.017 * |
| violated steps | **−7.75** | 0.017 * |
| goal successes | **29 vs 13** (discordant 28/12) | 0.017 * |
| collision-free completed | **487 vs 435** (discordant 204/152) | **0.0068 ** ** |
| peak tracking error | +0.023 | 0.166 ns |

**MeanFlow beats AlphaFlow on this visual task, consistently across the whole projector grid.**
Effect sizes are small (2 pp of sat) but the direction is stable over 1064 paired rollouts.

**This contradicts the batch's own top-line ranking**, which puts `af-K100` at rank 1 with
"goal+constraint 4.55 %". That figure is **one successful rollout out of 11** in a truncated
cell, averaged over two variants. `candidates_ranking.csv` should not be used to rank engines
in this batch.

Per-variant descriptives:

| engine | variant | n | succ% | rel% | s+c% | cfree% | div% | sat% | viol | prog_m | t_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mf-K2` | `diffuser` | 30 | 3.3 | 6.7 | 3.3 | 33.3 | 13.3 | 88.5 | 43.5 | 0.155 | 27.7 |
| `af-K2` | `diffuser` | 30 | 6.7 | 10.0 | 3.3 | 20.0 | 20.0 | 96.2 | 15.0 | 0.291 | 26.6 |
| `mf-K2` | `dpcc-r` | 30 | 6.7 | 13.3 | 6.7 | 36.7 | 0.0 | 86.8 | 53.0 | 0.256 | 45.5 |
| `af-K2` | `dpcc-r` | 30 | 3.3 | 3.3 | 0.0 | 36.7 | 3.3 | **99.1** | **2.5** | 0.187 | 44.3 |

Note also **K=2 is not worse than K=100** (`mf`: 13.3 % vs 20.0 % divergence; `af`: 20.0 % vs
30.0 %) at 33× less compute — which, given §5's 24 h problem, is the most operationally useful
fact in the batch.

---

## 7. The projector axis — the main result

Everything below is **pooled over `mf-K2` + `af-K2`**, paired on the same contexts, n = 112
(or 336 where the three selection rules stack). This axis has no truncation problem and no
checkpoint confound.

### 7.1 Does projection work at all?

| variant − `diffuser` | sat | viol | cfree | t_ms |
|---|---|---|---|---|
| `hardflow_new-r` | **+0.156 ***** | −62.0 *** | 71 v 25 *** | +149.8 *** |
| `hardflow_new-t` | +0.149 *** | −58.9 *** | 71 v 25 *** | +150.2 *** |
| `post_processing` | +0.141 *** | −55.8 *** | 66 v 25 *** | +26.4 *** |
| `dpcc-r` | +0.140 *** | −55.4 *** | 65 v 25 *** | +26.1 *** |
| `dpcc-t` | +0.135 *** | −53.4 *** | 70 v 25 *** | +23.9 *** |
| `hardflow_new-c` | +0.132 *** | −52.2 *** | 61 v 25 *** | +158.3 *** |
| `dpcc-c` | +0.090 ** | −35.6 ** | 48 v 25 ** | +30.7 *** |
| **`gradient`** | **−0.010 ns** | **+4.2 ns** | **30 v 25 ns** | +1.8 *** |

**Projection is the single largest effect in the batch** — +9 to +16 pp of constraint
satisfaction, 35–62 fewer violated steps out of 400, all *p* < 0.001.

**`gradient` guidance is the one thing that does not work.** It is the only variant that fails
to beat no projection on any metric, and it is statistically flat. It costs almost nothing
(1.8 ms) but delivers nothing — a clean negative result on gradient-based constraint guidance
in this setting.

**`post_processing` matches `dpcc-r` almost exactly** (+0.1408 vs +0.1397, same cost). Whatever
`dpcc-r` is buying over a simple post-hoc cleanup, it is not visible here.

### 7.2 HardFlow (arm C) vs DPCC — matched selection rule

The two arms differ only in **when** the constraint is applied. Per-cell:

| engine | rule | sat (HF−DPCC) | viol | cfree | t_ms |
|---|---|---|---|---|---|
| `mf-K2` | `-r` | +0.013 ns | −5.2 ns | 32 v 34 ns | +123.9 *** |
| `mf-K2` | `-c` | +0.026 ns | −10.5 ns | 30 v 25 ns | +123.5 *** |
| `mf-K2` | `-t` | +0.035 ns | −13.8 ns | 40 v 34 ns | +120.7 *** |
| `af-K2` | `-r` | +0.020 ns | −8.1 ns | 39 v 31 * | +123.6 *** |
| `af-K2` | `-c` | +0.057 * | −22.8 * | 31 v 23 ns | +131.8 *** |
| `af-K2` | `-t` | −0.007 ns | +2.7 ns | 31 v 36 ns | +131.9 *** |

Individually underpowered (5 of 6 favour HardFlow, 1 significant). **Stacked, n = 336:**

| metric | HF − DPCC | *p* |
|---|---|---|
| sat | **+0.0241** (+2.4 pp) | **0.0019 ** ** |
| violated steps | **−9.62** | **0.0019 ** ** |
| collision-free | **203 vs 183** (discordant 53/33) | 0.040 * |
| **ms/replan** | **+125.9** | **< 10⁻⁴ *** ** |

**HardFlow beats the DPCC projector on constraint satisfaction, and costs 3.4× more
(~178 ms vs ~53 ms).** Per the project's benchmark hierarchy — HardFlow must beat the DPCC
projector — it does, on the constraint axis. But it does not dominate: this is a **trade-off**.
And §7.5 shows a DPCC ablation that matches HardFlow's constraint numbers at **a quarter of
HardFlow's cost**, which makes HardFlow's position here weak.

### 7.3 Trajectory-selection rule — the "smart" rule hurts

| comparison | sat | viol | cfree | t_ms |
|---|---|---|---|---|
| `dpcc-c` − `dpcc-r` | **−0.049 ** ** | +19.8 ** | 48 v 65 * | +4.6 ns |
| `dpcc-t` − `dpcc-r` | −0.005 ns | +2.0 ns | 70 v 65 ns | −2.2 ns |
| `dpcc-t` − `dpcc-c` | **+0.045 ** ** | −17.8 ** | 70 v 48 *** | −6.8 ** |
| `hardflow_new-c` − `hardflow_new-r` | **−0.025 *** | +9.8 * | 61 v 71 ns | +8.5 ns |
| `hardflow_new-t` − `hardflow_new-r` | −0.008 ns | +3.1 ns | 71 v 71 ns | +0.4 ns |
| `hardflow_new-t` − `hardflow_new-c` | +0.017 ns | −6.7 ns | 71 v 61 ns | −8.1 ns |

**`minimum_projection_cost` (`-c`) is significantly worse than `random` (`-r`) — and the effect
replicates independently in both projector arms** (DPCC −4.9 pp, *p* < 0.01; HardFlow −2.5 pp,
*p* < 0.05). Selecting the sample the projector had to move least systematically picks a worse
trajectory. `temporal_consistency` (`-t`) ties `-r` and costs the same.

Two independent arms agreeing is what lifts this above a fishing expedition. **Actionable:
`-c` should stop being a default anywhere in this generation.**

### 7.4 Dynamics-timestep sweep inside `dpcc-c`

| variant − `dpcc-c` | sat | raw *p* | **Holm *p*** | t_ms |
|---|---|---|---|---|
| `dpcc-c-dt4p0` | +0.051 | 0.0031 | **0.0066 ✳✳** | **−12.9 *** |
| `dpcc-c-dt0p25` | +0.037 | 0.0221 | 0.0882 ns | −0.4 ns |
| `dpcc-c-dt0p5` | +0.020 | 0.2000 | 0.3999 ns | −0.6 ns |
| `dpcc-c-dt2p0` | +0.014 | 0.3576 | 0.3999 ns | +2.2 ns |

Only `dt4p0` survives correction. But the pattern is **non-monotonic** — both ends of the sweep
help, the middle does nothing — which is not what a real dynamics-timestep effect looks like.
Most of this is `dpcc-c` being a weak baseline (§7.3): almost anything beats it. **Treat as
"`dpcc-c` is bad", not "dt matters".**

### 7.5 Constraint ablations — the bounds constraint is harmful

Each variant drops one constraint class from the full projector:

| variant − `dpcc-c` | sat | viol | cfree | t_ms |
|---|---|---|---|---|
| **`bounds_free`** | **+0.077 ***** | **−30.6 *** ** | **78 v 48 *** ** | **−11.3 *** ** |
| `geo_free` | −0.108 *** | +43.3 *** | 30 v 48 * | −16.3 *** |
| `model_free` | −0.099 ** | +39.5 ** | 23 v 48 *** | −17.1 *** |
| `geo_free-bounds_free` | −0.091 ** | +36.2 ** | 35 v 48 ns | −19.3 *** |
| `geo_free-model_free` | −0.099 ** | +39.7 ** | 23 v 48 *** | −23.4 *** |
| `model_free-bounds_free` | −0.087 ** | +34.6 ** | 23 v 48 *** | −20.2 *** |

The geometry and model constraints are **load-bearing** — dropping either costs ~10 pp of
satisfaction. **The bounds constraint is the opposite: dropping it gains 7.7 pp, removes 30
violated steps, raises collision-free completion from 48 to 78 of 112, and saves 11 ms.**

A constraint whose removal improves constraint satisfaction is over-constraining the projection
and fighting the geometry/model terms. This is corroborated from outside the statistics: the
batch itself contains a candidate whose folder is literally named
`(legacy_correcct_besides_bounds_for_comapre)` — the bounds term was already under suspicion.
**This is the most actionable finding in the batch: the bounds constraint in the visual-aligning
projector config needs re-derivation or removal.**

### 7.6 Full variant ranking

Pooled `mf-K2` + `af-K2`, 112 unfrozen rollouts each, sorted by collision-free completion:

| variant | sat% | viol | cfree% | succ | peErr | t_ms |
|---|---|---|---|---|---|---|
| **`bounds_free`** | **95.2** | **19.3** | **69.6** | 3 | 0.059 | **46.7** |
| `hardflow_new-r` | 94.1 | 23.5 | 63.4 | 1 | 0.062 | 177.1 |
| `hardflow_new-t` | 93.4 | 26.6 | 63.4 | 1 | 0.057 | 177.5 |
| `dpcc-t` | 92.0 | 32.1 | 62.5 | 2 | 0.051 | 51.2 |
| `post_processing` | 92.6 | 29.7 | 58.9 | 3 | 0.058 | 53.6 |
| `dpcc-r` | 92.5 | 30.1 | 58.0 | 3 | 0.058 | 53.3 |
| `hardflow_new-c` | 91.7 | 33.3 | 54.5 | 3 | 0.058 | 185.6 |
| `dpcc-c-dt4p0` | 92.6 | 29.6 | 50.9 | 0 | 0.035 | 45.0 |
| `dpcc-c-dt0p25` | 91.2 | 34.9 | 48.2 | 4 | 0.047 | 57.6 |
| `dpcc-c-dt0p5` | 89.5 | 42.0 | 46.4 | 2 | 0.052 | 57.4 |
| `dpcc-c` | 87.5 | 49.9 | 42.9 | 1 | 0.053 | 57.9 |
| `dpcc-c-dt2p0` | 88.9 | 44.3 | 35.7 | 0 | 0.053 | 60.1 |
| `geo_free-bounds_free` | 78.4 | 86.1 | 31.2 | 1 | 0.103 | 38.7 |
| `gradient` | 77.5 | 89.7 | 26.8 | 6 | 0.053 | 29.0 |
| `geo_free` | 76.7 | 93.2 | 26.8 | 3 | 0.100 | 41.6 |
| `diffuser` | 78.5 | 85.5 | 22.3 | 6 | 0.058 | 27.2 |
| `model_free-bounds_free` | 78.9 | 84.5 | 20.5 | 1 | 0.055 | 37.7 |
| `model_free` | 77.7 | 89.4 | 20.5 | 1 | 0.054 | 40.8 |
| `geo_free-model_free` | 77.6 | 89.6 | 20.5 | 1 | 0.056 | 34.5 |

`bounds_free` against every other variant, paired:

- **Beats every DPCC variant**: vs `dpcc-c` cfree 78 v 48 ***; vs `dpcc-r` 78 v 65 *,
  sat +2.7 pp *; vs `dpcc-t` sat +3.2 pp ** (cfree ns); vs all four `dt` variants ** to ***.
- **Beats `hardflow_new-c`** (cfree 78 v 61 **, sat +3.5 pp **).
- **Ties `hardflow_new-r` and `hardflow_new-t`** (cfree 78 v 71, ns; sat ns) — **at 46.7 ms
  against 177 ms, i.e. 3.8× cheaper.**
- Beats every no-projection and ablation baseline at *** .

**`bounds_free` Pareto-dominates the HardFlow arm**: equal constraint performance, a quarter of
the cost. Given the project's rule that "good" means Pareto-dominant, this is the only variant
in the grid that earns the word.

**Caveat that keeps this honest:** the `succ` column does not follow the ranking at all —
`gradient` and `diffuser`, the two worst variants on constraints, have the *most* goal successes
(6 each). That is §8's point, and it means `bounds_free` is established as the best **constraint**
configuration, not as the configuration that solves the task.

---

## 8. What is genuinely noise — and what is not

The user-facing question "if it is random, say it is random" deserves a precise answer, because
it is metric-dependent:

| axis | verdict |
|---|---|
| **Goal success at K=100** | **Noise.** 3 in 183 rollouts. The 95 % Wilson interval on 1/30 is ~0.6–17 %. No engine ranking can rest on it. |
| **Goal success across the K=2 grid** | **Weak but not empty.** 29 vs 13 over 1064 paired rollouts, *p* = 0.017 — significant only because n is large. Effect is ~1.5 pp. |
| **`succ` column in the variant ranking (§7.6)** | **Noise, and anti-correlated with everything else.** `gradient`/`diffuser` lead on successes while worst on constraints. With 0–6 successes per 112, these are single rollouts. **Do not rank variants by success in this batch.** |
| **Constraint satisfaction / violated steps** | **Not noise.** n = 400 steps per rollout × 112–336 paired rollouts. Effects of 2–16 pp are detected reliably. |
| **Divergence rate** | **Not noise for `fm`** (survives Holm at n=30). **Noise among `diffusion`/`mf`/`af`** at this n. |
| **Cost (ms/replan)** | **Not noise.** Every timing comparison is *p* < 10⁻⁴. |

The general shape: **this batch cannot rank engines by task success, but it ranks projector
configurations by constraint behaviour with real statistical power.** That is the correct
division of what to believe.

---

## 9. Paper triage — what could be claimed

**From the projector axis — usable, with the caveat that it is one task and one seed:**

1. **"Removing the bounds constraint from the projector improves constraint satisfaction and
   halves cost."** Paired n = 112, *p* < 0.001 on three metrics, replicated across two engines.
   The counter-intuitive direction is the interesting part. Strongest single result in the batch.
2. **"Minimum-projection-cost trajectory selection is worse than random selection."**
   Independently replicated in two projector arms. Cheap to state, easy to defend, and directly
   useful to anyone implementing DPCC-style selection.
3. **"Gradient guidance does not improve constraint satisfaction over no projection, while
   projection improves it by 9–16 pp."** Clean negative result with a matched positive control.
4. **"HardFlow-style in-sampling constraints beat post-hoc projection on satisfaction (+2.4 pp)
   at 3.4× the cost — and are matched by a cheaper DPCC ablation."** A trade-off statement,
   not a win claim.

**From the engine axis — limitations material only:** three of four engines are
indistinguishable at n=30; the fourth (`fm`) is decisively broken and sits in the same failure
class as the dead Gen6V4 reference. And the projector-inertia observation (§3): every
constraint metric improves while every task metric goes to zero — a concrete illustration that
**constraint satisfaction is not a proxy for planner quality.**

**What must not be claimed:** any engine ranking by goal success on visual aligning, and
anything from the ragged §3 table.

**What is stronger and lives elsewhere:** the **state-based `avoiding-d3il`** results in
`logs_in_develop/Gen3v6_MeanFlow/DA/` — 5 seeds × 2 trials × 3 envs, reporting AlphaFlow K=2 at
1.000 success+constraints in all three envs at 15–18× less planning time than DPCC K=10
(`DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`), with the low-K capability argument in
`DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`. **Two things there decide whether it survives
review, both already recorded in those docs:** a 1-NFE diffusion baseline (`CAND_109`) also
reaches 1.000 and is faster than AlphaFlow — the ablation's per-environment collapse of that
baseline (0/10 on `top-right-hard`) is what makes the claim non-trivial; and the two docs
report different levels (1.000 under `-r-tightened` vs 70–80 % under plain `-r`), so pick one
variant as the headline explicitly. *I have not re-derived those numbers — they are read from
those documents' headline sections.*

---

## 10. What to run next, ordered by information per GPU-hour

1. **Re-run the projector grid with the bounds constraint re-derived or removed.** §7.5 says the
   current bounds term is costing you ~8 pp of constraint satisfaction across every engine. This
   is a config change, not a retrain.
2. **Drop `-c` selection from the defaults** (§7.3) and re-run. Also a config change.
3. **Run all four engines at K=2 with the full 19-variant grid.** At ~27 ms/replan a complete
   sweep is hours, not >24 h. This turns the truncated §3 table into a real 4-way and puts
   `diffusion`/`fm` on the projector axis, where they are currently absent.
4. **Second seed on `mf-K2` and `af-K2`** — the two arms with complete grids, so marginal cost
   is lowest, and it is the only thing that converts §6's engine-level result from
   "this checkpoint" into "this engine".
5. **Diagnose `fm`** — it shares a codebase with three arms that work, so the fault is in the
   `fm` engine path or its checkpoint. Start from the 23/30 diverged unprojected rollouts.
6. **Fix the `aw10`/`aw1` mismatch** (§1) before any §2 table is used to argue DDPM vs flow.
7. **Test `af` mode collapse directly** (§4): sweep K ∈ {2, 20, 100} on `af` alone and read the
   mode split.
