# DA — `fm` vs `mf`, three scenes, K=10 (batch `batch_uav_20260819_135638`)

**Date:** 2026-08-19 · **Batch:** `batch_uav_20260819_135638` (DA_UAV_v1, first real-data run)
**Arms:** `fm` (FlowMatchingODE, Gen11 engine) vs `mf` (MeanFlowODE, Gen3v6), both `unet`, K=10, seed 6
**Scenes:** corridor (10 trials) · pillars (3) · s_curve (3)
**Mask:** `proj_valid` throughout unless stated

---

## 0. TL;DR

**Corridor is a decisive `fm` win.** Of the 20 variants both arms ran, `fm` beats `mf` on
success-and-constraints in **11**, loses **1** (`geo_free`), and ties **8** — every tie at
0.00/0.00. On the three `dpcc` rows `fm` scores **S&C 1.00 against `mf`'s 0.70–0.80 while
also being ~25 ms/replan faster**, so it is Pareto-dominant there, not a trade-off.

**Pillars and s_curve are 0.00 strict success for both arms.** They separate on *how* they
fail, not whether, and the interesting split is goal distance, not success.

**`mf` loses at K=10 — which is the predicted result, not an anomaly.** MeanFlow's whole
advantage is removing Euler discretisation error, and that error is O(1/K). At K=10 there is
almost none left to remove, so `mf` pays the two-time objective's cost and collects none of
its benefit. **The experiment that would actually test MeanFlow (K=1–2) has never been run.**
See §6.5 — and read §0's first paragraph as "at K=10", not as a verdict on the objective.

---

## 1. First: why the batch "feels random"

The ranking table is dominated by noise, and it is worth naming the cause before reading it.

**28 of the 34 candidates are Gen11 archaeology.** Auto-scan pulled in every historical
plan directory under `logs/UAV_FM`, which turns out to be 13 different snapshot folders:

| n | folder |
|---|---|
| 8 | `plans(Bf_DC-FIX)` |
| 5 | `plans(Bf_U8)` |
| 3 | `plans` ← the only current ones |
| 2 | `plans(E6)`, `plans(E7_U3)` |
| 1 each | `plans(E7_U1)`, `plans(E7_U4_fix3)`, `plans(E7U3)`, `plans(Bf_Fix14)`, `plans(U1)`, `plans(no_GIF)`, `plans(with_gif_parts)` |

Only 6 candidates (29–34) are Gen15. Three consequences:

1. **All 28 Gen11 candidates have a blank `engine`**, and 17 have a blank `K`. Their paths
   predate Gen15's `E{engine}` exp-name token, so `parse_axes()` finds nothing to parse. The
   engine axis — the whole point of the panel — is empty for every baseline row.
2. **Rank 1 is candidate 9, `empty|K20|Gen11`, at 100% S&C.** That is the obstacle-free
   scene. It tops the table because it is trivial, and it is marked Pareto `FRONT`.
3. **Candidate 25 (`s_curve|K20|Gen11`) is also on the Pareto `FRONT` at 0.0% S&C** — it
   gets there on low time and low tracking error. A 0%-success row on the front means the
   front is being computed on axes that do not include success as a gate.

**Also: candidate-level means pool across all 23 variants.** `corridor|fm|K10` reports
"51.7% S&C" — that is the average of eight 1.00s and eleven 0.00s, a number that describes
no configuration that exists. Same for `NFE_effective`: corridor `fm` shows 16.5 and
corridor `mf` shows 10.0, which looks like an engine difference but is just variant mix —
`fm` had three HardFlow variants (~1.5× NFE) averaged in and `mf` had none.

> **Read this batch at the per-variant level (`candidates_per_variant.csv`), not the
> candidate level (`candidates_ranking.csv`).** Everything below does.

---

## 2. Corridor — `fm` wins, clearly (10 trials)

`succ` = strict success · `S&C` = success **and** constraints · `gdist` = final goal distance (m) · `tot` = ms/replan

| variant | fm succ | mf succ | **fm S&C** | **mf S&C** | fm gdist | mf gdist | fm tot | mf tot |
|---|---|---|---|---|---|---|---|---|
| `diffuser` (unguided) | **0.40** | 0.00 | 0.00 | 0.00 | **0.80** | 39.26 | 89 | 89 |
| `dpcc-r` | 1.00 | 1.00 | **1.00** | 0.80 | 0.29 | 0.29 | **245** | 271 |
| `dpcc-c` | 1.00 | 1.00 | **1.00** | 0.70 | 0.29 | 0.29 | **245** | 270 |
| `dpcc-t` | 1.00 | 1.00 | **1.00** | 0.70 | 0.29 | 0.29 | **244** | 273 |
| `dpcc-r-tightened` | 0.90 | 0.40 | **0.70** | 0.10 | 0.41 | 7.36 | **285** | 499 |
| `dpcc-c-tightened` | 0.80 | 0.60 | **0.60** | 0.10 | 0.44 | 5.67 | **282** | 430 |
| `dpcc-t-tightened` | 0.90 | 0.60 | **0.70** | 0.00 | 0.38 | 2.41 | **290** | 502 |
| `bounds_free` | 1.00 | 1.00 | **1.00** | 0.70 | 0.29 | 0.29 | **221** | 242 |
| `bounds_free-tightened` | 1.00 | 0.40 | **0.60** | 0.30 | **0.29** | 11.60 | **271** | 431 |
| `post_processing` | 1.00 | 0.10 | **1.00** | 0.00 | **0.29** | 13.94 | **119** | 196 |
| `post_processing-tightened` | 0.40 | 0.20 | **0.30** | 0.00 | 7.65 | 21.14 | **176** | 196 |
| `geo_free` | 0.80 | 0.70 | 0.40 | **0.50** | 6.78 | **1.08** | 168 | 185 |
| `geo_free-bounds_free` | 1.00 | 0.70 | **0.60** | 0.50 | **0.29** | 1.01 | **134** | 179 |
| `hardflow_new` | **1.00** | — | **1.00** | — | 0.29 | — | 725 | — |
| `hardflow_new-c` | **1.00** | — | **1.00** | — | 0.29 | — | 761 | — |
| `hardflow_new-t` | **1.00** | — | **1.00** | — | 0.29 | — | 731 | — |
| `gradient`, `gradient-t`, `model_free`×4, `geo_free-model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 4.3–7.3 | 4.7–61.3 | — | — |

### 2.1 The result

**`fm` Pareto-dominates `mf` on all three `dpcc` rows** — strictly higher S&C (1.00 vs
0.70–0.80) *and* lower time (244–245 ms vs 270–273 ms) at identical success, steps and goal
distance. That is a dominance claim, not a trade-off.

The `-tightened` rows are where the gap is widest: `fm` holds S&C 0.60–0.70 where `mf`
collapses to 0.00–0.10, and `mf` pays roughly **1.7× the wall clock** to do worse
(499 ms vs 285 ms on `dpcc-r-tightened`). Whatever tightening does to the plan, MeanFlow's
plans need far more projector work to absorb it and still end up outside the goal radius
(gdist 2.4–7.4 m vs `fm`'s 0.38–0.44 m).

`mf`'s single win is `geo_free` (S&C 0.50 vs 0.40, gdist 1.08 vs 6.78) — `fm` has an
outlier trial there dragging the mean.

### 2.2 Naive FM beats MeanFlow unguided **at K=10**

`diffuser` (no projector at all) is **0.40 success for `fm` and 0.00 for `mf`**, with final
goal distance **0.80 m vs 39.26 m** — a 49× gap. At K=10 MeanFlow's raw trajectory is not
merely unsuccessful, it is nowhere near the goal, whereas naive FM's is nearly usable
unprojected. This is the cleanest axis in the batch, because it isolates the objective from
the projector entirely.

⚠️ **Do not read this as MeanFlow failing the benchmark hierarchy.** That hierarchy's
requirement — MF beats naive FM — is a *low-K* requirement, and K=10 is where plain FM's
discretisation error has already vanished. What this row measures is the field-quality cost
of the two-time objective, which §6.5 traces to a 9.5× train/test generalisation gap. The
same row at K=2 is the one that would carry a verdict, and it has not been run.

### 2.3 HardFlow works on corridor

All three selections at **S&C 1.00, gdist 0.29 m**. This is the first clean HardFlow result
anywhere in Gen15 — corridor `mf` predates the U2 HardFlow arm, so there is no counterpart
to compare against, but it establishes the arm is sound on a solvable scene. Cost is
~725–761 ms/replan against ~245 ms for `dpcc`, i.e. **3× the wall clock for the same 1.00**,
so on corridor HardFlow is dominated by DPCC on time at equal quality.

---

## 3. Pillars — both arms fail, and they fail differently (3 trials)

Strict success is **0.00 for all 46 rows**. S&C is 0.00 everywhere. The signal is `gdist`:

| variant | fm gdist | mf gdist | fm tot | mf tot |
|---|---|---|---|---|
| `diffuser` | **1.23** | 118.07 | 89 | 89 |
| `dpcc-c` | **1.41** | 3.44 | 4136 | **1805** |
| `dpcc-t` | **0.85** | 50.39 | 4659 | **1972** |
| `dpcc-r-tightened` | **1.02** | 43.07 | 3004 | **1376** |
| `geo_free` | 0.69 | **0.59** | 141 | 150 |
| `geo_free-bounds_free` | 0.72 | **0.57** | 132 | 141 |
| `model_free` | **1.13** | 151.18 | 258 | 333 |
| `hardflow_new` | 3.40 | **0.29** | 5956 | **940** |
| `hardflow_new-c` | 3.19 | **0.51** | 5967 | **926** |
| `hardflow_new-t` ⚠️ | 2.74 | **0.78** (succ **0.67**) | 2521 | **1653** |

Two opposite readings, both true:

**`fm` is far more consistent.** Its goal distance sits in a **0.69–5.83 m** band across all
23 variants. `mf` ranges **0.29 m to 151 m** — it either nails the approach or leaves the
map entirely. On the unguided and model-free rows `fm` is two orders of magnitude closer.

**But `mf` is the only arm that actually reached the goal.** `mf`'s `hardflow_new` hits
0.29 m ×3 and `hardflow_new-t` is the single row on pillars with nonzero strict success
(0.67). `fm`'s HardFlow gets nowhere near (2.7–3.4 m) and costs **6.4× more projector time**
(5956 ms vs 940 ms) — a sign the IPOPT solve is thrashing on `fm`'s plans rather than
converging.

The same pattern shows on `dpcc`: `fm`'s projection costs **2.3–2.4× `mf`'s** (4136 vs
1805 ms on `dpcc-c`). `fm` produces plans that land in a tight, wrong place; `mf` produces
plans that are wilder but occasionally right, and cheaper to project.

---

## 4. s_curve — both zero, `fm` tracks far better (3 trials)

Strict success 0.00 everywhere, as for `mf` previously. The separation:

| variant | fm gdist | mf gdist | fm tot | mf tot |
|---|---|---|---|---|
| `geo_free` | **0.49** | 16.49 | **157** | 254 |
| `geo_free-bounds_free` | **0.49** | 36.52 | **146** | 222 |
| `dpcc-t` | **9.64** | 238.88 | **792** | 1121 |
| `dpcc-c` | **29.32** | 222.43 | 1008 | **1153** |
| `dpcc-r-tightened` | **12.92** | 145.25 | **839** | 1123 |
| `bounds_free-tightened` | **24.47** | 106.47 | 589 | 563 |

`fm`'s `geo_free` sits **0.49 m from the goal** — inside two goal radii — with relaxed
success 1.00 and safe 1.00, against `mf`'s 16.49 m. On the `dpcc` rows `fm` is 8–25× closer
and consistently faster. **`fm` dominates s_curve on every axis except the one that counts**
(strict success, 0.00 for both).

`fm`'s HardFlow rows are absent by design — `UAV_MIX_HF_OFF=1` was set because the s_curve
NLP is infeasible by construction (all four wall halfspaces held simultaneously; see
DA_20260816 §2). `mf`'s three HardFlow rows are present but void, at 5322–8765 ms/replan
for 0.00 — 8.8 seconds per replan against a 30.3 ms budget.

---

## 5. Data-integrity findings from this batch

Three things the DA surfaced that the eval logs alone did not.

### 5.1 Corridor `mf` predates Fix_1 — its timing split is unusable

| scene | engine | variants with `proj_ms` > 0 | mean `fm_ms` |
|---|---|---|---|
| corridor | fm | 20/23 | 96.4 |
| **corridor** | **mf** | **0/20** | **252.1** ⚠️ |
| pillars | fm | 20/23 | 94.9 |
| pillars | mf | 22/23 | 100.9 |
| s_curve | fm | 17/20 | 92.8 |
| s_curve | mf | 22/23 | 97.7 |

The corridor `mf` run was evaluated before the `projection_ms` fix, so **every** projector
cost was booked as generation time: `dpcc-c` reads `fm_ms=269.7, proj_ms=0.0` where true
generation is ~88 ms. Every other run in the batch has the fix.

**Consequence: the corridor `fm_ms` / `proj_ms` split must not be compared across engines.**
`total_ms` is unaffected and is what §2 uses. Reconstruct `mf`'s split as
`proj_ms ≈ total_ms − 88.5` if needed.

### 5.2 Pillars `fm` tripped the projection circuit breaker

Candidate 31 carries **6 CB sentinels** — the only candidate in the batch with any. Six
rollouts ran partly *unprojected* because SLSQP exceeded its budget, so their constraint
numbers describe a policy the variant name does not name. This is exactly why the
`proj_valid` mask exists; batch-wide it drops 694 → 688 rollouts. All numbers above use
`proj_valid`. It also corroborates §3: pillars `fm` is where projection is hardest.

### 5.3 Pillars `fm` `hardflow_new-t` has only 2 of 3 rollouts

The 24 h timeout cut trial 3. The DA carried the partial row through rather than dropping
it, so that one cell is a 2-trial mean — and it is precisely the variant that was `mf`'s
best pillars row. **The pillars HardFlow head-to-head is incomplete.**

### 5.4 The tool agrees with the hand analysis ✅

First real-data run, so worth pinning: `mf` corridor `dpcc-r` reads **S&C 0.80, succ 1.00,
gdist 0.29** here, matching `RUN_REPORT_Gen15_mf_corridor_K10.md` exactly; `dpcc-c` 0.70 and
`dpcc-t` 0.70 also match, as do the `fm` success rates against the 2026-08-17 eval logs.
Discovery, the diagnostics scan and the aggregator are behaving.

---

## 6. Verdict

1. **On corridor, `fm` beats `mf` outright** — Pareto-dominant on the `dpcc` rows (higher
   S&C, lower time, equal success), 11-1-8 across shared variants. Not a trade-off.
2. **`mf` loses on corridor at every projection setting, and unguided by 49× on goal
   distance — but this does not show MeanFlow failing its benchmark bar.** That bar is a
   *low-K* bar (§6.5). An earlier draft of this DA read item 2 as "MeanFlow is not clearing
   the bar it was adopted to clear"; that was wrong and is retracted.
3. **Pillars is the one place `mf` has something `fm` does not** — a HardFlow configuration
   that actually reaches the goal (0.29 m ×3, and the only nonzero strict success on the
   scene). That is a real result and it is the reason not to drop the `mf` arm.
4. **Neither engine solves pillars or s_curve at K=10.** Constraints are feasible but
   harmful on pillars (per DA_20260817 §7), and s_curve's HardFlow arm is void by
   construction. These are scene/projector problems, not objective problems.
5. **The K=10 operating point is the wrong one for this comparison** — see §6.5 below.
6. **Everything here is seed 6 only** (`N_Seeds=1` on all 34 candidates). No variance
   estimate exists. The corridor gap is large and consistent across 11 variants, which is
   reassuring, but the pillars and s_curve readings at n=3 are anecdotes.

### 6.5 🔴 Why `mf` loses here, and why that was predictable

Added 2026-08-19 after the K=10 result prompted "why is mf failing, that doesn't make sense".
It is a fair objection: MeanFlow **beats** FM+DPCC on visual-aligning. The resolution is that
it beats it *at low K*, and this DA measured K=10.

#### The mechanism

`logs_in_develop/Gen3v6_MeanFlow/Study/STUDY_why_mf_works_imf_fails_and_mf_beats_fm_dpcc.md`
— note the title ends **"at low K"** — states the case directly:

> For K = 20: `dt = 0.05`, global error ~ O(0.05). This is small enough to produce good
> results, **which is why FM at K=20 works fine.**

and its headline result is at **K=2**:

> MF at **2 NFE** dominates Diffusion and FM ODE at **10–20 NFE**, with 7–22× lower per-step
> cost.

Plain FM's Euler global error is **O(1/K)**. At K=10 it is ~0.1 and already small. MeanFlow's
two-time objective exists to remove that error; when the error is nearly zero there is nothing
to win, and the objective's extra difficulty becomes pure cost. **Gen15 ran the single K at
which MeanFlow has least to offer.**

#### The cost is real, and measurable in training

`a0_loss` is comparable across engines (both report it at `h=0`), unlike total loss:

| scene | engine | train/a0 | test/a0 | **gap** |
|---|---|---|---|---|
| corridor | **mf** | **0.00302** | 0.02855 | **9.5×** |
| corridor | fm | 0.01065 | **0.02221** | 2.1× |
| pillars | **mf** | 0.00855 | 0.06299 | **7.4×** |
| pillars | fm | 0.00426 | **0.01639** | 3.8× |
| s_curve | mf | 0.0127 | 0.0226 | 1.8× |
| s_curve | fm | 0.00992 | **0.01969** | 2.0× |

On corridor `mf` reaches the **lowest training loss of any run in this batch** and the
**worst generalisation gap**. It is memorising — spending capacity on interval consistency
that K=10 sampling never queries. That is precisely what produces §2.2's `diffuser` row
(unguided goal distance 39.26 m for `mf` against 0.80 m for `fm`).

Both arms trained 100 000 steps. `mf` took 2.7× the wall clock (the JVP), not fewer steps.

#### Ruled out

| suspected cause | finding |
|---|---|
| `dp0.5` = half the data? | No — it is `meanflow_data_proportion`, the r==t anchor fraction, **identical to Gen3v6** |
| wrong time sampling | No — `t_schedule` defaults to `logit_normal` (p_mean −0.4, p_std 1.0), matching Gen3v6; the `ts` token is merely absent from the UAV exp-name |
| optimiser stuck (`loss=1` flat) | No — that is the adaptive-loss normalisation; `mf_adp_p=1.0` makes the reported loss exactly 1 **by construction**. `final_test_loss` 0.93 vs `fm`'s 0.004 compares two different objectives and means nothing |

One real config gap remains: Gen3v6 used `action_weight=10`, UAV uses `aw1`
(deliberate — `config/uav_mix.py:321`). It cannot explain `fm` vs `mf`, since both Gen15 arms
use `aw1`, but it does mean UAV `mf` is not a like-for-like replication of the setup where
MeanFlow won.

#### What would settle it

PLAN §7.3 calls the K sweep "the experiment", and exactly one point of it has ever been run.
The claim — `mf`/`af` hold success where `fm` collapses — lives at K=1–2. **No retraining is
needed**: K is eval-time for both flow arms, so the existing checkpoints cover `{1,2,5,10,20}`.

Corridor is the scene to run it on: it is the only one where both arms succeed at K=10, so a
collapse curve is readable there and is not confounded by pillars/s_curve being unsolved for
unrelated reasons.

**Prediction, recorded before the run:** `fm` degrades sharply below K=5 while `mf` holds. If
`mf` does *not* hold at K=2 on UAV, then the generalisation gap above is the real story and
the problem is the UAV dataset, not MeanFlow.

---

## 7. What to do next

| # | action | why |
|---|---|---|
| **0** | **🔴 K sweep `{1,2,5,10,20}` on corridor, both arms** | **the actual experiment (§6.5); no retraining needed, K is eval-time** |
| 1 | **Re-run corridor `mf` at K=10** with current code | recovers the timing split lost to §5.1 and confirms the S&C gap on a Fix_1 build |
| 2 | **Finish pillars `fm` `hardflow_new-t`** | the one cell blocking the pillars HardFlow verdict (§5.3) |
| 3 | **Seeds 7–10 on corridor, both arms** | the headline claim rests on one seed |
| 4 | Exclude the 25 stale `plans(...)` dirs from auto-scan | 28-of-34 noise, §1 — prune `AUTO_ROOTS` or add a discovery ignore |
| 5 | Teach `parse_axes()` the Gen11 path shape | 28 candidates with blank `engine`, 17 with blank `K`, so the baseline is unslicable |
| 6 | Gate the Pareto front on success | a 0.0%-S&C row and the `empty` scene are both on the front (§1) |
| 7 | Raise pillars `n_trials` to 10 | §3's split rests on 3 trials |

Items 4–6 are DA_UAV_v1 work and belong in `logs_in_develop/DA_Code/DA_UAV_v1/`.
