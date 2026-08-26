# `uav-*` (Gen15 UAV Mix-ML) — env status, per projector, constraint-checked · **PILOT**

> 🚧 **PILOT / UNDER CONSTRUCTION — 2026-08-25.** This is the first whole-env pass over the UAV
> tree. It is written to the same contract as
> [`SNAPSHOT_20260823_visual_aligning_env_status.md`](SNAPSHOT_20260823_visual_aligning_env_status.md)
> and [`SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md`](SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md),
> but the underlying batch is **one seed, n = 10 per cell, one usable scene**. Nothing here is a
> paper number. Regenerated as new batches land; use the newest `SNAPSHOT_<date>_uav_*` file.

**Batch:** `batch_uav_20260825_143155` (DA_UAV_v1 — `temp/2508/…/per_rollout_detail.csv`)
**Scope:** full scan of `logs/UAV_MIX` + `logs/UAV_FM` →
**45 candidates · 623 units · 5 366 rollouts**, spanning Gen11 and Gen15, 4 scenes.
**Deployable core:** **13 candidates · 2 943 rollouts**, all `corridor`, all Gen15, all seed 6.
**Companions:** `logs_in_develop/Gen15/DA/` — [`…af_sit_K_sweep_corridor`](../../logs_in_develop/Gen15/DA/DA_20260824_af_sit_K_sweep_corridor.md)
(the SiT arm) · [`…fm_K_sweep_corridor`](../../logs_in_develop/Gen15/DA/DA_20260820_fm_K_sweep_corridor.md) ·
[`…fm_vs_mf_3scenes_K10`](../../logs_in_develop/Gen15/DA/DA_20260819_fm_vs_mf_3scenes_K10.md).

> ### The three answers up front
>
> **1. What does a constraint-clean corridor cost? → 112 ms/step, and the cost spread across
> configurations is 13×.** §4. Cheapest clean (10/10) cell: `fm` K10 `post_processing`, **111.9 ms**.
> Cheapest cell that is merely *good* (8/10): `af` K2 `dpcc-c`, **33.6 ms** — **3.3× cheaper for two
> rollouts**. Most expensive clean cell: `fm` K20 `hardflow_new`, 1 377 ms. **The projection solve,
> not generation, is 60–90 % of every one of these numbers.**
>
> ⚠️ **`budget_ms` = 30.3 ms is NOT a project target.** It is `1000/DATASET_HZ` — the rate the expert
> data happened to be recorded at — and the eval logs `over_budget_frac` against it purely as a
> reference scale. **FM-PCC has never set real-time deployment as a goal**, and the timing is cluster
> GPU wall-clock, which `behavior_logger.py:184` itself labels *"cluster latency, NOT target drone"*.
> Nothing in this file should be read as a real-time pass/fail. See §0.6.
>
> **2. Do MeanFlow / α-Flow beat naive FM? → YES below K = 5, NO above it. The split is total.**
> §3.1, §3.4. Same arm, same K, same 10 trials, `CF` (n = 10):
>
> | challenger | K1 | K2 | K5 | K10 | K20 | **total** |
> |---|---|---|---|---|---|---|
> | **`mf` vs `fm`** *(matched, 3.97 M vs 3.96 M U-Net)* | **W7 L0** T4 | **W6 L0** T5 | W0 **L10** T1 | W0 **L8** T3 | W0 **L7** T0 | **13 W / 25 L / 13 T** |
> | **`af` vs `fm`** ⚠️ *(10.00 M SiT — not matched)* | **W8 L0** T2 | **W9 L0** T1 | W4 L1 T5 | — | — | **21 W / 1 L / 8 T** |
>
> **Best-arm Pareto verdict vs `fm` at the same K:** `mf` **strictly dominates** at K1
> (0.80 @ 59 ms vs 0.00 @ 101 ms) and K2 (0.80 @ 65 ms vs 0.10 @ 79 ms), is **dominated** at K5 and
> K20, and **loses on cost 6.9×** at K10. `af` **dominates** at K1 and **ties `CF` 1.00 at 2.8× lower
> cost** at K5 (169 ms vs 475 ms).
>
> 🔴 Two things this is **not**. (a) `af` carries **2.53× the parameters**, so engine and capacity
> move together — appendix arm, not the matched claim (§3.3). (b) `mf`'s *unprojected* field reaches
> the goal radius in **0/10 at every K** and leaves the arena in 7–10/10, and its wins sit entirely
> on `dpcc-c` and the HardFlow arms while `dpcc-r` (random selection) never wins at any K — so the
> low-K result is **the selector and the projection, not the generator** (§3.2).
>
> **3. Is this a constrained-control benchmark yet? → Barely.** §2.5. Deleting the obstacle
> geometry (`geo_free`) costs `fm` K20 and `af` K5 **nothing — both stay at 10/10**. Deleting the
> workspace bounds (`bounds_free`) costs `fm` K10/K20 **nothing**. Only the **dynamics** class is
> load-bearing: `model_free` scores **0/10 at every K for every engine**. The corridor scene is
> currently a *trackability* task wearing a constraint task's clothes.

---

## Reporting rules

🔴 **1. No aggregation across projectors.** The unit is the **cell** =
`(candidate × projector × scene × geometry)`, each with its own `n`. There is no "model score".
The batch's own `candidates_ranking.csv` pools every arm into one percentage; **that column is not
used anywhere in this file** (see §6.3 for what it produces).

🔴 **2. Model-vs-model comparisons use each model's OWN best projector, and always name it.**
Best projector is selected on **`CF`** (§0.1), never on raw distance — raw distance is degenerate
here (§0.2).

🔴 **3. HardFlow (arm C) is never mixed into a DPCC (arm B) result**, and HardFlow is compared at
**measured `nfe_effective`, not at K**. Arm C spends **6 K + 2** network evaluations, not K and not
2 K (§5.1) — the tool's own README says 2×; the data says 6×.

🔴 **4. `S` (goal-reached-and-safe) is never quoted alone.** In this env **`CF` ≡ `S&C` ≡
`S&C_relaxed` in all 2 943 Gen15 corridor rollouts** (zero disagreements, either goal criterion) and
**every clean rollout also reached**, while **328 of 1 132 successes (29 %) were not clean**. So `S`
overstates by up to 0.7 per cell, `S&C` adds nothing over `CF`, and **the strict-vs-finish-line
choice adds nothing either** (§0.2). **One column, `CF`, is the instrument.** (§2.6)

🔴 **5. `goal_dist` means are never quoted.** The metric is trimodal by construction (§0.3): 40 %
of rollouts sit at the 0.30 m early-stop radius, 17 % end **outside the arena** (up to 158.7 m).
A mean over that mixture is dominated by *how far the wreck flew*, which is not a quality. Report
`reach`, `gone`, and the **median over in-arena rollouts only**.

🔴 **6. Cross-generation numbers are quarantined.** Every Gen11 candidate's folder says `K20`; its
measured generation time says **K ≈ 10** (§6.2). Gen11 is a *provenance* reference, not an opponent.

🔴 **7. Every row carries `n`, and all main tables are n = 10 cells only.** The 10 trials are the
same 10 `(homotopy, trial_seed)` initial conditions in the same order for every candidate, so
same-scene cells are **paired by `rollout_idx`**. Cells with n < 10 are quarantined in §7.

**Protocol:** seed 6 for every candidate in the batch. Single seed — the replication unit is the
**trial**, i.e. generalisation over initial conditions, not training variance.

---

## 0. Definitions

### 0.1 Metrics

| symbol | field | meaning | dir |
|---|---|---|---|
| **`CF`** | `collision_free_completed` | Fraction of the cell's rollouts with **zero** constraint violations at every step. **The instrument.** Identical to `n_success_and_constraints` in every Gen15 cell. Resolution 1/10. | higher |
| `S` | `n_success` | Goal reached **and** physically safe. Superset of `CF`. **Never quote alone** (rule 4). | higher |
| `reach` | `goal_reached` | **Strict.** Came within `GOAL_RADIUS = 0.30 m` of the route endpoint at some step (latched; triggers the early stop). | higher |
| **`cross`** | `goal_crossed_line` | **Relaxed (U7 finish line).** Latched the first time the drone is ever on the goal side of a vertical plane through the goal, perpendicular to the expert path's final approach heading. `reach ⇒ cross` by construction. **Reported beside `reach`, never instead of it** — see §0.2. | higher |
| `hold` | derived | `reach / cross` — of the rollouts that got to the goal, the fraction that *stayed*. The termination/braking diagnostic (§8.4). | higher |
| `S_rlx` | `success_relaxed` | `cross AND safe`. ⚠️ The canonical column `n_success_relaxed` is **empty in 4 966 of 5 366 rows** — a field-map bug in DA_UAV_v1 (§8.5). Use the raw `success_relaxed`. | higher |
| **`gone`** | *derived here* | Fraction of rollouts ending **> 8.4 m** from goal — outside the arena. The scene XMLs span \|x\| ≤ 3.6, \|y\| ≤ 1.6, z ≤ 2, so 8.4 m is the arena diagonal: past it the vehicle is **not in the room**. **Not a CSV column — computed in this file.** | lower |
| `gd_in` | derived | Median `goal_dist` over **in-arena rollouts only** (≤ 8.4 m). The only defensible distance number. | lower |
| `viol` | `n_violations` | Mean violating steps per rollout. Diagnostic only. | lower |
| **`ms`** | `avg_time_ms` | Wall-clock ms per control step = one plan (K net calls × MPC batch 4) **plus** the projection solve. | lower |
| `gen` / `prj` | `fm_ms` / `proj_ms` | The two halves of `ms`. | lower |
| `ovb` | `over_budget_frac` | Fraction of control steps exceeding `budget_ms` = 30.3 ms. ⚠️ **A reference scale, not a target** — `budget_ms` is `1000/DATASET_HZ`, inherited from the data-collection rate, and the timing is cluster latency. **Never a pass/fail.** See §0.6. | *(descriptive)* |
| **`NFE`** | `nfe_effective` | Measured network evaluations per plan. `= K` on arm B, **`= 6K + 2`** on arm C (§5.1). | lower |
| `match` | derived | Fraction of rollouts whose flown corridor side equals the trial's seeded `homotopy`. Diagnostic — the state-only policy is **never told** the homotopy (§8.2). | — |

### 0.2 🔴 Strict goal-reach vs the finish line — the criterion question, closed

The eval records **two** goal criteria: strict `goal_reached` (ended within 0.30 m) and relaxed
`goal_crossed_line` (ever passed the finish-line plane). The strict one is visibly harder. **It does
not matter for anything ranked in this file**, and the identity below is why.

| criterion | rate over the 2 943 deployable rollouts |
|---|---|
| `goal_reached` (strict, 0.30 m) | 0.4016 |
| **`goal_crossed_line`** (finish line) | **0.5420** — **+14 pp**, 413 extra rollouts |
| `phys_safe` | 0.4152 |
| `n_success` = reached AND safe | 0.3846 |
| `success_relaxed` = crossed AND safe | 0.4128 — only **+2.8 pp** |
| `n_success_and_constraints` | **0.2732** |
| **`n_success_relaxed_and_constraints`** | **0.2732** |
| **`collision_free_completed`** (`CF`) | **0.2732** |

**Two facts settle it.**

1. **The goal test is not the binding gate — `safe` is.** Relaxing the goal adds 14 pp on the goal
   axis and **2.8 pp** on success, because `phys_safe` (0.4152) is what most rollouts fail.
2. 🔴 **`CF` ≡ `S&C` ≡ `S&C_relaxed`, at 0.2732, with zero row-level disagreements.** Every
   constraint-clean rollout **already** cleared the strict 0.30 m radius (0 counterexamples), so
   `clean AND crossed` and `clean AND reached` are the same set. The 413 crossed-but-not-reached
   rollouts are **all** dirty, and the 791 crossed-but-not-clean rollouts were never going to be
   counted under either rule.

**Consequence:** every table in §2–§6, the §3 MeanFlow-vs-FM claim, the §4 cost frontier and
the §5 HardFlow verdict are **byte-identical under either criterion**. The same holds for the Gen11
reference (§6.3): C7's `cross` equals its `reach` at 0.30 on all four DPCC arms.

**Where the finish line *is* worth reading** is the failure anatomy, and this file reports it:
`cross − reach` separates *never got there* from *got there and could not stop*, and it is large
exactly where `CF = 0` (§8.4). That is a real diagnostic and it changes how §2.5 and §3.2 should be
worded — it is **not** a re-ranking.

### 0.3 🔴 Why `goal_dist` is not a distance metric here

Over the 2 943 deployable rollouts:

| band | meaning | share |
|---|---|---|
| ≤ 0.30 m | at the early-stop radius — **reached** | **40.2 %** |
| 0.30–8.4 m | in the arena, short of goal | 42.6 % |
| > 8.4 m | **outside the arena** (12.8 % are past 20 m; max 158.7 m) | **17.2 %** |

The episode *stops* at 0.30 m, so the low band is a hard floor, not a measurement — `min (clean)`,
the primary instrument in the aligning snapshot, is **0.28–0.30 in essentially every cell here and
carries no information**. The high band is a runaway integrator: `p_des += Δp_des` has no clamp, so
a lost policy commands a point tens of metres away (`Div_Abort` fires at arena + 3 m slack, 12 m/s,
or 5 m `p_des` lead — expert cruise is 0.3–0.5 m/s). **59.9 % of rollouts run to the 396-step cap.**

**Consequence:** the aligning snapshot's `dist` / `min (clean)` / `<15 cm clean` triple does not
port. Its replacement is `CF` (already constraint-checked) plus `reach` / `gone` for the failure
anatomy.

### 0.4 Projector arms

| class | variants | role |
|---|---|---|
| **Arm B — DPCC projector** | `dpcc-r` (random) · `dpcc-c` (min projection cost) · `dpcc-t` (temporal consistency) · `post_processing` (post-hoc, no selection), each ± `-tightened` | ✅ reported; model-vs-model uses each model's best of these |
| **Arm C — HardFlow** | `hardflow_new{,-c,-t,-r}` (in-loop NLP) | ✅ reported **separately** (§5), at `nfe_effective` |
| **Arm A — reference** | `diffuser` (unguided, no projection) | ⚪ the raw generator, read as a *generator diagnostic* (§3.2) |
| **Study-only** | `geo_free` · `bounds_free` · `model_free` + pairs (constraint-class **removal**) · `gradient` | ❌ not controllers — but §2.5 is the most informative table in this file |

### 0.5 Trial pool (what is paired with what)

10 trials per cell, `rollout_idx` 0–9, cycling homotopy **L, C, R, L, C, R, L, C, R, L** — so the
pool is **4 L / 3 C / 3 R**, mildly L-weighted, and **identical across every candidate**. Same-scene
cells are therefore paired by `rollout_idx`. `corridor` is the only scene with n = 10; `pillars` and
`s_curve` ran n = 3 (§7).

### 0.6 ⚠️ `budget_ms` / `over_budget_frac` — what this is NOT

`budget_ms = 1000 / control_hz` and `control_hz = DATASET_HZ = 33`
(`mix_uav_test/behavior_logger.py:67`). That is **the rate the UAV expert dataset was recorded at**,
carried into the eval so the replay loop matches how the data was collected. It is **not** a
requirement anyone set, and **real-time deployment is not and has never been a goal of FM-PCC**.

Two independent reasons no row in this file may be read as a real-time verdict:

1. **The number is a data-collection artefact.** 33 Hz is where the expert trajectories came from.
   Nothing selected it as a control-latency spec.
2. **The measurement is the wrong hardware.** The logger's own line is
   `real_time_safe=... (measured on {node} — cluster latency, NOT target drone)`
   (`behavior_logger.py:184`). These are shared-GPU wall-clocks with an SLSQP solve in the loop, on
   a machine that is not a flight controller.

**So `ovb` is reported here as a descriptive column only**, and `ms` is compared **between
configurations**, never against 30.3. Where a "× budget" ratio appears it is a **readability scale
for the spread**, not a score. An earlier draft of this file framed §4 as a "real-time gate" and led
with "is any configuration real-time? → No"; that was wrong on both counts above and has been
removed.

---

## 1. Roster

### 1.1 Deployable — Gen15, `corridor`, n = 10 (the whole of §2–§5)

| C | engine | backbone | params | K | arm B | arm C | model folder |
|---|---|---|---|---|---|---|---|
| 33/35/36/32/34 | **`fm`** naive Flow Matching (Gen11 FMv3ODE) | U-Net | **3 955 177 (3.96 M)** | 1/2/5/10/20 | ✅ | ✅ | `mix_uav_fm/…FlowMatchingODE_9D` |
| 38/40/41/37/39 | **`mf`** MeanFlow (Gen3v6) | U-Net | **3 969 222 (3.97 M)** | 1/2/5/10/20 | ✅ | ✅ (K20 ✗) | `mix_uav_mf/…MeanFlowODE_9D_dp0.5_bbunet` |
| 29/30/31 | **`af`** α-Flow (Gen3v7) | **SiT** | **10 003 654 (10.00 M)** ⚠️ | 1/2/5 | ✅ | ✅ | `mix_uav_af/…AlphaFlowODE_9D_as1_ae0_bbsit` |

⚠️ **`af` is not parameter-matched** (2.53×). SiT sizes from `dit_hidden_size`, not `freq_dim`;
`config/uav_mix.py:401-405` declares it the **deferred appendix arm**, never the
architecture-matched claim. The matched pair is **`fm` vs `mf`**, and that is where §3.1 puts the claim.

ℹ️ `dp0.5` on the `mf` rows is `meanflow_data_proportion` — the fraction of the training batch
forced to `r == t` (the FM-anchor ablation axis from Gen3v6). It is **not** a data-subset; `mf` sees
the same dataset as `fm` and `af`.

### 1.2 Everything else → §7

`pillars` / `s_curve` Gen15 (C42–C45, **n = 3**) · `empty` (C9, ill-defined goal) · 28 Gen11
candidates across 12 legacy `plans(<tag>)` trees, **all K-mislabelled** (§6.2), 21 of them with no
timing at all.

---

## 2. Cell tables — Gen15 `corridor`, geo `bounds+dynamics+geo_bounds+halfspace+obstacles`, n = 10

Sorted by K within engine. **Read `CF`, `gone`, `ms`, `ovb`.** `NFE` is measured, not assumed.
`reach` (strict, 0.30 m) and **`cross`** (the finish line) are both shown: they rank identically —
`CF` is unchanged by the choice (§0.2) — but their **gap is the termination diagnostic** (§8.4).

#### A. Arm B — DPCC, untightened

| C | engine | K | **projector** | NFE | **`CF`** | `S` | `reach` | **`cross`** | **`gone`** | `gd_in` | **`ms`** | `gen` | `prj` | **`ovb`** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 33 | `fm` | 1 | `dpcc-r` | 1 | **0.00** | 0.00 | 0.00 | 0.20 | 0.10 | 2.54 | 204.7 | 9.0 | 195.7 | 0.74 |
| 33 | `fm` | 1 | `dpcc-c` | 1 | **0.00** | 0.00 | 0.00 | 0.20 | 0.30 | 2.50 | 107.2 | 9.1 | 98.2 | 0.68 |
| 33 | `fm` | 1 | `dpcc-t` | 1 | **0.00** | 0.00 | 0.00 | 0.20 | 0.30 | 1.67 | 118.0 | 9.2 | 108.9 | 0.69 |
| 33 | `fm` | 1 | `post_processing` | 1 | **0.00** | 0.00 | 0.00 | 0.10 | 0.20 | 2.01 | 101.5 | 9.0 | 92.5 | 0.71 |
| 35 | `fm` | 2 | `dpcc-c` | 2 | **0.00** | 0.70 | 0.70 | 0.90 | 0.20 | 0.29 | 69.1 | 17.3 | 51.8 | 1.00 |
| 35 | `fm` | 2 | `dpcc-t` | 2 | **0.00** | 0.50 | 0.60 | 1.00 | 0.20 | 0.30 | 78.9 | 17.4 | 61.5 | 1.00 |
| 35 | `fm` | 2 | `dpcc-r` | 2 | **0.00** | 0.20 | 0.20 | 0.70 | 0.50 | 1.44 | 100.6 | 17.3 | 83.3 | 1.00 |
| 35 | `fm` | 2 | `post_processing` | 2 | **0.10** | 0.50 | 0.50 | 0.70 | 0.30 | 0.29 | 78.9 | 17.4 | 61.5 | 1.00 |
| 36 | `fm` | 5 | **`dpcc-r`** | 5 | **0.90** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 164.7 | 43.7 | 121.0 | 1.00 |
| 36 | `fm` | 5 | `dpcc-c` | 5 | **0.90** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 167.0 | 43.7 | 123.2 | 1.00 |
| 36 | `fm` | 5 | `dpcc-t` | 5 | **0.90** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 163.8 | 43.4 | 120.4 | 1.00 |
| 36 | `fm` | 5 | `post_processing` | 5 | **0.80** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | **70.7** | 43.5 | **27.2** | 1.00 |
| 32 | `fm` | 10 | `dpcc-r` / `-c` / `-t` | 10 | **1.00** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 240.8–241.9 | 85.0 | 155.8–156.9 | 1.00 |
| 32 | `fm` | 10 | **`post_processing`** | 10 | **1.00** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | **111.9** | 84.6 | **27.3** | 1.00 |
| 34 | `fm` | 20 | `dpcc-r` / `-c` / `-t` | 20 | **1.00** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 414.4–415.5 | 170.7 | 243.7–244.8 | 1.00 |
| 34 | `fm` | 20 | `post_processing` | 20 | **1.00** | 1.00 | 1.00 | 1.00 | 0.00 | 0.30 | 196.0 | 170.0 | 26.0 | 1.00 |
| 38 | `mf` | 1 | **`dpcc-c`** | 1 | **0.80** | 0.80 | 0.80 | 0.80 | 0.00 | 0.30 | **58.9** | 9.5 | 49.4 | **0.64** |
| 38 | `mf` | 1 | `dpcc-t` | 1 | 0.30 | 0.40 | 0.40 | 0.60 | 0.30 | 0.29 | 80.1 | 9.5 | 70.6 | 0.77 |
| 38 | `mf` | 1 | `dpcc-r` | 1 | 0.00 | 0.30 | 0.30 | 0.40 | 0.30 | 2.25 | 90.5 | 9.5 | 81.0 | 0.82 |
| 38 | `mf` | 1 | `post_processing` | 1 | 0.10 | 0.30 | 0.30 | 0.40 | 0.20 | 1.90 | 93.0 | 9.5 | 83.5 | 0.81 |
| 40 | `mf` | 2 | **`dpcc-c`** | 2 | **0.80** | 0.80 | 0.80 | 0.90 | 0.20 | 0.30 | 64.8 | 18.1 | 46.7 | 1.00 |
| 40 | `mf` | 2 | `dpcc-t` | 2 | 0.20 | 0.50 | 0.50 | 0.60 | 0.40 | 0.29 | 93.8 | 18.3 | 75.5 | 1.00 |
| 40 | `mf` | 2 | `dpcc-r` | 2 | 0.10 | 0.20 | 0.20 | 0.20 | 0.40 | 2.41 | 111.6 | 18.5 | 93.1 | 1.00 |
| 41 | `mf` | 5 | `dpcc-c` | 5 | 0.60 | 0.80 | 0.90 | 0.90 | 0.00 | 0.29 | 224.7 | 45.0 | 179.7 | 1.00 |
| 41 | `mf` | 5 | `dpcc-r` / `-t` | 5 | 0.50 | 0.70 | 0.70–0.80 | 0.80 | 0.00–0.10 | 0.30 | 238.3 / 269.7 | 45.0 | 193.3 / 224.6 | 1.00 |
| 37 | `mf` | 10 | **`dpcc-r`** / `-t` | 10 | **0.80** | 0.90 | 0.90 | 0.90–1.00 | 0.00 | 0.29 | 297.7 / 297.1 | 89.5–94.4 | 203.3 / 207.6 | 1.00 |
| 37 | `mf` | 10 | `dpcc-c` | 10 | 0.60 | 0.90 | 0.90 | 0.90 | 0.00 | 0.30 | 339.4 | 90.1 | 249.3 | 1.00 |
| 39 | `mf` | 20 | **`dpcc-c`** | 20 | **0.90** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 452.8 | 179.0 | 273.8 | 1.00 |
| 39 | `mf` | 20 | `dpcc-t` / `-r` | 20 | 0.80 / 0.70 | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 453.5 / 454.5 | 179.1 | 274.4 / 275.4 | 1.00 |
| 29 | `af` ⚠️ | 1 | `dpcc-c` / `-t` | 1 | 0.40 | 0.70 / 0.50 | 0.70 / 0.50 | 0.70–0.80 | 0.00 | 0.30 / 0.79 | 44.0 / 82.4 | 6.4 / 7.5 | 37.5 / 75.0 | **0.28** / 0.47 |
| 29 | `af` ⚠️ | 1 | `dpcc-r` | 1 | 0.20 | 0.50 | 0.60 | 0.60 | 0.00 | 0.29 | 68.8 | 6.4 | 62.4 | 0.44 |
| 30 | `af` ⚠️ | 2 | **`dpcc-c`** | 2 | **0.80** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | **33.6** | 12.1 | **21.5** | **0.17** |
| 30 | `af` ⚠️ | 2 | `dpcc-t` | 2 | 0.70 | 0.80 | 0.90 | 1.00 | 0.00 | 0.29 | 47.0 | 12.2 | 34.8 | **0.33** |
| 30 | `af` ⚠️ | 2 | `dpcc-r` | 2 | 0.40 | 0.50 | 0.50 | 0.70 | 0.00 | 1.06 | 91.7 | 12.2 | 79.6 | 0.53 |
| 31 | `af` ⚠️ | 5 | **`dpcc-r`/`-c`/`-t`** | 5 | **1.00** | 1.00 | 1.00 | 1.00 | 0.00 | 0.29 | 169.2–169.5 | 30.1 | 139.1–139.2 | 1.00 |
| 31 | `af` ⚠️ | 5 | `post_processing` | 5 | 0.70 | 0.70 | 0.70 | 0.80 | 0.00 | 0.30 | 78.3 | 30.1 | 48.2 | 1.00 |

#### B. Arm B — DPCC, `-tightened` 🔴 tightening is a *net loss* in this env

| C | engine | K | projector | `CF` untight → **tight** | `ms` untight → tight |
|---|---|---|---|---|---|
| 36 | `fm` | 5 | `dpcc-r` | 0.90 → **0.30** | 164.7 → 280.5 |
| 36 | `fm` | 5 | `dpcc-c` | 0.90 → **0.30** | 167.0 → 264.9 |
| 32 | `fm` | 10 | `dpcc-r` | 1.00 → **0.70** | 241.8 → 281.2 |
| 32 | `fm` | 10 | `dpcc-c` | 1.00 → **0.60** | 241.9 → 279.0 |
| 34 | `fm` | 20 | all three | 1.00 → **1.00** *(only K where tightening is free)* | 414–416 → 416–417 |
| 41 | `mf` | 5 | `dpcc-r`/`-c`/`-t` | 0.50–0.60 → **0.00** | 224–270 → 316–364 |
| 37 | `mf` | 10 | `dpcc-r` | 0.80 → **0.20** | 297.7 → 511.8 |
| 31 | `af` ⚠️ | 5 | all three | 1.00 → **0.30** | 169 → 314–339 |
| 32/34 | `fm` | 10/20 | `post_processing` | 1.00 → **0.30** | 111.9/196.0 → 173.7/274.6 |

**This inverts the aligning result**, where `-tightened` was the geometry on which constraint
numbers became trustworthy (`0-viol` 0.73→1.00). Here tightening costs **0.3–1.0 of `CF`** and adds
**15–70 % to `ms`** at nearly every operating point. `projection_backstop_hits = 0` and
`projection_cb_trips = 0` in **every** corridor cell, so this is **not** an infeasibility fallback
firing — the tightened solve returns a plan, and that plan is worse. **Mechanism unestablished; do
not guess it in a report.** The discriminating run is §10 item 4.

#### C. Arm C — HardFlow (reported at measured NFE, never at K)

| C | engine | K | **NFE** | best arm C | `CF` | `ms` | best arm B at **similar NFE** | its `CF` | its `ms` |
|---|---|---|---|---|---|---|---|---|---|
| 33 | `fm` | 1 | **8** | `hardflow_new-t` | 0.00 | 1 101 | `fm` K10 `post_processing` (NFE 10) | **1.00** | **111.9** |
| 35 | `fm` | 2 | **12** | `hardflow_new` | 0.10 | 514 | `fm` K10 `dpcc-t` (NFE 10) | **1.00** | 240.8 |
| 36 | `fm` | 5 | **32** | `hardflow_new{,-c,-t}` | **1.00** | 475–493 | `fm` K20 `post_processing` (NFE 20) | **1.00** | **196.0** |
| 32 | `fm` | 10 | **60** | `hardflow_new{,-c,-t}` | **1.00** | 720–756 | `fm` K10 `post_processing` (NFE 10) | **1.00** | **111.9** |
| 34 | `fm` | 20 | **120** | `hardflow_new{,-c}` | **1.00** | 1 360–1 377 | `fm` K10 `post_processing` (NFE 10) | **1.00** | **111.9** |
| 38 | `mf` | 1 | **8** | `hardflow_new-c` | **0.50** | 1 016 | `mf` K1 `dpcc-c` (NFE 1) | **0.80** | **58.9** |
| 40 | `mf` | 2 | **12** | `hardflow_new-c` | 0.50 | 460 | `mf` K2 `dpcc-c` (NFE 2) | **0.80** | **64.8** |
| 41 | `mf` | 5 | **32** | `hardflow_new-c`/`-t` | **0.80** | 480–501 | `mf` K20 `dpcc-c` (NFE 20) | **0.90** | 452.8 |
| 37 | `mf` | 10 | **60** | `hardflow_new{,-c}` | **1.00** | 772–780 | `mf` K20 `dpcc-c` (NFE 20) | 0.90 | 452.8 |
| 29 | `af` ⚠️ | 1 | **8** | `hardflow_new-t` | 0.40 | 173.9 | `af` K1 `dpcc-c` (NFE 1) | 0.40 | **44.0** |
| 30 | `af` ⚠️ | 2 | **12** | `hardflow_new-t` | **0.90** | 163.9 | `af` K2 `dpcc-c` (NFE 2) | 0.80 | **33.6** |
| 31 | `af` ⚠️ | 5 | **32** | `hardflow_new-c`/`-t` | **1.00** | 381–383 | `af` K5 `dpcc-c` (NFE 5) | **1.00** | **169.2** |

#### D. 🔴 Study-only ablations — the most informative table here (`CF`, n = 10)

The last column carries `model_free`'s **`cross`** beside its `CF`, because the two disagree and the
disagreement is the whole story (§2.5 item 1).

| engine | K | `dpcc-r`*(full)* | `bounds_free` | `geo_free` | **`model_free` `CF`** | *its* `cross` | `gradient` |
|---|---|---|---|---|---|---|---|
| `fm` | 1 | 0.00 | 0.00 | 0.00 | **0.00** | 0.20 | 0.00 |
| `fm` | 2 | 0.00 | 0.70 | 0.00 | **0.00** | 0.10 | 0.00 |
| `fm` | 5 | 0.90 | 0.90 | 0.20 | **0.00** | 0.00 | 0.00 |
| `fm` | 10 | 1.00 | **1.00** | 0.40 | **0.00** | 0.00 | 0.00 |
| `fm` | 20 | 1.00 | **1.00** | **1.00** | **0.00** | 0.00 | 0.00 |
| `mf` | 5 | 0.50 | 0.70 | 0.30 | **0.00** | **0.50** | 0.00 |
| `mf` | 10 | 0.80 | 0.50 | 0.50 | **0.00** | **0.50** | 0.00 |
| `mf` | 20 | 0.70 | 0.60 | 0.60 | **0.00** | 0.20 | 0.00 |
| `af` ⚠️ | 2 | 0.40 | 0.70 | 0.30 | **0.00** | **1.00** 🔴 | 0.00 |
| `af` ⚠️ | 5 | 1.00 | **1.00** | **1.00** | **0.10** | **0.90** | 0.00 |

### 2.5 What the ablations say about the benchmark

1. **`model_free` = 0/10 at every K for every engine** (one exception: `af` K5, 1/10). Removing the
   dynamics-consistency class does not merely relax safety — it **removes the controller's ability
   to fly legally**. Those rollouts stay in the arena (`gone` 0.00–0.80, `gd_in` ≈ 5–8 m) and
   accumulate **338 violating steps** on average with `phys_safe = 0.00`: the projected plan is no
   longer trackable by the PID layer, so the vehicle drifts and grounds (`phys_min_z` 0.10 m).
   **In this env the DPCC projector is not a safety filter bolted onto a policy — it is part of the
   policy.**

   🔴 **But "destroys the controller" is too blunt, and the finish line says why.** `af` K2
   `model_free` **crosses the goal line in 10/10** (`S_rlx` 0.70) while scoring `CF` 0.00; `mf` K5
   and K10 `model_free` cross in 5/10 from `reach` 0.00. So without the dynamics class the vehicle
   still **gets to the goal** — it just gets there illegally and cannot hold. The dynamics
   projection is what makes the trajectory **legal and terminable**, not what makes it goal-directed.
   For `fm` the blunt reading does hold (`cross` 0.00–0.20).
2. **`geo_free` — obstacles are free at the top of the ladder.** `fm` K20 and `af` K5 score **10/10
   with the obstacle geometry switched off entirely**. Once the generator is good the corridor walls
   are simply never approached.
3. **`bounds_free` — workspace bounds are free** for `fm` at K10/K20 (1.00 → 1.00).
4. **`gradient` guidance is dead: 0/10 in all 30 cells**, at every K, for all three engines.

🔴 **Consequence for the paper story.** The strong DPCC claim ("projection buys constraint
satisfaction the generator cannot") is **not demonstrable on `corridor` as configured**, because at
the operating points where anything works, the geometric constraints are not binding. What *is*
demonstrated is a different and weaker claim: **the dynamics-consistency projection is what makes a
learned trajectory flyable.** Say that one; it is supported.

### 2.6 What the constraint check changes

🔴 **Numbers that do not survive:**

| claim | raw | **constraint-checked** |
|---|---|---|
| "`fm` K2 `dpcc-c` reaches the goal in 7/10" | `S` 0.70 | **`CF` 0.00** — not one legal rollout. |
| "`fm` K2 `dpcc-t` **passes the finish line 10/10**" | `cross` 1.00 | **`CF` 0.00** — the most permissive goal rule available still yields nothing legal. |
| "`fm` K5 unprojected reaches 9/10" | `S` 0.90 | **`CF` 0.00** — the raw field reaches and violates on 124.6 steps/rollout. |
| "`af` K2 unprojected **passes the finish line 10/10**" | `cross` 1.00 | **`CF` 0.00** — and `reach` is only 0.30: it crosses and overshoots (§8.4). |
| "`af` K1 `dpcc-c` reaches 7/10" | `S` 0.70 | **`CF` 0.40**. |
| batch ranking's top row (C9, 100 %) | `S&C` 1.00 | **`CF` = `nan`** — `empty` records no constraint group at all (§7.2). |

**Relaxing the goal rule rescues none of these** (§0.2): `CF` ≡ `S&C_relaxed` exactly, so every row
above reads identically under `cross`. The finish line changes the *diagnosis*, never the verdict.

**What survives:** `fm` K10 `post_processing` and `fm` K10/K20 `dpcc-{r,c,t}` (**`CF` 1.00**);
`af` K5 `dpcc-{r,c,t}` and `hardflow_new-{c,t}` (**1.00**); `mf` K10 `hardflow_new{,-c}` (**1.00**).
Those six cells are the only 10/10 in the batch.

---

## 3. 🔴 The headline question — MeanFlow vs naive Flow Matching, matched backbone

### 3.1 The architecture-matched claim

`fm` (3.96 M U-Net) vs `mf` (3.97 M U-Net), **each at its own best arm-B projector**, paired over
the same 10 trials. Both trained by Gen15 `mix_uav`, same data, same 9-D schema, same PID stop-go
controller.

| K | NFE | **`mf` best arm** | **`mf` `CF`** | **`fm` best arm** | **`fm` `CF`** | Δ (`mf` − `fm`) | `ms` `mf` vs `fm` |
|---|---|---|---|---|---|---|---|
| **1** | 1 | `dpcc-c` | **0.80** | `dpcc-{r,c,t}`/`post` | **0.00** | **+0.80** | **58.9 vs 107.2 → 1.8× cheaper** |
| **2** | 2 | `dpcc-c` | **0.80** | `post_processing` | 0.10 | **+0.70** | 64.8 vs 78.9 |
| 5 | 5 | `dpcc-c` | 0.60 | `dpcc-{r,c,t}` | **0.90** | −0.30 | 224.7 vs 164.7 |
| 10 | 10 | `dpcc-{r,t}` | 0.80 | `post_processing` | **1.00** | −0.20 | 297.7 vs **111.9** |
| 20 | 20 | `dpcc-c` | 0.90 | `post_processing` | **1.00** | −0.10 | 452.8 vs **196.0** |

**Reading.** MeanFlow wins the **low-NFE regime decisively** — at K = 1 it is the only engine in the
matched pair that flies at all (8/10 vs 0/10) and it is **1.8× cheaper per control step**, because
its plans are cheaper to project (49.4 ms vs 98.2 ms of solve). At K ≥ 5 naive FM overtakes it on
both axes. `fm` is **monotone in K** (0.00 · 0.00/0.10 · 0.90 · 1.00 · 1.00); `mf` is **not**
(0.80 · 0.80 · 0.60 · 0.80 · 0.90) — at n = 10 that non-monotonicity is 1–3 rollouts and may be noise.

**This is the benchmark-hierarchy claim that matters** (MF must beat naive FM), and it holds only at
K ≤ 2. It should be quoted as *"at one and two network evaluations, MeanFlow flies the corridor and
naive Flow Matching does not"* — never as a general MeanFlow-beats-FM statement.

### 3.2 🔴 The caveat that reframes it: MeanFlow's generator does not *stop*

The **unprojected** arm (`diffuser`) is a pure generator diagnostic. Read `cross` (did the field ever
aim at the goal?) against `reach` and `gone` (did it stay there?):

| engine | metric | K1 | K2 | K5 | K10 | K20 |
|---|---|---|---|---|---|---|
| `fm` | `cross` | 0.50 | 0.90 | **1.00** | 0.70 | 0.40 |
| `fm` | `reach` | 0.00 | 0.30 | **0.90** | 0.40 | 0.40 |
| `fm` | `gone` | 1.00 | 0.50 | **0.00** | 0.00 | 0.00 |
| **`mf`** | **`cross`** | **0.20** | **0.20** | **0.30** | **0.50** | **0.30** |
| **`mf`** | **`reach`** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| **`mf`** | **`gone`** | **0.70** | **0.90** | **1.00** | **1.00** | **1.00** |
| `af` ⚠️ | `cross` | 0.10 | **1.00** | 0.90 | — | — |
| `af` ⚠️ | `reach` | 0.00 | 0.30 | 0.60 | — | — |
| `af` ⚠️ | `gone` | 0.30 | **0.00** | 0.00 | — | — |

**The finish line sharpens the diagnosis rather than softening it.** MeanFlow-U-Net's raw field
**crosses the goal plane in 2–5 of 10 rollouts** — so it is not aimless — and then **ends outside the
arena in 10/10 at every K ≥ 5** (mean `total_violations` 9 152 at K10). Its `reach` is **0.00 at
every K under either rule**: `hold` = 0.00 across the board. **The failure is termination, not
direction** — `p_des` is a free-running integrator (§8.1) and MeanFlow's `Δp_des` never goes to zero,
so the command point sails through the goal and keeps going.

So the 8/10 at K = 1 is still **not** the MeanFlow objective producing a *usable* trajectory; it is
`dpcc-c`'s **min-projection-cost selector** picking one candidate from a fan of 4 and the dynamics
projection supplying the braking the field lacks. The tell is the selector spread at K = 1:
`dpcc-c` **0.80** vs `dpcc-t` 0.30 vs `dpcc-r` **0.00** — a 0.80 range across selection rules alone,
where `fm` at K ≥ 5 shows 0.00 spread (all three arms identical). **When the fan is good the selector
is irrelevant; when the fan is garbage the selector is the policy.**

Corroboration in §2.5: `mf`'s `model_free` `CF` is **0.00 at every K** while its `cross` is
**0.50** at K5/K10 — same signature, the goal is found and not held.

**Honest statement for a report:** *"At K = 1–2 the MeanFlow arm is the only matched-parameter
configuration that completes the corridor, but its unprojected field crosses the goal and then
diverges in 7–10 of 10 rollouts and reaches the goal radius in none, so the result measures the
projector's ability to terminate a plan, not the generator's ability to produce one. Fixing the
MeanFlow UAV checkpoint — specifically its failure to drive `Δp_des` to zero near the goal — is a
prerequisite to claiming the objective."*

### 3.3 The α-Flow appendix arm ⚠️

`af` @ SiT is **10.00 M against 3.96 M** — engine and capacity move together, so **neither is
creditable alone**. Recorded because it is the only arm whose *generator* works at low NFE
(`diffuser` `gone` 0.00 and `S` 0.30 at K = 2, against `fm` 0.50/0.30 and `mf` 0.90/0.00), and
because it produces the batch's cheapest usable cell (33.6 ms at 8/10, §4). `config/uav_mix.py:401-405` already
declares it the deferred appendix arm. **There is no `af` @ U-Net corridor run**, so the backbone
cannot be isolated. Same framing as the D3IL side.

### 3.4 The full same-arm sweep — every arm, every K

§3.1 gives each engine its own best projector (rule 2 — the deployment question). This is the
**stricter** reading: both sides forced onto the *same* arm, so the projector cannot explain the
difference. 31 cells per challenger, `CF`, n = 10 each, paired over the same 10 trials.
**W / L / T is from the challenger's side.**

#### `mf` vs `fm` — matched backbone (3.97 M vs 3.96 M U-Net)

| arm | K1 | K2 | K5 | K10 | K20 |
|---|---|---|---|---|---|
| `dpcc-r` | 0.00 v 0.00 T | **0.10 v 0.00 W** | 0.50 v 0.90 L | 0.80 v 1.00 L | 0.70 v 1.00 L |
| **`dpcc-c`** | **0.80 v 0.00 W** | **0.80 v 0.00 W** | 0.60 v 0.90 L | 0.60 v 1.00 L | 0.90 v 1.00 L |
| `dpcc-t` | **0.30 v 0.00 W** | **0.20 v 0.00 W** | 0.50 v 0.90 L | 0.80 v 1.00 L | 0.80 v 1.00 L |
| `post_processing` | **0.10 v 0.00 W** | 0.10 v 0.10 T | 0.30 v 0.80 L | 0.00 v 1.00 L | 0.30 v 1.00 L |
| `dpcc-r-tightened` | 0.00 v 0.00 T | 0.00 v 0.00 T | 0.00 v 0.30 L | 0.20 v 0.70 L | 0.10 v 1.00 L |
| `dpcc-c-tightened` | **0.10 v 0.00 W** | **0.50 v 0.00 W** | 0.00 v 0.30 L | 0.30 v 0.60 L | 0.40 v 1.00 L |
| `dpcc-t-tightened` | 0.00 v 0.00 T | 0.00 v 0.00 T | 0.00 v 0.30 L | 0.20 v 0.70 L | — |
| `hardflow_new` | **0.10 v 0.00 W** | 0.10 v 0.10 T | 0.50 v 1.00 L | 1.00 v 1.00 T | — |
| `hardflow_new-c` | **0.50 v 0.00 W** | **0.50 v 0.00 W** | 0.80 v 1.00 L | 1.00 v 1.00 T | — |
| `hardflow_new-t` | **0.40 v 0.00 W** | **0.20 v 0.00 W** | 0.80 v 1.00 L | 0.90 v 1.00 L | — |
| `diffuser` *(no projector)* | 0.00 v 0.00 T | 0.00 v 0.00 T | 0.00 v 0.00 T | 0.00 v 0.00 T | 0.00 v 0.40 L |
| **tally** | **W7 L0 T4** | **W6 L0 T5** | **W0 L10 T1** | **W0 L8 T3** | **W0 L7 T0** |

**13 W / 25 L / 13 T overall — and the split is clean at K = 5.** Below it MeanFlow loses **zero**
cells; from it upward it wins **zero**. There is no arm on which it wins at high K and none on which
it loses at low K.

🔴 **Where the wins live matters.** `dpcc-r` — random candidate selection — is the **only arm
MeanFlow never wins on at any K**. Every win is on `dpcc-c` (min projection cost), `dpcc-t`
(temporal consistency) or an in-loop HardFlow arm. All of those *choose* or *repair*; `dpcc-r` does
neither. Same evidence as §3.2, from a different direction: at low K MeanFlow supplies a candidate
fan, and something downstream of the generator turns it into a controller.

#### `af` vs `fm` ⚠️ — **not** parameter-matched (10.00 M SiT vs 3.96 M U-Net)

| arm | K1 | K2 | K5 |
|---|---|---|---|
| `dpcc-r` | **0.20 v 0.00 W** | **0.40 v 0.00 W** | **1.00 v 0.90 W** |
| `dpcc-c` | **0.40 v 0.00 W** | **0.80 v 0.00 W** | **1.00 v 0.90 W** |
| `dpcc-t` | **0.40 v 0.00 W** | **0.70 v 0.00 W** | **1.00 v 0.90 W** |
| `post_processing` | **0.20 v 0.00 W** | **0.40 v 0.10 W** | 0.70 v 0.80 **L** |
| `dpcc-r-tightened` | 0.00 v 0.00 T | **0.10 v 0.00 W** | 0.30 v 0.30 T |
| `dpcc-c-tightened` | **0.30 v 0.00 W** | **0.30 v 0.00 W** | 0.30 v 0.30 T |
| `dpcc-t-tightened` | **0.20 v 0.00 W** | **0.20 v 0.00 W** | 0.30 v 0.30 T |
| `hardflow_new-c` | **0.20 v 0.00 W** | **0.70 v 0.00 W** | 1.00 v 1.00 T |
| `hardflow_new-t` | **0.40 v 0.00 W** | **0.90 v 0.00 W** | 1.00 v 1.00 T |
| `diffuser` *(no projector)* | 0.00 v 0.00 T | 0.00 v 0.00 T | **0.30 v 0.00 W** |
| **tally** | **W8 L0 T2** | **W9 L0 T1** | **W4 L1 T5** |

**21 W / 1 L / 8 T — one single losing cell in the whole sweep** (`post_processing` at K5), and
unlike `mf` it wins on `dpcc-r` too, and on the unprojected `diffuser` arm at K5. That last one is
the only place in the batch where an engine beats another **with no projector at all**. It is also
the arm where the 2.53× parameter confound is least excusable — see §3.3.

#### Best-arm Pareto verdict, same K

Each engine at its own best arm; the cheapest arm achieving that `CF`.

| K | `mf` best | `fm` best | verdict | `af` ⚠️ best | verdict |
|---|---|---|---|---|---|
| 1 | **0.80 @ 59 ms** `dpcc-c` | 0.00 @ 101 ms | 🟢 **strictly dominates** | **0.40 @ 44 ms** `dpcc-c` | 🟢 **strictly dominates** |
| 2 | **0.80 @ 65 ms** `dpcc-c` | 0.10 @ 79 ms | 🟢 **strictly dominates** | 0.90 @ 164 ms `hardflow_new-t` | 🟡 +0.80 `CF` at 2.1× cost — trade-off |
| 5 | 0.80 @ 480 ms `hardflow_new-c` | **1.00 @ 475 ms** | 🔴 **dominated** | **1.00 @ 169 ms** `dpcc-c` | 🟢 **ties `CF`, 2.8× cheaper** |
| 10 | 1.00 @ 772 ms `hardflow_new-c` | **1.00 @ 112 ms** | 🔴 ties `CF`, **6.9× costlier** | not run | — |
| 20 | 0.90 @ 453 ms `dpcc-c` | **1.00 @ 196 ms** | 🔴 **dominated** | not run | — |

**Both readings agree.** Same-arm and best-arm give the same crossover at K = 5 for `mf`, and the
same "wins everywhere it has been run" for `af`.

### 3.5 The same question in the other two environments

For context only — different tasks, different metrics, different sample sizes. Sources are the two
companion snapshots in this folder.

| env | matched? | `mf` vs naive FM | `af` vs naive FM |
|---|---|---|---|
| **`avoiding-d3il`** *(5 seeds × 30 eps — the strongest data we have)* | ✅ MF-UNet 4.0 M | K20: **11.2× cheaper** at equal S&C 1.000. Matched K = 2, **best-arm**: wins both axes (−14.7 steps, −0.32 s/ep). Matched K = 2, **same-arm `dpcc-c`**: **loses** (0.833 vs 1.000) | K1 **41.7×** vs the DPCC target; K10 is the study's **only strict Pareto domination** |
| **`aligning-d3il-visual`** *(1 seed, n = 30 paired)* | ✅ bone-matched | `mf` K2 vs `fm` K20: **0.2867 vs 0.3617 m**, t = −2.21, clean tail **10 % vs 0 %**, **20× cheaper** → win | `af` K2 vs `fm` K20: distance tie (ns), clean tail **17 % vs 0 %**, 20× cheaper → win on tail + cost |
| **`uav` corridor** *(this file, 1 seed, n = 10)* | ✅ MF-UNet 3.97 M | **Dominates at K ≤ 2, dominated at K ≥ 5** (§3.4) | **21 W / 1 L**; dominates at K1, ties `CF` at 2.8× lower cost at K5 |

🔴 **The pattern is identical in all three: the few-step engines win at low NFE and lose or tie at
high NFE.** UAV is the only env where the **crossover itself** is visible, because it is the only
one with a matched-K ladder on both sides — `aligning` never ran `fm` at K = 2 (§4 of that file),
and `avoiding` has multi-seed naive FM only at K20.

🔴 **And in the one env with a real target, the win is narrower than any of this suggests:** on
`avoiding`, MF-UNet beats the published diffusion-DPCC baseline on **one axis only**, and at matched
K = 10 **no flow engine beats diffusion-DPCC on cost at all**. On UAV there is no diffusion
checkpoint, so the ceiling of every claim above is *"beats naive FM"* (§6.1).

---

## 4. Compute cost per control step — and where it goes

⚠️ **Read §0.6 first.** `budget_ms = 30.3` is `1000/DATASET_HZ`, the expert data-collection rate;
it is **not a target**, real-time is **not a project goal**, and these are cluster wall-clocks on
non-flight hardware. The `× 30.3` column below is a **readability scale for the spread**, not a
score, and `ovb` is descriptive. **The comparison that means something is configuration vs
configuration.**

| rank | cell | `CF` | **`ms`** | × 30.3 *(scale only)* | `ovb` |
|---|---|---|---|---|---|
| — | *`budget_ms` reference line — not a target* | — | *30.3* | *1.0×* | *0.00* |
| 1 | **`af` ⚠️ K2 `dpcc-c`** | **0.80** | **33.6** | **1.1×** | **0.17** |
| 2 | `af` ⚠️ K2 `dpcc-t` | 0.70 | 47.0 | 1.6× | 0.33 |
| 3 | `af` ⚠️ K1 `dpcc-c` | 0.40 | 44.0 | 1.5× | 0.28 |
| 4 | `mf` K1 `dpcc-c` | 0.80 | 58.9 | 1.9× | 0.64 |
| 5 | `mf` K2 `dpcc-c` | 0.80 | 64.8 | 2.1× | 1.00 |
| 6 | `fm` K5 `post_processing` | 0.80 | 70.7 | 2.3× | 1.00 |
| 7 | **`fm` K10 `post_processing`** | **1.00** | **111.9** | **3.7×** | 1.00 |
| 8 | `af` ⚠️ K5 `dpcc-{r,c,t}` | **1.00** | 169.2 | 5.6× | 1.00 |
| 9 | `fm` K20 `post_processing` | **1.00** | 196.0 | 6.5× | 1.00 |
| 10 | `fm` K10 `dpcc-t` | **1.00** | 240.8 | 7.9× | 1.00 |

**The cost–quality frontier has two ends and a gap in the middle:**

- **Cheapest clean (10/10):** `fm` K10 `post_processing`, **111.9 ms**.
- **Cheapest good (8/10):** `af` K2 `dpcc-c`, **33.6 ms** — **3.3× cheaper, for two rollouts**.
- Between them nothing exists. Whether that trade is worth taking is a project decision, not a
  measurement — and it does **not** hinge on the 30.3 ms line (§0.6).
- **Cost spread: 7.2× across the ten rows above** (33.6 → 240.8 ms), **12.4× across all of §2A**
  (33.6 → 415.5 ms), **41× including arm C** (33.6 → 1 377 ms).

**Where the time goes.** Generation is *not* the bottleneck below K = 10: at K = 2, `gen` is
12–18 ms and `prj` is 21–93 ms. **The projection solve is 60–90 % of `ms` at every
operating point.** And it is **not a constant** — at identical K and identical geometry, `prj`
ranges 21.5 ms (`af` K2 `dpcc-c`) to 93.1 ms (`mf` K2 `dpcc-r`), a **4.3× spread driven entirely by
which engine produced the plan**. See §8.3.

**Consequence:** the useful low-NFE claim on this env is a **relative-cost** claim, not an accuracy
claim, and the lever that moves it most is the **NLP solve**, not K. Stated as a ratio between
configurations it survives the §0.6 caveat intact; stated against 30.3 ms it does not, because that
line is not a target and the hardware is not the target hardware.

---

## 5. Does HardFlow beat the DPCC projectors? — answered once

### 5.1 🔴 First fix the accounting

`nfe_effective`, measured per unit, is **6 K + 2** on every arm-C variant: K1 → **8**, K2 → **12**,
K5 → **32**, K10 → **60**, K20 → **120**. `Data_Analysis/DA_UAV_v1/README.md` states arm C
"evaluates the network **twice** per ODE step" — **the data says six times plus two.** Until that is
reconciled, **quote the measured `nfe_effective` column, not the README's factor, and not K.**

### 5.2 The verdict

**No — with one exception, and the exception is the broken engine.**

- **On `fm` (working generator):** arm C never wins. `fm` K5 hardflow (**NFE 32**, 475 ms) reaches
  1.00 — but `fm` K10 `post_processing` (**NFE 10**, 111.9 ms) also reaches 1.00, at **1/3 the NFE
  and 1/4 the wall-clock**. At the low end arm C is worse outright: NFE 12 → `CF` 0.10 against arm
  B's 1.00 at NFE 10.
- **On `af` (working generator):** arm C ties at K5 (both 1.00) at **6.4× the NFE and 2.3× the ms**;
  at K2 it edges arm B 0.90 vs 0.80 for **6× NFE and 4.9× ms**.
- **On `mf` (diverging generator):** arm C is the only thing that reaches **1.00** (K10, NFE 60,
  772 ms) where arm B tops out at 0.80–0.90. **This is the exception, and it says what §3.2 says —
  in-loop projection can rescue a bad field.** It is a statement about `mf`'s checkpoint, not about
  HardFlow's merit.

**One line:** *in-loop projection buys robustness exactly where the generator has failed, and buys
nothing where it has not — at 6× the network budget in both cases.*

---

## 6. The baseline problem

### 6.1 🔴 There is no diffusion-DPCC UAV checkpoint

`PLAN §1.5` and the batch's own NOTES both state it: **no Gen11 diffusion-engine UAV run exists.**
The standing benchmark hierarchy (diffusion-DPCC is THE baseline; MF/AF must also beat naive FM)
therefore **cannot be applied on this env**. The strongest available claim is *"vs naive FM +
DPCC"*, never *"beats DPCC"*. Producing a Gen15 `diffusion`-engine corridor candidate is the single
highest-value missing run (§10 item 1).

### 6.2 🔴 And the Gen11 rows are K-mislabelled — the timing proves it

Every Gen11 folder is tagged `K20` because `_load_base_cfg` injects `flow_steps_v3` from the plan
block, a key that did not exist in Gen11. Measured generation time settles what actually ran:

| generation | nominal K | measured `gen` (ms) |
|---|---|---|
| Gen15 `fm` | 1 / 2 / 5 / **10** / 20 | 9.1 / 17.6 / 44.1 / **85.7** / 171.9 |
| **Gen11, every candidate** (C2, C7, C9, C16, C22, C28) | "K20" | **82.6 – 94.0** |

**Every Gen11 candidate's generation cost sits on the Gen15 K = 10 line, not the K = 20 line.**
Read Gen11 as **K ≈ 10** throughout. This is a **provenance** correction, not a re-analysis: the
Gen11 rows stay quarantined either way (§7.1).

### 6.3 What the Gen11 corridor reference actually shows

C7 (`logs/UAV_FM/uav-corridor/plans/…/K20_mpc4_pid_stopgo_T0.5`) is the one non-fork Gen11 corridor
tree, same geo tag, same 10 trials:

| C | gen | arm | `CF` | `S` | `reach` | `viol` | `gd` med | `ms` |
|---|---|---|---|---|---|---|---|---|
| **7** | Gen11 (K ≈ 10) | `dpcc-{r,c,t}` | **1.00** | 0.30 | 0.30 | **0.0** | 0.51–0.58 | 251 |
| **32** | Gen15, K10 | `dpcc-{r,c,t}` | **1.00** | **1.00** | **1.00** | 0.0 | 0.29 | 241 |

🔴 **Note the shape difference.** In Gen11, `CF` = 1.00 while `S` = 0.30: its failures are
**clean-but-short** — the drone stops ~0.55 m out, half a body-length past the 0.30 m radius, with
**zero violations**. In Gen15, `CF` ≡ `S&C` ≡ reach and failures carry a **median 325 violating
steps**. Same geometry, same trials, different failure mode entirely.

**The gap between C7 (0.30) and C32 (1.00) at the same effective budget is unexplained and is not
attributed here.** Candidate causes, none tested: a different training run; the Gen15 eval-loop
changes (`Div_Abort`, the `executed_idx` fix on `dpcc-t`, PID stop-go); different MPC batch
handling. **Until it is attributed, no Gen15 number may be credited to the engine on the strength of
a Gen11 comparison.** §10 item 2.

---

## 7. Quarantine — nothing in §2–§6.3 uses these

### 7.1 Gen11 legacy trees — 28 candidates, 2 159 rollouts

Twelve parallel `plans(<tag>)` forks (`(E6)`, `(E7_U1)`, `(E7_U3)`, `(Bf_U8)`, `(Bf_DC-FIX)`,
`(Bf_Fix14)`, `(no_GIF)`, `(with_gif_parts)`, `(U1)` …) — development iterations, not curated
candidates. All K-mislabelled (§6.2); **21 units carry no timing at all**; several (C2, C9) predate
the constraint group in the npz and report `CF = nan`. Keep for provenance; rank on nothing.

### 7.2 🔴 `empty` (C9) — the ranking's #1 row is an artifact

`candidates_ranking.csv` puts C9 at the top with **100 % success + constraints, 117 ms, PARETO
FRONT**. It is n = 5, and:

- `empty` has a **random per-episode start/goal the state-only policy is never told**, so goal-reach
  is ill-defined there — the eval's own note says its success means *stable, safe flight only*.
- Its `CF` is **`nan` on every arm** — the run records no constraint group.
- Every arm scores 1.00, including `diffuser` with no projector.

**It is 5 rollouts of a drone hovering safely in an empty box.** Excluded from every table above,
and the reason rule 1 forbids the pooled ranking column.

### 7.3 `pillars` and `s_curve` — n = 3, and zero either way

| C | cell | best arm | `CF` | `ms` | note |
|---|---|---|---|---|---|
| 42 | `pillars` `fm` K10 | — | **0.00** on all 11 arms | 267–5 967 | 🔴 **circuit breaker tripped on 6 units**, `cb` 8.7 % — those rollouts ran partly **unprojected** |
| 43 | `pillars` `mf` K10 | `hardflow_new` reaches 3/3 | **0.00** | 926–2 009 | reaches the goal, never legally |
| 44 | `s_curve` `fm` K10 | — | **0.00** | 252–1 008 | |
| 45 | `s_curve` `mf` K10 | — | **0.00** | 172–8 765 | `gd` med **177–241 m** — gone |
| 16/22/28 | Gen11 `pillars`/`s_curve` | — | **0.00** | 137–8 036 | |

**Both hard scenes are at zero for every engine, every arm, both generations.** `pillars` also
breaks the solver: `prj` reaches **4 049–5 827 ms** — **15–22× the corridor's own projection cost**,
and 134–192× the `budget_ms` reference line (§0.6) — and it is
the only place the sustained-slowness breaker fires. n = 3 supports no claim in either direction —
**but nothing in the batch suggests these scenes are close.**

### 7.4 Sub-n cells inside the deployable block

`fm` K20 `hardflow_new-t` (**n = 1**) and `mf` K20 `dpcc-t-tightened` (**n = 2**) are truncated;
both are shown in §2 only where explicitly marked and support nothing.

---

## 8. Cross-cutting diagnostics

### 8.1 Failure is divergence, not paralysis

The aligning env's characteristic failure is the box **never moving** (`still` 83–100 % on the
baseline). Here it is the opposite: **17.2 % of deployable rollouts end outside the arena**, 12.8 %
beyond 20 m, one at 158.7 m. `p_des` is a free-running integrator with no clamp, so a lost policy
commands a point that runs away and the vehicle chases it. `Div_Abort` bounds the *episode*, not the
metric. **A UAV DA must report `gone` explicitly; a distance summary that pools runaways is
meaningless.**

### 8.2 Mode-following is at chance — but the policy was never told the mode

`match` (flown corridor side == the trial's seeded homotopy) across the arm-B cells: **0.10–0.90,
median ≈ 0.30**, against a 1/3 chance rate. Only `fm` K10 clears it (0.90 on `dpcc-{r,c,t}`).

🔴 **This is not disobedience.** The observation is 9-D `[p_des | p | v]` — **state-only**. The
`homotopy` label names the expert route the trial was *seeded from*; the policy never sees it. So
`match` measures whether the learned field happens to reproduce the seeded mode, and a mismatch is
the policy choosing its own. **The consequence is that the corridor task's multi-modality — the
entire reason D3IL-style benchmarks exist — is currently untested on the UAV side.** Either
condition the policy on the mode, or stop calling the homotopies a multi-modality result.

Per-homotopy `CF` on the 10/10 cells is uniform (1.00/1.00/1.00), so the 4 L / 3 C / 3 R pool
imbalance (§0.5) does not distort those rows; it may distort the partial cells, where the L quarter
carries 40 % of the weight.

### 8.3 Projection cost is a plan-quality measurement

At identical K, identical geometry and identical solver, `prj` varies with the engine that produced
the plan:

| K | `af` ⚠️ `dpcc-c` | `fm` `dpcc-c` | `mf` `dpcc-c` |
|---|---|---|---|
| 1 | **37.5** | 98.2 | 49.4 |
| 2 | **21.5** | 51.8 | 46.7 |
| 5 | **139.1** | 123.2 | 179.7 |
| 10 | — | **156.9** | 249.3 |
| 20 | — | **244.8** | 273.8 |

A plan that is nearly feasible is cheap to project; a plan that is far outside the feasible set
makes SLSQP work. **`proj_ms` is therefore a free, continuous proxy for raw-plan quality** — more
informative than `diffuser` `CF`, which is 0.00 almost everywhere and cannot rank anything. Worth
promoting to a first-class reported metric.

### 8.4 `cross − reach` is the overshoot / termination diagnostic

`reach` asks *did it stop at the goal*; `cross` asks *did it ever get there*. The gap is the
braking failure, and it is large **exactly where `CF = 0`** — which is why the strict rule costs
nothing in ranking (§0.2) and everything in diagnosis.

| cell | `CF` | `reach` | **`cross`** | Δ | reading |
|---|---|---|---|---|---|
| `af` ⚠️ K2 `diffuser` | 0.00 | 0.30 | **1.00** | **+0.70** | raw field flies the whole corridor, then overshoots — `gone` 0.00, it stays in the room |
| `af` ⚠️ K2 `model_free` | 0.00 | 0.30 | **1.00** | **+0.70** | goal found without the dynamics class; nothing legal or terminal (§2.5) |
| `fm` K2 `diffuser` | 0.00 | 0.30 | 0.90 | +0.60 | |
| `fm` K1 `diffuser` | 0.00 | 0.00 | 0.50 | +0.50 | crosses, then `gone` 1.00 |
| `mf` K10 `diffuser` | 0.00 | 0.00 | 0.50 | +0.50 | crosses, then `gone` 1.00 — the §3.2 signature |
| `mf` K5/K10 `model_free` | 0.00 | 0.00 | 0.50 | +0.50 | |
| `fm` K2 `dpcc-t` | 0.00 | 0.60 | **1.00** | +0.40 | **every** rollout crosses; none is legal |
| `mf` K1 `dpcc-t` / `-r-tightened` | 0.30 / 0.00 | 0.40 / 0.00 | 0.60 / 0.30 | +0.20 / +0.30 | |
| **every 10/10 cell** (`fm` K10/K20, `af` K5, `mf` K10 HF) | **1.00** | **1.00** | **1.00** | **+0.00** | at the top of the ladder the two rules coincide exactly |

**Pattern:** Δ is 0.00 in every cell that works and 0.10–0.70 in cells that do not. A configuration
that is failing is usually failing *at the goal*, not before it. `hold` (= `reach / cross`) on the
unprojected arm summarises it: `fm` 0.00 / 0.33 / **0.90** / 0.57 / 1.00 across K1–K20, `af` 0.00 /
0.30 / 0.67, **`mf` 0.00 at every K**.

**Report both columns.** Quoting `reach` alone makes several arms look directionally lost when they
are not; quoting `cross` alone makes them look nearly solved when they are not.

### 8.5 Data quality

27 of 623 units flagged. **21 = NO TIMING**, all Gen11 legacy trees (§7.1) — those units have no
time axis at all. **6 = circuit-breaker sentinels**, all on C42 `pillars` `fm` K10, `cb_tripped` 1/3
per unit: those rollouts ran partly on the **unprojected** trajectory, so their constraint numbers
describe a policy the variant name does not name. **Zero flagged units in the entire deployable
corridor block** — `projection_cb_trips` and `projection_backstop_hits` are **0.0 in all 130
corridor cells**. The §2 tables are clean.

---

## 9. Limits

- **One seed (6), n = 10 per cell.** `CF` resolves to 1/10. Every "Δ 0.20" is two rollouts. **No
  bootstrap, no CI, no significance test is reported in this file** — at n = 10 with one seed none
  would mean anything. The aligning snapshot's paired `t`/sign tests are deliberately **not** ported.
- **One usable scene.** `corridor` only. `pillars` and `s_curve` are n = 3 and at zero (§7.3);
  `empty` is not a task (§7.2).
- **No held-out split.** The DA writes `split=test`, but every corridor trial is drawn from the same
  route family the model trained on. There is no analogue of the aligning env's disjoint Test-30.
- **Architecture confound on `af`**: 10.00 M SiT vs 3.96/3.97 M U-Nets, 2.53×. Appendix arm by
  design; no `af` @ U-Net exists.
- **`mf`'s generator diverges** at every K (§3.2), so every `mf` row is a projector result. Its
  same-arm wins are confined to selecting/repairing arms (`dpcc-c`, `dpcc-t`, HardFlow); it never
  wins on `dpcc-r` at any K (§3.4).
- **§3.5 quotes the other two envs from their own snapshots**, at different tasks, metrics and
  sample sizes. It is context, not a pooled result — nothing there was recomputed here.
- **No diffusion baseline** (§6.1). The hierarchy cannot be applied.
- **Gen11 is K-mislabelled** (§6.2) and its Gen11-vs-Gen15 gap is unattributed (§6.3).
- **HardFlow NFE accounting disagrees with its own README** (§5.1).
- **`ms` is wall-clock on shared cluster GPUs** and includes the NLP solve; it is also
  **state-dependent** (§8.3), so it is not a clean per-engine constant. `behavior_logger.py:184`
  labels it *"cluster latency, NOT target drone"* — it supports **relative** comparisons between
  configurations only, never an absolute latency statement.
- ⚠️ **No real-time / deployability claim is made anywhere in this file.** `budget_ms` = 30.3 ms is
  `1000/DATASET_HZ`, an artefact of the expert data-collection rate, and real-time is not a project
  goal (§0.6). `ovb` is descriptive.
- **No smoothness, jerk or curvature metric** is recorded — same gap as the D3IL side.
- **`gone`, `gd_in` and `hold` are defined in this file**, not in the CSVs. The 8.4 m cutoff is the
  arena diagonal from the scene XMLs, chosen here; it is not a pipeline constant.
- **The goal criterion is settled, not assumed** (§0.2): strict `goal_reached` and relaxed
  `goal_crossed_line` give **identical** constraint-checked numbers (`CF` ≡ `S&C` ≡ `S&C_relaxed`,
  0 disagreements), so no result here depends on the harder rule. They differ only in failure
  anatomy (§8.4). Both are reported.

---

## 10. What this pilot needs next, in priority order

1. **A Gen15 `diffusion`-engine corridor candidate.** Without it there is no baseline and no
   hierarchy claim on this env (§6.1). `logs_in_develop/Gen15/U3/` already has the arm.
2. **Attribute the Gen11 → Gen15 corridor gap** (0.30 → 1.00 at the same effective budget, §6.3).
   Re-evaluate the Gen11 checkpoint under the Gen15 eval loop; that one run separates "new
   checkpoint" from "new eval".
3. **Fix or retire the `mf` UAV checkpoint.** Its unprojected field leaves the arena in 10/10
   rollouts at K ≥ 5 (§3.2). Until then no MeanFlow *objective* claim is available from this env.
4. **Explain the tightening inversion** (§2B): `-tightened` costs 0.3–1.0 `CF` here and helped on
   aligning, with zero backstop hits and zero breaker trips. Log per-class violation attribution and
   the tightened-solve residual.
5. **Multi-seed the corridor block** — seeds 7–10 at K ∈ {1, 2, 5, 10, 20} for `fm` and `mf`. n = 10
   × 1 seed cannot separate 0.80 from 1.00.
6. **`af` @ U-Net corridor**, to isolate backbone from objective (§3.3).
7. **A `corridor-hard` geometry.** With `geo_free` and `bounds_free` both free at the working
   operating points (§2.5), the current scene does not test constrained control. Narrow the walls or
   move the obstacles into the nominal path until `geo_free` costs something.
8. **Reconcile the HardFlow NFE factor** (§5.1) in `DA_UAV_v1/README.md` against the measured 6 K + 2.
9. **Promote `proj_ms` to a reported plan-quality metric** (§8.3), and add `gone` / `gd_in` to the
   aggregator so this file stops deriving them by hand.
10. **Decide what the homotopies are for** (§8.2): condition on the mode, or stop reporting `match`.

---

## 11. Verdict

**The env is at an earlier stage than either D3IL env.** One scene works, one seed ran, ten trials
per cell, and the two hard scenes are at zero for every engine and every projector.

What is genuinely established, at n = 10 and one seed:

- ✅ **`corridor` is solvable.** Six cells reach **10/10 constraint-clean**: `fm` K10 and K20 on
  `post_processing` and `dpcc-{r,c,t}`, `af` K5 on `dpcc-{r,c,t}` and `hardflow_new-{c,t}`, `mf` K10
  on `hardflow_new{,-c}`.
- ✅ **The low-NFE story has real content, and it is now measured on every arm** (§3.4): at K ≤ 2
  MeanFlow beats matched-parameter naive FM on **13 arms and loses 0**, and strictly Pareto-dominates
  it at K1 and K2; α-Flow goes **21 W / 1 L** across K1–K5. **From K = 5 up MeanFlow wins 0 of 25.**
  The crossover at K = 5 is the single most reproducible result in this batch — but §3.2/§3.4 say it
  measures the selector and the projection, not the objective, and `af` carries 2.53× the params.
- ✅ **The measurement apparatus is trustworthy on the deployable block:** zero breaker trips, zero
  backstop hits, zero timing gaps, 130/130 corridor cells clean.
- ⚖️ **Cost spans 13× across working configurations** (33.6 → 415 ms/step), and the projection solve
  is 60–90 % of every one of them. Cheapest clean cell 111.9 ms; cheapest 8/10 cell 33.6 ms.
  **No real-time claim is made or implied** — `budget_ms` is a data-rate artefact measured on cluster
  hardware, not a project goal (§0.6).
- 🔴 **The constraint benchmark is not yet a constraint benchmark.** Obstacles and bounds are free
  at the operating points that work; only the dynamics class is load-bearing.
- 🔴 **There is no baseline to beat**, and the previous generation's numbers are mislabelled and
  unattributed.

**One line:** *the UAV corridor is solved by three engines and by none of them in real time; the
low-NFE win is currently a projector result, not a generator result; and the env needs a diffusion
baseline and a binding geometry before any of it can be claimed.*

---

## 12. Reproduction

Candidate IDs are **this batch only** — they do not transfer between CSVs. (They happen to match
`batch_uav_20260824_091511` for C29–C31, and do **not** match `batch_uav_20260821_105229`.)

| config | ID | config | ID |
|---|---|---|---|
| `fm` K1/2/5/10/20 | C33/C35/C36/C32/C34 | `mf` K1/2/5/10/20 | C38/C40/C41/C37/C39 |
| `af` ⚠️ K1/2/5 | C29/C30/C31 | Gen11 corridor ref | **C7** (K ≈ 10) |
| Gen15 `pillars` `fm`/`mf` K10 | C42/C43 | Gen15 `s_curve` `fm`/`mf` K10 | C44/C45 |
| `empty` (quarantined) | C9 | Gen11 `pillars`/`s_curve` | C16/C22/C28 |

```
batch : Data_Analysis/DA_UAV_v1/main_da_batch.py --parent-path logs/UAV_MIX,logs/UAV_FM
        (cluster: sbatch Slurm_Codes/sbatch/DA/run_da_batch_uav.sh)
file  : per_rollout_detail.csv        # one row per rollout — everything here is a groupby of it
filter: generation == 'Gen15', scene == 'corridor', seed == 6,
        geo == 'corridor_bounds+dynamics+geo_bounds+halfspace+obstacles'
cell  : (Candidate, variant)          # n = 10, rollout_idx 0..9, paired across candidates
value : CF     -> mean(collision_free_completed)
                  [identical to n_success_and_constraints AND to
                   n_success_relaxed_and_constraints - 0 disagreements, see 0.2]
        reach  -> mean(goal_reached)                 # strict: ended within GOAL_RADIUS 0.30 m
        cross  -> mean(goal_crossed_line)            # relaxed: U7 finish line; reach => cross
        hold   -> reach / cross                      # termination quality, see 8.4
        S_rlx  -> mean(success_relaxed)              # RAW column: n_success_relaxed broken, 8.5
        gone   -> mean(goal_dist > 8.4)              [derived here; arena diagonal, not a column]
        gd_in  -> median(goal_dist | goal_dist <= 8.4)
        ms/gen/prj -> mean(avg_time_ms / fm_ms / proj_ms)   [JSON-sourced; NOT in the npz]
        ovb    -> mean(over_budget_frac)             # DESCRIPTIVE ONLY, see 0.6
        NFE    -> uav_units_long.csv, metric == 'nfe_effective', the n=1 rows
scales: GOAL_RADIUS 0.30 m · corridor max_episode_length 396
        budget_ms 30.3 = 1000/DATASET_HZ -> a REFERENCE line, not a gate or target (see 0.6);
        timing is cluster wall-clock, "NOT target drone" per behavior_logger.py:184
        arena from scene XML |x|<=3.6, |y|<=1.6, z<=2  ->  gone cutoff 8.4 m
no statistics are computed: one seed, n = 10 — see §9.
```
