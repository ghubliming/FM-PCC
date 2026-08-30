# DA — Gen15 UAV `pillars` K-sweep (fm / mf / af), Slurm log audit

**Date:** 2026-08-30
**Source:** `temp/3008/2026-08-27/`, `temp/3008/2026-08-28/`, `temp/3008/2026-08-30/`
(+ cross-check against `temp/3008/batch_uav_20260830_110536/` DA CSVs)
**Scope:** the 11 UAV eval jobs in this log drop — 10 × `pillars`, 1 × `s_curve`.
**Method:** Slurm logs read directly (banner, git rev, abort lines, per-variant summary,
TIMING); numbers cross-checked against `per_rollout_detail.csv` where both exist. No
cluster access, no re-run.

---

## 0. TL;DR

1. 🔴 **The `fm` K-ladder is split across a HardFlow solver swap.** K1/K2 ran on **IPOPT**,
   K5/K20 on **SLSQP**. The `hardflow_*` rows of that ladder are not a clean K sweep.
2. 🔴 **`pillars` produced 0 success+constraint rollouts out of 1707** — every engine, every
   K, every projector variant. Only **2/1707** rollouts are collision-free. No ranking is
   possible from this batch.
3. 🔴 **Two jobs were killed at the 24 h wall**: `25130` (fm K20, 4 of 17 variants done) and
   `25134` (diffusion s_curve, 8 of 10). Their rows are partial, not comparable.
4. 🟡 **The divergence abort fires on 14–78 % of trials**, and **83 % of those are `inverted`** —
   the one v2 trigger that shipped with zero calibration.
5. 🟢 **The abort code itself did NOT drift.** The v2 guard block is byte-identical across
   every rev in this drop, so aborts are at least internally consistent.
6. 🟡 **No `diffusion` (DPCC/GaussianDiffusion) arm exists for `pillars` at all** — the pinned
   DA target is missing for this scene.

---

## 1. Job inventory and code drift

The user's warning was right: the code moved under the sweep. Revs read from each log banner.

| job | engine | K | git rev | commit | HF NLP backend | outcome |
|---|---|---|---|---|---|---|
| 25127 | fm | 1 | `1897f4f` | K-SWEEP pipeline script | **IPOPT** | ok, 17 variants |
| 25128 | fm | 2 | `1897f4f` | ” | **IPOPT** | ok, 17 variants |
| 25129 | fm | 5 | `2c3c38d` | SLSQP artifact renaming | **SLSQP** | ok, 17 variants |
| 25130 | fm | 20 | `2c3c38d` | ” | n/a (never reached HF) | 🔴 **24 h TIME LIMIT** |
| 25131 | mf | 1 | `60f1b13` | `HFFM_SOLVERS` hotfix | SLSQP | ok, 17 variants |
| 25132 | mf | 2 | `60f1b13` | ” | SLSQP | ok, 17 variants |
| 25133 | mf | 5 | `60f1b13` | ” | SLSQP | ok, 17 variants |
| 25135 | af | — | `60f1b13` | ” | — | train, ok |
| 25136 | af | 1 | `60f1b13` | ” | SLSQP | ok, 17 variants |
| 25137 | af | 2 | `60f1b13` | ” | SLSQP | ok, 17 variants |
| 25138 | af | 5 | `60f1b13` | ” | SLSQP | ok, 17 variants |
| 25134 | diffusion | plan-block | `60f1b13` | ” | — | 🔴 **24 h TIME LIMIT** (s_curve) |

### 1.1 What actually changed between the revs

`1897f4f → 2c3c38d` (fires between fm **K2** and fm **K5**):

| file | change | affects results? |
|---|---|---|
| `mix_uav/sampling/hardflow_projection.py` | +181 lines: scipy **SLSQP** NLP backend added alongside IPOPT | 🔴 **YES** for `hardflow_*` |
| `mix_uav_test/eval_mix_uav.py` | +21: `artifact_variant_label(variant, resolve_nlp_backend())` — output dir carries the backend; two new summary fields | naming/grouping only |
| `mix_uav/sampling/projection.py` | +6: `self.last_solve_success` list | ❌ no — explicitly behaviour-neutral, DPCC still keeps `res.x` on non-convergence |

`2c3c38d → 60f1b13` (fires between fm K5/K20 and mf/af): **no UAV code** — docs, figures,
`MASTER_TEST_HISTORY.md`, and `mix_visual_avoiding` only.

**So: `diffuser` and `dpcc-*` rows are on behaviourally identical UAV code across the whole
drop. Only `hardflow_*` drifted.** Confirmed by the log text itself:

```
25127/25128:  [hardflow][NLP-FAILURE] first non-converged solve at tau=1.000.
              Falling back to IPOPT's last iterate ...
25129+:       [hardflow][NLP-FAILURE] first non-converged SLSQP solve at tau=1.000.
              Keeping scipy's last iterate ...
25133:        [hardflow][NLP-BACKEND] slsqp (scipy SLSQP via DPCC Projector — IPOPT built but idle)
```

The 08-27 solver bench (`25121`, same drop) measured the two backends as producing
**different projections** — `‖Π_IPOPT − Π_SLSQP‖` mean 2.87 / max 6.59 on the iterate regime,
with 7–17 / 50 IPOPT non-convergences. This is not a cosmetic swap.

> **Verdict:** `hardflow_new*` at fm K1/K2 and at fm K5 are two different projectors.
> Do not plot them as one ladder. The mf and af ladders are internally clean (all SLSQP).

### 1.2 What did NOT drift — the divergence abort

Guard-block md5 (`SCENE_FLIGHT_ENVELOPE` … end of `_check_divergence`):

```
1897f4f  3dd440f4a620      60f1b13  3dd440f4a620
2c3c38d  3dd440f4a620      HEAD     3dd440f4a620      WORKTREE 3dd440f4a620
```

Identical everywhere, and it is **v2** (`off_map` / `off_route` / `overspeed` / `inverted`;
`p_des_runaway` absent; slack 2.0 m, speed 6.0 m/s). Every job in this drop ran the same
guard. Aborts are comparable across the sweep.

---

## 2. The divergence abort in production — first real measurement

v2 shipped with "zero measurement" (see `aggregated_divergence_abort/
CHANGELOG_20260827_div_abort_v2_scene_envelope.md` §2.1). This is that measurement.

| job | eng | K | aborts | `inverted` | `off_route` | `overspeed` | `off_map` | `nan` | median abort step (frac of budget) |
|---|---|---|---|---|---|---|---|---|---|
| 25127 | fm | 1 | 97/170 (57 %) | 91 | 6 | 0 | 0 | 0 | 310/634 (0.49) |
| 25128 | fm | 2 | 58/170 (34 %) | 56 | 2 | 0 | 0 | 0 | 295/634 (0.47) |
| 25129 | fm | 5 | 61/170 (36 %) | 60 | 1 | 0 | 0 | 0 | 353/634 (0.56) |
| 25130 | fm | 20 | 31/40 (78 %) | 31 | 0 | 0 | 0 | 0 | 376/634 (0.59) |
| 25131 | mf | 1 | 56/170 (33 %) | 34 | 22 | 0 | 0 | 0 | 392/634 (0.62) |
| 25132 | mf | 2 | 57/170 (34 %) | 41 | 14 | 2 | 0 | 0 | 369/634 (0.58) |
| 25133 | mf | 5 | 79/170 (46 %) | 57 | 17 | 5 | 0 | 0 | 260/634 (0.41) |
| 25136 | af | 1 | 29/170 (17 %) | 21 | 7 | 1 | 0 | 0 | 384/634 (0.61) |
| 25137 | af | 2 | 23/170 (14 %) | 16 | 6 | 1 | 0 | 0 | 434/634 (0.68) |
| 25138 | af | 5 | 34/170 (20 %) | 24 | 10 | 0 | 0 | 0 | 385/634 (0.61) |
| 25134 | diffusion | — | 34/40 (85 %) | 34 | 0 | 0 | 0 | 0 | 404/871 (0.46) |
| **total** | | | **559** | **465 (83 %)** | **85 (15 %)** | **9 (2 %)** | **0** | **0** | |

### 2.1 `inverted` is carrying the guard

465/559. `off_map` and `nan_state` never fired once. Tilt at the moment of firing:

| tilt past vertical | n | cum |
|---|---|---|
| 90–92° | 104 | 22 % |
| 92–96° | 134 | 51 % |
| 96–102° | 122 | 77 % |
| 102–110° | 68 | 93 % |
| 110–120° | 19 | 100 % |

Max observed 120°. **This is the shape a crossing detector should have** — it fires the
instant `cos_tilt` goes negative, so the sample is the crossing distribution, not evidence of
marginality. Physically, past 90° a quadrotor's thrust has a downward world-z component and
the attitude loop is already commanding the flip, so recovery is not expected.

⚠️ **But that is reasoning, not measurement.** Because we abort at the crossing, this batch
contains **zero evidence** about what happens after 90°. If you want that evidence, the check
is one run with `inverted` disabled and the tilt logged — not a threshold change.

### 2.2 `off_route` is almost all the z-ceiling

Nearly every `off_route` line is `on z` at **p_z ≈ 3.33–3.38** against the 3.30 m trigger
(`pillars` envelope `[-5.2,-3.11,-1.3]..[5.2,3.11,3.3]`). The mf `diffuser` rows hit it at
**FM step 47–83** — the drone rockets straight up out of the scene in the first ~2 s. Two
`off_route` firings were `on x` at 5.21 vs the 5.20 limit.

`overspeed` fired 9 times, e.g. `|v|=6.10 m/s` vs the 6.0 cap — i.e. right at the threshold.

> **Read:** the envelope numbers are not obviously wrong, but `off_route`/`overspeed` are both
> firing within ~1 % of their thresholds, so they are effectively binary on this data. The
> abort's behaviour here is dominated by `inverted`.

---

## 3. Results — `pillars`, all engines

### 3.1 The finding that governs everything else

Over **1707** Gen15 `pillars` rollouts:

| metric | count |
|---|---|
| `success_strict_and_constraints` | **0 / 1707** |
| `collision_free_completed` | **2 / 1707** |
| `goal_reached` | 412 / 1707 (24 %) |
| `phys_safe` | 658 / 1707 (39 %) |
| `n_violations` per rollout | median **130**, mean 227, max 613 (of ≤634 steps) |

The drone reaches the goal in a quarter of rollouts **while violating the geometric
constraints for a median 130 steps**. Nothing in this batch is a success under S&C.

**Therefore no engine, K, or projector can be ranked from this batch.** Per the Pareto rule,
these are not trade-offs — they are all the same score (zero) on the axis that decides.

Candidate cause worth checking before re-running: after inflation `margin = 0.33`, the
`pillars` outer channels are **~12 cm wide** and the centre channel is `|y| ≤ 0.15`
(`config/uav_projection.yaml`, Fix_12 comment block). If the scored geometry is that tight,
near-zero collision-free is the expected outcome and the constraint set — not the policy — is
what this batch measured. 🔴 **This needs confirming before any pillars re-run.**

### 3.2 Per-variant summary, as printed by the eval

`success` (log's own field, n=10 trials/cell; fm K20 truncated):

| variant | fmK1 | fmK2 | fmK5 | fmK20 | mfK1 | mfK2 | mfK5 | afK1 | afK2 | afK5 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffuser | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 |
| dpcc-r | .00 | .00 | .00 | .00 | .30 | .20 | .00 | .00 | .00 | .00 |
| dpcc-c | .00 | .00 | .10 | .20 | .10 | .10 | .00 | .20 | .00 | .00 |
| dpcc-t | .00 | .00 | .00 | — | .00 | .10 | .00 | **.50** | .20 | .00 |
| dpcc-r-tightened | .00 | .00 | .00 | .00 | .20 | .00 | .00 | .00 | .00 | .00 |
| dpcc-c-tightened | .00 | .20 | .00 | — | .20 | .00 | .00 | .10 | .00 | .00 |
| dpcc-t-tightened | .00 | .00 | .00 | — | .00 | .00 | .00 | .40 | .30 | .10 |
| hardflow_new ❌ | .00 | .10 | .00 | — | .10 | .30 | .20 | .10 | .00 | .00 |
| hardflow_new-r ❌ | .00 | .00 | .00 | — | .20 | .00 | .00 | .00 | .00 | .00 |
| hardflow_new-c ❌ | .00 | .00 | .00 | — | .00 | .00 | .10 | .00 | .00 | .00 |
| hardflow_new-t ❌ | .00 | .00 | .10 | — | .20 | .30 | .10 | **.70** | .40 | .10 |

`goal_reached`:

| variant | fmK1 | fmK2 | fmK5 | fmK20 | mfK1 | mfK2 | mfK5 | afK1 | afK2 | afK5 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffuser | .00 | .00 | .00 | .10 | .00 | .00 | .00 | .00 | .00 | .00 |
| dpcc-c | .00 | .00 | .20 | .60 | .40 | .60 | .00 | .70 | **1.00** | .80 |
| dpcc-c-tightened | .00 | .20 | .50 | — | .40 | .50 | .10 | .20 | .90 | .80 |
| dpcc-t | .00 | .20 | .20 | — | .10 | .20 | .10 | .60 | .60 | .60 |
| hardflow_new ❌ | .00 | .10 | .20 | — | .70 | .60 | .60 | .80 | .90 | .40 |
| hardflow_new-r ❌ | .00 | .00 | .20 | — | **1.00** | .50 | .40 | .50 | .50 | .50 |

⚠️ Cells vary by ±0.1–0.3 with **n=10, one seed (6), no repeats** — that is 1–3 rollouts.
None of these differences is resolvable. Treat every number above as a sanity check, not a
result.

### 3.3 Confounds stacked on top of the above

| # | confound | who it hits |
|---|---|---|
| C1 | IPOPT (K1,K2) vs SLSQP (K5) | fm `hardflow_*` ladder |
| C2 | `af` trains on **SiT** (`bbsit`), `mf` on **UNet** (`bbunet`) | any af-vs-mf comparison — architecture-matched claim not available |
| C3 | HardFlow `noise_sigma`/`two_time` differ by engine: fm `0.5/False`, mf & af `1.0/True` | any cross-engine HardFlow comparison |
| C4 | `hardflow_*` at **K1 and K2** is degenerate | see §3.4 |
| C5 | fm K20 truncated to 4 variants, 40 rollouts | fm K ladder has no usable K20 endpoint |
| C6 | no `diffusion` arm for `pillars` at all | the pinned DA target is absent for this scene |
| C7 | single seed (6), n=10 | everything |

### 3.4 HardFlow low-K degeneracy — logged explicitly

```
[hardflow][DEGENERATE] K=1 A=0.5: n_active=1, n_genuine=0 — every NLP solve is the terminal
tau=1 solve, so this arm runs Pi_S(Euler sample): sample-then-project, == DPCC modulo
solver/variable-scope, NOT HardFlow.
[hardflow][DEGENERATE] first non-degenerate: K>=3 at A=0.5 ... for an attributable effect use
n_genuine>=2 — K>=5 at A=0.5
```

Confirmed present in **all six** K1 and K2 jobs (fm, mf, af). All rows marked ❌ above run **no
HardFlow math**. In this batch only **K5** is a genuine HardFlow row, and it is a single K —
there is no HardFlow ladder here at all.

---

## 4. Cost

`total_ms` per outer step (log TIMING lines):

| variant | fmK1 | fmK2 | fmK5 | fmK20 | mfK1 | mfK2 | mfK5 | afK1 | afK2 | afK5 |
|---|---|---|---|---|---|---|---|---|---|---|
| diffuser | 9.0 | 17.8 | 42.6 | 166.9 | 9.7 | 18.4 | 44.5 | 6.2 | 12.1 | 29.6 |
| dpcc-r | 131 | 172 | 1444 | **6766** | 85 | 91 | 1485 | 118 | 77 | 1683 |
| dpcc-c | 125 | 137 | 1209 | **7070** | 85 | 97 | 1470 | 58 | 51 | 1499 |
| dpcc-r-tightened | 162 | 160 | 1659 | **9474** | 109 | 91 | 1567 | 100 | 101 | 1660 |
| hardflow_new ❌ | 74 | 87 | 185 | — | 27 | 39 | 119 | 20 | 25 | 106 |
| hardflow_new-t ❌ | 440 | 313 | 539 | — | 98 | 111 | 374 | 57 | 80 | 305 |

Two things stand out:

* **`diffuser` (no projector) scales linearly in K** — 9 → 18 → 43 → 167 ms. Clean.
* **DPCC does not.** K2 → K5 is a **~9×** jump in `total_ms` for a 2.5× K increase, and K5 → K20
  another ~5×. `proj_ms` is the whole of it (`dpcc-c` fm K20: `fm_ms=189.9 proj_ms=6880.1`).
  This is what killed job 25130.

> ⚠️ The logs print `budget=30.3ms → real_time_OVER×3668`. **Ignore that ratio.** It is the
> 33 Hz data-rate artefact plus cluster latency, not a real-time pass/fail
> (see `uav-budget-ms-not-a-goal`).

### 4.1 Projection circuit breaker

`25130` (fm K20) tripped the CB **3 times** on `dpcc-c` alone —
`COST EXPLODED (sustained): 36/40 recent steps > 1000 ms` — 12 steps skipped across 3/10
trials, and those trials are flagged **UNPROJECTED**. `25129` (fm K5) and `25138` (af K5) also
show CB activity. Any K≥5 DPCC row must be checked against `PROJECTION_CB_TRIPPED.txt` before
it is read as a projected result.

---

## 5. The `s_curve` job in this drop (25134)

One job, `diffusion` engine (GaussianDiffusion K20), `n_trials=5`, s_curve.

* **34/40 rollouts aborted (85 %), all `inverted`**, median step 404/871.
* Killed at the **24 h wall** after 8 of 10 variants.
* `success=0.000 safe=0.000 goal_reached=0.000` on the last completed variant
  (`dpcc-r-geo_free`, 5/5 aborted).

This is the only `diffusion` arm anywhere in the drop, and it is both truncated and
near-totally aborted. 🔴 **Do not use it as the DPCC baseline.**

Note for context: the s_curve candidates in the DA batch dated **08-25/08-26** predate the v2
guard and were aborted by v1 `p_des_runaway` at 79–98 %. Those rows are dead. Candidate 65
(08-28) is this job.

---

## 6. Other jobs in the drop

| job | rev | status |
|---|---|---|
| 25121 solver bench | `1897f4f` | ok — IPOPT 4.4× slower than SLSQP (endpoint); 🔴 **both returned INFEASIBLE output** on the iterate regime |
| 25161 hardflow K10/K20 | `3df77e8` | 🔴 **crashed in 5 s** — `HFFM_FLOW_STEPS='10 20'` → `int()` ValueError |
| 25222 hardflow K10/K20 | `0c258d0` | ✅ ok, 2 h 49 m, all 4 backend×K passes, "Evaluation completed successfully" |
| 25215 visual aligning | `73adff1` | 🔴 failed — `MIX_PROJ_T='' is not a float` (both T passes) |
| 25216 visual aligning | `81e9ea7` | ✅ ok |

25161 → 25222 and 25215 → 25216 are both fix-then-rerun pairs; the HEAD-side commits
(`0c258d0`, `81e9ea7`) did their job.

---

## 7. What to do next

**Before any pillars re-run — settle the constraint set (§3.1).** 0 S&C in 1707 rollouts with
median 130 violating steps is either a real policy failure or an infeasible scored geometry.
Check whether an **expert** pillars trajectory scores collision-free under the same
`geo_tag=pillars_bounds+dynamics+geo_bounds+obstacles`. If it does not, the constraint set is
the bug and every number above is measuring the config.

Then, in order:

1. **Re-run the fm `hardflow_*` ladder on one backend.** K1/K2 (IPOPT) vs K5 (SLSQP) is not a
   ladder. Everything else in the drop is already on consistent code.
2. **Give K20 a realistic walltime or cut the variant list.** `dpcc-*` at K20 costs ~7–9 s per
   outer step; 17 variants × 10 trials × 634 steps does not fit in 24 h. Either
   `--time` up (24 h is the cap, so this means splitting the job per variant group) or drop to
   the 5 benchmark variants.
3. **Add the `diffusion` arm for pillars.** Without it there is no baseline for this scene.
4. **Drop or re-label the K1/K2 HardFlow rows.** They are `Π_S(Euler sample)`, not HardFlow —
   the log says so itself.
5. **Seeds.** n=10 on one seed cannot separate 0.2 from 0.5.
6. **Optional, cheap:** one run with `inverted` disabled and tilt logged, to convert §2.1 from
   reasoning into measurement.

---

## 9. `pillars` funnel — engines and projectors on the goal line

**Task** `uav-pillars` (state) · **Date** 2026-08-30
**Data** `temp/3008/batch_uav_20260830_110536/per_rollout_detail.csv`, Gen15 rows, K ∈ {1,2,5,20}
**Entrants** `fm` K1/K2/K5/K20 · `mf` K1/K2/K5 · `af` K1/K2/K5 (SiT bone) — **no `diffusion` arm exists for this scene**
**Protocol** seed 6, n = 10 trials/cell, `mpc_batch=4`, `pid_stopgo`, `T=0.5`, budget 634 steps
**Excluded** the `fm`/`mf` **K10** candidates (08-16 / 08-18): they predate the v2 divergence guard and
show 0 % aborts for that reason alone. Mixing them into a ladder whose other rungs abort 14–78 %
would be the same error §1 documents.

> ### How the funnel works
> **Stage 1 · reach** — the **unguided (`diffuser`) arm only**, no projector: *can the generative model
> fly the route at all?*
> **Stage 2 · constraints** — the projector enters. Of what gets down the route, how much is legal?
> **Stage 3 · projector** — HardFlow vs DPCC, paired on engine × K × selection rule.
> **Stage 4 · cost.**
> An arm leaves the moment it fails a stage.

**Lead metric: `goal_crossed_line`** (per request). It is the only metric on this scene with usable
spread — `success_strict_and_constraints` is **0/1707** (§3.1) and `collision_free_completed` is
**2/1707**, so both rank nothing. `goal_crossed_line` fires on **1214/1707 (71 %)** and is a strict
superset of `goal_reached` (all 412 goal-reached rollouts also crossed).

> 🪤 **The trap in this metric, stated up front.** Crossing the line is *necessary, not sufficient*.
> The `-geo_free` variants — projector on, **geometry off** — score **crossed = 0.98** pooled, the
> best number on the board, while violating constraints on **68 % of steps** and reaching the goal
> only **6–16 %** of the time. They fly straight through the pillars and over the line.
> **So `goal_crossed_line` is used here as a *gate*, never as the ranking.** Ranking is on the pair
> (crossed ∧ `viol/step`), with `goal_reached` as the tiebreak.

---

### The result

| entrant | Stage 1 · unguided | Stage 2 · constraints | Stage 3 · projector | Stage 4 · cost | |
|---|---|---|---|---|---|
| **αF K2** (SiT) | **0.90 crossed** ✅ | **0.10 viol/step · 1.00 crossed · 0.73 reached** ✅ | `dpcc-c` / `hardflow_sls` tie | **48–69 ms** | ✅ non-dom |
| αF K1 / K5 | 0.90 / 1.00 ✅ | 0.11 viol/step ✅ | HF cheaper at K5 | 19–280 ms (DPCC K5: 1 486) | ✅ |
| **MeanFlow K1/K2/K5** | **0.00 crossed** ❌ | *rescued by projector:* 0.10 viol/step, 0.36 reached | HF ≫ DPCC at K5 | 25–375 ms | ⚠️ projector-dependent |
| FlowMatching K1/K2/K5 | 0.85 crossed ✅ | **0.18 viol/step · 0.08 reached** ❌ | — | 240–590 ms | ❌ worst constrained arm |
| FlowMatching K20 | 0.70 ✅ | truncated, n=8 | — | 6 659–9 474 ms | ❌ 24 h wall |
| *diffusion* | *— no pillars run exists —* | | | | 🔴 |

# On `pillars` the ordering is **αF ≳ MeanFlow > FlowMatching** — **not** the expected `mf > fm > diffu`.

> Stated Pareto-correctly in §10.4 once the steps axis is added: **every `fm` cell is dominated**;
> `af` K1/K2 and `mf` K1/K2 are **mutually non-dominated** (αF reaches more and cheaper, MeanFlow is
> marginally cleaner), so αF-over-MeanFlow is a **trade-off, not a win**.

---

### 9.1 Stage 1 — unguided: can the model fly the route?

`variant='diffuser'`, no projector, n=10/cell.

| engine | K | crossed | goal_dist (median) | aborted | ms | |
|---|---|---|---|---|---|---|
| **fm** | 1 | 0.70 | 1.71 m | 8/10 | 8.9 | ✅ |
| **fm** | 2 | **1.00** | 1.21 m | 0/10 | 17.9 | ✅ |
| **fm** | 5 | **1.00** | 0.97 m | 1/10 | 42.6 | ✅ |
| fm | 20 | 0.70 | 1.01 m | 9/10 | 166.9 | ✅ |
| **af** | 1 | 0.90 | 1.17 m | 1/10 | 6.2 | ✅ |
| **af** | 2 | 0.90 | 1.27 m | 3/10 | 12.1 | ✅ |
| **af** | 5 | **1.00** | 1.07 m | 1/10 | 29.6 | ✅ |
| 🔴 **mf** | 1 | **0.00** | **6.46 m** | **10/10** | 9.2 | ❌ |
| 🔴 **mf** | 2 | **0.00** | **6.45 m** | **10/10** | 18.0 | ❌ |
| 🔴 **mf** | 5 | **0.00** | **6.51 m** | **10/10** | 44.2 | ❌ |

🔴 **MeanFlow unguided fails totally on `pillars`, at every K: 0/30 crossings, 30/30 aborts, median
final distance 6.5 m against fm/af's ~1.1 m.** The abort lines show it leaving through the ceiling
in the first ~2 s — `off_route ... on z` at `p_z ≈ 3.35` by **FM step 47–83** of 634.

**Gate: crossed ≥ 0.50 unguided. `fm` (0.85 pooled) and `af` (0.93 pooled) pass. `mf` (0.00) fails.**

⚠️ **`mf` is carried forward anyway**, against the funnel's own rule, because the Stage 2 data shows
the projector reverses this completely and that reversal is the most interesting thing in the batch.
It is marked ⚠️ everywhere below, and **no `mf` claim in this section is a claim about the generative
model on its own.**

---

### 9.2 Stage 2 — constraints: what the projector buys, and what it costs

Pooled over all engines and K, grouped by projector family and by whether the **geometric** constraint
group (`geo_bounds` + `obstacles`) is enforced:

| group | n | crossed | reached | aborted | **viol/step** | ms |
|---|---|---|---|---|---|---|
| unguided (`diffuser`) | 106 | 0.62 | 0.01 | 0.50 | 0.58 | 18 |
| **DPCC, geometry ON** | 606 | 0.57 | 0.32 | 0.47 | **0.14** | 181 |
| **HardFlow, geometry ON** | 377 | 0.58 | **0.41** | 0.46 | **0.11** | 141 |
| DPCC, geometry OFF (`-geo_free`) | 270 | **0.98** | 0.06 | **0.02** | 0.68 | 29 |
| HardFlow, geometry OFF | 270 | **0.98** | 0.16 | **0.02** | 0.68 | 78 |

**Three things fall out of this table.**

1. ✅ **The projector works.** Enforcing geometry cuts the violation rate **5×** (0.68 → 0.11–0.14)
   and raises `goal_reached` **2–7×** (0.06 → 0.32–0.41). That is the DPCC/HardFlow projector doing
   exactly its job.
2. 🔴 **And it destabilises the flight.** Geometry ON costs **0.98 → 0.57 crossings** and takes the
   abort rate from **2 % to 46 %**. Per-cell, `-geo_free` is the *only* configuration in the entire
   batch that essentially never diverges: abort **0.00–0.10** in every one of the 27 engine×K×selector
   cells, against 0.00–1.00 with geometry on.
3. 🪤 **Geometry OFF is not a win.** It scores top on the lead metric and bottom on everything that
   matters — 0.68 viol/step, `goal_reached` 0.06–0.16. It crosses the line by flying through the
   pillars. This is the trap flagged above, and it is why `crossed` alone cannot rank.

> **Read:** on `pillars` the geometric constraint set and flight stability are **in direct
> opposition**. That is the same finding §3.1 arrives at from the other side (2/1707 collision-free,
> ~12 cm inflated channels) and it is the single most actionable item in this DA.

**Engine ordering, geometry-ON projected rows only** (`dpcc-{r,c,t}` + `hardflow-{r,c,t}`, n=180 each):

| engine | crossed | **reached** | aborted | **viol/step** | ms |
|---|---|---|---|---|---|
| **af** (SiT) | **0.74** | **0.63** | **0.27** | 0.11 | **100** |
| ⚠️ **mf** (UNet) | 0.57 | 0.36 | 0.53 | **0.10** | 120 |
| **fm** (UNet) | 0.41 | **0.08** | **0.67** | 0.18 | 274 |

**αF > MeanFlow > FlowMatching**, and the gap between the top two and `fm` is not marginal —
`fm` reaches the goal in **8 %** of projected rollouts against 36 % and 63 %, aborts in **two thirds**,
and carries the worst violation rate. On this scene FlowMatching is the weakest arm by every column.

⚠️ **`af` is on a SiT bone; `mf` and `fm` are on UNet.** Per the architecture-matched rule this makes
the `af` lead **confounded and secondary**. The architecture-matched result is **`mf` > `fm`**, and
that one is clean: same bone, same K set, same solver, same code rev.

---

### 9.3 Stage 3 — HardFlow vs DPCC, paired

Same engine, same K, same selection rule, geometry ON, n=10/cell.
🔴 **K1 and K2 are excluded from the verdict** — the eval itself logs `n_genuine=0` there, i.e. those
rows run `Π(Euler sample)`, not HardFlow (§3.4). Only **K5** is a genuine HardFlow row.

| eng | K | sel | crossed HF/DPCC | reached HF/DPCC | abort HF/DPCC | viol/step HF/DPCC | ms HF/DPCC |
|---|---|---|---|---|---|---|---|
| af | 5 | c | 0.80 / 0.90 | 0.80 / 0.80 | **0.00** / 0.10 | **0.08** / 0.10 | **145** / 1485 |
| af | 5 | r | **0.60** / 0.50 | **0.50** / 0.30 | **0.50** / 0.60 | **0.10** / 0.18 | **279** / 1730 |
| af | 5 | t | **0.70** / 0.60 | 0.60 / 0.60 | **0.30** / 0.40 | 0.18 / **0.13** | **250** / 1583 |
| ⚠️ mf | 5 | c | **0.80** / 0.10 | **0.70** / 0.00 | **0.20** / 0.90 | **0.10** / 0.13 | **249** / 1397 |
| ⚠️ mf | 5 | r | **0.60** / 0.00 | **0.40** / 0.00 | **0.40** / 1.00 | 0.10 / **0.02** | **246** / 1441 |
| ⚠️ mf | 5 | t | **0.50** / 0.40 | **0.20** / 0.10 | 0.70 / 0.70 | 0.15 / **0.13** | **375** / 1523 |
| fm | 5 | c | **0.80** / 0.60 | 0.00 / **0.20** | **0.30** / 0.50 | 0.35 / **0.18** | **391** / 1146 |
| fm | 5 | r | 0.40 / **0.60** | 0.20 / **0.40** | 0.70 / **0.60** | **0.17** / 0.20 | **587** / 1329 |
| fm | 5 | t | 0.40 / 0.40 | 0.20 / 0.20 | **0.50** / 0.60 | 0.23 / **0.22** | **393** / 1518 |

**Tally over the 9 genuine cells: HardFlow better on cost 9/9, on aborts 7/9, on `goal_reached` 4/9,
on `viol/step` 4/9.**

> **Verdict: HardFlow matches DPCC on constraint quality and beats it by 4–6× on cost.**
> 145–587 ms against 1 146–1 730 ms, on the identical NLP, with the same or better violation rate.
> It does **not** beat DPCC on constraint satisfaction — it ties. The win is throughput.
>
> ⚠️ Reported at the **same** projection threshold (`T=0.5`), so this is not yet the lower-threshold
> claim the benchmark hierarchy asks HardFlow for.
>
> 🔴 The single most lopsided pair, `mf K5 dpcc-c` at **0.10 crossed / 0.90 abort**, is one n=10 cell
> and sits next to `mf K5 dpcc-r` at 0.00/1.00. Something is wrong with the `mf` + full-DPCC
> combination at K5 specifically; do not read the HF margin there as a HardFlow result until that is
> explained.

---

### 9.4 Stage 4 — cost

Median `avg_time_ms` per outer step, geometry-ON projected rows: **af 100 · mf 120 · fm 274.**

The dominant cost term is not the engine, it is **K through the DPCC projector**: `dpcc-*` runs
50–180 ms at K1/K2 and **1 150–1 730 ms at K5** — a ~10–20× jump for a 2.5× K increase. HardFlow at
the same K5 stays at 145–590 ms. The `fm` K20 DPCC rows reach 6 659–9 474 ms and are what killed job
25130 at the 24 h wall (§1, §4).

---

### 9.5 Does `mf > fm > diffu` survive?

**Partly — and only in the middle.**

| relation | on `pillars` | verdict |
|---|---|---|
| `mf` > `fm` | ✅ **holds, and cleanly** — matched UNet bone, same K set, same solver, same rev. Projected: reached **0.36 vs 0.08**, abort **0.53 vs 0.67**, viol/step **0.10 vs 0.18** | ✅ |
| `fm` > `diffu` | 🔴 **untestable — no `diffusion` arm exists for `pillars`** | 🔴 |
| `mf` on its own | 🔴 **reversed unguided**: 0/30 crossings, 30/30 aborts, 6.5 m out. `mf` is top-two *only* with a projector | 🔴 |
| vs `af` | ❌ **`af` beats both**, on every column — but on a **SiT** bone, so confounded | ⚠️ |

**So the honest statement for this scene is:**

> With a projector, **`af`(SiT) > `mf` > `fm`**. The architecture-matched part of that, **`mf` > `fm`**,
> holds. **Without** a projector the order inverts at the top: `fm` and `af` fly the route, `mf` does
> not fly at all. **The `diffu` leg cannot be closed on `pillars` — that run was never made.**

The missing leg has been measured on the *other* hard scene: `DA_20260827_s_curve_three_way_fm_mf_diffusion.md`
finds `diffusion` K20 + `dpcc-c` to be **the worst arm on `s_curve`** (203 violating steps,
`goal_dist` 2.34 m, 5 348 ms/step, 88/97 aborted). That is consistent with `diffu` sitting at the
bottom — but it is a **different scene, a different batch, and a different code rev**, so it is
supporting evidence, not a pillars result. 🔴 **A `diffusion` pillars run is the one job that would
close this question.**

---

### 9.6 Red flags on everything above

1. 🔴 **n = 10, one seed (6), no repeats.** A cell moving 0.6 → 0.9 is three rollouts. Every
   individual cell in §9.3 is inside noise; only the pooled n=180 engine rows in §9.2 carry weight.
2. 🔴 **`goal_crossed_line` is a gate, not a score.** See the trap box — it ranks the unconstrained
   projector first.
3. 🔴 **Zero rollouts satisfy success+constraints.** Everything here ranks arms on a task that no arm
   solves. This is a *relative* ordering inside a total failure, not a result.
4. ⚠️ **`af` vs `mf`/`fm` is architecture-confounded** (SiT vs UNet). The `af` row is secondary.
5. ⚠️ **The abort truncates episodes**, so `viol/step` on an aborted rollout is measured over a
   shorter flight. Recomputed on the **1045 non-aborted rollouts only** (full-length episodes,
   `dpcc-*` rows at K1/K2/K5) the engine ordering is unchanged — `af` **0.17 / 0.21 / 0.20**,
   `mf` **0.38 / 0.16 / 0.45**, `fm` **0.83 / 0.48 / 0.62** viol/step — so the conclusion is not an
   artefact of truncation. But per-cell numbers from high-abort cells should not be read closely.
   Note `mf` has **zero** non-aborted unguided rollouts, which is §9.1 restated.
6. ⚠️ **`hardflow_new` (fm K1/K2) and `hardflow_sls` (everything else) are the same variant on
   different NLP solvers** (§1.1). §9.3's verdict is drawn from K5 only, where all rows are SLSQP,
   so it is internally consistent — but the fm HardFlow K1→K2→K5 trend crosses the solver boundary
   and must not be read as a K effect.

---

## 10. Steps — the efficiency axis, and the Pareto verdict

Added per request, on the avoiding-DA convention: **how many control steps does a setup spend?**

### 10.1 Which steps column is usable

| column | defined on | verdict |
|---|---|---|
| `max_episode_length` | 1707/1707, always **634** | constant — useless |
| `n_steps` | 1705, median 634 | 🪤 **634 for every one of the 525 aborted rollouts** (U_13: a miss costs the full budget). Carries no signal beyond the abort rate |
| `n_fm_steps` / `n_proj_steps` | 1707 | cost proxies, not efficiency |
| **`steps_to_goal`** | **exactly the 412 goal-reached rollouts** (382–628, median 500) | ✅ the real one |

`steps_to_goal` is reported two ways, because on its own it is a **survivorship trap**:

* **`steps|reached`** — median over reached rollouts only. Answers *"when it gets there, how fast?"*
  🪤 A setup that reaches 7/100 times gets scored on those 7 easy contexts.
* **`chargedSteps`** — mean with every miss charged the full **634** budget. Answers *"what does a
  trial cost in expectation?"* This is the U_13 convention and the one that ranks.

### 10.2 The trap is real, and it is not small

| group | reached | `steps\|reached` | `chargedSteps` |
|---|---|---|---|
| `fm`, geometry ON | 0.11 | **461** ⚠️ *fastest on the board* | **616** ❌ *worst on the board* |
| `mf`, geometry ON | 0.37 | 508 | 588 |
| `af`, geometry ON | 0.63 | 511 | **555** ✅ |
| dpcc **geo-OFF** | 0.06 | **426** ⚠️ | 621 ❌ |
| HF **geo-OFF** | 0.16 | **428** ⚠️ | 601 ❌ |
| dpcc geo-ON | 0.33 | 502 | 589 |
| HF geo-ON | 0.41 | 511 | **585** |

🪤 **On the conditional metric, `fm` is the fastest engine and geometry-OFF is the fastest
configuration — and both readings are inverted.** `fm` reaches 37 times in 330 rollouts, and
`-geo_free` reaches by flying straight through the pillars without detouring. Charging the budget
flips both. **This is the same trap as `goal_crossed_line` in §9, on a different axis.**

> **Rule for this scene: quote `chargedSteps`. Quote `steps|reached` only with its n attached.**

### 10.3 Steps by setup

Geometry-ON projected rows (`dpcc-*` + `hardflow-*`, no `geo_free`), n = 100/cell (fm K20: n = 30).

| eng | K | crossed | reached | viol/step | `steps\|reached` | **chargedSteps** | ms |
|---|---|---|---|---|---|---|---|
| fm | 1 | 0.21 | 0.00 | 0.16 | —[0] | **634** | 174 |
| fm | 2 | 0.55 | 0.07 | 0.18 | 483 [7] | **626** | 184 |
| fm | 5 | 0.53 | 0.22 | 0.20 | 452 [22] | **596** | 1 192 |
| fm | 20 | 0.40 | 0.27 | 0.17 | 452 [8] | **588** | 7 064 |
| ⚠️ mf | 1 | 0.65 | 0.44 | **0.09** | 508 [44] | **579** | 91 |
| ⚠️ mf | 2 | 0.64 | 0.44 | **0.09** | 510 [44] | **581** | 83 |
| ⚠️ mf | 5 | 0.36 | 0.22 | 0.10 | 498 [22] | 604 | 1 264 |
| af | 1 | 0.76 | 0.56 | 0.10 | 504 [56] | **563** | **59** |
| **af** | **2** | **0.84** | **0.73** | 0.11 | 515 [73] | **545** | **59** |
| af | 5 | 0.66 | 0.60 | 0.12 | 504 [60] | 555 | 1 444 |

**`steps|reached` is flat — 452 to 515 across every engine, K and projector, on a 634 budget.** The
efficiency axis does not separate these arms; **it is the reach rate that separates them**, and
`chargedSteps` is essentially a monotone restatement of `reached`. That is worth knowing: on
`pillars`, *steps adds no independent ranking information*. It is a consistency check, and it passes.

### 10.4 Pareto verdict — the relationship holds, with one correction

Dominance on all four axes at once: **`reached`↑ · `viol/step`↓ · `chargedSteps`↓ · `ms`↓.**

| cell | verdict |
|---|---|
| `fm` K1 | 🔴 **dominated** by mfK1, mfK2, afK1, afK2 |
| `fm` K2 | 🔴 **dominated** by mfK1, mfK2, afK1, afK2 |
| `fm` K5 | 🔴 **dominated** by mfK1, mfK2, afK1, afK2 |
| `fm` K20 | 🔴 **dominated** by mfK1, mfK2, afK1, afK2, afK5 |
| `mf` K5 | 🔴 dominated by mfK1, mfK2 |
| `af` K5 | 🔴 dominated by afK2 |
| ✅ **`mf` K1 · `mf` K2 · `af` K1 · `af` K2** | **NON-DOMINATED** |

**Every single `fm` cell is Pareto-dominated — on all four axes simultaneously, by four different
cells.** That is the strongest form the `mf > fm` claim can take, and adding steps did not weaken it:

> ✅ **`mf` > `fm` holds on the steps axis too, and now as strict Pareto dominance.**
> Architecture-matched (both UNet), same K set, same solver, same code rev.

**The correction.** `af` K2 and `mf` K1/K2 are **mutually non-dominated**: `af` K2 reaches more
(0.73 vs 0.44), costs fewer charged steps (545 vs 579) and less time (59 vs 91 ms), while `mf` K1/K2
carry the lower violation rate (0.09 vs 0.11). **So §9's `af > mf` is a trade-off, not a win** — and
`af` is on a SiT bone regardless. The Pareto-safe statement is:

> **Non-dominated set: `af` K1/K2 (reaches more, cheaper) and `mf` K1/K2 (marginally cleaner).
> `fm` is dominated at every K.**
> ⚠️ The `mf` advantage is 0.09 vs 0.11 viol/step at n=100 pooled — well inside the noise this batch
> can resolve. Do not build on it.

### 10.5 HardFlow vs DPCC on steps

| projector | reached | `steps\|reached` | chargedSteps | ms |
|---|---|---|---|---|
| DPCC, geometry ON | 0.33 | 502 [190] | 589 | 181 |
| HardFlow, geometry ON | **0.41** | 511 [146] | **585** | **141** |

**A tie on steps** (589 vs 585, and 502 vs 511 conditional — both differences are ~1–2 % of budget).
This is the same verdict §9.3 reached from the constraint side: **HardFlow does not buy fewer steps or
better constraints than DPCC — it buys the same result 4–6× cheaper in wall time.** Steps confirm it
rather than adding to it.

### 10.6 K and steps

`chargedSteps` improves with K for `fm` (634 → 626 → 596 → 588) — the only engine where it does,
and it is bought at **174 → 7 064 ms**. For `mf` and `af` the best `chargedSteps` is at **K1/K2**
(579/581 and 563/545), and K5 is *worse* on both (604, 555) while costing ~20× the time.

> **More NFE does not buy fewer steps on `pillars` for `mf` or `af`.** Same direction as the s_curve
> DA's "more NFE makes it worse", and the opposite of what the `fm` column suggests in isolation —
> but `fm` starts from 0.00–0.07 reached, so its whole K trend is a climb out of total failure.

---

## 8. Provenance

* Logs: `temp/3008/2026-08-27/{18_34_21,18_34_44,18_31_47,18_39_02}_*.log`,
  `temp/3008/2026-08-28/`, `temp/3008/2026-08-30/`
* Aggregates cross-checked: `temp/3008/batch_uav_20260830_110536/{per_rollout_detail,
  candidate_axes,data_quality}.csv` (65 candidates / 968 units; pillars = 23 candidates)
* Abort counts derived from log lines agree with `per_rollout_detail.csv`
  `divergence_aborted` to the rollout.
* Guard-block parity checked by md5 over `git show <rev>:mix_uav_test/eval_mix_uav.py`.
* Diffs read with `git diff --stat` / `git diff` between `1897f4f`, `2c3c38d`, `60f1b13`.
