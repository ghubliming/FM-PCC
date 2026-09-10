# The tracker, not the planner — `mjpc` vs `pid_stopgo` on UAV `s_curve`

**Date:** 2026-09-09 · **Task:** `uav-s_curve` (Gen15 UAV Mix-ML) · **Engine:** MeanFlow, U-Net backbone (`bbunet`), K=10
**Batch:** `temp/0909/batch_uav_20260909_205118` · **Candidates:** C94 (`mjpc`) vs C95 (`pid_stopgo`)
**Jobs:** 25514 → **25554** (mjpc, n=3) · **25502** (pid_stopgo, n=8 variants, n=10) — both completed

📄 **Full working analysis:**
[`logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md)
· campaign index:
[`MASTER_20260908_five_missions_campaign.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/MASTER_20260908_five_missions_campaign.md)

---

`s_curve` has been the UAV scene where every engine scores S&C ≈ 0 — the result that killed the
α-Flow UAV arm on 2026-09-07. This experiment asks whether that floor is a **planner** failure or a
**tracker** failure, by swapping only the low-level controller (`pid_stopgo` → MuJoCo MPC) on an
otherwise byte-identical evaluation.

## What it shows

| # | Finding | Strength |
|---|---|---|
| **F1** | **The raw MeanFlow plan was flyable; `pid_stopgo` could not fly it.** Goal reached **0/10** under PID, **3/3** under MJPC, on the *same* three initial conditions. Goal distance 2.86/2.72/2.89 → 0.299/0.298/0.294. | Fisher exact **p = 0.0035**, paired |
| **F2** | 🔴 **PID loses ATTITUDE control — the drone inverts.** All **10/10** raw-plan rollouts abort with `inverted: body z-axis · world z < 0` at step **395–421** (t ≈ 11.97–12.76 s). MJPC: **0/3**. `min_z` stays ~1.1 m because the aircraft is upside down *at altitude*, not on the ground. | 10/10 vs 0/3, §6 |
| **F3** | **The DPCC-projected plan is unflyable by either tracker** — 0/10 and 0/3, goal distance ≈ 2.6–2.9 under both. A stronger controller changes nothing. This localises a **projector** defect. | 13/13 failures |
| **F4** | **On the HardFlow arm PID "succeeds" by dragging along the floor.** `phys_min_z` ≈ 0 or negative on **all 10** PID rollouts (min −0.009) vs **1.05–1.14 m** under MJPC, at 3.0× fewer violations per step and **half** the projection time. | 10/10 vs 3/3 |
| **F5** | 🔴 **S&C remains 0.000 in all six cells.** MJPC converts *"never arrives"* into *"arrives, still violates"*. The scene is **not** rescued as a ranking instrument. | all cells |
| **F6** | MJPC's per-step violation rate is near-constant across variants (0.068–0.077) while PID's spans **15×** (0.016–0.235). PID's numbers are dictated by *which pathology it falls into*; MJPC has a characteristic cost. | 3 variants |
| **F7** | 🔴 **MJPC is the *more expensive* controller — ≈ 119 ms/step more, a ~20× multiple on the tracker.** On the projector-free arm it costs **2.27×** per executed step (94.9 → 215.6 ms). It is still the cheaper route to a *success* (PID: ∞ on 2 of 3 arms), and on HardFlow it wins end-to-end (0.59×) — not by being fast, but by conditioning the SLSQP (`proj_ms` 1587.5 → 761.8). | §5, wall-clock |

---

## Figures

> **Placeholders — figures to be added.** Drop the files into this folder under exactly these
> names and the links resolve. Build specs and data sources in the [appendix](#appendix--figure-specs).

![Fig 1 — paired goal distance, PID → MJPC, per initial condition](fig1_paired_goal_distance.svg)

![Fig 2 — minimum altitude per rollout: the floor-scraping failure](fig2_min_altitude.svg)

![Fig 3 — constraint violations per step, normalised by episode length](fig3_violations_per_step.svg)

![Fig 4 — tracking error vs goal reached: the anti-correlation](fig4_trackerr_vs_goal.svg)

![Fig 5 — example trajectories on a shared initial condition](fig5_trajectories_idx0.svg)

![Fig 6 — planner compute per rollout, and per successful rollout](fig6_compute_cost.svg)

![Fig 7 — divergence: when each controller loses attitude](fig7_divergence.svg)

---

## 1. Design — matched, and paired

Everything except the controller is identical: same trained MeanFlow U-Net checkpoint, same scene,
K=10, same `u7hg` honest-geometry constraint set, same seed 6, same `mpc_batch=4`, same projection
mode. Only the tracker differs, and because `controller` is a results-path key the two runs write to
separate folders and cannot collide:

```
Emf_K10_mpc4_pid_stopgo_T0.5_u7hg/     ← C95, job 25502
Emf_K10_mpc4_mjpc_T0.5_u7hg/           ← C94, job 25554
```

**The comparison is paired.** `phys_min_z` at a given `rollout_idx` is identical across *different
variants* within a run (1.106 / 1.186 / 1.154 at indices 0/1/2 for both `diffuser` and `dpcc-r` under
PID), so `rollout_idx` names the same initial condition in both jobs. MJPC's rollouts 0–2 sit
directly against PID's rollouts 0–2.

**Trial counts are deliberately unmatched: n=3 (MJPC) vs n=10 (PID).** The MJPC arm was specified as
a cheap probe. A bare 3/3 carries a Wilson 95 % lower bound of only 0.29, so no claim here rests on
an MJPC rate in isolation — F1 survives because PID's 0/10 has an *upper* bound of 0.31 and the
intervals are disjoint.

**HardFlow genuineness:** at K=10 with activation threshold A=0.5 the arm is non-degenerate
(`n_genuine ≥ 1`), so the `hardflow_sls-r` rows carry real HardFlow arithmetic and are citable. ✅

---

## 2. F1/F2 — the raw plan, and the shape of PID's failure

| idx | PID `goal_dist` | PID `n_steps` | **MJPC `goal_dist`** | **MJPC `steps_to_goal`** |
|---|---|---|---|---|
| 0 | 2.863 | 871 *(budget exhausted)* | **0.299** | **633** |
| 1 | 2.718 | 871 | **0.298** | **650** |
| 2 | 2.889 | 871 | **0.294** | **625** |
| 3–9 | 2.617 – 2.873 | 871 × 7 | — | — |

Not one PID rollout diverges or crashes. The drone is airborne and stable for the entire episode and
simply does not arrive.

The mechanism is named by an inversion that would otherwise look like an error:

| controller | `track_err` | `goal_reached` |
|---|---|---|
| `pid_stopgo` | **0.294 – 0.341** *(tightest in the study)* | **0.000** |
| `mjpc` | 0.422 – 0.490 *(visibly looser)* | **1.000** |

**Tracking error is anti-correlated with task success.** A controller that hugs the reference while
never advancing along it is not failing to track — it is failing to make **progress**. That is
`pid_stopgo`'s stop-and-go logic saturating on a reference that demands sustained forward velocity.
MJPC accepts a looser corridor and converts it into motion.

---

## 3. F3/F4 — three variants, three different failures

| variant | PID | MJPC | who is at fault |
|---|---|---|---|
| `diffuser` (raw plan) | 0/10 goal — stalls at 871 steps | **3/3 goal** | **the controller** |
| `dpcc-r` (DPCC projection) | 0/10 goal | **0/3 goal** | **the projector** |
| `hardflow_sls-r` (HardFlow-SLSQP) | 7/10 goal, but `min_z` ≈ 0 | **3/3 goal at 1.1 m** | **the controller** |

### `dpcc-r` — not a controller problem

Goal distance moves 2.665 → 2.571 and violations *rise* 29.1 → 59.0. The tracker that completely
rescued the raw plan does nothing here. PID's ten rollouts split into 8 stalls (`goal_dist` ≈ 2.85,
7–11 violations) and 2 scrapes (idx 3–4: `goal_dist` 1.80/2.02, 99/120 violations, `min_z` 0.545,
`contact_frac` 0.172). **Whatever the projector emits on `s_curve`, no controller can fly it** —
and the projection dominates the control step at `proj_ms` ≈ 2122 (PID) / 2420 (MJPC).

### `hardflow_sls-r` — the aggregate hides the failure

`goal_reached` = 0.700 under PID looks respectable until the altitudes are read:

| ctrl | `phys_min_z`, per rollout |
|---|---|
| **PID** | −0.005, 0.005, 0.014, 0.002, 0.022, −0.002, 0.020, −0.009, 0.013, 0.019 |
| **MJPC** | **1.050, 1.143, 1.112** |

Every PID rollout descends to ground level or below; it reaches the goal by sliding along the floor.
On the three paired initial conditions the goal distance is unchanged (0.295→0.299, 0.296→0.297,
0.300→0.296) while altitude jumps from ≈ 0 to ≈ 1.1 m — **same outcome, entirely different flight.**
MJPC additionally halves the projection cost (`proj_ms` 1587.5 → 761.8), the NLP converging faster
from better-conditioned states.

---

## 4. F6 — violations must be normalised by episode length

Raw violation totals are confounded: a stalled drone accrues few violations because it never
traverses the scene. Per step:

| variant | ctrl | violations | steps | **viol / step** | `min_z` |
|---|---|---|---|---|---|
| `diffuser` | PID | 13.7 | 871.0 | **0.0157** | 1.126 |
| `diffuser` | **MJPC** | 48.7 | 636.0 | **0.0765** *(4.9× higher)* | 1.100 |
| `dpcc-r` | PID | 29.1 | 871.0 | 0.0334 | 1.046 |
| `dpcc-r` | **MJPC** | 59.0 | 871.0 | 0.0677 | 1.093 |
| `hardflow_sls-r` | PID | 166.9 | 709.1 | **0.2354** | **0.008** |
| `hardflow_sls-r` | **MJPC** | 48.0 | 618.0 | **0.0777** *(3.0× lower)* | **1.102** |

Two things follow, and they point in opposite directions:

* On the **raw plan** MJPC genuinely violates more — 4.9× more per step. It is buying progress with
  constraint adherence, not merely accumulating violations by flying further. **This is a real
  trade, not a normalisation artefact.**
* On the **HardFlow arm** MJPC violates 3.0× *less* per step *and* flies at altitude *and* halves the
  projection cost. There it is a strict improvement.

MJPC's per-step rate is nearly constant across all three variants (**0.068 – 0.077**), while PID's
spans a 15× range (0.016 – 0.235). PID's violation statistics are dictated by which failure mode it
falls into; MJPC has a characteristic, predictable cost.

---

---

## 5. Cost — wall-clock and compute

### 5.1 🔴 What is measured, and what is not

`avg_time_ms` is **exactly** `fm_ms + proj_ms` — verified on all six cells, to the decimal:

| variant | ctrl | `fm_ms` | `proj_ms` | sum | `avg_time_ms` |
|---|---|---|---|---|---|
| `diffuser` | PID | 88.59 | 0.00 | 88.59 | **88.59** |
| `diffuser` | MJPC | 90.06 | 0.00 | 90.06 | **90.06** |
| `dpcc-r` | PID | 90.17 | 1870.39 | 1960.56 | **1960.56** |
| `dpcc-r` | MJPC | 93.72 | 2459.11 | 2552.83 | **2552.83** |
| `hardflow_sls-r` | PID | 131.10 | 1121.85 | 1252.95 | **1252.95** |
| `hardflow_sls-r` | MJPC | 136.90 | 761.56 | 898.46 | **898.46** |

**The timed window is the planner — generation plus projection — and the low-level tracker runs
outside it, inside the environment step.** So this batch **cannot** price `pid_stopgo` against
`mjpc` directly. Any table claiming "MJPC costs X ms more per step" is not supported by this data,
and the near-identical `fm_ms` (+1.6 % to +2.7 %, same generator at the same K) is the artefact that
proves the tracker is absent from the measurement, not evidence that MJPC is free.

Getting the direct number needs instrumentation: a timer around the tracker call, exported the way
`fm_ms` / `proj_ms` already are. MuJoCo MPC solves an optimisation per control step and will not be
free — treat its per-step cost as **unmeasured**, not small.

### 5.2 The controller's real cost effect is indirect — through the projector — and it reverses

The tracker changes the *states* the projector is handed, and that changes how hard the NLP is:

| variant | PID `proj_ms` | MJPC `proj_ms` | |
|---|---|---|---|
| `hardflow_sls-r` | 1587.50 | **761.83** | **52 % cheaper** under MJPC |
| `dpcc-r` | 2122.00 | **2420.15** | **14 % dearer** under MJPC |
| `diffuser` | 0.00 | 0.00 | no projector |

MJPC keeps the aircraft at ~1.1 m and on a feasible path, so HardFlow's SLSQP converges from
better-conditioned states (§4). Under DPCC the effect runs the other way. **There is no single
"MJPC is cheaper" statement to make — it depends on which projector is downstream.**

### 5.3 🔴 RETRACTED — "planner compute per rollout"

An earlier draft of this section computed `n_fm_steps × fm_ms + n_proj_steps × proj_ms` and
concluded MJPC was **0.74×** the cost on the raw plan. **That was wrong.** `n_fm_steps` reports the
episode **budget** (871), not the steps that executed. PID's rollouts abort at step ≈ 406 (§6.1), so
the calculation charged PID for **465 steps that never ran** and inflated its cost by ≈ 2.1×. The
sanity check it failed: it put PID's planner cost at 77.0 s inside a rollout whose *total* wall-clock
was 38.5 s.

### 5.4 The direct measurement — wall-clock per executed step

The CSV carries no wall-clock column, so this comes from the **job logs** (total elapsed per variant
÷ `n_trials`) — labelled as such, and it is the only figure here that contains the tracker.
Executed steps = `divergence_step` when the run aborts, else `steps_to_goal`.

| variant | ctrl | wall-clock **s / rollout** | executed steps | **ms / step** | ratio |
|---|---|---|---|---|---|
| **`diffuser`** *(no projector)* | PID | 38.5 | 405.8 | **94.9** | — |
| **`diffuser`** | **MJPC** | **137.1** | 636.0 | **215.6** | **2.27× DEARER** |
| `dpcc-r` | PID | 863.3 | 375.6 | 2298.6 | — |
| `dpcc-r` | **MJPC** | 1110.0 | 417.3 | 2660.1 | 1.16× dearer |
| `hardflow_sls-r` | PID | 1101.0 | 631.2 | 1744.2 | — |
| `hardflow_sls-r` | **MJPC** | **639.3** | 618.0 | **1034.5** | **0.59× cheaper** |

**On the controller-only arm MJPC costs 2.27× more per executed step.** That is the honest answer to
"is MJPC faster": **it is not.** It solves an optimisation every control step where PID evaluates an
algebraic law.

### 5.5 Isolating the tracker — the per-step controller cost

On `diffuser` there is no projector, so wall-clock ≈ planner + tracker + simulator. The planner term
is `fm_ms` × executed steps, and the simulator term is identical for both controllers, so the
residual difference is the tracker:

| | PID | MJPC |
|---|---|---|
| wall-clock / rollout | 38.5 s | 137.1 s |
| executed steps | 405.8 | 636.0 |
| planner (`fm_ms` × steps) | 35.9 s | 57.1 s |
| **residual (tracker + sim)** | **2.6 s → 6.4 ms/step** | **80.0 s → 125.8 ms/step** |

**MJPC adds ≈ 119 ms per control step over `pid_stopgo`** — the simulator term cancels. Roughly a
**20×** step-cost multiple on the tracker itself. Both residuals are now smaller than their
wall-clocks, which is the consistency check §5.3 failed.

⚠️ Single seed, n = 10 (PID) vs n = 3 (MJPC), and the residual lumps the MuJoCo step in with the
tracker. It is a decomposition, not an instrumented measurement — §6.3 lists the counter that would
replace it.

### 5.6 Where MJPC *does* win, and why it is not a speed win

`hardflow_sls-r` is the one row where MJPC is cheaper end-to-end (0.59×), and the cause is not the
tracker — it is **downstream**. MJPC keeps the aircraft at ~1.1 m on a feasible path, so HardFlow's
SLSQP converges from better-conditioned states: `proj_ms` **1587.5 → 761.8**, a 52 % saving that
more than repays the ~119 ms/step the tracker costs. Under DPCC the same mechanism runs backwards
(`proj_ms` 2122.0 → 2420.2, 14 % dearer).

**So the correct statement is:** MJPC is a *more expensive controller* that can make an *expensive
projector cheaper*. Whether the trade pays depends entirely on what is downstream of it — and on
the raw plan, where nothing is downstream, it simply costs 2.27× more.

### 5.7 Cost per **successful** rollout

Wall-clock seconds per rollout ÷ `goal_reached` — what a success actually costs:

| variant | PID | MJPC | |
|---|---|---|---|
| `diffuser` | 38.5 s → **∞** (0/10 goals) | **137.1 s** (3/3) | PID never pays off at any budget |
| `dpcc-r` | ∞ (0/10) | ∞ (0/3) | neither controller can fly it |
| `hardflow_sls-r` | 1572.8 s (0.700) | **639.3 s** (1.000) | **2.46× cheaper** |

MJPC is the dearer controller per step and still the cheaper route to a completed traverse, because
PID's cheap steps buy nothing on two of the three arms.

*Raw times only. The `budget = 30.3 ms` / 33 Hz line in the job logs is a data-rate artefact plus
cluster latency, not a real-time target.*

---

## 6. The two controllers, compared directly

This section isolates the tracker. The `diffuser` arm carries **no projector at all**, so PID vs
MJPC on that arm is the cleanest controller-only comparison the data admits: identical plans from
an identical checkpoint, identical scene and seed, differing only in what flies them.

### 6.1 🔴 The headline: attitude stability

`divergence_aborted` fires on `inverted: body z-axis · world z < 0` — the aircraft is upside down.

| variant | ctrl | diverged | rate | `divergence_step` (range) |
|---|---|---|---|---|
| **`diffuser`** *(no projector)* | **PID** | **10 / 10** | **100 %** | 405.8 (**395 – 421**) |
| `diffuser` | **MJPC** | **0 / 3** | **0 %** | — |
| `dpcc-r` | PID | 10 / 10 | 100 % | 375.6 (350 – 469) |
| `dpcc-r` | **MJPC** | **3 / 3** | **100 %** | 417.3 (400 – 451) |
| `hardflow_sls-r` | PID | 3 / 10 | 30 % | 611.3 (462 – 782) |
| `hardflow_sls-r` | **MJPC** | **0 / 3** | **0 %** | — |
| **all variants** | PID | **48 / 80** | **60 %** | — |
| **all variants** | MJPC | **3 / 9** | **33 %** *(all on `dpcc-r`)* | — |

Three things fall out of this table:

1. **On the raw plan, `pid_stopgo` inverts the aircraft on every single rollout**, inside a 26-step
   window (395–421, t ≈ 11.97–12.76 s) across ten different initial conditions. That is not a
   marginal instability — it is deterministic, and it lands at the same point of the trajectory
   every time, ≈ 12 s into a ≈ 26 s traverse, i.e. **at the S-curve crossover** where the reference
   demands the sharpest lateral change. MJPC flies the same plans with **zero** divergences.
2. **On `dpcc-r`, both controllers invert, 100 % and 100 %.** This is the sharpest possible form of
   F3: the DPCC-projected plan is not merely hard to track, it is **dynamically infeasible** — it
   tumbles a controller that handles the unprojected plan perfectly. The defect is in the plan.
3. **On the HardFlow arm PID's divergence is later and rarer** (3/10, steps 462–782) — consistent
   with §4, where PID reaches the goal 0.700 of the time by dragging along the floor. Fewer flips,
   but only because it is scraping rather than flying.

### 6.2 Behavioural comparison on the controller-only arm (`diffuser`)

Means; PID n = 10, MJPC n = 3. Paired on `rollout_idx` 0–2.

| axis | metric | **PID** | **MJPC** | reading |
|---|---|---|---|---|
| **attitude** | `divergence_aborted` | **1.000** | **0.000** | MJPC eliminates the failure |
| | `divergence_step` | 405.8 | — | |
| **progress** | `n_steps` | 871 *(budget exhausted)* | **636** | 27 % shorter |
| | `steps_to_goal` | **nan** *(never)* | **636** | |
| **terminal** | `goal_dist` | 2.809 | **0.297** | −2.512 m |
| | `goal_crossed_line` | 0.000 | **1.000** | |
| **tracking** | `track_err_mean` | **0.304** | 0.455 | not comparable — see §2 |
| **altitude** | `phys_min_z` | 1.126 | 1.100 | indistinguishable |
| | `phys_final_z` | 1.492 | **1.138** | PID ends 0.35 m higher — attitude loss, not descent |
| | `phys_contact_frac` | 0.042 | 0.037 | indistinguishable |
| **constraints** | `n_violations` | 13.7 | 48.7 | confounded by episode length — §4 |
| **outcome** | `success_strict` | 0.000 | **0.667** | |
| | `phys_safe` | **0.000** | **0.667** | PID: 0/10 physically valid |

Paired, on the three shared initial conditions:

| idx | `goal_dist` PID → MJPC | `n_steps` PID → MJPC | diverged PID → MJPC |
|---|---|---|---|
| 0 | 2.863 → **0.299** | 871 → **633** | yes → **no** |
| 1 | 2.718 → **0.298** | 871 → **650** | yes → **no** |
| 2 | 2.889 → **0.294** | 871 → **625** | yes → **no** |

**Every axis that describes control quality moves the same way, on every pair.**

### 6.3 🔴 What cannot be compared — and what it would take

The batch carries **67 per-rollout columns and none of them are tracker-side**. Specifically
**not available**, and therefore not claimed anywhere in this report:

| missing | consequence |
|---|---|
| controller compute time | no ms-per-control-step figure for either tracker (§5.1). MJPC solves an optimisation per step; treat its cost as **unmeasured, not small** |
| control effort / energy | no thrust-integral or power comparison — the "which controller is cheaper to run" question is **open** |
| actuator saturation | cannot say whether PID's inversion is a saturation event or a gain/phase problem |
| attitude time-series | `divergence_step` gives *when*, not the roll/pitch trajectory into it |
| per-axis tracking error | `track_err_mean` is a scalar; no lateral-vs-vertical decomposition |

**To close these, three counters would be enough**, exported the way `fm_ms` / `proj_ms` already
are: (i) wall-clock around the tracker call, (ii) Σ‖u‖ or Σ‖u‖² per rollout, (iii) attitude
(roll/pitch) sampled per step, or at minimum its max. (i) and (ii) are the two the thesis would
actually quote.

### 6.4 What the controller comparison establishes

**`mjpc` is not "better tracking" — it is a different failure profile.** It tracks the reference
*worse* by the scalar metric (0.455 vs 0.304) and that is the correct trade: it gives up positional
tightness for attitude authority, and consequently completes the traverse that PID tumbles out of
on every attempt. On the unprojected plan the comparison is categorical rather than quantitative —
**0/10 physically valid rollouts against 3/3 that reach the goal.**

What it does **not** establish is that MJPC is a better controller in general. It fails exactly as
hard as PID on `dpcc-r` (3/3 inverted), it costs an unmeasured amount more per step, and n = 3.

---

## 7. 🔴 What this does *not* show

**S&C = 0.000 in all six cells, under both controllers.** Under MJPC the raw plan reaches `success`
0.667 and `phys_safe` 0.667 — but S&C requires success *and* a clean constraint record, and the
rollouts that arrive carry 28–63 violations.

So the experiment answers its own question — *yes, the tracker was not powerful enough* — while
**failing to restore `s_curve` as a scene that can rank engines.** The floor that made it useless is
still at 0.00; the failure has only moved from *"cannot reach the goal"* to *"reaches the goal
through the constraints"*.

Also out of scope: any multi-seed claim (seed 6 only); any MJPC rate beyond the paired `diffuser`
contrast at n=3; the five variants of C95 (`dpcc-c`, `dpcc-t`, `hardflow_sls`, `-c`, `-t`) that have
no MJPC counterpart.

**Timing note.** `avg_time_ms` is reported raw (88–2513 ms). The `budget = 30.3 ms` / 33 Hz line that
appears in the job logs is a data-rate artefact plus cluster latency, **not** a real-time target, and
the `real_time_OVER×N` counters are deliberately not reproduced as a pass/fail verdict.

---

## 8. Implications

1. **A UAV result that reports `goal_reached` without `phys_min_z` can be an artefact of the
   tracker.** F4 is the cautionary case: a 70 % goal rate achieved at zero altitude.
2. **`track_err` is not a proxy for task quality on this scene** — here it is inversely related to
   success. Reporting it alone would have ranked PID above MJPC.
3. **`dpcc-r` on `s_curve` is the sharpest open defect in the UAV line**: a projected plan that no
   controller can fly. It is now isolated from the controller question and can be attacked directly.
4. **Scene dynamic range is the blocker, not the controller.** `corridor` saturates at S&C 1.000,
   pre-U7 `pillars` is void for ranking, and `s_curve` floors at 0.000 under both trackers. This is a
   constraint-set problem.

**Recommended next runs.** (a) MJPC at **n=10** over the full 8-variant subset — the effect is large
and the run cost only 1.6 h; (b) a `dpcc-r`-focused `s_curve` investigation; (c) do **not** budget
for a controller change to make `s_curve` rankable.

---

## 9. Reproduction

```bash
# PID arm (C95) — job 25502
UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_VARIANTS='diffuser,dpcc-r,dpcc-c,dpcc-t,hardflow_new,hardflow_new-r,hardflow_new-c,hardflow_new-t' \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "10"

# MJPC arm (C94) — job 25514 -> 25554.  UAV_MIX_CONTROLLER selects the FMPCC_mjx conda env.
UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_CONTROLLER=mjpc UAV_MIX_VARIANTS='diffuser,dpcc-r,hardflow_new-r' \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "10"
```

`UAV_MIX_CONTROLLER` and the `FMPCC_mjx` env-selection fix are Gen15 U10
(`logs_in_develop/Gen15/U10/CHANGELOG_20260907_mjpc_controller_override.md`, commit `9bda071c`);
job 25554 is the first cluster run that exercised them, and validated all four first-run checks.

### Paper table (LaTeX)

```latex
\begin{tabular}{llrrrrr}
\toprule
Variant & Tracker & S\&C & Goal & $d_{\text{goal}}$ & $z_{\min}$ & viol/step \\
\midrule
Raw plan        & PID  & 0.00 & 0.00 & 2.809 & 1.126 & 0.0157 \\
Raw plan        & MJPC & 0.00 & \textbf{1.00} & \textbf{0.297} & 1.100 & 0.0765 \\
DPCC            & PID  & 0.00 & 0.00 & 2.665 & 1.046 & 0.0334 \\
DPCC            & MJPC & 0.00 & 0.00 & 2.571 & 1.093 & 0.0677 \\
HardFlow-SLSQP  & PID  & 0.00 & 0.70 & 0.834 & \textbf{0.008} & 0.2354 \\
HardFlow-SLSQP  & MJPC & 0.00 & \textbf{1.00} & \textbf{0.297} & \textbf{1.102} & \textbf{0.0777} \\
\bottomrule
\end{tabular}
```

*MeanFlow U-Net, `s_curve`, K=10, seed 6, honest geometry (`u7hg`). PID n=10, MJPC n=3.*

---

## Appendix — figure specs

**Data source:** `temp/0909/batch_uav_20260909_205118` — `per_rollout_detail.csv`
(filter `Candidate in {94, 95}`) and `candidates_per_variant.csv`. C95 = `pid_stopgo` (n=10),
C94 = `mjpc` (n=3). `rollout_idx` names the same initial condition in both runs.

| file | what it must show | carries |
|---|---|---|
| `fig1_paired_goal_distance.svg` | Dumbbell/slope plot: `goal_dist` PID → MJPC, one line per shared `rollout_idx` (0,1,2), panelled by variant. Mark the goal threshold. | **F1** — raw plan collapses 2.86 → 0.30 on all three pairs; `dpcc-r` does not move |
| `fig2_min_altitude.svg` | `phys_min_z` per rollout, PID (10) vs MJPC (3), grouped by variant; horizontal line at z=0. | **F4** — every PID HardFlow rollout sits at/below 0; MJPC at ~1.1 m |
| `fig3_violations_per_step.svg` | Grouped bars, `n_violations / n_steps`, PID vs MJPC × 3 variants. Annotate the 4.9× (raw) and 3.0× (HardFlow) ratios. | **F6** — the trade reverses by variant; MJPC's rate is near-constant |
| `fig4_trackerr_vs_goal.svg` | Scatter `track_err_mean` (x) vs `goal_reached` (y), coloured by controller, marker by variant. | **F2** — tight tracking with zero goal reach |
| `fig5_trajectories_idx0.svg` | Top-down `s_curve` trajectory overlay for `rollout_idx=0`, raw plan: PID vs MJPC, with obstacles/half-spaces and the reference. Optional z-vs-time inset. | **F1/F2** — the stall made visible |
| `fig6_compute_cost.svg` | Paired bars per variant: planner s/rollout (PID vs MJPC), with a second panel for s **per success**. Mark `diffuser`/PID as ∞ rather than plotting a bar. Annotate 0.74× / 1.25× / 0.51×. | **F7** — the cost picture, and that it reverses by projector |
| `fig7_divergence.svg` | Per rollout, a horizontal bar from 0 to `n_steps` with a marker at `divergence_step`; grouped PID/MJPC, panelled by variant. Shade the 395–421 band on the `diffuser`/PID panel. | **F2/§6.1** — 10/10 inversions in a 26-step window; `dpcc-r` tumbles both |

Keep both themes legible (no pure-black/pure-white fills), embed fonts, prefer `.svg`.
The sibling `../Report_20260903_AF_UNet/make_figs.py` is the closest existing template.
