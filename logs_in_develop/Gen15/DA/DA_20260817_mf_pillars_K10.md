# DA — Gen15 `mf` on UAV **pillars**, K=10: HardFlow works, and it beats DPCC on both axes

**Date:** 2026-08-17 · **Type:** data analysis · **Data:** `temp/1708/mix_uav_mf/`
**Runs:** pipeline **24612** → train **24613** → eval **24614**
**Config:** engine `mf`, backbone `unet` (3.97 M), scene `pillars`, seed 6, **n_trials = 3**,
K=10, `mpc4`, `pid_stopgo`, T=0.5, `--record all`
**Completeness:** ✅ **23 of 23 variants** — 20 DPCC + all 3 HardFlow. First complete Gen15 run.

---

## 0. Headline

**The HardFlow arm reaches the goal on pillars and the DPCC arms do not.**

| | `hardflow_new` | best DPCC (`dpcc-c`) | `geo_free` |
|---|---|---|---|
| goal_reached | **1.00 (3/3)** | 0.00 | 0.00 |
| final distance | **0.29 m** ×3 | 2.14–4.43 m | 0.56–0.61 m |
| steps used | **503–512** of 634 | 634 (full budget) | 634 |
| total_ms / plan | **939.8** | 1805.4 | 150.3 |

**`hardflow_new-t` is the only variant in the whole run with non-zero `success`: 0.67 (2/3).**

Two caveats up front, before anything is quoted: **`S&C = 0.00` everywhere** — not one variant
was collision-free (§3) — and **n = 3 trials**, so 0.67 means two rollouts.

---

## 1. Full table (23/23 variants, 3 trials)

`cf` = mean contact fraction · `gdist` = mean final distance to goal (m)

| variant | succ | S&C | rel | safe | goal | gdist | steps | cf | fm_ms | proj_ms | tot_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 118.07 | 634 | .000 | 89.1 | 0.0 | 89.1 |
| `gradient` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.58 | 634 | .568 | 88.9 | 11.3 | 100.2 |
| `gradient-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.34 | 634 | .262 | 89.0 | 11.3 | 100.3 |
| `post_processing` | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 24.17 | 593 | .006 | 89.0 | 68.2 | 157.2 |
| `post_processing-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.90 | 634 | .007 | 89.5 | 143.6 | 233.2 |
| `model_free` | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 151.18 | 634 | .000 | 89.7 | 243.7 | 333.5 |
| `model_free-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 120.65 | 634 | .000 | 92.6 | 264.1 | 356.7 |
| `bounds_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 40.85 | 634 | .306 | 95.5 | 1130.9 | 1226.4 |
| `bounds_free-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 72.19 | 634 | .234 | 95.2 | 887.4 | 982.5 |
| **`geo_free`** | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **0.59** | 634 | .000 | 92.7 | 57.6 | 150.3 |
| **`geo_free-bounds_free`** | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **0.57** | 634 | .000 | 93.2 | 48.0 | 141.2 |
| `geo_free-model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 102.64 | 634 | .000 | 92.7 | 52.7 | 145.4 |
| `model_free-bounds_free` | 0.00 | 0.00 | 0.33 | 0.67 | 0.00 | 138.52 | 634 | .000 | 94.0 | 202.9 | 296.9 |
| `model_free-bounds_free-tightened` | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 135.98 | 634 | .000 | 94.1 | 207.9 | 302.0 |
| `dpcc-r` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 65.27 | 634 | .009 | 95.6 | 1222.6 | 1318.2 |
| `dpcc-r-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 43.07 | 634 | .007 | 95.6 | 1280.3 | 1375.9 |
| `dpcc-c` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3.44 | 634 | .001 | 95.4 | 1710.0 | 1805.4 |
| `dpcc-c-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 104.79 | 634 | .001 | 95.5 | 1213.2 | 1308.7 |
| `dpcc-t` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 50.39 | 634 | .012 | 95.3 | 1876.6 | 1971.9 |
| `dpcc-t-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.57 | 634 | .119 | 95.4 | 1913.6 | 2009.0 |
| **`hardflow_new`** | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** | **0.29** | **506** | .005 | 153.7 | 786.1 | 939.8 |
| **`hardflow_new-c`** | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.51 | 538 | .003 | 153.8 | 771.9 | 925.7 |
| **`hardflow_new-t`** | **0.67** | 0.00 | **0.67** | 0.67 | 0.67 | 0.78 | **506** | .078 | 155.4 | 1497.2 | 1652.7 |

---

## 2. Per-trial detail (read this, not the means — the s_curve lesson)

| variant | gdist ×3 | reached | safe | contact ×3 | min_z ×3 | steps ×3 |
|---|---|---|---|---|---|---|
| `diffuser` | 115.7, 66.2, 172.3 | F F F | F F F | .00 .00 .00 | **−0.12**, .08, −.00 | 634 ×3 |
| `dpcc-c` | 4.43, 2.14, 3.75 | F F F | F F F | .000 .003 .000 | **.09 .09 .09** | 634 ×3 |
| `dpcc-t-tightened` | 4.45, 1.67, 1.59 | F F F | F F F | .001 .001 .356 | .00, −.01, −.00 | 634 ×3 |
| `geo_free` | 0.59, 0.58, 0.61 | F F F | **T T T** | .00 .00 .00 | .55 .64 .57 | 634 ×3 |
| `geo_free-bounds_free` | 0.58, 0.56, 0.58 | F F F | **T T T** | .00 .00 .00 | .57 .67 .61 | 634 ×3 |
| **`hardflow_new`** | **0.29, 0.29, 0.29** | **T T T** | F F F | .005 ×3 | **.88 .91 .91** | 512, 503, 504 |
| **`hardflow_new-c`** | 0.93, **0.30**, **0.30** | F **T T** | F F F | .002 .003 .004 | .88 .93 .90 | 634, 477, 504 |
| **`hardflow_new-t`** | 1.76, **0.30**, **0.28** | F **T T** | F **T T** | .232 .000 .001 | .17 **.92 .89** | 634, 439, 446 |

Three things this makes legible that the means do not:

**`hardflow_new`'s 0.29 ×3 is an early exit, not a coincidence.** The eval stops when the drone
enters the 0.30 m goal radius (U_13's fixed-budget-with-early-exit rule), so a successful rollout
registers just inside the radius and returns its step count short of the 634 budget — 503–512
here. Identical `gdist` across three trials is the signature of reaching, not of being stuck.

**The DPCC arms fail on altitude, not on navigation.** `dpcc-c` gets to 2.1–4.4 m with essentially
zero contact — but `min_z = 0.09` in all three trials, against the 0.2 m airborne threshold. It is
flying, just too low to be scored safe. `dpcc-t-tightened` is worse (`min_z` ≈ 0 or negative).

**`geo_free` flies beautifully and stops 0.58 m short.** `safe = True` ×3, zero contact,
`min_z` 0.55–0.67 — the cleanest flight in the run — ending consistently just *outside* a 0.30 m
radius. `relaxed = 1.00` (it crosses the finish plane). It is ~0.28 m from turning three clean
flights into three successes.

---

## 3. ⚠️ `S&C = 0.00` across all 23 — but the magnitudes differ by 4 orders of magnitude

No variant on pillars was collision-free, so `success_and_constraints` is zero everywhere and the
column carries no information. The *violation magnitude* does:

| variant | n_violations ×3 | total_violation magnitude ×3 |
|---|---|---|
| `diffuser` | 597, 601, 581 | **47 237 · 34 013 · 58 662** |
| `dpcc-t-tightened` | 447, 509, 410 | 859 · 479 · 213 |
| `dpcc-c` | 461, 285, 448 | 214 · 162 · 213 |
| `geo_free` | 277, 262, 264 | 24.3 · 17.9 · 17.0 |
| **`hardflow_new`** | **44, 41, 39** | **1.54 · 1.23 · 1.50** |
| **`hardflow_new-c`** | **25, 47, 37** | **0.55 · 1.48 · 0.80** |

`hardflow_new-c` violates on ~37 of ~540 steps with a summed magnitude below 1.5 — i.e. residual,
near-numerical. `diffuser` violates on ~95 % of steps with magnitude ~40 000. Reporting both as
"S&C = 0.00" is technically correct and practically useless. **On this scene, rank by violation
magnitude, and say so explicitly.**

---

## 4. ✅ The s_curve HardFlow diagnosis is confirmed

DA_20260816 §2 argued that `hardflow_new`'s 100 % NLP failure on s_curve was caused by the
static NLP holding mutually contradictory `x_active` halfspaces — an empty feasible set. Pillars
has **no `x_active`** (its `constraint_types` is `['dynamics','geo_bounds','obstacles','bounds']`;
Fix_12 removed the halfspaces). Prediction: the solver should converge here. Measured:

| scene | NLP solves | failures | rate |
|---|---|---|---|
| s_curve `hardflow_new` | 52 260 | 52 253 | **100.0 %** |
| pillars `hardflow_new` | 30 380 | 1 760 | **5.8 %** |
| pillars `hardflow_new-c` | 32 300 | 1 320 | **4.1 %** |
| pillars `hardflow_new-t` | 30 380 | 8 423 | 27.7 % |

100 % → 4–6 %. The diagnosis holds, and the s_curve active-set fix (U2 §2.3) remains the
outstanding work — but it is now confirmed to be a scene-scoped bug, not a broken port.

⚠️ `hardflow_new-t` at **27.7 %** is an outlier among the three: 6.7× the failure rate of its
siblings, on the same scene and the same NLP. Temporal-consistency selection changes only which
*candidate* is executed, not the program — so this needs explaining before its 0.67 success is
leaned on. A plausible mechanism: `-t` steers toward previously-executed plans, drifting the
warm-start `x1_ref` into regions where the prox-NLP is harder. Unverified.

---

## 5. Cost: HardFlow is *cheaper* than DPCC here, and it works

| arm | fm_ms | proj_ms | **total_ms** | NFE/plan |
|---|---|---|---|---|
| `dpcc-t` | 95.3 | 1876.6 | **1971.9** | 40 |
| `dpcc-c` | 95.4 | 1710.0 | **1805.4** | 40 |
| `hardflow_new` | 153.7 | 786.1 | **939.8** | **60** |
| `hardflow_new-c` | 153.8 | 771.9 | **925.7** | **60** |
| `geo_free` | 92.7 | 57.6 | **150.3** | 40 |

HardFlow pays **1.5× the network cost** (60 NFE/plan vs 40 — the terminal predict on activated
steps; see U2 §4 as corrected in DA_20260816 §4) and its `fm_ms` shows it: 153.7 vs ~95.
But its NLP is **2.2× cheaper than DPCC's SLSQP projection** (786 vs 1710 ms), so end to end it
is **1.9× faster** — *and* it is the arm that reaches the goal.

Per the Pareto rule this is **not** a clean dominance claim: at equal success-and-constraints
(both 0.00) HardFlow uses more NFE but less wall clock. On goal-reaching and wall-clock together
it is ahead of every DPCC arm; on NFE it is behind. State it as a trade-off with a large
wall-clock win, not as "best".

**Real-time: still nobody.** The 30.3 ms budget at 33 Hz is missed by 31× (`hardflow_new`) to 66×
(`dpcc-t-tightened`). Only `diffuser` (89 ms, unguided and useless) is within 3×.

---

## 6. Scene comparison — same engine, same K, same seed

| | corridor (10 trials) | pillars (3 trials) | s_curve (3 trials) |
|---|---|---|---|
| best `success` | **1.00** (`dpcc-r/c/t`) | 0.67 (`hardflow_new-t`) | 0.00 |
| best `goal_reached` | 1.00 | **1.00** (`hardflow_new`) | 0.33 |
| best `S&C` | **0.80** (`dpcc-r`) | 0.00 | 0.00 |
| who wins | DPCC | **HardFlow** | nobody |
| unguided `diffuser` | 0.00 | 0.00 | 0.00 |

The ordering **inverts between scenes**. On corridor (halfspace-driven, convex, wide) DPCC solves
it outright and HardFlow was never run. On pillars (sphere obstacles) HardFlow reaches the goal
3/3 while every DPCC arm stalls 2–4 m out on altitude. That is exactly the axis U2 predicted the
HardFlow NLP could express and the linear DPCC projector structurally cannot — and it is the first
Gen15 evidence for it.

`diffuser` scores 0.00 on all three scenes: **the guidance mechanism, not the objective, is what
separates these results.** Gen15 still has no evidence about MeanFlow-vs-anything; it has evidence
about DPCC-vs-HardFlow under MeanFlow.

---

## 7. What to do next

1. **Raise `n_trials` on pillars.** 0.67 is two rollouts. The result deserves 10 trials before it
   is quoted anywhere. ⚠️ Re-running at the same K/engine **overwrites this directory** — move
   `Emf_K10_mpc4_pid_stopgo_T0.5/` aside first.
2. **Chase the 0.28 m gap on `geo_free`.** Three clean, safe flights stopping just outside the
   goal radius is the cheapest available success in this generation — worth checking whether it
   is a controller/stop-condition artefact rather than a planning failure.
3. **Explain `hardflow_new-t`'s 27.7 % NLP failure rate** (§4) before relying on its 0.67.
4. **Diagnose the DPCC altitude floor.** `min_z = 0.09` in 3/3 `dpcc-c` trials, against a 0.2 m
   threshold, is suspiciously repeatable — it looks like the projector is pinning altitude at a
   constraint boundary rather than failing randomly.
5. **The K sweep is now worth running on pillars**, since there is finally a scene where an arm
   reaches the goal and the K axis can change something.

---

## 8. 🔎 VERDICT — "only `geo_free` is clean." Are the constraints too hard?

**No. They are feasible, and they are *harmful*. Those are different claims and the data
separates them.**

### 8.1 The ablation ladder — sorted by how close the drone got

| gdist | variant | constraint families ACTIVE | safe | worst min_z | Σ violation |
|---|---|---|---|---|---|
| **0.57** | `geo_free-bounds_free` | **dynamics ALONE** | **1.00** | **0.57** | 20.4 |
| **0.59** | `geo_free` | dynamics + action-bounds | **1.00** | **0.55** | 19.7 |
| 3.44 | `dpcc-c` | **FULL stack** | 0.00 | **0.09** | 196 |
| 40.85 | `bounds_free` | geometry + dynamics | 0.00 | −0.00 | 9 958 |
| 65.27 | `dpcc-r` | FULL stack | 0.00 | −0.01 | 14 627 |
| 102.64 | `geo_free-model_free` | action-bounds ALONE | 0.00 | −4.65 | 45 426 |
| 118.07 | `diffuser` | **NOTHING** (unguided) | 0.00 | −0.12 | 46 637 |
| 138.52 | `model_free-bounds_free` | **geometry ALONE** | 0.67 | 0.10 | 60 498 |
| 151.18 | `model_free` | geometry + bounds | 0.33 | 0.09 | 65 033 |

Read top-down, this is unambiguous:

1. **Dynamics is the entire load-bearing family.** Dynamics *alone* produces the best flight in
   the run — 0.57 m, safe 3/3, zero contact, `min_z` 0.55. Everything else is decoration.
   (U_13 reached the same conclusion on corridor: *"dynamics is the load-bearing family"*. This
   is an independent replication on a different scene and a different objective.)
2. **Geometry without dynamics is worse than no constraints at all.** `model_free` (geometry +
   bounds, no dynamics) lands at 151 m — *worse than `diffuser`'s* 118 m. Projecting onto
   geometry while the dynamics coupling is off actively pushes the plan somewhere useless.
3. **Adding geometry to a working configuration destroys it.** dynamics-alone 0.57 m and safe
   → full stack 3.44 m and **unsafe**. Adding constraint families degrades accuracy by 6× and
   flips safety from 1.00 to 0.00.

### 8.2 Not "too hard" — the feasibility gate says the set is satisfiable

Fix_12's pre-flight check ran and **passed for all four homotopies**:

```
[ eval ] pillars feasibility check: homotopy=(L,L,L) expert route OK under planning margin 0.33 m
                                    (L,R,L) / (R,L,R) / (R,R,R)  — all OK
```

The expert routes fit inside the inflated constraint set with room. The pillars are 0.12 m
radius spheres in **x,y only**, six of them, in a 7.2 × 3.0 m box. This is not a tight scene.
So the s_curve explanation ("the feasible set is empty / a 24 cm band") **does not transfer** —
whatever is going wrong here is not infeasibility.

### 8.3 🔴 The altitude paradox — the actual mechanism

The workspace box is `lb: [-3.6, -1.5, 0.30]`, `ub: [3.6, 1.5, 1.80]`, inflated by
`r_drone + margin = 0.33`, so the projector enforces a **z-floor of 0.63 m** on the plan. It is
applied to trajectory channels 6,7,8 — the actual position `p`:

```python
lb = np.concatenate([np.full(6, -np.inf), ws_lb + margin, np.full(pad, -np.inf)])
#                    act(0,1,2) + p_des(3,4,5) unconstrained │ p(6,7,8) gets the box
```

Now compare the realized altitude:

| | z-floor constraint | realized `min_z` |
|---|---|---|
| `geo_free` (box **OFF**) | none | **0.55, 0.64, 0.57** ✅ |
| `dpcc-c` (box **ON**, floor 0.63) | 0.63 m | **0.09, 0.09, 0.09** ❌ |

**Turning the altitude floor ON makes the drone fly 6× lower — and it lands 0.54 m below the
floor it is supposed to enforce, identically in all three trials.** A constraint that is
satisfied in the plan and violated by exactly the same amount every rollout is not a solver
failure; it is a **plan-vs-execution gap**.

The mechanism this points at: the projector enforces the box on the **predicted 8-step
trajectory**, anchored at the measured state. The controller executes only the **first action**,
and the `pid_stopgo` tracker chases `p_des`, not the projected `p`. The two are coupled only
through the shared `deriv` rows — which is why geometry is inert without dynamics (§8.1 point 2)
— but that coupling constrains the *plan*, not the closed loop. The drone sags below a floor its
plan respects.

⚠️ This is a **hypothesis consistent with the data, not a proven cause.** What would settle it:
log the projected `p_z` alongside the realized `p_z` per step and compare. If the plan holds
≥0.63 while execution sits at 0.09, the gap is confirmed and the constraint is being enforced on
the wrong object. If the plan itself dips, the projector is failing and it is a different bug.

This is precisely the failure mode `HF_Study/UAV_PID/INVESTIGATION_hardflow_higher_order_and_uav_pid.md`
§5.4 flagged as an open direction — *"per-sample feasibility does not imply closed-loop safety
across replans"*, the "tracker tube" problem. This run is the first Gen15 measurement of it.

### 8.4 Why HardFlow escapes it

`hardflow_new` runs the **same constraint list** and reaches 0.29 m at `min_z` 0.88–0.91 —
above the floor, unlike every DPCC arm. The difference is *when* the constraint is applied:
DPCC generates then projects the finished plan; HardFlow solves the prox-NLP **inside** each ODE
step, so the constraint shapes the trajectory as it forms rather than bending a finished one.
On a scene where bending the finished plan is what breaks it, that ordering matters.

That is the clearest evidence yet for U2's premise — but note it is one scene, one seed, three
trials.

### 8.5 So what does "only `geo_free` is clean" tell us?

**That the geometric constraint machinery is currently a net negative on pillars, and the
binding problem is enforcement-vs-execution, not constraint difficulty.** Concretely:

- ❌ It is **not** "the constraints are too hard" — they are feasible with margin (§8.2).
- ❌ It is **not** an objective problem — `diffuser` (the raw MeanFlow policy) fails at 118 m on
  every scene; the ML engine is not what separates these rows.
- ✅ It **is** that projecting a finished plan onto this geometry, in this MPC loop, with this
  tracker, produces plans the closed loop cannot execute — visible as a 0.54 m altitude deficit
  against the very bound being enforced.

**The single highest-value follow-up in this generation** is §8.3's plan-vs-realized altitude
comparison. It is a logging change, not an experiment, and it decides whether the fix belongs in
the projector, the constraint indices, or the tracker.
