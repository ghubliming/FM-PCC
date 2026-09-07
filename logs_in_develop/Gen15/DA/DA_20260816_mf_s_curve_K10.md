# DA — Gen15 `mf` on UAV **s_curve**, K=10: a total wipe-out, and the bug that explains one arm of it

**Date:** 2026-08-16 · **Type:** data analysis · **Data:** `temp/1608/`
**Runs:** train **24588** · eval **24589** (git `8a543b0`, node i6-gpu-1)
**Config:** engine `mf`, backbone `unet` (3.97 M), scene `s_curve`, seed 6, **n_trials=3**, K=10,
`mpc4`, `pid_stopgo`, T=0.5, `--record all`
**Completeness:** ⚠️ **21 of 23 variants.** `hardflow_new-c` was on trial 2/3 when the log ends;
`hardflow_new-t` never started. Everything below is the 21 that finished.

---

## 0. Headline

**Every variant scored `success = 0.00`** — but that headline is misleading on its own, and the
per-trial data (§1.5) tells a different story from the means. **One trial did reach the goal**
(`geo_free` trial 1, `goal_dist = 0.30 m`, `reached = True`); it scored 0 because it was not
*safe*. Six more trials finished within 4 m. The means hide this because s_curve failures are
bimodal — see §1.5, which is the section to read first.

> 🔴 **Correction (2026-08-16, same day).** The first version of this DA said "not one of the 21
> reached the goal". That was **wrong** — it read the `success` column and the *mean* `gdist`
> without opening the per-trial arrays. `geo_free`'s mean `gdist` of 16.49 is the average of
> **[46.8, 0.30, 2.35]**. Averaging a bimodal distribution produced a number that describes no
> trial that actually happened. §1.5 was added; §1's means are kept but must not be read alone.

This is not a Gen15 regression. `U_13`'s investigation already concluded s_curve is pathological
for the geometry-keeping variants (non-convex feasible set, ~24 cm bands, switched per segment,
871-step horizon), and it predicted exactly this ordering. What is new is that at K=10 the
MeanFlow arm fails there too, and that **one arm failed for a specific, fixable reason (§2).**

---

## 1. The table (21 completed variants, 3 trials each)

`succ`/`S&C` = strict success, and success∧constraints · `gdist` = final distance to goal (m)

| variant | succ | S&C | rel | safe | goal | gdist | steps | terr | fm_ms | **proj_ms** | tot_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.58 | 871 | 402.5 | 89.0 | **0.0** | 89.0 |
| `gradient` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.02 | 871 | 5.0 | 89.6 | 7.2 | 96.7 |
| `gradient-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.01 | 871 | 6.7 | 89.5 | 7.1 | 96.6 |
| `post_processing` | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 83.77 | 871 | 125.4 | 89.3 | 82.6 | 171.9 |
| `post_processing-tightened` | 0.00 | 0.00 | 0.33 | 0.67 | 0.00 | 148.81 | 871 | 131.2 | 89.3 | 78.2 | 167.5 |
| `model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.71 | 871 | 404.4 | 89.7 | 125.9 | 215.6 |
| `model_free-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.73 | 871 | 402.9 | 89.7 | 131.0 | 220.8 |
| `bounds_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 113.42 | 871 | 371.8 | 90.0 | 496.9 | 586.9 |
| `bounds_free-tightened` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 106.47 | 871 | 305.9 | 89.8 | 473.7 | 563.5 |
| **`geo_free`** | 0.00 | 0.00 | 0.00 | 0.00 | **0.33** | 16.49 | **799** | 68.2 | 89.6 | 164.9 | 254.5 |
| `geo_free-bounds_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 36.52 | 871 | 61.1 | 89.6 | 131.9 | 221.5 |
| `geo_free-model_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.56 | 871 | 403.4 | 89.4 | 53.6 | 143.0 |
| `model_free-bounds_free` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.64 | 871 | 406.4 | 89.4 | 96.0 | 185.5 |
| `model_free-bounds_free-tightened` | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 77.70 | 871 | 371.1 | 89.8 | 109.2 | 199.0 |
| `dpcc-r` | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 120.45 | 871 | 166.8 | 90.5 | 948.3 | 1038.8 |
| `dpcc-r-tightened` | 0.00 | 0.00 | 0.33 | 0.67 | 0.00 | 145.25 | 871 | 155.2 | 90.3 | 1032.4 | 1122.7 |
| `dpcc-c` | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 222.43 | 871 | 149.2 | 90.5 | 1062.8 | 1153.3 |
| `dpcc-c-tightened` | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 113.01 | 871 | 136.3 | 90.5 | 1119.4 | 1209.9 |
| `dpcc-t` | 0.00 | 0.00 | 0.33 | **1.00** | 0.00 | 238.88 | 871 | 159.7 | 90.4 | 1030.9 | 1121.3 |
| `dpcc-t-tightened` | 0.00 | 0.00 | 0.33 | **1.00** | 0.00 | 228.33 | 871 | 163.5 | 90.3 | 1103.2 | 1193.5 |
| **`hardflow_new`** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 6.31 | 871 | 166.4 | **150.0** | **5171.9** | **5321.9** |

**The DPCC arms trade goal-reaching for safety.** `dpcc-c`/`dpcc-t`/`dpcc-t-tightened` hit
`safe = 1.00` — contact-free, airborne, all three trials — while ending **222–239 m** from the
goal. The projector keeps the drone alive by refusing to let it commit to the corridor at all.
`geo_free` is the mirror image: drop the geometry and it gets closest (`gdist` 16.5, the only
`goal_reached > 0`, and the only variant to finish early at 799 steps) but is never safe.
That is `U_13`'s corridor-inverted ordering reproduced exactly, now on the MeanFlow objective.

---

### 1.5 🔴 Read this before §1 — the per-trial picture, and THREE distinct failure modes

`n_trials = 3`, so every mean in §1 is over three numbers. On this scene those three are usually
*not* near each other, and the mean lands in a gap where nothing happened.

**Per-trial final distance to goal (m), with contact fraction and minimum altitude:**

| variant | goal_dist ×3 | contact_frac ×3 | min_z ×3 | mode |
|---|---|---|---|---|
| `diffuser` | 6.7, 6.4, 6.6 | .004, **.956**, .292 | .09, .35, .08 | **stuck** |
| `model_free` | 6.7, 6.8, 6.6 | .41, .004, .228 | .06, .08, .07 | **stuck** |
| `hardflow_new` | 6.5, 6.2, 6.3 | .004, **.971**, **.937** | .07, .34, .16 | **stuck** |
| `gradient` | **3.4**, 5.5, 6.2 | .044, .872, .931 | .01, .06, .30 | crawls |
| **`geo_free`** | 46.8, **0.30** ✅, 2.35 | .042, .054, .501 | .00, .14, −.00 | **reaches, unsafely** |
| `geo_free-bounds_free` | **1.7**, **1.6**, 106.2 | .51, .45, .033 | .27, .01, .01 | close, scraping |
| `post_processing` | **1.2**, 70.9, 179.2 | .501, .002, .002 | .26, **−992.66**, 1.04 | close once, then diverges |
| `dpcc-c-tightened` | **1.2**, 163.3, 174.5 | .021, .004, .003 | .08, 1.01, .98 | close once, then flies off |
| `dpcc-r` | **3.8**, 177.5, 180.0 | .699, .001, .000 | .31, 1.07, .15 | close once, then flies off |
| `dpcc-c` | 220.5, 214.3, 232.5 | .002, .001, .002 | .91, 1.02, .99 | **clean flight, wrong way** |

**Mode A — stuck at the start (`gdist` ≈ 6.2–6.9, variance < 0.5 m).** Six variants sit in a tight
band with `contact_frac` up to **0.97** and `min_z` ≈ 0.07–0.09. The drone is *on the ground*, for
up to 97 % of an 871-step episode. It never departs. The ≈6.5 m is a fixed geometric quantity —
the start-to-goal separation — which is why six unrelated variants agree on it to within 0.5 m.
`diffuser`, `model_free`, `geo_free-model_free`, `model_free-bounds_free`, `hardflow_new` and
`model_free-tightened` are all in this mode. **`hardflow_new` being here is fully explained by
§2** (its NLP is infeasible, so it never produces a usable plan).

**Mode B — reaches or nearly reaches, but unsafely.** `geo_free` trial 1 lands at **0.30 m** and
is scored `reached = True`. It fails `success` on the safety half: `contact_frac = 0.054` and
`min_z = 0.14`, under the 0.2 m airborne threshold. It got there by scraping. `geo_free`
trial 3 (2.35 m) is worse — `contact_frac = 0.50`, `min_z` slightly **negative**, i.e. through
the floor. `geo_free-bounds_free` reaches 1.6–1.7 m with `contact_frac` 0.45–0.51: same story.

**Mode C — clean flight, wrong direction.** `dpcc-c` is the extreme: `contact_frac` 0.001–0.002,
`min_z` 0.91–1.02 — textbook stable flight at cruise altitude — ending **214–232 m** from the
goal in all three trials. The projector produces a perfectly safe trajectory that goes nowhere
near the corridor.

**So the answer to "did nothing get close?" is no.** Seven of 63 trials finished within 4 m, and
one reached. What no trial did was reach **and** stay safe — which is what `success` requires.

⚠️ **One numeric outlier to be aware of:** `post_processing` trial 2 records
`min_z = −992.66`. That is not a low flight; it is a divergence — the state left the world. Any
aggregate including that trial (its `gdist` mean of 83.8, for one) is meaningless.

---

## 2. 🔴 THE BUG: the HardFlow arm is structurally wrong on s_curve — 100 % NLP infeasibility

`hardflow_new` reports:

```
nlp_solves_total     52260
nlp_failures_total   52253      →  failure rate 100.0%   (7 solves converged, out of 52 260)
nfe_per_plan         60.0
proj_ms              5171.9 ms/plan
```

**Every single NLP solve failed.** IPOPT's `solve_limited()` raised, and the sampler fell back to
`opti.debug.value(x1)` — the last iterate — on essentially every step. So this arm spent
**5.2 seconds per plan** to enforce nothing at all. `success`, `safe` and `goal` are all 0.00,
and `gdist` 6.31 says it barely moved.

### 2.1 Why — and it is my bug, introduced in U2

s_curve is **the only scene with `x_active` halfspaces** (`config/uav_projection.yaml:217-220`):
each wall is live only over its own x-range, so the active wall set depends on where the drone
currently is. Gen11 handles this by rebuilding the projector every replan:

```python
_has_x_active = (variant != 'diffuser') and any(... x_active ...)
if rebuild_projector is not None:
    policy.projector = rebuild_projector(float(p[0]))     # eval_mix_uav.py:1058
```

`HardFlowPolicy` sets `self.projector = None` and builds its NLP **once**, in `__init__`, from a
static `constraint_list`. So that assignment is a **silent no-op for arm C**: the NLP holds *all
four* wall halfspaces for the entire flight.

Those four are mutually contradictory:

| segment | live over | constrains |
|---|---|---|
| seg1 (upper `below` y=−0.35, lower `above` y=−1.25) | x ∈ [−3.0, −0.5] | −1.25 ≤ y ≤ −0.35 |
| seg2 (lower `above` y=0.35, upper `below` y=1.25) | x ∈ [0.5, 3.0] | 0.35 ≤ y ≤ 1.25 |

**The intersection is empty.** Held simultaneously, they demand the drone be in two disjoint
y-bands at once, so the feasible set is empty by construction and IPOPT cannot converge — ever.
The 100.0 % failure rate is not a solver-tuning issue; it is an infeasible program.

### 2.2 What this does and does not invalidate

- ❌ **`hardflow_new` on s_curve is void.** Do not quote it. It is not a HardFlow result; it is
  a measurement of an empty feasible set. The same applies to the unfinished `hardflow_new-c`
  and `hardflow_new-t`.
- ✅ **Every DPCC row in §1 is valid** — those rebuild correctly.
- ✅ **HardFlow on corridor/pillars/empty is unaffected** — none of them declares `x_active`, so
  the static NLP is the correct NLP there.

### 2.3 The fix (not yet written)

`HardFlowPolicy` needs a `rebuild_nlp(current_x)` path so the eval's per-replan hook reaches arm
C, or the NLP must take the active set as a *parameter* rather than baking it into the
constraint rows. The second is better — CasADi `Opti` parameters exist for exactly this and it
avoids rebuilding the program 871 times per rollout — but it is a real change to the port's
formulation, not a patch. Until then, **arm C should be disabled on s_curve**
(`UAV_MIX_HF_OFF=1`), which also saves ~4 h/variant.

---

## 3. ✅ Fix_1 is confirmed working in production

This is the first run since Fix_1, and it validates cleanly:

- **`proj_ms` is non-zero on 20 of 21 rows.** The one zero is `diffuser`, which has no projector
  — correct.
- **`fm_ms` is now flat at 89–90.5 ms across every DPCC variant**, matching the unprojected
  `diffuser` (89.0) and gate G6's isolated K=10 measurement (87.2 ms). Pure inference is
  constant; the projector is what varies. Before Fix_1 this column read 96.6 → 1193.5 and would
  have looked like the MeanFlow *network* getting 13× more expensive under projection.

The HardFlow NLP timing (U2 §9) also works: `proj_ms = 5171.9` on arm C.

---

## 4. NFE accounting — correcting U2 §4

U2's changelog said HardFlow costs **2K** network evals to DPCC's K. Measured: `nfe_per_plan =
60.0` at K=10, B=4. The actual structure is

```
(10 reference steps + ~5 terminal predicts) × B=4 = 60      vs   DPCC's 10 × 4 = 40
```

The terminal predict only runs on **activated** steps (`activation_threshold = 0.5` → the late
half), so the real ratio is **1.5×, not 2×**. Quote `nfe_per_plan`; it is measured, and it moves
with the activation threshold.

---

## 5. Scene contrast: corridor vs s_curve, same engine, same K

| | corridor (K=10, 10 trials) | s_curve (K=10, 3 trials) |
|---|---|---|
| best `success` | **1.00** (`dpcc-r`, `dpcc-c`, `dpcc-t`, `bounds_free`) | **0.00** (all 21) |
| best `S&C` | 0.80 (`dpcc-r`) | 0.00 |
| `dpcc-c` steps | 259 / 396 | 871 / 871 (never finishes) |
| `dpcc-c` proj_ms | ~181 (reconstructed) | **1062.8** (measured) |
| `dpcc-c` total_ms | 269.7 | 1153.3 |
| unprojected `diffuser` | 0.00 success, gdist 39.3 | 0.00 success, gdist 6.58 |

The projector is **5.9× more expensive** on s_curve than corridor, and buys nothing. Note also
that the raw policy is *not* the differentiator: `diffuser` scores 0.00 on both scenes. On
corridor the DPCC projector converts that into 1.00; on s_curve it cannot.

**Training was healthy** and is not the explanation: `raw_mse_u` 0.401 train / 0.604 test (better
than corridor's 0.154/0.717 on the test side), `h_mse_b3` 0.309 (vs corridor's 0.555), zero
tracebacks. The s_curve model learned *better* than the corridor model by every logged metric and
still cannot fly the scene.

---

## 6. Real-time: not remotely

`total_over_budget = 2613` on every variant — i.e. **every step of every trial** exceeded the
30.3 ms budget. Measured per-plan cost: 89 ms unprojected, ~1.1 s under DPCC, **5.3 s** under the
(broken) HardFlow arm. At 33 Hz the budget is 30.3 ms, so DPCC on s_curve is **38× over** and
arm C is **176× over**.

---

## 7. What this run establishes

1. **s_curve at K=10 is not solved SAFELY by this arm under any of the 20 DPCC variants** — but
   it is not out of reach either. One trial reached the goal (unsafely) and six more finished
   within 4 m (§1.5). The binding failure is **safety, not navigation**: the trials that get
   close do it by scraping the ground (`contact_frac` 0.45–0.50, `min_z` ≈ 0), and the trials
   that fly cleanly go 200 m the wrong way. That is a much more tractable problem than "cannot
   get near the goal", and it points at the altitude/contact behaviour rather than the objective.
2. **The HardFlow arm has a scene-scoped structural bug** (§2) that must be fixed before arm C
   means anything on s_curve. Its corridor/pillars results remain valid.
3. **Fix_1 is verified in production**, so from here the `fm_ms` / `proj_ms` split can be trusted
   and quoted.
4. **The DPCC projector's behaviour on this scene is "stay safe, go nowhere"** — `safe = 1.00`
   at 222–239 m from goal. That is a real, reportable finding about the constraint geometry, not
   a failure of the objective.

### Suggested next steps

- **Do not** re-run s_curve arm C until §2.3 is fixed; set `UAV_MIX_HF_OFF=1` if s_curve is
  re-run for the DPCC rows.
- **pillars** is the scene that still needs running — it has sphere obstacles and no `x_active`,
  so it exercises HardFlow on a formulation that is actually correct.
- The `geo_free` row (closest approach, only early finish) suggests s_curve may be reachable
  with geometry off; a targeted `geo_free`-only run at higher `n_trials` would settle whether
  0.33 goal-reach is signal or one lucky trial out of three.
- ⚠️ **n_trials = 3.** Every rate in §1 is out of 3, so the resolution is 0.33. Treat
  `0.33 vs 0.67` differences as noise.
