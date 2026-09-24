# TO v3 — R45a is ready: the corridor re-scored with the finish line at the end of the corridor (x′ = 2.0 m)

**2026-09-24 · from the DA side (U19 corridor chat).** Nothing in `v3/` was touched.
> **Update (later 24-09): §6 is new.** The corridor frontier gets a **second reading**, the near-satisfaction bar
> (author). The tables stay strict.
- **Analysis of record:** `Data_Analysis/DA_in_Paper/analysis/DA_20260924_corridor_v3_R45a_clear_line.md`, script
  `corridor_v3_r45a_clear_line.py`. A row was added to `analysis/INDEX.md`. `DA_20260924_corridor_v3.md` has a banner
  pointing here; nothing else in it changed.
- **Data only, no run.** Same corpus as v3.80: `batch_uav_20260924_081422` + the npz, first ten flights, 1 360 flights.
- **The eval now uses the same rule** (author's go-ahead, 24-09; `logs_in_develop/Gen15/U19/CHANGELOG_20260924_U19_corridor_finish_line_coding2.md`).
  In `mix_uav_test/eval_mix_uav.py` a corridor flight's success now latches at x ≥ 2.0. The test runs after physics,
  the same timing as the old latch. The flights themselves do not change: no new early stop, and violations are counted
  as before. The goal-plane rule stays in `results.json` as `success.relaxed_goal_line`. So Ch 5 can describe x′ as the
  evaluator's own rule, not a re-scoring.
- Your preview (`R45_preview/`) reproduces exactly on the reading it used. **One flight** comes out differently when read
  at the eval's own latch timing (§2).

## 1 · The rule (Ch 5 success clause)

- **Success at x′:** the original success, **or** the vehicle's centre reaches **x ≥ 2.0 m** within the 396 steps, in
  controlled flight. x = 2.0 m is the end of the corridor: the walls and both test-time constraints end there.
- **S&C at x′:** success at x′ **and** no violating step over the whole flight, counted as before.
- **Where x is read:** on the position after each control step, which is when the eval tests its own finish latch.
- **Honesty clause** (spec §2): print the original counts beside the new ones. The DA gives both everywhere.

## 2 · The flight at x = 2.0 (the author's question)

Hump, baseline, per-step **r**, trial 7, route C. It is safe and violation-free.
- Furthest recorded position (the start of its 396th step): **x = 1.998785 m**, which is **1.2 mm short** of the line.
- After its 396th and last step: **x = 2.0167 m**, which is **16.7 mm past** the line, moving +18.8 mm per step.

**Read after each step (the eval's latch timing):** the flight is past the line, so hump r is **10/10** and it is the
baseline's best hump rule.

**Read on recorded positions only (your preview):** hump r is 9/10, and by the tie-break the best rule becomes c (9/10,
623.1 ms).

The DA uses the first reading. The choice between the two is the author's. The hump frontier is a trade-off under both
readings, and nothing else in the corpus depends on which one is chosen.

## 3 · What changes in Chapter 6 (baseline rows only)

**The baseline under per-step projection:**

| block | rule | S&C original → x′ | success original → x′ | ms/step |
| :-- | :-- | :-- | :-- | --: |
| tilt | r / c / t | 0 / 0 / 0 → **3 / 8 / 3** | 2 / 0 / 3 → 10 / 10 / 10 | 679.7 / 666.3 / 650.6 |
| hump | r / c / t | 3 / 3 / 6 → **10 / 9 / 9** | 3 / 3 / 7 → 10 / 9 / 10 | 654.2 / 623.1 / 625.9 |

**Tables:**
- `tab:uav-corridor-raw`: unchanged.
- `tab:uav-corridor-projection`: only the baseline row changes, as in the table above.
- `tab:uav-corridor`, the baseline row:
  - tilt: 0/10 c 666.3 → **8/10 c 666.3**;
  - hump: 6/10 t 625.9 → **10/10 r 654.2** (recorded-only reading: 9/10 c 623.1).
- Every flow-model row keeps its S&C.

**The frontier:**
- **Tilt:** none → **baseline per-step c (8/10, 666.3 ms)**, the only eligible point.
- **Hump:** FM $\nfe=20$ endpoint single (8/10, 291.0 ms) alone → FM $\nfe=20$ endpoint single **and** baseline per-step r
  (10/10, 654.2 ms). Neither dominates: this is a **trade-off**, two more violation-free successes for 2.2 × the time per
  action.

**Sentences that hold only under the original rule:**
- v3.80's "No frontier figure is drawn … because the frontier has at most one point". At x′ the hump has **two**
  non-dominated points, so a sentence in the text is enough. Note that `frontier.py::fig_uav_corridor_tradeoff` still
  draws the archived v2 corpus.
- "FM $\nfe=20$ endpoint is ahead of the baseline on the hump" and "the baseline point is dominated".
- The §6.4 facts "the best corridor configuration is FM $\nfe=20$ endpoint single; on the tilt no combination succeeds
  violation-free". At x′ no configuration is best:
  - on the **tilt** only the baseline succeeds violation-free (8/10 against 0/10 for every flow configuration);
  - on the **hump** FM $\nfe=20$ endpoint single and the baseline per-step r are both non-dominated.

  §6.4 is 🔒 and remains the author's call.
- §3.5 paths paragraph, "the baseline's unsuccessful flights run out of steps 0.16–0.39 m before the line". At x′, 41 of those
  42 flights had passed the end of the corridor and were still moving toward the goal (1.8–20.0 mm per step). The one that had
  not, hump c trial 7, ends at x = 1.23 m.

**Flow side:**
- No S&C value changes.
- Three $\nfe=2$ wall-drift flights reach x′ and count as successes: MeanFM 1 and CI-MeanFM 2, both rule t. None is
  violation-free, so their S&C stays 0.
- Where a sentence says "0 of 180 flights cross" at $\nfe \le 2$, the x′ reading is "3 of 180 reach the end of the corridor,
  all three after drifting across the wall's boundary".
- The budget facts (violating steps, depths, ms) do not move. Neither does the model comparison (the v3.80 wording "within two
  violating steps, each at its best rule").

## 4 · The author's four checks, on all 44 flights that become successes

1. **Past x′ before the cap:** all 44. The baseline's first recorded step past 2.0 is step 260–382 of 396. The §2 flight
   crosses during its last step.
2. **Safe, no abort:** all 44. No corridor flight aborted or was unsafe.
3. **Still moving toward the goal at the cap:** all 44. The baseline flights move 1.8–20.0 mm per step along x.
4. **No violating step past x′:** 41 of 44, every baseline flight among them. The three exceptions are the flow drift
   flights, which clip a wall-end cap at x 2.01–2.23 and are S&C 0 in any case.

Of the 41 baseline flights, 30 are violation-free and 11 are not. The 11 have 1–12 violating steps, all on the tilt and all
before x = 2.0.

## 5 · One correction to the spec

The hump baseline flight that ends at x ≈ 2.26 and drops out between x′ = 2.0 and 2.31 is **c, trial 8, route R**, not
trial 7. Trials are counted from 0, trial *i* is seeded `default_rng(10000 + i)`, and its route is `LCR[i % 3]`. Trial 7 of
that cell (route C) ends at x = 1.23 m and is short of every line.

## 6 · The corridor frontier gets a second reading: the near-satisfaction bar (author, later 24-09)

- **Definition.** A flight is *near-satisfying* when it succeeds at x′ **and** at least **95 %** of its control steps are
  violation-free. 0.90 gives the same frontier; 0.97 is the sensitivity to print.
- **Scope.** UAV-corridor only, and only the frontier. **The tables and every S&C column stay strict** (success at x′ and
  zero violating steps). Print the strict frontier (§3) beside this one.
- **Frontier at ≥ 0.95:**
  - tilt: **nfe = 1, per-step c, 10/10**: MeanFM 27.4 ms = FM 27.6 (CI-MeanFM 27.9);
  - hump: **nfe = 3, endpoint c, 10/10**: CI-MeanFM 71.9 ms = FM 71.9 = MeanFM 72.1.
- **What happens at ≥ 0.95.**
  - Every projected configuration reaches 10/10, except per-step projection at nfe ≤ 2 on the hump, where the flights stall
    on the roof. The baseline is 10/10 too.
  - The frontier is therefore simply the cheapest configuration.
  - The flow models are ahead of the baseline by **24× (tilt) and 9× (hump)** in time per action at the same 10/10.
- **What it does not show:** that MeanFM or CI-MeanFM beat FM. **The three flow models tie** (within 2 % at the same
  count). No reading of the corridor, strict or near, separates them.
- **Caveats to state with it.**
  - **Knife-edge:** violating MeanFM/CI-MeanFM flights are 0.80–0.99 violation-free (median 0.971). At 0.97 the tilt's
    nfe 1–2 cells fall to at most 4/10, and the tilt frontier becomes a staircase up to FM nfe=3 per-step (10/10, 79 ms).
  - **Floor:** unprojected flights are about 66 % (tilt) and 87 % (hump) violation-free. Below about 0.87 the bar would
    pass unprojected hump flights, so the defensible range is about 0.90–0.96.
  - **What 0.95 forgives:** up to about 14 steps (≈ 0.4 s) per flight inside the constraint, up to 10 cm deep (tilt,
    nfe 1–2).
- **Honesty.** This is the second change made on this scene after the data were seen: first the finish line, now this bar.
  Declare 0.95 as the bar, show 0.97, and keep the strict frontier beside it. The other environments keep the strict score.
- **Where:** `DA_20260924_corridor_v3_R45a_clear_line.md`, Appendix C. It has the frontier under each bar and every
  configuration's count (strict / ≥ 0.95 / ≥ 0.97, rule, ms).
