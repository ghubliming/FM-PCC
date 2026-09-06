# DA 2026-09-06 — Gen15 U7 honest geometry: first results

**Runs:** 25419–25424 (6 evals, `fm`/`mf` × `pillars`/`corridor`/`s_curve`, K=2, seed 6, tag `u7hg`)
**Code:** `963faed` · **Batch:** `batch_uav_20260906_230451`
**Predecessor:** [`DA_20260903_fix16_AB_mf_pillars.md`](DA_20260903_fix16_AB_mf_pillars.md) Part II §II.3
**Change under test:** [`Gen15/U7/CHANGELOG_20260904_honest_geometry_and_slack_gate.md`](../U7/CHANGELOG_20260904_honest_geometry_and_slack_gate.md)

---

## 0. TL;DR

1. 🟢 **`pillars` is no longer 0.** S&C goes **0.00 → 0.70** (`mf`, `dpcc-r`, K=2). The
   0/2876 result that stood across every engine, every K and both Fix_16 arms was a
   property of the benchmark, not of the engines.
2. 🟢 **`corridor` is solved.** S&C = **1.00** on 17 of 20 arm×variant cells, 0 violations,
   0 aborts. Its expert route previously cleared the planning set by **0.000 m** — provably
   infeasible.
3. 🔴 **The proof is a row where the trajectory cannot have changed.** `pillars`/`mf`/`diffuser`
   runs with **projection off**. Old vs new geometry: `track_err` 0.37 → 0.37, `goal_dist`
   0.66 → 0.66, `n_success` 0.30 → 0.30 — *identical*, as it must be. Yet **S&C 0.00 → 0.30**
   and violations **325.7 → 139.0**. Same trajectory, different verdict. The old score was
   measuring the arena, not the policy.
4. 🔴 **The dominant term was the arena box, not the obstacle margin.** `geo_bounds` at
   `y = ±1.5` sat *inside* the route the expert demonstrates, so a correct rollout booked a
   bounds violation on essentially every step. Widening to `y = ±2.5` removes a violation that
   was never about an obstacle.
5. 🟡 **`s_curve` is still broken, but the failure has moved and is now legible.** `fm` reaches
   the goal (S up to 0.80) with **S&C ≈ 0** — it flies the route and clips the physical margin.
   `mf` still aborts 80–100 %.
6. ⚠️ **No ranking claim is available on `pillars` or `corridor`.** Neither has a `diffusion`
   arm at `_hg`, so per [`benchmark-hierarchy-who-beats-whom`] the pinned DPCC target does not
   exist for these scenes yet. Everything below is *feasibility*, not a benchmark result.
7. ⚠️ **n = 10 rollouts, 1 seed, 1 K.** 0.70 vs 0.40 is ≈1.4 SE. Directional, not established.

---

## 1. The gate fired, and its numbers were as predicted

U7 added a bisection that reports the expert route's clearance in metres under the *planning*
inflation. Predicted in the changelog vs measured in the logs:

| scene · homotopy | predicted | measured | verdict |
|---|---|---|---|
| `pillars` (L,L,L) / (L,R,L) / (R,L,R) / (R,R,R) | 0.080 m | **0.078 m** | ✅ |
| `corridor` L, R | 0.020 m | **0.020 m** | ✅ |
| `corridor` C | — | **0.137 m** | new |
| `s_curve` default | 0.140 m | **0.121 m** | ✅ |

All four still trip `NEAR-ZERO SLACK` (threshold 0.30 m). **The geometry is honest now, not
generous** — the gate is still telling the truth about a tight benchmark. That it fires while
`corridor` scores 1.00 means the 0.30 m threshold is conservative for scenes whose failure mode
is not lateral clearance; it should not be read as a pass/fail line.

---

## 2. The headline: identical trajectory, opposite verdict

`pillars` · `mf` · K=2 · seed 6 · same checkpoint · `diffuser` = **projection off**:

| tag | geometry | terr | goal_dist | S | **S&C** | **collision_free** | violations |
|---|---|---|---|---|---|---|---|
| `fix16scaled` | old | 0.37 | 0.66 | 0.30 | **0.00** | **0.00** | 325.7 |
| `u7hg` | `_hg` | 0.37 | 0.66 | 0.30 | **0.30** | **0.30** | 139.0 |

With `proj=off` the rollout is a pure open-loop replay of the same weights. `track_err`,
`goal_dist` and `n_success` are byte-identical across the two rows, which is the control: the
policy did not change. Only the set it was scored against did.

**This retires the "engines cannot do `pillars`" reading of the 0/2876 result.** The engines were
producing rollouts that reach the goal; the scorer was rejecting all of them.

---

## 3. `pillars` — the full A/B

`mf`, K=2, seed 6. Old = `fix16scaled` (Fix_16 scaled eps, pre-U7 geometry), new = `u7hg`.

| variant | S&C old → new | collision_free old → new | viol old → new | terr old → new |
|---|---|---|---|---|
| `diffuser` | 0.00 → **0.30** | 0.00 → 0.30 | 325.7 → 139.0 | 0.37 → 0.37 |
| `dpcc-r` | 0.00 → **0.70** | 0.00 → 0.70 | 52.2 → 54.7 | 0.46 → 0.41 |
| `dpcc-c` | 0.00 → **0.40** | 0.00 → 0.40 | 65.6 → 108.8 | 0.45 → 0.37 |
| `dpcc-t` | 0.00 → **0.40** | 0.00 → 0.40 | 136.7 → 103.9 | 0.63 → 0.38 |
| `dpcc-r-tightened` | 0.00 → **0.40** | 0.00 → 0.40 | 41.9 → 107.6 | 0.45 → 0.38 |
| `dpcc-c-tightened` | 0.00 → **0.40** | 0.00 → 0.40 | 69.1 → 107.3 | 0.48 → 0.38 |
| `dpcc-t-tightened` | 0.00 → **0.40** | 0.00 → 0.40 | 114.8 → 105.5 | 0.56 → 0.38 |
| `dpcc-r-geo_free` | 0.00 → 0.20 | 0.00 → 0.30 | 283.0 → 123.8 | 0.35 → 0.35 |
| `dpcc-c-geo_free` | 0.00 → 0.30 | 0.00 → 0.30 | 231.4 → 121.9 | 0.38 → 0.36 |
| `dpcc-t-geo_free` | 0.00 → 0.20 | 0.00 → 0.20 | 191.7 → 124.7 | 0.41 → 0.35 |

**Every cell moves off zero.** Three further observations:

* **`geo_free` is now the *worst* group (0.20–0.30), not the best.** Under the old geometry
  `geo_free` looked like a rescue (§6.1 of the Fix_16 DA: abort 1.00 → 0.49 on `s_curve`)
  because deleting the geometry deleted a *spurious* constraint. With an honest set, keeping
  the geometry helps. That inverts the sign of the `geo_free` diagnostic and removes the
  awkward "it succeeds by flying through walls" caveat from the U7-era numbers.
* **Tracking error improves almost everywhere** (0.45–0.63 → 0.37–0.41). The projector is no
  longer fighting a box that excludes the demonstrated route.
* **Violation counts do not track S&C.** `dpcc-r-tightened` violations *rose* 41.9 → 107.6 while
  S&C rose 0.00 → 0.40. Violations are per-step against the scoring set; S&C is per-rollout
  collision. They answer different questions — do not use violations as a proxy.

### 3.1 Which `pillars` variant leads — carefully

`mf`/`dpcc-r` is the only cell at 0.70. It leads on S&C, `goal_dist` (0.39, lowest) and
violations (54.7, lowest), but its `track_err` is the **highest** of the projected group
(0.41 vs 0.35–0.38). Per [`pareto-definition-of-good`] this is a **trade-off, not a
Pareto-dominant result** — it buys goal attainment with looser tracking. At n=10/1 seed the
gap to the 0.40 cluster is ≈1.4 SE.

### 3.2 `fm` vs `mf` on `pillars`

| engine | best S&C cell | value | `diffuser` (unprojected control) |
|---|---|---|---|
| `mf` (U-Net, `dp0.5_bbunet`) | `dpcc-r` | **0.70** | 0.30 |
| `fm` (U-Net, `FlowMatchingODE_9D`) | `dpcc-r` / `dpcc-c-tightened` | **0.50** | 0.30 |

Both engines are U-Net, so this pair is **architecture-matched** — the comparison the thesis
needs per [`architecture-matched-beat-is-the-strong-claim`]. Both also clear their own
unprojected control (0.30), which is the MF-must-beat-naive-FM bar. 🔴 But with no `diffusion`
arm on `pillars` there is still **no DPCC baseline to beat**, so this is not yet a claim.

---

## 4. `corridor` — solved, and the cleanest evidence U7 worked

| engine | variants at S&C = 1.00 | exceptions | violations | aborts |
|---|---|---|---|---|
| `mf` | **10 / 10** | — | 0.0 | 0.00 |
| `fm` | 7 / 10 | `dpcc-t`, `dpcc-t-tightened`, `dpcc-t-geo_free` at 0.80 | 0.0 | 0.00 |

Zero violations and zero aborts across 200 rollouts. `corridor`'s L and R homotopies previously
had **0.000 m** slack — the expert route did not fit inside its own planning set, so S&C was
bounded at 0 by arithmetic. It now fits by 0.020 m, and both engines saturate the metric.

⚠️ **Saturation means `corridor` cannot rank anything.** With every cell at 1.00 there is no
signal left to separate engines, projectors or K. Its value from here is as a **feasibility
control** — a scene where we can show the pipeline is not broken — not as a benchmark.

The three `fm` `dpcc-t` cells at 0.80 come with the lowest `track_err` in the scene (0.41–0.47)
and the highest `goal_dist` (0.82–0.93): temporal-consistency selection is trading goal
attainment for smoothness here. Worth one look, not a claim at n=10.

---

## 5. `s_curve` — still failing, but the failure is now diagnosable

| engine | S (goal) | **S&C** | collision_free | abort | goal_dist |
|---|---|---|---|---|---|
| `fm` | 0.30 – **0.80** | 0.00 – 0.10 | 0.00 – 0.10 | 0.20 – 0.50 | 0.79 – 1.44 |
| `mf` | 0.00 – 0.30 | **0.00** | 0.00 | **0.60 – 1.00** | 1.68 – 2.96 |

Two *different* failures, which the old geometry had merged into one:

* **`fm` reaches the goal and fails the constraint check.** S = 0.80 with S&C = 0.10 means the
  route is flown and the physical margin is clipped. This is the honest-geometry split working
  as designed: U7 loosened the **planning** set (`planning_inflation`, margin 0) and
  deliberately left `_exec_constraint_violations` on the **physical** `inflation`. `fm` is now
  living in the gap between them.
* **`mf` still diverges.** 60–100 % abort, `goal_dist` up to 2.96 m. Geometry was not its
  problem. This is the `p_des_runaway` failure documented in the Fix_16 DA §6.2, and it
  survives U7 untouched — as §6.1 predicted it would ("unlikely, non-zero, you get it free").

**§6.1's forecast was right, and for the stated reason.** The Fix_16 DA reopened the `s_curve`
question at 🟡 "unlikely but non-zero". The outcome: geometry bought `fm` a large gain in *goal*
(S 0.00 → 0.80) and nothing in *S&C*, and bought `mf` nothing at all. Neither the optimistic
nor the flat reading was correct; the split was.

---

## 6. What can and cannot be said

**Can:**
- The pre-U7 `pillars` 0/2876 was a benchmark artifact. Demonstrated by a projection-off row
  whose trajectory is provably unchanged (§2).
- `corridor` was infeasible by construction (0.000 m slack) and is now feasible and saturated.
- Both `fm` and `mf` clear their own unprojected control on `pillars`, architecture-matched.
- `geo_free` inverted from apparent-rescue to worst-group once the geometry became honest.

**Cannot:**
- 🔴 Any *ranking* on `pillars` or `corridor` — no `diffusion` arm at `_hg`, so no pinned DPCC
  target exists ([`da-target-is-best-baseline-variant`]).
- 🔴 Any comparison of these numbers to pre-U7 numbers **as an engine result**. The geometry
  changed; only the §2 projection-off control is a valid cross-geometry comparison.
- 🔴 Anything about HardFlow. At K=2, A=0.5 the arm is degenerate (`n_genuine=0`) and the guard
  correctly dropped all 7 HF variants — the runs carry `HF_DEGENERATE_SKIPPED.txt` and report
  10 variants, not 17. Per [`hardflow-low-K-degeneracy`] no claim may cite these.
- 🟡 Anything from `avg_time` / `budget_ms`. Per [`uav-budget-ms-not-a-goal`] the 33 Hz line is
  a data-rate artefact; the `real_time_OVER×` figures in these logs are not a finding.

---

## 7. Next

| # | action | why |
|---|---|---|
| 1 | **`diffusion` arm on `pillars` + `corridor` at `_hg`** | the only thing standing between §3/§4 and an actual benchmark claim |
| 2 | **Seeds.** 3+ seeds at K=2 on `pillars` | 0.70 vs 0.40 at n=10/1 seed is ≈1.4 SE |
| 3 | **K sweep at `_hg`** | K=2 only; the Gen15 primary axis is untested under honest geometry |
| 4 | `s_curve`: close the planning/scoring gap for `fm` | S=0.80 → S&C=0.10 is a margin-bookkeeping question, possibly cheap |
| 5 | `s_curve`: `mf` `p_des_runaway` | untouched by U7; see Fix_16 DA §6.2.3 (silent SLSQP non-convergence is still the prime suspect) |
| 6 | Re-check the 0.30 m gate threshold | it fires on `corridor` at 0.020 m while `corridor` scores 1.00 — the threshold is not calibrated for non-lateral failure modes |

**In flight:** `af` `s_curve` arm A (train 25435 → evals 25436/37/38) and arm B (train 25440 →
evals 25441 ✅ K=1, 25442 ✅ K=2, 25443 running K=5). Arm B trained the real
`_as1_ae0.2_bbunet` tree — the first genuine α-Flow `s_curve` arm. Not analysed here.

---

## 8. `af` + U-Net on `s_curve` under `_hg` — a peek, not a result

Runs **25441 (K=1) / 25442 (K=2) / 25443 (K=5, still running)**, from train **25440**
(pipeline 25439). Submitted this session as the U6 arm-B resubmission; they landed on `963faed`,
so they picked up `_hg` geometry.

### 8.1 Provenance — every U6 gate is green, and it is parameter-matched

| gate | expected | measured | |
|---|---|---|---|
| `af_alpha_end` | > 0 | **0.2** | ✅ |
| `alpha(step 100000)` | `0.2000 ACTIVE` | **0.2000** | ✅ **not** snapped to 0 ⇒ genuine α-Flow, not MeanFlow |
| `discrete_frac` | > 0 | 0.50 / 0.25 (early epochs) | ✅ |
| checkpoint | `latest` | **`state_100000.pt`** | ✅ not `best` |
| backbone | `unet` | **`unet`, 4.0 M** | ✅ parameter-matched to `fm`/`mf` |
| geometry | `_hg` | `s_curve_hg_bounds+dynamics+geo_bounds+halfspace+obstacles` | ✅ |

**This is the first genuine, architecture-matched α-Flow arm on UAV.** Per
[`architecture-matched-beat-is-the-strong-claim`] it is the arm a thesis claim would have to rest
on — the ~9.4 M SiT arm is confounded and cannot carry one.

🔴 **Naming defect (mine).** The command re-issued the U6 tag `u6unet_ae02` unchanged, so these
runs are stamped `Eaf_K*_..._EPlatest_u6unet_ae02` while running **U7 geometry**. The tag says
U6, the run is U7. Same lying-tag class as `u6unet_ae0` in
[`Gen15/U6/RUNSTATUS_20260904_...`](../U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md) §4.
Disambiguate by `geo` (`s_curve_hg_…`), never by `run_tag`. Retag or re-run before publishing.

🟡 **K=5 is incomplete.** 25443 was still running at capture: 560 aggregated rows vs 840 for a
finished K, `dpcc-t-tightened` missing `collision_free` and `divergence_aborted`, and no
`geo_free` variants at all. **Nothing in §8.3's K=5 column is citable.**

### 8.2 Matched-K comparison, K=2, `s_curve`, `_hg`, seed 6, n=10

All three arms are U-Net at ~4.0 M, so this row is architecture-matched.

| engine | **S&C** | median S | best S (cell) | mean abort | best `goal_dist` |
|---|---|---|---|---|---|
| `fm` | 0.00–0.10 | **0.70** | 0.80 (`dpcc-t-geo_free`) | **0.34** | 0.79 |
| **`af`** | **0.00** | 0.20 | 0.70 (`dpcc-t`) | 0.68 | 0.82 |
| `mf` | 0.00 | 0.00 | 0.30 (`dpcc-t-tightened`) | 0.89 | 1.68 |

**`af` sits between `fm` and `mf`, closer to `mf`.** Its single best cell (0.70) is competitive
with `fm`'s, but its *median* cell is 0.20 against `fm`'s 0.70 — the arm is far less robust to
projector choice. 🔴 **S&C is 0.00 for `af` in all 10 variants at K=2**, so on the metric that
matters it is level with `mf` and behind `fm` (which has two cells at 0.10). Bringing af-unet
into UAV cannot be justified on this evidence.

### 8.3 `af` gets *worse* with more NFE — the one genuinely interesting signal

| K | best S | mean abort | note |
|---|---|---|---|
| 1 | **0.80** (`dpcc-t-geo_free`) | **0.50** | |
| 2 | 0.70 (`dpcc-t`) | 0.68 | |
| 5 | 0.20 (`diffuser`, `dpcc-t`) | 0.82 | 🟡 incomplete — not citable |

Monotone degradation in both columns as the NFE budget rises. That is backwards: K is the
compute axis, and more integration steps should not cost goal attainment. Two readings, not
separable here:

1. the α-Flow velocity field is inconsistent across `t`, so integrating it more finely
   accumulates error rather than reducing it; or
2. the failure is downstream — longer plans give the projector more opportunity to hit the
   silent SLSQP non-convergence path (Fix_16 DA §6.2.3), which is already the prime suspect for
   `s_curve` `p_des_runaway`.

Reading 2 predicts `fm`/`mf` show the same slope; reading 1 predicts they do not. **`fm`/`mf`
were only run at K=2 under `_hg`, so this is not yet testable** — a K sweep at `_hg` (§7 item 3)
separates them for free.

### 8.4 `dpcc-t` is the only selector that works for `af`

At K=1 and K=2, temporal-consistency selection is `af`'s best cell by a wide margin
(`dpcc-t-geo_free` 0.80 / abort 0.10 at K=1; `dpcc-t` 0.70 / abort 0.20 at K=2), while `dpcc-r`
and `dpcc-c` collapse to 0.00–0.30 with abort 0.70–0.90. `fm` shows the same preference more
weakly (`dpcc-t-geo_free` its best cell too). Consistent with an engine whose per-step output is
noisy across `t` — temporal smoothing is doing the work the velocity field should. Suggestive at
n=10, and it is the cheapest follow-up if af-unet is pursued.

### 8.5 Verdict for the "bring af-unet into UAV?" question

🔴 **Not on this evidence.** S&C = 0.00 in every `af` cell at K=1 and K=2; it is behind
architecture-matched `fm` on median success, abort rate and goal distance, and its K-scaling runs
the wrong way. The two things that would change the picture are (a) a scene where af-unet is not
being asked to solve `s_curve`, which is the scene U7 helped least, and (b) the K sweep in §8.3.
`pillars` under `_hg` — where `fm`/`mf` now score 0.50/0.70 — is the obvious test and has never
been run with af-unet at `_hg`.

⚠️ Also unresolved: **arm A** (`_ae0_bbunet`, α off — the control that separates *backbone* from
*objective*) is jobs 25435→25436/37/38, still queued at capture. Without it, α-on vs α-off is
confounded with nothing, but the U-Net-vs-SiT question stays open.

### 8.6 Defect noted in passing

Train log 25440 contains **tqdm progress bars** (`Epoch 0: 100%|██████████| 1000/1000 …`),
repeated per epoch. Batch logs must not carry live bars — they bloat the file and make `grep`
unreliable. `train_mix_uav.sh` / the af trainer should set `disable=True` (or `TQDM_DISABLE=1`)
when not attached to a TTY.
