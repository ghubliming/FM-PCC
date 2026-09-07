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

---

## 9. Three head-to-heads, asked directly

All at matched K=2, `_hg`, seed 6, n=10, architecture-matched U-Net (~4.0 M) throughout.

### 9.1 Does `mf > fm` hold in the **raw** output?

**No. It never holds unprojected.** `diffuser` = `proj=off`, so this is the engines' own output:

| scene | fm S&C | mf S&C | fm S | mf S | fm abort | mf abort | |
|---|---|---|---|---|---|---|---|
| `pillars` | 0.30 | 0.30 | 0.30 | 0.30 | 0.00 | 0.00 | tie |
| `corridor` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | tie |
| `s_curve` | 0.00 | 0.00 | **0.70** | **0.00** | **0.30** | **0.90** | 🔴 **fm** |

Two ties and one heavy loss. On `s_curve` the raw `mf` output reaches the goal in **0 of 10**
rollouts and diverges in 9.

### 9.2 Does it hold **under projection**?

**Yes — but only on the two scenes the geometry fix repaired.** Per-variant tally over the
9 projected variants + `diffuser`:

| scene | mf wins | fm wins | ties | verdict |
|---|---|---|---|---|
| `pillars` | **4** | 2 | 4 | mf ahead |
| `corridor` | **3** | 0 | 7 | mf ahead (fm loses only the 3 `dpcc-t` cells) |
| `s_curve` | 0 | **10** | 0 | 🔴 fm sweeps |

### 9.3 🔴 The finding: `mf`'s advantage is projection-mediated

`mf` is never better than `fm` on its own output, and better than `fm` on 7 of 20 projected cells
across `pillars`+`corridor`. **The projector is what converts `mf`, not the engine.** That is
exactly what the Fix_16 degenerate-action-channel story predicts: `mf`'s raw trajectory carries a
defect (§1.2 of [`DA_20260903_fix16_AB_mf_pillars.md`](DA_20260903_fix16_AB_mf_pillars.md) — the
b₁ gain is +0.1121/+0.0804/+0.0310 for `mf` against ~0 for `fm`/`af`), and the DPCC projection
repairs it.

Consequences for the claim ladder:
- ⚠️ A "MeanFlow beats Flow Matching" claim **cannot be made on generative quality**. The honest
  statement is *"`mf` + projection beats `fm` + projection on `pillars`/`corridor`; unprojected,
  `mf` is not better and on `s_curve` is far worse."*
- The margin is 4–2 and 3–0 at n=10 with **one seed**. Not established — §7 item 2.
- On `pillars`/`corridor` there are **zero aborts**, so S&C ≡ goal-reaching there; the whole
  comparison rests on one axis.

### 9.4 Is HardFlow better than the DPCC projector?

🔴 **No data. The question cannot be answered from these runs at all.**

Every run reports **10 variants, not 17** — at K=2 with A=0.5 the HardFlow arm is degenerate
(`n_genuine=0`: every NLP solve is the terminal τ=1 solve, making the arm
Π_S(Euler sample) = sample-then-project ≡ DPCC), and the guard correctly dropped all 7 HF
variants, writing `HF_DEGENERATE_SKIPPED.txt`. Per [`hardflow-low-K-degeneracy`] no claim may
cite a degenerate row, and none exists here to cite.

**To answer it:** K ≥ 3 at A=0.5 gives `n_genuine ≥ 1`; **K ≥ 5** is needed for an *attributable*
effect. The `_hg` K sweep (§7 item 3) delivers this for free if it includes K=5.

### 9.5 Is `af > mf`?  (§8 follow-up)

**On goal-reaching yes, on the metric that matters no — they are tied at zero.** `s_curve`, K=2:

| axis | af | mf | winner |
|---|---|---|---|
| **S&C** | **0.00 in all 10 variants** | **0.00 in all 10** | 🔴 **tie at zero** |
| S, per-variant tally | **8 wins** | 0 wins | af (2 ties) |
| S, median cell | **0.20** | 0.00 | af |
| abort, mean | **0.68** | 0.89 | af |

So `af` beats `mf` on the axis that does not produce a claim and ties it on the axis that does.

🔴 **And `af` does not clear the naive-FM bar.** Per
[`benchmark-hierarchy-who-beats-whom`], MF/AF must beat plain FM. On `s_curve`, `fm` has median
S = 0.70 and two cells at S&C = 0.10; `af` has median S = 0.20 and S&C = 0.00 throughout. `af`
loses to `fm` on the same scene, at the same K, with the same backbone and parameter count.

**Net ordering on `s_curve` @ K=2:  `fm`  >  `af`  >  `mf`.**  This is the reverse of the
`pillars`/`corridor` ordering in §9.2, where `mf` leads — a further sign that `s_curve` is
measuring a different failure (divergence, §5) than the other two scenes (goal-reaching).

---

## 10. Pareto analysis — is any projection variant actually superior?

Per [`pareto-definition-of-good`], "better" means: **at equal or higher S&C, fewer `n_steps` AND
lower `avg_time`**. Anything else is a trade-off. Three axes, K=2, `_hg`, seed 6, n=10.
`avg_time` is seconds per control step; `n_steps` is control steps per episode (lower = reaches
the goal sooner, or gets terminated later — read with S&C).

### 10.1 🟢 `pillars` / `mf` — **yes, one point, and it dominates outright**

| variant | S&C | n_steps | avg_time (s) | proj_ms | track_err | front |
|---|---|---|---|---|---|---|
| **`dpcc-r`** | **0.70** | **491.3** | **0.0603** | **42.0** | 0.413 | ★ |
| `dpcc-t-tightened` | 0.40 | 550.3 | 0.0804 | 62.0 | 0.376 | |
| `dpcc-t` | 0.40 | 550.8 | 0.0798 | 61.5 | 0.375 | |
| `dpcc-c-tightened` | 0.40 | 552.8 | 0.0802 | 61.8 | 0.376 | |
| `dpcc-r-tightened` | 0.40 | 552.9 | 0.0821 | 63.8 | 0.375 | |
| `dpcc-c` | 0.40 | 554.6 | 0.0806 | 62.2 | 0.374 | |
| `diffuser` (proj=off) | 0.30 | 570.0 | 0.0182 | 0.0 | 0.371 | ★ |
| `dpcc-c-geo_free` | 0.30 | 578.0 | 0.0296 | 11.3 | 0.358 | |
| `dpcc-t-geo_free` | 0.20 | 591.9 | 0.0297 | 11.3 | 0.350 | |
| `dpcc-r-geo_free` | 0.20 | 593.2 | 0.0297 | 11.3 | 0.351 | |

**`dpcc-r` Pareto-dominates all eight other projected variants simultaneously** — highest S&C,
fewest steps, *and* lowest `avg_time` of any variant that runs the projector. Against the
best of the rest (`dpcc-t-tightened`): **+0.30 S&C, −59 steps, −25 % `avg_time`**.

The time result is the surprising half. `dpcc-r`'s projector cost is **42.0 ms vs 61.5–63.8 ms**
for every other projected variant — it is *cheaper*, not bought with compute. The natural reading
is that random selection lands on plans the SLSQP solve converges on quickly, while the
cost/temporal selectors hand it harder starts. That is a testable prediction: `dpcc-r` should show
the lowest `nlp_failures` / non-convergence rate. **Not logged today** — see §7 and the Fix_16 DA
§6.2.3 (`last_solve_success` was already on the wish-list).

⚠️ The front is `{dpcc-r, diffuser}`, not `{dpcc-r}` alone. Unprojected `diffuser` survives only
on speed (0.0182 s, 3.3× faster) at 0.30 S&C. `dpcc-r` does **not** dominate it, and it does not
dominate `dpcc-r`. **Buying S&C 0.30 → 0.70 costs 3.3× per-step time.**

### 10.2 🟢 Cross-engine: `mf` Pareto-dominates `fm` on the same projector

Architecture-matched (both U-Net ~4.0 M), same variant, same K, `pillars`:

| variant | S&C fm → mf | n_steps fm → mf | avg_time fm → mf | verdict |
|---|---|---|---|---|
| **`dpcc-r`** | 0.50 → **0.70** | 538.3 → **491.3** | 0.0784 → **0.0603** | 🟢 **mf dominates** |
| `dpcc-t-tightened` | 0.20 → 0.40 | 594.1 → 550.3 | 0.1043 → 0.0804 | 🟢 mf dominates |
| `dpcc-t` | 0.30 → 0.40 | 572.3 → 550.8 | 0.0910 → 0.0798 | 🟢 mf dominates |
| `dpcc-c` | 0.20 → 0.40 | 595.0 → 554.6 | 0.0953 → 0.0806 | 🟢 mf dominates |
| `dpcc-r-tightened` | 0.40 → 0.40 | 560.0 → 552.9 | 0.0874 → 0.0821 | 🟢 mf dominates |
| `dpcc-c-tightened` | 0.50 → 0.40 | 540.4 → 552.8 | 0.0789 → 0.0802 | 🔴 fm dominates |
| `dpcc-r-geo_free` | 0.30 → 0.20 | 575.7 → 593.2 | 0.0288 → 0.0297 | 🔴 fm dominates |
| `diffuser`, `dpcc-c-geo_free`, `dpcc-t-geo_free` | — | — | — | neither |

**5 of 10 cells are strict `mf` dominance on all three axes, 2 are strict `fm`, 3 neither.**
Every `mf` win is a projected variant; both `fm` wins are `-tightened` / `geo_free` edge cases.
This is the strongest form the §9 result takes: not just "higher S&C" but *higher S&C at lower
cost on both axes at once*.

### 10.3 🟡 `corridor` — S&C saturates, so it is a pure trade-off curve

All 20 cells at S&C 1.00 except `fm`'s three `dpcc-t` variants (0.80). The gate is
non-discriminating, exactly as in
[`Report_20260819_MF_UNet`](../../../Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md)
§1, so the result reduces to (`n_steps`, `avg_time`). `mf` front:

| variant | S&C | n_steps | avg_time | note |
|---|---|---|---|---|
| `dpcc-t-tightened` | 1.00 | **248.5** | 0.1688 | fewest steps, 9.2× the time |
| `dpcc-r` | 1.00 | 261.1 | 0.0649 | |
| `dpcc-t-geo_free` | 1.00 | 262.3 | 0.0294 | |
| `dpcc-r-geo_free` | 1.00 | 266.5 | 0.0293 | |
| `diffuser` | 1.00 | 271.7 | **0.0183** | fastest, 23.2 more steps |

**No dominance — 5 non-dominated points.** Spending 9.2× per-step time buys 23.2 fewer steps
(8.5 %). Whether that trade is worth taking is a deployment question, not a result.

### 10.4 🔴 `s_curve` — no Pareto statement is possible

S&C is 0.00 or 0.10 in all 20 cells. With the gate at the floor, `n_steps` and `avg_time` describe
*how* runs fail, not how well they succeed. `mf`'s four worst variants sit at `n_steps` = 871.0 —
the full episode budget — i.e. they never terminate. Nothing here is rankable.

### 10.5 What this licenses

**Can say:** on `pillars` at K=2, `dpcc-r` is the sole Pareto-superior projector among the nine
projected variants, and `mf`+`dpcc-r` Pareto-dominates architecture-matched `fm`+`dpcc-r` on
S&C, steps and time simultaneously.

**Cannot say:** that this beats the baseline. 🔴 There is still **no `diffusion` arm on `pillars`
at `_hg`**, so per [`da-target-is-best-baseline-variant`] the pinned DPCC K20/aw10 target does not
exist for this scene and no benchmark claim follows. At n=10 with **one seed**, 0.70 vs 0.40 is
≈1.4 SE — the ordering is consistent across 5 of 10 variants, which is what makes it worth
reporting, but seeds are what would make it a result.

---

## 11. Why `s_curve` still fails — three causes, only one of them "hard"

Rollout-level data, `s_curve` @ `_hg`, n = 461 across `fm`/`mf`/`af`.

### 11.1 Cause A — the channel is genuinely narrow (this one *is* hard)

From `config/uav_projection.yaml` `s_curve_hg`, the halfspace walls are **unchanged by U7** — the
entry's own comment says `PHYSICAL inner faces`. Segment 1 confines `y ∈ [-1.25, -0.35]`,
segment 2 `y ∈ [0.35, 1.25]`:

| quantity | value |
|---|---|
| physical wall gap | **0.90 m** |
| drone diameter (2 × `r_drone` 0.31) | **0.62 m** |
| free centre-line half-width, planning | **0.140 m** |
| measured `fm` `track_err_mean` | **0.378 m** |
| ratio | **2.7×** |

🔴 **U7 could not fix this and should not have.** On `pillars` the binding constraint was a
*spurious* arena box (`y = ±1.5` sat inside the demonstrated route), so widening it to ±2.5 was
free. On `corridor`, adding `x_active` stopped the halfspaces binding outside the corridor
section — also free. On `s_curve` the binding constraint is a **real 0.90 m gap between real
walls**. Widening it would be lying about the scene. The `_hg` box change
(`x ±3.6→±4.0`, `y ±1.6→±1.8`) bought nothing here because the box was never what was binding.

⚠️ The 2.7× is an episode-mean tracking error against a *local* clearance. The halfspaces bind
only on `x ∈ [-3,-0.5] ∪ [0.5,3]`, so in-channel tracking must be better than 0.378 or nothing
would ever get through — 2 rollouts do, with `n_violations = 0`. Read 2.7× as the same
order-of-magnitude statement §II.3 of the Fix_16 DA made, not as a per-step figure.

### 11.2 Cause B — divergence is a *deterministic scene event* at the crossover

Abort position as a fraction of the 871-step budget:

| arm | n aborts | mean | median | IQR |
|---|---|---|---|---|
| `fm` K=2 | 34 | 54.5 % | 49.9 % | 47.6 – 59.0 % |
| `mf` K=2 | 89 | 46.9 % | **45.9 %** | **45.5 – 46.5 %** |
| `af` K=1 | 50 | 52.5 % | 46.6 % | 44.8 – 60.2 % |
| `af` K=2 | 68 | 47.6 % | 45.6 % | 45.4 – 46.5 % |

🔴 **Every engine dies at the same place — the midpoint.** `mf`'s interquartile range spans
**one percentage point across 89 rollouts**. A stochastic policy failure does not do that. This
is the scene, not the engine.

The midpoint is the **crossover**, and the mechanism is visible in the config. `x_active` gates
the halfspaces: segment 1 is live on `x ∈ [-3.0, -0.5]`, segment 2 on `x ∈ [0.5, 3.0]`. On
`x ∈ (-0.5, 0.5)` **no halfspace is active at all** — the only lateral geometry is two
`radius = 0.05` corner spheres. So at exactly the point where the drone must swing `y` from
−0.80 to +0.80, the projector's lateral walls vanish, and then segment 2's snap on.

**Testable prediction:** aborts should be `p_des_runaway` (they were 55–90 of s_curve aborts
pre-U7, vs **0** on `pillars`), and the runaway should start inside `|x| < 0.5`. Logging the
abort `x` would settle it in one run. This is a *constraint-switching* bug, not a capability
limit, and nothing in U7 touched it.

### 11.3 Cause C — the scoring set is 0.02 m tighter than the planning set (my U7 choice)

U7 deliberately split the two:

```yaml
inflation:          {r_drone: 0.31, margin_base: 0.02}   # SCORING  -> 0.33 m
planning_inflation: {r_drone: 0.31, margin_base: 0.00}   # PLANNING -> 0.31 m   (_hg entries)
```

| | half-width in the 0.90 m channel |
|---|---|
| what the projector plans to clear | (0.90 − 2×0.31)/2 = **0.140 m** |
| what the scorer requires | (0.90 − 2×0.33)/2 = **0.120 m** |
| band the projector may enter but the scorer rejects | **0.020 m** |

On `s_curve` that 2 cm is **14 % of the entire clearance budget**. And it is where the runs die:

| `fm` K=2, goal-reaching rollouts | n | `track_err` | `goal_dist` | `phys_safe` | `n_violations` |
|---|---|---|---|---|---|
| collision-**free** | **2** | 0.363 | 0.296 | 1.000 | **0.0** |
| collision-**violating** | **61** | 0.363 | 0.295 | **0.984** | **23.1** (6–89) |

🔴 **The 61 rejected rollouts are indistinguishable from the 2 accepted ones on every physical
axis** — identical tracking error, identical goal distance, and MuJoCo calls 98.4 % of them
*safe*. The only thing separating them is the analytic violation counter.

Scene-level, the two safety verdicts diverge exactly where clearance is tight:

| scene | n | `phys_safe` | `constraint_collision_free` | gap |
|---|---|---|---|---|
| `corridor` | 200 | 100 % | **100 %** | **0** |
| `pillars` | 200 | 100 % | 35 % | 130 |
| `s_curve` | 461 | 31 % | **1 %** | 141 |

### 11.4 So: is `s_curve` too hard?

**Partly. One of the three causes is real; two are not.**

| cause | real difficulty? | fixable |
|---|---|---|
| A — 0.90 m gap vs 0.62 m drone | 🔴 **yes, physically** | no — would require lying about the walls |
| B — no active halfspace on `|x| < 0.5` | 🟢 no, a constraint-switching gap | yes — overlap the `x_active` windows |
| C — 0.02 m scoring/planning offset | 🟢 no, a bookkeeping choice | yes — see below |

**Recommended order:**

1. **B first — it is free and it is the abort cause.** Overlap the `x_active` windows
   (e.g. seg 1 → `[-3.0, +0.5]`, seg 2 → `[-0.5, +3.0]`) so lateral geometry is continuous across
   the crossover. Predicted effect: the 45–47 % abort cluster disappears. ⚠️ Check it does not
   make the crossover infeasible — the two channels are on opposite sides of `y = 0`, so an
   overlap region binds both walls at once and may have empty interior. If so, the correct fix is
   a smooth switch, not an overlap.
2. **C second, and state it explicitly rather than tuning it away.** Either score at the same
   0.31 m the projector plans to (then S&C measures control, not the 2 cm band), or keep the
   0.02 m and report `phys_safe` alongside S&C so the disagreement is visible. 🔴 The one thing
   not to do is quietly lower the scoring margin to make the number go up — that is the mistake
   U7 was written to stop.
3. **A last, and honestly.** With 0.14 m of clearance and a policy that tracks at ~0.38 m
   episode-mean, `s_curve` may simply not be winnable by this policy class at this margin. That is
   a legitimate finding, and after B and C it can be stated cleanly instead of being confounded
   with a switching bug and a bookkeeping offset.

---

## 12. Do we need higher K? And does HardFlow earn a slot?

Answered from the **pre-U7 K sweeps already in the batch** (K = 1, 2, 5, 10, 20 for `fm`/`mf` on
all three scenes) — no new runs needed to decide. 🔴 All HardFlow rows with
`hf_degenerate = 1` / `n_genuine = 0` are excluded throughout, per
[`hardflow-low-K-degeneracy`]; 14 of the 29 HF cells in the batch are degenerate and were being
silently included in "best variant" reads.

### 12.1 🔴 `fm`'s K requirement was a **geometry artifact**

Best S&C by K, `corridor`, degenerate HF excluded:

| engine | K=1 | K=2 | K=5 | K=10 | K=20 | **U7 `_hg` @ K=2** |
|---|---|---|---|---|---|---|
| `fm` | 0.00 | 0.10 | 0.90 | 1.00 | 1.00 | 🟢 **1.00** |
| `mf` | 0.80 | 0.80 | 0.70 | 0.80 | 0.90 | 🟢 **1.00** |

Pre-U7, `fm` needed **K ≥ 5** to become usable and **K = 10** to saturate. Under honest geometry
it saturates at **K = 2**. The K-ladder was measuring the constraint set, not the sampler.

This matters for the thesis. A "MeanFlow works at low K where Flow Matching does not" claim reads
naturally off the pre-U7 ladder (`mf` 0.80 at K=1 vs `fm` 0.00) — and **that gap closes to zero
once the geometry is honest**. `mf`'s low-K advantage on `corridor` is not reproducible post-U7,
because there is nothing left to be better at.

### 12.2 Per scene: is more K worth buying?

| scene | pre-U7 K-slope | post-U7 @ K=2 | more K? |
|---|---|---|---|
| `corridor` | `fm` 0.00→1.00 over K1–10; `mf` flat ~0.80 | **1.00 / 1.00** | 🔴 **no** — saturated, nothing above 1.00 |
| `s_curve` | flat **~0.00 at every K, 1→20, both engines** | 0.00–0.10 | 🔴 **no** — K was never the limit; §11 causes B and C first |
| `pillars` | **0.00 at every K, every engine, incl. genuine HF at K=5/10** | 0.50 `fm` / 0.70 `mf` | 🟢 **yes — the only scene with headroom** |

`pillars` is the one place a K sweep can teach anything: it is unsaturated (0.50/0.70), and it has
**zero usable K information** because every pre-U7 cell was pinned at 0 by the spurious arena box.

### 12.3 HardFlow — the genuine rows, and why they are suspect anyway

Genuine (`n_genuine ≥ 1`, K ≥ 5) HF vs the best non-HF variant at the same cell:

| scene | engine | K | best non-HF | genuine HF | Δ |
|---|---|---|---|---|---|
| `corridor` | `fm` | 5 | 0.90 `dpcc-r` | **1.00** `hardflow_new` | **+0.10** |
| `corridor` | `fm` | 10 | 1.00 `dpcc-r` | 1.00 `hardflow_new` | 0 |
| `corridor` | `fm` | 20 | 1.00 `dpcc-r` | 1.00 `hardflow_new` | 0 |
| `corridor` | `mf` | 5 | 0.70 `bounds_free` | **0.80** `hardflow_new-c` | **+0.10** |
| `corridor` | `mf` | 10 | 0.80 `dpcc-r` | **1.00** `hardflow_new` | **+0.20** |
| `corridor` | `af` | 5 | 1.00 `dpcc-r` | 1.00 `hardflow_new-c` | 0 |
| `pillars` | all | 5, 10 | 0.00 | 0.00 | tie at zero |
| `s_curve` | all | 5, 10 | 0.00 | 0.00 | tie at zero |

HF wins 3 of 8 comparable cells by 0.10–0.20 and never loses. That looks like a case for it.

🔴 **It is probably the `geo_free` illusion again.** Every HF win is on `corridor` **pre-U7**,
where the constraint set was over-tight — and HardFlow's mechanism is precisely a *relaxed*
projection (lower threshold, τ-staged solves). On a set that wrongly excludes the demonstrated
route, relaxing the projector wins for the same reason deleting the geometry did. §3 of this
report already recorded that inversion: `geo_free` went from apparent rescue to **worst group**
the moment the geometry became honest.

**And post-U7 `corridor` is 1.00 at K=2 with a plain DPCC projector.** The entire regime in which
HF demonstrated an advantage no longer exists.

### 12.4 Recommendation — one run settles both questions

**`pillars` K sweep, `_hg`, including K=5.** It is the only unsaturated scene, and K=5 is exactly
the threshold at which HF becomes genuine (`n_genuine = 2`). So one job answers:

1. does `pillars` have K headroom above 0.50/0.70?
2. does af's backwards K-scaling (§8.3) hold on a scene that works?
3. does genuine HF beat the DPCC projector **under honest geometry** — the first such test?

⚠️ **Walltime risk.** Both `pillars` K=5 jobs in the Fix_16 A/B died at the 24 h wall, and K=5
runs 17 variants (HF re-enabled) rather than 10. Mitigations, in order of preference: pass
`UAV_EVAL_HOURS=24` explicitly; keep `n_trials = 10`; and note post-U7 `pillars` has **0 aborts**
and shorter episodes (`n_steps` 491 vs 570–595), so it should be materially cheaper than the runs
that died. There is **no variant-subset knob** in `eval_mix_uav.py` — if the wall is hit again,
adding one is the fix, not trimming K.

**Skip HF on `corridor` and `s_curve`.** Saturated and broken respectively; neither can produce a
usable HF comparison.

### 12.5 The strategic point

If the thesis claim is the
[`Report_20260819_MF_UNet`](../../../Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md)
one — *one-step sampling at baseline constraint quality* — then **low K is the product**, and
HardFlow structurally cannot support it: HF needs K ≥ 3 to run any HardFlow arithmetic at all and
K ≥ 5 for an attributable effect. HF is a claim about the **projector**, not the sampler
([`benchmark-hierarchy-who-beats-whom`]: HardFlow must beat the DPCC projector via a lower
projection threshold). Those are separate ladders and should stay separate. If GPU time is
scarce, the sampler ladder is the one carrying the thesis.
