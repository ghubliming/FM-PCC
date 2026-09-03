# DA — Fix_16 A/B: degenerate action channel, `mf` on UAV `pillars`

**Date:** 2026-09-03
**Source:** `temp/0309/2026-09-02/` (6 eval jobs 25316–25321 + 2 pipeline logs 25294/25295),
aggregated in `temp/0309/batch_uav_20260903_204120/`
**Git rev:** `def8fdf` — **all six jobs, one rev.** No code drift inside the A/B.
**Scope:** `pillars`, seed 6, engine `mf`, K ∈ {1, 2, 5}, two arms:
`FMPCC_SAFE_EPS_MODE=scaled` (Fix_16) vs `=legacy` (eps = 1.0, pre-fix behaviour).
**Method:** per-rollout CSV (`per_rollout_detail.csv`, 2876 pillars rollouts) plus the raw
Slurm logs. No cluster access, no re-run.
**Fix under test:** `logs_in_develop/Gen15/fix_16/CHANGELOG_fix16_degenerate_action_channel.md`
**Mechanism it was built from:** `logs_in_develop/Gen15/Study/STUDY_20260901_mf_unguided_failure_uav_pillars.md`

---

## 0. TL;DR

1. 🟢 **The falsifiable prediction passed.** `mf` unguided (`variant=diffuser`) on `pillars`
   went from **100 % divergence abort at every K** to **0 % at every K**. Goal distance
   6.50 / 6.46 / 6.49 m → **0.62 / 0.66 / 0.36 m**. The study's mechanism is confirmed.
2. 🟢 **The `legacy` arm reproduces the pre-fix numbers bit-for-bit** — every metric of
   candidates 52/55/58 equals candidates 51/54/57 to the printed decimal. So the A/B isolates
   exactly one variable (`eps`), and `legacy` is a byte-exact rollback switch.
3. 🟢 **The projected variants improved too**, without exception in direction: aborts fall on
   9 of 10 non-`geo_free` variants; e.g. K5 `dpcc-r` 100 % → 20 % abort, K1 `dpcc-t` 80 % → 0 %.
4. 🟢 **The `*-geo_free` arms — which were already healthy pre-fix — did not move**
   (K1 abort 0 % → 0 %, track err 0.329 → 0.325). The fix moves only what was broken. This is
   the internal control the study asked for.
5. 🔴 **Nothing here makes `pillars` solvable.** Success+constraints is **0 / 2876** rollouts,
   every engine, every K, both arms — unchanged from the 2026-08-30 batch. Collision-free
   completion: **4 / 2876**. No engine ranking can be drawn from this scene.
6. 🔴 **Both K5 jobs died at the wall clock** (25321 scaled, 25318 legacy). K5 `scaled` has
   5 of 17 variants. K1 and K2 are complete on both arms; **K5 is a partial result.**
7. 🟡 **The fix is not free.** Per-step projection time rises **1.2–1.6×** (K1 `dpcc-c`
   76 → 120 ms; K5 1425 → 1751 ms). Squeezing a decision variable to a ±3.1e-05 box turns it
   into a near-equality constraint and SLSQP works harder for it.
8. 🔴 **`fm` and `af` were never re-run with the fix.** Every cross-engine comparison below is
   fixed-`mf` vs **unfixed** `fm`/`af`. It is not an engine comparison.
9. 🟡 **The DA tooling mis-parses the tagged folder names** — `K` comes back empty and all six
   tagged runs collapse to one `FolderName`, so they are dropped from `candidates_detailed.csv`
   and mis-binned in `uav_k_sweep.csv`. Per-rollout data is intact; the aggregates are not.

---

## 1. What this test was for

Short recap, because the chain is three documents long.

`mf` on `pillars` failed *only* in the unguided `diffuser` variant, and failed totally — 30/30
divergence aborts — while `fm` and `af` flew the same scene with the same sampler family. The
study traced it to a four-link chain:

| link | what | shared with fm/af? |
|---|---|---|
| 1 | expert data has `Δz ≡ 0` exactly → **no vertical training signal at all** | shared |
| 2 | `SafeLimitsNormalizer(eps=1)` widens that constant channel by ±1, making `unnormalize` the identity: a saturated output commands **±1 m of altitude per step**, and `action_bounds:'auto'` hands the projector a ±1 m z-cap | shared |
| 3 | **`mf` learned a positive vertical feedback gain** (∂Δz_cmd/∂e_z ≈ **+0.11 /m**); `fm` and `af` sit at ≈ 0 | 🔴 **the differentiator** |
| 4 | `pid_stopgo` chases `p_des`, closing the loop | shared |

Link 3 is a property of the trained model and is not fixed here. **Fix_16 removes link 2** —
the amplifier — by deriving the widening from the data (`1e-3 ×` median half-width of the
non-degenerate dims) instead of hard-coding 1.0.

So this run answers exactly one question: *with the amplifier gone, does `mf` fly?*

---

## 2. Is the data OK?

**Usable, with three defects to work around.** Details:

### 2.1 🟢 Clean A/B design

All six jobs ran on git rev `def8fdf`, same node (`i6-gpu-1`), same seed, same command line
apart from `--flow-steps` and the two env vars. Unlike the 2026-08-30 pillars batch there is
**no solver swap and no rev drift** inside this comparison.

### 2.2 🟢 The `legacy` arm is a byte-exact rollback

| metric, `diffuser` | cand 51 (pre-fix, untagged) | cand 52 (`fix16legacy`) |
|---|---|---|
| abort % | 100.0 | 100.0 |
| goal_dist (m) | 6.50 | 6.50 |
| divergence step | 77 | 77 |
| track_err | 1.006 | 1.006 |
| phys_min_z / final_z | 0.56 / 2.15 | 0.56 / 2.15 |

Identical on every column, and the same holds for K2 (54 ↔ 55) and K5 (57 ↔ 58). Two
consequences: the eval is deterministic given the seed, and `SAFE_EPS_MODE=legacy` restores
pre-Fix_16 behaviour exactly rather than approximately. Everything the `scaled` arm does
differently is attributable to `eps`.

### 2.3 🔴 Both K5 jobs hit the SLURM time limit

| job | arm | K | variants completed | outcome |
|---|---|---|---|---|
| 25319 | scaled | 1 | 10 / 10 | ✅ complete |
| 25320 | scaled | 2 | 10 / 10 | ✅ complete |
| 25321 | scaled | 5 | 5 (2 of them partial: `dpcc-c-tightened` n=2) | 🔴 **TIME LIMIT** |
| 25316 | legacy | 1 | 10 / 10 | ✅ complete |
| 25317 | legacy | 2 | 10 / 10 | ✅ complete |
| 25318 | legacy | 5 | 11 (`hardflow_sls` n=9) | 🔴 **TIME LIMIT** |

The 10-of-17 variant count at K1/K2 is **not** truncation — it is the HardFlow low-K
degeneracy guard firing correctly and dropping the 7 `hardflow_*` variants (`K=1, A=0.5 →
n_genuine=0`). At K5 HardFlow is admissible again, which is why K5 has more variants to get
through, and why it is the one that ran out of time.

🟡 **The fix caused the timeout.** Pre-fix a `diffuser` rollout died at step ~77; post-fix it
flies ~600. K5 `dpcc-c-tightened` logged **1764.7 s for 2 trials** with "~7058.6 s to go" for
that single variant. A complete K5 sweep needs a substantially larger `--time`.

### 2.4 🟡 The DA discovery tooling mis-parses the `FMPCC_UAV_EVAL_TAG` suffix

The K-extractor does not survive a suffix after `T0.5`:

| candidate | folder | parsed `K` | parsed `FolderName` |
|---|---|---|---|
| 52 | `Emf_K1_..._T0.5_fix16legacy` | *(empty)* | `pillars\|mf\|bbunet\|dp0.5` |
| 53 | `Emf_K1_..._T0.5_fix16scaled` | *(empty)* | `pillars\|mf\|bbunet\|dp0.5` |
| 55/56 | K2 legacy/scaled | *(empty)* | *(same string)* |
| 58/59 | K5 legacy/scaled | *(empty)* | *(same string)* |

All six collapse to one `FolderName`, so they are **absent from `candidates_detailed.csv`**
(48 rows for 71 candidates) and **mis-binned in `uav_k_sweep.csv`**, which groups on `K`.

**They are fully present and correctly separated in `per_rollout_detail.csv`** (keyed on
`Candidate`), which is what every number in this DA is computed from. Nothing is lost — but do
not read the tagged runs out of the aggregate CSVs.

➡️ **Action:** extend the folder-name regex in the UAV DA discovery step to accept a trailing
`_<tag>`, and carry the tag as its own column. Until then, tagged runs must be read
per-rollout.

---

## 3. Headline: unguided `mf` (`variant=diffuser`, no projector)

n = 10 rollouts per cell. `pre` = untagged pre-fix run; `legacy` reproduces it exactly and is
omitted for width.

| K | arm | abort % | success % | goal line crossed % | goal_dist mean / median (m) | track_err | phys_safe | final z (m) |
|---|---|---|---|---|---|---|---|---|
| 1 | pre    | **100.0** | 0.0 | 0.0 | 6.50 / 6.46 | 1.006 | 0.00 | 2.15 |
| 1 | scaled | **0.0** | 10.0 | **100.0** | **0.62 / 0.66** | 0.338 | **1.00** | 1.13 |
| 2 | pre    | **100.0** | 0.0 | 0.0 | 6.46 / 6.45 | 0.798 | 0.00 | 1.24 |
| 2 | scaled | **0.0** | 30.0 | **100.0** | **0.66 / 0.78** | 0.371 | **1.00** | 1.13 |
| 5 | pre    | **100.0** | 0.0 | 0.0 | 6.49 / 6.51 | 1.002 | 0.00 | 3.26 |
| 5 | scaled | **0.0** | **90.0** | **100.0** | **0.36 / 0.30** | 0.446 | **1.00** | 1.13 |

Five things worth pulling out:

- **The pre-fix goal distances are pathologically tight**: all 30 rollouts land in
  [6.29, 6.68] m. A stochastic quality problem does not produce a 0.4 m spread across three
  different K and 30 seeds — a deterministic runaway does. This is independent corroboration
  of the study's mechanism and against the "the samples are just bad" reading.
- **`phys_final_z` is exactly 1.13 m in every scaled rollout, and equals `phys_min_z`.**
  Altitude is now dead flat, which is what a ~0-authority z channel is supposed to look like.
  Pre-fix it ran to 2.15 / 3.26 m — up against the 3.30 m ceiling trigger.
- **The gain in success is monotone in K** (10 → 30 → 90 %). Pre-fix it was 0 at every K,
  because the runaway killed the rollout before K could matter.
- **Generation time is untouched** (`fm_ms` 9.3/18.2/44.5 ms, unchanged to the decimal). The
  fix costs nothing in the model.
- 🟡 **`n_violations` rises** (28.5 → 286.1 at K1). This is survivorship, not regression:
  pre-fix the drone stopped accumulating violations when it died at step 77. Normalised per
  *live* step the picture inverts — see §5.

---

## 4. Projected variants — pre-fix → `fix16scaled`

Abort % / success %, n = 10 per cell. `*-geo_free` rows are the control (they were already
healthy).

### K1
| variant | abort % | success % | goal_dist (m) | track_err |
|---|---|---|---|---|
| `diffuser` | 100 → **0** | 0 → 10 | 6.50 → **0.62** | 1.006 → 0.338 |
| `dpcc-c` | 30 → **10** | 10 → 0 | 0.93 → 0.39 | 0.452 → 0.483 |
| `dpcc-c-tightened` | 60 → **20** | 20 → 0 | 1.15 → 0.89 | 0.494 → 0.475 |
| `dpcc-r` | 50 → **10** | 30 → 10 | 1.44 → 0.48 | 0.524 → 0.531 |
| `dpcc-r-tightened` | 50 → **30** | 20 → 0 | 1.47 → 0.70 | 0.493 → 0.455 |
| `dpcc-t` | 80 → **0** | 0 → **40** | 2.46 → 0.33 | 0.601 → 0.528 |
| `dpcc-t-tightened` | 50 → **40** | 0 → 0 | 1.13 → 1.43 | 0.635 → 0.637 |
| *`dpcc-c-geo_free`* | *0 → 0* | *0 → 0* | *0.68 → 0.65* | *0.329 → 0.325* |
| *`dpcc-r-geo_free`* | *0 → 0* | *0 → 10* | *0.65 → 0.58* | *0.329 → 0.338* |
| *`dpcc-t-geo_free`* | *0 → 0* | *0 → 40* | *0.58 → 0.45* | *0.325 → 0.377* |

### K2
| variant | abort % | success % | goal_dist (m) |
|---|---|---|---|
| `diffuser` | 100 → **0** | 0 → **30** | 6.46 → 0.66 |
| `dpcc-c` | 30 → **0** | 10 → 0 | 0.90 → 0.32 |
| `dpcc-c-tightened` | 40 → **10** | 0 → 0 | 0.94 → 0.62 |
| `dpcc-r` | 50 → **0** | 20 → 0 | 1.75 → 0.29 |
| `dpcc-r-tightened` | 40 → **20** | 0 → 0 | 1.17 → 0.66 |
| `dpcc-t` | 60 → **30** | 10 → **40** | 1.69 → 0.87 |
| `dpcc-t-tightened` | 70 → **0** | 0 → 10 | 1.96 → 0.33 |
| *`*-geo_free` (3 rows)* | *0 → 0* | *10/30/40 → 40/20/60* | *≈ unchanged* |

### K5 — 🔴 partial (`scaled` job killed after 5 variants)
| variant | abort % | success % | goal_dist (m) | note |
|---|---|---|---|---|
| `diffuser` | 100 → **0** | 0 → **90** | 6.49 → 0.36 | complete |
| `dpcc-c` | 90 → **20** | 0 → 10 | 3.02 → 0.64 | complete |
| `dpcc-r` | 100 → **20** | 0 → 0 | 4.99 → 0.73 | complete |
| `dpcc-r-tightened` | 90 → **0** | 0 → 0 | 3.83 → 0.31 | complete |
| `dpcc-c-tightened` | 80 → — | 0 → 0 | 3.48 → 0.76 | 🔴 n=2, do not cite |
| `dpcc-t*`, `*-geo_free` | — | — | — | 🔴 never ran |

**Reading.** Aborts fall on **9 of 10** non-`geo_free` variants across K1/K2 (the exception is
K1 `dpcc-t-tightened`, 50 → 40 %, which is within noise at n=10). Goal distance improves on
every single variant except K1 `dpcc-t-tightened`. Success is noisy at n=10 and moves both
ways — **do not read the success column as a fix effect at K1/K2**; the abort and goal-distance
columns are where the signal is.

**Abort reasons** (whole-job counts, all variants):

| K | arm | `inverted` | `off_route` | `overspeed` | total |
|---|---|---|---|---|---|
| 1 | legacy | 26 | 16 | 0 | 42 |
| 1 | scaled | 10 | 1 | 0 | **11** |
| 2 | legacy | 27 | 11 | 1 | 39 |
| 2 | scaled | 4 | 2 | 0 | **6** |
| 5 | legacy | 47 | 14 | 4 | 65 *(partial job)* |
| 5 | scaled | 4 | 1 | 0 | **5** *(partial job)* |

`overspeed` vanishes entirely under `scaled`. `off_route` falls hardest (16 → 1 at K1), which
is the signature of the runaway: the drone was being driven off the corridor, not failing to
track it.

---

## 5. Cross-engine context — 🔴 confounded

`fm` and `af` were **not** re-run with Fix_16, so they still carry `eps = 1.0`. This table is
fixed-`mf` against unfixed baselines and **cannot be used as an engine comparison**.

`pillars`, `variant=diffuser`, n = 10:

| engine | K | eps | abort % | success % | goal_dist (m) | viol / **live** step | track_err | phys_safe |
|---|---|---|---|---|---|---|---|---|
| af | 1 | 1.0 | 10 | 0 | 1.30 | 0.678 | 0.519 | 0.80 |
| af | 2 | 1.0 | 30 | 0 | 1.56 | 0.675 | 0.443 | 0.60 |
| af | 5 | 1.0 | 10 | 0 | 1.13 | 0.655 | 0.367 | 0.70 |
| fm | 1 | 1.0 | 80 | 0 | 2.11 | 0.579 | 0.594 | 0.00 |
| fm | 2 | 1.0 | 0 | 0 | 1.19 | 0.749 | 0.421 | 0.00 |
| fm | 5 | 1.0 | 10 | 0 | 1.18 | 0.724 | 0.532 | 0.80 |
| fm | 20 | 1.0 | 90 | 0 | 1.00 | 0.713 | 0.517 | 0.00 |
| mf | 1 | 1.0 | **100** | 0 | 6.50 | 0.400 | 1.006 | 0.00 |
| mf | 2 | 1.0 | **100** | 0 | 6.46 | 0.390 | 0.798 | 0.00 |
| mf | 5 | 1.0 | **100** | 0 | 6.49 | 0.510 | 1.002 | 0.00 |
| **mf** | **1** | **3.1e-05** | **0** | 10 | **0.62** | 0.458 | 0.338 | **1.00** |
| **mf** | **2** | **3.1e-05** | **0** | 30 | **0.66** | 0.538 | 0.371 | **1.00** |
| **mf** | **5** | **3.1e-05** | **0** | **90** | **0.36** | **0.306** | 0.446 | **1.00** |

"viol / live step" divides `n_violations` by `divergence_step` for aborted rollouts and by
`n_steps` otherwise — the raw count is uninterpretable when a rollout dies at step 77.
On that normalisation the pre-fix `mf` rows (0.39–0.51) look *better* than `fm`/`af`
(0.58–0.75) purely because they died early; the fixed `mf` K5 row (0.306) is the only one that
is low **and** flies the whole route.

Fixed-`mf` is ahead of unfixed `fm`/`af` on abort, success, goal distance, physical safety and
per-live-step violations at K5. Given the confound the honest statement is: **the fix restores
`mf` to at least parity with the unfixed baselines and the K5 numbers suggest more.**
An engine claim needs the same fix applied to `fm` and `af` — see §8.

---

## 6. What Fix_16 did **not** fix

- 🔴 **Success + constraints is 0 / 2876 pillars rollouts** — every engine, every K, both arms,
  including all 551 Fix_16 rollouts. Identical to the 2026-08-30 finding. `pillars` remains
  unsolved on the constraint axis and **no ranking may be published from it**.
- 🔴 **Collision-free completion is 4 / 2876** (all four in pre-fix `mf` K5 `dpcc-c*`). The
  fixed arm produced none — expected for `diffuser` (no projector) but not obviously so for
  the `dpcc-*` arms.
- 🔴 **The `mf` vertical gain itself (link 3) is untouched.** Fix_16 removes the amplifier, so
  a +0.11 /m gain now commands +0.11 × 3.1e-05 m/step instead of +0.11 × 1 m/step. The learned
  pathology is still in the checkpoint; it is simply no longer connected to anything with
  authority. Any future change that re-widens that channel will bring the failure straight back.
- 🔴 **`corridor` and `s_curve` were not tested.** On the older data in this batch, `corridor`
  `mf` shows the same catastrophic signature at every K (goal_dist ≈ 30 m, track_err ≈ 100,
  final z 0.09 m — floor impact, 0 % success), and `s_curve` aborts 100 % for **every** engine
  including the `diffusion` baseline, which points at a separate scene-level problem unrelated
  to Fix_16.

---

## 7. Costs and side effects

| | K1 `dpcc-c` | K2 `dpcc-c` | K5 `dpcc-c` |
|---|---|---|---|
| `fm_ms` (generation) pre → scaled | 9.4 → 9.4 | 18.3 → 18.3 | 45.2 → 44.7 |
| `proj_ms` pre → scaled | 75.9 → **120.2** | 78.4 → 62.8 | 1425 → **1751** |
| `total_ms` p95 pre → scaled | 266 → **547** | 248 → 273 | 6541 → **7759** |

Generation cost is unchanged, as it must be. **Projection cost rises 1.2–1.6× at K1 and K5.**
The likely cause is structural: the z decision variable now lives in a ±3.1e-05 box, which is a
near-equality constraint, and SLSQP spends iterations on it. K2 moves the other way, so n=3
cells is not enough to call the size of the effect — but the direction at K1 and K5 is large.

➡️ **Follow-up:** rather than bounding a degenerate action dimension to a hair-width box,
**eliminate it from the decision vector** and write the constant back afterwards. Cheaper,
better conditioned, and it removes the failure mode structurally instead of numerically.

(`over_budget_frac` rises to 1.00 in several cells. Per the standing convention, `budget_ms` /
33 Hz is a data-rate artefact, **not** a real-time target — not reported as pass/fail.)

---

## 8. Diagnostics — did the fix actually engage?

All four new log surfaces fired, on the cluster, as designed:

```
FIX_16:  SAFE_EPS_MODE=scaled  SAFE_EPS_FRAC=1e-3  EVAL_TAG=fix16scaled
[ utils/normalization ] Constant data in actions[2] | max = min = 9.895e-13
    → Fix_16 widened by eps=3.086e-05 (mode=scaled).
      unnormalize scale for this channel = 3.086e-05 data-units per unit output.
[ eval ] Fix_16 DEGENERATE actions[2]: constant in the expert data — no training signal.
[ eval ] Fix_16 projector action_bounds=auto
    → lb=[ 3.5000e-05 -3.9716e-02 -3.1000e-05]
      ub=[ 4.4061e-02  3.9704e-02  3.1000e-05]   (degenerate dims now bounded, not +/-1)
```

against `legacy`, same channel:

```
[ utils/normalization ] Constant data in actions[2] | max = min = 0.0
    → Fix_16 widened by eps=1.000e+00 (mode=legacy).
[ eval ] Fix_16 projector action_bounds=auto
    → lb=[ 3.5000e-05 -3.9716e-02 -1.0000e+00]
      ub=[ 0.044061  0.039704  1.      ]
```

The realised `eps` is **3.086e-05**, close to the 3.998e-05 predicted offline (the offline test
reconstructed the non-constant half-widths synthetically). The projector z-cap moved from
±1.000 m to ±3.1e-05 m — a **32,000× reduction** — which is the whole intervention.

Three incidental findings from the diagnostics:

1. 🟢 **The all-constant fallback path works.** `rewards[0]` is constant with no non-constant
   reference dim, so `_resolve_eps` logged
   `⚠ Fix_16: ALL dimensions constant — no reference scale; falling back to eps=1.0` and did.
   Harmless — rewards are not in the control path — but it confirms the branch.
2. 🟡 **The un-silenced clip warning fires on both arms, identically**, at
   `observations (-1.0617, 0.3512)` and `actions (-1.0017, 0.0816)`. Two things follow: the
   first sample is identical across arms (consistent with the study's finding that step-0
   actions are indistinguishable), and **the observation channel is also being clipped** —
   `-1.0617` is outside the normalizer range. That is a separate, still-unfixed issue and it
   affects every engine.
3. 🟡 **`actions[2]` reads `max = min = 9.895e-13` in the scaled arm and exactly `0.0` in the
   legacy arm.** Both are detected as constant and both round-trip, so it is immaterial here,
   but the two jobs did not compute a bit-identical `mins`/`maxs`. Unexplained; worth one look
   if the normalizer is touched again.

---

## 9. Verdict

**The study's mechanism is confirmed and the fix works.** The prediction on record was: *if
`mf` unguided still aborts 30/30 after the fix, the mechanism is wrong.* It aborts **0/30**,
with goal distance down 18×, physical safety 0.00 → 1.00, and the healthy `geo_free` control
arms unmoved. Combined with the bit-exact `legacy` rollback on the same rev, this is as clean
an attribution as this setup can produce.

**Scope of the claim, precisely:** Fix_16 removes a *scale-calibration defect* that gave a
zero-signal action channel ±1 m of authority per step. It does not improve the model, does not
make `pillars` solvable, and does not remove the learned vertical gain that made `mf` uniquely
sensitive to that defect.

---

## 10. Next runs

Ordered by value.

1. 🔴 **Re-run K5 with a larger `--time`.** Both K5 arms died at the wall; K5 is where the
   effect is largest (90 % success) and where the evidence is thinnest.
   ```bash
   FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled_v2 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf pillars "6" "5"
   ```
2. 🔴 **Apply the fix to `fm` and `af`** — without it every cross-engine number in §5 is
   confounded, and `fm` K1/K20 abort at 80/90 % which the same defect may explain.
   ```bash
   FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm pillars "6" "1 2 5"
   FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=fix16scaled \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "1 2 5"
   ```
3. 🟡 **`corridor` with the fix** — `mf` shows the same catastrophic signature there
   (goal_dist ≈ 30 m, floor impact) and it has never been A/B'd.
4. 🟡 **Fix the DA folder-name regex** (§2.4) before the next tagged batch, otherwise the
   aggregate CSVs keep silently dropping tagged runs.
5. 🟡 **Eliminate degenerate dims from the projector decision vector** (§7) instead of boxing
   them, and re-measure `proj_ms`.
6. ⚪ The 2×2 from the study (`mf` on the SiT/DiT bone, `mf_dit_trajectory.py`) is unaffected by
   this result — it targets link 3, why `mf` learned the gain, which Fix_16 does not address.

---

## 11. Provenance

| item | value |
|---|---|
| Slurm jobs | 25316 (legacy K1), 25317 (legacy K2), 25318 (legacy K5, killed), 25319 (scaled K1), 25320 (scaled K2), 25321 (scaled K5, killed) |
| pipeline jobs | 25294 (scaled), 25295 (legacy) |
| git rev | `def8fdf` — all six |
| node | `i6-gpu-1` |
| eval folders | `.../mix_uav_mf/H8_...bbunet/Emf_K{1,2,5}_mpc4_pid_stopgo_T0.5_fix16{scaled,legacy}/6/` |
| DA batch | `temp/0309/batch_uav_20260903_204120/` |
| primary table | `per_rollout_detail.csv`, candidates 51–59 (2876 pillars rollouts; 551 in the tagged arms) |
| ⚠ do not use | `candidates_detailed.csv`, `uav_k_sweep.csv` for the tagged runs — see §2.4 |
