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
9. 🟡 **Fix_16 is unlikely to rescue `s_curve` — but "impossible" was overstated** (§6.1).
   Against: `s_curve` aborts 87–94 % for *every* engine through `p_des_runaway`, a horizontal
   integrator gap that fires **0 times** in `pillars`; its altitude never misbehaved pre-fix
   (90.5 % abort among the 1297 benign-altitude rollouts); and deleting the *entire* action
   bound (`bounds_free`) barely moves it. For: the z channel is **not** separable inside the
   SLSQP solve (§6.2.2), and the raw `s_curve` output is broken too (`track_err` p90 175).
   `scaled` is the default now, so any new `s_curve` job gets the fix **free** — no dedicated
   A/B is warranted either way.
10. 🟢 **The `s_curve` lever is the projector geometry, not the normalizer.** `geo_free` (drop
   the geometric constraints, keep dynamics) takes abort **1.00 → 0.49** and `track_err` p90
   **175.5 → 0.38**; `geo_free-model_free` goes straight back to 1.00. The planning corridor is
   only **24 cm** wide after inflation, and SLSQP non-convergence is **silent by design** — the
   partly-converged iterate is flown. See §6.2.3.
11. 🔴 **The projector never plans in z** — every wall and pillar binds `(x, y)` only, so each
   obstacle is an **infinite cylinder**. It cannot route over or under anything (§6.2.1).
12. 🔴 **The scenes are geometrically under-dimensioned for this policy** — the tightest
   `pillars` planning channel is **6 cm** of slack against a **34 cm** median tracking error
   (`s_curve`: 12 cm vs 30 cm). This is engine-independent, fix-independent, and cannot be tuned
   away without discarding the drone's real 0.31 m rotor reach. It is the single largest fact in
   this document and it is **not** about Fix_16 — see **Part II**.
13. 🟡 **The DA tooling mis-parses the tagged folder names** — `K` comes back empty and all six
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

### 1.1 What Fix_16 actually is

**One number, in one file** — the `eps` that `SafeLimitsNormalizer` uses to widen a *constant*
data dimension (`mix_uav/datasets/normalization.py`).

`LimitsNormalizer` maps each channel through its own `mins`/`maxs`. If a channel is constant
then `maxs - mins == 0` and the map is `0/0`. `SafeLimitsNormalizer` avoids the NaN by pushing
the two bounds apart by `eps`:

```python
self.mins[i] -= eps
self.maxs[i] += eps
```

Inherited code hard-coded **`eps = 1.0`**. Crucially, `normalize()` sends constant data to the
midpoint `0` for *any* `eps` — which is why this went unnoticed for the whole project, and why
changing it **does not invalidate a checkpoint**. What `eps` actually sets is the channel's
physical scale on the way back *out*:

| | `eps = 1.0` (legacy) | `eps = 3.086e-05` (scaled, realised) |
|---|---|---|
| `unnormalize` scale for the channel | **1.0 data-unit per unit output** | 3.086e-05 |
| what a saturated (±1) model output commands | **±1 m of altitude per step** | ±3.1e-05 m per step |
| `action_bounds:'auto'` z-cap handed to the projector | **±1.000** | ±3.1e-05 |

So the one channel with **zero training signal** was the *loudest* channel in the action
vector, and the projector was told it was free to move a full metre of altitude per step.

Fix_16 derives the widening from the data instead, so a degenerate channel ends up quieter than
every real one rather than louder:

```python
halfw = (maxs - mins)[~const] / 2.0                 # half-widths of the NON-constant dims
eps   = max(median(halfw) * SAFE_EPS_FRAC, 1e-8)    # SAFE_EPS_FRAC default 1e-3
```

Knobs:

| env var | default | effect |
|---|---|---|
| `FMPCC_SAFE_EPS_MODE` | `scaled` | `legacy` restores `eps = 1.0` byte-for-byte, for A/B (§2.2) |
| `FMPCC_SAFE_EPS_FRAC` | `1e-3` | fraction of the median non-constant half-width |
| *(constant)* `SAFE_EPS_FLOOR` | `1e-8` | never collapse to exactly zero |
| *(fallback)* all dims constant | `eps = 1.0` | no reference scale exists; logs a ⚠ and says so (§8.1) |

Shipped in the same fix: the out-of-range clip in `unnormalize` was **silent** (the warning was
commented out) — it now warns once per normalizer and keeps `_clip_events` / `_clip_max`
counters, and eval logs the resolved `eps` plus the derived `action_bounds` (§8).

🔴 **What Fix_16 is not.** Not a model change, not a retrain, not a controller change, not a
projector change. The weights are byte-identical across both arms; only two env vars differ.
It is a **units** fix — the same network output now means 32,000× less altitude. That scope is
exactly why it repairs `pillars` `mf` (§3) and why it cannot touch `s_curve` (§6.1).

### 1.2 Why `fm` and `af` never hit this — the part that is *not* the normalizer

This is the question the fix does not answer by itself, and it matters, because the answer
decides whether Fix_16 is a repair or a papering-over.

**Links 1 and 2 are shared by all three engines.** Every engine trains on the same expert data
(`Δz ≡ 0`) and every engine loads through the same `SafeLimitsNormalizer(eps=1)`. So `fm` and
`af` also had a zero-signal z channel wired to ±1 m of authority per step. They simply **never
pushed on it**. The measured vertical feedback gain (§3.6.2 of the study — `Δz_cmd ~ b₁·e_z +
b₂·step + b₃`, step included as a covariate, matched window ≤150 steps):

| engine | K=1 | K=2 | K=5 | |
|---|---|---|---|---|
| `af` | −0.0018 | −0.0084 | −0.0123 | ✅ restoring |
| `fm` | −0.0163 | +0.0123 | −0.0265 | ✅ restoring / ~0 |
| 🔴 `mf` | **+0.1121** | **+0.0804** | **+0.0310** | 🔴 destabilising at every K |

`b₁ > 0` means *"the higher the drone is above cruise, the harder the model pushes it up"*. On
the **x and y** channels `mf`'s `b₁` sits at −0.011…+0.027 — inside the `fm`/`af` spread. The
instability is z-specific *and* mf-specific.

**Why is that gain free to be anything at all?** Because a constant channel has no gradient
constraining it. In normalized space the training target for `actions[2]` is exactly `0` at
every timestep, for every engine. Nothing in the loss says *"do not couple this output to the
position channels"* — the data is silent on the entire question. The gain is therefore **pure
inductive bias of the objective and the backbone**, and `mf` drew the bad number.

Two structural reasons why `mf` is the arm most exposed to drawing it:

**(a) Plain FM's target for a dead channel is self-contained; MeanFlow's is not.** On the
linear path `x_t = (1−t)·x₀ + t·x₁` with `x₁ = 0`, the velocity target for that channel is
`u = x₁ − x₀ = −x₀ = −x_t/(1−t)` — a function of *that channel's own* `x_t` and nothing else.
It is a 1-D regression the network can fit exactly, and cross-channel coupling earns no loss
reduction. MeanFlow instead regresses the **average** velocity and builds its target by
bootstrapping through the network's own JVP,
`u_tgt = v − (t−r)·(∂ₜu + v·∂ₓu)`. That `∂ₓu` is a directional derivative **across all input
channels jointly**. The dead channel's target is therefore a function of the network's Jacobian
coupling into the *live* position channels — and no data term pins that coupling to zero. A
learned dependence of `Δz` on the position channels is exactly what "a gain on `e_z`" means.

**(b) `mf` queries its sampler where it was never trained.** §3.6.7: simulating
`_sample_tau_pair` (400 k draws), `P(anchor r < 0.05)` is **0.0004** for `mf` against
**0.0739** for `fm`'s `t` — **185×** less training mass at the corner the sampler starts from,
and the K=1 corner `(r=0, h=1)` drew **zero** hits in 400 k samples. The first step off pure
noise carries the whole transport and it is **extrapolated, not interpolated** — which is
precisely where an unpinned Jacobian coupling surfaces. The study measured
`corr(b₁, coverage) = −0.994` across K, so this is a quantitative contributor, not a story.

🔴 **What this does not establish.** `mf` is the only two-time U-Net (`E(τ) + E_h(h)`) and `af`
is the only SiT — **engine and backbone are perfectly confounded, one cell per condition**. And
on `avoiding-d3il` the pairing *reverses*: MF-U-Net works, AF-U-Net fails. So "MeanFlow's
objective causes this" is **unearned** on current evidence; the mechanism above is a hypothesis
consistent with the measurements. The 2×2 that settles it — `mf` on the SiT/DiT bone, models
already present in `mix_uav/models/` — is §10 item 6.

➡️ **The consequence for Fix_16's status.** `fm` and `af` were not *safe*, they were *lucky*:
they sat next to the same live wire with a gain that happened to be ≈ 0. Fix_16 de-energises
the wire, so the luck is no longer load-bearing. But the `mf` checkpoint still carries
`b₁ = +0.11 /m`; anything that re-widens that channel brings the failure straight back (§6).

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

➡️ ✅ **FIXED same day.** `run_tag` is now parsed, labelled and used as a grouping key —
including in `K_SWEEP_KEYS`, without which the fixed regex would have *pooled* the two A/B arms
instead of dropping them, which is worse. `_reduce` also groups with `dropna=False` so a NaN
axis can never silently delete rollouts again. See
[`../../DA_Code/DA_UAV_v1/CHANGELOG_20260903_run_tag_axis.md`](../../DA_Code/DA_UAV_v1/CHANGELOG_20260903_run_tag_axis.md).
🔴 The numbers in **this** DA were computed from `per_rollout_detail.csv` before that fix and
are unaffected; re-run `main_da_batch.py` over the 0309 tree to regenerate correct aggregates.

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
  final z 0.09 m — floor impact, 0 % success) — that one *is* a plausible Fix_16 candidate and
  is worth a run. `s_curve` is a different case: it aborts 87–94 % for every engine including
  the `diffusion` baseline, and the evidence on whether Fix_16 can reach it is genuinely
  two-sided — §6.1 lays out both, §6.2 the mechanisms.

---


### 6.1 Can Fix_16 help `s_curve`? — 🟡 reopened, and the earlier "no" was too flat

An earlier revision of this section answered a flat **no** on the grounds that `s_curve` fails
horizontally and Fix_16 only touches z. That argument is still the strongest single one, but it
was **incomplete in two ways** worth stating before the verdict:

- It treated the NLP as if the z channel were separable from x/y. It is not — §6.2.2.
- It implied `s_curve` fails only through the controller. The **raw** `s_curve` output is also
  broken (`diffuser` `track_err` p90 = **175.5**), and a raw-output defect is exactly Fix_16's
  category.

So the question deserves a real answer rather than a dismissal. Here is the evidence on both
sides, then the verdict.

#### The case *for* — three genuine openings

1. **The raw output is broken on `s_curve` too.** `diffuser`: abort **1.00**, `goal_dist` 5.98 m,
   `track_err` median 1.158 but **p90 175.5** — a heavy tail, i.e. a subset of rollouts diverge
   catastrophically rather than drift. Fix_16 repairs a raw-output scale defect. The categories
   overlap.
2. **Fix_16 lands on the constraint family that demonstrably matters on `s_curve`.** It rescales
   the `dz` coefficient inside the *dynamics* deriv rows (§6.2.2), and the dynamics family is
   the only one doing any good on this scene: `geo_free` (dynamics ON) aborts **0.49** while
   `geo_free-model_free` (dynamics OFF) aborts **1.00**. That is not a null intersection.
3. **The NLP is solved jointly, so z is not separable in practice.** Pre-fix, every constraint
   row touching the z action carried a coefficient ~45× larger than the x/y rows purely because
   of `eps` (§6.2.2). SLSQP converges on one scalar tolerance for the whole system. Measured
   proof that the solver *is* sensitive to `eps` alone: `proj_ms` moved **1.2–1.6×** on
   `pillars` with nothing else changed (§7).

#### The case *against* — and it is heavier

1. 🔴 **The empirical ceiling.** `bounds_free` deletes the **entire** action-magnitude bound on
   `s_curve` and the abort rate barely moves: **0.98** vs 0.95–1.00 with it on. Fix_16 changes
   *one third* of that bound. Whatever the effect is, it is bounded above by an intervention
   that measurably does nothing.
2. 🔴 **The amplifier existed on `s_curve` but never engaged.** Pre-fix altitude is benign:
   `phys_min_z` median 1.02–1.10, `phys_final_z` median 1.25–1.43 (max 3.23), against `pillars`
   `mf` reaching `final_z` **92.29 m** / `min_z` **−7.04 m**. Restrict to the **1297 / 2040**
   `s_curve` rollouts whose altitude never misbehaved (`final_z < 3 m` and `min_z > −0.1 m`) and
   the abort rate is **90.5 %** — indistinguishable from the 87–94 % of the full set. There is
   no z pathology for Fix_16 to remove.
3. 🔴 **Different abort surface.** `s_curve` aborts are 55–90 % `p_des_runaway` (the rest
   `inverted`); `off_route`, `overspeed`, `off_map` and `nan_state` never fire. `p_des_runaway`
   fires **0 times** in `pillars` — including in all six Fix_16 jobs. Full per-job counts:

   | job | engine | K | `p_des_runaway` | `inverted` | aborts / 100 |
   |---|---|---|---|---|---|
   | 25077 | fm | 1 | 55 | 24 | 79 |
   | 25078 | fm | 2 | 65 | 15 | 80 |
   | 25079 | fm | 5 | 63 | 27 | 90 |
   | 25080 | fm | 10 | 68 | 23 | 91 |
   | 25072 | fm | 20 | 78 | 17 | 95 |
   | 25081 | mf | 1 | 72 | 16 | 88 |
   | 25082 | mf | 2 | 74 | 16 | 90 |
   | 25083 | mf | 5 | 78 | 20 | 98 |
   | 25084 | mf | 10 | 78 | 18 | 96 |
   | 25073 | mf | 20 | 90 | 8 | 98 |
   | 25075 | **diffusion** | plan-block | 15 | **75** | 90 |

4. 🔴 **It is engine-independent.** The `diffusion` / DPCC baseline aborts **90 %**. `pillars`
   `mf` was singular *because* of link 3, a learned `mf` gain; nothing analogous can explain a
   scene that defeats every engine including the baseline.
5. 🔴 **Fix_16 could plausibly make `s_curve` slightly *worse*.** It shrinks the `dz` coefficient
   in the dynamics deriv row by 32,000× (2.0 → 6.17e-05). A row with near-zero coefficients is
   near-vacuous under a shared tolerance — and the dynamics family is the one thing keeping
   `s_curve` alive (point 2 of the "for" list, read the other way).

#### 🟡 Verdict — low probability, non-zero, and **you get it for free**

Not "no". **"Unlikely to be the lever, cheap enough that it does not need its own run."**
`scaled` is now the **default** `SAFE_EPS_MODE`, so *any* new `s_curve` job carries Fix_16
whether or not it is the point of the job. There is no reason to spend a dedicated A/B on it
and no reason to avoid it either.

🟢 **What the same data says is the actual lever, with a demonstrated 20-point effect:**

| variant (s_curve, all engines pooled) | n | abort | success | `track_err` med / p90 | `goal_dist` |
|---|---|---|---|---|---|
| `diffuser` (no projector) | 160 | **1.00** | 0.00 | 1.158 / **175.5** | 5.98 |
| `dpcc-c` (full stack) | 140 | 0.97 | 0.01 | 0.336 / 1.46 | 5.09 |
| `bounds_free` (no action bound) | 75 | 0.98 | 0.00 | 0.437 / 54.6 | 5.17 |
| `model_free` (no dynamics) | 155 | 0.96 | 0.00 | 1.202 / 186.5 | 6.11 |
| 🟢 **`geo_free`** (no geometry) | 75 | **0.49** | **0.20** | **0.302 / 0.38** | **1.50** |
| 🟢 `geo_free-bounds_free` (dynamics only) | 75 | **0.49** | 0.17 | 0.297 / 0.32 | 1.46 |
| `geo_free-model_free` (bounds only) | 75 | 1.00 | 0.00 | 1.090 / 229.6 | 6.14 |

**Deleting the geometric constraints halves the abort rate and is the only thing on this scene
that produces flight** — `track_err` p90 collapses from 175.5 to **0.38**, and `goal_dist` from
6.0 m to 1.5 m. Deleting the dynamics constraints instead sends it straight back to 1.00. So on
`s_curve` the model can fly and the *projector's geometry* is what breaks the loop.

🔴 **`geo_free` is a diagnostic, not a solution** — read it honestly. Its success is bought by
flying **through the walls**: S&C = **0.013**, `collision_free` = **0.015**, and violations per
live step *rise* (290/796 = 0.36 vs `diffuser`'s 136/741 = 0.18). It does not solve `s_curve`;
it localises the failure to the constraint set. Why that set is near-infeasible is §6.2.3.


### 6.2 Why the raw output diverges, and why projection does not save it

Read from the code, not from the metrics. Three separate answers, and one design fact that
frames all of them.

#### 6.2.1 🔴 The projector never plans in z — every obstacle is an infinite cylinder

The natural expectation is that the NLP takes the FM/diffusion sample and re-routes it in
**xyz** around the geometry. It does not. In `setup_dpcc_projector`
(`mix_uav_test/eval_mix_uav.py:1063`, `:1134`):

```python
_DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'x': 6, 'y': 7, 'z': 8}
...
if 'halfspace' in ctypes and 'geo_free' not in variant:
    _hs = {'x': _DIM['x'], 'y': _DIM['y']}          # ← walls bind x and y ONLY
```

and every obstacle entry in `config/uav_projection.yaml` is
`{type: sphere_outside, dimensions: ['x', 'y'], ...}` — in all three scenes. A `sphere_outside`
on `(x, y)` is an **infinite cylinder in z**. The consequence:

| constraint family | binds | can it move z? |
|---|---|---|
| `halfspace` (walls) | p_x, p_y | ❌ never |
| `obstacles` (pillars, corner caps) | p_x, p_y | ❌ never |
| `geo_bounds` (workspace box) | p_x, p_y, p_z | 🟡 only the `[0.30, 1.80]` altitude slab |
| `bounds` (action magnitude) | Δp_des x, y, **z** | 🟡 a box, carries no geometry |
| `dynamics` (deriv rows) | p_des ← Δp, p ← Δp | 🟡 couples z to itself |

**So the projector solves a 2-D avoidance problem and z comes along as ballast.** It cannot
hop a pillar, duck under a wall, or trade altitude for clearance — not because it chose not to,
but because no constraint it holds is a function of `p_z` and an obstacle at the same time.

This is worth stating plainly because it re-frames Fix_16: the ±1 m z action box was not a
loose-but-useful degree of freedom that the NLP was exploiting for avoidance. It was **a large,
cheap, geometrically meaningless direction in a 72-variable problem** (9 dims × H=8). Nothing
was lost by closing it, which is consistent with §3–§4: aborts fall, geometry-related metrics
do not degrade.

➡️ If routing over obstacles is ever wanted, it is a **config + `_DIM` change**
(`dimensions: ['x','y','z']`, `sphere_outside` → capsule/box in 3-D), not a model change. Out of
scope here; recorded because the question comes up every time someone reads the pillars plots.

#### 6.2.2 The z channel is *not* separable from x/y inside the solve

The objective is separable — `Q = diag(1)` in normalized space
(`mix_uav/sampling/projection.py:67-75` — `cost_dims` is unset, so `Q = torch.eye(...)`), so the cost is `½‖z − ẑ‖²` with equal weight on all
72 variables. If that were the whole problem, the z subproblem would decouple and Fix_16 could
not possibly move x/y. (Note the `cost_dims` branch sets every weight to `1` as well — there is
currently **no** way to down-weight a channel in the objective, which is precisely the knob a
degenerate dimension would want.)

It is not the whole problem. **Every constraint row is scaled by its channel's normalizer
half-width**, and `eps` *is* that half-width for a degenerate channel:

```python
# SafetyConstraints.build_matrices  — bound rows
mat_append = mat_append * (x_max - x_min) / 2          # = eps for the dead channel

# DynamicConstraints.build_matrices — deriv rows
mat_append[i, i*T + dx_idx] = self.dt * dx_diff        # dt = 1.0, dx_diff = 2·eps
```

| row | x action | z action, `eps=1.0` | z action, `eps=3.086e-05` |
|---|---|---|---|
| action-bound coefficient | 0.022 | **1.0** (45× larger) | 3.1e-05 (713× smaller) |
| dynamics deriv coefficient | 0.044 | **2.0** (45× larger) | 6.2e-05 (713× smaller) |

`minimize(..., method='SLSQP', options={'maxiter': 1000})` converges on **one** scalar tolerance
for the whole KKT system, so rows differing by 45×–700× in scale spend the budget unevenly.
That is a real coupling: **Fix_16 can move the x/y solution, in either direction.** The
measured `proj_ms` shift of 1.2–1.6× from the `eps` change alone (§7) is direct evidence the
solver's trajectory through the problem changed.

➡️ This is the honest reason §6.1 is "unlikely" rather than "impossible", and it is also the
argument for the §7 follow-up: **eliminate the degenerate dimension from the decision vector**
rather than choosing between two badly scaled boxes for it.

#### 6.2.3 Three distinct ways a *projected* variant still crashes

| # | mechanism | code | fired here? |
|---|---|---|---|
| 1 | **Circuit breaker opens** — sustained-slow episode → `return trajectory, inf`, i.e. the **unprojected** sample is executed. A projected variant silently degenerates into `diffuser`. | `projection.py:116-131` (Fix_15.2), `last_proj_skipped` | ❌ **No** — `projection_cb_tripped = 0.00` on every `s_curve` variant. Ruled out. |
| 2 | 🔴 **Non-convergence is silent and the iterate is executed.** `# DPCC itself still silently keeps res.x on non-convergence — that is unchanged here on purpose` (`projection.py:182-186`). A near-infeasible NLP returns a partly-converged point and the loop flies it as if it were a solution. | `projection.py:182-233` | 🔴 **Prime suspect** on `s_curve` — see below. |
| 3 | **The projection is valid but the tracker cannot execute it.** `pid_stopgo` sets `v_des = np.zeros(3)` (`eval_mix_uav.py:1423`) — no velocity feed-forward — while `p_des += action` runs free (`:1416`). In a sustained turn the drone lags, `p_des` keeps advancing, the gap grows monotonically until `|p_des − p| > 5 m`. | `eval_mix_uav.py:1416-1436` | 🔴 **The measured abort reason**, 55–90 % of `s_curve` aborts. |

**Why mechanism 2 is the prime suspect on `s_curve`.** The feasible set is genuinely thin. The
scene's walls have inner faces 0.90 m apart; inflation is `r_drone 0.31 + margin_base 0.02 =
0.33` per side, so the planning corridor is **0.90 − 0.66 = 0.24 m** wide, with a crossover
corner gate of about the same. The set is also **non-convex and switched** — each wall
halfspace is live only over its own `x_active` interval, so the constraint set *changes between
replans*. SLSQP is being handed a 24 cm tube in a non-convex, time-varying problem, 1000
iterations, and no failure signal on the way out. `geo_free`'s 1.00 → 0.49 abort collapse
(§6.1) is what that looks like from the outside.

➡️ **The single highest-value `s_curve` experiment** is therefore not the normalizer. It is:
**(a)** log `res.status` / `last_solve_success` per solve and count non-convergence — the
plumbing already exists (`self.last_solve_success`, added by SolverSwap, currently only read by
`HardFlowNLP`), so this is a reporting change, not a solver change; and **(b)** sweep
`margin_base` down from 0.02 (and reconsider whether the 0.31 m per-axis rotor reach must be
applied to *both* sides of a 0.90 m gap) to see whether the corridor widens enough for the NLP
to converge. Both are cheaper than a model run and both attack the mechanism the data points at.

#### 6.2.4 🟡 The still-unfixed sibling defect — the observation clip

§8.2 records `observations (-1.0617, 0.3512)` clipped on **both** arms. With `cond_mode=pos_only`
the observation is `[p_des | p]`, so **`p_des` itself is an input**. When `p_des` runs away it
leaves the training range and `unnormalize`/`normalize` **clips it**: the model stops being able
to see how far the commanded point has drifted, and keeps emitting whatever it emits at
saturation. That is a lock-in for exactly the `p_des_runaway` failure mode, it is scene-general,
and Fix_16 did **not** address it — it only un-silenced the warning. Same file, same class of
defect (a normalizer range silently truncating the control path), different fix.

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
4. ✅ **Fix the DA folder-name regex** (§2.4) — **done 2026-09-03**; re-run `main_da_batch.py`
   over the 0309 tree to regenerate the aggregates (no eval needed, it only re-reads artifacts,
   and it is the only way to exercise the pandas-side changes).
5. 🟡 **Eliminate degenerate dims from the projector decision vector** (§7) instead of boxing
   them, and re-measure `proj_ms`.
6. ⚪ The 2×2 from the study (`mf` on the SiT/DiT bone, `mf_dit_trajectory.py`) is unaffected by
   this result — it targets link 3, why `mf` learned the gain, which Fix_16 does not address.

**On `s_curve` specifically** (§6.1, §6.2): don't book a dedicated Fix_16 A/B — `scaled` is the
default, so every new job carries it for free. Book these instead, in order:

- 🔴 **a. Count the silent NLP failures.** `self.last_solve_success` is already populated
   (`projection.py:186`) and currently read only by `HardFlowNLP`. Surfacing it per variant is a
   reporting change, no solver change — and it would say directly whether §6.2.3 mechanism 2 is
   what breaks `s_curve`.
- 🔴 **b. Sweep the inflation margin.** `r_drone 0.31 + margin_base 0.02` leaves a **24 cm**
   corridor in a 0.90 m gap. `geo_free` shows what happens when that pressure is removed
   (abort 1.00 → 0.49). Widening it is the intervention with a demonstrated effect size.
- 🟡 **c. A/B `pid_stopgo_anchorP`** — it re-anchors `p_des` to the drone, attacking the
   `p_des_runaway` mechanism (§6.2.3 #3) directly.
- 🟡 **d. Explain the Gen11 → Gen15 regression** — Gen11 `s_curve` candidates 26/27 report
   95 % / 100 % success against Gen15's ~2 %. A 50× drop on one scene is more likely a config or
   data-path regression than a modelling problem, and finding it costs no GPU time.
- ⚪ **e. Fix the observation clip** (§6.2.4) — `p_des` is an input under `cond_mode=pos_only`, so
   a runaway setpoint saturates the model's own view of the runaway. Scene-general, untouched by
   Fix_16.

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

---

# Part II — Is the 2-D projector a design bug? The Gen14 precedent, and what `pillars`/`s_curve` actually reveal

*Added 2026-09-04, prompted by "is the xyz 3D a logic bug in design?". Read from the code and
the configs; no new runs. §II.3 is the part that matters and it is not about Fix_16 at all.*

## II.1 How Gen14 V_A holds obstacles — the same way, and it already A/B's 2-D vs 3-D

`mix_visual_aligning_test/eval_mix_visual_aligning.py` uses the **identical** pattern:

```python
_DIM = {'dx':0,'dy':1,'dz':2, 'des_x':3,'des_y':4,'des_z':5, 'x':6,'y':7,'z':8}   # :209
...
_hs_indices = {'x': _DIM['x'], 'y': _DIM['y']}                                     # :283
```

Both generations inherit it from DPCC-avoiding, which is a **planar** task where a 2-D
reduction is exact. Two things follow that are directly relevant here:

**(a) 🟢 The 3-D machinery already exists and is already exercised — just not by the UAV.**
`ObstacleConstraints.build_matrices` (`projection.py:512-570`) loops `for dim in dims:` and
builds `P`, `q`, `v` for **any** number of dimensions. `config/visual_aligning_eval.yaml` ships
both spellings as a deliberate A/B pair:

```yaml
  - name: obstacle_only_1
      dimensions: ['x', 'y']       # 2D cylinder projection — avoiding-paper style
  - name: obstacle_only_2
      dimensions: ['x', 'y', 'z']  # true 3D sphere — stricter
```

So `dimensions: ['x','y','z']` in `config/uav_projection.yaml` would work **today, with no code
change**. The UAV — the one task that is genuinely 3-D — is the only one that never tried it.
The single real capability gap is **3-D halfspaces**, which the same yaml marks
`PENDING impl` ("3D requires a full plane normal — different yaml format").

**(b) 🟡 Gen14 hit the 2-D-blindness failure mode first, and built a guard.** From the D1
header (`eval_mix_visual_aligning.py:92-104`):

> *"The `obstacles` family is a sphere_outside cylinder on the EE position dims (6,7); it knows
> nothing about the box. So a box that starts inside the obstacle disc is NOT itself a
> constraint violation — but it is a guaranteed-futile rollout… surfacing downstream only as
> unexplained solver thrash."*

Gen14's response was a **pre-flight feasibility guard that skips the context** so "aggregate
metrics are not polluted by a rollout that never had a chance". The UAV's analogue is
`_warn_expert_route_infeasibility` (`eval_mix_uav.py:755`, Fix_12), and it is **weaker on two
axes**: it only *warns* (never skips), and it checks only the **expert reference route**. See
§II.3 for why that second limitation is the important one.

Gen14 aligning has no `s_curve` analogue — its geometry vocabulary is the same three families
(`geo_bounds` / `halfspace` / `obstacles`) but on a table-top EE workspace, with no switched
`x_active` walls and no non-convex crossover.

## II.2 Is the 2-D reduction a logic bug? — 🟢 **No.** It is documented, and made sound by the ceiling

I went looking for a bug and did not find one. The reduction is *deliberate* and each scene has
a stated reason why it is not lossy:

| scene | real geometry | why 2-D is sound |
|---|---|---|
| `pillars` | 6 pillars, **full-height** in the XML | a full-height cylinder **is** its own 2-D projection — the reduction is **exact**, not an approximation |
| `corridor` | walls top at 1.5 m | ceiling is `ub 1.80 − margin 0.33 = 1.47 m < 1.5` — the config comment says outright *"so walls can't be hopped"* |
| `s_curve` | walls top at 1.5 m | same ceiling argument, same comment |

So the projector cannot route over an obstacle **because in these three scenes there is no legal
over-the-top route to find**. The altitude ceiling was chosen to guarantee that. Planning and
scoring agree (both treat walls as z-invariant), so there is no plan/score mismatch either.

🟡 **The one genuine artifact** it produces: `_exec_constraint_violations` applies the 2-D wall
test at *any* altitude, so a diverged rollout at `phys_final_z = 92 m` is scored as colliding
with a 1.5 m wall it is nowhere near. This only touches already-failed rollouts, but it means
**`n_violations` is not a clean physical quantity on diverged runs** — one more reason not to
read violation counts without normalising by live steps.

➡️ The real cost of the 2-D reduction is not correctness, it is **§6.2.1**: the z action
dimension carries *zero* geometric information, which is exactly why the pre-Fix_16 ±1 m z box
was a large, cheap, meaningless direction in the NLP.

## II.3 🔴 The serious finding: the planning channel is **narrower than the policy's own tracking error**

This is what came out of the geometry that I did not expect, and it is scene-level, engine-
independent, and unaffected by every fix discussed above.

Inflation is `r_drone 0.31 + margin_base 0.02 = 0.33 m`, applied to **every** surface. Working
out the resulting free channels from the raw config numbers:

| scene · route | raw geometry | free channel after inflation | **half-width (slack)** |
|---|---|---|---|
| `pillars` · outer | pillar edge `0.6+0.12=0.72`; field ub `1.5` | `[1.05, 1.17]`, expert at 1.11 | 🔴 **0.06 m** |
| `pillars` · centre | two pillar rows at `y=±0.6`, r 0.12 | `[−0.15, +0.15]` | 🔴 **0.15 m** |
| `s_curve` · corridor | wall inner faces 0.90 m apart | `[−0.92, −0.68]`, expert at −0.8 | 🔴 **0.12 m** |
| `s_curve` · crossover | corner balls r `0.05+0.33=0.38` | gate ≈ 0.24 m | 🔴 **0.12 m** |

Now the measured tracking error of the policy that has to fly inside those channels
(`track_err_mean`, best-behaved variant per scene):

| scene | variant | n | median | p25 |
|---|---|---|---|---|
| `pillars` | `geo_free` | 26 | **0.336** | 0.331 |
| `pillars` | `dpcc-c` | 207 | 0.468 | 0.427 |
| `pillars` | `diffuser` | 237 | 0.486 | 0.398 |
| `s_curve` | `geo_free` | 75 | **0.302** | 0.237 |
| `s_curve` | `dpcc-c` | 140 | 0.336 | 0.291 |

🔴 **The tightest `pillars` channel is 6 cm of slack against a 34 cm median tracking error —
5.6× too wide. Even the 25th percentile is 4–5× over. `s_curve` is 12 cm against 30 cm.**

That single comparison explains, without reference to any engine, objective, normalizer or
solver:

- why **success + constraints is 0 / 2876** on `pillars` for every engine, every K, both Fix_16
  arms (§6) — the policy is never inside the tube long enough to score;
- why `geo_free` on `s_curve` "succeeds" 20 % of the time **by flying through the walls** (§6.1)
  — with the constraints off it flies fine; the constraints are what it cannot satisfy;
- why the projector "fights" the sample on every replan: SLSQP is repeatedly asked to move a
  point that is ~0.3 m outside a ~0.1 m tube back into it, in 8 steps, under a dynamics
  equality, inside `maxiter=1000`, non-convex on `s_curve` — and then keeps `res.x` **silently**
  if it fails (§6.2.3 #2).

**Why `_warn_expert_route_infeasibility` said "OK".** It passed on both scenes — `s_curve
homotopy=default expert route OK` (11 logs), `pillars homotopy=(L,L,L)/(L,R,L)/(R,L,R)/(R,R,R)
expert route OK` (6 logs each), all at `margin 0.33 m`. That is a true statement about the
**ideal expert route**, which threads the channel by construction. It says nothing about the
route the policy actually flies — and `_realized_homotopy`'s own docstring concedes the point:
*"the FM policy is unconditioned and never tracks that route"*. The gate answers *"could a
perfect pilot fit?"* when the question that decides the metrics is *"can this pilot, with 0.3 m
of error, fit?"*. Its own docstring already hedges in the right direction — *"treat a near-zero-
slack PASS here as a tightened FAIL"* — and 6 cm is a near-zero-slack pass.

🔴 **And this one cannot be tuned away.** To open the `pillars` outer channel to 0.34 m of
half-width you would need `margin ≈ 0.05 m`, i.e. to discard the 0.31 m rotor reach — which is
physically real (rotor centres at ±0.14/±0.18 plus 0.13 m rotor radius). **A drone with a 0.31 m
body and 0.34 m of tracking error does not fit between pillars 1.2 m apart on a 3.0 m field.**
The scene is under-dimensioned for this policy, not mis-implemented.

## II.4 What I checked and cleared — the negative results

Recorded so nobody re-derives them:

| # | suspicion | verdict |
|---|---|---|
| 1 | Batch initial-state pinning uses `trajectory_reshaped[0]` for **all** batch elements (`projection.py:151`, `:282`) | 🟢 **Not a bug.** All `mpc_batch` candidates are sampled from the same current observation, so they share the first transition by construction. |
| 2 | Planning margin `0.33` vs scoring margin `0.31` | 🟢 **Deliberate**, documented: scoring is *"physical collision truth — NOT the planning margin"*. |
| 3 | Constraint set infeasible at build time | 🟢 **No** — the Fix_12 gate passes on both scenes for every homotopy. (But see §II.3 for what that does and does not mean.) |
| 4 | 3-D obstacles need new solver code | 🟢 **No** — `ObstacleConstraints` is already N-dimensional; only 3-D *halfspaces* are unimplemented. |
| 5 | Circuit breaker silently degrading `s_curve` projected variants to `diffuser` | 🟢 **Ruled out** — `projection_cb_tripped = 0.00` on every `s_curve` variant. |
| 6 | `cost_dims` lets you down-weight a channel in the objective | 🔴 **It does not.** `costs[idx] = 1` sets the weight to the value it already had (`projection.py:68-72`) — the branch is a no-op and `Q` is the identity either way. Harmless today, but the knob a degenerate dimension would want does not actually exist. |

## II.5 What follows

Ordered by how much it changes the conclusions of this project.

1. 🔴 **Re-dimension the scenes, or state the limit.** `pillars` as configured cannot be solved
   on the constraint axis by any policy in this repo. Either widen the pillar spacing / field in
   the XML and regenerate the expert data, or keep the scene and **report it as a diagnostic
   scene with a stated geometric infeasibility**, never as a benchmark row. Publishing an engine
   ranking off a scene where every engine scores 0/2876 would be reporting solver noise.
2. 🔴 **Add a policy-aware feasibility gate**, the UAV analogue of Gen14's D1: compare each
   scene's channel half-width against the measured `track_err` and refuse-or-flag when slack <
   error. Cheap, and it would have caught this months ago.
3. 🟡 **Log NLP non-convergence.** `self.last_solve_success` is already populated and read only
   by `HardFlowNLP` (§6.2.3). Surfacing it per variant turns "the projector fights the sample"
   from an inference into a measurement.
4. ⚪ **Try `dimensions: ['x','y','z']` on `pillars`** — a one-line config A/B mirroring Gen14's
   `obstacle_only_1` vs `_2`. It will *not* help here (the pillars are full-height, so the 3-D
   sphere is strictly *looser* than the true geometry and the reduction was already exact), but
   it closes the "did we ever check?" question at zero cost, and it is the right shape for any
   future scene with finite-height obstacles.

