# Investigation — `dpcc-c` "crushed to a point" at K=2 (Gen3v6 U3, mf_dit)

**Date:** 2026-07-30 (updated with the K-sweep the same day) · **Scope:** `dpcc-c` /
`dpcc-c-tightened`, job 23981 (K=2, mpc=4, seed 6) plus the confirmation sweep jobs
**24021 (K=1) / 24022 (K=5) / 24023 (K=20)**, all mpc=4, seed 6, git `bed63b3`.
Follow-up to [`INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md`](INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md).
**Trigger:** visual inspection of `dpcc-c.png` showed the agent barely leaving its start point over
the full 200-step episode, in all 3 scenarios, while every other variant traj looked normal.

> **Reading order note.** §§1–4b were written from the K=2 run alone. §8 is the K-sweep that
> **confirmed the localization but falsified the proposed mechanism** in §3c. Where the two
> disagree, **§8 is authoritative** — §3c is kept, marked, as the record of a hypothesis that the
> data killed.

## Verdict (final — after the K-sweep)

**Not file corruption, not a K-sweep collision, not a hardflow-port bug, and not "the selection rule
is naive."** There is a real, measurable defect, and the sweep pins it exactly:

**The mf_dit MeanFlow checkpoint has a degenerate "stay put" generation mode that exists at K=2 and
*only* at K=2.** At K=2 it captures **28.1%** of Gaussian noise draws (449/1600 candidates); at
K=1, K=5 and K=20 it captures **exactly zero** (0/788, 0/432, 0/608). Not "rarer at other K" —
**absent**. Same checkpoint, same code, same seed, same selection rule; only the step count changes,
and `dpcc-c` goes from 0.0 success at K=2 to **1.0 on all three scenarios** at K=1, K=5 and K=20 (§8).

The collapsed output is not noise and not a dead field: it is a **coherent, valid-looking plan that
sits motionless on the robot's current position** — all 8 horizon waypoints within 1e-4 of each other
and of `obs_all` (§8b). `-c` then locks onto it every replan because a plan that never approaches a
constraint is, correctly by the metric's own definition, the cheapest one to leave alone. `-r` and
`-t` don't seek that property, so they survive the same broken field — which is exactly why the
failure looked like a `-c` bug and is not one.

**Corrected mechanism.** The earlier hypothesis (§3c) was "large h is undertrained, so bigger `h`
per step ⇒ worse." **The sweep falsified it:** K=1 uses `h = 1.0`, the largest interval possible, and
is completely clean (0% collapse, median plan span 0.085). The defect is not monotone in `h` — it is
localized to the specific `(r, t)` coordinates the K=2 sampler visits, `(r=0, t=0.5)` and
`(r=0.5, t=1.0)`, i.e. the **interior/midpoint** of the two-time field. K=1 only ever queries the
`(r=0, t=1)` corner — the canonical one-step MeanFlow target — and K=5/K=20 query short intervals
that damp any single bad `u`. K=2 is the one setting that stakes half the trajectory on a single
query at the field's weakest interior point. Pinning this below the trajectory level needs a direct
`u(x, r, t)` probe on the cluster (§6).

## 1. Ruled out first: is the data even real?

- `dpcc-c.npz` and `dpcc-c-tightened.npz` are **distinct files** (different md5 per scenario) — not an
  overwrite/collision artifact.
- Both files' `obs_all` arrays are **byte-identical to each other**, and **byte-identical across all
  3 halfspace scenarios** (`top-right-hard`, `top-left-hard`, `both-hard`) for the same trial index.
  This looks alarming in isolation but turns out to be the expected *signature* of the bug (§3), not
  evidence of a copy/write error — see below.
- The eval log for this job ends cleanly (`Evaluation completed successfully.`), and all 39 expected
  npz files are present (13 variants × 3 scenarios) — this run is not truncated.

## 1b. Direct confirmation from the raw realtime logs (no decoding involved)

Before trusting the npz-decode analysis below, this was cross-checked against the plain-text
`realtime_dpcc-c_trial0.log` the eval script prints live, step by step:

- `ACT (0.000,0.000)` or `(-0.000,0.000)` on **198 of 200 steps** — the executed action is a literal
  zero almost the entire episode.
- `realtime_dpcc-r_trial0.log` (same checkpoint/scenario/K): healthy varied actions (~0.007–0.012
  magnitude), clearly making progress.
- `diff` between `realtime_dpcc-c_trial0.log`'s and `realtime_dpcc-c-tightened_trial0.log`'s `OBS`
  lines: **zero differences** — confirms the tightened/untightened rollouts are identical, consistent
  with §3's mechanism (a candidate that never nears a constraint is unaffected by tightening it).
- Directory sanity check: single `results/` tree, all 13×3 npz/png written in one batch (`09:12`),
  no stale/duplicate files, no leftover pre-K-collision-fix artifacts.

## 2. What the PNG shows

`dpcc-c.png` (`both-hard`, both trials): `x` oscillates in a **[0.524, 0.527] band** and `y` in
**[-0.282, -0.277]** for the entire 200-step episode — a decaying oscillation around the start pose
that never breaks out toward the goal (green dot, off the bottom of frame) or through the gap in the
obstacle row (top of frame, y≈0.35). Compare `dpcc-r.png` (same checkpoint, same K, same scenario):
`x` sweeps 0.35→0.61, `y` sweeps -0.29→0.35, reaching the goal in ~65 steps. **The checkpoint's field
is fine** — `dpcc-r` and `dpcc-t` both traverse the full course. Only `-c` gets stuck.

## 3. Why: the candidate fan, decoded from `sampled_trajectories_all`

Each replan's raw 4-candidate fan is stored per step. Net horizon displacement (`‖end − start‖` over
the planned 8-step horizon) per candidate, at a few representative replans (`both-hard`, trial 0):

| replan step | c0 | c1 | c2 | c3 | argmin(disp) | realized net step |
|---|---|---|---|---|---|---|
| 0   | 0.0504 | 0.0271 | **0.0139** | 0.0744 | c2 | ~0 |
| 1   | 0.0396 | 0.0257 | **0.0155** | 0.0363 | c2 | ~0 |
| 50  | 0.0633 | 0.0583 | **0.0001** | 0.0102 | c2 | ~0 |
| 100 | 0.0320 | 0.0784 | **0.0001** | 0.0383 | c2 | ~0 |
| 150 | 0.0412 | **0.0001** | 0.0000 | 0.0000 | c2/c3 | ~0 |
| 198 | 0.0405 | **0.0001** | 0.0321 | 0.0274 | c1 | ~0 |

Full-episode scan (both trials, `both-hard`):

- **72% of the 199 replans** have at least one candidate with net horizon displacement **< 0.005**
  (mean of the per-step minimum = 0.0071, vs. a corridor that needs ~0.6 units of net travel).
- **Total net travel over the entire 200-step episode: 0.0018–0.0027 units** — i.e. the agent
  essentially never leaves its start cell.

## 3b. Per-candidate slot analysis — ruling out an indexing bug, finding the real one

First checked whether one specific fan slot (e.g. "candidate index 2") is structurally broken — that
would point at an indexing/broadcast bug. It is not:

| trial | mean disp c0/c1/c2/c3 | argmin-frequency c0/c1/c2/c3 (of 200 replans) |
|---|---|---|
| 0 | 0.031 / 0.031 / 0.031 / 0.030 | 49 / 44 / 54 / 53 (24.5% / 22% / 27% / 26.5%) |
| 1 | 0.029 / 0.029 / 0.029 / 0.029 | 48 / 52 / 48 / 52 (24% / 26% / 24% / 26%) |

All 4 slots have near-identical mean displacement and near-uniform argmin frequency (~25% each, as
expected for 4 symmetric slots) — **no positional/indexing bias**. The bug is not "slot N is broken."

**What the full displacement histogram shows instead** (pooling every individual candidate across
both trials, all 200 replans, 1600 samples total — not just the selected minimum):

```
[0.000, 0.001): 449 samples (28.1%)   <-- essentially exact zero
[0.001, 0.005):   3 samples ( 0.2%)   <-- near-empty gap
[0.005, 0.010):  20 samples ( 1.2%)
[0.010, 0.020): 105 samples ( 6.6%)
[0.020, 0.030): 192 samples (12.0%)
[0.030, 0.040): 241 samples (15.1%)
[0.040, 0.050): 241 samples (15.1%)
[0.050, 0.060): 159 samples ( 9.9%)
[0.060, 0.080): 163 samples (10.2%)
[0.080, 0.100):  27 samples ( 1.7%)
```

**This is bimodal, not a smooth tail.** 28.1% of every individual candidate ever sampled — not just
the ones `-c` picks — lands in a machine-precision-zero cluster, then there is a near-empty gap
(0.001–0.01: only 1.4% of samples), then a separate, healthy, continuously-distributed cluster from
0.01–0.09 (the remaining ~72%). This is the signature of **two distinct output regimes**, not
ordinary sampling noise around one mean. With 4 i.i.d. draws per replan at a 28% per-draw collapse
rate, the chance at least one of the 4 collapses is `1 − 0.72⁴ ≈ 73%` — matching the observed 72%
`argmin`-lock rate in §3 almost exactly. **This is the real bug: the checkpoint itself has an ~28%
chance, per K=2 sample, of emitting a near-zero-net field**, and `-c`'s cost metric then reliably
finds and selects it.

## 3c. ~~Why K=2 specifically triggers it — tied to a documented training weakness~~ (FALSIFIED by §8)

> ⚠️ **This section's hypothesis is wrong and is retained only as a record.** It predicted the
> collapse rate scales with `h`, and specifically that **K=1 (h=1.0) might be *worse* than K=2**.
> The sweep (§8) measured **0.0% collapse at K=1** vs 28.1% at K=2. Bigger `h` is not the driver.
> The `h_mse_b*` validation-noise correlation below is real but does **not** explain the K=2
> singularity. See the corrected mechanism in the Verdict and §8c.

At K=2, `dt = 0.5`, so **both** Euler macro-steps query the model at `h = 0.5` — the *largest*
interval this network is ever asked to average a velocity over. This is not a new hypothesis invented
for this report: the **U2 training insight for this exact checkpoint** already flagged, independently,
that its large-h validation buckets are its least reliable region —
[`INSIGHT_Gen3v6_U2_mf_dit_first_run.md`](../U2/INSIGHT_Gen3v6_U2_mf_dit_first_run.md) lines 53-57:
> "Train/val gap ... `per_dim_rms_u` train 0.195 vs val 0.447. The large-h validation buckets are very
> noisy (`h_mse_b1/b2/b3` val = 39 / 40 / 20, with single-batch maxima in the 10³–10⁴ range)"

K=2 forces every single query into exactly this documented weak spot — the largest-h bucket, with the
worst, noisiest validation error of any h-range this checkpoint has. The ~28% exact-zero collapse rate
measured directly above is consistent with that noise: a network whose large-h output is this
poorly-calibrated would be expected to occasionally saturate/cancel to ~0 for a meaningful fraction of
noise draws, rather than degrade gracefully. **This is a real defect in the mf_dit MeanFlow
checkpoint's large-h behavior, most exposed at K=2** — not an artifact of the selection code, the
projector math, or the eval driver (all independently audited clean in §4b).

**How the projector's cost formula turns "collapsed" into "selected":** it only snaps/corrects at the
*last* denoising step when `diffusion_timestep_threshold=0.5` and `K=2`
(`snapping_start_idx = int((1-0.5)*2) = 1`), so `infos['projection_costs']` reflects exactly **one**
correction magnitude per candidate. A collapsed candidate never comes near a constraint, so its
correction cost is ≈0 — the global minimum by construction, regardless of where the obstacles
actually are — and `argmin` picks it. This explains the byte-identical across-scenario trajectories
in §1: the winning candidate never approaches any obstacle, so it literally cannot tell the 3
scenarios apart. But this part of the mechanism was already true and correct-by-design in Gen12 too
(§4) — it only becomes catastrophic here because §3b/§3c's ~28% collapse rate hands it a degenerate
candidate to pick on 72% of replans, which Gen12's checkpoint apparently never does.

## 4. Why doesn't this happen elsewhere? (isolating the cause to the mf_dit checkpoint, not the code)

The selection/projection code (`policies.py`, `mf_diffusion.py`'s late-snap logic) is unchanged from
Gen12's, and threshold=0.5 is the shared default everywhere. Yet:

- **Gen12 baseline (FMv3ODE, non-MeanFlow), same K=2, same threshold, same selection code:**
  `dpcc-c-tightened` scored **100%** goal+constraint success
  ([`RESULTS_Gen12_Ksweep_lowK.md`](../../Gen12/fix_3/RESULTS_Gen12_Ksweep_lowK.md) §1).
- **Gen3v6 U2, iMF-DiT backbone (`imf_backbone='dit'`), same K=2, same selection code:**
  `dpcc-c-tightened` was "the star" — `g1/b1/v0` on all 3 scenarios
  ([`INSIGHT_Gen3v6_U2_mf_dit_first_run.md`](../U2/INSIGHT_Gen3v6_U2_mf_dit_first_run.md) line 90).
- **Gen3v6 U2/U3, mf_dit backbone, same K=2, same selection code:** `-c` collapses to 0% in both runs.

Holding the shared selection/projection code fixed and varying only the backbone flips the outcome
from 100% to 0% — this pins the cause on **mf_dit's K=2 sampling variance** (it produces
near-zero-net-displacement "round trip" candidates far more readily than the UNet/iMF-DiT backbones
do at the same K), not on anything introduced by the U3 HardFlow port or the eval driver.

## 4b. Code/math audit — is the `-c` logic itself wrong?

Read the actual implementation line-by-line (not just the selection wrapper) to rule out a genuine
defect, and re-checked all 3 scenarios' PNGs for `-c-tightened` (previously only `both-hard` had been
visually confirmed) since the user's initial read was "tightened is fine." **Correction: it is not
fine — `top-right-hard` and `top-left-hard`'s `dpcc-c-tightened.png` show the identical frozen
oscillation-near-start pattern as `both-hard`.** All three scenarios, both trials, both tightened and
untightened, are frozen. There is no scenario where tightened actually escapes the start.

**`projection.py::project()` cost formula (`flow_matcher_v3_meanflow/sampling/projection.py:88-145`):**
```python
r = -trajectory_reshaped @ Q                      # r = -Q·traj
...
cost_fun = lambda x: 0.5*x@Q@x + r_np_double[i]@x   # solver objective
...
projection_costs[i] = 0.5*sol@Q@sol + r_np[i]@sol + 0.5*traj@Q@traj
```
Substituting `r = -Q·traj`, both reduce algebraically to:
- solver objective ≡ `0.5·(x − traj)ᵀQ(x − traj)` (up to an additive constant — same argmin)
- reported cost ≡ `0.5·(sol − traj)ᵀQ(sol − traj)` **exactly**

This is precisely the correct, intended quantity: the squared distance (in the Q-metric) between the
projector's output and the raw candidate. **No sign error, no missing term, no batch/index
cross-talk.** The formula does exactly what "minimum projection cost" is documented to mean.

**Constraint swap for `-tightened` (`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:297-304`):**
```python
if 'model_free' in variant and 'tightened' in variant: constraints = constraint_list_without_prior_tightened
elif 'model_free' in variant and not 'tightened' in variant: constraints = constraint_list_without_prior
elif not 'model_free' in variant and 'tightened' in variant: constraints = constraint_list_tightened
else: constraints = constraint_list
```
`dpcc-c-tightened` genuinely receives `constraint_list_tightened` (the +0.025-margin halfspace/obstacle
set), not a reused/cached untightened list — verified by reading the branch directly. So the tightened
run is not silently skipping its own constraints; it really is evaluated under the tighter margin.

**Why tightened and untightened are then still byte-identical — also not a bug:** `torch.manual_seed(i)`
is called once per trial index `i`, identically for every variant
(`eval_flow_matching_v3_meanflow.py:372`), so `dpcc-c` and `dpcc-c-tightened` draw the exact same initial
noise for the same trial — intentional, for apples-to-apples reproducibility across variants. Given the
winning candidate's raw (pre-projection) trajectory never gets near ANY constraint (§3: net displacement
≈0), it is **already feasible under both the loose and the tight constraint set**, so the SLSQP solve is
a no-op in both cases (`sol == traj`, `res.x == x0`) and `projection_costs ≈ 0` either way. Tightening a
margin the trajectory never approaches cannot change anything downstream — the identical output is the
mathematically necessary consequence of the selection collapse, not evidence of a caching/reuse defect.

**Conclusion of the audit: no code bug, no math bug.** Every formula and every branch does exactly what
it is designed and documented to do. The failure is a **specification gap**: `-c`'s definition of
"quality" (least correction needed) silently assumes every candidate is at least attempting to make
progress, which is false at K=2 for mf_dit, where ~72% of replans include a candidate that isn't
attempting progress at all.

## 5. Is this a "bug"? (final)

**Yes — in the model/checkpoint, not in the selection or projection code.** The selection code
(`-c`'s argmin-over-cost) and the projector's cost formula are both correctly implemented and
match Gen12 verbatim (§4b) — that part is clean. The defect is **a degenerate "stay put" mode in the
mf_dit checkpoint's two-time field that exists only at the `(r,t)` coordinates the K=2 sampler
visits**, capturing 28.1% of noise draws there and 0% at every other K tested (§8). `-c` doesn't
cause the defect — it just has no way to avoid being fooled by it, because a motionless candidate is
(correctly, by the metric's own definition) the cheapest one to leave alone.

So: **real bug, located in the mf_dit/MeanFlow checkpoint's sampled field at K=2, not in the
DPCC-family selection/projection code** — which is shared, unmodified, and now proven fine three
independent ways: the Gen12 and iMF-DiT comparisons (§4), the line-by-line math audit (§4b), and the
K-sweep, where the *identical* `-c` code on the *identical* checkpoint scores 1.0 at K=1/5/20 (§8).

## 6. What's lacking / recommended follow-ups

1. ✅ **K-sweep — done, see §8.** It confirmed the localization and falsified §3c's mechanism.
2. **Probe the field directly (needs cluster/GPU).** The remaining open question is *why* `(r=0,
   t=0.5)` / `(r=0.5, t=1.0)` host a dead mode when `(r=0, t=1)` does not. The direct test: sweep a
   grid of `(r, t)` pairs, push N Gaussian draws through a single `u(x, r, t)` call each, and map the
   fraction of near-zero outputs. That turns the trajectory-level inference here into a measured
   property of the weights, and would show whether the dead region is a point, a band, or the whole
   interior. **This is the highest-value next check.**
3. **Only 1 seed / 2 trials per K.** The 28.1%/0% split is measured over 1600 (K=2) and 788/432/608
   (K=1/5/20) individual candidate samples — a large sample per configuration, but a single seed.
   Seeds 7-10 (user running manually) would confirm the K=2 singularity isn't seed-specific.
4. **Training-side fix, not eval-side.** Since the hole is in the field's interior, the remedy is
   better `(r,t)` interior coverage during MeanFlow training for `mf_dit` — e.g. checking what
   density the `logit_normal` `t_schedule` (`p_mean`/`p_std`) plus the `r`-sampling actually puts on
   mid-interval pairs, and whether the JVP target is well-conditioned there. Not a patch to `-c`'s
   selection formula, which is doing its documented job correctly on a field that has a real hole.
5. **Practical guidance meanwhile: do not run mf_dit at K=2.** K=5 and K=20 are the strong settings —
   `dpcc-c-tightened` is 1.0 goal-and-constraints on **all three** scenarios at both (§8a).
6. **The `hardflow_new-*` arm has the mirror-image defect — now investigated separately.**
   `hardflow_new-c` is clean at K=1/K=2 but freezes 200/200 at K=5 and K=20, with ~29–31% of its
   candidates collapsed where DPCC's fan from the *same checkpoint, seed and noise* has 0%. That is a
   **port** bug — ⚠️ **but see fix_4:** arm C was running at σ=0.5 against a σ=1.0 checkpoint, so
   "the in-loop NLP manufactures the motionless trajectories" is observed, not established, and the
   HF numbers need a re-run
   ([`../fix_4/CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md`](../fix_4/CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md)).
   Nothing in *this* document is affected — arms A/B never enter that code path. Written up in
   [`INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md`](INSIGHT_Gen3v6_U3_hardflow_first_run_K2.md)
   §"Two distinct `-c` collapses". The two failures share a symptom and nothing else.

## 8. K-sweep — the decisive test (jobs 24021 / 24022 / 24023)

Jobs `24021` (K=1), `24022` (K=5), `24023` (K=20), all `HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5`, seed 6,
git `bed63b3`, exported to `temp/2026-07-30/`. The `K2` folder in that same export is **byte-identical**
to job 23981's (`md5 3282c155…` on `dpcc-c.npz`) — it is the original run re-downloaded, not a re-run,
so K=2 here is the same data analyzed in §§1–4b.

### 8a. Control outcome — `dpcc-c` is fine at every K except 2

`dpcc-c` / `dpcc-c-tightened` success rate (2 trials/cell), and the fraction of executed actions that
are a literal zero (`ACT (±0.000,±0.000)`, from the plain-text realtime logs):

| K | h/step | `-c` succ (TR / TL / BH) | `-c-tightened` succ (TR / TL / BH) | frozen ACT steps, `-c`, `both-hard` |
|---|---|---|---|---|
| 1  | 1.00 | **1.0 / 1.0 / 1.0** | 0.0 / 1.0 / 1.0 | 2/83, 0/114 (~1%) |
| 2  | 0.50 | **0.0 / 0.0 / 0.0** | 0.0 / 0.0 / 0.0 | **198/200, 200/200 (~99%)** |
| 5  | 0.20 | **1.0 / 1.0 / 1.0** | **1.0 / 1.0 / 1.0** | 0/53, 0/55 (0%) |
| 20 | 0.05 | **1.0 / 1.0 / 1.0** | **1.0 / 1.0 / 1.0** | 1/59, 0/93 (~1%) |

*(TR = top-right-hard, TL = top-left-hard, BH = both-hard.)*

On goal **and** constraints, `dpcc-c-tightened` is **1.0 on all three scenarios at both K=5 and
K=20** with 0.00 violations — the best-performing arm anywhere in the sweep. The one non-K=2 zero,
`K=1 / top-right-hard / -c-tightened`, is **a different failure**: its realtime log shows the robot
travelling normally (`(0.525,-0.280) → (0.615,0.226)` over 76 steps, only 2 frozen actions) and simply
not reaching the goal in time. An ordinary miss, not the freeze.

### 8b. Candidate-fan collapse rate — the switch, measured

Same metric as §3b (net planned horizon displacement `‖obs[-1] − obs[0]‖` per candidate), pooled over
every candidate of every replan of both trials, `both-hard`:

| K | h/step | candidates | collapsed (`< 1e-3`) | median span | replans with ≥1 collapsed | selected candidate was collapsed |
|---|---|---|---|---|---|---|
| 1  | 1.00 | 788  | **0.0%** (0/788) | 0.0850 | 0.0% | 0.0% |
| 2  | 0.50 | 1600 | **28.1%** (449/1600) | 0.0316 | **71.5%** | **71.5%** |
| 5  | 0.20 | 432  | **0.0%** (0/432) | 0.0864 | 0.0% | 0.0% |
| 20 | 0.05 | 608  | **0.0%** (0/608) | 0.0842 | 0.0% | 0.0% |

Two things stand out:

- **Zero, not "fewer."** K=1, K=5 and K=20 produce not a single collapsed candidate in 1828 samples.
  A graded effect would have left a tail; there is none. This is a switch that is on at K=2 alone.
- **K=2 is globally attenuated too, not just bimodal.** Its *healthy* cluster has median span 0.0316
  vs ≈0.085 at every other K — even the non-collapsed K=2 candidates plan at roughly a third of the
  normal speed. And K=1 puts 15.4% of its candidates above 0.10 span, a bucket K=2 never reaches at
  all. The whole K=2 field is damped, with 28% of it pinned to zero.

**The collapse is i.i.d. in the noise draw, not state-driven.** Per-replan counts of how many of the
4 candidates collapsed: `[114, 155, 101, 28, 2]` for 0/1/2/3/4. If each draw collapses independently
with probability `p`, then `P(none of 4) = (1−p)⁴ = 114/400 = 0.285 ⇒ p = 0.27`, matching the measured
28.1% marginal rate. So it is a fixed-size basin in the noise → trajectory map, hit at random —
**not** a property of where the robot happens to be.

That also closes the last gap between §3's numbers and the realtime logs: `-c` locks onto a collapsed
candidate on 71.5% of replans, and on the other 28.5% it still takes the `argmin` of 4 healthy-but-
attenuated spans — the slowest available plan. Net executed motion over 200 steps: **0.002 units**.

### 8c. What the collapsed candidate actually is

Decoding one (trial 0, replan 12, candidate 1) against the same replan's healthy candidate 0:

```
COLLAPSED cand1, planned obs over the 8-step horizon:   HEALTHY cand0, same replan:
  [ 0.52476 -0.28123]                                     [ 0.52476 -0.28123]
  [ 0.52475 -0.28123]                                     [ 0.52449 -0.28075]
  [ 0.52471 -0.28123]                                     [ 0.52451 -0.27841]
  [ 0.52470 -0.28123]                                     [ 0.52501 -0.27216]
  [ 0.52470 -0.28123]                                     [ 0.52602 -0.26440]
  [ 0.52467 -0.28123]                                     [ 0.52731 -0.25612]
  [ 0.52467 -0.28123]                                     [ 0.52895 -0.24757]
  [ 0.52468 -0.28123]                                     [ 0.53100 -0.23903]
robot's actual position at that step (obs_all): (0.52461, -0.27879)
```

This rules out the two obvious "dead network" stories:

- **It is not `u ≈ 0`.** If the velocity field went to zero, `x` would remain the initial Gaussian
  draw — a wild scribble spanning ~±1 in normalized space, i.e. a *huge* span, not a zero one.
- **It is not garbage.** All 8 waypoints agree to ~1e-4 and sit on the robot's current pose, and the
  action rows are equally constant.

The network is actively and coherently generating **"stay exactly where you are"** — a well-formed,
perfectly feasible, entirely useless plan. That is mode collapse onto a degenerate fixed point of the
field, which is also why it is invisible to every constraint-based check: a stationary plan violates
nothing, so the projector leaves it untouched (`sol == traj`, cost ≈ 0) and `-c`'s `argmin` takes it.

## 7. Addendum — re-checking the "tightened works, untightened doesn't" premise

The investigation started from the user's initial read that `dpcc-c-tightened` looked fine and only
plain `dpcc-c` was broken. That premise does **not** hold in this job's data — re-verified directly
against the plain-text `eval_*.log` summaries (not a derived/decoded number) for all 3 scenarios:

```
top-right-hard  dpcc-c:            Success rate: 0.0
top-right-hard  dpcc-c-tightened:  Success rate: 0.0
top-left-hard   dpcc-c:            Success rate: 0.0
top-left-hard   dpcc-c-tightened:  Success rate: 0.0
both-hard       dpcc-c:            Success rate: 0.0
both-hard       dpcc-c-tightened:  Success rate: 0.0
```

Every one of the 6 logs reads `0.0`. Combined with the earlier `diff`-confirmed byte-identical
realtime action logs (§1b), there is no scenario in `job 23981` where `dpcc-c-tightened` succeeds
while plain `dpcc-c` fails — both are frozen, identically, everywhere. If a "tightened works" result
exists, it is not in this run; it would have to come from a different job/seed/K (e.g. Gen12's own
`dpcc-c-tightened`, a *different checkpoint*, which legitimately scored 100% at K=2 — see §4 — and is
easy to conflate with this Gen3v6 run since the variant name is identical) or from visually confusing
`dpcc-c-tightened.png` with a neighboring file like `dpcc-t-tightened.png`/`dpcc-r-tightened.png` in
the same folder (which genuinely do succeed in this run). Flagging this rather than resolving it,
since it could not be reproduced from any file found under `.../6/results/halfspace_*/` for this job.

**Update after the K-sweep (§8): the premise is right about the checkpoint, wrong about K=2.** The
sweep shows `dpcc-c-tightened` is not merely "fine" but the **single best arm in the whole sweep** —
1.0 goal-and-constraints on all three scenarios, 0.00 violations, at both K=5 and K=20. So the
instinct that `-c`/`-c-tightened` works on this checkpoint was correct; it just does not hold at K=2,
where the two move together and both read 0.0. Across all four K values there is **no configuration
where `-c-tightened` succeeds while plain `-c` freezes** — they are never separable, exactly as §4b's
argument predicts (a motionless candidate is already feasible under both constraint sets, so
tightening a margin it never approaches cannot change anything). The likeliest source of the original
"tightened is fine" read is therefore a K=5/K=20-style result, or the Gen12 `dpcc-c-tightened` 100%,
rather than anything inside job 23981.

## How this was checked

No cluster/Python packages available in this container for `.npz`/numpy natively; used a throwaway
venv (`pip install numpy`, scratchpad-only, not part of the repo) purely to decode the existing
`.npz` result files already produced on the cluster — no training/eval was run locally, consistent
with "debug/analysis only" tooling use.

Every headline number has a decode-free cross-check: the success rates come from the plain-text
`eval_*.log` summaries, and the frozen-action counts from the plain-text `realtime_*_trial*.log`
files, both read directly with `grep`. The npz decode is used only for the candidate-fan statistics
(§3b, §8b, §8c), which have no plain-text equivalent. Sources:
`temp/2907/` (job 23981, K=2) and `temp/2026-07-30/` (jobs 24021/24022/24023, K=1/5/20).
