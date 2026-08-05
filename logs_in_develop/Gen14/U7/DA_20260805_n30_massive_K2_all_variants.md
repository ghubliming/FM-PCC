# DA — Gen14 massive K=2 eval, n=30 contexts, all 19 variants × 2 geometries × 2 arms

**Date:** 2026-08-05
**Data:** `temp/0508/` — jobs **24281** (mf) and **24282** (af), i6-gpu-1, git rev `205c494`
**Scope:** the first Gen14 run with enough contexts to say anything statistically. 2280 rollouts.
**Figures:** `figs/fig1_distance_null.png`, `figs/fig2_pareto.png`, `figs/fig3_time.png`
**Script:** `da_20260805_n30_massive.py` (scratchpad venv — this container has no project env)

---

## 0. TL;DR

1. **The folder is 3.0 GB and the gifs are not the reason.** The 776 MB of gifs on disk are
   *stale* — left over from an older 10-context run. This run wrote **zero** gifs; `--record none`
   worked. The 2.2 GB it *did* write is per-rollout `_report.png` (671 MB) + `_mpc_foresight.svg`
   (748 MB) + `.npz` (494 MB) + realtime logs (243 MB), **none of which `--record` controls**.
   See §1.
2. **On final distance, nothing beats anything.** With `n=30` the run-to-run noise floor is
   **±0.135 m (mf) / ±0.068 m (af)** on a 30-rollout mean. The *entire* spread across all 19
   variants is 0.10–0.18 m. After correcting for 18 comparisons, **no variant is distinguishable
   from the unprojected baseline in any of the four cells.** See §3–§4.
3. **On constraints the signal is enormous and unambiguous.** Under `combined_5-tightened`,
   `dpcc-t` reaches **0.0 violated steps, 30/30 clean rollouts** against the unprojected 63.5.
   That is 200× the noise band. Projection works; it just doesn't buy you goal accuracy. See §5.
4. **HardFlow (U7) buys nothing.** It costs **145–194 ms/replan (3.5× DPCC, 6.5× unprojected)**
   and in every one of the four cells there is a ≤50 ms variant that matches or beats it on both
   axes. **This overturns the n=3 conclusion in the previous HardFlow DA** — that DA reported
   HardFlow as the best nominal-constraint cell (0.998 sat / 0.7 violated steps); at n=30 the same
   configuration gives 0.900 sat / 40.2 violated steps. It was noise. See §6.
5. **`bounds_free` is the surprise winner.** Removing the bounds rows from the projection NLP
   makes bounds violations *go down* (mf/tgt: 0.5 vs 10.9 for full `dpcc-c`), makes distance the
   best or near-best, and is faster. Consistent in all four cells. See §7.
6. **`af` is not better than `mf` anywhere, and `mf` is 2× noisier.** See §8.

---

## 1. Why the folder is still 3.0 GB

### 1.1 The gifs are stale — this run wrote none

```
category                        files        MB      %
gif (rollout video)               486     776.5   25.7   <-- STALE
svg (mpc foresight)              2128     747.6   24.7
png (per-rollout report)         2280     671.0   22.2
npz (bulk arrays)                 115     494.2   16.3
log (realtime per-rollout)       2280     243.4    8.1
png (summary grid)                161      74.7    2.5
log/json/other                   2603      15.3    0.5
TOTAL                           10053    3022.7
```

Three independent proofs the gifs are not from this run:

| evidence | value | implication |
|---|---|---|
| gifs per variant dir | **10** | this run did 30 rollouts |
| variant dirs holding gifs | **12 of 19** per geometry | the missing 7 are `dpcc-c-dt{0p25,0p5,2p0,4p0}` + the 3 `hardflow_new-*` — variants that **did not exist** when the gifs were written |
| gifs in the 6 HardFlow dirs | **0** | those dirs were created fresh by this run |

480 rollout gifs = 12 dirs × 2 geometries × 2 arms × 10, exactly. Plus 6 expert-reference gifs.
The eval log confirms the flag took effect: `[ eval ] Recording mode set to: none`.

**Reclaim 776 MB with:**
```bash
find /workspaces/FM-PCC/temp/0508 -name 'rollout_*.gif' -delete
```

### 1.2 `--record` only gates video — the real bulk is ungated

In `mix_visual_aligning_test/eval_mix_visual_aligning.py`, `_export_rollout_realtime()` gates
**only** the mp4/gif branch on `record_mode`:

- `:1460` `if self.record_mode != 'none' and self.video_frames:` → mp4 / gif
- `:1622` `fig.savefig(... f'rollout_{i}_report.png')` → **unconditional**
- `:1950` `fig_mpc.savefig(f'{_mpc_base}.svg', ...)` → **unconditional**
- `realtime_<variant>_rollout<i>.log` → **unconditional**, ~120 KB each

So per variant-cell at n=30, with `--record none`:

| item | per cell |
|---|---|
| `diagnostics/` (30 × report.png + 30 × foresight.svg) | **20 MB** |
| `<variant>.npz` | **6–11 MB** |
| 30 × `realtime_*.log` | 3.6 MB |
| summary png + json + txt | 0.8 MB |
| **total** | **~30–35 MB** |

76 cells × ~30 MB ≈ **2.2 GB**. That is the whole story. The growth vs. the earlier runs is
**not** recording — it is *n_contexts 3 → 30* (10×) plus 6 new HardFlow cells.

### 1.3 What to trim, ranked by payoff

| action | saves | cost |
|---|---|---|
| delete stale gifs (above) | 776 MB | none — they are wrong data |
| `_mpc_foresight` back to PNG | ~600 MB | the `.png` savefig at `:1949` is **commented out** in favour of `.svg`; svg averages 351 KB/file vs a dpi-200 png at ~80 KB |
| raise `mpc_foresight_stride` (yaml `:49`, currently 6) | scales ~linearly | fewer decision points drawn per plot |
| drop `sampled_trajectories_all` from the npz | ~450 MB | it is **90.3%** of every npz (9.89 of 10.96 MB); only `-c`/`-t` selection post-hoc needs it |
| gate `_report.png` behind a new `--diagnostics` level | ~670 MB | needs a code change; the report png is genuinely useful for debugging |

The first one is free. The rest are proposals — **no code changed for this DA.**

---

## 2. What actually ran

| | |
|---|---|
| arms | `mf` (Gen3v6 MeanFlow), `af` (Gen3v7 α-Flow) — locked U-Net + dual ResNet-18 |
| NFE | **K = 2** (`flow_steps_v3=2`, U6 default; log confirms `train=100 -> eval=2`) |
| projection threshold | **T = 0.5** → `snapping_start_idx = 1` → **1 projector call per replan** |
| MPC horizon | 4 |
| seed | 6 (single seed) |
| contexts | **30** (`n_contexts: 30` in the *cluster* yaml — see §9) |
| geometries | `combined_5`, `combined_5-tightened` |
| variants | 19 (12 legacy + 4 `dpcc-c-dt*` + 3 `hardflow_new-*`) |
| cells | 19 × 2 × 2 = **76**, all completed (`item 38/38`, "Job completed successfully" on both) |
| wall time | mf 15.0 h, af 15.2 h (summed variant elapsed) |
| per rollout | 31 s unprojected → 42 s DPCC → **95 s HardFlow** |

HardFlow selection rules genuinely dispatched this time — the log shows
`[ hardflow ] selection=minimum_projection_cost (from 'hardflow_new-c')` and
`selection=temporal_consistency`, confirming **U7 fix_1** (the suffix-stripping bug that silently
made `-c`/`-t` fall back to `random`) is fixed and exercised.

---

## 3. The noise floor — read this before any table below

`diffuser` is the unprojected variant. **It never touches the constraint set.** So its
`combined_5` and `combined_5-tightened` runs are the same policy, on the same 30 contexts, with
nothing different between them except the noise draw. That gives a free, direct measurement of
run-to-run noise:

| arm | rollouts that differ | max \|Δ\| | SD of the paired difference | 95% half-width on a 30-mean | Bonferroni (18 comparisons) |
|---|---|---|---|---|---|
| mf | **27 / 30** | 1.216 m | 0.376 m | **±0.135 m** | ±0.226 m |
| af | **26 / 30** | 0.480 m | 0.190 m | **±0.068 m** | ±0.114 m |

Now compare against the full spread of all 19 variant means:

| cell | spread (best→worst) | 95% floor | Bonferroni floor | verdict |
|---|---|---|---|---|
| mf / combined_5 | 0.179 m | 0.135 | 0.226 | **inside noise** |
| mf / tightened | 0.104 m | 0.135 | 0.226 | **inside noise** |
| af / combined_5 | 0.135 m | 0.068 | 0.114 | marginal |
| af / tightened | 0.138 m | 0.068 | 0.114 | marginal |

**The entire variant ordering on distance fits inside the noise band of one policy compared with
itself.** Not one variant against one other — the whole 19-wide spread.

Two consequences that matter more than any ranking in this document:

- **This overturns §12 of the U5 K=2 DA and the whole of the n=3 HardFlow DA.** Both ranked
  variants by distance. Those rankings were reading noise.
- **The generator is not deterministic across variants.** The U7 DA claimed "the generator is
  deterministic; scipy SLSQP is not," based on `diffuser` reproducing 3/3 at n=3. At n=30 it
  reproduces **3/30**. The likely mechanism is that the torch RNG stream is not reset per
  variant, so item 1 and item 20 of the loop draw different initial noise. That is a design
  property, not a bug — but it means **every cross-variant comparison is confounded with a
  different noise draw**, and it is cheap to remove (§10).

To resolve the observed best-vs-worst mf gap (0.179 m) at Bonferroni you would need **48 contexts**.
To resolve a genuine 0.05 m effect you would need **~613**.

---

## 4. Distance — the metric that matters

Per the standing instruction, success rate is not used here; success is 0.000–0.133 everywhere and
carries no information. Distance distributions are heavily right-skewed, so median and tail are
reported alongside the mean. *(fig1)*

**mf / combined_5** (n=30, sorted by median)

| variant | mean | med | p25 | p75 | p90 | max | >0.6 m | ms |
|---|---|---|---|---|---|---|---|---|
| hardflow_new-t | 0.307 | 0.281 | 0.237 | 0.368 | 0.526 | 0.613 | 2 | 174.9 |
| hardflow_new-r | 0.306 | 0.283 | 0.213 | 0.363 | 0.522 | 0.643 | 1 | 181.9 |
| gradient | 0.371 | 0.286 | 0.166 | 0.489 | 0.727 | 1.287 | 5 | 29.5 |
| dpcc-t | 0.287 | 0.288 | 0.146 | 0.403 | 0.510 | 0.687 | 1 | 52.8 |
| bounds_free | 0.295 | 0.309 | 0.174 | 0.383 | 0.487 | 0.717 | 1 | 48.7 |
| dpcc-c-dt0p5 | 0.306 | 0.305 | 0.235 | 0.392 | 0.476 | **0.561** | **0** | 59.2 |
| dpcc-r | 0.342 | 0.323 | 0.211 | 0.450 | 0.587 | 0.795 | 3 | 55.5 |
| dpcc-c-dt4p0 | 0.383 | 0.368 | 0.275 | 0.463 | 0.593 | 0.702 | 3 | 43.9 |
| **diffuser** | **0.466** | **0.440** | 0.180 | 0.616 | 0.887 | **1.477** | **10** | 27.9 |

**mf / combined_5-tightened**

| variant | mean | med | p90 | max | >0.6 m | ms |
|---|---|---|---|---|---|---|
| dpcc-c-dt0p25 | 0.287 | 0.245 | 0.536 | 0.768 | 3 | 47.0 |
| hardflow_new-r | 0.312 | 0.260 | 0.531 | 1.030 | 3 | 147.2 |
| diffuser | 0.345 | 0.263 | 0.561 | 1.012 | 3 | 23.7 |
| bounds_free | **0.284** | 0.282 | 0.519 | **0.560** | **0** | 37.5 |
| dpcc-t | 0.307 | 0.290 | 0.495 | 0.686 | 1 | 42.3 |
| dpcc-c-dt4p0 | 0.388 | 0.368 | 0.556 | 0.709 | 2 | 38.2 |

Full tables for both `af` cells are in fig1; the ordering there is equally unstable.

**What survives the noise floor:**

- Nothing on the mean or the median. Every "improvement" in the tables above is smaller than the
  ±0.135 / ±0.068 m band.
- **The one apparently-significant block is an artifact.** A paired bootstrap on mf/combined_5
  flags 11 of 18 variants as beating `diffuser` (e.g. `dpcc-t` d = −0.179 m, 95% CI
  [−0.307, −0.059]). But mf's `diffuser` drew 0.466 in `combined_5` and 0.345 in `tightened` —
  a 0.121 m swing from noise alone, the same magnitude as every "significant" effect. In
  mf/tightened, where `diffuser` drew the luckier 0.345, **nothing is significant**. The
  significance was one unlucky baseline draw, and the replicate proves it.
- **The tail is the only place a pattern is visible, and even that is weak.** `diffuser` in
  mf/combined_5 puts 10/30 rollouts past 0.6 m with a 1.477 m worst case; every projected variant
  cuts that to 0–3 with a worst case under 0.8 m. This is consistent with projection truncating
  catastrophic excursions rather than improving typical accuracy — but the same `diffuser` in
  mf/tightened only has 3/30 past 0.6 m, so the effect size is unmeasured. **Flagging as a
  hypothesis for the multi-seed run, not a result.**

---

## 5. Constraints — this is where the signal actually is

Unlike distance, the constraint axis separates far outside the noise band. The `diffuser`
replicate moves `viol` by 29.4 steps in mf and **0.2** steps in af; the observed range is 0–97.

**Under `combined_5-tightened`** (violated steps/rollout, lower is better):

| mf | viol | 0-viol | ms | | af | viol | 0-viol | ms |
|---|---|---|---|---|---|---|---|---|
| **dpcc-t** | **0.0** | **30/30** | 42.3 | | **dpcc-t** | **1.1** | 28/30 | 42.6 |
| hardflow_new-t | 0.3 | 29/30 | 145.6 | | hardflow_new-r | 3.1 | 27/30 | 154.4 |
| dpcc-c-dt4p0 | 0.4 | 29/30 | 38.2 | | bounds_free | 3.8 | 27/30 | 39.4 |
| bounds_free | 0.5 | 28/30 | 37.5 | | hardflow_new-c | 4.5 | 25/30 | 158.5 |
| hardflow_new-c | 2.2 | 26/30 | 148.3 | | hardflow_new-t | 4.9 | 26/30 | 156.4 |
| dpcc-r | 4.5 | 27/30 | 42.3 | | dpcc-c-dt4p0 | 5.3 | 21/30 | 39.8 |
| dpcc-c | 12.6 | 22/30 | 54.9 | | dpcc-c | 18.6 | 22/30 | 49.1 |
| *diffuser* | *63.5* | *11/30* | 23.7 | | *diffuser* | *81.4* | *6/30* | 23.3 |
| *geo_free* | *73.8* | *12/30* | 35.2 | | *geo_free* | *96.6* | *10/30* | 37.5 |

`dpcc-t` at 0.0 violated steps over 30 rollouts against an unprojected 63.5 is not a marginal
result — it is a 200× gap against a ±29 noise band.

**Under `combined_5` (nominal) nothing works well.** Best mf is `dpcc-c-dt4p0` at 22.0 violated
steps / 14 of 30 clean; best af is `bounds_free` at 27.4 / 18 of 30. The unprojected baselines are
92.8 and 81.5. So projection helps by ~3–4×, but the nominal constraint set is simply too tight for
K=2 with a single projector call per replan to close.

**The family breakdown says the geometry constraints do all the work.** Any variant with the
geometry block ablated (`geo_free`, `model_free`, `geo_free-model_free`, `geo_free-bounds_free`)
sits at the unprojected level in every cell:

| arm/geo | variant | bounds | halfspace | obstacles |
|---|---|---|---|---|
| mf/tgt | diffuser | 47.5 | 13.7 | 4.7 |
| mf/tgt | geo_free | 56.8 | 14.0 | 4.9 |
| mf/tgt | geo_free-bounds_free | 64.6 | 22.4 | 5.5 |
| mf/tgt | **dpcc-t** | **0.0** | **0.0** | **0.0** |
| mf/tgt | **bounds_free** | **0.5** | **0.0** | **0.0** |

---

## 6. HardFlow (U7) — works, but does not earn its cost

The port is functionally sound: all six HardFlow cells ran clean at n=30, all three selection rules
dispatched correctly, visual conditioning survived into the sampler, and the constraint numbers are
respectable (0.3–4.9 violated steps under tightened, 25–29 of 30 clean).

But:

| | HardFlow | best ≤50 ms alternative |
|---|---|---|
| mf/tgt viol | 0.3 (`-t`, 145.6 ms) | **0.0** (`dpcc-t`, 42.3 ms) |
| mf/nom viol | 40.2 (`-t`, 174.9 ms) | **22.0** (`dpcc-c-dt4p0`, 43.9 ms) |
| af/tgt viol | 3.1 (`-r`, 154.4 ms) | **1.1** (`dpcc-t`, 42.6 ms) |
| af/nom viol | 31.4 (`-r`, 177.5 ms) | **27.4** (`bounds_free`, 48.6 ms) |
| distance | inside the noise floor | inside the noise floor |
| cost | **145–194 ms** | 37–55 ms |

**In all four cells a DPCC-family variant matches or beats HardFlow on constraints at ~1/3.5 the
cost, and the distance axis cannot separate them.** HardFlow sits at ~5× the 33 ms 30 Hz budget;
the cheap alternatives sit at 1.1–1.7×. *(fig3 shows this as a clean right-hand cluster.)*

**Explicit correction to the previous HardFlow DA (n=3):**

| claim at n=3 | value at n=30 |
|---|---|
| "best nominal-constraint cell: 0.998 sat / 0.7 violated steps" | mf/nom `hardflow_new-t`: **0.900 sat / 40.2 violated steps** |
| "vs DPCC's best 0.932 / 27.0" | DPCC's best nominal is **0.945 / 22.0** — DPCC wins |
| ordering of the top 4 | completely reordered |

The n=3 caveat in that DA (spread across contexts was 6–9×, dropping one context moved a mean by
0.169 m) was correct and is now confirmed: the headline it carried did not survive.

There is one thing HardFlow does have — the tightest distance tails in mf/combined_5
(`hardflow_new-t/-r` max 0.613 / 0.643, only 1–2 rollouts past 0.6 m, vs `diffuser`'s 1.477 and
10). Whether that is real needs the multi-seed run.

---

## 7. The `bounds_free` paradox — worth a look

`bounds_free` removes the bounds rows from the projection NLP. The metric still counts bounds
violations. It should therefore be *worse* on bounds. It is dramatically **better**:

| cell | variant | bounds viol | halfspace | obstacles | total | dist | ms |
|---|---|---|---|---|---|---|---|
| mf/tgt | `bounds_free` | **0.5** | 0.0 | 0.0 | **0.5** | **0.284** | **37.5** |
| mf/tgt | `dpcc-r` | 2.8 | 3.9 | 0.0 | 6.7 | 0.314 | 42.3 |
| mf/tgt | `dpcc-c` | 10.9 | 2.6 | 0.0 | 13.5 | 0.363 | 54.9 |
| mf/nom | `bounds_free` | **11.5** | 28.0 | 1.0 | 40.5 | 0.295 | 48.7 |
| mf/nom | `dpcc-c` | 49.7 | 19.1 | 1.0 | 69.8 | 0.409 | 56.9 |
| af/nom | `bounds_free` | **18.2** | 7.8 | 1.5 | **27.6** | 0.331 | 48.6 |
| af/nom | `dpcc-c` | 57.5 | 40.2 | 1.2 | 99.0 | 0.350 | 55.4 |

`bounds_free` is on the Pareto frontier in **all four cells** — best-in-cell on distance in mf/tgt
(0.284 m), second in mf/nom, fourth in both af cells — at 12–32% less cost than full `dpcc-c`.
Against HardFlow it strictly dominates (better on distance,
violations *and* time) all three HardFlow variants in af/combined_5, two of three in each mf cell,
and none in af/tightened — where `hardflow_new-r` is genuinely ahead on both quality axes, at 4×
the cost.

Leading hypothesis: the bounds rows make the SLSQP problem harder without adding value — more
iterations, more failed/early-terminated solves, a worse projected trajectory returned. The
degradation then shows up on the bounds family itself, which is the paradoxical part. Note the
effect is **not** uniform across families: in af/combined_5 dropping the bounds rows improves
halfspace too (7.8 vs 40.2), but in mf/combined_5 it makes halfspace worse (28.0 vs 19.1) while
cutting bounds violations by 4×. So this is a solver-conditioning story, not "bounds constraints
are harmful" — the total is what improves, and it improves everywhere.

**This is not confirmed** — it needs a solver-exit-status histogram, which the current npz does not
record. Worth checking whether the bounds rows are correctly normalized before drawing conclusions
about the solver.

---

## 8. mf vs af

- **Distance:** best mf cell 0.284 m, best af cell 0.311 m; per-cell mf is ahead in 3 of 4. All
  inside the noise floor. **No conclusion.**
- **Constraints:** essentially tied. mf/tgt reaches 0.0; af/tgt bottoms out at 1.1.
- **Stability:** af's replicate SD is **0.190 m vs mf's 0.376 m** — af is 2× more reproducible, and
  its constraint metrics are nearly deterministic across the replicate (81.5 vs 81.4 violated steps,
  6 vs 6 clean rollouts) where mf's swing by 32% (92.8 vs 63.5, 10 vs 11).

The natural reading is that mf's rollouts sit near a bifurcation — a tiny perturbation flips the
trajectory into a different basin — while af is in a flatter region. **af is the better arm to run
ablations on**, purely because you need ~4× fewer contexts for the same statistical power. That is
an argument about measurement cost, not about policy quality.

---

## 9. Reproducibility warnings

1. **`n_contexts` is out of sync.** The run used **30** (proved by
   `6/config_snapshot_aligning-d3il-visual/visual_aligning_eval.yaml:44`), but the repo copy at
   `config/visual_aligning_eval.yaml:44` still says **3**. The cluster checkout has an uncommitted
   edit. Anyone reproducing from git gets a 10× smaller run.
2. **`config/visual_aligning_eval.yaml` is shared** with the Gen6V4 and Gen7 evals, so whatever
   value lands there changes their next eval too.
3. **Single seed (6).** All 76 cells come from one training seed and one eval seed. Everything in
   §4–§8 is within-seed.
4. **Train-set contexts** (`results_train_set`). No held-out evaluation in this run.
5. `dpcc-r` and `post_processing` are meant to be the same computation; they differ on 5–8 of 30
   rollouts, up to 1.63 m in mf/tgt. Consistent with the SLSQP nondeterminism noted in the U7 DA,
   now measured at a larger n.

---

## 10. Recommendations

**Measurement (blocking — everything else is guesswork without it):**

1. **Reseed the sampler RNG at the start of every variant.** Then all 19 variants share a noise
   draw per context, the comparison becomes properly paired, and most of the ±0.135 m collapses.
   This is the single highest-value change in this document and it costs a few lines.
2. **Multi-seed before any further ranking.** Gen14 fix_3 already added multi-seed support. Three
   seeds × 30 contexts gets mf into the range where a 0.08 m effect is detectable.
3. Record the **SLSQP exit status** per projector call in the npz — needed to test §7 and to
   quantify how often the projection silently fails.

**Configuration:**

4. **Drop the HardFlow variants from the default sweep.** Keep the code (the port is correct and
   the U7 fix matters); make it opt-in via `HFFM_VARIANTS`, which is already how it works. They
   cost 3× the wall time of the entire rest of the sweep and contribute nothing measurable.
5. **Promote `bounds_free` to a first-class variant** and investigate §7 before shipping any
   "DPCC projection" number as the headline.
6. `combined_5` nominal is not a useful operating point at K=2/T=0.5 — nothing gets below 22
   violated steps. Either raise the projection budget for that geometry or drop it from the sweep.

**Disk:**

7. `find temp/0508 -name 'rollout_*.gif' -delete` — 776 MB, free.
8. Consider the `_mpc_foresight` svg→png revert and dropping `sampled_trajectories_all` from the
   npz for non-`-c`/`-t` variants: another ~1 GB. Proposals only; **no code was changed for this DA.**

---

## 11. Reproduction

```bash
# what ran (2 jobs, ~15 h each, n_contexts=30 in the CLUSTER yaml)
for E in mf af; do
  HFFM_VARIANTS="hardflow_new-r hardflow_new-c hardflow_new-t" \
    ./Slurm_Codes/submit.sh \
      Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh $E 6 none 2
done
#                                                                    arm seed  |   |
#                                                            $3 = --record ----+   |
#                                                            $4 = NFE override ----+

# figures
/tmp/.../plotenv/bin/python logs_in_develop/Gen14/U7/da_20260805_n30_massive.py
```

Overwriting a previous run at the same path is safe as long as `n_contexts` is not *decreased* —
per-rollout artifacts are `rollout_<i>` for `i` in `0..n-1`, so a smaller n leaves the tail behind.
That is exactly how the 10-context gifs in §1 survived into a 30-context folder.
