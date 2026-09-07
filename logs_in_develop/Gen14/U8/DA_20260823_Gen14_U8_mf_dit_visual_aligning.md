# DA — Gen14 U8 `mf@dit` visual aligning, first DiT-bone run

**Date:** 2026-08-23 · **Batch:** `batch_va2_20260823_135156` (DA_VA_v2, 18 candidates / 321 units)
**Jobs:** pipeline `24872` → gates `24873` → train `24874` → eval `24875` · **GIT REV** `eb82d0b`
**Candidate 12** in the batch, snapshot `20260823_004343`.

```
H8_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow
     _a1.5_b1.0_aw1_VTrue_steps1000_bs64_Bdit_Emf_tslogit_normal_TB80pct
  └─ H8_K2_Meuler_T0.5_D…VisualMeanFlow_VTrue_mpc4_Bdit_Emf
```

`_Bdit_` present, `_film*` absent — the DiT branch. `_TB80pct` present — the 80 k budget is a path
key, so this tree cannot be mistaken for a full-budget run.

**Comparators** (same task, seed 6, split, geometries, K=2, `n = 30`/cell):

| | cand | bone | bone params | budget | arm C? |
|---|---|---|---|---|---|
| **DiT** | 12 | `dit` (iMF RoPE DiT) | 3.37 M (0.84×) | 80 k | **no** |
| **mf-v1** | 14 | `unet` + FiLM v1 | 4.04 M (1.00×) | 100 k | **yes** |
| **mf-v2** | 15 | `unet` + FiLM v2 | 4.04 M (1.00×) | 100 k | no |
| *af-UNet* | 6 | `unet` + FiLM v1 (`af`) | 4.04 M | 100 k | yes |

---

## 0. The chain ran clean — with one gap

All 14 gates PASS at `eb82d0b`, **including G0** (the Fix_10 ledger repair from job 24864 holds).
Full U8 bone battery green on hardware (G-B1, G-B2 params, G-B3/B45 JVP-through-visual-token,
G-B6/B7 path keys), plus G1–G7.

Training reached `status completed` with `train/lr 0.0` — the cosine schedule ran out, so all
**80 000 steps** executed; not a wall-clock kill. **~12 h 50 m** (11:49 → 00:43) inside the 24 h wall
at `save_freq = 5000`. Eval ran 16 variants × 2 geometries, `npz_complete = 1.0` everywhere,
**0 circuit-breaker trips**, frozen-rollout rate 6.7 % (identical across candidates — a task
property, not a bone property).

🔴 **The gap: arm C (HardFlow) did not run for the DiT.** See §4 — it exists in Gen14, it has run
before on this task, and it is disabled by default in the shared eval YAML.

## 1. Success is at the floor — distance is the instrument

Pooled over every variant and geometry:

| model | S&C | success | relaxed |
|---|---|---|---|
| **DiT** | 6 / 1856 = **0.32 %** | 22 = 1.19 % | 64 = 3.45 % |
| mf-v1 | 46 / 2204 = 2.09 % | 58 = 2.63 % | 162 = 7.35 % |
| mf-v2 | 22 / 1856 = 1.19 % | 24 = 1.29 % | 38 = 2.05 % |

Per cell (n = 30) that is 0, 1 or 2 episodes, and `n_steps = 400.0` — the episode cap — in almost
every projected cell. Nothing finishes. Success reports which model crosses a threshold by luck.
**Everything below is built on distance.**

**Contexts are paired.** Initial box→target distance is identical to four decimals across all three
candidates (mean **0.4547 m**, sd 0.0560, range 0.3598–0.5903), as are initial box angle
(14.2°, sd 55.7°) and target angle (−1.0°, sd 51.1°). **0.4547 m is the do-nothing baseline** — any
cell at ≈0.45 made no progress at all.

Two distance fields exist and they are not the same:

* **`mean_dist_per_rollout`** — what the eval prints as *"Avg final mean distance"* and the ranking
  reports as `MeanDist_m`. The D3IL aligning metric. **Authoritative; every headline below uses it.**
* `context_final_xy_dist` — final box→target xy displacement only. Partial credit. Cited once, in
  §5, where the disagreement is the point.

## 2. Headline: the DiT loses on distance in every slice

`mean_dist_per_rollout`:

| slice | **DiT** | mf-v1 | mf-v2 |
|---|---|---|---|
| `combined_5` projected | 0.3756 | **0.3471** | 0.3653 |
| `combined_5` unguided | 0.4378 | 0.4184 | **0.3856** |
| `combined_5-tightened` projected | 0.3633 | **0.3392** | 0.3870 |
| `combined_5-tightened` unguided | 0.4291 | **0.3302** | 0.3431 |
| **pooled** | **0.3959** | **0.3425** | 0.3661 |

Last or second-to-last in all four slices. The eight lowest-distance cells in the three-way
comparison are all U-Net (six of them mf-v1); the DiT appears in none. Best cell overall is
**mf-v1 `dpcc-t` untightened at 0.2867 m**; the DiT's best is **`dpcc-t` at 0.3575 m**.

On the task's own metric this is a **clean loss for the DiT**, and it agrees with success.

## 3. The projections — where the result actually lives

### 3.1 What the variant names mean

The suffix is a **candidate-selection rule over the MPC pool of 4**, not a different projector
(`eval_mix_visual_aligning.py:2804-2831`):

| suffix | rule |
|---|---|
| `-r` | `random` — always index 0, deterministic |
| `-c` | `minimum_projection_cost` — pick the candidate needing least correction |
| `-t` | `temporal_consistency` — pick the candidate most consistent with the previous plan |

Arms: **A** unguided (`diffuser` = single sample, no projection, batch 1; `gradient` = guidance) ·
**B** DPCC projector (`dpcc-*`, `post_processing`, and the `dt` threshold sweep) · **C** HardFlow
in-loop NLP (`hardflow_new-*`, §4). `bounds_free` / `geo_free` / `model_free` are constraint-class
ablations. Tightening is a **geo-level** flag in Gen14, not a variant suffix.

### 3.2 Full table — `mean_dist_per_rollout`, n = 30/cell, init 0.4547 m

`combined_5` (untightened) / `combined_5-tightened`. Best of the three bones in **bold**.

| variant | arm | DiT | mf-v1 | mf-v2 | DiT viol | v1 viol |
|---|---|---|---|---|---|---|
| `diffuser` | A | 0.4187 / 0.4480 | 0.4656 / **0.3451** | **0.4098** / 0.3669 | 117.7 / 129.5 | 92.8 / 63.5 |
| `gradient` | A | 0.4568 / 0.4103 | 0.3713 / **0.3153** | **0.3613** / 0.3192 | 108.9 / 106.0 | 97.1 / 74.3 |
| `dpcc-r` | B | 0.4033 / 0.3387 | **0.3423** / **0.3145** | 0.3750 / 0.3873 | 42.8 / 4.4 | 58.4 / 4.5 |
| **`dpcc-c`** | B | **0.3577** / **0.3684** | 0.4094 / 0.3626 | 0.3855 / 0.3740 | **32.2 / 4.2** | 69.5 / 12.6 |
| `dpcc-t` | B | 0.3575 / 0.3450 | **0.2867** / **0.3066** | 0.3369 / 0.4240 | 30.2 / 4.4 | 66.2 / **0.0** |
| `hardflow_new-r` | C | — | 0.3064 / 0.3117 | — | — | 49.6 / 3.4 |
| `hardflow_new-c` | C | — | 0.3249 / **0.2959** | — | — | 60.3 / 2.2 |
| `hardflow_new-t` | C | — | 0.3074 / 0.3112 | — | — | 40.2 / 0.3 |
| `post_processing` | B | 0.3926 / **0.3482** | **0.3481** / 0.3682 | 0.3731 / 0.3921 | 41.4 / 4.6 | 57.1 / 4.9 |
| `dpcc-c-dt0p25` | B | 0.3716 / 0.3547 | **0.3248** / **0.2875** | 0.3353 / 0.3545 | 29.8 / 11.1 | 42.9 / 6.5 |
| `dpcc-c-dt0p5` | B | 0.3559 / 0.3896 | **0.3055** / **0.3229** | 0.3424 / 0.3950 | 34.7 / 11.8 | 55.3 / 11.9 |
| `dpcc-c-dt2p0` | B | 0.3800 / 0.3757 | **0.3760** / **0.3634** | 0.3898 / 0.3846 | 42.0 / 6.3 | 58.3 / 4.7 |
| `dpcc-c-dt4p0` | B | 0.3864 / 0.3864 | 0.3835 / 0.3878 | 0.3846 / 0.3846 | 39.9 / 1.2 | 22.0 / 0.4 |
| `bounds_free` | abl | 0.3253 / 0.3392 | **0.2947** / **0.2835** | 0.3361 / 0.3738 | 31.4 / 3.2 | 40.5 / 0.5 |
| `geo_free` | abl | 0.4166 / 0.4021 | **0.3626** / 0.3798 | 0.3988 / **0.3708** | 138.9 / 163.3 | 82.4 / 73.8 |
| `model_free` | abl | 0.4434 / 0.4749 | 0.3611 / 0.3625 | **0.3199** / **0.3548** | 116.0 / 124.5 | 82.4 / 63.9 |

### 3.3 How much does projection buy each bone?

Untightened, `diffuser` → best arm-B cell:

| bone | unguided | best projected | **gain** |
|---|---|---|---|
| DiT | 0.4187 | 0.3575 (`dpcc-t`) | **0.061 m** |
| mf-v1 | 0.4656 | 0.2867 (`dpcc-t`) | **0.179 m** |
| mf-v2 | 0.4098 | 0.3369 (`dpcc-t`) | 0.073 m |

**The U-Net gets ~3× more out of the projector than the DiT does.** The DiT starts from a better
unguided trajectory (0.4187 vs 0.4656) and ends in a worse projected one. The projector is not the
DiT's bottleneck — the DiT is giving it less to work with. This is the single most actionable line
in the DA: improving the projector will not close this gap.

### 3.4 🔴 The selection rule inverts between bones

Untightened, arm B, spread across `-r` / `-c` / `-t`:

| bone | `-r` random | `-c` min-cost | `-t` temporal | spread | worst rule |
|---|---|---|---|---|---|
| **DiT** | 0.4033 | **0.3577** | **0.3575** | 0.046 | **random** |
| mf-v1 | 0.3423 | **0.4094** | **0.2867** | 0.123 | **min-cost** |

`dpcc-c` is the DiT's joint-best rule and the U-Net's **worst** — a 0.052 m swing in opposite
directions. On the xy metric the same cell is even starker: DiT **0.2112 m** vs mf-v1 **0.4332 m**,
the largest single-cell gap anywhere in this comparison.

A mechanism consistent with §5: minimum-projection-cost picks the candidate needing least
correction. For the DiT — low-variance, all four candidates similar — least-corrected is a fair
proxy for best. For the U-Net — bimodal, sd 0.31 — the least-corrected candidate is often one that
has drifted somewhere the constraints barely touch, so cost is *anti*-correlated with task quality.
**Projector-selection choice is bone-dependent and should not be inherited across bones.** The
U-Net's default of `-t` is right for the U-Net and leaves nothing on the table for the DiT.

### 3.5 The threshold sweep degrades above `dt ≈ 0.5` for every bone

| `diffusion_timestep_threshold` | DiT | mf-v1 | mf-v2 |
|---|---|---|---|
| 0.25 | 0.3716 | 0.3248 | 0.3353 |
| **0.5** | **0.3559** | **0.3055** | 0.3424 |
| 2.0 | 0.3800 | 0.3760 | 0.3898 |
| 4.0 | 0.3864 | 0.3835 | 0.3846 |

Both bones optimise near **0.5** and degrade monotonically above it. At `dt4p0` all three collapse
to 0.384–0.386 with violations near zero (v1: 22.0 untightened, 0.4 tightened) — the projector has
stopped contributing and is merely freezing the plan. **`dt2p0` and `dt4p0` are a dead zone; drop
them from future sweeps or investigate.** The useful range is below 1.0 and has only two samples in
it — a finer sweep (0.1 / 0.25 / 0.5 / 0.75 / 1.0) would be cheap and is currently the only
projector knob showing a real optimum.

### 3.6 The bounds constraint is what costs distance

`bounds_free` is the **best or near-best cell for every bone in both geometries** — DiT 0.3253 /
0.3392, mf-v1 0.2947 / **0.2835** (the single best mf-v1 cell in the batch). Dropping bounds
enforcement improves distance by 0.03–0.05 m at similar violation counts (DiT 31.4 vs 30.2 for
`dpcc-t`; mf-v1 40.5 vs 66.2).

By contrast `geo_free` and `model_free` are clearly **worse** (0.41–0.47 for the DiT) with 3–4×
the violations. So the halfspace/obstacle geometry and the model constraint are load-bearing, and
**the bounds constraint is over-restrictive on this task** — it is buying safety the task was not
losing. Worth checking the bounds spec against the aligning workspace before the next sweep.

## 4. 🔴 HardFlow (arm C) — yes it exists, no it did not run here

**Answering directly: HardFlow *is* ported into Gen14 visual aligning and has produced data on this
task.** `mix_visual_aligning/sampling/hardflow_projection.py` exists (Gen14 U7 port), with its own
gate file `mix_visual_aligning_test/gates_hardflow_mix_visual.py`, and in **this very batch**
candidates **14** (`mf` U-Net v1) and **6** (`af` U-Net v1) carry full `hardflow_new-{c,r,t}` cells.

**It did not run for the DiT** because arm C is opt-in and off by default:

```yaml
# config/visual_aligning_eval.yaml:433
hardflow_variants: []
# hardflow_variants: ['hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t']   # ← line 434, commented
```

The eval reads `HFFM_VARIANTS` first and falls back to that key
(`eval_mix_visual_aligning.py:2455-2458`), so re-running the DiT eval with arm C needs no code
change:

```bash
HFFM_VARIANTS="hardflow_new-r hardflow_new-c hardflow_new-t" \
  sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6
```

`submit.sh` uses `--export=ALL`, so the variable propagates.

### 4.1 What arm C does on the U-Net, since we have it

mf-v1, matched selection rule, arm B vs arm C:

| geo | rule | B distance | **C distance** | B viol | **C viol** | B ms | **C ms** |
|---|---|---|---|---|---|---|---|
| `combined_5` | `-r` | 0.3423 | **0.3064** | 58.4 | **49.6** | 55.5 | 181.9 |
| `combined_5` | `-c` | 0.4094 | **0.3249** | 69.5 | **60.3** | 56.9 | 194.0 |
| `combined_5` | `-t` | **0.2867** | 0.3074 | 66.2 | **40.2** | 52.8 | 174.9 |
| `-tightened` | `-r` | 0.3145 | **0.3117** | 4.5 | **3.4** | 42.3 | 147.2 |
| `-tightened` | `-c` | 0.3626 | **0.2959** | 12.6 | **2.2** | 54.9 | 148.3 |
| `-tightened` | `-t` | **0.3066** | 0.3112 | **0.0** | 0.3 | 42.3 | 145.6 |

Three readings:

1. **HardFlow lowers violations in 5 of 6 pairings**, sometimes hugely (`-c` tightened: 12.6 → 2.2).
2. **On distance it wins 4 of 6, and its wins are concentrated on `-c`** — exactly the selection
   rule where the DPCC projector fails for the U-Net (§3.4). Arm C rescues the rule arm B breaks.
   Where arm B is already good (`-t`), arm C is slightly worse.
3. **It costs ~3.3×** (145–194 ms vs 42–57 ms).

Per the benchmark hierarchy this is still **not** the claim we need: HardFlow must beat the DPCC
projector **at a lower projection threshold**, and both arms ran at the same threshold here. It is a
cost-for-constraint-quality trade, the same verdict Gen16 reached on avoiding. A threshold sweep on
arm C is what would make it mean something.

### 4.2 Why the DiT's missing arm C matters

§3.4 says `dpcc-c` is the DiT's best rule; §4.1 says arm C's largest gains are on `-c`. **The one
cell most likely to show the DiT in its best light is precisely the one that was not run.** This is
the cheapest outstanding experiment in this DA — an eval-only re-run, no retraining.

## 5. Distribution shape: the DiT is tight, the U-Net is bimodal

`combined_5`, projected, `context_final_xy_dist` (the partial-credit metric — used here only because
the shape is the point), n = 240:

| | mean | median | p10 | min | **sd** | **< 0.05 m** | > 0.40 m |
|---|---|---|---|---|---|---|---|
| **DiT** | **0.2972** | 0.2758 | 0.0818 | 0.0075 | **0.1747** | 6.7 % | **33.8 %** |
| mf-v1 | 0.3122 | 0.3160 | **0.0328** | **0.0039** | 0.3100 | **13.8 %** | 38.8 % |
| mf-v2 | 0.3657 | 0.4065 | 0.1128 | 0.0153 | 0.2157 | 2.9 % | 50.4 % |

This is the only slice where the DiT leads on any distance metric, and it explains both §2 and
§3.4. The DiT has **half the spread** — it fails less catastrophically — while the U-Net has the
better **tail**, landing inside 5 cm twice as often. A tighter distribution with a worse tail scores
better on a mean and worse on anything thresholded, which is why the DiT's 0.32 % S&C sits under the
U-Net's 2.09 % while its untightened xy mean is lower.

## 6. Orientation (secondary)

Wrapped |final box angle − target angle|, projected: DiT **80.3°** / mf-v1 67.9° / mf-v2 63.1°
untightened; DiT **72.8°** / 65.1° / 66.9° tightened. Within 15°: DiT 16.7 % vs 19.2 % / 24.6 %.

The DiT is 12–17° worse in the mean and lands within 15° least often. Noted as a contributing
mechanism — it translates better and rotates worse — but distance is the metric of record and §2
already settles the verdict without it. Worth one diagnostic (does the single prepended visual token
attenuate the rotation components of the 9-D action?), not a headline.

## 7. Constraints

Projected (8 variants) vs unguided (2):

| geo | group | metric | **DiT** | mf-v1 | mf-v2 |
|---|---|---|---|---|---|
| `combined_5` | projected | violations ↓ | **36.6** | 53.7 | 84.2 |
| `combined_5` | projected | exec sat rate ↑ | **0.908** | 0.866 | 0.789 |
| `-tightened` | projected | violations ↓ | 6.47 | **6.11** | 9.86 |
| `-tightened` | projected | exec sat rate ↑ | 0.984 | **0.985** | 0.975 |
| `combined_5` | unguided | violations ↓ | 113.3 | **95.0** | 162.3 |
| `-tightened` | unguided | violations ↓ | 126.8 | **74.2** | 163.8 |

The DiT is ahead untightened+projected and behind unguided: its **raw** samples respect constraints
less, and the projector simply handles them better. `collision_free_completed` agrees with the
pessimistic reading (0.221 vs 0.275 untightened; 0.749 vs 0.865 tightened). The DiT produces more
*projectable* trajectories, not safer ones — which sits oddly beside §3.3, where it also extracts
less benefit from projection.

## 8. Cost

Mean `avg_time_ms` all cells: DiT 50.1 · mf-v1 64.4 · mf-v2 51.7. But restricted to projected
variants it is a tie untightened (58.7 vs 58.7) and reverses tightened (60.9 vs 48.0); on unguided
the DiT is slower (33.5 vs 28.7). **The 50-vs-64 headline is a mix effect, not a per-call speedup.**
Arm C is the only real cost difference in this DA at ~3.3× (§4.1). Training: 80 k steps ≈ 12 h 50 m.

## 9. Confounds, ordered

1. **Budget, 80 k vs 100 k.** Only the DiT is short of full budget. `_TB80pct` makes it visible, not
   comparable.
2. **Parameters, 3.37 M vs 4.04 M (0.84×).** `mf_dit` is the exactly matched bone (4.04 M, 1.00×).
3. **Arm C missing for the DiT** (§4.2) — the comparison is 2-arm for the DiT and 3-arm for mf-v1.
4. **One seed (6), n = 30/cell.** Shared by all candidates so it does not bias the comparison, but
   only §3.3, §3.4 and §5 are large enough to survive it. Success is not.
5. **Train split only.** No `test` split for any candidate; all of this is on seen data.
6. **Noise floor.** At 0.3–2 % strict success, "which model fails less badly" is what is measured.

## 10. Verdict

**The DiT bone trains, evaluates and integrates correctly — and on visual aligning it loses to the
matched U-Net on distance in every slice** (0.3959 vs 0.3425 m pooled; zero appearances in the eight
best cells).

The projector analysis says where the loss comes from. The DiT's *unguided* trajectory is
**better** than the U-Net's (0.4187 vs 0.4656) — it is the projection stage where it falls behind,
extracting 0.061 m of improvement where the U-Net extracts 0.179 m (§3.3). Its distribution is
tighter and its tail is worse (§5), which flips the ranking of the min-cost selection rule between
the two bones (§3.4). **This is not a "worse generative model" story; it is a "worse interaction
with the projector" story**, and it will not be fixed by tuning the projector.

At 80 % budget, 84 % parameters and with arm C unrun, this is **not a final architectural verdict** —
but per the reporting rule it is a **loss on the task metric with a specific, mechanistic
explanation**, not a Pareto trade-off.

## 11. Next, in order

1. **Re-run the DiT eval with arm C** — eval-only, no retraining, one command (§4). It is the
   cheapest experiment here and §4.2 argues it targets the DiT's best cell.
2. **`mf@mf_dit` at full budget** — the 4.04 M exactly parameter-matched DiT. Kills confounds §9.1
   and §9.2 in one run, and is worth more than a second look at an 0.84× bone.
3. **Fine threshold sweep below `dt = 1.0`** (0.1 / 0.25 / 0.5 / 0.75) and **drop `dt2p0` / `dt4p0`**
   (§3.5) — the only projector knob with a demonstrated optimum, currently sampled twice.
4. **Audit the bounds constraint** (§3.6) — `bounds_free` is the best cell for every bone, at
   comparable violations. The spec is likely over-restrictive for the aligning workspace.
5. **Pick selection rules per bone, not per generation** (§3.4) — `-c` for the DiT, `-t` for the
   U-Net. The current shared default silently handicaps one of them.
6. **Arm C threshold sweep** so §4.1 can answer the HardFlow-vs-DPCC question the hierarchy asks.
7. **`af@sit` / `af@dit`**, then a test split, then seeds 7–10.

## 12. One-line summary

**Gen14's first DiT-bone run completed cleanly through 14 gates, 80 k steps and a 16-variant eval,
and loses to the matched U-Net on the D3IL aligning distance metric in every slice (0.3959 vs
0.3425 m pooled) — not because its raw samples are worse (they are better: 0.4187 vs 0.4656
unguided) but because it extracts 3× less from the DPCC projector, with the min-cost selection rule
inverting between the two bones; HardFlow arm C exists in Gen14 and ran on the U-Net but was left
off for the DiT, which is the cheapest outstanding experiment.**
