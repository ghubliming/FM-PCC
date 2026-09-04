# PLAN 2026-09-04 — Making α-Flow win on Visual Aligning: the attack plan

*Goal: reproduce the `avoiding-d3il` AF-UNet result
([`Report_20260903_AF_UNet`](../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md))
on `aligning-d3il-visual`. Written against
[`DA_20260904_Gen14_U12_alpha_floor_and_latest_checkpoint.md`](DA_20260904_Gen14_U12_alpha_floor_and_latest_checkpoint.md),
which is the run this plan is reacting to.*

**The headline finding of this plan: we never tested the cell where α-Flow wins.**
Avoiding's win is specifically at **K = 1**. Every Gen14 α-Flow evaluation ever run is at **K = 2**.
K is an **inference-only** knob, so testing it costs **zero training** — and there is no `mf` K = 1
row on this scene either, so the matched comparison does not currently exist in any form.

---

## 1. What "winning" means here — decided before the runs

Fixed now so it cannot be reinterpreted after the numbers land.

| rank | claim | requirement |
|---|---|---|
| **W1 — the real win** | α-Flow **Pareto-dominates** MeanFlow | at equal `S&C`, **fewer `n_steps` AND lower `avg_time`**. Same K, same bone, same seed, same contexts, same variant. |
| **W2 — acceptable** | α-Flow is **non-dominated** vs MeanFlow | better on one axis, worse on another, both reported. The word is "trade-off", never "beats". |
| **W3 — the ladder** | `af > mf > fm` | must **name the axis**. §3.1 Q3 of the DA shows it currently holds on 0-viol only and inverts on task progress. |
| **W0 — the blocker** | any of the above is *citable* | at least one arm must reach **`S&C > 0`**. Today every arm — including the pinned `diffusion` K20 aw10 target — is at **`S&C = 0.000`**. **Until W0 clears, W1–W3 cannot be claimed at all.** |

W0 is the honest gate and it is not negotiable: with the primary metric at zero for every engine
*and the baseline*, a "win" on a secondary axis inside failed rollouts has no anchor. Every phase
below is therefore scored on **W0 first** — does anything move `S&C` off the floor — and only then
on W1/W2.

---

## 2. Avoiding WIN vs Gen14 LOSS — knob by knob

Avoiding checkpoint: `…AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.2_ag25.0_rf0.5`
Gen14 checkpoint:    `H8_<AF>_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Eaf_tslogit_normal_afschsigmoid_AFAFend0p2`

| knob | avoiding (WIN) | Gen14 (LOSS) | verdict |
|---|---|---|---|
| `af_alpha_init` | 1.0 | 1.0 | ✅ same |
| **`af_alpha_end`** | **0.2** | **0.2** (cand 9) | ✅ same |
| `af_alpha_gamma` | 25.0 | 25.0 | ✅ same |
| `af_ratio_fm` | 0.5 | 0.5 | ✅ same |
| `t_schedule` | `logit_normal` | `logit_normal` | ✅ same |
| checkpoint selector | `latest` | `latest` | ✅ same |
| **K (NFE)** | **1** ← *the winning cell* | **2, and only 2** | 🔴 **NEVER TESTED — lever #1** |
| **n rollouts** | **20** | **10** | 🟠 half the power — lever #4 |
| headline variant | `dpcc-t-tightened` | reported on `combined_5` | 🟠 read the right cell |
| **`discrete_frac`, final** | **0.25 – 0.41** | **0.50173** | ⚠️ **unexplained** — §5 |
| backbone | 4.0 M state U-Net | 26.4 M visual U-Net + `MultiImageObsEncoder` | ⛔ structural, cannot match |
| `action_weight` | `aw10` | `aw1` | ⛔ **NOT a lever** — see §4 |
| baseline `S&C` | **1.00 reachable** | **0.000 for everyone** | ⛔ the W0 blocker |

**The α-Flow objective is already configured identically.** Every knob that defines the method
matches the winning recipe. What differs is *where we looked* (K, variant, power) and *what the
scene can support* (W0).

---

## 3. The three leverage points, in order

### Lever 1 — K = 1. Free, and it is the whole ballgame.

Avoiding's report is explicit that the ordering lives at low NFE:

> `top-left-hard`, K = 1, `dpcc-t-tightened`, all at S&C 1.00 — **α-Flow 57.20 steps < MeanFlow
> 60.75 < naive FM 65.65**… **`both-hard` puts `A1t` alone at the front. MeanFlow has *no* S&C = 1.00
> row at K = 1 here** — its cheapest clearing point is `M2t` at K = 2.

That last sentence is the mechanism. The bootstrapped target buys **few-step quality**; by K = 2
MeanFlow has caught up. **Gen14 has run α-Flow at K = 2 and nowhere else** — i.e. exclusively at the
budget where avoiding says the advantage has already washed out.

`flow_steps_v3` is eval-time (`[ config->pkl ] INFO flow_steps_v3: train=100 -> eval=2`), passed as
`$4` of the eval sbatch. **No retraining.** There is also **no `mf` K = 1 row and no `fm` K = 1 row**
on this scene, so the matched ladder has to be built from scratch — four eval jobs.

### Lever 2 — the mid-curriculum checkpoint. Free.

α at the steps actually on disk (`sigmoid`, γ = 25, `save_freq = 20 000`):

| step | α (`ae0.05`) | α (`ae0.2`) | what it is |
|---|---|---|---|
| 20 000 | 1.000 | 1.000 | pure FM |
| 40 000 | 0.928 | 0.939 | ~FM |
| 50 000 | 0.525 | 0.600 | the crossover — **no checkpoint** |
| **60 000** | **0.122** | **0.261** | **genuinely mid-curriculum** |
| 100 000 | 0.050 | 0.200 | the floor ← what U12 deployed |

γ = 25 compresses the entire FM → MeanFlow transition into steps 40 k–60 k — **20 % of the budget**.
And the strongest `af` arm on task in the whole DA is the shipped `α_end=0` arm evaluated at
**`best`**, which is selected on `test_loss ≈ 0.75 + 0.25·α` and therefore *structurally prefers a
mid-curriculum checkpoint*. **The best "α-Flow" model we own may be a mid-curriculum one, by
accident** — and U12's contribution was to move away from it. `state_60000.pt` already exists on
both new trees.

### Lever 3 — γ. One train job.

If lever 2 fires, γ is the principled version of it: `MIX_AF_ALPHA_GAMMA=5` spreads the homotopy
across most of training instead of 20 % of it. Wired already (`config/aligning-d3il-visual.py:1545`),
path key `_AFg5`. Pair it with `MIX_SAVE_EVERY=5000` so the curve is sampled 20× instead of 5×.

---

## 4. Ruled out — do not spend GPU here

| candidate | why not |
|---|---|
| **`action_weight` 1 → 10** | Looks like a 10× difference from avoiding's `aw10`. It is **cosmetic on both sides.** `mix_visual_aligning/models/af_diffusion.py:67-70`: *"FIX-3: action_weight / loss_discount are KEPT (utils + folder naming read them) but are deliberately **NOT applied** to the α-Flow loss… **DO NOT "fix" this back**."* Same in `avoiding-d3il.py:885`. The token differs; the loss does not. |
| **backbone** | 4.0 M state U-Net vs 26.4 M visual U-Net + vision encoder. Not matchable, and matching it would destroy the task. This is the one genuinely structural difference and it is a *finding*, not a bug to fix. |
| **more α floors first** | The three we have are **non-monotone** in progress (0.140 / 0.042 / 0.072 for α = 0 / 0.05 / 0.2). At n = 1 seed that shape is as likely noise as signal. Floors come *after* K and seeds, not before. |
| **`af_adp_eps` 1e-3 → 0.01** | Real and uncontrolled (`config/aligning-d3il-visual.py:1675-1678` calls the 10× gap deliberate), but it is **also 1e-3 on avoiding**, where α-Flow wins. It cannot explain the difference. Park it; it only matters for a *clean one-knob* af-vs-mf claim later, and it needs a code edit. |

---

## 5. One thing we do not understand

`train/discrete_frac` at the final epochs: **avoiding 0.25 – 0.41**, **Gen14 0.50173**.

With `af_ratio_fm = 0.5` forcing half the batch to `r == t` (the FM anchors), 0.5 is the *ceiling* —
Gen14 sits exactly at it, meaning **every** non-anchor sample takes the bootstrapped branch. On
avoiding a substantial fraction is instead routed to the exact-JVP branch by the `clamp_value` snap
(`flow_matcher_v3_alphaflow/models/af_diffusion.py:437-441`). Same α, same `rf`, different routing.

Not a lever yet — but if levers 1–3 all miss, this is the next thing to instrument, because it means
the two runs are training on different *mixtures* of target despite identical α knobs.

---

## 6. The plan

### Phase 0 — free. Eight eval jobs, zero training. **Submit this now.**

```bash
# ── P0.1  THE K LADDER — the single highest-value test in this plan ──────────────────
# af, alpha floored 0.2, endpoint checkpoint, at K=1 and K=5.
# K=2 already exists (DA candidate 9) — do NOT resubmit it.
# Keep FMPCC_RUN_MSG identical so the ladder is ONE family and K differentiates the folder.
for K in 1 5; do
  MIX_AF_ALPHA_END=0.2 MIX_EPOCH=latest FMPCC_RUN_MSG=afon02_s6 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh \
    af 6 all $K
done

# The comparators at K=1 DO NOT EXIST on this scene. Without these the ladder is unmatched.
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 all 1
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh fm 6 all 1

# ── P0.2  MID-CURRICULUM SWEEP — tests whether we deployed the wrong point ───────────
# alpha at these steps on the ae0.2 tree: 0.939 / 0.261 / 0.200
for EP in 40000 60000 80000; do
  MIX_AF_ALPHA_END=0.2 MIX_EPOCH=$EP FMPCC_RUN_MSG=afon02_ep${EP}_s6 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh \
    af 6 all 1
done

# ── P0.3  THE LOW FLOOR AT K=1 — cheap, completes the floor × K grid ─────────────────
MIX_AF_ALPHA_END=0.05 MIX_EPOCH=latest FMPCC_RUN_MSG=afon005_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh \
  af 6 all 1
```

Every job writes its own `H8_K<k>_…_EP<sel>_msg…` directory. **Nothing existing is touched.**

### Phase 1 — one train job each. Submit only if Phase 0 moves `S&C` or reverses the af/mf order.

```bash
# P1.1  gamma=5 — make the curriculum an actual curriculum, and sample it 20x not 5x
MIX_AF_ALPHA_END=0.2 MIX_AF_ALPHA_GAMMA=5 MIX_SAVE_EVERY=5000 MIX_EPOCH=latest \
  FMPCC_RUN_MSG=afon02_g5_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
  af 6 1
#   -> checkpoint tree gains '_AFend0p2-g5';  eval at K=1

# P1.2  a higher floor — avoiding's own next step was alpha->0.4
MIX_AF_ALPHA_END=0.4 MIX_EPOCH=latest FMPCC_RUN_MSG=afon04_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh \
  af 6 1
#   -> checkpoint tree '_AFend0p4';  eval at K=1
```

✅ Verified: the pipeline signature is `<engine> <seeds> <flow_steps>`
(`mix_visual_aligning_pipeline.sh:79-86`) and it forwards `"$FLOW_STEPS"` to the eval stage at
line 300, so the trailing `1` in both commands above **is** K = 1. Gates → train → eval are
chained with `afterok`, so a failed stage cancels only its own downstream.

### Phase 2 — power. Only for whatever configuration survives Phases 0–1.

```bash
# seeds 7 and 8 on the winner (substitute the winning knobs)
MIX_AF_ALPHA_END=0.2 MIX_EPOCH=latest FMPCC_RUN_MSG=afon02_s7 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 7 1
MIX_AF_ALPHA_END=0.2 MIX_EPOCH=latest FMPCC_RUN_MSG=afon02_s8 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 8 1
```

**Optional, and think before doing it:** raising `n_contexts` 10 → 20 in
`config/visual_aligning_eval.yaml` doubles the power and matches avoiding's `n_trials = 20`. But that
yaml is **shared with the Gen6V4 and Gen7 evals**, so it changes the benchmark for arms that are
already banked. If you do it, re-run the `mf` comparator at the same count in the same batch.

---

## 7. Reproducibility hazard to fix first

**The repo and the cluster disagree on `n_contexts`.** `config/visual_aligning_eval.yaml:44` says
`n_contexts: 3` at every commit in the range these runs used (`ba05cb7c`, `43d684cb`), yet jobs
25373 / 25377 evaluated **10 rollouts per variant** (`Zero-violation rollouts: 9 / 10`). The cluster
copy is locally edited and **not** tracked. Consequences:

- a fresh clone reproduces this DA at **3 contexts**, not 10;
- Phase 0 will silently inherit whatever the cluster copy currently holds.

**Before submitting Phase 0**, check it and record the value in the batch notes:

```bash
grep -n 'n_contexts' /data/home/llim/FMPCC/FM-PCC/config/visual_aligning_eval.yaml
```

Every Phase-0 cell must run at the **same** count as candidates 8/9 or the K ladder is not paired.

---

## 8. How to read the results, and when to stop

**Read in this order.** For each new cell, from `constraint_metrics.json` + the aggregated CSVs:

1. **`S&C` (`n_success_and_constraints`)** — the W0 gate. Anything > 0 is the first real news this
   scene has produced.
2. **progress** = `context_init_xy_dist − context_final_xy_dist`. The DA's §3.0 table is the
   reference; a new cell joins that table or it means nothing.
3. **0-viol** (`collision_free_completed`) and `constraint_exec_sat_rate`.
4. **`n_steps` and `avg_time_ms`** — only meaningful *at equal S&C*, per W1.

**Report the sweep as a sweep.** Phase 0 produces 8 cells; Phase 1 adds 2 more. If one of them beats
`mf`, the artefact is **all** cells plus the α and K curves — not the single flattering cell. Picking
a winner post hoc out of a family this size is precisely how the non-monotone floor result in §4
would turn a null into a false positive.

**Kill criteria — stop and write the negative result if:**

- Phase 0 leaves `S&C = 0.000` in every cell **and** `mf` still leads on progress at K = 1.
  That is a matched, well-powered-in-cells negative and it is publishable as scene-dependence:
  *α-Flow's few-step advantage is real on a 4.0 M state-space U-Net and does not survive a 26.4 M
  visual encoder.*
- Or Phase 1 changes nothing after γ and the floor have both moved.

At that point the AF budget goes to **`s_curve` (Gen15)**, which has a live `diffusion` target arm and
can actually rank a result — carrying **`α_end=0.2 @latest`, and now also `K=1`**, per
[`RUNSTATUS_20260904`](../Gen15/U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md).

---

## 9. Summary — one line per phase

| phase | cost | tests | kills the plan if |
|---|---|---|---|
| **P0.1 K ladder** | 4 evals, **0 GPU-train** | Does α-Flow's few-step edge exist here at all? | af ≤ mf at K=1 on progress |
| **P0.2 mid-curriculum** | 3 evals, **0 GPU-train** | Did U12 deploy the wrong checkpoint? | step 60 000 ≤ step 100 000 |
| **P0.3 low floor @ K=1** | 1 eval, **0 GPU-train** | Completes the floor × K grid | — |
| **P1.1 γ = 5** | 1 train + 1 eval | Is γ=25 too sharp to be a curriculum? | no change vs γ=25 |
| **P1.2 α_end = 0.4** | 1 train + 1 eval | Does a higher floor help, as on avoiding? | worse than 0.2 |
| **P2 seeds 7, 8** | 2 trains + 2 evals | Is the winner real or n=1 noise? | margin inside seed spread |

**Start with Phase 0. It is eight eval jobs and no training, and P0.1 alone tests the one hypothesis
that best explains why avoiding won and aligning did not.**
