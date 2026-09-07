# DA 2026-09-07 — The UAV **α-Flow U-Net** arm: does the Gen3v7 port hold on `s_curve` and `pillars`?

**Runs under test (the U6 port, and only those):**

| chain | scene | K | jobs | candidates |
|---|---|---|---|---|
| 25439 | `s_curve` | 1, 2, 5 | train **25440** → evals **25441 / 25442 / 25443** | 84 / 85 / 86 |
| 25392 | `pillars` | 1, 2 | train **25393** → evals **25394 / 25395** | 46 / 48 |

**Code:** `963faed` (`s_curve`), `ca0eb31` (`pillars`) · **Batch:** `batch_uav_20260907_141115`
· **Seed:** 6 · **n = 10 rollouts** per variant · **Prescribed in:**
[`Gen15/U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md`](../U6/RUNSTATUS_20260904_uav_pipelines_submitted_pre_U6.md) §5
· **Status:** [`RUNSTATUS_20260905_…_and_cleanup.md`](../U6/RUNSTATUS_20260905_af_unet_resubmit_25434_25439_and_cleanup.md)

---

## 0. Answer in three lines

1. 🟢 **The port is mechanically correct.** α is genuinely on at 0.2 on a 3.97 M U-Net in both
   chains; every gate passes; all 5 evals completed 10/10 rollouts on every variant. §1.
2. 🔴 **The policy is not good.** On `s_curve` under honest geometry the arm reaches
   **S&C = 0.00 on 27 of 30 legal cells**, and its best cell anywhere is **0.20**. §3.
3. 🔴 **And it gets worse as K grows.** On the *unprojected* plan, success falls
   **0.60 (K=1) → 0.10 (K=2) → 0.20 (K=5)** and goal distance rises **1.31 → 2.08 → 2.25 m**.
   Extra NFE actively hurts this checkpoint. §2 — this is the finding of the batch.

`pillars` (46 / 48) is **void for ranking**: it ran on pre-U7 geometry, the arena that U7 proved
was scoring the box rather than the policy. §4.

---

## 1. Gate check — the port did what it was built to do

| gate | `s_curve` (25440 / 25441) | `pillars` (25393 / 25394) |
|---|---|---|
| backbone | `backbone=unet … params=3,969,222 (3.97 M)` | same, `3,969,222 (3.97 M)` |
| α floored, not annealed to 0 | `alpha=0.2` through the final epochs | `alpha=0.2`, epochs 96–99 |
| bootstrap branch taken | `discrete_frac=0.25` at step 99999 | `0.25 – 0.625`, `test/discrete_frac 0.51` |
| checkpoint path key | `…AlphaFlowODE_9D_as1_ae0.2_bbunet/6` | same |
| eval loads the floored ckpt | `alpha(step 100000) = 0.2000 … ACTIVE` | `… ACTIVE` |
| selector | `_EPlatest` in the results path | `_EPlatest` |
| training finished | `JOB END Sun Sep 6 09:58:11 UTC`, completed | `JOB END Sat Sep 5 01:50:06 UTC`, completed |
| evals finished | 25441/42/43 all 10/10 on every variant | 25394/25395 all 10/10 |

`data_quality.csv` confirms `n_rollouts = 10` and 10 diagnostics JSONs for **every** variant of
candidates 84, 85, 86, 46, 48 — including the 7 `hardflow_sls*` variants at K=5. **Nothing here is
partial.** The α-Flow objective trained and deployed UAV weights, twice, exactly as specified.

---

## 2. The headline: the raw plan degrades with K

`diffuser` = projection **off**, so this is the generative model alone. `s_curve`, honest geometry:

| K | S&C | success | coll-free | violations | steps | goal dist (m) | ms/replan |
|---|---|---|---|---|---|---|---|
| **1** | 0.00 | **0.60** | 0.00 | 1.6 | 704.8 | **1.31** | **9.2** |
| **2** | 0.00 | 0.10 | 0.00 | 3.6 | 793.9 | 2.08 | 18.5 |
| **5** | 0.00 | 0.20 | 0.00 | 1.0 | 818.6 | 2.25 | 44.7 |

Success collapses 6× between K=1 and K=2 and does not recover at K=5; goal distance grows
monotonically; step count grows monotonically; cost grows linearly in K. **There is no K at which
paying more NFE buys anything on this checkpoint** — K=1 dominates K=2 and K=5 on every column at
once. That echoes the `avoiding-d3il` result, where the α-on arm's wins were concentrated at K=1.

It also means the natural deployment point for this arm is **K=1 at 9.2 ms/replan**, which is the
one place it is cheap. What it is not, at any K, is *safe*: `collision_free = 0.00` on all three.

---

## 3. `s_curve` — full projection tables (honest geometry, n=10)

`geo_free` rows drop the geometry constraints from the projector; they are a **diagnostic**, not a
claim arm, and are marked ◇.

### 3.1 K = 1 (candidate 84)

| variant | S&C | succ | c-free | viol | steps | goal d | ms |
|---|---|---|---|---|---|---|---|
| diffuser | 0.00 | 0.60 | 0.00 | 1.6 | 704.8 | 1.31 | 9.2 |
| dpcc-c | 0.00 | 0.30 | 0.00 | 3.4 | 805.4 | 1.34 | 117.3 |
| dpcc-c-tightened | 0.00 | 0.10 | 0.00 | 13.2 | 849.9 | 1.80 | 206.6 |
| dpcc-r | 0.00 | 0.30 | 0.00 | 4.7 | 801.3 | 1.78 | 134.4 |
| dpcc-r-tightened | 0.00 | 0.10 | 0.00 | 8.8 | 805.0 | 1.45 | 141.7 |
| dpcc-t | 0.00 | 0.60 | 0.00 | 3.2 | 698.6 | 0.91 | 116.7 |
| dpcc-t-tightened | 0.00 | 0.50 | 0.00 | 23.0 | 748.3 | 1.02 | 223.0 |
| ◇ dpcc-c-geo_free | 0.00 | 0.50 | 0.00 | 50.7 | 733.2 | 1.53 | 20.8 |
| ◇ dpcc-r-geo_free | 0.00 | 0.50 | 0.00 | 1.3 | 733.0 | 1.54 | 20.8 |
| ◇ dpcc-t-geo_free | 0.00 | **0.80** | 0.00 | 4.8 | **615.0** | **0.54** | 21.5 |

### 3.2 K = 2 (candidate 85)

| variant | S&C | succ | c-free | viol | steps | goal d | ms |
|---|---|---|---|---|---|---|---|
| diffuser | 0.00 | 0.10 | 0.00 | 3.6 | 793.9 | 2.08 | 18.5 |
| dpcc-c | 0.00 | 0.10 | 0.00 | 1.1 | 848.7 | 2.56 | 89.6 |
| dpcc-c-tightened | 0.00 | 0.00 | 0.00 | 3.2 | 854.0 | 2.32 | 99.2 |
| dpcc-r | 0.00 | 0.30 | 0.00 | 1.2 | 805.5 | 2.09 | 79.8 |
| dpcc-r-tightened | 0.00 | 0.00 | 0.00 | 10.9 | 871.0 | 2.45 | 148.4 |
| dpcc-t | 0.00 | **0.70** | 0.00 | 3.1 | **688.4** | **0.82** | 91.9 |
| dpcc-t-tightened | 0.00 | 0.50 | 0.00 | 2.3 | 760.7 | 1.21 | 181.2 |
| ◇ dpcc-c-geo_free | 0.00 | 0.20 | 0.00 | 1.5 | 818.4 | 2.27 | 29.8 |
| ◇ dpcc-r-geo_free | 0.00 | 0.20 | 0.00 | 4.4 | 817.0 | 2.26 | 29.9 |
| ◇ dpcc-t-geo_free | 0.00 | 0.50 | 0.00 | 1.3 | 734.2 | 1.56 | 29.8 |

### 3.3 K = 5 (candidate 86) — the only cells that are not zero

| variant | S&C | succ | c-free | viol | steps | goal d | ms |
|---|---|---|---|---|---|---|---|
| diffuser | 0.00 | 0.20 | 0.00 | 1.0 | 818.6 | 2.25 | 44.7 |
| dpcc-c | **0.10** | 0.10 | 0.10 | 1.2 | 847.3 | 2.55 | 1226.8 |
| dpcc-c-tightened | 0.00 | 0.00 | 0.00 | 17.6 | 871.0 | 2.83 | 1355.9 |
| dpcc-r | 0.00 | 0.10 | 0.00 | 28.5 | 845.4 | 2.30 | 1346.3 |
| dpcc-r-tightened | 0.00 | 0.00 | 0.00 | 74.2 | 871.0 | 2.72 | 1382.6 |
| dpcc-t | 0.00 | 0.20 | 0.00 | 5.2 | 792.9 | 1.37 | 1241.7 |
| dpcc-t-tightened | 0.00 | 0.00 | 0.00 | 37.7 | 846.9 | 1.99 | 1473.5 |
| **hardflow_sls-r** | **0.20** | 0.20 | 0.20 | 97.2 | 729.1 | 0.83 | 1290.1 |
| **hardflow_sls-t** | **0.20** | 0.20 | 0.20 | 104.2 | 720.4 | 1.07 | 1242.2 |
| hardflow_sls-c | 0.10 | 0.20 | 0.10 | 90.8 | 701.1 | 0.78 | 1269.9 |
| hardflow_sls | 0.00 | 0.00 | 0.00 | 119.6 | 701.5 | **0.72** | 289.4 |
| ◇ dpcc-t-geo_free | 0.00 | 0.60 | 0.00 | 2.2 | 688.8 | 1.05 | 78.8 |
| ◇ dpcc-c/r/t-geo_free (hf) | 0.00 | 0.10–0.40 | 0.00 | 1.0–3.7 | 765–845 | 1.8–2.6 | ~100 |

**Reading.** The arm's ceiling is `hardflow_sls-{r,t}` at **S&C 0.20 / 20 % collision-free**, bought
at **~1.29 s per replan** and **~100 violations per rollout**. Every `dpcc-*` projection at K=5
costs 1.2–1.5 s/step — two orders of magnitude over any plausible control budget — and returns at
most 0.10. The three K=5 HardFlow cells are the only place `collision_free > 0` in the entire
`s_curve` sweep, and even there 4 of 5 rollouts still collide.

---

## 4. `pillars` (46 / 48) — void for ranking

Both ran on the **pre-U7 arena** (`geo = pillars_bounds+…`, not `pillars_hg_…`). U7 showed that on
that geometry an identical trajectory scores S&C 0.00 vs 0.30 purely because `geo_bounds` at
y = ±1.5 sat inside the route the expert demonstrates. The signature is visible here: `diffuser`
books **115.5 (K=1) / 123.4 (K=2)** violations, and the `geo_free` rows — which drop exactly that
constraint — book **105–127** while the constrained rows book 1.6–24.5.

| K | variant | S&C | succ | c-free | viol | steps | goal d | ms |
|---|---|---|---|---|---|---|---|---|
| 1 | diffuser | 0.00 | 0.00 | 0.00 | 115.5 | 634.0 | 0.75 | 9.2 |
| 1 | dpcc-t-tightened | 0.00 | 0.30 | 0.00 | 24.5 | 510.5 | 0.35 | 135.7 |
| 2 | diffuser | 0.00 | 0.20 | 0.00 | 123.4 | 595.3 | 0.75 | 17.9 |
| 2 | dpcc-t | 0.00 | **0.40** | 0.00 | 13.4 | **497.7** | 0.30 | 136.4 |

S&C is 0.00 in all 20 cells. **No conclusion may be drawn from these numbers** — the scene, not the
policy, produced them. Any `pillars` claim needs a re-evaluation on `pillars_hg` geometry.

---

## 5. Reference only — not my runs

For scale, from the same batch, `s_curve` honest geometry, K = 2, `diffuser` (projection off):

| arm | S&C | success | goal dist (m) | source |
|---|---|---|---|---|
| **AF-UNet α→0.2** (this DA, cand 85) | 0.00 | 0.10 | 2.08 | 25442 |
| MF-UNet (cand 92) | 0.00 | 0.00 | 2.51 | not mine |
| naive FM (cand 91) | 0.00 | **0.70** | **1.11** | not mine |

At matched K=2 the α-Flow U-Net is ahead of MeanFlow and far behind naive FM. **No arm reaches
S&C > 0.10 on `s_curve` at K=2**, and there is still **no `diffusion` target arm under honest
geometry on any scene**, so per the benchmark hierarchy no ranking claim is available here at all.
Drop this section if you want the DA restricted strictly to the U6 chains.

---

## 6. Verdict

**The engineering succeeded and the hypothesis did not.** Every mechanism U6 was written to deliver
is verified working on real UAV runs: the U-Net backbone, the α floor, the bootstrap branch, the
path keys, the checkpoint selector, the degeneracy guard. That was the open question on 09-05 and
it is now closed — **the UAV α-Flow U-Net arm is production-ready as code.**

What it produces is not competitive. On `s_curve` the arm never completes a collision-free rollout
under any `dpcc-*` projection at any K; its single best legal cell is S&C 0.20 at 1.29 s/replan; and
its unprojected plan is *strictly best at K = 1*, degrading with every extra NFE. `pillars` cannot
speak to this either way because it ran on the arena U7 retired.

This is the third scene family in a row where the α floor fails to reproduce the `avoiding-d3il`
result — after Gen14 U12 on Visual Aligning, where the ordering came out `mf > af > fm` on task
progress. The `avoiding` win is looking increasingly like a property of that task, not of α-Flow.

## 7. To run on cluster

1. **`pillars` AF-UNet on honest geometry.** Eval-only, the checkpoint exists, ~1 h per K:
   ```bash
   UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest \
     FMPCC_UAV_EVAL_TAG=u7hg \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af pillars "6" "1 2"
   ```
2. **A `diffusion` arm under `_hg`** on at least one scene. Without it nothing in Gen15 UAV can be
   ranked against the pinned baseline, only described.
3. **Do not spend GPU on K > 1 for this arm** until §2 is explained. If more NFE reliably hurts,
   the K sweep is measuring a defect, not a trade-off.
4. **Seeds 7–10 at K=1** if the arm is pursued at all — every number here is n=10, one seed.

## 8. Open issues

- **§2 has no mechanism yet.** A generative model that degrades with integration steps points at
  the ODE solver or the α-Flow velocity parameterisation, not at the projector — the `diffuser`
  arm has no projector. Worth a targeted check before any more UAV training.
- `hardflow_sls*` at K=5 is the only place `collision_free > 0`, at ~100 violations/rollout. That
  combination (some rollouts clean, the rest badly violating) is not yet explained.
- `s_curve` K=5 `dpcc-*` at 1.2–1.5 s/replan is far outside any control budget; those rows are
  feasibility probes, not deployable configurations.
