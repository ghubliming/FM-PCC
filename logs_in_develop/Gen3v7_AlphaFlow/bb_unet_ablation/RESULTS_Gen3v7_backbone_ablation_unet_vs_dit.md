# RESULTS — Backbone ablation: UNet vs DiT for the MeanFlow-family objective (α-Flow & MeanFlow)

**Date:** 2026-07-25 · **Type:** results analysis · **Status:** ⚠️ seed 6, n=2 trials — smell-test scale, but the effect is huge
**Follows:** [`../init/INSIGHT_Gen3v7_first_train_curve.md`](../init/INSIGHT_Gen3v7_first_train_curve.md) · [`../init/NOTE_backbone_fidelity_Gen3v4_v6_v7.md`](../init/NOTE_backbone_fidelity_Gen3v4_v6_v7.md)
**New evidence:**
- α-Flow **UNet**: `temp/Gen3V7/2507/23_49_20_af_train_23810.log` + `..._af_eval_23811.log` (K∈{1,2,5,10})
- MeanFlow **UNet**: `temp/Gen3v6/2507/23_49_59_mf_train_23813.log` + `..._mf_eval_23814.log`
- (baselines) α-Flow **DiT**: `temp/Gen3V7/00_36_18_eval_alphaflow_23786.log` · MeanFlow **DiT**: `temp/Gen3v6/23_41_30_eval_meanflow_23777.log`

---

## 0. Headline

**The backbone dominates the objective. The DiT was the *good* backbone all along; the UNet fails both MeanFlow-family objectives on this task.**

The UNet run was launched to test whether the h-only DiT (backbone note §4) was the bottleneck behind the "terrible" α-Flow-DiT MSE. **The opposite is true:** swapping the DiT for the UNet makes *both* α-Flow and MeanFlow **3–7× worse on planning success** and collapses the raw field to ~noise scale. The DiT's ugly-looking training curve was the *best* of the four cells.

## 1. The 2×2 (objective × backbone), seed 6

### 1.1 Planning — `avoiding-d3il`, DPCC-projected, goal **and** constraints satisfied (higher = better)

| goal+constr success | **DiT** | **UNet** | DiT / UNet |
|---|---|---|---|
| **MeanFlow** (Gen3v6) | **0.49**  (11 full-1.0 blocks / 39) | 0.14  (1 / 39) | **3.5×** |
| **α-Flow** (Gen3v7) | **0.50**  (56 / 156) | 0.07  (0 / 156) | **7×** |

### 1.2 Goal reached at all (the generative brain, before projection)

| goal success | **DiT** | **UNet** |
|---|---|---|
| MeanFlow | 0.94 | 0.74 |
| α-Flow | 0.91 | **0.67** |

On the UNet the *generative field itself* often fails to reach the target (α-Flow: goal SR 0.67, and **zero** blocks achieve full goal+constraint success). On the DiT the field reaches the goal ~90 % of the time and projection recovers constraints in half.

### 1.3 Raw field quality — final train `raw_mse_u` (SUM/48) and `per_dim_rms_u`

| final per-dim RMS | **DiT** | **UNet** |
|---|---|---|
| MeanFlow | ~0.2–0.5 | **~0.98** (raw_mse_u ≈ 46, spiked to 90 mid-run) |
| α-Flow | ~0.2–0.3 (b0 ~2; best 0.11) | **~0.96** (raw_mse_u ≈ 44, rising in the α=0 tail) |

**`per_dim_rms_u ≈ 1.0` means the UNet's per-dimension error equals the full normalised data scale** — the field is barely fitting. Both objectives land there. On the DiT the same objectives sit at 0.2–0.5.

## 2. What this settles

1. 🔴 **The h-only-DiT hypothesis is refuted as the bottleneck.** The DiT (h-only conditioning, backbone note §4/R1) is **far better**, not worse, than the UNet. The h-only limitation is real but second-order; it does not explain the α-Flow-DiT curve. **R1 (ablate `dit_condition_on_t`) drops in priority.**
2. ✅ **Keeping the DiT for the Gen3v4/v6/v7 A/B was the right call** — and now empirically, not just for control.
3. ⭐ **The failure is objective-independent → it is the backbone.** MeanFlow-UNet and α-Flow-UNet collapse *identically* (per-dim ~0.97, planning ~0.1). The two-time average-velocity field `u(z, τ, h)` needs the transformer's in-context h/attention tokens; the UNet's scalar-embedding h-conditioning is too weak to represent an h-dependent velocity, regardless of which target trains it.
4. ⚠️ **"The trusted 100 %-safe FM baseline used the UNet" does not transfer.** That baseline learned a *single*-time field `v(z, τ)`. The MeanFlow-family two-time field is a strictly harder representational ask, and the UNet does not meet it here.
5. 🟢 **The DiT α-Flow result is re-affirmed as the best cell** — and α-Flow-DiT edges MeanFlow-DiT on full-success *rate* (56/156 = 0.36 vs 11/39 = 0.28), consistent with the training story (§7 of the insight: the homotopy makes b1/b2 learnable).

## 3. Caveats (do not over-read)

- **Seed 6 only, n=2 trials/cell.** Granularity is 0.0/0.5/1.0. These are smell tests. The backbone effect (3–7×) dwarfs seed/trial noise, but the *within-DiT* α-Flow-vs-MeanFlow gap (0.36 vs 0.28) is within the noise and **not yet a claim**.
- **K not fully matched across objectives.** α-Flow evals swept K∈{1,2,5,10} (156 blocks); MeanFlow evals were single-K (39 blocks). The **DiT-vs-UNet contrast within α-Flow is matched** (same K set) and within MeanFlow is matched (same single K) — so the headline backbone comparison is clean. Do **not** compare α-Flow's 156-block aggregate head-to-head against MeanFlow's 39-block aggregate without normalising (done in §2.5 via full-success *rate*).
- **UNet training was unstable** (MeanFlow-UNet spiked to per-dim 1.37 / raw 90 at ~45k). Not a single bad seed — the whole run sits at ~noise scale.
- No `losses.pkl` for the UNet runs yet, so no h-stratified buckets — but at per-dim ~1.0 the buckets are moot (nothing is fitting).

## 4. Recommended next steps

1. ✅ **Drop the UNet line.** It is a dead end for this objective family; do not sweep LR/clip on it hoping to rescue it — per-dim ~1.0 on *both* objectives is a representational failure, not a tuning one.
2. 🔴 **Back to the DiT, fix the two real DiT defects** (insight §7.2/§7.3): the grad-clip saturation (short LR 5e-4→2e-4 or clip 1.0→5.0 probe) and the under-sampled/unstable few-NFE bucket b3. These are where the actual α-Flow verdict will be decided.
3. **Scale the DiT eval to all 5 seeds** and run the endpoint-error diagnostic — the matched-K safety-vs-s/plan table over seeds is still the real deliverable, and it exists only for DiT.
4. **De-prioritise backbone-note R1** (`dit_condition_on_t`) given this result; **R2** (a faithful adaLN sibling backbone) is now the more interesting architecture question, but only *after* the DiT α-Flow verdict is settled.

## 5. One-line verdict

**Backbone ≫ objective on avoiding-d3il: the DiT carries the MeanFlow-family field and the UNet cannot. The α-Flow-DiT run — ugly MSE and all — remains the best result and the only line worth continuing.**
