# DA — Gen16 `mf` visual avoiding, first cluster results

**Date:** 2026-08-23 · **Batch:** `batch_avoiding_combined_20260823_085253` (169 candidates, 3 roots)
**New:** `logs/avoiding-d3il-visual-mix/plans/mix_visual_avoiding_mf/…` — Gen16's first evaluated cell,
written 2026-08-22 20:47.

```
H8_Dmix_visual_avoiding.models.visual_mf_diffusion.VisualMeanFlow
     _a1.5_b1.0_aw1_VTrue_steps200_bs64_filmv1_Emf_tslogit_normal
  └─ H8_K2_Meuler_T0.5_A0.5_D…VisualMeanFlow_VTrue_mpc4_filmv1_Emf
```

**Bone = U-Net.** No `_B..` fragment in the path, and `_filmv1_` is present — `_mix_bone_keys()`
emits `film_mode` only on the U-Net branch and `ml_bone` only on DiT/SiT
(`config/avoiding-d3il-visual-mix.py:505-524`). So this is the **architecture-matched row**, the one
that can be compared to the U-Net baseline without a backbone confound.

**Sample:** seed **6 only**, `n_trials = 30` (per-seed S&C lands on multiples of 1/30) → **30 episodes
per cell**. The pinned DPCC target is 5 seeds × 20 = 100. `s/ep = n_steps × avg_time`.

---

## 1. It ran, completely — that is the first result

All **13 variants × 3 halfspaces**, arms A, B **and C**:

| | Gen16 mf-visual | Gen9 visual tree | state MeanFlow |
|---|---|---|---|
| arm A `diffuser` | ✅ | ✅ | ✅ |
| arm B `dpcc-{c,r,t}[-tightened]` | ✅ | ✅ | ✅ |
| **arm C `hardflow_new-*`** | ✅ **first time on a visual task** | ❌ never run | ✅ |
| NFE instrumented | ✅ (`nfe_total`, `nlp_solves/failures`) | ❌ | ✅ |
| candidate-fan parity | ✅ `hf_batch_size = 4` in **both** arms B and C | n/a | ⚠️ `B1` — arm C ran at fan **1** |

The two-time engine trains under the JVP, conditions on the pre-encoded `visual_latent`, drives the
gym loop through `VisualPolicy`, and feeds both the DPCC projector and the HardFlow in-loop NLP. The
whole Gen16 chain is exercised. **B4_PARITY holds here and does not hold in the state MeanFlow K2 run
it is being compared against** — see §5.

## 2. Every failure is a task failure, never a violation

In all nine `*-tightened` cells, `total_violations = 0.000` and **S&C is exactly equal to
`n_success`**:

| variant | TL | TR | BH |
|---|---|---|---|
| `dpcc-c-tightened` | 1.00 / 1.00 | **0.77 / 0.77** | 1.00 / 1.00 |
| `dpcc-t-tightened` | 1.00 / 1.00 | **0.83 / 0.83** | 1.00 / 1.00 |
| `hardflow_new-c-tightened` | 1.00 / 1.00 | **0.87 / 0.87** | 1.00 / 1.00 |

*(S&C / n_success, 30 episodes each)*

Nothing the tightened projector emits ever violates a constraint. Where Gen16 loses, it loses because
the **policy does not reach the goal**, and only on `top-right-hard`. That splits the diagnosis
cleanly: the constraint layer is correct on visual; the generative model has a mode-coverage hole.

The **untightened** arms are a different story — on `top-left-hard` every one of them scores S&C 0.00
at `n_success` 1.00 with residual violation 0.84–0.98. The base projector does not close on TL at all;
tightening is not an optimisation here, it is load-bearing.

## 3. Against the pinned baseline — dominant on two geometries, beaten on one

`dpcc-c-tightened`, vs **DPCC K20 aw10, 20 trials** (`H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials`):

| | S&C TL | S&C TR | S&C BH | steps TL/TR/BH | **s/ep TL/TR/BH** |
|---|---|---|---|---|---|
| DPCC K20 (target) | 1.00 | **0.95** | 1.00† | 70.0 / 77.6 / 59.4 | 39.1 / 40.2 / 36.5 |
| **Gen16 mf-visual K2** | 1.00 | 0.77 | 1.00 | **55.9 / 62.4 / 52.0** | **1.7 / 2.0 / 2.3** |

† the target's BH cell is seed-6-only (carried over from the 2026-08-19 DA).

* **top-left and both-hard: strict Pareto dominance.** Equal S&C, *fewer* control steps, *and* lower
  time per step — all three axes. **23× and 16× lower wall-clock per episode.**
* **top-right: a real loss.** 0.77 vs 0.95. Gen16's best variant there is `hardflow_new-c-tightened`
  at 0.87, still below the target.

Worst-halfspace, the number that decides it: **0.77 (Gen16, arm B) / 0.87 (Gen16, arm C) vs 0.95
(target)**. So the honest verdict is **non-dominated, not "better"** — a large compute win against a
success deficit concentrated in one geometry. Per the Gen16 §9 reporting rule this is a
**trade-off**, and calling it a win would be wrong.

## 4. Against its own comparators

`dpcc-c-tightened`. Steps and s/ep are Gen16's consistent strength.

| model | seeds × trials | S&C TL/TR/BH | steps TL/TR/BH | s/ep TL/TR/BH |
|---|---|---|---|---|
| **Gen16 mf-visual K2** (U-Net) | 1 × 30 | 1.00 / **0.77** / 1.00 | **55.9 / 62.4 / 52.0** | **1.7 / 2.0 / 2.3** |
| state MeanFlow K2 A0.5 B1 (U-Net) | 5 × 20 | 0.99 / 0.93 / **0.86** | 95.0 / 100.2 / 98.8 | 2.6 / 2.7 / 2.7 |
| state FM K2 | 5 × 20 | 1.00 / 1.00 / 1.00 | — | 1.8 / 1.9 / 1.9 |
| Gen9 visual DPCC K20 | 5 × 2 | 0.70 / 1.00 / 1.00 | 67.2 / 68.0 / 63.8 | 51.6 / 34.1 / 37.2 |
| Gen9 visual FM K20 filmv1 | 1 × 2 | 1.00 / 1.00 / 1.00 | — | 26.6 / 32.8 / 136.0 ⚠️ |

* **vs its state twin (MeanFlow K2):** genuinely mixed, not a uniform degradation. Gen16 is *better*
  on both-hard (1.00 vs 0.86) and *worse* on top-right (0.77 vs 0.93), and it reaches the goal in
  **~40 % fewer control steps** on every geometry (52–62 vs 95–100). Adding vision did not simply
  cost accuracy here.
* **vs Gen9 visual DPCC** (the architecture-matched *visual* baseline): wins TL (1.00 vs 0.70), loses
  TR (0.77 vs 1.00), ties BH — at **17–30× less wall-clock**. Gen9's cells are 2 trials, so its TR
  1.00 is 10 episodes and carries little weight.
* **vs state FM K2 — Gen16 does not clear it.** 1.00 across all three at comparable s/ep. Per the
  benchmark hierarchy, mf/af must beat naive FM; on this evidence **it does not**.

## 5. Arm C: beats arm B on the failing geometry, at 3.5× the cost

Within Gen16, on `top-right-hard`: `hardflow_new-c-tightened` **0.87** vs `dpcc-c-tightened` **0.77**
(2.0 → 7.0 s/ep, 24 300 NFE / 8 100 NLP solves / 40 failures). Elsewhere the two tie at 1.00 and arm C
is simply 3.5× more expensive.

That is a **cost-for-success trade**, not the claim the hierarchy asks for: HardFlow has to beat the
DPCC projector **at a lower projection threshold**, and both ran here at `hf_act_threshold = 0.5`. A
threshold sweep is what would make this result mean something.

⚠️ **Do not compare Gen16's arm-C cost to the state MeanFlow K2 run's.** That run is tagged `B1` — its
arm C ran at candidate fan **1** while its arms A/B ran at fan 4 (`hf_batch_size` in the CSV: 1 vs 4).
Gen16 runs fan 4 in both. The NFE totals show it: 19–24 k (Gen16, fan 4) vs 3.9–4.3 k (state, fan 1).
The state run's arm C looks ~5× cheaper because it is doing ~5× less work — the exact B4_PARITY defect
Gen16 was built to avoid.

## 6. Confounds, in the order they matter

1. **One seed.** 30 episodes on seed 6. Seed variance is *unmeasured*, and the comparators carry 5
   seeds. The state MeanFlow K2 spread across seeds on TR is the kind of thing that could swallow the
   0.77-vs-0.95 gap whole, or confirm it. **Nothing in §3 survives contact with a second seed
   without being re-checked.**
2. **`aw1` vs `aw10`.** Gen16's mf block inherits `action_weight = 1` from `_PARENT_FM`
   (`config/avoiding-d3il-visual-mix.py:284`, "Gen7's value, not Gen6V4's"). *Every* comparator in §3
   and §4 was trained at `aw10` — the DPCC target, state MeanFlow (`…_aw10_objmeanflow_bbunet…`), state
   FM. This is a training-config mismatch on the loss term that weights exactly the action dims the
   task is scored on, and it is a plausible cause of the TR deficit. **This is the confound I would
   remove first**, and it is cheaper than a seed sweep.
3. **No Gen16 `diffusion` arm.** The rebuild has no internal control: nothing here proves Gen16
   reproduces Gen9's visual DPCC row, so a Gen16-specific regression and a genuine mf property are
   currently indistinguishable.
4. **Gen9 comparators are `n_trials = 2`** (10 episodes/cell). Their 1.00s are weak.
5. K=2 vs K=20 is *not* a confound to fix — each model is at its own operating point, which is the
   point of the comparison. But it does mean "Gen16 is 23× faster" is a statement about mf-at-K2, not
   about vision.

## 7. What to run next, in order

1. **Retrain mf at `aw10`** and re-eval. Single largest confound, one training run, directly targets
   the only cell Gen16 loses.
2. **Seeds 7–10.** Turns every number above from indicative into reportable.
3. **The `diffusion` arm.** The missing control (§6.3). Must reproduce Gen9's visual DPCC row.
4. **`af`**, the other engine Gen16 exists to test — still zero data.
5. **`hf_act_threshold` sweep on arm C**, so §5 can answer the HardFlow-vs-DPCC question properly.
6. **Diagnose `top-right-hard` specifically.** It is the only geometry where anything fails, the
   failures are goal-reaching rather than violations (§2), and `n_success` on the *unguided* arm is
   0.73 there vs 1.00 elsewhere — so the hole is in the generative model, before any projection.

## 8. One-line summary

**Gen16's visual mf pipeline works end-to-end including arm C, is 16–23× cheaper per episode than the
pinned DPCC baseline, matches it on two of three geometries and loses the third (0.77 vs 0.95) — on
one seed, at `aw1` against `aw10` comparators, and without clearing naive state FM.**
