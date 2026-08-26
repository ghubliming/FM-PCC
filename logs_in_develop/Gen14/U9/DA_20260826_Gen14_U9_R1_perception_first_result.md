# Gen14 U9 — R1 result: the perception-first stack, measured

> **DA 2026-08-26.** First and so far only U9 run to reach eval.
> Companion to [`PLAN`](PLAN_Gen14_U9_perception_first_ml.md) (what was proposed),
> [`ARCH`](ARCH_20260825_perception_stack_and_conditioning_path.md) (what the parameters are) and
> [`CHANGELOG`](CHANGELOG_Gen14_U9_perception_first_ml.md) (what was built, and the three cluster
> failures on the way).
> Reporting conventions inherited from
> [`SNAPSHOT_20260823_visual_aligning_env_status.md`](../../../Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260823_visual_aligning_env_status.md) — §0 restates them and adds one.

**Run:** jobs 25045 (pipeline) → 25046 (gates) → 25047 (train) → 25048 (eval), all on `i6-gpu-1`,
git rev `6c2df73`. **Batch:** `batch_va2_20260826_142750` (DA_VA_v2, 19 candidates / 353 units /
0 failed). **R1** = `mf` engine, `ml_bone=mf_dit`, `vis_pretrained=True`, `vis_lr_scale=0.1`,
`vis_cond_mode=adaln`, seed 6, 100 k steps. It is **candidate 13** in that batch.

> ### The answer up front
>
> **U9-R1 did not work, and the one thing that looked like it worked is a measurement artifact.**
>
> 1. **The machinery is correct.** 17/17 gates PASS, all three knobs verifiably reached the model,
>    the path key rendered as designed, no U8 or pre-U8 path changed. §2.
> 2. **No instrument improved.** Goal success 1/320. Best legal rollout **0.1302 m** against
>    0.1208 m (U8 DiT) and **0.0258 m** (`mf` U-Net v1) on the same contexts. Constraint-clean
>    tail at or below both comparators everywhere. §4.
> 3. 🔴 **The apparent constraint win is truncation.** U9-R1 reads `0-viol` = 1.00 on three
>    tightened arms — but it is the only candidate in the batch evaluated *with* the Div_Abort
>    guard (added 2026-08-23 20:09, after every comparator's eval), and **25 % of its arm-B
>    rollouts are cut short by it.** Filter all three candidates to guard-surviving rollouts and
>    **all three read 1.00**. The win disappears. §5.
> 4. 🔴 **And so does the "lowest tracking error in the batch".** `MaxPhysErr` = 0.1599 is the
>    lowest of 19 candidates because the guard *truncates the rollout at the moment tracking
>    error crosses 0.25 m*. Diverged rollouts: mean 0.2520. Full-length: 0.1074. §5.2.
> 5. **The one statistically supported effect is negative** — on `dpcc-c`, U9-R1 is **+0.11 to
>    +0.14 m worse**, paired, **0 of 10 contexts won**, t(9) = 2.57 / 3.48. §4.3.
> 6. 🔴 **R1 cannot be attributed.** It moved three knobs at once and **its matched-bone control
>    (R4) has never been run.** Candidate 12 is a *different bone* (`dit`, not `mf_dit`) at a
>    *different budget* (80 k, not 100 k). There is no clean control in this batch. §7.

---

## Reporting rules

Rules 1–6 are inherited verbatim from the 2026-08-23 env snapshot; **rule 7 is new and is the one
that matters most here.**

🔴 **1. No aggregation across projectors.** The unit is the **cell** =
`(model × projector × geometry × split)`, each with its own `n`.

🔴 **2. Model-vs-model comparisons use each model's OWN best projector, and always name it.**
Best projector is selected on the constraint-clean tail, not raw distance.

🔴 **3. Arm C (HardFlow) is never mixed into an arm B (DPCC) result.** R1 has no arm C.

🔴 **4. Every distance claim is constraint-checked.** A short rollout that clipped an obstacle is
not a result. `min (clean)` and `<15cm clean` are the numbers that count.

🔴 **5. Goal success is not used for ranking.** It is at the floor across the whole environment.
Here it is quoted once, in §4.1, purely to say that it did not move.

🔴 **6. Every row carries `n`.** The env snapshot's main tables are n = 30 cells only. **R1 is
n = 10** — see rule 7.

🔴 **7. NEW — R1's n = 10 is a paired prefix, and its rollouts were measured under a different
guard.** Two separate facts, both load-bearing:
  - **(a) Pairing is fine.** R1's 10 rollouts are byte-identical in initial box pose to contexts
    0–9 of Train-30 (verified on `context_box_init_xy_{x,y}` and `context_box_angle_deg`). So R1
    *can* be compared to the n = 30 candidates — **paired over those 10 contexts**, which is how
    every comparison below is run. It is not an unpairable orphan.
  - **(b) The measurement regime differs.** R1 is the **only** candidate in this batch whose eval
    ran with `ALIGN_DIVERGENCE_ABORT` (commit `1288118a`, 2026-08-23 20:09 UTC). Candidate 12's
    eval finished at 2026-08-23 00:43, candidates 15/16 in early August. **Their `0` divergence
    counts mean the guard did not exist, not that they never ran away.** Any statistic sensitive to
    episode length — `0-viol`, `viol`, `n_steps`, `MaxPhysErr` — is therefore **not directly
    comparable** between R1 and anything else in the batch. §5 handles this explicitly; nothing
    outside §5 quotes those four fields cross-candidate without the filter.

---

## 0. Definitions — what "good" means here

### 0.1 Metrics

Inherited from the env snapshot. Direction column is which way is better.

| symbol | field | meaning | dir |
|---|---|---|---|
| **`dist`** | `mean_dist_per_rollout` | The D3IL aligning metric, cell mean. | lower |
| **`0-viol`** | `constraint_exec_zero_violation` | Fraction of rollouts with **zero** constraint violations at every step. | higher |
| `viol` | `constraint_exec_total_viol_count` | Mean violating steps per rollout. | lower |
| **`min (clean)`** | derived | Lowest `dist` **among zero-violation rollouts only** — the best result the controller actually achieved *legally*. `— none` = no clean rollout exists. | lower |
| `min (any)` | derived | Lowest `dist` ignoring constraints. **Never quote alone.** | lower |
| **`<15cm` clean** | derived | Fraction of the cell **both** under 0.15 m **and** zero-violation. **The primary ranking metric.** | higher |
| `ms` | `avg_time_ms` | Wall-clock ms per control step, including the projection solve. | lower |
| `still` | derived | `abs(final_xy_dist − init_xy_dist) < 0.02` — the box never moved. | lower |
| **`div`** | `divergence_aborted` | Episode cut short by the Div_Abort guard. **New in this DA.** | lower |

**Do-nothing reference: `dist` = 0.3985 m.** Measured over the 5 418 rollouts in the 08-23 batch
where the box moved < 2 cm. A cell at ≈ 0.40 did nothing. *(The 0.4547 m figure quoted in earlier
U8/U9 documents is initial box→target xy distance — a different quantity. Do not use it as the
`dist` reference; this DA does not.)*

### 0.2 The definition of good, stated once

Ranked, and in this order:

1. **A short rollout that violated nothing.** `min (clean)` and `<15cm clean`. A 6 mm rollout that
   clipped an obstacle is not a result; neither is a clean rollout that sat at 0.40 m.
2. **A high `0-viol` at a low `ms`** — constraint satisfaction is the part of this stack that
   already works (env snapshot §11.1: `0-viol` = 1.00 at 42 ms), so a new model must not lose it.
3. **`dist` below the 0.3985 do-nothing line**, paired over shared contexts, with a sign count.
4. **Goal success is not an instrument** (rule 5). Every deployable cell in this environment scores
   0–2 of 30.

**And one negative criterion, which R1 is the first candidate to fail:** *the episode must survive
to the horizon.* A run that trips the guard has not produced a measurement — it has produced a
truncation that flatters every length-sensitive statistic it is scored on.

### 0.3 Projectors

| class | variants | role |
|---|---|---|
| **Arm B — DPCC** | `dpcc-r` · `dpcc-c` · `dpcc-t` · `post_processing` | ✅ reported |
| **Arm A — reference** | `diffuser` (unguided) | ⚪ no-projector control |
| **Study-only — excluded** | `geo_free` · `bounds_free` · `model_free` + pairs · `gradient` · `dpcc-c-dt{0p25,0p5,2p0,4p0}` | ❌ not controllers |

> ⚠️ R1's **single** goal success in the whole eval lands in `geo_free-bounds_free` — a study-only
> ablation with two constraint classes *deleted*. Under rule 5 and this table it is worth nothing,
> and it is the reason the batch's `candidates_ranking.csv` shows R1 with a non-zero `goal` and a
> zero `goal+constraint`.

### 0.4 The pool

**Train-10** — contexts 0–9 of Train-30, seed 6, verified identical initial box pose across
candidates 12, 13, 15, 16. Every comparison in §4–§5 is **paired over these 10**. No test split
exists for any Gen14 candidate; everything here is on **seen training contexts**.

---

## 1. What U9 actually changed — the setup

U9 is **three independent ML-side knobs and nothing else**. No control-theory code, no horizon, no
projector, no eval path was touched. All three default to the pre-U9 value, and at their defaults
the config emits `{}` — so every pre-U9 path key, checkpoint and results folder is untouched. That
additivity was verified three ways before submission (§2).

### 1.1 The problem being attacked

From the U9 ARCH doc: `velocity_net` on the `mf_dit` bone is **26,430,834** parameters, of which
the dual ResNet-18 encoder is **22,394,176 — 84.7 %**. `layer4` alone is 8,393,728 per tower
(**75 % of a tower**), and it runs on a **3 × 3** feature map. All of it is fitted from scratch to
**900 episodes**. The hypothesis: the bottleneck is encoder data-efficiency, not the generative
objective or the bone.

### 1.2 Knob 1 — `vis_pretrained` (ImageNet init)

`config → VisualDiTTwoTime → MultiImageObsEncoder → get_resnet(pretrained=)` →
`robomimic VisualCore` → `torchvision.resnet18(weights=IMAGENET1K_V1)`.

`d3il/agents/models/vision/model_getter.py` gained one backwards-compatible kwarg
(`pretrained: bool = False`); `visual_unet_twotime.py` and `visual_dit_twotime.py` pass it through.
**`visual_unet.py` was deliberately not touched** — it is a G0 `VERBATIM` file.

> 🔴 **The caveat that was written into the code comment before the run and is now relevant.**
> `use_group_norm=True` replaces all **40** `BatchNorm2d` layers with fresh `GroupNorm`, discarding
> 19,200 affine parameters *and* the matching running statistics. **An ImageNet-initialised encoder
> therefore arrives decalibrated** — it keeps the convolution filters and loses the normalisation
> that made them work. This is why knob 2 is a *scale* and not a *freeze*.

### 1.3 Knob 2 — `vis_lr_scale` (two-group optimiser)

`mix_visual_aligning/utils/training_twotime.py:181`. At `vis_lr_scale == 1.0` **nothing runs** and
the optimiser is byte-for-byte the pre-U9 single-group object — deliberately, because
`_restore_optimizer_state()` loads pre-U9 checkpoints and a 1-group state_dict cannot enter a
2-group optimiser. Above/below 1.0 the optimiser is rebuilt with the encoder in its own group.

Resolution goes through the wrapper's own `_visual_backbone()` helper rather than a hard-coded
attribute chain — see §2.3 and the CHANGELOG §6b.6 for why that mattered.

### 1.4 Knob 3 — `vis_cond_mode` (where vision enters)

The U8 bone prepends the visual latent as **one token**. U9 adds the option to route it into the
**adaLN modulation** path instead:

```python
c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
if self.use_visual and self.vis_cond_mode in ('adaln', 'both'):
    c = c + self.vis_projector(self._pool_cond(cond))
```

| mode | vis token | params | vs U8 |
|---|---|---|---|
| `token` (U8 default) | 1 | 4,036,658 | — |
| **`adaln` (R1)** | **0** | 4,036,338 | **−320** |
| `both` | 1 | 4,036,658 | +0 |

A **parameter-free** A/B: `adaln` is 320 parameters *smaller* (it drops the learned `vis_token`)
and `both` is exactly equal. Whatever `adaln` changed, it did not change capacity.

Grafted as **pure insertions** into `mf_dit_official_trajectory.py` (+92/−3) and
`af_sit_trajectory.py` (+86/−3), so the G0 `GRAFTED_DIFF` pins — which pin *removed* lines — held.
`token` stays bit-identical to U8, asserted by gate G-B9.

### 1.5 How R1 was selected

R1 sets all three at once: `vis_pretrained=1`, `vis_lr_scale=0.1`, `vis_cond_mode=adaln`. That was
a deliberate "best shot first" — and it is exactly why §7 cannot attribute the outcome.

---

## 2. Verification that the setup ran as designed

This section is the good news, and it is unambiguous.

### 2.1 Gates — 17/17 PASS (job 25046)

`GB1 GB6 GB7 GB2 GB3 GB45 GB8 GB9 GB11 G0 G1 G2 G3 G4 G5 G6 G7` — all PASS, including the three
new U9 gates and the reverted G-B7. G0: 17 `VERBATIM` files byte-identical, 6 `GRAFTED_DIFF` pins
held.

### 2.2 All three knobs verifiably reached the model (job 25047)

```
[ train ] U9: vis_pretrained=1  vis_lr_scale=0.1  vis_cond_mode=adaln
[ VisualDiTTwoTime ] vis_pretrained=True  (ImageNet ResNet-18 init)
[ VisualDiTTwoTime ] vis_cond_mode=adaln  (visual tokens in sequence: 0)
[ utils/training ] U9 vis_lr_scale=0.1 — 125 trunk tensors @ 0.0002, 129 encoder tensors @ 2e-05
```

plus, from torchvision itself, `weights=ResNet18_Weights.IMAGENET1K_V1`. `lr_vis` tracked at
exactly 0.1 × the trunk LR for all 100 000 steps, both decaying to 0 on the cosine schedule.
Training ran 100 epochs / 100 k steps in ~9.7 h with no crash, no resume, no loss spike.

### 2.3 The path key rendered as predicted — R1 cannot overwrite R4

```
..._Bmf_dit_Emf_tslogit_normal_VPTrue_VLR0.1_VCadaln/6
```

`VPTrue_VLR0.1_VCadaln`, exactly as designed. This was an open question before the run (how
`watch()` renders a bool and a float) and it is now settled empirically. **R4 — the same bone with
U9 defaults off — will land in a parallel directory.** The manual check that replaced the reverted
G-B7 is therefore satisfied.

### 2.4 What it cost to get here

Three cluster failures preceded this run (jobs 25034, 25038, 25043 — CHANGELOG §6b). All three were
caught by gates or by U9's own guard, none consumed more than ~90 s of GPU time, and none reached
training. The `vis_lr_scale` guard's first catch was an error in the code that added it.

---

## 3. Roster for this DA

| C | gen | folder | engine | bone / cond | K | budget | n (Train) | guard |
|---|---|---|---|---|---|---|---|---|
| **13** | **Gen14 U9** | `mix_visual_aligning` | `mf` | **`mf_dit`, adaLN, ImageNet, 0.1× enc-LR** | 2 | 100 k | **10** | ✅ **on** |
| 12 | Gen14 U8 | `mix_visual_aligning` | `mf` | `dit`, token | 2 | **80 k** | 30 | ❌ off |
| 15 | Gen14 | `mix_visual_aligning` | `mf` | U-Net FiLM v1 | 2 | 100 k | 30 | ❌ off |
| 16 | Gen14 | `mix_visual_aligning` | `mf` | U-Net FiLM v2 | 2 | 100 k | 30 | ❌ off |

🔴 **Candidate 12 is not R1's control.** Different bone (`dit` vs `mf_dit`), different budget
(80 k vs 100 k), different conditioning. It is the *nearest available* comparator, not a matched
one. **The matched control is R4, and R4 has never been run.**

---

## 4. The result

All tables **paired over Train-10**, seed 6, arm B + arm A only.

### 4.1 Goal success — did not move

| candidate | successes / rollouts (whole eval) | rate | 95 % CI |
|---|---|---|---|
| **U9-R1** | **1 / 320** | 0.31 % | [0.06, 1.75] |
| U8 DiT | 11 / 960 | 1.15 % | [0.64, 2.04] |
| `mf` U-Net v1 | 29 / 1140 | 2.54 % | [1.78, 3.63] |
| D3IL baseline (6 seeds) | 8 / 2804 | 0.29 % | [0.14, 0.56] |

R1's single success is in a **study-only** ablation (§0.3) and does not count at all. Under rule 5
none of this ranks anything; it is here to record that the floor did not move.

### 4.2 Cell table — `combined_5-tightened`, Train-10

**Read `min (clean)` and `<15cm clean`.** Do-nothing line = 0.3985.

| cand | arm | n | `dist` | `0-viol` | `viol` | **min (clean)** | **`<15cm` cln** | `ms` | `still` | **`div`** |
|---|---|---|---|---|---|---|---|---|---|---|
| **U9-R1** | `dpcc-r` | 10 | 0.3877 | **1.00** | 0.0 | **0.1302** | **10 %** | 47.2 | 20 % | **3** |
| **U9-R1** | `dpcc-t` | 10 | 0.4225 | **1.00** | 0.0 | 0.2239 | 0 % | 41.5 | 20 % | **2** |
| **U9-R1** | `post_proc` | 10 | 0.3816 | **1.00** | 0.0 | 0.1619 | 0 % | 43.8 | 20 % | **3** |
| **U9-R1** | `dpcc-c` | 10 | 0.4804 | 0.90 | 1.7 | 0.2712 | 0 % | 75.3 | 20 % | **4** |
| U8 DiT | `dpcc-r` | 10 | 0.4233 | 0.90 | 1.9 | 0.1473 | 10 % | 59.9 | 40 % | 0* |
| U8 DiT | `dpcc-t` | 10 | 0.4344 | 0.80 | 7.0 | 0.1627 | 0 % | 54.9 | 50 % | 0* |
| U8 DiT | `post_proc` | 10 | 0.4260 | 0.90 | 1.9 | 0.1473 | 10 % | 60.1 | 40 % | 0* |
| U8 DiT | `dpcc-c` | 10 | 0.3689 | 0.90 | 0.2 | 0.1208 | 10 % | 57.3 | 50 % | 0* |
| `mf` U-Net v1 | `dpcc-r` | 10 | 0.3693 | 0.80 | 18.3 | 0.0768 | **20 %** | 46.2 | 10 % | 0* |
| `mf` U-Net v1 | `dpcc-t` | 10 | 0.3487 | **1.00** | 0.0 | 0.1394 | 10 % | 43.0 | 30 % | 0* |
| `mf` U-Net v1 | `post_proc` | 10 | 0.3676 | 0.90 | 15.1 | 0.0768 | **20 %** | 45.0 | 10 % | 0* |
| `mf` U-Net v1 | `dpcc-c` | 10 | 0.3703 | 0.90 | 9.1 | **0.0258** | 10 % | 47.3 | 50 % | 0* |

`*` = **guard was not running** (rule 7b). Not a zero.

**On the primary metric, R1 is last.** Best legal rollout 0.1302 m vs 0.1208 (U8 DiT) and
**0.0258** (`mf` U-Net v1). Clean tail 10 % at best, vs 20 % for the U-Net on two arms. Every R1
cell sits at or above the 0.3985 do-nothing line except `dpcc-r` (0.3877) and `post_processing`
(0.3816) — i.e. within noise of doing nothing.

### 4.3 The one statistically supported effect, and it is negative

Paired `dist`, R1 minus comparator; negative = R1 better.

| geo | arm | vs U8 DiT | t(9) | won | vs `mf` U-Net v1 | t(9) | won |
|---|---|---|---|---|---|---|---|
| `combined_5` | `diffuser` | −0.0059 | −0.08 | 6/10 | −0.1112 | −1.07 | 6/10 |
| `combined_5` | **`dpcc-c`** | **+0.1445** | **2.57** | **0/10** | **+0.1242** | **1.95** | 2/10 |
| `combined_5` | `dpcc-r` | −0.0062 | −0.08 | 6/10 | −0.0075 | −0.09 | 4/10 |
| `combined_5` | `dpcc-t` | −0.0552 | −0.63 | 4/10 | **+0.1114** | **2.30** | 2/10 |
| `combined_5` | `post_proc` | −0.0441 | −0.57 | 6/10 | −0.0418 | −0.44 | 5/10 |
| tightened | `diffuser` | −0.0584 | −1.35 | 8/10 | −0.0307 | −0.37 | 3/10 |
| tightened | **`dpcc-c`** | **+0.1115** | **3.48** | **0/10** | **+0.1101** | **2.25** | 2/10 |
| tightened | `dpcc-r` | −0.0355 | −0.88 | 6/10 | +0.0185 | 0.21 | 5/10 |
| tightened | `dpcc-t` | −0.0119 | −0.20 | 4/10 | +0.0738 | 1.58 | 2/10 |
| tightened | `post_proc` | −0.0444 | −1.03 | 5/10 | +0.0140 | 0.15 | 5/10 |

Everything is noise **except `dpcc-c`**, where R1 is worse against both comparators, on both
geometries, and **loses every single one of the 10 paired contexts** against U8 DiT (sign test
p = 0.002 two-sided). t(9) = 3.48 tightened.

`dpcc-c` selects the MPC candidate with **minimum projection cost**. Losing specifically there —
while `dpcc-r`, `dpcc-t` and `post_processing` are flat — says R1's plan *distribution* changed
shape: the cheapest-to-project candidate in its pool is systematically a worse candidate than it
used to be. That is a real, localised, reproducible effect, and it is the wrong direction.

### 4.4 What did not change at all

- **Orientation.** Mean |box angle − target angle| starts at **39.7°** and ends at 35–54° on every
  R1 arm — the same failure the env snapshot §9 records for every Train-30 cell. Nothing about the
  perception stack touched it.
- **`still`.** R1 leaves the box unmoved on 20 % of arm-B rollouts vs 40–50 % for U8 DiT. **R1 is
  *less* static than its predecessor** — it moves the box, to the wrong place. The "the encoder
  collapsed to a constant and the policy froze" story is *not* what happened.

---

## 5. 🔴 The measurement-regime problem — and what survives it

This is the most important section in the document. Two of R1's three apparent advantages are
artifacts of a guard that only its eval ran.

### 5.1 R1 is the only candidate measured with Div_Abort

`ALIGN_DIVERGENCE_ABORT` (`eval_mix_visual_aligning.py:652`) stops an episode when the *commanded*
end-effector runs away from the real one — `|des_c_pos − c_pos|_xy > 0.25 m` — because
`aligning_sim` integrates `pred_action + des_robot_pos` with no absolute clamp. It landed in commit
`1288118a`, **2026-08-23 20:09 UTC**. Candidate 12's eval finished **2026-08-23 00:43**; 15 and 16
ran in early August. **Their zeroes mean the guard was absent.**

Every R1 abort is `des_runaway`, all tripping at 0.251–0.257 m — i.e. the instant the threshold is
crossed, not a catastrophic blow-up.

**Retro-check.** The guard's trigger is the same XY quantity the eval already reports as
`max_physical_tracking_error` (stated in the code comment), so `MaxPhysErr > 0.25` reconstructs
what the guard *would* have done. On R1 the proxy is exact — 20 predicted, 20 actual on arm B
(125 vs 127 over all variants). Applied to the pre-guard candidates:

| candidate | arm-B rollouts | would have tripped | rate | recorded `div` |
|---|---|---|---|---|
| **U9-R1** | 80 | 20 | **25.0 %** | 20 |
| U8 DiT | 240 | 32 | 13.3 % | 0 (no guard) |
| `mf` U-Net v1 | 240 | 31 | 12.9 % | 0 (no guard) |
| `mf` U-Net v2 | 240 | 45 | 18.8 % | 0 (no guard) |
| `af` U-Net v1 | 240 | 42 | 17.5 % | 0 (no guard) |
| `diffusion` K20 | 99 | 32 | 32.3 % | 0 (no guard) |

**Command runaway is a property of the whole environment, not of U9.** R1 is roughly 2× the two
`mf` comparators and better than `diffusion` K20. It is elevated, not unique — and it is **not**
a categorical U9 regression, which is what the raw `0 vs 127` counts would have suggested.

### 5.2 The two artifacts this creates

**(a) `MaxPhysErr` = 0.1599, "lowest of 19 candidates", is meaningless.** The guard truncates the
rollout *at the moment tracking error crosses 0.25 m*, so it caps the very statistic being
reported:

| R1 arm-B rollouts | n | mean `MaxPhysErr` | max |
|---|---|---|---|
| diverged (truncated) | 20 | 0.2520 | 0.2556 |
| ran to full length | 60 | 0.1074 | 0.2436 |

**(b) `0-viol` = 1.00 is inflated.** A rollout aborted at step 99 has 301 fewer steps in which to
violate anything.

### 5.3 The apples-to-apples table — filter all three the same way

Restrict **every** candidate to rollouts the guard would not have truncated
(`MaxPhysErr ≤ 0.25` and not aborted). `combined_5-tightened`, Train-10:

| candidate | arm | n surv | `dist` | `0-viol` | **min (clean)** | **`<15cm` cln** |
|---|---|---|---|---|---|---|
| **U9-R1** | `dpcc-r` | 7 | 0.3654 | **1.00** | 0.1302 | 14 % |
| **U9-R1** | `dpcc-t` | 8 | 0.3848 | **1.00** | 0.2239 | 0 % |
| **U9-R1** | `post_proc` | 7 | 0.3566 | **1.00** | 0.1619 | 0 % |
| **U9-R1** | `dpcc-c` | 6 | 0.5002 | 0.83 | 0.2712 | 0 % |
| U8 DiT | `dpcc-r` | 9 | 0.4206 | **1.00** | 0.1473 | 11 % |
| U8 DiT | `dpcc-t` | 8 | 0.3928 | **1.00** | 0.1627 | 0 % |
| U8 DiT | `post_proc` | 9 | 0.4206 | **1.00** | 0.1473 | 11 % |
| U8 DiT | `dpcc-c` | 10 | 0.3689 | 0.90 | 0.1208 | 10 % |
| `mf` U-Net v1 | `dpcc-r` | 7 | 0.3383 | **1.00** | **0.0768** | **29 %** |
| `mf` U-Net v1 | `dpcc-t` | 9 | 0.3594 | **1.00** | 0.1394 | 11 % |
| `mf` U-Net v1 | `post_proc` | 8 | **0.3185** | **1.00** | **0.0768** | **25 %** |
| `mf` U-Net v1 | `dpcc-c` | 8 | 0.3424 | 0.88 | **0.0258** | 12 % |

🔴 **The constraint win is gone.** All three candidates read `0-viol` = 1.00 on `dpcc-r`, `dpcc-t`
and `post_processing`. R1 was never cleaner; it was measured on shorter episodes.

🔴 **And `mf` U-Net v1 dominates on the primary metric** — best `dist` (0.3185), best `min (clean)`
(0.0258) and best clean tail (25–29 %) — on the same 10 contexts, under the same filter.

**Nothing about R1 survives the filter as an improvement.**

### 5.4 One thing that does survive, stated at its true weight

Among R1's **full-length** rollouts on tightened geometry, **27 of 28 are zero-violation (96 %)**.
That is genuine — it is not truncation. But it is 28 rollouts, both comparators reach 1.00 under
the same filter, and on **untightened** geometry R1's full-length rollouts are only 3/32 clean
(≈ 9 %), *worse* than either comparator. It supports no claim.

---

## 6. Why the headline batch ranking says something different

`candidates_ranking.csv` places R1 16th of 19 with `goal+constraint` = 0.00 %, `goal` = 0.31 %,
`collision-free` = 47.8 %, 47.7 ms, `MeanDist` 0.4231, `MaxPhysErr` **0.1599 (best in batch)**.

Three of those five numbers are artifacts of rules this file applies and that file does not:

| number | why it is not usable |
|---|---|
| `goal` 0.31 % | the single success is in a **study-only** ablation (§0.3) |
| `collision-free` 47.8 % | pools all 32 variants **including study-only**, and is inflated by truncation (§5.2b) |
| `MaxPhysErr` 0.1599 | **capped by the guard** at the trip point (§5.2a) |
| `MeanDist` 0.4231 | pools study-only variants; the per-cell arm-B numbers are in §4.2 |
| 47.7 ms | ✅ real, and the one genuinely good number — see §9 |

This is the same failure the env snapshot §8 records for candidate 5's "4.55 % headline" (1 success
in 11). **The aggregate ranking is a discovery index, not a result.**

---

## 7. 🔴 What this run does and does not license

**Does:** U9's three knobs are implemented correctly, are additive, are verifiable, and one
configuration of them (`VPTrue / VLR0.1 / VCadaln`, 100 k, `mf_dit`, seed 6) produces no
improvement on any instrument and a significant regression on `dpcc-c`.

**Does not:** attribute that to any individual knob, or to U9 as a direction.

1. **R1 moved three knobs at once.** With one arm there are no marginal effects.
2. **The matched control does not exist.** R4 = `mf_dit` at U9 defaults, 100 k, seed 6 — never run.
   Candidate 12 differs from R1 in bone *and* budget *and* conditioning.
3. **R6, the init control** (`vis_pretrained=False`, everything else as R1), is unrun. Without it,
   the decalibration concern in §1.2 — ImageNet filters arriving with their BatchNorms stripped —
   is untested. **It is entirely possible that `vis_pretrained=True` is a net negative under
   `use_group_norm=True`, and R1 cannot tell us.**
4. **One seed, 10 contexts, seen data, no test split.**
5. **The comparators were measured under a different guard** (§5).

The honest one-line summary: **the perception-first hypothesis has not been tested yet. R1 tested
one point of a three-dimensional space against no control.**

---

## 8. Limits

- **n = 10**, paired to Train-30's first 10 contexts. Recording was left at `all`, which caps
  rollouts at 10; the other candidates ran 30.
- **Seed 6 only.** Pairing is across contexts, not training variance.
- **No test split** for any Gen14 candidate.
- **Budget mismatch** — R1 100 k vs candidate 12's 80 k.
- **Bone mismatch** — `mf_dit` vs `dit`.
- **Guard mismatch** — §5, the dominant one.
- **`min (clean)` is an extreme-value statistic** and R1's cells have 6–10 rollouts. Treat the
  §5.3 column as indicative.
- **No orientation-specific metric.** §4.4 uses final-vs-target box angle, which does not separate
  "never rotated" from "rotated wrongly".
- **`avg_time_ms` is wall-clock on shared GPUs.**

---

## 9. Verdict

**U9-R1 is a clean negative result with a corrected headline.**

1. **The build is correct and cheap to iterate on.** 17/17 gates, three knobs verified live, path
   keys parallel, zero pre-U9 paths disturbed, three pre-training failures caught for ~90 s of GPU
   each. §2.
2. **No instrument improved.** Best legal rollout 0.1302 m vs 0.0258 m for `mf` U-Net v1 on the
   same contexts; clean tail at or below both comparators; `dist` at the do-nothing line. §4.2.
3. 🔴 **The apparent constraint win was truncation.** Filter all three candidates identically and
   all three read `0-viol` = 1.00. §5.3.
4. 🔴 **"Lowest tracking error in the batch" was the guard capping the metric it reports.** §5.2a.
5. **The one supported effect is a regression on `dpcc-c`** — +0.11 to +0.14 m, 0/10 contexts won,
   t(9) = 3.48. The min-projection-cost selection rule now picks worse plans. §4.3.
6. **Command runaway is environment-wide, not U9's fault.** The retro-check puts every candidate at
   11–32 %; R1's 25 % is elevated ~2× over the `mf` comparators but better than `diffusion` K20.
   §5.1. **This finding is worth more than R1 itself** — it means every pre-08-23 visual-aligning
   number in the archive was measured on episodes that included runaway command integration.
7. ✅ **One real positive: 41.5–47.2 ms per control step** with the full projection solve in the
   loop — 21–24 Hz, the fastest arm-B cells in the batch, and faster than candidate 12 (54–60 ms)
   on the same arms. The `adaln` route removes a token from the sequence and it shows.
8. **Nothing is attributable.** §7.

**Order of work.**

- **(a) R4 — the matched-bone control.** `mf_dit`, U9 defaults, 100 k, seed 6. Without it R1 means
  nothing. Highest priority, and it is one job.
- **(b) R6 — the init control.** `vis_pretrained=False`, `VLR0.1`, `adaln`. Directly tests §1.2's
  decalibration concern. Explicitly not optional.
- **(c) Re-run R1's eval at 30 rollouts**, recording off, so it enters the n = 30 tables.
- **(d) 🔴 Re-run at least candidate 15's eval under the current guard.** Every cross-candidate
  `0-viol` / `MaxPhysErr` / `n_steps` comparison in the archive is otherwise regime-mixed, and this
  DA had to filter around it. **This is now a whole-environment measurement debt, not a U9 task.**
- **(e) The P2 latent-informativeness probe** (`probe_latent_informativeness.py`, unrun; no sbatch
  wrapper exists yet). If the visual latent carries no incremental R² over state onto the action
  chunk, no encoder-side knob can help and R4/R6 are answering the wrong question. **One GPU-hour
  against a GPU-day per arm** — arguably it should run before (a).
- **(f) Investigate the `dpcc-c` regression directly** — it is the only reproducible signal U9 has
  produced, and it is about plan-distribution shape, which is diagnostic.

---

## 10. Reproduction

```bash
# R1 as run (jobs 25045–25048, git 6c2df73)
MIX_BONE_MF=mf_dit MIX_VIS_PRETRAINED=1 MIX_VIS_LR_SCALE=0.1 MIX_VIS_COND=adaln \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6

# (a) R4 — matched-bone control: same bone, U9 knobs at their defaults
MIX_BONE_MF=mf_dit \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6

# (b) R6 — init control: R1 minus the ImageNet weights
MIX_BONE_MF=mf_dit MIX_VIS_PRETRAINED=0 MIX_VIS_LR_SCALE=0.1 MIX_VIS_COND=adaln \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6
```

**Analysis.** Batch `batch_va2_20260826_142750`; all per-cell numbers derive from
`per_rollout_detail.csv`. R1 = `Candidate == '13'`; comparators 12 / 15 / 16. Every comparison is
filtered to `split == 'train'` and `rollout_idx < 10`. Arm B = `{dpcc-r, dpcc-c, dpcc-t,
post_processing}`; study-only variants excluded per §0.3. Guard-survival filter (§5.3) is
`max_phys_error_per_rollout <= 0.25 and not divergence_aborted`. Context pairing verified on
`context_box_init_xy_{x,y}` + `context_box_angle_deg`.

**Cluster logs.** `temp/2608/2026-08-25/11_14_19_{gates_mix_visual_25046,
train_mix_visual_aligning_25047, eval_mix_visual_aligning_25048}.log`.
