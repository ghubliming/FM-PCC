# Gen14 U9 — Architecture: the perception stack, and where the image enters

**Date:** 2026-08-25 · **Generation:** Gen14 (`mix_visual_aligning` ↔ `mix_visual_aligning_test`)
**Unit:** U9 (perception-first) · **Task:** D3IL aligning, visual (dual-camera)
**Evidence:** a static read of the model sources at commit `2368aa1a`, plus the U8 gate log
`temp/2208/2026-08-22/13_49_30_gates_*_24873.log` for the one measured figure this document leans on.
**Status:** 🔴 **no U9 run exists yet.** Every number below is analytic or inherited from U8.

Sibling documents: the plan is [`PLAN_Gen14_U9_perception_first_ml.md`](PLAN_Gen14_U9_perception_first_ml.md),
the implementation record is [`CHANGELOG_Gen14_U9_perception_first_ml.md`](CHANGELOG_Gen14_U9_perception_first_ml.md),
and the argument for doing perception at all is
[`../U8/ARCH_20260823_dit_vs_visual_unet.md`](../U8/ARCH_20260823_dit_vs_visual_unet.md) §13.

---

## 0. What this document is

U8's ARCH document studied **the 15 % of the model that plans a trajectory**. This one studies
**the 85 % that looks at the pictures**, and the one design decision U8 deliberately did not make:
whether the image should reach the transformer as a *token* or as *modulation*.

It answers three questions U8 never asked:

1. Where, exactly, do 22.4 M parameters go, component by component?
2. What does `use_group_norm=True` actually do to a pretrained ResNet?
3. What is the parameter cost of moving the image from the sequence into `c`? (**Answer: −320.**)

---

## 1. What U9 changes, and what it deliberately does not

| | pre-U9 (= U8) | U9 |
|---|---|---|
| Encoder architecture | `MultiImageObsEncoder`, dual ResNet-18, SpatialSoftmax(32 kp), `Linear→64` | **unchanged** |
| Encoder **weights** | random init | `vis_pretrained` → ImageNet |
| Encoder **learning rate** | `train_lr`, same as the bone | `vis_lr_scale` → `train_lr × s` |
| Where the image enters the DiT | one prepended token | `vis_cond_mode` → token \| **adaln** \| both |
| `LATENT_DIM` | 128 | **unchanged** |
| `cond_dim` reaching the bone | 128 | **unchanged** |
| Horizon, obs window, `TRANSITION_DIM` | 8, 1, 9 | **unchanged** |
| Projector, constraints, MPC fan, eval variants | — | **untouched. No control-theory code was modified.** |

Keeping `LATENT_DIM` fixed is the load-bearing choice: `cond_dim` never moves, so **every U8 gate
remains valid exactly as written** and the new gates are additions rather than rewrites.

---

## 2. The perception stack, component by component

Counted analytically from the sources (no torch in this container). The ResNet-18 trunk reproduces the
canonical **11,176,512** (torchvision `resnet18` minus its 513,000-parameter `fc`), which is the
arithmetic check that the rest of the column is trustworthy.

### 2.1 One camera tower — 11,197,088

| Component | Shape | Params | Share |
|---|---|---:|---:|
| stem `conv1` 7×7 s2 + norm | 3 → 64 | 9,536 | 0.09 % |
| `layer1` — 2 BasicBlocks | 64 → 64 | 147,968 | 1.3 % |
| `layer2` — 2 BasicBlocks (+downsample) | 64 → 128 | 525,568 | 4.7 % |
| `layer3` — 2 BasicBlocks (+downsample) | 128 → 256 | 2,099,712 | 18.8 % |
| **`layer4` — 2 BasicBlocks (+downsample)** | 256 → 512 | **8,393,728** | **75.0 %** |
| `SpatialSoftmax` — `Conv2d(512, 32, 1)` | 512 → 32 kp | 16,416 | 0.15 % |
| `Linear` (feature_dimension) | 64 → 64 | 4,160 | 0.04 % |
| **Total per camera** | | **11,197,088** | |

`share_rgb_model=False`, so there are **two independent towers**: `agentview_image` and
`in_hand_image` share no weights.

### 2.2 The whole visual arm

| | Params |
|---|---:|
| Two camera towers | **22,394,176** |
| `mf_dit` bone, `vis_cond_mode='token'` | 4,036,658 |
| **Total `velocity_net`** | **26,430,834** |

**Encoder share of the trainable model: 84.7 %.**

> ⚠️ **Minor correction to U8 ARCH §1.** That table gives the encoder as "≈ 22.36 M", obtained by
> subtracting two already-rounded gate-log figures (26.4 − 4.04). The encoder is **22,394,176
> (22.39 M)**. The conclusion it supported — ~85 % of the model is perception — is unaffected, and
> `26,430,834` rounds to the gate log's `26.4 M` exactly.

### 2.3 The two facts that matter

**(a) 75 % of a camera tower is `layer4`, and `layer4` runs at 3×3.** `ResNet18Conv.output_shape`
is `ceil(96/32) = 3` (`base_nets.py:535-537`), so the trunk emits `512 × 3 × 3`. Three quarters of
the perception budget is spent mixing 512 channels over **nine spatial positions**.

**(b) The information bottleneck is 32 keypoints over a 3×3 grid.** SpatialSoftmax takes a
`512×3×3` map to 32 (x, y) pairs — 64 numbers — then `Linear(64, 64)`. Per camera, **11.2 M
parameters produce 64 numbers**, and the softmax that localises each keypoint has nine cells to
choose between. Sub-cell precision comes only from the softmax's weighted average.

This is the sharpest structural finding in this document, and it is the mirror image of U8 §5. There,
~80 % of the U-Net bone ran on a length-1 *temporal* sequence. Here, 75 % of the encoder runs on a
3×3 *spatial* grid. **Both halves of this model spend most of their parameters at a resolution where
the thing they are named for has already collapsed.**

---

## 3. What `use_group_norm=True` does to a pretrained trunk

`MultiImageObsEncoder` runs, at `multi_image_obs_encoder.py:62-69`:

```python
replace_submodules(root_module=this_model,
                   predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                   func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16,
                                               num_channels=x.num_features))
```

Per camera that is **20 `BatchNorm2d` layers** (1 stem + 4 in `layer1` + 5 in each of `layer2/3/4`,
the extra one per stage being the downsample branch) — **40 across both towers**.

| What is replaced | Per camera | Both |
|---|---:|---:|
| Normalisation layers | 20 | **40** |
| Affine parameters discarded (γ, β) | 9,600 | 19,200 |
| Running statistics discarded (μ, σ²) — buffers | 9,600 | 19,200 |

**19,200 parameters is 0.086 % of the trunk — and it is the 0.086 % that makes the other 99.91 %
numerically usable.** A pretrained conv stack whose normalisation has been re-initialised to
γ=1, β=0 with no running statistics does not produce pretrained *activations*; it produces pretrained
*filters* feeding a mis-scaled normaliser. The filters are the transferable part and they survive
intact, so this is not fatal — robomimic and `diffusion_policy` both do exactly this and it works —
but it has one concrete architectural consequence:

> **A hard freeze is the weakest form of the pretrained-encoder idea.** Freezing locks in the
> decalibration and gives the network no way to recover the scale. This is why U9's headline knob is
> `vis_lr_scale=0.1` (adapt slowly) rather than `mf_freeze_vision_encoder=True` (do not adapt), and
> why R5/R6 exist to measure the freeze case rather than assume it.

`imagenet_norm=True` is already on, so the *input* statistics do match the pretrained distribution.
The mismatch is internal, not at the input.

---

## 4. Where the image enters — the decision U8 left open

U8 ARCH §4 documented three mechanisms. U9 adds the fourth, which is the transformer analogue of the
best-scoring one.

| Bone | Mechanism | Reaches a trajectory token via | Score (U8, pooled) |
|---|---|---|---:|
| U-Net **v1** | `cond_mlp(latent)` concatenated onto `t`, widening `embed_dim` 32 → 64 | an **additive bias in every residual block**, unconditionally | **0.3425** |
| U-Net **v2** | `film_proj(cond) → (γ, β)`; `(1+γ)·f + β` | a **multiplicative gate in every block**, unconditionally | 0.3661 |
| DiT `token` | `vis_token + vis_projector(latent)` prepended | **one attention read per block**, which the model must learn to make | 0.3959 |
| **DiT `adaln`** (U9) | `c += vis_projector(latent)` | **adaLN scale/shift on every token in every block**, unconditionally | **never run** |

The ordering of the first three is the whole argument. Both U-Net variants condition
*unconditionally and everywhere*; the DiT conditions *through an attention budget it has to learn to
allocate*. `adaln` is the first transformer configuration in this repo that conditions the way the
winning U-Net does.

Concretely, in `mf_dit_official_trajectory.py::forward`:

```python
c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
if self.use_visual and self.vis_cond_mode in ('adaln', 'both'):
    c = c + self.vis_projector(self._pool_cond(cond))
```

`c` then drives `adaLN_modulation(c) → (shift, scale, gate) × 2` inside **every** `DiTBlock`, and
both `FinalLayer`s. The image is no longer something the trajectory tokens must look up — it is part
of what every block is conditioned on, at every position, by construction.

This was not an oversight in U8. `DECISION_Gen14_U8_injection_choice.md` chose the token
deliberately, following `diffusion_policy`'s convention. U9 tests the other branch of that decision.

### 4.1 🔴 The comparison is parameter-free

| mode | `vis_projector` | `vis_token` | `pos_embed` rows | Bone params | vs U-Net v1 |
|---|---:|---:|---:|---:|---:|
| `token` (U8) | 20,640 | 160 | 9 | 4,036,658 | 1.0002× |
| `adaln` | 20,640 | **absent** | **8** | **4,036,338** | 1.0002× |
| `both` | 20,640 | 160 | 9 | 4,036,658 | 1.0002× |

`adaln` is **320 parameters smaller** (one token embedding, one `pos_embed` row). `both` costs
**exactly zero** extra parameters over `token`, because the modulation path reuses the same
`vis_projector`.

This matters more than it looks. U8's headline comparison was confounded — a 3.37 M RoPE DiT against
a 4.04 M U-Net, 0.84× (U8 DA §9.2). **U9's C3 varies the conditioning mechanism at a constant
parameter count of 4.04 M**, ±0.008 %. It is the cleanest architecture A/B this repo has been able to
run, and it needs no bracket caveat at all.

### 4.2 Sequence geometry

| mode | tokens | conditioning share | attention |
|---|---:|---:|---:|
| `token` | 1 + 8 = **9** | 11 % | 9×9 |
| `adaln` | **8** | **0 %** | 8×8 |
| `both` | 9 | 11 % | 9×9 |

`num_visual_tokens` is set **before** `num_tokens` is computed, so `pos_embed`, the sin-cos
initialisation table and the strip `x = x[:, self.num_visual_tokens:]` in `forward()` all follow from
the one assignment. The invariant G-B6 already asserts therefore covers all three modes for free —
which is the reason the graft could be written as pure insertions.

Compare U8 ARCH §5's finding for the RoPE bone: 16 tokens of which half were conditioning. In
`adaln`, **100 % of the sequence is trajectory.**

### 4.3 Interaction with adaLN-zero

Every `adaLN_modulation[-1]` and both `FinalLayer`s are zero-initialised
(`mf_dit_official_trajectory.py:347-357`). So in `adaln` mode the image's influence starts at
**exactly zero** and is learned upward through the same gate the timestep uses — the image and the
time signal are treated identically by construction, which is precisely what adaLN-zero was designed
for.

It also means **a step-0 measurement of "does the image matter" reads 0.0 in every mode**, for the
same reason job 24834 produced a false `G-B3 FAIL` on 2026-08-21. G-B9 therefore warms up 5 Adam
steps at lr 1e-2 before measuring, exactly as the repaired G-B3 does. This is not a workaround; it is
the only correct way to measure a zero-init network.

---

## 5. Optimisation: two groups, one schedule

| | trunk | encoder |
|---|---|---|
| Parameters | 4.04 M | 22.39 M |
| LR at `vis_lr_scale=1.0` (pre-U9) | `train_lr` | `train_lr` |
| LR at `vis_lr_scale=0.1` (U9 headline) | `train_lr` | `train_lr / 10` |
| LR at `vis_lr_scale=0.0` | `train_lr` | 0 (frozen) |

The schedule is `get_cosine_schedule_with_warmup`, a `LambdaLR`, which multiplies **each group's own
`initial_lr`** by one shared lambda — so both groups warm up and anneal on the same curve, scaled.
There is no second schedule and no second warmup.

**At `vis_lr_scale = 1.0` the two-group code does not execute at all.** The optimiser object, its
group count and its `state_dict` layout are the pre-U9 ones. That is not tidiness: the trainer calls
`optimizer.load_state_dict()` on resume (`training_twotime.py:509-510`), a one-group checkpoint
cannot load into a two-group optimiser, and this pipeline auto-resumes near the 24 h wall routinely.

`logs["lr"]` reports group 0 (the trunk), so `logs["lr_vis"]` is emitted whenever the split is live.
Without it the encoder rate would be absent from every log and every `lr_history` plot — the classic
shape of a knob that silently did nothing.

---

## 6. Checkpoint compatibility

| Across | Interchangeable? | Why |
|---|---|---|
| `vis_pretrained` True ↔ False | ✅ **yes** | only the initial values differ; every shape is identical |
| `vis_lr_scale` values | ✅ **yes** | an optimiser-side knob; the model `state_dict` is untouched |
| `token` ↔ `both` | ✅ yes | identical parameter set |
| `token`/`both` ↔ **`adaln`** | ❌ **no** | `vis_token` is absent and `pos_embed` is `(1, 8, D)` not `(1, 9, D)` |
| U8 bone ↔ U9 bone at defaults | ✅ **bit-identical** | G-B9(a) asserts state_dict equality |

All three knobs are **path keys**, emitted only when non-default (`_mix_u9_keys()` returns `{}` at
the defaults), so incompatible checkpoints can never land in the same directory — and a
default-valued U9 run resolves to the *same* path as U8, which is what makes the R0 reproduction
check meaningful rather than a new tree that trivially passes.

---

## 7. Compute

Nothing in U9 changes the cost profile materially:

| | change | why |
|---|---|---|
| Forward FLOPs | ≈ 0 | the encoder dominates and is untouched; `adaln` removes one of nine tokens |
| Backward FLOPs at `vis_lr_scale=0.0` | **lower** | gradients still flow but no Adam state is updated for 22.39 M parameters |
| Optimiser memory at `vis_lr_scale=0.0` | ~**−170 MB** | Adam holds two moments per trainable tensor; 22.39 M × 2 × 4 B |
| Wall-clock | ≈ U8 | U8 measured ~12 h 50 m for 80 k steps on the RoPE bone |

The interesting cost is not FLOPs. It is that **22.39 M parameters are currently being fitted from
900 episodes**, and `vis_lr_scale` is the knob that changes how much of that fitting happens.

---

## 8. What is measured, and what is not

**Measured (analytic, this document):** every parameter count in §2 and §4.1; the trunk figure
reproduces canonical `resnet18` exactly, and the §2.2 total reproduces the U8 gate log's `26.4 M`.

**Measured (U8, inherited):** the pooled distances in §4, single seed, `n = 30`/cell, train split.

**Not measured — no U9 run exists:**

* whether ImageNet initialisation survives the GroupNorm surgery usefully (§3);
* whether `adaln` beats `token` (§4) — the whole point of the unit;
* whether the 128-D latent is informative at all, before or after C1/C2. This is what
  `probe_latent_informativeness.py` exists to answer, and it runs on the U8 checkpoint that already
  exists;
* whether any of it leaves the 0.29–0.47 m band.

---

## 9. Predictions, stated before the runs

Recorded so they can be wrong in public.

1. **`adaln` beats `token` on the DiT.** The mechanism ordering in §4 is the argument, and §4.1 makes
   it a clean test. Confidence: moderate. This is U9's most likely genuine finding, and it is
   *architectural* — it survives whatever happens to perception.
2. **ImageNet init helps less than a fresh reader expects.** §3 (40 re-initialised norms) and §2.3
   (a 3×3 map, 32 keypoints over nine cells) both cap the ceiling. If the probe stays low after C1,
   the suspect is **input resolution**, not the weights — and that is U10, not a U9 patch.
3. **The band holds.** ARCH §13.3(a): three trunks, two engines, two projector arms, eleven variants,
   all inside 0.29–0.47 m. U9 attacks the largest thing upstream of the bone, but a 400-step cap that
   nothing ever reaches is not obviously a perception problem.

If 1 holds and 3 holds, U9 still produced a real architectural result and a probe number explaining
why the result did not translate. That is the outcome this unit was designed to make possible.

---

## 10. Next

1. **[probe]** `probe_latent_informativeness.py` on the U8 checkpoint. Minutes, no training. It is the
   only item here that can invalidate C1/C2 before a GPU-week is spent.
2. **[gates]** `gates_mix_visual.sh bone` — nine gates, ~1 min. G-B9(a) is the additivity proof;
   G-B11 is the did-the-weights-load proof.
3. **[train]** R4 (matched bone, defaults) — ARCH §11.1's clean claim, and R1's baseline.
4. **[train]** R1 (pretrained + slow encoder + adaln) — the headline.
5. **[train]** R6 (frozen, **random**) — the control that separates "pretrained weights help" from
   "fewer trainable parameters help". Not optional.
6. **[train]** R3 (`adaln` alone, defaults elsewhere) — the parameter-free conditioning A/B of §4.1,
   and the one run whose result stands independently of the perception story.
7. **[U10, not U9]** input resolution, the observation window, and the 400-step harness ceiling.

---

## 11. One line

**U9 leaves the trajectory bone alone and moves on the 84.7 % of the model that looks at the
pictures** — 22,394,176 parameters of from-scratch ResNet whose `layer4` spends three quarters of the
budget on a 3×3 grid and whose 40 normalisation layers are re-initialised the moment
`use_group_norm=True` runs — while settling, at **zero parameter cost**, the one conditioning
question U8 deliberately left open: whether a transformer should read the image through attention or
be modulated by it.
