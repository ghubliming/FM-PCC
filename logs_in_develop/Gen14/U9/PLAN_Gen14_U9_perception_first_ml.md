# Gen14 U9 — PLAN: perception-first ML upgrade

**Date:** 2026-08-24 · **Generation:** Gen14 (`mix_visual_aligning` ↔ `mix_visual_aligning_test`)
**Unit:** U9 · **Task:** D3IL aligning, visual (dual-camera) · **Status:** planned, not implemented
**Predecessor:** U8 (`ARCH_20260823_dit_vs_visual_unet.md`, `DA_20260823_…_mf_dit_visual_aligning.md`)

---

## 0. What U9 is, in one paragraph

U8 asked *"which trajectory bone?"* and got a clean answer to the wrong question: every bone lands in
the same 0.29–0.47 m band against a 0.4547 m do-nothing baseline, at the 400-step cap, below 2.1 %
success. U9 stops varying the 15 % of the model that does the task and starts on the **85 % that
looks at the pictures** — while completing the one bone experiment U8 left unrun. Four ML changes,
every one a flag defaulting OFF so U8 reproduces bit-identically, plus one new gate that turns the
central question into a *measurement* instead of an argument.

**U9 touches no control code.** Not the projector, not the constraints, not the MPC fan, not the eval
variants, not the horizon. §3 lists the untouched surface explicitly so a reviewer can check it.

---

## 1. The three findings U9 is built on

**(F1) The failure is invariant to the bone.** Pooling U7 + U8: three structurally different trunks
(U-Net v1 concat-cond, U-Net v2 FiLM, iMF RoPE DiT), two engines (`mf`, `af`), two projector arms,
two geometries, eleven variants. Every cell lands in **0.29–0.47 m**; changing the bone moves ~0.05 m
and nothing leaves the band. The bottleneck is upstream of the bone.

**(F2) 85 % of the trainable model is a from-scratch ResNet pair on 900 episodes.** The dataset is
**900 episodes** (`config/aligning-d3il-visual.py:422`). The model is **26.4 M trainable, of which
22.36 M is a randomly-initialised dual ResNet-18** learned end-to-end, because
`'mf_freeze_vision_encoder': False` (`config/aligning-d3il-visual.py:1308, 1348`). We are fitting two
ImageNet-scale convnets to 900 demonstrations and spending 15 % of the model on the trajectory.
**This is the single largest design risk in the stack, and it is five times bigger than the bone
question U8 studied.**

**(F3) Vision-as-modulation on a transformer has never existed here.** The adaLN bones are conditioned
on `c = t_emb + r_emb + w_emb` (`mf_dit_official_trajectory.py:263`) and the visual latent is
*"PREPENDED as one token (**not summed into adaLN's `c`**)"* (`:278`). So the 2×2 has an empty cell:

| trunk | modulation conditioning | token conditioning |
|---|---|---|
| conv U-Net | v1 concat-into-`t` **0.3425** · v2 FiLM 0.3661 | — |
| transformer | **EMPTY** | prefix token (`dit`) 0.3959 |

Our best mechanism is U-Net **v1**: fold `cond_mlp(latent)` into the embedding that drives every
block. Its exact transformer analogue is `c += vis_proj(latent)` — adaLN's native class-conditioning
path, which adaLN-zero was designed for. We built the adaLN bones and routed the image around it.
That was a deliberate, documented decision (`DECISION_Gen14_U8_injection_choice.md`), not a defect —
U9 tests the other design point.

---

## 2. The four changes

All four are independent. All default to the U8 value. Nothing is coupled.

### C1 — `vis_pretrained`: ImageNet initialisation of the encoder

**Why.** F2. Under a 900-episode budget the encoder is the wrong thing to learn from scratch.

**Mechanism.** `d3il/agents/models/vision/model_getter.py::get_resnet` hard-codes
`backbone_kwargs=dict(input_coord_conv=False, pretrained=False)`, and `base_nets.py:510` passes it
straight to `vision_models.resnet18(pretrained=…)`. Add a `pretrained: bool = False` kwarg to
`get_resnet` — **backwards-compatible by default, so no other generation changes behaviour** — and
set it from the `rgb_model` block of the encoder config.

**⚠️ Three encoder blocks must move together.** `visual_unet.py:38`, `visual_unet_twotime.py:82`,
`visual_dit_twotime.py:101` each build the same `OmegaConf` encoder spec, and the latter two carry a
🔴 *"BYTE-IDENTICAL to …, any drift silently breaks the U-Net-vs-DiT comparison"* comment. Edit all
three or the comparison U9 exists to make is void. G-B8 enforces this.

**🔴 Known interaction — the GroupNorm surgery discards part of the pretraining.**
`use_group_norm=True` runs `replace_submodules(…, isinstance(x, nn.BatchNorm2d), → nn.GroupNorm(C//16, C))`
(`multi_image_obs_encoder.py:62-69`). **Every BatchNorm2d in the pretrained ResNet is replaced by a
freshly-initialised GroupNorm** — affine parameters and running statistics thrown away. The conv
filters survive, which is where the transferable structure lives (robomimic and `diffusion_policy`
both do exactly this), but the network arrives **decalibrated**. This is the reason C2 exists and the
reason a hard freeze is *not* the headline configuration.

### C2 — `vis_lr_scale`: a separate, lower learning rate for the encoder

**Why.** C1's decalibration means the pretrained encoder needs *some* adaptation, but at 900 episodes
full-rate end-to-end training is what wastes the prior. The right answer is neither "train it" nor
"freeze it" — it is **train it slowly**.

**Mechanism.** Split the optimiser into two parameter groups in `train_mix_visual_aligning.py`:
encoder parameters at `lr × vis_lr_scale`, everything else at `lr`. `vis_lr_scale = 1.0` reproduces
U8 exactly. `0.1` is the headline. `0.0` is equivalent to freezing and is the ablation floor.

The existing `mf_freeze_vision_encoder` flag (`visual_mf_diffusion.py:29,35,51`,
`visual_af_diffusion.py:29,34,45`, `train_mix_visual_aligning.py:500`) stays as the hard-freeze
extreme; C2 is the graded version and supersedes it as the default control.

### C3 — `vis_cond_mode`: vision into the adaLN conditioning vector

**Why.** F3 — the empty cell, and the mechanism our best result already uses.

**Mechanism.** One flag on the adaLN bones, `vis_cond_mode ∈ {token, adaln, both}`:

* `token` — U8 behaviour, **bit-identical**. Default.
* `adaln` — `c = t_emb + r_emb + w_emb + vis_projector(latent)`; the prefix loses `vis_token` and its
  `pos_embed` row, so the sequence is **one token shorter** and the bone is **marginally smaller**.
  The 1.00× bracket holds; G-B2 checks it.
* `both` — token *and* modulation, the maximal-conditioning cell.

**Scope: `mf_dit` and `sit` only.** The RoPE bones (`dit`, `af_dit`) are `forward(x, cos, sin)` and
have **no adaLN pathway at all** (ARCH §4, point 3) — they are untouched by C3 and keep U8 semantics.

### C4 — `ml_bone: mf_dit` at 4.04 M, full 100 k budget

**Why.** U8's trained bone was the RoPE DiT at **3.37 M (0.84×) on an 80 % budget**. Its two headline
confounds (DA §9.1, §9.2) both die in this one run, and `mf_dit` is the exactly parameter-matched
bone (4.04 M, 1.00× against the U-Net's 4,035,666).

**Mechanism.** Zero new model code — U8 already built, gated and parameter-matched this bone
(G-B1/B2/B6/B7 pass on it); it has simply never been *trained* on aligning. It is
`MIX_BONE_MF=mf_dit` plus dropping the `_TB80pct` budget tag.

---

## 3. What U9 does NOT touch

Stated explicitly so the U8↔U9 comparison is auditable cell-for-cell.

| Untouched | Why it matters |
|---|---|
| `config/visual_aligning_eval.yaml` | same variants, same constraint classes, same thresholds, same tightening |
| `mix_visual_aligning/sampling/**` (DPCC projector, HardFlow) | **no control-theory change of any kind** |
| MPC candidate fan `B = 4`, selection rules `-r/-c/-t` | selection is a known bone-dependent confound (DA §3.4) — freezing it keeps U9 readable |
| `horizon = 8` | ARCH §5 says H is worth changing; doing it here would confound the perception result |
| `window_size = 1`, `obs_seq_len = 1` | single-frame conditioning stays — it is a real suspect (ARCH §13.3c) but belongs to U10 |
| `LATENT_DIM = 128` | keeping it fixed means `cond_dim` never moves and every U8 gate stays valid unmodified |
| encoder architecture (ResNet18Conv + SpatialSoftmax 32 kp + Linear→64), `use_group_norm`, `imagenet_norm`, `share_rgb_model` | only the *weights* change, never the shape |
| engines (`mf`, `af`), `t_schedule`, `dual_head`, EMA, batch size, seeds | U9 varies perception and one conditioning path, nothing else |

---

## 4. Implementation

| # | File | Edit | ~lines |
|---|---|---|---:|
| 1 | `d3il/agents/models/vision/model_getter.py` | `get_resnet(..., pretrained: bool = False)` → into `backbone_kwargs` | 3 |
| 2 | `mix_visual_aligning/models/visual_unet.py` | `'pretrained': vis_pretrained` in the `rgb_model` block | 3 |
| 3 | `mix_visual_aligning/models/visual_unet_twotime.py` | same, byte-identical to #4 | 3 |
| 4 | `mix_visual_aligning/models/visual_dit_twotime.py` | same, byte-identical to #3; plus forward `vis_cond_mode` to the bone | 6 |
| 5 | `mix_visual_aligning/models/mf_dit_official_trajectory.py` | `vis_cond_mode`; `adaln`/`both` add `vis_projector(cond)` into `c`; `token` unchanged | ~20 |
| 6 | `mix_visual_aligning/models/af_sit_trajectory.py` | same graft (SiT bone) | ~20 |
| 7 | `mix_visual_aligning_test/train_mix_visual_aligning.py` | two optimiser param groups for `vis_lr_scale`; forward the new keys | ~15 |
| 8 | `config/aligning-d3il-visual.py` | `vis_pretrained`, `vis_lr_scale`, `vis_cond_mode` in the mf/af blocks + `_mix_bone_keys` path-key handling | ~15 |
| 9 | `mix_visual_aligning_test/gates_mix_visual.py` | G-B8…G-B11, extend G-B2/G-B3 | ~120 |
| 10 | `Slurm_Codes/sbatch/mix_visual_aligning/*.sh` | `MIX_VIS_PRETRAINED`, `MIX_VIS_LR_SCALE`, `MIX_VIS_COND` env passthrough | ~10 |

**Path keys.** `vis_pretrained` / `vis_lr_scale` / `vis_cond_mode` must appear in the checkpoint path
**only when non-default**, exactly as `_TB80pct` and `_Bdit_` do — so a U9 tree can never be confused
with a U8 tree, and a default-valued U9 run lands on the *same* path as U8 (which is what makes the
reproduction check in §6 meaningful). Use the `_DROP` sentinel mechanism already in
`_mix_train_block`.

**⚠️ Operational: no internet on compute nodes.** `pretrained=True` triggers a torchvision download
into `~/.cache/torch/hub/checkpoints/`. Pre-fetch **once on the login node** inside the FMPCC env:

```bash
conda activate FMPCC
python -c "import torchvision as tv; print(tv.__version__); tv.models.resnet18(pretrained=True)"
ls -lh ~/.cache/torch/hub/checkpoints/
```

The version print matters: `pretrained=` was removed in torchvision 0.15. The current code passes
`pretrained=False` and runs, so the env is ≤0.14 — confirm rather than assume (`requirements.txt:114`
leaves torchvision unpinned; `d3il/install.sh:34` pins 0.14.0). If it is ≥0.15, the fix is
`weights='IMAGENET1K_V1'` at `base_nets.py:510`, one line. **G-B11 makes this failure loud instead of
silent.**

---

## 5. Gates

U8's battery (G0, G1–G7, G-B1…G-B7) stays and must still pass. New:

| Gate | Asserts | Why |
|---|---|---|
| **G-B8** | The `rgb_model` spec is **identical across all three** encoder blocks (`visual_unet.py`, `visual_unet_twotime.py`, `visual_dit_twotime.py`) under every flag combination | The 🔴 byte-identical contract is now flag-dependent; drift here voids the U-Net-vs-DiT comparison silently |
| **G-B9** | `vis_cond_mode='token'` produces **bit-identical** output to the U8 code path at fixed seed; `adaln` shortens the prefix by exactly one and changes output when the latent changes; `both` does neither-nor | The regression guard. Without it U9 cannot claim U8 reproduction |
| **G-B10** | **Latent informativeness.** Given a checkpoint, freeze the encoder, fit a linear head from the 128-D latent to ground-truth box pose + target pose, report R² / mean error | **The instrument.** Turns "is perception the bottleneck?" from an argument into a number reported on every run |
| **G-B11** | With `vis_pretrained=True`, conv weights are **provably not random** — checksum against a freshly downloaded reference, and assert the load did not silently fall back | A silent fallback to random init would make U9's headline result meaningless and look like a null architecture result |
| G-B2 *(extended)* | Bracket holds in all three `vis_cond_mode` values | `adaln` removes `vis_token`; the 1.00× claim must be re-checked, not assumed |
| G-B3 *(extended)* | The warmed-up gradient + latent-sensitivity check passes in `adaln` mode | In `adaln` the visual path runs through `adaLN_modulation`, which is **zero-initialised** — the U8 warm-up (5 Adam steps @ 1e-2) is load-bearing here, not optional |

**G-B10 is the one that matters most.** It is minutes of compute, it runs on the U8 checkpoint we
already have, and it should be run **before** implementing C1/C2:

* latent **decodes box and target pose accurately** → the encoder is fine, F2 is wrong, C1/C2 are
  dead, and U9 collapses to C3 + C4;
* latent **is uninformative** → F2 confirmed, C1/C2 are the headline and C3/C4 are the free riders.

---

## 6. Run matrix

Every run: engine `mf`, seed 6, full 100 k budget, `n = 30`/cell, the untouched eval config of §3.

| # | Name | `ml_bone` | `vis_pretrained` | `vis_lr_scale` | `vis_cond_mode` | Purpose |
|---|---|---|---|---|---|---|
| R0 | **U8 reproduction** | `dit` | False | 1.0 | `token` | Must land on U8's path and metrics. Proves the flags are inert when off. *(gate-only if paths collide)* |
| **R1** | **U9 headline** | `mf_dit` | **True** | **0.1** | **`adaln`** | All four changes. The best shot. |
| R2 | perception only | `mf_dit` | True | 0.1 | `token` | Isolates C1+C2 from C3 |
| R3 | conditioning only | `mf_dit` | False | 1.0 | `adaln` | Isolates C3 from C1+C2 |
| R4 | matched-bone control | `mf_dit` | False | 1.0 | `token` | **ARCH §11.1** — the clean matched claim U8 lacks. Also R1's true baseline |
| R5 | freeze floor | `mf_dit` | True | **0.0** | `adaln` | Trainable 26.4 M → 4.0 M. The pure data-efficiency cell |
| R6 | init control | `mf_dit` | False | 0.0 | `adaln` | **Control for R5** — separates "pretrained weights help" from "training fewer parameters helps" |

**R4 first, then R1.** R4 is the honest baseline for everything else and costs one run. **R6 is not
optional**: without it a win at R5 is confounded between the weights and the parameter count, and
those two claims generalise very differently to the next task.

G-B10 is reported on **every** checkpoint above, so the mechanism is visible independently of the
distance metric.

---

## 7. Risks and how each is caught

| Risk | Catch |
|---|---|
| Silent fallback to random weights (no internet, wrong torchvision) | **G-B11** |
| The three encoder blocks drift apart | **G-B8** |
| `adaln` mode breaks the U8 path | **G-B9** (bit-identity in `token` mode) |
| `adaln` measured at step 0 reads as "image-blind" | **G-B3** warm-up — `adaLN_modulation` is zero-init; this already bit us on 2026-08-21 (job 24834) |
| Parameter bracket silently drifts | **G-B2** extended to all modes |
| Pretraining is worth little at 96×96 | Genuine, and **not** caught by a gate. `ceil(96/32) = 3` (`base_nets.py:535-537`) → the trunk emits `512×3×3` and SpatialSoftmax places 32 keypoints over **9 spatial positions**. If G-B10 stays low *after* C1, the suspect is **input resolution**, not the weights — and that is U10, not a U9 patch |
| EMA (0.995) interacting with two LR groups | EMA copies parameters, not optimiser state; unaffected. Noted so nobody re-derives it |
| Result confounded across four changes | The R2/R3/R4/R6 ablations exist precisely for this. **Do not report R1 alone** |

---

## 8. How to read the outcome

| G-B10 after R1 | R1 distance | Reading |
|---|---|---|
| high (latent decodes pose) | leaves the 0.29–0.47 m band | **Perception was the bottleneck and pretraining fixed it.** The strongest possible U9 result |
| high | stays in the band | Perception is now fine, so the bottleneck is downstream: horizon, single-frame conditioning, or the 400-step harness ceiling → U10 |
| low | either | Pretraining did not make the latent informative. Given the 3×3 bottleneck, suspect **resolution**, not weights. C1/C2 are answered and closed |
| — | R3 ≫ R4 | **F3 confirmed independently**: modulation beats token on a transformer, which is a publishable architectural finding on its own and survives whatever happens to perception |

---

## 9. What would make U9 a failure

Not "R1 does not win". U9 fails if it produces a number nobody can interpret:

1. **R1 run without R4/R6.** Four changes and no ablations is a press release, not a result.
2. **G-B10 not reported.** Then U9 is another distance number in the same band with no mechanism, and
   it repeats U8's central weakness.
3. **Any control-side edit sneaking in.** The moment the projector, the constraint YAML or the
   candidate fan moves, U9 stops being comparable to U8 and the whole unit is spent.

---

## 10. One line

**U9 stops varying the 15 % of the model that does the task and starts on the 85 % that looks at the
pictures** — ImageNet initialisation and a slow encoder LR against the 900-episode budget, plus the
one conditioning path (vision → adaLN) that our best-performing mechanism has and no transformer bone
here has ever had — with a latent-probe gate so that, for the first time, the result comes with its
mechanism attached.
