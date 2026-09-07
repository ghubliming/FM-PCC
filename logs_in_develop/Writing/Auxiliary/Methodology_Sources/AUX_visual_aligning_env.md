# AUX — `aligning-d3il-visual`: how the visual environment was built on D3IL

**Thesis home:** `sec:method:visual`, `sec:setup:tasks` · **TARGET** §5.1
**Primary dev logs:** `D3IL_Code_Guide/D3IL_Native_Visual_Aligning_Pipeline_Guide.md` ·
`Add_Visual_to_ML_bones/INVESTIGATION_visual_dit_sit_backbones.md` (the clearest single source) ·
`D3IL_Visual_Aligning_RUN/` · `Gen6_dpcc_Engine_for_visual_aligning/` ·
`Gen7_FMPCC_Viusal_Aligning/` · `Gen14/Fix_9/` (FiLM modes) · `Gen14/U8/` (DiT/SiT attempt)

---

## 1. What the thesis must say

`aligning-d3il-visual` is the **modality-transfer environment**: the same manipulator task as the
state benchmark, but the policy sees two RGB cameras instead of a 20-D state vector. It is the
environment where the high-`K` regime lives, and therefore where Goal B is decided.

The section has to establish three things, in this order:

1. **The task is D3IL's, the vision *pipeline* is D3IL's, and the pairing is ours.** The upstream
   D3IL benchmark ships a native visual aligning pipeline (`run_vision.py`,
   `aligning_vision_config.yaml`, `Aligning_Img_Dataset`, `Aligning_Sim`). We did not re-implement
   perception; we reused it and attached our generative engines beneath it.
2. **The 128-D latent is the clean interface boundary**, and this is what makes the
   architecture-matched claim honest. Everything above it — image preprocessing, the two ResNets,
   window pooling — is backbone-agnostic and identical across every engine. Everything below it is
   encoder-agnostic. **The generative backbone never sees an image.** So when the thesis says "same
   4.0 M U-Net, only the objective differs", the perception stack is provably held fixed rather than
   assumed fixed.
3. **Conditioning is FiLM, and the mode is a recorded knob.** `film_mode` `v1` is the default on all
   four Gen14 arms and every reported flagship number; `v2` exists and is reachable. Because the
   value used to arrive by *config inheritance* rather than being stated in the arm blocks, the
   checkpoint tag carries it (`filmv1`) — the thesis should quote the tag rather than the prose.

---

## 2. The architecture, as built

```
  agentview_image ─► ResNet-18 (GroupNorm) ─► 64-D ┐
                                                    ├─ concat ─► 128-D visual latent
  in_hand_image   ─► ResNet-18 (GroupNorm) ─► 64-D ┘        │
                                                             │  FiLM conditioning (v1)
                                                             ▼
                                        UNet1DTemporalCondModel — 1-D temporal conv U-Net
                                        (Conv1d residual blocks, stride-2 down/up,
                                         time embedding via MLP)   → 4.0 M params
```

| item | value | source |
|---|---|---|
| encoder | D3IL `MultiImageObsEncoder`, **vendored unchanged** | `INVESTIGATION_visual_dit_sit_backbones.md` §1.2 |
| per-camera model | `agents.models.vision.model_getter.get_resnet` → ResNet-18, GroupNorm | ibid. §"rgb_model" table |
| weight sharing | `share_rgb_model = False` — **independent ResNet per camera** | ibid. |
| latent | 2 × 64-D → concat → **128-D** | ibid. |
| images | 2 × RGB, 96 × 96 | `D3IL_Native_Visual_Aligning_Pipeline_Guide.md` §2 |
| state part of obs | `obs_dim: 3` (robot EE position) | ibid. |
| generative backbone | `VisualUNet` / `VisualUNetTwoTime`, **4.0 M** | CLOSURE_20260907 §2 |
| conditioning | FiLM, `film_mode = v1` on every reported run | `Gen14/Fix_9/` |
| horizon | H8 | run tags |

**Encoder universality is a documented D3IL design property**, not our inference: the identical
`MultiImageObsEncoder` + `get_resnet` config appears across all 11 D3IL vision agents and all 5
vision scenes. That is the sentence that justifies calling it a fixed perception stack.

---

## 3. Two implementation points that are load-bearing for the *method* chapter

### 3.1 The pre-encode short-circuit — why MeanFlow can be trained at all here

MeanFlow's target needs a **JVP**, and every component inside the JVP must be forward-AD friendly.
A ResNet inside the differentiated closure would be both wrong and ruinously expensive. The visual
pipeline solves this with a **pre-encode short-circuit** (`visual_unet_twotime.py:19-41`): the
encoder runs **once, outside the JVP**, and the resulting 128-D tensor is captured as a *constant*
inside the closure.

This is not a performance trick — it is what makes the two-time objective compatible with a visual
observation at all, and it belongs in `sec:method:engine` alongside the MeanFlow target. It also
explains a real bug the corpus carries: the `_encode_once` `AttributeError` that killed the first
`fm` K=20/T=0.2 job (2026-09-01) is exactly this short-circuit failing on an engine that did not
implement it.

### 3.2 Why there is no DiT/SiT visual result worth citing

Every active visual pipeline from Gen6 → Gen14 uses the temporal U-Net. The DiT/SiT investigation
concluded that using them properly requires **spatial** ResNet features and cross-attention rather
than the pooled 128-D vector — a different architecture, not a swap. The `af`-SiT rows that do exist
are disqualified twice over (9.4 M vs 4.0 M, and `ae0.0` ⇒ MeanFlow mislabelled). The thesis should
present the U-Net as the deliberate architecture-matched choice and the transformer route as scoped
out with a reason.

---

## 4. Holes — fill before writing

- [ ] **The success criterion is a known problem and must be written as one.** Binary success needs
      `pos ≤ 1.8 cm` **and** `rot ≤ 8.64°`, rotation is uncontrolled on every arm, and success is
      0–2/30 everywhere — it ranks nothing. Every V_A result is therefore reported on
      `context_final_xy_dist`. This is a *task-design* caveat, not a result, and it belongs in
      `sec:setup:metrics` + `sec:disc:threats`.
- [ ] **`mean_dist_per_rollout` is not a distance** (`0.5·(pos_dist_3D + rot_err/π)`). Define both
      metrics explicitly once so no table is misread.
- [ ] **Dataset provenance and split**: 900 episodes / 168 274 windows (horizon 8, traj_dim 9) appear
      in eval logs; the train/test split protocol and whether the reported numbers are train-split
      are stated per-DA but not in one place. **Most V_A results are train-split — say so.**
- [ ] **Whose checkpoint, whose training run** — `steps1000_bs64` and the EMA policy are in the tags;
      the training recipe (optimiser, LR schedule, epochs, action weight `aw`) needs one table.
- [ ] **Camera placement and rendering settings** for the two views are not documented on our side —
      they are inherited from D3IL and should be cited, not restated from memory.
- [ ] **Licence and version of the vendored D3IL encoder** for `app:repro` (§5.3).
