# Gen14 U9 — CHANGELOG: perception-first ML upgrade

**Date:** 2026-08-24 · **Generation:** Gen14 (`mix_visual_aligning` ↔ `mix_visual_aligning_test`)
**Unit:** U9 · **Plan:** [`PLAN_Gen14_U9_perception_first_ml.md`](PLAN_Gen14_U9_perception_first_ml.md)
**Status:** implemented, **not yet run on the cluster** (no Python in this container — first real
execution will be the gate job, exactly as U8 was)

---

## 0. The one-line claim, and how it is enforced

**U9 is purely additive: with no environment variables set, every file behaves exactly as it did
before, and every checkpoint path is character-for-character unchanged.**

That is not a promise, it is checked three ways:

| Guarantee | Enforced by | Verified |
|---|---|---|
| Config emits no new path keys at the defaults | `_mix_u9_keys()` returns `{}`; `watch()` skips undefined keys | ✅ run locally — `{}` at defaults and at explicit defaults |
| Grafted bones/trainers stayed additive | G0 pins *removed/rewritten* source lines; every U9 edit is a pure **insertion** | ✅ all 6 `GRAFTED_DIFF` entries hold their pins |
| VERBATIM copies untouched | G0 byte-compare | ✅ 17/17 identical |
| `vis_cond_mode='token'` is bit-identical to U8 | **G-B9(a)** — state_dict equality against a pre-U9 build | cluster |
| Pretrained weights really loaded | **G-B11** | cluster |

---

## 1. Files changed

| File | Change | Additive? |
|---|---|---|
| `d3il/agents/models/vision/model_getter.py` | `get_resnet(..., pretrained: bool = False)` | ✅ new kwarg, default = the previously hard-coded value |
| `mix_visual_aligning/models/visual_unet_twotime.py` | `'pretrained': _vis_pretrained` in the encoder spec + a U9 log line | ✅ |
| `mix_visual_aligning/models/visual_dit_twotime.py` | same encoder edit + `vis_cond_mode` resolution and forwarding | ✅ |
| `mix_visual_aligning/models/mf_dit_official_trajectory.py` | `vis_cond_mode` (`token`/`adaln`/`both`), `_pool_cond()` | ✅ **+92 / −3**, pin held |
| `mix_visual_aligning/models/af_sit_trajectory.py` | same graft | ✅ **+86 / −3**, pin held |
| `mix_visual_aligning/utils/training_twotime.py` | `vis_lr_scale` two-group optimiser, `lr_vis` logging | ✅ **+84 / −3**, pin held |
| `mix_visual_aligning_test/train_mix_visual_aligning.py` | forwards `vis_lr_scale` to the two-time trainer | ✅ |
| `mix_visual_aligning_test/gates_mix_visual.py` | `_build()` gains 2 kwargs; **G-B8, G-B9, G-B11**; `--gate bone` extended | ✅ |
| `mix_visual_aligning_test/probe_latent_informativeness.py` | **new** — the P2 probe | ✅ new file |
| `config/aligning-d3il-visual.py` | `_mix_u9_keys()`, 3 watch-list entries, splat into the mf/af blocks | ✅ |
| `Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh` | 3 env knobs with narrowing + validation | ✅ |

**Deliberately NOT touched:** `mix_visual_aligning/models/visual_unet.py` (a G0 **VERBATIM** file —
editing it would force it out of the ledger for no experimental gain; it is the fm/diffusion arm and
U9 runs mf/af), the projector, HardFlow, `config/visual_aligning_eval.yaml`, the MPC fan, the
selection rules, the horizon, the observation window, `LATENT_DIM`, and the encoder architecture.
**No control-theory code was modified.**

---

## 2. The three knobs

### C1 · `vis_pretrained` — ImageNet init of the dual ResNet-18

`MIX_VIS_PRETRAINED=1`. 22.36 M of 26.4 M trainable parameters are the encoder, fitted from scratch
to 900 episodes; this changes only the **weights**, never a shape, so `cond_dim` never moves and every
U8 gate remains valid as written.

The kwarg lands in the vendored `d3il` file, which Gen6V4, Gen7, `imf_visual_aligning` and
`mix_visual_avoiding` all import — **none of them pass it, so none of them change.**

🔴 Documented in the docstring because it is invisible otherwise: `use_group_norm=True` replaces
**every** `BatchNorm2d` in the pretrained trunk with a fresh `GroupNorm`
(`multi_image_obs_encoder.py:62-69`), discarding the pretrained affine params and running stats. The
conv filters survive. The network therefore arrives **decalibrated**, which is precisely why C2
exists and why a hard freeze is *not* the headline configuration.

### C2 · `vis_lr_scale` — a slower encoder, not a frozen one

`MIX_VIS_LR_SCALE=0.1`. Encoder parameters at `train_lr × scale`, everything else at `train_lr`.

🔴 **Implemented as a conditional rebuild, not an edit of the optimiser line.** At `1.0` the new code
does not execute at all, so the optimiser object, its param-group count and its `state_dict` layout
are the pre-U9 ones. This matters concretely: `_restore_optimizer_state()` calls
`optimizer.load_state_dict()` on pre-U9 checkpoints, a one-group checkpoint cannot be loaded into a
two-group optimiser, and this pipeline auto-resumes near the 24 h wall as a matter of routine. One
throwaway `Adam` construction buys that guarantee without a single conditional in the resume path.

The scheduler is `LambdaLR`-based (`get_cosine_schedule_with_warmup`), which scales **each group's**
`initial_lr`, so both groups anneal correctly. `logs["lr"]` reports group 0, so a second key
`logs["lr_vis"]` is emitted whenever the split is live — otherwise the encoder rate would be
invisible in every log and every `lr_history` plot.

If the encoder cannot be found at `model.velocity_net.obs_encoder`, the trainer **raises**. A
silently-ignored encoder LR is exactly how a null result gets manufactured.

### C3 · `vis_cond_mode` — vision into the adaLN path

`MIX_VIS_COND=adaln`. Where the visual latent enters the transformer.

The motivation is U8's own data. Our best-scoring mechanism is VisualUNet **v1** (0.3425 pooled),
which folds `cond_mlp(latent)` into the embedding driving every block. Its exact transformer analogue
is `c += vis_projector(latent)` — adaLN's native conditioning path. And the adaLN bones never used it:
`mf_dit_official_trajectory.py:263` conditions on `c = t_emb + r_emb + w_emb`, with `:278` stating the
latent is *"PREPENDED as one token (**not summed into adaLN's `c`**)"*. **That cell of the 2×2 has
never been trained** — U8 chose the token deliberately (`DECISION_Gen14_U8_injection_choice.md`), and
U9 tests the other design point. This is a new design point, not a bug fix.

| mode | sequence | `c` | `vis_token` in state_dict |
|---|---|---|---|
| `token` (default) | +1 visual token | t+r+w | yes — **bit-identical to U8** |
| `adaln` | unchanged | t+r+w **+ vis** | **deleted** (would be a dead parameter) |
| `both` | +1 visual token | t+r+w **+ vis** | yes |

`num_visual_tokens` is overridden *before* `num_tokens` is computed, so `pos_embed`, the sin-cos
table and the strip in `forward()` all follow automatically — the invariant G-B6 already asserts now
covers all three modes for free. In `adaln` the `pos_embed` is one row shorter, so **state_dicts are
not interchangeable across modes** — which is correct, and `vis_cond_mode` is a path key so the trees
never collide.

**adaLN bones only.** `mf_dit` and `sit`. The RoPE bones (`dit`) are `forward(x, cos, sin)` with no
adaLN pathway; passing the knob would be swallowed by their `**unused` and silently do nothing, so
`VisualDiTTwoTime` **raises** instead — and so does the config, and so does the sbatch script. Three
independent guards on one silently-ignorable knob is deliberate.

---

## 3. New gates

| Gate | GPU | Asserts |
|---|---|---|
| **G-B8** | no | The `rgb_model` spec is identical between `visual_unet_twotime.py` and `visual_dit_twotime.py`, and contains `'pretrained'`. The byte-identical contract was a comment before U9; U9 puts a flag in that block, which is exactly what gets added to one file and forgotten in the other — and the failure is silent because both models still train. **Verified locally: identical.** |
| **G-B9** | yes | (a) `token` state_dict is identical to a pre-U9 build — the additivity guarantee; (b) `adaln` has `num_visual_tokens == 0`, no `vis_token`, and still responds to the latent; (c) `both` keeps the token and responds. Warms up 5 Adam steps first, for the same reason G-B3 does — these are adaLN-**zero** bones and a step-0 measurement reads 0.0 in every mode (job 24834, 2026-08-21). |
| **G-B11** | no | `pretrained=True` produces **seed-independent** weights (⇒ loaded from a file) while `pretrained=False` is seed-dependent (⇒ the control is not vacuous). Needs no checksum and no network. Catches the failure that matters most: a download that silently fell back to random init, or a torchvision ≥ 0.15 env where the `pretrained=` kwarg was removed — either produces a null result that looks like an architecture finding. |

`--gate bone` now runs `gb1 gb6 gb7 gb2 gb3 gb45 gb8 gb9 gb11`.

---

## 4. The P2 probe (new file)

`mix_visual_aligning_test/probe_latent_informativeness.py` — read-only, minutes, **runs on the U8
checkpoint that already exists.**

Ridge-regresses three feature sets onto the H×3 action chunk on a held-out split: state only (the
free 6-D anchor), latent only, and both. **The number that matters is the incremental R² of the
latent over the state** — the image is only worth 22.36 M parameters if it explains variance the
state does not. G-B3 proves gradient *reaches* `vis_projector`; it never proved the latent is
*informative*.

Scope stated in the docstring rather than buried: `ParityAligningDataset` yields only
`(trajectory, cond[0], images)` and carries **no box or target pose** (`sequence.py:42-47`), so this
cannot regress onto box pose. The action chunk is the closest available target and arguably the more
relevant one. A low score does not prove the image lacks box information in principle — it proves the
trained encoder does not expose it linearly for the thing the policy must do.

---

## 5. How to run

**Once, on the login node** (compute nodes have no internet):

```bash
conda activate FMPCC
python -c "import torchvision as tv; print(tv.__version__); tv.models.resnet18(pretrained=True)"
ls -lh ~/.cache/torch/hub/checkpoints/
```

The version print matters — `pretrained=` was removed in torchvision 0.15. The current code passes
`pretrained=False` and runs, so the env is ≤ 0.14, but confirm rather than assume. G-B11 makes the
failure loud either way.

**Gates** (~1 min, and they exercise all three knobs):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh bone
```

**P2, before committing a GPU-week** — it decides whether C1/C2 are worth running at all:

```bash
python mix_visual_aligning_test/probe_latent_informativeness.py \
    --checkpoint logs/.../state_80000.pt
```

**R4 — the matched-bone control, ARCH §11.1. Run this before the headline:**

```bash
MIX_BONE_MF=mf_dit ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6
```

**R1 — the U9 headline:**

```bash
MIX_BONE_MF=mf_dit MIX_VIS_PRETRAINED=1 MIX_VIS_LR_SCALE=0.1 MIX_VIS_COND=adaln \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6
```

Full matrix (R0–R6, including the **non-optional** R6 init control) in PLAN §6.

---

## 6. What is verified and what is not

**Verified in this container (static only):**

* All 6 `GRAFTED_DIFF` pins hold — every U9 edit is a pure insertion, so **G0 will pass unchanged.**
* All 17 `VERBATIM` files still byte-identical.
* `_mix_u9_keys()` returns `{}` at the defaults, produces the right keys for R1/R6, and raises on
  every malformed input (bad bool, bad mode, negative scale, adaLN on a RoPE bone).
* G-B8's comparison logic run by hand: the two encoder specs are identical and carry `'pretrained'`.
* `py_compile` clean on all 10 Python files; `bash -n` clean on the sbatch script.

**NOT verified — needs the cluster:**

* G-B9's bit-identity claim (needs torch).
* G-B11 (needs torch + the weights in cache).
* That `pretrained=True` actually loads under this env's torchvision version.
* That the two-group optimiser trains stably, and that auto-resume survives the split.
* Any statement about whether U9 moves the metric.

**Nothing is committed.** No `git commit` was run, per the standing rule.

---

## 7. Honest caveat

ARCH §13.3(a) still applies: three trunks, two engines, two projector arms and eleven variants all
land in 0.29–0.47 m. U9 attacks the largest thing upstream of the bone, but **my expectation is that
it improves the DiT's standing without leaving the band**, and P2 is in this unit precisely so that
outcome still yields a mechanism rather than another uninterpretable distance number.

**Do not report R1 alone.** Without R4 (matched-bone control) and R6 (init control, which separates
"pretrained weights help" from "training 4 M instead of 26 M helps"), a win at R1 is four changes and
no ablation.
