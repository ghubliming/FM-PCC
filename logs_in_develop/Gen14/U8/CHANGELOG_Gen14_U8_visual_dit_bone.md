# CHANGELOG — Gen14 U8: visual DiT/SiT ML-bone for the mf / af arms

> **Date**: 2026-08-20 · **Status**: implemented, **not yet executed** (no Python in this container — run on cluster)
> **Plan**: [`PLAN_Gen14_U8_visual_dit_bone.md`](./PLAN_Gen14_U8_visual_dit_bone.md)
> **Decision**: [`DECISION_Gen14_U8_injection_choice.md`](./DECISION_Gen14_U8_injection_choice.md)
> **Upstream refs pulled for this unit**: `aux_repo/visual_transformer_refs_(Claude_pulled)/`

---

## 0. What changed, in one paragraph

The four transformer backbones in `mix_visual_aligning/models/` were state-only and blocked
from visual runs by one `raise` per arm. They now accept a **visual token**: a `cond_dim`
constructor argument turns on a learned token carrying the 128-D dual-cam latent, prepended
to the trajectory sequence. A new `VisualDiTTwoTime` wrapper owns the vision encoder and
delegates to whichever bone is selected. A new `ml_bone` config key routes the mf/af arms to
it, and — critically — is a **checkpoint-path key**, so DiT and U-Net runs can never collide.
`film_mode` is **deleted** from DiT blocks rather than merely unset. Six gates were added.
**No objective, sampler, JVP, dataset or evaluation math was touched.**

---

## 1. Files changed

| File | Change |
|:--|:--|
| `models/mf_dit_trajectory.py` | `cond_dim` arg; `vis_tokens` + `vis_projector`; visual token appended to `_build_sequence`; **`prefix_tokens` and the RoPE table both +1** |
| `models/af_dit_trajectory.py` | identical (the two files differ only by class name) |
| `models/mf_dit_official_trajectory.py` | `cond_dim` arg; `vis_token` + `vis_projector`; `_prepend_visual()`; `pos_embed` sized over `num_tokens`; visual position stripped before the u/v heads |
| `models/af_sit_trajectory.py` | identical treatment (pos-embed stays **frozen**, as α-Flow's SiT) |
| **`models/visual_dit_twotime.py`** | 🆕 `VisualDiTTwoTime` — vision encoder + bone, `encode_visual` / `resolve_visual_cond` contract |
| `models/mf_trajectory_model.py` | visual branch routes `unet` / `mf_dit` / `dit`; unknown key still raises |
| `models/af_trajectory_model.py` | visual branch routes `unet` / `sit` / `dit`; unknown key still raises |
| `models/{mf,af}_engine.py`, `{mf,af}_trajectory_model.py` | stale `FIX_8_BACKBONE_DEFAULT` comments rewritten (they cited the `raise` that U8 removed) |
| `config/aligning-d3il-visual.py` | `_ml_bone()`, `_mix_bone_keys()`, `_DROP` sentinel, `ml_bone` in both watch lists, plan-block identity-key removal |
| `mix_visual_aligning_test/train_mix_visual_aligning.py` | bone validation + print; `imf_backbone` and `dit_*` finally threaded into `model_config` |
| `mix_visual_aligning_test/eval_mix_visual_aligning.py` | reports the bone from the pkl; **suppresses the FiLM breadcrumb on a DiT checkpoint** |
| `mix_visual_aligning_test/gates_mix_visual.py` | `_build(..., ml_bone=)`; six new gates; `--gate bone`. **Post-run (§6):** `_vnet()` engine unwrap, dit geometry passed as engine kwargs, `GRAFTED_DIFF` additive-graft ledger in G0 |
| `Slurm_Codes/sbatch/mix_visual_aligning/*.sh` | `MIX_BONE` / `MIX_BONE_<ARM>` knob with the same narrowing discipline as `MIX_FILM_MODE` |

---

## 2. The mechanism

Per the DECISION doc: **a token, not adaLN.** `diffusion_policy` — verified to be the upstream
of this repo's own vision encoder — tokenises the obs latent for its transformer
(`cond_obs_emb`, one token per obs step) and reserves modulation (FiLM) for its U-Net. Our
`VisualUNet` already occupies the modulation design point, and the two RoPE bones have no
adaLN pathway at all, so the token is both the referenced design and the only one that spans
all four bones.

```
 dual-cam images ──▶ MultiImageObsEncoder ──▶ 128-D latent  (UNCHANGED, byte-identical cfg)
                                                   │
                                    Linear(128, d) + learned token
                                                   │
   [class, omega, t_min, t_max, time, ▸VIS◂, x_0 … x_7]      ← RoPE bones (dit / af_dit)
   [▸VIS◂, x_0 … x_7]  + pos_embed                            ← adaLN bones (mf_dit / sit)
```

**Two bones, two placements, one idea.** On the RoPE bones the token joins the existing
in-context prefix, appended **last** so every pre-existing token keeps its RoPE position. On
the adaLN bones there is no prefix, so the token is prepended and `pos_embed` grows by one row.

### 2.1 The bookkeeping that had to move together

`mf_dit_trajectory.py` uses `prefix_tokens` in two places — to size the RoPE table
(`prefix_tokens + num_patches`) and to strip the prefix before the u/v heads. A `+1` applied to
one and not the other produces a model that **trains fine and reads the wrong positions**. Both
now derive from a single `num_visual_tokens`, and gate **G-B6** asserts they agree.

The RoPE buffers are `persistent=False`, so resizing them cannot corrupt checkpoint loading —
only the two constants matter.

---

## 3. Two regressions I introduced and then fixed

Recording these because both were silent and both were caught by inspecting output, not by an error.

### 3.1 🔴 The `_Bunet` fragment would have orphaned every existing checkpoint

First cut had `_mix_bone_keys('mf')` return `{'ml_bone': 'unet', ...}` for the baseline bone.
That put a `_Bunet` fragment into `exp_name` **and** `diffusion_loadpath` — so every Gen14
U-Net checkpoint already on the cluster would have become unreachable.

**Fix**: the U-Net bone emits **no** `ml_bone` key at all. `watch()` skips keys the args object
lacks (`diffuser/utils/setup.py:25`), so the U-Net path is byte-identical to pre-U8, while DiT
blocks *do* define it and therefore carry a fragment the U-Net lacks. Collision is still
impossible; nothing existing moves. Same trick `n_diffusion_steps` and `film_mode` already use.

Verified:
```
UNET   …_bs{train_batch_size}_film{film_mode}_E{engine}_ts{t_schedule}          ← pre-U8, unchanged
MF_DIT …_bs{train_batch_size}_B{ml_bone}_E{engine}_ts{t_schedule}               ← new tree
```

### 3.2 🔴 `film_mode` survived onto DiT blocks by inheritance

Omitting `film_mode` from `_mix_bone_keys` was not enough: the mf/af arms inherit it from
their parent block (`fm_visual_aligning`), so a DiT path still read `_filmv1_` — a directory
name asserting a FiLM mode for a model with no FiLM path. The same leak existed on the **plan**
side via `_mix_plan_common`, which would have labelled DiT *results* folders `_filmv1_` too.

**Fix**: a `_DROP` sentinel that `_mix_train_block` removes from the merged block, plus a loop
in `_mix_plan_block` that strips any identity key the training block does not have. Verified
across `{}`, `MIX_BONE=dit`, and `{MIX_BONE_MF: mf_dit, MIX_BONE_AF: sit}` — `film_mode` is
`ABSENT` on both train and plan blocks for every DiT bone, present for every U-Net bone, and
the `diffusion`/`fm` arms are untouched in all cases.

---

## 4. Safety properties, and how each is enforced

| Property | Enforcement |
|:--|:--|
| State-only generations (Gen3v4/v6/v7) unaffected | every edit guards on `cond_dim > 0`; at `cond_dim=0` the state_dict is unchanged — **G-B1** |
| A visual run cannot be image-blind | `cond_dim>0` + no latent ⇒ `raise` in the bone; gradient into `vis_projector` asserted — **G-B3** |
| The JVP still works | latent is pre-encoded by the engine and captured as a constant; unchanged by U8 — **G-B4/5** |
| Checkpoints cannot collide | `ml_bone` is a path key present only on DiT blocks — **G-B7** |
| No lying directory names | `film_mode` deleted from DiT train *and* plan blocks — **G-B7** |
| The A/B is not confounded | `dit_hidden_size=160` (~3.9 M vs the U-Net's ~4.0 M); build-time warning at 256; ratio asserted — **G-B2** |
| `diffusion` / `fm` arms untouched | they are single-time; `ml_bone != 'unet'` on them is a hard `SystemExit` in the train script |

🔴 **On `dit_hidden_size`**: the *engine's* default is 256 — the state-only width, deliberately
left alone. The visual config sets 160 explicitly. Because the engine value is now forwarded to
the wrapper as the single source of truth, a build that reaches 256 with `if_vision=True` prints
a loud warning naming the retraction it would reproduce (PLAN §1.2c).

---

## 5. What was NOT done

- **adaLN injection** (Option 1) — rejected, DECISION §4.1. ~30 lines on top of this wrapper if
  the token result is negative and the mechanism needs ruling out.
- **Spatial visual tokens** (Option 3) — deferred to Phase B; `act` was pulled for it.
- **Cross-attention decoder** — DECISION §4.3.
- **`diffusion` / `fm` arms** — no single-time visual transformer exists in this repo.
- `MASTER_TEST_HISTORY.md` — not edited.

---

## 6. Verification status — FIRST CLUSTER RUN, 2026-08-21

Jobs `24815` (`gates … bone`) and `24830` (pipeline `gates … all`) on i6-gpu-1, git rev `6200512`.

```
GB1 PASS   GB6 PASS   GB7 PASS   GB45 PASS      G1..G7 PASS
GB2 FAIL   GB3 FAIL                             G0  FAIL
```

**The model path was proven healthy.** G-B1/G-B6/G-B45 are the substantive gates and all four
bones passed them: state-only builds are unchanged at `cond_dim=0`, prefix/`pos_embed`
bookkeeping agrees on every bone, outputs come back `(2, 8, 9)`, and one real loss step is
finite through both the MeanFlow JVP and the alpha-Flow bootstrap with the visual token live.

**All three failures were in gate code, not in the trained path.** Fixed below; no model,
config, or sbatch file needed a change. Train `24831` / eval `24832` were cancelled by
`--dependency=afterok`, which is the chain behaving correctly.

### 6.1 G-B2 / G-B3 — `AttributeError: 'MeanFlowEngine' object has no attribute 'velocity_net'`

`_build()` returns the ENGINE for mf/af (`wraps_unet=True`, see `engine_registry.py`), and
`velocity_net` lives one level down on the trajectory model at `.model`. G-B45 never hit this
because it only touches `diffusion.loss()`. Fixed with a `_vnet()` helper that resolves either
shape; both gates now go through it.

### 6.2 🔴 The gates were building at `dit_hidden_size=256`, not the matched 160

Visible in the log as the `VisualDiTTwoTime` warning firing on every bone, and as
`params=32.6M` where the U-Net is `26.4M`. Cause: `_build()` set `cfg.dit_hidden_size = 160`
as a config attribute only, but `VisualDiTTwoTime._knob()` gives the **engine-passed kwarg
precedence** — and `MeanFlowEngine`'s own default is the state-only `256`. The cfg value could
never win. Fixed by passing the geometry as engine kwargs, mirroring
`train_mix_visual_aligning.py:418-422`, and setting the cfg attrs in sync so the fallback can
never disagree.

**This did not affect training.** The train script forwards `args.dit_hidden_size` explicitly,
and `_mix_bone_keys()` sets it to 160, so a real run was always going to be parameter-matched.
The warning fired exactly as designed — it was written for this failure mode and it caught it.

### 6.3 G0 — the four bones are no longer verbatim copies

Correct and expected: U8 grafted a visual token into them. But dropping them into the
existence-only `GRAFTED` ledger would remove them from G0's coverage, and their upstreams
(Gen3v6/Gen3v7) are still actively edited. Instead they moved to a new `GRAFTED_DIFF` ledger
that keeps a real check, weakened only where U8 needed it: **the graft must stay additive.**
Each entry pins the number of source lines U8 rewrote — 4 on the RoPE bones, 3 on the adaLN
bones — and G0 fails if that count moves in either direction.

Verified locally (`gate_g0` is pure stdlib): `+46/-4`, `+45/-3`, `+46/-4`, `+45/-3` — G0 PASS,
19 verbatim files still match.

### 6.4 What is still unverified

- G-B2's parameter ratio has never actually been evaluated (it died before the check). At
  d=160 the arithmetic predicts ~0.98x for the adaLN bones and ~0.82x for the RoPE bones
  against the 4.0 M U-Net, both inside the 0.75-1.35 band — **predicted, not measured.**
- G-B3 has never confirmed gradient reaches `vis_projector`. Until it does, "the visual token
  is live" rests on construction plus a finite loss, which is weaker than it sounds.
- Nothing has trained.

Re-run `gates_mix_visual.sh bone` (~1 min) before the next pipeline submission.

## 7. Expected result — read PLAN §1.3 before budgeting GPU time

The only parameter-matched backbone evidence this repo owns says the **U-Net wins** on the
deployable tightened DPCC arms (`AF UNet@32` 0.958 vs `AF SiT` 0.722), and the one result that
said otherwise was retracted. This unit exists to ask the question honestly in the visual
setting, not to deliver a win. "We built it, parameter-matched it, and the U-Net still wins"
is a legitimate and citable outcome — plan the DA around being able to state that cleanly.
