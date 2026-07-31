# PLAN — Gen14: **Visual-Mix-ML** — one frame, four ML engines activated by config

**Date:** 2026-07-31 · **Type:** implementation plan · **NO CODE WRITTEN YET**
**Status:** draft for review — nothing may be implemented until the §1 decisions are confirmed
**Goal:** reassemble **Gen6V4 (DDPM)** + **Gen7 (FM-ODE)** + **Gen3v6 (MeanFlow)** + **Gen3v7 (α-Flow)**
into one visual-aligning frame where the ML engine is a config switch.

> ## 🔑 The governing principle
>
> **Write the least code. Reassemble, don't rewrite.**
> Every file is a **copy** of a currently-working file, edited as little as possible.
> **Redundancy is explicitly allowed and preferred** over clever de-duplication.
> **Fidelity to the working code is non-negotiable** — if a "copy" comes out with an unexplained
> code diff, the merge assumption broke: stop and re-open this plan.
>
> The measure of success for this generation is **how few lines were newly authored**, not how
> elegant the result is. Target: **< 400 newly-authored lines** across the whole generation (§3.2).

**Sources — all four are robust, current, ready to use as-is**
| | model folder | test folder | engine class |
|---|---|---|---|
| Gen6V4 | [`diffuser_visual_aligning/`](../../../diffuser_visual_aligning) | [`diffuser_visual_aligning_test/`](../../../diffuser_visual_aligning_test) | `GaussianDiffusion` → `VisualGaussianDiffusion` |
| Gen7 | [`fm_visual_aligning/`](../../../fm_visual_aligning) | [`fm_visual_aligning_test/`](../../../fm_visual_aligning_test) | `FlowMatchingODE` → `VisualFlowMatching` |
| Gen3v6 | [`flow_matcher_v3_meanflow/`](../../../flow_matcher_v3_meanflow) | [`FM_v3_meanflow_test/`](../../../FM_v3_meanflow_test) | `MeanFlowODE` + `MFTrajectoryModel` |
| Gen3v7 | [`flow_matcher_v3_alphaflow/`](../../../flow_matcher_v3_alphaflow) | [`FM_v3_alphaflow_test/`](../../../FM_v3_alphaflow_test) | `AlphaFlowODE` + `AFTrajectoryModel` |

**☠️ Gen8 (`imf_visual_aligning/`) is DEAD CODE — reference only.** See §2. Nothing is copied from it.
**iMF is NOT an arm of Gen14.** Gen3v4-iMF is `abandoned` in `MASTER_TEST_HISTORY.md` and stays that way.

**Prior plans to read first:** [`Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md`](../../Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md) · [`Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md`](../../Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md)

---

## 0. Why this generation exists

Gen3v6 (MeanFlow) and Gen3v7 (α-Flow) are the two live answers to the Gen13 iMF refutation, and both
are **state-only, avoiding-d3il, DiT/SiT-backbone** experiments. The thesis payload — D3IL **visual
aligning** under the DPCC safety cage — runs only two engines today: Gen6V4 (DDPM) and Gen7 (FM-ODE).
The two newest objectives have never been tested where the results matter.

**And the reassembly is nearly free.** Gen6V4 vs Gen7 differ in exactly two real files:

```
diff -rq diffuser_visual_aligning/ fm_visual_aligning/
  models/diffusion.py                  differ  ★ THE ENGINE
  models/visual_gaussian_diffusion.py  differ  ★ THE VISUAL ENGINE SUBCLASS
  models/__init__.py                   differ  → re-export names
  models/helpers.py                    differ  → ONE import line (package name)
  models/visual_unet.py                differ  → THREE import lines (package name)
  utils/config.py, utils/setup.py      differ  → package name inside a comment
  datasets/sequence.py                 differ  → 93 lines of docstring, 0 lines of code
  sampling/projection.py               IDENTICAL   ← the entire DPCC projector, byte-for-byte
```

Two generations of the visual pipeline are **already** one frame with a swapped engine file. Gen14
makes that explicit and admits two more engines. Nothing is being invented here — it is a copy job
with two small, well-identified grafts (§6).

---

## 1. Decisions (confirm before any code)

1. **Generation number: Gen14, name `Visual_Mix_ML`.** Parallel to Gen13 `HF_Mix_ML` — the same
   cross-architecture question asked under a different constraint mechanism.
   ⚠️ **Gen13 = HardFlow math; Gen14 = DPCC math. Never mix their results.**
2. **Folder pair: `mix_visual_aligning/` ↔ `mix_visual_aligning_test/`.**
   (Rejected: extending `fm_visual_aligning/` in place — breaks Gen7 rollback and invalidates every
   existing Gen7 checkpoint path.)
3. **Frame base: `fm_visual_aligning/` @ HEAD.** It is the newest folder carrying every C4/C5/C6/D1/U19
   upgrade, and it is commit-for-commit in sync with Gen6V4 (`git log -- <folder>` confirms: both
   share every commit from `1b4851b8` onward).
4. **Engines shipped: `ddpm | fm | mf | af`. Four. No iMF, now or later.**
5. **Backbone LOCKED to the `VisualUNet` stack** (dual-ResNet `MultiImageObsEncoder` → 128-D FiLM
   latent → 1-D temporal U-Net). The Gen3v6 `mf_dit` / Gen3v7 `af_sit`/`af_dit` backbones are **not**
   ported — none has a visual-conditioning input. Locking the backbone is also what makes the
   four-way comparison *architecture-controlled*: only the objective and sampler vary.
6. **Task/data/dims frozen:** D3IL visual aligning, `ParityAligningDataset`, 9-D trajectory
   `[act(3) | des_c_pos(3) | c_pos(3)]`, `action_dim=3`, `obs_dim=6`, `horizon=8`.
   Non-visual (`if_vision=False`, 23-D `StateOnlyAligningDataset`) stays available on all four arms.
7. **Constraint layer untouched.** `sampling/projection.py` copied byte-identical;
   `config/visual_aligning_eval.yaml` reused unchanged. Constraints belong to the *task*, not the
   engine — that is what keeps the arms comparable.
8. **HardFlow is OUT of scope.** Gen3v6/v7 both carry `sampling/hardflow_projection.py`, but the
   HardFlow NLP is built on a **state-only linear-dynamics `.npz` fitted in the avoiding normalizer's
   units** (the Gen12 refit warning). Retargeting it to visual aligning is its own generation.
   Deferred → §11.
9. **Zero retraining of existing work.** Gen6V4 and Gen7 checkpoints stay where they are and remain
   evaluable from their own folders. Gen14 writes under its own `mix_visual_aligning/` prefix.

---

## 2. ☠️ Gen8 — what to learn, what never to touch

`imf_visual_aligning/` is the only prior attempt at this exact merge (iMF into the Gen7 visual path).
It is **dead code**: forked from Gen7 before C4/C5/C6/D1/U19 and never re-synced. Verified:

| Gen7 / Gen6V4 @ HEAD | Gen8 |
|---|---|
| `_save_partial_npz` (C5 crash-safe NPZ) | **absent** |
| `_scan_box_obstacle_conflicts` (D1 box×obstacle guard) | **absent** |
| C5 consolidated-NPZ schema, C6 circuit breaker, U19 foresight plot | **absent** |
| shares every commit from `1b4851b8` onward | shares **none** of them |

**Rule: no file, function, or line is copied out of `imf_visual_aligning/`.** It may be *read* to
learn three design answers, all of which Gen14 re-derives from the live Gen3v6/v7 sources:

- **L1 — the visual cond-dict wrapper.** `visual_imf_diffusion.py:24-61` shows the shape: a subclass
  whose `loss()` repacks the trainer's `(trajectories, conditions)` into
  `{0: obs_anchor, 'visual': (primary_img, wrist_img, obs_seq)}`, and whose `forward()` unpacks the
  eval wrapper's `{0: (bp_imgs, inhand_imgs, obs_seq)}`. Gen14 copies this shape **from Gen7's
  `VisualFlowMatching`**, which is the same 40 lines and is alive.
- **L2 — "Option A" engine-wraps-backbone wiring.** For a two-time `(u,v)` engine the chain is
  `Engine(if_vision=True, vis_config=args)` → `TrajectoryModel` → builds `VisualUNet` **internally**
  as its `velocity_net` (`imf_trajectory_model.py:38-44`). Gen14 uses the same shape (§5).
- **L3 — `model_config.pkl` then describes the *engine*, not the U-Net**, so the eval loader must
  reconstruct the engine. Gen14 inherits this and adds the arm-identity assert (§5).

Everything else in Gen8 — its train script, its eval script, its config blocks — is stale and is to
be treated as if it did not exist.

---

## 3. The architecture: a frame, and four self-contained stacks

### 3.1 The one rule that makes fidelity structural

> **The `ddpm` and `fm` arms import ONLY verbatim copies.**
> Every newly-authored line lives in a file that only `mf` and `af` import.

This is not a style preference. It means Gen14's reproduction of Gen6V4 and Gen7 is guaranteed **by
construction** rather than verified by testing — the DDPM and FM code paths are byte-identical
copies of code that works today, reachable through a frame that only chooses between them.
Gate G1 then becomes a cheap confirmation rather than a load-bearing check.

The cost is duplication: two backbone files that share ~85 % of their lines, two trainer files that
share ~70 %. **That duplication is the point.** De-duplicating them is what turned Gen8 into dead
code — a shared file drifts under whichever generation last touched it.

### 3.2 Copy ledger — the budget this plan is accountable to

| Kind | Meaning | Files | Newly-authored lines |
|---|---|---|---|
| **V** verbatim | byte-identical copy | 14 | 0 |
| **S** sed-only | copy + `s/fm_visual_aligning/mix_visual_aligning/g` | 9 | 0 |
| **G** graft | copy of file A + a block **pasted verbatim** out of file B | 3 | ~40 (glue + comments) |
| **N** new | genuinely new code | 4 | ~350 |
| | | | **≈ 390 total** |

Anything that pushes **N** past ~400 lines means the plan drifted from reassembly into rewriting.

### 3.3 File tree with provenance and kind

```
mix_visual_aligning/
├── __init__.py                          S  ← Gen7
├── setup.py                             S  ← Gen7
├── datasets/                            S  ← Gen7 (normalization.py, sequence.py, __init__.py)
├── sampling/
│   ├── __init__.py                      S  ← Gen7
│   └── projection.py                    V  ← Gen7  (== Gen6V4 byte-for-byte; DPCC projector)
├── utils/
│   ├── arrays.py constraints_helpers.py logger.py plot.py progress.py serialization.py
│   │                                    V  ← Gen7
│   ├── config.py  setup.py              S  ← Gen7
│   ├── training.py                      S  ← Gen7 VERBATIM      → used by ddpm, fm
│   └── training_twotime.py              S  ← Gen3v7 VERBATIM    → used by mf, af   (§4)
└── models/
    ├── __init__.py                      N  ← exports the four engines (~20 lines)
    ├── engine_registry.py               N  ← the dispatch table (~60 lines)          (§5)
    ├── helpers.py                       S  ← Gen7  (apply_conditioning, Losses)
    │
    │  ── arm: ddpm ──────────────────────────────────────────────────────────────
    ├── diffusion.py                     S  ← Gen6V4 `GaussianDiffusion`
    ├── visual_gaussian_diffusion.py     S  ← Gen6V4 `VisualGaussianDiffusion`
    │
    │  ── arm: fm ────────────────────────────────────────────────────────────────
    ├── fm_diffusion.py                  S  ← Gen7 `models/diffusion.py` (renamed file only)
    ├── visual_fm_diffusion.py           S  ← Gen7 `visual_gaussian_diffusion.py` (renamed)
    │
    │  ── shared by ddpm + fm — UNTOUCHED, guarantees §3.1 ────────────────────────
    ├── visual_unet.py                   S  ← Gen7 (film_mode v1/v2 selector intact)
    ├── unet1d_temporal_cond.py          S  ← Gen7 `UNet1DTemporalCondModel`
    ├── unet1d_temporal_film.py          S  ← Gen7 `UNet1DTemporalFiLMModel`
    │
    │  ── arm: mf ────────────────────────────────────────────────────────────────
    ├── mf_diffusion.py                  S  ← Gen3v6 `MeanFlowODE`
    ├── mf_engine.py                     G  ← Gen3v6 + `if_vision`/`vis_config` passthrough
    ├── mf_trajectory_model.py           G  ← Gen3v6 + visual branch (L2 shape, ~15 lines)
    ├── visual_mf_diffusion.py           N  ← `VisualMeanFlow`  (~90 lines; §6.1)
    │
    │  ── arm: af ────────────────────────────────────────────────────────────────
    ├── af_diffusion.py                  S  ← Gen3v7 `AlphaFlowODE`
    ├── af_engine.py                     S  ← Gen3v7 (mf_engine's twin; same graft, copied)
    ├── af_trajectory_model.py           S  ← Gen3v7 (same graft, copied)
    ├── visual_af_diffusion.py           N  ← `VisualAlphaFlow` (~90 lines; §6.2)
    │
    │  ── shared by mf + af only — the ONE real graft ─────────────────────────────
    ├── unet1d_twotime_cond.py           G  ← Gen3v6 `Flow_matcher_U_Net_v2`
    │                                         + Gen7's cond_mlp block pasted in   (§6.3)
    └── visual_unet_twotime.py           N  ← VisualUNet twin: `h=` + latent cache (~80 lines)

mix_visual_aligning_test/
├── train_mix_visual_aligning.py         G  ← Gen7 train + `--engine` dispatch (~40 lines)
├── eval_mix_visual_aligning.py          G  ← Gen7 eval (2838 lines) + engine dispatch (~30 lines)
├── gates_mix_visual.py                  N  ← G0…G6 of §8 (~120 lines)
├── plot_yaml_constraints.py             V  ← Gen7
└── README_plot_constraints.md           V  ← Gen7

config/aligning-d3il-visual.py           APPENDED — 8 blocks by inheritance (§7). Nothing edited.
Slurm_Codes/sbatch/mix_visual_aligning/  S  ← Gen7's trio + one new gates job (§9)
```

**Note `af_engine.py` / `af_trajectory_model.py` are marked S, not G.** Gen3v6's and Gen3v7's copies
of these two files are already twins of each other; the visual graft is written **once** into the mf
pair and then **copied** into the af pair. Two near-identical files, zero extra authored lines —
this is the redundancy principle doing its job.

---

## 4. Trainers: two files, side by side, neither merged

Gen7's trainer is 343 lines; Gen3v7's is 495. The delta is entirely additive and all of it is wanted
by `mf`/`af` — but **merging them into one file is exactly the kind of rewrite this plan forbids.**
So: copy both, verbatim, and let the frame pick.

| Gen3v6/v7 trainer feature | Needed by | Why |
|---|---|---|
| `EXTRA_METRIC_KEYS` passthrough of `_build_info` | mf, af | MF reports `raw_mse_u`, `raw_mse_v`, `h_mse_b0..b3`, `h_mean`, `fm_frac`; AF adds `alpha`, `discrete_frac`, `clamp_frac`. Without it the only visible number is the adaptive loss, which is **pinned at its ceiling by construction** and says nothing about convergence (COMPARE §7.1). |
| `gradient_clip` actually applied before `optimizer.step()` | mf, af | Was a dead config key in this lineage while Gen3v4/Gen13 logged 65–500× loss spikes. |
| `split_seed=42` on `random_split` | mf, af | Unseeded split re-splits on resume, leaking test trajectories into training. |
| `set_train_step(self.step)` before the loss call | **af (mandatory)** | α-Flow's `current_alpha()` reads it. No-op for the others. |

**Why Gen3v7's copy and not Gen3v6's:** Gen3v7's is a strict superset (it adds only the α telemetry
keys and the `set_train_step` hook), and it has **LF line endings** — Gen3v6's `training.py` is
**CRLF**, which makes every subsequent `diff` in this generation unreadable. Copy from Gen3v7.

**The one thing the copied `training_twotime.py` must gain (graft, not rewrite):** Gen7's wandb
curve-logging and visual dataloader wiring. Paste those blocks in from Gen7's `training.py` verbatim.

`split_seed=42` in the two-time trainer vs Gen7's unseeded split means **`mf`/`af` train on a
different train/test split than `ddpm`/`fm`.** That is a real confound for cross-arm comparison.
Decision needed in §1 review: either accept it and document, or set `split_seed` in the Gen7 trainer
too — which breaks §3.1's "verbatim" guarantee for the ddpm/fm arms. **Recommendation: accept and
document**, then compare arms on task success (which is split-independent at eval time, since eval
rollouts use env contexts, not held-out trajectories), never on `test_loss`.

---

## 5. The frame: one config key, one dispatch table

`engine` ∈ `{ddpm, fm, mf, af}`. **No `if engine == ...` chains anywhere in the train/eval scripts** —
every branch lives in `engine_registry.py`.

```python
# mix_visual_aligning/models/engine_registry.py            (N, ~60 lines)
_P = 'mix_visual_aligning.models.'
ENGINES = {
 'ddpm': dict(diffusion=_P+'visual_gaussian_diffusion.VisualGaussianDiffusion',
              model=_P+'visual_unet.VisualUNet',            wraps_unet=False,
              trainer='utils.training',          nfe_key='n_diffusion_steps'),
 'fm':   dict(diffusion=_P+'visual_fm_diffusion.VisualFlowMatching',
              model=_P+'visual_unet.VisualUNet',            wraps_unet=False,
              trainer='utils.training',          nfe_key='flow_steps_v3'),
 'mf':   dict(diffusion=_P+'visual_mf_diffusion.VisualMeanFlow',
              model=_P+'mf_engine.MeanFlowEngine',          wraps_unet=True,
              trainer='utils.training_twotime',  nfe_key='flow_steps_v3'),
 'af':   dict(diffusion=_P+'visual_af_diffusion.VisualAlphaFlow',
              model=_P+'af_engine.AlphaFlowEngine',         wraps_unet=True,
              trainer='utils.training_twotime',  nfe_key='flow_steps_v3'),
}
```

**`wraps_unet` is the one structural asymmetry and it stays explicit.**
`ddpm`/`fm` hand a `VisualUNet` straight to the engine: `GaussianDiffusion(model=VisualUNet(args))`.
`mf`/`af` need the two-time `(u,v)` surface, so it is
`MeanFlowODE(model=MeanFlowEngine(..., if_vision=True, vis_config=args))`, and the engine's
`MFTrajectoryModel` builds `VisualUNetTwoTime` internally as its `velocity_net` — Gen8's L2 shape,
re-derived from the live Gen3v6 source.

Consequence for eval: `load_diffusion_with_override()` reconstructs whatever `model_config.pkl`
names, so it works unchanged **provided the arm matches**. Add one assert —
`assert args.engine == ckpt_args['engine']`. Today an arm mismatch surfaces as an opaque
`state_dict` key error minutes into a GPU allocation.

---

## 6. The three grafts — the only places real thinking happens

### 6.1 🔴 MeanFlow's JVP would run *through the vision encoder*

`MeanFlowODE._p_losses_meanflow` (`mf_diffusion.py:443-454`):

```python
def _u_of(z_in, r_in, h_in):
    u, _v = self._predict_uv(z_in, cond, r_in, h=h_in, returns=returns)
    return u
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
```

The closure captures `cond`. State-only, that is a cheap U-Net call. **Visual**, `cond['visual']`
holds `(primary_img, wrist_img, obs_seq)` and `VisualUNet.forward` calls `encode_visual()` → two
ResNet-18 forwards on `(B·T, 3, 96, 96)` **inside the JVP**. Three consequences:

1. ~2× compute and ~2× activation memory on the ResNets for nothing — the image latent is constant
   w.r.t. all three tangents (`z`, `r`, `h`), so its forward-mode derivative is identically zero.
2. `_predict_uv` is called again at `mf_diffusion.py:467` for the `v` head → a **third and fourth**
   ResNet forward per training step.
3. Forward-mode AD through torchvision ResNet + `GroupNorm` is the single most likely thing in this
   plan to simply raise (`NotImplementedError: jvp for aten::...`). **Smoke-test it on the cluster
   before anything else in P3.**

**Fix — pre-encode once, pass a tensor** (this is what `visual_unet_twotime.py` exists for):

```python
# VisualMeanFlow.loss(), BEFORE building the closure:
vis_latent = self.model.model.velocity_net.encode_visual(primary_img, wrist_img)   # (B, 128)
cond = {0: obs_0, 'visual_latent': vis_latent}          # ← tensor; no images downstream
```

`VisualUNetTwoTime.forward` short-circuits: if `cond` carries `'visual_latent'`, use it and skip
`encode_visual`. The JVP then differentiates the 1-D U-Net only, and `vis_latent` is a captured
constant, so its tangent is zero **by construction** rather than by hope.

🔴 **Do not wrap `encode_visual` in `no_grad` by default.** The vision encoder is trained end-to-end
in Gen6V4/Gen7 (it lives inside `VisualUNet`, in the optimizer's param list). Pre-encoding *under
grad* keeps it trainable through both the `u` and `v` paths; only the JVP tangent needs to be zero,
and capturing the tensor already achieves that. Expose `mf_freeze_vision_encoder: False` as an
explicit ablation knob, default OFF, and record which way each checkpoint was trained.

**Second-order note:** the MeanFlow target is `.detach()`-ed (`mf_diffusion.py:461`), so no
second-order graph through the encoder is ever built. Keep it that way.

**Time-schedule collision.** Gen3v6 draws `(t, r)` from **two independent logit-normals**
(`_sample_tau_pair`, `p_mean=-0.4`, `p_std=1.0`); the visual blocks are built around
`1 − Beta(1.5, 1.0)`. Gen3v6 already carries the Beta path as a legacy ablation arm, so both work —
but `t_schedule` **must** be a checkpoint-path key. Default for `mf`/`af`: `logit_normal`.
🔴 The sign trap at `mf_diffusion.py:360` carries over verbatim: **`−p_mean`, not `+p_mean`.**

### 6.2 🔴 α-Flow needs the global step, and evaluates the net a second time

**(a) Step plumbing.** `AlphaFlowODE.current_alpha()` reads `self._train_step`, pushed in by the
trainer via `set_train_step()`. Covered by copying Gen3v7's trainer (§4) — but respect the assert:

```python
# af_diffusion.py:161-168
if af_n_train_steps is not None and scheduler not in ('constant','step'):
    assert int(af_n_train_steps) == self.af_alpha_end_step
```

The visual blocks train `n_train_steps = 1e5`; α-Flow's default `af_alpha_end_step` is `100000`.
They coincide **today**, and that coincidence is a trap: bump the visual budget to 2e5 and you get an
assert; delete the assert and you get an α-anneal that finishes at the halfway mark, silently
training the back half as pure MeanFlow. **Derive `af_alpha_end_step` and `af_n_train_steps` from the
single `n_train_steps` variable in the config — never two literals.**

**(b) The bootstrap doubles the network cost.** `compute_u_target` (`af_diffusion.py:529-635`)
evaluates the net at the shifted point `(z_r + dt·v, r+dt, h−dt)` under `no_grad`. In visual mode
that is another full ResNet pair unless §6.1's cached latent is reused. **Implement §6.1 first — the
α-Flow arm depends on it.** With the latent cached, per-step cost is ~1 ResNet pair + 3 U-Net calls,
comparable to Gen7's 1 + 1.

Carried over unchanged: the target clamp at `4.0` (watch `clamp_frac` — a rising value means the
bootstrap is diverging *before* the anneal reaches it), and the α = 0 branch, which is literally
Gen3v6's JVP loss and therefore inherits §6.1 wholesale.

### 6.3 The backbone graft — one paste, ~15 lines

Verified by inspection:

| | `h_mlp` (two-time) | `cond_mlp` / `use_cond_projection` (visual FiLM) |
|---|---|---|
| Gen7 `UNet1DTemporalCondModel` | ❌ | ✅ (`unet1d_temporal_cond.py:97,120-131,220-227`) |
| Gen3v6 `Flow_matcher_U_Net_v2` | ✅ (`:120, :245`) | ❌ |
| Gen3v7 `Flow_matcher_U_Net_v2` | identical to Gen3v6 modulo package name | ❌ |

So `unet1d_twotime_cond.py` = **copy Gen3v6's `Flow_matcher_U_Net_v2` verbatim**, then **paste Gen7's
`cond_mlp` construction block and its `forward` integration block in verbatim**. Both grafts are
additive; neither touches the `h_mlp` path or the interval-CFG path. Gen7's own file is never edited,
so the `ddpm`/`fm` arms keep their exact current behaviour (§3.1).

`visual_unet_twotime.py` = copy Gen7's `visual_unet.py`, point it at `unet1d_twotime_cond`, add
`h=None` to the `forward` signature and pass it through, and add the `'visual_latent'` short-circuit
of §6.1. ~80 lines, of which ~70 are the copy.

### 6.4 Smaller items, all real

- **NFE key collision.** `ddpm` reads `n_diffusion_steps`; `fm`/`mf`/`af` read `flow_steps_v3`.
  Resolve via the registry's `nfe_key`, emit one `K` fragment in the plan prefix, and make the eval
  banner print `engine=<e> NFE=<k>` unambiguously.
- **🔴 Low-NFE projector hook — a latent Gen7 bug.** Gen3v6 guards the projector with
  `near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)`
  (`mf_diffusion.py:283-284`). Gen7's FM loop has **only** the threshold term
  (`fm_visual_aligning/models/diffusion.py:178`). At `flow_steps_v3=1` with a low threshold the DPCC
  projection can be skipped **entirely**, and the run reports "FM is unsafe" when in fact nothing was
  ever projected. Gate G6 catches it. **Report it as a Gen7/Gen6V4 finding — do not silently absorb
  it into Gen14**, and note that fixing it inside Gen14's `fm` arm would breach §3.1, so the fix
  belongs upstream in Gen7 (its own hotfix commit), after which Gen14 re-copies.
- **`goal_dim`.** All four engines slice `x[:, :, :-goal_dim]` before projecting. Visual aligning runs
  `goal_dim=0`, so the branch is inert — but every
  `apply_conditioning(..., goal_dim=self.goal_dim)` call must be preserved verbatim, not simplified.
- **Temporal-consistency candidate selection.** Gen3v6/v7 `sampling/policies.py` carry the fixes from
  `ecbae16f` and `a6a7a8ad` (reference index / double-permutation). The visual path selects MPC
  candidates inside `eval_*.py`'s `VisualAgentWrapper`, not in `policies.py`. **Audit the visual
  selector against those two commits.** If the same double-permutation exists there, it is a real bug
  in current Gen6V4/Gen7 results and must be reported before any Gen14 number is read.

---

## 7. Config blocks (`config/aligning-d3il-visual.py`, appended — existing blocks untouched)

Eight blocks: four train + four plan, built by **inheritance from `fm_visual_aligning`** so drift is
structurally impossible.

```python
args_to_watch_mix_visual_train = args_to_watch_fm_visual_train + [
    ('engine', 'E'),          # ddpm | fm | mf | af   ← the arm identity key
    ('t_schedule', 'ts'),     # logit_normal | beta   (mf/af only; watch() skips absent keys)
]

base['mix_visual_aligning_fm'] = {                 # reference arm — must equal Gen7 exactly
    **base['fm_visual_aligning'],
    'engine': 'fm',
    'model':     'mix_visual_aligning.models.visual_unet.VisualUNet',
    'diffusion': 'mix_visual_aligning.models.visual_fm_diffusion.VisualFlowMatching',
    'prefix':    'mix_visual_aligning/',
    'exp_name':  watch(args_to_watch_mix_visual_train),
}
base['mix_visual_aligning_ddpm'] = {**base['mix_visual_aligning_fm'], 'engine': 'ddpm',
    'diffusion': _P+'visual_gaussian_diffusion.VisualGaussianDiffusion',
    'n_diffusion_steps': 100, 'predict_epsilon': True}

base['mix_visual_aligning_mf'] = {**base['mix_visual_aligning_fm'], 'engine': 'mf',
    'model': _P+'mf_engine.MeanFlowEngine', 'diffusion': _P+'visual_mf_diffusion.VisualMeanFlow',
    't_schedule': 'logit_normal', 'p_mean': -0.4, 'p_std': 1.0,
    'mf_adp_p': 1.0, 'mf_adp_eps': 0.01, 'meanflow_data_proportion': <Gen3v6 value>,
    'gradient_clip': 1.0, 'split_seed': 42, 'mf_freeze_vision_encoder': False}

base['mix_visual_aligning_af'] = {**base['mix_visual_aligning_mf'], 'engine': 'af',
    'model': _P+'af_engine.AlphaFlowEngine', 'diffusion': _P+'visual_af_diffusion.VisualAlphaFlow',
    'af_alpha_scheduler': 'sigmoid', 'af_alpha_init': 1.0, 'af_alpha_end': 0.0,
    'af_alpha_gamma': 25.0, 'af_alpha_clamp': 0.005,
    'af_alpha_end_step': _N_TRAIN_STEPS, 'af_n_train_steps': _N_TRAIN_STEPS}   # one variable, twice
```

🔴 **The oldest trap in this repo, verbatim from the Gen8 block comments:** each `plan_*` block's
`diffusion_loadpath` must reproduce `args_to_watch_mix_visual_train` **exactly**, key for key. A
missing `_E{engine}` or `_ts{t_schedule}` fragment resolves to a non-existent directory and the eval
dies minutes into a GPU allocation. **Derive the plan format string from the same list** — never
retype it.

---

## 8. Gates — run on the cluster, in order, before any result is believed

| | Gate | Pass condition |
|---|---|---|
| **G0** | **Copy fidelity, mechanical.** `diff` every V/S file against its source, modulo the package-name `sed`. | Zero code diffs. This is the gate that enforces the whole plan. |
| **G1** | **Reference-arm equivalence.** `engine=fm` and `engine=ddpm`, seed 0, 50 train steps; compare `losses.pkl` to Gen7 / Gen6V4. | Identical to float64. Guaranteed by §3.1 — a failure means the frame leaked into the reference path. |
| **G2** | **JVP survives the visual path.** One `engine=mf` step, `if_vision=True`. | No `NotImplementedError`; `du_dr` finite; peak GPU memory within 1.3× of the `fm` arm — proving §6.1's pre-encoding is actually live. |
| **G3** | **MeanFlow identity at h=0.** Force `meanflow_data_proportion=1.0`. | `u_target == v_inst` to `1e-6`; the loss reduces to the FM loss. |
| **G4** | **α spans the real budget.** Read `alpha` from the metric stream at steps 0, N/2, N. | `α(0)≈1.0`, `α(N)≈0.0`, monotone. Flat or early-saturating ⇒ the §6.2(a) trap fired. |
| **G5** | **α→0 limit is MeanFlow.** Pin `af_alpha_init=af_alpha_end=0.0`, same seed as the `mf` arm. | `raw_mse_u` matches the `mf` arm to solver noise. |
| **G6** | **Projector fires at K=1.** `engine=mf`, `flow_steps_v3=1`, projection ON; then repeat on `engine=fm`. | `projection_costs[0]` populated and non-trivial. The `fm` leg is expected to **FAIL** — that is §6.4's Gen7 bug, and the failure is the finding. |

**Read `raw_mse_u`, never `diffusion_loss`** — the adaptive loss sits at its ceiling by construction.
Per the Gen13 verdict: **rank arms by unguided task success only**; post-projection roughness measures
the NLP, not the model.

---

## 9. SLURM

`Slurm_Codes/sbatch/mix_visual_aligning/` — copy Gen7's trio, add one:

- `train_mix_visual_aligning.sh` — `--engine` as `$1`, default `fm`.
- `eval_mix_visual_aligning.sh` — same, plus the §5 arm assert.
- `mix_visual_aligning_pipeline.sh` — train → eval chain, per engine.
- `gates_mix_visual.sh` — **new**: runs G0–G6 in one short job. This is what keeps the merge honest;
  without it nobody re-checks fidelity after the next upstream sync.

Standing cluster rules apply: `--time` = 2× expected (24 h cap), no `tqdm`/live progress bars in batch
logs, never break GPU/EGL isolation. Register runs in
`Slurm_Codes/logs/important_runs/important_runs.md`.

---

## 10. Assembly order — each phase ends green before the next starts

| Phase | What | Newly-authored lines | Gate |
|---|---|---|---|
| **P0** | Create the folder pair as a **pure `sed` rename of Gen7**. Zero behaviour change. | 0 | G0 |
| **P1** | Copy Gen6V4's two engine files in; add `engine_registry` + the `--engine` switch. Two arms live. | ~120 | G0, G1 |
| **P2** | Copy Gen3v7's trainer in as `training_twotime.py` + graft Gen7's wandb/dataloader blocks. Still two arms. | ~30 | G1 again |
| **P3** | `mf` arm: copy Gen3v6's engine stack, build the two backbone files (§6.3), write `VisualMeanFlow` (§6.1). | ~190 | G2, G3, G6 |
| **P4** | `af` arm: copy Gen3v7's engine stack (same graft, copied), write `VisualAlphaFlow` (§6.2). | ~50 | G4, G5 |

P2 is deliberately a **no-op phase** — the two-time trainer is added but nothing uses it yet, so G1
must still pass. That isolates "did the frame break the reference arms?" from "does MeanFlow work?".

---

## 11. What this generation does NOT claim, and what is deferred

- **Not a new objective.** MeanFlow and α-Flow are Gen3v6/Gen3v7's contributions. Gen14 moves them
  onto the visual task and the DPCC cage; that is all.
- **A merged codebase does not make few-NFE work.** The honest prior from the Gen3v7 plan stands:
  α-Flow's own margin over MeanFlow is real but modest, and the aligning dataset is still the
  aligning dataset. Pre-register kill criteria per arm before looking at results.
- **Not comparable to Gen13.** Different constraint mechanism, different task.
- **Deferred, each its own plan:** HardFlow sampler on the visual normalizer (§1.8); the
  `mf_dit`/`af_sit` backbones (needs a visual-conditioning path first); the avoiding-task port
  (Gen9's `fm_visual_avoiding` would be the frame base, not this one).

---

## 12. Hand-off checklist

- [ ] §1 decisions confirmed — especially the folder name and §4's `split_seed` confound call.
- [ ] P0 done and **G0 green** before a single engine is added.
- [ ] `git diff --stat` after P0 shows *only* renames and package-name string changes.
- [ ] Nothing anywhere imports from, or was copied out of, `imf_visual_aligning/` (§2).
- [ ] §6.1 pre-encoded-latent path implemented and **G2 green** before α-Flow is touched.
- [ ] `af_alpha_end_step` and `af_n_train_steps` both derive from one config variable.
- [ ] Gen7's missing terminal-step projector fallback (§6.4) **reported upstream as a Gen7 finding**,
      fixed in Gen7, then re-copied — not patched inside Gen14.
- [ ] `VisualAgentWrapper` candidate-selection audit done against `ecbae16f` / `a6a7a8ad`.
- [ ] Final newly-authored line count reported against the §3.2 budget (~390).
- [ ] `MASTER_TEST_HISTORY.md` row **drafted below, NOT applied** — the master index is never edited
      without an explicit instruction.

### Draft MASTER row (prepared, NOT applied)

| **Gen14 (Visual Mix-ML)** | `mix_visual_aligning/` | `mix_visual_aligning_test/` | July 2026 | ⚠️ **Gen14 = DPCC math; Gen13 = HardFlow math — never mix results.** One visual-aligning **frame** with four config-activated ML engines — **DDPM (Gen6V4) + FM-ODE (Gen7) + MeanFlow (Gen3v6) + α-Flow (Gen3v7)** — on a locked `VisualUNet` backbone, so the four-way comparison is architecture-controlled. Built by **reassembly, not rewriting**: ~390 newly-authored lines, everything else a verbatim/sed copy, with the `ddpm`/`fm` arms importing **only** verbatim copies so reference-arm fidelity is structural. Key engineering: a pre-encoded visual latent so MeanFlow's JVP and α-Flow's bootstrap never differentiate the ResNets. **No iMF arm** (Gen3v4 abandoned); Gen8 `imf_visual_aligning/` is dead code, reference-only. HardFlow deferred. Plan: [`Gen14/init/PLAN_Gen14_visual_mix_ml.md`](./Gen14/init/PLAN_Gen14_visual_mix_ml.md). | planning |
