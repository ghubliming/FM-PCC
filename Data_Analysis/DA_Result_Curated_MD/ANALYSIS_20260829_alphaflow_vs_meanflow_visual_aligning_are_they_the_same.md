# `af` vs `mf` on visual-aligning — is the training target changed, and what is left once the bone is shared?

**Date** 2026-08-29 · **Scope** Gen14 `mix_visual_aligning`, `engine=mf` (`cand14`) vs `engine=af`
(`cand6`), both `unet` bone / `filmv1` / `K = 2`.
**Method** source diffs + verbatim quotes from this repo and the two upstreams in `/workspaces/aux_repo/`.
Nothing was executed (no torch in this container); the α curve is `_get_ratio` evaluated in pure Python.

---

## The five answers, up front

| question | answer |
|---|---|
| **Q1. Is the training target changed?** | **Yes — but only through one scalar α, and at α = 0 the α-Flow target is the MeanFlow target, branch-for-branch.** |
| **Q2. Where is α-Flow's upgrade point over MeanFlow?** | **The target's *schedule*. Not the ML bone, not the sampler, not the time distribution.** Upstream held the network fixed too — their MeanFlow baseline is α-Flow's own loss with `alpha: constant 0.0`. |
| **Q3. Training only, or eval too?** | **Training only.** α never appears at inference. `diff` of the two samplers = 29 lines, *all inside comments*. |
| **Q4. On V_A, same visual U-Net — what remains?** | **Three things, none of them "a different objective":** (a) the optimisation *path* (28.8 % pure FM → 42.4 % bootstrapped → **28.8 % identical MeanFlow loss**), (b) one constant, `af_adp_eps` 1e-3 vs `mf_adp_eps` 1e-2, which the code calls near-inert, (c) the different weights those produce — visible as the **inverted K response** (§4.3). |
| **Q5. Then why not leave α on?** | 🔴 **We should. The α → 0 snap costs ~2.5× on `raw_mse_u`** — af hits 2.657 at α ≈ 0.007 (step 70 k) and jumps to 8.504 the step α is clamped to 0, landing on MeanFlow's own plateau and never recovering. Measured in Gen14 U5 on 2026-08-04; the fix was logged as the highest-value training job and **never run**. §5. |

> **One-line summary.** α-Flow *is* a training curriculum for MeanFlow — that is its own paper's
> claim (*"AlphaFlow: Understanding and **Improving MeanFlow** Models"*). On V_A we removed the one
> axis on which the two originally differed in *architecture*, so what separates `cand6` from
> `cand14` is a curriculum, not two engines. **Do not report them as two independent wins.**
>
> **And the curriculum ends in the wrong place.** The V_A numbers do not measure α-Flow at its own
> operating point — they measure the model that is left after α is clamped to 0 at step 71 180,
> which this repo has already measured as 2.5× worse than the same run 1 000 steps earlier (§5).
> The one experiment that would settle mf-vs-af — a **constant-α** arm on the same visual U-Net —
> has never been run.

---

# Q1. Is the training target changed?

**Yes.** It is the only thing that is. Here are both targets, side by side, verbatim.

### MeanFlow — the JVP target

```python
# mix_visual_aligning/models/mf_diffusion.py:454-461
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
...
# MeanFlow-Identity target (START-anchored): u ← v + h·du/dr ; stop-gradient.
u_target = (v_inst + h_exp * du_dr).detach()
```

One target, always. `u_tgt = v + h·du/dr`, with tangents `(v_inst, +1, −1)` because `h = t − r`.

### α-Flow — a **three-way** target selected by α

```python
# mix_visual_aligning/models/af_diffusion.py:552-583, 592-632  (compute_u_target, condensed)
if alpha <= 0.0:
    # CONTINUOUS BRANCH (α = 0) — Gen3v6's `_p_losses_meanflow` body, UNMODIFIED.
    _u_primal, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
    u_target = (v_inst + h_exp * du_dr).detach()          # ← identical to MeanFlow's
else:
    if alpha >= 1.0:
        u_target = v_inst.clone()                         # ← pure Flow Matching
    else:
        dt      = alpha * h
        z_shift = x_r + dt_exp * v_inst                   # travel dt toward data at velocity v
        with torch.no_grad():
            u_next, _ = self._predict_uv(z_shift, cond, r + dt, h=h - dt)
            # (dt·v + (h−dt)·u_next) / h  ==  α·v + (1−α)·u_next
            u_target = (dt_exp * v_inst + (h_exp - dt_exp) * u_next) / h_safe
            u_target = u_target.clamp(-self.af_clamp_utgt, self.af_clamp_utgt)
            u_target = torch.where(h_exp > 0, u_target, v_inst)   # FM anchors keep v
```

| α | target | needs a JVP? | equals |
|---|---|---|---|
| **1.0** | `u_tgt = v` | no | **plain Flow Matching** |
| 0 < α < 1 | `u_tgt = α·v + (1−α)·u_next`, clamped ±4 | no — a *second forward* instead | the α-Flow-specific bootstrap |
| **0.0** | `u_tgt = v + h·du/dr` | **yes** | **MeanFlow, bit-identical** |

The docstring states the two endpoints as gates:

```python
# af_diffusion.py:677-680
#   α = 0 ⇒ dt = 0 ⇒ the JVP branch, character-for-character Gen3v6's
#           `_p_losses_meanflow`. First-order expansion of u_next recovers the
#           MeanFlow identity u = v + h·D_tot — PLAN §3.4.                 [G2]
```

**The mechanism of the change matters:** MeanFlow's target contains a *derivative of the network*,
so the target moves as the network moves. α-Flow's bootstrapped target is a **fixed tensor** (one
extra `no_grad` forward). That is the substantive idea — *"a fixed target has no blind direction, so
unlike the MeanFlow residual it cannot hide an error with δ_u = h·δ_D"* (`af_diffusion.py:667-668`).

### Everything else in the loss is α-gated too

```python
# af_diffusion.py:736-738
w_br = torch.where(discrete_mask, torch.full_like(err_u, alpha), torch.ones_like(err_u))
loss = (w_br * self._adaptive(err_u) + self._adaptive(err_v)).mean()
```

`discrete_mask = mf_mask & (alpha > 0.0)`. At α = 0 it is all-False ⇒ `w_br ≡ 1` ⇒ this reduces to
MeanFlow's line exactly:

```python
# mf_diffusion.py:473
loss = (self._adaptive(err_u) + self._adaptive(err_v)).mean()
```

The `af_clamp_utgt = 4.0` clamp lives only inside the discrete branch and is unreachable at α = 0.

---

# Q2. Where is α-Flow's upgrade point over MeanFlow?

**The target's schedule — and nothing else.** The upgrade is *not* a bone, *not* a sampler, *not* a
new time distribution. Here is the evidence, axis by axis.

## 2.1 🔑 Upstream's own MeanFlow baseline is α-Flow with one line changed

Not a Gen14 artefact. In the α-Flow authors' repo both methods are the **same loss class, same
network, same budget**; the experiment file differs by the α scheduler:

```yaml
# aux_repo/alphaflow/infra/experiments/experiments-alphaflow.yaml:133-136   ← their MeanFlow
alphaflow-meanflow-latentspace-B-2:
  _inherit_: alphaflow-latentspace-B-2
  overrides:
    loss: {alpha: {scheduler: constant, initial_value: 0.0}, ratio_fm: {scheduler: constant, initial_value: 0.75}}

# ...:152-155                                                              ← their α-Flow
alphaflow-sigmoid-latentspace-B-2:
  _inherit_: alphaflow-latentspace-B-2
  overrides:
    loss: {alpha: {scheduler: sigmoid, initial_value: 1.0, end_value: 0, change_init_steps: 0, change_end_steps: 400000, clamp_value: 0.005, gamma: 25.0}, ratio_fm: {scheduler: constant, initial_value: 0.5}}
```

Both `_inherit_` the identical `num_blocks: 12, dim: 768, num_heads: 12` and `max_steps: 400000`
(line 86-91). **The upgrade point is the `alpha:` line.** Measured gain, same repo, same net:
FID 43.1 → 40.2 at NFE-1 for B/2 (`aux_repo/alphaflow/README.md:32-33`).

Note `change_end_steps: 400000 == max_steps: 400000` — **upstream's anneal also spans the whole
budget**, so upstream's α-Flow-B/2 also finishes ~28.8 % of training at α = 0 (§2.2). Gen14 copied
the recipe faithfully and inherited the property with it.

## 2.2 The schedule, on this task

```python
# config/aligning-d3il-visual.py:1435-1446  (af arm) — the upstream recipe, ported
    # The alpha anneal: 1 -> 0 means training starts as plain flow matching and becomes
    # MeanFlow. 🔴 end_step is bound to _MIX_N_TRAIN_STEPS, the same name that sets
    # n_train_steps below.
    'af_alpha_scheduler': 'sigmoid',
    'af_alpha_init': 1.0,
    'af_alpha_end': 0.0,
    'af_alpha_init_step': 0,
    'af_alpha_end_step': _MIX_N_TRAIN_STEPS,
    'af_alpha_gamma': 25.0,
    'af_alpha_clamp': 0.005,
    'n_train_steps': _MIX_N_TRAIN_STEPS,
```

`_MIX_FULL_N_TRAIN_STEPS = int(1e5)` (config line 877); `cand6`'s folder carries no `TB` budget tag,
so it is the full 100 000 steps. The clamp forces *exact* endpoints rather than asymptotic ones:

```python
# af_diffusion.py:472-475  (_get_ratio tail)
if ratio < clamp_value:
    ratio = 0.0
elif ratio > 1.0 - clamp_value:
    ratio = 1.0
```

Evaluating `_get_ratio` over the real budget (sigmoid, γ = 25, clamp = 0.005, end = 1e5):

| optimizer steps | α | what is actually being trained | share |
|---|---|---|---|
| 0 → 28 820 | **1.0 exactly** | **pure Flow Matching** (`u_target = v_inst.clone()`) | 28.8 % |
| 28 830 → 71 170 | 1.0 → 0.0 | bootstrapped target, sample weight α | 42.4 % |
| **71 180 → 100 000** | **0.0 exactly** | **MeanFlow — `cand14`'s exact objective** | **28.8 %** |

**The checkpoint behind every `af` number in the V_A reports was last optimised against MeanFlow's
loss, for 28 820 steps.**

## 2.3 Not the ML bone — and Gen14 removed the one place it ever was

Originally the two methods *did* ship different networks. This repo ports both, and its own header
tabulates the difference:

```python
# mix_visual_aligning/models/af_sit_trajectory.py:5-16
# It is NOT MeanFlow's MFDiT and NOT the iMF DiT ... All three are the DiT/SiT
# family but differ on every learned detail:
#
#                   α-Flow SiT (here)                 MeanFlow MFDiT            iMF DiT ('dit')
#   norm            **LayerNorm** (affine off, fp32)   RMSNorm                  RMSNorm
#   QK-norm         **OFF** (`qk_norm=False`)          ON (QK-RMSNorm)          ON (RoPE + QK-RMSNorm)
#   positions       frozen sin-cos (requires_grad=F)   learned sin-cos (grad)   RoPE
#   time embed      freq=256, **no scale**             freq=256, **scale=1000**  (rope)
#   heads           **single** (u only) + analytic v   twin u/v FinalLayers     shared trunk → head blocks
```

Both native bones exist here (`af_sit_trajectory.py`, `mf_dit_official_trajectory.py`) and are
reachable via `MIX_BONE_AF=sit` / `MIX_BONE_MF=mf_dit`. **Neither was used.** Every V_A number is
`imf_backbone='unet'`, and the config says why:

```python
# config/aligning-d3il-visual.py:1455-1457  (af arm)
    # Gen14 U8: same bone treatment as the mf arm. MIX_BONE_AF=sit|dit. The mf-vs-af
    # comparison is architecture-controlled only if BOTH arms sit on the same bone — use
    # the bare MIX_BONE to move them together.
```

That is the *right* call for an architecture-controlled claim — and it is also why the answer to Q4
is as thin as it is. Note that upstream is architecture-matched too (§2.1), so **the bone was never
where α-Flow's gain came from** in the first place.

## 2.4 Not the time schedule, not the FM-anchor ratio — the upstreams already agreed

| knob | MeanFlow upstream | α-Flow upstream | Gen14 |
|---|---|---|---|
| time schedule | `time_dist=['lognorm', -0.4, 1.0]` (`MeanFlow/meanflow.py:72`) | `logit_norm`, `location: -0.4`, `scale: 1.0` | both `logit_normal`, `p_mean −0.4`, `p_std 1.0` |
| FM-anchor fraction | `data_proportion: 0.5` (`imeanflow/imf.py:66`) | `ratio_fm: 0.5` in the headline recipes | `meanflow_data_proportion: 0.5` = `af_ratio_fm: 0.5` |

The Gen14 comment on `_sample_tau_pair` says it outright: *"the two upstreams agree here, so nothing
had to change"* (`af_diffusion.py:489-491`). **Gen14 is in fact more controlled than upstream's own
headline B/2 pair**, which differs on `ratio_fm` (0.75 vs 0.5) as well as α.

---

# Q3. Training only, or eval too?

**Training only.** Three independent confirmations.

**1. The sampler is byte-identical.** `diff` of `_predict_uv` … `sample()`
(`mf_diffusion.py:155-340` vs `af_diffusion.py:210-398`) returns **29 lines, every one inside a
comment**. The α-Flow copy states the consequence:

```python
# af_diffusion.py:270-273
# ⚠️ SAMPLER — DO NOT TOUCH. The interval-jump update x += dt·u with h = dt = 1/N is
# already faithful (U10 audit F1, re-verified); iMF, MeanFlow and α-Flow all share
# the identical sampler. ✅ α is TRAINING-ONLY — it does not appear here, which is
# exactly what makes the three-way comparison clean (PLAN §5.3).
```

**2. α is pushed in by the *trainer*, and only there.**

```python
# mix_visual_aligning/utils/training_twotime.py:281-282
if hasattr(self.model, 'set_train_step'):
    self.model.set_train_step(self.step)
```

`MeanFlowODE` has no `set_train_step`, so the same trainer drives both arms; the hook is a no-op on
`mf`. Nothing in `p_sample_loop` reads `self._train_step` or calls `current_alpha()`.

**3. The eval-side config is pinned equal.**

```python
# config/aligning-d3il-visual.py:1516-1524  (af plan block)
        # ── U6 ── see the mf block above for the full rationale. Kept equal to mf's on
        # purpose: NFE is an operating point, and an mf-vs-af comparison at different K
        # would confound the objective with the step budget.
        'flow_steps_v3': 2,
```

and the registry gives both arms the same trainer, wrapper semantics and NFE key:

```python
# mix_visual_aligning/models/engine_registry.py:65-86
    'mf': dict(label='MeanFlow (Gen3v6)',   wraps_unet=True, two_time=True,
               trainer=_U+'training_twotime.Trainer', nfe_key='flow_steps_v3', ...),
    'af': dict(label='alpha-Flow (Gen3v7)', wraps_unet=True, two_time=True,
               trainer=_U+'training_twotime.Trainer', nfe_key='flow_steps_v3', ...),
```

**⇒ At eval time — which is all the V_A results measure — `mf` and `af` execute the same code with
different weights.**

---

# Q4. Same visual U-Net on V_A — what actually remains?

## 4.1 What is provably shared

**The backbone is the same class, not merely the same architecture.** Both trajectory models
instantiate the *one* `VisualUNetTwoTime` in the tree (`visual_unet_twotime.py:54`):

```python
# mf_trajectory_model.py:88-91        AND      af_trajectory_model.py:88-91   (identical)
from .visual_unet_twotime import VisualUNetTwoTime
self.velocity_net = VisualUNetTwoTime(
    vis_config, dual_head=dual_head, interval_cfg=interval_cfg)
state_dim = VisualUNetTwoTime.TRANSITION_DIM
```

`diff mf_trajectory_model.py af_trajectory_model.py` → 66 changed lines, **28 non-comment, every one
a class rename or a branch the `unet` bone never enters** (`sit`/`mf_dit` alternatives, error
strings, a print banner). `diff mf_engine.py af_engine.py` → **8 non-comment lines, all renames**.
The visual wrappers are near-duplicates by design:

```python
# visual_af_diffusion.py:1-6
"""Gen14 — `VisualAlphaFlow`: the alpha-Flow (Gen3v7) engine on the visual-aligning task.

Deliberate near-duplicate of `visual_mf_diffusion.VisualMeanFlow` — same shape, same two
boundary repacks, different base class. ...
```

and the architecture flags are pinned equal on purpose:

```python
# config/aligning-d3il-visual.py:1430-1434  (af arm)
    # 🔴 Same architecture flags as the mf arm — Gen3v7 ships them identically ...
    # Keeping mf and af equal here is also what makes the MeanFlow-vs-alpha-Flow
    # comparison architecture-controlled.
    'dual_head': True,
    'interval_cfg': False,
```

Both arms also inherit one parent block (`_mix_train_block(..., 'fm_visual_aligning', ...)`), so the
budget (1e5), batch (64), LR (2e-4), EMA (0.995), grad-accum (2) and `split_seed` (42) are identical.

## 4.2 What remains — exactly three things

| # | what remains | size |
|---|---|---|
| **1** | **The optimisation path.** `cand6` saw 28 820 steps of pure FM, then 42 400 of bootstrapped target, then 28 820 of the loss `cand14` ran for all 100 000. | the whole difference |
| **2** | **One constant.** `mf_adp_eps = 0.01` vs `af_adp_eps = 1e-3` in the adaptive weight — the *only* numerical difference in the α = 0 loss path. | near-inert (below) |
| **3** | **The weights those two produce**, and everything downstream of them. | measurable — §4.3 |

Everything else in the α = 0 loss matches:

| knob | `mf` | `af` | same? |
|---|---|---|---|
| FM-anchor fraction | `meanflow_data_proportion: 0.5` | `af_ratio_fm: 0.5` | ✅ |
| adaptive exponent | `mf_adp_p: 1.0` → `.pow(1.0)` is a no-op | no exponent knob, fixed 1 | ✅ |
| **adaptive epsilon** | **`mf_adp_eps: 0.01`** | **`af_adp_eps: 1e-3`** | ❌ **the only one** |
| τ schedule | `logit_normal`, −0.4, 1.0 | identical | ✅ |
| `dual_head` / `interval_cfg` | `True` / `False` | `True` / `False` | ✅ |
| budget · batch · lr · EMA · grad-accum · split_seed | 1e5 · 64 · 2e-4 · 0.995 · 2 · 42 | identical (shared parent) | ✅ |

The eps split is deliberate, and the code argues it barely acts:

```python
# config/aligning-d3il-visual.py:1425-1428  (af arm)
    # alpha-Flow's OWN constants. ⚠️ af_adp_eps=1e-3 is DELIBERATELY != MeanFlow's 0.01
    # (af_diffusion.py:97) — different method, different constant. Do NOT harmonise them.

# af_diffusion.py:521-524
# Practical consequence: with SUM, `err ≫ eps` almost always, so the
# adaptive weight is ≈1 and this term is near-inert — exactly as in Gen3v6 ...
```

The Gen3v7 pre-flight gate independently confirms it is the only compensation needed:

```python
# FM_v3_alphaflow_test/gates_alphaflow.py:16-20
  G2  α = 0 ⇒ MeanFlow      the α=0 target must equal **Gen3v6**'s `_p_losses_meanflow`
                            target on identical inputs (<1e-5), and the scalar losses must
                            agree once the adaptive eps is matched.
```

⚠️ That gate applies to *these exact files*: `diff` of `flow_matcher_v3_alphaflow/models/af_diffusion.py`
against `mix_visual_aligning/models/af_diffusion.py` → **0 changed lines**, same for `mf_diffusion.py`.
It has never been run with `if_vision=True` (§6).

## 4.3 The results confirm both halves

**They agree, as the config predicts.** Paired over the same 30 contexts, matched bone, each at its
own best DPCC projector
([`Report_20260829_VA_funnel`](Report_20260829_VA_funnel/README.md)):

| | `dist` | ≤ 15 cm ∧ clean | `0-viol` | `ms` |
|---|---|---|---|---|
| `mf` K2 `dpcc-t` | 0.193 m (0.51×) | 7/30 | 0.27 | 53 |
| `af` K2 `dpcc-t` | 0.264 m (0.63×) | 7/30 | 0.40 | 53 |
| paired | Δ −0.036 m, closer 19/29 · **sign p = 0.136 · Wilcoxon p = 0.393** | | | |

Tightened, the same: 10/30 vs 8/30, `0-viol` 1.00 vs 0.93, 42 vs 43 ms. **This is not a null result
to explain away — it is what §2.2 predicts.**

**And they disagree in the one way that proves distinct weights.** On the unguided arm the two react
to sampler steps in *opposite* directions
([`DA_20260826_K_sampler_steps_visual_aligning.md`](DA_20260826_K_sampler_steps_visual_aligning.md)):

| engine | K = 100 | K = 2 | paired sign / Wilcoxon |
|---|---|---|---|
| MeanFlow | **0.28×** | 0.60× | 0.136 / 0.068 — high K closer |
| alpha-Flow | 0.69× | **0.29×** | **0.008** / **0.031** — 🔴 **low K closer** |

Same class, same sampler, same NFE key, inference-only contrast within each row — and α-Flow gets
*worse* with 50× more Euler steps on 22 of 30 contexts while MeanFlow gets better. **A model whose
converged ODE solution is worse than its 2-step approximation has a different velocity field**, and
the FM-first 28.8 % is the only place that difference can have come from.

---

# Q5. Then why not just leave α on? 🔴 The α→0 snap costs 2.5× on this task

The obvious follow-up to Q1–Q4 is: *if the AF target is the interesting one, why do we schedule it
off?* The answer is that we should not — and this repo already measured what it costs, on
2026-08-04, in
[`logs_in_develop/Gen14/U5/DA_20260804_mf_af_visual_aligning_first_run.md`](../../logs_in_develop/Gen14/U5/DA_20260804_mf_af_visual_aligning_first_run.md) §3.

## 5.1 The measurement

`af_alpha_clamp = 0.005` snaps α to exactly 0 between logged steps 71 000 and 72 000. From that
run's own training log:

| step | α | `discrete_frac` | **test raw MSE (u)** |
|---:|---:|---:|---:|
| 69 000 | 0.008577 | 0.406 | 2.779 |
| 70 000 | 0.006693 | 0.547 | **2.657**  ← af's best |
| 71 000 | 0.005220 | 0.484 | 2.911 |
| **72 000** | **0.000000** | **0.000** | **8.504** |
| 73 000 – 99 000 | 0 | 0 | 6.3 – 8.6, never recovers |

**2.911 → 8.504 in one logging interval — a 2.9× step change**, against a curve that had descended
monotonically for 70 k steps (12.58 → 2.66).

**And the band it lands in is MeanFlow's band.** `mf`, which runs the α = 0 JVP branch from step 0,
plateaus at test MSE 7–10 from ~10 k on and finishes at 7.29 (best 6.65 @ 91 k). Two independently
initialised, independently trained runs land on the same test-error floor as soon as — and only as
soon as — they use the same estimator.

> **α-Flow at α ≈ 0.005–0.02 reaches 2.657: about 2.5× better than either arm ever achieves under
> the JVP target.** The U5 DA's conclusion, verbatim: *"`af_alpha_end: 0.0` is throwing away the best
> model this generation has produced."*

The per-interval buckets say the same thing where it matters most for few-NFE sampling: af's
long-interval `h_mse_b1`/`h_mse_b3` drop to ~1e-3 over steps 55 k–71 k and jump back to 10⁰–10¹ at
the cliff.

## 5.2 Two readings, still unresolved — and separable in one run

1. **The JVP target is genuinely worse on this problem.** `d/dr[u]` is a single-sample derivative
   taken through a vision-conditioned network; the small-α bootstrap is a two-point finite
   difference of the *same* quantity, and finite differences are better conditioned than exact
   derivatives when the function is noisy.
2. **The clamp is the bug, not the branch.** α does not *decay* to 0 — it is **snapped** there from
   0.0052, and `discrete_frac` goes 0.48 → 0.00 in one step. That is a distribution shift in the
   **target**, after 70 k steps of adapting to a target that always had a discrete component.

Reading 2 predicts the jump shrinks or vanishes at `af_alpha_clamp = 1e-4` or `af_alpha_end = 0.01`.
Reading 1 predicts it persists — **which would itself be a result**, and a more interesting one than
anything in the eval tables.

## 5.3 ⚠️ A second defect that may be hiding the good checkpoint

Eval loads `state_best.pt` (`diffusion_epoch: 'best'`, `config/aligning-d3il-visual.py:775`,
inherited by every mix plan block through `_mix_plan_common`). That file is written here:

```python
# mix_visual_aligning/utils/training_twotime.py:337-339
if test_loss < self.best_test_loss:
    self.best_test_loss = test_loss
    self.save_best()
```

`test_loss` is the **adaptive** loss — which the U5 DA measured as pinned near 1.0 (mf 0.926,
af 0.989 at the end) and describes as carrying **no signal**, exactly as `af_diffusion.py:521-524`
predicts it must under SUM reduction. `test_raw_mse` — the quantity that actually moves, and the one
the α cliff shows up in — is appended two lines above (`:330`) and **is never used for selection**.

**Consequence:** the 2.657 checkpoint at step 70 k would be selected only by accident. Which step
`cand6`'s `state_best.pt` actually holds is a cluster-side question — it cannot be read from the
batch CSVs — but the selection rule is documented as signal-free, so it should be checked before any
`af` number is taken as that arm's ceiling.

## 5.4 Why this has not been fixed

It was written up as follow-up **#3** in the U5 DA — *"Highest-value **training** job"* — on
2026-08-04, and never run. There is **no environment override** for the α knobs (unlike
`MIX_BONE_*`, `MIX_FILM_MODE_*`, `MIX_TRAIN_STEPS`); changing them is a config edit at
`config/aligning-d3il-visual.py:1438-1444`.

## 5.5 What this does to the rest of this document

Q1–Q4 stand as written: the target *is* the only difference, α is training-only, and at α = 0 the
two arms share an objective. What changes is the **reading** of that convergence.

| | before | after §5 |
|---|---|---|
| the α → 0 anneal | makes the mf-vs-af comparison thin | makes it thin **and costs the af arm ~2.5× on `raw_mse_u`** |
| "af ≈ mf in the results" | expected — they converge on one objective | expected **and partly self-inflicted**: we are comparing MeanFlow against a model that was α-Flow until step 71 k and MeanFlow after |
| the constant-α run | a nice-to-have control | **the experiment that should have been run first** |

**The V_A numbers do not measure α-Flow at its own operating point.** They measure the post-cliff
model. Every `af` row in [`Report_20260829_VA_funnel`](Report_20260829_VA_funnel/README.md) should
be read that way.

---

## 6. How to report this

0. 🔴 **Label every `af` row as post-cliff.** The evaluated `af` checkpoint is the α = 0 model, not
   α-Flow at its best setting (§5). "α-Flow underperforms / matches MeanFlow on V_A" is not
   supported by this batch; "α-Flow **as annealed to 0** matches MeanFlow" is.
1. **Not two independent confirmations.** Same bone, same sampler, same trainer, same budget, and —
   for the final 28.8 % of training — the same objective. Report as **one result with two training
   curricula**, and say so.
2. **Do not call `af` "a different method" without the α curve.** The honest description on this
   config is *"MeanFlow trained with a flow-matching warm-up (α: 1 → 0, sigmoid, γ = 25)"* — which
   is also what its authors call it.
3. **The K-response inversion (§4.3) is the one place `af` earns a separate row.** Real, significant,
   mechanistically interesting — and currently filed as a red flag rather than a result.
4. Anywhere a claim rests on `mf` and `af` agreeing, that agreement is **near-tautological** and
   carries far less independent evidence than two unrelated engines would.

## 7. Open items (all need the cluster — no torch in this container)

- **🔑🔑 Disambiguate the α cliff — one training run.** `af_alpha_clamp = 1e-4`, or
  `af_alpha_end = 0.01`, or `af_alpha_scheduler='constant'` with `af_alpha_init = af_alpha_end ≈ 0.05`
  (`config/aligning-d3il-visual.py:1438-1444`; no env override exists). If the 2.9× jump vanishes the
  clamp is an artefact; if it persists, **the JVP target is genuinely worse on vision-conditioned
  trajectories** — a result worth more than any eval table in this generation. Logged as U5 follow-up
  #3 on 2026-08-04 and still open.
- **🔑 Free check first: re-evaluate `af` at step 70 000** (pre-cliff) at K = 2, if that checkpoint
  survived. Zero training cost, and it tests whether §5's cliff is what the eval is measuring.
- **Check which step `cand6`'s `state_best.pt` holds**, and whether `save_best` should track
  `test_raw_mse` rather than the signal-free adaptive `test_loss` (§5.3,
  `training_twotime.py:337-339`). Affects **both** arms, not just `af`.
- **🔑 Run G1/G2 against the Gen14 visual arms.** `gates_alphaflow.py` lives only in
  `FM_v3_alphaflow_test/` (state-based). `af_diffusion.py` is byte-identical across the two trees so
  it should pass unchanged, but it has never run with `if_vision=True` on the `VisualUNetTwoTime`
  bone — where the JVP-tangent story (`visual_*_diffusion.py:11-15`) is the new risk, not the α branch.
- **The decisive architecture test:** on the `unet` bone the two trajectory models build identical
  module trees, so `cand6`'s `state_dict` should load into a `MeanFlowEngine` **without renaming**.
  If it does, "same architecture" stops being an inference from source and becomes a measurement.
  (`assert_engine_matches`, `engine_registry.py:169`, refuses it on the recorded `engine` key —
  bypass for the test only.)
- **Run each arm on its OWN native bone** (`MIX_BONE_AF=sit`, `MIX_BONE_MF=mf_dit`). Both are
  implemented; **neither has ever been run on visual-aligning**. Removing that axis was right for an
  architecture-controlled claim, but it means the repo has never measured the two methods as their
  authors built them.
- **⚠️ Config asymmetry found while checking this.** The `af` block binds its budget to the
  overridable `_MIX_N_TRAIN_STEPS` (line 1445); the `mf` block sets **no** `n_train_steps` and
  inherits the literal `1e5` from `fm_visual_aligning` (line 531). `MIX_TRAIN_STEPS=50000` therefore
  shortens **`af` only** — while `_mix_train_block` stamps `TB50pct` on *both* folders. A
  reduced-budget sweep would silently compare a 50 k `af` against a 100 k `mf`, both labelled 50 %.
  Not triggered by any run in the 0823 batch (no `TB` tag on any folder), but live.

---

*Repo quotes verbatim from `mix_visual_aligning/models/{visual_af_diffusion,visual_mf_diffusion,af_diffusion,mf_diffusion,af_engine,mf_engine,af_trajectory_model,mf_trajectory_model,af_sit_trajectory,engine_registry}.py`,
`mix_visual_aligning/utils/training_twotime.py`, `config/aligning-d3il-visual.py`,
`FM_v3_alphaflow_test/gates_alphaflow.py`. Upstream quotes from
`/workspaces/aux_repo/alphaflow/{infra/experiments/experiments-alphaflow.yaml,configs/loss/alphaflow.yaml,README.md}`,
`/workspaces/aux_repo/MeanFlow/meanflow.py`, `/workspaces/aux_repo/imeanflow/imf.py`. The α table in
§2.2 is `_get_ratio` evaluated over `0 … 1e5` with this task's constants. The training-curve table in
§5.1 is quoted from `logs_in_develop/Gen14/U5/DA_20260804_mf_af_visual_aligning_first_run.md` §3 —
the only place in the repo where these two arms' training logs have been read side by side. Diffs are `difflib`
unified diffs with comment-only lines filtered. Empirical rows from
`temp/2508/batch_va2_20260823_135156/per_rollout_detail.csv`.*
