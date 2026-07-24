# Audit: `thought&theory.md` vs. the actual iMF code

**Scope:** `flow_matcher_v3_imeanflow/` (model package) + `FM_v3_imeanflow_test/` (train/eval
scripts), cross-checked against the reference `/workspaces/imeanflow` repo the port is
adapted from. Every claim below is backed by a file:line citation — nothing is inferred from
memory or from the theory doc's own framing.

**Verdict up front:** the theory doc's *high-level* claim — "location-dependence lives in the
conditioning variable `c`, not in the noisy tensor `z_t`" — is directionally correct and the
codebase does achieve it. But three of the theory doc's *specific, literal* mathematical/
architectural claims do **not** match what's actually implemented in this repo. This isn't
"the code is broken" — the code works by a different, still-defensible mechanism in each case
— but an engineer reading `thought&theory.md` and expecting to find Fig. 5 token-conditioning,
Eq. 17 CFG-target-mixing, or a default `meanflow_jvp` objective in this codebase will not find
them where the doc implies they are.

**Separately — and this one is not about the theory doc at all —** Finding 6 digs into whether
this repo's DATA-AT-1 time convention (flipped from the reference repo's DATA-AT-0) actually
breaks anything at runtime, beyond just confusing a reader. Answer: the core flow-matching
mechanics (interpolant, velocity target, JVP tangents) were flipped correctly and consistently
(6a) — but the `t`-sampling schedule's `p_mean=-0.4` hyperparameter was copied from the
reference repo **without** flipping its sign, and this audit confirmed numerically (6b) that
this makes FM-PCC train with the sample-density bell centered on the *opposite* side of the
noise/data midpoint from what that hyperparameter was originally tuned to achieve. This is the
one finding in this document that looks like a real, fixable bug rather than a documentation
mismatch.

---

## Finding 1 — Conditioning is inpainting, not token-concatenation (theory doc §0, §1 wrong for this repo)

**Theory doc claims** (`thought&theory.md` lines 5, 13, 23):
> "`c` is fed to the network as an extra input alongside `z_t, t` — architecturally, exactly
> like Fig. 5's token concatenation, just swap 'class tokens' for 'state/goal tokens.'"
> "**c must be clean, unnoised information that travels alongside `z_t`, not inside it.**"

**What the code actually does:**
- `c = (s_0, s_goal)` is correctly identified — `SequenceDataset.get_conditions()`
  (`flow_matcher_v3_imeanflow/datasets/sequence.py:87-92`) returns `{0: observations[0]}`
  (the trajectory's own first row, i.e. `s_0`), and `goal_dim` (constant-across-horizon
  observation dims, `sequence.py:95-102`) supplies `s_goal`.
- But injection is via `apply_conditioning()` (`flow_matcher_v3_imeanflow/models/helpers.py:143-166`),
  which **overwrites slots of the trajectory tensor `x` itself**:
  ```python
  x[:, t, action_dim:] = val.clone() if not noise else 0        # helpers.py:161
  x[:, :, -goal_dim:] = conditions[0][:, -goal_dim:]...          # helpers.py:164
  ```
  This is Diffuser-style (Janner et al.) inpainting/clamping — `c` is written **inside** `z_t`
  at specific (timestep, dim) coordinates, the opposite of "travels alongside, not inside."
- The network backbones DO accept a `cond` parameter for API shape — but never use it:
  - UNet: `Flow_matcher_U_Net_v2.forward(self, x, cond, time, ...)`
    (`unet1d_temporal_cond.py:214`) — `cond` appears in the signature and is never referenced
    again in the ~90-line body (verified by reading the full function).
  - DiT: `IMFDiTTrajectory.forward(self, x, cond, time, ...)`
    (`imf_dit_trajectory.py:356`), whose own docstring at line 358 admits it:
    *"`cond`/`returns`/`use_dropout` accepted for parity."* The DiT's `class_tok` mechanism
    (which genuinely IS Fig. 5-style token conditioning, and does exist in this file) is wired
    to a **hardcoded constant class index**, not to `cond`:
    ```python
    y_idx = torch.full((b,), self.num_classes if force_dropout else 0, ...)  # imf_dit_trajectory.py:349
    ```
    `y_idx` is always `0` ("sole conditioning class") except on CFG-dropout, where it's the
    null index. It never carries `s_0`/`s_goal`.
- **Cross-check against the reference repo confirms the theory doc correctly describes
  `/workspaces/imeanflow`, just not this port.** `imeanflow/models/imfDiT.py:192,303,309`
  really does route a per-sample class label through `y_embedder` into `class_tokens` — genuine
  token-conditioning, exactly Fig. 5. That mechanism exists in `imf_dit_trajectory.py` too (it's
  a structural copy) but is never fed real per-sample state — only the constant `y_idx=0`.

**Assessment:** Not a bug — inpainting conditioning is a legitimate, independently-established
technique (this is literally how the original Diffuser paper conditions on start/goal state).
The JVP z-tangent is even zeroed at conditioned dims (`imf_diffusion.py:546`,
`v_inst = apply_conditioning(..., noise=True)`), which is a self-consistent way to encode
"`c` doesn't move" *within* the inpainting paradigm. But it is a **different mechanism** than
what §0/§1 of the theory doc describes, and the theory doc's own stated invariant ("not inside
it") is the one thing this implementation violates by construction.

---

## Finding 2 — Default training objective is `fm_equivalent`, not the JVP math in theory doc §2

**Theory doc claims** (`thought&theory.md` lines 37-51): iMF training regresses to
`u_tgt = (e−x) − (t−r)·JVP(...)`, i.e. the real MeanFlow-Identity via a JVP.

**What the code actually does:** this exists, correctly gated behind a flag that is **not the
default**:
```python
imf_objective: str = 'fm_equivalent',     # imf_diffusion.py:47 — the DEFAULT
...
if self.imf_objective == 'meanflow_jvp':
    return self._p_losses_meanflow_jvp(x_start, cond, t, returns=returns)   # imf_diffusion.py:417-418
# else falls through to the finite-difference u_target = (x_t - x_r) / h   (imf_diffusion.py:444)
```
The `'fm_equivalent'` path (the class default) is described in-code as the thing that
"collapses to FM, the A/B baseline" (`imf_diffusion.py:45`) — i.e. plain conditional Flow
Matching with a finite-difference average-velocity target, **not** the JVP-based
MeanFlow-Identity the theory doc's §2 derives.

**Whether this matters in practice depends on which config is actually run** — checked both
configs that set this flag:
- `config/avoiding-d3il.py:480`: `'imf_objective': 'meanflow_jvp'` — the real JVP path IS the
  configured default for this task, plus `dual_head=True` (`:488`), `interval_cfg=True`
  (`:489`), `meanflow_cfg_omega=4.0` (`:495`).
- `config/aligining-d3il-visual.py:771,791`: `'imf_objective': 'fm_equivalent'` — but this
  config belongs to `imf_visual_aligining_test` (a separate sibling package that imports its
  own `imf_visual_aligining.*` modules, confirmed via
  `imf_visual_aligining_test/eval_imf_visual_aligining.py:50-51`), **not**
  `flow_matcher_v3_imeanflow`/`FM_v3_imeanflow_test` — out of this audit's named scope, noted
  for awareness only.
- `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py:28` defaults `dataset =
  'avoiding-d3il'` — so the package actually being audited here does default to the real JVP
  objective in practice, even though the class-level default inside `imf_diffusion.py` itself
  is the FM-equivalent fallback.

**Assessment:** Correct in the configuration that matters for this package, but the theory doc
presents the JVP objective as *the* iMF training procedure without noting it is one of two
selectable objectives, opt-in via `imf_objective`, and silently falls back to plain FM if a
future config forgets to set it.

---

## Finding 3 — The CFG target-mixing formula (theory doc's Eq. 17/62) is not implemented; ω is an input embedding instead

**Theory doc claims** (`thought&theory.md` lines 59-63), calling it "straight from the appendix
pseudocode, Alg. 2":
```
v_cfg = (e−x) + (1 − 1/ω)·[ u_θ(z_t | t,t,c) − u_θ(z_t | t,t,∅) ]
```

**Reference repo does implement exactly this**, at training time, as the regression target:
```python
v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)     # imeanflow/imf.py:315 (guidance_fn)
v_g = v_t + (1 - 1 / w) * (v_c - v_u)        # imeanflow/imf.py:320
...
u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)   # imf.py:373
```
`v_c` and `v_u` are two separate forward passes (conditioned / null-class), explicitly mixed
via `(1-1/w)` **before** being handed to the JVP as the target-construction ingredient.

**FM-PCC's `_p_losses_meanflow_jvp`** (`imf_diffusion.py:497-609`) does not do this anywhere.
It calls the u-network **once** per JVP (`imf_diffusion.py:571`,
`u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))`) and regresses directly to
`u_target = v_inst + h_expand * du_dr` (`:578`) — no `v_c`/`v_u` pair, no `(1-1/ω)` mixing into
the target anywhere in this function. The one place `(1 - 1.0/w)` *does* appear in this
codebase is `imf_dit_trajectory.py:343`:
```python
w_arg = torch.where(w > 0, 1.0 - 1.0 / w.clamp(min=1e-6), torch.zeros_like(w))   # DiT only
omega_tok = self.omega_tokens[None] + self.omega_embedder(w_arg)[:, None]
```
— but this feeds `ω` as an **additive input embedding** to the network (asking it to learn the
CFG-corrected output directly from seeing `ω` as a condition), not as a target-mixing formula.
The UNet backbone's `interval_cfg` path (`unet1d_temporal_cond.py:248-254`) does the analogous
thing without even the `(1-1/w)` transform — `omega` is embedded raw via `self.omega_mlp`.

Separately, **inference-time CFG does match the theory doc's simpler Eq. 13** (plain-FM
formula, §1): `_predict_velocity`'s interval-CFG block
(`imf_diffusion.py:187-190`) computes `velocity = uncond + cfg_scale * (velocity - uncond)`,
which is algebraically `ω·v_cond + (1-ω)·v_uncond` — correct, and this part of the theory doc
holds. It's specifically the *training-time* Eq. 17/62 mixing that's absent.

**Assessment:** A real design divergence, not a bug per se — "let the network learn the
guidance manifold from an ω-input" is a legitimate alternative to "bake the CFG-mixed target
in via two forward passes," and the code's own comments frame it as a deliberate choice
(`imf_diffusion.py:59` block comment references the "official recipe" for *sampling* `ω`, not
for target construction). But it means the theory doc's Eq. 17/62, presented as directly
implemented, is not — what's implemented is a different (also-plausible) CFG mechanism.

---

## Finding 4 — JVP differentiation direction differs from the reference repo — RE-DERIVED here: appears mathematically sound, not the likely defect

Reference repo differentiates the **end-anchored** identity w.r.t. `t` (`z`-tangent `v_c`,
`t`-tangent `+1`, `r`-tangent `0`) — `imeanflow/imf.py:373`. FM-PCC's
`_p_losses_meanflow_jvp` differentiates the **start-anchored** identity w.r.t. `r`
(`z`-tangent `v_inst`, `r`-tangent `+1`, `h`-tangent `-1`) — `imf_diffusion.py:571`. The code's
own docstring (`imf_diffusion.py:504-522`) flags this as unverified — *"the prime suspect is the
JVP sign / the h-tangent"* — so this audit re-derived it symbolically from the two conventions'
own defining formulas, rather than taking either the doc's caution or the theory doc's
confidence at face value.

**Why the two repos differentiate w.r.t. different variables in the first place — this is a
consequence of which endpoint is "known" during sampling, not an arbitrary choice:**
- Reference repo's sampler (`imeanflow/imf.py:33,42,90-114`) starts `z_t` at `t=1` (pure
  noise) and steps with **decreasing** `t` toward `r<t`: `z_new = z_t − (t−r)·u`. The *known*
  point is `z_t`, so the identity is naturally parametrized as `u(z_t, r, t)` and differentiated
  w.r.t. `t` (the known point's own time coordinate).
- FM-PCC's sampler (`imf_diffusion.py:217,266-327`) starts `x` at `t=0` (pure noise, matching
  its own `q_sample` convention) and steps with **increasing** `t`: `x = x + velocity·dt`. The
  *known* point is `x` at the CURRENT loop time, which plays the role of `r` (anchor), stepping
  toward the FUTURE `t=r+h`. So FM-PCC's identity is naturally `u(z_r, r, t)`, differentiated
  w.r.t. `r` (the known point's own time coordinate) — the mirror-image setup, correctly
  matched to a start-anchored (forward) sampler instead of an end-anchored (backward) one.

**Re-deriving FM-PCC's own identity from scratch, to check the sign:** start from
`(t−r)·u(z_r,r,t) = z_t − z_r`. Differentiate both sides w.r.t. `r`, holding `t` FIXED, using
`dz_r/dr = v(z_r,r)` (the point moves along the ODE as its own anchor time varies):
```
LHS: d/dr[(t−r)u] = −u + (t−r)·du/dr
RHS: d/dr[z_t − z_r] = −dz_r/dr = −v(z_r,r)
⟹ u = v(z_r,r) + (t−r)·du/dr        — matches imf_diffusion.py:513-517 exactly
```
`u_θ` is parametrized as `u_θ(z, r, h)` with `h=t−r` (not `u_θ(z,r,t)`), so with `t` fixed,
`dh/dr = −1`. The chain rule for `du/dr` through this parametrization is
`∂_z u·(dz_r/dr) + ∂_r u·1 + ∂_h u·(−1)` — **exactly** the tangent tuple the code passes:
```python
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))   # imf_diffusion.py:571
```
And the z-tangent itself checks out against FM-PCC's own interpolant: `x_r = (1−r)x_base +
r·x_start` (`q_sample`, `imf_diffusion.py:199`) ⟹ `dx_r/dr = x_start − x_base = v_inst` exactly
as used (`imf_diffusion.py:545`).

**Conclusion of the re-derivation: the sign/tangent choice is internally consistent and
correctly derived for FM-PCC's own (DATA-AT-1, start-anchored) convention** — it is a genuinely
different but equally valid MeanFlow-Identity variant, not a transcription error inherited from
blindly copying the reference repo's t-anchored formula. This *raises* confidence relative to
the code's own self-flagged caution, but a written derivation is still not the same as a
verified forward/backward pass — the recommended 1-NFE reconstruction check on the cluster
remains the authoritative confirmation and should still be run before fully retiring this flag.

---

## Finding 5 — Notation mismatch: theory doc's sampling formulas assume the *original* paper's time convention, not this repo's

Theory doc (`thought&theory.md:68`, `:84`, `:90`): `x̂ = z_1 − u_θ(z_1 | c)`, "`z_1` is still
pure noise." This is the **reference repo's** convention: `imeanflow/imf.py` implicitly has
noise at `t=1` (data-at-0). FM-PCC explicitly flips this — **DATA-AT-1**:
```python
return (1.0 - t_cont) * noise + t_cont * x_start    # imf_diffusion.py:199 (q_sample)
```
At `t=0`: pure noise. At `t=1`: data. `p_sample_loop` starts from `x = torch.randn(shape)`
(`imf_diffusion.py:217`, comment: *"sigma=1.0 to match q_sample training noise"*) and
integrates **forward**, `t: 0→1` (`imf_diffusion.py:266-327`), the opposite direction of the
theory doc's `z_1 → x̂` formula. The underlying *point* (location-dependence lives in `c`, not
in the noise tensor, at either end) is still true regardless of which end is labeled 0 vs. 1 —
but the literal formula `x̂ = z_1 − u_θ(z_1|c)` would be wrong if applied verbatim against this
code's actual convention (in FM-PCC's convention the 1-NFE analog is closer to
`x̂ = z_0 + u_θ(z_0|c)` with `z_0` the noise). Worth fixing in the doc so a future reader
doesn't transcribe the formula literally.

---

## Finding 6 — Deep dive: does the DATA-AT-1/DATA-AT-0 flip cause a REAL problem, not just a notation mismatch?

User asked to dig past "the doc uses the wrong notation" and check whether the convention flip
actually breaks anything at runtime. Two separate places depend on the flip; they resolve
oppositely.

### 6a. The flow-matching mechanics themselves (interpolant + velocity target + JVP): flipped CORRECTLY, both endpoints checked

- Interpolant: reference `z_t=(1−t)x+te` (`imf.py:350`) vs. FM-PCC `x_r=(1−r)noise+r·x_start`
  (`imf_diffusion.py:199`) — a consistent full flip (not a partial one): every `t` in the
  reference maps to `(1 − t)` in FM-PCC, and the *labels* "noise"/"data" swap endpoints
  accordingly.
- Velocity target: reference `v_t = e − x` (noise minus data, `imf.py:351`, points toward
  *increasing* t = more noise, matching their t-increasing-is-noisier convention) vs. FM-PCC
  `v_inst = x_start − x_base` (data minus noise, `imf_diffusion.py:545`, points toward
  *increasing* t = more data, matching FM-PCC's t-increasing-is-more-data convention). These
  are sign-mirrors of each other, and **each is correct within its own repo's convention** —
  confirmed by direct differentiation of each repo's own interpolant formula above.
- JVP tangents (Finding 4): re-derived from scratch above and found internally consistent with
  FM-PCC's own interpolant and its own start-anchored sampler design.

**Verdict for 6a: no real bug found.** The core flow-matching math was adapted as a coherent,
fully-flipped system, not a half-translated one — every place checked (interpolant, velocity
target, JVP tangents, sampler step direction) uses the SAME (DATA-AT-1) convention consistently.

### 6b. The time-SAMPLING SCHEDULE hyperparameter: NOT flipped — this appears to be a real, quantifiable inversion

This is a different kind of dependency: `P_mean`/`p_mean` is not a formula that mechanically
transforms under `t → 1−t` — it's an **empirically-tuned hyperparameter** whose correct value
under a flipped convention is `−p_mean`, not `p_mean`, and nothing in the code enforces that
relationship automatically.

- Reference repo's `t` = **fraction of noise** (`z_t=(1−t)x+te`, t=1 is pure noise). Its
  schedule (`imf.py:120-124`) is `t = sigmoid(randn·P_std + P_mean)`, `P_mean=−0.4` ⟹
  `median(t) = sigmoid(−0.4) ≈ 0.401`. In their convention this means: **median training
  sample has ≈40% noise-fraction** — i.e. the schedule is tuned to spend more density on the
  *lower-noise* (more-data-like, "harder to get exactly right") regime, per the standard
  diffusion-training intuition that near-pure-noise steps are uninformative and near-data steps
  are comparatively easy, so weight should concentrate somewhat below the midpoint of the
  *noise* axis.
- FM-PCC's `t` = **fraction of data** (`x_r=(1−r)noise+r·x_start`, t=1 is pure data) — the
  literal opposite semantic. FM-PCC copied the schedule verbatim
  (`imf_diffusion.py:397-400`, `config/avoiding-d3il.py:540`, both **`p_mean=-0.4`**,
  unflipped) ⟹ `median(t) = sigmoid(−0.4) ≈ 0.401` again — but this time **0.401 is the
  DATA-fraction**, so the median training sample now has ≈**60% noise-fraction**, the opposite
  side of 50% from what the reference schedule was tuned to hit (≈40% noise-fraction).
- **This isn't speculative reasoning about what "should" happen — it's directly corroborated by
  this repo's own prior investigation.** `logs_in_develop/Gen3v4_imf/U7/PLAN_Logit_Normal_Schedule.md`
  (the doc that introduced this exact schedule) justifies the bell-shaped distribution with:
  *"Near t=0 (noise), the model output is almost pure noise anyway ... Near t=1 (data), the
  problem is near-trivial"* — this sentence is written entirely in FM-PCC's OWN convention
  (t=0 is noise, t=1 is data — U7's author correctly identified FM-PCC's semantics) — but the
  `P_mean=−0.4` value adopted right below that reasoning was copied from the reference schedule,
  which was tuned for the OPPOSITE t-semantics. The U7 doc's own conceptual framing and its own
  chosen number are talking about two different axes without reconciling them.
- **Net effect (if this reasoning holds): FM-PCC trains with roughly 60/40 noise-heavy sampling
  bias instead of the reference's intended 40/60 (i.e. data-heavy) bias** — the sample-density
  peak sits on the wrong side of the midpoint relative to what the schedule's own tuned
  hyperparameter was meant to achieve. This is configured for the actual audited task —
  `config/avoiding-d3il.py:539-541` sets `t_schedule='logit_normal'`, `p_mean=-0.4` as the
  active training config, not a dormant/unused branch.

**Verdict for 6b: this looks like a real, quantifiable bug** — not a notation issue. The fix
(if the reasoning above is confirmed) would be `p_mean = +0.4` (⟹ `median(t) ≈ 0.599`, i.e.
≈40% noise-fraction in FM-PCC's own convention, reproducing the reference schedule's actual
intent). This audit stops short of asserting it's *confirmed* — the argument rests on reading
both repos' formulas and one planning doc's prose, not on empirical loss curves — but the
direction of the argument is concrete and falsifiable: plotting a histogram of `t` samples next
to a histogram of `1 - t_reference` samples would immediately confirm or refute the claimed
mismatch, with no cluster/training run required (pure `torch.sigmoid`/`numpy` arithmetic).

### 6c. Interval-CFG's default eval operating point: checked, inconclusive, minor

`meanflow_cfg_t_min=0.4, meanflow_cfg_t_max=0.6` (`config/avoiding-d3il.py`, `imf_diffusion.py`
defaults) is roughly centered on the midpoint rather than skewed toward either "noise" or
"data" under either convention — so even if guidance is conventionally more valuable
near-data in standard practice, this default's near-symmetric placement makes it much less
sensitive to the DATA-AT-1 flip than 6b's schedule bias. Noted for completeness, not raised as
a separate confirmed finding — the asymmetry here is far too mild to draw a conclusion from a
code read alone.

---

## Summary table

| Theory doc claim | Location | Matches code? | Where |
|---|---|---|---|
| `c = (s_0, s_goal)`, extracted from `x` itself | §0 | ✅ Yes | `sequence.py:87-102` |
| `c` fed as extra network input / token concat (Fig. 5) | §0, §1 | ❌ No — inpainting instead | `helpers.py:143-166`; `unet1d_temporal_cond.py:214` (`cond` unused); `imf_dit_trajectory.py:349,358` (`y_idx` constant) |
| Plain-FM training objective `v_θ(z_t,t\|c)` | §1 | ✅ Structurally yes (via `fm_equivalent` path) | `imf_diffusion.py:420-468` |
| Plain-FM inference CFG (Eq. 13) | §1 | ✅ Yes | `imf_diffusion.py:187-190` |
| iMF JVP training objective (Eq. 9/12 conditional) | §2 | ⚠️ Exists, opt-in, not the class default | `imf_diffusion.py:47,417-418,497-609` |
| `v_θ ≡ u_θ(·,t,t\|c)` boundary trick | §2 | ⚠️ Not literally found as an explicit call site — see note below | — |
| Flexible-guidance training target (Eq. 15/17/62) | §2 | ❌ No — ω is an input embedding, not a target-mixing formula | `imf_diffusion.py:497-609` (absent); `imf_dit_trajectory.py:343-344` (embedding only) |
| 1-NFE / K-NFE sampling formulas | §2 | ⚠️ Right idea, wrong time-convention (z_1 vs z_0) for this repo | `imf_diffusion.py:199,217,266-327` |
| "location lives in `c`, not in `z_t`" (core claim) | §3 | ✅ Yes, by a different mechanism than described | overall |
| *(not a theory-doc claim)* Does the DATA-AT-1 flip break the core FM/JVP math? | Finding 6a | ✅ No — re-derived, fully and consistently flipped | interpolant/velocity/JVP, all cross-checked |
| *(not a theory-doc claim)* Does the DATA-AT-1 flip break the `t`-sampling schedule? | Finding 6b | ❌ Likely yes — `p_mean=-0.4` ported unflipped, inverts the intended noise/data density bias | `imf_diffusion.py:397-400`; `config/avoiding-d3il.py:539-541`; corroborated by `U7/PLAN_Logit_Normal_Schedule.md`'s own reasoning |

**Note on the boundary-condition trick** (`v_θ(z_t,t|c) ≡ u_θ(z_t,t,t|c)`): not found as an
explicit `r=t` call anywhere outside training — `_p_losses_meanflow_jvp` samples `r=t` for a
*fraction* of the batch (`meanflow_r_equals_t_frac`, `imf_diffusion.py:535-536`) as an anchor
regularizer, which is functionally related but not the same statement as "define `v` to literally
be `u` evaluated at `r=t`." Flagged as unverified rather than confirmed either way — would need
to check whether any code path explicitly calls `u_fn(..., r=t)` at inference to stand in for
`v`, which was not found in the files read for this audit.

## Files read for this audit
- `logs_in_develop/Gen3v4_imf/Condition_Analysis/thought&theory.md` (the audited doc)
- `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (full file, 659 lines)
- `flow_matcher_v3_imeanflow/models/imf_engine.py` (full file, 193 lines)
- `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` (full file, 204 lines)
- `flow_matcher_v3_imeanflow/models/imf_losses.py` (full file — confirmed legacy/unused compat
  surface, not the active loss path)
- `flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py` (forward() body, lines 214-303)
- `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py` (conditioning/forward sections)
- `flow_matcher_v3_imeanflow/models/helpers.py` (`apply_conditioning`, lines 143-166)
- `flow_matcher_v3_imeanflow/datasets/sequence.py` (`get_conditions`/`get_goal_dim`, lines 87-109)
- `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` (Parser defaults)
- `config/avoiding-d3il.py`, `config/aligining-d3il-visual.py` (grepped for `imf_objective`
  and related flags)
- `/workspaces/imeanflow/imf.py` (full file, 404 lines — reference training loop, CFG,
  guidance_fn, JVP call site)
- `/workspaces/imeanflow/models/imfDiT.py` (conditioning sections — `y_embedder`, `class_tokens`)

## What was NOT verified (out of scope / needs cluster runtime)
- Numeric correctness of the JVP sign convention (Finding 4) — re-derived symbolically here and
  found self-consistent, but a written derivation is not a substitute for the 1-NFE
  reconstruction check the code's own comments already call for.
- Whether the `fm_equivalent` vs `meanflow_jvp` objectives produce measurably different
  eval behavior — needs actual training runs, not a code read.
- Runtime shape/dtype correctness of `torch.func.jvp` usage — not exercised (no torch in this
  environment).
- **Finding 6b's schedule-inversion claim — the cheap numeric check does NOT need a cluster,
  so it was run here** (pure NumPy, `sigmoid(randn(200000)·1.0 − 0.4)`, seed 0):
  ```
  FM-PCC t (data-fraction) samples:  mean=0.4186  median=0.4023
    => implied noise-fraction (1−t): mean=0.5814  median=0.5977
  sigmoid(-0.4) exact = 0.4013  |  sigmoid(+0.4) exact = 0.5987
  ```
  This confirms the arithmetic in Finding 6b exactly: FM-PCC's `p_mean=-0.4` produces a `t`
  distribution with **median data-fraction ≈40%**, i.e. **median noise-fraction ≈60%** — the
  mirror image, across 50%, of the reference schedule's own stated target (~40% noise-fraction,
  same numeric formula, opposite semantic label). The distribution shape (bell, `P_std=1.0`)
  transfers correctly across the flip — only the sign of `p_mean` does not. **What is still
  NOT verified** (genuinely needs cluster/training): whether this ~20-percentage-point shift in
  where training density concentrates actually produces a measurable difference in trained
  policy quality, or whether the loss landscape is flat enough there that it doesn't matter in
  practice. The arithmetic mismatch is now confirmed; its downstream *impact* is not.
