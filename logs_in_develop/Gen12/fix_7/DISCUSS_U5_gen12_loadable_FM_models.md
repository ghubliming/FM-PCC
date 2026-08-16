# Gen12 (HardFlow / FMv3ODE): which FM models can it actually load?

**Question (user):** what kind of FM model can Gen12 currently load? Can we load the **iMF (Gen3v4)**,
**MeanFlow (Gen3v6)**, **α-Flow (Gen3v7)** models? Why / why not? And — known to be impossible —
what about **Gen13** models?

**Short answer:**

| model | Gen12 **diffuser + dpcc** arms | Gen12 **HardFlow** arm (the whole point of Gen12) |
|---|---|---|
| **plain FMv3 / FlowMatchingODE** (the baseline living in every sibling folder) | ✅ works | ✅ works — this is what Gen12 is built for |
| **iMF** (Gen3v4 `imf_diffusion`) | ⚠️ runs natively, *semantically OK* | ❌ **no** — wrong velocity field |
| **MeanFlow** (Gen3v6 `mf_diffusion`) | ⚠️ runs natively, *semantically OK* | ❌ **no** — wrong velocity field |
| **α-Flow** (Gen3v7 `af_diffusion`) | ⚠️ runs natively, *semantically OK* | ❌ **no** — wrong velocity field (α=1 endpoint only) |
| **Gen13** (HardFlow-native + iMF) | ❌ not even loadable | ❌ not even loadable |

The nuance that makes this interesting: Gen12 has **two different sampling paths**, and they impose
**different** requirements on the checkpoint. The `diffuser`/`dpcc` arms are field-agnostic; the
**HardFlow** arm is not.

---

## 1. What Gen12 actually requires from a checkpoint

Gen12 loads the checkpoint's **own** diffusion class natively — the eval driver sets
`target_class = None` on purpose:

```python
# FM_v3_hardflow_test/eval_FM_v3_hardflow.py:148
target_class = None            # load the pickle's OWN class (FlowMatchingODE), no override
...
:171  fm_experiment = load_diffusion_with_override(*loadpath_parts, target_class=None, ...)
:172  fm_model = fm_experiment.diffusion
```

So "what can Gen12 load" is really "what interface does each of Gen12's three arms call on
`fm_model`."

### 1a. The `diffuser` and `dpcc` arms — go through the **native** sampler

Both use `Policy.__call__`, which just calls the diffusion object's own `forward`:

```python
# flow_matcher_v3_hardflow/sampling/policies.py:52
samples, infos = self.model(conditions, returns=..., projector=projector, ...)
#                └─ fm_model.forward → conditional_sample → p_sample_loop  (the CHECKPOINT's own)
```

- **`diffuser`** = the native sampler, unguided.
- **`dpcc`** = the native sampler + **post-hoc** SLSQP projection on the *finished* trajectory
  (`projection.py::Projector`). Projection operates on the output plan `(H, T)`; it does **not**
  care whether that plan came from an instantaneous-velocity ODE or a mean-velocity jump.

→ These two arms are **field-agnostic**. Whatever native sampler the checkpoint ships (FM Euler, or
iMF/MeanFlow interval sampler) runs as-designed, and DPCC filters the result. This is why the table
marks iMF/MF/αF as "runs natively, semantically OK" for these two arms.

### 1b. The `HardFlow` arm — bypasses the native sampler, needs **instantaneous velocity**

`HardFlowSampler.sample()` does **not** call `p_sample_loop`. It runs its **own** Euler ODE and, at
each active step, queries the model for the *instantaneous* velocity `v(x, τ)`:

```python
# flow_matcher_v3_hardflow/sampling/hardflow_projection.py:389-390
with torch.no_grad():
    v = self.model._predict_velocity(traj, cond, t, returns=returns)   # expects v(x,t), 3 time-args
```

and then builds the **endpoint prediction** the in-loop NLP projects:

```python
# hardflow_projection.py (active step)
V_next = self._velocity_batch(X_ref, tau_next, ...)      # instantaneous v at τ_next
X1_ref = X_ref + (1.0 - tau_next) * V_next               # x1 = x_τ + (1−τ)·v   (linear-FM endpoint)
```

This is the **linear flow-matching identity** `x₁ = x_τ + (1−τ)·v(x_τ,τ)` — it is only valid when the
network returns the **instantaneous** velocity of the probability-flow ODE. The whole HardFlow
mechanism (Euler-integrate `dx/dτ = v(x,τ)`, predict the endpoint, project it, snap back) is built on
`v(x,τ)`.

---

## 2. Why iMF / MeanFlow / α-Flow cannot drive the HardFlow arm

All three are **mean-flow** models. They do **not** parameterize `v(x,t)`; they parameterize the
**average velocity over an interval** `u(x, t, r)` — the network takes **two** time arguments:

```python
# flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py:361
def forward(self, x, cond, time, returns=None, ..., h=None, omega=None, t_min=None, t_max=None, return_v=False):
    ...
    return u                      # the MEAN-velocity head (v-head is discarded at inference)

# flow_matcher_v3_imeanflow/models/imf_diffusion.py:174
def _predict_velocity(self, x, cond, t, h=None, ..., omega=None, t_min=None, t_max=None, cfg_scale=0.0):
    velocity, _aux = self._predict_uv(x, cond, t, h=h, ...)   # u over [t−h, t], NOT instantaneous v
```

(MeanFlow `mf_diffusion` and α-Flow `af_diffusion` share this two-time `u(x,t,r)` shape — α-Flow is a
homotopy whose target is FM at α=1 and MeanFlow at α=0, but the *network* is still a mean-velocity net
with the interval inputs.)

Three concrete failure modes, in increasing order of how badly they bite:

1. **Interface / silent-wrong-field (the killer).** Gen12 calls `_predict_velocity(traj, cond, t,
   returns=…)` — three positional args, **no `h` / `t_min` / `t_max` / `omega`**. On the iMF class
   this doesn't *crash* (`h=None` is a default), so it would return `u(x, t, r=?)` with a degenerate
   interval. But `u` is the **average** velocity over `[r,t]`, not the instantaneous `v(x,t)`.
   Feeding `u` into HardFlow's Euler step (`X + dt·V`) and endpoint identity
   (`X1 = X + (1−τ)·V`) integrates and extrapolates the **wrong field**. The run would produce
   plausible-looking numbers and quietly-wrong trajectories — the worst kind of "loads fine."

2. **Wrong sampling regime.** Mean-flow's reason to exist is **1-step (or few-step)** generation:
   `x₁ = x₀ + u(x₀, 1, 0)` in a single jump. HardFlow instead runs a **K=20-step** Euler integration
   and projects at ~half the steps. There is no per-step instantaneous `v` for the NLP to project
   against in the mean-flow parameterization — the object HardFlow needs simply isn't what these
   nets emit.

3. **Loader / config drift.** If one instead tried to force the plain `GaussianDiffusion` wrapper
   onto an iMF checkpoint, `load_diffusion_with_override` **drops** every diffusion kwarg the plain
   class doesn't accept (`eval:96-98` — interval-CFG, aux weight, two-time schedule…), and
   `_predict_velocity` collapses to `self.model(x, cond, t)` (`diffusion.py:76`) — again calling the
   DiT with no interval and taking the `u`-head. Same wrong-field outcome, now with silently dropped
   hyperparameters on top.

**Bottom line for the HardFlow arm:** loading iMF/MeanFlow/α-Flow is **not a config swap**. HardFlow's
in-loop projection is derived for an instantaneous-velocity probability-flow ODE. To use it with a
mean-velocity model you would have to **re-derive** the in-loop projection for `u(x,t,r)` (what is the
"endpoint to project" when the field is an interval average? how do you project at an intermediate τ
when the model is trained for a single jump?). That is a research change, not a load.

---

## 3. The one thing that DOES load: the plain FMv3 baseline

Every sibling (`flow_matcher_v3_imeanflow`, `_meanflow`, `_alphaflow`) also carries a **plain**
`models/diffusion.py::GaussianDiffusion` — the vanilla FMv3 baseline copied in by the copy-modify
convention. Its `_predict_velocity` is **byte-identical** to Gen12's:

```python
# flow_matcher_v3_{imeanflow,meanflow,alphaflow}/models/diffusion.py:90   (== hardflow diffusion.py:71)
def _predict_velocity(self, x, cond, t, returns=None):
    ...
    return self.model(x, cond, t)          # instantaneous v(x,t)
```

A checkpoint trained with **that** class (the FM baseline, not the iMF/MF/αF engine) loads into Gen12
and drives the HardFlow arm perfectly — it's the same `v(x,t)` contract. So the precise statement is:

> Gen12 can load the **FMv3 baseline checkpoint** that happens to live inside the v4/v6/v7 folders,
> but **not** the iMF / MeanFlow / α-Flow *engines* themselves — because those change the field
> parameterization from `v(x,t)` to `u(x,t,r)`.

Do not confuse "the folder is Gen3v4" with "the checkpoint is iMF." Only checkpoints trained by
`imf_diffusion` / `mf_diffusion` / `af_diffusion` are the real mean-flow models, and those are the
ones the HardFlow arm rejects.

---

## 4. Gen13 (HardFlow-native + iMF) — impossible, and for a deeper reason

Gen13 is **not a sibling of Gen12**. Per `MASTER_TEST_HISTORY.md`:

> ⚠️ *Gen13 is built on HardFLOW, whereas Gen12 is built on DPCC/FMv3ODE. Their deep math and robotic
> mechanisms are fundamentally different.*

Gen13 lives in the vendored `HardFlow/` upstream repo (`hardflow/models_flow/imf/`), with its **own**
trajectory layout, **own** NLP formulation, **own** training/checkpoint format, and — being the
HardFlow-native line — it already did the exact "integrate iMF into an in-loop constrained sampler"
work that §2 says Gen12 would have to re-derive. So Gen13 checkpoints are unloadable by Gen12 not just
on the velocity-field grounds of §2 but on **every** axis: different package, different state-dict,
different constraint engine, different data contract. There is no `load_diffusion_with_override` path
that reaches them.

Worth noting the punchline from Gen13's own results: even in HardFlow's *native* code the iMF
efficiency thesis was **refuted** (FM@K=2 beats iMF at every matched K; `raw_mse_u` is a residual, not
accuracy). So porting iMF into the Gen12 line wouldn't just be hard — the evidence so far says it
wouldn't pay off either.

---

## 5. Summary

- **Gen12's HardFlow arm requires an instantaneous-velocity field `v(x,t)`** queried as
  `_predict_velocity(x, cond, t)` and extrapolated via the linear-FM endpoint `x₁ = x_τ + (1−τ)·v`.
  Only FMv3 / FlowMatchingODE checkpoints satisfy this.
- **iMF / MeanFlow / α-Flow are mean-velocity `u(x,t,r)` models.** Their `diffuser`/`dpcc` arms would
  run natively (DPCC is post-hoc and field-agnostic), but the **HardFlow arm cannot use them** — it
  would silently integrate the wrong field. Making it work is a re-derivation of the in-loop
  projection for an interval-average field, not a checkpoint swap.
- **Gen13 is a different codebase (HardFlow-native) with fundamentally different math** and is
  unloadable by Gen12 on every axis — and its own results already refuted the iMF efficiency case.

> **Refinement — see §6.** The "re-derivation" verdict above is for *upgrading Gen12* to consume the
> mean-flow **u-field sampler** directly. There is a cleaner alternative (the user's proposal): port
> HardFlow's modules **into** v6/v7 and query the u-head at the **`h=0`** anchor, where
> `u(x,t,0)=v(x,t)` exactly — which makes the projection math valid with **no** re-derivation.

---

## 6. Next-step proposal: port HardFlow modules **into** Gen3v6/v7 (not upgrade Gen12)

**User's proposal:** rather than upgrade Gen12 to support mean-flow, add the HardFlow modules into
Gen3v6 / Gen3v7 so those siblings can eval **diffuser + dpcc + hardflow** together on the mean-flow
checkpoints. **Verdict: yes — this is the better move, and it's cleaner than it looks, *provided* the
HF arm queries the mean-flow net at `h=0`.** Here's the full reasoning.

### 6a. Why it's clean: the MeanFlow identity gives you a real instantaneous `v`

§2 said HardFlow needs `v(x,τ)` and mean-flow emits `u(x,t,r)`. That is *not* a dead end, because the
mean-flow objective **grounds `u` on the instantaneous velocity at zero interval**. Directly from the
Gen3v4 iMF loss derivation (identical family in Gen3v6/v7):

```python
# flow_matcher_v3_imeanflow/models/imf_diffusion.py:544  (JVP loss docstring)
# "At the r==t anchor (h=0) this reduces to  u_target = v_inst — the FM velocity — which
#  grounds the field."
#   ⇒  u(x, t, h=0) = v(x, t)     exactly, by construction of the training target
```

So a mean-flow checkpoint **can** hand HardFlow a genuine instantaneous velocity — you just query the
u-head with **`h = 0` (r = t)** instead of the native `h = dt` interval the mean-flow sampler uses:

```python
# native mean-flow sampler (few-step): interval average
#   imf_diffusion.p_sample_loop → u(x, τ, h=dt)          # NOT instantaneous
# HardFlow port (what to call instead):
#   _predict_velocity(x, cond, t, h=torch.zeros_like(t)) # = v(x, τ), valid for Euler + endpoint
```

This is the key design decision. With `h=0`, HardFlow's Euler step (`x + dt·v`) and its endpoint
identity (`x₁ = x_τ + (1−τ)·v`) are **mathematically valid with no re-derivation** — you're feeding
the projection engine exactly the field it was built for. With `h=dt` you'd reintroduce the §2 bug
(projecting an interval-average as if it were instantaneous). **Commit to `h=0` for the HF arm.**

### 6b. Why porting-into-v6/v7 beats upgrading Gen12

| | **Upgrade Gen12 to load mean-flow** | **Port HardFlow into v6/v7** ✅ |
|---|---|---|
| what changes | re-derive the in-loop projection for an interval-average field `u(x,t,r)`; new loader path; new endpoint semantics | copy the 3 HardFlow modules + one arm into v6/v7; feed `h=0` to `_predict_velocity` |
| math risk | **high** — "what is the endpoint to project when the field is an interval average?" is an open question | **low** — instantaneous `v` via `u(·,·,0)`; projection math unchanged |
| touches validated Gen12 | yes (regression risk to fix_7) | **no** — Gen12 untouched |
| repo convention | breaks copy-modify isolation (one folder trying to be two engines) | **matches it** — Gen12 itself was FMv3ODE + HardFlow modules copied together |
| what you can compare | mean-flow vs FM, but only via a bespoke new sampler | **diffuser + dpcc + hardflow, 3-arm, on each model family**, apples-to-apples |

The HardFlow modules (`hardflow_projection.py` — the casadi NLP, `HardFlowSampler`, `HardFlowPolicy`)
are **field-agnostic once fed an instantaneous `v`**. Porting is mostly plumbing: the only
model-specific line is the velocity query (`_velocity_batch` → call the mean-flow `_predict_velocity`
with `h=0`). Everything downstream (endpoint, NLP, tightening, DPCC-parity batching from fix_7) carries
over unchanged.

### 6c. What the HF arm then measures — state this in the results

Running HardFlow's K-step Euler on `u(x,τ,0)` **uses the mean-flow model as a plain FM field** and
therefore **gives up mean-flow's few-step speed for that arm.** So the comparison the ported HF arm
enables is:

> *"Under identical in-loop constrained sampling, is the field iMF/MeanFlow/α-Flow **learned** better
> or worse than the FMv3 field?"*

— i.e. a **field-quality** A/B, **not** a "few-step generation speed" claim. That's exactly the right
question for a DPCC-vs-HardFlow safety/quality study, but it must be labelled as such, or a reader will
wrongly expect the mean-flow speed advantage to show up in HF timings (it can't — every active step is
a full net eval, same as FM).

### 6d. Costs / caveats before committing

1. **Three-way sync burden.** HardFlow modules would now live in Gen12, v6, and v7. The repo already
   syncs fixes across siblings (Gen11↔Gen7↔Gen6V4), so this is the normal tax — but fix_7-class
   changes now have three homes. Consider a thin shared import if the drift becomes painful.
2. **`h=0` vs the v-head.** Prefer the `u(x,t,0)` identity over the separate aux/v-head — iMF's
   `_predict_velocity` comment (`imf_diffusion.py:176-199`) explicitly **discards** the aux head at
   inference because it "varies step-to-step" and injected jitter. `u(·,·,0)` is the grounded,
   stable instantaneous field. (For **α-Flow**, note the field is a homotopy: it *is* the FM velocity
   at α=1 and drifts toward mean-flow as α→0, so `h=0` is exactly right there too.)
3. **Temper the expectation.** Gen13 already did HardFlow+iMF *natively* and the efficiency thesis was
   **refuted** (FM@K=2 beat iMF at every matched K; `raw_mse_u` is a residual, not accuracy). This
   port answers a *different, cleaner* question (same-sampler field quality under DPCC vs HF), but
   don't expect the HF arm to suddenly favor the mean-flow models.
4. **Checkpoints must be the real mean-flow engines.** Point the port at `imf_diffusion` /
   `mf_diffusion` / `af_diffusion` checkpoints — not the plain FM baseline sitting in the same
   folder (§3), or you'd just be re-running Gen12 under a new name.

### 6e. Recommendation

**Do it, as a copy-modify sibling** (e.g. `flow_matcher_v3_meanflow_hardflow/` ↔ test, and the α-Flow
analog), with the HF arm querying the u-head at **`h=0`**. It is lower-risk than upgrading Gen12, it
respects the repo's isolation convention, it reuses the fix_7-validated HardFlow engine almost
verbatim, and it yields the 3-arm (diffuser/dpcc/hardflow) comparison you want — as long as the writeup
is explicit that the HF arm is a **field-quality** test, not a few-step-speed test.

---

### Code references
- Gen12 native-load (no override): `FM_v3_hardflow_test/eval_FM_v3_hardflow.py:148, :171`; override/kwarg-drop `:87-98`.
- MeanFlow identity `u(x,t,0)=v` (enables the port): `flow_matcher_v3_imeanflow/models/imf_diffusion.py:544` (loss docstring); query API `_predict_velocity(x,cond,t,h=…)` `:174`.
- Native mean-flow sampler uses `h=dt` (interval, NOT instantaneous): `imf_diffusion.py` `p_sample_loop` (`h_batch = dt`).
- Field-agnostic arms: `flow_matcher_v3_hardflow/sampling/policies.py:52`; DPCC post-hoc `sampling/projection.py::Projector`.
- HardFlow instantaneous-v requirement: `flow_matcher_v3_hardflow/sampling/hardflow_projection.py:389-390` (query), endpoint `X1_ref = X_ref + (1−τ)·V_next`.
- Plain FM contract (identical across siblings): `flow_matcher_v3_hardflow/models/diffusion.py:71-76` ≡ `flow_matcher_v3_{imeanflow,meanflow,alphaflow}/models/diffusion.py:90-95`.
- Mean-flow two-time interface: `flow_matcher_v3_imeanflow/models/imf_diffusion.py:167-199`; DiT forward `imf_dit_trajectory.py:361-381`.
- Gen13 separateness: `logs_in_develop/MASTER_TEST_HISTORY.md` (Gen13 row).
