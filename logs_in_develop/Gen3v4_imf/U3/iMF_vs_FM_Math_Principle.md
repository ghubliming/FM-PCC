# iMF vs FM — Will iMeanFlow Actually Beat Flow Matching? (Math / Principle)

**Date:** 2026-06-13
**Scope:** Gen3v4 `flow_matcher_v3_imeanflow/` as the concrete example.
**Question:** Principally — by the math — is iMF *better* or *worse* than the original FM?
**Short answer:** As currently implemented, iMF is **mathematically equivalent to FM** (it
cannot be better), and carries dead machinery that can only add variance. A *properly*
implemented MeanFlow *could* be better, but only on one axis (few-step inference speed), and
at a real cost (training-time JVP, optimization instability). Details below.

> **Clarification added 2026-06-13 (after reading the official repo `/workspaces/imeanflow`).**
> The "FM-equivalent" claim is about the **training target**, not a claim that the port is
> sloppy or "fake." On the contrary — §8 shows Gen3v4 **faithfully reproduces the official
> iMF *inference algorithm and dual-head architecture***. The single thing that diverges is
> the **training objective**: the official PyTorch repo is *inference-only* and ships **no
> training code** (the JVP/MeanFlow-Identity training is JAX-only), so Gen3v4 substituted a
> finite-difference target that — for the linear interpolant — reduces to the FM velocity.
> That is the precise, narrow sense in which "iMF ≈ FM" here. See §8 and §9.

---

## 1. What FM does (the baseline target)

Original FMv3 ODE flow matching trains a network to predict the **instantaneous velocity**
of a linear (OT) interpolant between noise `ε` and data `x₁`:

```
x_τ = (1−τ)·ε + τ·x₁           (DATA-AT-1: τ=0 noise, τ=1 data)
v(x_τ, τ) = d/dτ x_τ = x₁ − ε    ← constant along the straight path
```

The training target is `v_target = x₁ − ε` (see `imf_diffusion.py:293`,
`v_target = x_start − x_base`). Inference integrates this velocity field with an N-step ODE
solver (Euler), `x ← x + v·dt`. Config: `ode_inference_steps_v3 = 10`
(`config/avoiding-d3il.py:488`).

That is the whole baseline: **predict `x₁ − ε`, integrate over N steps.**

---

## 2. What MeanFlow does *in theory* (the real idea)

MeanFlow (Geng et al. 2025) does not predict the instantaneous velocity `v(z,t)`. It predicts
the **average velocity over an interval** `[r, t]`:

```
                1     ⌠ t
u(z, r, t) =  ─────   │   v(z_s, s) ds
              t − r   ⌡ r
```

The point of this is the **MeanFlow Identity**, obtained by differentiating the definition:

```
u(z, r, t) = v(z, t) − (t − r) · d/dt u(z, r, t)
                                  └────────┬────────┘
                                  total derivative (a JVP):
                          d/dt u = ∂u/∂t + (∂u/∂z)·v
```

The payoff: if the network directly knows the **average** velocity across the whole interval,
you can jump `t=0 → t=1` in **one Euler step** (`x₁ ≈ x₀ + 1·u(x₀, 0, 1)`) instead of
integrating `v` over many steps. **Few-step / one-step generation is the entire reason
MeanFlow exists.** For a real-time controller (the 30 ms / 33 Hz budget in
`REALTIME_RECORDING/IDEAS.md`) that is a genuinely attractive property: fewer NFE = lower
`fm_ms`.

The crucial term is the **JVP `d/dt u`**. It is what makes `u` differ from `v`. It captures
the *curvature of the marginal (expectation) velocity field* — the field you actually
integrate at inference, where many data points share the same `z_t`.

---

## 3. What Gen3v4's iMF *actually* does (read the code)

Look at `imf_diffusion.py:p_losses` (lines 258–313). It builds the mean-flow target by
**finite-differencing the sample-level interpolant**:

```python
r = t * torch.rand_like(t)          # r ~ U(0, t)
h = t - r                           # interval size
x_t = q_sample(x_start, t, noise)   # = (1−t)ε + t·x₁
x_r = q_sample(x_start, r, noise)   # = (1−r)ε + r·x₁
u_target = (x_t - x_r) / h          # "mean flow" over [r,t]
```

Now do the algebra on that last line. Both `x_t` and `x_r` are on the **same straight line**
(same `ε`, same `x₁`):

```
x_t − x_r = [(1−t)ε + t·x₁] − [(1−r)ε + r·x₁] = (t − r)(x₁ − ε)

u_target = (x_t − x_r) / (t − r) = x₁ − ε          ← identical to FM's v_target
```

**The mean-flow target collapses to the constant instantaneous velocity `x₁ − ε`.** The code
even says so (`imf_diffusion.py:286-288`): *"For the linear interpolant this equals the
constant v = x_data − noise, independent of which endpoint we condition on."*

And the aux/v head target is `v_target = x₁ − ε` (`imf_diffusion.py:293`) — **the exact same
vector.** So:

| Head | Target in Gen3v4 | Same as FM? |
|---|---|---|
| `u` (mean-flow) | `x₁ − ε` | **Yes — identical** |
| `v` (aux/instantaneous) | `x₁ − ε` | **Yes — identical** |

Then at **inference**, the aux head is **discarded entirely** (`_predict_velocity`,
`imf_diffusion.py:115-131`, "FIX-3 / Deviation A": *"reference iMF's inference uses ONLY the u
head and explicitly DISCARDS the v head"*), and sampling runs a **10-step Euler loop**
(`p_sample_loop`, `flow_steps_v3`), not a one-step jump.

### What is left after the collapse

The deployed model is: **predict `x₁ − ε`, integrate 10 Euler steps.** That is *exactly*
FMv3 ODE. The iMF additions that survive are:

1. an **auxiliary head** trained on the same target as the main head, then thrown away;
2. an **`h` (interval-size) conditioning input** to the UNet (`unet1d_temporal_cond.py`),
   which — since the target does **not** depend on `h` — the network must learn to *ignore*.

---

## 4. Why, by the math, the current iMF cannot beat FM

The collapse in §3 is not a bug to be tuned away — it is **inherent to finite-differencing the
sample-level interpolant**. The genuine MeanFlow signal lives in the JVP term `d/dt u`, which
only appears when you differentiate the **marginal/expected** velocity field (averaging over
all data that pass through a given `z_t`). The sample-level path of a *single* `(ε, x₁)` pair
is a straight line traversed at constant speed; its average velocity over any sub-interval is
trivially that same constant. **No interval `[r,t]` choice can recover curvature that the
sample-level construction does not contain.**

Consequences, strictly from the math:

- **Capability ceiling = FM.** Same model class, same regression target. iMF here cannot
  represent anything FM cannot. Best case it *matches* FM.
- **The one advantage (few-step) is not taken.** Inference is 10-step, identical NFE budget to
  FM. So even the speed argument — the only place a real MeanFlow wins — is off the table in
  the current config.
- **Strictly added variance.** The discarded aux head and the must-be-ignored `h` input are
  extra parameters and extra loss terms (`aux_loss_weight`, `imf_diffusion.py:302-304`) that
  consume capacity and inject gradient noise without changing the target. Expected effect:
  **neutral-to-slightly-worse**, plus a harder optimization landscape (this is consistent with
  the project's own history — `fix_3_major_rebuild`, `fix_5_major_rebuild`: the dual-target
  curriculum was unstable and had to be collapsed to "FM-style main loss + small aux," per
  `imf_losses.py` docstring).

> **Bottom line for Gen3v4 as it stands:** iMF ≈ FM in capability, ≤ FM in robustness. It is
> "FM wearing an iMF costume." The marginal-improvement-but-still-feels-bad observation in the
> project notes is exactly what this predicts.

---

## 5. When a *proper* iMF would be better — and when worse

To make iMF actually iMF, you must compute the JVP `d/dt u = ∂u/∂t + (∂u/∂z)·v` (forward-mode
autodiff through the network) and train against the **MeanFlow Identity**, not the
finite-difference of a single interpolant. Then:

### Better (the upside)
- **Few-step / one-step inference.** Direct average-velocity prediction lets you cut NFE from
  ~10 to 1–2. For FM-PCC real-time control this is the headline win: lower `fm_ms`, easier to
  fit the 30 ms budget. This is the *only* axis on which iMF can beat FM in trajectory
  *quality-per-compute*.
- **Smoother few-step samples.** A correct `u` integrates curvature analytically, so 1–2 step
  rollouts avoid the discretization error that cripples 1–2 step plain-FM Euler.

### Worse (the cost / risk)
- **Training cost.** The JVP adds a forward-over-reverse autodiff pass every step — materially
  more expensive and memory-hungry than FM's single forward.
- **Optimization instability.** The identity is *self-referential* (`u` appears on both sides,
  and the target depends on the network's own derivative). This is a notoriously stiff target;
  it is precisely why the project's curriculum (`u_first → balanced`) kept diverging and was
  abandoned.
- **No gain at matched NFE.** If you still sample with 10+ steps, a correct iMF buys you almost
  nothing over FM — the curvature correction matters most exactly when steps are few. Paying
  the JVP cost without dropping NFE is strictly a loss.

---

## 6. Verdict table

| Property | Original FM | Gen3v4 iMF (as-is) | Proper iMF (with JVP) |
|---|---|---|---|
| Training target | `x₁ − ε` | `x₁ − ε` (collapsed) | average `u` via MeanFlow Identity |
| Captures field curvature? | No (not needed; multi-step) | **No** | **Yes** (the JVP term) |
| Inference NFE | ~10 | ~10 (advantage unused) | **1–2** (the point) |
| Extra machinery | — | aux head + `h` cond (both inert/discarded) | aux + JVP (used) |
| Training cost | baseline | ~baseline (+ wasted aux) | **higher** (JVP) |
| Optimization stability | stable | ≤ FM (extra noise) | **harder** (self-referential) |
| Expected quality vs FM | — | **≈ FM, possibly slightly worse** | better *only* at low NFE |

---

## 7. Recommendation

1. **Do not expect the current Gen3v4 iMF to beat FM.** By the algebra in §3 it is the same
   model; any measured difference is regularization noise from the inert aux head / `h` input,
   not a new capability. The "marginal improvement, still feels bad" result is the expected
   outcome.
2. **If iMF is worth pursuing, commit to the real thing:** implement the JVP `d/dt u` and train
   on the MeanFlow Identity, **and** drop inference to 1–2 NFE — otherwise there is no point.
   Measure success as *quality-at-low-NFE* and *`fm_ms`*, not as quality at 10 steps (where FM
   already wins on simplicity).
3. **If the goal is just trajectory quality at fixed NFE, prefer FM.** It is the same target
   with less to go wrong. Spend the iMF complexity budget only where its one real lever —
   few-step inference for the real-time control loop — is actually exercised.

---

## 8. Cross-check against the official iMF repo (`/workspaces/imeanflow`)

I read the official PyTorch re-implementation to verify the claim is fair rather than
"bombastic." Here is exactly what matches and what does not.

### 8.1 The decisive fact: the official PyTorch repo has NO training code

`imf.py:29` hard-asserts inference: `assert eval, 'The current codebase only supports
inference mode'`. The README states plainly: *"We only provide inference code and pre-trained
checkpoints in this repo. For training code, please refer to the original JAX implementation."*

**Implication:** the JVP / MeanFlow-Identity training — the mathematical heart of the method —
is **not present in the repo Gen3v4 was ported from.** Gen3v4 could not copy it because it was
never there. So Gen3v4's `p_losses` is an *original* training objective, not a port, and that
is where (and the only place where) it diverges from true MeanFlow.

### 8.2 What Gen3v4 ports FAITHFULLY (credit where due)

| Aspect | Official repo | Gen3v4 | Match? |
|---|---|---|---|
| **Dual-head architecture** (shared backbone → `u_heads` + `v_heads`) | `imfDiT.py:278-286, 353-390` returns `(u, v)` | `iMeanFlowEngine` returns `(u, aux)`; UNet has u + aux head | ✅ Faithful |
| **v-head dropped at eval** | `eval_mode=True` ⇒ `v_heads` not instantiated (`imfDiT.py:286`); `u_fn(...)[0]` uses u only (`imf.py:93,135`) | `_predict_velocity` uses u, discards aux (`imf_diffusion.py:115-131`, "FIX-3") | ✅ Faithful |
| **Interval (`h = t−r`) conditioning of the network** | `forward(x, t, h, w, t_min, t_max, y)` (`imfDiT.py:353`) | `model(x, t, h=h, cond=...)` (`imf_engine.py:68-86`) | ✅ Faithful |
| **Sampling update `z ← z ∓ (t−r)·u`** | `z_t - (t-r)*u` (`imf.py:95,136`) | `x + velocity·dt` (`imf_diffusion.py:184`) | ✅ Faithful (sign = convention) |

So the **inference path and the model are genuine iMF.** This is not a costume — the scaffolding
is correctly transplanted.

### 8.3 What diverges — and why it matters

| Aspect | Official iMF | Gen3v4 | Match? |
|---|---|---|---|
| **Training target for `u`** | average velocity via **MeanFlow Identity** `u = v − (t−r)·d/dt u` (JVP), JAX-only | finite difference `(x_t − x_r)/h` of the **sample-level** interpolant ⇒ `= x₁ − ε` | ❌ **Diverges** — JVP term absent |
| **Inference NFE** | **1–2** (the entire selling point; README FID 3.32 @ NFE=1) | **10** (`ode_inference_steps_v3: 10`) | ❌ few-step capability unused |

The JVP term is exactly the part that makes `u` a *true* average velocity (coupled to the
marginal field) rather than the instantaneous conditional velocity. Without it:

- the learned `u(x, r, h)` regresses `E[x₁ − ε | x_r]`, which **does not depend on `h`** → `h`
  is an inert input (the network's Bayes-optimal output ignores it);
- that learned `u` **equals FM's `v`** in expectation;
- so the few-step update `z ← z − (t−r)·u`, run at low NFE, would be plain low-step Euler —
  inaccurate — which is *why* Gen3v4 has to run 10 steps and gets no speed win.

### 8.4 Honest restatement of the verdict

> Gen3v4 is a **faithful port of iMF inference on top of an FM-equivalent training objective.**
> It is "real iMF" everywhere except the one place that gives MeanFlow its power — the
> JVP-based average-velocity training — which the source PyTorch repo did not contain. The
> earlier "FM in a costume" phrasing was too dismissive of the (correct) architecture/inference
> work; the accurate statement is: **the body is iMF, the learning signal is FM.**

---

## 9. Verdict on OUR iMF on DPCC avoiding (the thing you actually run)

This is the concrete question: with the SLSQP projector in the loop on `avoiding-d3il`, what
does Gen3v4 iMF give us versus the FMv3ODE baseline?

### 9.1 What actually executes

On avoiding, `iMeanFlowODE.p_sample_loop` (`imf_diffusion.py:141-215`) runs a **10-step Euler
rollout** of the u-head, and the **DPCC projector snaps near the end** of the rollout
(`snapping_start_idx = (1 − threshold)·flow_steps`, lines 187-206). Because the u-head was
trained to `x₁ − ε` (§3), this is **operationally identical to running FMv3ODE with the same
10-step sampler and the same end-of-rollout projection.**

### 9.2 The verdict

| Question | Answer for Gen3v4 iMF on avoiding |
|---|---|
| Is it broken / "fake" output? | **No.** It produces valid trajectories; the projector enforces constraints exactly as in FMv3ODE. |
| Does it beat the FMv3ODE baseline? | **No reason to, by the math.** Same target, same NFE, same projector → same capability. Any delta is regularization noise from the (discarded) aux head + inert `h` input. |
| Does it explain "marginal improvement, still feels bad"? | **Yes — exactly.** That is the predicted signature of an FM-equivalent model with extra inert machinery. |
| Is the constraint satisfaction affected? | **No.** Projection acts on `x` post-velocity-step; it is agnostic to whether the velocity came from a u-head or a v-head. |
| Where could a *proper* iMF help avoiding? | **Only at low NFE.** A JVP-trained u at 1–2 NFE would cut `fm_ms` for the real-time loop (`REALTIME_RECORDING/IDEAS.md`, 30 ms budget). That is the sole avoiding-relevant upside. |

### 9.3 The avoiding-specific catch if you ever do the real iMF

Even a correct JVP-trained iMF would **not** be a drop-in win on DPCC avoiding, because the
projector logic is **NFE-coupled**: it starts snapping at `(1 − threshold)·flow_steps`. At
NFE=1–2 the "near-end" window is the *entire* rollout, so the projector would fire on an
almost-pure-noise or single-jump iterate — a very different geometry than the 10-step case it
was tuned for. **Going few-step changes when/how DPCC can act.** So the proper-iMF upgrade is
not just "add the JVP"; it is "add the JVP **and** re-derive the projection schedule for low
NFE, then re-validate constraint satisfaction." Budget for both or neither.

### 9.4 Recommendation for the avoiding line

1. **Treat current Gen3v4 iMF as an FMv3ODE-equivalent on avoiding.** Do not report it as a
   distinct, stronger model; report it as "FM-equivalent (iMF inference scaffold, FM training)."
   If its avoiding numbers ≈ FMv3ODE, that is the correct, expected result — not a failure.
2. **Only invest in iMF if the goal is real-time few-step control.** Then implement the JVP
   training (port from the JAX repo referenced in the README), drop NFE to 1–2, *and* re-tune
   the DPCC snap schedule. Measure success as constraint-satisfaction-at-low-NFE and `fm_ms`,
   never as quality at 10 steps (where FMv3ODE already suffices).
3. **Otherwise, prefer FMv3ODE for the avoiding baseline** — identical capability, less to break.

---

## 10. First principles: why FM-over-diffusion was a free swap, but iMF-over-FM is not

You asked the exact right question: *FM replaced diffusion with no drama — why is iMF
replacing FM problematic? Did we get the first principles wrong?*

**Answer: you did not make a coding mistake. You made a category mistake** — you assumed iMF is
a *label swap* like FM was. It is not. It belongs to a different class of objective. Here is the
precise reason, and the recipe to fix it.

### 10.1 The hidden property that made diffusion→FM trivial

Look at what the network regresses in each method, and ask one question: **does the training
target contain the network itself?**

| Method | Per-sample target the net regresses to | Target contains the net? | Training kind |
|---|---|---|---|
| **Diffusion (DDPM)** | `ε` (the sampled noise) | **No** | plain MSE to a label |
| **Flow Matching** | `v = x₁ − ε` | **No** | plain MSE to a label |
| **MeanFlow (iMF)** | `u = v − (t−r)·d/dt u_θ` | **YES — `d/dt u_θ` is the network's own derivative** | **bootstrapped / self-referential** |

Diffusion and FM share one decisive property: **the target is a closed-form label you compute
from the data sample alone** (`ε`, or `x₁−ε`). Training is "draw a sample, compute the label,
MSE." Swapping diffusion→FM is therefore *just changing the label and the sampler* — a few
lines. That is why it dropped in cleanly.

**MeanFlow breaks that property on purpose.** Its target is defined by the **MeanFlow
Identity**:

```
u(z_t, r, t) = v(z_t, t) − (t − r) · d/dt u_θ(z_t, r, t)
                                     └──────────┬──────────┘
                         total derivative along the trajectory:
                         d/dt u_θ = ∂_t u_θ + v · ∂_z u_θ     (a JVP of the network)
```

The target **literally contains a derivative of the network being trained**. This is the same
structural class as consistency models, score distillation, and TD-learning — *not* the class
that diffusion and FM live in. You cannot express it as "compute a label, MSE." That is the
first-principles reason it is not a free swap.

### 10.2 Why our finite-difference shortcut HAD to degenerate to FM

Gen3v4 tried to stay in the comfortable "compute a label" world by approximating the average
velocity with a finite difference of the **sample-level** interpolant:

```
u_target = (x_t − x_r) / (t − r)
```

But on a single `(ε, x₁)` pair the interpolant is a **straight line at constant speed**, so
this is *exactly* `x₁ − ε` for any `(r, t)` — the FM label (§3). The finite difference can never
produce the `d/dt u_θ` term, because that term is a property of the **marginal** field (the
average over all data sharing a given `z_t`), which a per-sample difference cannot see. **The
shortcut was doomed by construction, not by a bug.** The only way to get the real signal is to
actually differentiate the network. There is no label-only path to MeanFlow.

### 10.3 What "real iMF" requires — the three things we are missing

1. **A JVP through the network at train time.** Compute `d/dt u_θ = ∂_t u_θ + v·∂_z u_θ` with
   forward-mode autodiff (`torch.func.jvp`), tangent `v` on the `z` input and `1` on the `t`
   (and `h`) input.
2. **A stop-gradient on the target.** The bracket `v − (t−r)·d/dt u_θ` is the *target*; detach
   it. Gradients flow **only** through the predicted `u_θ`, never through the JVP. (Otherwise
   you need second-order grads and it blows up.) This is what makes a self-referential objective
   trainable.
3. **A mixed `(t, r)` sampling schedule with a real fraction at `r = t`.** The `r=t` samples
   have `(t−r)=0`, so their target is exactly `v = x₁−ε` — they **anchor** the field to the true
   instantaneous velocity (this is your FM signal, kept as a special case). The `r<t` samples
   propagate the average-velocity consistency. MeanFlow uses ~25–50% of the batch at `r=t`.
   Gen3v4 currently samples `r ~ U(0,t)` with **no mass at `r=t`** — so even the anchor is weak.

Plus the two things from §8 we already flagged:
4. **Low NFE inference (1–2)** to actually reap the speed — otherwise there is no point.
5. **Adaptive loss weighting** (e.g. `w = (‖Δ‖² + c)^(−p)` with `Δ` detached, `p≈0.5–1`,
   `c≈1e-3`) — the JVP targets have wildly varying magnitude; without normalization training is
   unstable. This is in the MeanFlow recipe and is not optional.

### 10.4 Concrete migration for Gen3v4 (`imf_diffusion.py:p_losses`)

The architecture is already correct (u-head + `h`-conditioning exist). The change is the
**objective**. Sketch, in Gen3v4's DATA-AT-1 convention (`q_sample`, `_predict_uv`):

```python
import torch
from torch.func import jvp   # forward-mode AD

def p_losses(self, x_start, cond, t, returns=None):
    eps   = torch.randn_like(x_start)                       # noise  (x_base)
    eps   = apply_conditioning(eps, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

    # (3) mixed interval sampling: a fraction with r == t  (anchor = FM)
    r = t * torch.rand_like(t)
    anchor = torch.rand_like(t) < 0.25                      # ~25% at r=t
    r = torch.where(anchor, t, r)
    h = (t - r)

    z_t = self.q_sample(x_start=x_start, t=t, noise=eps)    # point the net sees
    z_t = apply_conditioning(z_t, cond, self.action_dim, goal_dim=self.goal_dim)

    v_inst = x_start - eps                                  # conditional instantaneous velocity (FM target)
    v_inst = apply_conditioning(v_inst, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

    # (1) JVP: total derivative d/dt u_theta along the trajectory.
    #     tangents: dz/dt = v_inst on the latent, dt = 1 on time, dh = 1 (h=t-r, r fixed).
    def u_only(z_in, t_in, h_in):
        u, _aux = self._predict_uv(z_in, cond, t_in, h=h_in, returns=returns)
        return u
    ones = torch.ones_like(t)
    u_pred, dudt = jvp(u_only, (z_t, t, h), (v_inst, ones, ones))

    # (2) MeanFlow Identity target, STOP-GRADIENT.
    h_b = h.view(-1, *([1] * (x_start.ndim - 1)))
    u_target = (v_inst - h_b * dudt).detach()

    # (5) adaptive weighting on the detached residual
    delta = (u_pred - u_target)
    w = 1.0 / (delta.detach().pow(2).mean(dim=tuple(range(1, delta.ndim)), keepdim=True) + 1e-3).pow(0.5)
    main_loss = (w * delta.pow(2)).mean()

    # optional: keep the tiny aux v-head on v_inst purely as a stabilizer (discarded at sampling)
    ...
    return main_loss, info
```

Then:
- **Inference (`p_sample_loop`)**: set `flow_steps` to **1–2** for the iMF benefit (keep a
  10-step path available for A/B). The update `x ← x + (t_next−t_cur)·u` is already correct.
- **DPCC projector schedule (§9.3)**: at NFE 1–2 the "near-end" snap window is the whole
  rollout — **re-derive `snapping_start_idx`** and re-validate constraint satisfaction. Do not
  ship low-NFE iMF on avoiding until this is checked.

### 10.5 Honest cost / risk of doing it right

- **Compute:** the JVP roughly doubles the per-step forward cost and memory at train time. Fine
  for trajectory-sized tensors (`H=8`), far cheaper than for images.
- **Functional purity:** `torch.func.jvp` needs the forward to be functional — watch
  `condition_dropout` (disable or make deterministic inside the JVP), in-place ops, and EMA.
- **Stability:** self-referential objectives can collapse (commonly *back* to the instantaneous
  solution — i.e. to FM — if the `r=t` anchor dominates, or diverge if the JVP target is
  unweighted). The `r=t` fraction (3) and adaptive weight (5) are the two dials that decide
  success. Expect to tune them; this is the part that "feels hard," and it is hard *for everyone*
  — it is intrinsic to the method, not a Gen3v4 defect.
- **Payoff is conditional:** if after all this you still sample at 10 NFE, you will have spent
  the cost for ≈ no gain (§5). Commit to low-NFE inference or do not start.

### 10.6 One-paragraph answer to "are we doing first principles wrong?"

No — the FM and diffusion targets are *self-contained labels*, so swapping between them is a
label change. MeanFlow's target is *self-referential* (it contains the network's own
directional derivative), which puts it in the consistency-model family, not the
diffusion/FM family. Gen3v4 implemented iMF's **architecture and inference** faithfully but
kept an **FM-style label objective**, which provably degenerates to FM on the linear
interpolant. To get the real power you must add the JVP, stop-gradient the identity target,
sample a real fraction at `r=t`, weight adaptively, and sample at 1–2 NFE. That is a bounded,
well-specified upgrade — not a rewrite — but it is a genuinely *different kind* of training loop
than the one FM let you get away with.

---

## Caveats

- Code read on 2026-06-13, branch `update_into_FM`:
  `flow_matcher_v3_imeanflow/models/imf_diffusion.py`, `imf_engine.py`, `imf_losses.py`;
  config `config/avoiding-d3il.py:437-494`.
- The "collapse to FM" conclusion is exact for the **linear/OT interpolant** used here
  (`q_sample`, `imf_diffusion.py:133-139`). A curved interpolant would not collapse the same
  way — but Gen3v4 uses the straight-line interpolant.
- This is a principled (math) analysis, not a measured benchmark. It predicts *why* the eval
  numbers look the way they do; confirm against the actual `U3/EVAL_STATUS.md` results.
