# HF·iMF — Blending HardFlow (hard-constrained sampling) with iMeanFlow (average-velocity flows)

> **See also `THEORY_DeepMix_HF_iMF.md`** — the deep-math companion: proves HardFlow's terminal prediction is exactly the posterior mean (mode average, not the sample endpoint), derives the Newton pull-back, collapses OC-Flow into an algebraic NLP, and validates all of it numerically in closed form (`validate_theory.py`). This file is the engineering/repo-audit side.

**Date:** 2026-07-02
**Sources read (code-level):**
- `/workspaces/HardFlow` — Li, Alim, Azizan, *HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization*, TPAMI 2026 (`d3il` branch, avoiding benchmark, PyTorch + CasADi/IPOPT)
- `/workspaces/imeanflow` — Lyy et al., *Improved Mean Flows: On the Challenges of Fastforward Generative Models*, arXiv 2512.02012 (official JAX; a PyTorch port exists on the repo's `torch` branch)

**Verdict up front:** the two methods fix each other's central weakness. HardFlow's entire error structure comes from a **first-order Euler extrapolation of the terminal sample** (`x̂1 = z + (1−t)·v`), which iMF's average-velocity field `u` replaces **exactly** (up to training error) at the same 1-NFE cost. Conversely, iMF gives fast unconstrained samples but has no mechanism for hard constraints; HardFlow's prox-NLP + pull-back is exactly that mechanism, and it survives the substitution unchanged. The result is a hard-constrained sampler at 2–4 model evaluations instead of ~60.

---

## 1. The two ingredients

### 1.1 HardFlow in one paragraph

HardFlow (`hardflow/models_flow/flow_policy.py:1286`, `hardflow_new_forward`) steers a pretrained flow-matching sampler so the generated trajectory satisfies **hard** constraints (obstacle quadrilaterals, fitted linear dynamics `A·s + B·a + c = s'`, action box), solved with IPOPT. Its convention: `z_τ = τ·x1 + (1−τ)·x0`, **τ=0 noise → τ=1 data**, forward Euler. Per step `k` at `τ_k = k/N` (default `N = ode_t_steps = 20`):

```
1. Euler ref step:        z_ref  = z_k + v(z_k, τ_k)·Δτ                        (1 NFE)
2. Terminal prediction:   x̂1     = z_ref + (1 − τ_{k+1})·v(z_ref, τ_{k+1})     (1 NFE)   ← the seam
3. Constrained prox NLP:  X1*    = argmin ½ρ·τ_{k+1}²·‖X − x̂1‖²  s.t. X ∈ C    (IPOPT)
4. Pull-back:             z_{k+1} = z_ref + τ_{k+1}·(X1* − x̂1)
```

Step 2 (`flow_policy.py:1339-1340`; same construction in `x1_estimate()` at `:227`) shoots the instantaneous velocity linearly to τ=1. Step 4 rewrites the interpolant with the endpoint swapped and the noise anchor kept (`∂z_τ/∂x1 = τ`). The NLP itself is purely algebraic — the network never enters the solver (that is `hardflow_new`'s whole point vs. the l4casadi variant).

Cost per action: warm-start rollout (N NFE) + 2N NFE + N IPOPT solves ≈ **60 NFE + 20 NLPs** at defaults.

### 1.2 iMeanFlow in one paragraph

MeanFlow-family models learn the **average velocity** over a time interval instead of the instantaneous one. In the official convention (`imf.py:350`: `z_t = (1−t)·x + t·e`, **t=0 data → t=1 noise**, v = e − x):

```
u(z_t, t, r)  ≜  (1/(t−r)) ∫ᵣᵗ v(z_s, s) ds        ⇒        z_r = z_t − (t−r)·u(z_t, t, t−r)   exactly
```

(`sample_one_step`, `imf.py:90-114`; sampling walks `t_steps = linspace(1, 0)`). Training enforces the MeanFlow identity `u = v − (t−r)·du/dt` via a JVP; iMF's improvements over vanilla MeanFlow — **predicted-v as the JVP tangent** (`imf.py:373`), adaptive loss normalization (`:380-382`), an **auxiliary v-head** so one network emits both `(u, v)` (`u_fn`, `:185-210`), CFG scale/interval as conditioning inputs — are what push it to FID 1.70 at **1 NFE** on ImageNet-256. Those same improvements are exactly what a constrained sampler needs from `u`: accuracy and a co-trained instantaneous head.

### 1.3 ⚠ The conventions are reversed — translation table

This is the #1 integration hazard. Everything below states the math in **HardFlow's convention** (τ=0 noise), with the mapping:

| | HardFlow (`τ`) | official iMF (`t`) | map |
|---|---|---|---|
| noise / data ends | τ=0 noise, τ=1 data | t=1 noise, t=0 data | `τ = 1 − t` |
| interpolant | `z = τ·x1 + (1−τ)·x0` | `z = (1−t)·x + t·e` | identical curve |
| instantaneous velocity | `v_HF = x1 − x0` | `v_iMF = e − x` | `v_HF = −v_iMF` |
| exact jump | `z_{τ'} = z_τ + (τ'−τ)·u_HF` | `z_r = z_t − (t−r)·u_iMF` | `u_HF = −u_iMF` |
| terminal (data) prediction | `x̂1 = z + (1−τ)·u_HF(z, τ→1)` | `x̂ = z_t − t·u_iMF(z_t, t→0)` | same object |
| interval-size input `h` | `τ' − τ` | `t − r` | equal |

A sign slip here produces a sampler that walks *toward* noise; any implementation must fix the convention in ONE wrapper and test it with a 1-NFE reconstruction check before anything else.

---

## 2. The seam: where the two meet

HardFlow's step 2 is a **first-order proxy for exactly the quantity iMF learns.**

| | HardFlow today | with iMF |
|---|---|---|
| Terminal prediction | `x̂1 = z + (1−τ)·v(z, τ)` | `x̂1 = z + (1−τ)·u(z, τ, h=1−τ)` |
| Exact when | flow is straight (rectified) | always (up to training error ε_u) |
| NFE | 1 (biased) | 1 (unbiased) |

Everything else in HardFlow — the prox NLP, the `τ²` weighting, the pull-back — depends only on the interpolation-path geometry, **not** on how `x̂1` was produced. The substitution is one call-site.

### 2.1 The proxy-bias bound (the math that makes this a real upgrade)

Let `z_s` solve the marginal-flow ODE from `z` at time τ. Then

```
x1 − x̂1_Euler = (1−τ)·(u − v(z,τ)) = ∫_τ¹ [v(z_s, s) − v(z_τ, τ)] ds
```

so with `L ≜ sup_s ‖(d/ds) v(z_s, s)‖` (velocity drift along the flow — the **flow curvature**):

```
‖x1 − x̂1_Euler‖ ≤ ½·L·(1−τ)²
```

Two observations make this bound bite hard on HardFlow's own benchmark:

1. **`L` is largest exactly at small τ.** The avoiding data is multi-modal (many homotopy classes around the obstacles). Near τ=0 the marginal velocity is an average over modes and bends sharply as the sample commits to one; near τ=1 the flow is locally straight. The Euler proxy is worst *precisely when constraining should start*.
2. **Three of HardFlow's design choices are crutches for this bias**, and all three relax once `x̂1` is exact:
   - `hardflow_activation='late'` (`flow_policy.py:1330`) — skip constraining in the first half because early `x̂1` is unreliable;
   - the `τ²` prox weight (`flow_policy.py:710-715`) — make deviations from an unreliable early reference cheap;
   - `N = 20` Euler steps — keep Δτ small so per-step extrapolation errors stay correctable.

**iMF replaces the structural bias `½L(1−τ)²` with a τ-uniform network error ε_u** — and iMF's ImageNet numbers are the empirical evidence that ε_u can be driven small enough to *sample from directly*, which is a far harsher test than serving as an NLP reference point.

### 2.2 The pull-back carries over unchanged (consistency lemma)

Step 4 keeps the noise-side anchor `x0` implied by `(z, x̂1)` and swaps the endpoint:

```
z' = τ·X1* + (1−τ)·x0 = z + τ·(X1* − x̂1)      given  z = τ·x̂1 + (1−τ)·x0
```

The identity holds for *any* endpoint estimate, so the update formula is untouched. What changes is its meaning: with Euler-`x̂1` the implied `x0` is a first-order fiction; with iMF-`x̂1` the pair `(x0, x1)` is the actual coupling the learned flow transports, so the pulled-back `z'` lies on a genuine model interpolant and the next `u`-evaluation re-attracts it to the data manifold from a consistent point. The `τ²` prox weight also keeps its justification (a terminal move Δ costs τ·Δ in z-space, so `τ²‖Δ‖²` is the z-space metric) — keep it.

---

## 3. The blend, in increasing depth

### Level 0 — naive: iMF as the velocity field inside unmodified HardFlow

The official model's `u_fn` returns `(u, v)` — the auxiliary v-head IS an instantaneous velocity field (`imf.py:185-210`, "By default, we use auxiliary v-head"). So an iMF checkpoint drops into HardFlow's `flow_model(x, τ) → v` slot through a sign/convention shim:

```python
def v_HF(x, tau):                    # HardFlow calls this
    t = 1.0 - tau                    # convention flip
    u, v = imf_u_fn(x, t, h=0)       # h→0 ⇒ u = v; or take the v-head directly
    return -v                        # sign flip (v_HF = −v_iMF)
```

Zero math changes. **Not interesting on its own** — it inherits all of HardFlow's proxy bias — but it validates conventions, normalizer and checkpoint plumbing before the real steps, and isolates "model quality" from "algorithm change" in the eval.

### Level 1 — the surgical substitution (the core of the blend)

Replace the two `flow_eval_np` call-sites inside `hardflow_new_forward`:

```python
# flow_policy.py:1339-1340  — today (Euler extrapolation):
#   v_next = flow_eval_np(x_next_ref, t_k + dt)
#   x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next
# blended (exact jump to the data end):
u_next = imf_eval_np(x_next_ref, tau=t_k + dt, h=1.0 - t_k - dt)
x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * u_next

# flow_policy.py:1324-1325 — today (Euler transport):
#   x_next_ref = x_k + flow_eval_np(x_k, t_k) * dt
# blended (exact interval jump):
x_next_ref = x_k + dt * imf_eval_np(x_k, tau=t_k, h=dt)
```

Consequences, in order of importance:

1. `hardflow_activation='all'` becomes trustworthy **from step 0**. Constraints then shape the sample throughout generation — *steering between modes* (choosing a homotopy class that clears the obstacle) rather than *deforming within a mode* (bending a committed path until it barely clears). HardFlow-'late' can only do the second.
2. `N` collapses. The only remaining reasons for multiple outer steps are (a) re-attracting to the manifold after each constraint correction and (b) letting constraints influence mode choice progressively. Both need K ≈ 2–4, not 20 — iMF is *designed* for 1–2-step sampling.
3. NFE per action: ~60 → 2K ≈ 4–8. IPOPT solves: 20 → K. The NLP is unchanged (same 44-dof variables, same constraints); there are simply 5–10× fewer solves — and the solves dominate HardFlow's wall-time.
4. The diagnostic `x1_estimate()` chain (`:227`) becomes exact too — HardFlow's own visualization of "where the sampler thinks it's going" stops lying at small τ.

### Level 2 — K-step iMF-HardFlow (the concrete proposed algorithm)

```
Input: anchor times 0 = τ_0 < τ_1 < … < τ_K = 1 (K ∈ {2,3,4}), trained iMF u(·,·,h), constraint set C
z ← N(0, I);  apply conditioning
for k = 0 … K−1:
    z_ref ← z + (τ_{k+1} − τ_k) · u(z, τ_k, h = τ_{k+1} − τ_k)           # exact interval jump
    x̂1    ← z_ref + (1 − τ_{k+1}) · u(z_ref, τ_{k+1}, h = 1 − τ_{k+1})   # exact terminal estimate
    X1*   ← argmin_X  ½ρ·τ_{k+1}²·‖X − x̂1‖²   s.t.  X ∈ C               # HardFlow prox NLP, unchanged
    z     ← z_ref + τ_{k+1} · (X1* − x̂1)                                  # HardFlow pull-back, unchanged
return X1*                                          # last solve has τ_K = 1 ⇒ output ∈ C exactly
```

Degenerate case **K=1**: sample noise → one exact terminal prediction (= iMF 1-NFE sample) → one projection. That is "constrained projection of the 1-NFE iMF sample" — the cheapest possible hard-constrained sampler and a mandatory baseline. K ≥ 2 differs from it in one specific, testable way: after each pull-back, the next `u`-evaluation **re-attracts the corrected point to the data manifold** (off-flow, `u` acts as a learned denoiser), so constraint corrections get re-naturalized instead of leaving projection artifacts. This gives a clean reading of the whole method family:

> **HardFlow = alternating (manifold re-attraction via the flow, constraint prox in terminal space).**
> **iMF makes the re-attraction step exact in one evaluation, turning a 20-step crutch into a 2–4-step algorithm.**

### Level 3 — multiple-shooting over anchor times (new formulation unlocked by the two-time field)

HardFlow explicitly abandoned putting the network inside the optimizer (the l4casadi bridge was "unnecessary overhead" — their README): with an instantaneous field, neural dynamics would enter the NLP at all N=20 fine steps. iMF's **two-time** field changes the economics: generation "dynamics" become K ≈ 3 big-step maps, admitting a direct multiple-shooting transcription:

```
variables:   z_1, …, z_K                                        (K × dof — small)
minimize     Σ_k ½ρ·τ_k²·‖z_k − ẑ_k‖²                           (proximity to unconstrained rollout)
subject to   z_{k+1} = z_k + (τ_{k+1}−τ_k)·u(z_k, τ_k, ·)        (shooting constraints, SQP-linearized)
             z_K ∈ C                                             (hard constraints on the terminal sample)
```

Each SQP outer iteration costs K NFE (values) + K JVPs (sensitivities `∂u/∂z` — free machinery in both JAX and the torch branch; it is the same JVP the training loss already uses). Unlike Levels 1–2 (greedy per-step corrections), this optimizes the whole generation path *jointly* — trading early-τ freedom against late-τ correction optimally instead of via the `τ²` heuristic. This is the genuinely-new-math part rather than substitution: **constrained sampling posed as a K-stage trajectory optimization whose stage dynamics are the learned average-velocity field.** It is intractable with an instantaneous field (K would be 20+); K=3 makes the NLP small.

---

## 4. A required training tweak, found by checking interval coverage — do not skip

Levels 1–3 query `u` on intervals **ending exactly at the data end** (`h = 1−τ`, i.e. iMF's `r = 0`) for *arbitrary* current times. But the official trainer (`imf.py:126-139`, `sample_tr`) draws `t, r` both **logit-normal** and takes max/min — logit-normal puts nearly zero mass at the boundary, so pairs `(t, r≈0)` are a thin slice of training. A stock iMF checkpoint is likely **under-trained on exactly the query the blend needs most.**

The fix is one line in `sample_tr`, mirroring the existing `data_proportion` mechanism (which pins `r = t` on half the batch to anchor `u ≈ v` at `h=0`):

```python
# after computing t, r, fm_mask:
term_mask = (jax.random.uniform(self.make_rng("gen"), (bz,1,1,1)) < self.term_proportion)
r = jnp.where(term_mask & ~fm_mask, 0.0, r)        # interval [0, t] — the data-end prediction query
```

Call it **terminal-anchored interval sampling** (suggest `term_proportion = 0.25`). It costs nothing — same JVP objective; the MeanFlow identity holds for every `(r,t)` pair — and it converts the u-head into a purpose-built x̂-predictor at every anchor time. Symmetry worth noting: `fm_mask` anchors the field at `h=0` (instantaneous end), `term_mask` anchors it at `h=t` (full-jump end); the blend is what makes the second anchor matter. Image-generation iMF never notices this gap because its sampler only jumps on a fixed grid; **constrained sampling is the application that exposes it.** This is a concrete, novel, cheaply-testable contribution that only becomes visible when composing the two papers.

---

## 5. What it takes to actually run this (both repos as-is)

The two codebases share the *benchmark* but not the *checkpoint*:

| Item | HardFlow | official iMF | consequence |
|---|---|---|---|
| Domain | avoiding trajectories, `[action(2)\|state(4)]`, H=8, dof=44 | ImageNet images, imfDiT, class labels | **iMF must be retrained on the trajectory domain** — there is no shared checkpoint |
| Framework | PyTorch (+ CasADi) | JAX (torch port on the repo's `torch` branch; `models/torch_models.py` here is flax-with-torch-init, *not* a port) | use the `torch` branch as reference, or bridge via numpy at the two call-sites (HardFlow already round-trips through numpy for IPOPT — `flow_eval_np`) |
| Time convention | τ=0 noise → 1 data | t=0 data → 1 noise | §1.3 wrapper, fixed once |
| Conditioning | inpainting (`apply_conditioning`, cond={0: s0}) | class labels + CFG (ω, t_min, t_max as network inputs) | for trajectories: drop labels (unconditional net) + keep HardFlow's inpainting; ω-conditioning optional (train with ω=1 or keep the interval-CFG machinery — it is orthogonal) |
| Backbone | UNet1D (`hardflow/models_flow/unet.py`) | imfDiT | the iMF *recipe* (logit-normal (t,r), predicted-v JVP, adaptive weighting, dual head) is backbone-agnostic; an (8×6) trajectory needs a small 1D net, not a DiT-XL |

So the honest shape of the project: **train an iMF-recipe model on HardFlow's own avoiding dataset** (their `hardflow/datasets` + normalizer, their inpainting conditioning), then patch two call-sites in `hardflow_new_forward`. The recipe transfers; the weights don't.

### Implementation plan

1. **iMF-for-trajectories trainer** inside the HardFlow repo (new `hardflow/models_flow/imf_matcher.py`, sibling of `flow_matcher.py`): official loss (`imf.py:331-401`) transcribed to PyTorch (`torch.func.jvp`), on HardFlow's UNet1D extended with an `h` (and optional ω) embedding and a dual `(u, v)` head, **plus the §4 terminal-anchoring flag**. Convention: adopt HardFlow's τ (data-at-1) so no flip is needed at inference; flip the identity's sign once here instead.
2. **Sanity gate**: 1-NFE reconstruction on held-out avoiding trajectories (`x̂1 = z + 1·u(z, 0, h=1)` from pure noise, compare distributions), and `u(·,·,h→0) ≈ v` against the co-trained v-head. Do not proceed while either fails — every downstream symptom would be ambiguous.
3. **Patch `hardflow_new_forward`**: new `guidance_method='hardflow_imf'` = same function with the two `flow_eval_np` sites swapped to the u-head (§3 Level 1) and `ode_t_steps` reinterpreted as K.
4. **Eval grid** on their existing pipeline (`run_scripts/`, `trajectories.csv`): HardFlow N=20 (paper baseline) · iMF-HF K∈{1,2,4} · Level-0 sanity (iMF v-head inside stock HardFlow, N=20) · ablation ±terminal-anchoring. Metrics: their success/violation rates + wall-time and NFE.
5. **Level 3 shooting** only if Level 2 shows the expected win; it reuses the same checkpoint and adds one SQP loop (CasADi callback or plain torch SQP).

(Our FM-PCC repo has a local trajectory-iMF port that could prototype step 1 faster, but nothing above depends on it.)

## 6. Compute accounting

| Method | NFE / action | IPOPT solves | terminal-estimate bias |
|---|---|---|---|
| HardFlow (N=20, defaults) | ~60 (warm-start 20 + 2×20) | 20 | ½L(1−τ)² — worst at small τ |
| iMF-HF, K=4 | 8 | 4 | ε_u (τ-uniform) |
| iMF-HF, K=2 | 4 | 2 | ε_u |
| iMF-project, K=1 | 2 | 1 | ε_u (no re-naturalization) |
| Level 3 shooting, K=3, ~5 SQP iters | ~30 (15 evals + 15 JVPs) | 1 (larger) | ε_u, jointly optimized |

The IPOPT solves dominate HardFlow's wall-time (44-dof NLP, ~50 obstacle constraints, per ODE step); cutting 20 → 2–4 solves matters more in practice than the NFE cut.

## 7. Honest caveats

1. **ε_u is not zero.** The argument trades a *structural* bias for a *learned* one. iMF's ImageNet results (FID 1.70 @ 1 NFE) are strong evidence ε_u can be small, but trajectories ≠ images; the sanity gate in §5.2 is the checkpoint's burden of proof.
2. **Off-manifold queries.** After a pull-back, `z'` is slightly off the learned flow; `u` was trained on-path. This is the same situation HardFlow's `v` already faces (their method works regardless), but it is the residual failure mode under aggressive corrections — and the honest reason K=2–4 may beat K=1.
3. **Coverage** — addressed by terminal-anchored sampling (§4). Without it, expect degraded `x̂1` at small τ and don't blame the algorithm.
4. **The NLP is unchanged, so is its failure mode**: IPOPT returning locally-infeasible iterates on the nonconvex obstacle constraints. HardFlow already falls back to the last iterate (`flow_policy.py:1354-1358`); nothing new.
5. Level 3 is a proposal with a plausible cost model, not verified code. Levels 0–2 are call-site-level patches of an existing, working codebase plus one retraining run.
6. The convention flip (§1.3) is trivial and lethal; it gets a dedicated unit test before anything else runs.

---

*Load-bearing code, for whoever picks this up:*
`HardFlow/hardflow/models_flow/flow_policy.py:1286-1435` (`hardflow_new_forward` — the seam is :1324-1325 and :1339-1340), `:683-751` (NLP formulation, τ² cost), `:227-245` (`x1_estimate`), `:350-475` (dynamics/box constraints), `:753-795` (warm-start);
`imeanflow/imf.py:90-114` (exact-jump sampler), `:126-139` (`sample_tr` — §4 patches here), `:185-210` (`u_fn` dual head), `:331-401` (training loss: predicted-v JVP :373, compound target :376, adaptive weighting :380);
`imeanflow` `torch` branch — PyTorch reference for step 1 of the plan.
