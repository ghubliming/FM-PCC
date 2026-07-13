# Investigation — after CFG-off: is Gen3v4 even *real* iMeanFlow? (vanilla vs improved MeanFlow fidelity audit)

**Gen:** Gen3v4_imf / U10 · **Date:** 2026-07-13
**Follows:** `INVESTIGATION_new_vs_upstreams_KILL_TABLE.md` (the CFG kill chain) and its config-override-pkl fix.
**Upstream reference:** `/workspaces/aux_repo/imeanflow/imf.py` (`iMeanFlow` class).

> **Reference convention:** pointers are `file · function/logic`, not line numbers (lines rot). Durable anchor = the named function + the described math.

---

## 1. Cluster result (visual inspection) — the kill chain is CONFIRMED

Ran on the cluster: **DiT backbone (`bbdit`), 10-step Euler, CFG OFF** (`condition_guidance_w=0`, `meanflow_cfg_omega=0`, `returns_condition=False`, reaching the model via the config-override-pkl fix).

- ✅ **No more exploding.** Closing gates #1/#2 stopped the blow-up → kill-table rows 1/2/3 were the cause; the `bbdit` checkpoint itself was healthy. The sampler was poisoning a fine model.
- ❌ **Still not smooth.** Bounded but jittery — **loses to the old UNet-FM (Gen7 FMv3ODE) and even DPCC.**

So: **correctness problem solved, quality problem remains.** The question below is the quality one.

## 2. The question (user)

> "You told me to check the imeanflow repo — and now you're saying Gen3v4 is *still not real iMeanFlow*? What is lacking / deviated? We must be faithful to iMeanFlow."

Correct. **Despite the name, the Gen3v4 training objective is *vanilla* MeanFlow, not *improved* MeanFlow (iMF).** The single change that *defines* iMF over MeanFlow was reverted. Details below.

## 3. First, a correction to be faithful — the SAMPLER is fine

My earlier "10-step = you're running it like FM" was imprecise. Official iMF's `sample_one_step` also steps with `h = t−r = 1/N` and a single `u`-call per step — **identical** to Gen3v4's `iMeanFlowODE.p_sample_loop` (`x += dt·u(x,t,h=dt)`). So the sampler is **faithful**. Running 10 steps is a *usage* choice: iMF's advantage is **1–2 NFE**; at 10 steps a perfect iMF can at best *tie* FM. This is not a deviation — the deviations are all in **training**.

## 4. The aha — why it's rough and loses to FM

iMF's headline improvement over MeanFlow (verbatim, `imf.py` · `forward`): *"Different from original MeanFlow, we use predicted v in the jvp."* The MeanFlow-Identity target contains a total derivative `d/d·[u]`, computed by a JVP whose **z-tangent is the instantaneous velocity along the trajectory**. Two choices:

- **Original MeanFlow:** tangent = the **analytic per-sample** velocity `v = x_data − noise` — *high variance* (a single noisy sample of the marginal velocity).
- **improved MeanFlow (iMF):** tangent = the network's **predicted `v_c`** (from the v-head / `guidance_fn`) — *low variance*.

Why it decides smoothness: the JVP target is `u_target = v + h·du_d·`, and `du_d·` contains `∂_z u · v_tangent` — the **Jacobian × the tangent**. Feeding the noisy analytic `v` there injects Jacobian-amplified label noise into the (stop-gradient) regression target. That is *strictly more* label noise than FM's plain `v = x−noise` target. Bumpy targets → bumpy learned average-velocity field → **non-smooth few-step samples**. UNet-FM regresses to a clean low-variance `v` → smoother field → it wins.

**Gen3v4 uses the ANALYTIC `v_inst = x_start − x_base` as the JVP tangent** (`iMeanFlowODE._p_losses_meanflow_jvp`), i.e. **original MeanFlow**, not iMF — even though the v-head *exists* (`dual_head=True`, `meanflow_aux_weight=0.05`) and could supply the low-variance `v_c`. The one ingredient that makes iMF smooth is present but **not wired into the JVP**.

Net: Gen3v4-iMF = *MeanFlow's harder, higher-variance objective* − *iMF's variance-reduction fix* − *few-step training coverage* (§5, dev 3). It is dominated by plain FM on every axis → rough, loses. Not a tuning gap.

## 5. Deviation audit — `imf.py` (`forward`) vs Gen3v4 (`_p_losses_meanflow_jvp` / `p_sample_loop`)

| # | Component | Official iMF (`imf.py`) | Gen3v4 (ours) | Severity for quality/fidelity |
|---|---|---|---|---|
| **D1** | **JVP z-tangent** | **predicted `v_c`** — `jax.jvp(u_fn,(z_t,t,r),(v_c,1,0))` (the defining iMF change) | **analytic `v_inst = x_start − x_base`** in `_p_losses_meanflow_jvp` | **CRITICAL** — *this is what makes it vanilla MeanFlow.* High-variance target → rough field → the #1 reason it loses to FM |
| **D2** | **Regression target / CFG** | regress `V` → **guided `v_g`** from `guidance_fn`; null token trained by `cond_drop` (10%/batch); `v_fn`/`v_cond_fn` build `v_c`,`v_u` | target = **unguided** `v_inst + h·du_dr`; `guidance_fn`/`v_fn`/`cond_drop` **not ported**; null token gets zero gradient | **CRITICAL** — the kill-table explosion cause; also leaves ω-conditioning **dead** (net trained to ignore ω) |
| **D3** | **(t,r) sampling** | `sample_tr`: **two INDEPENDENT logit-normals**, `t=max,r=min`; `data_proportion=0.5` → **50% forced r=t** (FM anchors) | `r = t·U(0,1)` (so **h ≤ t always**); only **25%** anchors (`meanflow_r_equals_t_frac=0.25`) | **HIGH** — narrower `(t,h)` coverage, weaker FM grounding, worse few-step (1-NFE queries `h≈1` are undertrained) |
| **D4** | **v-head role** | v-head output **IS `v_c`**, fed into the JVP tangent (D1) AND trained via `loss_v` toward `v_g` | v-head is a **side stabilizer only** (regressed to `v_inst`), **never fed into the JVP** | **HIGH** — same root as D1; the wiring `v_head → tangent` is missing |
| **D5** | **Loss composition** | `loss = loss_u + loss_v`, both → `v_g`; adaptive `w=(loss+eps)^p` with **p=1.0, eps=0.01**, per-sample **SUM** over dims | `u`-loss × **DPCC per-dim `loss_weights`** (action_weight=10, discount); adaptive **p=0.5, c=1e-3**, per-sample **MEAN**; aux `0.05·‖v−v_inst‖²` | **MED** — different gradient geometry; DPCC weighting injected (not in iMF) |
| **D6** | **DiT sizing** | presets `imfDiT_B/M/L/XL` (hidden 768+, depth 12+, heads 12+, tokens 4/4/8) | toy: hidden 256, depth 8, 4 heads, tokens 2/2/1 (`config dit_*`) | LOW–MED — capacity; fine for H=8 but not comparable to paper |
| **F1** | **Sampler** | `sample_one_step`: `z − (t−r)·u`, one `u`-call, `h=1/N` | `p_sample_loop`: `x += dt·u`, `h=dt=1/N` | **FAITHFUL** ✔ |
| **F2** | time-flip (DATA-AT-1), real-RoPE, V-form↔residual-form loss algebra | — | re-derived / re-implemented | **FAITHFUL** ✔ (verified in kill-table rows 12/16/17) |

**Bottom line of the audit:** the deviations that make it *worse than FM* are **D1 + D3 + D4** (rough, ungrounded field); **D2** adds the CFG breakage (now disabled) and dead ω-conditioning; **D5/D6** are secondary. **D1 is the one that means "not real iMF."**

## 6. The decisive test (confirms the mechanism, no retrain — cluster)

Run the **same `bbdit` checkpoint** at **N = 1, 10, 50** steps:
- **N=50 ≈ UNet-FM, N=10 < FM, N=1 garbage** → field is *unbiased but rough / undertrained-at-large-h* → confirms D1+D3 (variance + coverage). The fix is the objective, not knobs.
- **N=50 still < FM** → field is *biased* (JVP sign/derivation) — but that was re-derived and checked (F2), so I expect the former.

## 7. Faithful-iMF fix list (ranked; D1 first)

1. **D1 — feed predicted `v_c` into the JVP z-tangent.** *The* change that turns this from MeanFlow into iMF. In `_p_losses_meanflow_jvp`, obtain `v_c` from the v-head and pass it as the tangent instead of `v_inst`. Retrain. **Highest expected smoothness gain.**
2. **D3 — `sample_tr` fidelity:** two independent logit-normals (`t=max,r=min`) + `data_proportion=0.5` (50% FM anchors), replacing `r=t·U(0,1)` + 25%. Broadens `(t,h)` coverage and grounds the field.
3. **D2 — CFG: either port it properly or drop it.** Faithful path: port `guidance_fn`+`v_fn`+`cond_drop` (guided `v_g` target + trained null token) and delete the sampling-time output-mix. Pragmatic path for a clean baseline: `interval_cfg=False`, retrain — removes the dead ω-conditioning entirely.
4. **D5 — match the loss:** `loss_u+loss_v` both → `v_g`; adaptive `p=1.0,eps=0.01`, sum; reconsider whether the DPCC `loss_weights` injection belongs in a MeanFlow objective.
5. Only after D1–D3: revisit DiT sizing (D6) if still short of the paper.

**Recommendation:** do the §6 test first (cheap, confirms the story), then implement **D1 (+D3)** as one retrain — that is the minimum change to make Gen3v4 *actually* improved-MeanFlow. If the goal is 10-step generation and D1/D3 don't close the gap to UNet-FM, the honest conclusion is that FM is the right tool at 10 steps and iMF only pays off at 1–2 NFE.

---

## 8. "Can we write: *we faithfully use iMF in FM-PCC*?" — **NO** (honest verdict + defensible wording)

**Direct answer: no.** As of this checkpoint, a claim of *faithful iMF replication* would be **inaccurate**. What FM-PCC contains is the **iMF DiT architecture + faithful sampler**, wrapped around an objective that is **original MeanFlow (not improved), with CFG half-ported and DPCC loss-weighting injected**, reimplemented in PyTorch. Faithful in *architecture*, not in *method*.

### Faithfulness by axis (be precise — "iMF" is several things)

| Axis | Faithful to `imeanflow/imf.py`? | Why |
|---|---|---|
| **Few-step sampler** (`z − h·u`, one `u`-call, `h=1/N`) | ✅ **Yes** | matches `sample_one_step` (F1) |
| **DiT backbone** (RoPE, QK-RMSNorm, SwiGLU, dual u/v heads, zero-init gates, in-context tokens) | ⚠️ **Mostly** | faithful port (F2), but **toy-sized** vs paper presets (D6); real-valued RoPE reimpl (verified equivalent). *Only the `bbdit` arm — see below.* |
| **Training objective — the JVP itself** | ❌ **No** | **D1: analytic `v_inst` tangent, not predicted `v_c`** ⇒ this is **original MeanFlow**, i.e. *not the "improved" in improved-MeanFlow* |
| **CFG** (`guidance_fn`, `v_fn`, `cond_drop`, guided `v_g` target) | ❌ **No** | **D2:** not ported; null token untrained; ω-conditioning dead |
| **(t,r) time sampling** | ❌ **No** | **D3:** `r=t·U(0,1)` + 25% anchors vs two independent logit-normals + 50% |
| **Loss / adaptive weight** | ❌ **No** | **D5:** DPCC per-dim `loss_weights` + `p=0.5`/mean vs `p=1.0`/sum, `loss_u+loss_v→v_g` |
| **Backbone actually used if `imf_backbone='unet'`** | ❌ **Not iMF at all** | the UNet arm is the FM UNet (`Flow_matcher_U_Net_v2`); only `bbdit` is architecturally iMF |
| **Implementation** | — | **PyTorch reimplementation**, not the JAX/Flax repo code |
| **Domain** | — | images/VAE-latents → **trajectories** (a necessary, discloseable adaptation, not an infidelity) |

### The disqualifier
A claim of "faithful iMF" hinges on the **method**, and the method's *defining* feature — **predicted-`v` in the JVP (D1)** — is exactly what was reverted. With D1 reverted, **the objective that trains is original MeanFlow**, so "we use improved-MeanFlow" is not currently true. D2/D3/D5 compound it.

### What you CAN honestly write (defensible today)
- *"We adapt the improved-MeanFlow (iMF) **DiT architecture** and its **few-step sampler** to trajectory generation, reimplemented in PyTorch within the DPCC pipeline."*
- *"Our engine is **iMF-inspired**: it uses the iMF backbone and MeanFlow-Identity training, with deviations (JVP tangent, CFG, time sampling) documented in [this file]."*
- *"a **MeanFlow-family** trajectory engine."*

### What you must NOT write (false as stated)
- ✗ *"we faithfully use / replicate iMeanFlow"*
- ✗ *"we use the iMF objective / improved-MeanFlow training"* — currently it is **original MeanFlow**
- ✗ any "iMF" claim for the **`unet` backbone** runs (that arm is not iMF)

### What would MAKE the claim true
Implement **D1** (feed predicted `v_c` into the JVP tangent) — then "we use the improved-MeanFlow objective" becomes accurate. Add **D3** (faithful `sample_tr`) and either port or drop CFG (**D2**) for a clean claim of *faithful iMF training*. Until D1 lands, keep the wording to "iMF-**inspired** / iMF **architecture**," not "faithful iMF."

---

*Awaiting instruction before implementing any of the above.*
