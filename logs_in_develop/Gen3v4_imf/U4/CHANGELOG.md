# U4 — CHANGELOG: imfv2 (flag-gated MeanFlow-JVP objective) implemented

**Date:** 2026-06-13
**Branch:** `update_into_FM`
**Implements:** [`U4/PLAN_Unleash_Full_iMF.md`](./PLAN_Unleash_Full_iMF.md) — Phase 1 (objective),
Phase 2 (low-NFE inference path), §2A (keep-vs-drop = flag-gated imfv2).
**Status:** Code complete, **untested** (Docker = AI-coding only; no Python runtime here). The
correctness gate is a cluster 1-NFE reconstruction check — see §6.

---

## 1. What changed (summary)

Replaced *nothing destructively*. Added a **second training objective behind a flag**, exactly
per §2A "keep, don't drop":

```
imf_objective: 'fm_equivalent'   # DEFAULT — legacy finite-diff target (unchanged FM baseline arm)
             | 'meanflow_jvp'     # NEW — real MeanFlow Identity via forward-mode JVP (imfv2)
```

Default is `fm_equivalent`, so **every existing run is byte-for-byte unaffected**. imfv2 is opted
into by one config key.

---

## 2. Files touched

| File | Change |
|------|--------|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | `__init__`: 5 new params (`imf_objective`, `meanflow_r_equals_t_frac`, `meanflow_adaptive_p`, `meanflow_adaptive_c`, `meanflow_aux_weight`) + storage. `p_losses`: dispatch on `imf_objective`. **New** `_p_losses_meanflow_jvp(...)`. |
| `config/avoiding-d3il.py` (`flow_matching_v3_imeanflow` block) | Added `imf_objective: 'fm_equivalent'` (default) + the 4 meanflow knobs, with switch instructions. |

The legacy `_p_losses` body, `_predict_uv`, `_predict_velocity`, `p_sample_loop`, sampler, and
serialization are **untouched**. (`p_sample_loop` already accepts low `flow_steps`, so Phase 2
needs only a config change — no code.)

---

## 3. The objective that was implemented (`_p_losses_meanflow_jvp`)

In Gen3v4's DATA-AT-1 convention (τ=0 noise, τ=1 data; sampler anchors at the noise-side point
`z_r` at time `r` and steps forward to `t=r+h`), the **MeanFlow Identity** derived for the
START-anchored query is:

```
(t − r)·u(z_r, r, t) = z_t − z_r
  ⇒ differentiate w.r.t. r at FIXED t, along the trajectory (dz_r/dr = v):
  u(z_r, r, t) = v(z_r, r) + (t − r) · d/dr u
              = v_inst       + h · JVP[u ; tangents (∂z=v_inst, ∂time_r=+1, ∂h=−1)]
```

Implementation steps:
1. `r ~ U(0,t)`, with **25%** of the batch forced to `r=t` (the FM anchor; at `h=0` target = `v_inst`).
2. `z_r = q_sample(x_start, r, noise=ε)`, conditioned.
3. `v_inst = x_start − ε` (conditioned, noise-masked) — serves as both the identity's `v(z_r,r)`
   and the JVP z-tangent.
4. `u_pred, du_dr = torch.func.jvp(u_head, (z_r, r, h), (v_inst, +1, −1))`.
5. `u_target = (v_inst + h·du_dr).detach()` — **stop-gradient** (regress to it; never backprop
   through the JVP — that would need 2nd-order grads).
6. Loss = adaptive-weighted, `loss_weights`-scaled MSE:
   `sq = (u_pred−u_target)² · loss_fn.weights`; `w = (mean(sq).detach() + c)^(−p)`;
   `loss = mean(w · mean(sq))`.
7. Optional aux v-head MSE on `v_inst` (gated by `meanflow_aux_weight`, default 0 = off).

This is the part Gen3v4 never had (U3 §3/§10): the `d/dr u` (JVP) term that makes `u` a *true*
average velocity instead of collapsing to the FM velocity.

---

## 4. Why the sign differs from the official repo

The official `imf.py` uses `u = v − (t−r)·d/dt u`, anchoring at the **larger-time** endpoint and
differentiating w.r.t. that endpoint (noise-at-1 convention). Gen3v4 integrates **forward** and
the sampler anchors at the **smaller-time / noise-side** point `z_r`, so the correct, derivation-
matched identity is `u = v + h·d/dr u` with the `h`-tangent **−1** (since `h=t−r`, `dh/dr=−1`).
The `+` sign and the `−1` h-tangent are the two things to flip first if the cluster check fails.

---

## 5. How to run an imfv2 experiment

In `config/avoiding-d3il.py` `flow_matching_v3_imeanflow` block:
```python
'imf_objective': 'meanflow_jvp',     # turn on real iMF
'ode_inference_steps_v3': 1,         # or 2 — few-step is the whole point (Phase 2)
# optional: 'meanflow_aux_weight': 0.05  to enable the aux stabilizer
```
Train as usual; then re-tune the DPCC projector snap schedule for low NFE (Phase 4 — **still TODO**,
required before trusting avoiding constraint satisfaction) and A/B vs `fm_equivalent` @10 NFE.

---

## 6. Verification status

### 6.0 Done locally — JVP sign/tangent derivation ✅ (2026-06-14)

A pure-numpy finite-difference check (`/tmp/jvp_sign_check.py`) validated the §3/§4 derivation on a
**non-linear** velocity field (linear interpolant collapses `du/dr→0` and can't catch a sign error):

```
IMPLEMENTED  u = v_inst + h·du/dr   max|err| = 1.86e-06   (integration-error floor) ✅
WRONG SIGN   u = v_inst − h·du/dr   max|err| = 3.42e-01    (184,000× worse)
```

⇒ the implemented sign `+` and the `(v_inst, +1, −1)` tangent are **mathematically correct**. The
"flip the sign first" advice below is therefore a *fallback*, not the expected outcome — if 1-NFE
diverges on the cluster, the cause is more likely forward-AD/encoder behaviour (§6.2) than the sign.

### 6.1+ REQUIRED on the cluster (cannot be done here — no torch/GPU)

This objective is numerically delicate and **was not executed** (no local runtime). Before
trusting any imfv2 number, run on the cluster:

1. **1-NFE reconstruction** — train briefly with `imf_objective='meanflow_jvp'`, sample at
   `ode_inference_steps_v3=1`, measure RMS vs ground-truth trajectories. Low RMS ⇒ the JVP +
   sign are correct (now corroborated by §6.0). High/diverging RMS ⇒ check forward-AD (§6.2)
   before flipping the sign (§4), since the sign is independently verified.
2. **(§6.2) forward-mode AD support** — confirm `torch.func.jvp` runs through the temporal UNet without
   erroring (some ops lack forward-AD rules; condition-dropout randomness inside the JVP is the
   other suspect — set `condition_dropout` low or deterministic for the imfv2 run if it errors).
3. **A/B** — `meanflow_jvp`@1–2 NFE vs `fm_equivalent`@10 NFE on quality **and** `fm_ms`. Success
   = match quality at a fraction of latency (U3 §5).

---

## 7. Not done (deferred, per plan)

- **Phase 3** (improved-iMF interval CFG) — not implemented; optional/advanced.
- **Phase 4** (DPCC projector low-NFE re-tune) — **required before shipping** on avoiding; the
  snap schedule is NFE-coupled (U3 §9.3). Left as the next coding task.
- No retrain/eval run (cluster).

---

## Caveats

- Default behaviour unchanged (`fm_equivalent`); zero risk to existing runs.
- The `meanflow_jvp` path is **unverified** — treat §6 as a hard gate.
- Same objective is mirrored in the Gen8 visual fork — see
  [`../../Gen8/Epoch_1/U3/CHANGELOG.md`](../../Gen8/Epoch_1/U3/CHANGELOG.md).
