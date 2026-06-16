# U4 Fix 1 — Re-audit + official-repo cross-check (why results are still poor)

**Date:** 2026-06-15
**Audited:** [`./CHANGELOG.md`](./CHANGELOG.md) (the crash fix) + the `meanflow_jvp` objective
**Cross-checked against:** `/workspaces/imeanflow` **`torch` branch** (official PyTorch *inference*
re-impl; README confirms **training is JAX-only**, not shipped here).

---

## A. The crash fix itself — verdict: correct & complete

`train_flow_matching_v3_imeanflow.py` Fix 1 (getattr for `ode_inference_steps_v3` + forwarding the
5 imfv2 params) is **right**, and the param list is complete — the model's other ctor args
(`flow_steps_v3`, `time_beta_*`, loss weights) were already forwarded. Training now runs to
completion, which is consistent with "trained, but results not good." **No further crash-side fix
needed.**

**Two lines in CHANGELOG.md are now stale and should not be trusted:**
- It inherits the framing "prime suspect = JVP sign / h-tangent." **The sign is verified** (numpy
  finite-diff, exact to 1.9e-6; see [U4 CHANGELOG §6.0](../CHANGELOG.md)). Stop blaming the sign.
- "Next run … watch for forward-AD/JVP errors" is fine, but the *quality* problem below is not an
  error-throwing bug — it trains cleanly and still underperforms.

---

## B. Cross-check vs official `imf.py` — the convention mirrors, no sign bug

| | Official (`imf.py`) | Gen3v4 |
|---|---|---|
| noise side | τ=1 | τ=0 |
| sampler start | `z` at τ=1 | `z` at τ=0 |
| step | `z ← z − (t−r)·u`, t:1→0 | `x ← x + u·dt`, τ:0→1 |
| network time-arg | **t** (current, larger) | **r** (current, smaller) |
| h argument | `t−r` | `t−r` (= `dt`) |

These are exact mirror images under τ ↔ 1−τ. Gen3v4 is **internally consistent** (training *and*
sampler both anchor at the current point `r` with `h=t−r` and use `v_inst = data−noise`), so the
mirror does **not** introduce a sign error. The numpy check already confirmed the identity
`u = v_inst + h·du/dr` with tangents `(v_inst, +1, −1)` for exactly this "anchor-at-r" convention.
**The objective is faithful.** So the poor result is not the math — it's coverage. ⇩

---

## C. Leading suspect — the training (t, r) schedule starves the few-step regime

Gen3v4 keeps the **FM time schedule**: `t = 1 − Beta(1.5, 1.0)`, `r ~ U(0,t)` (+25% `r=t`). That
schedule concentrates the interval **near the noise side with small width**. Measured over 2e5
draws:

```
t (data-side endpoint):  t∈[.75,1] = 12.5%,   t≥0.9 = 3.1%
h (interval width):      h≥0.5 = 9.2%,         h≥0.75 = 1.0%
```

So the model is trained almost entirely on **short, noise-side intervals** and **almost never** on
the long / near-data intervals that few-step iMF integrates through.

**Why this is fatal for `meanflow_jvp` but harmless for `fm_equivalent`:**
- `fm_equivalent`'s target is the **constant** `v = data − noise`, identical at every `(r,t)`. Sparse
  coverage near τ=1 is fine — the network just generalizes the same constant. That's why the FM arm
  "works" at 10 NFE.
- `meanflow_jvp`'s target is **bootstrapped and non-constant**: `v_inst + h·du/dr`, where `du/dr` is
  the network's *own* curvature. In undertrained `(r,t)` regions `du/dr` is garbage → the target is
  garbage → it poisons exactly the long-interval / end-of-trajectory predictions that sampling
  relies on. Few-step (and even the last steps of 10-NFE) land on these starved regions.

This is the single most likely reason training completes but quality is poor.

---

## D. Secondary gaps vs the official model (smaller, real)

1. **No interval-CFG.** The official `u_fn` conditions on `(omega, t_min, t_max)` and the README's
   own eval commands rely on it (`--cfg-omega 8–10.5`, `--interval-min/max ≈ 0.4–0.6`). Gen3v4 has
   none. This is the documented Phase 3; it materially helps low-NFE quality.
2. **Dropout (0.1) inside the JVP.** The bootstrapped target is differentiated through a stochastic
   net; dropout injects noise into `du/dr`. Consider dropout→0 for the `meanflow_jvp` arm (or at
   least know it adds target variance).

---

## E. Recommended next fix (Fix 2 candidate) — re-balance the interval schedule

Before chasing anything else, **change the `(t, r)` sampling for the `meanflow_jvp` arm only** so it
covers long, near-data intervals:
- sample `t` ~ U(0,1) (or skew *toward* data), not `1−Beta(1.5,1)`;
- keep `r ~ U(0,t)` + the `r=t` anchor, but ensure a healthy fraction of large `h` (e.g. mix in
  `r ~ U(0,t)` with occasional `r=0` so `h≈t≈1`).
This is a few-line, flag-gated change in `_p_losses_meanflow_jvp` (leave `fm_equivalent` untouched).

**Verify by:** re-run, then the **1-NFE and 4-NFE reconstruction RMS** — coverage fix should move
both sharply. Only after that is the schedule eliminated should interval-CFG (D.1) be added.

> Net: the crash fix is correct; the *objective* is mathematically faithful; the poor result is a
> **data-coverage / schedule** problem, not a sign or wiring bug.

---

## F. Schedule lean — confirmed against the source-of-truth repo (`/workspaces/SafeFlowMPC`)

The `t = 1 − Beta(1.5, 1.0)` schedule did not originate in FM-PCC; it was ported from
**`SafeFlowMPC/train_imitation_learning.py:85-103`**. Checking the origin removes all ambiguity:

```python
x_1 = trajs                  # data        x_0 = randn   # noise
t = Beta(1.5, 1.0).sample(); t = 1 - t
path = AffineProbPath(CondOTScheduler())   # x_t = t·x_1 + (1-t)·x_0  ⇒ t=1 data, t=0 noise
# t = torch.rand(...)        # ← uniform alternative, left commented out
```

- **Convention:** `t=1` = data, `t=0` = noise (identical to Gen3v4). ✓ The port is faithful.
- **Lean direction = NOISE, not data.** `Beta(1.5,1)` mean = `0.6`; `t = 1−Beta` mean = `0.4` ⇒ the
  average sample is `0.4·data + 0.6·noise`. So the established FM schedule deliberately leans toward
  the **noise** side (and the repo even shows plain uniform `t=torch.rand` as the considered
  alternative). This triple-confirms §C: the lean is real and noise-ward, fine for FM, wrong for iMF.

## G. The schedule fix is config-only — `Beta(1,1) ≡ Uniform(0,1)`

§E proposed a code change, but it is **not required** to get uniform `t`: `Beta(1,1)` is *exactly*
`Uniform(0,1)`, and `1 − Uniform = Uniform`. So uniform coverage is reachable by editing two config
keys, no code:

```python
# config/avoiding-d3il.py  (flow_matching_v3_imeanflow block, ~L499-500)
'time_beta_alpha_v3': 1.0,   # was 1.5  → Beta(1,1) = Uniform(0,1)
'time_beta_beta_v3':  1.0,
# config/aligning-d3il-visual.py — same two keys, add as explicit overrides in imf_visual_aligning
```

Effect: data-side coverage `P(t≥0.9)` rises `3.1% → 10%`; long intervals get exercised.

**Caveat:** `time_beta_*` feeds `loss()` for **both** objectives in the block, so `1.0/1.0` also
moves the `fm_equivalent` baseline to uniform. That is fine (equal-footing A/B). Only if FM must
*keep* `1.5/1.0` while iMF goes uniform is the §E code branch needed — otherwise the config edit
supersedes §E as the simplest Fix 2. After uniform `t`, the remaining lever is widening `h`
(occasionally forcing `r≈0` so `h≈t`), which still needs the small code change.
