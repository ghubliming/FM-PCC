# U4 — PLAN: Unleash the FULL iMF Power in Gen3v4

**Date:** 2026-06-13
**Scope:** `flow_matcher_v3_imeanflow/` (Gen3v4), reusing `/workspaces/imeanflow` (official iMF).
**Status:** Plan / principle only — **no production code in this document.** Pseudocode is
logic-level illustration; the implementing agent writes the real code.
**Predecessor:** [`../U3/iMF_vs_FM_Math_Principle.md`](../U3/iMF_vs_FM_Math_Principle.md) — §10 has
the first-principles diagnosis and the objective sketch this plan operationalises.

---

## 0. The one-sentence problem and fix

> Gen3v4 today is **iMF body + FM brain**: it faithfully ports iMF's *architecture* and
> *inference* (§8 of U3) but trains an **FM-equivalent objective** (finite-difference target that
> collapses to `x₁−ε`). **To unleash full iMF we replace the brain — the training objective —
> with the JVP-based MeanFlow Identity, and switch inference to low NFE.** The body stays.

---

## 1. What we can REUSE directly from `/workspaces/imeanflow`

The official repo is **inference-only** (`imf.py:29`; README: *"For training code, refer to the
JAX implementation"*). So we reuse its **inference + architecture logic**, and **build** the
training. Reuse map:

| Asset to reuse | Official location | How it maps into Gen3v4 |
|---|---|---|
| **One-step / few-step update** `z ← z − (t−r)·u` | `imf.py:71-95` (`sample_one_step`), `imf.py:97-138` (`generate`) | Replaces the 10-step Euler in `imf_diffusion.py:p_sample_loop`; logic identical up to DATA-AT-1 sign |
| **u-only at inference** (`u_fn(...)[0]`, v discarded) | `imf.py:93,135` | Already done in Gen3v4 (`_predict_velocity`, "FIX-3") — keep |
| **Dual-head split** (shared backbone → `u_heads` + `v_heads`, v dropped at eval) | `imfDiT.py:264-295, 374-390` | Gen3v4 already has u + aux head; confirm the split depth mirrors `aux_head_depth` |
| **h-only conditioning** (condition on `h=t−r`, NOT on `t`) | `imfDiT.py:370-371` (cites arXiv 2502.13129) | **Divergence to resolve** — Gen3v4 conditions on *both* `t` and `h` (see Phase 0) |
| **Improved-iMF guidance** (CFG via `omega`, interval `[t_min,t_max]`) | `imfDiT.py:223-246, 308-351`; `imf.py:44-69` | Maps to Gen3v4's returns-guidance; **optional advanced phase** (Phase 3) |
| **Time/interval embedders, in-context conditioning tokens** | `embedder.py`, `imfDiT.py:_build_sequence` | Reference design only — Gen3v4 uses a 1D temporal UNet, not a DiT, so port the *pattern* not the module |

> **Key takeaway:** the repo gives us a *verified reference* for inference and architecture, so
> we don't have to guess those. The **JVP training objective is the only genuinely new code**,
> and its specification comes from the MeanFlow paper / JAX repo, transcribed in §3 below.

---

## 2. What we must BUILD (absent from the PyTorch repo)

1. **JVP-based MeanFlow-Identity training target** (the heart).
2. **Stop-gradient discipline** on that target.
3. **Mixed `(t, r)` sampling** with a real fraction at `r = t`.
4. **Adaptive loss weighting** to stabilise the bootstrapped objective.
5. **Low-NFE inference** path (1–2 steps) wired to the config.
6. **DPCC projector schedule** re-derived for low NFE (avoiding-specific).

---

## 2A. Keep vs drop the old iMF code — flag-gated "imfv2" (code disposition)

**Decision: KEEP everything; replace exactly one method; gate old-vs-new with a config flag.**

The old code is **not broken** — it produces valid FM-quality output. It is *mislabeled*: an
FM model wearing correctly-ported iMF scaffolding (verified against `/workspaces/imeanflow`,
U3 §8). The only **wrong** piece is the `p_losses` target. So this is a **surgical objective
swap, not a rewrite, and not a new folder/generation.**

### What is wrong vs reusable

| Piece | Status | Action |
|---|---|---|
| `p_losses` finite-diff target (collapses to FM) | the FM-brain | **Replace** (the only wrong part) |
| Dual-head (u + aux) network | real iMF needs exactly this | **Keep** |
| h/t conditioning | real iMF needs the interval input | **Keep** |
| u-only inference, aux discarded | correct iMF inference (matches official repo) | **Keep** |
| Few-step Euler `x + Δt·u` | correct | **Keep** |
| DPCC projector, trainer, config, serialization (+ visual wiring in Gen8) | infrastructure | **Keep** |

### "imfv2" = a flag, not a fork

Gate the objective behind one config switch (default = current behaviour, so existing runs do
not change):

```
imf_objective: 'fm_equivalent'   # current finite-diff path — kept as the A/B baseline arm
             | 'meanflow_jvp'     # imfv2 — the real JVP MeanFlow-Identity objective (Phase 1)
```

`p_losses` branches on the flag: `fm_equivalent` runs today's code untouched; `meanflow_jvp`
runs the §3/§4 JVP objective. Three reasons this beats deleting the old path:

1. **No regression** — the working FM-equivalent path stays available.
2. **It becomes the honest A/B baseline** — the only way to measure what the JVP actually buys
   is real-iMF @1–2 NFE *vs* FM-equivalent @10 NFE on identical data/seeds (Phase 5).
3. **One flag covers both Gen3v4 and Gen8** — they share `imf_diffusion.py`, so the in-place
   flag propagates to the visual variant automatically. A new folder would fork that and create
   drift.

> **Do not create an `imf_v2/` folder or a new generation.** imfv2 is the `meanflow_jvp` mode of
> the existing shared `imf_diffusion.py`. Keep the FM-equivalent mode as the labelled baseline,
> never present it as a distinct stronger model (U3 §9.2).

---

## 3. The math we implement (Gen3v4 DATA-AT-1 convention)

Interpolant and instantaneous velocity (per sample):
```
z_τ = (1−τ)·ε + τ·x₁ ,   τ∈[0,1] (τ=1 data) ;   v(z,τ) ≡ dz/dτ ,  sample-level v = x₁ − ε
```

Average velocity over `[r, t]` (t>r) and the **MeanFlow Identity** (differentiate the definition
`(t−r)·u = ∫ᵣᵗ v ds` w.r.t. `t` at fixed `r`):
```
u(z_t, r, t) = v(z_t, t) − (t − r) · d/dt u(z_t, r, t)
d/dt u       = ∂_t u + v · ∂_z u                      ← total derivative = a JVP of the network
```

This is the line Gen3v4 never implemented; its finite-difference shortcut drops the `d/dt u`
term and so collapses to `u = v = x₁−ε` (U3 §3). **The whole plan is: put that term back, the
right way.**

Training target (note **stop-gradient** — we regress to it, we do not backprop through it):
```
target = stopgrad[ v_inst − (t−r) · (∂_t u_θ + v_inst · ∂_z u_θ) ]
loss   = w · ‖ u_θ(z_t, r, t) − target ‖²
```
where `v_inst = x₁ − ε` and the JVP is taken with tangents `(∂z=v_inst, ∂t=1, ∂h=1)`.

Sampling (reuse `imf.py` logic), 1–2 NFE:
```
z ← z + (t_next − t_cur) · u_θ(z, r, t)        (DATA-AT-1 forward; sign-flip of imf.py)
```

---

## 4. Phased execution plan

### Phase 0 — Convention & architecture alignment (decide before touching the loss)

- **0a. `t` vs `h` conditioning.** The official model conditions on **`h=t−r` only**
  (`imfDiT.py:370-371`). Gen3v4 conditions on **both `t` and `h`** (and U3-B1 explicitly warned
  *never freeze t*). Decision: **match the reference (h-centric)**, or keep the dual `t,h`
  input. Recommendation: keep `t` *and* `h` (more information; Gen3v4's UNet already trained that
  way), but be aware the JVP tangents must then cover **both** `t` and `h` (`∂t=1, ∂h=1`).
  Document the choice; it changes the JVP call signature.
- **0b. Head split.** Confirm Gen3v4's u-head / aux-head depth split mirrors the reference
  `aux_head_depth` shared-backbone pattern. The v/aux head is kept **only for training**
  (optional regulariser); it is discarded at sampling, exactly as the reference drops `v_heads`
  at `eval_mode`.
- **0c. Convention lock.** Stay in DATA-AT-1 (τ=1 data) to avoid churning the rest of the stack;
  every identity/sign in this plan is already written for it. Record the one sign-flip vs
  `imf.py` (which integrates 1→0).

### Phase 1 — Replace the training objective (the core, in `imf_diffusion.py:p_losses`)

Logic (no production code — see U3 §10.4 for the illustrative sketch):
1. Sample `ε`, `x₁`, `t∈(0,1]`. Sample `r∈[0,t]` **with ~25% of the batch forced to `r=t`** (the
   anchor — at `r=t` the target reduces to `v_inst`, i.e. pure FM, which grounds the field).
2. Form `z_t = (1−t)ε + t·x₁`; apply conditioning (inpaint actions) on `z_t`.
3. Compute `v_inst = x₁ − ε` (+ conditioning, noise-masked dims).
4. **JVP:** evaluate `u_θ` and its total derivative `d/dt u_θ` in one forward-mode pass with
   tangents `(∂z=v_inst, ∂t=1, ∂h=1)`. (Use `torch.func.jvp`; the function must be **purely
   functional** — see Phase 1 risks.)
5. **Target:** `target = detach( v_inst − (t−r)·d/dt u_θ )`.
6. **Adaptive weight:** `w = (‖u_θ − target‖²_detached + c)^(−p)`, `p≈0.5–1`, `c≈1e-3`,
   per-sample, detached.
7. **Loss:** `w · ‖u_θ − target‖²`; optionally add a small `‖aux − v_inst‖²` regulariser
   (discarded at sampling). Drop the old `u_mix/v_mix/aux` finite-difference machinery.

**Phase 1 risks / must-handle:**
- **Functional purity for JVP.** `condition_dropout` (Bernoulli CFG mask), any in-place op, and
  EMA reads break `torch.func.jvp`. Disable/serialise dropout inside the JVP (use the
  deterministic conditioning path), and ensure `apply_conditioning` is functional.
- **Second-order guard.** Never let gradients flow through the JVP target → the `detach()` in
  step 5 is mandatory; verify with a gradient check that `target.requires_grad is False`.
- **Cost.** JVP ≈ one extra linearised forward (≈2× train compute / memory). Trivial at `H=8`.

### Phase 2 — Low-NFE inference (`imf_diffusion.py:p_sample_loop`, config)

- Add a config knob `ode_inference_steps_v3 ∈ {1, 2, 4, 10}`; **default low (1–2)** for the iMF
  benefit, but keep 10 available for A/B against the FM baseline.
- The update rule is already correct (`x + (t_next−t_cur)·u`); just let `flow_steps` be small.
- Validate one-step reconstruction quality before scaling (the iMF "free lunch" only shows up
  here — if 1-NFE samples are good, the JVP training worked).

### Phase 3 — Improved-iMF guidance (OPTIONAL / ADVANCED)

The "improved" in iMF is **interval CFG**: guidance scale `omega` applied only on a time
interval `[t_min, t_max]` (`imfDiT.py:_build_sequence`, `imf.py:u_fn`). Map this onto Gen3v4's
**returns-conditioning** (`returns_condition`, `condition_guidance_w`):
- Replace the plain CFG mix in `_predict_velocity` with interval-gated guidance: apply guidance
  only when `t∈[t_min,t_max]`.
- This is a *quality* refinement on top of working iMF — **do not** attempt it until Phases 1–2
  pass; it adds three conditioning inputs and is the most fiddly part of the reference.

### Phase 4 — DPCC projector re-tune for low NFE (avoiding-specific, REQUIRED before shipping)

At 1–2 NFE the projector's `snapping_start_idx = (1−threshold)·flow_steps` makes the "near-end"
window the **entire** rollout (U3 §9.3). Therefore:
- Re-derive the snap schedule for low NFE (e.g. always-project at the final step, or project
  post-jump on the single-step path).
- Re-validate constraint satisfaction on `avoiding-d3il` — **the safety guarantee must hold at
  the new NFE**, not just at 10. This is a gate, not a nicety.

### Phase 5 — Validation & A/B

- **Sanity:** 1-NFE reconstruction RMS on held-out trajectories (the iMF "did the JVP work?"
  test).
- **A/B vs FM:** same data, same seeds, three columns —
  (i) FMv3ODE @10 NFE, (ii) Gen3v4-iMF-old @10 NFE (the current FM-equivalent), (iii)
  full-iMF @1–2 NFE. Report **quality** *and* **`fm_ms`** (the REALTIME budget metric).
- **Success = iMF @1–2 NFE matches FM @10 NFE quality at a fraction of the latency.** That is
  the only outcome that justifies the method (U3 §5).

---

## 5. Stability dials (where success/failure is decided)

| Dial | Effect if wrong | Safe starting point |
|---|---|---|
| `r=t` fraction | too high → collapses back to FM; too low → field never anchors, diverges | 25% |
| Adaptive-weight `p`, `c` | unweighted → JVP-magnitude outliers blow up loss | `p=0.5`, `c=1e-3` |
| JVP `detach` | missing → second-order, NaN/blowup | always detach target |
| Dropout inside JVP | non-functional → `torch.func` error / silent wrong grad | deterministic path in JVP |
| NFE at inference | stay at 10 → no speed gain, JVP cost wasted | 1–2 (with 10 as A/B) |

---

## 6. File-touch map (for the implementing agent)

| File | Change |
|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | **`p_losses`**: branch on `imf_objective` (§2A) — `fm_equivalent` keeps today's finite-diff target; `meanflow_jvp` runs JVP MeanFlow-Identity target + stopgrad + adaptive weight + `r=t` sampling. **`p_sample_loop`**: allow low `flow_steps`. **`_predict_velocity`**: optional interval-CFG (Phase 3). |
| `flow_matcher_v3_imeanflow/models/imf_engine.py` / `unet1d_temporal_cond.py` | Ensure forward is **functional** for `torch.func.jvp`; expose `t` and `h` tangent inputs cleanly. |
| `config/avoiding-d3il.py` (`flow_matching_v3_imeanflow` block) | add **`imf_objective: 'fm_equivalent'`** (default) — flip to `'meanflow_jvp'` for imfv2; `ode_inference_steps_v3 → 1–2` for the imfv2 run; add `r_equals_t_frac`, adaptive-weight `p/c`, interval-CFG `t_min/t_max/omega`. |
| `flow_matcher_v3_imeanflow/sampling/projection.py` (or projector schedule) | Phase 4 low-NFE snap schedule + constraint re-validation. |

---

## 7. Decision points to confirm before coding

1. **Condition on `t` and `h`, or `h` only?** (Phase 0a) — affects JVP tangents. *Lean: keep both.*
2. **Keep the aux/v-head as a training regulariser, or drop it entirely?** *Lean: keep, tiny weight, discard at sampling (mirrors reference).*
3. **Target NFE for the shipped avoiding model?** 1 (max speed) vs 2 (safer quality). *Lean: A/B both.*
4. **Do Phase 3 (interval CFG) now or defer?** *Lean: defer until 1–2 pass.*

---

## 8. What this plan deliberately does NOT do

- Does **not** rewrite the architecture — the body is already correct iMF.
- Does **not** touch the FMv3ODE baseline or the state `diffuser/` — those remain the clean
  references for A/B and for any DPCC-paper comparison.
- Does **not** claim low-NFE iMF is safe on avoiding until Phase 4 re-validates the projector.

---

## 9. One-paragraph summary

Reuse the official repo's **inference and architecture** (verified, low-risk) and **build the one
missing piece**: the JVP-based MeanFlow-Identity training objective, with stop-gradient, a 25%
`r=t` anchor, and adaptive weighting — then move inference to 1–2 NFE and re-tune the DPCC snap
schedule for that regime. That converts Gen3v4 from "iMF body, FM brain" into a true MeanFlow
that can match FM quality at a fraction of the inference latency — the only axis on which iMF
beats FM, and the one that matters for real-time FM-PCC control.

---

## Caveats

- Reference read 2026-06-13: `/workspaces/imeanflow/imf.py`, `models/imfDiT.py`, `README.md`.
  The repo is inference-only; the JVP training spec is transcribed from the MeanFlow method, not
  copied from this repo.
- All file:line refs are as of branch `update_into_FM`, 2026-06-13; re-locate if files move.
- This is a logic/principle plan. Sign conventions, JVP tangent wiring, and functional-purity
  details must be verified empirically by the implementing agent (1-NFE reconstruction is the
  fastest correctness probe).
