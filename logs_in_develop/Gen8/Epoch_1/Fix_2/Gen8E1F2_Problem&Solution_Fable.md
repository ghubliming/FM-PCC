# Gen8E1F2 — Problem & Solution

**Date:** 2026-06-10 · **Updated:** 2026-06-11 (review integration, see §Review) · **Scope:** iMF Visual Aligning — exploded chaotic lines at eval (post Fix_2.5)
**Inputs:** `Epoch_1/{PLAN.md, Fix_1, Fix_2}`, `imf_visual_aligning/models/*`, Gen3v4 history (`fix_6/AUDIT_REPORT_INTEGRATION_BUG.md`, u2 `fix_3` Deviation A/B), reference-iMF audit notes
**Principle applied:** code/math/ML-logic bugs first.

---

## Problem

After Fix_2.5 finally let the checkpoint load (dim=32 rebuild), the Gen8 iMF visual aligning eval runs but produces **exploded chaotic lines** — the same class of symptom Gen3v4 hit twice before (fix_1 over-scaled target; fix_6 single-step Euler). Training loss was healthy.

## Root causes found (math/ML logic — ranked)

### B1 — Frozen `t = 0.5` at inference contradicts a t-conditioned model ⚠️ PRIMARY

`imf_visual_aligning/models/imf_diffusion.py:187` (`T_CONST_INFERENCE = 0.5`, inherited from Gen3v4u2 fix_3 "Deviation B"): every Euler step queries the model with the **same constant t**, while `h = 1/flow_steps`.

Why this is invalid: the rationale ("reference iMF ignores t") only holds for a model **trained without t-conditioning** — the reference learns to infer progress from x alone. **Our model was trained with the true t** (`p_losses` line 312 passes `t` into `time_mlp`), so it learned `u(x, t, h)` *relying* on t to interpret x. At inference:

- Step 0 feeds pure noise x ~ N(0,1) labelled t=0.5. A genuine t=0.5 interpolant has std ≈ 0.5–0.7 (half noise, half normalized data) — the input is badly out-of-distribution **and** mislabelled.
- Integrating a time-dependent field's **t=0.5 slice** as if it were an autonomous field cannot transport noise→data; the field at t=0.5 expects half-denoised inputs at every step.

The model never learned to infer t from x (it never had to), so the time information is simply destroyed. Garbage velocity at every step → chaotic lines. This deviation "fix" converted a working FM-style sampler into a mathematically wrong one.

### B2 — Endpoint reversal in the data-at-1 port (breaks the actual iMF promise)

Reference iMF (data-at-0) conditions `u(z_t, r, t)` on the point the sampler is currently AT (the noise-side endpoint) and steps toward data. Our port (data-at-1) trains `u` conditioned on **x_t — the data-side endpoint** of [r, t] (`p_losses`: `_predict_uv(x_t, cond, t, h)` with target `(x_t − x_r)/h`), but the sampler steps **forward from the noise-side point**. The conditioning endpoint was not swapped when the time convention was flipped.

Consequences: for small h with the **correct** t this degenerates to vanilla-FM Euler (only O(h) bias) — tolerable. For large h / one-step (the entire point of iMF) it is fully inconsistent: a one-step query would need `u(x_data, t=1, h=1)` but receives pure noise. So even after fixing B1, Gen8-as-trained is at best a vanilla FM, never a one-step iMF.

### Context — third incident in this family

Gen3v4 fix_6 already diagnosed "chaotic trajectories" from `flow_steps_v3: 1`; Gen3v4u2 fix_1 from an over-scaled target. The recurring source is iMF **inference-time deviations** layered on an FM-PCC-convention engine. Each deviation was justified by reference parity while ignoring that the reference's training differs.

### Exonerated

Fix_1 import merges, Fix_2.5 loader (dim validation correct; state_dict loads strictly), VisualUNet FiLM wiring (h threaded end-to-end via `h_mlp`, visual cond via `cond_mlp` concat), `apply_conditioning` string-key guard, σ=1.0 noise consistency train↔sample.

## Solution (for the implementing agent)

1. **No-retrain fix (do first):** in `p_sample_loop`, replace `t_const` with the true integration time per step — **`t_i = i / flow_steps`** (the point the sampler is currently AT). ~~or `(i+1)/flow_steps`; the right endpoint matches the training conditioning more closely~~ ← retracted per review: training always paired x with its *own* time, so feeding (x@τ, t=τ) is the in-distribution choice; labelling the current x with the *destination* time `(i+1)/N` would mislabel the pair and amplify the B2 mismatch. Keep `h = 1/flow_steps`. The (x, t, h) triples are then in-distribution (training had `h = t − r, r ~ U(0,t)`, so small h at any t was seen). Expected: behaves like Gen7 vanilla FM multi-step. Apply the same change to the Gen3v4 source (`flow_matcher_v3_imeanflow/models/imf_diffusion.py:187`) — same bug.
2. **A/B validate via pseudo-run logic, real run on cluster:** eval the existing checkpoint with (a) frozen t=0.5 vs (b) true t, same seed/flow_steps. (b) producing structured trajectories confirms B1 as root cause with zero retraining cost. **Prerequisite (B3, from review): clean up the redundant sampling paths first** so the A/B is guaranteed to exercise `p_sample_loop` — see §Review B3.
3. **Retrain fix (restores genuine iMF, optional second phase):** swap the conditioning endpoint in `p_losses` — condition on `(x_r, r, h)` and predict `(x_t − x_r)/h`, sampler steps `x ← x + h·u(x, r=current_t, h)`. This is the correct data-at-1 transcription of the reference. Only after this does few-step/one-step iMF sampling become legitimate.
4. **Guardrail:** add an assertion/warning in `p_sample_loop` if the model has an active `time_mlp` while t is being frozen — make this class of deviation impossible to reintroduce silently.

**Decision guidance:** if step 1+2 yields aligning performance ≈ Gen7 FM, Gen8's added value (one-step) still requires step 3 + retrain; decide then whether iMF-one-step is worth the retrain or Gen8 closes as "iMF == FM under multi-step."

---

## Review integration (2026-06-11, from `Gen8E1F2 — Technical Review.md`)

The review was checked against the code and is correct on all points. B1/B2/Steps 1–4
confirmed (the reviewer's initial objection to Step 3 was self-retracted: the target
`(x_t − x_r)/h = data − noise` is endpoint-independent for the linear interpolant, so
the proposed `(x_r, r, h)` conditioning with the same target and same forward update is
the correct data-at-1 transcription). Two genuine misses were found and are adopted:

### B3 (review, verified) — Redundant inconsistent sampling paths must be removed before the A/B

`imf_engine.py:96-147` (`iMFEngine.sample()`) and `imf_trajectory_model.py:92-124`
(`sample_trajectory()`) are **independent sampling loops** that differ from
`p_sample_loop` on both axes under investigation:

- they pass the **true `t_cur`** (no B1 bug), and
- they **mix the aux branch** into the velocity (`u_weight·u + 0.1·v_weight·aux`),
  contradicting Deviation A (`p_sample_loop` discards aux).

Verified: neither has any caller in the live eval path (eval goes through
`p_sample_loop`), so they are dead-but-dangerous code. If any debug/eval harness picks
them up, the B1 A/B comparison is confounded on two axes at once. **Action: delete
both standalone samplers (preferred) or synchronize them with `p_sample_loop`,
*before* running Step 2.**

### B4 (review, verified) — h-distribution at inference vs training: noted, not blocking

Training draws `r ~ U(0,t)` with `t ~ 1−Beta(1.5,1)`, so `h = t−r` spans (0,1) and
large h *is* seen. Multi-step inference uses small constant `h = 1/N` — in-distribution.
The gap only matters for the few-step/one-step regime (Step 3 territory): one-step
quality depends on density of large-h training pairs at the specific (x, t) contexts
the sampler visits. Monitor at Step 3 evaluation; no action now.

### Step 1 refinement (review, adopted)

Use `t_i = i / flow_steps` — not `(i+1)/flow_steps`. The original parenthetical
suggesting the right endpoint is struck above; training only ever saw (x, t) pairs
where the time labels x's own position, so self-consistent labelling is the
in-distribution choice.
