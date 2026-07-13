# Gen3v4_imf — What is NEW vs both upstreams, and what kills the eval (KILL TABLE)

**Generation:** Gen3v4_imf / U9 · **Date:** 2026-07-13
**Question:** training is healthy, eval is total nonsense — theory says this can't happen, so a *hybridization defect* must exist. Enumerate everything in `flow_matcher_v3_imeanflow` that exists in **neither** upstream (`aux_repo/dpcc`, `aux_repo/imeanflow`), rate each for danger.
**Failing run (from opened scene.json):** `H8_D…iMeanFlowODE_a1.0_b1.0_aw1_objmeanflow_jvp_bbdit_tslogit_normal / H8_K10_Meuler_T0.5` → **DiT backbone, meanflow_jvp objective, logit-normal, euler**.
**Prior note:** normalization was audited and cleared in `INVESTIGATION_diffuser_exploding_traj_normalization.md`.

---

## Executive verdict — the kill chain (found, line-verified)

> **At every sampling step, the model output is mixed with an "unconditional" prediction produced by a network branch that was NEVER TRAINED — then that garbage is amplified up to ~8×. Neither upstream can have this failure; it exists only in our hybrid.**

The chain, with evidence:

1. **Training config** (`config/avoiding-d3il.py:518,523,495`): `include_returns: True`, `condition_guidance_w: 1.2`, `meanflow_cfg_omega: 4.0`.
2. **Train script** (`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py:227-228`): `returns_condition=args.include_returns` → the pickled `diffusion_config` carries **`returns_condition=True, condition_guidance_w=1.2, meanflow_cfg_omega=4.0`** into eval (eval rebuilds from the pickle, `eval…py:95`).
3. **Sampling, gate #1 — returns-CFG (DPCC/decision-diffuser pattern)** (`imf_diffusion.py:181-184`): `returns_condition and returns is not None and w>0` — **all true at eval** (policy always passes `returns=test_ret·1`, `policies.py:42`). So EVERY step computes
   `velocity = 2.2·u_cond − 1.2·u_uncond(force_dropout=True)`.
4. **Sampling, gate #2 — interval-CFG (our own invention at sample time)** (`imf_diffusion.py:187-190`, `:274`): `meanflow_cfg_omega=4.0 > 0` → for τ∈[0.4,0.6] a SECOND uncond call and a second mix:
   `velocity = u_uncond + 4·(velocity − u_uncond)` ⇒ net **`8.8·u_cond − 7.8·u_uncond`** on the mid-trajectory steps.
5. **The uncond branch is untrained.** `force_dropout=True` makes the DiT embed the **null class token `y_embedder(1)`** (`imf_dit_trajectory.py:349`). Training **never** calls with `force_dropout=True` and **never applies random condition dropout**: the DiT accepts `use_dropout` and *ignores it* (`imf_dit_trajectory.py:356-358` — "accepted for parity"; zero uses in the file). The official iMF trains the null label via `cond_drop` (`imf.py:263-293`, 10% of every batch); DPCC trains its uncond branch via Bernoulli returns-dropout (`condition_dropout=0.25`, actually applied in its UNet, `unet1d_temporal_cond.py:259-262` — but only when `returns_condition=True`). **We ported neither mechanism into the meanflow_jvp path.** The null-token embedding sits at random init and receives **zero gradient for all 100k steps**.
6. **Result:** `u_uncond` = the network evaluated on a token it has never seen — out-of-distribution, arbitrary. Every step injects `−1.2×` (and mid-steps `−7.8×`) of that into the velocity. `x` leaves the training manifold, subsequent queries are OOD too, and the trajectory saturates far outside `[-1,1]` → `LimitsNormalizer` clips to the data-box corners → screen-spanning lines / corner-seeking rollouts.

**Why every observed symptom matches:**
- **Training curves look perfect** — the uncond branch is never exercised at train time; the u-loss sees only healthy inputs. ✔
- **Eval is catastrophic** — both gates open only at sampling. ✔
- **`diffuser` variant is the worst** — `projector=None` (`eval…py:249`), nothing re-snaps `x` after each poisoned step; `dpcc*` variants get projected back into the feasible set every step, partially masking the poison. ✔
- **The DiT run explodes** while earlier UNet runs did not: the UNet arm is constructed with **`returns_condition=False` hardcoded** (`imf_trajectory_model.py:85`) → `force_dropout` is a **no-op** there → `u_uncond ≡ u_cond` → both mixes algebraically collapse to identity. **The bug bites only `bbdit`.** ✔

---

## THE TABLE — everything new-in-Gen3v4 (in neither upstream), rated

Provenance: **[INVENTED]** = exists in neither upstream · **[RECOMBINED]** = each half exists in one upstream, the combination is ours · **[FLIPPED]** = deliberate re-derivation. Danger = probability × severity of causing *this* eval catastrophe. **Active** = live in the failing `bbdit` run.

| # | What | Where (ours) | Upstream truth | Prov. | Active | Danger |
|---|------|--------------|----------------|-------|--------|--------|
| **1** | **Sampling-time CFG mixing against an UNTRAINED uncond branch** — returns-CFG `2.2u_c−1.2u_u` every step; null token gets zero gradients all training | `imf_diffusion.py:181-184`; `imf_dit_trajectory.py:349,356`; `train…py:227-228` | DPCC mixes but *trains* uncond via Bernoulli dropout; iMF *trains* null label via `cond_drop` (`imf.py:263`) and **never mixes at sampling** | RECOMBINED (worst halves of both) | **YES** | **10/10 — the kill** |
| **2** | **Interval-CFG output-space mixing at sampling**, ω=4 for τ∈[0.4,0.6], stacked ON TOP of #1 → `8.8u_c−7.8u_u` mid-trajectory | `imf_diffusion.py:187-190,274`; `config:869-871` | Official iMF sampler is a **single** `u_fn` call (`imf.py:112`) — guidance is baked into the *training target*, never output-mixed | **INVENTED** | **YES** | **9/10 — the amplifier** |
| **3** | **No condition dropout anywhere in training** (DiT ignores `use_dropout`; jvp loss never force-drops) | `imf_dit_trajectory.py:356`; `_p_losses_meanflow_jvp` (no dropout call) | Both upstreams train their uncond branch | INVENTED (by omission) | **YES** | **9/10 — root enabler of #1/#2** |
| **4** | **Interval-CFG trained with an unguided target**: net conditioned on per-sample ω but regressed to the ω-independent `v_inst + h·du/dr`; official modifies the target via `guidance_fn` (`v_g = v + (1−1/ω)(v_c−v_u)`) — `guidance_fn`/`v_fn` **not ported** | `imf_diffusion.py:557-583` vs `imf.py:295-325,358` | iMF's ω-conditioning is meaningful *only* because the target changes with ω | INVENTED (half-port) | YES | 6/10 — trains net to ignore ω; not explosive alone but made #2 look safe on paper |
| **5** | **Returns plumbing is fictional**: `returns` accepted by `_predict_uv` then silently dropped (`engine.forward_train` has no `returns` param); `include_returns=True, returns_scale=400` condition nothing | `imf_diffusion.py:159-164`; `imf_engine.py:165-193` | DPCC actually feeds returns to the backbone | INVENTED (by accident) | YES | 5/10 — its *only* live effect is gating #1 ON (`returns is not None`) |
| 6 | `repeat_last` reinterpreted in a flow: extra loop iterations **re-add** `u·dt` past t=1 (in DPCC diffusion the repeated last step is a contraction + projection polish) | `imf_diffusion.py:223,266-267` (same in `flow_matcher_v3/models/diffusion.py:172-174`) | DPCC-only concept; harmless there | RECOMBINED | no (=0 in this eval; no `sample_kwargs`) | 6/10 latent — overshoot bomb if ever set |
| 7 | torchdiffeq multi-stage solvers feeding u(x,t,h) as an *instantaneous* RHS (documented as empirical gamble in-file) | `imf_diffusion.py:276-321` | iMF never sub-steps u; one `z−h·u` per interval | INVENTED | no (`legacy_euler` active, `config:855`) | 4/10 latent |
| 8 | JVP tangent = **analytic** `v_inst = x−e`, not the **predicted** `v_c` — reverts iMF's headline improvement ("we use predicted v in the jvp", `imf.py:372`) to original-MeanFlow style | `imf_diffusion.py:548,574` | iMF: predicted v; orig. MeanFlow: analytic v (works, published) | FLIPPED to older lineage | YES | 3/10 — quality, not explosion |
| 9 | (t,r) sampling: `r = t·U(0,1)` (+25% r=t anchor) vs official *two independent logit-normals* + min/max + 50% r=t | `imf_diffusion.py:537-539` vs `imf.py:126-139` | different (t,h) joint; ours has fatter h tail, fewer FM anchors | INVENTED | YES | 3/10 |
| 10 | DPCC per-dim `loss_weights` (action_weight=10, discount) **multiplied inside** the MeanFlow loss, then adaptive-weighted — each upstream has one, neither has both | `imf_diffusion.py:587-591` | — | RECOMBINED | YES | 3/10 |
| 11 | Adaptive weighting `p=0.5, c=1e-3` on per-sample **mean** vs official `p=1.0, eps=0.01` on per-sample **sum** | `imf_diffusion.py:590` vs `imf.py:71-72,380-386` | different effective loss geometry | INVENTED (re-tuned) | YES | 2/10 |
| 12 | **DATA-AT-1 time flip** (t=0 noise → t=1 data; official is t=1 noise → t=0 data). I re-derived the START-anchored identity, tangents `(v, +1, −1)`, and the sampler anchoring — **internally consistent** (verified this session) | `imf_diffusion.py:199,509-522` vs `imf.py:42,114,350` | — | FLIPPED | YES | 2/10 — checked, sound |
| 13 | Conditioning by **inpainting inside the MeanFlow JVP/targets** (x_r pinned, tangent & target zeroed at cond dims → u trained to 0 there) — iMF has no inpainting concept at all | `imf_diffusion.py:534,544,549,582`; `helpers.py:143-161` | DPCC-only pattern, transplanted into a JVP objective | RECOMBINED | YES | 2/10 — self-consistent with sampler pinning |
| 14 | `eval_use_ema=False` (DPCC-legacy raw weights) vs iMF default EMA-at-sampling | `config:877` vs `imeanflow/utils/sample_util.py` | — | RECOMBINED | YES | 3/10 — few-step MeanFlow is known EMA-sensitive |
| 15 | DiT scaled to toy size (256/8/4 heads, tokens 2/2/1 vs official 768/12+/12, tokens 4/4/8) | `config:505-510` vs `imfDiT.py:393-427` | — | re-scaled | YES | 2/10 |
| 16 | Real-valued RoPE reimplementation (JVP-safe) — pairing & frequencies verified equivalent to the complex bitcast this session | `imf_dit_trajectory.py:123-150` vs `imfDiT.py:370-385` | — | re-implemented | YES | 1/10 — checked, equivalent |
| 17 | Loss form `‖u − sg(v+h·du/dr)‖²` vs official `‖(u+h·sg(du/dt)) − v_g‖²` — **gradients identical** (residual algebra checked this session) | `imf_diffusion.py:581-586` vs `imf.py:376-385` | — | re-arranged | YES | 1/10 — checked, equivalent |

---

## The one-shot falsification test (NO retraining — run on cluster)

Rows 1–3 are all **eval-time gates on scalars**. Kill them with a 2-line patch in the eval, right after the model is loaded (`eval…py:139ff`, before `policy = Policy(...)`):

```python
fm_model.condition_guidance_w = 0.0    # closes gate #1 (returns-CFG mixing)
fm_model.meanflow_cfg_omega  = 0.0     # closes gate #2 (interval-CFG mixing)
```

Re-run **only the `diffuser` variant on the existing `bbdit` checkpoint**.

- **If trajectories become sane/bounded** → kill chain confirmed. The checkpoint itself was always fine; the sampler was poisoning it. (Quality may still be mediocre — rows 4, 8, 9, 14 degrade the model vs true iMF — but "total nonsense" ends.)
- **If it still explodes identically** → rows 1–3 are exonerated; escalate to row 12/8 (add the 1-NFE reconstruction check already recommended in `imf_diffusion.py:524`'s own docstring) and row 14 (`eval_use_ema=True`).

Cheap corroboration in the same run: print `‖u_cond − u_uncond‖ / ‖u_cond‖` at one step — if the untrained-token theory is right, this ratio is O(1) garbage rather than a small guidance correction.

## If confirmed — the proper fix (in order of scientific soundness)

1. **Minimal (recommended for the rescue shot):** run with CFG fully OFF (both scalars 0). This is the honest ablation arm — official iMF *without* guidance is still a working generative model; guidance is a quality add-on, not a correctness requirement.
2. **Full port (only if CFG is wanted):** port `cond_drop` (train the null token, ~10%/batch) **and** `guidance_fn`/`v_fn` (guided target `v_g`) from `imf.py:263-325` — and **delete** the sampling-time output-space mixing entirely (`imf_diffusion.py:181-190`), replacing it with the official single-call ω-conditioned `u_fn`. Half-porting is what created this bug.
3. Independently of CFG: stop passing `returns` into `Policy`/model for this generation (it conditions nothing — row 5) so the dead gate can never re-arm.

---

*Verification status of "cleared" rows: 12, 16, 17 were manually re-derived/checked line-by-line this session; normalization was cleared in the prior MD. Everything else in the table is a live deviation whose danger rating reflects residual uncertainty.*
