# Gen3v4_imf — What is NEW vs both upstreams, and what kills the eval (KILL TABLE)

**Generation:** Gen3v4_imf / U10 · **Date:** 2026-07-13
**Question:** training is healthy, eval is total nonsense — theory says this can't happen, so a *hybridization defect* must exist. Enumerate everything in `flow_matcher_v3_imeanflow` that exists in **neither** upstream (`aux_repo/dpcc`, `aux_repo/imeanflow`), rate each for danger.
**Failing run (from opened scene.json):** `H8_D…iMeanFlowODE_a1.0_b1.0_aw1_objmeanflow_jvp_bbdit_tslogit_normal / H8_K10_Meuler_T0.5` → **DiT backbone, meanflow_jvp objective, logit-normal, euler**.
**Prior note:** normalization was audited and cleared in `INVESTIGATION_diffuser_exploding_traj_normalization.md`.

> **Reference convention:** every code pointer is anchored to **file · `function/class` · logic** — because line numbers rot as files are edited. Any `(~Lnn)` is only a "where it was at time of writing" hint; if it has moved, find the named function and the described logic — that is the durable anchor. The full row-by-row function map is in **§ WHERE THE CODE ACTUALLY IS** below.

---

## Executive verdict — the kill chain (found, function/logic-verified)

> **At every sampling step, the model output is mixed with an "unconditional" prediction produced by a network branch that was NEVER TRAINED — then that garbage is amplified up to ~8×. Neither upstream can have this failure; it exists only in our hybrid.**

The chain, by function/logic (line hints in parens; trust the function name):

1. **Training config** — `config/avoiding-d3il.py` · `flow_matching_v3_imeanflow` train block: `include_returns: True`, `condition_guidance_w: 1.2`, `meanflow_cfg_omega: 4.0`.
2. **Train wiring** — `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` (the diffusion-construction call, ~L227-228): passes `returns_condition=args.include_returns` → the pickled `diffusion_config` carries **`returns_condition=True, condition_guidance_w=1.2, meanflow_cfg_omega=4.0`** into eval, which rebuilds the model from that pickle in `load_diffusion_with_override`.
3. **Sampling gate #1 — returns-CFG (DPCC/decision-diffuser pattern)** — `imf_diffusion.py` · **`iMeanFlowODE._predict_velocity`**, the `if self.returns_condition and returns is not None and self.condition_guidance_w > 0:` block. All three are true at eval (the policy, `policies.py` · **`Policy.__call__`**, always passes `returns = test_ret·ones`). So EVERY step computes
   `velocity = (1+w)·u_cond − w·u_uncond` = `2.2·u_cond − 1.2·u_uncond`, with `u_uncond` from a `_predict_velocity` call with `force_dropout=True`.
4. **Sampling gate #2 — interval-CFG (our own invention at sample time)** — same function **`iMeanFlowODE._predict_velocity`**, the `if cfg_scale > 0:` block; the τ-gate that sets `cfg_scale` lives in **`iMeanFlowODE.p_sample_loop`** (`step_cfg` computed per step). With `meanflow_cfg_omega=4.0`, for τ∈[0.4,0.6] this adds a SECOND uncond call + mix:
   `velocity = u_uncond + 4·(velocity − u_uncond)` ⇒ net **`8.8·u_cond − 7.8·u_uncond`** on mid-trajectory steps.
5. **The uncond branch is untrained.** `force_dropout=True` makes the DiT select the **null class token** (`y_idx = num_classes`) in `imf_dit_trajectory.py` · **`IMFDiTTrajectory._build_sequence`**. Training **never** calls with `force_dropout=True` and **never applies random condition dropout**: **`IMFDiTTrajectory.forward`** accepts `use_dropout` and ignores it ("accepted for parity"; grep the file → zero uses), and the JVP loss **`iMeanFlowODE._p_losses_meanflow_jvp`** has zero `force_dropout` calls. Official iMF trains the null label in `imf.py` · **`cond_drop`** (~10%/batch); DPCC trains its uncond branch via Bernoulli returns-dropout in its UNet (`unet1d_temporal_cond.py` · `forward`, active only when `returns_condition=True`). **We ported neither into the meanflow_jvp path** → the null-token embedding sits at random init with **zero gradient for all 100k steps**.
6. **Result:** `u_uncond` = the network on a token it has never seen — arbitrary/OOD. Every step injects `−1.2×` (mid-steps `−7.8×`) of that into the velocity. `x` leaves the training manifold, subsequent queries are OOD too, the trajectory saturates far outside `[-1,1]` → `LimitsNormalizer.unnormalize` clips to the data-box corners → screen-spanning / corner-seeking rollouts.

**Why every observed symptom matches:**
- **Training curves look perfect** — the uncond branch is never exercised at train time; the u-loss sees only healthy inputs. ✔
- **Eval is catastrophic** — both gates live in `_predict_velocity`, which only runs at sampling. ✔
- **`diffuser` variant is the worst** — the eval main loop sets `projector = None` for it, so `p_sample_loop`'s projection block never re-snaps `x` after a poisoned step; `dpcc*` variants project back into the feasible set every step, partially masking the poison. ✔
- **The DiT run explodes while earlier UNet runs did not** — the UNet arm is constructed with **`returns_condition=False` hardcoded** in `imf_trajectory_model.py` · **`iMFTrajectoryModel.__init__`** → `force_dropout` is a **no-op** there → `u_uncond ≡ u_cond` → both mixes algebraically collapse to identity. **The bug bites only `bbdit`.** ✔

---

## THE TABLE — everything new-in-Gen3v4 (in neither upstream), rated

Provenance: **[INVENTED]** = exists in neither upstream · **[RECOMBINED]** = each half exists in one upstream, the combination is ours · **[FLIPPED]** = deliberate re-derivation. Danger = probability × severity of causing *this* eval catastrophe. **Active** = live in the failing `bbdit` run.

> The `Where (ours)` column below gives **quick `file:line` hints only**. The **durable anchor** for every row — file · `function/class` · logic to look for — is the row-by-row map in **§ WHERE THE CODE ACTUALLY IS**. If a line hint no longer matches, use that section.

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

## WHERE THE CODE ACTUALLY IS — row → file · function · lines (check it yourself)

All paths are under `flow_matcher_v3_imeanflow/models/` unless noted. Upstreams: `OURS` = this repo; `iMF` = `/workspaces/aux_repo/imeanflow/`; `DPCC` = `/workspaces/aux_repo/dpcc/`. Line numbers drift as the file is edited — trust the **function name** first, then the lines.

| # | OUR code — file · **function** · lines | The exact thing to look at | Upstream to diff against |
|---|----------------------------------------|----------------------------|--------------------------|
| **1** | `imf_diffusion.py` · **`_predict_velocity`** · 181-184 | the `if self.returns_condition and returns is not None and self.condition_guidance_w > 0:` block → `velocity = (1+w)*velocity − w*uncond_vel`. `uncond_vel` comes from `_predict_uv(..., force_dropout=True)`. The untrained token it hits: `imf_dit_trajectory.py` · **`_build_sequence`** · 349 (`y_idx = num_classes if force_dropout`). Train wiring that arms it: `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py:227-228` | DPCC `diffuser/models/diffusion.py` (its `p_mean_variance` / guidance path **trains** the uncond via Bernoulli dropout); iMF `imf.py` · `sample_one_step` (no mixing at all) |
| **2** | `imf_diffusion.py` · **`_predict_velocity`** · 187-190 (the mix) + **`p_sample_loop`** · 274 (the `step_cfg` τ-gate) | second block `velocity = uncond + cfg_scale*(velocity − uncond)`; `cfg_scale` set at :274 only when `t_min ≤ τ ≤ t_max`. Config values: `config/avoiding-d3il.py` · `plan_fm_v3_imeanflow` · `meanflow_cfg_omega/t_min/t_max` | iMF `imf.py` · **`sample_one_step`** · 112 — a **single** `u_fn(...)[0]` call, no output mixing |
| **3** | `imf_dit_trajectory.py` · **`IMFDiTTrajectory.forward`** · 356 (accepts `use_dropout`, never reads it) **and** `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 500-615 (never calls with `force_dropout=True`, no random cond-drop) | grep `use_dropout` in the DiT file → only the signature. grep `force_dropout` in `_p_losses_meanflow_jvp` → **zero hits**. So the null token gets no gradient. | iMF `imf.py` · **`cond_drop`** · 263-293 (drops ~10%/batch, training the null label); DPCC UNet `unet1d_temporal_cond.py:259-262` Bernoulli returns-dropout |
| **4** | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 557-583 | ω/t_min/t_max sampled at 558-560 and fed to the net, but `u_target` (581) = `v_inst + h·du_dr` has **no ω term** → net learns to ignore ω | iMF `imf.py` · **`guidance_fn`** · 295-325 (builds `v_g = v_t+(1−1/ω)(v_c−v_u)`) + **`v_fn`** · 235-261. Neither is ported. |
| **5** | `imf_diffusion.py` · **`_predict_uv`** · 159-164 (takes `returns=`, drops it) → `imf_engine.py` · **`forward_train`** · 165-193 (no `returns` param) → `imf_trajectory_model.py` · **`forward`** (no `returns`) | trace `returns` from `_predict_uv` down: it dies at the engine boundary. `include_returns/returns_scale` in `config/avoiding-d3il.py` condition nothing. | DPCC `diffuser/models/temporal.py` (returns MLP actually feeds the backbone) |
| 6 | `imf_diffusion.py` · **`p_sample_loop`** · 223 (`total_steps = flow_steps + repeat_last`) + 266-267 (loop; `loop_idx` clamps but `x += u·dt` still runs) | with `repeat_last>0`, extra iterations keep integrating past t=1 | DPCC `diffusion.py` repeated last step = contraction+projection, not re-integration. Sibling ref: `flow_matcher_v3/models/diffusion.py:172-174` |
| 7 | `imf_diffusion.py` · **`p_sample_loop`** · 276-321 (the `use_torchdiffeq` branch) + nested **`ode_rhs`** · 291 | multi-stage solvers call `_predict_velocity` (a mean-velocity u) as if instantaneous. Inactive: `ode_solver_backend_v3='legacy_euler'` in config | iMF has no sub-stepping — `sample_one_step` does one `z − h·u` per interval |
| 8 | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 548 (`v_inst = x_start − x_base`) → 574 (used as the JVP z-tangent) | tangent is the analytic v, not a predicted `v_c` | iMF `imf.py:372-373` (uses predicted `v_c` in `jax.jvp`) |
| 9 | `imf_diffusion.py` · **`loss`** · 400 (t via logit-normal) + **`_p_losses_meanflow_jvp`** · 537-539 (`r = t·U(0,1)`, 25% anchor r=t) | our (t,r) joint | iMF `imf.py` · **`sample_tr`** · 126-139 (two logit-normals, min/max, 50% r=t) |
| 10 | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 587-591 | `sq = delta² * self.loss_fn.weights` then `w = 1/(…)^p` — both DPCC weights AND iMF adaptive weight applied | DPCC weights: `get_loss_weights` (146). iMF adaptive: `imf.py:380-386`. Neither upstream stacks both. |
| 11 | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 590 | `p=meanflow_adaptive_p (0.5)`, `c (1e-3)`, on per-sample **mean** (589) | iMF `imf.py:71-72` (`norm_p=1.0, norm_eps=0.01`) + `380-386` (on per-sample **sum**) |
| 12 | `imf_diffusion.py` · **`q_sample`** · 193-199 (`(1−t)·noise + t·x_start`) + **`_p_losses_meanflow_jvp`** docstring · 509-522 (derivation) | our t=0-noise→t=1-data convention and the START-anchored identity | iMF `imf.py:350` (`z_t=(1−t)x+t·e`, i.e. t=0 data) + `sample_one_step:114`. **Opposite** time direction — verified self-consistent. |
| 13 | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 534,544,549,582 (calls `apply_conditioning(..., noise=True/False)`) → `helpers.py` · **`apply_conditioning`** · 143-161 | observed dims pinned in x_r and zeroed in tangent+target | DPCC-only inpainting pattern; iMF has none |
| 14 | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py:143-144` (`fm_model = fm_experiment.ema if use_ema else …`); flag in `config/avoiding-d3il.py` · `plan_fm_v3_imeanflow` · `eval_use_ema` | which weights sampling uses | iMF `imeanflow/utils/sample_util.py` (defaults `ema=True`) |
| 15 | `imf_dit_trajectory.py` · **`IMFDiTTrajectory.__init__`** · 247-314; sizes in `config/avoiding-d3il.py` · `flow_matching_v3_imeanflow` block · `dit_*` (504-510) | hidden 256 / depth 8 / 4 heads / tokens 2·2·1 | iMF `imfDiT.py:393-427` (the `imfDiT_B/M/L/XL_2` presets) |
| 16 | `imf_dit_trajectory.py` · **`precompute_rope_cos_sin`** · 123-132 + **`apply_rotary_pos_emb`** · 135-150 | real-valued interleaved rotation | iMF `imfDiT.py` · `precompute_rope_freqs` · 370 + `apply_rotary_pos_emb` · 379 (complex bitcast). Verified equivalent. |
| 17 | `imf_diffusion.py` · **`_p_losses_meanflow_jvp`** · 581-586 | `u_target = v_inst + h·du_dr` (detached), then `delta = u_pred − u_target` | iMF `imf.py:375-385` (`V = u + (t−r)·sg(du_dt)`, loss vs `v_g`). Verified same gradient. |

**Fastest way to self-check the kill (rows 1–3), no cluster:**
```bash
cd /workspaces/FM-PCC
# gate #1 + #2 live here — read _predict_velocity in full:
sed -n '166,191p' flow_matcher_v3_imeanflow/models/imf_diffusion.py
# proof the uncond branch is never trained — expect ZERO hits:
grep -n 'force_dropout' flow_matcher_v3_imeanflow/models/imf_diffusion.py | grep -i loss
grep -n 'use_dropout' flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py   # only the signature
# the untrained null token:
sed -n '331,353p' flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py
# compare to how iMF DOES train it:
sed -n '263,293p' /workspaces/aux_repo/imeanflow/imf.py       # cond_drop
sed -n '90,115p'  /workspaces/aux_repo/imeanflow/imf.py       # sample_one_step (single call, no mixing)
```

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
