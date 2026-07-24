# Investigation — `diffuser` variant produces exploding trajectories (normalization audit)

**Generation:** Gen3v4_imf / U9
**Date:** 2026-07-13
**Symptom:** In eval, the `diffuser` projection variant (i.e. **projection disabled**) draws trajectory lines that explode across the whole plot. Projected variants (`dpcc*`) look fine. Training curves all look sane.
**Hypothesis under test (user):** the normalize / unnormalize boundary is the culprit.

---

## TL;DR verdict

**Normalization is very unlikely to be the *root cause* of the explosion, and the normalize/unnormalize code is a faithful copy of DPCC.** The audit found:

1. **DPCC/diffuser upstream DID normalize/unnormalize**, and our `flow_matcher_v3_imeanflow/sampling/policies.py` is a byte-for-byte faithful copy of it (only cosmetic diffs). No normalization bug was introduced at this layer.
2. **The imeanflow upstream does NOT have any trajectory normalizer** — it is an *image-generation* codebase (VAE latents, FID/IS, `Normalize(0.5, 0.5)`). Only the *velocity/mean-flow parametrization* was grafted into DPCC's pipeline; DPCC owns all trajectory normalization. So there is **no competing normalization convention** fighting at the boundary.
3. **The aligning task uses `LimitsNormalizer`, which *clips* to `[-1, 1]` on `unnormalize`** (`normalization.py:168-170`). That mathematically **bounds** both unnormalized observations and actions to the training-data range — so the unnormalization step *cannot itself* turn a finite trajectory into screen-spanning lines.

The explosion therefore almost certainly originates **upstream of unnormalization**, in the *generative ODE output itself*, and is exposed **only in `diffuser`** because that is the one variant with **no projection to sanitize / rebound the trajectory** (`eval...:249` → `projector = None if variant == 'diffuser'`). The single realistic path by which a bad ODE output survives to the screen is **NaN/Inf passthrough** (there is **no finite-guard anywhere** — see §4), because `np.clip(nan) = nan` defeats the LimitsNormalizer clamp.

---

## 1. Three-way normalization comparison

| Layer | DPCC upstream (`aux_repo/dpcc/diffuser/sampling/policies.py`) | Gen3v4_imf (`flow_matcher_v3_imeanflow/sampling/policies.py`) | imeanflow upstream (`aux_repo/imeanflow/`) |
|---|---|---|---|
| Normalize **conditions/obs** before model | `_format_conditions` → `normalizer.normalize(..., 'observations')` (L103-108) | **Identical** (L97-108) | **None** — no trajectory normalizer exists |
| Unnormalize **observations** out | `normalizer.unnormalize(..., 'observations')` (L59, L62) | **Identical** (L53, L56) | n/a |
| Unnormalize **actions** out | `normalizer.unnormalize(actions, 'actions')` (L89) | **Identical** (L83) | n/a |
| Normalizer class (aligning) | `LimitsNormalizer` (maps data→`[-1,1]`, **clips** on inverse) | **same** (`config/aligning-d3il-visual.py:185,303`) | Image `Normalize(0.5,0.5)` + fixed VAE mean/std (`utils/data_util.py:39`, `utils/vae_util.py:54-57`) |

**Only differences** between our `policies.py` and DPCC's:
- L70 `self.prev_observations = ... observations[which_trajectory] ...` vs DPCC's `observations[0]` (tagged `MPC_NPZ_PATCH`) — cosmetic, only affects temporal-consistency ordering.
- The `GaussianInvDynDiffusion` inverse-dynamics branch is dropped (not used here).

→ **No normalization semantics changed.** Whatever DPCC's `diffuser` baseline did, ours does too.

---

## 2. The actual data-flow (annotated)

```
obs (env, raw)
  └─ normalize('observations')            policies.py:99   → conditions in normalized space
     └─ p_sample_loop:
          x0 = torch.randn(shape)  (N(0,1), σ=1)           imf_diffusion.py:217   ← t=0 prior
          for step: x = x + v·dt   (Euler / torchdiffeq)   imf_diffusion.py:321/327
                    v = u-head(x, t, h)  (NO finite guard)  imf_diffusion.py:179
          → x ≈ normalized data in [-1,1]  (IF model good)
     └─ unnormalize('observations')  ── LimitsNormalizer CLIPS to [-1,1] ── normalization.py:168-175
     └─ unnormalize('actions')       ── same clip ──                        normalization.py:164-175
action → env.step                                          eval...:320-345
```

Flow convention (confirmed): `q_sample = (1-t)·noise + t·x_start` (`imf_diffusion.py:199`), **t=0 = noise, t=1 = data**; sampler integrates `x += v·dt` from 0→1 (`:266-327`). Training seeds noise with `randn` σ=1 (`:421`), matching the sampler prior (`:217`). So train and sample priors agree — no scale mismatch there.

---

## 3. Why unnormalization can't be the amplifier (for aligning)

`LimitsNormalizer.unnormalize` (`normalization.py:164-175`):

```python
if x.max() > 1+eps or x.min() < -1-eps:
    x = np.clip(x, -1, 1)          # <-- hard clamp
return x*(maxs-mins) + mins        # result ∈ [mins, maxs]  (the TRAINING data range)
```

So **for finite input**, the unnormalized obs/action is *guaranteed* to lie inside the dataset's own `[mins, maxs]` box — it cannot draw a line "across the whole screen" beyond the data range. The projected variants and the `diffuser` variant pass through the *same* clamp; if unnormalization were the amplifier, **all** variants would explode. They don't. This exonerates the unnormalize step as the direct cause.

**Caveat — the clamp is defeated by NaN/Inf:** `np.clip(nan, -1, 1) == nan`, and `nan*(maxs-mins)+mins == nan`. A NaN obs/action then goes to `env.step`, which typically diverges or returns garbage positions → screen-spanning lines in the `obs_buffer` plot (`eval...:373`). See §4.

---

## 4. The real suspects (ranked)

**(A) — Unguarded generative ODE + no projection safety net. [most likely]**
`diffuser` is the *only* variant with `projector = None` (`eval...:249`). In every other variant, `projector.project(...)` / `compute_gradient(...)` runs near the end of sampling (`imf_diffusion.py:331-350`) and **snaps `x` back into the constraint-feasible, bounded set each step** — which incidentally also rescues a drifting/degenerate MeanFlow trajectory. Remove it and the raw few-step ODE output is exposed directly to unnormalize+env.
There is **no `isnan`/`isfinite`/`nan_to_num`/`clamp` guard anywhere** in `imf_diffusion.py` or `policies.py` (the only `clip*` hit is `clip_denoised`, a **dead DDPM flag**, unused on the FM path). So a single NaN/Inf from the u-head (out-of-distribution `x` at few steps, or CFG amplification `(1+w)·v − w·v_uncond` at `:184/:190`) propagates unchecked → defeats the clamp → env divergence.

**(B) — MeanFlow few-step sampling correctness (independent of normalization). [likely co-factor]**
The huge `(t,h)`-domain commentary in `imf_diffusion.py:230-320` documents a *recurring* struggle to keep sampler queries in-domain (`t+h ≤ 1`). If few-step Euler/torchdiffeq lands `x` well outside `[-1,1]` at t=1, LimitsNormalizer clips it to a **corner of the data box** every MPC step → the robot is repeatedly commanded toward an extreme corner → looks like an "explosion" toward the plot edges even *without* any NaN. This is a *model/sampler* issue, not a normalization issue, but it manifests at the plot.

**(C) — Normalizer stats mismatch train vs eval. [low, but cheap to rule out]**
Eval rebuilds the normalizer via `dataset = dataset_config()` from the saved `dataset_config.pkl` (`eval...:93,127`), i.e. stats are **recomputed from the dataset at eval time**, not loaded from the training checkpoint. If the eval dataset differs from training's (different D3IL split/version), `mins/maxs` differ → the `[-1,1]→[mins,maxs]` mapping is miscalibrated. This *distorts* trajectories but is still **bounded** (can't explode unless the stats themselves are absurd). Worth a 1-line check anyway.

---

## 5. Concrete checks to run **on cluster** (fast, targeted)

1. **NaN/Inf probe (settles A immediately).** In `policies.py.__call__`, right after `samples, infos = self.model(...)` (L46), add a temporary:
   ```python
   assert np.isfinite(utils.to_np(samples)).all(), f"non-finite in raw samples: nan={np.isnan(utils.to_np(samples)).sum()}"
   ```
   Run only the `diffuser` variant. If it trips → confirms (A): fix by adding a finite-guard / `torch.nan_to_num` + a soft clamp on `x` inside the sampler loop, **or** simply accept that `diffuser` (no brakes) is expected to be unstable for this MeanFlow checkpoint.

2. **Print pre-clip range.** Temporarily log `x.min()/x.max()` of the normalized `samples` before unnormalize. If it's finite but e.g. `±50`, that's suspect (B): the model is not landing in `[-1,1]` — a sampler/`flow_steps`/`(t,h)` problem, not normalization. If it's `±1`-ish and env still explodes, look at the action-space mapping / env dynamics.

3. **Sanity: does DPCC's own `diffuser` baseline also explode?** The unprojected diffusion baseline in stock DPCC is known to violate constraints but should stay *bounded*. If DPCC-`diffuser` is bounded but ours explodes with the *same* `policies.py`, the delta is 100% the **imeanflow engine's generative output** (A/B), conclusively not normalization.

4. **Stats check (rules out C).** Print `dataset.normalizer.normalizers['observations'].mins/maxs` at eval and compare to the values implied by `dataset_config.pkl` used in training. Expect identical.

---

## 6. Bottom line for the user

- Your instinct to check normalization was worth doing, but the evidence clears it: **the normalize/unnormalize path is DPCC's, faithfully copied, and `LimitsNormalizer` clips — it bounds, it does not amplify.** imeanflow contributes *no* trajectory normalization to conflict with it.
- The explosion is gated on **"projection off"**, and the one mechanism that lets a bad generative step reach the screen is **unguarded NaN/Inf** (there is no finite-guard) — with **few-step MeanFlow sampling drift** as the underlying generator of those bad steps.
- **Fastest disambiguation:** the finite-assert in check #1. It will tell you in one run whether this is NaN-passthrough (add a guard) or bounded-but-degenerate sampling (a `flow_steps`/`(t,h)`/checkpoint-quality problem).

---

### Appendix — file:line index
- `flow_matcher_v3_imeanflow/sampling/policies.py:53,56,83,99` — normalize/unnormalize calls
- `flow_matcher_v3_imeanflow/models/imf_diffusion.py:179-191` — u-head velocity + CFG (amplification site, no guard)
- `imf_diffusion.py:199,217,321,327` — flow convention, prior, Euler/torchdiffeq step
- `imf_diffusion.py:331-350` — projection block (skipped when `projector=None`)
- `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py:249` — `projector = None if variant=='diffuser'`
- `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py:373,383` — trajectory plotting (where explosion is seen)
- `aux_repo/dpcc/diffuser/sampling/policies.py:59,62,89,104` — DPCC upstream (identical)
- `aux_repo/dpcc/diffuser/datasets/normalization.py:152-175` — `LimitsNormalizer` (clip on inverse)
- `config/aligning-d3il-visual.py:185,303` — `'normalizer': 'LimitsNormalizer'`
- `aux_repo/imeanflow/utils/data_util.py:39`, `utils/vae_util.py:54-57` — imeanflow is an image model (no traj normalizer)
