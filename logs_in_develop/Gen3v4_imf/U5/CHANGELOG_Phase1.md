# U5 Phase 1 — CHANGELOG (real-iMF on UNet: dual v-head + interval-CFG + placeholder)

**Date:** 2026-06-15
**Plan:** [PLAN_100_Real_iMF.md](./PLAN_100_Real_iMF.md) Phase 1.
**Status:** Code complete, all `py_compile` clean, **untested** (no local runtime).
**Default behaviour: byte-for-byte unchanged** — every addition is gated OFF by default
(`dual_head=False`, `interval_cfg=False`, `meanflow_cfg_omega=0`). Schedule kept as Beta (config).

---

## Files touched

| File | Change |
|---|---|
| `flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py` | **1b** shared-backbone `v_final_conv` (reads the same post-up `trunk` as `final_conv`) returned via `return_v`. **1c** optional `omega/tmin/tmax` sinusoidal embeds summed into the time/h embedding (mirrors `h_mlp`). Gated by new `dual_head`/`interval_cfg` ctor flags. |
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | **1a** documented as the `IMFBackbone` swap point + `# TODO(real-iMF-NN)` DiT placeholder. Routes `v` from the shared backbone when `dual_head` (else legacy orphan aux MLP). Threads `omega/t_min/t_max`. |
| `flow_matcher_v3_imeanflow/models/imf_engine.py` | Threads `dual_head`/`interval_cfg` to the backbone; `forward_train` forwards `omega/t_min/t_max`. |
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | New ctor knobs `meanflow_cfg_omega/t_min/t_max` (+storage). `_predict_uv`/`_predict_velocity` thread CFG; **1c** interval-CFG guided sampling in `p_sample_loop` (`u_cfg = u_uncond + ω·(u_cond−u_uncond)`, applied only for `τ∈[t_min,t_max]`). Train-time CFG conditioning in `_p_losses_meanflow_jvp` (held **constant** through the JVP). Aux loss now hits the **shared** v-head when `dual_head`. |
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | Forwards `dual_head`/`interval_cfg` → `model_config`; `meanflow_cfg_*` → `diffusion_config` (all `getattr`-defaulted). |
| `config/avoiding-d3il.py` | New keys in the `flow_matching_v3_imeanflow` train block **and** `plan_fm_v3_imeanflow` block: `dual_head`, `interval_cfg`, `meanflow_cfg_omega/t_min/t_max` (all OFF). |

---

## What is now real (vs official `/workspaces/imeanflow`)

- ✅ MeanFlow-Identity JVP objective (U4) — unchanged, sign-verified.
- ✅ **Shared-backbone v-head** — the official `u_heads`/`v_heads` split (was an orphan MLP). With
  `dual_head=True`, `meanflow_aux_weight>0` now actually regularizes the u-trunk.
- ✅ **Interval-CFG** — `(omega, t_min, t_max)` conditioning + guided sampling restricted to the
  interval (matches official eval recipe).
- ✅ **UNet kept** + `IMFBackbone` placeholder for the future DiT NN (same `forward→(u,v)` contract).
- ➖ Beta schedule unchanged (your `time_beta_*=1.0/1.0` flip for uniform, per audit).

## How to turn it on (a real-iMF run)

Training block (`flow_matching_v3_imeanflow`):
```python
'imf_objective': 'meanflow_jvp',
'dual_head': True,
'meanflow_aux_weight': 0.05,     # now meaningful (shared backbone)
'interval_cfg': True,            # only if you want CFG; then also set ω at plan time
# optional uniform schedule: 'time_beta_alpha_v3': 1.0, 'time_beta_beta_v3': 1.0
```
Plan block (`plan_fm_v3_imeanflow`) — **must match the trained flags**:
```python
'dual_head': True, 'interval_cfg': True,
'meanflow_cfg_omega': 4.0, 'meanflow_cfg_t_min': 0.4, 'meanflow_cfg_t_max': 0.6,
'flow_steps_v3': 4,              # or 1–2
```

## Required cluster verification (cannot run here)

1. **Forward-AD still OK** — JVP now also linearizes through the v-head branch + CFG embeds; UNet is
   InstanceNorm (safe), but confirm `torch.func.jvp` runs.
2. **Dual-head parity** — `dual_head=True, ω=0` should ≈ the U4 single-head result (v-head only adds
   the aux regularizer); large divergence ⇒ check the shared-trunk wiring.
3. **CFG sanity** — ω sweep at 1–2 NFE should monotonically trade diversity/quality.
4. Defaults (all OFF) must reproduce the current `fm_equivalent`/U4 numbers exactly.

## Not done (per scope)
- Gen8 visual mirror (separate fork; not part of this Gen3v4 U5 coding).
- DPCC low-NFE re-tune (Phase 4 domain gate, tracked separately).
- No commit/push.
