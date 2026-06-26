# U6 — CHANGELOG: Config-Switchable Official-iMF DiT Backbone

**Date:** 2026-06-17
**Plan:** [PLAN_Switchable_DiT_Backbone.md](./PLAN_Switchable_DiT_Backbone.md)
**Status:** Code complete, all `py_compile` clean, **untested** (no local torch runtime — Docker is
AI-coding-only; forward + JVP must be cluster-verified).
**Default behaviour: byte-for-byte unchanged** — `imf_backbone` defaults to `'unet'` everywhere, so
every existing run is identical. The DiT only activates when you flip one config key.

---

## What this delivers

A faithful port of the official `/workspaces/imeanflow/models/imfDiT.py` transformer, adapted to 1D
trajectories, selectable via a single config flag `imf_backbone: 'unet' | 'dit'`. It slots in behind
the U5 `IMFBackbone`/`velocity_net` contract — **the objective, JVP, interval-CFG sampler, and DPCC
path are untouched.**

---

## Files touched

| File | Change |
|---|---|
| `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py` | **NEW.** The trajectory iMF DiT + all ported primitives. |
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | Added `imf_backbone` + `dit_*` ctor args; dispatch `velocity_net` to UNet **or** DiT; DiT always uses the `(u, v)` shared-head path. |
| `flow_matcher_v3_imeanflow/models/imf_engine.py` | Added `imf_backbone` + `dit_*` args; forwarded to `iMFTrajectoryModel`. |
| `flow_matcher_v3_imeanflow/models/__init__.py` | Export `IMFDiTTrajectory`. |
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | Thread `imf_backbone` + `dit_*` into `model_config` (all `getattr`-defaulted to UNet values). |
| `config/avoiding-d3il.py` | New keys in **train** + **plan** blocks; `('imf_backbone','bb')` added to the watch list; `_bb{imf_backbone}` added to plan `prefix` + `diffusion_loadpath`. |

---

## 1. `imf_dit_trajectory.py` (new) — what was ported and what was adapted

**Ported verbatim from the official repo (same architecture):**
- `RMSNorm`, `SwiGLUMlp`, `TorchLinear` (scaled-variance / zeros init) — from `torch_models.py`.
- `TimestepEmbedder` (sinusoidal → SiLU MLP) — from `embedder.py`.
- `RoPEAttention` (QK-RMSNorm + RoPE), `TransformerBlock` (**zero-init residual vector-gates**
  `attn_scale=mlp_scale=0`), `FinalLayer` (RMSNorm + zero-init linear) — from `imfDiT.py`.
- **Shared backbone → equal-depth `u_heads` / `v_heads`** split; v-head dropped at inference.
- **In-context conditioning tokens**: separate learnable tokens for `h, ω, τ_min, τ_max` (+ class),
  each summed with its embedder output and **prepended** to the content tokens.
- `ω` conditioned as `1 − 1/ω` (official recipe); scaled-variance init constants kept.

**Adapted for trajectories (`[B, H, D]` instead of `[B, C, H, W]`):**
- `PatchEmbedder` (2D conv) → **`TrajPatchEmbedder`**: a linear lift over `patch_size` consecutive
  timesteps (`patch_size=1` ⇒ one token per step). `unpatchify` reshapes back to `[B, H, D]`.
- `LabelEmbedder(class y)` → a **null-class embedding used only as the CFG content-dropout switch**:
  normal `y=0`; on `force_dropout=True` (the interval-CFG unconditional branch) `y=num_classes` (null).
  Trajectory conditioning proper is the pinned observation, applied to `x` **externally** by
  `apply_conditioning`, exactly as for the UNet.
- Conditions on `h` always; on `t` only if `dit_condition_on_t=True` (official conditions on `h` only).

**The one deliberate deviation — JVP-safe RoPE:**
- The official RoPE uses a complex bitcast `x.view(torch.complex64) * (cos + i·sin)`. That cast is not
  reliably differentiable under `torch.func.jvp` (forward-mode AD), which the **MeanFlow objective
  requires**. Replaced with a **real-valued interleaved rotation** that is *mathematically identical*:
  `out_even = x_even·cos − x_odd·sin`, `out_odd = x_even·sin + x_odd·cos`. Same rotation, same learned
  weights, AD-friendly. **No learned component changed.**

**Contract conformance (drop-in for `velocity_net`):**
```
forward(x, cond, time, returns=None, use_dropout=True, force_dropout=False,
        h=None, omega=None, t_min=None, t_max=None, return_v=False)
  -> u            (return_v=False)
  -> (u, v)       (return_v=True)
```
Matches `Flow_matcher_U_Net_v2.forward` argument-for-argument, so `iMFTrajectoryModel` calls it with no
change. `cond`/`returns`/`use_dropout` are accepted for parity (trajectory conditioning is external).

**JVP-safety design (the hard gate, to confirm on cluster):**
- RMSNorm is **per-token** (no batch coupling) — safe, like the UNet's InstanceNorm.
- Attention/softmax/SwiGLU are pointwise + matmul — safe.
- RoPE is real-valued (above) — safe.
- `(ω, τ_min, τ_max)` are captured **constants** in the JVP closure (zero tangent), so the `torch.where`
  / `clamp` in the `1 − 1/ω` guard never sees a differentiated input.

---

## 2. Backbone dispatch (`imf_trajectory_model.py`)

`iMFTrajectoryModel.__init__` now branches on `imf_backbone`:
- `'unet'` (default) → `Flow_matcher_U_Net_v2` exactly as U5 (unchanged).
- `'dit'` → `IMFDiTTrajectory(horizon, transition_dim, dit_hidden_size, dit_depth, dit_num_heads,
  dit_aux_head_depth, dit_patch_size, dit_condition_on_t)`.
- anything else → `ValueError`.

`forward` uses the shared `(u, v)` path when `dual_head OR imf_backbone=='dit'` — the DiT carries
native v-heads, so it never falls back to the legacy orphan aux MLP.

---

## 3. Config surface (`avoiding-d3il.py`)

**Train block `flow_matching_v3_imeanflow`** (new keys, all default to UNet/no-op):
```python
'imf_backbone': 'unet',      # 'unet' | 'dit'
'dit_depth': 8,
'dit_hidden_size': 256,      # small for H=8 (DiT image scale would be wasteful)
'dit_num_heads': 4,
'dit_aux_head_depth': 2,     # private blocks per u/v head
'dit_patch_size': 1,         # must divide horizon (8)
'dit_condition_on_t': False, # official recipe: condition on h only
```

**Watch list** `args_to_watch_fmv3_imf_train`: added `('imf_backbone', 'bb')` ⇒ training folder name
ends `…_obj{imf_objective}_bb{imf_backbone}`. **UNet and DiT checkpoints live in separate dirs — no
collision.**

**Plan block `plan_fm_v3_imeanflow`:** mirrored `imf_backbone` + `dit_*` (MUST equal training), and
appended `_bb{imf_backbone}` to both `prefix` and `diffusion_loadpath` so eval resolves the matching
checkpoint. The saved `model_config.pkl` also records `dit_*`, so checkpoints are self-describing.

---

## 4. How to run the DiT (a real-iMF DiT run)

Training block — flip the backbone (everything else stays U5 all-power):
```python
'imf_backbone': 'dit',
# optional: 'dit_depth': 12, 'dit_hidden_size': 384, 'dit_aux_head_depth': 4
```
Plan block — **must match** the trained flags:
```python
'imf_backbone': 'dit',
# same dit_depth / dit_hidden_size / dit_num_heads / dit_aux_head_depth / dit_patch_size / dit_condition_on_t
```
Revert to the UNet by setting `imf_backbone: 'unet'` in both blocks (the default).

---

## 5. Required cluster verification (cannot run here)

1. **JVP through the DiT** — the hard gate. Run `meanflow_jvp` with `imf_backbone='dit'`; confirm
   `torch.func.jvp` completes (real-RoPE + RMSNorm should be safe; verify no complex/in-place trips).
2. **Forward shape parity** — DiT `u`/`v` must be `[B, 8, D]`, same as the UNet, for identical inputs.
3. **1-NFE reconstruction** sanity (same check U4/U5 used).
4. **Interval-CFG** — ω sweep monotonic; `force_dropout` (null class) produces a distinct uncond branch.
5. **A/B (Phase 4)** — UNet-iMF vs DiT-iMF vs FM at 1/2/4 NFE; report quality **and** `fm_ms`.

---

## 6. Not done (per scope)
- DiT at ImageNet scale — deliberately small for `H=8` (see plan §7 risk).
- Gen8 visual fork mirror — separate fork.
- DPCC low-NFE snap-schedule re-tune — orthogonal domain gate, tracked separately.
- No commit/push (per policy).
