# CHANGELOG — U10: faithful improved-MeanFlow objective (`imf_objective='imf_official'`)

**Date:** 2026-07-13 · **Gen:** Gen3v4_imf / U10 · **Plan:** `PLAN_faithful_imf_replication.md`
**Goal:** replicate `imeanflow/imf.py` math 1:1 so the claim "we faithfully use iMF" is true. Legacy arms kept for A/B (checkpoints auto-separate via `obj{...}_bb{...}` prefix). **Nothing runs locally — validate on cluster.**

## Files changed (6)

| File · symbol | Change |
|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py · iMeanFlowODE` | **New `_p_losses_imf_official`** (W3/W4/W6/W7); `loss()` dispatch (W1); `__init__` +3 knobs; `_sample_cfg_scale(..., s_max=)` (W2); `p_sample_loop` CFG-as-net-input branch (W8) |
| `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py · IMFDiTTrajectory._build_sequence` | `force_dropout` now accepts a **per-sample bool tensor** (W5) — required for `cond_drop`; plain bool still works |
| `config/avoiding-d3il.py` (train + plan blocks) | `imf_objective: imf_official`; `imf_backbone: unet→dit` (required); new `meanflow_cfg_smax/data_proportion/class_dropout_prob`; plan eval point `meanflow_cfg_omega 0.0→1.0` (1=off), `eval_use_ema False→True` |
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | pass the 3 new kwargs into `diffusion_config` |

## What `imf_official` fixes (vs legacy `meanflow_jvp`) — the audited deviations

- **D1 (critical):** JVP z-tangent is the **predicted `v_c`** (via `torch.func.jvp(..., has_aux=True)`), not analytic `v_inst`. *This is the line that makes it improved-MeanFlow instead of vanilla.*
- **D2:** guided target `v_g` (`guidance_fn` ported) + **`cond_drop` trains the null token** (per-sample W5); the broken output-space CFG mixes are **deleted** for this objective — CFG is now a net input (ω=1 off, ω>1 guides), no untrained branch to poison sampling.
- **D3:** two **independent** logit-normals + 50% FM anchors (`data_proportion=0.5`).
- **D3b (NEW bug found while coding):** legacy `loss()` drew `sigmoid(randn·σ + p_mean)` on our τ-axis → mass near **noise**. Faithful axis is `−p_mean` → `sigmoid(N(+0.4,1))`, mass near **data**. Fixed in the new branch.
- **D5:** official loss `adp(loss_u)+adp(loss_v)`, `p=1.0/eps=0.01`, per-sample **SUM**, **no DPCC `loss_weights`**.
- **Faithful, unchanged:** sampler (`z±h·u`, single call), DiT port, time-flip/RoPE/loss-algebra (verified equivalences).

## Verification

- ✅ `py_compile` on all 6 files.
- ✅ Method re-read line-by-line vs `imf.py`: tangent=`v_c.detach()`, u-query uses **ungated** ω + **dropped** labels, v-head dummies `h=0,t_min=0,t_max=1`, `v_g←v_t` on dropped/FM rows, interval gate on `s_anchor=1−r` (official s-convention).
- ⏳ **Cluster-only (run on i6-gpu-1):** per PLAN §6 — (1) τ-histogram mass near data; (2) short train: `loss_u/loss_v` finite & falling, **null-token grad-norm > 0** (old bug = exactly 0); (3) eval N∈{1,2,10,50}, ω=1, EMA — N=1/2 coherent (the capability the old arm lacked), N=10 ≥ old arm; (4) A/B `imf_official` vs `meanflow_jvp` vs UNet-FM/DPCC.

## Notes / risk
- `torch.func.jvp(has_aux=True)` must keep the primal grad-connected for the outer backward — the legacy arm already relies on jvp→params→backward, so the pattern holds; the extra aux is standard. Confirm on first cluster run.
- 3 extra v-head forwards/step (official pays the same; grad-free via `no_grad`). Batch cond/uncond later if a bottleneck.
- **Out of scope:** UNet arm (not iMF), Gen8 `imf_visual_aligning` sync (do only after cluster validation), DiT scale-up.
- Uncommitted (user commits manually).
