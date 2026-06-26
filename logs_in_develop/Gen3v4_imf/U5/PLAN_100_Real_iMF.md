# U5 — PLAN: 100% Real iMF on a UNet backbone (rises from U4)

**Date:** 2026-06-15
**Premise:** U4 landed the *real* MeanFlow-Identity objective (JVP, sign-verified). U5 closes the
**remaining faithfulness gaps** so it becomes the **full real iMF method** — kept on the **UNet**
backbone, with a **placeholder hook** to later swap in the proposed iMF NN (DiT).
**Schedule:** stays the **Beta** schedule (config-controlled — you flip to uniform via params; not a
code task). **Only 2 phases: (1) coding, (2) train/eval vs FM.**
**Companion:** [NEXT_STEPS.md](./NEXT_STEPS.md) · audit [U4/fix_1/AUDIT_crosscheck.md](../U4/fix_1/AUDIT_crosscheck.md).

---

## 0. Cross-check vs `/workspaces/imeanflow` — what "real iMF" requires

From the official `imfDiT.forward(x, t, h, w, t_min, t_max, y)` (`models/imfDiT.py:353-390`):

| Official iMF ingredient | Evidence | Our status after U4 |
|---|---|---|
| MeanFlow-Identity target via JVP (stop-grad) | JAX (paper) | ✅ real, verified (U4) |
| `r=t` anchor + adaptive loss weight | paper | ✅ 25%, p=0.5 |
| Conditions on **`h=t−r`** (not t) | `:370` *"only on h = t − r"* | ✅ UNet has `h_mlp`; we *also* feed `t` (richer, fine) |
| Conditions on **`omega, t_min, t_max`** (interval-CFG) | `:342-344` tokens | ❌ **missing → Phase 1** |
| **Shared backbone → `u_heads` + `v_heads`** | `:374-388` | ⚠️ orphan aux MLP, not shared → **Phase 1** |
| `v` head **dropped at eval** | `eval_mode` | ✅ sampler already u-only |
| Interval-CFG guided sampling | eval cmds (`--cfg-omega`, `--interval-min/max`) | ❌ **missing → Phase 1** |
| Few-step jump sampler | `imf.py:135-136` | ✅ structurally correct |

**Conclusion:** exactly **two** ingredients are missing — the **shared-backbone v-head** and
**interval-CFG** (conditioning + guided sampling). Both are bounded code changes on the UNet. The
UNet is **not** a barrier (InstanceNorm → JVP-safe; already conditions on `h`). Everything else
already rose from U4.

---

## Phase 1 — Coding (everything, one pass)

> Flag-gated under `imf_objective='meanflow_jvp'`; `fm_equivalent` stays byte-for-byte unchanged.

**1a. Backbone abstraction + placeholder for the real iMF NN.**
Introduce a thin `IMFBackbone` boundary: `forward(x, h, cond, t=None, omega=None, t_min=None,
t_max=None) -> (u, v)`. Provide the **UNet implementation now**; leave a clearly-marked
`# TODO(real-iMF-NN): DiT dual-head drop-in` stub so the proposed iMF architecture can replace the
UNet later without touching the objective or sampler.
- Files: `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py`, `imf_engine.py`.

**1b. Faithful dual-head (shared backbone u + v).**
Replace the detached `aux_head` MLP (`imf_trajectory_model.py:46-66`) with a **v-head that branches
off `velocity_net`'s shared features**, trained on the stable FM target `data − noise`. This is the
official `u_heads`/`v_heads` split. Then `meanflow_aux_weight` actually regularizes the field
(today it does nothing — audit aux note).
- Files: `imf_trajectory_model.py` (head wiring), `imf_diffusion.py:_p_losses_meanflow_jvp` (v-loss).

**1c. Interval-CFG (the published-quality lever).**
- **Conditioning:** add sinusoidal embeddings for `omega, t_min, t_max`, summed into the time/h
  embedding exactly like the existing `h_mlp` (`unet1d_temporal_cond.py:211`).
- **Sampling:** `u_cfg = u_uncond + omega·(u_cond − u_uncond)`, applied **only inside the guidance
  interval** `[t_min, t_max]`.
- Files: `unet1d_temporal_cond.py` (3 embeds), `imf_diffusion.py:p_sample_loop`, both configs
  (`omega`, `t_min`, `t_max`, default omega=0 ⇒ off ⇒ unchanged).

**1d. Keep (no code):** the U4 JVP objective + `r=t` anchor + adaptive weight; and the **Beta
schedule** (`time_beta_alpha_v3/beta_v3` — you set `1.0/1.0` for uniform when wanted).

**Safety / correctness notes:**
- Forward-AD (JVP) survives the UNet (InstanceNorm, Mish/SiLU — verified). Hold `omega/t_min/t_max`
  **constant** through the JVP closure (they are CFG knobs, not differentiated inputs).
- All new conditioning defaults to *off* so existing runs are unaffected.

**Phase 1 done when:** code runs, `py_compile` clean, dual-head + CFG present, placeholder stub in,
defaults reproduce current behaviour.

---

## Phase 2 — Train / Eval / Compare to FM

1. **Train** `imf_objective='meanflow_jvp'` on the Beta schedule (set `1.0/1.0` for uniform if you
   want the broad-interval coverage the audit recommends; your call via params).
2. **Eval** at **1 / 2 / 4 NFE** with interval-CFG (omega sweep, e.g. 0 vs ~4–8).
3. **A/B vs FM** — same data/seeds, three columns:
   (i) FMv3ODE @10 NFE · (ii) FM-equivalent iMF @10 NFE · (iii) **real iMF @1–2 NFE + CFG**.
   Report **quality** *and* **`fm_ms`** (latency).
4. **Success = real iMF @1–2 NFE matches FM @10 NFE quality at a fraction of the latency.**

> (DPCC projector low-NFE re-tune remains a separate domain gate for the *avoiding* constraint task;
> not part of the iMF-method comparison above.)

---

## Placeholder contract — the real iMF proposed NN

`IMFBackbone` is the single swap point. UNet today; the official-style **DiT dual-head** (shared
blocks → `u_heads`/`v_heads`, RoPE, conditions on `h, omega, t_min, t_max, y`) drops in later by
implementing the same interface — **no change to the objective, JVP, sampler, or configs.** That is
how U5 stays "100% real iMF *method*" now while leaving the proposed *architecture* as a clean
future swap.

## What U5 deliberately does NOT do
- ❌ Change the schedule in code (Beta kept; uniform is a param flip).
- ❌ Replace the UNet with a DiT now (placeholder only).
- ❌ Re-litigate the JVP sign (verified, audit §B).
