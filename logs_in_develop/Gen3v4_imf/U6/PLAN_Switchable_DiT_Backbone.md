# U6 — PLAN: Config-Switchable Original DiT Backbone (rises from U5)

**Date:** 2026-06-16
**Premise:** U5 made the iMF *method* real on the UNet and left a single, clean swap point — the
`IMFBackbone` contract `forward(x, t, h, cond, ω, t_min, t_max) -> (u, v)`
(`imf_trajectory_model.py:11`, the `# TODO(real-iMF-NN)` stub). U6 **cashes in that placeholder**: add
a config switch that selects the backbone — keep the UNet as default, or drop in the **original iMF
DiT** (`/workspaces/imeanflow/models/imfDiT.py`) — **without touching the objective, JVP, sampler, or
DPCC path.**
**Motivation:** [UNet_vs_DiT_for_iMF_Principle](../U5/UNet_vs_DiT_for_iMF_Principle.md) showed the
UNet's principled iMF weakness is the **conditioning bottleneck** + **shallow heads**; the DiT closes
both natively. U6 makes that an A/B you can flip, not a rewrite.
**Scope:** concepts and config surface only — no implementation in this doc.

---

## 0. Design rule (the whole plan in one sentence)

> The backbone is the **only** thing that changes. Everything downstream of
> `IMFBackbone.forward(...) -> (u, v)` — the MeanFlow-JVP loss, the stop-grad target, interval-CFG, the
> Euler sampler, the DPCC projector — must run **byte-for-byte identically** whether the backbone is the
> UNet or the DiT. If a change is needed outside the backbone, the contract was wrong, not the plan.

---

## 1. The switch point (recap of the U5 boundary)

U5 already isolated the network behind one interface inside `iMFTrajectoryModel`:

```
IMFBackbone.forward(x, t, h, cond, omega, t_min, t_max, return_v) -> (u, v)
```

Today that resolves to `Flow_matcher_U_Net_v2`. U6 turns the resolution into a **dispatch** on a new
config key, with the UNet as the default branch and a new **trajectory-DiT** as the alternate branch.
No caller of `forward(...)` learns which branch ran.

---

## 2. Config surface (the new keys — naming only)

Add to **both** the train block (`flow_matching_v3_imeanflow`) and the plan block
(`plan_fm_v3_imeanflow`), so training and eval agree:

| Key | Role | Default |
|---|---|---|
| `imf_backbone` | `'unet'` \| `'dit'` — selects the `IMFBackbone` implementation | `'unet'` (unchanged) |
| `dit_depth` | total transformer blocks | (DiT-only) |
| `dit_hidden_size` | token width | (DiT-only) |
| `dit_num_heads` | attention heads | (DiT-only) |
| `dit_aux_head_depth` | private blocks per `u`/`v` head (the Stress-C lever) | (DiT-only) |
| `dit_condition_on_t` | feed `t` as a token, or follow official `h`-only | `False` (official recipe) |

**Defaults keep `imf_backbone='unet'`** ⇒ every existing run is unaffected. The DiT keys are read with
`getattr` fallbacks and ignored entirely when the UNet branch is active.

**Folder-name + loadpath plumbing (mandatory):** add `('imf_backbone', 'bb')` to the train watch list
(`args_to_watch_fmv3_imf_train`) and inject `_bb{imf_backbone}` into the plan block's
`diffusion_loadpath`/`prefix`. A DiT checkpoint and a UNet checkpoint must live in **distinct folders**
and the planner must resolve the matching one — otherwise eval silently loads the wrong architecture.

---

## 3. Adapting the image-DiT to trajectories (concepts)

The official `imfDiT` is built for **images** (`PatchEmbedder` over a 2D grid, `LabelEmbedder` for
ImageNet class `y`, patchify/unpatchify). A **trajectory-DiT** must re-interpret these for a
`[B, H, D]` sequence. The conceptual mapping:

| Image-DiT ingredient | Trajectory re-interpretation |
|---|---|
| 2D patches of an image | **Per-timestep tokens** of the trajectory (`H` tokens, "patch_size=1"; each step's `D`-vector → one token) |
| `PatchEmbedder` (conv patchify) | A linear lift `D → hidden_size` per timestep |
| RoPE over 2D spatial positions | **RoPE over the horizon axis** (1D positional structure of `H` steps) |
| `LabelEmbedder(y)` class conditioning | **No class label.** Map to FM-PCC's conditioning: the pinned-observation `cond` and/or `returns`. A null/learned token when unconditional (this is also how CFG dropout is realized) |
| `unpatchify` → image | Linear projection `hidden_size → D` per timestep → `[B, H, D]` |
| `eval_mode` drops `v_heads` | Same — instantiate `v_heads` only at train; sampler uses `u` only |

Everything iMF-specific is **already a token** in the official model and maps directly: `h`, `ω`,
`τ_min`, `τ_max` each keep their **own learnable conditioning tokens** prepended to the sequence
(`imfDiT.py:333-344`) — which is exactly the **Stress-A** fix the UNet lacks.

---

## 4. Contract-conformance requirements (the non-negotiables)

For the DiT branch to be a true drop-in, it must satisfy:

1. **Signature & output.** Accept `(x, t, h, cond, ω, t_min, t_max, return_v, force_dropout)` and return
   `u` (and `(u, v)` when `return_v=True`). Internally it may ignore `t` if `dit_condition_on_t=False`
   (official recipe), but it must **accept** the argument so the JVP closure is unchanged.
2. **JVP-safety (the hard gate).** The MeanFlow objective differentiates the backbone with
   `torch.func.jvp` (forward-mode AD) through `(z, r, h)`. The DiT must be **functionally pure** under
   that transform:
   - **No batch-coupled norms.** DiT uses **RMSNorm** (per-token) — JVP-safe, like the UNet's
     InstanceNorm. (A vanilla DiT with adaLN over batch stats would *not* be — confirm none is used.)
   - **CFG knobs `(ω, t_min, t_max)` held constant** through the JVP, identical to U5 — they are
     conditioning, not differentiated inputs.
   - **Dropout / CFG-mask determinism inside the JVP** — the conditioning dropout must not inject
     randomness on the differentiated path (same care as the UNet branch).
   - **Verify on cluster** with the existing 1-NFE reconstruction check; a JVP failure here is the
     single most likely blocker.
3. **CFG mechanism parity.** Unconditional prediction (`force_dropout=True`) must zero the *content*
   conditioning (cond/returns) while **keeping** the interval tokens — so interval-CFG
   `u_cfg = u_uncond + ω·(u_cond − u_uncond)` behaves as in U5.
4. **Device / dtype / EMA** behavior identical to the UNet branch (the trainer and EMA wrap the
   diffusion module, not the backbone — should be free, but confirm parameter registration).

---

## 5. Checkpoint & state-dict implications

- **No cross-loading.** A UNet checkpoint and a DiT checkpoint have disjoint parameter trees; the
  existing legacy-remap logic (`imf_diffusion.py:_remap_state_dict_for_compatibility`) is UNet-specific.
  U6 must ensure the planner **never** tries to load a DiT checkpoint into a UNet model or vice-versa —
  the `_bb{imf_backbone}` folder tag (§2) is the guard. Fail loudly on mismatch.
- **Self-describing checkpoints.** The saved `model_config.pkl` already records constructor args; adding
  `imf_backbone` + `dit_*` there makes each checkpoint declare its own architecture for safe reload.

---

## 6. Phases

**Phase 1 — Switch plumbing (UNet still the only real branch).**
Add `imf_backbone` dispatch + config keys + watch/loadpath tags. DiT branch is a stub that raises
"not yet implemented." `imf_backbone='unet'` reproduces U5 exactly. *Done when:* default runs unchanged,
folder names carry `_bb`.

**Phase 2 — Trajectory-DiT implementation behind the contract.**
Implement the trajectory re-interpretation (§3) satisfying the §4 contract. Keep depth/heads small first
(trajectories are `H=8`; do not copy ImageNet-scale `depth=28, hidden=1152`). *Done when:* `py_compile`
clean, forward returns `(u, v)`, shapes match the UNet branch for the same inputs.

**Phase 3 — JVP-safety + parity validation (cluster).**
1. **JVP runs** through the DiT (forward-mode AD; the §4.2 gate).
2. **1-NFE reconstruction** sanity, same check U4/U5 used.
3. **Dual-head + interval-CFG** behave (ω sweep monotonic).
*Done when:* DiT trains the `meanflow_jvp` objective without AD errors and CFG responds sanely.

**Phase 4 — A/B: DiT vs UNet at matched objective.**
Same data, seeds, schedule, NFE, projector. Three-way compare:
(i) UNet real-iMF, (ii) DiT real-iMF, (iii) FM baseline — report **quality** *and* **`fm_ms`** at
1/2/4 NFE. Tests the U5 hypothesis that the DiT's tokenized conditioning + deep heads help low-NFE
quality where the UNet's additive-bottleneck does not.

---

## 7. Risks & mitigations

| Risk | Why | Mitigation |
|---|---|---|
| **JVP breaks on the DiT** | forward-mode AD through attention/norms is stricter than conv | RMSNorm only; deterministic dropout in the JVP closure; 1-NFE check first (Phase 3.1) |
| **DiT overkill at H=8** | image-scale depth/width wastes params, slows `fm_ms`, overfits | start tiny (`dit_depth` small, `aux_head_depth` 2–4); the receptive-field win is irrelevant at H=8 (U5 §4) |
| **Silent checkpoint mismatch** | UNet vs DiT trees disjoint | `_bb` folder tag + self-describing `model_config.pkl`; hard-fail on mismatch (§5) |
| **CFG dropout semantics differ** | tokens vs additive bias zero differently | mirror U5: drop content tokens, keep interval tokens (§4.3) |
| **DPCC schedule still 10-step-tuned** | orthogonal to backbone, but bites any low-NFE run | inherit the U5 low-NFE snap-schedule caveat; re-derive before reporting constraint sat |

---

## 8. What U6 deliberately does NOT do

- ❌ Change the objective, JVP, sampler, or DPCC code — backbone-only (the §0 rule).
- ❌ Remove or weaken the UNet — it stays the default and the A/B control.
- ❌ Port ImageNet-scale DiT hyperparameters — trajectory scale demands a small DiT.
- ❌ Re-open the FM-vs-iMF objective question (settled, U3) or the JVP sign (verified, U4).
- ❌ Commit/push (per policy).

---

## 9. Success criterion

A single config flip — `imf_backbone: 'unet' → 'dit'` (train + plan) — trains and evaluates the **same**
real-iMF method on the original DiT, with checkpoints kept separate, JVP intact, and a clean three-way
A/B that answers: **does the DiT's tokenized conditioning + deep dual heads beat the UNet at 1–2 NFE on
avoiding?** If yes, the U5 principle analysis is confirmed and the DiT becomes the default; if no, the
UNet stays and we have proof the conditioning bottleneck was not the limiter.
