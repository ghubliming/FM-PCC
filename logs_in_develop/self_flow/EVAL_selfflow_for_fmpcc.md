# EVAL — Self-Flow (BFL, ICML'26) for FM-PCC

**Date:** 2026-08-09
**Upstream evaluated:** `/workspaces/aux_repo/Self-Flow` (github.com/black-forest-labs/Self-Flow, commit `8e65aef`)
**Paper:** Chefer\*, Esser\*, Lorenz, Podell, Raja, Tong, Torralba, Rombach — *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*, arXiv:2603.06507 (ICML 2026)

---

## 0. Verdict

**DO NOT ADOPT. The `aux_repo/Self-Flow` checkout can be deleted.**

Not because the paper is weak — it is a strong, well-scaled paper — but because **every one of its three components lands outside FM-PCC's operating point**:

| Self-Flow component | Status w.r.t. FM-PCC |
| :--- | :--- |
| Representation loss (EMA-teacher feature prediction) | Not in the repo (inference-only release). Would need full reimplementation from the PDF. Demonstrated at 290M–1B params / 6M–200M samples; FM-PCC's planner is ~1–2M params / hundreds of D3IL demos. |
| Dual-Timestep Scheduling (per-token heterogeneous noise) | Mechanism needs a large redundant token grid (256 tokens). FM-PCC plans are **8 tokens** (`horizon: 8`, `patch_size: 1`). Contested by arXiv:2607.02508 as data augmentation, which is exactly the part that does not survive 256 → 8. |
| Per-token adaLN conditioning (the only portable code) | **Already implemented in FM-PCC** — `flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py:69` `modulate()` accepts `(B,T,D)` token-level scale/shift. Zero to gain. |

And the decisive framing point: **Self-Flow is a training-time-only method with no effect on NFE**. It cannot help the axis FM-PCC is actually being judged on (Gen3v6/Gen3v7/Gen14 low-K sweeps, closed-loop latency vs diffusion-DPCC).

---

## 1. What the paper actually proposes

Self-Flow removes REPA's dependency on an external vision encoder (DINOv2/SigLIP) by making the model supervise its own representations.

**Dual-Timestep Scheduling.** Sample two independent timesteps `t, s ~ p(t)`. Draw a token mask `M = {i : u_i < R_M}`, `u_i ~ U(0,1)`, with mask ratio `R_M ≤ 0.5`. Masked tokens are noised to level `s`, unmasked tokens to level `t`, giving a per-token time vector `τ` and input `x_τ = diag(1-τ)x_0 + diag(τ)x_1`. This creates *information asymmetry*: some tokens are cleaner than others, and the model must infer the corrupted ones from the clean ones. `R_M ≤ 0.5` and the two-level (rather than fully per-token independent) design are the compromise against the train/inference gap, since inference uses a uniform `t`.

**Objective.**

```
L_gen = E ‖ f_θ(x_t, t) − (x_1 − x_0) ‖²                                  (standard CFM)
L_rep = − E [ cos( h_θ^(l)(x_τ, τ),  f_θ'^(k)(x_{τmin}, τmin) ) ]        τmin = min(t, s)
L     = L_gen + γ · L_rep
```

`f_θ'` is an EMA teacher run on the **cleaner** input; `h_θ^(l)` is an MLP head on student layer `l` predicting teacher layer `k`, with `l < k` (early student layer predicts late teacher layer). Cosine similarity is required — ℓ1 was numerically unstable.

**Results.** ImageNet 256 SiT-XL: FID 5.70 vs REPA 5.89 vs vanilla at 4M steps; with RAE 2.95 vs 3.24 at 1M steps. T2I: FID 3.61 vs REPA 3.92. Video: FVD 47.81 vs REPA-DINOv2 49.59 vs vanilla 50.95 (external alignment *hurts* video — the paper's strongest argument). Audio: FAD 145.6, where MERT alignment gives no gain. Scaling: the Self-Flow-vs-REPA gap **widens** with model size; 625M Self-Flow beats 1B REPA.

**Ablations.** Removing `L_rep` costs ~+4 FID (dominant term); removing masking costs ~+1 FID; restricting `s ∈ [t−0.2, t]` is equivalent to removing masking.

**Cost.** One extra teacher forward per training step. **No inference overhead, no change to the ODE solver, no change to NFE.**

---

## 2. What the repo actually contains

`aux_repo/Self-Flow` is **1,213 LOC of inference code** for one ImageNet-256 checkpoint. It is not a training release.

| File | Content | Portable to FM-PCC? |
| :--- | :--- | :--- |
| `sample.py` | 50k-sample FID generation driver, `torchrun` multi-GPU | No |
| `src/model.py` | `SelfFlowPerTokenDiT` (SiT-XL/2), `modulate_per_token`, `PerTokenDiTBlock`, `PerTokenFinalLayer`, class-label `y_embedder`, `SimpleHead` projector stub (`:233`), `return_features` layer hook (`:379-382`), per-token `t` embed (`:463-468`) | Already have equivalents |
| `src/sampling.py` | Linear ICPlan transport, Euler-Maruyama / Heun **SDE** sampler, CFG, `get_score_from_velocity` (`:123`) | No — see §3.4 |
| `src/utils.py` | 2-D sin-cos position encoding | No |

**The paper's entire contribution is absent from the code.** There is no training loop, no `L_rep`, no EMA teacher, no mask sampling, no `γ`. `model.py:380` merely exposes `zs = self.projector(x)` behind a flag; the objective that consumes `zs` was never released. Adopting Self-Flow means reimplementing it from the PDF — the repo saves nothing.

Confirming this, `sampling.py:477` raises `NotImplementedError("Only SDE mode is currently supported")`. The released sampler is 250-step stochastic.

---

## 3. Why it fails for FM-PCC — five specific reasons

### 3.1 Eight tokens is not a token grid

Self-Flow's asymmetry mechanism runs on SiT-XL/2 over a 32×32 latent with patch 2 → **256 tokens**, masking up to 128 of them. FM-PCC's planner state is `[B, H=8, 9]` with `patch_size: 1` → **8 tokens** (`config/aligning-d3il-visual.py:355`, `mf_dit_official_trajectory.py:276-285`). At `R_M ≤ 0.5` that is at most 4 masked positions, over a horizon whose adjacent steps are already near-collinear at 33 Hz control rate. There is no redundancy to exploit and no population to mask. On the U-Net backbones (Gen3v6/Gen3v7 default, Gen7/Gen14 visual, `dim: 32`, `dim_mults: (1,2,4,8)`) it is worse: the stride-2 stack collapses 8 → 4 → 2 → 1, so per-position noise labels are destroyed by the second downsample, and "intermediate layer features" have no `l < k` token-aligned meaning.

### 3.2 The mechanism is contested precisely where FM-PCC is weakest

arXiv:2607.02508 (*From SRA to Self-Flow: Data Augmentation or Self-Supervision?*, July 2026) introduces **Attention Separation**: keep Self-Flow's dual-timestep input but **block attention between tokens at different noise levels**. Performance does not degrade and sometimes improves. That kills the stated causal story (clean tokens informing noisy ones) and attributes the gain to augmentation — "splitting a single image into multiple effective training parts to expand the training data."

If the gain is augmentation-by-token-splitting, it scales with token count and intra-sample redundancy. FM-PCC has 8 highly correlated tokens. The follow-up therefore predicts **≈ zero transfer** to trajectory planning. Worse, "block attention between noise groups" is undefined for a conv U-Net — there is no attention to separate, and the receptive field mixes noise levels unconditionally.

### 3.3 Three orders of magnitude off the demonstrated scale

Demonstrated at 290M / 420M / 625M / 1B params, depth 8–28, on 6M–200M samples, 1M–4M training steps. FM-PCC's temporal U-Net at `dim: 32, dim_mults (1,2,4,8)` is ~1–2M params trained on a few hundred D3IL demonstrations. The paper's own scaling figure shows the Self-Flow advantage **growing with scale** — i.e. it is smallest exactly at our end. And the headline ImageNet number is 5.89 → 5.70 FID (~3%) at XL scale after 4M steps.

### 3.4 The released sampler points the wrong way

FM-PCC's thesis is a *deterministic* engine replacing stochastic diffusion, currently pushed toward low NFE (Gen14 automated K sweeps, Gen13 closure: FM@K=2 at 0.1894 s/plan, 100% safe). The only implemented sampler here is 250-step SDE. `get_score_from_velocity` (`sampling.py:123`, ~8 lines) would let an FM checkpoint be sampled as an SDE, but that is an ablation against the thesis and is trivially rewritten from the linear-interpolant identities — not a reason to keep a repo.

### 3.5 Wrong metric, and a cost we cannot absorb

Self-Flow is scored on FID/sFID/IS/FVD/FAD. FM-PCC ranks on **unguided task success + constraint satisfaction + NFE/latency vs diffusion-DPCC** — and Gen13's closure was specifically the lesson that generative proxy metrics (`raw_mse_u`) do not map to control quality. Adopting Self-Flow costs an extra EMA-teacher forward per step (~+30–50% train time) and adds five coupled hyperparameters — `γ`, `R_M`, `l`, `k`, EMA decay — on top of a noise schedule `p(t)` that the paper explicitly says **requires tuning because it determines the masking behaviour**. That is a large sweep on i6-gpu-1 for a mechanism with no NFE benefit and an adverse scale argument.

---

## 4. The robotics result, and why it does not rescue the case

The paper *does* include robotics: joint video-action fine-tuning on **RT-1 (73.5k episodes)**, evaluated on **SIMPLER** success rates, with Self-Flow keeping an advantage over vanilla flow matching on complex multi-object / sequential tasks (Move Near, Open-and-Place) while converging with vanilla on simple Pick / Open-Close.

This looks like the strongest pro-adoption evidence and is actually the opposite:

1. It initialises from a **video-weighted mixed-modality pretrained model**. The gain is inherited from large-scale video representation learning — the asset FM-PCC does not have and is not trying to build.
2. **No action-only ablation is reported.** Actions are a side output of a video generator; there is no evidence the mechanism does anything when video tokens are removed. FM-PCC's planner is action/state-only (or visual-conditioned, not visual-*generating*).
3. The paper gives **no description of how actions are tokenized**, so there is nothing to port even in principle.
4. The advantage appears on *semantic* task complexity (identifying and sequencing objects) — a perception/language problem. FM-PCC's residual failures are *geometric/dynamic* (constraint satisfaction, plan smoothness under projection, low-K fidelity).

---

## 5. The one idea worth remembering (and its better source)

Strip away the representation loss, and Dual-Timestep Scheduling leaves a primitive that is genuinely interesting for receding-horizon MPC: **heterogeneous noise level per horizon step**. In closed-loop control there is a real, structural information asymmetry that the image domain has to manufacture artificially — at each replan, the near-future portion of the plan is largely determined (previous plan's tail, executed history, current state), while the far-future is open. Conditioning the velocity field on a per-step `τ` vector that decays along the horizon would let a single model serve both plan-refinement and plan-extension, and would amortise across replans.

**But do not source that from Self-Flow.** Self-Flow neither proposes nor evaluates it for this reason; its `R_M ≤ 0.5` two-level scheme exists purely to limit the train/inference gap for uniform-`t` sampling. The proper upstreams are **Diffusion Forcing** and **Rolling Diffusion**, which target exactly this and come with sequential-decision-making evaluations. If this direction is ever opened (natural home: a Gen15/Gen16 sibling on the DiT backbones, where `modulate()` already accepts token-level params), pull those repos — not this one.

---

## 6. What would reverse this verdict

Concrete, falsifiable conditions. If none hold, the verdict stands:

1. **BFL releases the training code** (`L_rep`, teacher, mask sampling) — removes the reimplementation cost, though §3.1–3.3 still apply.
2. **FM-PCC moves to a token-rich backbone** — Gen10's planned VAE+Transformer with action chunking at `H ≫ 8`, or a visual generation (not just visual conditioning) branch producing image tokens. Below ~64 tokens the mechanism has no substrate.
3. **A published action-only ablation** shows Self-Flow gains without video co-generation and without large-scale pretraining.
4. **arXiv:2607.02508 is itself refuted** and the interaction mechanism is re-established — this is the load-bearing one for §3.2.

---

## 7. Decision

- `aux_repo/Self-Flow`: **delete.** Inference-only, wrong modality, wrong scale, and its single portable code idea (per-token adaLN) is already present at `flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py:69`.
- Priority ranking unchanged: low-NFE objectives (Gen3v6 MeanFlow, Gen3v7 α-Flow, Gen14 Visual-Mix-ML) remain the correct investment, because they move the axis FM-PCC is measured on and Self-Flow does not.
- No code, config, or sbatch changes made. This document is the entire artifact.

---

## Sources

- [Self-Flow — arXiv:2603.06507](https://arxiv.org/abs/2603.06507) · [PDF](https://arxiv.org/pdf/2603.06507) · [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2603.06507)
- [Black Forest Labs — Self-Flow project page](https://bfl.ai/research/self-flow)
- [GitHub — black-forest-labs/Self-Flow](https://github.com/black-forest-labs/Self-Flow/)
- [ICML 2026 poster](https://icml.cc/virtual/2026/poster/65011)
- [From SRA to Self-Flow: Data Augmentation or Self-Supervision? — arXiv:2607.02508](https://arxiv.org/abs/2607.02508)
