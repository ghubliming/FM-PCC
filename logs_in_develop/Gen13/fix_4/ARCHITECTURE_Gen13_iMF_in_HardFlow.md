# Gen13 architecture — how iMF is injected into HardFlow

**Date:** 2026-07-19 · **Purpose:** one document to fully understand Gen13, from the abstract idea down to "which file, which function".
**Scope:** everything under `HardFlow/hardflow/models_flow/imf/` + the three `run/*_imf.py` entries. All Gen13 code is **additive** — no pre-existing HardFlow file is modified.
**Companions:** `../init/PLAN_Gen13_iMF_backbone_in_HardFlow.md` (the plan), `../../HF_iMF/Research/BLEND_HardFlow_iMeanFlow.md` + `THEORY_DeepMix_HF_iMF.md` (the math), `../fix_3/INSIGHTS_Gen13_first_run.md` (results).

---

## LEVEL 0 — The one-paragraph idea

HardFlow is a **constrained sampler**: it runs a flow-matching model's ODE forward, and at every ODE step it (a) predicts where the trajectory will *end up*, (b) hard-projects that prediction onto the feasible set with an IPOPT NLP, and (c) pulls the correction back into the current state. Step (a) is HardFlow's weak point — it estimates the endpoint by **Euler-extrapolating the instantaneous velocity**: `x̂₁ = z + (1−τ)·v(z,τ)`, a first-order guess. iMeanFlow (iMF) learns the **average velocity** `u(z, τ, h)` over a time interval, for which the endpoint is `x̂₁ = z + (1−τ)·u(z, τ, 1−τ)` — the *exact* flow endpoint (up to training error), at the same 1 network call. **Gen13 swaps that one estimator and changes nothing else.** The NLP, constraints, pull-back, and controller are untouched.

**Result:** the same constrained-sampling guarantee at ~2× fewer network evaluations (21 vs 41 NFE at K=5).

---

## LEVEL 0.5 — ⚠️ EXACTLY what was swapped: **HardFlow's U-Net, iMF's math**

"iMF backbone" is ambiguous and is the single most common misreading of Gen13, so state it plainly:

| Component | Source | Detail |
|---|---|---|
| **Network architecture** | 🟦 **HardFlow's `TemporalUnet`** | 1-D temporal conv U-Net for trajectories. `temporal_imf_unet.py` **imports 8 blocks directly** from `hardflow/models_flow/unet.py`: `Conv1dBlock`, `Downsample1d`, `Upsample1d`, `ResidualTemporalBlock`, `SinusoidalPosEmb`, `LinearAttention`, `PreNorm`, `Residual`. Same `dim=32`, `dim_mults=(1,4,8)`, same channel dims `[(6,32),(32,128),(128,256)]` as the FM baseline. |
| **What the network predicts** | 🟩 **iMF** | average velocity `u(z,τ,h)` + auxiliary `v` (dual head) instead of `v(z,τ)` |
| **Training objective** | 🟩 **iMF** | MeanFlow identity via JVP, predicted-v tangent, adaptive loss |
| **Sampler** | 🟩 **iMF** | K exact interval jumps (NFE = K) |
| **iMF's DiT (`imfDiT`)** | ❌ **NOT USED** | **zero transformer / DiT code exists anywhere in Gen13** — verified by grep across the whole package |

**So Gen13 = HardFlow's proven trajectory architecture + iMF's field definition, loss, and sampler.** The only architectural *changes* to the U-Net are the two minimal additions needed to host iMF's math (L3.1): a second time-embedding branch for the interval width `h`, and a doubled final conv so one network emits both `(u, v)`. The U-Net body — down/mid/up blocks — is untouched.

**Why not iMF's DiT** (plan decision D2): the official iMF is an ImageNet model where DiT is the natural choice. Here the data is 96 demonstration *trajectories* (H=16 × 6 dims), not images — a transformer would be badly data-starved, and Gen3v4 already showed DiT struggling at this scale. Keeping HardFlow's U-Net also means the FM baseline and iMF differ **only** in the field/objective, making the comparison clean: any measured difference is attributable to the average-velocity idea, not to a change of architecture.

*(Consequence: the 3.69M-param iMF net is slightly larger than the FM one solely because of the extra `h` embedding and the doubled output head.)*

---

## LEVEL 1 — The two paradigms side by side

### What FM (baseline) learns
A flow-matching model learns the **instantaneous velocity field** `v(z, τ)` — "at state z, at time τ, which direction is the data?" To generate, you integrate it with many small Euler steps (HardFlow uses `ode_t_steps=10`). To guess the endpoint from partway along, you can only extrapolate linearly — accurate only for small `1−τ`.

### What iMF learns
iMF learns the **average velocity over an interval**, `u(z, τ, h)` — "starting at z at time τ, what constant velocity would take me exactly to time τ+h?" This makes the jump *exact*:
```
z_{τ+h} = z_τ + h · u(z_τ, τ, h)          ← exact, one network call
x̂₁      = z_τ + (1−τ) · u(z_τ, τ, 1−τ)   ← the endpoint, one network call
```
Generation with K steps needs only K calls (K=1 is legal). This is the "fast-forward generative model" property.

### The consequence for HardFlow
| | FM | iMF (Gen13) |
|---|---|---|
| Endpoint estimate | Euler shot `z + (1−τ)v` — 1st-order **approximation** | `z + (1−τ)u` — **exact** map |
| Steps to traverse [0,1] | 10 (needs small steps) | K ∈ {1,2,4,5} |
| NFE per plan | ~41 | 5 / 9 / 17 / 21 |
| What the NLP sees | a biased endpoint prediction | a (in principle) unbiased one |

---

## LEVEL 2 — The convention hazard (why `convention.py` exists)

The single most dangerous integration detail. The two codebases run **time in opposite directions**:

| | HardFlow | official iMF (aux repo) |
|---|---|---|
| interpolant | `z_τ = τ·x₁ + (1−τ)·x₀` | `z_t = (1−t)·x + t·e` |
| direction | **τ=0 noise → τ=1 data** | **t=0 data → t=1 noise** |
| velocity | `v_HF = x₁ − x₀` | `v_iMF = e − x` |

Mapping: `τ = 1 − t`, `u_HF = −u_iMF`, interval width `h` identical. A sign slip produces a sampler that walks *toward noise* — and (this is the trap) it still passes naive tests.

**Design decision:** rather than convert at runtime, the whole package is written **natively in HardFlow's convention**, and *all* mapping reasoning is quarantined in one file.

📄 **`imf/convention.py`** (103 lines) — the module docstring carries the full derivation; no other file reasons about signs.
- `sample_tau_h()` :65 — draws `(τ, h)`; implements the official logit-normal `(t,r)` scheme through the `τ = 1−t` flip.
- `jvp_tangents()` :90 — the tangent triple `(v_c, +1, −1)` for the identity.
- `endpoint_from_u()` :96, `jump_from_u()` :101 — the two exact formulas above.

> **Real incident (fix_1):** the identity's `h`-term sign was wrong in the first implementation. Gate G1-A (the `h→0` limit) *passed* — because at `h=0` the term vanishes — while G1-B/C/D failed with 1-NFE samples overshooting (mean|x| 2.499 vs 2.0). Corrected identity: **`u = v + h·D_tot`**, so the training compound is `V = u − h·sg(D_tot)`. This is why the gate exists.

---

## LEVEL 3 — The ML: network, objective, sampler

### 3.1 The network — two-time conditioning + dual heads
📄 **`imf/temporal_imf_unet.py`** → `class TemporalImfUnet` :33, `forward()` :150

`v(z,τ)` takes one time argument; `u(z,τ,h)` takes **two**. And the training objective needs both `u` and an auxiliary `v`. So, relative to HardFlow's `TemporalUnet` (whose building blocks are **imported, not copied**):

1. **Second time embedding** — a `h_mlp` sinusoidal branch alongside `time_mlp`; the two embeddings are **summed** into the single conditioning vector the residual blocks already expect (so the U-Net body is unchanged).
2. **Dual head** — final conv emits `2 × transition_dim` channels, split by `torch.chunk` into `(u, v)`.

Signature: `(x, tau, h) → (u, v)`, each `(B, H, transition_dim)`. 3.69M params.
*Why a U-Net and not iMF's DiT (plan D2): trajectories ≠ images, and at 96 demos a DiT is data-starved.*

### 3.2 The objective — improved MeanFlow, CFG stripped
📄 **`imf/imf_matcher.py`** → `class ImfMatcher` :33, `loss()` :68

Ported from the official JAX `imf.py forward()`. The MeanFlow identity in HF convention (derived in `convention.py`):
```
u = v + h · D_tot ,   D_tot = JVP of u(z,τ,h) along tangents (v_c, +1, −1)
```
Enforced as a **v-form regression**:
```python
V      = u - h * sg(D_tot)                    # should equal the FM target
loss_u = adp( Σ (V - v_target)² )             # MeanFlow identity term
loss_v = adp( Σ (v - v_target)² )             # auxiliary v-head
adp(L) = L / sg((L + eps)^p)                  # adaptive normalization
```
Three iMF-specific features preserved:
- **Predicted-v tangent** — the JVP direction is the network's *own* detached `v` prediction (the "improved" in improved-MeanFlow), computed by an extra no-grad forward.
- **Adaptive loss** `L/(L+ε)^p` — equalizes per-sample gradient magnitude.
- **`data_proportion`** — a fraction of the batch gets `h=0` (pure flow-matching anchors).

⚠️ **The adaptive loss is bounded and FLAT BY CONSTRUCTION** (it sat at 1.996–1.999 for the entire real run). **Never judge convergence by it.** `loss()` returns `raw_mse_u`, `raw_mse_v`, `a0_mse` for that purpose.

**CFG dropped entirely** (plan D3): HardFlow conditions by *state-inpainting*, not class labels, so `omega/t_min/t_max`/null-token machinery is removed — `u_fn` simplifies to `u(x, τ, h)`.

### 3.3 The sampler — K exact jumps
📄 **`imf/imf_sampler.py`** → `imf_sample()` :22
```python
for i in range(K):
    τ = i/K ;  dt = 1/K
    u, _ = model(x, τ, dt)
    x = x + u * dt          # exact jump over [τ, τ+dt]
```
Conditioned entries are masked exactly as HardFlow's `ConditionedODESolver` does. **NFE = K.**

---

## LEVEL 4 — The injection: where iMF meets HardFlow

📄 **`imf/imf_flow_policy.py`** → `class ImfFlowPolicy(FlowPolicy)` :45 — **this is the heart of Gen13.**

It *subclasses* HardFlow's `FlowPolicy`, inheriting the CasADi NLP construction, obstacle/dynamics constraints, value model, and pull-back untouched. It overrides only what must change because the network signature is now `(x,τ,h)→(u,v)`.

### 4.1 THE SEAM — two lines, the entire scientific content of Gen13
📄 `imf_hardflow_new_forward()` :263 — an adapted copy of the base `hardflow_new_forward` (`flow_policy.py:1286`). Diff vs base:

| | FM baseline (`flow_policy.py`) | Gen13 iMF (`imf_flow_policy.py`) |
|---|---|---|
| **(1) reference step** | `x_next_ref = x_k + v_k * dt` :1325 | `x_next_ref = x_k + u_k * dt` **:306** — exact jump |
| **(2) endpoint** | `x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next` :1340 | `... = x_next_ref + h_terminal * u_terminal` **:324**, `h_terminal = 1 − t_next` — exact map |

**Everything after those two lines is byte-identical logic:** `oc_cs_opti.solve_limited()`, the pull-back `x_next = x_next_ref + t_next·(x_terminal_predicted − x_terminal_predicted_ref)`, control bookkeeping, unnormalization, timing.

### 4.2 Supporting overrides
| Method | Line | Why it must change |
|---|---|---|
| `_u()` | :82 | single `u`-evaluation + NFE accounting |
| `x1_estimate()` | :103 | base uses `x + (1−t)·v`; iMF uses exact `x + (1−t)·u` |
| `warmstart()` | :126 | base rolls Euler `v`-steps; iMF uses exact `u`-jumps |
| `hardflow_formulate()` | :164 | **reuses the base CasADi build** via a guidance-name shim (temporarily sets `guidance_method="hardflow_new"` so the base assertion passes, restores it in `finally`) — so **zero NLP code is duplicated** |
| `constrained_u_fn_torch()` | :182 | dof-space `u` evaluation (mirror of base `constrained_flow_fn_torch`) |
| `imf_original_forward()` | :206 | unguided K-step sampling (the `original` baseline equivalent) |

### 4.3 Dispatch and safety
`__call__` :90 accepts only `original_imf` / `hardflow_new_imf`; FM guidance names **raise** (their `(x,t)` signature is incompatible). `IMF_GUIDANCE_METHODS` :47.

### 4.4 Instrumentation (fix_4)
- **NFE accounting** — `_nfe` buckets (warmstart / sampling / diag), `_nfe_info()` :73. Verified exact: K=2 → 2 + 4 + 3 = 9 ✓ matches logs.
- **NLP health** — `_nlp_solves` / `_nlp_failures`, `reset_nlp_stats()` :62, `nlp_stats()` :67; a failed solve prints a loud greppable WARNING. First run: **0 failures in 100 episodes**, which is what let us rule out solver failure as the cause of residual violations.

---

## LEVEL 5 — Entry points and control flow

📄 **`run/train_imf.py`** — sibling of `run/train.py`. Builds `TemporalImfUnet` + `ImfMatcher`; reuses `SequenceDataset` unchanged. tensorboard optional; always writes `metrics.csv`; cosine LR over the **full** budget; final checkpoint at cp index 4. tqdm gated behind `sys.stdout.isatty()` (fix_4).

📄 **`run/eval_imf.py`** — sibling of `run/eval.py`. Reuses `ProxyValueModel` + `check_violation` by **import**; forks `run_env` → `_run_env_quiet` (quiet logging only — identical semantics, so CSVs stay comparable with the frozen FM baselines). Adds `nfe_*` and `nlp_*` CSV columns.

📄 **`run/imf_gates.py`** — G0 (shapes) + G1 (1-D GMM end-to-end: `h→0` limit, **1-NFE sign gate**, K1≈K2, jump composition). Exit-code gated; the training sbatch runs it first and **aborts on failure**.

### Full call chain for one planning step (guided)
```
run_env loop (eval_imf.py)
  └─ policy(conditions)                       → ImfFlowPolicy.__call__            :90
      └─ imf_hardflow_new_forward()                                               :263
          ├─ warmstart()          → K× _u()  → TemporalImfUnet.forward(x,τ,h)     :126
          └─ for k in range(K):
               ├─ u_eval_np(x_k, t_k, dt)      ← SEAM (1) exact jump              :306
               ├─ u_eval_np(x_ref, t_next, h)  ← SEAM (2) exact endpoint          :324
               ├─ oc_cs_opti.solve_limited()   ← INHERITED IPOPT NLP (unchanged)
               └─ pull-back x_next             ← INHERITED (unchanged)
```

---

## LEVEL 6 — Why K means something different here (the key Gen13 finding)

`K == cfg.ode_t_steps` controls **two things at once** in the guided path:
1. the number of sampler steps (NFE), and
2. **the number of prox-NLP constraint projections** — one per step.

Gen3v4 established that iMF is **K-invariant for generation** (more steps don't improve sample quality), which is why plan D7 set K∈{1,2}. **That reasoning does not transfer to the guided path**, because of (2). Measured:

| K | Safety | NFE | Projections |
|---|---|---|---|
| 1 | 80% | 5 | 1 |
| 2 | 94% | 9 | 2 |
| 4 | 96% | 17 | 4 |
| 5 | 98% | 21 | 5 |
| FM | 100% | ~41 | 10 |

Safety rises monotonically with **projection count**, not with generative quality. (Diminishing after K=2; see `../fix_3/INSIGHTS_Gen13_first_run.md` §9–11 — and note the K≥2 differences vs FM are **not statistically significant** at n=50.)

**Related insight:** HardFlow enforces dynamic feasibility as a *hard NLP constraint*, so a coarse `u`-field gets projected onto the feasible manifold anyway. This is why unguided iMF solves ~0–2% of episodes yet guided iMF solves 94–98%. See `../../HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md`.

---

## LEVEL 7 — File map (quick reference)

| File | Lines | Role |
|---|---|---|
| `imf/convention.py` | 103 | **all** time-convention/sign logic; `(τ,h)` sampling |
| `imf/temporal_imf_unet.py` | 188 | `(x,τ,h)→(u,v)` backbone |
| `imf/imf_matcher.py` | 119 | improved-MeanFlow training objective (JVP + adaptive) |
| `imf/imf_sampler.py` | 49 | K-step exact-jump sampler |
| `imf/imf_flow_policy.py` | 423 | **the seam** + all HardFlow integration |
| `imf/imf_config.py` | 48 | config dataclasses (children of FM configs) |
| `imf/README_PROVENANCE.md` | — | what was ported from where |
| `run/train_imf.py` | ~150 | training entry |
| `run/eval_imf.py` | ~350 | eval entry (+ quiet `run_env` fork) |
| `run/imf_gates.py` | ~180 | G0/G1 correctness gates |
| `run_scripts/{train_imf,eval_original_imf,eval_hardflow_new_imf}.sh` | — | paper-parameter wrappers |
| `Slurm_Codes/sbatch/hardflow/{train_imf,eval_imf,imf_pipeline}_hardflow.sh` | — | cluster jobs |

**Reading order for a newcomer:** `convention.py` docstring → `imf_matcher.loss()` → `imf_sampler.imf_sample()` → the two SEAM lines in `imf_flow_policy.py:306,324`. That is the entire method.

---

## LEVEL 8 — What is genuinely new vs inherited

**New (Gen13):** the average-velocity backbone, its training objective, the K-step sampler, and **two lines** replacing the endpoint estimator — plus NFE/NLP instrumentation.

**Inherited unchanged from HardFlow:** the IPOPT prox-NLP and its formulation, obstacle + fitted-dynamics + action-bound constraints, the τ-weighted pull-back, `ProxyValueModel`, the receding-horizon controller (H=16, replan 8), dataset/normalizer, env loop, CSV schema.

**Inherited unchanged from the FM path (so results are comparable):** every eval hyper-parameter — `random_repeat=50`, `constraint="novel"`, `obstacle_margin=0.02`, cost scales, `hardflow_activation="all"`.

**Not built (deliberate):** the Newton/MF pull-back gain `∇F = I + (1−τ)∇u` from `THEORY_DeepMix_HF_iMF.md` — Gen13 is "Level 1" (seam only, plan D8); the pull-back still uses HardFlow's `τ` gain. That is the natural Gen13-continuation experiment.
