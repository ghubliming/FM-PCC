# Is the FM-PCC / HardFlow iMF reimplementation wrong? — and should we instead train inside the `imeanflow` repo?

**Date:** 2026-07-22 · **Type:** audit + design analysis · **no code change**
**Question (user):** *"Could Gen13's and Gen3v4's reimplementation of iMF into FMPCC/HF be wrong? Maybe we should build the iMF training on the `imeanflow` repo itself, then use the trained weights in FMPCC/HF. Currently we rebuild iMF from aux_repo/imeanflow to replace each old trainer, but the bone is FMPCC/HF — could that be wrong? Should I rather feed the avoiding dataset into imeanflow directly instead of ImageNet, then use the weights to eval?"*

**Sources read this session (line-level):**
- `/workspaces/aux_repo/imeanflow` @ `bf60cd7` (JAX `main`) — `imf.py`, `models/imfDiT.py`, `train.py`, `scripts/*.sh`, `utils/input_pipeline.py`
- `/workspaces/aux_repo/imeanflow` @ `04687983` (`origin/torch`, fetched this session) — `imf.py`, `README.md`, `requirements.txt`
- `HardFlow/hardflow/models_flow/imf/` — `convention.py`, `imf_matcher.py`, `imf_sampler.py`, `temporal_imf_unet.py`, `imf_config.py`
- `flow_matcher_v3_imeanflow/models/` — `imf_diffusion.py::_p_losses_imf_official`, `imf_dit_trajectory.py`; `config/avoiding-d3il.py`
- Dev logs: `Gen13/fix_7/RESULTS_..._VERDICT_imf_refuted.md`, `HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md`, `Gen3v4_imf/U10/debug_notes/{INVESTIGATION_imf_fidelity...,POST_U10_II,POST_U10_III}.md`, `Gen3v4_imf/U10/K2_train_eval/ANALYSIS_...md`

---

## 0. TL;DR — direct answers

1. **The port is not wrong in the way you fear.** The MeanFlow identity, its sign, the `(t,r)`→`(τ,h)` flip, the sampler, and the adaptive loss are all correct in both ports. I re-derived them against upstream this session. §2.
2. **But it is not clean either — I found 3 real deviations, one of them new and Gen13-only.** ⭐ **Gen13's `TemporalImfUnet` conditions on BOTH `τ` and `h`; upstream `imfDiT` deliberately conditions on `h` ONLY** (`imfDiT.__call__` takes `t` and never uses it — explicit comment citing the MeanFlow paper). Gen3v4 got this right (`dit_condition_on_t: False`). §2.3
3. **This is not a sign bug — Gen13's math is self-consistent — but it changes the learning problem**: Gen13's JVP carries an extra `∂u/∂τ` term upstream never has, and it hands the network a spare input through which COMPARE §8.2's blind direction can be satisfied. It also means **Gen13 and Gen3v4 were never architecturally comparable**, which weakens the cross-codebase comparison in COMPARE §2. §2.3
4. **Second deviation (Gen13):** the JVP tangent `v_c` is the v-head evaluated at the *actual* `h`; upstream evaluates it at `h = 0` (`v_cond_fn` hardcodes `h = zeros`). Low-to-medium severity — `loss_v` pushes the v-head toward `h`-invariance — but it is a deviation. Gen3v4 is faithful here (`h=h0=0`). §2.4
5. 🔴 **The thing that has never been done in either codebase: a numerical parity check against upstream.** Every verification to date (gates G0/G1, kill-table) is *internal self-consistency* on synthetic data. Nobody has ever fed identical tensors to `imeanflow/imf.py` and to our port and compared the numbers. **That is the actual gap your instinct is pointing at** — and it is cheap to close. §6
6. 🔴 **"Train in the imeanflow repo with minimal modification" is inverted — it is the *maximal*-modification path.** Upstream is JAX/Flax + `jax[tpu]==0.4.27` + TF 2.15 + orbax + `pmap`, over VAE latents, class-conditional, evaluated by FID/IS. The only piece that transfers unchanged is `imf.py::forward` — ~70 lines — **which is exactly the piece already ported twice.** Everything else (data pipeline, 2D patchify + 2D RoPE, class conditioning, FID eval, TPU infra) must be replaced. §3
7. 🔴 **The torch branch does not rescue this.** It exists (`origin/torch`, fetched OK), but it is **inference-only** — `assert eval, 'The current codebase only supports inference mode'`, no `forward()`, no loss, no training loop. Its own README: *"We only provide inference code and pre-trained checkpoints… For training code, please refer to the original JAX implementation."* So the weight-reuse plan requires a **JAX→PyTorch conversion you would have to write yourself**. §3.2
8. **Even a bit-exact upstream training would very likely land in the same place.** The two documented failure mechanisms — the loss's blind direction `δ_u = h·δ_D` at large `h` (COMPARE §8.2) and the 96-episode data scale — are **properties of the objective and the task, not of the port.** Changing repos does not touch either. §4
9. ⭐ **The highest-value new idea is not "train in imeanflow" — it is distillation.** You already have a *validated* teacher: FM @ K=2, 100% safe, 0.1894 s/plan (fix_7.3 §2). Regress `u(z,τ,h)` directly onto the FM teacher's integrated displacement. That deletes the JVP, deletes the blind direction, deletes the `h`-coverage gap (you sample `h=1` as often as you like), and is plain supervised regression. **No log in this repo has proposed it** (grep: 10 incidental hits, none on this). §7.1
10. **Recommendation:** run the parity harness (§6, ~1 day, decides "is our port wrong" outright), fix the two deviations in §2.3/§2.4, and put the real effort into §7.1 distillation. Treat Path U (§5) as fully specified but **not recommended** — I have written it out in full because you asked for it.

---

## 1. First, a correction to the premise: there is no single "the reimplementation"

The question says *"the bone is FMPCC/HF"*. That is right, but there are **two independent ports with different fidelity profiles**, and conflating them has already caused confusion in earlier notes:

| | **Gen3v4** (`flow_matcher_v3_imeanflow`) | **Gen13** (`HardFlow/hardflow/models_flow/imf`) |
|---|---|---|
| host | FMPCC / DPCC / diffuser stack | HardFlow stack |
| backbone | `imf_dit_trajectory.py` — a real port of `imfDiT` (RoPE, QK-norm, SwiGLU, dual heads, in-context tokens) | `temporal_imf_unet.py` — a **1D temporal U-Net**, not a DiT at all |
| horizon × dim | 8 × 6 = 48 | 16 × 6 = 96 |
| CFG | ✅ **ported** (`guidance_fn`, `v_fn`, `cond_drop`, guided `v_g`, null token) | ❌ **dropped by decision D3** ("HardFlow conditions by state-inpainting") |
| conditions on `t`? | ❌ No — `dit_condition_on_t: False` ✅ **matches upstream** | ⚠️ **Yes** — `time_mlp(tau) + h_mlp(h)`, unflagged ❌ **deviates** |
| `v_c` tangent at | `h = 0` ✅ **matches upstream** | actual `h` ❌ **deviates** |
| FM anchors | 0.5 (setup1) / 0.25 (setup2) | 0.25 |
| val split | present but **leaks** (window-level `random_split`) | **none at all** |

So "is the reimplementation wrong?" has to be answered twice. **Gen3v4-`imf_official` is the more faithful of the two.** Gen13 is a deliberate simplification (no CFG) plus two undocumented drifts.

---

## 2. Correctness audit vs upstream — graded by confidence

I read `imeanflow/imf.py` and both ports line by line. Grades are honest: ✅ means I verified the algebra/code myself this session; ⚠️ means a real deviation; ❓ means *nobody has ever tested this*.

### 2.1 ✅ Verified correct — high confidence

| item | upstream | port | verdict |
|---|---|---|---|
| **MeanFlow identity + sign** | `u = v − (t−r)·du/dt`; `V = u + (t−r)·sg(du_dt)`, loss `‖V − v_g‖²` | Gen13 `convention.py` derives `u = v + h·D_tot` under `τ=1−t`, enforces `V = u − h·sg(D_tot)` | ✅ **correct.** I re-derived it: with `g(τ)=z_s−z_τ=h·u`, `dg/dτ=−v` and `dh/dτ=−1` ⇒ `−u + h·D_tot = −v` ⇒ `u = v + h·D_tot`. Substituting `τ=1−t`, `u_HF=−u_iMF`, `v_HF=−v_iMF` reproduces the official form exactly. **The sign is right.** |
| **JVP tangent triple** | `(v_c, dtdt=1, dtdr=0)` over primals `(z_t, t, r)`; since `h=t−r`, tangent in `h` is `+1` | Gen13 `(v_c, +1, −1)` over `(z, τ, h)`; Gen3v4 same | ✅ **correct** — the `h`-tangent sign flips with the time axis, as it must |
| **Sampler** | `sample_one_step`: `z − (t−r)·u`, one `u`-call, `h = 1/N` | Gen13 `imf_sample`: `x += dt·u(x, τ, dt)`; Gen3v4 `p_sample_loop` identical | ✅ **faithful** (already established in the U10 fidelity audit F1; re-confirmed) |
| **Adaptive loss** | `L / sg((L+eps)^p)`, `p=1, eps=0.01`, per-sample **sum** over dims, `loss_u + loss_v` | both ports identical | ✅ **faithful**, including the saturation-at-2.0 artefact that fooled the W&B reading |
| **`(t,r)` sampling** | two independent logit-normals, `t=max, r=min`, `data_proportion` forces `r=t` | Gen13 `sample_tau_h` negates `p_mean` for the `τ` axis, takes `τ=min, s=max`; Gen3v4 same | ✅ **correct under the flip.** The `−p_mean` negation is right and is the trap Gen3v4 explicitly documents at `imf_diffusion.py:666` |
| **Predicted-`v` tangent (the defining iMF feature, D1)** | `v_c` from the v-head, not the analytic `v` | ✅ present in Gen13 and in Gen3v4-`imf_official` | ✅ **present** (this was the U10 fix; the legacy `meanflow_jvp` arm was vanilla MeanFlow) |

**Bottom line of §2.1: the parts you would most expect to be wrong are right.** The sign bug hunt is over; `convention.py` is a genuinely good piece of work and gate G1 exercised it.

### 2.2 ⚠️ Known, *deliberate*, documented deviations

- **Gen13 drops CFG entirely** (decision D3). Defensible — HardFlow conditions by inpainting, and the U9/U10 kill-chain showed the half-ported CFG was what made Gen3v4 explode. But it means **Gen13 is not "improved MeanFlow" in the paper's full sense**; it is MeanFlow + predicted-`v` tangent + adaptive loss + v-head. The `INVESTIGATION_imf_fidelity...md` §8 wording rules still apply: do not write *"we faithfully replicate iMF"* for the Gen13 arm.
- **Gen13's backbone is a U-Net, not `imfDiT`.** Also fine as an engineering choice, but any claim about "the iMF architecture" belongs to the Gen3v4 arm only.
- **Domain adaptation** (images/VAE latents → trajectories, class labels → state inpainting) — necessary and disclosable, not an infidelity.

### 2.3 ⭐ NEW deviation found this session — Gen13 conditions on `τ`; upstream does not

`models/imfDiT.py`:

```python
def __call__(self, x, t, h, w, t_min, t_max, y):
    # We don't explicitly condition on time t, only on h = t - r
    # following https://arxiv.org/abs/2502.13129
    seq = self._build_sequence(x, h, w, t_min, t_max, y)   # <-- t is NOT passed
```

`t` is accepted and **discarded**. The only time signal the network sees is `h`.

`HardFlow/.../temporal_imf_unet.py:163`:

```python
t = self.time_mlp(tau) + self.h_mlp(h)     # <-- BOTH, unconditionally, no flag
```

Gen3v4 handles this correctly and knows it does — `imf_dit_trajectory.py:258` `condition_on_t: bool = False,  # official conditions only on h`, and `config/avoiding-d3il.py:516` sets `'dit_condition_on_t': False`.

**Why this matters (three ways, none of which is "sign bug"):**

1. **It changes the JVP.** Upstream's `du_dt = ∂u/∂z·v_c + ∂u/∂h·1` — two terms, because `∂u/∂t ≡ 0` by construction. Gen13's `D_tot = ∂u/∂z·v_c + ∂u/∂τ·1 + ∂u/∂h·(−1)` — three terms. Gen13's derivation correctly accounts for the extra term, so the loss is *self-consistent*; but it is solving a **strictly harder and differently-conditioned problem** than the one the paper tuned.
2. **It widens the blind direction.** COMPARE §8.2 showed the residual is blind to any error with `δ_u = h·δ_D`. Giving the network `τ` as a free input adds a whole extra channel (`∂u/∂τ`) through which it can manufacture the `δ_D` that hides a `δ_u`. Upstream removed exactly that freedom. **This is a plausible partial explanation for Gen13's flat-in-K endpoint error (fix_7.3 §4: 0.1539/0.1538/0.1595/0.1572), and it is testable.**
3. **It breaks the Gen13↔Gen3v4 comparison.** COMPARE §2 concluded "both trainings converge by comparable amounts (~10×), the `a0` curves are near-identical." That comparison implicitly assumed the two were the same method on different backbones. They are not: different conditioning inputs, different JVP term count, CFG vs no CFG. The convergence observation survives; the *inference* that Gen13's objective is behaving like Gen3v4's is weaker than stated.

**Severity: MEDIUM-HIGH.** Not a bug, but the single largest un-documented divergence from the official recipe in the Gen13 arm, and it sits directly on the mechanism that fix_7.3 blamed for the failure.

**Fix cost: ~3 lines** — add a `condition_on_tau: bool = False` flag to `TemporalImfUnet`, drop the `time_mlp(tau)` term when false, and drop the `+1` τ-tangent in `convention.py::jvp_tangents` accordingly. ⚠️ **Both must change together** — the tangent triple is only valid for the architecture it was derived for.

### 2.4 ⚠️ NEW deviation — Gen13's `v_c` tangent is evaluated at the wrong `h`

Upstream, `imf.py::v_cond_fn`:

```python
# Set h, t_min, t_max to dummy values for v prediction
h = jnp.zeros_like(t)          # <-- v-head is queried at h = 0
t_min = jnp.zeros_like(t); t_max = jnp.ones_like(t)
v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]
```

Gen13, `imf_matcher.py`:

```python
with torch.no_grad():
    _, v_c = self.model(z, tau, h)      # <-- queried at the ACTUAL h
```

Gen3v4 is faithful: `_v_head` calls `self._predict_uv(..., h=h0, ...)` with `h0 = zeros_like(r)`.

**Severity: LOW–MEDIUM.** `loss_v` trains the v-head toward `v_target` at whatever `h` the batch drew, so at convergence `v(z,τ,h) ≈ v(z,τ,0)` and the two agree. Early in training they do not, and the tangent is what multiplies the Jacobian — a bad tangent produces exactly the spike pathology both runs show (`raw_mse` max spike 7,548 in Gen13's 300k run). **Worth fixing; cheap; one-line.**

### 2.5 ❓ Never verified by anyone — the real gap

| never tested | consequence |
|---|---|
| **Numerical parity vs upstream** — identical inputs, identical weights, compare `loss_u`, `loss_v`, `u`, `du_dt` | **We have no external oracle.** G0/G1 test self-consistency (does `u→v` as `h→0`? does 1-NFE land on data? is `K1≈K2`?) — all of which a *consistently wrong* implementation can also pass. |
| **Generalization** — Gen13 has no split; Gen3v4's split leaks (POST_U10_III §4.2) | Cannot distinguish underfit from memorising 96 episodes. |
| **`h`-stratified residual** — COMPARE §7.4.1 called it *"the single highest-value change in this document"* | ⚠️ **Confirmed not implemented** (`grep -rn "stratif\|h_bucket" HardFlow/run HardFlow/.../imf/` → no hits). The leading hypothesis for the whole failure has been sitting untested for a month. |

**This §2.5 is where your suspicion is justified.** Not "the port is wrong," but "**the port has never been checked against the thing it is a port of.**"

---

## 3. Why "train in the `imeanflow` repo with minimal modification" is the *maximal*-modification path

### 3.1 What upstream actually is

| axis | `imeanflow` (JAX `main`) | what avoiding/FM-PCC needs |
|---|---|---|
| framework | **JAX + Flax**, `jax[tpu]==0.4.27`, `flax>=0.8`, `optax` | PyTorch (the whole DPCC/HardFlow/CasADi/l4casadi stack) |
| parallelism | `jax.pmap` over `jax.local_device_count()` — **TPU-pod shaped** | 1 GPU on Slurm |
| checkpointing | `orbax-checkpoint==0.6.4` + `tensorstore` | `torch.save` |
| extra deps | `tensorflow==2.15.0`, `keras<3`, `tensorflow_datasets` | none of it |
| data | ImageNet → **VAE latents** via `utils/vae_util.LatentManager`, `LatentDataset`, batches reshaped `(local_device_count, -1, H, W, C)` | 96 episodes → ~13.6k overlapping windows, `(H, 6)`, `LimitsNormalizer` |
| tokenisation | 2D patchify of a 32×32×4 latent + **2D RoPE** | 1D sequence over horizon |
| conditioning | **1000-way class embedding + null token + CFG** (`omega`, `t_min`, `t_max` in-context tokens) | **state inpainting** — architecture-free |
| eval | FID/IS on 50k samples, `utils/jax_fid/inception.py` | rollout success + constraint violations in MuJoCo |

**The only file that survives contact with the new domain is `imf.py`, and within it essentially only `forward()` — ~70 lines.** That is precisely the code that has already been ported, twice, and that §2.1 says is correct.

> **The current architecture ports the small correct part and reuses the large working part (data, env, projection, eval). The proposal inverts it: keep the small part, rebuild the large part.** That is why the instinct — reasonable on its face — points the wrong way.

### 3.2 The torch branch is inference-only — the easy version of the plan does not exist

`origin/torch` @ `04687983` fetched successfully (network works from this container). Contents:

```
evaluate.py  imf.py  models/{embedder,imfDiT,torch_models}.py
utils/{fidelity_wrapper,torch_dist_util,torch_util,vae_util}.py  requirements.txt
```

Note what is **missing**: no `train.py`, no `main.py`, no `configs/`, no optimizer, no dataloader. And in `imf.py`:

```python
assert eval, f'The current codebase only supports inference mode'
```

There is **no `forward()`** — the training loss is not in this branch at all. The README is explicit:

> *"We only provide inference code and pre-trained checkpoints in this repo. For training code, please refer to the original JAX implementation."*

**Consequences:**
- ❌ You cannot train in PyTorch upstream. There is no upstream PyTorch trainer.
- ✅ You *can* use it as an **architecture and sampler oracle** — a reference `imfDiT` and `sample_one_step` in PyTorch. That is genuinely useful (§6, §7.3).
- ❌ The released `.pth` checkpoints are ImageNet class-conditional latents — **zero transfer value** for avoiding.

### 3.3 Port-cost, honestly counted

If you go ahead anyway (§5), here is what has to be written:

| # | work item | size | risk |
|---|---|---|---|
| 1 | trajectory dataset → replace `input_pipeline.py` + `LatentDataset`/`LatentManager` | ~200 LOC | low |
| 2 | 2D patchify + 2D RoPE → 1D over horizon, in `imfDiT` | ~80 LOC | **medium — this is the part that silently produces a plausible-but-wrong model** |
| 3 | class conditioning → inpainting; thread `cond` through the Flax module; pin conditioned dims in `z`, `v_target`, `v_c`, `v_g` | ~60 LOC | medium |
| 4 | strip FID/IS/inception; replace with npz dump | ~100 LOC deleted | low |
| 5 | `jax[tpu]` → `jax[cuda]`, TF/orbax pins, separate conda env on i6-gpu-1 | infra | **medium — a pinned 0.4.27/TF-2.15 stack next to the FMPCC env** |
| 6 | **JAX Flax params → PyTorch `state_dict`** converter (Dense kernel `[in,out]`→`[out,in]` transpose, LayerNorm/QK-norm naming, RoPE buffers, dual-head prefix tokens) | ~250 LOC | 🔴 **high — silent numerical drift is the default failure mode** |
| 7 | torch-side `imfDiT` with the *same* 1D modification (mirror of #2), wrapped as an `ImfFlowPolicy` for the DPCC projection | ~150 LOC | medium |
| 8 | verify #6 by matching JAX and torch forward passes to ~1e-5 | ~80 LOC | — |

**Total: a larger project than either existing port, ending at the same objective the existing ports already implement.** And items #2/#3/#7 mean the result is *not* "unmodified upstream" either — you would be maintaining a third fork.

---

## 4. Even if you did it perfectly, it probably would not change the answer

This is the part that should decide the question. The failure is already diagnosed, and **both mechanisms are repo-independent**:

| mechanism | evidence | does training in upstream fix it? |
|---|---|---|
| **The residual's blind direction.** The loss sees only `δ_u − h·δ_D`; any error with `δ_u = h·δ_D` is invisible. Conditioning degrades as `h→1` — *exactly* where 1-NFE lives. | COMPARE §8.2 (exact algebra on the loss as written — and it is the **same loss** upstream writes) | ❌ **No.** It is a property of the iMF objective, identical in JAX. |
| **Data scale.** 96 episodes; `u(z,τ,h)` is a two-time object, far more data-hungry than `v(z,τ)`. | POST_U10_III §7.2 — 13,632 windows but ~96 independent samples; "past batch ~512 brute force buys nothing" | ❌ **No.** Same dataset either way. |
| **Error tolerance.** FID forgives a slightly-wrong field; a 5 cm obstacle does not. Measured `x̂1` error **15.4 cm**. | COMPARE §8.5.2 | ❌ **No.** Task property. |
| **`h`-coverage.** K=1 queries `h=1.0`, which receives **0.11%** of training mass. | COMPARE §7.3 (Monte-Carlo of `sample_tau_h`, matches logged `h_mean` to 3 s.f.) | ❌ **No** — upstream uses the *same* logit-normal sampler. |

And the empirical verdict is already in — `RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md`, pre-registered interpretation, matched-NFE battery:

> **FM wins or ties in every single cell, on both axes.** FM@K=2: **100% safe, 0.1894 s/plan**. iMF@K=5: 95% safe, 0.4923 s/plan. iMF's "exact" endpoint map is **4–6× less accurate** than the Euler shot it was meant to replace, and **flat in K** — the signature of field/training error, not discretisation error.

**A faithful upstream retrain has to beat FM@K=2, not FM@K=10.** Nothing in §2's deviation list plausibly closes a 4–6× endpoint-accuracy gap *and* a 2.6× speed gap. The §2.3 τ-conditioning fix is the most promising of them and I would still not bet on it clearing that bar.

> **Honest counterweight:** §2.5 is real. Because no parity test exists, I cannot *prove* the refutation isn't measuring a subtly broken port. That is exactly why §6 comes before everything else — it is the cheapest way to make the refutation trustworthy (or to overturn it).

---

## 5. Path U — the "train in imeanflow" plan, fully specified (asked for; **not recommended**)

If you decide to do it regardless, do it in this order. Do **not** start at U1 — start at U0.

**U0 — decide what the deliverable is.** Two very different goals hide behind this proposal:
- *(a) "Prove our port is right."* → **Path U is the wrong tool.** Use §6, which achieves this in ~1 day instead of ~3 weeks.
- *(b) "Be able to write 'trained with the official iMF implementation' in a paper."* → Path U is the only route, and the cost below is the price.

**U1 — trajectory dataset shim.** Export avoiding windows once, offline, from the existing `SequenceDataset` to `.npy`: `x` `(N, H, D)` float32 already `LimitsNormalizer`-scaled, plus `cond` `(N, D)` for the `t=0` inpainting state. ⚠️ **Split by EPISODE, not window** (POST_U10_III §4.2 — at H=8 adjacent windows share 7 of 8 frames; a window-level split reports a train loss and calls it validation). Hold out ~19 of 96 episodes. Replace `utils/input_pipeline.py::create_split` with a numpy loader; keep the `(local_device_count, -1, ...)` reshape (it is a no-op at 1 device).

**U2 — conditioning: use inpainting, keep the class axis trivial.** ⭐ This is the part that makes Path U *less* absurd than it first looks: **DPCC/HardFlow conditioning is architecture-free.** `apply_conditioning` pins observed dims into `z`; nothing in `imfDiT` needs to learn it. So:
- `num_classes = 1`, every sample class 0, null token retained → CFG machinery stays structurally intact.
- Set `cfg_beta`/`s_max` so `omega ≡ 1` (CFG off) for the first run; a 1-class CFG steers nothing.
- In `imf.py::forward`, after `z_t = (1-t)*x + t*e`, pin conditioned dims of `z_t`, and zero the same dims of `v_t`, `v_c`, `v_g`. ~6 lines, mirroring `_p_losses_imf_official` W-steps exactly.

**U3 — architecture: 1D tokens.** `imfDiT` patchifies a square latent and uses 2D RoPE. For `(H=8..16, D=6)`, the clean move is `patch_size=1` over the horizon axis, `D` as channels ⇒ `H` tokens, and **1D RoPE over the horizon**. 🔴 This is the highest-silent-risk item: a wrong RoPE trains fine and generates plausible garbage. Mitigate by porting Gen3v4's `imf_dit_trajectory.py` RoPE choice, which was already verified equivalent to the official real-valued form (kill-table rows 12/16/17).

**U4 — infra.** New conda env on i6-gpu-1: `jax[cuda]` (not `jax[tpu]`), `flax`, `optax`, `orbax-checkpoint`, `ml_collections`. Drop `tensorflow`/`keras`/`tfds` (only used by the FID path) and stub `utils/fid_util.py` + `utils/sample_util.get_fid_evaluator`. Keep `pmap` — it degrades to 1 device cleanly. New sbatch script under `Slurm_Codes/` following the existing submit pipeline.

**U5 — instrument BEFORE launching.** POST_U10_III §6 is right and it applies doubly here: a 20-hour job that logs only the adaptive `loss` reproduces the U9 mistake at 4× cost. Log from step 0: `raw_mse_u`/`raw_mse_v`, per-dim RMS, **`h`-stratified residual** (buckets `[0]`, `(0,0.3)`, `[0.3,0.6)`, `[0.6,1.0]`), endpoint error `‖x̂1 − x1‖` at the sampler's own grid `{(0,1), (0,0.5), (0.5,0.5)}`, and episode-level val versions of all four.

**U6 — weight export.** Flax pytree → torch `state_dict`. Use the `origin/torch` `imfDiT` as the target definition (with U3's 1D modification mirrored). Transpose every `Dense` kernel (`[in,out]`→`[out,in]`); map `scale`/`bias` for LayerNorm and QK-RMSNorm; carry the prefix tokens (`class/omega/t_min/t_max/time`) as parameters, not buffers. **Gate:** identical input ⇒ JAX and torch `u`,`v` agree to `<1e-5` relative. Do not proceed on a failed gate.

**U7 — eval bridge.** Wrap the torch model as an `ImfFlowPolicy` so `imf_sample` + the DPCC/CasADi projection can drive it, then run the **matched-NFE battery from fix_7.3** — same K grid, same n, same pre-registered rules. Anything less is not comparable to the existing verdict.

**Realistic cost:** ~2–3 weeks of implementation before the first meaningful number, with two high-risk silent-failure points (U3, U6). **Expected outcome given §4: the same refutation, more expensively.**

---

## 6. ⭐ Path P — the parity harness (this is what your instinct actually wants)

**Goal:** decide "is our iMF wrong?" against an external oracle, without training anything.

**Why it works:** `imf.py::forward` is a pure function of `(images, labels, params, rng)`. Nothing about it needs TPUs, ImageNet, or VAEs. Feed it a `(B, 4, 4, 1)` toy tensor and it runs on CPU JAX in seconds.

**Procedure:**
1. New conda env on the cluster (or a CPU box): `jax[cpu]`, `flax`, `numpy`. No TF, no orbax, no TPU.
2. Instantiate upstream `iMeanFlow` with the **smallest** `imfDiT` preset, `num_classes=1`, tiny `img_size`.
3. Export its Flax params to a torch `state_dict` **for a torch mirror of `imfDiT`** — take the mirror from `origin/torch`, which removes all guesswork about layer semantics. (This is U6 at toy scale: same code, 1/10 the risk, and it de-risks U6 if you later go that way.)
4. Fix `rng` and all sampled quantities (`t, r, fm_mask, e, omega, t_min, t_max, drop`) — **pass them in from numpy** rather than sampling inside, on both sides.
5. Compare, in order: `z_t` → `v_t` → `v_c` → `v_g` → `u` → `du_dt` → `V` → `loss_u`, `loss_v`. **The first tensor that disagrees localises the bug exactly.**
6. Repeat with the `τ = 1 − t` flip applied to the inputs, against `ImfMatcher` / `_p_losses_imf_official`.

**What it decides:**

| outcome | conclusion |
|---|---|
| all tensors match to ~1e-5 | **The port is correct.** §4's refutation stands, the objective/data are the problem, and Path U is definitively pointless. |
| divergence at `u`/`du_dt` | architecture or JVP-wiring bug — localised to one call |
| divergence at `v_c`/`v_g` | the §2.4 tangent issue and/or CFG wiring — localised to `guidance_fn` |
| divergence at `V`/`loss` | sign or algebra — the one thing §2.1 says is right, so this would be the big catch |

**Cost:** ~1 day. **No GPU. No retraining. No cluster queue.** It is strictly dominated-by-nothing: it is a prerequisite for Path U (step U6 is the same code) *and* it settles the question Path U was proposed to answer.

⚠️ Caveat: step 3 is not free — the toy converter is real work, and the ports use *different backbones* (Gen13 U-Net, Gen3v4 DiT). For Gen13 the honest scope is **objective-level parity only**: wrap the *same* toy torch network in both `ImfMatcher` and a torch transcription of upstream `forward`, and compare losses. That still tests everything in §2.1–§2.4 and needs no converter at all — **do this variant first, it is half a day.**

---

## 7. New ideas — ranked by expected value

### 7.1 ⭐⭐ Distil the validated FM teacher into the `u`-field (the one I would actually do)

**The observation nobody has written down in this repo:** you already own a teacher that works. fix_7.3 §2: **FM @ K=2 → 100% safe, 0.1894 s/plan.** iMF's entire problem is that its `u`-field is trained by a *self-referential residual* with a blind direction. **But `u` has a closed-form supervised target once you have a good `v`:**

```
u*(z, τ, h) = (1/h) ∫_τ^{τ+h} v_FM(z_s, s) ds        computed by rolling the FROZEN FM teacher
loss = || u_θ(z, τ, h) − u*(z, τ, h) ||²             plain regression. no JVP.
```

**What this deletes, one for one, from the diagnosed failure list:**

| failure mechanism | status under distillation |
|---|---|
| blind direction `δ_u = h·δ_D` (COMPARE §8.2) | **gone** — the target is a fixed tensor, not a function of the network |
| `h`-coverage gap, 0.11% mass at `h=1` (COMPARE §7.3) | **gone** — sample `h` uniformly, or *only* at `{1.0, 0.5}`, the grid you actually use |
| JVP variance spikes (7,548 max) | **gone** — no JVP |
| "converged means self-consistent, not correct" (COMPARE §8.3) | **gone** — the loss now measures `u` accuracy directly, the first metric in this project that does |
| 96-episode data ceiling | **improved** — the teacher supplies a *dense* target at every `(z, τ, h)` you care to sample, including off-manifold `z`. Data augmentation for free. |

**Cost:** low. Reuses the existing `TemporalImfUnet`/`imf_dit_trajectory` (unchanged), the existing sampler (unchanged), and the existing eval battery (unchanged). The only new code is a target generator: roll the frozen FM teacher `M` sub-steps over `[τ, τ+h]` and store the displacement. `M=8–16` sub-steps is plenty and it is a one-off cost per batch, no gradient through it.

**Risk:** the student inherits the teacher's ceiling — it cannot beat FM@K=2 on *accuracy*. That is fine: **the goal is to match FM's quality at K=1 with 1 NFE**, i.e. a genuine 2× speedup over the best known config, on a fair matched-budget comparison. That is a real, defensible claim, and it is exactly how few-step generative models are made to work in practice.

**Kill criterion, pre-registered:** if distilled-iMF @ K=1 does not reach ≥95% safety at <0.12 s/plan (FM@K=1 is 95% @ 0.1119), stop — the u-field is not representable at this capacity/data scale and the honest conclusion is the fix_7.3 one.

### 7.2 ⭐ Fix §2.3 + §2.4 and re-run the `h`-stratified diagnostic

The cheapest experiment with a real chance of moving the needle:
1. `condition_on_tau=False` in `TemporalImfUnet` + matching `jvp_tangents` change (§2.3) — brings Gen13 in line with both upstream *and* its Gen3v4 sibling, and removes a channel for the blind direction.
2. `v_c` tangent at `h=0` (§2.4).
3. **Implement the `h`-stratified residual** — COMPARE §7.4.1, *"nearly free, and it either proves or kills the leading hypothesis"*, **still not implemented**. The per-sample errors already exist before `.mean()`; this is ~5 lines.
4. Train-where-you-sample: force a fixed fraction of each batch onto `{(0,1), (0,0.5), (0.5,0.5)}`.

One retrain, four changes, all cheap. **Do (3) even if you do nothing else** — it is a log-only change that finally makes the existing metric mean something.

### 7.3 Use `origin/torch` as an architecture oracle (do this regardless)

Vendor `origin/torch`'s `imfDiT` into `third_party/` and diff it against `flow_matcher_v3_imeanflow/models/imf_dit_trajectory.py`. It is an *official PyTorch* transcription of the same JAX module — it settles RoPE conventions, QK-norm placement, SwiGLU details, zero-init gates, and prefix-token handling **without any JAX at all**. Half a day, zero risk, permanently useful, and it is a prerequisite for both §6 step 3 and §5 U6.

### 7.4 The honest reframe of the whole line of work

fix_7.3 §7 already found the publishable result and it is not about iMF: **HardFlow-FM runs at K=2 with 100% safety — a 4.5× speedup over its own default K=10.** If iMF stays dead after §7.1/§7.2, the paper writes itself as *"MeanFlow's few-NFE advantage does not transfer to low-data constrained control, because the identity's conditioning degrades precisely in the large-`h` regime few-NFE requires"* (COMPARE §8.6) — a legitimate negative result, backed by a pre-registered matched-budget battery. **That is a better paper than a marginal iMF win would have been**, and §7.1 is the one experiment that could still overturn it.

---

## 8. Recommendation

| priority | action | cost | decides |
|---|---|---|---|
| **1** | §6 Path P — objective-level parity harness (the half-day variant first) | 0.5–1 day, no GPU | **"Is our iMF wrong?"** — outright, against an external oracle |
| **2** | §7.2 items (1)(2)(3) — τ-conditioning, `v_c` at `h=0`, `h`-stratified residual | ~1 day + 1 retrain | closes the two real deviations; tests the leading failure hypothesis |
| **3** | §7.3 — vendor the torch `imfDiT` as an architecture oracle | 0.5 day | permanent de-risking, prerequisite for anything else |
| **4** | §7.1 — **FM→iMF distillation** | ~1 week | the only remaining route to a *positive* iMF result |
| **5** | §5 Path U — train inside `imeanflow` | ~2–3 weeks, 2 silent-failure points | only worth it for a *"trained with the official implementation"* paper claim, and only after 1–4 |

**Direct answer to the question as asked:** the port is not the problem, but it has never been proved not to be — so run the parity harness (§6) rather than the repo migration (§5). "Feed avoiding data into imeanflow instead of ImageNet" is a bigger rewrite than the port it would replace, its PyTorch escape hatch does not exist (inference-only branch), and the two mechanisms that killed iMF (§4) follow you across the repo boundary. **If you want iMF to work on this task, distillation (§7.1) is the idea worth spending the weeks on.**

---

## 9. ⭐ ADDENDUM (same day, later) — two new upstreams landed in `aux_repo`, and they change §3.2, §6 and §7

The user added **`/workspaces/aux_repo/MeanFlow`** and **`/workspaces/aux_repo/alphaflow`** after the above was written. Both bear directly on this document. **Two conclusions above are now superseded — flagged 🔄 below.** Everything in §2 (the deviation audit) and §4 (why upstream training wouldn't help) stands unchanged.

### 9.1 The MeanFlow lineage, straight

| repo | paper | impl | trains? | role for us |
|---|---|---|---|---|
| `MeanFlow` (haidog-yaqub) | MeanFlow 2505.13447 **+ iMF 2512.02012** | **PyTorch, unofficial** | ✅ **yes — both modes** | ⭐ **the missing oracle** |
| `imeanflow` (Lyy-iiis) | iMF 2512.02012 | JAX `main` / torch branch | ✅ JAX only | authoritative loss; arch oracle |
| `alphaflow` (snap-research) | α-Flow 2510.20771 | **PyTorch, official** | ✅ yes | ⭐ **attacks our exact failure mode** |
| — | "Understanding, Accelerating, and Improving MeanFlow Training" (FID 2.87) | — | — | not vendored; worth pulling |

Per the user's note the lineage is: **MeanFlow (baseline, FID 3.43)** → independent 2025–26 follow-ups — **iMF (FID 1.72)**, **α-Flow**, and the "Understanding/Accelerating" paper (FID 2.87) — each attacking a *different* weakness of the same baseline (training-target instability, gradient conflict, training dynamics). **We bet the entire Gen3v4/Gen13 line on one branch (iMF) without knowing the others existed.**

### 9.2 🔄 Supersedes §3.2 — a PyTorch iMF trainer *does* exist

§3.2 concluded *"You cannot train in PyTorch upstream. There is no upstream PyTorch trainer."* **True of the official repo, false in general.** `aux_repo/MeanFlow` is 831 LOC of PyTorch that trains both MeanFlow and iMF:

```python
# meanflow.py — mode='i-meanflow'
v_est  = u_p + (t_ - r_) * stopgrad(dudt)
mf_loss = adaptive_l2_loss(v_est - stopgrad(v_hat), ...)     # the iMF V-form
```

That is **exactly the objective both our ports implement.** ✅ Independent third-party corroboration that §2.1's algebra is right.

Three implementation differences worth noting:

1. **JVP under `no_grad` + a separate grad-enabled forward.** It computes `dudt` with `torch.autograd.functional.jvp(..., create_graph=False)` inside `no_grad`, then re-runs `u_p, v_p = model(...)` with grad. Semantically identical to our `torch.func.jvp(..., has_aux=True)` (grad through `u`, `dudt` detached) but costs an extra forward. **Our approach is the more efficient one** — no change needed.
2. **`model(z, t, r, ...)` — it passes BOTH `t` and `r`,** unlike official `imfDiT` which discards `t`. So the ecosystem is genuinely split on §2.3, and Gen13's choice is *not* unique. This softens §2.3 from "undocumented drift" to "a real fork in the design space that Gen13 took silently and Gen3v4 took the other way." **The §2.3 recommendation still stands** — make it a flag, and test both — but it is less damning than written.
3. `adaptive_l2_loss` uses `mean` over dims with `p = 1 − γ`, `γ=0` default; ours uses `sum` with `p=1`. Under Adam a constant rescale is absorbed (COMPARE §5), so this is cosmetic — but it is a third data point that nobody normalises this the same way.

**Consequence for §6:** the parity harness gets much cheaper and better. Instead of writing a JAX↔torch converter, wrap the **same torch network** in `MeanFlow.loss(mode='i-meanflow')` and in our `ImfMatcher`, feed identical `(x, t, r, e)`, and diff `u`, `dudt`, `v_est`, `loss`. **No JAX, no converter — a few hours, not a day.** Do this version. Keep the JAX comparison as a tie-breaker only if torch-vs-torch disagrees.

### 9.3 ⭐⭐ α-Flow attacks precisely the mechanism COMPARE §8.2 blamed

This is the important one. α-Flow's contribution is **replacing the JVP target with a self-bootstrapped finite-difference target**, `src/training/loss.py::AlphaFlowLoss`:

```python
# _compute_mean_velocity_d  (all under @torch.no_grad)
x_t_minus_dt = x_t - dt * velocity_cfg
mean_velocity_next = net(x_t_minus_dt, sigma_next=t_next, sigma=t - dt, ...)
u_tgt = (dt * velocity_cfg + (t - dt - t_next) * mean_velocity_next) / (t - t_next)
u_tgt = torch.clip(u_tgt, -clamp_utgt, +clamp_utgt)         # clamp_utgt: 4.0
```

with `dt = alpha * (t − t_next)`, and the headline recipe annealing **α: 1.0 → 0 on a sigmoid schedule** (`experiments-alphaflow.yaml:155`: `{scheduler: sigmoid, initial_value: 1.0, end_value: 0, change_end_steps: 400000, gamma: 25.0, clamp_value: 0.005}`).

**Read what that does:**
- **α = 1** ⇒ `dt = t − t_next` ⇒ `u_tgt = velocity_cfg` — pure flow matching, **no JVP at all**.
- **α → 0** ⇒ the continuous limit ⇒ recovers the MeanFlow/iMF JVP objective (`_compute_mean_velocity_c`, which *is* a JVP with tangent `velocity_cfg`).
- So α-Flow is a **curriculum from FM to MeanFlow**, starting at the well-posed end and introducing the differential constraint only as the field becomes accurate.

**Why this matters here, point for point against our diagnosis:**

| our diagnosed failure | what α-Flow does about it |
|---|---|
| **blind direction `δ_u = h·δ_D`** (COMPARE §8.2) — the loss cannot see errors in `u` because the target contains the net's own JVP | at α>0 the target is a **fixed tensor computed under `no_grad`** ⇒ **the blind direction is gone** for that fraction of training. This is the same structural fix as §7.1 distillation, **but self-bootstrapped — no teacher required.** |
| **JVP variance spikes** (Gen13 max 7,548; Gen3v4 max 327) | no JVP at α>0, **plus explicit `clamp_utgt: 4.0`** — a target clamp, which neither of our codebases has |
| **`h`-coverage: 0.11% mass at `h=1`** (COMPARE §7.3) | `ratio_fm` is scheduled and set **0.25–0.75** (vs our 0.25/0.5), and the α-anneal means large-`h` samples get a *tractable* target early |
| **"converged = self-consistent, not correct"** (COMPARE §8.3) | at α>0 the loss measures `u` against a fixed target ⇒ **the metric means what we always assumed it meant** |
| **early-training instability** | starting at α=1 means the first phase is ordinary FM — the regime our FM baseline already nails at 100% safety |

⚠️ Note the released *default* `configs/loss/alphaflow.yaml` ships `alpha.initial_value: 0.0` — i.e. plain JVP MeanFlow. The α-anneal lives in the **experiment** configs. Read `infra/experiments/experiments-alphaflow.yaml`, not the loss default, or you will replicate the baseline by accident.

### 9.4 🔄 Revised recommendation

α-Flow's α-anneal is **cheaper than §7.1 distillation and attacks the same root cause**: both replace a self-referential target with a fixed one; distillation buys the target from a validated FM teacher, α-Flow bootstraps it from the network itself. Distillation is the stronger guarantee (our FM teacher is *measured* at 100% safe @ K=2); α-Flow is the smaller change and needs no second model. **They compose** — α-anneal with the FM teacher supplying `velocity_cfg` is a coherent hybrid.

Updated priority (replaces §8):

| priority | action | cost | changed? |
|---|---|---|---|
| **1** | §6 parity harness, **torch-vs-torch against `aux_repo/MeanFlow`** | **~half a day** | 🔄 cheaper, no JAX |
| **2** | §7.2 — τ-conditioning flag, `v_c` at `h=0`, **`h`-stratified residual** | ~1 day + retrain | unchanged (§2.3 softened by §9.2.2) |
| **3** | ⭐ **Port α-Flow's α-anneal + `clamp_utgt` into the existing `ImfMatcher`** | ~2–3 days | 🆕 **new — now the top experiment** |
| **4** | §7.1 FM→iMF distillation | ~1 week | demoted below 3; still the strongest guarantee |
| **5** | §7.3 vendor torch `imfDiT` as arch oracle | 0.5 day | unchanged |
| **6** | §5 Path U — train inside `imeanflow` | 2–3 weeks | 🔄 **weaker than ever.** `aux_repo/MeanFlow` gives a PyTorch iMF trainer for ~none of the cost; the only remaining reason is a literal *"official implementation"* paper claim. |

Priority 3 is small: `ImfMatcher.loss` already has `tau`, `h`, `v_c` and a conditioning path. Adding α means computing `x_{τ+dt}`, one extra `no_grad` forward for `u_next`, the two-line target blend, the clamp, and an α schedule on `cur_step`. **`α=1` reduces it to the FM objective we know works — so the curriculum has a verified floor**, which is exactly the property every previous iMF attempt lacked.

### 9.5 Caveats on this addendum

- **`aux_repo/MeanFlow` is explicitly unofficial** (its own README says so). Excellent as a cross-check and a starting point; **not** citable as "the official implementation".
- I read `meanflow.py` and `AlphaFlowLoss` in full, but **did not run either**, and did not audit α-Flow's DiT, sampler, or hydra/torchrun infra. α-Flow is a heavier codebase than `imeanflow` (hydra + `torchrun` + video-shaped `[b,t,c,h,w]` tensors throughout — note the 5-D assert, which a trajectory adaptation must deal with).
- **α-Flow's numbers are ImageNet FID/FDD, on 400k–1.2M steps.** Its reported margin over MeanFlow at B/2 is real but modest (FID 43.1→40.2 no-cfg; 3.47→2.95 at XL/2-cfg). §4's transfer-gap argument still applies: **FID forgives what a 5 cm obstacle does not.** α-Flow makes the objective better-posed; it does not create data, and 96 episodes is still 96 episodes.
- The "Understanding, Accelerating, and Improving MeanFlow Training" paper (FID 2.87) is **not** in `aux_repo`. If its focus is gradient conflict / training dynamics it may be the closest match of all to our spike pathology — worth pulling before committing to priority 3.

---

## 10. What I did *not* verify — read this before acting

- **No code was run.** No Python in this container (per CLAUDE.md); everything above is line-level reading plus algebra done by hand. The identity re-derivation in §2.1 is mine and independent, but it is not a numerical check — that is precisely what §6 is for.
- **§2.3's third consequence is a hypothesis, not a measurement.** That τ-conditioning materially widens the blind direction is a plausible mechanism consistent with fix_7.3's flat-in-K endpoint error; it is not proven. §7.2 tests it.
- **§7.1's cost and risk estimates are engineering judgement**, not measurements. The teacher-rollout cost per batch in particular depends on `M` and has not been profiled.
- **I did not audit** `imf_flow_policy.py` (510 LOC), the projection/seam path, or `eval_imf.py`. COMPARE §6 named the inference/seam path as a leading suspect and **"Test A" (sample from the v-head through the identical sampler) does not appear to have been run** — I did not find a result log for it. If it hasn't been, it is cheap and it isolates sampler bugs from objective bugs. Worth adding to §8 priority 2.
- **Upstream is a moving target** — `bf60cd7` today; the torch branch was fetched fresh this session. Re-check before quoting.
- Per repo convention I have **not** touched `MASTER_TEST_HISTORY.md`. If you want an entry for this doc, say so and I'll add one.
