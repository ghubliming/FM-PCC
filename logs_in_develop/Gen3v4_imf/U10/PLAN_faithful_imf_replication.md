# PLAN — U10 "one last shot": 100% faithful replication of the iMeanFlow (iMF) math in FM-PCC

**Gen:** Gen3v4_imf / U10 · **Date:** 2026-07-13 · **Status:** PLAN — implementation to be done by a coding agent, training/eval on cluster (**i6-gpu-1**).
**Ground truth:** `/workspaces/aux_repo/imeanflow/imf.py` (class `iMeanFlow`) + `models/imfDiT.py`.
**Target:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (`iMeanFlowODE`), `imf_dit_trajectory.py` (`IMFDiTTrajectory`), `imf_engine.py`/`imf_trajectory_model.py` (plumbing), `config/avoiding-d3il.py`, `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`.
**Prereq reading:** `debug_notes/INVESTIGATION_new_vs_upstreams_KILL_TABLE.md` (kill chain, fixed), `debug_notes/INVESTIGATION_imf_fidelity_vanilla_vs_improved_meanflow.md` (deviations D1–D6).

> **Reference convention:** pointers are `file · function/logic`; line numbers rot. The math below is the spec — the coding agent must match the FORMULAS, not paraphrase them.

---

## 0. Problem statement (why this plan exists)

Confirmed on cluster (U10): with CFG gates off, the `bbdit` checkpoint no longer explodes but is **bounded-and-jittery** — worse than the old UNet-FM and DPCC. Root cause audit: **despite the name, Gen3v4 trains ORIGINAL MeanFlow, not improved MeanFlow.** The deviations from `imf.py` (D1 tangent, D2 CFG, D3 time-sampling, D5 loss) remove exactly the parts that make iMF smooth and few-step-capable. This plan replicates the official math 1:1 (modulo three *declared* domain adaptations, §5) so the claim "we faithfully use iMF in FM-PCC" becomes true.

---

## 1. Ground-truth math (from `imf.py` — official convention: **s=0 data, s=1 noise**)

Per training batch (`forward`):
1. **Times** (`sample_tr`): `t = LN(P_mean=−0.4, P_std=1)`, `r = LN(−0.4, 1)` — two INDEPENDENT logit-normals `sigmoid(N(μ,σ))`; then `t = max, r = min`; first `data_proportion=0.5` of the batch forced `r = t` (**fm_mask**, 50% FM anchors).
2. **Interpolant:** `z_t = (1−t)·x + t·e`, `e ~ N(0,1)`; **instantaneous velocity** `v_t = e − x`.
3. **CFG knobs:** `ω = exp(u·log1p(s_max))` ∈ [1, 1+s_max] (`sample_cfg_scale`, β=1 branch; s_max default 7.0); `t_min ~ U(0,0.5)`, `t_max ~ U(0.5,1)` (`sample_cfg_interval`); fm_mask rows get `t_min=0, t_max=1`.
4. **Guided target** (`guidance_fn`):
   - `v_c, v_u = v_fn(z_t, t, ω, y)` — v-head (h=0 dummy) on a DUPLICATED batch: conditioned `(y, ω)` and unconditional `(y_null, ω=1)`.
   - `v_g_fm = v_t + (1 − 1/ω)(v_c − v_u)`
   - gate: `ω ← where(t_min ≤ t ≤ t_max, ω, 1)`; recompute `v_c = v_cond_fn(z_t, t, ω_gated, y)`.
   - `v_g = v_t + (1 − 1/ω_gated)(v_c − v_u)`; `v_g ← where(fm_mask, v_g_fm, v_g)`.
   - returns `(v_g, v_c)` — **v_c (gated) is the JVP tangent**.
5. **Label dropout** (`cond_drop`, `class_dropout_prob=0.1`): ~10% of rows get `y ← y_null` AND `v_g ← v_t`. (Official quirk: it drops the first `num_drop` rows — a batch prefix. Replicate the *semantics*: per-row Bernoulli(0.1) mask is acceptable and better-randomized; note the divergence in a comment.)
6. **JVP** (the defining iMF step): warped `u_fn(z,t,r) := net_u(z, t, h=t−r, ω, t_min, t_max, y)`;
   `u, du_dt, v = jvp(u_fn, (z_t, t, r), tangents=(v_c, 1, 0), has_aux=True)` → **tangent is the PREDICTED `v_c`**, not the analytic `v_t`.
7. **Compound + losses:** `V = u + (t−r)·stop_grad(du_dt)`; `v_g ← stop_grad(v_g)`;
   `loss_u = Σ_dims (V − v_g)²` per sample; `loss_v = Σ_dims (v − v_g)²` per sample (v = v-head at the SAME warped query, from jvp aux);
   **adaptive:** `adp(L) = L / stop_grad((L + 0.01)^{1.0})` (p=1, eps=0.01, on the per-sample **SUM**);
   `loss = mean(adp(loss_u) + adp(loss_v))` — **u and v equally weighted; no other weights.**
8. **Sampling** (`sample_one_step` / `generate`): `t_steps = linspace(1,0,N+1)`; each step ONE u-call: `z ← z − (t−r)·u(z, t, h=t−r, ω, t_min, t_max, y)`; ω/t_min/t_max are **network inputs, constant across steps**; **no output-space mixing, no dropout call at sampling**; **EMA weights** (`sample_util` `ema=True`).

---

## 2. Convention map (official s-axis → our DATA-AT-1 τ-axis) — THE TRANSCRIPTION TABLE

Gen3v4 keeps **DATA-AT-1** (τ=0 noise → τ=1 data; verified internally consistent — kill-table F2). Map: **τ = 1 − s**. Every formula must pass through this table:

| Quantity | Official (s-axis) | Ours (τ-axis) |
|---|---|---|
| interpolant | `z_s=(1−s)x+s·e` | `x_τ=(1−τ)e+τ·x` (**existing `q_sample`** ✔) |
| instantaneous v | `v = e−x` | `v = x−e` (= `x_start − x_base`, existing ✔) — all v/u flip sign together; formulas below are written in OUR convention and need no further sign work |
| anchor (net query point) | `z_t`, noise side, time `t=max` | `x_r`, noise side, **`r = min(τ₁,τ₂)`** |
| far end | `r = min` | `t = max(τ₁,τ₂)` |
| `h` | `t−r` | `t−r` (same, positive) |
| **time draws** ⚠️ | `s ~ sigmoid(N(−0.4, 1))` (mass near data, s small) | `τ ~ 1 − sigmoid(N(−0.4,1)) = sigmoid(N(+0.4, 1))` → **code: `torch.sigmoid(randn·p_std − p_mean)` with config `p_mean=−0.4`** (mass near data, τ large). **Current code uses `+p_mean` → mass near NOISE → WRONG (new deviation D3b).** |
| JVP identity | `V = u + h·sg(du/dt)` regress → `v_g` | `u_target = v_g + h·du/dr` (detached) regress `u` → target; tangents `(v_c.detach(), +1, −1)` for `(x_r, r, h)` — residual/gradient equivalence verified (kill-table row 17) |
| interval gate ⚠️ | `t_min ≤ s ≤ t_max` at the sample's noise-side time s | compute **`s_anchor = 1 − r`** and gate `t_min ≤ s_anchor ≤ t_max` — keep `t_min,t_max` in OFFICIAL s-convention everywhere (sampling of them, embeddings, eval inputs). Do NOT gate on τ directly. |
| sampler step | `z ← z − h·u` | `x ← x + h·u` (**existing `p_sample_loop` Euler** ✔ faithful) |

**Faithful already (do not touch):** `q_sample`, the Euler sampler stepping, `_sample_cfg_scale`, `_sample_cfg_interval`, DiT block port (RoPE/RMSNorm/SwiGLU/gates), `1−1/ω` embedding arg.

---

## 3. Work items (W1–W8) — file · function · exact change

### W1 — New objective branch, flag-gated (repo convention: never overwrite the old arm)
`imf_diffusion.py · iMeanFlowODE`: add `imf_objective='imf_official'` dispatch in `loss()`/`p_losses()` → new method `_p_losses_imf_official`. Keep `'meanflow_jvp'` and `'fm_equivalent'` untouched for A/B. Config `prefix` already embeds `obj{imf_objective}` → checkpoints auto-separate. ✔

### W2 — Time sampling (fixes D3 + D3b) — inside `_p_losses_imf_official`
```python
# two INDEPENDENT logit-normals on the τ axis (τ = 1 − s ⇒ negate P_mean; see plan §2)
tau1 = torch.sigmoid(torch.randn(B)*p_std - p_mean)   # p_mean=-0.4 ⇒ sigmoid(N(+0.4,1))
tau2 = torch.sigmoid(torch.randn(B)*p_std - p_mean)
t, r = torch.maximum(tau1,tau2), torch.minimum(tau1,tau2)
fm_mask = torch.rand(B) < data_proportion              # NEW config key, default 0.5 (was 0.25)
r = torch.where(fm_mask, t, r); h = t - r
```
Move time sampling INTO the new loss (bypass `loss()`'s single-t path for this branch). Add config `meanflow_data_proportion: 0.5`.

### W3 — Guided target `v_g` + predicted tangent `v_c` (fixes D1 + D2-target + D4)
All velocities in OUR convention (`v_t := x_start − x_base`, masked at cond dims as now):
```python
s_anchor = 1.0 - r                                    # official-convention time of the anchor
omega  = sample_cfg_scale(...)                        # ∈[1,1+s_max]  (existing port ✔)
t_min, t_max = sample_cfg_interval(...)               # official s-convention (existing ✔)
t_min = torch.where(fm_mask, zeros, t_min); t_max = torch.where(fm_mask, ones, t_max)

v_c_raw, v_u = v_head(x_r, h=0, omega, tmin=0, tmax=1, y=cond_class),  v_head(x_r, h=0, omega=1, ..., y=null)
v_g_fm  = v_t + (1 - 1/omega)*(v_c_raw - v_u)
omega_g = torch.where((s_anchor >= t_min) & (s_anchor <= t_max), omega, ones)
v_c     = v_head(x_r, h=0, omega_g, tmin=0, tmax=1, y=cond_class)      # GATED — this is the tangent
v_g     = v_t + (1 - 1/omega_g)*(v_c - v_u)
v_g     = torch.where(fm_mask[:,None,None], v_g_fm, v_g)
```
`v_head(...)` = the DiT's v output via `_predict_uv(..., )[1]` with **h=0, t_min=0, t_max=1 dummies** (official `v_cond_fn`); uncond via per-sample null token (W5) with **ω=1**. Apply `apply_conditioning(·, noise=True)` (zero pinned dims) to `v_t`, `v_g`, `v_c` exactly as the current code does for `v_inst` — the pinned dims carry no velocity (declared adaptation §5).

### W4 — cond_drop (trains the null token; fixes the kill-chain root at its source)
```python
drop = torch.rand(B) < class_dropout_prob             # NEW config key, default 0.1
y    = torch.where(drop, NULL, y)                     # per-sample null token (W5)
v_g  = torch.where(drop[:,None,None], v_t, v_g)
```

### W5 — Per-sample dropout in the DiT (implementation gap)
`imf_dit_trajectory.py · IMFDiTTrajectory._build_sequence`: `force_dropout` is currently a **batch-wide bool** (`y_idx = num_classes if force_dropout else 0`). Change to also accept a **per-sample bool tensor** → `y_idx = drop_mask.long() * num_classes`. Thread through `forward` / `iMFTrajectoryModel.forward` / `iMeanFlowEngine.forward_train` / `_predict_uv` (accept `force_dropout: bool | Tensor`). Backward-compatible: bools keep working.

### W6 — JVP with predicted tangent + aux v (the defining change, D1)
```python
def _uv_of(z, r_in, h_in):
    u, v_aux = self._predict_uv(z, cond, r_in, h=h_in, returns=None,
                                omega=omega, t_min=t_min, t_max=t_max, force_dropout=drop)
    return u, v_aux
(u_pred, du_dr), v_aux = ... = torch.func.jvp(_uv_of, (x_r, r, h),
                                (v_c.detach(), ones, -ones), has_aux=True)
u_target = (v_g + h_expand * du_dr).detach()          # ours-form of V=u+h·sg(du/dt); equiv. verified
u_target = apply_conditioning(u_target, ..., noise=True)
```
Notes: tangent **`v_c.detach()`** (official's `stop_gradient(du_dt)` kills that grad path — detaching the tangent is the torch equivalent); ω passed to the u-query is the **UNGATED** ω (official `forward` does); labels are POST-drop (official). `torch.func.jvp(..., has_aux=True)` exists in torch ≥ 2.0 — keep the functorch fallback the current file has.

### W7 — Loss composition (fixes D5): NO DPCC weights, official adaptive
```python
loss_u = (u_pred - u_target).pow(2).sum(dim=(1,2))    # per-sample SUM (official), NOT mean
loss_v = (v_aux  - v_g.detach()).pow(2).sum(dim=(1,2))
adp    = lambda L: L / (L + 0.01).detach().pow(1.0)   # p=1.0, eps=0.01
loss   = (adp(loss_u) + adp(loss_v)).mean()
```
**Do NOT multiply `self.loss_fn.weights`** (action_weight/discount) in this branch — official has none. Keep `info['raw_mse']`, `a0_loss` etc. as **metrics only** for W&B parity. Ignore `meanflow_adaptive_p/c` and `meanflow_aux_weight` in this branch (log a warning if set).

### W8 — Sampler + eval semantics (fixes D2-sampling; mostly deletion)
`imf_diffusion.py · _predict_velocity`: for this branch **delete both output-space mixes** — the returns-CFG block AND the `cfg_scale>0` interval mix. Sampling = ONE u-call per step with constant inputs `(ω_eval, t_min_eval, t_max_eval)` fed to the net; **ω=1 ⇒ guidance off** (not 0 — new semantics!). `p_sample_loop`: drop the `step_cfg` τ-gating (the interval behavior lives in the net's weights).
Config split (train ceiling vs eval operating point — they were conflated):
- `meanflow_cfg_smax: 7.0` (NEW; training `sample_cfg_scale` ceiling — official default)
- `meanflow_cfg_omega: 1.0` (now EVAL operating point only; 1.0 = off) + `meanflow_cfg_t_min/t_max` (eval inputs, official s-convention)
- `eval_use_ema: True` (official samples with EMA)
Remove `returns` plumbing from this generation's Policy→model path (it conditions nothing; its only historical effect was arming the broken gate). Update `Slurm_Codes` sbatch/plan block for the new keys (per repo convention).

---

## 4. Traps for the coding agent (each has bitten this repo already)
1. **P_mean sign** (W2): `sigmoid(randn·σ + (−0.4))` on the τ-axis is WRONG; must be `− p_mean` (τ = 1−s). The current U7 comment *claims* it matches the reference — it doesn't; the axis is flipped.
2. **Interval gate in s-space** (W3): gate on `s_anchor = 1−r`, never on τ/r directly. Keep `t_min,t_max` official-convention end-to-end (train sampling, embeddings, eval inputs).
3. **Tangent detach** (W6): pass `v_c.detach()`; do NOT let loss_u backprop into the v-head through the JVP.
4. **ω domain** (W3/W8): ω ∈ [1, 1+s_max]; "off" = 1, not 0. The DiT's `w_arg = where(w>0, 1−1/w, 0)` maps ω=1→0 correctly — leave it.
5. **SUM not MEAN** (W7): official adaptive normalizes the per-sample **sum**; using mean changes the effective eps regime.
6. **v-head query dummies** (W3): `h=0, t_min=0, t_max=1` for every v evaluation (official `v_cond_fn`), while the u JVP query uses the real `(h, t_min, t_max)`.
7. **Per-sample force_dropout** (W5): a batch-wide bool cannot express cond_drop; the DiT change is mandatory, and must stay backward-compatible with plain bools.
8. **Don't touch the siblings**: `imf_visual_aligning/` (Gen8) mirrors this engine — do NOT sync until this arm is validated on cluster (commit-message convention will carry the sync later).
9. **Cost**: W3 adds 3 extra v-head forwards per step (official pays the same). Batch the cond/uncond pair (official `v_fn` concatenates) if easy; correctness first.

## 5. Declared domain adaptations (NOT infidelities — cite these three, nothing else)
1. **DATA-AT-1 time flip** (τ=1−s) with the §2 transcription table — algebra-equivalent (verified).
2. **Inpainting conditioning** (`apply_conditioning`): pin observed dims of `x_r`; zero those dims in `v_t/v_c/v_g/u_target` and the tangent. Trajectories need a current-state anchor; images don't.
3. **Trajectory DiT sizing** (`dit_*` config) + single-class labels (y∈{0}, null=1) + DPCC projector at sampling (unchanged, runs after the faithful generative step).

## 6. Verification protocol (cluster; nothing runnable locally)
1. **Unit — gradient equivalence:** toy tensors, assert ours-form residual `u − (v_g + h·du/dr).detach()` ≡ official V-form `(u + h·sg(du/dr̂)) − sg(v_g)` gradients (they were proven equal for the vanilla case; re-assert with v_c tangent).
2. **Unit — FM anchor:** on fm_mask rows (h=0): u_target reduces to `v_g_fm`; and `u(x,r,h=0) ≈ v_head(x,h=0)` after some training.
3. **Unit — time distribution:** histogram τ draws; mass must sit near **τ=1 (data)**, median ≈ 0.60.
4. **Train smoke (short run):** `loss_u`, `loss_v` both finite/decreasing; null-token embedding grad-norm **> 0** (the old bug's signature was exactly 0).
5. **Eval sweep, same checkpoint:** N ∈ {1, 2, 10, 50}, ω=1, EMA on. Success criteria: N=1/2 produce coherent (not garbage) trajectories — the iMF capability the old arm never had; N=10 ≥ old arm at N=10.
6. **A/B:** `obj=imf_official` vs `obj=meanflow_jvp` (old), same seeds/epochs — smoothness (visual + tracking error) and success-rate vs UNet-FM (Gen7) and DPCC baselines.
7. **CFG check (after base passes):** eval ω ∈ {1, 1.5, 2, 4} × interval `[0.4,0.6]` (net inputs only) — quality should now RESPOND to ω (the old arm's ω was dead).

## 7. Acceptance criteria for the claim "we faithfully use iMF"
- [ ] Predicted-`v_c` JVP tangent (W6) — the defining iMF feature — in the training path actually used.
- [ ] Guided `v_g` target + `cond_drop`-trained null token (W3/W4); `loss_u+loss_v` official adaptive (W7).
- [ ] Two-logit-normal (correct axis) + 50% FM anchors (W2).
- [ ] Sampler: single u-call, input-conditioned CFG, EMA (W8).
- [ ] §5's three adaptations documented wherever the claim is made.
- [ ] §6 checks 1–5 pass. → Then the sentence *"we faithfully replicate the iMF training objective and sampler, with three documented domain adaptations"* is TRUE. Until all boxes tick, keep "iMF-inspired".

## 8. Out of scope (explicitly)
UNet arm (`imf_backbone='unet'` — not iMF, leave as A/B relic) · Gen8 `imf_visual_aligning` sync · DiT scale-up (D6, revisit only if quality still short AFTER fidelity) · `repeat_last` latent bomb (kill-table row 6) · MASTER_TEST_HISTORY update (offer separately, never self-edit).
