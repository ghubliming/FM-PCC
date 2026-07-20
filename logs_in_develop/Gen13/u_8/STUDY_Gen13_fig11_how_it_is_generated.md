# STUDY — How the Fig.11 ODE-step grid is generated (math + code path)

**Date:** 2026-07-20 · **Code:** `run/eval_imf.py` (capture), `run/make_fig11_ode_grid.py` (render)
**Track back:** `grep -rn "u_8.2" HardFlow/run/eval_imf.py`
**Purpose:** document exactly what each cell of the grid *is*, mathematically and in code, so the figure can be read and trusted.

---

## 0. What the grid shows

For **one planning instance** (one replan), a 2×N grid across ODE steps `k = 0 … N`:

| row | symbol | meaning |
|---|---|---|
| top | **`x_τ`** | the *actual* trajectory variable at ODE step k — where the sampler currently is |
| bottom | **`x̂₁`** | the *terminal prediction* decoded from that state — where the model thinks it will end up |

With `--both`, two backbones stack into 4 rows (iMF `x_τ` / iMF `x̂₁` / FM `x_τ` / FM `x̂₁`).

**Grid sizes:** FM has `ode_t_steps=10` ⇒ **11** chain states (subsampled to 6 columns: 0,2,4,6,8,10). iMF at K=5 has **6** states ⇒ all shown.

---

## 1. Convention (HardFlow's, used throughout)

```
z_τ = τ·x₁ + (1−τ)·x₀        τ = 0 → noise ,  τ = 1 → data
```
so the sampler integrates **from noise up to data**. (Official iMF runs the opposite direction; the mapping is quarantined in `imf/convention.py` — see `../fix_1/`.)

Per ODE step, `dt = 1/N` and `τ_k = k·dt`.

---

## 2. Top row — how `x_τ` is produced

**Important:** for the *guided* runs this is **not** a plain ODE integration. It is HardFlow's projected chain — each step is reference-step → NLP → pull-back. From `imf_hardflow_new_forward` (iMF) / `hardflow_new_forward` (FM):

**(a) reference step** — advance the state
```
FM :  x_ref = x_k + dt · v(x_k, τ_k)                    (Euler step)
iMF:  x_ref = x_k + dt · u(x_k, τ_k, dt)                (exact interval jump)
```

**(b) terminal prediction** — shoot to τ=1
```
FM :  x̂₁_ref = x_ref + (1 − τ_{k+1}) · v(x_ref, τ_{k+1})
iMF:  x̂₁_ref = x_ref + (1 − τ_{k+1}) · u(x_ref, τ_{k+1}, 1 − τ_{k+1})
```
⟵ **this is the Gen13 seam**: the only line the backbone swap changes.

**(c) hard projection** — CasADi/IPOPT prox-NLP on the *predicted endpoint*
```
X* = argmin_X  ½ ρ τ²_{k+1} ‖X − x̂₁_ref‖²      s.t.  X ∈ C
```
where `C` = obstacle quadrilaterals ∩ fitted linear dynamics (`A·s + B·a + c = s'`) ∩ action bounds.

**(d) pull-back** — map the endpoint correction back to the current state
```
x_{k+1} = x_ref + τ_{k+1} · (X* − x̂₁_ref)
```
The gain `τ_{k+1}` comes from `∂z_τ/∂x₁ = τ` for the linear interpolant: early in the trajectory (small τ) the current state barely moves even when the endpoint is corrected a lot. (`THEORY_DeepMix_HF_iMF.md` measures this gain as delivering only ~11% of the requested correction at τ=0.1 — the motivation for the un-built Newton upgrade.)

The stored chain is `x_chain[k] = x_k`, assembled at `flow_policy.py:1375–1392` (dof → full trajectory → `unnormalize_chain`). **Top row = these states.**

> For **unguided** runs there are no steps (b)–(d): the chain is the plain sampler path (`imf_sample`, K exact jumps / `ConditionedODESolver`, Euler). That is why unguided fans look chaotic — nothing is dragging them onto the feasible manifold (`../fix_7/RESULTS_Gen13_fix7_smoothness_2x2.md`).

---

## 3. Bottom row — how `x̂₁` is produced

Computed by `x1_estimate(x_chain, conditions)`, looping over **every** stored chain state with `t = k/(n_steps−1) = τ_k`:

```python
# FM   (flow_policy.py:233-244)
current_v = self.flow_model(current_x, t)
predicted_x = current_x + (1.0 - t) * current_v

# iMF  (imf/imf_flow_policy.py, x1_estimate)
current_u = self._u(current_x, t, 1.0 - t)
x1_estimation[:, k] = current_x + (1.0 - t) * current_u
```

So:
```
FM :  x̂₁(k) = x_k + (1 − τ_k) · v(x_k, τ_k)
iMF:  x̂₁(k) = x_k + (1 − τ_k) · u(x_k, τ_k, 1 − τ_k)
```

Note this is a **diagnostic re-evaluation** at each stored state — it re-queries the network, which is why it is billed to the `nfe_diag` bucket in the NFE accounting.

---

## 4. The math that makes the bottom row the interesting one

Let `x₁` be the true endpoint of the flow through `(x_k, τ_k)`.

**FM (Euler shot).** The exact endpoint is
```
x₁ = x_k + ∫_{τ_k}^{1} v(x_s, s) ds
```
but FM approximates the integrand by its value at the left edge:
```
x̂₁ = x_k + (1 − τ_k) · v(x_k, τ_k)
```
The error is therefore
```
x₁ − x̂₁ = ∫_{τ_k}^{1} [ v(x_s, s) − v(x_k, τ_k) ] ds  =  O( (1 − τ_k)² )
```
— a **first-order quadrature error that grows quadratically with the remaining time**. Worst at `τ = 0`, vanishing as `τ → 1`. This is precisely the appendix's observation that FM's early-step `x̂₁` shows *"positional shifting or deformation"*.

**iMF (exact endpoint map).** The average velocity is *defined* as the interval mean
```
u(z, τ, h) ≜ (1/h) ∫_{τ}^{τ+h} v(z_s, s) ds
```
so with `h = 1 − τ` the shot is exact by construction:
```
x_k + (1 − τ_k) · u(x_k, τ_k, 1 − τ_k)  =  x_k + ∫_{τ_k}^{1} v ds  =  x₁
```
The only error is the **network's training error on `u`** — which has no `(1−τ)²` factor and is roughly τ-independent.

**Prediction:** at **ODE step 0** (`τ = 0`, `1−τ = 1`, FM's error maximal) iMF's bottom-left cell should already sit near the final path, while FM's should be visibly displaced. By the last column both converge, since `(1−τ) → 0` kills the FM error too. **The left-hand columns of the bottom rows are the discriminator; the right-hand columns are expected to agree.**

Caveat: iMF's advantage is bounded by its own field quality, which is measurably coarser here (≈0.37/dim vs Gen3v4's ≈0.25/dim), so the effect may be partly masked.

---

## 5. Code path (capture → storage → render)

```
policy(conditions)                          # returns (action, traj, x_chain, x1_est, info)
  └─ x_chain  : (1, N+1, H, T)   ← top row,    all ODE states
  └─ x1_est   : (1, N+1, H, T)   ← bottom row, all ODE states
        │
run/eval_imf.py  _run_env_quiet()           # only when cfg.imf_plot_fan
  └─ chain_full.append(x_chain[0]) ; x1_full.append(x1_est[0])
        │                                    (u_8.2 — previously sliced [0,-1]
        │                                     and 10 of 11 states were discarded)
  └─ np.savez_compressed("{run_id}_fan.npz", chain_full=…, x1_full=…, real=…)
        │
run/make_fig11_ode_grid.py                  # pure post-processing, no GPU/sim
  ├─ pick replan instance p (default middle), subsample ODE steps to --n_cols
  ├─ per cell: _configure_axis(compact) → grey executed rollout → plot_single_trajectory(
  │            traj[:, action_dim:], style="predicted") → add_environment_elements()
  └─ save
```

**Index detail:** `plot_single_trajectory` reads x,y from **columns 2,3** (observation layout), while chain states are full transitions with x,y at `action_dim+2, +3`. Hence the `traj[:, action_dim:]` slice.

---

## 6. How to read the figure

| Observation | Interpretation |
|---|---|
| Top row chaotic at step 0 → clean at step N | normal — the sampler starts from noise; convergence is the expected behaviour |
| Top row still ragged at the LAST column | the plan itself is rough — but note the NLP normally prevents this (fix_7: guided plans are smooth for both backbones) |
| **Bottom row already localised at step 0** | the endpoint estimator is accurate early ⇒ expected for **iMF** |
| **Bottom row displaced/deformed at step 0, converging later** | the `O((1−τ)²)` Euler error ⇒ expected for **FM** |
| Bottom rows agree in the last column | expected for both — `(1−τ) → 0` |

**What would falsify the Gen13 story:** iMF's bottom-left cell being as displaced as FM's. That would mean the exact endpoint map is not delivering its theoretical advantage in practice — most likely because the `u`-field's training error dominates the Euler error at this data scale.

---

## 7. Scope note

This figure is **illustrative, not evidential** — a single planning instance cannot support a quantitative claim. The measured results live in `../u_5/RESULTS_Gen13_u5_paired_n200.md` (safety/efficiency, n=200) and `../fix_7/RESULTS_Gen13_fix7_smoothness_2x2.md` (roughness). Its value is showing the *mechanism* those numbers come from.
