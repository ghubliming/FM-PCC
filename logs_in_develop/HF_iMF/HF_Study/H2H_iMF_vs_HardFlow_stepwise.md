# H2H — iMF vs. original HardFlow, step by step (real code + math)

**Date:** 2026-07-23 · **Scope:** the `avoiding` task, Gen13 code in `/workspaces/FM-PCC/HardFlow`
**Companions:**
`MAP_Algorithm1_to_AvoidingCode.md` (paper Algorithm 1 → FM code — read that first if the paper side is unclear)
`../Research/BLEND_HardFlow_iMeanFlow.md` (why the seam is where it is) ·
`../Research/THEORY_DeepMix_HF_iMF.md` (Thm 1/2 — which *field* each side estimates) ·
`../../Gen13/u_8/STUDY_Gen13_fig11_how_it_is_generated.md` (the figure that renders this comparison) ·
`../../Gen13/fix_7/RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md` (**what actually happened when measured**).

**What this doc is:** the two samplers laid side by side, one algorithm step per section, with the
exact file:line that executes each step and the math that line implements. Both columns are running
code today — `run/eval_imf.py --backbone fm|imf` drives them through *one* entry point, so this is a
literal A/B, not a paper-vs-paper comparison.

**Bottom line up front:** the two paths differ in **exactly two lines of arithmetic** inside the
per-step loop (plus the network that those two lines query, and its training objective). Everything
else — warmstart structure, the CasADi NLP, the pull-back, the chain bookkeeping, the diagnostics —
is the *same object*, reused by subclassing, not copied. And when measured at matched budget, the
two changed lines made things **worse**, for a reason that is not in the algorithm (§10).

---

## 0. The two objects being compared

| | **original HardFlow (FM)** | **Gen13 (iMF)** |
|---|---|---|
| Field learned | instantaneous velocity `v(z, τ)` | **average** velocity `u(z, τ, h)` over `[τ, τ+h]` |
| Defining identity | `v = E[ẋ_τ ¦ z_τ = z]` | `u(z,τ,h) ≜ (1/h)·∫_τ^{τ+h} v(z_s, s) ds` ⇒ `z_{τ+h} = z + h·u` **exactly** |
| Network | `TemporalUnet(x, t) -> v` | `TemporalImfUnet(x, tau, h) -> (u, v)` |
| net file | `models_flow/unet.py:347` `forward(x, time)`, final conv → `transition_dim` ch. | `models_flow/imf/temporal_imf_unet.py:150` `forward(x, tau, h)`, final conv → `2*transition_dim` ch., `torch.chunk` → `(u, v)` (`:134-137`, `:187`) |
| time conditioning | `t = time_mlp(t)` | `t = time_mlp(tau) + h_mlp(h)` (`temporal_imf_unet.py:163`) — two sinusoidal MLPs summed |
| Policy class | `FlowPolicy` (`flow_policy.py:117`) / `InstrumentedFlowPolicy` (`imf_flow_policy.py:99`) | `ImfFlowPolicy(WarmstartCaptureMixin, FlowPolicy)` (`imf_flow_policy.py:118`) — **additive subclass, base never edited** |
| guided entry | `hardflow_new_forward` (`flow_policy.py:1286`) | `imf_hardflow_new_forward` (`imf_flow_policy.py:348`) |
| unguided entry | `__call__` "original" branch (`flow_policy.py:147,173-197`), `ConditionedODESolver`, Euler | `imf_original_forward` (`imf_flow_policy.py:291`) → `imf_sample` (`imf/imf_sampler.py:22`) |
| `ode_t_steps` default | `20` (`config/flow_matching.py:65`) | `2` = K (`imf/imf_config.py:67`) |

Time convention is **HardFlow-native on both sides**: `z_τ = τ·x1 + (1−τ)·x0`, `τ=0` noise → `τ=1`
data. The official iMeanFlow repo runs the opposite direction; that mapping is quarantined in
`imf/convention.py` (module docstring, lines 1-53) and **no sign flip exists anywhere at sampling
time**. If a sign is ever suspected, audit that one file.

---

## 1. Step 0 — warmstart (draw `x_0`)

Algorithm 1 says "draw `x̄_0 ~ p_0`". Both implementations instead roll out `warmstart_batch`
unconstrained candidates and keep the one the value model likes best (engineering, not math —
`MAP_Algorithm1_to_AvoidingCode.md` §6).

| | FM `flow_policy.py:753-795` | iMF `imf_flow_policy.py:199-238` |
|---|---|---|
| per-step update | `dx_dt = flow_model(x, t)` ; `x = x + dx_dt*dt` (`:769,773`) | `u = self._u(x, t, dt, bucket="warmstart")` ; `x = x + u*dt` (`:215,219`) |
| math | Euler: `x_{k+1} = x_k + dt·v(x_k, τ_k)`, local error `O(dt²)` | exact jump: `x_{k+1} = x_k + dt·u(x_k, τ_k, dt)` — **no discretisation error by definition of `u`** |
| conditioning | `apply_conditioning_from_conditioned_x` zeroes the update at inpainted state slots (`:770`) | identical (`:216`) |
| selection | `argmax` of `value_model(x_chain[-1])`, then split into `(s0, dof_chain)` (`:776-793`) | byte-identical logic (`:222-238`) |
| `no_grad` | ❌ **absent** — the base builds an autograd graph here | ✅ `with torch.no_grad()` (`:214`) |

⚠️ **Asymmetry worth knowing:** the base FM `warmstart` never wraps in `no_grad`. It is outside the
timed region (`t_start` is set *after* warmstart on both sides — `flow_policy.py:1305`,
`imf_flow_policy.py:369`), so it does not corrupt `computation_time`, but it does mean the FM path
allocates a graph it never uses. Not a Gen13 change; inherited.

Both then stash the raw pre-projection chain via `WarmstartCaptureMixin._stash_warmstart`
(`imf_flow_policy.py:60-61`, called at `:109` for FM and `:237` for iMF) — that is how the
raw-vs-projected diagnostic works identically for both backbones.

---

## 2. Step 1 — the loop and the grid

Identical on both sides:

```python
dt = 1.0 / self.oc_N_steps          # flow_policy.py:1320   |  imf_flow_policy.py:384
for k in range(self.oc_N_steps):    #             :1321     |              :385
    t_k = k * dt                    #             :1322     |              :386
    x_k = X_optimized[k]            #             :1323     |              :387
```

`oc_N_steps = cfg.ode_t_steps` (`flow_policy.py:700`) — uniform grid, so `Δt_i` is a constant and the
paper's `1/Δt_i` is pre-folded into `hardflow_reg_scale`. **`N` means different things by default**
(FM 20 / iMF 2) — matching it is the whole point of the fix_7.3 battery (§10).

---

## 3. Step 2 — the reference step ⟵ **SEAM #1**

**Paper:** `x̄_{i+1} = x_i + v_{t_i}(x_i)·Δt_i`.

| | code | math |
|---|---|---|
| **FM** | `flow_policy.py:1324-1325`<br>`v_k = flow_eval_np(x_k, t_k)`<br>`x_next_ref = x_k + v_k * dt` | `x̄_{k+1} = x_k + dt·v(x_k, τ_k)` — literal Euler, local truncation `O(dt²)` |
| **iMF** | `imf_flow_policy.py:389-391`<br>`u_k = u_eval_np(x_k, t_k, dt)`<br>`x_next_ref = x_k + u_k * dt` | `x̄_{k+1} = x_k + dt·u(x_k, τ_k, dt)` = `z_{τ_k+dt}` **exactly** (definition of `u`), so the transport is discretisation-free |

Both `*_eval_np` closures do the same three things — reshape the free-DOF vector to a full trajectory,
splice the conditioned `s0` back in, call the net, strip `s0` out again:

* FM: `flow_eval_np` (`:1310-1314`) → `constrained_flow_fn_torch` (`:1695-1719`)
* iMF: `u_eval_np` (`:374-378`) → `constrained_u_fn_torch` (`:267-287`) — a line-for-line mirror with the extra `h` argument and NFE accounting.

The `s0` splice exists because `oc_dof = horizon*(state+action) − state_dim` (`flow_policy.py:701`):
the optimizer never touches `s0`, but the network always sees it. Unchanged by the swap.

---

## 4. Step 3 — the terminal prediction ⟵ **SEAM #2 (the one that matters)**

**Paper:** `x̄_N = M_{t_{i+1}}(x̄_{i+1})`, which under the CFM scheduler collapses to
`x̄_N = x̄_{i+1} + (1−t_{i+1})·v_{t_{i+1}}(x̄_{i+1})`.

```python
# FM  — flow_policy.py:1339-1340
v_next = flow_eval_np(x_next_ref, t_k + dt)
x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next

# iMF — imf_flow_policy.py:405-409
t_next     = t_k + dt
h_terminal = 1.0 - t_next
u_terminal = u_eval_np(x_next_ref, t_next, h_terminal)
x_terminal_predicted_ref = x_next_ref + h_terminal * u_terminal
```

Same shape, one argument different: FM asks for the velocity **at a point**, iMF asks for the average
velocity **over the whole remaining interval** `h = 1−τ'`.

### The math, stated precisely (this is where the naive story needs correcting)

Write `τ' = τ_{k+1}` and let `x₁` be the true endpoint of the PF-ODE trajectory through `(x̄, τ')`.

**iMF.** By definition of `u`,
```
x̄ + (1−τ')·u(x̄, τ', 1−τ')  =  x̄ + ∫_{τ'}^{1} v(z_s, s) ds  =  x₁
```
so the shot is **exact by construction**; the only error is the network's error on `u`. This object is
`F(z,τ)` in `THEORY_DeepMix_HF_iMF.md` §1 — the *flow-map endpoint*, which commits to one mode.

**FM — two different framings, and the sharper one wins.**

* *Quadrature framing* (used in the u_8.2 STUDY doc): treating `x̂₁ = x̄ + (1−τ')·v(x̄,τ')` as a
  left-endpoint quadrature of `∫v ds`, the error is `∫_{τ'}^1 [v(z_s,s) − v(x̄,τ')] ds = O((1−τ')²)` —
  worst at τ=0, vanishing as τ→1.
* *Posterior-mean framing* (Theorem 1, `THEORY…` §1, verified numerically to `2.2e-16`): for the
  linear interpolant, `v(z,τ) = E[x1−x0 ¦ z_τ=z]`, hence
  ```
  z + (1−τ)·v(z,τ) = τE[x1¦z] + (1−τ)E[x0¦z] + (1−τ)(E[x1¦z] − E[x0¦z]) = E[x1 ¦ z_τ = z] = PM(z,τ)
  ```
  **identically, not to first order.**

So the honest statement of the seam is **not** "approximate vs exact endpoint" — it is
**"exact estimate of `PM` (the mode average) vs. estimate of `F` (the mode-committed endpoint)"**.
`PM` and `F` differ maximally at small τ (measured `mean|PM−F|`: 1.077 → 0.024 across τ = 0.05 → 0.9)
and coincide as τ→1. On the avoiding task the obstacle sits *between* the two modes, so early-τ `PM`
is a phantom path straight through the pillar — that is Defect A, and the real reason
`hardflow_activation='late'` exists.

The Gen13 bet was: swapping `PM → F` removes the phantom, so fewer, better-targeted corrections are
needed. The bet is theoretically sound. §10 records what measurement said.

### Boundary behaviour (both sides, identical)

At the last step `k = N−1`: `τ' = 1`, so `1−τ' = 0`. FM multiplies `v_next` by 0; iMF queries
`u(·, 1, 0)` (which, by `h→0`, is just `v` at τ=1) and multiplies by 0. **Both waste exactly one
network evaluation on the final step** and both reduce to `x̂₁_ref = x̄_N`. This is what makes
Proposition 1 (the output satisfies `h(x)≤0`) survive the swap untouched — see §7.

---

## 5. Step 4 — the prox-NLP: **literally the same object**

```
X* = argmin_X  ½·ρ·τ'²·‖X − x̂₁_ref‖²      s.t.   X ∈ C
```
`ρ = hardflow_reg_scale` (default 1.0), `C` = obstacle keep-outs (`_apply_obstacle_constraints`,
`flow_policy.py:280-348`) ∩ optional fitted linear dynamics `A·s + B·a + c = s'`
(`:350-441`, gated by `cfg.dynamics_constraint`).

Built **once** in `hardflow_formulate` (`flow_policy.py:683-751`); the loop only updates parameters
and re-solves.

iMF does **not** reimplement any of it. `ImfFlowPolicy.hardflow_formulate` (`imf_flow_policy.py:240-265`)
temporarily renames `guidance_method` to `"hardflow_new"` so the base assertion (`:693-696`) passes,
calls `super()`, restores the name in a `finally`, and then re-declares the solver purely to silence
CasADi's timing table (`silence_casadi_timing`, `:63-80`; FM gets the same treatment via
`InstrumentedFlowPolicy.hardflow_formulate`, `:112-115`).

| | FM `flow_policy.py:1342-1358` | iMF `imf_flow_policy.py:411-435` |
|---|---|---|
| set `τ'` | `set_value(oc_t_param, t_k + dt)` | `set_value(oc_t_param, t_next)` |
| set reference | `set_value(oc_X_terminal_predicted_ref, …)` | identical |
| NLP warm start | `set_initial(oc_X_terminal_predicted, x_terminal_predicted_ref)` | identical |
| solve | `solve_limited()` (IPOPT) | identical |
| on `RuntimeError` | `print("Solver failed…")`, fall back to `debug.value` | same fallback **+ counters** `_nlp_solves` / `_nlp_failures` (`:419,427`, fix_4) and a `[ eval_imf ] WARNING` line |

Only difference: **failure bookkeeping**. The optimisation itself is byte-identical, which is exactly
why the comparison is clean — any measured difference is attributable to the field, not the solver.

---

## 6. Step 5 — the pull-back (identical, and provably so)

**Paper:**
`x_{i+1} = α_{t'}·x̂*_N + β_{t'}·W_{t'}(x̄_{i+1})`, which under CFM simplifies (see
`MAP_Algorithm1_to_AvoidingCode.md` §3, full expansion) to:

```python
# FM  — flow_policy.py:1360-1362        |  iMF — imf_flow_policy.py:437-439
x_next = x_next_ref + (t_k + dt) * (x_terminal_predicted - x_terminal_predicted_ref)
```

i.e. `x_{k+1} = x̄_{k+1} + τ'·(X* − x̂₁_ref)`. **Character-for-character the same line on both sides.**

*Why it carries over unchanged* (BLEND §2.2, "consistency lemma"): the gain `τ'` comes from
`∂z_τ/∂x₁ = τ` for the **linear interpolant**, which both backbones share — iMF's `u` is trained on
the same `z = τ·x1 + (1−τ)·x0` (`imf_matcher.py:81`) as FM's `v` (`flow_matcher.py:71`). The seam
changes *what x̂₁ is*, not *how a correction to x̂₁ maps back to z*.

*But note* (Defect B, `THEORY…` §2): `τ·I` is **not** the inverse Jacobian of the endpoint map. At
τ=0.1 it delivers ~11% of the requested correction. That defect is present on **both** sides — Gen13
implemented Level 1 (seam only, decision D8) and deliberately left the Newton pull-back
(`THEORY…` §3) unbuilt. So this is a shared handicap, not a differentiator.

Then, identically on both sides:

```python
x_next = np.array(x_next).flatten()
u_ctrl = (x_next - x_next_ref) / dt      # flow_policy.py:1368  |  imf_flow_policy.py:445
```
— the recovered control `u_i*` of the paper's Problem 3, stored only for its norm diagnostic. (FM
prints that norm on **every plan**, `flow_policy.py:1394-1399`; iMF gates it behind
`cfg.imf_verbose_control`, `:472-477`, after it produced thousands of noise lines in the n=200 run.)

`hardflow_activation` gating (`"all"` vs `"late"`, skip constraining for `k < N//2`) is duplicated
verbatim: `flow_policy.py:1327-1336` ↔ `imf_flow_policy.py:393-402`. When skipped, both do
`x_next = x_next_ref`.

---

## 7. Step 6 — output, chain assembly, and why Proposition 1 survives

Identical code (`flow_policy.py:1373-1392` ↔ `imf_flow_policy.py:449-468`): stack `X_optimized`,
tile `s0` back into the middle of every row, reshape to `(1, N+1, H, T)`, unnormalize.

**Proposition 1 in this code, for both backbones.** At `k = N−1`, `τ' = 1`:
`x̂₁_ref = x̄_N + 0·(field) = x̄_N`, so the pull-back becomes
`x_N = x̄_N + 1·(X* − x̄_N) = X*` — the NLP solution itself, which satisfies `h(·) ≤ 0` by
construction. The argument uses only `(1−τ')=0` and the gain `τ'=1`; it never touches which field
produced `x̂₁`. **The safety guarantee is therefore backbone-independent** — and indeed both columns
hit 100% safe at K=10 (§10).

---

## 8. Step 7 — the `x̂₁` diagnostic chain (bottom row of the Fig.11 grid)

Called once per plan, **inside the timed region** on both sides (`flow_policy.py:1424`,
`imf_flow_policy.py:502`).

```python
# FM  — flow_policy.py:227-245
t = to_torch(k / (n_steps - 1))
current_v = self.flow_model(current_x, t)               # NOT under no_grad
predicted_x = current_x + (1.0 - t) * current_v

# iMF — imf_flow_policy.py:176-197
t = k / (n_steps - 1)
with torch.no_grad():
    current_u = self._u(current_x, t, 1.0 - t, bucket="diag")
x1_estimation[:, k] = current_x + (1.0 - t) * current_u
```

Math: `x̂₁(k) = x_k + (1−τ_k)·v(x_k, τ_k)` vs `x_k + (1−τ_k)·u(x_k, τ_k, 1−τ_k)` — the same seam,
re-evaluated at every stored chain state rather than only at `x̄_{k+1}`.

⚠️ **Second `no_grad` asymmetry**, and this one *is* inside the timed region: the base FM
`x1_estimate` builds an autograd graph for `N+1` forward passes on every plan; iMF does not. It
biases `computation_time` **in iMF's favour**, and iMF still lost on wall-clock (§10) — so the
conclusion is safe, but any future timing claim should fix this first.

---

## 9. The rest of the pipeline (context for reading the numbers)

### 9.1 Training

| | FM `flow_matcher.py:39-56` | iMF `imf/imf_matcher.py:68-119` |
|---|---|---|
| sample | `t ~ U(0,1)`, `xt = t·x1 + (1−t)·x0` (`:90-97, 69-71`) | `(τ, h)` from paired logit-normals, `τ = min`, `h = s − τ`, with `data_proportion=0.25` forced to `h=0` (`convention.py:65-87`) |
| target | `ut = x1 − x0` (`:83-85`) | same `v_target = x1 − x0` (`:75`) |
| loss | `mse(model(xt,t), ut)` — one head, one time | `V = u − h·sg(D_tot)`; `adp(‖V−v_target‖²) + adp(‖v−v_target‖²)` where `D_tot` is a `torch.func.jvp` with tangents `(v_c, +1, −1)` (`:95-109`) |
| identity | — | `u = v + h·D_tot` — derivation and sign audit live **only** in `convention.py:22-36`; the sign was fixed empirically by gate G1 |
| judge convergence on | `loss` | **`raw_mse_u` / `raw_mse_v` / `a0_mse`** — the adaptive `loss` is flat by construction (`imf_matcher.py:21-23`) |

The iMF net must fit a **two-time** object `u(z,τ,h)` from the same 96 demonstrations that determine
FM's one-time `v(z,τ)`. That data-hunger was the pre-registered Gen13 risk, and it is what actually
decided the experiment.

### 9.2 Unguided sampling (the "no NLP" control)

| FM | iMF |
|---|---|
| `ConditionedODESolver`, `t_span = linspace(0,1,N+1)[:-1]`, Euler (`flow_policy.py:173-197`) | `imf_sample` (`imf/imf_sampler.py:37-43`): `τ = i/K`; `x += dt·u(x, τ, dt)` |
| `N` Euler steps, error `O(1/N)` globally | `K` exact jumps, error = field error only |

Same NFE count for the same step count; the iMF claim was that K could be *much* smaller.

### 9.3 NFE accounting (per plan, `hardflow_activation="all"`)

Tracked explicitly on the iMF side only (`_nfe` buckets, `imf_flow_policy.py:143-159`), but the
structure is identical:

| bucket | FM | iMF |
|---|---|---|
| warmstart | `N` | `N` |
| sampling (reference + terminal) | `2N` | `2N` |
| diagnostic `x̂₁` | `N+1` | `N+1` |
| **total** | **`4N+1`** | **`4N+1`** |

**Identical at matched `N`.** So iMF can only win by needing a smaller `N`, or by being cheaper per
call. It is neither: the dual-head two-time UNet is *more* expensive per evaluation.

### 9.4 Where you see it — the figures

Both are pure post-processing of `{run_id}_fan.npz` (dumped by `run/eval_imf.py` when
`cfg.imf_plot_fan`; no GPU, no simulator):

* `run/make_fig11_comparison.py` — one representative planning instance per panel, iMF vs FM, in
  upstream's magenta `style="predicted"` (`:61`) + executed rollout in `"actual"` (`:70`). Why this
  style belongs here and not on the foresight fan: `../Research/MEMO_hardflow_fig11_predicted_style.md`.
* `run/make_fig11_ode_grid.py` — 2×N grid, top row `x_τ`, bottom row `x̂₁`, across ODE steps; `--both`
  stacks the two backbones into 4 rows. The **bottom-left cells are the discriminator**: that is
  where `(1−τ)` is largest and `PM` and `F` are furthest apart.
* Index gotcha, both scripts: `plot_single_trajectory` reads x,y from **columns 2,3** (observation
  layout) while chain states are full transitions ⇒ the `traj[:, action_dim:]` slice.
* `run/analyze_x1_accuracy.py` — the quantitative version of the bottom row:
  `err(k) = mean_H ‖x̂₁(k) − x_final‖`. Only `err(τ=0)` carries information; `err(τ=1) = 0`
  identically for both by construction, so the printed `decay` column is meaningless.

---

## 10. What the H2H actually measured — and the verdict

Full record: `../../Gen13/fix_7/RESULTS_Gen13_fix7.3_VERDICT_imf_refuted.md` (job 23612, interpretation
pre-committed before the data existed).

**Matched budget (equal K ⇒ equal NFE ⇒ equal projections), n=20/cell:**

| K | FM safe | FM s/plan | iMF safe | iMF s/plan |
|---|---|---|---|---|
| 1 | **95%** | **0.1119** | 75% | 0.1357 |
| 2 | **100%** | **0.1894** | 85% | 0.2434 |
| 5 | **100%** | **0.4331** | 95% | 0.4923 |
| 10 | 100% | **0.8456** | 100% | 0.9224 |

**The seam itself, measured** — terminal-prediction error at τ=0, exactly where the swap was supposed
to pay:

| K | iMF err(τ=0) | FM err(τ=0) | |
|---|---|---|---|
| 1 | 0.1539 | 0.0260 | iMF 5.9× worse |
| 2 | 0.1538 | 0.0303 | iMF 5.1× worse |
| 5 | 0.1595 | 0.0356 | iMF 4.5× worse |
| 10 | 0.1572 | 0.0384 | iMF 4.1× worse |

iMF's error is **flat in K** (0.1539 → 0.1572) — the signature of a fixed **field/training** error,
not a discretisation error. The `u`-field's error (~0.155) dwarfs the Euler/`PM` gap it was meant to
eliminate (~0.026–0.038). Corroborating: `raw_mse_u` plateaued ≈13 (≈0.37/dim, vs Gen3v4's 0.25/dim)
and `a0_mse` never reached the <0.15 reference.

> **Verdict.** Every line of the swap is correct and verified. The **mechanism** is refuted *in
> practice at this data scale*: the exact endpoint map is only as good as the field that computes it,
> and 96 demonstrations do not determine a two-time `u(z,τ,h)` well enough to beat a one-time `v(z,τ)`.
> The earlier "1.74× speedup" was an artefact of pairing iMF@K=5 against FM@K=10.
>
> **The genuine finding is about HardFlow, not iMF:** FM runs at **K=2 with 100% safety** — a 4.5×
> speedup over its own default K=10. The default is over-provisioned.

Rescuing iMF would require fixing the *field* (much more data, or a far better schedule) and would
then have to beat FM@K=2 (100% @ 0.19 s/plan), not FM@K=10.

---

## 11. One-screen summary — the complete diff

Everything Gen13 changes inside the guided sampler, in full:

```python
#            ORIGINAL HardFlow (FM)                    Gen13 (iMF)
# ---------------------------------------------------------------------------------
# reference step   flow_policy.py:1324-1325            imf_flow_policy.py:390-391
  v_k = flow_eval_np(x_k, t_k)                  →      u_k = u_eval_np(x_k, t_k, dt)
  x_next_ref = x_k + v_k * dt                          x_next_ref = x_k + u_k * dt

# terminal shot    flow_policy.py:1339-1340            imf_flow_policy.py:408-409
  v_next = flow_eval_np(x_next_ref, t_k+dt)     →      u_terminal = u_eval_np(x_next_ref, t_next, 1-t_next)
  x̂1_ref = x_next_ref + (1-t_k-dt) * v_next            x̂1_ref = x_next_ref + (1-t_next) * u_terminal

# everything else: IDENTICAL
#   NLP build/solve · pull-back x + τ'(X*-x̂1) · control recovery · chain assembly
#   · unnormalize · conditioning masks · activation gating · Proposition 1
```

> **Read this diff correctly — it is NOT a rename, and NOT a repackaging.**
> `t_next = t_k+dt` and `h_terminal = 1-t_next` *are* pure renames (hoisted only because iMF reuses
> `t_next` at `:411` and `:437`, where FM re-types `t_k+dt` inline). The **surrounding algebra is
> identical on both sides**: `x̂1 = x̄ + (1−τ')·FIELD(x̄, τ')`. The entire change is *which function is
> evaluated* — a **third argument** and a **different head**:
> `flow_eval_np → flow_model(x,t) -> v` (`:1310-1314`) vs `u_eval_np → flow_model(x,τ,h) -> (u,v)`, keeps `u` (`:374-378, 155-159`).
> Note `(1−τ')` is passed **twice** — as the multiplier *and* as the interval width `h`. That coupling
> is the whole mechanism: with `h` absent the product is a linear extrapolation (= `PM` exactly, Thm 1);
> with `h = 1−τ'` the same product becomes the exact integral `∫_{τ'}^1 v ds`. Set `h → 0` and iMF's
> line degenerates to FM's line exactly, since `u(z,τ,0) = v(z,τ)`.

Plus, outside the loop: the network (`+h` embedding, dual head), its training objective (JVP
identity), the `x̂₁` diagnostic (same seam), and NFE/NLP counters. **Nothing in `flow_policy.py` was
edited** — the Gen13 additive rule held.

### Invariants to preserve if this is ever touched again

1. **Convention lives in one file.** All sign/direction logic is in `imf/convention.py`. Never
   introduce a flip at a call site.
2. **The NLP must stay shared.** iMF reuses the base `hardflow_formulate` through a name shim
   (`imf_flow_policy.py:247-256`). Duplicating it would destroy the A/B's cleanliness.
3. **Match K before comparing anything.** The single hardcoded `k_steps=10` in a diagnostic script is
   what hid the refutation through four rounds of analysis.
4. **Judge iMF training on `raw_mse_u` / `a0_mse`, never on `loss`** (adaptive ⇒ flat by design).
5. **Fix the two `no_grad` asymmetries** (`flow_policy.py:769`, `:237`) before making any new
   wall-clock claim — both currently favour iMF.
