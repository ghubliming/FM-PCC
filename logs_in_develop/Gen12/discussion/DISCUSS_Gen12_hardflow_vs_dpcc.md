# Gen12 — DISCUSSION: does in-loop constrained sampling (HardFlow) beat post-hoc projection (DPCC)?

**Date:** 2026-07-26 · **Type:** discussion / synthesis (not a changelog)
**Scope:** where Gen12 stands after init → fix_1..fix_6, U4, U4.2, and the Gen13 U10 cross-check.
**One-line state:** the port is correct and cheap; the *scientific* verdict is still open, but the
evidence so far says **arm C ties DPCC at best and loses at low K** — a (soft) negative, consistent
with Gen13's "the projection dominates."

---

## 1. What Gen12 set out to answer

Port HardFlow's `hardflow_new` (in-loop constrained sampling) onto FMPCC's own FMv3ODE model and ask
PLAN §5's question:

> **Does enforcing constraints *during* sampling (arm C, HardFlow) beat projecting *after* sampling
> (arm B, DPCC) — at equal compute, on FMPCC's own model?**

Three arms, always matched-K:
- **A** `diffuser` — unguided FMv3ODE (field-quality floor)
- **B** `dpcc-c-tightened` — DPCC post-hoc `Projector` (the incumbent)
- **C** `hardflow_new` — per-step prox-NLP on the predicted terminal (the contribution)

## 2. The road so far (why each fix happened)

| step | what | why it mattered |
|---|---|---|
| init | port hardflow_new → FMv3ODE, gates G0–G3 | premise: HardFlow's contribution is eval-time only, reuse the checkpoint |
| fix_1 | eval single-sourced on the Gen12 yaml | stop `projection_eval.yaml` leaking into Gen12 |
| fix_2 | G2 gate fixed (τ-invariance was bitwise) | false-negative gate, not a sampler bug |
| fix_3 | **K-sweep results** | the first real finding — see §3 |
| fix_5 | FMv3ODE path layout (`<train>/<eval-name>/…`) | make results discoverable by the DA |
| fix_6 | **threshold polarity flipped to DPCC** | our threshold was inverted vs DPCC — now identical |
| U4 | late-activation **threshold** | skip early-step NLPs (paper-endorsed) |
| U4.2 | **MPC candidate fan** + DPCC-style `-c/-r/-t` selection | close the batch confound (B fanned 4, C ran 1) |
| Gen13 U10 | same threshold on the **original** HardFlow (n=50) | independent validation on the paper's own code/env |

## 3. What the data actually says

### 3.1 Low K — arm C collapses, DPCC is rock-solid (fix_3, the important one)
K-sweep {2,5,10}, n=6, seed 6:

| K | A goal+constr | **B (DPCC)** | **C (hardflow)** |
|---|---|---|---|
| 2 | 0.17 | **1.00** | 0.33 |
| 5 | 0.00 | **1.00** | 0.50 |
| 10 | 0.17 | **1.00** | 1.00 |

DPCC is 100% from K=2 up; arm C **fails at low K** and only catches up at K≥10. This *refuted* the
pre-registered hope that in-loop guidance rescues coarse-field (low-K) trajectories. Mechanism: the
terminal prediction `x̂₁ = x_ref + (1−τ)v` is garbage at low K, and projecting onto garbage yields a
feasible-but-wrong plan.

### 3.2 High K (K=20, saturated) — everyone is safe; U4 makes arm C cheap
At K=20 all guided arms are 100% safe, so the axis is compute. **U4 threshold 0.5** cut arm C's NLP
solves ~46% and time ~35% with zero safety cost — bringing arm C to **cost-parity with DPCC**
(0.488 vs 0.477 s/step). So U4 removes arm C's cost disadvantage; it does not give it an advantage.

### 3.3 Gen13 U10 (original HardFlow, n=50) — threshold 0.5 is a free lunch, terminal-only isn't
On HardFlow's own algorithm/env: threshold 0.5 = identical safety+quality at −35% compute; pushing
to terminal-only (0.0 DPCC) degrades safety (1.00→0.98) and path length. Two independent codebases
agree: **threshold 0.5 is a safe ~⅓ compute saving.**

## 4. Are the two NLPs the same? — the actual math + code

**No. Same feasible set `S`, different optimisation problem, different decision variable, different
solver, different call site.** Here are both, verbatim from the code.

### 4.1 DPCC (arm B) — post-hoc projection of the SAMPLED trajectory
Code: `flow_matcher_v3/sampling/projection.py::Projector.project` (L70–155); called *after* the ODE
step, only near the end, in `flow_matcher_v3/models/diffusion.py::p_sample_loop` L178/L193:
```
near_end = loop_idx >= (1 - T)·K ;  if near_end:  x, cost = projector.project(x, ...)
```
Decision variable = the **full sampled trajectory** `z = x ∈ ℝ^{H·T}` (all H steps stacked).
The solve (SLSQP, L133–142):
```
min_z   ½ zᵀQz + rᵀz          with r = −z_rawᵀQ   ⟺   min_z ½‖z − z_raw‖²_Q
s.t.    A z = b               (Euler-kinematics equalities + fixed x₀)
        C z ≤ d               (halfspaces, velocity bounds)
        zₜᵀP zₜ + qᵀzₜ ≤ v    (sphere/obstacle, per step t=1..H−1)
```
i.e. **Euclidean(-Q) projection of the already-generated sample onto `S`**, solved **once**, on the
real trajectory. The network does not appear in the solve.

### 4.2 HardFlow (arm C) — in-loop prox-NLP on the PREDICTED TERMINAL
Code: `flow_matcher_v3_hardflow/sampling/hardflow_projection.py`. Per active ODE step k
(`HardFlowSampler.sample` L444–448):
```
x_ref   = x_k + v(x_k,τ_k)·dt
x̂₁_ref = x_ref + (1 − τ_{k+1})·v(x_ref, τ_{k+1})     # 1-step terminal extrapolation
x̂₁*    = HardFlowNLP.solve(x̂₁_ref, τ_{k+1})
x_{k+1} = x_ref + τ_{k+1}·(x̂₁* − x̂₁_ref)             # pull-back, blended by τ
```
The NLP (`HardFlowNLP`, L155–174) has decision variable = the **predicted terminal** `x̂₁ ∈ ℝ^{dof}`
(dof = H·T − state_dim, s₀ pinned), objective (L155–157):
```
min_{x̂₁}   ½ · reg · τ² · ‖x̂₁ − x̂₁_ref‖²        s.t.   h(x̂₁) ≤ 0   (same S as §4.1)
```
solved with **CasADi/IPOPT** (L174, `solve_limited` L309), **once per active step** (up to K times a
plan).

### 4.3 Why they are not the same (side-by-side)

| | DPCC (arm B) | HardFlow (arm C) |
|---|---|---|
| decision variable | sampled trajectory `z = x` | predicted terminal `x̂₁ = x_ref + (1−τ)v` |
| objective | ½‖z − z_raw‖²_Q (Q-weighted) | ½·reg·τ²·‖x̂₁ − x̂₁_ref‖² |
| when | **once**, post-hoc, near end | **per active ODE step**, in-loop |
| operates on | the real iterate | a 1-step *extrapolation* of the terminal |
| result mapped back? | no (z is the sample) | yes: `x_{k+1}=x_ref+τ(x̂₁*−x̂₁_ref)` |
| solver | scipy SLSQP, dim H·T | CasADi/IPOPT, dim H·T−state_dim |
| feasible set `S` | **identical** (Gen12 builds arm C's `h` from the same yaml geometry) | **identical** |

**Only `S` is shared.** So at threshold 0.5 both *act over the last half of the ODE and enforce the
same constraints*, but they solve **different programs on different variables** → not the same NLP.
A `hardflow_new.npz` and a `dpcc-c-tightened.npz` are a fair **outcome** comparison (success / safety
/ time), never a solver identity. The per-row `activation_threshold` / `nlp_solves` (loaded after
DA-U7) make the distinction explicit.

## 4.4 What does "HF beats DPCC" MEAN? — the interpretation table

Because §4.3 shows **both enforce the identical feasible set `S`**, safety alone often can't separate
them (both hit 100% once K is large enough). So read the two axes the DA reports —
**success** `s = n_success_and_constraints` (primary) and **avg_time** `t` (secondary) — with this logic:

Let `s_H, t_H` = hardflow, `s_D, t_D` = DPCC, at **matched K, matched n, same seeds, same S**.

| observation | conclusion | what it means scientifically |
|---|---|---|
| `s_H > s_D` | **HF wins on quality** ⭐ | in-loop steering reaches feasible-AND-good solutions that post-hoc projection cannot — the projection lands the raw sample in a bad feasible basin, in-loop avoids it. **This is the only outcome that justifies the contribution.** |
| `s_H ≈ s_D` and `t_H < t_D` | HF wins on cost only | same quality, cheaper. Useful but weak — it's an efficiency claim, not "constrained sampling is better." Depends on solver tuning, not method. |
| `s_H ≈ s_D` and `t_H ≈ t_D` | tie | no reason to prefer HF over the simpler incumbent (Occam → keep DPCC). |
| `s_H ≈ s_D` and `t_H > t_D` | HF loses on cost | DPCC dominates (this was fix_3/U4 at high K before U4; U4 made it a tie). |
| `s_H < s_D` | **HF loses** | post-hoc projection is strictly better; in-loop buys nothing. Any time advantage is irrelevant — you don't trade safety/success for speed. |

**The decisive caveat (why "faster" is the weak claim):** at saturation (high K) both reach `s≈1.00`,
so a time win is the *only* thing left — but a time win at equal quality is a **solver/efficiency**
result, not evidence that in-loop constrained sampling is *better than* post-hoc projection. The
strong, publishable claim requires **`s_H > s_D` in a regime where DPCC genuinely fails** — i.e. where
projecting the finished sample onto `S` cannot recover a good trajectory but in-loop steering can.
fix_3 tested exactly that regime (low K) and found the **opposite** (`s_H < s_D`). So:

- **"HF avg-time beats DPCC" (with `s_H ≈ s_D`)** → *nice-to-have efficiency*, not a scientific win.
  Report as "matches DPCC safety at lower/again-equal cost." Do **not** claim in-loop superiority.
- **"HF success beats DPCC" (`s_H > s_D`, especially at low K / hard constraints)** → *the real result*:
  in-loop constrained sampling is genuinely better because it shapes the trajectory before commitment,
  not after. This is what to hunt for, and what the U4∩U4.2 low-K run (§7) is designed to expose.
- **Same-safety, same-time** → the honest headline is "the projection dominates the outcome regardless
  of when it is applied" (the Gen13 finding), and HardFlow is a more complex equal.

Concretely for the pair you're inspecting (`hardflow_new` @0.5 vs `dpcc-c-tightened`, both in the
`K20_thres0.5_mpc1_n2` folder): both are 100% safe at K=20, so this pair can **only** show a time
difference → it is the *weak* (efficiency) axis. The *quality* axis needs the **low-K** run.

## 5. Where this leaves the central question

So far, across every regime tested:
- **low K:** DPCC ≫ hardflow (hardflow fails);
- **high K:** DPCC ≈ hardflow (tie on safety; U4 ties them on cost too);
- **never:** hardflow strictly beats DPCC.

That is a **clean negative trending**, and it echoes the Gen13 finding that *the projection dominates
outcomes regardless of the field*. If it holds up, Gen12's contribution (`hardflow_new` on FMPCC) is
a correct, now-cheap method that **buys nothing over DPCC** — a legitimate, publishable negative.

**BUT one thing could still overturn it** — the confound U4.2 exists to test:

> At **low K**, arm C ran **batch 1** while DPCC ran **batch 4 + candidate selection**. Does a
> batch-4, min-cost-selected arm C (`hardflow_new-c`, mpc4, thres0.5) *recover* the low-K failure?
> If yes, fix_3's negative was partly the missing candidate fan, and the story changes. If no, the
> negative is real and controlled.

This single experiment (U4 ∩ U4.2 at low K) is the crux left to run.

## 6. Decisions / positions to take

1. **Adopt threshold 0.5 as the default** for HardFlow-style sampling. Validated twice (Gen12 K=20,
   Gen13 n=50): free ~⅓ compute, zero safety cost. Low risk.
2. **Treat Gen12 as heading toward a controlled negative** — unless the low-K batch-4 arm-C run
   (§5) flips it. Frame the write-up accordingly, and don't over-claim arm C parity as a "win."
3. **Power it before concluding:** everything so far is seed 6, n≤6 (Gen12) — only Gen13 U10 is
   n=50. A real verdict needs n≥100 and multiple seeds (needs seeds 7–10 trained, or more trials on
   seed 6).
4. **Keep DPCC as the incumbent.** Nothing here suggests replacing it; hardflow is at best an equal
   at higher cost/complexity.

## 7. The one experiment that decides Gen12

```
config/hardflow_projection_eval.yaml:
  projection_variants: [dpcc-c-tightened, hardflow_new-c]   # B vs arm-C-with-fan
  hardflow: { batch_size: 4, activation_threshold: 0.5, candidate_cost: prox }
sweep K ∈ {2, 5, 10}, n_trials high (→ n≥100 on seed 6)
```
- If `hardflow_new-c @ mpc4, thres0.5` is **100% safe at K=2/5** → the fix_3 collapse was the fan;
  arm C is competitive → Gen12 is a *positive* (in-loop matches DPCC once fairly resourced).
- If it **still fails at low K** → the negative is real and fully controlled → Gen12 concludes that
  post-hoc projection dominates in-loop constrained sampling on FMPCC.

Either outcome is a clean, reportable result. This is the next run.

---

### Pointers
- Results: [`../fix_3/RESULTS_Gen12_Ksweep_lowK.md`](../fix_3/RESULTS_Gen12_Ksweep_lowK.md),
  [`../U4/RESULTS_Gen12_U4_threshold_K20.md`](../U4/RESULTS_Gen12_U4_threshold_K20.md),
  [`../../Gen13/U_10/RESULTS_Gen13_U10_threshold_sweep.md`](../../Gen13/U_10/RESULTS_Gen13_U10_threshold_sweep.md)
- Plans: [`../U4/PLAN_Gen12_U4_late_activation_threshold.md`](../U4/PLAN_Gen12_U4_late_activation_threshold.md),
  [`../U4/PLAN_Gen12_U4.2_mpc_candidate_selection.md`](../U4/PLAN_Gen12_U4.2_mpc_candidate_selection.md)
- Polarity fix: [`../fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md`](../fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md)
- DA support: [`../../DA_Code/v3/U7/CHANGELOG_U7_hardflow_variants.md`](../../DA_Code/v3/U7/CHANGELOG_U7_hardflow_variants.md)
