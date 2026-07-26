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

## 4. The "is it the same NLP?" question (asked during DA work)

**No — different solvers, comparable outcomes.**
- **DPCC (arm B):** *post-hoc* — sample the full trajectory, then one SLSQP projection snaps it onto
  the feasible set. No network in the solve.
- **HardFlow (arm C):** *in-loop* — at each active ODE step, a small prox-NLP nudges the *predicted
  terminal* `x̂₁` toward feasibility, blended back by τ.

At threshold 0.5 both act over the *last half* of the ODE (same schedule region, DPCC polarity after
fix_6), but they are **not the same optimisation**. So the arms are a fair *outcome* comparison
(success / safety / time, all now loadable in the DA after DA-U7), not a "same-NLP" identity.

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
