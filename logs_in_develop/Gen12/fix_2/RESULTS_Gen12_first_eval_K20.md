# Gen12 — first end-to-end eval: arms A/B/C on FMv3ODE (K=20, smoke)

**Date:** 2026-07-24 · **Type:** results / insight · **Status:** preliminary (underpowered smoke)
**Run:** jobs 23796 (gates) → 23797 (eval) → 23798 (aggregate), git `46fa799`, node i6-gpu-1
**Logs:** `temp/Gen12/14_43_11_hffm_{gates,eval,load}_*.log`
**Model:** FMv3ODE checkpoint `flow_matching_v3_ode_selectable/H8_…FlowMatchingODE_a1.5_b1.0_aw10`, seed 6, step 98000
**Results on cluster:** `logs/avoiding-d3il/plans/flow_matching_v3_hardflow/H8_K10_…FlowMatchingODE/6/results/halfspace_<variant>/K20_n2/`

> ⚠️ **Read the caveats (§4) before quoting any number.** This is a **6-episode smoke run at a
> single K=20** — it is a plumbing check, not an experiment. The K is also **20, not the intended
> 10** (a cluster-side config hand-edit; see fix_2 debug notes). Treat this as "the pipeline works
> and here is the shape of the answer", not as a result.

---

## 1. The three arms (PLAN §5)

| arm | method | what it tests |
|---|---|---|
| **A** `diffuser` | unguided FMv3ODE ODE | field-quality floor |
| **B** `dpcc-c-tightened` | DPCC `Projector`, post-hoc per step | the incumbent |
| **C** `hardflow_new` | in-loop constrained sampling (the port) | the contribution |

All three ran at the **same K=20** (matched-budget, PLAN §5 — this part was correct despite the K
being 20 not 10). Aggregated over 3 halfspace variants × n_trials=2 = **6 episodes per arm**, seed 6.

## 2. Aggregate results (from job 23798)

| metric | A `diffuser` | B `dpcc-c-tightened` | C `hardflow_new` |
|---|---|---|---|
| success (goal) | **1.00** | **1.00** | **1.00** |
| success (goal + constraints) | **0.00** | **1.00** | **1.00** |
| constraints satisfied | **0.00** | **1.00** | **1.00** |
| avg violations | 18.5 ± 10.4 | **0.0** | **0.0** |
| avg total violation | 4.855 ± 4.592 | **0.000** | **0.000** |
| avg steps to goal | 65.0 ± 8.0 | 62.2 ± 5.3 | 69.0 ± 10.8 |
| **s / plan** | 0.173 | **0.474** | 0.746 |
| **success·s⁻¹** (goal+constr / s·plan) | 0.00 | **2.11** | 1.34 |
| NFE (sum) | 31 680 | 30 320 | 16 800 |
| NLP solves (sum) | 0 | 0 | **8 400** |
| NLP failures | 0 | 0 | **0** |
| batch size | 4 | 4 | 1 (faithful, §3.4) |

Per-halfspace breakdown (eval log 23797): the pattern is identical in all three variants —
A always violates (top-right total-viol 11.07, both-hard 3.35, top-left 0.14), B and C always
0 violations.

## 3. What the numbers say

**3.1 Guidance is doing all the safety work.** The unguided field (A) reaches the goal **every
time** (1.00) but is safe **never** (0.00 collision-free) — it drives straight through obstacles
(18.5 violations/episode). Both B and C take that same field and make it 100% safe. So the FMv3ODE
model is "goal-capable but unsafe on its own", and the constraint machinery — not the field — is
what produces safe trajectories. This echoes the Gen13 finding that *the projection dominates
outcomes regardless of the field* (PLAN §5.1).

**3.2 On the headline question — does in-loop (C) beat post-hoc (B) at equal compute? — no.**

- **Outcome: a tie.** At K=20 both B and C are saturated: 100% goal, 100% constraints, 0
  violations. On the metric the plan says to rank by (task success, §5.1), they are
  indistinguishable here.
- **Compute: B wins.** B is ~1.6× faster (0.474 vs 0.746 s/plan) and therefore better on
  success-per-second (**2.11 vs 1.34**). C pays for its guarantee with ~1 400 NLP solves per
  episode (8 400 total, all converged, 0 failures) run *inside* the sampling loop, versus B's
  cheaper post-hoc SLSQP projection.

This is a **clean negative** for the "constrained-sampling beats post-hoc-projection" hypothesis —
and PLAN §6 pre-registered that a clean negative is itself a valid result, consistent with Gen13:
the projection is what matters; enforcing constraints *during* sampling matches it at higher cost.

**3.3 The port is correct (PLAN §6 minimum criterion — MET).** Arm C runs end-to-end, is at least
as safe as arm B (both 0 violations), the NLP never fails, and NFE/NLP-solve accounting is exactly
2K per step / K solves per plan as designed. The gates (G0–G3) passed. So Gen12's contribution
*works*; it just doesn't *win* in this regime.

## 4. Why this is not yet a result — caveats

1. **n = 6.** Two trials × three halfspace variants × one seed. A "1.00" here is 6/6 — the
   confidence interval is enormous. PLAN §5 calls for **n ≥ 100** and multiple seeds. Nothing here
   is statistically separable.
2. **Single K = 20, and the saturated regime.** At K=20 the field is good enough that *both* arms
   are perfect, so success **cannot** discriminate them — the run only exposes the cost gap. The
   scientifically interesting regime is **low K** (K ∈ {2, 5, 10}), where field quality is poor and
   in-loop guidance *might* rescue trajectories that post-hoc projection cannot. **That regime is
   completely untested here.** (And the K should have been 10 — the 20 came from a cluster config
   hand-edit overriding the intended value.)
3. **batch asymmetry.** B fans 4 candidates and selects; C is faithful batch-1 (PLAN §3.4). So the
   s/plan comparison is deployment-honest but not a like-for-like per-trajectory cost. The batch-4
   arm-C counter-run (§3.4) has not been done.
4. **One checkpoint, one model class.** FMv3ODE only, as designed (fix_1 §8). No claim generalises
   beyond this checkpoint.

## 5. Verdict and next steps

**Verdict:** pipeline validated; preliminary shape is a *tie on safety, loss on speed* for arm C —
a clean negative, consistent with Gen13, but on far too little data to assert.

**To turn this into a result:**
1. Re-run at the **intended K, and as a sweep**: K ∈ {2, 5, 10, (20)} — the low-K regime is the
   whole point (PLAN §5). Fix the K back to config-driven 10 first (verify no cluster hand-edit).
2. **n ≥ 100**, multiple seeds. Raise `n_trials` and add seeds in the hardflow eval yaml.
3. Report success **and** s/plan **and** NFE/NLP-solves per arm at each K (the table above is the
   template).
4. Optional: the **batch-4 arm-C** counter-run to check whether B's candidate fan explains any of
   its edge (§3.4).
5. Rank arms by **task success**, not smoothness (PLAN §5.1); record roughness only as a descriptor.

**Pre-registered kill/interest criterion:** if at low K arm C's success stays at/above arm B while
arm B degrades (post-hoc projection can't fix a bad field), that would be the first real evidence
*for* the contribution. If C only ever ties B at high K and costs more, Gen12 confirms the Gen13
"projection dominates" story and should be written up as such.
