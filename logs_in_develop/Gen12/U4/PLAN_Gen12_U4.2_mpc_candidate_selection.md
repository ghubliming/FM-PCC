# Gen12 U4.2 — PLAN: MPC candidate fan + DPCC-style selection for `hardflow_new`

**Date:** 2026-07-25 · **Status:** plan only, **no code written**
**Sibling of:** [`PLAN_Gen12_U4_late_activation_threshold.md`](PLAN_Gen12_U4_late_activation_threshold.md) (orthogonal; combinable)
**Motivation:** close the fix_3 confound — arm B (DPCC) runs `batch_size=4` + candidate selection,
arm C (`hardflow_new`) runs faithful `batch_size=1` with no selection. Give arm C the **same MPC
machinery** (a 4-candidate fan and the `-c/-r/-t` selection rules) so the comparison isolates
*method* (in-loop vs post-hoc) from *candidate fan*.
**Verdict: ✅ possible, and the fan already works — only the selection layer is missing.**

---

## 1. The question

Does `hardflow_new` have DPCC's MPC candidate mechanism (`batch_size=4` + `dpcc-c/-r/-t`
selection)? Currently **no**: it runs one candidate and takes it. fix_3 §5 flagged this as the
**primary threat** to the fix_3 conclusion — B's low-K robustness might come from its 4-candidate
fan + selection, not from being post-hoc. U4.2 removes that confound and, as a bonus, gives arm C
the full DPCC MPC toolkit for a fair head-to-head.

## 2. What already exists vs. what to build

**Already works (verified in `sampling/hardflow_projection.py::HardFlowSampler.sample`):**
```python
for b in range(batch_size):        # <- the candidate fan already loops here
    ... independent NLP-guided ODE chain per candidate ...
```
So `hardflow.batch_size: 4` already produces 4 independent NLP-steered candidates. No structural
change needed for the fan. (Gen12 already generalised past upstream's `batch_size==1` hard-assert.)

**Missing — two things:**
1. **Selection is hardcoded** in `HardFlowPolicy.__call__`: `which_trajectory = 0`. No `-c/-r/-t`.
2. **No per-candidate cost** is collected — `infos['projection_costs'] = {}`, and `nlp.solve()`
   returns only the solution, not its objective. So DPCC-c ("minimum cost") has nothing to rank by.

## 3. DPCC's three selection rules and their `hardflow_new` analogs

DPCC's `Policy` (`sampling/policies.py`) selects among the `batch_size` projected candidates by
`trajectory_selection`:

| DPCC variant | rule | `hardflow_new` analog | portable? |
|---|---|---|---|
| `dpcc-r` | **random** (`which=0`) | index 0 (today's behaviour) | ✅ trivial |
| `dpcc-t` | **temporal consistency**: candidate whose obs are closest to the *previous* plan (`argsort ‖obs[:,:-1] − prev_obs[:,1:]‖`) | **identical** — operates on observations + `prev_observations`, model-agnostic | ✅ reuse DPCC code verbatim |
| `dpcc-c` | **minimum projection cost**: candidate whose SLSQP projection moved it least (raw sample already closest to feasible ⇒ higher quality) | **minimum NLP intervention**: candidate with the smallest accumulated prox deviation `Σ_k ‖x1_proj − x1_ref‖²` (equivalently control effort `Σ ‖u_k‖`) | ✅ principled analog (see §3.1) |

### 3.1 Why "minimum NLP intervention" is the right analog of DPCC-c

DPCC-c prefers the candidate the projection had to move the least — i.e. whose *unguided* sample was
already nearest the feasible set, hence most faithful to the field. The HardFlow paper's
**minimal-intervention principle** (U4 §2.3) is the same idea: the best sample is the one requiring
the least control. So for `hardflow_new` the natural per-candidate cost is the **total intervention**
the NLP applied — `Σ_k ‖x1_proj,k − x1_ref,k‖²` over the active steps (or the control-effort norm
`Σ_k ‖u_k‖`, `u_k = (x_next − x_ref)/dt`, which upstream already prints). Smallest = least
intervention = closest to the nominal field = highest expected quality. This is the direct,
theory-consistent counterpart of DPCC-c.

## 4. The upgrade

### 4.1 Sampler — return a per-candidate cost

In `HardFlowSampler.sample`, accumulate per candidate `b`:
- `candidate_cost[b] = Σ_k ‖x1_proj − x1_ref‖²` over active steps (prox deviation), **and/or**
- `control_effort[b] = Σ_k ‖u_k‖` (already computable from the pull-back).

Return `infos['candidate_costs']` (shape `(batch,)`) and keep the per-candidate observations (already
in `out`). Pick **one** cost as the ranking key (recommend prox deviation — it is exactly the NLP
objective's data term); expose the other as a descriptor.

### 4.2 Policy — the three selection rules (mirror DPCC)

In `HardFlowPolicy.__call__`, replace `which_trajectory = 0` with the DPCC logic, driven by a
`trajectory_selection` arg:
- `'random'` → `0`;
- `'temporal_consistency'` → the DPCC `argsort` over `‖obs[:,:-1] − prev_observations[:,1:]‖`
  (copy the DPCC block verbatim — it is model-agnostic);
- `'minimum_projection_cost'` → `argmin(infos['candidate_costs'])`.

Keep `self.prev_observations` updated with the *selected* candidate each replan, exactly as DPCC does,
so temporal consistency works across the MPC loop.

### 4.3 Eval wiring — variant suffixes

The eval already routes `variant.startswith('hardflow')` to arm C. Extend the suffix parse to mirror
DPCC's:
- `hardflow_new` / `hardflow_new-r` → random,
- `hardflow_new-t` → temporal_consistency,
- `hardflow_new-c` → minimum_projection_cost.

`hardflow.batch_size` already exists (default 1 = faithful; set 4 for the fan). Encode
`selection + batch` in the results provenance (dir/npz) so a selection sweep never collides.

### 4.4 No change to the NLP, loading, or U4's threshold

Purely a candidate-fan + selection layer on top of the existing per-candidate sampler. Orthogonal to
U4 (activation threshold) — the two combine (fan of late-activated candidates).

## 5. Gates (extend `gates_hardflow.py`)

- **G7 fan independence:** with `batch_size=4`, the 4 candidates differ (distinct initial noise →
  distinct chains) and each terminal is feasible (safety holds per candidate, U4 §2.1).
- **G8 selection correctness:** `minimum_projection_cost` returns the `argmin` of the reported
  candidate costs; `temporal_consistency` reproduces DPCC's `argsort[0]` on a stubbed prev-plan;
  `random` returns index 0. Deterministic checks on stub data.
- **G9 cost monotonicity (sanity):** a candidate the NLP moved more has a strictly larger reported
  `candidate_cost` — confirms the ranking key measures intervention.

## 6. Experiment design — the confound-closing matrix

Matched-K (PLAN §5), matched batch, seed 6, raise `n_trials` toward n ≥ 100.

| factor | values |
|---|---|
| K | {2, 5, 10} |
| method × selection | B: `dpcc-c` / `dpcc-r` / `dpcc-t`  ·  C: `hardflow_new-c` / `-r` / `-t` |
| batch_size | 4 (matched); plus C at batch 1 as the faithful reference |

**The questions:**
1. **Does batch-4 + selection fix arm C's low-K failure?** (fix_3 §5's decisive test.) If
   `hardflow_new-c @ batch4` becomes 100%/100% at K=2/5, then fix_3's low-K collapse was largely the
   missing fan, not the method — and the fix_3 negative must be softened.
2. **At matched K *and* matched batch *and* matched selection rule, does C beat B?** This is the
   cleanest possible in-loop-vs-post-hoc comparison — every other factor held equal.
3. **Which selection rule is best for C?** (Does `-c` dominate `-r/-t` as it tends to for DPCC?)

Combine with U4: the most interesting single cell is **`hardflow_new-c`, batch 4, activation
threshold 0.5** — a late-activated, cost-selected 4-candidate fan, i.e. arm C with DPCC's full
machinery at DPCC-like cost.

## 7. Traps

1. **Keep the faithful batch-1 variant.** It is the honest comparison to Gen13 and upstream (which
   asserts batch==1). Do not delete it when adding the fan; make batch a config knob (it already is).
2. **Cost definition must be fixed and documented.** Prox deviation vs control effort rank candidates
   slightly differently; pick one as the key (recommend prox deviation) and record both.
3. **`prev_observations` threading.** Temporal consistency needs the *selected* candidate stored each
   replan; a stale/҂wrong prev-plan silently degrades `-t`. Mirror DPCC exactly.
4. **Compute scales with batch.** batch-4 arm C solves ~4× the NLPs of batch-1 → ~4× wall time. Report
   s/plan and NLP-solves per cell; the fair "success-per-second" comparison must account for it.
5. **Provenance.** selection × batch × K × (U4 threshold) is a big grid — encode all in the results
   path/npz or a sweep overwrites (PLAN §3.6, bitten twice already).
6. **Selection needs candidate diversity.** All 4 candidates share the conditioning `s0`; they differ
   only by initial noise. At very low K the chains may be near-identical → selection has little to
   choose from. Note this when reading K=2 results.

## 8. Success criteria

- **Minimum (confound closed):** batch-4 arm C runs with all three selection rules; G7–G9 pass; we can
  state whether the fix_3 low-K failure survives once B and C are matched on batch + selection.
- **Target:** at matched K/batch/selection, a clear verdict on in-loop vs post-hoc — either C ties B
  (clean negative, now *fully* controlled) or C wins somewhere (first positive for the contribution).
- **Stretch:** `hardflow_new-c @ batch4 + threshold 0.5` (U4 ∩ U4.2) beats `dpcc-c @ batch4` on
  success-per-second at some K.

## 9. Out of scope

- Retraining (Gen12 is eval-only); `linear_fit` dynamics; iMF/MeanFlow backbones (FMv3ODE-only).
- Value-model candidate ranking (upstream HardFlow's warmstart used a value model; FMPCC has none —
  the cost-based ranking in §3.1 replaces it).

---

### Appendix — code touch-points

- `flow_matcher_v3_hardflow/sampling/hardflow_projection.py`
  - `HardFlowSampler.sample`: accumulate `candidate_costs` per `b`; return in `infos`.
  - `HardFlowPolicy.__init__`: add `trajectory_selection`; `__call__`: replace `which_trajectory = 0`
    with the r/t/c logic (copy DPCC's `temporal_consistency` block).
- `FM_v3_hardflow_test/eval_FM_v3_hardflow.py`: map `hardflow_new-{c,r,t}` suffix → `trajectory_selection`
  (mirror the existing `dpcc-{c,r,t}` parse); pass `hardflow.batch_size` (already read) through.
- `config/hardflow_projection_eval.yaml`: add `hardflow_new-c/-r/-t` to `projection_variants`;
  `hardflow.batch_size: 4`; `hardflow.candidate_cost: prox|control` (ranking key).
- Reference: DPCC selection in `flow_matcher_v3/sampling/policies.py` (`trajectory_selection` block).
