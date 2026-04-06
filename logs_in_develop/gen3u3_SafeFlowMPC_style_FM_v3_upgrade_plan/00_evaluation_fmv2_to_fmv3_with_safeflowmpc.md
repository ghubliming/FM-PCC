# 00 Evaluation First: FM-v2 to FM-v3 Upgrade (SafeFlowMPC-aligned)

Date: 2026-04-06
Status: Review Pending
Scope: FM-PCC only, with SafeFlowMPC as reference design

## First Principle

Use the same SafeFlowMPC FM engine semantics, but implemented inside FM-PCC structure.

This means:
1. preserve SafeFlowMPC FM meaning (continuous-time velocity-field learning + ODE rollout semantics),
2. keep FM-PCC project structure, dataset flow, logging, and experiment interfaces,
3. avoid introducing behavior that silently departs from SafeFlowMPC FM assumptions.

---

## 1) Why this document exists

This is the pre-plan evaluation for an FM-v3 upgrade path.

Goal of this document:
1. Decide whether FM-v3 upgrade is justified now.
2. Define exact upgrade boundaries before implementation planning.
3. Identify risks that must be controlled in v3.

This is intentionally evaluation-only.
Plan document 01 is not created yet and should be written only after review approval.

---

## 2) Current FM-v2 baseline (what we already have)

Current v2 behavior in FM-PCC:
1. SafeFlowMPC-style Beta time sampling in training (`time_beta_alpha_v2`, `time_beta_beta_v2`).
2. Decoupled time knobs:
   - `vf_time_bins_v2` for model time-conditioning representation.
   - `ode_inference_steps_v2` for Euler integration resolution at inference.
3. Config path split is explicit:
   - training block: `flow_matching_v2`
   - planning/eval block: `plan_fm_v2`
4. Known and accepted interpretation:
   - changing ODE inference steps does not require retraining,
   - changing VF time-bin representation is a training-design change.

Baseline conclusion:
FM-v2 is conceptually correct and usable. FM-v3 must deliver measurable value, not just naming or structural churn.

### 2.1) Clarified meaning of old FM-v2 reported results

For the historical mismatch window, old reported numbers should be interpreted as:
1. **Eval ODE=20 rollout results**,
2. with **Beta-time FM training**,
3. not an Eval ODE=10 benchmark.

Reason this matters for v3 evaluation baseline:
1. v3 must compare against correctly labeled v2 reference points,
2. v3 claims must not use mislabeled historical baselines,
3. all comparison tables must include explicit eval solver-step labels.

---

## 3) What FM-v3 should mean (evaluation definition)

For this project, FM-v3 should mean:
1. Same SafeFlowMPC FM engine semantics inside FM-PCC structure.
2. Better reliability and experiment integrity than v2.
3. Cleaner SafeFlowMPC alignment where it improves correctness or maintainability.

FM-v3 should not mean:
1. uncontrolled architectural changes,
2. silent parameter behavior changes,
3. reintroducing train/eval mismatch risk.

---

## 4) Evaluation criteria (go/no-go gate)

FM-v3 is justified only if it can satisfy all required criteria below.

### Required criteria

1. Parameter integrity
   - Prevent accidental drift between train and eval critical knobs.
   - At minimum guard: `flow_steps_v3` (with `ode_inference_steps_v3` compatibility alias) consistency checks.

2. Reproducibility
   - Every run must record effective runtime config and checkpoint provenance.
   - Evaluation reports must include solver steps explicitly.
   - Historical and new results must use consistent labels (for example: `Eval ODE=20`, `Eval ODE=10`).

3. Backward comparability
   - FM-v3 must support controlled A/B against v2 on same dataset and seeds.
   - Must provide a clean migration story for existing v2 checkpoints or clearly declare incompatibility.

4. SafeFlowMPC-consistent semantics
   - Preserve continuous-time FM interpretation.
   - Preserve distinction between training-time time sampling and inference-time ODE solver resolution.
   - In v3 path, prefer SafeFlow-style continuous-time model query semantics.
   - Keep behavior parity with SafeFlowMPC FM engine semantics unless explicitly documented as an intentional divergence.

5. Operational simplicity
   - No duplicated knobs with ambiguous meaning.
   - Naming and script paths must be obvious and stable.

### Optional high-value criteria

1. Better runtime-performance tradeoff for equal quality.
2. Better failure messaging when config is invalid.
3. Cleaner report automation for sweeps.

---

## 5) Gap analysis: FM-v2 vs desired FM-v3

### Gaps already observed in v2 workflow

1. Human confusion around time knobs
   - Frequent confusion between VF time bins and ODE steps.

2. Drift risk in split config blocks
   - Training and planning blocks can diverge unless manually synchronized.

3. Evaluation labeling risk
   - Results can be misinterpreted if solver-step metadata is omitted.

4. Naming history complexity
   - Legacy naming variants caused script/doc mismatch risk.

### Consequence if not fixed

1. Hard-to-interpret experiments.
2. Accidental invalid comparisons.
3. Reduced trust in conclusions despite valid code.

---

## 6) Risk evaluation for FM-v3 effort

### Technical risks

1. Hidden behavior regressions during refactor.
2. Over-expansion of scope (architecture changes mixed with reliability upgrades).
3. Incompatibility with existing result-loading tooling.

### Process risks

1. Premature plan execution before evaluation agreement.
2. Unclear acceptance criteria leading to endless iteration.

### Mitigation requirements

1. Keep v3 scope narrow and explicit.
2. Add fail-fast runtime checks before expensive runs.
3. Define mandatory experiment matrix before coding begins.

---

## 7) Suggested acceptance tests for go decision

A go decision should require all tests below to be feasible and planned.

1. Integrity test
   - Intentionally misalign critical knobs and verify hard error with clear message.

2. Equivalence test
   - Same checkpoint, same seed, compare ODE steps sweep (for example 10/12/16/20).

3. Stability test
   - Repeat selected setting across multiple seeds and verify metric variance is reasonable.

4. Reporting test
   - Confirm each output artifact includes exact effective parameters.
   - Confirm each result row includes explicit eval solver-step label and train/eval experiment keys.

---

## 8) Go/No-Go recommendation (evaluation result)

Recommendation: Conditional Go.

Reasoning:
1. FM-v2 is already methodologically sound.
2. Main remaining weakness is reliability/guardrail/traceability, not FM math.
3. A tightly scoped FM-v3 focused on guardrails and experiment integrity is justified.

Condition for go:
1. Keep v3 focused on reliability and reproducibility improvements first.
2. Defer major architecture experimentation to a later v3.x branch unless explicitly requested.

---

## 9) Proposed FM-v3 scope boundary for next document (01 Plan)

If approved, 01 Plan should include:
1. Naming and file map for v3 scripts/config blocks.
2. Runtime guardrail design (consistency checks + fail-fast).
3. Logging/report schema for effective params and checkpoint identity.
4. Experiment matrix and success thresholds.
5. Rollout strategy with rollback path.

Not in initial 01 Plan unless explicitly approved:
1. major backbone redesign,
2. broad objective-function replacement,
3. cross-repo deep dependency rework.

---

## 10) Decision request

Review this 00 evaluation and confirm:
1. Approve conditional-go direction,
2. Keep initial v3 scope reliability-first,
3. Proceed to write 01 implementation plan.
