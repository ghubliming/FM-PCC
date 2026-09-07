---
name: hardflow-low-K-degeneracy
description: HardFlow rows at low K run NO HardFlow math — never build an HF claim on them
metadata:
  type: feedback
---

Before quoting ANY `hardflow_new-*` row, check `n_genuine = max(K − floor((1−A)·K), 1) − 1`.
`n_genuine == 0` ⇒ the row is `Π_S(Euler sample)` — sample-then-project, i.e. DPCC's algorithm with
IPOPT instead of SLSQP. **At the shipped A = 0.5 (every generation except Gen12), K=1 and K=2 are
degenerate; K=3 is alive but marginal (n_genuine=1); K≥5 is genuine.** Gen12 ships A = 1.0 → degenerate
at K=1 only. `A = 0.0` is degenerate at every K.

**Why:** the user has flagged this twice, the second time because a DA I wrote picked a K=1 row as
"HardFlow's best result" and credited a K=2 cell with "in-loop beats post-hoc". Both are wrong by
construction — no in-loop guidance step executed in either. A degenerate row is still a valid
*solver* comparison (SLSQP vs IPOPT on one terminal projection); it is never evidence about HardFlow.

**How to apply:** tag every row ❌ degenerate / ⚠️ marginal / ✅ genuine in the table itself, not in a
footnote. Compute best-of / win-count / Pareto claims over ✅ rows ONLY. When a sweep's only HF rows
are K ≤ 2, say the sweep carries no HardFlow signal and ask for K ≥ 5. Full derivation:
`logs_in_develop/aggregated_hardflow_lowK/REGISTER_20260824_degenerate_HF_rows_and_warnings.md` §0–1
and `logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`.
Related: [[benchmark-hierarchy-who-beats-whom]], [[pareto-definition-of-good]].
