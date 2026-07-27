# Gen12 U5 — code verification: does `hardflow_new-*-tightened` actually tighten (like DPCC)?

**Question (user):** is `hardflow_new-c-tightened` vs `hardflow_new-c` a *real* tightened-vs-exact
difference the way `dpcc-c-tightened` vs `dpcc-c` is — or does the margin silently get dropped
somewhere on the HardFlow path (which uses a casadi/ipopt NLP, not DPCC's scipy projector)?

**Verdict: it really works.** The HardFlow variants consume the **exact same** `constraint_list`
objects DPCC does; the only difference between the tightened and non-tightened arm is the
geometric margin baked into those objects, and that margin provably reaches the NLP. Traced below.

---

## The chain (file:line)

### 1. Two constraint lists are built, identically for both engines
`FM_v3_hardflow_test/eval_FM_v3_hardflow.py:218-240`

```
constraint_list            <- enlarge = 0        (exact)
constraint_list_tightened  <- enlarge = enlarge_constraints (=0.025 for avoiding)
```

| constraint | exact (`constraint_list`) | tightened (`constraint_list_tightened`) | line |
|---|---|---|---|
| halfspace | `formulate_halfspace_constraints(c, 0, …)` | `formulate_halfspace_constraints(c, enlarge, …)` | 223 / 224 |
| obstacle | `radius` | `radius + enlarge` | 232 / 233 |
| bounds | lb/ub | **same lb/ub (NOT enlarged)** | 228 / 229 |
| dynamics | identical | identical | 239 / 240 |

This is exactly DPCC's tightening rule (halfspace + obstacle geometry only; bounds & dynamics
untouched).

### 2. The dispatch picks the tightened list for hardflow-tightened variants
`eval_FM_v3_hardflow.py:265-272`
```python
elif not 'model_free' in variant and 'tightened' in variant:
    constraints = constraint_list_tightened      # <- hardflow_new-*-tightened lands here
else:
    constraints = constraint_list                # <- hardflow_new-* (exact)
```
`hardflow_new-c-tightened` contains `tightened` and not `model_free` → **tightened list**.
`hardflow_new-c` → **exact list**. (Same branch DPCC arms use — nothing hardflow-specific.)

### 3. The SAME object is handed to the HardFlow policy
`eval_FM_v3_hardflow.py:310-313`
```python
policy = HardFlowPolicy(..., constraint_list=constraints, ...)
```
No copy, no filtering — the tightened/exact `constraints` object flows straight in. (Arm B's
`Projector(..., constraint_list=constraints, solver='scipy')` at :327 receives the *identical*
object, which is why the two engines are matched on the feasible set.)

### 4. The NLP consumes the margin symbolically
`flow_matcher_v3_hardflow/sampling/hardflow_projection.py:197-236` (`HardFlowNLP._apply_constraints`,
always called from `__init__` at :160)

- **halfspace** `:204-210` — `c_row, d = spec[1]; opti.subject_to(c_row·x_unnorm <= d)`.
  The margin lives in `d` (see §5). Applied for every step `t ∈ [1, H)`.
- **obstacle** `:227-234` — `radius = float(spec[3]); opti.subject_to(sq >= radius**2)`.
  The tightened list passed `radius + enlarge` here, so the NLP enforces the **enlarged** circle.
- **bounds** `:214-225`, **dynamics** `:238-253` — unchanged between the two lists, as intended.

So the enlarged geometry is a hard casadi constraint, not a cosmetic tag.

### 5. `formulate_halfspace_constraints` returns `(C_row, d)` with the margin in `d`
`flow_matcher_v3_hardflow/utils/constraints_helpers.py:4-20`
```python
points_enlarged = [c[0] + enlarge*n, c[1] + enlarge*n]   # shift boundary by enlarge along normal
d = points_enlarged[0][1] - m*points_enlarged[0][0]
return C_row, d
```
Return type `(C_row, d)` matches the NLP's unpack `c_row, d = spec[1]` at :205 exactly — no scipy
callable that would break the tuple unpack, no shape mismatch. The margin enters through `d`.

---

## Numeric proof the margin is non-zero and correctly signed

Recomputing `d` (halfspace) and `r` (obstacle) at `enlarge=0` vs `0.025` for the avoiding tasks:

```
HALFSPACE (enforced C_row·x <= d ; SMALLER d = tighter feasible region)
constraint              d(enlarge=0)  d(enlarge=0.025)      Δd
both-hard[2] (\)             2.5444         2.4521      -0.0923
both-hard[3] (/)            -1.0111        -1.1034      -0.0923
top-right[0] (\)             1.5000         1.4327      -0.0673
top-left[1]  (/)            -1.0000        -1.0673      -0.0673

OBSTACLE (enforced sq >= r^2 ; LARGER r = tighter)
  base r=0.06 -> 0.085   (r^2 0.00360 -> 0.00722)
  base r=0.08 -> 0.105   (r^2 0.00640 -> 0.01103)
```

Both move in the **restrictive** direction (halfspace `d` shrinks, obstacle `r` grows) by a
material amount. The tightened NLP genuinely solves against a smaller feasible set than the exact
NLP — the tightened-vs-exact difference is real, in the solver, for the HardFlow path.

---

## Honest scope / caveats

- **What is provably identical to DPCC:** the *margin* and the *constraint objects*. Both engines
  read the same `constraint_list` / `constraint_list_tightened`, so tightening semantics match by
  construction (that's the whole point of arm B being the safeguard — it hit the same 3.70e-8 as
  the standalone DPCC eval, §4a of the run analysis).
- **What differs by design (not a bug):** the *solver* (casadi/ipopt in-loop vs scipy post-hoc)
  and *when* the constraint is enforced. The NLP applies ineq/obstacle to steps `t ∈ [1, H)` and
  bounds to action dims from `t=0` — the code comments (`:220-222`, `:242-249`) assert this mirrors
  DPCC's `SafetyConstraints.build_matrices`. This timestep coverage is identical between the
  tightened and exact hardflow arms, so it does **not** confound the tightened-vs-exact comparison.
- **Consistency with the observed data:** verified path predicts exactly what the run shows —
  `hardflow_new-c` (exact) leaves small residual violations (rides the zero-margin boundary),
  `hardflow_new-c-tightened` reaches 0 violations / 100% succ+con. The +40 pt jump is the margin
  doing its job through the NLP, confirming the code trace.

**Conclusion:** `hardflow_new-*-tightened` tightens the same constraints, by the same margin, as
`dpcc-*-tightened`; the enlargement is enforced inside the casadi NLP (halfspace `d`, obstacle
`r²`), and the exact-vs-tightened distinction is genuine, not a naming artifact.
