# SolverSwap fix — `nlp_failures` was blind to the hard solver failures

**Date** 2026-08-30 · **Found by** gates job 25206 (`gates_mix_visual`, G0 FAIL) on `3df77e8`
**Introduced by** `ee9a4fc4` *(add scipy SLSQP backend option to HardFlow solver)*, 2026-08-27
**Files** `mix_visual_aligning/`, `mix_visual_avoiding/`, `mix_uav/` `sampling/projection.py`
· `mix_visual_aligning_test/gates_mix_visual.py`
**Status** code only. **Nothing re-run** (no torch in the AI container); the accounting fix is
proven by a behavioural simulation of the loop, not by a cluster run.

---

## 1. The bug

SolverSwap added per-solve convergence telemetry to the DPCC projector:

```python
self.last_solve_success = []
for i in range(batch_size):
    try:
        res = minimize(...)                      # SLSQP
    except _SolveBudgetExceeded:
        sol_np[i] = trajectory_np_double[i]      # keep it UNPROJECTED
        projection_costs[i] = np.inf
        print('[ projector ] solve backstop hit ...')
        continue                                 # 🔴 appends NOTHING
    self.last_solve_success.append(bool(res.success))
```

The consumer, identical in six generations (`hardflow_projection._solve_slsqp`):

```python
n_bad = sum(1 for ok in getattr(self.projector, 'last_solve_success', ()) if not ok)
self.n_failures += n_bad          # -> `nlp_failures` in the run summary
```

**The `continue` fires before the append**, so the backstop path contributes no entry. The
consequence is the worst possible polarity:

| outcome | severity | recorded before the fix |
|---|---|---|
| SLSQP converged | none | `True` ✅ |
| SLSQP returned, `success=False` | soft — scipy's last iterate kept, may be infeasible | `False` ✅ counted |
| **60 s backstop hit** | **hard — trajectory kept UNPROJECTED at `cost=inf`** | **nothing ❌ invisible** |

`nlp_failures` counted the soft failures and **under-reported exactly the failures that matter
most.** A run could time out on solve after solve, keep every one of those plans unprojected, and
still report `nlp_failures = 0`.

A second, latent consequence: the list stopped being index-aligned with the batch. Every consumer
today only *counts*, so nothing is currently misattributed — but `last_solve_success[i]` was no
longer element `i`, which is a trap for the next caller that indexes.

### Why it survived three days — the second bug

`ee9a4fc4` grafted this into `projection.py` but left the file in G0's **verbatim-COPIED** list, so
every gates run since has reported:

```
G0 FAIL — the copy assumption broke. Re-open the plan; do NOT patch over it:
  ! mix_visual_aligning/sampling/projection.py: DIFFERS from fm_visual_aligning/...
```

The gate was right, and the failure was load-bearing: it was flagging an unaudited graft, and that
graft had a bug in it. But because the entry looked like stale bookkeeping, nobody re-read the code.
**A permanently-red gate is a gate nobody reads.**

## 2. The fix

### 2.1 Record the backstop as the failure it is — 3 files

```python
    except _SolveBudgetExceeded:
        ...
        self.last_solve_success.append(False)    # ← added, BEFORE the continue
        continue
```

Applied identically to `mix_visual_aligning`, `mix_visual_avoiding` and `mix_uav`. The three
`flow_matcher_v3_*` copies carry the same telemetry but have **no** backstop branch, so they have no
early exit to skip and needed no change (verified).

Behavioural check on a batch of `['ok', 'backstop', 'nonconv', 'ok', 'backstop']` — 3 true failures:

| | entries | `n_bad` | truth | index-aligned |
|---|---|---|---|---|
| before | 3/5 | **1** | 3 | ❌ misaligned |
| after | 5/5 | **3** | 3 | ✅ aligned |

### 2.2 Register the graft so G0 goes meaningfully green

`projection.py` moved out of `COPIED` and into `GRAFTED_DIFF` with **`removed = 0`** — the strongest
form of that entry. The graft is purely insertive, so zero upstream lines were rewritten; any future
edit that changes an *existing* line in the file fails G0 immediately. Verified: `added=18,
removed=0` against the `fm_visual_aligning` reference.

## 3. Scope — what is and is not affected

- ✅ **No result changes.** `last_solve_success` is telemetry only; `sol_np`, `projection_costs` and
  the circuit breaker are untouched. Every number in every existing DA and report stands.
- ⚠️ **`nlp_failures` in past runs is a lower bound**, not a count. Any HardFlow-SLSQP run since
  2026-08-27 reporting a small or zero `nlp_failures` cannot be read as "the solver was healthy" —
  backstop hits were unrepresented. Re-reading the `[ projector ] solve backstop hit` lines in those
  job logs is the only way to recover the true figure retrospectively.
- The sibling gates (`gates_mix_visual_avoiding.py`, `mix_uav`) do **not** check `projection.py` at
  all, so there was no equivalent ledger entry to correct there.

## 4. Noted, deliberately not changed

- `mix_visual_avoiding/sampling/projection.py` diverges from its own reference by **119 added / 9
  removed** — far beyond this graft. Its gate does not check the file, so this is invisible today.
  Out of scope here; worth a separate audit.
- `projection.py:127` raises `SyntaxWarning: invalid escape sequence '\h'` (a LaTeX `\hat` in a
  non-raw docstring). It is **pre-existing and present in the untouched `fm_visual_aligning`
  reference**, so it is not from this work — and fixing it in the mix copy alone would add a removed
  line and break the `removed=0` additive property just registered. Fix upstream first, if at all.

## 5. Verify on the cluster

1. `gates_mix_visual.sh` → **G0 PASS** with `projection.py` listed under
   *"grafted, additive-only"* with `(+18 lines, -0)`.
2. On the next HardFlow-SLSQP eval, confirm a deliberate backstop (lower `_PROJ_SOLVE_BACKSTOP_S`)
   increments `nlp_failures` rather than passing silently.
