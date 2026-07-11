# Fix_15 — Projection cost-explosion guard ("kill a runaway solve, don't burn 24 h")

**Gen11 / Epoch9 (PCC Constraints)** · status: **PROBLEM ANALYSIS + PROPOSAL (not yet implemented)**
Trigger log: `temp/20_59_54_eval_fm_uav_23265.log` (JOB 23265, `eval_fm_uav`, scene=pillars, seed=6, git `6d251f6`).

---

## 1. What happened

A single-seed eval job (**1 seed**, `pillars`, 20 projection variants × 10 trials) hit the **24 h SLURM
wall-clock limit and was CANCELLED mid-way through variant 15/20 (`dpcc-r`)** — it never finished.
The last 6 variants (`dpcc-r`, `dpcc-r-tightened`, `dpcc-c`, `dpcc-c-tightened`, `dpcc-t`,
`dpcc-t-tightened`) produced **zero results**.

The 24 h was not spread evenly. Two variants ate ~73 % of the entire budget:

| # | variant | wall time (10 trials) | proj_ms avg / p95 | note |
|---|---|---|---|---|
| 1 | diffuser | 600 s | 0 / 0 | proj off |
| 2 | gradient | 675 s | 0 / 0 | |
| 3 | gradient-tightened | 601 s | 0 / 0 | |
| 4 | post_processing | 1 923 s | 220 / 2 886 | |
| 5 | post_processing-tightened | 2 060 s | 255 / 3 843 | |
| 6 | model_free | 1 671 s | 168 / 363 | |
| 7 | model_free-tightened | 1 672 s | 183 / 388 | |
| **8** | **bounds_free** | **23 898 s ≈ 6.6 h** | **3 966 / 40 665** | ⚠ |
| **9** | **bounds_free-tightened** | **39 243 s ≈ 10.9 h** | **6 402 / 112 177** | ⚠⚠ |
| 10 | geo_free | 947 s | 56 / 142 | |
| 11 | geo_free-bounds_free | 888 s | 46 / 138 | |
| 12 | geo_free-model_free | 766 s | 27 / 129 | |
| 13 | model_free-bounds_free | 1 601 s | 157 / 342 | |
| 14 | model_free-bounds_free-tightened | 1 670 s | 167 / 399 | |
| 15 | dpcc-r | **KILLED** (time limit) | — | never finished |

`bounds_free` + `bounds_free-tightened` alone = **~63 140 s ≈ 17.5 h**. The other 12 completed
variants together took ~4.2 h.

### The real-time context (why 24 h is absurd)

- Control rate: **33 Hz** (`DATASET_HZ`, `eval_fm_uav.py:58`) → **real-time budget ≈ 30.3 ms/step**
  (matches `budget=30.3ms` in every TIMING line).
- Episode length: **634 steps** (`max_episode_length`, U_13 fixed budget).
- A real-time-faithful episode is therefore **634 × 30.3 ms ≈ 19 s**.
- `bounds_free-tightened` averaged **3 924 s per trial** → **~200× slower than the episode it is
  supposedly controlling in real time.** Its worst single projection step took **112 s** (p95),
  i.e. one 30 ms control tick expanded by **~3 700×**.

The `real_time_OVER×NNNN` flag in the log already *knows* every step blew the budget — but the flag
is post-hoc telemetry printed *after* the variant finishes. Nothing acts on it, so a variant is free
to run for 11 hours and only then report that it was 5 777× over budget.

---

## 2. Root cause: SLSQP thrashing on the `bounds_free` feasible region

The projector is DPCC's `Projector.project()`, copied verbatim into
`flow_matcher_v3_uav/sampling/projection.py` (imported at `eval_fm_uav.py:660`). Per **FM step**,
per **MPC replan**, per **batch element (B=4)**, it solves:

```python
res = minimize(fun=cost_fun, x0=trajectory, constraints=constraints,
               method='SLSQP', jac=jac_cost_fun,
               bounds=Bounds(-5, 5),
               tol=1e-6, options={'maxiter': 1000, 'disp': False})   # projection.py:135-142
```

The obstacle constraints are a **nonconvex QCQP**: one `{'type':'ineq'}` python-lambda per obstacle
**per horizon step** (`projection.py:118-122`), each doing `xᵀPx + qᵀx` inside the SLSQP inner loop.

`bounds_free` **removes the box/polytopic bound constraints** (`C`,`d`) while keeping
obstacles + geo_bounds + dynamics. Right before variant 8 starts, SLSQP begins emitting:

```
scipy/optimize/_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds
during a minimize step, clipping to bounds
```

That warning is the signature of the pathology: with the bound rows gone, SLSQP's linearized QP
sub-problem repeatedly proposes iterates **outside** the remaining feasible set, the line search
backtracks/clips, and the solver grinds toward `maxiter=1000` **without early exit**. Contrast the
sibling variants that instead drop the *obstacle* constraints:

- `geo_free` (obstacles/geo_bounds removed): **proj_ms ≈ 56**, ~100× faster.
- `bounds_free` (bounds removed, obstacles kept): **proj_ms ≈ 3 966**.

So the cost is **not** proportional to constraint count — it is a specific bad interaction
(`bounds_free` = nonconvex obstacles with the bounds that used to regularize the QP removed) that
sends SLSQP into near-worst-case iteration counts. `-tightened` enlarges the obstacle surfaces,
shrinking the feasible region further and making it worse still (6.4 s avg, 112 s p95).

---

## 3. How the original DPCC repo handles this → **it doesn't**

`/workspaces/aux_repo/dpcc/diffuser/sampling/projection.py` is the parent of our copy, and its
**only** guard is the same one we inherited:

```python
options={'maxiter': 1000, 'disp': False}     # dpcc projection.py:142
tol=1e-6
```

- **No wall-clock timeout** on the `minimize` call.
- **No per-step / per-episode / per-variant time budget.**
- **No "abort if over real-time budget" logic** anywhere.
- **No detection of the "clipping to bounds" non-convergence** — the `RuntimeWarning` is swallowed.

DPCC gets away with it because it evaluates on **D3IL `avoiding`/`aligning`** — short-horizon,
low-obstacle-count, offline benchmark episodes where even a maxed-out SLSQP solve is cheap and the
task list is small. **`maxiter=1000` bounds the *iteration count*, not the *wall time*** — and on our
UAV workload (634 steps × MPC-per-step × B=4 × nonconvex obstacles) a single 1000-iteration solve can
still take **>100 s**. DPCC's safeguard simply does not cover our regime. **There is no upstream
safeguard to port; we have to add one.**

---

## 4. Proposal — a projection cost-explosion guard

**Principle:** a controller that claims to run at 33 Hz should never be allowed to spend hours on one
episode. If a solve/step/episode runs grossly past its real-time budget, **abort it, record it as
`cost_exploded`, and move on** — do not let it starve the rest of the job.

Three nested guards, cheapest first. (1) is the minimum viable fix; (2)+(3) make it clean.

### Guard 1 — per-solve wall-clock cap inside `minimize` (the real fix)

`scipy`'s SLSQP has no native timeout, but the callback/constraint hooks run in-process, so we raise
out of them once a deadline passes:

```python
class _SolveBudgetExceeded(Exception): pass

def _project_one(..., solve_budget_s):
    t0 = time.perf_counter()
    def _deadline_cb(xk):                      # SLSQP calls this each iteration
        if time.perf_counter() - t0 > solve_budget_s:
            raise _SolveBudgetExceeded()
    try:
        res = minimize(..., method='SLSQP', callback=_deadline_cb,
                       options={'maxiter': 1000, 'disp': False})
    except _SolveBudgetExceeded:
        return x0, np.inf, True     # (fallback = unprojected traj, cost=inf, exploded=True)
    return res.x, cost, False
```

- `solve_budget_s` default e.g. **2.0 s** (≈ 66× the 30 ms real-time tick — generous, still bounds a
  step at seconds not minutes). Configurable via `config/uav_projection.yaml`
  (`projection_solve_budget_s`).
- On abort, **fall back to the unprojected (FM) trajectory** for that batch element — the eval still
  produces a rollout, just an unconstrained-at-that-step one, and the metrics record it as unsafe.
  This is strictly better than the current behavior (the whole *job* dies).
- `callback` only fires between iterations, so a single pathological inner line-search can overrun the
  deadline slightly — acceptable. (Harder guarantee = run the solve in a worker
  process/`SIGALRM`, deferred unless Guard-1 proves insufficient.)

### Guard 2 — per-episode budget kill (bound the blast radius)

In `rollout_one`, track cumulative projection time; if one episode exceeds
`episode_proj_budget_s` (e.g. **N× the real-time episode length**, `634 × 30.3 ms × N`), **stop the
episode early**, mark it `cost_exploded=True`, and return partial stats. Prevents a single trial from
consuming 6 000 s even if individual solves stay under Guard-1.

### Guard 3 — per-variant abort + explicit console/log marker (what the user asked for)

In `_run_variant` / the trial loop (`eval_fm_uav.py:1322-1350`), if the running average per trial
projects the variant to exceed a variant budget (or if ≥K trials came back `cost_exploded`),
**abort the remaining trials of that variant** and emit a loud, greppable marker to **both stdout and
the written eval log** (`artifacts.write_eval_log`):

```
[ eval ] pillars variant=bounds_free-tightened >>> COST EXPLODED: aborted after 3/10 trials
         (avg 3924.0s/trial ≫ variant budget 600s; 3 trials hit solve/episode cap) — SKIPPING rest
```

and stamp `summary['cost_exploded'] = True` + `summary['status'] = 'cost_exploded'` in
`results.json` so the aggregator (`aggregate_scene_summaries.py`) and MASTER history can show the
variant as *deliberately skipped for cost*, not silently missing. This directly answers the ask:
**"a 400-step 33 Hz job running for hours should be directly killed and marked as cost-exploded in the
console and the real log outputs."**

### Where the budgets live

Add to `config/uav_projection.yaml` (per the `.yaml = constraint-projection config` convention):

```yaml
projection_solve_budget_s:   2.0     # Guard 1: per SLSQP solve
projection_episode_budget_x:  8      # Guard 2: episode cap = x · (max_ep_len · 1/hz)
projection_variant_budget_s: 900     # Guard 3: hard cap per variant (0 = disabled)
```

All default-on but tunable; setting to `0`/`inf` restores today's unguarded behavior for A/B.

### Non-goals / do-not-touch

- **Do not change the solver math** (`maxiter`, `tol`, `Bounds`, constraint construction). The guard
  is purely a *watchdog* around the existing solve — it must not alter feasible results, only cap
  runaway ones.
- **Do not silently drop the variant** — a cost-exploded variant must appear in outputs, flagged.
- Keep the change inside the UAV sibling (`flow_matcher_v3_uav` / `FM_v3_uav_test`); mirror to other
  active gens (Gen7/Gen6V4) only after it's validated here, per the copy-modify sibling convention.

---

## 5. Follow-ups (separate from the guard)

- **Investigate the `bounds_free` pathology directly**: warm-start `x0` from the previous step's
  solution, or re-add a loose box bound so SLSQP's QP sub-problem stays regularized even in the
  "bounds_free" constraint set. The guard stops the bleeding; this would remove the cause.
- **Split `bounds_free*` into their own low-`n_trials` job** as an interim, so a normal 20-variant
  run fits in 24 h.
- Consider a **convex QP solver** (proxsuite/OSQP, already referenced by the `solver=` arg) for the
  obstacle constraints via sequential convexification, instead of raw SLSQP on the nonconvex QCQP.

---

*Run all validation on the cluster (i6-gpu-1); no Python executes in this container.*
