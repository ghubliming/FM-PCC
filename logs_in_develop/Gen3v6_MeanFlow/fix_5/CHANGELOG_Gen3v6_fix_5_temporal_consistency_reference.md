# Gen3v6 fix_5 — `prev_observations` indexes the wrong candidate in the `-t` branch

**Date:** 2026-07-31 · **Gen:** Gen3v6 (MeanFlow) · **Epoch:** fix_5
**File changed:** `flow_matcher_v3_meanflow/sampling/policies.py:58-70` (1 logical change)
**Affects:** `dpcc-t`, `dpcc-t-tightened` at `batch_size > 1` only
**Status:** applied, **not committed**, **not yet re-run on cluster**

---

## 1. The bug

`policies.py` selects a candidate and then stores the executed plan as the reference for the
next replan's temporal-consistency sort. Before this fix:

```python
if trajectory_selection == 'temporal_consistency' and ...:
    order = np.argsort(...)                 # order[0] = index into the ORIGINAL batch
    which_trajectory = order[0]
    observations = observations[order]      # observations is now SORTED
elif trajectory_selection == 'minimum_projection_cost' and ...:
    which_trajectory = np.argmin(costs_total)
else:
    which_trajectory = 0
self.prev_observations = np.repeat(np.expand_dims(observations[which_trajectory], 0), batch_size, 0)
```

`which_trajectory` indexes the **original** batch — correctly so, because `actions` is never
reordered and line 86 does `actions[which_trajectory, 0]`. But in the `-t` branch
`observations` **has** been reordered, so `observations[which_trajectory]` evaluates to
`observations_original[order[order[0]]]`, which is not the executed plan.

The executed plan sits at `observations_sorted[0]`.

## 2. Severity — exact, not estimated

Correct **iff** `order[0] == 0`. Proof: if `order[0] = j ≠ 0`, then value `j` is already
consumed at position 0 of the permutation, so `order[j] ≠ j` — never correct. Enumerating all
`4! = 24` permutations at `batch_size=4`:

| `order[0]` | permutations | stored == executed |
|---|---|---|
| 0 | 6 | 6 (100%) |
| 1 | 6 | 0 |
| 2 | 6 | 0 |
| 3 | 6 | 0 |
| **total** | **24** | **6 (25%)** |

**~75% of replans stored a phantom reference** — a valid trajectory from the fan, but not the
one executed. The next replan then ran temporal consistency against a plan the agent never
followed. The executed action itself was always correct (`actions` is unsorted).

## 3. Provenance — a fix that broke a live path to repair a dead one

Introduced by commit `82c151c2` (2026-06-21, *"Danger Giant MPC traj npz saving patch"*),
documented as JOB C in `logs_in_develop/MPC_traj_saved_in_npz/CHANGELOG.md:52-59`:

> **Bug:** `observations[0]` → **Fix:** `observations[which_trajectory]`
> This ensures that when `minimum_projection_cost` (`dpcc-c`) selects a non-zero trajectory, the
> temporal consistency window for the **next** step references the trajectory that was actually
> executed, not always candidate 0.

Two problems with that reasoning:

1. **The intended beneficiary is a dead path.** `trajectory_selection` is fixed at construction
   (`policies.py:32`), so a `-c` policy never enters the `temporal_consistency` branch —
   `prev_observations` is **write-only** for `-c`. JOB C had no effect where it was aimed.
2. **The `-t` reorder was not accounted for**, so the patch violated in `-t` the exact invariant
   it was written to establish.

Upstream (`/workspaces/aux_repo/dpcc/diffuser/sampling/policies.py:76`) uses `observations[0]`
and is **correct**, because `observations_sorted[0] == observations_original[order[0]]`. The
rest of the block is byte-identical to upstream.

## 4. The fix

Introduce `executed_idx` — the index of the executed plan *within `observations` as it stands at
the assignment* — and leave `which_trajectory` (which indexes `actions`) untouched:

| branch | `which_trajectory` (→ `actions`) | `executed_idx` (→ `observations`) |
|---|---|---|
| `temporal_consistency` | `order[0]` | **`0`** (observations sorted) |
| `minimum_projection_cost` | `argmin(costs)` | `which_trajectory` (not reordered) |
| random / fallback | `0` | `0` |

This is behaviourally identical to upstream for `-t` and `random`, and keeps JOB C's genuine
(if unobservable) improvement for `-c`. Chosen over "drop the `observations = observations[order]`
reorder" because the reorder is load-bearing elsewhere: `eval:451` reads
`samples.observations[0, 1, ...]`, which is the executed plan for `-t` only because of the sort.

## 5. Blast radius

**Affected** (`batch_size > 1` only): `dpcc-t`, `dpcc-t-tightened`.
**Not affected:**

- all `hardflow_new-*` — `HardFlowPolicy._select` (`hardflow_projection.py:653,667`) never
  reorders `observations`, so its `observations[which_trajectory]` was always the executed plan.
  **Arm C was correct; arm B was not.**
- `dpcc-r*`, `diffuser` — `which_trajectory = 0`, no sort.
- `dpcc-c*` — `prev_observations` write-only.
- everything at `batch_size == 1` — `order = [0]` trivially.

## 6. Impact on the fix_4 sweep (jobs 24034-24038, `batch=4`)

Pre-fix `-t` numbers, g&c summed over 3 halfspaces (max 3.0):

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `dpcc-t-tightened` (buggy) | 3.0 | **2.5** | 3.0 | 3.0 | 3.0 |
| `hardflow_new-t-tightened` (correct) | 2.5 | **3.0** | 3.0 | 3.0 | 3.0 |

**The headline survives.** `dpcc-t-tightened` is already at ceiling at four of five K — a correct
implementation cannot score higher — so "arm C *matches*, does not beat, DPCC" stands, and was
if anything conservative about the baseline.

**One claim is now un-makeable until re-run:** the K=2 cell, the only place arm C leads. It is
bug-affected, and the 3.0-vs-2.5 gap is already inside the noise floor declared in
`../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md` §8 (`n_trials: 2`). Do not report it.

No emergency re-run of jobs 24034-24038 is required; the correction is subsumed by the
full-seed sweep. Re-run `dpcc-t*` **after** this fix and before publishing any baseline table.

## 7. Not fixed here (deliberate)

- **The 10 sibling copies.** `MPC_NPZ_PATCH` is repo-wide: `diffuser/sampling/policies.py:76`
  (the vendored DPCC baseline itself), `flow_matcher_v3_alphaflow/:70` (Gen3v7),
  `flow_matcher_v3_hardflow/`, `flow_matcher_v3_imeanflow/`, `flow_matcher_v3_drifting/`,
  `flow_matcher_v3_ode_selectable/`, `flow_matcher_v3_uav/`, `flow_matcher_v3/`,
  `flow_matcher_v2/`, `flow_matcher_unet_v2/`, `flow_matcher/`. All carry the identical defect
  and all need the same edit before any `batch_size > 1` `-t` result from them is trusted.
  **Gen3v7 (`flow_matcher_v3_alphaflow`) is the urgent one** — same class of sibling-sync failure
  that produced the fix_4 σ bug. Awaiting explicit go-ahead.
- **JOB B (`desired_next_pos`)**, deferred in the original changelog: `eval:451` reads
  `samples.observations[0, 1, ...]`, which is slot 0 rather than the executed candidate for `-c`.
  Still dead in practice — tracking error is printed only for `diffuser` (`eval:509`), which is
  why `terr` is null in every other cell.
- **The `cand_prox` ranking key** (`hardflow_projection.py:515-516`) — the *other* open defect,
  unrelated to this one. See `../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md` §5.3/§5.7.

## 8. Validation

Not run — no Python in this container. To verify on the cluster: at `batch_size=4`, assert each
replan that `prev_observations[0]` equals the plan whose first action was executed. Behavioural
signature: `dpcc-t*` results change at `batch_size>1` and are bit-identical at `batch_size=1`.
