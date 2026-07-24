# C6 — Run report: Fix_15 guard fires en masse on `dpcc-r × combined_5` (JOB 23293)

**Module:** `fm_visual_aligning/` (Gen7) · **Log:** `temp/23_39_40_eval_fm_visual_aligning_23293.log`
**Run:** `eval_fm_visual_aligning`, JOB 23293, NODE i6-gpu-1, **GIT `c54feb9`** (includes Fix_15 guard),
seed 6, geo `combined_5`, 24 items (variant × tightened). Related: `CHANGELOG_projection_cost_explosion_guard.md`,
Gen11 `Fix_15_projection_cost_explosion_guard/`.

## TL;DR

The Fix_15 guard **worked** (no hard hang/death), but it fired **6,470 times — all inside a single
variant, `dpcc-r`** (item 2/24). This is the **same SLSQP pathology** as the UAV `bounds_free` case:
on the `combined_5` geometry the projection QCQP does not converge, so nearly every solve runs to the
2.0 s cap and falls back to the unprojected trajectory. The guard turned an infinite hang into a slow
bleed and, crucially, **labelled the broken (variant × geometry) pair** — but `dpcc-r` results here are
invalid and the job still won't finish in budget.

## What the log shows

| item | variant | proj | ~time/replan | COST EXPLODED | outcome |
|---|---|---|---|---|---|
| 1/24 | `diffuser` | off | ~0.33 s | **0** | clean, completed |
| 2/24 | `dpcc-r` | on | 0.58–1.53 s | **6,470** | all 25 trial summaries `Success: False`; log cut off still inside item 2 |

- **Every explosion in the entire 8,421-line log is in item 2 (`dpcc-r`).** Item 1 had none.
- SLSQP signature present: `_slsqp_py.py:437: RuntimeWarning: Values in x were outside bounds during a
  minimize step, clipping to bounds` — iterates leave the feasible set immediately.
- Explosions spread evenly across all 4 MPC batch elements (~1,585–1,684 each) → not one bad sample,
  the whole constraint problem is pathological.
- `6,470 × 2.0 s ≈ 3.6 h` burned in item 2 **alone**; the run never reached item 3/24.

## Interpretation

1. **Guard verified (the win).** Without Fix_15 this job hangs on the first bad `dpcc-r` solve and dies
   at the SLURM time limit with no signal. Instead it capped each runaway solve at 2 s, kept moving,
   and made the failure visible + greppable. The guard is now effectively a **diagnostic** flagging
   `dpcc-r × combined_5` as a broken projection.

2. **`dpcc-r` results are NOT valid — read them as unprojected.** Every exploded solve keeps the
   **unprojected FM trajectory**, so the DPCC safety projection *did not run* on those solves. All 25
   `dpcc-r` summaries are `Success: False`. Do **not** read this as "DPCC projects badly here" — read
   it as "**DPCC projection didn't actually execute here**"; `dpcc-r` silently degraded toward plain
   `diffuser`.

3. **Still too slow to finish.** The guard caps solve *duration*, not the *count* of runaway solves.
   At ~3.6 h per pathological variant and 24 items, the job still blows the time budget (it was cut off
   mid-item-2).

## Root cause (the real bug to fix)

`dpcc-r × combined_5` is a **constraint-feasibility / conditioning problem**, not merely performance.
The near-universal `clipping to bounds` warning means SLSQP's linearized QP sub-problem is infeasible
or badly scaled for this geometry. Same failure family as the UAV `bounds_free` set.

## Recommended next steps

- **Fast triage (no code change):** set `FMPCC_PROJ_SOLVE_BUDGET_S=0.3` so a broken variant fails in
  minutes instead of burning 2 s per dead solve — same pattern, far cheaper to observe.
- **Implement Guard 3 (per-variant abort):** if the explosion rate in a variant exceeds a threshold,
  abort the remaining trials, stamp `status=cost_exploded` in the results, and move on — so `dpcc-r`
  can't eat 3.6 h producing invalid output while starving the other 22 items.
- **Fix the projection feasibility for `combined_5`:** warm-start `x0` from the previous step's
  solution; verify the `combined_5` constraint set is jointly feasible; check normalization/scaling of
  the QCQP. This is the actual defect the guard exposed.

*All validation runs on the cluster (i6-gpu-1); no Python executes in the dev container.*
