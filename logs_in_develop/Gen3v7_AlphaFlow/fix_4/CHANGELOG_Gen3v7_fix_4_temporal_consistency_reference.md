# Gen3v7 fix_4 — `prev_observations` indexes the wrong candidate in the `-t` branch

**Date:** 2026-07-31 · **Gen:** Gen3v7 (α-Flow) · **Epoch:** fix_4
**File changed:** `flow_matcher_v3_alphaflow/sampling/policies.py:58-81` (1 logical change)
**Mirrors:** Gen3v6 fix_5 — commit `ecbae16f`, `../../Gen3v6_MeanFlow/fix_5/CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md`
**Affects:** `dpcc-t`, `dpcc-t-tightened` at `batch_size > 1` only
**Status:** applied, **not committed**, **re-eval required** (see §5)

---

## 1. The bug

Gen3v7 carried the pre-fix block byte-identical to Gen3v6:

```python
order = np.argsort(...)                 # a permutation of the batch
which_trajectory = order[0]
observations = observations[order]      # observations now SORTED
...
self.prev_observations = np.repeat(np.expand_dims(observations[which_trajectory], 0), batch_size, 0)
```

`which_trajectory` indexes the **original** batch — correctly, since `actions` is never reordered
and the executed action is `actions[which_trajectory, 0]`. But `observations` **has** been
reordered, so the stored reference applies `order` a second time:

| quantity | value |
|---|---|
| plan actually executed | `observations_old[order[0]]` = `observations_sorted[0]` |
| plan actually stored | `observations_sorted[order[0]]` = `observations_old[order[order[0]]]` |

**Equal iff `order[0] == 0`.** Proof: if `order[0] = j ≠ 0` then value `j` is already consumed at
position 0 of the permutation, so `order[j] ≠ j` — never equal. At `batch_size = 4`, 6 of the
`4! = 24` permutations have `order[0] = 0`, so **~75% of replans stored a plan the agent never
followed**, and the next replan then ran its temporal-consistency sort against that phantom.

This is a double-applied permutation, not a modelling choice. Two independent confirmations:

- Upstream DPCC (`/workspaces/aux_repo/dpcc/diffuser/sampling/policies.py:76`) writes
  `observations[0]`, which **is** correct, because `observations_sorted[0] == observations_old[order[0]]`.
  The rest of the block is byte-identical to upstream.
- The line was *changed away* from `observations[0]` by `82c151c2` (MPC_NPZ_PATCH, JOB C). It is a
  regression, and one aimed at a dead path: `trajectory_selection` is fixed at construction
  (`policies.py:32`), so a `-c` policy never enters the `temporal_consistency` branch and its
  `prev_observations` is write-only. JOB C had no effect where it was aimed, and broke `-t` in
  exactly the invariant it was written to establish.

## 2. Severity — calibrated

The wrong reference is still a *valid* trajectory drawn from the same conditional fan, so nothing
crashes, no constraint is violated, and **the executed action was always correct**. What degrades is
the heuristic itself: temporal consistency stops meaning "stay close to the plan I am following" and
becomes "stay close to a random sibling of it." A degraded heuristic, not a safety defect — which is
why the `-t` scores in the U3 sweep are 2.5–3.0 rather than garbage.

## 3. The fix

Introduce `executed_idx` — the index of the executed plan *within `observations` as it stands at the
assignment* — and leave `which_trajectory` (which indexes `actions`) untouched:

| branch | `which_trajectory` (→ `actions`) | `executed_idx` (→ `observations`) |
|---|---|---|
| `temporal_consistency` | `order[0]` | **`0`** (observations sorted) |
| `minimum_projection_cost` | `argmin(costs)` | `which_trajectory` (not reordered) |
| random / fallback | `0` | `0` |

Behaviourally identical to upstream for `-t` and `random`, and keeps JOB C's genuine (if
unobservable) intent for `-c`. Chosen over "drop the `observations = observations[order]` reorder"
because the reorder is load-bearing elsewhere: the eval reads `samples.observations[0, 1, ...]`,
which is the executed plan for `-t` only because of the sort.

The code is now identical to `flow_matcher_v3_meanflow/sampling/policies.py` apart from the comment
tag (`fix_4` here vs `fix_5` there).

## 4. Blast radius

**Affected** (`batch_size > 1` only): `dpcc-t`, `dpcc-t-tightened`.
**Not affected:**

- All `hardflow_new-*`. Gen3v7's `HardFlowPolicy._select`
  (`flow_matcher_v3_alphaflow/sampling/hardflow_projection.py:701-721`) *returns* `int(order[0])` and
  never reorders `observations`, so line 693's `observations[which_trajectory]` was always the
  executed plan. **Arm C was correct; arm B was not** — same asymmetry as Gen3v6.
- `dpcc-r*`, `diffuser` — `which_trajectory = 0`, no sort.
- `dpcc-c*` — `prev_observations` write-only (see §1).
- Everything at `batch_size == 1` — `order = [0]` trivially.

## 5. Re-eval — required, and more urgent than it was for Gen3v6

The U3 sweep (jobs 24044–24048, `temp/3107/`, git `cb859e3`) ran `HFFM_BATCH=4`, so it is
bug-affected. **`../U3/RESULTS_Gen3v7_U3_hardflow_K_sweep.md` must be re-run before any `-t` number
in it is published.**

Gen3v6 fix_5 §6 could defer its re-run because `dpcc-t-tightened` was at the 3.0 ceiling in 4 of 5 K
— a correct implementation had nowhere to go, so the "arm C matches, does not beat" headline was
conservative. **Gen3v7 does not have that safety margin:**

| | K=1 | K=2 | K=5 | K=10 | K=20 | headroom |
|---|---|---|---|---|---|---|
| Gen3v6 `dpcc-t-tightened` | 3.0 | 2.5 | 3.0 | 3.0 | 3.0 | 0.5 |
| **Gen3v7 `dpcc-t-tightened`** | 3.0 | **2.5** | 3.0 | **2.5** | **2.5** | **1.5** |
| **Gen3v7 `dpcc-t`** | 3.0 | 1.0 | 2.5 | 2.5 | 2.0 | **4.0** |

The U3 headline is the tightened **r + t** subtotal: **DPCC 28.5 vs HardFlow 29.0** out of 30. The
bug sits *exclusively* in the arm that loses, and arm C's corresponding cells are correct. A repaired
`dpcc-t-tightened` can gain up to 1.5 — enough to reach 30.0 and **invert the ranking**.

This is a directional bias, not two-sided noise. That distinction is the whole reason to re-run: the
existing gap cannot be defended as "within the noise floor" when the error term has a known sign.

### 5.1 What happens if we do **not** re-eval

Ranked by how much damage each costs.

1. **The paper headline flips or dies.** §1 item 6 / §3's "HardFlow edges ahead on the tightened r+t
   subtotal (29.0 vs 28.5)" is the only quantitative claim of HardFlow superiority in the document.
   It rests on a 0.5-point gap measured against a baseline handicapped by up to 1.5. Publishing it is
   claiming a win produced by our own bug. This is the single reason re-eval is not optional.
2. **§3.1's cross-generation table becomes unreadable.** It compares Gen3v7 `dpcc-t-tightened`
   (2.5 at K=10/20) against Gen3v6 `dpcc-t-tightened` (3.0 at K=10/20) — but as of `ecbae16f` the
   Gen3v6 row will be *re-measured fixed* while the Gen3v7 row stays buggy. A mixed-provenance table
   invites exactly the wrong conclusion, that α-Flow's DPCC baseline is *worse* than MeanFlow's, when
   the difference may be entirely the fix.
3. **Two secondary tables carry silent errors.** §4 freeze rate `dpcc-t-tightened`
   (0.8/2.2/1.8/1.0/1.0%) and §7 compute `dpcc-t-tightened` (0.015–0.968 s, and the 0.3%/0.8% budget
   overruns at K=1/2) all shift, because the trajectory actually flown changes. §7's *conclusion*
   survives — the real-time claim rests on `dpcc-r-tightened`, which is unaffected — but the printed
   `-t` cells will not reproduce.
4. **Any future `-t` result is non-comparable.** Seeds 7–10, already queued in §9/§10 as the fix for
   the n=2 noise floor, will run on fixed code. Pooling them with the current sweep silently mixes
   two different algorithms under one variant name.

**What survives untouched even with no re-eval**, and can be cited today:

- §2's 84-cell port-isolation control — both sides (`temp/2807` pre-port, `temp/3107` post-port) ran
  the *same* buggy `policies.py`, so the bug cancels exactly. The control is still valid.
- The entire `-c` story: §4, §5, §5(b′) bit-identical frozen counts, §6 K=2 collapse. `-c` never
  reads `prev_observations`, and HardFlow's `-c` ranks by `candidate_costs`.
- §7's real-time finding (`dpcc-r-tightened` at K ≤ 2, and HardFlow's 100% overrun at every K).
- §8's U2-INV reconciliation — `dpcc-c` / `dpcc-r` violation counts.
- §3.1's `hardflow_new-r-tightened` cross-generation row.

### 5.2 Cost of the re-eval

The five jobs ran serially 21:06 → 00:30 = **~3 h 25 m** total on i6-gpu-1. Re-run **all five K**, not
just `dpcc-t*`: at this price a surgical re-run only buys a table whose cells come from two different
code revisions.

```bash
Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh
```

(matched-K sweep, `HFFM_BATCH=4`, `HFFM_ACT_THRESHOLD=0.5` — unchanged from jobs 24044–24048; the
only delta is the git rev). Gates should still report 4/4; this fix touches nothing they assert.

## 6. Not fixed here (deliberate)

The 8 remaining sibling copies of `MPC_NPZ_PATCH`, all carrying the identical defect:
`diffuser/sampling/policies.py:76` (the vendored DPCC baseline itself), `flow_matcher_v3_hardflow/`,
`flow_matcher_v3_imeanflow/`, `flow_matcher_v3_drifting/`, `flow_matcher_v3_ode_selectable/`,
`flow_matcher_v3_uav/`, `flow_matcher_v3/`, `flow_matcher_v2/`, `flow_matcher_unet_v2/`,
`flow_matcher/`. Each needs the same edit before any `batch_size > 1` `-t` result from it is trusted.
Awaiting explicit go-ahead.

Also still open and unrelated: the `cand_prox` ranking key
(`hardflow_projection.py:515-516`) — see `../U3/RESULTS_Gen3v7_U3_hardflow_K_sweep.md` §5.

## 7. Validation

Not run — no Python in this container. To verify on the cluster: at `batch_size=4`, assert each
replan that `prev_observations[0]` equals the plan whose first action was executed. Behavioural
signature: `dpcc-t*` results change at `batch_size > 1` and are bit-identical at `batch_size == 1`.
