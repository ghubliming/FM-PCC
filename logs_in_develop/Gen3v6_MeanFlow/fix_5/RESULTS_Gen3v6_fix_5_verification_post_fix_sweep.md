# RESULTS — Gen3v6 fix_5 verification: the temporal-consistency fix is live, and correctly scoped

**Date:** 2026-08-01 · **Verifies:** [`CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md`](CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md)
**Supersedes the 🔶 caveat in** [`../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md`](../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md) §8

---

## 0. Verdict

**Yes — fix_5 works, and it does exactly what it was predicted to do, nothing more.**

The evidence is stronger than usual because the eval turned out to be **bit-deterministic**:
across 195 (variant × K × halfspace) cells, the 165 cells fix_5 was predicted not to touch
came back **byte-identical on every behavioural metric**, and the 30 cells it was predicted to
touch are the only ones that moved. There is no ambiguity about whether the fix is live.

What it does **not** establish is that the fixed arm is *better* — at `n_trials: 2` on a
single seed, the magnitudes are inside the noise floor. §5 is explicit about this.

---

## 1. The A/B

| | BEFORE | AFTER |
|---|---|---|
| Jobs | 24034–24038 (2026-07-30) | **24074–24078** (2026-07-31) |
| Git | `87b01d9` | `b5846ee` |
| `policies.py:70` | `observations[which_trajectory]` ❌ | `observations[executed_idx]` ✅ |
| Console logs | `temp/2026-07-30/II/` | `temp/3107/3107/2026-07-31/14_18_43_*` |
| Results tree | — | `temp/0108/flow_matching_v3_meanflow/` |

Everything else is held fixed: seed 6 only, the same `state_best.pt` (seed 6 was not
retrained — jobs 24069/24100 touched seeds 7–10 only), `HFFM_BATCH=4`,
`HFFM_ACT_THRESHOLD=0.5`, `n_trials: 2`, the same 13 `projection_variants`, the same five
matched-K points {1, 2, 5, 10, 20}.

`ecbae16f` (fix_5) is an **ancestor** of `b5846ee`, and `git show b5846ee:...policies.py`
confirms `executed_idx` is present in the AFTER run and absent in `87b01d9`. The
`temp/0108/` npz tree was cross-checked against the console logs cell-by-cell — same run,
same numbers.

---

## 2. The decisive evidence: blast radius

fix_5 touched one line, guarded by `trajectory_selection == 'temporal_consistency'` on the
DPCC path. So **only `dpcc-t` and `dpcc-t-tightened` may change; the other 11 arms must not.**

Comparing all 195 cells on the six behavioural metrics (`sr`, `cs`, `gc`, `steps`, `nviol`,
`tviol` — `ctime` excluded, it is wall-clock):

| Arm | Cells differing (of 15) |
|---|---|
| `diffuser` | 0 |
| `dpcc-r`, `dpcc-r-tightened` | 0, 0 |
| `dpcc-c`, `dpcc-c-tightened` | 0, 0 |
| **`dpcc-t`** | **14** |
| **`dpcc-t-tightened`** | **14** |
| `hardflow_new-{r,c,t}` | 0, 0, 0 |
| `hardflow_new-{r,c,t}-tightened` | 0, 0, 0 |

**28 of 195 cells changed, and all 28 are `dpcc-t*`.** Three separate claims from the fix_5
changelog are confirmed by this table, each by an arm that did *not* move:

1. **`hardflow_new-t*` unchanged** ⇒ `hardflow_projection.py:653` was already correct. It
   stores `observations[which_trajectory]` but never reorders `observations`, so its index
   was always valid. fix_5 correctly left it alone.
2. **`dpcc-c*` unchanged** ⇒ the `-c` branch's `prev_observations` write really is **dead**.
   `trajectory_selection` is fixed at construction, so a `-c` policy never reads the
   temporal-consistency reference. This is what made the original `MPC_NPZ_PATCH` so
   deceptive: it "repaired" a path nobody reads while breaking the one that matters.
3. **`dpcc-r*` and `diffuser` unchanged** ⇒ the `else` branch and the projection-free arm are
   untouched, as intended.

The 165 zero-difference cells double as a determinism check: identical `steps`, `nviol` and
`tviol` to full printed precision means the environment, sampler and solver replay exactly,
so **any** difference in the `dpcc-t*` cells is attributable to the code change and nothing
else. That is a stronger control than this benchmark usually affords.

---

## 3. What changed numerically

### `dpcc-t-tightened` — now at ceiling across the whole sweep

g&c summed over the 3 halfspaces (max 3.0):

| K | 1 | 2 | 5 | 10 | 20 | sweep total (max 15) |
|---|---|---|---|---|---|---|
| before | 3.00 | **2.50** | 3.00 | 3.00 | 3.00 | 14.50 |
| after | 3.00 | **3.00** | 3.00 | 3.00 | 3.00 | **15.00** |

The single sub-ceiling cell in the entire before-sweep was K=2 / `top-right-hard` (0.50), and
it is exactly the cell that repaired to 1.00. Constraint violations were already 0.00
everywhere on this arm, before and after.

### `dpcc-t` (untightened) — noisier, net positive

| K | 1 | 2 | 5 | 10 | 20 | sweep total |
|---|---|---|---|---|---|---|
| before | 0.50 | 1.50 | 1.00 | 1.00 | 1.50 | 5.50 |
| after | 1.00 | 1.00 | 1.50 | **2.50** | 1.00 | **7.00** |

Up at K=1/5/10, down at K=2/20. The K=10 cell is the most interesting one, because the
constraint metrics move coherently with the success rate rather than independently:

| K=10, `dpcc-t`, summed over halfspaces | before | after |
|---|---|---|
| g&c | 1.00 | **2.50** |
| avg # constraint violations | 9.50 | **1.00** |
| avg total violation | 0.27 | **0.00** |

K=5 shows the same direction (violations 14.50 → 8.50, total violation 0.82 → 0.33). That is
the signature you would expect if the temporal-consistency reference is now the plan actually
executed: the replanner is tracking its own previous trajectory instead of an arbitrary
sibling from the batch, so successive plans are consistent and drift into the obstacle less.

---

## 4. Impact on the fix_4 headline

fix_4 concluded: **arm C (HardFlow) matches, but does not beat, DPCC.** That claim had exactly
one exception — K=2, where `hardflow_new-t-tightened` (3.00) beat `dpcc-t-tightened` (2.50),
arm C's only lead anywhere in the sweep. The changelog flagged that cell as bug-affected.

It was. After fix_5:

| K=2, tightened | before | after |
|---|---|---|
| `dpcc-t-tightened` (arm B) | 2.50 | **3.00** |
| `hardflow_new-t-tightened` (arm C) | 3.00 | 3.00 |

**Arm C's last lead is gone.** On this benchmark, at seed 6, HardFlow now beats DPCC at *no*
value of K. The fix_4 conclusion was conservative and survives intact — it is now simply
cleaner: `dpcc-t-tightened` 15.00/15.00 vs `hardflow_new-t-tightened` 14.50/15.00 over the
whole sweep.

The 🔶 bullet in [`../fix_4/RESULTS_...md`](../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md)
§8 marking the `dpcc-t*` column as superseded can now be resolved: the column has been
re-measured and this file is the replacement.

---

## 5. What this does NOT establish

**The magnitudes are inside the noise floor.** `n_trials: 2` per halfspace means each cell is
one of {0.0, 0.5, 1.0} and each sweep column is 6 trials. Concretely:

- The `dpcc-t-tightened` K=2 repair is **one trial**.
- The `dpcc-t` K=10 gain of +1.50 is **three trials**, and the K=2/K=20 losses are one each.
- `dpcc-t`'s whole-sweep +1.50 is 3 trials out of 30.

So: *"fix_5 is live and correctly scoped"* is **proven**. *"fix_5 makes the arm better"* is
**consistent with the data but not demonstrated** — the only reason to believe the direction
is the mechanism (§3, the violation counts moving coherently with success), not the sample
size. `dpcc-t-tightened` was already at 14.50/15.00 before the fix, i.e. near ceiling, so
there was never room for the fix to show a large effect there.

**Seed 6 only.** Seeds 7 and 8 finished on 2026-07-31, seed 9 resumed on 08-01, seed 10 is
pending. Nothing here is a multi-seed result and none of it should be published as one.

---

## 6. Confirmed independent: the `dpcc-c` K=2 collapse

`dpcc-c-tightened` sits at **0.00/3.00 at K=2** while scoring 3.00 at K≥5 — the anomaly
investigated in [`../fix_4/RESULTS_...md`](../fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md)
§5.8. It is **bit-identical before and after fix_5**, which settles one hypothesis: the
collapse has nothing to do with the temporal-consistency reference. Combined with §5.8's
ruling-out of a solve-abort mechanism, the `-c` ranking key itself (`cand_prox`, measured on
the extrapolation `X1_ref = X_ref + (1−τ_next)·V_next`, unweighted) remains the standing
suspect. Still open.

---

## 7. Next

1. **Multi-seed.** Re-run the sweep with `seeds: [6,7,8,9,10]` (uncomment
   `config/meanflow_projection_eval.yaml:21`) once seed 10 is trained. That is the only thing
   that converts §3 from "consistent with" to "shows".
2. **Mirror fix_5 to the remaining siblings.** `flow_matcher_v3_alphaflow/` already has it
   (`a6a7a8ad`, Gen3v7 fix_4). Still unpatched: `diffuser/sampling/policies.py:76` (the
   vendored DPCC baseline — note upstream is *correct* there, so this is a no-op check rather
   than a fix), `flow_matcher_v3_hardflow/`, `_imeanflow/`, `_drifting/`, `_ode_selectable/`,
   `_uav/`, `flow_matcher_v3/`, `_v2/`, `flow_matcher_unet_v2/`, `flow_matcher/`.
3. **`cand_prox`** ranking key — the second defect from fix_5 §7, untouched and now the
   leading explanation for §6.
4. Raise `n_trials` above 2 before any of these columns goes in a paper table.

---

## Appendix — reproducing this comparison

```bash
# BEFORE: temp/2026-07-30/II/*.log            (24034-24038, git 87b01d9)
# AFTER:  temp/3107/3107/2026-07-31/14_18_43_*.log  (24074-24078, git b5846ee)
# parse both, diff every cell on sr/cs/gc/steps/nviol/tviol (NOT ctime — wall-clock)
```

Parser and diff scripts used: `cmp.py` / `cmp3.py` in the session scratchpad; they extract
per-cell metrics from the `------Running <env> - <halfspace> - <variant> (<seed>) - K=<K>------`
blocks that both log generations emit identically.
