# CHANGELOG — Gen3v6 fix_7: HardFlow-arm plot crash (`IndexError`) + figure leak

**Date:** 2026-08-01 · **Follows:** [`../fix_6/CHANGELOG_Gen3v6_fix_6_resume.md`](../fix_6/CHANGELOG_Gen3v6_fix_6_resume.md)
**Trigger:** `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh`
died at `JOB END: Sat Aug 1 22:52:57 UTC 2026`, on the first HardFlow variant.

---

## 1. Symptom

```
------Running avoiding-d3il - top-right-hard - hardflow_new-r (7) - K=2------
RuntimeWarning: More than 20 figures have been opened. ...
Traceback (most recent call last):
  File ".../eval_flow_matching_v3_meanflow.py", line 491, in <module>
    curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, obs_indices['x']], ...)
IndexError: index 1 is out of bounds for axis 0 with size 1
```

Diffuser + all six DPCC arms had already completed. It died on variant **8 of 13**.

## 2. Root cause — a local/global `batch_size` mismatch in the plot loop

`batch_size` is rebound **per arm**:

| | line | value |
|---|---|---|
| arms A/B (diffuser / dpcc-\*) | `eval:345` | `batch_size = args.batch_size` → **4** (`config/avoiding-d3il.py:1008`) |
| arm C (hardflow_new-\*) | `eval:316` | `batch_size = hf_batch_size` → **1** |

and `hf_batch_size` (`eval:75`) resolves
`HFFM_BATCH` → `hardflow.batch_size` → `1`, with
`config/meanflow_projection_eval.yaml:96` saying `batch_size: 1`.

The foresight-fan plot loop then iterated the **global** value:

```python
for ___ in range(min(args.batch_size, 4)):        # 4 …
    curr_ax.plot(sampled_trajectories_all[i][__][___, ...])   # … but the array has 1 row
```

So on arm C it indexed row 1 of a 1-row candidate array. Arms A/B are immune because there the
two names hold the same number.

**This is a port defect, not a design question.** Both siblings already read the local name:

| file | line | form |
|---|---|---|
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | 445 | `range(min(batch_size, 4))` ✅ |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | 534 | `range(min(batch_size, 4))` ✅ |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | 489 | `range(min(args.batch_size, 4))` ❌ |

Gen12's HardFlow eval had it right; the **U3 port into Gen3v6 introduced the HardFlow branch's
`batch_size` override but left the plot loop on `args.batch_size`.** The fix is sibling-parity
restoration, nothing more.

Every generation *without* a HardFlow arm (`scripts/eval.py`, `FM_v3_test/`, `FM_v2_test/`,
`FM_v3_drifting_test/`, `FM_v3_imeanflow_test/`, `FM_v3_ode_selectable_test/`,
`FM_Unet_v2_test/`, `FM_hp_tune_test/`, `FM_test/`) still carries `min(args.batch_size, 4)` and
is **correct as written** — with no per-arm override the two names cannot disagree. No mirroring
needed there.

## 3. Why it never fired before

Every sweep in the fix_4 / fix_5 record went through `eval_meanflow_hardflow.sh`, which exports
`HFFM_BATCH` (`:85`). The matched-K runs (jobs 24034–24038, 24074–24078) used `HFFM_BATCH=4`,
which coincidentally made `hf_batch_size == args.batch_size == 4` and hid the bug completely.

`eval_meanflow.sh` exports **nothing** — no `HFFM_BATCH`, no `HFFM_ACT_THRESHOLD`, no
`HFFM_FLOW_STEPS` — so the yaml default of 1 applied and the mismatch surfaced immediately.

The two scripts otherwise call the identical entry point with no `--config`, so both resolve to
`config/meanflow_projection_eval.yaml` (`eval:48`, repointed at U3) — i.e. `eval_meanflow.sh`
*does* run the HardFlow arms, which is why it could reach the crash at all. The comment at
`eval_meanflow.sh:81` claiming it reads `config/projection_eval.yaml` is **stale** (that file has
zero `hardflow` entries). Left untouched by fix_7; see §7.

## 4. Blast radius of the crashed run

The crash is inside the trial loop, **before** the `np.savez` at `:518`, and the sbatch runs
under `set -e`:

| | status |
|---|---|
| `top-right-hard`, arms A/B (7 variants) | ✅ `.npz` written |
| `top-right-hard`, arm C (6 variants) | ❌ lost — died on the first one |
| `top-left-hard` (all 13) | ❌ never ran |
| `both-hard` (all 13) | ❌ never ran |

**No previously-analysed result is affected.** Nothing was corrupted — the run simply stopped.

## 5. Changes

Both in `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` (+10 / −1). No other file.

### (a) `:489` — the crash

```python
# 🔴 fix_7 — iterate the LOCAL batch, not args.batch_size. Arm C
# overrides it (batch_size = hf_batch_size, :316) and the yaml
# default is 1, so with HFFM_BATCH unset this asked for index 1 of
# a 1-row candidate array -> IndexError. Arms A/B are unaffected
# (batch_size = args.batch_size, :345). Restores parity with the
# alphaflow/hardflow siblings, which already read `batch_size`.
for ___ in range(min(batch_size, 4)):
```

`batch_size` is bound on **both** branches of the `if is_hardflow:` split, so it is always in
scope at this point. The change is a **no-op for every run in the record** (all had the two equal
at 4) — no existing number, plot or `.npz` changes.

### (b) `:546` — the figure leak

```python
fig_all.savefig(f'{save_path}/all.png')
plt.close(fig_all)   # fix_7: was the only figure never closed -> "More than 20
                     # figures have been opened" once the variant list grew to 13.
```

`fig` was already closed (`:538`) and `figs_all_seeds` at `:564`; `fig_all` was the only one left
open. Cosmetic, but at 13 variants × 3 halfspace variants × 5 seeds the leak is real memory.

## 6. Found but deliberately NOT fixed

`fig_all` is created at **`:374`, inside the per-variant loop**, while `ax_all[i, variant_idx]` is
written across variants — so `all.png` only ever contains the last variant's column, the rest
being blank axes. Upstream DPCC (`/workspaces/aux_repo/dpcc/scripts/eval.py:124`) creates it
**outside** that loop, which is the intended cross-variant comparison figure.

The regression is old and lineage-wide (`scripts/eval.py:238` has it too), so it predates Gen3v6
by many generations. Fixing it here alone would change `all.png` for Gen3v6 only and break
sibling parity for a plot nobody currently reads — the analysis all runs off the `.npz`. Recorded
here so it is not rediscovered as new; deferred to a lineage-wide sweep.

## 7. Operational note — which sbatch to use

Fixing the crash does **not** make `eval_meanflow.sh` the right launcher for a HardFlow sweep:

- no `HFFM_ACT_THRESHOLD` ⇒ HardFlow runs at the yaml's `1.0`, not the DPCC-parity `0.5`;
- no `HFFM_FLOW_STEPS` ⇒ each arm uses its own plan `flow_steps` instead of matched K, and the
  results path carries no `_K{K}_` token, so runs collide;
- at mpc-batch 1 the `-r` / `-c` / `-t` suffixes all collapse to index 0 (`eval:319`), so the six
  HardFlow variants would produce six identical result sets.

**Use `eval_meanflow_hardflow.sh` for anything comparable to the fix_4 / fix_5 tables.** After
fix_7, `eval_meanflow.sh` at least completes rather than crashing — it is a valid arms-A/B run
with a degenerate arm C.

## 8. Verification

- `ast.parse` on the edited file — pass.
- `git diff --stat` — 1 file, +10 / −1, both hunks as intended.
- Line endings: file is LF; `file` reports no CRLF introduced.
- Scope check: `grep` confirms every other `min(args.batch_size, 4)` site lacks a per-arm
  override, so none needs the change.
- **Not run locally** (no python env in this container). Cluster job required:

```bash
# the crashed configuration, now expected to complete all 13 variants × 3 exps
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh

# the real sweep launcher (unchanged behaviour — the fix is a no-op at HFFM_BATCH=4)
HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS=2 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

Regression check on the second command: its numbers must be **bit-identical** to jobs
24074–24078, since the fix cannot alter behaviour when the two batch sizes agree.

## 9. Still open

- `eval_meanflow.sh:81` stale comment (`config/projection_eval.yaml`) — not touched.
- `fig_all` per-variant recreation, §6 — lineage-wide, deferred.
- Pre-existing from fix_5/fix_6: `dpcc-c-tightened` K=2 collapse (`cand_prox` ranking key),
  Fix A (τ² weight, `hardflow_projection.py:515-516`), Fix B (`candidate_cost: control`),
  `n_trials: 2` and seed-6-only in the eval yaml.
