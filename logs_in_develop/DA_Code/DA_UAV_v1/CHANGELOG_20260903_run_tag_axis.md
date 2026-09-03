# CHANGELOG — DA_UAV_v1: `run_tag` becomes a real axis

**Date:** 2026-09-03
**Tool:** `Data_Analysis/DA_UAV_v1/`
**Trigger:** `logs_in_develop/Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md` §2.4
**Severity:** 🔴 silent data loss — whole runs disappeared from the output CSVs with no error

---

## 1. The problem

`mix_uav_test/eval_mix_uav.py:_uav_eval_tag()` appends `FMPCC_UAV_EVAL_TAG` to the eval folder
name, so that two runs differing only in an env knob cannot overwrite each other's output:

```
Emf_K1_mpc4_pid_stopgo_T0.5              # untagged
Emf_K1_mpc4_pid_stopgo_T0.5_fix16scaled  # FMPCC_UAV_EVAL_TAG=fix16scaled
Emf_K1_mpc4_pid_stopgo_T0.5_fix16legacy  # FMPCC_UAV_EVAL_TAG=fix16legacy
```

`EVAL_TAG_RE` anchored on `_T{threshold}$`, so **a tagged folder did not match it at all.**

### 1.1 The failure chain

| # | what | consequence |
|---|---|---|
| 1 | `EVAL_TAG_RE` fails to match | `parse_eval_tag` returns `{}` |
| 2 | every path-encoded axis is empty; `K` is `None` → `NaN` | the run has no engine, no K, no controller |
| 3 | `display_name()` drops `K` from the label | all six tagged runs collapse to the **same** name, `pillars\|mf\|bbunet\|dp0.5` |
| 4 | `_reduce()` groups on `AGG_KEYS`, which contains `K` | pandas `groupby(dropna=True)` **deletes every row with a NaN key** |
| 5 | `agg_long` never sees them → `candidate_stats` never sees them | ⚠️ **absent from `candidates_detailed.csv` and `candidates_ranking.csv`** |
| 6 | `_build_k_sweep` filters `K.notna()` | ⚠️ **absent from `uav_k_sweep.csv`** |

No exception, no warning, no log line. `candidates_detailed.csv` carried **48 rows for 71
candidates** and nothing said which 23 were missing or why.

### 1.2 What it cost

The 2026-09-03 Fix_16 A/B — six cluster runs, ~14 h of GPU time, 551 rollouts — was invisible
in every aggregate CSV. It was recovered only because `per_rollout_detail.csv` keys on
`Candidate` (not on the axes) and still held the data. **A reader who trusted the summary
tables would have concluded the A/B was never run.**

### 1.3 The trap in the obvious fix

Making the regex match is *not* sufficient, and on its own is **worse than the bug**.

`K_SWEEP_KEYS = ['scene', 'engine', 'geo', 'variant', 'split', 'K']` deliberately excludes
`Candidate` — K *is* the candidate axis in that table. So once `K` parses, the pre-fix,
`fix16legacy` and `fix16scaled` runs of `mf`/`pillars`/`K1`/`diffuser` land in **one cell** and
get averaged: a 100 %-abort arm blended with a 0 %-abort arm into a meaningless ~66 %. A
silent drop is at least visible as a gap. A silent *pool* is not.

`run_tag` therefore has to be a key, not just a parsed field.

---

## 2. What changed

### 2.1 `config.py`

- `EVAL_TAG_RE` and `EVAL_TAG_RE_GEN11` gained an optional trailing
  `(?:_(?P<run_tag>.+))?` group. Safe at the end: the greedy `controller` backtracks past
  `_T{threshold}`, so `pid_stopgo` / `pid_const_v` / `pid_stopgo_anchorP` still parse and an
  untagged folder still yields `run_tag == ''`.
- **New** `EVAL_TAG_PREFIX_RE = r'^(?:E[A-Za-z0-9]+_)?K\d+_mpc\d+_'` — deliberately looser than
  `EVAL_TAG_RE`, used only to tell two failure modes apart (see §2.4).

### 2.2 `discovery.py`

- `parse_eval_tag` returns `run_tag`; `parse_axes` carries it into the axes dict.
- `display_name` appends `|@{run_tag}` → `pillars|mf|K1|bbunet|dp0.5|@fix16scaled`.
  **Without this the two arms of an A/B read as one candidate in every by-name table.**
- `_make_unit` puts `run_tag` on every unit, alongside the other axes.

### 2.3 `aggregator.py` / `reporter.py`

- `run_tag` added to `AXIS_COLUMNS` (both files) — it now rides `ID/UNIT/AGG_KEYS` and lands as
  a column in `run_config.csv`, `per_rollout_detail.csv` and the long tables.
- 🔴 `run_tag` added to **`K_SWEEP_KEYS`** — the fix for §1.3. Arms never pool.
- The "cell pools more than one candidate" warning now names run tag among the suspects.

### 2.4 Defence in depth — the class of bug, not just this instance

Two changes so this cannot recur silently:

1. **`_reduce()` groups with `dropna=False`.** A NaN in *any* key column no longer deletes
   rollouts. This is the mechanism that did the damage; it is now inert.
2. **`parse_eval_tag` warns when a folder is eval-tag-*shaped* but does not parse** (prefix
   matches, full regex fails), once per distinct name. This separates the two cases that were
   previously indistinguishable:
   - a **legacy Gen11 model folder** used as the candidate (`H8_Dmodels.diffusion.FlowMatchingODE`)
     legitimately has no K → **stays quiet** (17 such folders in the 0309 batch);
   - an **eval-tag folder the parser missed** → **loud warning naming the folder** and pointing
     at `EVAL_TAG_RE`.

   The `k_sweep` exclusion notice was correspondingly relaxed from `warning` to `info`, since
   discovery now raises the alarm precisely and by name.

### 2.5 `test_discovery_offline.py`

**+33 checks**, stdlib-only, runs in this container:

- tagged folders parse: engine / K / mpc / controller / threshold all still correct, `run_tag`
  captured, for both the Gen15 and Gen11 spellings;
- untagged folders keep `run_tag == ''` and do **not** absorb the controller or threshold;
- a tag containing the separators the sanitiser permits (`ab-1.2_x`) parses;
- `display_name` keeps the two arms of an A/B — and an arm vs. the untagged run — distinct,
  while an untagged label is byte-identical to before;
- the quiet/loud split of §2.4, including once-per-name suppression.

---

## 3. Verification

```
$ python3 Data_Analysis/DA_UAV_v1/test_discovery_offline.py
  … [eval-tag parser] 30 PASS …
ALL CHECKS PASSED
```

Replayed against the **71 real eval folders** of `temp/0309/batch_uav_20260903_204120/`:

| | before | after |
|---|---|---|
| tagged folders parsed | 0 / 6 | **6 / 6**, correct K (1, 2, 5) and controller |
| legacy model folders warning noisily | — | 0 / 17 (correctly silent) |
| tag-shaped-but-unparsable folders warned by name | 0 | all |

`python3 -m py_compile Data_Analysis/DA_UAV_v1/*.py` clean.

🔴 **Not verified end-to-end.** `aggregator.py` and `reporter.py` need pandas, which this
container does not have. The `dropna=False`, `AXIS_COLUMNS` and `K_SWEEP_KEYS` changes are
**unexercised** until a batch runs on the cluster.

---

## 4. How to confirm on the cluster

Re-run the DA over the existing 0309 tree — no eval needed, it only re-reads artifacts:

```bash
python Data_Analysis/DA_UAV_v1/main_da_batch.py --scenes pillars --engines mf
```

What must be true afterwards:

1. `candidates_detailed.csv` has a row for **all** candidates (was 48 of 71).
2. `candidate_axes.csv` shows `K = 1/2/5` and `run_tag = fix16scaled|fix16legacy` for the six.
3. `uav_k_sweep.csv` has **separate** cells per `run_tag`; the `mf`/`pillars`/`K1`/`diffuser`
   scaled cell reads `divergence_aborted ≈ 0.0` and the legacy cell `≈ 1.0`. If one cell shows
   an intermediate value with `n_candidates > 1`, `run_tag` did not reach `K_SWEEP_KEYS`.
4. No `looks like an eval tag but does NOT match EVAL_TAG_RE` warnings in `logs/loading.log`.

---

## 5. Scope

- **`DA_UAV_v1` only.** `FMPCC_UAV_EVAL_TAG` is written by `mix_uav_test/eval_mix_uav.py`, so
  the UAV tree is the only one that can carry a tagged folder today. 🟡 If the tag mechanism is
  ever copied into the visual-aligning eval, `DA_VA_v2` and `DA_Visual_Aligning` will need the
  same three changes — their eval-tag regexes are independent copies.
- **No change to any metric, reduction or ranking rule.** Purely parsing, keying and reporting.
- **Existing untagged outputs are unaffected** — `run_tag` is `''` for every pre-2026-09-02 run
  and the display names are byte-identical, so old and new batches stay comparable.

---

## 6. Files touched

| file | change |
|---|---|
| `Data_Analysis/DA_UAV_v1/config.py` | run-tag suffix on both eval-tag regexes; new `EVAL_TAG_PREFIX_RE` |
| `Data_Analysis/DA_UAV_v1/discovery.py` | parse/carry `run_tag`; label it; loud warning on a tag-shaped miss |
| `Data_Analysis/DA_UAV_v1/aggregator.py` | `run_tag` in `AXIS_COLUMNS` **and `K_SWEEP_KEYS`**; `dropna=False`; notice re-levelled |
| `Data_Analysis/DA_UAV_v1/reporter.py` | `run_tag` in `AXIS_COLUMNS` and the `run_config` axes block |
| `Data_Analysis/DA_UAV_v1/test_discovery_offline.py` | +33 checks |
| `Data_Analysis/DA_UAV_v1/README.md` | `run_tag` row in the axes table + the "not decoration" note; changelog pointer corrected to `DA_Code/DA_UAV_v1/` |
