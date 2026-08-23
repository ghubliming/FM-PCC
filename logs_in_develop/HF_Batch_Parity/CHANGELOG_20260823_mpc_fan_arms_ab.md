# CHANGELOG 2026-08-23 — `FMPCC_MPC_BATCH`: the arms-A/B MPC candidate fan is now settable

**Date:** 2026-08-23 · **Severity:** 🟡 capability gap (not a wrong number) · **Type:** knob + path guard, cross-generation
**Scope:** the three `avoiding-d3il` generations that carry an arm C — **Gen3v6** (MeanFlow), **Gen3v7** (α-Flow), **Gen12** (HardFlow→FMv3ODE)
**Follows:** [`CHANGELOG_20260820_HF_batch_parity.md`](./CHANGELOG_20260820_HF_batch_parity.md) — same confound, the other half of it
**Motivation:** *"is the MPC candidate selection dragging the performance?"* — the control that answers it (**both** arms at one candidate) could not be expressed.
**Validation:** ⚠️ in-container only — import/behaviour test of the config module and `hf_paths` (§5), `py_compile` + `bash -n` on every touched file. **No numeric result here. Run on cluster.**

---

## 1. The gap

There are **TWO independent MPC candidate fans** in an arms-A/B/C eval, and only one of them was reachable:

| arm | variants | fan comes from | settable before today |
|---|---|---|---|
| A / B | `diffuser`, `dpcc-{r,c,t}[-tightened]` | `args.batch_size` ← the **plan block** in `config/avoiding-d3il.py` | ❌ **hardcoded `4`** |
| C | `hardflow_new-{r,c,t}[-tightened]` | `hardflow.batch_size` / `HFFM_BATCH` → `resolve_hf_batch_size()` | ✅ yes |

`--batch-size` on the CLI cannot reach it either: `utils.Parser.add_extras` is commented out
(`diffuser/utils/setup.py:77`), so the flag is silently ignored.

The 2026-08-20 fix made `HFFM_BATCH` default to 4 so arm C would **match** arms A/B. That closed
the direction "arm C is secretly cheap". It left the opposite direction wide open: **you could not
turn the fan down.** Any experiment of the form *"hold every arm at a single candidate"* — which is
exactly what isolates how much of DPCC's success rate is **MPC candidate SELECTION** rather than the
projector — was unexpressible without a git edit.

That question is live. From `DA_20260820` (C136, K10, 5 seeds × 20 trials, `both-hard`):

| variant | fan | S&C |
|---|---:|---:|
| `dpcc-r-tightened` | 4 | 0.840 |
| `dpcc-c-tightened` | 4 | 0.860 |
| `dpcc-t-tightened` | 4 | **0.970** |
| `hardflow_new-*-tightened` | **1** | 0.637 |

A 0.97-vs-0.64 gap where one side picks the best of 4 plans and the other has no choice at all is
not yet attributable to the projector. Matching the fan at **1** is what separates the two.

---

## 2. What changed

### 2.1 `config/avoiding-d3il.py` — one env-read, three plan blocks

```python
_mpc_batch = int(os.environ.get('FMPCC_MPC_BATCH', 4))
```

used as `'batch_size': _mpc_batch` in **`plan_fm_v3_hardflow`**, **`plan_fm_v3_meanflow`** and
**`plan_fm_v3_alphaflow`**.

**Deliberately NOT applied** to `plan`, `plan_fm`, `plan_fm_v3`, `plan_fm_v3_ode_selectable`,
`plan_fm_v3_drifting`, `plan_fm_v3_imeanflow`, … — those have no arm C, so there is nothing to
match, and leaving their literal `4` in place keeps every historic path and command untouched.
Verified in §5.

**Default `4` ⇒ every existing command, path and result is byte-identical.**

### 2.2 The three eval drivers — mismatch warning + path tag

`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`,
`FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py`,
`FM_v3_hardflow_test/eval_FM_v3_hardflow.py` each now read the same variable **before the first
`Parser().parse_args()`** (that call is what imports `config/<exp>.py`, so a later read would be too
late for the folder name) and:

1. **Warn on a mismatch** — `FMPCC_MPC_BATCH != HFFM_BATCH` prints a `B4_PARITY` warning naming both
   numbers. It is a warning, not an abort: an intentional 1-vs-4 sweep is a legitimate experiment.
2. **Reject `< 1`.**
3. **Auto-tag the results path** (§3).
4. **Print the resolved pair**, e.g. `[ eval ] mpc fan: arms A/B=1, arm C=1`.

Gen12 additionally records `mpc_batch_arms_ab` and `run_msg` in its `provenance.write(...)` block —
it builds `savepath` by hand and so gets no `args.json`, which is why those two knobs had to be
added explicitly.

### 2.3 `FM_v3_hardflow_test/hf_paths.py` — `sanitize_msg()` + `resolve_run_msg()`

`eval_name()` gained an optional `run_msg` (default `None` ⇒ auto-resolve from the environment), so
**the eval and the aggregator stay in sync without either of them changing a call site** — the whole
point of that module. `sanitize_msg` is byte-identical to `config/avoiding-d3il.py::_sanitize_msg`.

### 2.4 The three sbatch entrypoints

`Slurm_Codes/sbatch/{MeanFlow/eval_meanflow_hardflow.sh, AlphaFlow/eval_alphaflow_hardflow.sh, hardflow_fmv3/eval_fmv3_hardflow_job.sh}`:

* `export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"` next to the existing `HFFM_*` exports, documented
  in the same knob list, and echoed alongside `HFFM_BATCH` so **both fans are visible in the job log**.
* `"$@"` is now forwarded to the `python` call, so `./submit.sh <script> --config <yaml>` reaches the
  eval. Previously the drivers accepted `--config` but no entrypoint could pass it.

---

## 3. 🔴 The path-collision guard (the part that is easy to get wrong)

`batch_size` is **not** a results-folder token in any generation. Without a guard, an `mpc4` and an
`mpc1` run at the same `K/A/T` write to the **same directory** and clobber each other — the exact
hazard `args_to_watch_fmv3_hf_plan` exists to prevent, and the same one U10 hit with `replan_steps`.

Promoting it to a real token would rename **every historic path**. So, following U10's precedent, a
non-default fan **auto-tags itself** through the existing custom-message slot:

| gen | mechanism | result at `FMPCC_MPC_BATCH=1` |
|---|---|---|
| Gen3v6 | `FMPCC_RUN_MSG=mpc1` → `watch_plan` | `…_T0.5_A0.5_B1_D…_msgmpc1/` |
| Gen3v7 | same | `…_T0.5_A0.5_B1_D…_msgmpc1/` |
| Gen12 | `hf_paths.resolve_run_msg()` → `eval_name` | `K2_thres0.5_mpc1_n2_msgmpc1/` |

Gen3v6 **composes** with the replan tag: `MF_REPLAN_STEPS=8` + `FMPCC_MPC_BATCH=1` ⇒ `msgr8-mpc1`.
An explicit `FMPCC_RUN_MSG` always wins over both.

⚠️ **Gen12 specifically.** Its `mpc<N>` token has *always* been fed `hf_batch_size`, i.e. it describes
**arm C only**. The historic `B1` runs therefore already sit in `…_mpc1_n…/` directories while their
DPCC arms ran at 4. Without the `_msgmpc1` suffix a genuine all-arms-at-1 run would land on top of
them — two different controllers in one folder. This is why Gen12 needed the extra `hf_paths` work
rather than just the config line.

---

## 4. How to run the study

Both fans at one candidate, matched K=2, avoiding:

```bash
# MeanFlow flagship (UNet@32, H8)
FMPCC_MPC_BATCH=1 HFFM_BATCH=1 HFFM_FLOW_STEPS=2 HFFM_ACT_THRESHOLD=0.5 DPCC_THRESHOLD=0.5 \
MF_BACKBONE=unet MF_HORIZON=8 \
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh

# FMv3ODE (Gen12) — note HFFM_ACT_THRESHOLD is MANDATORY here: that yaml ships 1.0,
# the MeanFlow yaml ships 0.5, and an unmatched arm C makes the two runs incomparable.
FMPCC_MPC_BATCH=1 HFFM_BATCH=1 HFFM_FLOW_STEPS=2 HFFM_ACT_THRESHOLD=0.5 DPCC_THRESHOLD=0.5 \
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```

The `mpc4` control is the same command with `FMPCC_MPC_BATCH=4 HFFM_BATCH=4` (or nothing at all).

### 4.1 Reading the results

At a fan of **1 the selection RULES collapse**: `dpcc-r`/`-c`/`-t` all execute index 0
(`sampling/policies.py` — `which_trajectory = 0` in every branch) and so do
`hardflow_new-r`/`-c`/`-t` (`hardflow_projection.py::_select`). Expect those triples to come back
**numerically identical** — that is the correctness check, not a bug. Anyone wanting to save wall
clock can list one of each instead of the trio; leaving all six in costs time but nothing else.

At **K=2, threshold 0.5** the two arms are unusually well matched on solve count:
`snapping_start_idx = int((1−0.5)·2) = 1` ⇒ DPCC active at `loop_idx ∈ {1}`, and HardFlow's
`k >= int((1−0.5)·2) = 1` ⇒ `k ∈ {1}`. **One NLP solve per candidate per plan in both arms.**
The residual asymmetry is NFE: arm C burns 3 per plan against arms A/B's 2.

**Expect DPCC's S&C to FALL at mpc=1.** That is the measurement, not a regression.

---

## 5. Validation performed (in-container, no GPU/MuJoCo)

| check | result |
|---|---|
| `py_compile` on all 5 touched `.py` + `load_results_FM_v3_hardflow.py` | ✅ |
| `bash -n` on all 3 sbatch scripts | ✅ |
| config module import, `FMPCC_MPC_BATCH` unset / `4` / `1` / `8` | ✅ the 3 arm-C plan blocks track it; `plan`, `plan_fm_v3` stay pinned at 4 |
| `hf_paths.eval_name` unset / `=4` | ✅ `K2_thres0.5_mpc1_n2` — byte-identical to the historic name |
| `hf_paths.eval_name` `=1` | ✅ `K2_thres0.5_mpc1_n2_msgmpc1` |
| explicit `FMPCC_RUN_MSG` wins over the auto-tag | ✅ `…_msg20trials` |
| `config/avoiding-d3il.py` CRLF line endings preserved (1690/1690) | ✅ |

**Not validated in-container:** that a DPCC `Projector` run at `batch_size == 1` behaves — the code
paths are fan-generic (`for i in range(batch_size)`, `np.zeros(batch_size)`, `min(batch_size, 4)` in
the plot loop) and arm C has shipped at 1 for months, but **arms A/B have never actually run at 1.**
First cluster job is the real test; watch for an `IndexError` in the plotting block the way `fix_7`
did on the arm-C side.

---

## 6. ⚠️ Sync requirement

`FMPCC_MPC_BATCH` does **nothing** until this commit is on the cluster. A job submitted with
`FMPCC_MPC_BATCH=1` against an un-synced checkout runs arms A/B at **4** and says nothing about it —
the variable is simply unread. Confirm from the job log, which now prints:

```
[ hardflow ] HFFM_BATCH=1 (arm C)  FMPCC_MPC_BATCH=1 (arms A/B)  ...
[ eval ] mpc fan: arms A/B=1, arm C=1
```

and from the results path carrying `_msgmpc1`. **No `_msgmpc1` in the path ⇒ arms A/B ran at 4.**

---

## 7. Files touched

```
config/avoiding-d3il.py                                    _mpc_batch + 3 plan blocks
FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py      fan read, warning, composed auto-tag
FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py    fan read, warning, auto-tag
FM_v3_hardflow_test/eval_FM_v3_hardflow.py                 fan read, warning, auto-tag, provenance
FM_v3_hardflow_test/hf_paths.py                            sanitize_msg, resolve_run_msg, eval_name(run_msg=)
Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh      export + echo + "$@"
Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh    export + echo + "$@"
Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh export + echo + "$@"
```

No YAML was changed — the `hardflow.batch_size: 4` defaults from `B4_PARITY` stand, and the arm-C
fan is overridden per job with `HFFM_BATCH`.
