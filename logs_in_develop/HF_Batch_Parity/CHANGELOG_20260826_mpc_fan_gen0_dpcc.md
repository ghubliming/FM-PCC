# CHANGELOG 2026-08-26 — `FMPCC_MPC_BATCH` reaches the Gen0 DPCC baseline

**Date:** 2026-08-26 · **Severity:** 🟡 capability gap (not a wrong number) · **Type:** knob + path guard
**Scope:** Gen0 DPCC baseline only — `config/avoiding-d3il.py::plan`, `scripts/eval.py`, `Slurm_Codes/sbatch/eval_dpcc_job.sh`
**Follows:** [`CHANGELOG_20260823_mpc_fan_arms_ab.md`](./CHANGELOG_20260823_mpc_fan_arms_ab.md) — same knob, the block that was deliberately left out
**Validation:** ⚠️ in-container only — stub-import behaviour test of the config module (§4), `py_compile` + `bash -n` on every touched file. **No numeric result here. Run on cluster.**

---

## 1. The gap

The 2026-08-23 change wired `FMPCC_MPC_BATCH` into the three plan blocks that carry an arm C
(`plan_fm_v3_{hardflow,meanflow,alphaflow}`) and **deliberately excluded `plan`** — the Gen0 DPCC
baseline — to keep every historic path untouched.

That left the DA target unreachable. The paper baseline is pinned to **DPCC K20 / aw10 /
`models.GaussianDiffusion`**, and it has only ever run at a candidate fan of **4**. So the question
the whole `FMPCC_MPC_BATCH` line of work exists to answer —

> how much of DPCC's success rate is MPC candidate **SELECTION** rather than the **projector**?

— could be asked of MeanFlow, α-Flow and FMv3ODE, but **not of the baseline they are all measured
against**. Jobs 24991/24992 (`DA_20260824_mpc1_parity_MF_vs_FM.md`) answered it for the `dpcc-*`
arms *wrapped around a flow generator*; the diffusion baseline itself was never in scope.

## 2. What changed

### 2.1 `config/avoiding-d3il.py` — one line in the `plan` block

`'batch_size': 4` → `'batch_size': _mpc_batch`, reusing the module-level
`_mpc_batch = int(os.environ.get('FMPCC_MPC_BATCH', 4))` already defined at line 71.

**Still deliberately NOT applied** to `plan_fm`, `plan_fm_unet_v2`, `plan_fm_v2`, `plan_fm_v3`,
`plan_fm_v3_ode_selectable`, `plan_fm_v3_drifting`, `plan_fm_v3_imeanflow`, `plan_fm_hp_tune` —
asserted in §4.

**Default `4` ⇒ every existing DPCC command, path and result is byte-identical.**

### 2.2 `scripts/eval.py` — read + reject + auto-tag, before the first `parse_args`

Placed after the YAML load and **before `for exp in exps:`** (guard at lines 74–88, first
`Parser().parse_args(experiment='plan', …)` at line 116). That call is what imports
`config/<exp>.py`, so a later read would be too late for **both** the fan and the folder name.

Three deliberate differences from the arm-A/B/C drivers:

1. **No `B4_PARITY` mismatch warning.** Gen0 has no arm C — there is no second fan to disagree
   with, so `HFFM_BATCH` is not read here at all.
2. **Rejects `< 1`.**
3. **Prints `[ eval ] mpc fan: DPCC baseline arm=<N>`.**

### 2.3 `Slurm_Codes/sbatch/eval_dpcc_job.sh`

* `export FMPCC_MPC_BATCH="${FMPCC_MPC_BATCH:-4}"`, documented in the same style as the three
  hardflow entrypoints, echoed alongside the resolved `FMPCC_RUN_MSG` so the fan is visible in the
  first lines of the job log.
* `"$@"` is now forwarded to `python scripts/eval.py`, matching what `83471f8d` did for the other
  three entrypoints (`scripts/eval.py` already accepts `--seed` / `--aggregate-only` and passes the
  remainder through to `Parser`).

## 3. 🔴 The path-collision guard

`batch_size` is **not** a results-folder token in the `plan` block either, so an `mpc4` and an
`mpc1` run at the same `H/K/T` would write to the **same** directory and clobber each other.

Unlike Gen12, no new machinery was needed: the `plan` block **already** carries
`'custom_msg': custom_msg` and its `exp_name` already ends in `_msg_suffix(args)`. So the same
guard the other generations use applies unchanged — a non-default fan auto-tags itself:

| `FMPCC_MPC_BATCH` | `FMPCC_RUN_MSG` | resulting leaf |
|---:|---|---|
| 4 (default) | unset | `H8_K20_T0.5_D…` *(unchanged)* |
| 1 | unset → auto `mpc1` | `H8_K20_T0.5_D…_msgmpc1` |
| 1 | explicit `20trials-mpc1` | `H8_K20_T0.5_D…_msg20trials-mpc1` |

An explicit `FMPCC_RUN_MSG` always wins over the auto-tag.

⚠️ The existing 5-seed × 20-trial mpc4 baseline sits at `…_msg20trials` (C11), so `_msgmpc1` alone
would not collide — but it would also not record the trial count. Set the run message explicitly
when the trial count is non-default.

## 4. Validation (in-container)

`py_compile` on `config/avoiding-d3il.py` + `scripts/eval.py`, `bash -n` on `eval_dpcc_job.sh`.

Behaviour test — the config module loaded five times under different environments with
`diffuser.utils.watch` stubbed:

| env | `plan.batch_size` | `exp_name` leaf suffix |
|---|---:|---|
| *(unset)* | **4** | *(none)* |
| `FMPCC_MPC_BATCH=4` | **4** | *(none)* |
| `=1`, `RUN_MSG=mpc1` | **1** | `_msgmpc1` |
| `=1`, `RUN_MSG=20trials-mpc1` | **1** | `_msg20trials-mpc1` |
| `=2`, `RUN_MSG=mpc2` | **2** | `_msgmpc2` |

Non-regression, same load at `FMPCC_MPC_BATCH=1`:

* untouched at 4 — `plan_fm`, `plan_fm_v3`, `plan_fm_v3_imeanflow`, `plan_fm_v3_ode_selectable`,
  `plan_fm_v3_drifting`, `plan_fm_hp_tune`
* wired, follow the knob — `plan_fm_v3_hardflow`, `plan_fm_v3_meanflow`, `plan_fm_v3_alphaflow`

Fan plumbing re-checked in `scripts/eval.py`: `args.batch_size` reaches the sampler at line 349
(`policy(…, batch_size=args.batch_size, …)`) with no per-variant override, and the only other use
is the plot loop's `range(min(args.batch_size, 4))` at line 411, which degrades safely to
`range(1)`.

## 5. Eval shape — NOT touched here, set it at run time

**No eval YAML was changed by this changelog.** `config/projection_eval.yaml` and
`config/alphaflow_projection_eval.yaml` are run-time knobs, owned by whoever launches the job.
For the record, they currently sit at:

| file | seeds | n_trials |
|---|---|---:|
| `config/projection_eval.yaml` (Gen0 DPCC) | `[6,7,8,9,10]` | **2** |
| `config/alphaflow_projection_eval.yaml` (Gen3v7) | `[6]` | **2** |

⚠️ `n_trials` is **not** a folder token, so a 2-trial and a 20-trial run at the same `H/K/T`
collide exactly the way the fan does — which is why the 20-trial baselines on disk carry a
hand-set `_msg20trials` tag. An mpc1 run tagged `_msg20trials-mpc1` is safe from both.

**What the shape buys.** Inferred from metric granularity in
`temp/2508/batch_avoiding_combined_20260825_143212`: S&C takes only `{0, 0.5, 1.0}` in a 2-trial
run and 21 distinct values in a 20-trial run. The 2026-08-23 B=1 runs (C147, C64) were **2 trials,
seed 6** — precisely why `DA_20260824` could not quote a single S&C difference. A 2-trial mpc1 run
compared against a 20-trial mpc4 baseline would repeat that.

## 6. Reading the result

⚠️ At a fan of 1 the DPCC selection rules **collapse**: `dpcc-r`, `dpcc-c` and `dpcc-t` all execute
index 0 (`sampling/policies.py`, `which_trajectory = 0` in every branch). The trio is redundant
compute at `mpc1`, not three arms — expect three identical rows, and treat a difference between
them as a bug, not a finding.

This run moves the **DA target**. If B=1 changes the baseline, the comparison gains a second
reference point rather than replacing the first — which of the two the paper quotes is a decision,
not an outcome of this run.
