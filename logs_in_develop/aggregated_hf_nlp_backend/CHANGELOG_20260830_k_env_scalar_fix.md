# CHANGELOG 2026-08-30 — `HFFM_FLOW_STEPS` scalar/list collision (K_ENV_SCALAR)

**Trigger:** job **25161** (`temp/3008/2026-08-28/15_18_34_eval_fmv3_hardflow_job_25161.log`),
the first submission of the IPOPT-vs-SLSQP A/B. Started 2026-08-29 22:17:47, **dead 5 s later**.
No data written.

## 1. The bug

`HFFM_FLOW_STEPS` had **two readers that disagreed on its type**.

- `Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh` treats it as a space-separated
  **list**, loops it, and hands each K down the CLI as `--flow-steps "$K"`.
- `config/avoiding-d3il.py:1441-1442` reads the **same variable name** as a scalar,
  `int(os.environ.get('HFFM_FLOW_STEPS', 2))`, at **module-import time** — i.e. before argparse
  can apply `--flow-steps`.

The sbatch never removed the variable from the environment, so the child saw the whole list:

```
File "config/avoiding-d3il.py", line 1441, in <module>
    'flow_steps_v3': int(os.environ.get('HFFM_FLOW_STEPS', 2)),
ValueError: invalid literal for int() with base 10: '10 20'
```

A **single** value reads identically both ways, which is why every single-K job ever submitted
worked. The multi-K sweep documented in the script's own header (`HFFM_FLOW_STEPS="2 5 10"`) had
**never actually been exercised**. This is latent in the K loop and independent of the SolverSwap
work — it fires on any multi-K submission, with or without `HFFM_SOLVERS`.

## 2. Fix

Both files in `Slurm_Codes/sbatch/hardflow_fmv3/`, tagged `K_ENV_SCALAR` in-line:

**`eval_fmv3_hardflow_job.sh`**
- The K switch snapshots the list into `HFFM_K_LIST` and **`unset HFFM_FLOW_STEPS`**, so no child
  can ever import a config with a non-integer value in the environment.
- Each entry is validated as a positive integer up front (`case "$K" in ''|*[!0-9]*)` → `exit 2`).
  A typo'd grid now fails in 1 s instead of after the first K has burned hours.
- `run_eval` builds an `env` prefix: `env "HFFM_FLOW_STEPS=$K"` when a K override is active,
  `env -u HFFM_FLOW_STEPS` when it is not. The empty-string case is stripped rather than passed
  through, because `int('')` is also a `ValueError`.
- Net effect: the env value and the `--flow-steps` argument now **always** agree, so the
  config-derived `exp_name` (`_K{K}_`) always names the K that was actually evaluated.

**`load_results_hardflow_fmv3.sh`** — same defect, same fix. `load_results_FM_v3_hardflow.py`
imports the same config, so a multi-K *aggregation* would have died the same way. Fixed together
because it is the immediate next step after the eval.

## 3. Not changed

- `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` copies `HFFM_FLOW_STEPS` into
  `FLOW_STEPS_GRID` and loops it without unsetting, so it carries the **same latent defect**. Its
  own comments call that path a "single-K run", so it is documented as scalar-only and does not
  crash in practice. Left alone pending a decision on whether to mirror.
- `config/avoiding-d3il.py` untouched — the scalar `int()` read is correct behaviour for the
  config; the sbatch was the side passing the wrong type.

## 4. Verification (local, no cluster)

- `bash -n` on the LF form of both scripts: **syntax OK** (the working-tree copies are CRLF, which
  `bash -n` rejects spuriously; `.gitattributes` `* text=auto` means the committed/cluster blob
  is LF). Line endings preserved by the patch.
- The real patched block was extracted from the file and run against a stubbed `python` that
  echoes its environment. Five cases, all correct:

| case | result |
|---|---|
| `HFFM_FLOW_STEPS="10 20"` + `HFFM_SOLVERS="ipopt slsqp"` (the 25161 case) | 4 invocations; each sees scalar `10`/`20` matching its own `--flow-steps` |
| single `K=20`, no solver sweep | `HFFM_FLOW_STEPS=20`, `--flow-steps 20` |
| no K set (plan-block path) | `HFFM_FLOW_STEPS` **unset**, no `--flow-steps` |
| stale multi-K exported, plan-block path | `HFFM_FLOW_STEPS` **unset** — stale value cannot leak |
| `HFFM_FLOW_STEPS="10 2O"` (typo) | fails fast, `exit 2`, names the offending entry |

**Still to run on the cluster:** the A/B itself. `_solve_slsqp` has still never executed.
