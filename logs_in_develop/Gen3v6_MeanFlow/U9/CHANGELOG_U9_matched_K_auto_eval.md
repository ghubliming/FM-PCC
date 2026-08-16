# CHANGELOG — U9: matched-K auto-eval for Gen3v6 (MeanFlow) — parity with Gen3v7 (AlphaFlow)

**Date:** 2026-08-09 · **Generation:** Gen3v6 (MeanFlow) · **Epoch:** U9
**Retrieval flag:** `🔵 U9` — find every site with:

```bash
grep -rn "U9 MATCHED-K\|🔵 U9" --include='*.py' --include='*.sh' . | grep -v Archived_Codes
```

**Status:** code only — **not yet run on the cluster.** Nothing here has executed; the mechanism
is a verbatim port of the Gen3v7 path that has been running since Gen3v7 init.

---

## 1. One-line summary

Gen3v7 (AlphaFlow) has had the NFE-budget grid **built into its eval sbatch since day one** —
one job sweeps K ∈ {1, 2, 5, 10} automatically. Gen3v6 (MeanFlow) never got it: every MeanFlow
eval ran the single `HFFM_FLOW_STEPS` default (K=2), and a K sweep meant **four hand-typed
resubmits**. This update ports the AlphaFlow mechanism to MeanFlow, end to end (eval + aggregation,
DPCC-only entry point + unified HardFlow entry point).

## 2. Why this was a real hazard, not a convenience gap

⚠️ **MATCHED BUDGET OR NOTHING** (PLAN §7 / fix_7.3 §9). The entire Gen13 claim died because one
hard-coded `k_steps=10` made the decisive control unrunnable, and the confound survived four rounds
of analysis.

The Gen3v6 fix_4 post-fix sweep is the shape of the problem:

```bash
# CHANGELOG_Gen3v6_fix_4_hardflow_init_noise.md:188-191 — four separate submits, by hand
HFFM_FLOW_STEPS=1  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh …eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=2  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh …eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=5  HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh …eval_meanflow_hardflow.sh
HFFM_FLOW_STEPS=20 HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh …eval_meanflow_hardflow.sh
```

One forgotten line and the matched-K control silently does not exist. Compare Gen3v7, where the
same sweep is `./submit.sh …eval_alphaflow.sh` and the loop is in the script
(`Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh:90-98`).

Note the Gen3v6 grid is now `{1, 2, 5, 10}`, matching the Gen3v7 grid and PLAN §7 — **not** the
`{1, 2, 5, 20}` fix_4 used. Comparisons against Gen3v7/Gen3v4 need the same rungs.

## 3. Changes

### 3.1 `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`

- **NEW `--flow-steps K` CLI flag** (`:49`). K becomes a first-class knob, so the grid can be a
  loop in the sbatch instead of an env edit someone has to remember.
- **Mechanism (identical to AlphaFlow):** patch the *config module's* plan block before any
  `Parser` reads it. `utils.Parser.read_config` does `importlib.import_module(args.config)` and
  copies `base[experiment]` key by key, and Python caches modules — so this is the intended data
  path, not a monkey-patch. `exp_name`, `savepath` and the diffusion kwargs all follow.
- **Both K knobs are patched together** (`flow_steps_v3` for arms A/B, `flow_steps` for arm C's
  HardFlow Euler K). Patching only one would move arms A/B and leave every arm-B-vs-arm-C table
  comparing different NFE budgets — the exact class of confound this update exists to prevent.
- **🔴 Reordering (required, not cosmetic):** the `FIX_9_CFG_PROVENANCE` block that publishes
  `FMPCC_PROJ_CFG` / `FMPCC_DPCC_THRESHOLD` / `HFFM_ACT_THRESHOLD` / `HFFM_BATCH` now runs
  **before** the `exps`/`seeds` block and the `--flow-steps` patch (`:65-97`). Reason: the patch
  calls `importlib.import_module('config.' + exp)`, which executes `config/avoiding-d3il.py`'s
  module-level env reads and caches the result. Publishing after that import would be too late for
  the `T`/`A`/`B` folder tokens on any `--flow-steps` run — i.e. it would reintroduce Fix_9's
  "path that lies about what ran" bug on exactly the runs this update adds. Gen3v7 already had the
  block in this order for the same reason.

### 3.2 `FM_v3_meanflow_test/load_results_flow_matching_v3_meanflow.py`

- **NEW `--flow-steps K`** mirroring the eval script. Results live in a per-K directory
  (`flow_steps_v3` is watched as `K` in `args_to_watch_fmv3_hf_plan`), so aggregating without the
  matching K would silently report the K=2 default no matter which budget was run.
- **🔴 Also publishes `FMPCC_PROJ_CFG`.** This script loads `config/meanflow_projection_eval.yaml`
  (Gen3v6-dedicated) but never told the config module, so the `T` token it searched under was
  derived from the **shared** `config/projection_eval.yaml` — a file Gen3v6 never opens. Both
  currently hold `0.5`, so this is a no-op today and a `FileNotFoundError` the day they diverge.
  Same defect and same one-line fix as Fix_9 §2.1, in the one script Fix_9 did not cover.

### 3.3 `Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh`

K grid loop added, verbatim shape of `eval_alphaflow.sh`. Override with
`MF_FLOW_STEPS="2" ./Slurm_Codes/submit.sh …` (Gen3v7 uses `AF_FLOW_STEPS`; the prefix is per-gen
so a chained submit can drive the two independently).

### 3.4 `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh`

Same loop, **with a backward-compatibility branch** — this is the entry point fix_4 and U3 actually
used, and its documented commands must keep working:

- `HFFM_FLOW_STEPS` set → **pins the job to that single K** (every command in the fix_4 changelog
  behaves exactly as before).
- `HFFM_FLOW_STEPS` unset → the whole `{1,2,5,10}` grid runs in one job. Override with
  `MF_FLOW_STEPS`.

### 3.5 `Slurm_Codes/sbatch/MeanFlow/load_results_meanflow.sh`

Same K grid loop around the aggregation call, matching `load_results_alphaflow.sh`.

## 4. Why per-K results cannot collide

`flow_steps_v3` is the `K` token in `args_to_watch_fmv3_hf_plan` (`config/avoiding-d3il.py:113`),
so `exp_name` → `savepath` carries `_K{K}_` and each budget writes its own results directory. This
is the property that makes the in-job loop safe; it is also why K had to flow through the **config**
rather than a post-load patch on the model.

## 5. Verification — RUN ON CLUSTER

Nothing below has been executed (this container has no Python env).

1. **Cheap smoke test — one K, one seed:**
   ```bash
   MF_FLOW_STEPS="2" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh
   ```
   Expect in the log: `[ eval ] NFE budgets to evaluate: 2` then
   `[ eval ] Overriding flow_steps_v3 / flow_steps (K) from config to: 2`, and results under a
   path containing `_K2_`.

2. **Backward compatibility of the HardFlow entry point** — must be byte-identical in behaviour to
   the fix_4 runs:
   ```bash
   HFFM_FLOW_STEPS=2 HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   ```
   Expect `[ eval ] HFFM_FLOW_STEPS is set -> single-K run: 2` and the same `_K2_A0.5_B4_` path
   fix_4 produced.

3. **The actual feature — full grid in one job:**
   ```bash
   HFFM_BATCH=4 HFFM_ACT_THRESHOLD=0.5 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   ```
   Expect four `[ eval ] K = …` banners and **four distinct** `_K1_ _K2_ _K5_ _K10_` result dirs.
   ⏱️ This is 4× the wall time of a single-K job — the `--time=24:00:00` cap already in the script
   may not be enough for a 4-seed × 13-variant sweep. Check the fix_4 single-K durations first and
   split the grid with `MF_FLOW_STEPS` if needed.

4. **Aggregation:**
   ```bash
   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/load_results_meanflow.sh
   ```
   Expect four `[ load_results ] K = …` blocks with **different** numbers. Identical numbers across
   K means the patch is not reaching the path — that is the failure signature to watch for.

## 6. Not done (deliberately)

- **`gates_hardflow_meanflow.py` is still not called** by `eval_meanflow_hardflow.sh`.
  `RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md:349` records that the gates have **never executed** in
  Gen3v6 and recommends adding the call, as Gen3v7 does (`eval_alphaflow_hardflow.sh` §5). It is one
  line, `set -e` is already active — but it can abort jobs, so it is a behaviour change outside this
  update's scope. **Recommended as U9.1 or fix_10.**
- **`HFFM_BATCH` default stays `1`** in `eval_meanflow_hardflow.sh` (Gen3v7 moved its default to
  `4` to close the fix_3 confound). Changing it would silently change what every future MeanFlow
  headline number means. Left as an explicit decision for the user.
- **`plt.close(fig_all)` (fix_7) is retained** in the MeanFlow eval. Gen3v7 lacks it; that is an
  AlphaFlow gap, not something to "sync" backwards.

## 7. Files touched

| File | Change |
|---|---|
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | `--flow-steps` + config-module patch; FIX_9 block reordered above it |
| `FM_v3_meanflow_test/load_results_flow_matching_v3_meanflow.py` | `--flow-steps` + `FMPCC_PROJ_CFG` publication |
| `Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh` | K grid loop (`MF_FLOW_STEPS`) |
| `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` | K grid loop + `HFFM_FLOW_STEPS` single-K pin |
| `Slurm_Codes/sbatch/MeanFlow/load_results_meanflow.sh` | K grid loop |

Reference implementation throughout: `FM_v3_alphaflow_test/` + `Slurm_Codes/sbatch/AlphaFlow/`.
