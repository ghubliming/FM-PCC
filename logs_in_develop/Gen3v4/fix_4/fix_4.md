# Gen3v4 Fix 4: iMF CLI Parser Compatibility Repair

**Date**: 2026-05-13
**Scope**: `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`, `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`, `diffuser/utils/setup.py`
**Status**: ✅ Fixed and recorded

---

## Summary

This fix restores the standard FM-PCC parser contract for the iMeanFlow entrypoints.

The run failed because the iMF train script passed top-level CLI flags such as `--seeds`, `--use-wandb`, and `--wandb-project` into the shared `Parser`, which only accepts the config flags it defines internally. The parser instance also lacked the FM-PCC `dataset` / `config` metadata that `parse_args()` expects. That combination caused `argparse` to reject the run before the iMF config block could load.

---

## Failure Mode

Observed error pattern:

```text
usage: [] [-h] [--config CONFIG] [--seed SEED]
[]: error: unrecognized arguments: --seeds 6 7 8 9 10 --use-wandb --wandb-project FMPCC-iMF
```

Root cause:
- The iMF train path was not isolating its own seed/W&B CLI arguments before invoking the shared FM-PCC `Parser`.
- The parser instance was created without the required `dataset` / `config` metadata, so `parse_args()` could not resolve the FM-PCC config module.
- The shared parser saw the outer command-line flags and aborted.
- A previous compatibility assumption around `exe_name` was also present, but the real failure was the argv handoff pattern plus missing parser metadata.

---

## Fix Applied

### 1. Parser compatibility retained

`diffuser/utils/setup.py` now accepts `exe_name` as a no-op compatibility field:

- Prevents legacy call sites from crashing.
- Keeps the parser interface tolerant while preserving existing behavior.

### 2. Train entrypoint normalized

`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` now:

- Saves the original `sys.argv`
- Replaces it with the remaining FM-PCC config arguments only
- Uses a local `Parser(utils.Parser)` subclass with `dataset='avoiding-d3il'` and `config='config.avoiding-d3il'`
- Calls `Parser(exe_name='train')` and `parse_args(experiment='flow_matching_v3_imeanflow', seed=seed)`
- Builds dataset, model, diffusion, and trainer explicitly via `utils.Config(...)`
- Replaces the invalid `args.trainer` access with the standard FM-PCC instantiation flow
- Restores `sys.argv` after the loop

This matches the established FM-PCC pattern and prevents top-level CLI options from leaking into `argparse`.

### 3. Eval entrypoint normalized

`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` was aligned to the same parser isolation pattern and the stale duplicate `main()` block was removed.

It also uses the same local parser subclass and loads the `plan` config block for evaluation.

### 4. Runtime device fallback added

`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` now derives a safe runtime `device` locally instead of assuming the config provides one.

- Uses `args.device` when available
- Falls back to `'cuda'` if CUDA is available
- Otherwise falls back to `'cpu'`

This prevents the training loop from crashing on `args.device` and keeps model/trainer construction consistent with the runtime environment.

### 5. Namespace alias added for iMF imports

FM-PCC's `diffuser.utils.config.import_class()` always resolves class strings under the `diffuser.*` namespace. The real iMF implementation lives at the repository root (`flow_matcher_v3_imeanflow/`), so the config loader could not import it directly.

To fix that, a compatibility namespace was added:

- `diffuser/flow_matcher_v3_imeanflow/__init__.py`
- `diffuser/flow_matcher_v3_imeanflow/models/__init__.py`
- `diffuser/flow_matcher_v3_imeanflow/models/imf_engine.py`
- `diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py`

These modules forward the iMF classes from the real root package into the namespace expected by the FM-PCC config loader. This resolves the `ModuleNotFoundError: No module named 'diffuser.flow_matcher_v3_imeanflow'` crash without changing the existing config loader behavior.

### 6. iMF wrapper API aligned with the real U-Net

The trajectory model wrapper was updated to match the real `Flow_matcher_U_Net_v2` constructor and call signature.

- The wrapper now passes `horizon`, `transition_dim`, and `cond_dim` instead of the nonexistent `input_dim` / `output_dim` arguments.
- The `forward()` call now uses the correct `self.u_net(x, cond, t)` ordering.
- The wrapper now exposes the `forward_train()` and `sample()` methods expected by `iMFDiffusion`.

This removes the next API-layer failure in the iMF stack and keeps the wrapper consistent with both the engine and diffusion adapter.

---

## Why This Fix Makes Sense

Yes, this fix is structurally correct.

It solves the problem at the interface boundary instead of masking the symptom:

- The parser now accepts the legacy compatibility keyword.
- The entrypoints now separate top-level orchestration flags from FM-PCC config flags.
- The code path now follows the same seed-driven pattern used elsewhere in the repository.

That means the failure is handled at the right layer:
- **CLI orchestration** stays outside the shared parser.
- **Config parsing** only receives the arguments it knows how to interpret.
- **Training/eval behavior** remains consistent with the rest of FM-PCC.

---

## Verification Notes

- The parser crash path is removed.
- The iMF train loop now proceeds through seed parsing and config instantiation.
- The iMF train loop now proceeds through seed parsing, config instantiation, and runtime device setup.
- The iMF config loader can now import the model and diffusion classes through the expected `diffuser.*` namespace.
- The iMF wrapper API now matches the real U-Net and the diffusion adapter expectations.
- Remaining tool warnings about `wandb`, `torch`, and `numpy` are environment import-resolution warnings, not parser errors.

---

## Outcome

The iMF entrypoints are now consistent with FM-PCC’s standard configuration flow, and the earlier silent-dead run condition is resolved.

**Conclusion**: This is a valid fix, not a workaround.
