# CHANGELOG — Gen3v6 fix_1: `diffuser.*` namespace shim (training crash at step 0)

**Date:** 2026-07-22 · **Type:** bugfix changelog · **Job:** Slurm `23739` (`mf_train`, i6-gpu-1), died 4 s in
**Follows:** [`../init/CHANGELOG_Gen3v6_coding1.md`](../init/CHANGELOG_Gen3v6_coding1.md)
**Nothing committed.**

---

## 1. Symptom

First real submission of `Slurm_Codes/sbatch/MeanFlow/meanflow_pipeline.sh`. The dataset built
fine (`observations: (96, 150, 4)`, `actions: (96, 150, 2)`), then:

```
File ".../FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py", line 225, in <module>
    model_config = utils.Config(
File ".../diffuser/utils/config.py", line 15, in import_class
    module = importlib.import_module(f'{repo_name}.{module_name}')
ModuleNotFoundError: No module named 'diffuser.flow_matcher_v3_meanflow'
```

Chained `mf_eval` was cancelled by `afterok`, as designed. No checkpoint or partial state was
written beyond the (empty) savepath directory and its config snapshot.

## 2. Root cause

`diffuser/utils/config.py:import_class()` **hard-prefixes every config class string with the repo
package name**:

```python
repo_name = __name__.split('.')[0]                      # -> 'diffuser'
module = importlib.import_module(f'{repo_name}.{module_name}')
```

So the config values

```python
'model':     'flow_matcher_v3_meanflow.models.MeanFlowEngine',
'diffusion': 'flow_matcher_v3_meanflow.models.MeanFlowODE',
```

are resolved as `diffuser.flow_matcher_v3_meanflow.models.*`, which did not exist.

Gen3v4 solved this with a **shim namespace package** at `diffuser/flow_matcher_v3_imeanflow/`
that simply re-exports the real top-level package:

```
diffuser/flow_matcher_v3_imeanflow/__init__.py
diffuser/flow_matcher_v3_imeanflow/models/__init__.py            # from .imf_engine import iMeanFlowEngine ...
diffuser/flow_matcher_v3_imeanflow/models/imf_engine.py          # from flow_matcher_v3_imeanflow.models.imf_engine import iMeanFlowEngine
diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py       # from flow_matcher_v3_imeanflow.models.imf_diffusion import iMeanFlowODE
```

**Why it was missed in the coding pass:** the shim lives under `diffuser/`, which PLAN §1 lists as
`Untouched (hard rule)`. The copy-modify sweep was scoped to the two sibling folders plus the
config and sbatch dirs, so a **required** file outside those four locations was never considered.
Nothing in the plan, the folder-pair convention, or the Gen3v4 model package points at it — the
dependency is invisible until `diffuser.utils.Config` is actually called with a string class.

## 3. Fix — files created (4, all new; nothing existing modified)

```
diffuser/flow_matcher_v3_meanflow/__init__.py                    # docstring only, explains the shim
diffuser/flow_matcher_v3_meanflow/models/__init__.py             # from .mf_engine import MeanFlowEngine
                                                                 # from .mf_diffusion import MeanFlowODE
diffuser/flow_matcher_v3_meanflow/models/mf_engine.py            # from flow_matcher_v3_meanflow.models.mf_engine import MeanFlowEngine
diffuser/flow_matcher_v3_meanflow/models/mf_diffusion.py         # from flow_matcher_v3_meanflow.models.mf_diffusion import MeanFlowODE
```

Structurally 1:1 with the Gen3v4 shim, renamed. `diffuser/flow_matcher_v3_imeanflow/` is untouched.

Because the shim re-imports rather than redefines, the resolved class object keeps its **real**
`__module__` (`flow_matcher_v3_meanflow.models.mf_diffusion`). That matters downstream: the pickled
`diffusion_config._class` and the eval-time `target_class` string therefore agree, so the eval
script's config-overrides-pkl class check stays a no-op instead of firing a spurious
"Pickled diffusion class does not match" override. Same behaviour as Gen3v4.

## 4. Why eval never needed the shim

The eval script imports `flow_matcher_v3_meanflow.utils`, whose **own** `import_class` carries a
guard that `diffuser`'s copy does not:

| file | line | behaviour |
|---|---|---|
| `flow_matcher_v3_meanflow/utils/config.py` | 15 | `module_path = module_name if module_name.startswith(repo_name) else f'{repo_name}.{module_name}'` |
| `diffuser/utils/config.py` | 15 | `importlib.import_module(f'{repo_name}.{module_name}')` — unconditional |

`'flow_matcher_v3_meanflow.models'.startswith('flow_matcher_v3_meanflow')` is true, so eval imports
the real package directly. **Only the train script routes through `diffuser.utils.Config`**, which
is why the crash was training-only. (This asymmetry is inherited from DPCC upstream; not changed
here — fixing `diffuser/utils/config.py` would touch shared code across every generation.)

## 5. Coverage check — the other three `utils.Config` calls in the train script

| call | class argument | resolves to | needs shim? |
|---|---|---|---|
| `dataset_config` | `'datasets.SequenceDataset'` (str) | `diffuser.datasets.SequenceDataset` — real, exists | no |
| `model_config` | `'flow_matcher_v3_meanflow.models.MeanFlowEngine'` (str) | shim | **yes — fixed** |
| `diffusion_config` | `'flow_matcher_v3_meanflow.models.MeanFlowODE'` (str) | shim | **yes — fixed** |
| `trainer_config` | `MeanFlowTrainer` (class object) | `import_class` returns it unchanged | no |

So the two names exported by the shim are exactly the two that need it. No further gaps on the
train path.

## 6. Verification done / not done

- ✅ Shim file tree matches Gen3v4's byte-for-byte in structure; the two exported names match the
  two class strings in `config/avoiding-d3il.py:flow_matching_v3_meanflow`.
- ✅ Confirmed the guard asymmetry above by reading both `config.py` files.
- ❌ **Not executed** — no Python in the AI-coding container. The import only resolves on the
  cluster (PYTHONPATH puts the repo root ahead of the stale `diffuser.egg-info` install, same as
  Gen3v4).

## 7. What to re-run

Sync, then resubmit the pipeline unchanged:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/meanflow_pipeline.sh
```

The failed run left an empty
`logs/avoiding-d3il/flow_matching_v3_meanflow/H8_D..._dp0.5/6/` with only `dataset_config.pkl/json`
and the config snapshot — harmless; the rerun overwrites it. No cleanup needed.

Next failure mode to watch for, if any, is the first `model.loss()` call (the JVP): that is what
`FM_v3_meanflow_test/gates_meanflow.py` covers in seconds, and it is still unrun.

## 8. Note for Gen3v7 (AlphaFlow)

Add `diffuser/flow_matcher_v3_alphaflow/` (or whatever the package is named) to the scaffold
checklist **at folder-creation time**. PLAN §0.3 argued Gen3v6 exists partly to de-risk the shared
scaffolding on the cheap generation — this is precisely that payoff: the copy-modify checklist is
five locations, not four.

| # | location | Gen3v6 |
|---|---|---|
| 1 | `flow_matcher_v3_<gen>/` | ✅ |
| 2 | `FM_v3_<gen>_test/` | ✅ |
| 3 | `config/avoiding-d3il.py` (2 blocks + args_to_watch) | ✅ |
| 4 | `Slurm_Codes/sbatch/<Gen>/` | ✅ |
| 5 | **`diffuser/<package_name>/` shim** | ✅ **fix_1** |
