# iMF-PCC Fix #5: Rebuild Mission Briefing

**Date**: 2026-05-14
**Scope**: iMeanFlow rebuild for FM-PCC training, evaluation, and config alignment
**Status**: Completed in code, verification intentionally not run per request

## Mission

Replace the unstable iMF glue code with a stable FMv3ODE-aligned implementation so that:
- training uses a normal flow-matching loss curve instead of the earlier catastrophic dual-target behavior
- eval loads checkpoints and runs with the same control flow style as FMv3ODE
- the iMF engine remains an actual wrapper around the FM-PCC stack, not a parallel ad hoc pipeline

## What Was Rebuilt

### 1. Core iMF model path
Files:
- [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)
- [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)

Changes:
- Replaced the old dual-velocity training target with a stable FM-style velocity backbone plus a small auxiliary residual branch
- Kept the iMF shape/API surface, but made the auxiliary branch non-dominant so it cannot overpower the main flow field
- Sampling now uses explicit Euler integration with conditioning preserved

### 2. Diffusion wrapper
File:
- [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)

Changes:
- Reworked loss computation to match FMv3ODE behavior more closely
- Main training signal is now the flow velocity loss
- Auxiliary residual is only a small regularizer
- Fixed checkpoint loading so the wrapper loads through the normal `nn.Module` path instead of dropping wrapper state
- Sampling now applies conditioning and uses the same ODE-style rollout structure

### 3. Config alignment
File:
- [config/avoiding-d3il.py](../../../config/avoiding-d3il.py)

Changes:
- Changed iMF loss weights from balanced dual-loss settings to a stable main-loss-plus-small-aux setup
- Disabled the old curriculum schedule that was contributing to instability
- Kept the FMv3ODE-derived inference contract intact

### 4. Eval path
File:
- [FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py)

Changes already present before this rebuild and kept in place:
- Seed resolution uses the iMF experiment path instead of `plan`
- Checkpoint lookup now matches the real iMF log tree
- Legacy checkpoint configs with duplicate `model` keys are handled
- Evaluation uses dataset conditioning instead of unconditional sampling

## Behavioral Result

Expected behavior after rebuild:
- training loss should track the FM-style objective, with the auxiliary branch staying small
- eval should load the same checkpoint layout as the train job writes
- the runtime should look like FMv3ODE with iMF as the underlying engine, not like a separate experimental pipeline

## Files Touched In This Rebuild

- [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)
- [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)
- [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)
- [config/avoiding-d3il.py](../../../config/avoiding-d3il.py)

## Notes

- I did not run verification because you explicitly asked for no verification.
- The `/bin/bash: libtinfo.so.6` warning is environment noise and not part of the iMF logic.
- The rebuild keeps the iMF namespace, but the learning dynamics now follow the FMv3ODE contract much more closely.
