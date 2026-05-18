# Recover Success From Legacy `ddpm_encdec_vision`

This note explains how to recover the working behavior by restoring the legacy contract that the visual pipeline depended on.

## Short Version

The current tree is failing for the same reason many refactors fail: it became more flexible before every call site was updated to match the new contract.

The two changes most likely to break the run are:

1. `VisualUNet` now trusts `config.obs_dim`, but the active visual config still carries a dummy `obs_dim: 128` value.
2. `Scaler` no longer clamps low-variance dimensions with the legacy `1e-2` safety floor.

If you want the fastest recovery path, restore the legacy working behavior first, then reintroduce flexibility one branch at a time.

## What Legacy Got Right

The legacy run was stable because it kept a fixed visual contract:

- the visual backbone expected the same transition shape every time
- normalization was conservative and tolerant of constant dimensions
- trainer and model assumptions stayed aligned
- evaluation logic stayed close to the training data shape

That consistency matters more than configurability when the pipeline mixes vision, state, and diffusion.

## Recovery Order

### 1. Restore the legacy normalization floor

Current risk:
- [current scaler](../../../../ddpm_encdec_vision/utils/scaler.py)

Legacy behavior:
- [legacy scaler](../../../../ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/utils/scaler.py)

Change the current scaler back to the legacy safety behavior:

- use a `1e-2` minimum standard deviation floor
- apply it to both input and output scaling
- do not rely on a tiny `1e-12` epsilon for real data stability

Why this comes first:
- it prevents unstable scaling from hiding the real shape bug
- it is low risk and easy to verify with a short train step

### 2. Restore the legacy visual shape contract

Current risk:
- [current VisualUNet](../../../../ddpm_encdec_vision/models/visual_unet.py)

Legacy behavior:
- [legacy VisualUNet](../../../../ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/models/visual_unet.py)

The visual pipeline should not build its transition dimension from the dummy visual `obs_dim: 128` field. That value is not a real image or proprioception dimension.

The recovery fix is one of these:

- ignore `config.obs_dim` for visual runs and force the legacy working dim
- or make the config explicit so the model only sees the real transition size

The goal is to keep the backbone input contract identical to the legacy working run.

### 3. Make the trainer branch on batch layout

Current risk:
- [current trainer](../../../../ddpm_encdec_vision/utils/training.py)

Legacy behavior:
- [legacy trainer](../../../../ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/utils/training.py)

The current trainer still indexes batch slots as if every batch has the visual layout. That is unsafe if the code now accepts both visual and state-only batches.

Fix:

- check the batch length before touching `batch[2]` and `batch[3]`
- only scale the visual slots when they exist
- keep state-only and visual paths separate until they are proven equivalent

This prevents silent batch-shape crashes during training.

### 4. Make the visual bridge explicit

Current risk:
- [d3il visual bridge](../../../../ddpm_encdec_vision/models/d3il_visual_bridge.py)

Legacy behavior:
- fixed contract, fewer config-dependent branches

If the bridge reads `obs_seq_len` and `action_seq_len` from config, the config must be complete and mode-specific.

Fix:

- set explicit sequence lengths for the visual run
- do not let dummy or reused config values decide the model layout
- verify the bridge builds the same sequence shape as the working legacy run

### 5. Keep evaluation close to training assumptions

Current risk:
- [evaluation entrypoint](../../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)

The evaluation wrapper has extra state-only assumptions that are not part of the legacy success path. That is fine later, but it should not be the first place you debug the visual pipeline.

Fix:

- first validate the pure legacy visual path
- only after that, test the state-only branch
- keep evaluation shape logic simple while recovering the run

## Most Likely Root Cause

The most likely direct cause is this combination:

- visual config still exposes a dummy `obs_dim: 128`
- current `VisualUNet` trusts that value
- the backbone gets built for the wrong transition dimension
- scaler changes make the pipeline less robust to low-variance data

That is enough to break training or inference even if the rest of the code looks correct.

## Practical Recovery Plan

1. Patch the scaler back to the legacy `1e-2` floor.
2. Force the visual model to use the working legacy transition shape.
3. Add batch-length guards in the trainer.
4. Explicitly set sequence lengths for the visual bridge.
5. Run one short visual train step and confirm the tensor shapes match the legacy run.
6. Only then re-enable any state-only or dynamic config behavior.

## What To Verify After The Patch

- `VisualUNet` builds the same transition dimension as the legacy run.
- scaler output stays bounded when a dimension is constant or nearly constant.
- trainer no longer assumes a fixed batch length.
- evaluation can load and step without shape mismatches.

## Recovery Principle

When the legacy version is the only one that works, the safest fix is not to keep adding branches.

First restore the legacy contract.
Then add flexibility behind explicit checks.
Then verify each new mode separately.
