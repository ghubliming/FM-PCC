# Legacy vs Current `ddpm_encdec_vision`

This note compares the working legacy snapshot under `ddpm_encdec_vision/ddpm_encdec_vision_Legacy` with the current code under `ddpm_encdec_vision` and `ddpm_encdec_vision_test`.

## Bottom Line

The current tree is not just a cleanup of the legacy visual pipeline. It adds state-only support, DPCC plumbing, and dynamic config handling, but it also changes a few core assumptions that the legacy visual run depended on.

The two highest-risk regressions are:

1. `VisualUNet` now reads `config.obs_dim` literally, but the visual config still carries a dummy `obs_dim: 128` field. That makes the current visual backbone build with the wrong transition dimension.
2. `Scaler` no longer uses the legacy `1e-2` minimum standard deviation floor. It now adds `1e-12`, which reintroduces numerical instability for constant or near-constant dimensions.

## Main Differences From Legacy

### 1. Visual backbone dimension handling changed

Legacy `VisualUNet` hardcoded the visual trajectory shape to the working 6D layout:
- action dims: 3
- proprioception dims: 3
- total transition dim: 6

Current `VisualUNet` now computes the observation dimension from config:
- [current VisualUNet](ddpm_encdec_vision/models/visual_unet.py#L62)
- [legacy VisualUNet](ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/models/visual_unet.py#L60)

That looks flexible, but it is unsafe for the visual config because the active config still defines a dummy `obs_dim: 128`:
- [visual config](config/aligning-d3il-visual.py#L107)

So the current model can build a backbone with `action_dim + obs_dim = 3 + 128 = 131` instead of the legacy 6D layout. That is a direct shape-contract break.

### 2. Scaler safety margin was weakened

Legacy scaler behavior:
- clamps standard deviations with `min=1e-2`
- logs stabilized std values
- protects constant dimensions from exploding during normalization

Current scaler behavior:
- [current scaler](ddpm_encdec_vision/utils/scaler.py#L35)
- legacy scaler clamp: [legacy scaler](ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/utils/scaler.py#L37)

The current code changes the safe std from `clamp(min=1e-2)` to `+ 1e-12`.

That is a big regression because the aligning data has dimensions that can be nearly constant. With the smaller epsilon, normalization can produce very large values, unstable losses, and bad rollouts.

### 3. Trainer still assumes the legacy batch layout

The current diffusion loss now accepts both 5-item vision batches and 3-item state-only batches:
- [current diffusion loss](ddpm_encdec_vision/models/visual_gaussian_diffusion.py#L13)

But the trainer still unconditionally indexes batch slots as if the batch always has vision layout:
- [current trainer scaling and slicing](ddpm_encdec_vision/utils/training.py#L136)
- [legacy trainer parity logic](ddpm_encdec_vision/ddpm_encdec_vision_Legacy/ddpm_encdec_vision/utils/training.py#L136)

That means the state-only path is not actually end-to-end safe yet. If the nonvisual dataset is used, `batch[3]` does not exist and the trainer will fail before the model loss runs.

### 4. Evaluation wrapper now has extra state-only assumptions

Current evaluation adds a new nonvisual branch that:
- fabricates a 20D observation by concatenating a desired position with raw state
- updates only the first two coordinates of the mental position
- keeps a 3D action default in several places

Relevant locations:
- [state-only inference branch](ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py#L443)
- [action slice selection](ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py#L541)
- [mental position update](ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py#L560)

This is a second-order risk rather than the primary break, but it is another place where the current code makes stronger assumptions than the legacy run.

### 5. Config coupling is now more dynamic, but also more fragile

Current bridge code reads `obs_seq_len` and `action_seq_len` from config instead of using a fixed visual contract:
- [current bridge config mapping](ddpm_encdec_vision/models/d3il_visual_bridge.py#L72)

That is fine only if the config keys are guaranteed to match the intended mode. If the config is partially populated or reused across visual and nonvisual runs, the bridge can silently instantiate the wrong sequence setup.

## Most Likely Root Cause

If the legacy visual run is the only one that works, the most likely failure point in the current tree is the `VisualUNet` transition dimension change combined with the still-present dummy `obs_dim: 128` in the visual config.

That mismatch alone can break the whole training/inference contract:

- config says `obs_dim = 128`
- current `VisualUNet` uses `config.obs_dim`
- current visual trajectory data is still 6D
- backbone is built for the wrong channel count

The scaler regression is the second likely contributor because it makes the pipeline much less tolerant of constant or low-variance dimensions.

## What To Check First

1. Verify the instantiated `VisualUNet.backbone.transition_dim` in a visual run.
2. Restore the legacy scaler floor and re-run a short train step.
3. If state-only mode is intended, branch the trainer on batch length before indexing `batch[2]` and `batch[3]`.
4. Keep the visual config from leaking dummy `obs_dim: 128` into the backbone constructor.

## Short Comparison

- Legacy: fixed visual contract, safer normalization, fewer mode branches.
- Current: more flexible, but the added flexibility is not consistently propagated through trainer, model, and config.
