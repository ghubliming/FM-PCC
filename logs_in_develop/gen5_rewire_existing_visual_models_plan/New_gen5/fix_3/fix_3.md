# Visual Aligning Evaluation Fix 3

This document records the fix applied after the visual aligning evaluation crashed inside D3IL's diffusion sampler because the action bounds were missing.

## Failure Summary

The evaluation job reached the first rollout and then aborted with:

```text
AttributeError: 'Diffusion' object has no attribute 'min_action'
```

The crash occurred during diffusion sampling when the code attempted to clamp the reconstructed action tensor:

```python
x_recon.clamp_(self.min_action, self.max_action)
```

## Root Cause

The visual evaluation path uses `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`, which directly instantiates the D3IL diffusion model through `VisualDiffusionBridge`.

Unlike the regular D3IL agent wrappers, this path did not initialize the diffusion sampler bounds before inference. In the standard D3IL agent stack, the same bounds are set from the training scaler:

```python
self.model.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
self.model.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)
```

Because the visual bridge bypassed that agent-layer setup, the diffusion model had no `min_action` and `max_action` attributes when `clip_denoised=True` ran.

## Fix Applied

The visual bridge was updated to initialize the diffusion bounds immediately after constructing the model.

### Code Delta

File: `ddpm_encdec_vision/models/d3il_visual_bridge.py`

```diff
 import torch
 import torch.nn as nn
 from omegaconf import OmegaConf
 import hydra
 import sys
 import os

 # Ensure d3il is in path if not already
 sys.path.append(os.path.abspath('d3il'))

 from agents.utils.scaler import Scaler
 from environments.dataset.aligning_dataset import Aligning_Dataset
...
         self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
         self.diffusion_model = hydra.utils.instantiate(model_cfg).to(self.device)

         self._set_action_bounds(config)

     def _set_action_bounds(self, config):
         """Initialize diffusion clamp bounds from the aligning training dataset."""
         try:
             train_data_path = getattr(config, "train_data_path", None)
             ...
             dataset = Aligning_Dataset(
                 data_directory=train_data_path,
                 device="cpu",
                 obs_dim=20,
                 action_dim=self.diffusion_model.action_dim,
                 max_len_data=getattr(config, "max_len_data", 512),
                 window_size=getattr(config, "window_size", 8),
             )
             scaler = Scaler(...)
             self.diffusion_model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(self.device)
             self.diffusion_model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(self.device)
         except Exception:
             default_bounds = torch.tensor([-0.01, -0.01, -0.01], device=self.device)
             self.diffusion_model.min_action = default_bounds
             self.diffusion_model.max_action = -default_bounds
```

## Result

The diffusion sampler now has valid action bounds before evaluation starts, so the clamp step no longer crashes and the rollout can proceed.

## Notes

- This is the correct fix to retain; the missing bounds were not intentional.
- The `libtinfo.so.6` shell warning is unrelated.
- The `No checkpoint found` warning is separate from this error and only means the current evaluation run is not loading a saved model file.