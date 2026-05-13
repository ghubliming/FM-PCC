# iMeanFlow Training Fix 1: Keyword Argument Correction

## 1. Context & Problem
During the initialization of the iMeanFlow (Improved Mean Flows) training pipeline, the script crashed immediately after data generation.

**Error Message:**
```
TypeError: TimeConditionedDualVelocity.__init__() got an unexpected keyword argument 'use_jvp'
```

## 2. Root Cause
The `TimeConditionedDualVelocity` model class (defined in `flow_matcher_v3_imeanflow/models/imf_velocity.py`) defined its optional Jacobian-Vector Product (JVP) guidance parameter as `include_jvp`. However, the training and evaluation entry points were passing it as `use_jvp`.

## 3. Solution
Corrected the keyword argument name in the following files:
- `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`
- `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`

Modified call:
```python
self.model = TimeConditionedDualVelocity(
    state_dim=state_dim,
    hidden_dim=256,
    time_dim=128,
    include_jvp=False,  # Renamed from use_jvp
).to(device)
```

## 4. Impact
The training script now correctly instantiates the model and proceeds to the training loop.

## 5. Time Embedding Dimension Fix
Resolved a `RuntimeError` during the forward pass:
```
RuntimeError: expand(torch.cuda.FloatTensor{[32, 1, 20, 128]}, size=[32, 20, -1]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (4)
```

**Root Cause:**
The model was unconditionally trying to `unsqueeze(1).expand(...)` the time embedding if the input state `x` was 3D. However, the training script passes a 2D time tensor `t` (B, T), which already produces a 3D time embedding `(B, T, d)`. Unsqueezing this made it 4D, causing the `expand` call to fail.

**Solution:**
Updated `flow_matcher_v3_imeanflow/models/imf_velocity.py` to only expand the time embedding if it is 2D and the state input is 3D.

```python
if x.dim() == 3 and t_embed.dim() == 2:
    t_embed = t_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
```
