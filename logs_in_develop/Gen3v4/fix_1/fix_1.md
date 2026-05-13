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
