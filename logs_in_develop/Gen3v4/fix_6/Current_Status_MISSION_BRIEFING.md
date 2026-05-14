# Mission Briefing: iMeanFlow (iMF) Pipeline Standardization (Fix #6)

**Status**: Completed  
**Target**: Standardize the iMeanFlow (iMF) experimental engine to match the `FMv3ODE` pipeline architecture while preserving its unique dual-velocity mathematical properties.

---

## 1. Mathematical Foundation: iMeanFlow vs. FMv3ODE

Both engines utilize the same **Linear Probability Path** ($x_t = (1-t)x_0 + tx_1$) and **Beta-Skewed Time Sampling** ($t \sim 1 - \text{Beta}(1.5, 1.0)$). The distinction lies in how the velocity field is structured and used across Training and Inference.

### 1.1 TRAINING PHASE: The Learning Objective

| Feature | FMv3ODE (Standard) | iMeanFlow (iMF) |
| :--- | :--- | :--- |
| **Model Output** | Single velocity vector: $v$ | Dual streams: Main $u$ and Auxiliary $v_{aux}$ |
| **Loss Function** | Simple MSE on velocity | **Weighted Sum** of Mean and Residual losses |
| **Target** | $v^* = x_1 - x_0$ | $u^* = x_1 - x_0$ AND $v_{aux}^* = \mathbf{0}$ |

**iMF Training Proof (`imf_diffusion.py`):**
```python
# Dual-head prediction and split loss
velocity_pred, aux_pred = self._predict_uv(x_t, cond, t)
main_loss, _ = self.loss_fn(velocity_pred, v_target)
aux_loss = F.mse_loss(aux_pred, torch.zeros_like(aux_pred)) # Residual constraint
total_loss = main_loss + self.aux_loss_weight * aux_loss
```

### 1.2 INFERENCE PHASE: Sampling & Fusion

During sampling, FMv3ODE relies on high-order ODE solvers, whereas iMF uses a specialized **Velocity Fusion** to stabilize the path.

| Feature | FMv3ODE (Standard) | iMeanFlow (iMF) |
| :--- | :--- | :--- |
| **Velocity Step** | $v_{step} = \text{Model}(x, t)$ | $v_{step} = u + \lambda \cdot v_{aux}$ |
| **Solver** | Flexible (`torchdiffeq` RK4, etc.) | Standard Euler (Fixed-step) |

**iMF Inference Proof (`imf_diffusion.py`):**
```python
# The model 'fuses' the streams only during inference
def _predict_velocity(self, x, cond, t):
    velocity, aux = self._predict_uv(x, cond, t)
    return velocity + self.sample_aux_weight * aux # u + lambda * v_aux
```

---

## 2. Standardization Roadmap (Fix #6)

### A. Evaluation Parity (`eval_flow_matching_v3_imeanflow.py`)
- **Engine Bridge**: Completely replaced the legacy iMF evaluation script with the robust `FMv3ODE` version.
- **Variant Looping**: Restored support for `config/projection_eval.yaml`, enabling the standard suite of DPCC-R/T/C benchmarks.
- **Serialization**: Integrated the `Plan` config system, ensuring that evaluation metadata is saved alongside the training artifacts.

### B. Directory Hierarchy
Evaluation results are now deterministically nested:
`logs/<experiment>/ddpm_encdec_vision/<H>/<seed>/eval/<plan_name>/`

---

## 3. Conclusion
With Fix #6, the iMeanFlow engine is mathematically distinct in its **Dual-Velocity Decomposition** but structurally identical in its **Deployment** pipeline. The auxiliary residual stream acts as an "active dampener," smoothing out the high-frequency noise that often causes standard ODE solvers to wobble in complex tasks like `aligning`.
