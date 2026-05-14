# Mission Briefing: Drifting (FM-D) Pipeline Standardization (Fix #1)

**Status**: Completed  
**Target**: Align the Drifting Flow Matching (FM-D) pipeline with the standardized FM-PCC architecture while documenting its unique distribution-guidance mechanics.

---

## 1. Mathematical Foundation: Drifting (FM-D) vs. FMv3ODE

Both engines share the **Linear Probability Path** ($x_t = (1-t)x_0 + tx_1$) and **Beta-Skewed Time Sampling** ($t \sim 1 - \text{Beta}(1.5, 1.0)$). The "Drifting" engine adds a second layer of control: **Manifold Guidance**.

### 1.1 TRAINING PHASE: Expert Distribution Learning

While FMv3ODE only learns individual transitions, Drifting learns the **entire expert distribution manifold**.

| Feature | FMv3ODE (Standard) | Drifting (FM-D) |
| :--- | :--- | :--- |
| **Learning Goal** | Individual $v = x_1 - x_0$ | $v$ + Manifold Density $P(x)$ |
| **Extra Components** | None | **Memory Bank** + **Drift Encoder** |
| **Memory Bank** | N/A | Circular buffer of 5000 expert trajectories |

**Drifting Training Logic (`drift_loss.py`):**
During training, expert trajectories are encoded into a latent space and stored. The model learns to minimize the **Drift Loss** ($L_{drift}$), which measures the KL-Divergence or MMD between sampled and expert distributions.
$$L_{drift} = D_{KL}(Q_{sampled} || P_{expert})$$

### 1.2 INFERENCE PHASE: Gradient-Based Guidance

This is the "Drifting" effect. The ODE solver does not just follow the model's velocity; it is **steered** by the gradient of the learned manifold.

| Feature | FMv3ODE (Standard) | Drifting (FM-D) |
| :--- | :--- | :--- |
| **Velocity Step** | $v_{step} = \text{Model}(x, t)$ | $v_{step} = v_{model} - \eta \cdot \nabla_x L_{drift}$ |
| **Dynamics** | Purely Generative | **Guided "Drift" towards Expert Manifold** |
| **Solver** | `torchdiffeq` Standard | `DriftODESolver` (Guided Integration) |

**Drifting Inference Proof (`drift_ode_solvers.py`):**
```python
# Drifting guidance modifies the velocity field in real-time
def guided_ode_rhs(t, x):
    v_fm = model.predict_velocity(x, t)    # Standard FM velocity
    v_drift = drift_loss.get_gradient(x)   # Gradient towards expert manifold
    return v_fm - eta * v_drift            # The "Drift" correction
```

---

## 2. Standardization Roadmap (Fix #1)

### A. Evaluation Parity (`eval_flow_matching_v3_drifting.py`)
- **Manifold Integration**: Successfully ported the legacy distribution-matching logic into the standardized `Plan/Variant` framework.
- **Guidance Scaling**: Exposed the $\eta$ (drift scale) hyperparameter to the YAML config, allowing for `dpcc-r/t/c` benchmarking.
- **Serialization**: Integrated `Plan` config system for deterministic result nesting.

### B. Directory Hierarchy
Standardized output path:
`logs/<experiment>/ddpm_encdec_vision/<H>/<seed>/eval/<plan_name>/`

---

## 3. Conclusion
The Drifting (FM-D) engine provides **Distribution-Aware Control**. By using a learned "Critic" (the Drift Loss) to steer the ODE integration, FM-D effectively "pulls" the robot back to expert-like behavior whenever the generative model begins to diverge.

> [!TIP]
> Use Drifting when the agent suffers from **Compounding Error** (drift) over long horizons. The manifold guidance acts as a "magnetic pull" toward valid states.
