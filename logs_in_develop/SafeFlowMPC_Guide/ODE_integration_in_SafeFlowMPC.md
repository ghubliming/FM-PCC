# ODE Integration Used in SafeFlowMPC (Vector Field Only)

This note documents the ODE integration method used by SafeFlowMPC for integrating the vector field (flow matching step).

## Summary
- **SafeFlowMPC does NOT use torchdiffeq or any advanced ODE solver for vector field integration.**
- The integration is performed using a simple **forward Euler method** (manual fixed-step update loop).
- The vector field is computed by a neural network (FlowMatchingField), and the state is updated as:
  
  $$ x_{k+1} = x_k + \Delta t \cdot \text{velocity\_field}(x_k, t, \text{condition}) $$
- The integration loop is implemented directly in Python, with one step per flow step (see `SafeFlowMPC.step()` and `FlowMatchingField.compute_velocity()`).
- There is **no call to odeint, rk4, or any adaptive ODE solver** for the vector field.

## Key Code Locations
- `safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` — main integration loop in `step()`
- `safe_flow_mpc/SafeFlowMPC/FlowMatchingField.py` — neural network velocity field, called each step

## Details
- For each flow step, the code computes the neural velocity field and updates the state with a fixed step size (`dt = 1.0 / flow_steps`).
- If safety filtering is enabled, the updated state is projected onto the safe set after each step, but the ODE integration itself is always forward Euler.
- There is no support for higher-order or adaptive ODE solvers in the vector field integration path.

## Why Only Forward Euler?

The SafeFlowMPC codebase uses only the forward Euler method for vector field integration because:

- **Simplicity and Control:** Forward Euler is easy to implement and debug, and gives full control over each integration step. This is important when integrating with neural network-based velocity fields and safety filters.
- **Neural Network Coupling:** The vector field is computed by a neural network, which may not be smooth or differentiable enough for higher-order/adaptive ODE solvers to provide significant benefit. Each step is a black-box neural net evaluation.
- **Safety Filtering:** After each Euler step, the state may be projected onto a safe set (using a safety filter). This projection breaks the assumptions of higher-order ODE solvers, which expect continuous, unconstrained dynamics.
- **Determinism and Reproducibility:** Fixed-step Euler integration ensures deterministic and reproducible results, which is important for debugging and safety-critical applications.
- **No Need for Adaptive Methods:** The time step is chosen small enough (via `flow_steps`) to ensure stability and accuracy for the application, so more complex solvers are not required.

If you want to use a more advanced ODE solver, you would need to:
- Remove or redesign the safety filter/projection to allow continuous integration.
- Ensure the neural network velocity field is smooth enough for higher-order methods.
- Implement or integrate a different solver (e.g., RK4, torchdiffeq) in the integration loop.

## Conclusion
If you want to change the ODE integration method for the vector field in SafeFlowMPC, you would need to manually implement a different integrator (e.g., RK4, midpoint, or use torchdiffeq). By default, only forward Euler is used.
