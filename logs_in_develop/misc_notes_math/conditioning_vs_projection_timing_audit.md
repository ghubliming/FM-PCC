# Audit: Internal Sampling Logic Timing (Re-anchoring vs. Projection)

This note documents the precise timing of state re-anchoring (Conditioning) and constraint satisfaction (Projection) within the 10-step ODE integration loop.

## 1. Re-anchoring (Conditioning) Timing: 10/10 Steps
The "Hard Reset" of the starting state (Step 0) happens at **every single ODE iteration**, regardless of whether we are at the beginning or end of the sampling process.

### Code Evidence (`diffusion.py:261`):
```python
for i in range(total_steps):
    # ... Integration ...
    # This happens EVERY iteration (no 'if near_end' here)
    x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
```
*   **Mathematical Reason:** To prevent numerical drift of the boundary condition ($x_{H=0} = s_{\text{real}}$). Even if the neural network predicts a velocity that moves the start point during integration, we forcefully snap it back to reality to ensure the plan always starts exactly where the robot is currently standing.

## 2. Projection (Obstacle Avoidance) Timing: "Last Half" Only
Unlike re-anchoring, the **Safety Shield / Projector** (which pushes the plan out of obstacles) is typically only active during the final stages of the flow integration ($t \rightarrow 1.0$).

### The `near_end` Logic:
```python
# Robust logic: Ensure final steps are prioritized
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
```
*   **Mathematical Reason:** At $t=0$, the path is pure Gaussian noise. Projecting random noise into obstacle-free space is mathematically unstable and often counter-productive. By waiting until the model has "sculpted" the general shape of the path (e.g., after the threshold), the projector works on a much more coherent candidate.

## 3. Comparison Summary

| Mechanism | Frequency | Scope | Goal |
| :--- | :--- | :--- | :--- |
| **Apply Conditioning** | **100% (Every Step)** | **Horizon Index 0** | **Re-anchoring**: Keeps the start of the plan pinned to the robot's actual position. |
| **Projector / Shield** | **Variable (Last Half)** | **Entire Horizon** | **Safety**: Pushes the "Imagined" future steps out of obstacles. |

### Final Result
After 10 iterations of this interleaved process, you get a trajectory that is **perfectly anchored at the start** and **locally safe** at the end of the imagination horizon.
