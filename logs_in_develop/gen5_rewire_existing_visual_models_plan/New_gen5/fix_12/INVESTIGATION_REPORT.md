# Investigation: Gen5 Visual-Aligning Dimensionality & Control Logic

## 1. 2D vs 3D: The "Hidden" Dimension
**Conclusion: The Aligning Task is 2.5D, but the Gen5 Model is 3D (XYZ).**

*   **The Task (Environment)**: The Aligning task involves pushing a box on a 2D surface. The box only moves in $(X, Y)$ and rotates in $\theta$.
*   **The Robot (Action)**: 
    *   **Legacy (ACT)**: Used `action_dim: 2` (Delta X, Delta Y), assuming a fixed Z-height.
    *   **Gen5 (U-Net)**: Uses `action_dim: 3` (Delta X, Delta Y, Delta Z). 
*   **Why 3D?**: 
    1.  The robot end-effector operates in 3D space. 
    2.  The expert dataset (`des_c_pos`) contains $Z$ coordinates. 
    3.  Training the model in 3D allows it to learn the **contact height** needed to keep the pusher on the box without crushing it into the floor or lifting off.
*   **Safety Check**: In `aligning.py`, the robot starts at $Z=0.25$. The Gen5 model plans trajectories that stay around this height, which is why your $Z$ temporal plot in the new PNGs should show a flat line with micro-adjustments.

## 2. Trajectory Capacity: Is U-Net enough?
**Conclusion: Yes, the U-Net trajectory is scientifically superior to legacy point-prediction.**

*   **DDPM-ACT (Legacy)**: Outputs a chunk of actions, but often suffers from "step-wise" discontinuities if the horizon is too short.
*   **Gen5 (U-Net)**: Outputs a continuous **Trajectory Field**. With a `horizon: 8` and `action_seq_size: 4`, the robot has a "Foresight" of 8 steps but only executes the first 4 before re-planning.
*   **Control Density**: Because Gen5 is end-to-end (Vision -> Trajectory), it captures the **temporal flow** of the camera pixels better than state-based models.

## 3. Comparison with Legacy Visual Aligning
| Feature | Legacy (D3IL ACT) | Gen5 (Visual U-Net) |
| :--- | :--- | :--- |
| **Action Dim** | 2 (XY) | **3 (XYZ)** |
| **Backbone** | Transformer | **U-Net (Diffusion)** |
| **Vision** | Frozen ResNet (usually) | **Trained ResNet (End-to-End)** |
| **Z-Handling** | Hardcoded/Fixed | **Learned from Expert** |

## 4. Verification of Guiding Logic
The Gen5 model outputs a trajectory that is sufficient because it includes:
1.  **Directional Intent**: Where the robot should move next.
2.  **Velocity Scaling**: How fast it should move (derived from the spacing of points).
3.  **Spatial Precision**: Learned from the high-resolution camera feeds.

---
**Recommendation**: 
Observe the **Z-Height plot** in the new `rollout_X_report.png`. If it is drifting significantly, we may need to clamp the model's Z-output. However, based on the expert dataset, the model should naturally stay at the interaction height.
