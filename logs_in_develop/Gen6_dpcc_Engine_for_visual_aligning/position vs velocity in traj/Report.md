# Gen6 Verdict: The True Nature of "Traj", MuJoCo Inputs, and IK

> [!NOTE]
> This document provides a mathematically and architecturally rigorous analysis of the trajectory representation ("traj"), the physical control inputs fed into MuJoCo, the role of Inverse Kinematics (IK), and the forensic code details of the **ddpm_encdec_vision** model architecture in the **Visual Aligning** task under the Gen6 DPCC engine.

---

## 1. What is the Trajectory (`traj`)?
Besides the high-dimensional visual observations (i.e. the Body-Perspective camera image `bp_image` and the In-Hand camera image `inhand_image`), the trajectory representation consists of two distinct components:

### A. The State Observation Trajectory ($s_t$)
* **Dimensions:** 3D Cartesian coordinates ($x, y, z$).
* **Physical Meaning:** The desired target position coordinates of the robot's end-effector.
* **Interface role:** Fed into the neural network (U-Net) along with the transposed visual frames to establish the current starting state context.

### B. The Action Velocity Trajectory ($a_t$)
* **Dimensions:** 3D spatial deltas ($vx, vy, vz$).
* **Physical Meaning:** The spatial displacement (step delta) between sequential positions.
* **Dataset Formula:** 
  $$\Delta x = x_{t+1} - x_t$$

---

## 2. What is Fed Into MuJoCo?
At every simulation step in `aligning_sim.py`, the following bridge is executed:

1. **Inference Output:** The model predicts the next delta action:
   $$\text{action\_delta} = [\Delta x, \Delta y, \Delta z]$$
2. **Integration Step:** This delta is added directly to the current robot Cartesian desired position:
   $$\text{pred\_action} = \text{action\_delta} + \text{des\_robot\_pos}$$
3. **Quaternion Coupling:** The 3D position is concatenated with a fixed target orientation quaternion `[0, 1, 0, 0]` to form a 7D pose:
   $$\text{mujoco\_action} = [x_{\text{des}}, y_{\text{des}}, z_{\text{des}}, 0, 1, 0, 0]$$
4. **Simulator Step:** This 7D coordinate is fed directly into the environment:
   ```python
   # aligning_sim.py (Line 90)
   obs, reward, done, info = env.step(pred_action)
   ```

---

## 3. Do We Solve the Joint-Level IK?
### ❌ Verdict: No joint-level Inverse Kinematics (IK) is solved!

By tracing the controller down to the MuJoCo simulation layer, we discover that **no joint angle calculations or joint-level IK solvers** are used.

#### The Mathematical Proof from the Controllers:
1. The robot environment uses a Cartesian robot model (`panda_rod_invisible.xml`) in MuJoCo.
2. The active controller is `CartPosQuatCartesianRobotController` ([IKControllers.py: Line 459](file:///workspaces/FM-PCC/d3il/environments/d3il/d3il_sim/controllers/IKControllers.py#L459)).
3. Inside this controller, the `getControl` method simply returns the Cartesian setpoints **directly** to the MuJoCo actuators:
   ```python
   def getControl(self, robot):
       robot.des_c_pos = self.desired_pos[:3]
       robot.des_c_vel = np.zeros((3,))
       if self.desired_pos.shape[0] > 3:
           robot.des_quat = self.desired_pos[3:]
       robot.des_quat_vel = np.zeros((4,))

       return self.desired_pos  # <-- Returns absolute 7D Pose directly!
   ```

---

## 4. Architectural Comparison: Avoiding vs. Visual Aligning

| Comparative Axis | Avoiding (`dpcc` & `FMv3ODE`) | Visual Aligning (Gen6 `ddpm_encdec_vision`) |
| :--- | :--- | :--- |
| **Model Modality** | State-Only (unconstrained diffuser) | Multi-Modal (Dual RGB Cameras + End-Effector State) |
| **Action Variable** | 2D Spatial Delta ($vx, vy$) | 3D Spatial Delta ($vx, vy, vz$) |
| **Observation Variable** | 4D Cartesian coordinates ($x_{des}, y_{des}, x, y$) | 3D Cartesian coordinates ($x, y, z$) |
| **Trajectory Representation** | 6D Unified vector: `[vx, vy, x_des, y_des, x, y]` | 6D Unified vector: `[vx, vy, vz, x, y, z]` |
| **DPCC Snapping Target** | **Position Only:** Snaps position coordinates to bypass spherical obstacles and halfspace planes | **Position Only:** Snaps position coordinates to bypass flat table heights and boundary cages |
| **MuJoCo Command Input** | 2D absolute position setpoints | 7D absolute pose setpoints (3D Cartesian + Fixed Quaternion) |
| **Inverse Kinematics (IK)** | **Bypassed:** Actuators track Cartesian setpoints directly via `CartPosQuatCartesianRobotController` | **Bypassed:** Actuators track Cartesian setpoints directly via `CartPosQuatCartesianRobotController` |

---

## 5. Mathematical Proof: The Myth of "Physical Velocity"

There is **no physical velocity** (i.e., meters per second or joint angular velocity) used anywhere in the dataset, model, or simulator. The label "velocity" (`vx`, `vy`, `vz`) is purely a nomenclature choice.

### Proof A: Dimensional Analysis of the Dataset
In both the Avoiding and Visual Aligning datasets, the "velocity" action `vel_state` is constructed via:
```python
# aligning_dataset.py (Line 243) & avoiding_dataset.py (Line 59)
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
```
* **Dimensionality Check:** 
  $$\text{Units}(\text{vel\_state}) = \text{meters} - \text{meters} = \text{meters}$$
* Since there is **no division by the time step $dt$**, the trajectory dimensions are strictly **spatial displacements (delta positions)**.

### Proof B: Dimensional Legality of the Control Loop
During simulation rollouts, the predicted model action is added directly to the robot's current Cartesian coordinate:
```python
# aligning_sim.py (Line 87) & avoiding_sim.py (Line 64)
pred_action = pred_action[0] + des_robot_pos
```
* **Dimensional Addition Check:**
  $$\text{Units}(\text{pred\_action}) = \text{Units}(\text{pred\_action}[0]) + \text{Units}(\text{des\_robot\_pos})$$
  $$\text{meters} = \text{meters} + \text{meters}$$
* If `pred_action[0]` were a physical velocity (m/s), this addition would violate basic dimensional analysis laws (adding meters to meters per second is mathematically illegal).

### Proof C: Bypassing Velocity Actuators in MuJoCo
The Cartesian controller in MuJoCo receives the target position directly and ignores the velocity channels:
```python
# IKControllers.py (Line 473-474)
robot.des_c_pos = self.desired_pos[:3]
robot.des_c_vel = np.zeros((3,)) # <-- Hardcoded to ZERO velocity!
```
* Therefore, the simulator's physical solver operates strictly under position-tracking controllers.

---

## 6. Forensic Deep Dive into the `ddpm_encdec_vision` (Encdec) Codebase
By investigating the raw implementation inside `/workspaces/FM-PCC/ddpm_encdec_vision`, we uncover the following exact software architecture:

### A. The Denoising Diffeomorphism Loop (`VisualGaussianDiffusion`)
* **File Reference:** [`ddpm_encdec_vision/models/visual_gaussian_diffusion.py`](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py)
* **Joint Trajectory Construction (Line 18):** Action deltas (`act`) and observation position coordinates (`obs`) are concatenated into a unified 6D joint trajectory variable `x`:
  ```python
  x = torch.cat([act, obs], dim=-1)
  ```
* **Tuple Unpacking and Context Alignment (Lines 68-84):** During stochastic evaluation, `cond[0]` is received as a tuple `(bp_imgs, inhand_imgs, pos)`. The class wraps `bp_imgs` and `inhand_imgs` for the encoder while aligning `pos[:, -1]` as the target coordinate for the $t=0$ trajectory snap:
  ```python
  bp_imgs, inhand_imgs, pos = cond[0]
  visual_cond = (bp_imgs, inhand_imgs, pos)
  snapping_cond = {0: pos[:, -1]}
  ```
* **Wide Selective Clamping (Lines 45-49):** When safety clamps are enabled (`clip_denoised = True`), it clamps only the action/velocity dimensions and excludes the observation dimensions to prevent proprioceptive spatial distortion:
  ```python
  if self.clip_denoised:
      x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)
  ```

### B. The Visual Encoding Network (`VisualUNet`)
* **File Reference:** [`ddpm_encdec_vision/models/visual_unet.py`](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_unet.py)
* **ResNet Image Encoding (Lines 20-42):** Instantiates a `MultiImageObsEncoder` to encode 3-channel visual frames resized to `[3, 96, 96]` for both `agentview_image` (`bp_cam`) and `in_hand_image` (`inhand_cam`).
* **Safe Temporal Padding (Lines 51-56):** To support the 1D temporal UNet's downsampling structure (which requires sequence lengths that are multiples of 8), it pads the trajectory and visual embeddings dynamically to `padded_horizon` before backbone execution and crops it back afterward:
  ```python
  self.target_horizon = config.horizon
  self.padded_horizon = ((self.target_horizon + 7) // 8) * 8
  ```
