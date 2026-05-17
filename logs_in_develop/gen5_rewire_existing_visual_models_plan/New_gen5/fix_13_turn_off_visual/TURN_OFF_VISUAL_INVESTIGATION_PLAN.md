# Investigation & Plan: Supporting Non-Visual (State-Only) Control in Gen5

## 1. Investigation: How D3IL Handles "No Visual"

In the D3IL environment (`Robot_Push_Env`), the observation logic is split by the `if_vision` flag.

### A. The Visual Observation (Current Gen5)
*   **Toggle**: `if_vision=True`
*   **Returns**: `(robot_ee_pos, bp_image, inhand_image)`
*   **Data Type**: `(3, )`, `(3, 96, 96)`, `(3, 96, 96)`
*   **Neural Path**: Images $\to$ ResNet-18 $\to$ 128-dim Embedding $\to$ FiLM U-Net.

### B. The State-Only Observation (Native D3IL)
*   **Toggle**: `if_vision=False`
*   **Returns**: `env_state` (17-dimensional vector)
*   **Vector Components**:
    1.  **Robot EE Pos**: `[x, y, z]` (3)
    2.  **Box Position**: `[x, y, z]` (3)
    3.  **Box Quaternion**: `[w, x, y, z]` (4)
    4.  **Target Position**: `[x, y, z]` (3)
    5.  **Target Quaternion**: `[w, x, y, z]` (4)
*   **Total Dimension**: 17 dimensions.

### C. Current Gen5 & D3IL Limitations
Investigation of the D3IL `agents/` directory reveals that they do not support a "Hot Toggle" for vision. Instead, they use separate specialized classes:
*   **State-Only**: `ActAgent` (`act_agent.py`) expects a flat vector.
*   **Visual**: `ActVisionAgent` (`act_vision_agent.py`) hardcodes the unpacking of `(bp_image, inhand_image, robot_pos)`.

**Key Finding**: In both D3IL and the current Gen5, you cannot "turn off" visual for a visual model. If you try to pass `if_vision=False` to a visual agent, it will crash or raise a `NotImplementedError` because the `predict()` method is hardcoded to receive visual arguments.

---

## 2. Our Gen5 Goal: The "Smart Backbone" (Unified Support)
Unlike D3IL's separate-file approach, we aim for a **Unified Gen5 Interface** that can handle both paths within a single class structure.

The goal is to allow Gen5 to run in **"State-Only"** mode while maintaining the same U-Net backbone, allowing for a direct comparison between visual and proprioceptive performance.

### Step 1: Implementation of "Blind Mode" in `VisualAgentWrapper`
We will implement a middle-ground "Blind Mode" for current visual models:
*   **Logic**: If `if_vision=False` but the model is `visual_input=True`, the wrapper will use a **Zero/Mean Visual Embedding** or the **Initial Frame Embedding** as a constant condition.
*   **Function**: Allows testing how well the robot can push the box based only on its proprioceptive tracking of its "mental map" of where the box *was* at the start.

### Step 2: Modular Bridge in `VisualUNet`
Update `VisualUNet.py` to support a `state_only` toggle:
*   **Structural Change**: Add a `cond_projector` (Linear Layer: $17 \to 128$) to the model.
*   **Forward Logic**:
    *   `if visual_input`: Use ResNet as usual.
    *   `if state_only`: Pass the 17-dim `env_state` through the `cond_projector` to reach the 128-dim latent space.
*   **Benefit**: This allows the **Exact Same U-Net Backbone** to be trained/evaluated on either pixels or coordinates.

### Step 3: Evaluation Engine Alignment
Update `eval_ddpm_encdec_vision.py` to handle the D3IL state vector:
*   **Detection**: Automatically detect if the environment is returning a flat vector or a tuple of images.
*   **Formatting**: Package the 17-dim vector into the `cond` dictionary for the model.

---

## 3. Implementation Checklist

- [ ] **Task 1: D3IL State Mapping**
    - [ ] Create a `StateBridge` class in `ddpm_encdec_vision/models/state_bridge.py` that maps the 17-dim D3IL state to a 128-dim embedding.
- [ ] **Task 2: VisualUNet Generalization**
    - [ ] Modify `VisualUNet.__init__` to optionally instantiate a Linear state-projector.
    - [ ] Update `VisualUNet.forward` to branch based on input type.
- [ ] **Task 3: Wrapper Update**
    - [ ] Fix `VisualAgentWrapper.predict` to handle the `else: (if_vision=False)` branch.
    - [ ] Implement `env_state` parsing for state-only rollouts.
- [ ] **Task 4: Validation**
    - [ ] Train a small state-only Gen5 baseline for comparison.
    - [ ] Verify that `eval_ddpm_encdec_vision.py --no-vision` successfully rollouts.

---

## 4. Why This Matters for the Thesis
By supporting "Turn-Off Visual" in the **same framework**, we can provide a scientifically sound **Ablation Table**:

| Model | Conditioning | Success Rate | Path Smoothness |
| :--- | :--- | :--- | :--- |
| **Gen5-Visual** | Pixels (ResNet) | TBD | High (U-Net) |
| **Gen5-Blind** | Initial Pixel + Open-Loop | TBD | High (U-Net) |
| **Gen5-State** | Coordinates (17D Vector) | TBD | High (U-Net) |
| **ACT-State** | Coordinates (Baseline) | TBD | Low (Transformer) |

**Status**: Ready for Implementation (Fix #13).
