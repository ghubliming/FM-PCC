# D3IL Aligning Task & DPCC/DDPM Implementation Overview

This document provides a detailed explanation of the "Aligning" task in the D3IL environment, the multi-model DDPM implementation within the DPCC framework, and how we leverage D3IL's native architectures.

---

## 1. The "Aligning" Task in D3IL

### Objective
The **Aligning** task (defined in `gym_aligning`) is a vision-based robotic manipulation challenge. The goal is for a Franka Panda robot to push a physical box (`aligning_box`) from a random starting configuration to match a ghost target box (`target_box`) in both **position** and **orientation**.

### Observation Space (Vision-Centric)
The agent receives high-dimensional pixel input from two distinct viewpoints:
1.  **Cage Camera (`BPCageCam`):** A fixed, top-down "bird's-eye" view of the workspace. This provides global context for the box and target positions.
2.  **In-hand Camera:** A camera mounted on the robot's end-effector. This provides fine-grained visual feedback as the robot approaches the box.
3.  **Proprioception:** The current 3D position of the robot's end-effector.

---

## 2. Flexible Model Architectures (D3IL + DPCC)

The D3IL repository and the DPCC framework support multiple diffusion-based backbones. 

### A. D3IL Native ML Candidates
The following architectures are implemented in `d3il/agents/models/diffusion/diffusion_models.py`:
1.  **DDPM-MLP (`DiffusionMLPNetwork`):** Simple Multi-Layer Perceptron. processes observations as flattened vectors.
2.  **DDPM-GPT (`DiffusionTransformerNetwork`):** Causal Transformer (Self-Attention). **[Most similar to DPCC in purpose]**
3.  **DDPM-ACT (`DiffusionEncDec`):** Action Chunking Transformer (Encoder-Decoder).

### B. The DPCC Contribution (1D Temporal U-Net)
**The DPCC backbone is "Total New" relative to D3IL.** 
- **Module:** `diffuser.models.UNet1DTemporalCondModel`
- **Architecture:** 1D Convolutional U-Net.
- **Unique Status:** D3IL does not contain any convolutional temporal backbones. The U-Net was integrated via the `diffuser` library to support global trajectory denoising.

---

## 3. Implementation Workflow (Gen5 Current Status)

### **Current Backbone: U-Net (Hybrid Architecture)**
As of the current Gen5 implementation, we have opted for a **Hybrid Approach**:
- **Vision Model:** `MultiImageObsEncoder` (from D3IL).
- **Backbone Model:** `UNet1DTemporalCondModel` (from DPCC/Diffuser).

**Rationale:** The 1D U-Net is retained because of its superior performance in generating smooth, temporally-consistent trajectories that are compatible with the DPCC geometric projection operators.

### **The Gen5 Full Pipeline: From Pixels to Projection**

The Gen5 pipeline is a visual evolution of the state-based FMv3ODE pipeline.

| Stage | FMv3ODE (State-based) | **Gen5 (Visual-based)** |
| :--- | :--- | :--- |
| **1. Input** | Robot Pos, Obstacle Pos (Vectors) | **Raw Images** from 2 Cameras (Pixels) |
| **2. Encoder** | None (Direct Input) | **ResNet-18 (D3IL native)** |
| **3. Feature** | N/A | **128-dim Latent Vector** |
| **4. Condition** | "Don't hit the vector (x,y)" | "Move box to match pixels in target image" |
| **5. Backbone** | 1D U-Net | **1D U-Net (DPCC native)** |
| **6. Projection** | Safety Shield (Active) | **Inactive (Aligning task has no obstacles)** |
| **7. Result** | Safe path around obstacle | Path that pushes box to target |

#### **Technical Deep Dive: The Vision Encoder**
The "ResNet-18" and "128-dim feature" components are directly derived from the D3IL codebase:
- **Origin:** We instantiate `agents.models.vision.model_getter.get_resnet` (D3IL).
- **Structure:**
    - **Camera 1 (Cage):** ResNet-18 → 64-dim latent.
    - **Camera 2 (In-hand):** ResNet-18 → 64-dim latent.
- **Total Latent:** 64 + 64 = **128-dim conditioning vector**.

### **Input/Output Strategy: Sequence-to-Sequence**

| Model Type | Input | Output | Strategy |
| :--- | :--- | :--- | :--- |
| **D3IL MLP** | Image (Current) | Action (Next) | **Reactive** (Step-by-step) |
| **D3IL Transformer (ACT)** | Images (History Window) | Actions (Chunk) | **Sequence-to-Sequence** |
| **Gen5 (FMPCC/DPCC)** | **Images (History Window)** | **Trajectory (Full Plan)** | **Sequence-to-Sequence** |

---

## 4. Minimal Architectural Change: The Power of Conditioning

The core architectural innovation in Gen5 is the **Conditioning Swap**. 

The **U-Net** and the **Diffusion Engine** remain 100% identical to the state-based version. By simply encoding raw pixels into the same latent space where obstacle coordinates previously lived, we can repurpose the entire DPCC planning infrastructure for visual manipulation.

- **Internal Mechanism:** The 128-dim visual embedding is injected into the U-Net via **FiLM** (Feature-wise Linear Modulation) layers, which bias the convolutional features to generate a trajectory that achieves the visual goal.

---

## 5. Architectural Power Analysis: U-Net vs. D3IL Candidates

### The "U-Net Power" (DPCC)
The U-Net's power comes from its **Convolutional 1D Kernels**:
*   **Global Smoothing:** Unlike an MLP, the U-Net treats the entire trajectory as one "image." This physically prevents "jagged" or "jumpy" paths as the convolutions enforce temporal consistency.
*   **Projection Ready:** It is the **only** architecture compatible with our **Projection Operators**. Because it outputs a full trajectory, we can "project" that trajectory onto a safety manifold. Reactive models (MLP) cannot do this.

### D3IL Model Pipelines
*   **D3IL MLP:** **Step-by-Step.** It is "Image $\rightarrow$ Action." It has no concept of a trajectory and is often prone to high-frequency jitter.
*   **D3IL Transformer (ACT):** **Sequence-to-Sequence.** This is the only D3IL model that matches our pipeline (History Window $\rightarrow$ Action Chunk).

---

## 6. Why Modularize?

By utilizing the **D3IL native code for vision**, we ensure that our visual features are 1:1 with benchmark baselines. By keeping the **DPCC U-Net**, we preserve the framework's core ability to handle safety and control constraints. This modularization allows us to swap the perception backbone without losing the planning stability.

---

## 7. Technical Audit: Legacy DPCC vs. D3IL Semantics

| Component | D3IL Semantic Category | Implementation Module |
| :--- | :--- | :--- |
| **Backbone** | **None (Unique)** | `diffuser.models.UNet1DTemporalCondModel` |
| **Engine** | **Trajectory DDPM** | `diffuser.models.GaussianDiffusion` |
| **Environment** | **State-based Avoiding** | `d3il.environments.gym_avoiding` |

---

## 8. Summary: The DPCC-D3IL Relationship

To be mathematically precise for the thesis/documentation:

*   **Legacy DPCC (State-based):** 
    *   **Utilized from D3IL:** Environment logic (`gym_avoiding`) and Expert Datasets.
    *   **Custom ML Engine:** Built on the `diffuser` library (1D U-Net). **Zero dependency** on D3IL's agent/model code.
*   **Gen5 Aligning (Visual):** 
    *   **Utilized from D3IL:** Environment, Datasets, **AND** native **Vision Encoder** (`MultiImageObsEncoder`).
    *   **Current Backbone:** **U-Net.** 
    *   **Rationale:** Gen5 = D3IL Vision (Parity) + DPCC U-Net (Trajectory Quality).

### Comparison: D3IL Transformer vs. DPCC Trajectory U-Net

| Feature | D3IL Transformer (Similar) | DPCC U-Net (Current Gen5) |
| :--- | :--- | :--- |
| **Type** | **Attention-based** | **Convolution-based** |
| **Temporal View** | Global via Self-Attention tokens. | Global via 1D Sliding Kernels. |
| **Logic** | Predicts Action Chunks (Short). | Predicts Whole Trajectory (Long). |
| **Control** | Reactive actions. | **Plan-based Projection Control.** |

**Conclusion:** The DPCC U-Net provides a "Global Safety Planning" capability that is architecturally absent from the D3IL native candidate pool.

---

**Document updated for FM-PCC Diagnostic Phase 8.**
