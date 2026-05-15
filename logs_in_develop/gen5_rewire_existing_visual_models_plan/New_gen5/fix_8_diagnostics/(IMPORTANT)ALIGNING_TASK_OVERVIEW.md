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

---

## 6. Data Provenance: Training & Trajectory Comparison

### A. Learning from Expert Demos
During the training phase, the Gen5 model is supervised by the D3IL **Expert Trajectory** dataset. 
- **Input:** Goal images + current observation images.
- **Target:** The ground-truth sequence of robot end-effector positions from successful human/expert pushes.
- **Result:** The model learns to map visual scenes to optimal, high-quality motion plans.

---

## 7. Evaluation Structure & Console Logging

When running the evaluation, the console provides real-time progress updates. Here is a guide to interpreting them:

### A. Context vs. Rollout
- **`Context X`**: This refers to a **Unique Problem Setup**. In the Aligning task, a "Context" defines the starting (x,y) position of the box and the rotation of the target. The evaluation set typically evaluates **30 unique contexts**.
- **`Rollout Y`**: This refers to a **Single Attempt** at a context. Because Diffusion models are probabilistic, we often run multiple attempts (e.g., 5 rollouts) for the same context to ensure the model's success is not a "fluke."

---

## 8. Why Modularize?

By utilizing the **D3IL native code for vision**, we ensure that our visual features are 1:1 with benchmark baselines. By keeping the **DPCC U-Net**, we preserve the framework's core ability to handle safety and control constraints. This modularization allows us to swap the perception backbone without losing the planning stability.

---

## 11. Data Transmission - What is Learned?

A key question for architectural analysis is exactly what is being learned and transmitted to the robot's hardware.

### A. The Target: End-Effector Positions
In all discussed architectures (D3IL and Gen5), we are transmitting **End-Effector Targets $(x, y, z)$**. 
*   **The Robot's Role:** The robot uses an internal **Inverse Kinematics (IK)** controller to follow the coordinate points provided by the model. 
*   **What is NOT learned:** We are **not** learning joint torques, voltages, or low-level motor currents. The "intelligence" lives at the task-planning level (where should the hand be?).

### B. "Trajectory" vs. "Action"
While both models output $(x, y, z)$ points, the **form** of that data differs significantly:

1.  **D3IL (DDPM-ACT): "Reactive Steps"**
    *   **Learns:** How to map the last few images to the next few small moves.
    *   **Transmits:** A "Chunk" of future actions (e.g., the next 10 small displacements).
    *   **Concept:** It is a reactive policy that continuously "corrects" the robot's course.

2.  **Gen5 (FMPCC/DPCC): "Global Trajectories"**
    *   **Learns:** How to map a visual scene to a **full, continuous geometric curve**.
    *   **Transmits:** A high-resolution trajectory plan covering the entire goal-reaching motion.
    *   **Concept:** It is a global planner. The U-Net treats the entire motion as a single "temporal image," ensuring the points are physically smooth and mathematically projectable.

---

## 12. Mathematical Distinction: Action Sequences vs. Trajectory Planning

To understand why the DPCC framework is unique, we must distinguish between the mathematical representation of **Actions** and **Trajectories**.

### A. The Action Sequence (Relative Steps)
In models like DDPM-ACT, the output is a set of **Action Deltas**:
$$A = \{\Delta x_1, \Delta x_2, \dots, \Delta x_k\}$$
*   **Recursive Dependency:** The robot's position at step $t+1$ is dependent on the previous state: $x_{t+1} = x_t + \Delta x_t$.
*   **Mathematical Limitation:** If any $\Delta x$ in the sequence is noisy, the error **compounds** over time (drift). There is no "global awareness" of where the robot ends up.

### B. The Trajectory Plan (Temporal Function)
In Gen5 (DPCC), the output is a **Temporal Trajectory**:
$$\tau = \{x_1, x_2, \dots, x_H\} \in \mathbb{R}^{H \times 3}$$
*   **Global Awareness:** The U-Net treats the entire plan as a single high-dimensional point in "Trajectory Space." All points $x_1 \dots x_H$ are predicted **simultaneously**.
*   **Mathematical Strength:** Because we have the entire curve $\tau$, we can apply a **Projection Operator** $P(\tau)$. 
    *   If $\tau$ hits an obstacle, the operator $P$ pushes the **entire curve** into a safe region while maintaining smoothness. 
    *   **You cannot "project" a single action** because an action is just a direction; it has no concept of "later" consequences.

---

## 13. Control Paradigm: MPC vs. Action Chunking

A final conceptual distinction for thesis documentation is the control loop philosophy.

### A. Gen5 (DPCC): True Generative MPC
The Gen5 pipeline implements a **Model Predictive Control (MPC)** loop:
*   **Model:** The 1D U-Net serves as the "Generative Model" of possible future behaviors.
*   **Receding Horizon:** At every control tick, the system generates a full future horizon (e.g., the next 8-16 steps). It executes only the **first few steps** of this plan.
*   **Re-planning:** The system then immediately captures a new visual observation and **re-plans the entire trajectory** from the new position. This closed-loop "receding horizon" behavior is the definition of MPC.

### B. DDPM-ACT: Open-Loop "Action Chunking"
The ACT architecture uses a different philosophy:
*   **Execution:** It generates an "Action Chunk" (a sequence of steps) and executes them in an **open-loop** fashion (or uses "Temporal Ensembling" to smooth overlapping chunks).
*   **Philosophy:** It is a high-capacity **Reactive Policy**. It does not "re-plan" the global path at every tick in the way a Generative MPC system does; it simply samples a likely next sequence of actions.

---

## 14. Failure Analysis: The Impact of MPC Step Size

In high-precision tasks like Aligning, the **Step Size** (the number of planned steps executed before re-planning) is a critical performance bottleneck.

### Why "Big Steps" Lead to Failure:
1.  **Feedback Latency:** If the robot executes 4 steps of an 8-step plan without looking at the cameras, it is essentially "driving blind" for half the horizon. In a dynamic task, the box may have shifted, making the remainder of the plan invalid.
2.  **Contact Dynamics Instability:** Pushing a box requires microscopic adjustments. A large MPC step often results in an aggressive, high-velocity command that can cause the box to rotate or "squirt" out from under the robot's end-effector.
3.  **Tracking Error Drift:** The MPC loop assumes the robot starts each new plan exactly where the *previous* plan ended. If the robot's physical controller (IK) only completes 90% of a "big step," the next plan starts from a hallucinated state, causing the model to diverge from the goal.

### Optimization Strategy:
To maximize accuracy, we aim for a **High-Frequency Receding Horizon** (e.g., executing only 1 or 2 steps per re-plan). This increases the computational cost but ensures the robot is always reacting to the most recent visual state of the box.

---

## 15. Parameter Tuning Guide

To optimize the Gen5 Visual-Aligning pipeline, you should focus on the following parameters across two configuration files.

### A. Evaluation Parameters (`config/visual_aligning_eval.yaml`)
*   **`n_contexts` (The Judge of Generalization)**: This defines the number of unique "problem scenes" the model must solve (Default: 30).
    *   **What is a Context?** Each context is a unique starting configuration (e.g., Box is at $A$, Target is at $B$).
    *   **The Scientific Meaning:** If you only test 5 contexts, you might get "lucky." Testing **30-100 contexts** ensures the model hasn't just memorized a few paths, but has truly learned the **spatial logic** of the task.
    *   **Thesis Rigor:** Higher values provide a much stronger scientific proof of **generalization** across the entire robot workspace.
*   **`seeds`**: The random seeds for simulation initialization.
    *   *Expected Change:* Evaluating across multiple seeds ensures the success rate is statistically stable.

### B. Architecture & Planning Parameters (`config/aligning-d3il-visual.py`)
These parameters live in the `ddpm_encdec_vision` block and affect both training and inference.

*   **`action_seq_size` (MPC Chunk Size)**: **CRITICAL.** This is the number of steps executed before re-planning (Default: 4).
    *   *Tuning:* Reduce this to 1 or 2 for higher precision.
    *   *Expected Change:* Significant increase in success rate at the cost of slower real-world execution (more compute).
*   **`horizon` (Planning Horizon)**: Total length of the planned trajectory (Default: 8).
    *   *Expected Change:* Longer horizons give the model better "foresight," but make the denoising task harder for the U-Net.
*   **`obs_seq_len` (Visual History)**: Number of historical image frames provided to the ResNet encoder (Default: 5).
    *   *Expected Change:* Increasing this helps the model understand the **momentum** of the box, but may introduce lag if the history window is too long.
*   **`n_diffusion_steps`**: Number of denoising iterations (Default: 16-20).
    *   *Expected Change:* Higher values improve trajectory quality (less noise) but significantly slow down the inference speed.

---

## 16. The Visual Encoder: Dimensionality Reduction via ResNet

A common question in robotic vision is why we cannot feed raw images directly into the U-Net. This section explains the encoding process.

### A. The Challenge: The Curse of Dimensionality
A single image from our cameras is $128 \times 128 \times 3 = 49,152$ pixels. 
*   **If we used raw pixels:** The U-Net's "Conditioning" input would be nearly 50,000 dimensions. This would require billions of parameters to train and would likely never converge.
*   **The Solution:** We need a **Visual Encoder** to compress this massive amount of data into a small, information-dense **Latent Vector**.

### B. The Encoder: ResNet-18
We use a **ResNet-18** (Residual Network) as our backbone. The process works as follows:
1.  **Input:** The raw RGB tensor $[3, 128, 128]$.
2.  **Feature Extraction:** The ResNet's convolutional layers scan the image. 
    *   Early layers find simple **Edges**.
    *   Middle layers find **Shapes** (the box corners).
    *   Deep layers find **Semantic Meaning** (the relative distance between the box and the target ghost).
3.  **Global Pooling:** The final layer of the ResNet collapses the spatial dimensions into a single vector.
4.  **Projection:** A small Multi-Layer Perceptron (MLP) maps this vector to our desired **64-dimensional latent**.

### C. Multi-View Fusion
Because we have two cameras, we perform this process twice:
*   **Cage Cam Encoder:** Produces a 64-dim "Global Intelligence" vector.
*   **In-hand Cam Encoder:** Produces a 64-dim "Precision Intelligence" vector.
*   **Fusion:** We **concatenate** these two vectors into a single **128-dimensional Conditioning Vector**.

---

## 17. End-to-End Training Philosophy

A final clarification for the Gen5 architecture: The ResNet Encoder and the U-Net Backbone are **NOT trained as separate blocks**.

### A. The Unified "Brain"
In the Gen5 pipeline, the entire network is trained **End-to-End**. 
*   **Simultaneous Optimization:** During the training phase, the ResNet and the U-Net are optimized together in the same loop. 
*   **Gradient Flow:** If the U-Net predicts an incorrect trajectory, the mathematical "error" signal (the gradient) travels all the way back through the U-Net, through the FiLM conditioning layers, and directly into the ResNet weights.

### B. Task-Specific Vision
Because of this end-to-end training, the ResNet does not just learn to "see everything." It learns **Task-Specific Vision**. It ignores irrelevant background details (like the table texture) and learns to emphasize only the specific pixels that are most useful for generating a successful trajectory plan. 

### C. Unified Evaluation
When we evaluate the model, we are testing the **collective intelligence** of the combined vision-planning loop. We do not evaluate them in isolation because their strength lies in their integration.

---

**Document updated for FM-PCC Diagnostic Phase 9.**
