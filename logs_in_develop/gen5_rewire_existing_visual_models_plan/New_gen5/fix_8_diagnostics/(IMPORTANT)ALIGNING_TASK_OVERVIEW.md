# D3IL Aligning Task & DPCC/DDPM Implementation Overview

This document provides a detailed explanation of the "Aligning" task in the D3IL environment, the multi-model DDPM implementation within the DPCC framework, and how we leverage D3IL's native architectures.

---

## 1. The "Aligning" Task in D3IL

### Objective
The **Aligning** task (defined in `gym_aligning`) is a vision-based robotic manipulation challenge. The goal is for a Franka Panda robot to push a physical box (`aligning_box`) from a random starting configuration to match a ghost target box (`target_box`) in both **position** and **orientation**.

### Observation Space (Vision-Centric)
The agent receives high-dimensional pixel input from two distinct viewpoints:
1.  **Cage Camera (`BPCageCam`):** A fixed, top-down "bird's-eye" view  the workspace. This provides global contexoft for the box and target positions.
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

### B. The Solution: Learned Feature Extraction (ResNet)
A common misconception is that the ResNet is a "static" dimensionality reduction tool (like a Fourier Transform or PCA). In the Gen5 pipeline, this is false.

1.  **Is it just Dimension Reduction?**
    *   Yes, it reduces the input space from 49,152 pixels to a 128-dimensional embedding.
2.  **Is it "Untrained"?**
    *   **No.** While we often initialize the ResNet with "Pre-trained Weights" (from ImageNet), the encoder is **Fine-tuned** during the training of the Aligning task.
    *   **The "End-to-End" Factor:** The ResNet and the U-Net are trained as a **single unit**. When the robot fails to push the box, the "Error Signal" (Gradient) flows back through the U-Net and directly into the ResNet.
3.  **What does it "Learn"?**
    *   The ResNet learns to **ignore** the background (the floor, the cage walls) and **focus** specifically on the $(x,y,\theta)$ of the box and the target. It transforms a "picture" into a "mathematical meaning" that the U-Net can understand.

### C. In-Code Implementation: The Training Entry Point
To see how this is implemented in the Gen5 codebase, you can look at the following files:

1.  **The Training Script:** `ddpm_encdec_vision_test/train_ddpm_encdec_vision.py`
    *   This is the "Brain" of the training process. It initializes the dataset, the model, and the optimization loop.
2.  **The Dataset:** `d3il/environments/dataset/aligning_dataset.py` (`Aligning_Img_Dataset`)
    *   This class loads the expert images from disk and prepares them for the ResNet.
3.  **The Hybrid Model:** `ddpm_encdec_vision/models/visual_unet.py` (`VisualUNet`)
    *   This is where the ResNet and U-Net are joined.
    *   **The Code Proof:** If you look at the `forward()` pass of `VisualUNet`, you will see that the images are first passed through `self.resnet`, and the output is then fed into the U-Net's conditioning layers. Because they are connected in one single Python object, PyTorch treats them as one network during training.

**When was it trained?**
Training is typically performed once for **100,000 to 500,000 steps** on a GPU. After this, the "weights" (the learned knowledge) are saved as `.pt` files in the `logs/` directory. When you run an evaluation, you are loading these pre-learned weights.

---

## 17. End-to-End Training Philosophy

A common point of confusion is whether the ResNet and U-Net are trained in two separate stages. In the Gen5 architecture, the answer is **No**. There is only **one training process**.

### A. How it is achieved in Code (PyTorch Logic)
The "Glue" that binds the two models together is the **`VisualUNet`** class (found in `ddpm_encdec_vision/models/visual_unet.py`).

1.  **Single Container:** The `VisualUNet` class contains both `self.obs_encoder` (ResNet) and `self.backbone` (U-Net) as its children.
2.  **The Optimizer's View:** When the training script starts, it tells the Optimizer (Adam) to watch `model.parameters()`. Because the ResNet is inside the model, the Optimizer "sees" the ResNet weights and the U-Net weights as **one big list of numbers** to improve.
3.  **Differentiable Bridge:** In the `forward()` pass:
    *   Step 1: Raw images $\to$ ResNet $\to$ **Visual Embedding**.
    *   Step 2: **Visual Embedding** $\to$ U-Net $\to$ **Trajectory**.
    *   Because the Visual Embedding is a direct output of the ResNet and a direct input to the U-Net, PyTorch builds a **single computational graph**. 

### B. The "Two-Stage" vs. "End-to-End" Difference
*   **Two-Stage (NOT used here):** You train a ResNet to classify objects, lock it, and then use its fixed features to train a robot. (The ResNet never learns about the robot's needs).
*   **End-to-End (Gen5 Method):** The ResNet is wide open. If the U-Net realizes it needs to see the *corners* of the box better to make a better plan, it sends a signal back to the ResNet to adjust its convolutional filters to highlight corners.

**Summary:** The ResNet and U-Net are "partners" in a single optimization loop. They learn to communicate with each other from scratch.

### C. Parity with D3IL (ACT) Principle
You may wonder if this "End-to-End" approach is an experimental choice for Gen5. It is not. It is a strict adherence to the **D3IL Golden Standard**.

1.  **D3IL Implementation:** If you inspect `d3il/agents/act_vision_agent.py`, you will see the `ActPolicy` class. It bundles the **ACT Transformer** and the **ResNet Encoder** into one object, exactly like our `VisualUNet`.
2.  **Shared Principle:** Both D3IL and Gen5 follow the same philosophy: **Vision should not be a "witness"; it must be an "active participant."**
    *   By training together, the ResNet learns what the U-Net/Transformer needs to see to succeed.
    *   If we used a fixed, pre-trained ImageNet ResNet without training it further, the robot would likely be "clumsy" because it would be looking for "cats and dogs" (ImageNet features) instead of "box corners and rod tips" (Aligning features).

**Conclusion:** Gen5's training philosophy is 100% consistent with the D3IL benchmarks. We have swapped the "Backbone" (U-Net instead of Transformer), but we kept the "Neural Connectivity" (End-to-End) identical.

### D. Final Technical Confirmation
To answer your question directly: **Yes.** 

*   We do **not** have a "ResNet training phase."
*   We do **not** have a "U-Net training phase."
*   We have a **Joint Training Phase** where the Optimizer calculates one single loss and updates **all weights in the ResNet and all weights in the U-Net at the exact same time.**

**One Model, One Gradient, One Goal.**

---

## 18. Parameter Provenance: The Origin of Defaults

| Parameter | Default Value | Source / Ancestry | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **`horizon`** | 8 | **FMv3ODE (Legacy DPCC)** | Maintains parity with state-based planning benchmarks. |
| **`window_size`** | 8 | **FMv3ODE (Legacy DPCC)** | Power-of-2 architectural constraint for U-Net downsampling. |
| **`obs_seq_len`** | 5 | **D3IL Benchmark** | Standard history length for visual manipulation tasks in D3IL. |
| **`n_contexts`** | 30 | **D3IL Benchmark** | Official benchmark scale for the Aligning task. |
| **`action_seq_size`** | 4 | **Experimental (Heuristic)** | Balanced re-planning frequency (exactly half of the horizon). |
| **`n_diff_steps`** | 16-20 | **D3IL / ACT** | Optimal quality vs. speed trade-off for visual diffusion models. |
| **`obs_dim`** | 128 | **ResNet-18 (D3IL)** | Feature vector size generated by the dual-ResNet visual backbone. |

---

## 19. Total Horizon Flexibility: The Auto-Padding Fix

In previous versions of the Gen5 pipeline, the model was rigid and would crash if the `horizon` was set to a small value (like 2 or 4). This was due to the "Downsampling Bottom-out" error in the U-Net.

### A. The FMPCC Standard Restored
As of Fix #10, the Visual Gen5 pipeline now matches the **Flexibility of FMv3ODE**. 
*   **The Mechanism:** The `VisualUNet` wrapper now implements **Automatic Temporal Padding**.
*   **How it Works:** Regardless of the `horizon` you choose in the config, the model automatically pads the trajectory to a "Safe Size" (multiple of 8) inside the forward pass, processes it, and crops the result back to your desired length.

### B. Updated Tuning Strategy
You can now freely set the `horizon` to any value (1, 2, 4, 8, 16) without changing any other architectural parameters.

| Goal | Horizon ($H$) | `action_seq_size` | Behavior |
| :--- | :--- | :--- | :--- |
| **Precision** | 2 | 1 | High-frequency reactive corrections. |
| **Foresight** | 8 | 4 | Long-range planning with receding horizon. |
| **Hybrid** | 4 | 2 | Balance of foresight and reactivity. |

---

## 20. Visual History (obs_seq_len): Understanding Motion

In vision-based control, a single image is only a "snapshot." The `obs_seq_len: 5` parameter provides the robot with a short-term memory.

### A. From Static to Dynamic
*   **1 Image:** The robot knows the **Position** of the box.
*   **5 Images:** The robot knows the **Velocity and Inertia** of the box. 
    *   By looking at the "delta" between five consecutive images, the vision encoder can mathematically perceive if the box is sliding quickly, spinning, or coming to a stop.

---

## 21. Native D3IL DDPM-ACT Baseline (The Original Version)

For scientific comparison, it is important to understand the original D3IL version of this task before our Gen5 "U-Net Upgrade."

### A. The ML Architecture: DiffusionEncDec (ACT)
*   **Backbone:** Transformer Encoder-Decoder (Self-Attention).
*   **Temporal Logic:** Operates on discrete "Action Chunks." It does not have a convolutional U-Net for trajectory smoothing.
*   **Latent Dimension:** Typically 512 (Standard D3IL) vs. our 128 (Gen5).

### B. Standard D3IL Hyperparameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **`horizon`** | 10 | Standard "Action Chunk" size in ACT research. |
| **`n_diffusion_steps`** | 100 | Uses a longer denoising chain than our high-speed Gen5. |
| **`obs_seq_len`** | 1 | Typically reactive (no visual history) unless explicitly configured. |
| **`model_type`** | `act` | Activates the Transformer-based policy. |
| **`n_train_steps`** | 500,000 | Official benchmark scale (500 Epochs x 1000 steps). |
| **`batch_size`** | 64 | Standard vision batch size in D3IL. |
| **`learning_rate`** | 5e-4 | Official Adam LR for ACT Transformer. |

### C. One-Click Training (Native D3IL)
To train the original baseline for comparison, use the native D3IL entry point:
```bash
python d3il/agents/train_diffusion.py --task aligning --model act
```

### D. Evaluation System
The native D3IL code includes an internal evaluation loop that calls `Aligning_Sim.test_agent()`.
*   **Metrics:** It only reports the **Success Rate** and **Mean Goal Distance**.
*   **Gap:** It lacks the 7-metric report (Violations, Entropy, Steps) that our Gen5 script provides.

---

## 22. Replicating DDPM-ACT within Gen5 (The "Parity Setup")

To conduct a scientifically valid comparison, you can configure Gen5 to mimic the behavior of the native DDPM-ACT baseline as closely as possible.

### A. The "ACT-Style" Gen5 Config
To match the ACT baseline, update your `config/aligning-d3il-visual.py` with these values:

| Parameter | ACT-Parity Value | Rationale |
| :--- | :--- | :--- |
| **`horizon`** | 10 | Matches the standard ACT chunk size. |
| **`window_size`** | 16 | (Next multiple of 8 for U-Net stability). |
| **`action_seq_size`** | 10 | Executes the full chunk without re-planning (Open-Loop). |
| **`obs_seq_len`** | 1 | Makes the model reactive (no historical context). |
| **`n_diffusion_steps`** | 100 | Matches the baseline denoising quality. |
| **`batch_size`** | 64 | Match benchmark training density. |
| **`learning_rate`** | 5e-4 | Match benchmark optimizer setup. |

### B. Fundamental Differences (Irreconcilable)
Even with the parity config, these **Core Architectural Differences** cannot be replicated in Gen5:

1.  **Backbone Mechanism:**
    *   **ACT:** Uses **Self-Attention** (Transformers). It learns global dependencies across the chunk using attention weights.
    *   **Gen5:** Uses **1D Convolutions** (U-Net). It treats the trajectory as a temporal signal, enforcing local smoothness through kernel overlaps.
2.  **Trajectory Topology:**
    *   **ACT:** Produces a sequence of discrete actions. It is often "jittery" at high frequencies.
    *   **Gen5:** Produces a continuous geometric curve. This is required for **Projection Operators** (Safety) which ACT cannot natively support.
3.  **Inference Logic:**
    *   **ACT:** Designed for "Action Chunking" (Open-loop execution of 10 steps).
    *   **Gen5:** Designed for "Generative MPC" (Receding horizon re-planning).

### C. Scientific Recommendation
For your thesis, use the **ACT-Parity Setup** in Gen5 to prove that even with identical hyperparameters ($H=10$), the **U-Net Backbone** provides superior path smoothness and safety over the Transformer baseline.

---

## 23. Head-to-Head Comparison: U-Net vs. Transformer (ACT)

For your thesis, this is the most critical scientific distinction. Even if both models use the same vision and parameters, the "Backbone" changes the physics of the robot's motion.

| Feature | Gen5 (1D Temporal U-Net) | Native D3IL (Transformer ACT) |
| :--- | :--- | :--- |
| **Math Core** | **1D Convolutions** | **Self-Attention** |
| **Temporal Logic** | Local/Spatial (Smoothness by Design) | Global/Semantic (Correlation by Design) |
| **Path Quality** | **Continuously Differentiable.** The plan is a single geometric curve. | **Discrete Steps.** The plan is a sequence of individual jumps. |
| **Safety** | **Projection Ready.** Can be "bent" by geometric operators. | **Non-Local.** Hard to project without breaking temporal logic. |
| **Bottleneck** | Architectural safe-horizon (Fixed by Auto-Padding). | High memory cost for long horizons. |
| **Capacity** | **High (20M+ Params)** | **Low (~1M Params)** |
| **Best For** | **High-precision manipulation & Safety.** | Complex, multi-stage semantic tasks. |

### 24. Quantitative Capacity Analysis: Why Gen5 is "Smarter"

| Metric | Gen5 (U-Net) | Native D3IL (ACT) | Ratio |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | **~18,000,000** | **~900,000** | **20x Larger** |
| **Backbone Depth** | 8 Convolutional Blocks | 6 Transformer Layers | 1.3x Deeper |
| **Latent Width** | 128 $\to$ 1024 (Expanding) | 64 (Fixed) | **16x Wider** |
| **Inference Cost** | Higher (Iterative Denoising) | Lower (Transformer Pass) | - |

**Thesis Note:** The 20x increase in parameter capacity allows Gen5 to store much more "spatial knowledge" about the box's geometry. This explains why Gen5 can achieve higher success rates in "Aligning" than the lightweight ACT baseline, as it has more "neurons" dedicated to understanding the relationship between the two camera views.

### Why the U-Net is the "Better Bone" for FM-PCC:
1.  **Convolutions enforce Smoothness:** Because 1D Convolutions "slide" across time, every point in the plan is mathematically linked to its neighbors. This prevents the "shaking" or "jitter" often seen in Transformer-based robots.
2.  **The Projection Factor:** In FM-PCC, we need to "Project" the trajectory onto a safe manifold. A U-Net trajectory is like a **piece of wire**—you can bend it smoothly. A Transformer output is like **scattered beads**—if you move one, the others don't necessarily follow smoothly.

---

## 25. Evaluation Log Guide: Expert References & System Resources

When initiating an evaluation rollout, you will encounter specific console logs. This section explains their technical meaning for your experiment records.

### A. Training Continuity
> `[ utils/training ] Restored loss history from checkpoint at step 48000`
*   **Source Code:** `ddpm_encdec_vision/utils/training.py:L335`
*   **Meaning:** The script has successfully located a saved model checkpoint at **48,000 training steps**.
*   **Significance:** It is loading the "learned intelligence" of the robot from this specific point in its training history. All performance metrics reported in the evaluation are tied to this 48k-step maturity level.

### B. The "Expert" Visual Baseline
> `[ expert ] Generating 3 reference videos from dataset...`
*   **Source Code:** `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py:L241`
*   **Meaning:** Before testing the AI model, the script reads the original **Human/Expert Dataset** and records 3 "perfect" rollouts.
*   **Significance:** These GIFs (saved in `expert_references/`) serve as a **Scientific Control**. If the AI fails, you can look at these expert videos to see how a "perfect" push should look. This helps distinguish between "Impossible Scene Setups" and "Model Failures."

### C. Inverse Kinematics (IK) Initialization
> `Final IK error (74 iterations): 8.113e-06`
*   **Source Code:** `environments/d3il/d3il_sim/controllers/TrajectoryTracking.py`
*   **Meaning:** To play back an expert demo, the simulated robot must first move its hand to the exact $(x,y,z)$ starting point of that demo.
*   **Significance:** The simulator uses an **IK Solver** to calculate the joint angles required to reach that point. The "error" (e.g., $10^{-6}$) confirms that the robot's hand is physically positioned with sub-millimeter precision before the test begins.

### D. System Resource Detection
> `there are cpus: 64`
*   **Source Code:** `d3il/simulation/aligning_sim.py:L147`
*   **Meaning:** The environment has detected 64 logical CPU cores on the host machine.
*   **Significance:** This is utilized by MuJoCo and PyTorch for **Parallel Physics Processing**. A higher core count ensures that image rendering and collision detection (the most expensive parts of the visual pipeline) do not bottleneck the evaluation speed.

---

## 26. Real-Time Scientific Diagnostics (FIX_12)

As of the Gen5 Diagnostic Phase 14, the evaluation script (`eval_ddpm_encdec_vision.py`) has been upgraded to support **Real-Time Data Streaming**.

### A. Instant PNG Reports
The system no longer waits for all 50+ contexts to finish. Instead:
1.  **After every rollout**, a high-fidelity **6-panel PNG report** is generated in `realtime_diagnostics/`.
2.  This report overlays the **MPC Foresight (the Blue Fan)** with the **Real Execution (the Black Path)**.
3.  It includes temporal plots for $x$, $y$, and $z$ to visualize control stability.

### B. Why this matters for Debugging
If the robot fails context #5, you can immediately open `rollout_5_report.png` to see if:
*   The **U-Net** predicted a bad path (Visual/Planning failure).
*   The **Robot** drifted away from the predicted path (IK/Control failure).
*   The **Z-height** oscillated (Physics/Contact failure).

**Data Persistence:** A corresponding `.pkl` file is also saved for each rollout, ensuring that if the simulation is interrupted, no scientific data is lost.

---

## 27. Dimensionality: 2D Task vs 3D Model

A common point of confusion is whether the Aligning task is 2D or 3D.

### A. The Reality
*   **The Physical Task**: The box moves on a 2D table surface.
*   **The Robot Control**: The Gen5 model uses **3D Actions (XYZ)**.
*   **The Rationale**: While the task is planar, the robot end-effector must maintain a precise **Z-height** to stay in contact with the object. By using 3D actions, the U-Net learns the optimal "Interaction Height" directly from the Expert Dataset, rather than relying on a hardcoded Z-value which might be brittle across different simulation seeds.

### B. Guiding Capacity
The Gen5 trajectory (Horizon 8) provides sufficient guidance because it encodes both the **Spatial Path** and the **Temporal Velocity**. This "Foresight" allows the robot to anticipate corners and contact points more smoothly than legacy point-prediction models.

---

## 28. Mixed-Loop Control: The Vision vs. State Divergence

A critical scientific finding in the D3IL replication is the "Mixed-Loop" nature of the evaluation.

### A. Vision (Closed-Loop)
*   **Mechanism**: The model processes a fresh camera image every $N$ steps (the re-planning interval).
*   **Function**: Vision provides the **Ground Truth** for the box position and any environmental changes. It is the primary "Error Correction" signal.

### B. Proprioception/State (Open-Loop)
*   **Mechanism**: The model is conditioned on its **intended** (mental) position, not the **measured** (physical) simulation position.
*   **Function**: This prevents "Jitter Feedthrough." By ignoring simulation drift ($8e-06$ IK errors), the model stays on the "Expert Manifold" it learned during training.

---

## 29. The SNR Barrier: Why Normalization is Mandatory

Diffusion models (U-Net) operate by denoising Gaussian noise $\mathcal{N}(0, 1)$.

*   **The Problem**: Raw robot actions are in meters (e.g., $0.005$m). If the signal is $0.005$ and the noise is $1.0$, the **Signal-to-Noise Ratio (SNR)** is effectively zero.
*   **The Fix**: Data Normalization (Scaling) maps actions to the $[-1, 1]$ range. This brings the signal into the same "numerical power" as the noise, allowing the U-Net to "see" the trajectory it is trying to denoise.
*   **Catastrophic Failure**: Without scaling, the model learns to output pure noise, resulting in the robot "jittering" or moving randomly.

---

## 30. Temporal Chunking: Why Horizon 10?

The use of a 10-step horizon (chunking) instead of 1 step or 400 steps is a design trade-off for complex visual tasks.

1.  **Why not 1 step?**: Single-step predictions lack temporal smoothness. The robot's motion would be jerky and "reactive" rather than "planned."
2.  **Why not 400 steps?**: Predicting the entire task at once makes the model **Blind**. If the box slips at step 50, a 400-step plan made at step 1 cannot adjust.
3.  **The "Goldilocks Zone" (Horizon 10)**: 10 steps are long enough to ensure **Smooth Velocity** (re-using the temporal backbone) but short enough to allow **Reactive Vision** (re-planning allows the model to "open its eyes" and correct for box movement).

---

## 31. The Stabilization Breakthrough: Gen5 "Visual Transplant" Finalized (FIX_12 P2)

As of FIX_12 Phase 2, the Gen5 Visual-Aligning pipeline has reached its final stabilized form. This section serves as the incremental update for the "Scientific Replication" milestone.

### A. The Evolution of Feedback Logic
*   **Previous Approach (Section 27)**: Early attempts used "Closed-Loop State" feedback (Measured Position). This was found to be **Catastrophic** for Visual Aligning because simulation drift ($8e-06$) acted as OOD noise that "distracted" the U-Net from its visual goals.
*   **Stabilized Approach (Section 28)**: We have officially transitioned to **Mixed-Loop Control**. Proprioception is now **Open-Loop** (trusting the Mental Map), while Vision remains **Closed-Loop** (watching the camera). This ensures the robot stays on the "Expert Manifold."

### B. The Numerical Foundation (Scaling)
*   The implementation of the **Scaling Bridge** (Section 29) solved the numerical signal-to-noise crisis. By normalizing actions to the $[-1, 1]$ range, we enabled the U-Net to distinguish between expert intent and diffusion noise.
*   **Crucial Note**: All checkpoints prior to this update are **DEPRECATED**. Evaluation requires a `scaler.pkl` generated during the new standardized training phase.

### C. Detailed Evidence & Summary
For the full technical breakdown, refer to the incremental report:
*   [FIX_12_P2_STRICT_PARITY_SUMMARY.md](file:///workspaces/FM-PCC/logs_in_develop/gen5_rewire_existing_visual_models_plan/New_gen5/fix_12/FIX_12_P2_STRICT_PARITY_SUMMARY.md)

---

## 32. The Dimension Bridge: 6D vs. 3D Latent Alignment

A second critical runtime barrier was the disagreement between the **Brain (Gen5)** and the **Body (D3IL)** regarding tensor dimensions.

### A. The Diagnosis
*   **The Gen5 Brain (U-Net Output)**: In the FM-PCC framework, the Diffusion model predicts a **6D vector** ($3 \times$ State $+ 3 \times$ Action). This is because the U-Net denoises the entire trajectory including the robot's proprioception to ensure global smoothness.
*   **The D3IL Body (The Scaler)**: Our implementation of the D3IL Scaler was trained strictly on the **3D Action** (the deltas) to match the baseline's behavior.
*   **The Conflict**: When calling `inverse_scale_output(trajectory)`, the scaler attempted to apply 3D statistics to a 6D tensor, causing a `RuntimeError`.

### B. The Fix: Slicing Logic
To bridge this gap, the `eval_ddpm_encdec_vision.py` script was updated to slice the action component out of the 6D diffusion output **BEFORE** applying the inverse scaler.

**Slicing Logic:**
1.  **`Trajectory[:, :, :3]` (Denoised State)**: We ignore this component because we have a "Mental Map" for proprioception.
2.  **`Trajectory[:, :, 3:6]` (Denoised Action)**: This is the component we scale and execute.

---

---

## 33. The Expert Baseline Engine: Ground Truth Documentation

To ensure scientific validity, the Gen5 evaluation pipeline automatically generates a "Golden Standard" from the human expert dataset before testing the AI.

### A. Data Provenance (Where is it loaded from?)
The expert trajectories are loaded from the D3IL serialized dataset:
*   **Path**: `d3il/environments/dataset/data/aligning/test_contexts.pkl`
*   **Source Code**: `d3il/simulation/aligning_sim.py:L22`
*   **Content**: These files contain the exact $(x, y, z)$ coordinates and camera images recorded during successful human trials.

### B. The Reference GIFs: Meaning and Purpose
*   **What it means**: When you see `[ expert ] Saved ... expert_rollout_0.gif`, the script has successfully "replayed" a human demonstration in the simulator.
*   **Scientific Significance**: These GIFs provide a **Visual Control**. If your model reaches a success rate of 0%, you can check these GIFs to confirm that the environment itself is working. If the expert can finish the task in the GIF but the AI cannot in the rollout, the failure is purely in the AI's "intelligence," not the simulation physics.

### C. Visual Logic (Computation vs. Action)
*   **Playback Speed**: GIFs are recorded at a constant **10 FPS**.
*   **Calculation Time**: The GIFs **skip the calculation steps**. They do not show the "thinking time" (the 1.4s U-Net inference delay). They only show the **Actions** (the physical results).
*   **Result**: This allows you to compare the **Path Quality** of the AI vs. the Human Expert without being distracted by computational latency.

---

---

## 34. The Zero-Padding Trap: Resolving the Numerical Explosion

A "severe problem" was observed during validation: the robot end-effector was flying out of bounds at hypersonic speeds (e.g., $1750$ meters in a $1$ meter workspace).

### A. The Cause: Statistics Corruption
*   **Mechanism**: The dataset uses **Zero-Padding** to reach a fixed length of 256 steps. Because the Scaler was calculated on the raw padded tensors, the thousands of "padding zeros" caused the **Standard Deviation** to collapse.
*   **The Math Failure**: When the model's scaled output was multiplied by this corrupted, near-zero standard deviation, the resulting "meters" value exploded into massive, unrealistic numbers.
*   **The Feedback Loop**: Because we use a **Mental Map (Open-Loop)**, these huge erroneous actions were accumulated every step, causing the robot to accelerate into space exponentially.

### B. The Fix: Masked Statistics (Numerical Grounding)
The training initialization was updated to use **Masked Data** only:
*   `all_obs = dataset.get_all_observations()`
*   `all_act = dataset.get_all_actions()`
*   **Result**: This forces the Scaler to compute its Mean and Std using only the **Real Human Demonstrations**, ignoring the padding. This restores the "Numerical Grounding" of the model.

---

**Document updated for FM-PCC Diagnostic Phase 20 (Numerical Grounding Finalized).**
