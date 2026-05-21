# Implementation Plan: Unified Visual-DPCC Joint Trajectory Paradigm (3D Cartesian Edition)

This document presents a comprehensive, scientifically rigorous research and implementation plan to transition the **Gen6 (D3IL Visual Bridge)** architecture into a unified, **single-horizon joint trajectory DPCC model** (matching the standard DPCC diffuser design), adapted for full **3D (XYZ) spatial dimensions**.

---

## 🗺️ Architectural Concept: The 3D Joint State-Action Space

In the standard state-only DPCC way, the trajectory is formulated as a flat joint space: $x \in \mathbb{R}^{H \times (d_a + d_o)}$. 
In our **3D Cartesian Edition**, the trajectory is formulated as a flat 9D joint space over the horizon $H=8$:

$$x_t = \begin{bmatrix} a_t & s_t^{proprio} \end{bmatrix} \in \mathbb{R}^{3 + 6} = \mathbb{R}^9$$

where:
* **3D Actions ($a_t$):** $\begin{bmatrix} dx_t & dy_t & dz_t \end{bmatrix}$ (commanded displacement delta)
* **6D Observation Poses ($s_t^{proprio}$):** $\begin{bmatrix} des\_x_t & des\_y_t & des\_z_t & x_t & y_t & z_t \end{bmatrix}$ (commanded target + actual feedback robot coordinates)

```
       [Dual Camera Streams (Primary + Wrist)]
                          │
                          ▼
            [Spatial Softmax ResNet Encoder]
                          │
                          ▼
         [Low-Dimensional Visual Latents z] (e.g., 64D * 2 = 128D)
                          │
                          ▼
   [Linear Projector] ───> [Conditioning Vector emb] (256D)
                          │
                          ▼
            ┌───────────────────────────┐
            │   1D Temporal U-Net Core  │  ◄── [Denoises 9D joint trajectory grid H=8]
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │  3D SLSQP QP Projector    │  ◄── [Convex QP table/obstacle boundaries in 3D]
            └─────────────┬─────────────┘
                          │
                          ▼
                  [Executed Action a_0] (3D XYZ)
```

---

## 🔬 Core Innovations for Visual-DPCC Ingestion

We merge D3IL's high-fidelity visual data with standard DPCC diffusion/Flow Matching backbones through a unified conditional joint diffusion model:

### 1. Spatial Softmax ResNet Visual Conditioning (Conditioned Diffusion)
* **Mechanism:** Rather than treating images as external inputs that pollute the diffusion space, we keep the U-Net joint space focused on the 9D physical control variables ($x_t = [a_t, s_t^{proprio}] \in \mathbb{R}^9$).
* **Image Context Processing:** Dual camera frames at $t=0$ are encoded into compact spatial softmax latent vectors using pre-trained ResNet blocks.
* **Integration:** A linear projection layer merges these visual latents with the starting proprioceptive state to create a unified $256D$ conditioning vector `emb`. The temporal U-Net layers utilize FiLM linear projections to guide the trajectory denoising process:
  $$\hat x_{t-1} = \operatorname{UNet}(x_t, t \mid \operatorname{FiLM}(\text{emb}))$$
* **Key Benefit:** Complete separation of visual representation complexity from the joint diffusion dimensions, ensuring stable convergence and direct parity with state-only models.

---

## 📋 Phase-by-Phase Migration Plan

To implement this paradigm in your codebase, follow this structured plan:

### 🚀 Phase 1: The D3IL-to-DPCC 3D Visual Dataloader
1. **Target File:** Create a custom sequence dataset parser `AligningImgSequenceDataset` inside [diffuser_visual_aligning/datasets/sequence.py](file:///workspaces/FM-PCC/diffuser_visual_aligning/datasets/sequence.py).
2. **Implementation:**
   * Instantiate D3IL's untouched `Aligning_Img_Dataset` to load Dual camera frames.
   * Parse the raw state `.pkl` files to extract 3D desired coordinates and 3D physical feedback poses.
   * Concatenate them into the 6D sequence observation `[des_x, des_y, des_z, x, y, z]`.
   * Package them into sequence-based trajectory batches of length $H=8$.

---

### 🚀 Phase 2: Joint U-Net and Normalizer Adaptations
1. **Limits Normalization:** Set up `LimitsNormalizer` to scale 6D observations and 3D actions individually to $[-1, 1]$ boundaries.
2. **Transition Dimension scaling:**
   * Set the U-Net transition dimension to $d_a + d_{proprio} = 3 + 6 = 9$.
   * Lock the initial state ($t=0$) for the 6D proprioception state inside `apply_conditioning()` in `helpers.py`.

---

### 🚀 Phase 3: 3D Receding Horizon Controller
1. **Control Loop Migration:** Implement closed-loop step replanning in the online evaluation evaluator `eval_visual_aligning_dpcc.py`:
   ```python
   def step(self, obs_state, image_frames):
       # 1. Package current state condition (6D stacked observation)
       cond = {0: obs_state}
       
       # 2. Call joint trajectory model (denoises full H=8 trajectory)
       # And applies the 3D SLSQP projector on intermediate timesteps
       trajectory, infos = self.model(cond, projector=self.projector)
       
       # 3. Extract ONLY the first step action command a_0 (3D XYZ delta)
       action = trajectory[0, :self.action_dim]
       return action
   ```

---

## 🏆 Scientific Value and Thesis Contribution
By adapting Gen6 to the Visual-DPCC 3D joint trajectory paradigm:
1. **Real-time Obstacle Avoidance:** Eliminates the latency of chunked planning, enabling the robot to dynamically evade obstacles during visual rollouts.
2. **Safety Guarantees:** Applies 3D QP boundary projections directly to continuous physical trajectories, ensuring tabletop and collision limits are never violated.
3. **Multi-Modal Integration:** Combines robust, pre-trained visual feature representations with direct, low-dimensional trajectory diffusion for sample-efficient imitation learning.
